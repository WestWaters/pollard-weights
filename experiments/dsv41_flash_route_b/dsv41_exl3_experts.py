#!/usr/bin/env python3
"""dsv41_exl3_experts.py — route-B EXL3 quantizer for the DeepSeek-V4.1-Flash ROUTED experts (EXL3_PORT_PLAN.md section (e)).

Drives exllamav3's own quantizer (`quantize_exl3` / `quantize_exl3_batch`: LDLQ + mul1 trellis codebook) per expert matrix from
the capture kit's per-layer dumps instead of running exllamav3's model forward — DeepSeek's reference `model.py` produced the
activations (dsv41_capture.py --hessian-expert-mode dump), so the only exllamav3 code in the loop is the quantizer itself.

Inputs
  1. `ffn_in_L{LL}.safetensors` dumps: {"x": bf16 [R*S, 5120] (MoE input after ffn_norm, every token), "route_idx": int16 [R*S, 6],
     "route_w": bf16 [R*S, 6]}, optional metadata {layer, rows, seq, row_start}. The capture is ROW-SPLIT across nodes (10-way:
     row starts 0,39,...,351; earlier 2-way 0/192) -> up to 10 files per layer with the SAME file name in per-node dirs
     (/data/dsv41cap/hess_r0/, hess_r39/, ...). Pass every dir (or a glob) with --dumps; ALL matching files of a layer are merged
     (Hessians are additive, token counts add). Metadata is used only for sanity (layer match, duplicate row_start warning).
  2. bf16 expert weights `layers.N.ffn.experts.E.{w1,w3,w2}.weight` ([2304,5120], [2304,5120], [5120,2304]) by HTTP byte-range from the
     upscaled checkpoint (default http://STORE_HOST:8767/bf16, index from .../fp4), or a local dir with the same shard files.
  3. optional bit recipe (JSON or YAML): per (layer, "gu"|"down") -> K in 2..8. Default: uniform K = --bits. Formats accepted:
        {"default": {"gu": 3, "down": 4}, "layers": {"5": {"gu": 4, "down": 4}, "12": {"down": 5}}}
        {"5.gu": 4, "5.down": 4, "layers.12.down": 5}          (flat; unspecified -> default)
     cuda-exl3 forces one K for all experts of a layer and the same K for gate and up (moe.py:100-112), so this is the granularity.

Hessians (per layer, per expert e, from the rows routed to e):
  H_gu   = sum x x^T           fp32 [5120,5120]  — ONE H_data dict shared by w1 and w3 (exllamav3's own shared-H group path: one LDLQ
                                                    over gate|up concatenated along out_features, identical su)
  H_down = sum a a^T           fp32 [2304,2304]  — a = silu(min(w1 x, 10)) * clamp(w3 x, -10, 10) with the bf16 (unquantized) weights,
                                                    rounded to bf16 like the reference (Expert.forward l.841-851) unless --no-act-bf16.
           --w2-weighting none : a as above (matches cuda-exl3, which applies the routing weight in the down epilogue, moe.py:444-447)
           --w2-weighting route: a := route_w * a (matches the reference's pre-w2 multiply, l.849-851) -> H gets route_w^2 per token
  Both are SUMS (not means): finalize_capture_H divides by `count` (quantize.py:871), so count = tokens routed to the expert.

Outputs (per layer, in --out)
  exl3_L{LL}.safetensors : layers.N.ffn.experts.E.w{1,3,2}.{trellis,suh,svh,mul1} exactly as pack_trellis emits them
                           (trellis int16 (in/16, out/16, 16K); suh fp16 (in,); svh fp16 (out,); mul1 int32 scalar)
  exl3_L{LL}.json        : ledger — per tensor proxy_err, K, bpw, tokens, seconds, apply_out_scales, g_scale, q_fallback (+ rfn_out if
                           --verify-rows), per-layer totals; `lines` = flat list for the allocator (exl3_depth_recipe.py plan input)
  parts/                 : per-chunk part files (resume granularity = chunk of experts; the ledger is per expert); merged and deleted
                           when the layer's expert range is complete (--keep-parts keeps them)

Memory: never more than one chunk of expert Hessians + weights on the device (a 5120^2 fp32 H is 105 MB; 384 = 40 GB — hence
--chunk). x for the whole layer (384x2048 tokens = 8 GB bf16) is resident once; per-expert rows are gathered by a CSR built from
route_idx, so x is not re-streamed per chunk.

usage (one GB10, inside the vLLM image with exllamav3 installed):
  dsv41_exl3_experts.py --dumps '/data/dsv41cap/hess_r*' --layers 0-3 --bits 3 --out /data/dsv41exl3   # all 10 row shards merged
  dsv41_exl3_experts.py --dumps ... --layers 5 --recipe recipe.json --w2-weighting route --out ... --verify-rows 4096
  dsv41_exl3_experts.py --selftest            # CPU, tiny random matrices; skips the quantizer part if exllamav3/CUDA are absent
  dsv41_exl3_experts.py --dry-run --layers 0  # print the memory / time plan for the given chunking, no GPU work
"""
import argparse, json, math, os, struct, sys, time, urllib.request, zlib
import concurrent.futures as cf

D_MODEL, D_INTER, N_EXPERTS, TOPK = 5120, 2304, 384, 6
MUL1_MULT = 0x83DCD12D          # exllamav3 quantize.py:19 codebook_mul1_mult
MATS = ("w1", "w3", "w2")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dumps", nargs="*", default=[], help="dirs (one per capture node), globs, or files: every ffn_in_L##.safetensors found for the layer is merged")
ap.add_argument("--http", default="http://STORE_HOST:8767/bf16", help="bf16 shard source: http(s) base URL or a local dir")
ap.add_argument("--index-http", default="http://STORE_HOST:8767/fp4", help="where model.safetensors.index.json lives (same names)")
ap.add_argument("--layers", default="0-39"); ap.add_argument("--experts", default=f"0-{N_EXPERTS - 1}")
ap.add_argument("--chunk", type=int, default=48, help="experts per chunk (Hessians + bf16 weights of one chunk resident)")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--bits", type=int, default=3, help="uniform K (2..8) when no --recipe / for layers the recipe does not name")
ap.add_argument("--mats", default="w1,w3,w2", help="matrix types to quantize (subset of w1,w3,w2); e.g. --mats w2 cooks only the down projections (gu still fetched for the Hessian)")
ap.add_argument("--recipe", default="", help="JSON/YAML per-(layer, gu|down) K map (see module doc)")
ap.add_argument("--w2-weighting", default="none", choices=["none", "route"])
ap.add_argument("--no-act-bf16", action="store_true", help="keep the w2 input activations in fp32 (reference casts to bf16 before w2)")
ap.add_argument("--out", default="/data/dsv41exl3"); ap.add_argument("--parts", default="", help="part dir (default OUT/parts)")
ap.add_argument("--gu-mode", default="shared", choices=["shared", "batched"],
                help="shared: per expert one LDLQ over [w1|w3] with the shared H (exllamav3 semantics); batched: --gu-batch experts' w1+w3 stacked (ldlq_batched, distinct H)")
ap.add_argument("--gu-batch", type=int, default=4); ap.add_argument("--down-batch", type=int, default=16, help="experts' w2 stacked per ldlq_batched call")
ap.add_argument("--tok-chunk", type=int, default=65536, help="tokens per matmul step when building an expert's Hessians")
ap.add_argument("--sigma-reg", type=float, default=0.025, help="exllamav3 sigma_reg (H diag += sigma_reg * mean diag)")
ap.add_argument("--out-scales", default="always", choices=["always", "never", "auto"], help="quant_args apply_out_scales")
ap.add_argument("--verify-rows", type=int, default=0, help="reconstruct each quantized matrix and report ||X(Wq-W)||/||XW|| on up to N routed rows")
ap.add_argument("--seed-base", type=int, default=0, help="xor'ed into the per-tensor crc32 seed")
ap.add_argument("--min-tokens", type=int, default=D_MODEL // 4, help="flag experts with fewer routed tokens in the ledger (low_tokens)")
ap.add_argument("--tf32", action="store_true", help="allow TF32 in the Hessian matmuls (default: strict fp32)")
ap.add_argument("--keep-parts", action="store_true"); ap.add_argument("--merge-only", action="store_true", help="only merge existing parts into exl3_L##")
ap.add_argument("--prefetch", type=int, default=8, help="parallel weight fetches")
ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--selftest", action="store_true")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_range(s, lo_default=0, hi_default=0):
    """'3', '3-7', '0-3,8,10-11' -> sorted list of ints."""
    out = set()
    for part in str(s).split(","):
        part = part.strip()
        if not part: continue
        a, _, b = part.partition("-"); out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


# ---------------------------------------------------------------------------------------------------------------------
# recipe
# ---------------------------------------------------------------------------------------------------------------------
def load_recipe(path, default_bits):
    """Returns k_for(layer, kind) with kind in {"gu","down"}; K validated to 2..8."""
    table = {}
    dflt = {"gu": int(default_bits), "down": int(default_bits)}
    if path:
        txt = open(path).read()
        try: raw = json.loads(txt)
        except json.JSONDecodeError:
            try: import yaml
            except ImportError: raise SystemExit(f"recipe {path}: not JSON and PyYAML not importable")
            raw = yaml.safe_load(txt)
        if not isinstance(raw, dict): raise SystemExit(f"recipe {path}: top level must be a mapping")
        if "default" in raw and isinstance(raw["default"], dict): dflt.update({k: int(v) for k, v in raw["default"].items() if k in dflt})
        for key, val in (raw.get("layers") or {}).items():
            table[int(key)] = {k: int(v) for k, v in dict(val).items()}
        for key, val in raw.items():
            if key in ("default", "layers"): continue
            k = str(key)
            if k.startswith("layers."): k = k[len("layers."):]
            L, _, kind = k.partition(".")
            if not L.isdigit(): continue          # allocator metadata keys
            if kind not in ("gu", "down"): raise SystemExit(f"recipe {path}: bad key {key!r} (want <layer>.gu|down)")
            table.setdefault(int(L), {})[kind] = int(val)
    def k_for(layer, kind):
        K = table.get(int(layer), {}).get(kind, dflt[kind])
        if not 2 <= K <= 8: raise SystemExit(f"recipe: K={K} for layer {layer} {kind} outside 2..8")
        return K
    return k_for


def tensor_seed(name, base=0):
    return (zlib.crc32(name.encode()) & 0x7FFFFFFF) ^ (int(base) & 0x7FFFFFFF)


# ---------------------------------------------------------------------------------------------------------------------
# weight fetch (byte-range HTTP or local dir), same header/offset logic as dsv41_exl3_probe.py
# ---------------------------------------------------------------------------------------------------------------------
class Fetcher:
    def __init__(self, base, index_base, retries=4):
        self.base, self.index_base, self.retries = base.rstrip("/"), index_base.rstrip("/"), retries
        self.local = not base.startswith(("http://", "https://")); self.hcache = {}; self.wm = None

    def _read(self, path, start=None, end=None):
        """bytes [start, end] inclusive (HTTP Range semantics) from shard `path` (relative name)."""
        for attempt in range(self.retries):
            try:
                if self.local:
                    with open(os.path.join(self.base, path), "rb") as f:
                        if start is None: return f.read()
                        f.seek(start); return f.read(end - start + 1)
                hdr = {} if start is None else {"Range": f"bytes={start}-{end}"}
                r = urllib.request.Request(f"{self.base}/{path}", headers=hdr); buf = bytearray()
                with urllib.request.urlopen(r, timeout=600) as rr:
                    while True:
                        b = rr.read(32 << 20)
                        if not b: break
                        buf += b
                if start is not None and len(buf) != end - start + 1: raise IOError(f"short read {len(buf)} != {end - start + 1} for {path}")
                return bytes(buf)
            except Exception as e:
                if attempt == self.retries - 1: raise
                log(f"  fetch retry {attempt + 1} {path} [{start}-{end}]: {type(e).__name__}: {str(e)[:80]}"); time.sleep(2 * (attempt + 1))

    def weight_map(self):
        if self.wm is None:
            if self.index_base.startswith(("http://", "https://")):
                self.wm = json.load(urllib.request.urlopen(f"{self.index_base}/model.safetensors.index.json", timeout=120))["weight_map"]
            else: self.wm = json.load(open(os.path.join(self.index_base, "model.safetensors.index.json")))["weight_map"]
        return self.wm

    def header(self, shard):
        if shard not in self.hcache:
            n = struct.unpack("<Q", self._read(shard, 0, 7))[0]
            self.hcache[shard] = (json.loads(self._read(shard, 8, 8 + n - 1)), 8 + n)
        return self.hcache[shard]

    def get(self, key, torch):
        """bf16 tensor [out, in] on CPU."""
        shard = self.weight_map()[key]; h, base = self.header(shard); info = h[key]
        if info["dtype"] != "BF16": raise ValueError(f"{key}: dtype {info['dtype']} (need the bf16 upscale, not the fp4 dir)")
        s, e = info["data_offsets"]; buf = bytearray(self._read(shard, base + s, base + e - 1))
        return torch.frombuffer(buf, dtype=torch.bfloat16).reshape(info["shape"]).clone()

    def prefetch_experts(self, layer, experts, torch, workers=8):
        """Submit the chunk's 3*len(experts) fetches to one persistent pool; returns {(e, mat): Future}. Call gather() to consume."""
        if not hasattr(self, "pool"): self.pool = cf.ThreadPoolExecutor(max_workers=workers)
        return {(e, m): self.pool.submit(self.get, f"layers.{layer}.ffn.experts.{e}.{m}.weight", torch) for e in experts for m in ("w1", "w3", "w2")}   # gu always needed for the down Hessian

    @staticmethod
    def gather(futs):
        return {k: f.result() for k, f in futs.items()}


# ---------------------------------------------------------------------------------------------------------------------
# dumps
# ---------------------------------------------------------------------------------------------------------------------
def find_dump_files(specs, layer):
    """Every ffn_in_L{LL}.safetensors reachable from the --dumps entries: dirs (searched non-recursively, then one level of subdirs so a
    parent like /data/dsv41cap holding hess_r0/ hess_r39/ ... works), shell globs (dirs or files), or explicit files."""
    import glob
    name = f"ffn_in_L{layer:02d}.safetensors"; files = []
    expanded = []
    for s in specs:
        hits = glob.glob(os.path.expanduser(s))
        if not hits: raise SystemExit(f"--dumps entry matches nothing: {s}")
        expanded += hits
    for s in expanded:
        if os.path.isdir(s):
            p = os.path.join(s, name)
            if os.path.exists(p): files.append(p)
            files += [q for q in glob.glob(os.path.join(s, "*", name)) if os.path.isfile(q)]
        elif os.path.isfile(s):
            if os.path.basename(s) == name: files.append(s)
            elif "ffn_in_L" not in os.path.basename(s): raise SystemExit(f"--dumps file is not a ffn_in dump: {s}")
    files = sorted({os.path.realpath(f) for f in files})
    return files


def load_dumps(files, layer, device, torch):
    """Merge all row-split dump files of a layer: x (bf16 [N, D] on device, preallocated once), route_idx (int64 [N, TOPK]),
    route_w (fp32 [N, TOPK]), per-file meta. Files are concatenated in (row_start if present, path) order; metadata is optional."""
    from safetensors import safe_open
    infos = []
    for p in files:
        with safe_open(p, "pt", device="cpu") as f:
            md = f.metadata() or {}; sl = f.get_slice("x"); shape = sl.get_shape(); dt = sl.get_dtype()
            if md.get("layer") not in (None, str(layer)): raise SystemExit(f"{p}: metadata layer {md.get('layer')} != {layer}")
            if dt != "BF16" or len(shape) != 2 or shape[1] != D_MODEL: raise SystemExit(f"{p}: x is {dt} {shape}, expected BF16 [N, {D_MODEL}]")
            rs = md.get("row_start"); infos.append((int(rs) if rs is not None else None, p, int(shape[0]), md))
    starts = [i[0] for i in infos if i[0] is not None]
    if len(starts) != len(set(starts)):
        dup = sorted({s for s in starts if starts.count(s) > 1}); log(f"  WARNING: dump files share row_start {dup} — duplicated rows would double-count tokens: {[i[1] for i in infos if i[0] in dup]}")
    infos.sort(key=lambda i: (i[0] if i[0] is not None else 1 << 40, i[1]))
    N = sum(i[2] for i in infos); x = torch.empty(N, D_MODEL, dtype=torch.bfloat16, device=device)
    ri = torch.empty(N, TOPK, dtype=torch.int64, device=device); rw = torch.empty(N, TOPK, dtype=torch.float32, device=device); metas = []; pos = 0
    for rs, p, n, md in infos:
        with safe_open(p, "pt", device="cpu") as f:
            xt = f.get_tensor("x"); rit = f.get_tensor("route_idx"); rwt = f.get_tensor("route_w")
        if rit.shape != (n, TOPK) or rwt.shape != (n, TOPK): raise SystemExit(f"{p}: route_idx {tuple(rit.shape)} / route_w {tuple(rwt.shape)} vs x rows {n}")
        x[pos:pos + n].copy_(xt); ri[pos:pos + n].copy_(rit.long()); rw[pos:pos + n].copy_(rwt.float()); pos += n
        metas.append({**md, "file": p, "tokens": n}); del xt, rit, rwt
    if ri.min() < 0 or ri.max() >= N_EXPERTS: raise SystemExit(f"route_idx out of range [{ri.min()}, {ri.max()}]")
    return x, ri, rw, metas


class RoutingCSR:
    """token lists per expert from route_idx [N, TOPK] (flattened argsort -> CSR)."""
    def __init__(self, route_idx, route_w, torch):
        flat = route_idx.reshape(-1); order = torch.argsort(flat, stable=True)
        self.counts = torch.bincount(flat, minlength=N_EXPERTS); self.offsets = torch.zeros(N_EXPERTS + 1, dtype=torch.long, device=flat.device)
        self.offsets[1:] = torch.cumsum(self.counts, 0)
        self.tok = order // TOPK; self.w = route_w.reshape(-1)[order]
    def rows(self, e):
        a, b = int(self.offsets[e]), int(self.offsets[e + 1]); return self.tok[a:b], self.w[a:b]


# ---------------------------------------------------------------------------------------------------------------------
# Hessians
# ---------------------------------------------------------------------------------------------------------------------
def act_down_input(xf, w1f, w3f, wrow=None, round_bf16=True, torch=None, limit=10.0):
    """a = silu(min(x w1^T, limit)) * clamp(x w3^T, -limit, limit) [* route_w] -> bf16-rounded fp32 (reference Expert.forward l.841-851)."""
    import torch.nn.functional as F
    gate = (xf @ w1f.T).clamp_(max=limit); up = (xf @ w3f.T).clamp_(-limit, limit)
    a = F.silu(gate) * up
    if wrow is not None: a = a * wrow[:, None]
    if round_bf16: a = a.to(torch.bfloat16).float()
    return a


def build_expert_hessians(x, csr, e, w1, w3, torch, w2_weighting="none", tok_chunk=65536, round_bf16=True, keep_rows=0):
    """H_gu [D,D], H_down [I,I] fp32 sums on x.device, n_tok; optionally the first keep_rows (x rows, a rows) for verification."""
    dev = x.device; tok, wr = csr.rows(e); n = int(tok.numel())
    H_gu = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float32, device=dev); H_dn = torch.zeros(D_INTER, D_INTER, dtype=torch.float32, device=dev)
    w1f, w3f = w1.to(dev).float(), w3.to(dev).float(); Xk, Ak = [], []
    for s in range(0, n, tok_chunk):
        xf = x[tok[s:s + tok_chunk]].float()
        H_gu.addmm_(xf.T, xf)
        a = act_down_input(xf, w1f, w3f, wr[s:s + tok_chunk] if w2_weighting == "route" else None, round_bf16, torch)
        H_dn.addmm_(a.T, a)
        if keep_rows and sum(t.shape[0] for t in Xk) < keep_rows: Xk.append(xf[:keep_rows]); Ak.append(a[:keep_rows])
        del xf, a
    keep = (torch.cat(Xk)[:keep_rows], torch.cat(Ak)[:keep_rows]) if Xk else None
    return H_gu, H_dn, n, keep


def make_H_data(H, name, n_tok, device, torch):
    """Mirror of exllamav3 Linear.init_H_data (modules/linear.py) after capture: H is the SUM over rows, count = rows."""
    k = H.shape[0]
    return {"H": H, "first_key": name, "count": int(n_tok), "finalized": False, "num_total": int(n_tok) * k,
            "inf_nan": torch.zeros(2, dtype=torch.long, device=device), "device": torch.device(device)}


def fallback_H_data(k, name, device, torch):
    """count=0 -> finalize_capture_H takes the uncalibrated (meta) path: for experts that saw no tokens."""
    return {"H": torch.empty(k, k, device="meta"), "first_key": name, "count": 0, "finalized": False, "num_total": 0,
            "inf_nan": torch.zeros(2, dtype=torch.long, device=device), "device": torch.device(device)}


def make_quant_args(K, name, out_scales, sigma_reg, seed_base, device_index=0):
    return {"K": int(K), "seed": tensor_seed(name, seed_base), "devices": [int(device_index or 0)], "device_ratios": None,
            "apply_out_scales": {"always": True, "never": False, "auto": None}[out_scales], "mul1": True, "sigma_reg": sigma_reg}


def bpw_of(out_tensors, numel):
    return sum(t.numel() * t.element_size() for t in out_tensors.values()) * 8 / numel


def reconstruct(out_tensors, torch):
    """Dequantize (in, out) fp32 from trellis/suh/svh (exl3_kernel_truth.py recipe: ext.reconstruct + both Hadamards + sign/scale vectors)."""
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant.exl3_lib.quantize import preapply_had_l, preapply_had_r, had_k, had_n
    tr = out_tensors["trellis"]; k, n, K = tr.shape[0] * 16, tr.shape[1] * 16, tr.shape[2] // 16
    w = torch.empty((k, n), dtype=torch.half, device=tr.device)
    ext.reconstruct(w, tr.contiguous(), K, False, True)
    w = preapply_had_l(w, had_k); w *= out_tensors["suh"].to(w.device).unsqueeze(1); w = preapply_had_r(w, had_n); w *= out_tensors["svh"].to(w.device).unsqueeze(0)
    return w.float()


def rfn_out(X, W_t, out_tensors, torch):
    """||X (Wq - W)|| / ||X W|| with W_t (in, out) fp32 on device."""
    Wq = reconstruct(out_tensors, torch).to(W_t.device); ref = X @ W_t
    return ((X @ (Wq - W_t)).norm() / (ref.norm() + 1e-12)).item()


# ---------------------------------------------------------------------------------------------------------------------
# per-chunk quantization
# ---------------------------------------------------------------------------------------------------------------------
def quantize_chunk(layer, experts, x, csr, weights, k_for, a, torch, dev):
    """experts: list of expert ids; weights: {(e, mat): bf16 [out,in] CPU}. Returns (tensors {name: cpu tensor}, ledger {e: {...}})."""
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3, quantize_exl3_batch
    K_gu, K_dn = k_for(layer, "gu"), k_for(layer, "down"); tensors, ledger = {}, {}
    round_bf16 = not a.no_act_bf16; keep = a.verify_rows
    pending_down = []     # (e, name, W_t (in,out) fp32 CPU, H_data, quant_args, verify rows)
    pending_gu = []       # batched gu mode: (e, w1 name, w3 name, W1_t, W3_t, H_data, qa1, qa3, X rows)
    t_h = 0.0

    def finish_tensor(e, mat, name, res, qa, W_t, X, t0, n_tok):
        proxy_err, out = res
        out = {k: v.detach().cpu() for k, v in out.items()}
        for k, v in out.items(): tensors[f"{name}.{k}"] = v
        rec = {"K": qa["K"], "proxy_err": float(proxy_err), "bpw": bpw_of(out, W_t.numel()), "secs": round(time.time() - t0, 2),
               "apply_out_scales": bool(qa.get("apply_out_scales")), "g_scale": float(qa.get("g_scale", 0.0)), "q_fallback": bool(qa.get("q_fallback", False)),
               "trellis_shape": list(out["trellis"].shape)}
        if X is not None and not rec["q_fallback"]:
            try: rec["rfn_out"] = rfn_out(X.to(dev), W_t.to(dev).float(), {k: v.to(dev) for k, v in out.items()}, torch)
            except Exception as ex: rec["rfn_out_error"] = f"{type(ex).__name__}: {str(ex)[:80]}"
        ledger.setdefault(e, {})[mat] = rec
        log(f"    L{layer:02d} e{e:03d} {mat} K={qa['K']} proxy {float(proxy_err):.5f} bpw {rec['bpw']:.2f} tok {n_tok}"
            + (f" rfn_out {rec['rfn_out']:.4f}" if "rfn_out" in rec else "") + f" {rec['secs']:.0f}s")

    def flush_down():
        nonlocal pending_down
        if not pending_down: return
        t0 = time.time()
        res = quantize_exl3_batch([p[2] for p in pending_down], [p[3] for p in pending_down], [p[4] for p in pending_down]) \
            if len(pending_down) > 1 else [quantize_exl3(pending_down[0][2].to(dev).float(), pending_down[0][3], pending_down[0][4], False)[1:]]
        for (e, name, W_t, Hd, qa, A, n_tok), r in zip(pending_down, res): finish_tensor(e, "w2", name, r, qa, W_t, A, t0, n_tok)
        pending_down = []; torch.cuda.empty_cache() if dev.type == "cuda" else None

    def flush_gu():
        nonlocal pending_gu
        if not pending_gu: return
        t0 = time.time(); ws, hs, qs = [], [], []
        for (e, n1, n3, W1t, W3t, Hd, qa1, qa3, X, n_tok) in pending_gu: ws += [W1t, W3t]; hs += [Hd, Hd]; qs += [qa1, qa3]
        res = quantize_exl3_batch(ws, hs, qs)
        for i, (e, n1, n3, W1t, W3t, Hd, qa1, qa3, X, n_tok) in enumerate(pending_gu):
            finish_tensor(e, "w1", n1, res[2 * i], qa1, W1t, X, t0, n_tok); finish_tensor(e, "w3", n3, res[2 * i + 1], qa3, W3t, X, t0, n_tok)
        pending_gu = []; torch.cuda.empty_cache() if dev.type == "cuda" else None

    for e in experts:
        t0 = time.time(); w1, w3, w2 = weights[(e, "w1")], weights[(e, "w3")], weights[(e, "w2")]
        H_gu, H_dn, n_tok, kept = build_expert_hessians(x, csr, e, w1, w3, torch, a.w2_weighting, a.tok_chunk, round_bf16, keep)
        t_h += time.time() - t0
        names = {m: f"layers.{layer}.ffn.experts.{e}.{m}" for m in ("w1", "w3", "w2")}
        ledger.setdefault(e, {})["tokens"] = n_tok
        if n_tok < a.min_tokens: ledger[e]["low_tokens"] = True
        if n_tok == 0:
            Hd_gu = fallback_H_data(D_MODEL, names["w1"], dev, torch); Hd_dn = fallback_H_data(D_INTER, names["w2"], dev, torch); log(f"    L{layer:02d} e{e:03d}: NO routed tokens -> uncalibrated fallback")
        else:
            Hd_gu = make_H_data(H_gu, names["w1"], n_tok, dev, torch); Hd_dn = make_H_data(H_dn, names["w2"], n_tok, dev, torch)
        X_rows, A_rows = (kept if kept is not None else (None, None))
        # exllamav3 wants (in_features, out_features) = transpose of the checkpoint's [out, in]; kept bf16 on the CPU for the batch path
        # (its _WeightStager uploads and casts to fp32 on the device), cast to fp32 only where quantize_exl3 asserts it
        W1t, W3t, W2t = (w.T.contiguous() for w in (w1, w3, w2))
        di = dev.index if dev.type == "cuda" else 0
        qa1, qa3 = (make_quant_args(K_gu, names[m], a.out_scales, a.sigma_reg, a.seed_base, di) for m in ("w1", "w3"))
        qa2 = make_quant_args(K_dn, names["w2"], a.out_scales, a.sigma_reg, a.seed_base, di)
        if "w1" not in MATS: pass                       # --mats w2: down-only pass
        elif a.gu_mode == "shared" or n_tok == 0:
            t1 = time.time()
            res = quantize_exl3_batch([W1t, W3t], [Hd_gu, Hd_gu], [qa1, qa3]) if n_tok else \
                [quantize_exl3(W1t.to(dev).float(), Hd_gu, qa1, False)[1:], quantize_exl3(W3t.to(dev).float(), Hd_gu, qa3, False)[1:]]
            finish_tensor(e, "w1", names["w1"], res[0], qa1, W1t, X_rows, t1, n_tok); finish_tensor(e, "w3", names["w3"], res[1], qa3, W3t, X_rows, t1, n_tok)
            Hd_gu.pop("dev_cache", None); Hd_gu["H"] = None; Hd_gu["L"] = None
        else:
            pending_gu.append((e, names["w1"], names["w3"], W1t, W3t, Hd_gu, qa1, qa3, X_rows, n_tok))
            if len(pending_gu) >= a.gu_batch: flush_gu()
        if "w2" not in MATS: pass                       # --mats w1,w3: gu-only pass
        elif n_tok == 0:
            t1 = time.time(); finish_tensor(e, "w2", names["w2"], quantize_exl3(W2t.to(dev).float(), Hd_dn, qa2, False)[1:], qa2, W2t, None, t1, 0)
        else:
            pending_down.append((e, names["w2"], W2t, Hd_dn, qa2, A_rows, n_tok))
            if len(pending_down) >= a.down_batch: flush_down()
        del H_gu, H_dn
    flush_gu(); flush_down()
    for e in experts: ledger[e]["hessian_secs_share"] = round(t_h / max(len(experts), 1), 2)
    return tensors, ledger


# ---------------------------------------------------------------------------------------------------------------------
# parts / ledger / merge
# ---------------------------------------------------------------------------------------------------------------------
def part_paths(parts_dir, layer, experts):
    stem = os.path.join(parts_dir, f"exl3_L{layer:02d}_e{experts[0]:03d}-{experts[-1]:03d}")
    return stem + ".safetensors", stem + ".json"


def done_experts(parts_dir, layer):
    """experts with a finished part (ledger json + tensor file present and header lists all 12 tensors per expert)."""
    from safetensors import safe_open
    done = {}
    for fn in sorted(os.listdir(parts_dir)) if os.path.isdir(parts_dir) else []:
        if not (fn.startswith(f"exl3_L{layer:02d}_e") and fn.endswith(".json")): continue
        jp = os.path.join(parts_dir, fn); sp = jp[:-5] + ".safetensors"
        if not os.path.exists(sp): continue
        led = json.load(open(jp))
        try:
            with safe_open(sp, "pt", device="cpu") as f: keys = set(f.keys())
        except Exception as ex: log(f"  ignoring unreadable part {sp}: {ex}"); continue
        for e_s, rec in led["experts"].items():
            e = int(e_s)
            if all(f"layers.{layer}.ffn.experts.{e}.{m}.{t}" in keys for m in MATS for t in ("trellis", "suh", "svh", "mul1")): done[e] = (sp, rec)
    return done


def ledger_lines(layer, experts_led):
    return [{"name": f"layers.{layer}.ffn.experts.{e}.{m}", "layer": layer, "expert": e, "mat": m, "kind": "down" if m == "w2" else "gu", **{k: v for k, v in rec[m].items()}, "tokens": rec["tokens"]}
            for e, rec in sorted(experts_led.items()) for m in MATS if m in rec]


def merge_layer(parts_dir, out_dir, layer, expected, a, meta=None):
    """Concatenate all parts of a layer into exl3_L##.safetensors + exl3_L##.json when every expected expert is present."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    done = done_experts(parts_dir, layer); missing = [e for e in expected if e not in done]
    if missing: log(f"L{layer:02d}: {len(missing)} experts not done yet (e.g. {missing[:6]}) — not merging"); return False
    tensors, led = {}, {}; files = sorted({sp for sp, _ in done.values()})
    for sp in files:
        with safe_open(sp, "pt", device="cpu") as f:
            for k in f.keys():
                e = int(k.split(".experts.")[1].split(".")[0])
                if e in expected: tensors[k] = f.get_tensor(k)
    for e in expected: led[e] = done[e][1]
    lines = ledger_lines(layer, led); import statistics
    summ = {}
    for m in MATS:
        v = [r["proxy_err"] for r in lines if r["mat"] == m and r["proxy_err"] >= 0]
        Km = sorted({r["K"] for r in lines if r["mat"] == m})
        summ[m] = {"K": Km[0] if len(Km) == 1 else Km, "proxy_err_mean": statistics.fmean(v) if v else None, "proxy_err_max": max(v) if v else None,
                   "bpw_mean": statistics.fmean([r["bpw"] for r in lines if r["mat"] == m]) if lines else None, "n": len(v)}
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values()); numel = len(expected) * 3 * D_MODEL * D_INTER
    out = {"layer": layer, "experts": {str(e): led[e] for e in expected}, "summary": summ, "bpw_layer": total_bytes * 8 / numel, "bytes": total_bytes,
           "w2_weighting": a.w2_weighting, "gu_mode": a.gu_mode, "sigma_reg": a.sigma_reg, "out_scales": a.out_scales, "act_bf16": not a.no_act_bf16,
           "seed_base": a.seed_base, "dumps": meta, "low_tokens": [e for e in expected if led[e].get("low_tokens")], "lines": lines,
           "tensor_naming": "layers.N.ffn.experts.E.w{1,3,2}.{trellis,suh,svh,mul1}", "mul1_multiplier": MUL1_MULT}
    sp = os.path.join(out_dir, f"exl3_L{layer:02d}.safetensors"); tmp = sp + ".tmp"
    save_file(tensors, tmp, metadata={"layer": str(layer), "format": "pt", "producer": "dsv41_exl3_experts.py"}); os.replace(tmp, sp)
    json.dump(out, open(os.path.join(out_dir, f"exl3_L{layer:02d}.json"), "w"), indent=1)
    if not a.keep_parts:
        for f_ in files: os.remove(f_); os.remove(f_[:-len(".safetensors")] + ".json")
    log(f"L{layer:02d} MERGED {len(expected)} experts -> {sp} ({total_bytes / 1e9:.2f} GB, {out['bpw_layer']:.3f} bpw) "
        + " ".join(f"{m}:{summ[m]['proxy_err_mean']:.5f}" for m in MATS if summ[m]["proxy_err_mean"] is not None))
    return True


# ---------------------------------------------------------------------------------------------------------------------
# plan / estimates
# ---------------------------------------------------------------------------------------------------------------------
def plan(a, layers, experts):
    n_tok = 384 * 2048; per_e = n_tok * TOPK / N_EXPERTS
    x_gb = n_tok * D_MODEL * 2 / 1e9; h_chunk = a.chunk * (D_MODEL ** 2 + D_INTER ** 2) * 4 / 1e9; w_chunk = a.chunk * 3 * D_MODEL * D_INTER * 2 / 1e9
    ldlq = (D_MODEL * 2 * D_INTER * 4 + D_MODEL ** 2 * 4 * 2 + a.down_batch * (D_INTER * D_MODEL * 4 + D_INTER ** 2 * 4)) / 1e9
    out_layer = 3 * D_MODEL * D_INTER * len(experts) * a.bits / 8 / 1e9
    flops = len(experts) * per_e * (D_MODEL ** 2 + 2 * D_MODEL * D_INTER + D_INTER ** 2) * 2 / 1e12
    print(f"plan: layers {layers[0]}-{layers[-1]} ({len(layers)}), experts {experts[0]}-{experts[-1]} ({len(experts)}), chunk {a.chunk} -> {math.ceil(len(experts) / a.chunk)} chunks/layer")
    print(f"  device resident: x {x_gb:.1f} GB (all row shards) + Hessians <= 1 H_gu + {a.down_batch} H_down ({(D_MODEL ** 2 + a.down_batch * D_INTER ** 2) * 4 / 1e9:.1f} GB; chunk-of-{a.chunk} upper bound {h_chunk:.1f} GB) + chunk bf16 weights {w_chunk:.1f} GB (+{w_chunk:.1f} GB prefetch) + LDLQ work ~{ldlq:.1f} GB"
          f" + part tensors (CPU) {out_layer / math.ceil(len(experts) / a.chunk):.2f} GB  => peak ~{x_gb + h_chunk + 2 * w_chunk + ldlq + 3:.0f} GB")
    print(f"  Hessian build: ~{flops:.0f} TFLOP fp32/layer (~{flops / 25:.0f}-{flops / 12:.0f} s on GB10 at 12-25 TFLOPS fp32); quantization: same LDLQ as exllamav3 => ~60-75 min/layer at 384 experts (GLM-5.3 ref 42-52 min for 9.7B/layer)")
    print(f"  output: ~{out_layer:.2f} GB/layer at K={a.bits} ({len(layers)} layers -> {out_layer * len(layers):.1f} GB); dumps needed per layer: {x_gb:.1f} GB")


# ---------------------------------------------------------------------------------------------------------------------
# selftest (CPU; the quantizer stage only if exllamav3 + CUDA are importable)
# ---------------------------------------------------------------------------------------------------------------------
def selftest(a):
    try: import torch
    except ImportError: print("SELFTEST-SKIP: torch not importable on this machine"); return 0
    import tempfile
    torch.manual_seed(0); dev = torch.device("cpu"); ok = True
    def check(cond, msg):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + msg); ok &= bool(cond)
    # 1. recipe
    with tempfile.TemporaryDirectory() as td:
        rp = os.path.join(td, "r.json"); json.dump({"default": {"gu": 3, "down": 4}, "layers": {"5": {"gu": 4}}, "7.down": 5, "layers.9.gu": 2}, open(rp, "w"))
        kf = load_recipe(rp, 3)
        check(kf(0, "gu") == 3 and kf(0, "down") == 4 and kf(5, "gu") == 4 and kf(5, "down") == 4 and kf(7, "down") == 5 and kf(9, "gu") == 2, "recipe: default/layers/flat forms")
        check(load_recipe("", 6)(3, "down") == 6, "recipe: uniform default")
        check(tensor_seed("layers.3.ffn.experts.0.w1") != tensor_seed("layers.3.ffn.experts.0.w3") and tensor_seed("a") == tensor_seed("a"), "seed: deterministic, distinct per tensor")
    # 2. routing CSR vs mask gather, on tiny random data
    N, E = 3000, N_EXPERTS
    ri = torch.stack([torch.randperm(E)[:TOPK] for _ in range(N)]); rw = torch.rand(N, TOPK)
    csr = RoutingCSR(ri, rw, torch)
    e = int(ri[0, 0]); tok, w = csr.rows(e); mask = (ri == e).any(1)
    check(torch.equal(tok.sort().values, mask.nonzero().flatten()) and int(csr.counts.sum()) == N * TOPK, "CSR rows == mask rows; counts sum")
    wref = rw[ri == e]; check(torch.allclose(w.sort().values, wref.sort().values), "CSR route_w aligned with tokens")
    # 3. Hessians: CSR/chunked build == direct formula (small dims via monkeypatched globals)
    global D_MODEL, D_INTER
    D0, I0 = D_MODEL, D_INTER; D_MODEL, D_INTER = 256, 128
    try:
        x = torch.randn(N, D_MODEL, dtype=torch.bfloat16); w1 = (torch.randn(D_INTER, D_MODEL) * 0.05).bfloat16(); w3 = (torch.randn(D_INTER, D_MODEL) * 0.05).bfloat16()
        H_gu, H_dn, n_tok, kept = build_expert_hessians(x, csr, e, w1, w3, torch, "none", tok_chunk=7, round_bf16=False, keep_rows=5)
        xf = x[mask].float(); ref_gu = xf.T @ xf
        g = (xf @ w1.float().T).clamp(max=10); u = (xf @ w3.float().T).clamp(-10, 10); aa = torch.nn.functional.silu(g) * u; ref_dn = aa.T @ aa
        check(n_tok == int(mask.sum()) and torch.allclose(H_gu, ref_gu, rtol=1e-4, atol=1e-3), "H_gu == sum x x^T over routed rows (chunked)")
        check(torch.allclose(H_dn, ref_dn, rtol=1e-4, atol=1e-3), "H_down == sum a a^T, a = silu(min(g,10))*clamp(u,±10)")
        check(kept is not None and kept[0].shape == (5, D_MODEL) and kept[1].shape == (5, D_INTER), "verification rows kept")
        H_gu_w, H_dn_w, _, _ = build_expert_hessians(x, csr, e, w1, w3, torch, "route", tok_chunk=1000, round_bf16=False)
        ww = rw[ri == e]; order = csr.rows(e)[0]  # rows are in argsort order; recompute reference with matching weights
        xw = x[order].float(); gw = (xw @ w1.float().T).clamp(max=10); uw = (xw @ w3.float().T).clamp(-10, 10); aw = torch.nn.functional.silu(gw) * uw * csr.rows(e)[1][:, None]
        check(torch.allclose(H_gu_w, ref_gu, rtol=1e-4, atol=1e-3) and torch.allclose(H_dn_w, aw.T @ aw, rtol=1e-4, atol=1e-3), "--w2-weighting route: H_gu unchanged, H_down gets route_w^2")
        hd = make_H_data(H_gu, "t", n_tok, dev, torch)
        check(set(hd) == {"H", "first_key", "count", "finalized", "num_total", "inf_nan", "device"} and hd["count"] == n_tok and hd["num_total"] == n_tok * D_MODEL, "H_data keys mirror exllamav3 init_H_data")
        fb = fallback_H_data(D_MODEL, "t", dev, torch); check(fb["H"].is_meta and fb["count"] == 0, "fallback H_data is meta/count 0")
        qa = make_quant_args(3, "layers.0.ffn.experts.0.w1", "always", 0.025, 0)
        check(qa["K"] == 3 and qa["apply_out_scales"] is True and qa["mul1"] is True and qa["devices"] == [0], "quant_args shape")
        # 4. ledger / part / merge round-trip with fake out_tensors of the right shapes
        with tempfile.TemporaryDirectory() as td:
            from safetensors.torch import save_file
            parts = os.path.join(td, "parts"); os.makedirs(parts); layer, exps, K = 3, [0, 1, 2], 3
            class A: pass
            aa_ = A(); aa_.w2_weighting = "none"; aa_.gu_mode = "shared"; aa_.sigma_reg = 0.025; aa_.out_scales = "always"; aa_.no_act_bf16 = False; aa_.seed_base = 0; aa_.keep_parts = False
            for grp in ([0, 1], [2]):
                T, L = {}, {"layer": layer, "experts": {}}
                for ee in grp:
                    L["experts"][str(ee)] = {"tokens": 10}
                    for m in MATS:
                        k_, n_ = (D_MODEL, D_INTER) if m != "w2" else (D_INTER, D_MODEL); nm = f"layers.{layer}.ffn.experts.{ee}.{m}"
                        out = {"trellis": torch.zeros(k_ // 16, n_ // 16, 16 * K, dtype=torch.int16), "suh": torch.ones(k_, dtype=torch.half), "svh": torch.ones(n_, dtype=torch.half),
                               "mul1": torch.tensor(MUL1_MULT, dtype=torch.uint32).view(torch.int32)}
                        for t_, v_ in out.items(): T[f"{nm}.{t_}"] = v_
                        L["experts"][str(ee)][m] = {"K": K, "proxy_err": 0.01, "bpw": bpw_of(out, k_ * n_), "secs": 1.0, "apply_out_scales": True, "g_scale": 1.0, "q_fallback": False, "trellis_shape": list(out["trellis"].shape)}
                sp, jp = part_paths(parts, layer, grp); save_file(T, sp); json.dump(L, open(jp, "w"))
            d = done_experts(parts, layer); check(sorted(d) == exps, "done_experts finds all experts in parts")
            check(merge_layer(parts, td, layer, exps, aa_), "merge_layer merges when complete")
            from safetensors import safe_open
            with safe_open(os.path.join(td, "exl3_L03.safetensors"), "pt") as f: keys = list(f.keys())
            led = json.load(open(os.path.join(td, "exl3_L03.json")))
            check(len(keys) == len(exps) * 12 and len(led["lines"]) == len(exps) * 3 and abs(led["bpw_layer"] - K) < 0.3, "merged file: 12 tensors/expert, ledger lines, bpw ~K")
            check(not os.listdir(parts), "parts removed after merge")
    finally:
        D_MODEL, D_INTER = D0, I0
    # 5. the quantizer itself (needs exllamav3 + its CUDA extension)
    try:
        import exllamav3  # noqa: F401
        from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3, quantize_exl3_batch  # noqa: F401
        have_exl3 = True
    except Exception as ex:
        have_exl3 = False; print(f"  skip quantizer stage: exllamav3 not importable ({type(ex).__name__}: {str(ex)[:60]})")
    if have_exl3 and torch.cuda.is_available():
        try:
            cd = torch.device("cuda:0"); k_, n_ = 256, 128; Xq = torch.randn(4000, k_, device=cd); W = torch.randn(k_, n_, device=cd) * 0.05
            hd = make_H_data(Xq.T @ Xq, "st.w1", 4000, cd, torch); qa1 = make_quant_args(3, "st.w1", "always", 0.025, 0); qa3 = make_quant_args(3, "st.w3", "always", 0.025, 0)
            res = quantize_exl3_batch([W.cpu(), (W * 1.1).cpu()], [hd, hd], [qa1, qa3])
            (pe, out) = res[0]
            check(out["trellis"].shape == (k_ // 16, n_ // 16, 48) and out["suh"].shape == (k_,) and out["svh"].shape == (n_,) and int(out["mul1"].view(torch.uint32)) == MUL1_MULT, "quantize_exl3_batch shared-H: trellis (in/16,out/16,16K), suh, svh, mul1")
            check(0 <= pe < 0.5, f"proxy_err sane ({pe:.4f})")
            r = rfn_out(Xq[:64], W, {k: v.to(cd) for k, v in out.items()}, torch); check(r < 0.3, f"reconstruct path rfn_out {r:.4f}")
        except Exception as ex:
            check(False, f"quantizer stage raised {type(ex).__name__}: {str(ex)[:120]}")
    elif have_exl3: print("  skip quantizer stage: no CUDA device")
    print("SELFTEST-OK" if ok else "SELFTEST-FAIL"); return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------------
def main():
    global MATS
    a = ap.parse_args()
    MATS = tuple(x for x in ("w1", "w3", "w2") if x in {t.strip() for t in a.mats.split(",")})
    if not MATS or (("w1" in MATS) != ("w3" in MATS)): raise SystemExit("--mats: choose w2, w1,w3 or w1,w3,w2 (gate/up share one Hessian and are cooked together)")
    if a.selftest: sys.exit(selftest(a))
    layers = parse_range(a.layers); experts = parse_range(a.experts)
    if not layers or not experts or experts[0] < 0 or experts[-1] >= N_EXPERTS: raise SystemExit("bad --layers/--experts")
    if not 2 <= a.bits <= 8: raise SystemExit("--bits must be 2..8")
    if a.dry_run: plan(a, layers, experts); return
    import torch
    from safetensors.torch import save_file
    if not a.tf32: torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(a.device)
    if dev.type == "cuda": torch.cuda.set_device(dev)
    os.makedirs(a.out, exist_ok=True); parts_dir = a.parts or os.path.join(a.out, "parts"); os.makedirs(parts_dir, exist_ok=True)
    k_for = load_recipe(a.recipe, a.bits); fetch = Fetcher(a.http, a.index_http)
    plan(a, layers, experts)
    for layer in layers:
        t_layer = time.time(); final = os.path.join(a.out, f"exl3_L{layer:02d}.safetensors")
        if os.path.exists(final) and os.path.exists(final[:-len(".safetensors")] + ".json"):
            log(f"L{layer:02d}: {final} exists -> skip"); continue
        done = done_experts(parts_dir, layer); todo = [e for e in experts if e not in done]
        if a.merge_only: merge_layer(parts_dir, a.out, layer, experts, a); continue
        log(f"L{layer:02d}: K gu={k_for(layer, 'gu')} down={k_for(layer, 'down')} w2-weighting={a.w2_weighting}; {len(done)} experts done, {len(todo)} to do")
        if not todo: merge_layer(parts_dir, a.out, layer, experts, a); continue
        files = find_dump_files(a.dumps, layer)
        if not files: raise SystemExit(f"L{layer:02d}: no ffn_in_L{layer:02d}.safetensors under {a.dumps}")
        log(f"  L{layer:02d}: {len(files)} dump file(s): " + ", ".join(os.path.relpath(f, os.path.commonpath(files)) if len(files) > 1 else f for f in files))
        t0 = time.time(); x, ri, rw, meta = load_dumps(files, layer, dev, torch); csr = RoutingCSR(ri, rw, torch); del ri, rw
        log(f"  dumps loaded: {x.shape[0]} tokens from {len(files)} file(s) in {time.time() - t0:.0f}s; routed tokens/expert min {int(csr.counts.min())} med {int(csr.counts.median())} max {int(csr.counts.max())}")
        chunks = [todo[i:i + a.chunk] for i in range(0, len(todo), a.chunk)]
        fut = fetch.prefetch_experts(layer, chunks[0], torch, a.prefetch)
        for ci, ch in enumerate(chunks):
            t0 = time.time(); weights = Fetcher.gather(fut); t_fetch = time.time() - t0
            if ci + 1 < len(chunks): fut = fetch.prefetch_experts(layer, chunks[ci + 1], torch, a.prefetch)
            log(f"  L{layer:02d} chunk {ci + 1}/{len(chunks)} experts {ch[0]}-{ch[-1]} (weights ready after {t_fetch:.0f}s)")
            tensors, led = quantize_chunk(layer, ch, x, csr, weights, k_for, a, torch, dev)
            del weights
            sp, jp = part_paths(parts_dir, layer, ch)
            save_file(tensors, sp + ".tmp", metadata={"layer": str(layer), "format": "pt"}); os.replace(sp + ".tmp", sp)
            json.dump({"layer": layer, "experts": {str(e): led[e] for e in ch}, "secs": time.time() - t0}, open(jp, "w"))
            pe = [led[e][m]["proxy_err"] for e in ch for m in MATS if m in led[e] and led[e][m]["proxy_err"] >= 0]
            log(f"  L{layer:02d} chunk {ci + 1} done {time.time() - t0:.0f}s, mean proxy_err {sum(pe) / max(len(pe), 1):.5f} -> {sp}")
            if dev.type == "cuda": torch.cuda.empty_cache()
        del x, csr
        if dev.type == "cuda": torch.cuda.empty_cache()
        merge_layer(parts_dir, a.out, layer, experts, a, meta=[{k: v for k, v in m.items()} for m in meta])
        log(f"L{layer:02d} done in {(time.time() - t_layer) / 60:.1f} min")
    log("DSV41-EXL3-EXPERTS-DONE")


if __name__ == "__main__":
    main()
