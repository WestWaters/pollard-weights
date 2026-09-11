#!/usr/bin/env python3
"""Gate-0 for the EXL3 Pollard route on DeepSeek-V4.1-Flash: does an EXL3 trellis at 2-4 bpw reproduce QAT-FP4 expert weights
(a discrete 8-level-per-block grid) as well as it reproduces continuous bf16 weights? Runs exllamav3's real quantizer
(quantize_exl3, LDLQ + mul1 codebook) on real expert matrices from the exact bf16 upscale, with a synthetic Hessian (Gaussian inputs,
optionally with log-normal per-channel scales to mimic activation outliers) and a Gaussian CONTROL matrix of matching row statistics.
Reports per bit level: proxy_err (trace form), weight rfn ||Wq-W||/||W||, output rfn ||X(Wq-W)||/||XW||, and grid-exact fraction.
usage: dsv41_exl3_probe.py --http http://STORE_HOST:8767/bf16 --layer 3 --experts 0-7 --bits 2,2.5,3,3.25,3.5,4 --out DIR"""
import argparse, json, struct, urllib.request, time, math, os, torch
ap = argparse.ArgumentParser(); ap.add_argument("--http", default="http://STORE_HOST:8767/bf16"); ap.add_argument("--index-http", default="http://STORE_HOST:8767/fp4", help="where to read model.safetensors.index.json (the bf16 dir gets its index only when the upscale finishes; shard names/keys are identical)"); ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--experts", default="0-7"); ap.add_argument("--bits", default="2,2.5,3,3.25,3.5,4"); ap.add_argument("--rows", type=int, default=8192)
ap.add_argument("--skew", type=float, default=0.0, help="log-normal sigma for per-channel input scales (0 = isotropic)"); ap.add_argument("--out", default="/data/dsv41_probe")
ap.add_argument("--dense", action="store_true", help="also probe the layer's dense projections (wq_a, wq_b, wo_b, shared w1)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = torch.device("cuda:0")
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
def log(m): print(time.strftime("[%H:%M:%S]"), m, flush=True)
wm = json.load(urllib.request.urlopen(f"{a.index_http}/model.safetensors.index.json", timeout=120))["weight_map"]
hcache = {}
def header(shard):
    if shard in hcache: return hcache[shard]
    r = urllib.request.Request(f"{a.http}/{shard}", headers={"Range": "bytes=0-7"}); n = struct.unpack("<Q", urllib.request.urlopen(r, timeout=60).read())[0]
    r = urllib.request.Request(f"{a.http}/{shard}", headers={"Range": f"bytes=8-{8 + n - 1}"}); hcache[shard] = (json.loads(urllib.request.urlopen(r, timeout=60).read()), 8 + n); return hcache[shard]
def get(key):
    shard = wm[key]; h, base = header(shard); info = h[key]; s, e = info["data_offsets"]; assert info["dtype"] == "BF16", (key, info["dtype"])
    r = urllib.request.Request(f"{a.http}/{shard}", headers={"Range": f"bytes={base + s}-{base + e - 1}"}); buf = bytearray()
    with urllib.request.urlopen(r, timeout=600) as rr:
        while True:
            b = rr.read(32 << 20)
            if not b: break
            buf += b
    return torch.frombuffer(buf, dtype=torch.bfloat16).reshape(info["shape"]).clone()
def probe(name, W, bits, X):
    """W: [out, in] bf16. Returns list of dicts per bit level."""
    Wf = W.to(dev).float(); Wt = Wf.T.contiguous()  # exllamav3 wants (in_features, out_features)
    H = (X.T @ X); n = X.shape[0]; ref_out = X @ Wt
    rows = []
    for K in bits:
        H_data = {"H": H.clone(), "count": n, "device": dev, "finalized": False, "L": None, "diag": None, "q_fallback": False, "first_key": name}
        qa = {"seed": 0, "K": K, "devices": [0], "device_ratios": None, "apply_out_scales": True, "debug_dir": a.out, "mul1": True}
        t0 = time.time()
        try:
            Wq, proxy_err, out_tensors = quantize_exl3(Wt.clone(), H_data, qa, return_weight_q=True)
        except Exception as e:
            rows.append({"name": name, "K": K, "error": f"{type(e).__name__}: {str(e)[:120]}"}); log(f"  {name} K={K}: ERROR {e}"); continue
        Wq = Wq.to(dev).float(); d = Wq - Wt
        rfn_w = (d.norm() / Wt.norm()).item(); rfn_out = ((X @ d).norm() / ref_out.norm()).item()
        exact = (Wq.to(torch.bfloat16) == Wt.to(torch.bfloat16)).float().mean().item()
        bpw = sum(t.numel() * t.element_size() for t in out_tensors.values()) * 8 / Wt.numel()
        rows.append({"name": name, "K": K, "proxy_err": float(proxy_err), "rfn_weight": rfn_w, "rfn_output": rfn_out, "grid_exact_frac": exact, "bpw_stored": bpw, "secs": time.time() - t0})
        log(f"  {name:38s} K={K:<4} proxy {float(proxy_err):.5f} rfn_w {rfn_w:.4f} rfn_out {rfn_out:.4f} exact {exact:.3f} bpw {bpw:.2f} {time.time() - t0:.0f}s")
        del Wq, d, out_tensors; torch.cuda.empty_cache()
    return rows
bits = [int(b) for b in a.bits.split(",")]; lo, hi = [int(x) for x in a.experts.split("-")]
torch.manual_seed(0); results = {"layer": a.layer, "rows": a.rows, "skew": a.skew, "bits": bits, "results": []}
def make_X(k_in):
    X = torch.randn(a.rows, k_in, device=dev)
    if a.skew > 0: X = X * torch.exp(a.skew * torch.randn(k_in, device=dev))[None, :]
    return X
Xin = make_X(5120); Xmid = make_X(2304)
for e in range(lo, hi + 1):
    for mat, X in (("w1", Xin), ("w3", Xin), ("w2", Xmid)):
        key = f"layers.{a.layer}.ffn.experts.{e}.{mat}.weight"; W = get(key)
        log(f"{key} {tuple(W.shape)} absmax {W.float().abs().max():.4f} zero-frac {(W == 0).float().mean():.3f}")
        results["results"] += probe(key, W, bits, X)
        if e == lo:  # control: continuous Gaussian weights with the same per-row std (what the GLM bf16 cook saw)
            Wc = (torch.randn_like(W.float()) * W.float().std(dim=1, keepdim=True)).to(torch.bfloat16)
            results["results"] += probe(f"CONTROL-gaussian.{mat}", Wc, bits, X)
if a.dense:
    for mat in ("attn.wq_a", "attn.wq_b", "attn.wo_b", "ffn.shared_experts.w1"):
        key = f"layers.{a.layer}.{mat}.weight"
        if key not in wm: continue
        try: W = get(key)
        except AssertionError as ex: log(f"skip {key}: {ex}"); continue
        results["results"] += probe(key, W, bits, make_X(W.shape[1]))
json.dump(results, open(os.path.join(a.out, f"exl3_probe_L{a.layer}_skew{a.skew}.json"), "w"), indent=1)
# summary table: mean over experts per (matrix, K)
import collections; agg = collections.defaultdict(list)
for r in results["results"]:
    if "error" in r: continue
    kind = r["name"].split(".")[-2] if "CONTROL" not in r["name"] else r["name"]
    agg[(kind, r["K"])].append((r["rfn_weight"], r["rfn_output"], r["proxy_err"], r["grid_exact_frac"]))
print("\n== mean over experts: kind  K  rfn_w  rfn_out  proxy  grid_exact")
for (kind, K), v in sorted(agg.items()):
    m = [sum(x[i] for x in v) / len(v) for i in range(4)]; print(f"{kind:26s} {K:<5} {m[0]:.4f} {m[1]:.4f} {m[2]:.5f} {m[3]:.3f}")
log("DSV41-EXL3-PROBE-DONE")
