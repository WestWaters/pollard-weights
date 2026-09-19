#!/usr/bin/env python3
"""pollard-probe -- the CHEAP, any-box sensitivity profile. Same output as
pollard-sensitivity, a fraction of the cost, so measured allocation runs on a
laptop instead of needing a big-GPU GGUF sweep.

pollard-sensitivity is the ground truth: it CRUSHES each tensor group to a real
GGUF quant and runs a full perplexity KL pass -- 2*layers subprocess builds + KL
evals. Accurate, but heavy and GPU-hungry. This does the same measurement the
cheap way EXL3 does it: perturb one group at a time IN-PROCESS (torch), read the
logit-KL directly, restore. No GGUF, no imatrix, no separate eval binary -- one
forward pass per group. The RANKING is what the allocator needs, and injected
quant-error ranks the groups the same way for a tiny cost.

Emits the identical `sensitivity.json` schema pollard-fit consumes:
  {"ffn": {"<layer>": kl_cost}, "attn": {...}, "noise": {type: uniform_kl}, ...}

    pollard-probe --model <hf-dir-or-id> --eval held-out.txt --out model.sensitivity.json
    pollard-fit --gguf model-f16.gguf --ram 16 --sensitivity model.sensitivity.json

Note: this is the torch/RTN proxy for the GGUF crush -- the per-group ranking
matches; absolute KL is a proxy, not the ik_llama trellis error. For the final
published card, confirm the winner with a pollard-sensitivity run on the box.
"""
import argparse, glob, json, os, re, sys

# `--device cpu` has to MEAN cpu, and this has to happen BEFORE torch is imported.
# device_map="auto" enumerates every visible device, so accelerate places layers on the GPU even
# when the CPU was asked for; worse, some model code launches a Triton kernel whenever CUDA merely
# looks available, and then meets a CPU tensor:
#     ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)
# Setting this after `import torch` is too late -- torch caches cuda availability, so is_available()
# keeps answering True. On a shared box it also matters that we do not quietly take the GPU.
if "--device" in sys.argv[1:-1] and sys.argv[sys.argv.index("--device") + 1] == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch, torch.nn.functional as F
from pollard_load import load_backbone, text_layers


def _weights_bytes(model_id):
    """On-disk weight bytes, so we can tell BEFORE loading whether this fits. 0 = unknown (hub id)."""
    tot = 0
    for pat in ("*.safetensors", "*.bin"):
        for f in glob.glob(os.path.join(model_id, pat)):
            tot += os.path.getsize(f)
    return tot


def _accel_bytes(dev):
    """Usable accelerator memory, or 0 if `dev` is the CPU."""
    if dev == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.get_device_properties(0).total_memory * 0.90)
    if dev == "mps":
        rec = getattr(torch.mps, "recommended_max_memory", None)
        return int(rec() * 0.90) if rec else 0
    return 0


def _win_mem():
    """(total, avail) physical bytes on Windows. There is no sysconf and no /proc there, so a
    POSIX-only probe reads 0 RAM and offloads to disk on a box with plenty free."""
    import ctypes

    class _MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    m = _MS()
    m.dwLength = ctypes.sizeof(_MS)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return int(m.ullTotalPhys), int(m.ullAvailPhys)
    return 0, 0


def _host_bytes():
    """RAM we can actually SPEND right now, on macOS, Linux and Windows alike.

    Not the nameplate: the build box has 31.8GB installed but 15.4GB free with a desktop session
    on it, and budgeting a 22GB model against the 31.8 lands it in swap. Total is only the
    fallback for a platform that won't tell us what's free."""
    try:
        from pollard_calc import detect_available_ram_gb
        avail = detect_available_ram_gb()
        if avail:
            return int(avail * 1e9)
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            tot, avail = _win_mem()
            return avail or tot
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def plan_placement(model_id, dev, offload_dir):
    """Where does this model actually go?

    The probe used to pin the WHOLE model to one device. That silently worked for every model small
    enough to fit and then OOM'd on the first one that wasn't -- and because the caller treated a
    probe failure as "no profile", the build quietly degraded to a UNIFORM allocation, which is the
    one thing pollard-fit warns has no quality win. Measure first, then place:

      fits the accelerator      -> use it (fast path, unchanged)
      fits host RAM             -> CPU (slower, still exact)
      within OVERSPILL of RAM   -> CPU anyway, and let the OS page it
      bigger than that          -> shard across accelerator+CPU, offload the tail to disk

    The third case is deliberate. accelerate's disk-offload path took an access violation on Windows
    loading a 22.3GB model against 14.2GB free, before writing a single offload file; the OS pager
    does the same job for a modest overspill and does not crash. Reserve offload for the case where
    paging really cannot cope -- a model several times larger than RAM.

    Returns (device, load_kwargs, note).
    """
    OVERSPILL = 2.0                     # page up to 2x RAM before trusting disk offload
    need = _weights_bytes(model_id)
    if not need:                                            # hub id / unknown: keep the old behaviour
        return dev, {}, ""
    G = 1 << 30
    need_hdr = int(need * 1.15)                             # weights + activations//logits headroom
    accel, host = _accel_bytes(dev), _host_bytes()
    if accel and need_hdr <= accel:
        return dev, {}, f"{need/G:.1f}GB fits {dev} ({accel/G:.1f}GB)"
    if host and need_hdr <= host * 0.85:
        why = f"{need/G:.1f}GB exceeds {dev} ({accel/G:.1f}GB)" if accel else f"{need/G:.1f}GB"
        return "cpu", {}, f"{why} -> CPU ({host/G:.1f}GB RAM)"
    if host and need_hdr <= host * OVERSPILL:
        return "cpu", {}, (f"{need/G:.1f}GB over {host/G:.1f}GB free RAM -> CPU, OS-paged "
                           f"(under {OVERSPILL:g}x; disk offload is the fragile path)")
    os.makedirs(offload_dir, exist_ok=True)
    if accel and sys.platform == "darwin":
        # Apple Silicon is UNIFIED memory: the GPU and the CPU spend the same pool, so budgeting
        # them separately would promise ~2x the RAM that exists and thrash swap. One budget, split.
        # Carve the CPU share OUT of the budget, never in addition to it: with budget < 4,
        # budget//4 is 0 and the 1GiB floor used to be ADDED on top, so a 3GiB budget was handed
        # out as 3+1=4 -- the double-count this branch exists to stop, visible only when the box
        # is already short on RAM, which is exactly when it matters.
        budget = max(int(host * 0.70 / G), 2)
        cpu_share = max(budget // 4, 1)
        mm = {dev: f"{max(budget - cpu_share, 1)}GiB", "cpu": f"{cpu_share}GiB"}
    else:
        mm = {"cpu": f"{max(int(host * 0.70 / G), 2)}GiB"}
        if accel:
            mm[0 if dev == "cuda" else dev] = f"{max(int(accel / G), 1)}GiB"
    kw = {"device_map": "auto", "max_memory": mm, "offload_folder": offload_dir}
    return dev, kw, (f"{need/G:.1f}GB exceeds both {dev} ({accel/G:.1f}GB) and RAM "
                     f"({host/G:.1f}GB) -> sharded, tail offloaded to {offload_dir}")

# LADDER types -> (bpw, RTN bits) so the cheap noise curve keys match pollard-fit.
LADDER_BITS = [("q6_K", 6), ("q5_K", 5), ("iq4_xs", 4), ("iq3_s", 3),
               ("iq2_s", 2), ("iq2_xxs", 2)]
GROUP_ATTR = {"ffn": ("mlp", ("gate_proj", "up_proj", "down_proj")),
              "attn": ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj"))}

# The imatrix path works in GGUF tensor namespace, not HF module names. Both MoE spellings are
# here because the imatrix covers whatever the GGUF actually contains.
#
# "attn" is really the SEQUENCE-MIXING group -- whatever moves information between positions --
# as opposed to "ffn", which mixes channels. On a plain transformer that is q/k/v/output. On a
# HYBRID (Mamba/SSM + attention) model most blocks mix with a state-space operator instead, and
# its matmuls (fused attn_qkv, attn_gate, ssm_in/out, the alpha/beta projections) belong in the
# same group for allocation: the dense recipe protects the mixing path and crushes the FFN, and
# that reasoning does not change because the mixer is an SSM.
#
# Qwen3.8-27B is exactly this shape -- 48 of its 65 blocks are SSM, only 17 carry real attention.
# Leaving the SSM names out scored 256 of 496 covered matmuls and handed back 48 layers at cost
# 0.0, which reads to the allocator as "free to crush". Hence the invariant enforced below.
GROUP_GGUF = {"ffn": ("ffn_gate", "ffn_up", "ffn_down",
                      "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
                      "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"),
              "attn": ("attn_q", "attn_k", "attn_v", "attn_output",
                       "attn_qkv", "attn_gate",
                       "ssm_in", "ssm_out", "ssm_alpha", "ssm_beta", "ssm_x", "ssm_dt")}


def _gguf_slot(name, groups):
    """`blk.<N>.<base>.weight` -> (group, layer), or None for anything not in a scored group."""
    m = re.match(r"^blk\.(\d+)\.(.+)\.weight$", name)
    if not m:
        return None
    layer, base = int(m.group(1)), m.group(2)
    for g in groups:
        if base in GROUP_GGUF.get(g, ()):
            return g, layer
    return None


def _hessians_from_imatrix(path):
    """tensor name -> E[x_j^2], from EITHER imatrix format. Returns {} if neither parses."""
    from pollard_calc import read_imatrix_legacy
    try:                                                    # ik_llama's format, and ours
        raw = read_imatrix_legacy(path)
        return {n: [v / max(e["ncall"], 1) for v in e["values"]]
                for n, e in raw.items() if e["values"]}
    except Exception:
        pass
    try:                                                    # llama.cpp's newer GGUF imatrix
        from gguf import GGUFReader
        rd = GGUFReader(path)
        sums = {t.name[:-len(".in_sum2")]: t for t in rd.tensors if t.name.endswith(".in_sum2")}
        cnts = {t.name[:-len(".counts")]: t for t in rd.tensors if t.name.endswith(".counts")}
        out = {}
        for n, t in sums.items():
            c = float(cnts[n].data.reshape(-1)[0]) if n in cnts else 1.0
            out[n] = (t.data.reshape(-1).astype("float64") / max(c, 1.0)).tolist()
        return out
    except Exception:
        return {}


def _imatrix_sensitivity(gguf_path, imatrix_path, groups, probe_bits, ladder_bits):
    """The measured profile with NO forward pass, NO model load, and O(one tensor) memory.

    The one-pass estimator scores a group as sum dW^2 * h, where h_j = E[x_j^2] is the Hessian
    diagonal it gathers with forward hooks. An imatrix IS that quantity -- llama-imatrix accumulates
    exactly sum_j x_j^2 per tensor, which is why it can steer quantization at all. So whenever an
    imatrix exists (and for any real build one does, because the imatrix IS the calibration) the
    expensive half is already paid for -- on the full model, in full precision, over far more data
    than a probe run: 510 chunks of Calib 3.0 versus the probe's default 4.

    That leaves only dW = W - RTN(W,b), which needs one tensor at a time and streams off the GGUF.
    No torch model, no accelerate, no device_map, no disk offload. Model size stops mattering:
    a 27B profile becomes possible on a box that could not page 51.7GB of HF weights through the
    Windows commit limit (OSError 1455), which is the wall the HF probe hit.

    Same formula, same JSON, better activation statistics. The HF probe stays for the case this
    cannot serve: no imatrix yet, or a model with no GGUF."""
    import numpy as np
    from gguf import GGUFReader

    H = _hessians_from_imatrix(imatrix_path)
    if not H:
        raise SystemExit(f"could not read an imatrix from {imatrix_path} (tried legacy .dat and "
                         f"GGUF). Build one with llama-imatrix first.")
    rd = GGUFReader(gguf_path)
    bitset = sorted({probe_bits} | {b for _, b in ladder_bits})
    cost = {g: {} for g in groups}
    noise = {t: 0.0 for t, _ in ladder_bits}
    layers, scored, skipped = 0, 0, []
    nscored = {}                                            # (group, layer) -> tensors scored
    pinned = {}                                             # layer -> uncovered tensor names
    seen_covered = set()                                    # imatrix tensors we actually scored

    for t in rd.tensors:
        slot = _gguf_slot(t.name, groups)
        if slot is None:
            continue
        g, i = slot
        layers = max(layers, i + 1)
        nscored.setdefault((g, i), 0)
        hj = H.get(t.name)
        if hj is None:
            # In the GGUF, never calibrated -- an MTP/`nextn` head is the usual case. It must be
            # PINNED by the allocator, not scored: emitting 0.0 here would read as "costs nothing
            # to crush", which is the opposite of the truth for an uncalibrated tensor.
            pinned.setdefault(i, []).append(t.name)
            continue
        try:
            W = torch.from_numpy(np.asarray(_dequant(t))).float()
        except Exception as e:
            skipped.append(f"{t.name} ({e})")
            continue
        if W.ndim != 2 or W.shape[1] != len(hj):
            skipped.append(f"{t.name} (shape {tuple(W.shape)} vs imatrix {len(hj)})")
            continue
        h = torch.tensor(hj, dtype=torch.float32).unsqueeze(0)
        for b in bitset:                                    # ONE read, every bit-width off it
            dW = W - _rtn(W, b).float()
            c = float((dW * dW * h).sum())
            if b == probe_bits:
                cost[g][str(i)] = cost[g].get(str(i), 0.0) + c
            for ty, tb in ladder_bits:
                if tb == b:
                    noise[ty] += c
        scored += 1
        nscored[(g, i)] += 1
        seen_covered.add(t.name)
        del W

    if not scored:
        raise SystemExit("no scored tensors -- the imatrix and the GGUF do not share tensor names.")
    if skipped:
        # A partial profile that LOOKS measured is the failure mode this whole tool guards against.
        raise SystemExit(f"\n  {len(skipped)} tensors could not be scored, so the profile would be "
                         f"partial.\n  REFUSING to emit it. first: " + ", ".join(skipped[:3]))

    # THE INVARIANT: the imatrix is the ground truth for what actually gets quantized -- it holds an
    # entry for every matmul llama-quantize will touch. So a covered tensor we did not score means
    # this architecture has a weight family the group map has never heard of, and every layer built
    # from it silently lands at cost 0.0 -- i.e. "free to crush", the most damaging thing we can
    # tell the allocator. Qwen3.8-27B is how this was found: 48 of 65 blocks mix with an SSM rather
    # than attention, 240 of 496 covered matmuls went unscored, and the profile looked fine.
    missed = sorted({re.sub(r"^blk\.\d+\.", "", n) for n in H
                     if re.match(r"^blk\.\d+\..+\.weight$", n) and n not in seen_covered})
    if missed:
        raise SystemExit(
            f"\n  {len(missed)} calibrated tensor KIND(S) were not scored, so whole layers would "
            f"come back at cost 0.0\n  (= 'free to crush'). REFUSING to emit a profile this tool "
            f"cannot account for.\n  unscored: " + ", ".join(missed) +
            "\n  Add them to GROUP_GGUF in pollard_probe.py -- sequence mixers (attention or SSM) "
            "go in\n  'attn', channel mixers in 'ffn'.")

    # Drop any layer that scored nothing rather than emitting a zero for it.
    for g in groups:
        for i in [i for (gg, i) in nscored if gg == g and nscored[(gg, i)] == 0]:
            cost[g].pop(str(i), None)
    if not any(cost[g] for g in groups):
        raise SystemExit("every layer scored zero tensors -- nothing to allocate on.")

    print(f"  scored {scored} tensors across {layers} layers", flush=True)
    if pinned:
        ex = sorted(pinned)[:4]
        print(f"  {sum(len(v) for v in pinned.values())} tensors in {len(pinned)} layer(s) are NOT "
              f"in the imatrix (layers {ex}{' ...' if len(pinned) > 4 else ''}) -- left out of the "
              f"profile so the allocator PINS them rather than reading 0.0 as free.", flush=True)
    return cost, noise, layers


def _dequant(t):
    """Tensor data as a 2-D float array, whatever the GGUF stored it as."""
    import numpy as np
    from gguf import GGUFReader                             # noqa: F401  (import-time type table)
    if t.tensor_type.name in ("F32", "F16", "BF16"):
        return t.data.astype(np.float32)
    from gguf.quants import dequantize                      # lets a Q6_K host serve as the source
    return dequantize(t.data, t.tensor_type).astype(np.float32)


def _chunks(tok, text, seqlen, n):
    ids = tok(text, return_tensors="pt").input_ids[0]
    step = max(1, (len(ids) - seqlen) // max(n, 1)) if len(ids) > seqlen else seqlen
    out = [ids[i:i + seqlen] for i in range(0, max(1, len(ids) - seqlen + 1), step)][:n]
    return [c for c in out if len(c) >= 8] or [ids[:seqlen]]


@torch.no_grad()
def _rtn(W, bits, gs=64):
    """Per-row absmax symmetric RTN to `bits`, group size gs -- the actual quant
    error we perturb with (deterministic, cheap). Returns the quantized weight."""
    if bits >= 16:
        return W
    q = 2 ** (bits - 1) - 1 or 1
    out, D = W.float(), W.shape[1]
    r = out.reshape(out.shape[0], -1, min(gs, D)) if D % min(gs, D) == 0 else out.unsqueeze(1)
    scale = r.abs().amax(-1, keepdim=True).clamp_min(1e-8) / q
    r = (r / scale).round().clamp(-q - 1, q) * scale
    return r.reshape_as(W).to(W.dtype)


@torch.no_grad()
def _logits(model, chunks, dev):
    return [model(c.unsqueeze(0).to(dev)).logits[0, :-1].float().log_softmax(-1) for c in chunks]


@torch.no_grad()
def _kl_vs(model, chunks, ref_logp, dev):
    """Mean KL(clean || perturbed) over the eval chunks."""
    tot = ntok = 0.0
    for c, lp0 in zip(chunks, ref_logp):
        lp1 = model(c.unsqueeze(0).to(dev)).logits[0, :-1].float().log_softmax(-1)
        p0 = lp0.exp()
        tot += (p0 * (lp0 - lp1)).sum(-1).sum().item()
        ntok += lp0.size(0)
    return tot / max(ntok, 1)


def _linears(model, layer, group):
    """The linears of `group` in `layer` that actually EXIST.

    `hasattr` is not enough: an architecture whose layers differ from one another declares the full
    set of submodules and leaves the ones a given layer does not use set to None (Gemma4 does this).
    Taking those at face value put a None in the hook list and killed the probe with
    'NoneType has no attribute register_forward_hook' -- after loading 12B of weights. A layer with
    none of a group is legitimate; it simply contributes nothing to that group's cost."""
    parent, names = GROUP_ATTR[group]
    mod = getattr(text_layers(model)[layer], parent, None)
    if mod is None:
        return []
    return [m for m in (getattr(mod, n, None) for n in names) if m is not None]


class WeightSource:
    """A linear's weight, even when accelerate left it on the meta device.

    The one-pass estimator needs two different things: E[x_j^2], which comes from HOOKS and so only
    needs a forward pass, and dW = W - RTN(W), which needs the weights themselves. Those have very
    different memory costs, and conflating them caps the probe at models that fit in memory. Once a
    model is big enough to be offloaded, most of its weights sit on the meta device holding NO data
    -- reading them there yields a wrong answer, and a wrong sensitivity profile is worse than none,
    because it produces a confidently bad allocation that still looks like a measured build.

    So resolve the weight from the checkpoint on disk instead, one tensor at a time: O(1) memory,
    and the probe stops caring how big the model is."""

    def __init__(self, model, model_dir):
        self.names = {id(m): n for n, m in model.named_modules()}
        self.dir = model_dir
        self.map = {}                         # tensor key -> shard filename
        self.missing = 0
        self.unresolved = []
        idx = os.path.join(model_dir or "", "model.safetensors.index.json")
        single = os.path.join(model_dir or "", "model.safetensors")
        try:
            if os.path.isfile(idx):
                with open(idx, encoding="utf-8") as f:
                    self.map = json.load(f).get("weight_map", {})
            elif os.path.isfile(single):
                from safetensors import safe_open
                with safe_open(single, framework="pt") as f:
                    self.map = {k: "model.safetensors" for k in f.keys()}
        except Exception:
            self.map = {}

    def get(self, lin):
        w = getattr(lin, "weight", None)
        if w is None:
            return None
        if not (w.is_meta or w.device.type == "meta"):
            return w.data
        key = self.names.get(id(lin))
        shard = self.map.get(f"{key}.weight") if key else None
        if not shard and key:
            # The module path and the checkpoint key need not agree: a vision-language checkpoint
            # nests the text stack (model.language_model.layers.N...), and wrappers can add or drop
            # a prefix. The tail is what identifies the tensor, so match on it.
            tail = key.split(".")
            for n in range(min(5, len(tail)), 1, -1):
                suffix = ".".join(tail[-n:]) + ".weight"
                hits = [k for k in self.map if k.endswith(suffix)]
                if len(hits) == 1:
                    shard, key = self.map[hits[0]], hits[0][:-len(".weight")]
                    break
        if not shard:
            if len(self.unresolved) < 5:
                self.unresolved.append(key or f"<unnamed {type(lin).__name__}>")
            self.missing += 1
            return None
        try:
            from safetensors import safe_open
            with safe_open(os.path.join(self.dir, shard), framework="pt") as f:
                return f.get_tensor(f"{key}.weight")
        except Exception as e:
            if len(self.unresolved) < 5:
                self.unresolved.append(f"{key}: {type(e).__name__}")
            self.missing += 1
            return None


@torch.no_grad()
def _stream_sensitivity(model, chunks, dev, groups, layers, probe_bits, ladder_bits,
                        model_dir=None):
    """ONE forward pass over the calib set, sensitivity for EVERY group at once -- for models where
    the perturb+KL loop (layersxgroups full passes) is infeasible (744B over 1.5TB).

    For a linear y=Wx, the expected output error from quantizing W->W_hat is
        E||(W-W_hat)x||^2 = sum_ij (W-W_hat)_ij^2 * E[x_j^2]
    i.e. the squared quant error weighted by the Hessian DIAGONAL h_j = E[x_j^2]. We accumulate h_j
    per target linear with hooks in a single pass (no per-group forward), then score each group as
    sum deltaW^2 * h. Same ranking the perturb+KL probe gives, at a fraction of the cost."""
    h, cnt, index = {}, {}, {}
    hooks = []

    def mk(lid):
        def hook(mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            s = (x * x).sum(0)
            h[lid] = s if lid not in h else h[lid] + s
            cnt[lid] = cnt.get(lid, 0) + x.shape[0]
        return hook

    for i in range(layers):
        for g in groups:
            for lin in _linears(model, i, g):
                lid = id(lin); index[lid] = (g, i)
                hooks.append(lin.register_forward_hook(mk(lid)))
    for c in chunks:
        model(c.unsqueeze(0).to(dev))
    for hk in hooks:
        hk.remove()

    def cost_at(bits):
        cost = {g: {} for g in groups}
        for i in range(layers):
            for g in groups:
                tot = 0.0
                for lin in _linears(model, i, g):
                    lid = id(lin)
                    if lid not in h or cnt.get(lid, 0) == 0:      # module never fired (unused/pruned expert)
                        continue
                    W = weights.get(lin)
                    if W is None:                                # unreadable and unresolvable
                        continue
                    hj = (h[lid] / cnt[lid]).to(W.device)        # E[x_j^2]
                    dW = W.float() - _rtn(W, bits).float()
                    tot += float((dW * dW * hj.unsqueeze(0)).sum().item())
                cost[g][str(i)] = tot
        return cost

    weights = WeightSource(model, model_dir)
    profile = cost_at(probe_bits)
    noise = {t: sum(sum(cost_at(bits)[g].values()) for g in groups) for t, bits in ladder_bits}
    if weights.missing:
        # Silently dropping tensors would hand back a profile that looks measured and is not.
        raise SystemExit(f"\n  {weights.missing} weights were unreadable (offloaded to the meta "
                         "device and not resolvable from the checkpoint on disk).\n"
                         "  REFUSING to emit a partial sensitivity profile -- a wrong allocation "
                         "that looks measured is worse than none.\n  first unresolved: "
                         + ", ".join(weights.unresolved))
    return profile, noise


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    # Neither is needed by --from-imatrix (it reads the GGUF and the imatrix, nothing else), so
    # they are validated per-mode below rather than demanded up front.
    ap.add_argument("--model", help="HF model dir/id (the forward-pass probe)")
    ap.add_argument("--eval", help="held-out text (disjoint from any calib); forward-pass probe only")
    ap.add_argument("--out", help="profile path (default <model>.sensitivity.json)")
    ap.add_argument("--groups", default="ffn,attn")
    ap.add_argument("--probe-bits", type=int, default=2, help="RTN bits to crush a group to (default 2)")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--device", default="auto",
                    help="auto (default: cuda>mps>cpu, and shard+offload if the model is bigger "
                         "than the accelerator) or an explicit cuda/mps/cpu")
    ap.add_argument("--offload-dir", help="where to spill layers when the model fits neither the "
                                          "accelerator nor RAM (default: alongside --out)")
    ap.add_argument("--stream", action="store_true",
                    help="ONE-pass Hessian-diagonal estimator instead of per-group perturb+KL -- for "
                         "models too big to run layersxgroups forward passes (744B-scale)")
    ap.add_argument("--from-imatrix", metavar="IMATRIX",
                    help="derive the profile from an EXISTING imatrix + --gguf, with no forward "
                         "pass and no model load (O(one tensor) memory -- any model size on any "
                         "box). The imatrix already IS E[x_j^2]; reuse it instead of re-measuring.")
    ap.add_argument("--gguf", help="source GGUF for --from-imatrix (f16 preferred; a quantized "
                                   "host works and is dequantized per tensor)")
    a = ap.parse_args()

    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    if a.from_imatrix:
        if not a.gguf:
            sys.exit("--from-imatrix needs --gguf (the weights the imatrix was measured against).")
        for p in (a.from_imatrix, a.gguf):
            if not os.path.exists(p):
                sys.exit(f"not found: {p}")
        print(f"== pollard-probe :: {a.gguf}  probe={a.probe_bits}bit  "
              f"from-imatrix (no forward pass)", flush=True)
        profile, noise, layers = _imatrix_sensitivity(a.gguf, a.from_imatrix, groups,
                                                      a.probe_bits, LADDER_BITS)
        out = a.out or (os.path.splitext(a.gguf)[0] + ".sensitivity.json")
        json.dump({**profile, "noise": noise, "probe": f"rtn{a.probe_bits}", "layers": layers,
                   "source": a.gguf, "method": "imatrix-hessian"}, open(out, "w"), indent=2)
        for g in groups:
            vals = [v for v in profile[g].values()]
            if vals:
                lo, hi = min(vals), max(vals)
                print(f"  {g}: spread {hi/max(lo,1e-9):.1f}x  (min {lo:.4f}  max {hi:.4f})", flush=True)
        print(f"\ndone: {out}\n  feed it:  pollard-fit --gguf <f16>.gguf --ram <GB> "
              f"--sensitivity {out}", flush=True)
        return
    if not a.model or not a.eval:
        sys.exit("the forward-pass probe needs --model and --eval "
                 "(or use --from-imatrix IMATRIX --gguf MODEL.gguf).")

    from transformers import AutoTokenizer
    dev = a.device
    if dev == "auto":
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    elif dev == "mps" and not torch.backends.mps.is_available():
        dev = "cpu"
    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    offdir = a.offload_dir or os.path.join(os.path.dirname(a.out or ".") or ".", "pollard-offload")
    dev, loadkw, note = plan_placement(a.model, dev, offdir)
    print(f"== pollard-probe :: {a.model}  probe={a.probe_bits}bit  dev={dev}", flush=True)
    if note:
        print(f"   placement: {note}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if dev == "cpu":
        # Eager attention on the CPU path. A fused/Triton attention is selected on the strength of
        # CUDA merely LOOKING available -- torch.cuda.is_available() reports the driver, not the
        # visible devices -- and then meets a CPU tensor:
        #     ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)
        # The probe only needs the activations, so the plainest kernel is the right one.
        loadkw.setdefault("attn_implementation", "eager")
    model = load_backbone(a.model, torch.float16, dev, **loadkw)
    if loadkw.get("device_map"):            # accelerate hooks move inputs; stage them on the CPU
        dev = "cpu"
    layers = len(text_layers(model))
    # A DENSE model has no expert redundancy for the allocator to reallocate, and a measured
    # profile does NOT beat uniform there -- pollard-sensitivity refuses dense outright for this
    # reason. This tool is the cheap estimator for the same thing, so it inherits the caveat:
    # measured here on Qwen2.5-0.5B at matched size, imatrix-only IQ3_S came out +7.48% over fp16
    # while imatrix + measured allocation came out +14.17%. The profile made the build WORSE, and
    # nothing said so, because only the expensive path carried the warning.
    if not any(getattr(l, "mlp", None) and hasattr(getattr(l, "mlp"), "experts")
               for l in text_layers(model)):
        print("   NOTE: this model looks DENSE. A measured profile is the MoE lever -- on dense it\n"
              "         does not beat uniform (no expert redundancy to reallocate) and can lose to\n"
              "         plain imatrix-guided quantization. Dense -> imatrix directly; measure only\n"
              "         if you intend to compare the two.", flush=True)
    ch = _chunks(tok, open(a.eval, encoding="utf-8").read(), a.seqlen, a.chunks)

    if a.stream:
        print(f"  {layers} layers, {len(ch)} calib chunks -- one-pass Hessian-diagonal estimator", flush=True)
        profile, noise = _stream_sensitivity(model, ch, dev, groups, layers, a.probe_bits,
                                             LADDER_BITS, model_dir=a.model)
        method = "pollard-probe stream (Hessian-diagonal proxy)"
        for g in groups:
            vals = list(profile[g].values())
            print(f"  {g}: streamed {len(vals)} layers", flush=True)
    else:
        method = "pollard-probe (torch RTN proxy)"
        ref = _logits(model, ch, dev)
        print(f"  {layers} layers, {len(ch)} eval chunks, clean logits cached", flush=True)

        # per-model noise curve: RTN-crush EVERY group to each rung, measure KL. Cheap
        # (len(LADDER) forwards) and it's what lets the allocator see catastrophic rungs.
        noise = {}
        saved_all = {}
        for t, bits in LADDER_BITS:
            for i in range(layers):
                for g in groups:
                    for lin in _linears(model, i, g):
                        saved_all.setdefault(id(lin), lin.weight.data.clone())
                        lin.weight.data = _rtn(saved_all[id(lin)], bits)
            noise[t] = _kl_vs(model, ch, ref, dev)
            for i in range(layers):                                  # restore
                for g in groups:
                    for lin in _linears(model, i, g):
                        lin.weight.data = saved_all[id(lin)].clone()
            print(f"  noise {t:8} ({bits}b): KL={noise[t]:.4f}", flush=True)

        # per-(group,layer) sensitivity: crush ONE group at ONE layer, measure KL hit.
        profile = {g: {} for g in groups}
        for i in range(layers):
            row = []
            for g in groups:
                lins = _linears(model, i, g)
                saved = [l.weight.data.clone() for l in lins]
                for l in lins:
                    l.weight.data = _rtn(l.weight.data, a.probe_bits)
                profile[g][str(i)] = _kl_vs(model, ch, ref, dev)
                for l, w in zip(lins, saved):
                    l.weight.data = w
                row.append(f"{g}={profile[g][str(i)]:.4f}")
            print(f"  layer {i:>3}/{layers}  " + "  ".join(row), flush=True)

    if a.out:
        out = a.out
    else:
        try:
            # NOT `, os`: a function-local import of a module already imported at file scope makes
            # the name local to the WHOLE function, so every earlier use in main() raises
            # UnboundLocalError -- which is how this failed on the box and not here.
            import pollard_workspace as ws
            out = os.path.join(ws.calibration_dir(a.model, create=True),
                               ws.model_basename(a.model) + ".sensitivity.json")
        except Exception:
            out = a.model.rstrip("/").split("/")[-1] + ".sensitivity.json"
    payload = {**profile, "noise": noise, "probe": f"rtn{a.probe_bits}",
               "layers": layers, "source": a.model, "method": method}
    json.dump(payload, open(out, "w"), indent=2)
    for g in groups:
        vals = list(profile[g].values())
        lo, hi = min(vals), max(vals)
        print(f"  {g}: spread {hi/max(lo,1e-9):.1f}x  (min {lo:.4f}  max {hi:.4f})", flush=True)
    print(f"\ndone: {out}\n  feed it:  pollard-fit --gguf <f16>.gguf --ram <GB> --sensitivity {out}", flush=True)


if __name__ == "__main__":
    main()
