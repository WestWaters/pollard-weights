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
import argparse, glob, json, os, sys

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
from pollard_backbone import load_backbone, text_layers


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
        budget = max(int(host * 0.70 / G), 2)
        mm = {dev: f"{budget - budget // 4}GiB", "cpu": f"{max(budget // 4, 1)}GiB"}
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
        if not shard:
            self.missing += 1
            return None
        try:
            from safetensors import safe_open
            with safe_open(os.path.join(self.dir, shard), framework="pt") as f:
                return f.get_tensor(f"{key}.weight")
        except Exception:
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
                         "that looks measured is worse than none.")
    return profile, noise


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True, help="held-out text (disjoint from any calib)")
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
    a = ap.parse_args()

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
