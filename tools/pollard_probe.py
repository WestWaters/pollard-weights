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

STABILITY (--stability N / --stability-files a.txt,b.txt). A ranking is measured ON a calibration
set, and a different set could rank the groups differently -- Jurly (@jurlycat) named this as the
next bottleneck when Pollard was written up in Sept 2026. So the probe can measure its own ranking
twice over: the per-chunk costs it already computes are split into N slices (contiguous slices of a
multi-domain set, or one slice per --stability-files corpus), each slice ranks the groups on its own,
and the profile records how much they agree -- Spearman between slice rankings per group, and how
much of the protect set (the most sensitive quarter) survives from one slice to the next. STABLE
means the ranking is the model's; UNSTABLE means it is the corpus's, and the card must say so.
"""
import argparse, glob, json, math, os, re, sys

# `--device cpu` has to MEAN cpu, and this has to happen BEFORE torch is imported.
# device_map="auto" enumerates every visible device, so accelerate places layers on the GPU even
# when the CPU was asked for; worse, some model code launches a Triton kernel whenever CUDA merely
# looks available, and then meets a CPU tensor:
#     ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)
# Setting this after `import torch` is too late -- torch caches cuda availability, so is_available()
# keeps answering True. On a shared box it also matters that we do not quietly take the GPU.
# "-1", not "": on Windows an EMPTY mask leaves torch.cuda.is_available() True with zero devices, and the first
# device query (torchao's Triton check, pulled in by importing transformers) dies with IndexError.
# Same block: linear-attention models (Qwen3.5 / FrogNano Gated DeltaNet) pick up `fla` whenever it is importable,
# with no device check, and fla's Triton kernels then start a CUDA driver on a CPU run -- a native access violation
# (exit 3221225477) with no traceback. Hiding fla and causal_conv1d makes transformers use its own torch path.
if "--device" in sys.argv[1:-1] and sys.argv[sys.argv.index("--device") + 1] == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.modules["fla"] = None
    sys.modules["causal_conv1d"] = None

import torch, torch.nn.functional as F


def load_backbone(model_id, dtype=None, device="cpu", eval_mode=True, **kw):
    """This tool's OWN loader -- model tooling does not depend on a shared/brain-side one.

    AutoModelForCausalLM refuses a vision-language config outright ("Unrecognized configuration
    class Qwen2VLConfig for this kind of AutoModel"), so fall back to the vision-language auto
    classes: a VL model's text stack quantizes like any other. Both failures are reported if
    neither works, not just the last one.

    Pass `device_map=` to shard across GPU+CPU+disk; accelerate owns placement after that, so
    `device` is ignored in that case.
    """
    import torch as _t, transformers as _tf
    from transformers import AutoModelForCausalLM
    if dtype is None:
        dtype = _t.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kw)
    except ValueError as text_only_err:
        model, errs = None, [f"AutoModelForCausalLM: {text_only_err}"]
        for _n in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            _c = getattr(_tf, _n, None)
            if _c is None:
                continue
            try:
                model = _c.from_pretrained(model_id, dtype=dtype, **kw)
                break
            except Exception as e:
                errs.append(f"{_n}: {e}")
        if model is None:
            raise SystemExit(f"could not load {model_id!r} as a causal LM or a vision-language "
                             "model:\n  " + "\n  ".join(str(e)[:160] for e in errs)) from None
    if kw.get("device_map") is None:            # accelerate already placed a dispatched model
        model = model.to(device)
    return model.eval() if eval_mode else model


def text_layers(model):
    """This tool's OWN layer walk. A VL model keeps its text stack under `model.language_model`."""
    for path in ("model.language_model", "language_model.model", "model"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        layers = getattr(node, "layers", None) if node is not None else None
        if layers is not None:
            return layers
    layers = getattr(model, "layers", None)
    if layers is not None:
        return layers
    raise SystemExit(f"could not find the decoder layers on {type(model).__name__}; "
                     "this tool's text_layers() needs a path for this architecture")


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


def plan_placement(model_id, dev, offload_dir, need=None):
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
    # `need` lets a caller state the weight size instead of having a checkpoint on disk. Exercising
    # the too-big-for-this-machine path used to mean truncating a 400GB file into a temp dir, which
    # is only free on a filesystem with sparse files -- on NTFS it writes real bytes and filled the
    # build box's disk to 560MB mid-quantize. Nothing about this decision needs the file to exist.
    if need is None:
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
# Each group is a list of (parent module, linears) -- a layer contributes whichever parent it has. "attn" is the
# SEQUENCE-MIXING group (see GROUP_GGUF below): on a hybrid such as Qwen3.5 most layers mix with `linear_attn`
# (Gated DeltaNet) instead of `self_attn`. With only self_attn listed, the forward probe scored those layers
# 0.0 and the allocator crushed their mixers to the floor as "free" -- clef-flash, 2026-10-09: 24 of 32 layers
# at attn 0.0, every one of their attn_qkv / attn_gate planned iq2_xxs in the IQ3_S rung. The GGUF path
# already counted them; this is the same set by HF module name.
GROUP_ATTR = {"ffn": [("mlp", ("gate_proj", "up_proj", "down_proj"))],
              "attn": [("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                       ("linear_attn", ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"))]}

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
    excluded = {}                                          # known tensors not allocated by this profile

    for t in rd.tensors:
        if re.fullmatch(r"blk\.\d+\.ffn_gate_inp\.weight", t.name):
            # llama-quantize keeps MoE routers unchanged unless its separate
            # router-type option is supplied. They are not expert FFN weights.
            excluded[t.name] = {"source_type": t.tensor_type.name,
                                "reason": "MoE router; retain source precision"}
            continue
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
            h = _imatrix_weights(W, hj, t.name.endswith("_exps.weight"))
        except Exception as e:
            skipped.append(f"{t.name} ({e})")
            continue
        rows = W.reshape(-1, W.shape[-1])
        for b in bitset:                                    # ONE read, every bit-width off it
            dW = W - _rtn(rows, b).reshape(W.shape).float()
            c = float((dW * dW * h).sum())
            if not math.isfinite(c):
                raise SystemExit(f"non-finite probe cost for {t.name}; refusing to emit a profile")
            if b == probe_bits:
                cost[g][str(i)] = cost[g].get(str(i), 0.0) + c
            for ty, tb in ladder_bits:
                if tb == b:
                    noise[ty] += c
        scored += 1
        nscored[(g, i)] += 1
        seen_covered.add(t.name)
        del W

    if skipped:
        # A partial profile that LOOKS measured is the failure mode this whole tool guards against.
        raise SystemExit(f"\n  {len(skipped)} tensors could not be scored, so the profile would be "
                         f"partial.\n  REFUSING to emit it. first: " + ", ".join(skipped[:3]))
    if not scored:
        raise SystemExit("no scored tensors -- the imatrix and the GGUF do not share tensor names.")

    # THE INVARIANT: the imatrix is the ground truth for what actually gets quantized -- it holds an
    # entry for every matmul llama-quantize will touch. So a covered tensor we did not score means
    # this architecture has a weight family the group map has never heard of, and every layer built
    # from it silently lands at cost 0.0 -- i.e. "free to crush", the most damaging thing we can
    # tell the allocator. Qwen3.8-27B is how this was found: 48 of 65 blocks mix with an SSM rather
    # than attention, 240 of 496 covered matmuls went unscored, and the profile looked fine.
    missed = sorted({re.sub(r"^blk\.\d+\.", "", n) for n in H
                     if re.match(r"^blk\.\d+\..+\.weight$", n)
                     and n not in seen_covered and n not in excluded})
    if missed:
        # An unfamiliar architecture is an ONBOARDING case, not a dead end. Pollard's convention is
        # to forward it so the next person's model of that family one-shots -- every onboarding
        # feeds the tool. We are better placed than most to do that: we know the exact tensor kinds
        # that matched nothing, which is the finding an onboarding contribution needs.
        raise SystemExit(
            f"\n  {len(missed)} calibrated tensor KIND(S) were not scored, so whole layers would "
            f"come back at cost 0.0\n  (= 'free to crush'). REFUSING to emit a profile this tool "
            f"cannot account for.\n\n  unscored: " + ", ".join(missed) +
            f"\n\n  -> onboard this architecture, and paste the line above into the findings:\n"
            f"       pollard-onboard --model {gguf_path} --contribute\n"
            "     Then add these kinds to GROUP_GGUF in pollard_probe.py -- sequence mixers\n"
            "     (attention OR a state-space operator) go in 'attn', channel mixers in 'ffn'.\n"
            "     Qwen3.8-27B is how this check was born: 48 of its 65 blocks mix with an SSM.")

    # Drop any layer that scored nothing rather than emitting a zero for it.
    for g in groups:
        for i in [i for (gg, i) in nscored if gg == g and nscored[(gg, i)] == 0]:
            cost[g].pop(str(i), None)
    if not any(cost[g] for g in groups):
        raise SystemExit("every layer scored zero tensors -- nothing to allocate on.")

    print(f"  scored {scored} tensors across {layers} layers", flush=True)
    if excluded:
        cost["excluded_tensors"] = excluded
        print(f"  {len(excluded)} MoE routers excluded from the score; retain their source "
              "precision. Quantizing routers separately requires a separate measurement.", flush=True)
    if pinned:
        ex = sorted(pinned)[:4]
        print(f"  {sum(len(v) for v in pinned.values())} tensors in {len(pinned)} layer(s) are NOT "
              f"in the imatrix (layers {ex}{' ...' if len(pinned) > 4 else ''}) -- left out of the "
              f"profile so the allocator PINS them rather than reading 0.0 as free.", flush=True)
    return cost, noise, layers


def _imatrix_weights(weights, importance, merged_experts=False):
    """Align dense or expert-specific input-channel importance with weight rows."""
    h = torch.tensor(importance, dtype=torch.float32)
    if not torch.isfinite(weights).all() or not torch.isfinite(h).all() or (h < 0).any():
        raise ValueError("weights and nonnegative importance must be finite")
    if h.ndim != 1 or any(size == 0 for size in weights.shape):
        raise ValueError("empty weights or non-vector importance")
    if weights.ndim == 2 and weights.shape[-1] == h.numel():
        return h.unsqueeze(0)
    if weights.ndim == 3 and merged_experts:
        experts, _, channels = weights.shape
        if experts * channels == h.numel():
            return h.reshape(experts, 1, channels)
    raise ValueError(f"weight shape {tuple(weights.shape)} vs imatrix {h.numel()}")


def _dequant(t):
    """Decode stored GGUF values, preserving logical tensor dimensions."""
    import numpy as np
    from gguf import GGUFReader                             # noqa: F401  (import-time type table)
    if t.tensor_type.name in ("F32", "F16"):
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
def _kl_per_chunk(model, chunks, ref_logp, dev):
    """(sum KL, tokens) per eval chunk -- the pieces every slice-level mean is made of."""
    out = []
    for c, lp0 in zip(chunks, ref_logp):
        lp1 = model(c.unsqueeze(0).to(dev)).logits[0, :-1].float().log_softmax(-1)
        p0 = lp0.exp()
        out.append(((p0 * (lp0 - lp1)).sum(-1).sum().item(), float(lp0.size(0))))
    return out


def _mean_kl(per_chunk, idx=None):
    rows = per_chunk if idx is None else [per_chunk[i] for i in idx]
    tot = sum(r[0] for r in rows); ntok = sum(r[1] for r in rows)
    return tot / max(ntok, 1)


@torch.no_grad()
def _kl_vs(model, chunks, ref_logp, dev):
    """Mean KL(clean || perturbed) over the eval chunks."""
    return _mean_kl(_kl_per_chunk(model, chunks, ref_logp, dev))


def _spearman(a, b):
    """Rank correlation of two equal-length sequences (average ranks for ties)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    if n < 2:
        return 1.0
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 1.0


def stability_report(per_chunk_profile, slices, protect_frac=0.25):
    """How much the ranking depends on WHICH calibration text measured it.

    per_chunk_profile: {group: {layer: [(kl_sum, ntok) per chunk]}}; slices: [[chunk idx], ...].
    Per group: every slice ranks the layers on its own; report the minimum and mean pairwise
    Spearman between slice rankings, and the overlap (Jaccard) of each slice's protect set -- the
    most sensitive `protect_frac` of layers -- with every other slice's. Verdict thresholds:
    STABLE   min Spearman >= 0.8 and min protect overlap >= 0.6
    MIXED    min Spearman >= 0.5
    UNSTABLE otherwise -- the ranking is the corpus's, not the model's."""
    out = {"slices": len(slices), "groups": {}, "protect_frac": protect_frac}
    worst_rho, worst_ov = 1.0, 1.0
    for g, layers in per_chunk_profile.items():
        keys = sorted(layers, key=lambda k: int(k))
        per_slice = [[_mean_kl(layers[k], idx) for k in keys] for idx in slices]
        n_prot = max(1, int(round(len(keys) * protect_frac)))
        prot = [set(sorted(range(len(keys)), key=lambda i: -v[i])[:n_prot]) for v in per_slice]
        rhos, ovs = [], []
        for i in range(len(slices)):
            for j in range(i + 1, len(slices)):
                rhos.append(_spearman(per_slice[i], per_slice[j]))
                ovs.append(len(prot[i] & prot[j]) / max(1, len(prot[i] | prot[j])))
        rho_min = min(rhos) if rhos else 1.0; ov_min = min(ovs) if ovs else 1.0
        worst_rho, worst_ov = min(worst_rho, rho_min), min(worst_ov, ov_min)
        out["groups"][g] = {"spearman_min": round(rho_min, 3),
                            "spearman_mean": round(sum(rhos) / len(rhos), 3) if rhos else 1.0,
                            "protect_overlap_min": round(ov_min, 3),
                            "protect_overlap_mean": round(sum(ovs) / len(ovs), 3) if ovs else 1.0,
                            "protect_sets": [sorted(keys[i] for i in p) for p in prot],
                            "per_slice_costs": [[round(v, 5) for v in row] for row in per_slice]}
    out["verdict"] = ("STABLE" if worst_rho >= 0.8 and worst_ov >= 0.6 else
                      "MIXED" if worst_rho >= 0.5 else "UNSTABLE")
    out["spearman_min"], out["protect_overlap_min"] = round(worst_rho, 3), round(worst_ov, 3)
    out["advice"] = {
        "STABLE":   "the ranking is the model's, not the corpus's -- allocate on it",
        "MIXED":    "the coarse order holds but the protect set moves with the corpus -- allocate on the "
                    "UNION of the slices (a multi-domain calibration set), not on one domain",
        "UNSTABLE": "different text ranks the groups differently -- do not allocate on a single-domain "
                    "set; use Calib 3.0-style multi-domain text and re-probe, and say so on the card",
    }[out["verdict"]]
    return out


def _linears(model, layer, group):
    """The linears of `group` in `layer` that actually EXIST.

    `hasattr` is not enough: an architecture whose layers differ from one another declares the full
    set of submodules and leaves the ones a given layer does not use set to None (Gemma4 does this).
    Taking those at face value put a None in the hook list and killed the probe with
    'NoneType has no attribute register_forward_hook' -- after loading 12B of weights. A layer with
    none of a group is legitimate; it simply contributes nothing to that group's cost."""
    blk = text_layers(model)[layer]
    out = []
    for parent, names in GROUP_ATTR[group]:
        mod = getattr(blk, parent, None)
        if mod is not None:
            out += [m for m in (getattr(mod, n, None) for n in names) if m is not None]
    return out


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
    ap.add_argument("--stability", type=int, default=0, metavar="N",
                    help="also split the eval chunks into N contiguous slices and report how much the "
                         "ranking agrees between them (Spearman, protect-set overlap). 0 = off")
    ap.add_argument("--stability-files", default="",
                    help="comma list of extra calibration corpora; each becomes its own slice "
                         "(--chunks chunks from each) alongside --eval. Implies --stability")
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
        # The module-level guard above only sees an explicit --device cpu. Placement lands here too -- a
        # model bigger than the GPU and about the size of RAM is paged on the CPU -- and fla was then still
        # importable: clef-flash (Qwen3.5, 17.8GB vs a 16GB GPU, 2026-10-09) died in fla's Triton l2norm,
        # "Pointer argument cannot be accessed from Triton (cpu tensor?)". The model module is not imported
        # yet, so hiding them here still sends transformers down its torch path.
        sys.modules["fla"] = None
        sys.modules["causal_conv1d"] = None
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
    # Dense profiles are produced, not refused. The only dense measurement we have is a 0.5B
    # (below), which is far too small to settle the question for a 12B or a 27B -- so it is
    # stated as the data point it is and the user benches the two arms.
    if not any(getattr(l, "mlp", None) and hasattr(getattr(l, "mlp"), "experts")
               for l in text_layers(model)):
        print("   NOTE: dense model. Per-layer allocation has the most headroom on MoE. The one\n"
              "         dense measurement on record is Qwen2.5-0.5B at matched size -- imatrix-only\n"
              "         IQ3_S +7.48% over fp16 vs imatrix+profile +14.17% -- i.e. the profile lost\n"
              "         on a 0.5B. Larger dense models are untested; bench both arms.", flush=True)
    ch = _chunks(tok, open(a.eval, encoding="utf-8").read(), a.seqlen, a.chunks)
    slices, slice_names = [], []
    if a.stability_files:
        slices.append(list(range(len(ch)))); slice_names.append(os.path.basename(a.eval))
        for f in [x.strip() for x in a.stability_files.split(",") if x.strip()]:
            extra = _chunks(tok, open(f, encoding="utf-8").read(), a.seqlen, a.chunks)
            slices.append(list(range(len(ch), len(ch) + len(extra)))); slice_names.append(os.path.basename(f))
            ch = ch + extra
    elif a.stability and a.stability > 1:
        n = a.stability
        per = max(1, len(ch) // n)
        slices = [list(range(i * per, (i + 1) * per if i < n - 1 else len(ch))) for i in range(n)]
        slice_names = [f"slice{i}" for i in range(n)]
        if len(ch) < 2 * n:
            print(f"   NOTE: --stability {n} with only {len(ch)} chunks; raise --chunks for a meaningful split", flush=True)

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
        per_chunk = {g: {} for g in groups}
        for i in range(layers):
            row = []
            for g in groups:
                lins = _linears(model, i, g)
                saved = [l.weight.data.clone() for l in lins]
                for l in lins:
                    l.weight.data = _rtn(l.weight.data, a.probe_bits)
                pc = _kl_per_chunk(model, ch, ref, dev)
                per_chunk[g][str(i)] = pc
                profile[g][str(i)] = _mean_kl(pc)
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
    if slices and not a.stream:
        st = stability_report(per_chunk, slices)
        st["slice_names"] = slice_names
        payload["stability"] = st
        print(f"\n  stability across {len(slices)} calibration slices ({', '.join(slice_names)}):", flush=True)
        for g, r in st["groups"].items():
            print(f"    {g}: Spearman min {r['spearman_min']:.2f} mean {r['spearman_mean']:.2f}  "
                  f"protect-set overlap min {r['protect_overlap_min']:.2f}", flush=True)
        print(f"  RANKING {st['verdict']} -- {st['advice']}", flush=True)
    elif slices:
        print("  (stability needs the forward-pass sweep; --stream keeps no per-chunk costs)", flush=True)
    json.dump(payload, open(out, "w"), indent=2)
    for g in groups:
        vals = list(profile[g].values())
        lo, hi = min(vals), max(vals)
        print(f"  {g}: spread {hi/max(lo,1e-9):.1f}x  (min {lo:.4f}  max {hi:.4f})", flush=True)
    print(f"\ndone: {out}\n  feed it:  pollard-fit --gguf <f16>.gguf --ram <GB> --sensitivity {out}", flush=True)


if __name__ == "__main__":
    main()
