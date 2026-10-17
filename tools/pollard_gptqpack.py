#!/usr/bin/env python3
"""Pollard's own GPTQ checkpoint writer -- the streamed engine behind `pollard-export --engine pollard`.

GPTQ (Frantar et al.) quantizes one column at a time and pushes each column's rounding error into the columns not
yet quantized through the inverse Hessian H^-1 (H = X^T X of the layer's real inputs). This module runs that per
decoder block as pollard_stream feeds it (one block in memory, its inputs = the previous QUANTIZED blocks'
outputs -- sequential GPTQ), and writes the result in the GPTQ v1 checkpoint layout vLLM's gptq / gptq_marlin
loaders read (and gptqmodel / AutoGPTQ write):

    <module>.qweight   int32 [in * bits / 32, out]   codes packed along `in`, a little-endian bit stream per column
                                                    (2 / 4 / 8 bits: 32/bits codes per word; 3 bits: 32 codes per
                                                    3 words, codes straddling word boundaries -- AutoGPTQ's layout)
    <module>.qzeros    int32 [groups, out * bits / 32]  (zero - 1), packed the same way along `out` (v1 format)
    <module>.scales    float16 [groups, out]
    <module>.g_idx     int32 [in]                     column -> group (desc_act False: i // group_size)
    dequant: W[o, i] = scales[g, o] * (q[i, o] - (qzeros[g, o] + 1))

Symmetric quantization (sym=True, zero = 2^(bits-1)), group size 128, desc_act False: the Marlin fast path for
4 and 8 bits, vLLM's plain GPTQ kernel for 2 and 3. quantization_config carries `dynamic` (per-layer bit overrides
from Pollard's measured allocation, and `-:` for every layer linear left unquantized -- routers, odd shapes).

Every tensor outside the decoder blocks is copied through: towers, talkers, projectors at source precision, byte
for byte; an encoded source tensor outside the blocks (a ggml Q8_0 lm_head) is written as bf16, since a GPTQ
checkpoint carries no other encoding. A tensor already quantized in the source is quantized FROM its dequantized
values, and the sidecar records it.
"""
from __future__ import annotations

import json, os, re

GPTQ_BITS = (2, 3, 4, 8)
SIDECAR = "pollard-gptq.json"


# ----------------------------------------------------------------------------------------------- math

def gptq_sym(W, H, bits, group=128, percdamp=0.01, n_tokens=None, block=128):
    """GPTQ with symmetric per-group scales. W [rows, cols] float, H [cols, cols] (or None = RTN).
    Returns (dequantized W float32, codes int32 [rows, cols] in 0..2^bits-1, scales float32 [rows, groups])."""
    import torch
    W = W.detach().clone().float()
    rows, cols = W.shape
    group = cols if not group or group < 0 else min(group, cols)
    maxq = 2 ** bits - 1
    zero = 2 ** (bits - 1)
    if H is None:
        H = torch.eye(cols, device=W.device)
    H = H.detach().clone().float().to(W.device)
    diag = torch.diag(H)
    meandiag = diag[diag > 0].mean().clamp(min=1e-8) if bool((diag > 0).any()) else diag.new_tensor(1.0)
    dead = diag <= 1e-10 * meandiag
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    if n_tokens is not None and 0 < n_tokens < cols:          # rank-deficient: keep only the diagonal
        H = torch.diag(torch.diag(H))
    base = torch.mean(torch.diag(H)).clamp(min=1e-8)
    Hinv = None
    for attempt in range(6):
        use64 = attempt >= 3
        Hd = (H.double() if use64 else H).clone()
        Hd[range(cols), range(cols)] += percdamp * (10 ** (attempt % 3)) * base
        try:
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True).float()
            break
        except Exception:                                          # noqa: BLE001
            continue
    if Hinv is None:
        raise RuntimeError(f"GPTQ: a {cols}-column Hessian stayed non-positive-definite after damping x100 + fp64")
    Q = torch.zeros_like(W)
    C = torch.zeros(rows, cols, dtype=torch.int32, device=W.device)
    ng = (cols + group - 1) // group
    S = torch.zeros(rows, ng, device=W.device)
    blk = max(group, (block // group) * group)
    scale = None
    for b0 in range(0, cols, blk):
        b1 = min(b0 + blk, cols)
        Wb = W[:, b0:b1].clone()
        Eb = torch.zeros_like(Wb)
        for j in range(b1 - b0):
            i = b0 + j
            if i % group == 0:
                g = Wb[:, j:j + group]
                amax = g.abs().amax(1).clamp(min=1e-8)
                scale = 2 * amax / maxq
                S[:, i // group] = scale
            w = Wb[:, j]
            q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
            deq = scale * (q - zero)
            Q[:, i] = deq
            C[:, i] = q.to(torch.int32)
            err = (w - deq) / Hinv[i, i]
            Wb[:, j:] -= err.unsqueeze(1) * Hinv[i, i:b1].unsqueeze(0)
            Eb[:, j] = err
        if b1 < cols:
            W[:, b1:] -= Eb @ Hinv[b0:b1, b1:]
    # the scales are stored as float16: dequantize with exactly those
    S16 = S.half().float()
    Q = (C.float() - zero) * S16.repeat_interleave(group, 1)[:, :cols]
    return Q, C, S16


# ----------------------------------------------------------------------------------------------- packing

def pack_stream(codes, bits):
    """Pack codes [n, m] (uint values < 2^bits) along axis 0 into int32 words [n * bits / 32, m]: a little-endian
    bit stream per column (code t of a column sits at bit bits*t). Matches AutoGPTQ / gptqmodel for 2/3/4/8 bits."""
    import numpy as np
    c = np.asarray(codes, np.uint64)
    n, m = c.shape
    if (n * bits) % 32:
        raise ValueError(f"{n} codes x {bits} bits is not a whole number of int32 words")
    words = np.zeros(((n * bits) // 32, m), np.uint64)
    for t in range(n):
        pos = bits * t
        w, off = pos // 32, pos % 32
        words[w] |= (c[t] << np.uint64(off)) & np.uint64(0xFFFFFFFF)
        if off + bits > 32:
            words[w + 1] |= c[t] >> np.uint64(32 - off)
    return words.astype(np.uint32).view(np.int32)


def unpack_stream(words, bits, n):
    import numpy as np
    w = np.asarray(words).view(np.uint32).astype(np.uint64)
    m = w.shape[1]
    out = np.zeros((n, m), np.uint64)
    mask = np.uint64((1 << bits) - 1)
    for t in range(n):
        pos = bits * t
        i, off = pos // 32, pos % 32
        v = w[i] >> np.uint64(off)
        if off + bits > 32:
            v |= w[i + 1] << np.uint64(32 - off)
        out[t] = v & mask
    return out.astype(np.int32)


def pack_module(codes, scales, bits, group):
    """codes int [out, in] (0..2^bits-1), scales float [out, groups] -> the v1 GPTQ tensors (numpy)."""
    import numpy as np
    codes = np.asarray(codes)
    out_f, in_f = codes.shape
    ng = scales.shape[1]
    zero = 2 ** (bits - 1)
    qweight = pack_stream(codes.T, bits)                                   # [in*bits/32, out]
    qzeros = pack_stream(np.full((out_f, ng), zero - 1, np.uint64), bits).T.copy()   # [groups, out*bits/32]
    sc = np.asarray(scales, np.float32).T.astype(np.float16)              # [groups, out]
    g_idx = (np.arange(in_f) // group).astype(np.int32)
    return {"qweight": qweight, "qzeros": qzeros, "scales": sc, "g_idx": g_idx}


def dequant_module(t, bits, in_features):
    """The float32 [out, in] a GPTQ v1 module stores (the reference decode, used by tests and verify)."""
    import numpy as np
    q = unpack_stream(t["qweight"], bits, in_features)                    # [in, out]
    out_f = q.shape[1]
    zeros = unpack_stream(np.asarray(t["qzeros"]).T, bits, out_f).T + 1    # [groups, out]
    g = np.asarray(t["g_idx"]).astype(np.int64)
    sc = np.asarray(t["scales"]).astype(np.float32)
    return ((q.astype(np.float32) - zeros[g].astype(np.float32)) * sc[g]).T


# ----------------------------------------------------------------------------------------------- allocation

ATTN_RX = r"(self_attn|attention|attn|linear_attn|conv|mixer\.(?:q_proj|k_proj|v_proj|o_proj|in_proj|out_proj))"
FFN_RX = r"(mlp|feed_forward|block_sparse_moe|experts|shared_experts?|mixer\.(?:fc\d_latent_proj|experts|shared_experts|up_proj|down_proj))"
LAYER_RX = re.compile(r"\.layers\.(\d+)\.")


def side_of(key):
    """'attn' (token mixing: attention / conv / mamba) or 'ffn' (MLP / experts) for a block linear's key."""
    rel = key.split(".layers.", 1)[-1].split(".", 1)[-1]
    if re.match(r"(mlp|feed_forward|block_sparse_moe)\.", rel) or re.search(r"\.(experts|shared_experts?)\.", "." + rel) \
            or re.match(r"mixer\.(fc\d_latent_proj|experts|shared_experts|up_proj|down_proj)", rel):
        return "ffn"
    return "attn"


def bits_for(key, alloc, low):
    m = LAYER_RX.search(key)
    if not m or not alloc:
        return low
    g = alloc.get(int(m.group(1))) or {}
    return g.get(side_of(key), low)


def component_roots(keys):
    """The roots of the non-text components among `keys` (thinker.visual, thinker.audio_tower, talker, code2wav, mlp1,
    vision_model ...): the part of each key up to and including the component's own name."""
    import pollard_complete as PC
    roots = set()
    for k in keys:
        for _, rx in PC.COMPONENTS:
            m = rx.search(k)
            if m:
                roots.add(k[:m.end() - 1])
                break
    return sorted(roots)


def dynamic_rules(keys_bits, skipped, low, components=()):
    """gptqmodel / vLLM `dynamic`: a `+:` override per (layer, side) whose bits differ from the base, and a `-:`
    rule per layer linear left unquantized. Per (layer, side) patterns keep the map small on a 128-expert model.
    `components`: roots of the towers / projector / talker copied at source precision -- one `-:` rule each, so a
    loader that hands the quant config to the vision or speech modules builds them unquantized."""
    over = {}
    for k, b in keys_bits.items():
        if b == low:
            continue
        m = LAYER_RX.search(k)
        if not m:
            over[r"+:^" + re.escape(k[:-len(".weight")]) + "$"] = {"bits": b}
            continue
        side = side_of(k)
        rx = ATTN_RX if side == "attn" else FFN_RX
        over[rf"+:.*\.layers\.{int(m.group(1))}\.{rx}.*"] = {"bits": b}
    # vLLM / gptqmodel take the FIRST matching rule: the `-:` exclusions go first, or a layer's `+:` override would
    # claim its router too
    out = {r"-:^" + re.escape(root) + r"\..*": {} for root in components}
    out.update({r"-:^" + re.escape(mod) + "$": {} for mod in sorted(skipped)})
    out.update(dict(sorted(over.items())))
    return out


# ----------------------------------------------------------------------------------------------- what to quantize

def quantizable(key, shape, group):
    """(True, '') for a decoder-block linear this lane quantizes; (False, why) otherwise."""
    if not key.endswith(".weight") or len(shape) != 2:
        return False, "not a 2-D weight"
    mod = key[:-len(".weight")]
    last = mod.rsplit(".", 1)[-1]
    if "norm" in last:
        return False, "norm"
    if re.search(r"\.(gate|router)$", mod) or mod.endswith("shared_expert_gate"):
        return False, "MoE router (kept at source precision: it picks the experts)"
    if group and group > 0 and shape[1] % group:
        return False, f"in_features {shape[1]} is not a multiple of group {group}"
    if (shape[1] * 4) % 32 or shape[0] % 8:
        return False, "shape not packable"
    return True, ""


# ----------------------------------------------------------------------------------------------- the streamed build

def copy_plain(src, writer, key):
    """A tensor this lane does not quantize: byte copy when it is a plain tensor, bf16 when the source encoded it
    (a GPTQ checkpoint holds no modelopt / ggml encodings)."""
    import pollard_stream as S
    if src.encoding(key) is not None or src.modified(key):
        S.write_float(src, writer, key, "BF16")
    else:
        S.copy_raw(src, writer, key)


def write_config(src_dir, out, qcfg, drop_source_quant=True):
    p = os.path.join(out, "config.json")
    cfg = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
    if drop_source_quant:
        cfg.pop("quantization_config", None)
        for k in ("hf_quant_config.json",):
            if os.path.exists(os.path.join(out, k)):
                os.remove(os.path.join(out, k))
    cfg["quantization_config"] = qcfg
    json.dump(cfg, open(p, "w", encoding="utf-8"), indent=2)


def build_stream(model, out, calib, alloc=None, low=4, group=128, nsamples=128, seqlen=2048, device="cpu",
                 trust_remote_code="auto", max_ram_gb=None, percdamp=0.01, log=print, rows=None):
    """Sequential GPTQ streamed block by block from disk (pollard_stream) -> a GPTQ v1 checkpoint at `out`.
    alloc: {layer: {"attn": bits, "ffn": bits}} (pollard-export's measured allocation); low = the body bits.
    Returns the sidecar dict."""
    import time
    from collections import Counter
    import torch
    import pollard_stream as S
    t0 = time.time()
    bb = S.Backbone(model, trust_remote_code, log=log)
    src = bb.src
    log(f"   backbone: {bb.how}  ({bb.n_layers} blocks; {len(bb.other_keys)} tensors copied through)")
    plan = S.fit_memory(bb, nsamples, seqlen, S.ram_budget(max_ram_gb), log=log)
    if rows is None:
        rows = S.calib_rows(src.base_dir if src.overlay else model, calib, plan["nsamples"], seqlen, trust=bb.remote)
    rows = rows[:plan["nsamples"]]
    log(f"   calibration: {len(rows)} rows x <= {seqlen} tokens; memory plan peak ~{plan['peak_gb']} GB")
    bb.dtype = S.plan_dtype(plan)
    eng = S.Engine(bb, device=device, log=log)
    eng.record(rows)
    w = S.ShardWriter(out)
    keys_bits, skipped, why_skip = {}, set(), {}
    hess = Counter()
    encoded_from = Counter()
    chunks = max(1, int(plan.get("expert_chunks", 1)))
    for i in range(bb.n_layers):
        eng.load_layer(i)
        keys = bb.layer_keys[i]
        quant = [k for k in keys if quantizable(k, src.shape(k), group)[0]]
        for k in keys:
            if k not in quant and k.endswith(".weight") and len(src.shape(k)) == 2:
                skipped.add(k[:-len(".weight")])
                why_skip[k[:-len(".weight")]] = quantizable(k, src.shape(k), group)[1]
        probe = S.InputStats(eng, i, hessian=False, amax=False)
        lin_keys, ex_keys = probe.keys_for()
        experts = sorted({e for (_, e) in ex_keys})
        per = -(-len(experts) // chunks) if experts else 0
        deq = {}
        for c in range(chunks if experts else 1):
            rng = (experts[c * per], experts[min(len(experts), (c + 1) * per) - 1] + 1) if experts and per else None
            with S.InputStats(eng, i, hessian=True, amax=False, expert_range=rng,
                              only=None if c == 0 else set()) as st:
                eng.run_layer(i, keep_outputs=False)
            todo = [k for k in quant if (c == 0 and st._expert_key(k) is None)
                    or (st._expert_key(k) is not None and rng and rng[0] <= st._expert_key(k)[1] < rng[1])]
            for k in todo:
                H, n = st.hessian_for(k)
                b = bits_for(k, alloc, low)
                Wt = src.read_torch(k, torch.float32).to(device)
                Q, C, Sc = gptq_sym(Wt, H.to(device) if H is not None else None, b, group, percdamp,
                                    n_tokens=n if n else None)
                hess["hessian" if H is not None and n else ("block-input" if H is not None else "none (RTN)")] += 1
                mod = k[:-len(".weight")]
                for suf, arr in pack_module(C.cpu().numpy(), Sc.cpu().numpy(), b, group).items():
                    w.add(f"{mod}.{suf}", {"qweight": "I32", "qzeros": "I32", "scales": "F16", "g_idx": "I32"}[suf], arr)
                if f"{mod}.bias" in src.logical:
                    copy_plain(src, w, f"{mod}.bias")
                deq[k] = Q.cpu()
                keys_bits[k] = b
                enc = src.encoding(k)
                if enc is not None:
                    encoded_from[enc["algo"]] += 1
                del Wt, Q, C
            del st
        done = set(deq) | {k[:-len(".weight")] + ".bias" for k in deq}
        for k in keys:
            if k not in done:
                copy_plain(src, w, k)
        bb.load(list(deq), device, tensors=deq)          # the block, quantized, feeds the next one
        outs = eng.run_layer(i)
        bb.free_layer(i)
        eng.hidden = outs
        log(f"   block {i + 1}/{bb.n_layers}: {len(deq)} linears "
            f"({', '.join(f'{v}x int{b}' for b, v in sorted(Counter(keys_bits[k] for k in deq).items()))})"
            f"  [{(time.time() - t0) / 60:.1f} min, peak RSS {eng.peak_rss / 1e9:.1f} GB]")
    for k in bb.outer_keys + bb.other_keys:
        copy_plain(src, w, k)
    w.close()
    S.copy_side_files_of(src, out)
    qcfg = {"quant_method": "gptq", "bits": low, "group_size": group, "desc_act": False, "sym": True,
            "lm_head": False, "checkpoint_format": "gptq", "pack_dtype": "int32", "true_sequential": True,
            "damp_percent": percdamp, "dynamic": dynamic_rules(keys_bits, skipped, low,
                                                                   component_roots(bb.outer_keys + bb.other_keys)) or None,
            "meta": {"quantizer": ["pollard-export:stream"]}}
    write_config(src.base_dir, out, qcfg)
    n_par = sum(_numel(src.shape(k)) for k in keys_bits)
    avg = sum(keys_bits[k] * _numel(src.shape(k)) for k in keys_bits) / max(1, n_par)
    side = {"engine": "pollard-stream", "model": os.path.abspath(model), "bits": low, "group_size": group,
            "avg_bits": round(avg + 16.0 / group, 3), "quantized": len(keys_bits),
            "bits_hist": dict(Counter(keys_bits.values())), "skipped": dict(sorted(why_skip.items())),
            "hessians": dict(hess), "dequantized_source": dict(encoded_from),
            "rows": len(rows), "tokens": sum(len(r) for r in rows), "backbone": bb.describe(),
            "memory_plan": plan, "peak_rss_gb": round(eng.peak_rss / 1e9, 2),
            "minutes": round((time.time() - t0) / 60, 2)}
    json.dump(side, open(os.path.join(out, SIDECAR), "w"), indent=1)
    return side


def _numel(shape):
    n = 1
    for x in shape:
        n *= int(x)
    return n


def module_bits(qcfg, mod):
    """The bits module `mod` is stored at under a GPTQ quantization_config: the FIRST `dynamic` rule whose regex
    re.match()es the module decides (vLLM's and gptqmodel's rule: `-:` = left unquantized -> None, `+:` or no prefix =
    an override), else the base `bits`."""
    bits = qcfg.get("bits")
    for rx, ov in (qcfg.get("dynamic") or {}).items():
        if rx.startswith("-:"):
            if re.match(rx[2:], mod):
                return None
        elif re.match(rx[2:] if rx.startswith("+:") else rx, mod):
            return (ov or {}).get("bits") or bits
    return bits


def dequantized_state(rung_dir):
    """{checkpoint key: float32} of a GPTQ v1 checkpoint (every quantized module decoded) -- the reference loader
    for verify on a machine with no GPTQ kernels."""
    import numpy as np
    import pollard_stream as S
    cfg = json.load(open(os.path.join(rung_dir, "config.json"), encoding="utf-8"))
    q = cfg["quantization_config"]
    src = S.Source(rung_dir)
    out = {}
    mods = {k[:-len(".qweight")] for k in src.entries if k.endswith(".qweight")}
    for k in src.entries:
        if any(k.endswith(s) for s in (".qweight", ".qzeros", ".scales", ".g_idx")):
            continue
        out[k] = src.read(k)
    for m in mods:
        t = {s: src.read_raw(f"{m}.{s}")[1] for s in ("qweight", "qzeros", "scales", "g_idx")}
        bits = module_bits(q, m) or q["bits"]      # unprefixed rules too (pollard-export / gptqmodel write those)
        out[m + ".weight"] = dequant_module(t, bits, t["g_idx"].shape[0]).astype(np.float32)
    return out
