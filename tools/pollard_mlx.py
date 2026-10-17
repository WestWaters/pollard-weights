#!/usr/bin/env python3
"""pollard-mlx -- emit an MLX (Apple Silicon) mixed-bit model carrying Pollard's measured allocation.

The Apple deployment lane: llama.cpp/GGUF is the CPU/portable flagship, GPTQ (pollard-export) is the
vLLM/SGLang server lane, and THIS is the on-device Apple lane (M-series, ANE/Metal via MLX). Same
Pollard brain -- detect arch, allocate bits by measured sensitivity under a size budget -- a different
EMITTER. MLX group-quant supports mixed bits via a per-module `quant_predicate`, so the profile's hot
modules stay high (8-bit) and the tolerant body is crushed (4-bit); router + shared experts pinned high.

  pollard-mlx --model Qwen/Qwen2.5-1.5B-Instruct --out ./Qwen2.5-1.5B-Pollard-MLX
  pollard-mlx --model <hf> --sensitivity q15b.sensitivity.json --hot-frac 0.35 --plan-only
  # then run on a Mac:  mlx_lm.generate --model ./...-Pollard-MLX --prompt "hi"

MLX group-quant bits are {2,3,4,5,6,8}; we mix 4/8 (like the GPTQ lane) by default. A LADDER of rungs
is --bits N per rung: the body at N bits, the measured-hot layers one step up (--high-bits to override),
--bits 8 = uniform 8-bit. --plan-only needs no deps; the actual convert needs `pip install mlx_lm` on
Apple Silicon (this box has `mlx`).

  pollard-mlx --model <hf> --sensitivity s.json --bits 3 --out ./M-Pollard-MLX-3bit   # one ladder rung
  pollard-mlx --model <hf> --bits 4 --imatrix model.imatrix --out ./M-MLX-4bit          # weighted range search

--range search (default) picks each group's scale/bias by an imatrix-weighted range search instead of min/max;
the files are standard MLX affine (mlx-lm loads them unchanged). --range minmax = MLX's own choice."""
import argparse, json, os, re, struct, sys

HIGH, LOW = 8, 4
MLX_BITS = (2, 3, 4, 5, 6, 8)


#: rung body bits -> hot-layer bits: the protected set stays well above the body at the low rungs
RUNG_HIGH = {2: 4, 3: 6, 4: 8, 5: 8, 6: 8, 8: 8}


def rung_bits(bits, high=None):
    """(low, high) for a ladder rung: the body at `bits`, the measured-hot layers at RUNG_HIGH[bits]."""
    if bits not in MLX_BITS:
        raise ValueError(f"MLX quantizes to {MLX_BITS} bits, not {bits}")
    high = RUNG_HIGH[bits] if high is None else high
    if high not in MLX_BITS or high < bits:
        raise ValueError(f"--high-bits must be one of {MLX_BITS} and >= --bits")
    return bits, high


def detect_moe(model_id, layers_hint=0):
    import pollard_workspace as _ws
    cfg = _ws.text_config_dict(_ws.config_dict(model_id))    # plain JSON: no checkpoint code runs to read it
    if not cfg:
        return False, layers_hint
    blocks = cfg.get("block_configs") or cfg.get("layers_block_type") or cfg.get("hybrid_override_pattern")
    n = cfg.get("num_hidden_layers", 0) or (len(blocks) if blocks else 0) or layers_hint   # Bolt lists blocks only
    moe = any(isinstance(cfg.get(k), int) and cfg.get(k) > 1
              for k in ("num_experts", "num_local_experts", "n_routed_experts"))
    return bool(moe), int(n)


def allocate(sens, n_layers, hot_frac, focus=None, low=None, high=None):
    """Pollard profile -> {layer: {"attn":bits, "ffn":bits}}. Top `hot_frac` of layers stay HIGH.
    `focus` (layer indices) are forced HIGH regardless of the profile (steer the budget)."""
    LOW, HIGH = low or globals()["LOW"], high or globals()["HIGH"]
    focus = set(focus or ())
    ffn = {int(k): float(v) for k, v in (sens.get("ffn") or {}).items()}
    attn = {int(k): float(v) for k, v in (sens.get("attn") or {}).items()}
    if not ffn and not attn:
        return {i: {"attn": HIGH if i in focus else LOW,
                    "ffn": HIGH if i in focus else LOW} for i in range(n_layers)}

    def hot(d):
        k = max(1, int(round(hot_frac * len(d))))
        return set(sorted(d, key=lambda i: d[i], reverse=True)[:k])
    ha, hf = hot(attn or ffn) | focus, hot(ffn or attn) | focus
    return {i: {"attn": HIGH if i in ha else LOW, "ffn": HIGH if i in hf else LOW}
            for i in range(n_layers)}


def layer_of(path):
    m = re.search(r"\.layers\.(\d+)\.", path)
    return int(m.group(1)) if m else None


# A vision or audio tower lives INSIDE the checkpoint on this lane -- unlike GGUF, where the
# projector ships as a separate mmproj that Pollard never quantizes. Nothing below named them, so
# they fell through: `visual.merger.mlp.0` (the PROJECTOR) matched no rule and took LOW, and
# `vision_tower.encoder.layers.2.*` matched `.layers.2.` and took the LANGUAGE layer 2's
# allocation -- a text-sensitivity profile applied to a vision block.
# The encoder and the projector are what carry modality alignment; a model that cannot tell red
# from blue has lost something no text perplexity will show. Held high by default, and
# --vision-bits exists because compressing the tower is a legitimate choice, just not a silent one.
VISION_RE = re.compile(r"(^|\.)(visual|vision_tower|vision_model|vision_encoder|image_encoder"
                       r"|patch_embed|audio_tower|audio_encoder|speech_encoder)", re.I)
PROJ_RE = re.compile(r"(multi_modal_projector|mm_projector|mm_proj|\bmerger\b"
                     r"|modality_project|resampler|perceiver|connector|(^|\.)mlp1\.|sound_projection)", re.I)
# (`mlp1` is the InternVL-style projector Nemotron omni / Bolt ship; mlx-vlm keeps the name as mlp1.layers.N)
# A speech-output stack (Qwen3-Omni's talker / code2wav) has decoder layers of its own: `talker.model.layers.3.`
# would take the language layer 3's allocation. It is left at the checkpoint's precision.
SPEECH_RE = re.compile(r"(^|\.)(talker|code2wav|token2wav|code_predictor|speech_decoder|vocoder)(\.|$)", re.I)


EXIT_HELD = 3


def mlx_engine(model_dir):
    """Which converter ships this checkpoint WHOLE: ("mlx_lm", None) for a text model; ("mlx_vlm", model_type) for a
    model with a vision / audio tower, projector or speech output (mlx-lm drops every one of them); or
    ("hold", why) when mlx-vlm is missing or has no such model."""
    import pollard_complete as PC
    comps = PC.inventory(model_dir)["components"] if os.path.isdir(model_dir) else {}
    if not comps:
        return "mlx_lm", None
    parts = ", ".join(f"{k} ({len(v)} tensors)" for k, v in comps.items())
    try:
        from mlx_vlm.utils import get_model_type
    except Exception:                                       # noqa: BLE001
        return "hold", (f"this model has {parts}; mlx-lm would drop them and mlx-vlm is not installed "
                        "(pip install mlx-vlm)")
    import importlib, json as _j
    cfg = _j.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
    mt = get_model_type(cfg)
    try:
        mod = importlib.import_module(f"mlx_vlm.models.{mt}")
    except Exception as e:                                  # noqa: BLE001
        hint = f" -- {MLX_VLM_NEWER}" if cfg.get("model_type") == "nemotron_h_omni" else ""
        return "hold", f"this model has {parts}; mlx-vlm has no {mt!r} model ({type(e).__name__}){hint}"
    import pollard_workspace as _ws
    if _ws.text_config_dict(cfg).get("block_configs"):
        tc = getattr(mod, "TextConfig", None)
        if tc is None or not hasattr(tc, "for_layer"):     # an older mlx-vlm built every layer from ONE config
            return "hold", (f"this checkpoint gives each layer its own block config (NVIDIA puzzle: expert count / "
                            f"width per layer) and this mlx-vlm's {mt!r} builds every layer from one -- "
                            f"{MLX_VLM_NEWER}")
    return "mlx_vlm", mt


def needs_mlx_vlm(model_dir):
    """True when this checkpoint carries a vision / audio tower, projector or talker: its MLX lane is mlx-vlm (decided
    from the safetensors index and config, nothing imported)."""
    import pollard_complete as PC
    return bool(os.path.isdir(model_dir) and PC.inventory(model_dir)["components"])


#: Nemotron-3.5-Bolt (nemotron_h_omni over a nemotron_h_puzzle decoder) landed in mlx-vlm after 0.7.8 (#2528)
MLX_VLM_NEWER = ('needs an mlx-vlm newer than 0.7.8: pip install -U "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm"')


def encoded_source(model_dir):
    """The checkpoint's already-quantized linears -- {module: encoding} (modelopt NVFP4 / FP8, ggml blocks packed in
    safetensors as NVIDIA's Bolt-24GB ships its experts, ...); {} for a float checkpoint."""
    if not os.path.isdir(model_dir):
        return {}
    try:
        import pollard_stream as S
        return S.Source(model_dir).modules
    except (SystemExit, Exception):                         # noqa: BLE001 -- unreadable = let the converter say
        return {}


def decode_to_float(model_dir, out, log=print):
    """A BF16 copy of an already-quantized checkpoint, for a converter that reads float weights: every encoded linear
    decoded (pollard_stream's readers -- the same decode the streamed lanes use), every other tensor byte-for-byte,
    side files copied, the quantization config dropped. The disk it needs is checked first."""
    import shutil
    import pollard_stream as S
    src = S.Source(model_dir)
    need = sum((S._prod(src.shape(k)) * 2 if src.encoding(k) else src.nbytes(k)) for k in src.logical)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(out)) or ".").free
    if free < need + (2 << 30):
        raise SystemExit(f"ERROR: decoding the quantized source to BF16 for MLX needs {need / 1e9:.1f} GB at {out} "
                         f"and {free / 1e9:.1f} GB is free. Free the space, or start from the float release "
                         f"(for Bolt: nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16).")
    log(f"   source is already quantized ({len(src.modules)} encoded linears): decoding to BF16 -> {out} "
        f"({need / 1e9:.2f} GB) for mlx-vlm")
    w = S.ShardWriter(out)
    for k in sorted(src.logical):
        if src.encoding(k) is None and src.dtype(k) not in S.FLOAT_DT:
            S.copy_raw(src, w, k)                           # integer side tensors (RADIO's summary_idxs)
        else:
            S.write_float(src, w, k, "BF16")
    w.close()
    S.copy_side_files_of(src, out)
    cfg_p = os.path.join(out, "config.json")
    cfg = json.load(open(cfg_p, encoding="utf-8"))
    cfg.pop("quantization_config", None)
    json.dump(cfg, open(cfg_p, "w", encoding="utf-8"), indent=1)
    hq = os.path.join(out, "hf_quant_config.json")
    if os.path.exists(hq):
        os.remove(hq)
    return out


def bits_for(path, alloc, is_moe, gsize, vision_bits=None, low=None, high=None):
    """Per-module bit-width for MLX (returns a {'bits','group_size'} dict, or False to skip).
    Mirrors the Pollard policy: attn follows the attn profile; the FFN/expert body follows ffn;
    MoE router (`.gate`, not `gate_proj`) and shared experts are pinned HIGH; embeddings kept HIGH.
    The vision/audio tower and the modality projector are held at `vision_bits` (default HIGH) --
    they are matched FIRST, because `vision_tower.encoder.layers.2.` also matches the language
    layer pattern and would otherwise be given a text layer's allocation."""
    LOW, HIGH = low or globals()["LOW"], high or globals()["HIGH"]
    if SPEECH_RE.search(path):                          # speech output: its own layers, not the text profile
        return False
    if PROJ_RE.search(path):                            # modality alignment: never crushed, any rung
        return {"bits": globals()["HIGH"], "group_size": gsize}
    if VISION_RE.search(path):
        return {"bits": vision_bits or globals()["HIGH"], "group_size": gsize}
    if "embed" in path or path.endswith("lm_head"):    # embed_tokens (llama) OR embedding (Spark2_5, tied)
        return {"bits": HIGH, "group_size": gsize}      # vocab carriers: keep high
    if is_moe and (re.search(r"\.(mlp|block_sparse_moe|feed_forward)\.gate$", path)
                   or "shared_expert" in path):
        return {"bits": HIGH, "group_size": gsize}      # router + shared: selection integrity
    if "self_attn" in path and path.endswith("g_proj"):  # head-wise attn output gate (Spark2_5): selection-critical
        return {"bits": HIGH, "group_size": gsize}
    li = layer_of(path)
    if li is None or li not in alloc:
        return {"bits": LOW, "group_size": gsize}
    if "self_attn" in path or re.search(r"\.(mixer|conv)\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj)$", path):
        return {"bits": alloc[li]["attn"], "group_size": gsize}   # NemotronH Mamba2 / LFM2 short-conv token mixers
    if ("mlp" in path or "block_sparse_moe" in path or "experts" in path or "latent_proj" in path
            or "feed_forward" in path):
        return {"bits": alloc[li]["ffn"], "group_size": gsize}
    return {"bits": LOW, "group_size": gsize}


# ----------------------------------------------------------------- range search (imatrix-weighted)
#
# MLX affine group quant stores, per group of 64 weights, a scale and a bias: w ~= scale * q + bias, q in
# [0, 2^bits). mx.quantize takes them from the group's min and max. That spends the grid on the extremes: one
# outlier stretches the range and every other weight in the group lands on a coarser grid. The GPQ experiment
# (Graft, 2026-10-15, Qwen3-0.6B) measured a per-group range SEARCH -- shrink the low and high ends over a small
# grid, keep the pair with the least error WEIGHTED by each input channel's activation energy (the imatrix) --
# at about half the KL of min/max at 4 bits (0.147-0.178 vs 0.322).
#
# This is that search, clean-room for this lane: the OUTPUT is unchanged standard MLX affine (uint32 codes in
# MLX's little-endian bitstream, scale/bias per group in the model dtype), so mlx-lm loads it as any other
# quantized model -- only which scale and bias each group gets is different. MLX's own min/max answer is
# always one of the candidates, so no group ever ends up worse than mx.quantize on the weighted error.

#: fractions of the group's range shaved off the low / high end (GPQ's grid; 0 = min/max itself)
SHRINK = (0.0, 0.02, 0.04, 0.07, 0.10, 0.14, 0.19, 0.25)
RANGE_MODES = ("search", "minmax")


def _round_store(x, dtype, xp):
    """Round to the dtype the scales/biases are stored in -- the error has to be measured on what ships."""
    name = str(dtype).split(".")[-1]
    if xp.__name__.startswith("mlx"):
        return x.astype(getattr(xp, name)).astype(xp.float32) if name != "float32" else x
    import numpy as np
    if name == "float16":
        return x.astype(np.float16).astype(np.float32)
    if name == "bfloat16":                                        # numpy has no bf16: round-to-nearest-even
        u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
        return u.view(np.float32)
    return x.astype(np.float32)


def _codes(g, s, b, n, xp):
    q = xp.round((g - b[..., None]) / s[..., None])
    return xp.clip(q, 0, n)


def _err(g, q, s, b, imp):
    e = g - (q * s[..., None] + b[..., None])
    e = e * e
    if imp is not None:
        e = e * imp
    return e.sum(-1)


def search_groups(g, bits, imp=None, store="float32", xp=None, grid=SHRINK, refits=2, init=None):
    """Per-group (scale, bias) for g [..., G] (float32) minimising sum(imp * (w - (s*q + b))^2).

    Candidates: every (lo, hi) shrink pair on `grid`, plus `init` (MLX's own min/max scale/bias) when given;
    then `refits` rounds of weighted least squares for (s, b) at the chosen codes, kept only where they
    lower the error. Scales/biases are rounded to `store` before scoring. Returns (q, s, b, err)."""
    import numpy as np
    xp = xp or np
    n = 2 ** bits - 1
    mn, mxv = g.min(-1), g.max(-1)
    R = xp.maximum(mxv - mn, 1e-12)
    best = None
    cands = [(a, b_) for a in grid for b_ in grid]
    for a, b_ in cands:
        lo, hi = mn + a * R, mxv - b_ * R
        s = _round_store(xp.maximum((hi - lo) / n, 1e-12), store, xp)
        b = _round_store(lo, store, xp)
        q = _codes(g, s, b, n, xp)
        e = _err(g, q, s, b, imp)
        if best is None:
            best = [q, s, b, e]
            continue
        m = e < best[3]
        best = [xp.where(m[..., None], q, best[0]), xp.where(m, s, best[1]), xp.where(m, b, best[2]),
                xp.where(m, e, best[3])]
        if hasattr(xp, "eval"):                                   # MLX is lazy: keep the graph one step deep
            xp.eval(*best)
    if init is not None:                                          # MLX's own answer is always on the table
        s0, b0 = (_round_store(x.astype(xp.float32), store, xp) for x in init)
        s0 = xp.where(s0 == 0, 1e-12, s0)
        q0 = _codes(g, s0, b0, n, xp)
        e0 = _err(g, q0, s0, b0, imp)
        m = e0 < best[3]
        best = [xp.where(m[..., None], q0, best[0]), xp.where(m, s0, best[1]), xp.where(m, b0, best[2]),
                xp.where(m, e0, best[3])]
    w = imp if imp is not None else 1.0
    for _ in range(refits):                                       # weighted LS refit at the chosen codes
        q = best[0]
        wq = w * q
        Sw = (w * xp.ones_like(g)).sum(-1)
        Sq, Sqq = wq.sum(-1), (wq * q).sum(-1)
        Sg, Sqg = (w * g).sum(-1), (wq * g).sum(-1)
        det = Sw * Sqq - Sq * Sq
        ok = xp.abs(det) > 1e-20
        safe = xp.where(ok, det, 1.0)
        s = xp.where(ok, (Sw * Sqg - Sq * Sg) / safe, best[1])
        b = xp.where(ok, (Sg - s * Sq) / xp.maximum(Sw, 1e-30), best[2])
        s = _round_store(xp.where(s > 1e-12, s, best[1]), store, xp)
        b = _round_store(b, store, xp)
        q = _codes(g, s, b, n, xp)
        e = _err(g, q, s, b, imp)
        m = e < best[3]
        best = [xp.where(m[..., None], q, best[0]), xp.where(m, s, best[1]), xp.where(m, b, best[2]),
                xp.where(m, e, best[3])]
    return best


def pack_mlx(q, bits):
    """uint8/int codes [..., K] -> MLX's packed uint32 [..., K*bits/32]: a little-endian bitstream (element 0
    in the lowest bits of word 0), the layout mx.quantize writes for every width 2/3/4/5/6/8."""
    import numpy as np
    q = np.asarray(q).astype(np.uint32)
    lead, K = q.shape[:-1], q.shape[-1]
    if (K * bits) % 32:
        raise ValueError(f"K={K} at {bits} bits does not fill whole uint32 words")
    bitsarr = ((q[..., None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
    packed = np.packbits(bitsarr.reshape(*lead, K * bits), axis=-1, bitorder="little")
    return np.ascontiguousarray(packed).view(np.uint32).reshape(*lead, K * bits // 32)


def unpack_mlx(packed, bits, K):
    """The inverse of pack_mlx (for tests and audits)."""
    import numpy as np
    p = np.ascontiguousarray(np.asarray(packed, dtype=np.uint32))
    lead = p.shape[:-1]
    b = np.unpackbits(p.view(np.uint8).reshape(*lead, -1), axis=-1, bitorder="little")[..., :K * bits]
    b = b.reshape(*lead, K, bits).astype(np.uint32)
    return (b << np.arange(bits, dtype=np.uint32)).sum(-1)


def encode_affine(w, bits, group_size=64, imp=None, method="search", store=None, xp=None, init=None,
                  max_elems=1 << 22):
    """Quantize w [..., K] to standard MLX affine: (packed uint32 [..., K*bits/32], scales, biases [..., K/G]).

    method "minmax" = the classic min/max grid; "search" = the imatrix-weighted range search. `imp` = per input
    channel importance [K] (activation E[x^2], an imatrix row), or [..., K] per row block. Works on numpy or,
    with xp=mlx.core, on the GPU. Rows are processed in chunks of ~max_elems weights -- cast to float32, searched and
    packed one chunk at a time, so the working set is one chunk, never a float32 copy of the whole tensor. `init` =
    MLX's own (scales, biases) for the whole tensor, or a callable (row0, row1) -> the same for those rows."""
    import numpy as np
    xp = xp or np
    if bits not in MLX_BITS:
        raise ValueError(f"MLX affine takes {MLX_BITS} bits, not {bits}")
    store = store or str(getattr(w, "dtype", "float32")).split(".")[-1]
    shape = tuple(w.shape)
    K = shape[-1]
    if K % group_size:
        raise ValueError(f"last dim {K} not divisible by group_size {group_size}")
    rows = 1
    for d in shape[:-1]:
        rows *= d
    W = w.reshape(rows, K // group_size, group_size)
    I = None
    if imp is not None:
        I = xp.array(imp) if xp is not np else np.asarray(imp, dtype=np.float32)
        I = I.astype(xp.float32)
        if I.size == K:
            I = I.reshape(1, K // group_size, group_size)
        elif I.size == rows * K:
            I = I.reshape(rows, K // group_size, group_size)
        elif I.size % K == 0 and rows % (I.size // K) == 0:      # per expert [E, K] over [E, out, K] rows
            e = I.size // K
            I = xp.repeat(I.reshape(e, 1, K // group_size, group_size), rows // e, axis=1).reshape(
                rows, K // group_size, group_size)
        else:
            raise ValueError(f"importance has {I.size} values; weight needs {K} (per channel)")
        I = xp.maximum(I, 0.0) + 1e-8 * (I.mean() if I.size else 1.0)
    get_init = None
    if callable(init):
        get_init = init
    elif init is not None:
        s0, b0 = init
        s0, b0 = s0.reshape(rows, K // group_size), b0.reshape(rows, K // group_size)
        get_init = lambda r0, r1: (s0[r0:r1], b0[r0:r1])               # noqa: E731
    step = max(1, max_elems // max(1, K))
    ps, ss, bs = [], [], []
    for r in range(0, rows, step):
        g = W[r:r + step].astype(xp.float32)
        ii = None if I is None else (I if I.shape[0] == 1 else I[r:r + step])
        if method == "minmax":
            q, s_, b_, _ = search_groups(g, bits, ii, store, xp, grid=(0.0,), refits=0)
        else:
            q, s_, b_, _ = search_groups(g, bits, ii, store, xp,
                                         init=None if get_init is None else get_init(r, min(rows, r + step)))
        q8 = np.asarray(q.astype(xp.uint8)) if xp is not np else q.astype(np.uint8)
        ps.append(pack_mlx(q8.reshape(-1, K), bits))
        if hasattr(xp, "eval"):                                   # MLX is lazy: drop the chunk's graph now
            xp.eval(s_, b_)
        ss.append(s_)
        bs.append(b_)
        del g, q, q8
    cat = (lambda a: xp.concatenate(a, 0)) if xp is not np else (lambda a: np.concatenate(a, 0))
    s = cat(ss).reshape(*shape[:-1], K // group_size)
    b = cat(bs).reshape(*shape[:-1], K // group_size)
    return np.concatenate(ps, 0).reshape(*shape[:-1], K * bits // 32), s, b


def weighted_error(w, packed, scales, biases, bits, group_size=64, imp=None):
    """sum(imp * (w - dequant)^2) of a standard-MLX encoding, in numpy (for tests and the plan report)."""
    import numpy as np
    w = np.asarray(w, dtype=np.float32)
    K = w.shape[-1]
    q = unpack_mlx(packed, bits, K).astype(np.float32).reshape(*w.shape[:-1], K // group_size, group_size)
    deq = q * np.asarray(scales, np.float32)[..., None] + np.asarray(biases, np.float32)[..., None]
    e = (w.reshape(deq.shape) - deq) ** 2
    if imp is not None:
        e = e * np.asarray(imp, np.float32).reshape(1, K // group_size, group_size)
    return float(e.sum())


# GGUF imatrix names -> the HF / mlx-lm module path they describe (llama-family layout)
_GGUF_TO_HF = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
               "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
               "ffn_down": "mlp.down_proj", "ffn_gate_exps": "mlp.switch_mlp.gate_proj",
               "ffn_up_exps": "mlp.switch_mlp.up_proj", "ffn_down_exps": "mlp.switch_mlp.down_proj"}


def gguf_name_to_path(name):
    """blk.3.attn_q.weight -> model.layers.3.self_attn.q_proj ; output.weight -> lm_head ; else None."""
    n = name[:-len(".weight")] if name.endswith(".weight") else name
    if n == "output":
        return "lm_head"
    m = re.match(r"blk\.(\d+)\.(\w+)$", n)
    if m and m.group(2) in _GGUF_TO_HF:
        return f"model.layers.{m.group(1)}.{_GGUF_TO_HF[m.group(2)]}"
    return None


def _read_imatrix(path):
    """GGUF tensor name -> mean x^2 per input channel, from EITHER llama.cpp imatrix format (torch-free)."""
    import numpy as np
    try:                                                    # legacy .dat (ik_llama's and ours)
        from pollard_calc import read_imatrix_legacy
        raw = read_imatrix_legacy(path)
        return {n: np.asarray(e["values"], np.float64) / max(e["ncall"], 1) for n, e in raw.items() if e["values"]}
    except Exception:                                       # noqa: BLE001
        pass
    from gguf import GGUFReader                             # llama.cpp's GGUF imatrix
    rd = GGUFReader(path)
    sums = {t.name[:-len(".in_sum2")]: t for t in rd.tensors if t.name.endswith(".in_sum2")}
    cnts = {t.name[:-len(".counts")]: t for t in rd.tensors if t.name.endswith(".counts")}
    out = {}
    for n, t in sums.items():
        c = float(cnts[n].data.reshape(-1)[0]) if n in cnts else 1.0
        out[n] = t.data.reshape(-1).astype(np.float64) / max(c, 1.0)
    return out


def load_importance(path):
    """{module path: per-input-channel E[x^2]} from a llama.cpp imatrix (legacy .dat or GGUF; GGUF tensor
    names mapped to module paths) or an .npz / .safetensors keyed by module path."""
    import numpy as np
    if path.endswith(".npz"):
        z = np.load(path)
        return {k: np.asarray(z[k], np.float32) for k in z.files}
    if path.endswith(".safetensors"):
        from safetensors.numpy import load_file
        return {k: np.asarray(v, np.float32) for k, v in load_file(path).items()}
    out = {}
    for name, vals in _read_imatrix(path).items():
        p = gguf_name_to_path(name)
        if p:
            out[p] = np.asarray(vals, np.float32)
    return out


class RangeSearchQuantize:
    """Context manager: while active, mlx-lm's quantization goes through encode_affine (method "search").

    mlx-lm calls quant_predicate(path, module) and then module.to_quantized(), which calls mx.quantize on that
    module's weight -- so the predicate records the path (and the weight shape, to be sure) and the patched
    mx.quantize looks up that path's imatrix row. Everything else (mode != affine, a shape that does not match)
    falls through to MLX unchanged; the files mlx-lm writes are standard MLX affine either way."""

    def __init__(self, importance=None, method="search"):
        self.importance, self.method = importance or {}, method
        self.current, self.current_w = None, None
        self.stats = {"searched": 0, "weighted": 0, "passthrough": 0}

    def note(self, path, module):
        w = getattr(module, "weight", None)
        self.current = (path, tuple(w.shape) if w is not None else None)
        self.current_w = w

    def __enter__(self):
        import mlx.core as mx
        self._orig = mx.quantize

        def quantize(w, group_size=None, bits=None, mode="affine", **kw):
            gs, b = group_size or 64, bits or 4
            if mode != "affine" or kw.get("global_scale") is not None or b not in MLX_BITS:
                self.stats["passthrough"] += 1
                return self._orig(w, group_size, bits, mode=mode, **kw)
            # QuantizedLinear.__init__ quantizes a RANDOM placeholder before from_linear quantizes the real
            # weight: only the module's own weight array is worth searching
            if w is not getattr(self, "current_w", None):
                self.stats["passthrough"] += 1
                return self._orig(w, group_size, bits, mode=mode, **kw)
            path, shape = self.current or (None, None)
            imp = self.importance.get(path) if (path and shape == tuple(w.shape)) else None
            if imp is not None and imp.size not in (w.shape[-1], w.size) and imp.size % w.shape[-1]:
                imp = None
            init = self._orig(w, gs, b)[1:]
            packed, s, bb = encode_affine(w, b, gs, imp=imp, method=self.method, xp=mx, init=init)
            self.stats["searched"] += 1
            self.stats["weighted"] += imp is not None
            return mx.array(packed), s.astype(w.dtype), bb.astype(w.dtype)
        mx.quantize = quantize
        return self

    def __exit__(self, *exc):
        import mlx.core as mx
        mx.quantize = self._orig
        return False



# ----------------------------------------------------------------- streamed build: one tensor in memory at a time
#
# mlx-lm and mlx-vlm already build the model LAZILY: mx.load reads nothing until an array is evaluated, sanitize
# (expert stacking, renames, transposes) and nn.quantize only add graph nodes. Two things made a big model sit in
# memory whole anyway:
#   * the range search ran eagerly inside mx.quantize (numpy in the loop), so every quantized tensor was computed and
#     held before the first byte was written -- the whole quantized model in RAM (LFM2.5-24B: ~13.6 GB at 3 bits);
#   * mx.save_safetensors evaluates a whole shard (up to 5 GB of output, and every source tensor feeding it) at once.
# The streamed build keeps both converters' own code -- load, sanitize, quant predicate, config, tokenizer, processor
# files -- and replaces exactly those two calls: mx.quantize returns MLX's own LAZY answer as a placeholder (and notes
# which module's weight it stands for), and mx.save_safetensors writes the header from the shapes, then evaluates and
# writes ONE tensor at a time (running the range search for a placeholder right there), freeing it before the next.
# Peak memory = one source tensor (an expert stack: one projection of one layer) + its encoding, preflighted
# (plan_build) and guarded (free RAM checked while writing). The bytes are the non-streamed build's: same header
# (MLX's own key order, read from a one-element probe file), same encoding (encode_affine, chunk for chunk).

STREAM_MODES = ("auto", "on", "off")
#: python + mlx + the converter's imports + tokenizer / processor, measured ~0.6-0.9 GB on an M-series Mac
BASE_GB = 0.9
#: the range search's per-chunk working set (a 4M-weight chunk, its candidates and their errors), GB
SEARCH_GB = 0.35
_WRITTEN = None


class StreamedQuantize(RangeSearchQuantize):
    """While active, mx.quantize is MLX's own -- lazy -- and, for --range search, its outputs are placeholders: the
    writer (StreamWriter) replaces each with encode_affine's answer when it reaches that tensor."""

    def __init__(self, importance=None, method="search"):
        super().__init__(importance, method)
        self.pending = {}                      # id(placeholder) -> (job, index, placeholder)

    def __enter__(self):
        import mlx.core as mx
        self._orig = mx.quantize

        def quantize(w, group_size=None, bits=None, mode="affine", **kw):
            gs, b = group_size or 64, bits or 4
            out = self._orig(w, group_size, bits, mode=mode, **kw)
            if (self.method != "search" or mode != "affine" or kw.get("global_scale") is not None
                    or b not in MLX_BITS or w is not getattr(self, "current_w", None)):
                self.stats["passthrough"] += 1
                return out
            path, shape = self.current or (None, None)
            imp = self.importance.get(path) if (path and shape == tuple(w.shape)) else None
            if imp is not None and imp.size not in (w.shape[-1], w.size) and imp.size % w.shape[-1]:
                imp = None
            out = (out[0], out[1].astype(w.dtype), out[2].astype(w.dtype))
            job = {"w": w, "bits": b, "gs": gs, "imp": imp, "path": path, "out": None, "left": 3}
            for i, a in enumerate(out):
                self.pending[id(a)] = (job, i, a)
            self.stats["searched"] += 1
            self.stats["weighted"] += imp is not None
            return out
        mx.quantize = quantize
        return self

    def source_of(self, a):
        """The source weight a placeholder stands for (None once encoded, or for any other array)."""
        ent = self.pending.get(id(a))
        return ent[0]["w"] if ent is not None and ent[2] is a else None

    def take(self, a):
        """The final array for placeholder `a` (the search runs at the first of its three outputs), else None."""
        ent = self.pending.get(id(a))
        if ent is None or ent[2] is not a:
            return None
        job, i, _ = ent
        if job["out"] is None:
            job["out"] = self.encode(job)
            job["w"] = None
        out = job["out"][i]
        del self.pending[id(a)]
        job["left"] -= 1
        if not job["left"]:
            job["out"] = None
        return out

    def encode(self, job):
        import mlx.core as mx
        w, gs, b = job["w"], job["gs"], job["bits"]
        mx.eval(w)                                         # the source tensor (an expert stack) once, then by rows
        K = w.shape[-1]
        w2 = w.reshape(-1, K)
        orig = self._orig
        init = (lambda r0, r1: orig(w2[r0:r1], gs, b)[1:])  # MLX's own answer, chunk by chunk
        packed, s, bb = encode_affine(w, b, gs, imp=job["imp"], method=self.method, xp=mx, init=init)
        out = (mx.array(packed), s.astype(w.dtype), bb.astype(w.dtype))
        mx.eval(*out)
        return out


def _np_bytes(a):
    """A materialized mx.array as a contiguous numpy view of its bytes (bf16 through uint16)."""
    import mlx.core as mx
    import numpy as np
    if a.dtype == mx.bfloat16:
        return np.ascontiguousarray(np.array(a.view(mx.uint16), copy=False))
    return np.ascontiguousarray(np.array(a, copy=False))


class MemoryGuard:
    """Stops a streamed step cleanly when free RAM drops under `min_free_gb` (checked at most every `every_s`), and
    keeps the numbers: peak MLX memory, peak RSS, lowest free RAM seen."""

    def __init__(self, min_free_gb=None, every_s=2.0, log=print):
        self.min_free = float(os.environ.get("POLLARD_MIN_FREE_GB", 2.0)) if min_free_gb is None else min_free_gb
        self.every, self.log = every_s, log
        self.last, self.low = 0.0, None

    def check(self, what, force=False):
        import time
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        import pollard_stream as S
        free = S.free_ram_gb()
        if free is None:
            return
        self.low = free if self.low is None else min(self.low, free)
        if free < self.min_free:
            raise S.OutOfMemoryBudget(f"stopped at {what}: free RAM is {free:.2f} GB, under the {self.min_free:g} GB "
                                      "floor -- nothing more is written; free memory and run it again")

    def numbers(self):
        import pollard_stream as S
        try:
            import mlx.core as mx
            mlx_peak = mx.get_peak_memory() / 1e9
        except Exception:                                  # noqa: BLE001
            mlx_peak = None
        return {"peak_rss_gb": round(S.peak_rss_bytes() / 1e9, 2),
                "peak_mlx_gb": None if mlx_peak is None else round(mlx_peak, 2),
                "min_free_ram_gb": None if self.low is None else round(self.low, 2)}


class StreamWriter:
    """While active, mx.save_safetensors writes a shard one tensor at a time (same file bytes as MLX's own)."""

    def __init__(self, quantizer=None, guard=None, log=print, cache_limit_gb=0.25, mirror=None, out_bytes=0):
        self.q, self.guard, self.log = quantizer, guard or MemoryGuard(log=log), log
        self.cache_limit = int(cache_limit_gb * (1 << 30))
        self.n, self.bytes, self.files = 0, 0, 0
        self.mirror, self.out_bytes = mirror, out_bytes      # a remote source: fetch what each tensor reads

    def _reserve_gb(self):
        """Disk the mirror must leave free: the rest of this rung + a margin."""
        return max(0, self.out_bytes - self.bytes) / 1e9 + 3.0

    def __enter__(self):
        import mlx.core as mx
        self._orig = mx.save_safetensors
        mx.save_safetensors = self.save
        try:
            self._old_cache = mx.set_cache_limit(self.cache_limit)
        except Exception:                                  # noqa: BLE001
            self._old_cache = None
        return self

    def __exit__(self, *exc):
        import mlx.core as mx
        mx.save_safetensors = self._orig
        if self._old_cache is not None:
            mx.set_cache_limit(self._old_cache)
        return False

    def mlx_layout(self, arrays, metadata):
        """MLX's own data order and dtype names for this key set: a probe file of one-element arrays written by
        mx.save_safetensors itself (its map iterates the keys in an order only it knows)."""
        import mlx.core as mx
        import tempfile
        probe = {k: mx.zeros((1,), dtype=v.dtype) for k, v in arrays.items()}
        fd, tmp = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        try:
            self._orig(tmp, probe, metadata=metadata) if metadata is not None else self._orig(tmp, probe)
            import pollard_stream as S
            hdr, _ = S.st_header(tmp)
        finally:
            os.remove(tmp)
        hdr.pop("__metadata__", None)
        order = sorted(hdr, key=lambda k: hdr[k]["data_offsets"][0])
        return order, {k: hdr[k]["dtype"] for k in hdr}

    def save(self, file, arrays, metadata=None):
        import mlx.core as mx
        import time
        file = str(file)
        if not file.endswith(".safetensors"):
            file += ".safetensors"
        order, dts = self.mlx_layout(arrays, metadata)
        hdr, off = {}, 0
        for k in order:
            n = int(arrays[k].nbytes)
            hdr[k] = {"data_offsets": [off, off + n], "dtype": dts[k], "shape": list(arrays[k].shape)}
            off += n
        if metadata:
            hdr["__metadata__"] = dict(metadata)
        h = json.dumps(hdr, sort_keys=True, separators=(",", ":")).encode()
        tmp = file + ".partial"
        t0, total = time.time(), len(order)
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(h)))
            f.write(h)
            for j, k in enumerate(order):
                a = arrays[k]
                deps = set()
                if self.mirror is not None:
                    src = self.q.source_of(a) if self.q is not None else None
                    deps = self.mirror.deps(src if src is not None else a)
                    if deps:
                        self.mirror.ensure(deps, reserve_gb=self._reserve_gb())
                got = self.q.take(a) if self.q is not None else None
                if got is None:
                    mx.eval(a)
                    got = a
                arrays[k] = _WRITTEN                       # the caller only reads the keys back
                del a
                buf = _np_bytes(got)
                if buf.nbytes != hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0]:
                    raise RuntimeError(f"{k}: {buf.nbytes} bytes, the header says "
                                       f"{hdr[k]['data_offsets'][1] - hdr[k]['data_offsets'][0]}")
                f.write(memoryview(buf).cast("B"))
                if deps:
                    self.mirror.done(deps)
                self.n += 1
                self.bytes += buf.nbytes
                del got, buf
                if mx.get_cache_memory() > self.cache_limit:
                    mx.clear_cache()
                self.guard.check(f"{os.path.basename(file)}:{k}")
                if j % 400 == 0 or j == total - 1:
                    nb = self.guard.numbers()
                    mir = (f"; mirror fetched {self.mirror.fetched / 1e9:.1f} GB, cached "
                           f"{self.mirror.resident_gb():.1f} GB") if self.mirror is not None else ""
                    self.log(f"   wrote {j + 1}/{total} tensors of {os.path.basename(file)}  "
                             f"({self.bytes / 1e9:.2f} GB, {time.time() - t0:.0f}s; peak MLX {nb['peak_mlx_gb']} GB, "
                             f"peak RSS {nb['peak_rss_gb']} GB, free RAM low {nb['min_free_ram_gb']} GB{mir})",
                             flush=True)
        os.replace(tmp, file)
        self.files += 1


def plan_build(model_dir, low=4, high=8, method="search", encoded=False, avg_bits=None):
    """The streamed build's peak RAM and output size from the shapes on disk (pollard_stream.Source reads headers
    only). The unit of work is one output tensor: an expert stack (one projection of one layer, every expert) or a
    plain tensor -- its source bytes twice (the per-expert reads and the stack live together while it is built) +
    its encoding at the high bits + the range search's chunk + the base process."""
    import pollard_stream as S
    src = S.Source(model_dir)
    groups = {}
    for k in src.logical:
        g = re.sub(r"\.experts\.\d+\.", ".experts.*.", k)
        n = S._prod(src.shape(k))
        b = n * 2 if src.encoding(k) else src.nbytes(k)
        e = groups.setdefault(g, [0, 0, 0])
        e[0] += n
        e[1] += b
        e[2] += 1
    worst, wname, src_bytes, out_bytes = 0.0, None, 0, 0.0
    for g, (n, b, cnt) in groups.items():
        src_bytes += b
        work = b * (2 if cnt > 1 else 1) + n * max(high, 8) / 8
        out_bytes += n * ((avg_bits or low) + 0.5) / 8 if b // max(n, 1) >= 2 and n >= 4096 else b
        if work > worst:
            worst, wname = work, g
    peak = worst / 1e9 + BASE_GB + (SEARCH_GB if method == "search" else 0.0)
    return {"tensors": len(groups), "largest": wname, "largest_gb": round(worst / 1e9, 2),
            "peak_gb": round(peak, 2), "source_gb": round(src_bytes / 1e9, 2), "out_gb": round(out_bytes / 1e9, 2),
            "largest_bytes": int(worst), "source_bytes": int(src_bytes), "out_bytes": int(out_bytes),
            "encoded": bool(encoded)}


def preflight_build(plan, out_dir, max_ram_gb=None, log=print):
    """Refuse (OutOfMemoryBudget, with the numbers) a streamed build whose peak does not fit free RAM minus the
    floor, or whose output does not fit the disk."""
    import shutil
    import pollard_stream as S
    floor = float(os.environ.get("POLLARD_MIN_FREE_GB", 2.0))
    free = S.free_ram_gb()
    budget = min([x for x in (max_ram_gb, (free - floor) if free is not None else None) if x is not None] or [None])
    log(f"   memory plan: peak ~{plan['peak_gb']} GB (largest unit {plan['largest']} = {plan['largest_gb']} GB, "
        f"base {BASE_GB} GB) | free RAM {('%.1f' % free) if free is not None else '?'} GB, floor {floor:g} GB"
        + (f", cap {max_ram_gb:g} GB" if max_ram_gb else ""))
    if budget is not None and plan["peak_gb"] > budget:
        raise S.OutOfMemoryBudget(
            f"the streamed MLX build needs ~{plan['peak_gb']} GB at its peak (one tensor: {plan['largest']}, "
            f"{plan['largest_gb']} GB, + {BASE_GB} GB base) and {budget:.1f} GB is usable here "
            f"({'%.1f' % free if free is not None else '?'} GB free, {floor:g} GB floor"
            + (f", --max-ram-gb {max_ram_gb:g}" if max_ram_gb else "") + ")")
    parent = os.path.dirname(os.path.abspath(out_dir)) or "."
    while not os.path.exists(parent) and os.path.dirname(parent) != parent:
        parent = os.path.dirname(parent)
    disk = shutil.disk_usage(parent).free / 1e9
    if disk < plan["out_gb"] + 1.0:
        raise S.OutOfMemoryBudget(f"the MLX rung needs ~{plan['out_gb']:.1f} GB of disk at {out_dir} and "
                                  f"{disk:.1f} GB is free")


# ----------------------------------------------------------------- streamed fidelity: block by block, in MLX
#
# The verify gate compares the rung's logits with the source's over the same held-out chunks. Loading the source
# whole (48 GB of bf16 for LFM2.5, 70 GB for clef-omni) cannot happen on a 16 GB Mac, so the model runs LAYER-MAJOR:
# the decoder blocks are swapped for taps, one forward per chunk records each block's call (its masks, positions,
# rotary tables -- whatever that architecture passes) without running any block, then every chunk goes through
# block 0, block 0 is emptied, every chunk through block 1, ... Only the embeddings, one block and the chunks'
# hidden states are ever in memory; a last forward per chunk with the final hidden state injected runs the model's
# own norm + head. The model code is the converter's own (mlx-lm, or mlx-vlm's language model for a VLM / omni);
# towers, talkers and code2wav are never evaluated.


class _Tap:
    """Stands in for one decoder block: records its call (record), passes the hidden state through, or hands the
    final hidden state to the model's head (final). Attribute reads go to the real block (is_attention_layer, ...)."""

    def __init__(self, inner, i, run):
        self.__dict__.update(_inner=inner, _i=i, _run=run)

    def __getattr__(self, k):
        return getattr(self.__dict__["_inner"], k)

    def __call__(self, x, *args, **kw):
        return self.__dict__["_run"].tap(self.__dict__["_i"], x, args, kw)


def _decoder(model):
    """(language model to call, its block list): the longest list of blocks outside any tower / talker."""
    import mlx.nn as nn
    lm = getattr(model, "language_model", None) or model
    best = None
    for name, mod in lm.named_modules():
        for attr, val in list(mod.items()) if isinstance(mod, dict) else []:
            if isinstance(val, list) and val and all(isinstance(x, nn.Module) for x in val):
                path = f"{name}.{attr}" if name else attr
                if SPEECH_RE.search(path) or VISION_RE.search(path) or PROJ_RE.search(path):
                    continue
                if best is None or len(val) > len(best[1]):
                    best = (path, val)
    if best is None:
        raise SystemExit("could not find the decoder blocks of this MLX model")
    return lm, best[1]


def _lazy_load(path, engine):
    """The model built lazily from `path` (nothing read until evaluated)."""
    from pathlib import Path
    if engine == "mlx_vlm":
        from mlx_vlm.utils import load_model
        return load_model(Path(path), lazy=True)
    from mlx_lm.utils import load_model
    return load_model(Path(path), lazy=True)[0]


def _materialize(x):
    import mlx.core as mx
    if isinstance(x, mx.array):
        mx.eval(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            _materialize(y)
    elif isinstance(x, dict):
        for y in x.values():
            _materialize(y)
    return x


class StreamedLM:
    """One model (source or rung), run layer-major over `rows` (lists of token ids). final_h[b] after run()."""

    def __init__(self, path, engine="mlx_lm", guard=None, log=print):
        import contextlib
        self.path, self.engine, self.log = path, engine, log
        self.guard = guard or MemoryGuard(log=log)
        self.mirror = RemoteMirror.of(path, log) if os.path.isdir(str(path)) else None
        with (self.mirror.hook() if self.mirror is not None else contextlib.nullcontext()):
            self.model = _lazy_load(path, engine)
        self.lm, self.layers = _decoder(self.model)
        self.mode, self.calls, self.h, self.final = None, {}, {}, {}

    def tap(self, i, x, args, kw):
        if self.mode == "record":
            if i == 0:
                self.h[self.b] = _materialize(x)
            self.calls[(self.b, i)] = (_materialize(list(args)), _materialize(dict(kw)))
            return x
        if self.mode == "final" and i == len(self.layers) - 1:
            return self.final[self.b]
        return x

    def _forward(self, ids):
        import mlx.core as mx
        out = self.lm(mx.array([ids]))
        return getattr(out, "logits", out)

    def _swap(self, taps):
        orig = list(self.layers)
        for i in range(len(self.layers)):
            self.layers[i] = _Tap(orig[i], i, self) if taps else orig[i]
        return orig

    def run(self, rows):
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_map
        self.rows = rows
        mir = self.mirror
        if mir is not None:                                # embeddings / final norm / head: kept for the whole run
            blockdeps = set()
            for blk in self.layers:
                blockdeps |= mir.deps(*[v for _, v in tree_flatten(blk.parameters())])
            mir.ensure(mir.deps(*[v for _, v in tree_flatten(self.lm.parameters())]) - blockdeps, reserve_gb=3.0)
        orig = self._swap(True)
        try:
            self.mode = "record"
            for b, ids in enumerate(rows):
                self.b = b
                self._forward(ids)                         # taps only: no block runs, the logits are never evaluated
            self.mode = None
            for i, blk in enumerate(orig):
                deps = mir.deps(*[v for _, v in tree_flatten(blk.parameters())]) if mir is not None else set()
                if deps:
                    mir.ensure(deps, reserve_gb=3.0)
                for _, p in tree_flatten(blk.parameters()):  # one tensor at a time: an expert stack's reads + the
                    mx.eval(p)                             # stack never sit beside the rest of the block's stacks
                for b in range(len(rows)):
                    args, kw = self.calls.pop((b, i))
                    self.h[b] = blk(self.h[b], *args, **kw)
                    mx.eval(self.h[b])
                blk.update(tree_map(lambda _: mx.array([]), blk.parameters()))
                if deps:
                    mir.done(deps)
                mx.clear_cache()
                self.guard.check(f"block {i}")
                if i % 8 == 0 or i == len(orig) - 1:
                    nb = self.guard.numbers()
                    self.log(f"   {os.path.basename(str(self.path).rstrip('/'))}: block {i + 1}/{len(orig)}  "
                             f"(peak MLX {nb['peak_mlx_gb']} GB, peak RSS {nb['peak_rss_gb']} GB, free RAM low "
                             f"{nb['min_free_ram_gb']} GB)", flush=True)
            self.final = dict(self.h)
            self.h = {}
        finally:
            self.mode = None                               # the taps stay: logits() runs the head through them
        return self

    def logits(self, b):
        """float32 numpy [len(row), vocab] for row b: the model's own norm + head on the streamed final state."""
        import mlx.core as mx
        import numpy as np
        self.mode, self.b = "final", b
        try:
            out = self._forward(self.rows[b])[0].astype(mx.float32)
            mx.eval(out)
            return np.array(out)
        finally:
            self.mode = None


def plan_verify(source, rung=None, length=256, chunks=16, vocab=None):
    """Peak RAM of the streamed verify: the biggest decoder block of the source (bf16, + one tensor's transient) and
    its embeddings, the logits of one chunk (source + rung, float32 + the float64 KL math), the base process."""
    import pollard_stream as S
    src = S.Source(source)
    blocks, outer, biggest, vocab_seen = {}, 0, 0, 0
    for k in src.logical:
        n = S._prod(src.shape(k))
        b = n * 2 if src.encoding(k) else src.nbytes(k)
        m = re.match(r"^(.*?(?:layers|h|blocks))\.(\d+)\.", k)
        if m and not SPEECH_RE.search(k) and not VISION_RE.search(k):
            key = (m.group(1), int(m.group(2)))
            blocks[key] = blocks.get(key, 0) + b
            g = re.sub(r"\.experts\.\d+\.", ".experts.*.", k)
            biggest = max(biggest, b)
            blocks[(g, -1)] = blocks.get((g, -1), 0) + b
        elif "embed" in k or k.endswith("lm_head.weight"):
            outer += b
            if len(src.shape(k)) == 2 and not vocab:
                vocab_seen = max(vocab_seen, src.shape(k)[0])
    stacks = [v for (g, i), v in blocks.items() if i == -1]
    layer = max([v for (g, i), v in blocks.items() if i != -1] or [0])
    V = vocab or vocab_seen or 152064
    logits = length * V * 4 * 3 + 64 * V * 8 * 5            # rung + reference + head output; KL 64 rows at a time
    peak = (layer + max(stacks or [biggest]) + outer + logits) / 1e9 + BASE_GB
    return {"layer_gb": round(layer / 1e9, 2), "outer_gb": round(outer / 1e9, 2),
            "logits_gb": round(logits / 1e9, 2), "peak_gb": round(peak, 2)}


# ----------------------------------------------------------------- remote source: a sparse, disk-budgeted mirror
#
# A source that does not fit this disk (clef-omni: 70.8 GB of bf16 on a Mac with ~70 GB free, before the rung) or
# that sits on the build box already is MIRRORED instead of downloaded: the folder holds the small files whole and each
# weight shard as a SPARSE file -- its safetensors header, every tensor's bytes a hole -- so every reader of headers
# (mlx-lm / mlx-vlm loading lazily, pollard_stream.Source, pollard-complete) sees the checkpoint as it is. A tensor's
# bytes are fetched over ssh just before MLX reads them, and kept as a cache within a disk budget (least recently used
# tensors punched out again when the disk is needed). Which tensors an output needs is READ OFF MLX'S GRAPH: every
# array mx.load returns is a Load node, and mx.export_to_dot of an output lists the Load nodes it depends on -- the
# expert stack of one layer, the conv weight a sanitize transposed -- never guessed from names. A read from a hole
# would be zeros, so a Load that is not registered or a byte range not fetched is an error, never a silent zero.

MIRROR_MARK = ".pollard-remote"                  # dotfiles: the converters copy *.json / *.py, never these
MIRROR_STATE = ".pollard-remote-present"
_LOAD_RE = re.compile(r"\{ (\d+) \[label =\"Load\"")
_BLOCK = 4096


def _parse_ssh(url):
    from urllib.parse import urlparse, parse_qs, unquote
    u = urlparse(url)
    if u.scheme != "ssh" or not u.hostname:
        raise SystemExit(f"{url}: expected ssh://user@host[:port]/<remote dir>?python=<python>&key=<identity>")
    q = {k: v[-1] for k, v in parse_qs(u.query).items()}
    path = unquote(u.path)
    return {"host": f"{u.username}@{u.hostname}" if u.username else u.hostname, "port": str(u.port or 22),
            "dir": path[1:] if len(path) > 2 and path[2] == ":" else path, "python": q.get("python", "python3"),
            "key": os.path.expanduser(q["key"]) if q.get("key") else None}


def _load_ids(*arrays):
    """The Load nodes (mx.load reads) the arrays depend on, from MLX's own graph."""
    import io
    import mlx.core as mx
    buf = io.StringIO()
    mx.export_to_dot(buf, *arrays)
    return {int(x) for x in _LOAD_RE.findall(buf.getvalue())}


def punch(path, off, n):
    """Give back the disk blocks lying wholly inside [off, off + n) -- F_PUNCHHOLE on macOS, fallocate(PUNCH_HOLE)
    on Linux; bytes read there become 0. Blocks shared with a neighbouring tensor are never touched."""
    a, b = -(-off // _BLOCK) * _BLOCK, (off + n) // _BLOCK * _BLOCK
    if b <= a:
        return 0
    fd = os.open(path, os.O_RDWR)
    try:
        if sys.platform == "darwin":
            import fcntl
            fcntl.fcntl(fd, 99, struct.pack("IIqq", 0, 0, a, b - a))    # F_PUNCHHOLE
        else:
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]
            if libc.fallocate(fd, 0x03, a, b - a):                     # PUNCH_HOLE | KEEP_SIZE
                raise OSError(ctypes.get_errno(), "fallocate(PUNCH_HOLE)")
    finally:
        os.close(fd)
    return b - a


class RemoteMirror:
    """A local, sparse stand-in for a checkpoint folder on another machine (see above). `ensure(keys)` fetches,
    `done(keys)` unpins (the bytes stay as cache), eviction punches the least recently used when the disk is needed."""

    def __init__(self, local_dir, log=print, runner=None):
        import collections
        self.dir, self.log = str(local_dir), log
        meta = json.load(open(os.path.join(self.dir, MIRROR_MARK), encoding="utf-8"))
        self.url, self.shards = meta["url"], meta["shards"]
        self.ssh = _parse_ssh(self.url)
        self.run_ = runner
        import pollard_stream as S
        self.entries = {}
        for name in self.shards:
            hdr, base = S.st_header(os.path.join(self.dir, name))
            for k, info in hdr.items():
                if k != "__metadata__":
                    a, b = info["data_offsets"]
                    self.entries[k] = (name, base + a, b - a)
        self.present = collections.OrderedDict()               # key -> True, least recently used first
        try:
            for k in json.load(open(os.path.join(self.dir, MIRROR_STATE), encoding="utf-8")):
                if k in self.entries:
                    self.present[k] = True
        except (OSError, ValueError):
            pass
        self.pinned, self.by_load, self.reserve_gb = set(), {}, 3.0
        self.fetched, self.evicted = 0, 0
        import shutil
        self.free_bytes = lambda: shutil.disk_usage(self.dir).free

    @classmethod
    def of(cls, local_dir, log=print):
        return cls(local_dir, log) if os.path.isfile(os.path.join(str(local_dir), MIRROR_MARK)) else None

    # -- ssh
    def _cmd(self):
        s = self.ssh
        return ["ssh"] + (["-i", s["key"]] if s["key"] else []) + ["-o", "BatchMode=yes", "-p", s["port"], s["host"],
                                                                     f'"{s["python"]}" -']

    def _remote_json(self, code):
        import subprocess
        r = (self.run_ or subprocess.run)(self._cmd(), input="import json, os, sys, base64\n" + code,
                                         capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"{self.ssh['host']}: remote python exited {r.returncode}: {(r.stderr or '')[-600:]}")
        return json.loads([x for x in r.stdout.splitlines() if x.strip()][-1])

    def _stream(self, ranges):
        """[(file, offset, n)] read on the remote side, in order, as one byte stream; yields (file, offset, bytes)."""
        import subprocess
        rd = self.ssh["dir"].rstrip("/")
        code = ("import sys\nR = %s\no = sys.stdout.buffer\nfor f, off, n in R:\n"
                "    with open(%r + '/' + f, 'rb') as h:\n        h.seek(off)\n        while n:\n"
                "            b = h.read(min(n, 1 << 24))\n            if not b:\n                sys.exit(5)\n"
                "            o.write(b)\n            n -= len(b)\no.flush()\n") % (json.dumps(ranges), rd)
        p = subprocess.Popen(self._cmd(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.stdin.write(code.encode())
        p.stdin.close()
        try:
            for f, off, n in ranges:
                got, pos = 0, off
                while got < n:
                    b = p.stdout.read(min(n - got, 1 << 24))
                    if not b:
                        raise RuntimeError(f"{self.ssh['host']}: stream ended {n - got} bytes short in {f}")
                    yield f, pos, b
                    got += len(b)
                    pos += len(b)
        finally:
            rc = p.wait()
            if rc:
                raise RuntimeError(f"{self.ssh['host']}: remote read exited {rc}: {p.stderr.read()[-400:]!r}")

    # -- creation
    @classmethod
    def create(cls, url, local_dir, log=print, small_max_gb=2.0):
        """Mirror the remote folder at local_dir: small files whole, shards as sparse files (header only)."""
        import base64
        self = cls.__new__(cls)
        self.ssh, self.run_, self.log = _parse_ssh(url), None, log
        rd = self.ssh["dir"].rstrip("/")
        files = self._remote_json(
            f"d = {rd!r}\nfs = {{f: os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) "
            "if os.path.isfile(os.path.join(d, f))}\nidx = os.path.join(d, 'model.safetensors.index.json')\n"
            "sh = sorted(set(json.load(open(idx))['weight_map'].values())) if os.path.exists(idx) else "
            "[f for f in fs if f.endswith('.safetensors')]\nprint(json.dumps({'files': fs, 'shards': sh}))\n")
        os.makedirs(local_dir, exist_ok=True)
        shards = {s: files["files"][s] for s in files["shards"]}
        heads = self._remote_json(
            f"d = {rd!r}\nout = {{}}\nfor s in {sorted(shards)!r}:\n    with open(os.path.join(d, s), 'rb') as h:\n"
            "        n = int.from_bytes(h.read(8), 'little')\n        h.seek(0)\n"
            "        out[s] = base64.b64encode(h.read(8 + n)).decode()\nprint(json.dumps(out))\n")
        for s, size in shards.items():
            p = os.path.join(local_dir, s)
            with open(p, "wb") as f:
                f.write(base64.b64decode(heads[s]))
                f.truncate(size)                                 # a hole: no disk until a tensor is fetched
        small = [(f, n) for f, n in files["files"].items() if f not in shards and n <= small_max_gb * 1e9]
        skipped = [f for f, n in files["files"].items() if f not in shards and n > small_max_gb * 1e9]
        for f, n in small:
            with open(os.path.join(local_dir, f), "wb") as h:
                for _, _, b in self._stream([(f, 0, n)]):
                    h.write(b)
        json.dump({"url": url, "shards": shards, "created": __import__("time").strftime("%Y-%m-%dT%H:%M:%S")},
                  open(os.path.join(local_dir, MIRROR_MARK), "w", encoding="utf-8"), indent=1)
        log(f"   mirror: {len(shards)} shard(s) ({sum(shards.values()) / 1e9:.1f} GB, sparse) + {len(small)} file(s) "
            f"of {url} -> {local_dir}" + (f"; not mirrored (too big): {skipped}" if skipped else ""))
        return cls(local_dir, log)

    # -- MLX graph -> tensors
    def hook(self):
        """While active, every mx.load of a mirrored shard registers its arrays' Load nodes by tensor name."""
        import contextlib
        import mlx.core as mx
        mirror = self

        @contextlib.contextmanager
        def ctx():
            orig = mx.load

            def load(file, *a, **k):
                out = orig(file, *a, **k)
                name = os.path.basename(str(file))
                if (os.path.normcase(os.path.dirname(os.path.abspath(str(file)))) == os.path.normcase(
                        os.path.abspath(mirror.dir)) and name in mirror.shards and isinstance(out, dict)):
                    for key, arr in out.items():
                        for i in _load_ids(arr):
                            mirror.by_load[i] = key
                return out
            mx.load = load
            try:
                yield mirror
            finally:
                mx.load = orig
        return ctx()

    def deps(self, *arrays):
        """The mirrored tensors the arrays read (only the ones not evaluated yet)."""
        out = set()
        for i in _load_ids(*arrays):
            k = self.by_load.get(i)
            if k is not None:
                out.add(k)
        return out

    # -- cache
    def _save_state(self):
        tmp = os.path.join(self.dir, MIRROR_STATE + ".tmp")
        json.dump(list(self.present), open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, os.path.join(self.dir, MIRROR_STATE))

    def _evict_for(self, need, keep):
        free = self.free_bytes()
        drop = []
        for k in list(self.present):
            if free - need >= self.reserve_gb * 1e9:
                break
            if k in keep or k in self.pinned:
                continue
            drop.append(k)
            free += self.entries[k][2]
        if free - need < self.reserve_gb * 1e9:
            import pollard_stream as S
            raise S.OutOfMemoryBudget(f"disk: the mirror needs {need / 1e9:.2f} GB more and only "
                                      f"{(free - self.reserve_gb * 1e9) / 1e9:.2f} GB can be freed over the "
                                      f"{self.reserve_gb:.1f} GB reserve at {self.dir}")
        if drop:
            for k in drop:                                        # forget first, punch second: a crash in between
                del self.present[k]                               # only means a refetch, never a zero read
            self._save_state()
            for k in drop:
                f, off, n = self.entries[k]
                punch(os.path.join(self.dir, f), off, n)
                self.evicted += n

    def ensure(self, keys, reserve_gb=None):
        """Fetch what of `keys` is not on disk (after making room), pin all of them."""
        if reserve_gb is not None:
            self.reserve_gb = float(reserve_gb)
        keys = set(keys)
        self.pinned |= keys
        todo = sorted((k for k in keys if k not in self.present), key=lambda k: self.entries[k][:2])
        for k in keys:
            if k in self.present:
                self.present.move_to_end(k)
        if not todo:
            return 0
        need = sum(self.entries[k][2] for k in todo)
        self._evict_for(need, keys)
        ranges = []
        for k in todo:                                           # adjacent tensors of one file: one range
            f, off, n = self.entries[k]
            if ranges and ranges[-1][0] == f and ranges[-1][1] + ranges[-1][2] == off:
                ranges[-1][2] += n
            else:
                ranges.append([f, off, n])
        fds = {}
        try:
            for f, pos, b in self._stream([tuple(r) for r in ranges]):
                if f not in fds:
                    fds[f] = os.open(os.path.join(self.dir, f), os.O_RDWR)
                os.pwrite(fds[f], b, pos)
        finally:
            for fd in fds.values():
                os.close(fd)
        for k in todo:
            self.present[k] = True
        self.fetched += need
        self._save_state()
        return need

    def ensure_keys(self, keys):
        return self.ensure([k for k in keys if k in self.entries])

    def done(self, keys):
        """Unpin: the bytes stay as cache until the disk is needed."""
        self.pinned -= set(keys)

    def resident_gb(self):
        return sum(self.entries[k][2] for k in self.present) / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF model dir or id (FP16/BF16)")
    ap.add_argument("--sensitivity", help="Pollard sensitivity.json (else uniform 4-bit)")
    ap.add_argument("--out", help="output MLX model dir (required unless --plan-only)")
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--hot-frac", type=float, default=0.35)
    ap.add_argument("--bits", type=int, choices=MLX_BITS, default=None,
                    help="ladder rung: body bits (hot layers one step up; 8 = uniform 8-bit). Default: the 4/8 mix")
    ap.add_argument("--high-bits", type=int, choices=MLX_BITS, default=None,
                    help="bits for the measured-hot layers / router / embeddings (default: from --bits)")
    ap.add_argument("--focus-layers", help="force these layers to HIGH (8-bit) regardless of the profile, "
                    "e.g. '3,4,8' or '3-8,16' -- steer the budget to layers you care about")
    ap.add_argument("--group-size", type=int, default=64, help="MLX quant group size (default 64)")
    ap.add_argument("--vision-bits", type=int, default=None,
                    help="bits for a vision/audio tower inside the checkpoint (default: 8, same as "
                         "embeddings). The modality PROJECTOR is always held high regardless. "
                         "Lower it only if you have measured the cost with pollard-mmeval.")
    ap.add_argument("--trust-remote-code", default="auto", choices=["auto", "on", "off"],
                    help="run a model's own modeling code (custom archs); 'auto' = only if config has auto_map. "
                         "Note: MLX needs the arch supported in mlx_lm to convert a truly custom model.")
    ap.add_argument("--moe", dest="moe", action="store_true", default=None,
                    help="force MoE policy (default: auto-detect)")
    ap.add_argument("--dense", dest="moe", action="store_false", help="force dense policy")
    ap.add_argument("--range", default="search", choices=RANGE_MODES,
                    help="per-group scale/bias: 'search' (default) = imatrix-weighted range search, never worse "
                         "than min/max per group; 'minmax' = MLX's own. Output is standard MLX affine either way")
    ap.add_argument("--imatrix", help="activation importance for --range search: a llama.cpp imatrix (.dat/.gguf, "
                    "llama-family names mapped) or .npz/.safetensors keyed by module path. Without it the "
                    "search minimises plain weight error")
    ap.add_argument("--stream", default="auto", choices=STREAM_MODES,
                    help="write the model one tensor at a time (bounded RAM, any size; the default). off = the "
                         "converter's own save (the whole quantized model in memory first)")
    ap.add_argument("--max-ram-gb", type=float, default=None,
                    help="cap the streamed build's RAM (default: free RAM minus the POLLARD_MIN_FREE_GB floor, 2 GB)")
    ap.add_argument("--remote-source", metavar="SSH_URL",
                    help="--model is (or becomes) a sparse local MIRROR of the checkpoint at ssh://user@host[:port]/"
                         "<dir>?python=<python>&key=<identity>: headers and small files local, every tensor fetched "
                         "over ssh when MLX reads it and cached within the free disk (for a source this disk cannot "
                         "hold next to its rung)")
    ap.add_argument("--mirror-only", action="store_true", help="with --remote-source: create the mirror and stop")
    ap.add_argument("--plan-only", action="store_true", help="print the per-module bit plan, build nothing")
    a = ap.parse_args()

    if a.remote_source and not os.path.isfile(os.path.join(a.model, MIRROR_MARK)):
        if os.path.isdir(a.model) and os.listdir(a.model):
            sys.exit(f"ERROR: {a.model} exists and is not a mirror -- name an empty or new folder for --remote-source")
        RemoteMirror.create(a.remote_source, a.model)
    if a.mirror_only:
        return
    sens = json.load(open(a.sensitivity)) if a.sensitivity else {}
    low, high = rung_bits(a.bits, a.high_bits) if a.bits else (LOW, a.high_bits or HIGH)
    n_layers = a.layers or int(sens.get("layers") or 0)
    auto_moe, det = detect_moe(a.model, n_layers)
    n_layers = n_layers or det
    if not n_layers:
        sys.exit("ERROR: could not read layer count -- pass --layers.")
    is_moe = auto_moe if a.moe is None else a.moe
    import pollard_workspace as ws
    focus = ws.parse_layers(a.focus_layers)
    if focus:
        print(f"   focus-layers: forcing layers {sorted(focus)} to 8-bit (steered budget)")
    alloc = allocate(sens, n_layers, a.hot_frac, focus=focus, low=low, high=high)
    avg = sum(g["attn"] + g["ffn"] for g in alloc.values()) / (2 * len(alloc))
    kind = "MoE" if is_moe else "dense"
    print(f"== pollard-mlx :: {a.model}  [{kind}]  {n_layers} layers  |  mix {low}/{high}-bit "
          f"avg {avg:.2f}  |  group_size {a.group_size}")
    if is_moe:
        print(f"   MoE: router + shared experts pinned {high}-bit; cold experts {low}-bit")

    stream = a.stream != "off"
    mirror = RemoteMirror.of(a.model) if os.path.isdir(a.model) else None
    if mirror is not None and not stream:
        print("   (the source is a remote mirror: the build streams, --stream off ignored)")
        stream = True
    if a.plan_only and stream and os.path.isdir(a.model):
        try:
            mp = plan_build(a.model, low, high, a.range, bool(encoded_source(a.model)), avg_bits=avg)
            print(f"   streamed build: peak ~{mp['peak_gb']} GB RAM (largest unit {mp['largest']} = "
                  f"{mp['largest_gb']} GB + {BASE_GB} GB base), source {mp['source_gb']} GB -> rung ~{mp['out_gb']} GB")
        except (Exception, SystemExit) as e:                # noqa: BLE001 -- a plan is advice, never a failure
            print(f"   (memory plan unavailable: {e})")
    if a.plan_only:
        # representative module paths for the first 2 layers + the vocab carriers
        print("   per-module plan (sample):")
        samples = ["model.embed_tokens"]
        for i in range(min(2, n_layers)):
            samples += [f"model.layers.{i}.self_attn.q_proj",
                        f"model.layers.{i}.mlp.experts.0.gate_proj" if is_moe
                        else f"model.layers.{i}.mlp.gate_proj",
                        f"model.layers.{i}.mlp.down_proj"]
            if is_moe:
                samples.append(f"model.layers.{i}.mlp.gate")
        for p in samples:
            print(f"     {p:52s} -> "
                  f"{bits_for(p, alloc, is_moe, a.group_size, a.vision_bits, low, high)['bits']}-bit")
        return

    if not a.out:                                          # no --out -> organized workspace path
        try:
            import pollard_workspace as ws
            a.out = ws.resolve_out(a.model, "mlx", tag=f"{a.bits}bit" if a.bits else f"{low}.{high}")
            print(f"   (no --out) -> workspace: {a.out}")
        except Exception:
            sys.exit("ERROR: --out required unless --plan-only.")
    engine, why = mlx_engine(a.model)
    if engine == "hold":
        print(f"HELD: {why} -- an MLX build of the text alone would not be this model")
        sys.exit(EXIT_HELD)
    try:
        if engine == "mlx_vlm":                             # towers / projector / talker ship inside the MLX model
            from mlx_vlm import convert
            print(f"   mlx-vlm ({why}): the whole model -- towers and projector at {a.vision_bits or HIGH}-bit, "
                  "speech output at the checkpoint's precision")
        else:
            from mlx_lm import convert
    except Exception:
        sys.exit(f"ERROR: {engine} not installed. On Apple Silicon: pip install {engine.replace('_', '-')} ; then rerun.")

    rs = None
    imp = load_importance(a.imatrix) if (a.imatrix and a.range == "search") else {}
    if a.range == "search":
        print(f"   range: imatrix-weighted search ({len(imp)} module rows from {a.imatrix})" if imp else
              "   range: search on weight error (no --imatrix; pass one for the activation-weighted search)")
    if stream:
        rs = StreamedQuantize(imp, a.range)
    elif a.range == "search":
        rs = RangeSearchQuantize(imp, "search")

    def predicate(path, module, config=None):              # mlx_lm calls with (path, module)
        if rs is not None:
            rs.note(path, module)                          # the next mx.quantize is this module's weight
        return bits_for(path, alloc, is_moe, a.group_size, a.vision_bits, low, high)

    import contextlib, shutil
    import pollard_workspace as ws
    import pollard_stream as S
    src_dir, staging = a.model, None
    # mlx-vlm reads float weights, and so does mlx-lm for most encodings: an already-quantized source is decoded to a
    # BF16 copy first, itself streamed tensor by tensor (pollard_stream's readers)
    if (engine == "mlx_vlm" or stream) and encoded_source(a.model):
        if mirror is not None:
            sys.exit("ERROR: an already-quantized source cannot be read through a remote mirror -- fetch it")
        staging = a.out.rstrip("/\\") + ".float-src"
        shutil.rmtree(staging, ignore_errors=True)
        src_dir = decode_to_float(a.model, staging)
    guard = MemoryGuard()
    bplan = None
    if stream and os.path.isdir(src_dir):
        try:
            bplan = plan_build(src_dir, low, high, a.range, bool(staging), avg_bits=avg)
            preflight_build(bplan, a.out, a.max_ram_gb)
        except S.OutOfMemoryBudget as e:
            if staging:
                shutil.rmtree(staging, ignore_errors=True)
            print(f"HELD: {e}")
            sys.exit(EXIT_HELD)
    # mlx-vlm builds the model from its own code: the checkpoint's modelling files are never needed there
    trc = False if engine == "mlx_vlm" else ws.resolve_trust_remote_code(a.model, a.trust_remote_code)
    ckw = dict(mlx_path=a.out, quantize=True, q_bits=low, q_group_size=a.group_size, quant_predicate=predicate)
    existed = os.path.exists(a.out)
    try:
        with (rs if rs is not None else contextlib.nullcontext()), \
                (mirror.hook() if mirror is not None else contextlib.nullcontext()), \
                (StreamWriter(rs, guard, mirror=mirror, out_bytes=(bplan or {}).get("out_bytes", 0))
                 if stream else contextlib.nullcontext()):
            try:
                convert(src_dir, trust_remote_code=trc, **ckw)     # newer mlx_lm accepts it
            except TypeError:
                if trc:
                    print("   (this mlx_lm lacks trust_remote_code; a truly custom arch may not convert)")
                convert(src_dir, **ckw)
    except S.OutOfMemoryBudget as e:
        if not existed:
            shutil.rmtree(a.out, ignore_errors=True)        # a half-written rung is never left to look built
        print(f"HELD: {e}")
        sys.exit(EXIT_HELD)
    except BaseException:
        if stream and not existed:
            shutil.rmtree(a.out, ignore_errors=True)
        raise
    finally:
        if staging:
            shutil.rmtree(staging, ignore_errors=True)
    if stream:
        nb = guard.numbers()
        print(f"   streamed build: peak MLX {nb['peak_mlx_gb']} GB, peak RSS {nb['peak_rss_gb']} GB, "
              f"lowest free RAM {nb['min_free_ram_gb']} GB"
              + (f"; mirror fetched {mirror.fetched / 1e9:.1f} GB, evicted {mirror.evicted / 1e9:.1f} GB"
                 if mirror is not None else ""))
    if os.path.isdir(a.model):                             # the whole model or nothing: no silent text-only MLX
        import pollard_complete as PC
        if mirror is not None and not PC.check(a.model, a.out, "mlx")["ok"]:   # sidecars copy source bytes: fetch
            out_keys = PC._out_tensors(a.out)
            mirror.ensure_keys([k for ks in PC.inventory(a.model)["components"].values() for k in ks
                                if not PC._carried(k, out_keys)])
        PC.fill(a.model, a.out, "mlx")                      # processor files a converter did not write
        res = PC.check(a.model, a.out, "mlx")
        if not res["ok"]:
            sys.exit("ERROR: the MLX build is missing " + "; ".join(res["missing"]))
    if rs is not None:
        print(f"   range search: {rs.stats['searched']} tensors searched "
              f"({rs.stats['weighted']} imatrix-weighted)")
    try:
        import pollard_workspace as ws
        # record the ACTUAL avg bpw + a distinct tag so each ladder rung is listable on the card
        # (was always "4.8" regardless of the mix, so rungs collided)
        ws.record_build(a.model, "mlx", a.out, tag=(f"MLX-{a.bits}bit" if a.bits else f"mlx-{avg:.1f}bpw"),
                        bpw=round(avg, 2))     # same files/tag either --range: the encoding is the same format
    except Exception as e:
        print(f"   (warning: could not record build to the workspace manifest: {e})")
    print(f"wrote MLX model -> {a.out}\n"
          f"  run:  mlx_lm.generate --model {a.out} --prompt \"hello\"\n"
          f"  (mixed {low}/{high}-bit, Pollard-allocated; runs on Apple Silicon via MLX/Metal.)")


if __name__ == "__main__":
    main()
