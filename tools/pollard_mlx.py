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
import argparse, json, os, re, sys

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
    if is_moe and (re.search(r"\.(mlp|block_sparse_moe)\.gate$", path)
                   or "shared_expert" in path):
        return {"bits": HIGH, "group_size": gsize}      # router + shared: selection integrity
    if "self_attn" in path and path.endswith("g_proj"):  # head-wise attn output gate (Spark2_5): selection-critical
        return {"bits": HIGH, "group_size": gsize}
    li = layer_of(path)
    if li is None or li not in alloc:
        return {"bits": LOW, "group_size": gsize}
    if "self_attn" in path or re.search(r"\.mixer\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj)$", path):
        return {"bits": alloc[li]["attn"], "group_size": gsize}   # NemotronH: attention / Mamba2 token mixers
    if "mlp" in path or "block_sparse_moe" in path or "experts" in path or "latent_proj" in path:
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
    with xp=mlx.core, on the GPU; rows are processed in chunks of ~max_elems weights."""
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
    W = w.reshape(rows, K // group_size, group_size).astype(xp.float32)
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
    if init is not None:
        s0, b0 = init
        init = (s0.reshape(rows, K // group_size), b0.reshape(rows, K // group_size))
    step = max(1, max_elems // max(1, K))
    qs, ss, bs = [], [], []
    for r in range(0, rows, step):
        g = W[r:r + step]
        ii = None if I is None else (I if I.shape[0] == 1 else I[r:r + step])
        if method == "minmax":
            q, s_, b_, _ = search_groups(g, bits, ii, store, xp, grid=(0.0,), refits=0)
        else:
            q, s_, b_, _ = search_groups(g, bits, ii, store, xp,
                                         init=None if init is None else (init[0][r:r + step],
                                                                         init[1][r:r + step]))
        qs.append(np.asarray(q.astype(xp.uint8)) if xp is not np else q.astype(np.uint8))
        ss.append(s_)
        bs.append(b_)
    q = np.concatenate(qs, 0).reshape(*shape[:-1], K)
    cat = (lambda a: xp.concatenate(a, 0)) if xp is not np else (lambda a: np.concatenate(a, 0))
    s = cat(ss).reshape(*shape[:-1], K // group_size)
    b = cat(bs).reshape(*shape[:-1], K // group_size)
    return pack_mlx(q, bits), s, b


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
    ap.add_argument("--plan-only", action="store_true", help="print the per-module bit plan, build nothing")
    a = ap.parse_args()

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
    if a.range == "search":
        imp = load_importance(a.imatrix) if a.imatrix else {}
        rs = RangeSearchQuantize(imp, "search")
        print(f"   range: imatrix-weighted search ({len(imp)} module rows from {a.imatrix})" if imp else
              "   range: search on weight error (no --imatrix; pass one for the activation-weighted search)")

    def predicate(path, module, config=None):              # mlx_lm calls with (path, module)
        if rs is not None:
            rs.note(path, module)                          # the next mx.quantize is this module's weight
        return bits_for(path, alloc, is_moe, a.group_size, a.vision_bits, low, high)

    import contextlib, shutil
    import pollard_workspace as ws
    src_dir, staging = a.model, None
    if engine == "mlx_vlm" and encoded_source(a.model):     # mlx-vlm reads float weights: decode first
        staging = a.out.rstrip("/\\") + ".float-src"
        shutil.rmtree(staging, ignore_errors=True)
        src_dir = decode_to_float(a.model, staging)
    # mlx-vlm builds the model from its own code: the checkpoint's modelling files are never needed there
    trc = False if engine == "mlx_vlm" else ws.resolve_trust_remote_code(a.model, a.trust_remote_code)
    ckw = dict(mlx_path=a.out, quantize=True, q_bits=low, q_group_size=a.group_size, quant_predicate=predicate)
    try:
        with (rs if rs is not None else contextlib.nullcontext()):
            try:
                convert(src_dir, trust_remote_code=trc, **ckw)     # newer mlx_lm accepts it
            except TypeError:
                if trc:
                    print("   (this mlx_lm lacks trust_remote_code; a truly custom arch may not convert)")
                convert(src_dir, **ckw)
    finally:
        if staging:
            shutil.rmtree(staging, ignore_errors=True)
    if os.path.isdir(a.model):                             # the whole model or nothing: no silent text-only MLX
        import pollard_complete as PC
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
