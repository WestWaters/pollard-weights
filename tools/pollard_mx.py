#!/usr/bin/env python3
"""pollard-mx -- emit an FP4/FP8 checkpoint for Blackwell/Rubin (NVFP4 default, MXFP4 experimental), carrying
Pollard's measured-sensitivity allocation. The 5th output lane: GGUF / GPTQ / MLX / EXL3 / **MX (FP4)**.

NVFP4 is Blackwell's native 4-bit float (E2M1, per-group-16 local scale + per-tensor global scale); MXFP4
is the OCP microscaling variant (E2M1, shared E8M0 scale per 32). Both run on the FP4 tensor cores of
Blackwell and Vera Rubin (same formats, no new one on Rubin) via vLLM (compressed-tensors format). This lane produces a **vLLM-loadable** checkpoint -- same idea as the
GPTQ lane, targeting the FP4 cores instead of INT4.

Pollard's edge (vs a uniform NVFP4 dump): the measured-sensitivity profile decides **which layers/roles
stay high-precision (FP8) and which drop to FP4** -- protect the residual writers / salient layers, crush
the cold bulk to 4-bit -- instead of flattening everything to FP4. Same "one allocator, many emitters".

LOW-BIT: precondition first with `pollard-hf-smooth`. FP4's tiny dynamic range is exactly what massive-
activation channels blow out; smoothing the fp16 model migrates the outliers so the FP4 groups aren't
wrecked (llm-compressor pairs SmoothQuant with NVFP4 for the same reason). Or `pollard-doctor --repair`.

  pollard-mx --model Qwen/Qwen3-30B-A3B --sensitivity sens.json --calib cal.txt --out ./M-NVFP4   # emit
  pollard-mx --model <hf> --sensitivity sens.json --plan-only                                       # recipe only
  pollard-mx --model <hf> --scheme MXFP4 --hot-frac 0.25 --out ./M-MXFP4                            # OCP MX
  pollard-mx --model <moe> --sensitivity sens.json --calib cal.txt --moe-protect-attn-only --kv-fp8 # MoE, FP8 KV
  pollard-mx --model <hf> --scheme FP8 --calib cal.txt --out ./M-FP8                                # pure FP8 W8A8
  pollard-mx --model <hf> --sensitivity s.json --calib cal.txt --fp4-scale 4over6 --imatrix m.imatrix --out ./M-NVFP4-4o6

FOUR OVER SIX (--fp4-scale 4over6; MIT HAN Lab, arXiv 2512.02010, MIT licence): each NVFP4 block's E4M3 scale maps
the block max to 4 OR 6, whichever reconstructs it better -- scored by Pollard's imatrix-weighted error (--imatrix),
not plain MSE. Applied after llm-compressor writes the checkpoint; still standard NVFP4 (vLLM loads it unchanged).

PURE FP8 (--scheme FP8 / FP8_DYNAMIC): every linear at FP8 W8A8 except the MoE routers and lm_head (and the
vision tower / projector), which stay at the checkpoint's precision -- Cohere's published recipe. No
hot/cold split: at 8 bits there is nothing to protect FROM, only the selection- and vocab-carrying matrices
to leave alone. FP8 = static per-tensor activation scales (calibrated, needs --calib); FP8_DYNAMIC =
per-channel weights + dynamic per-token activations (no calibration). Hopper/Ada/Blackwell/Rubin FP8 cores.

Real emit needs `llm-compressor` on a CUDA box (`pip install llmcompressor`); it writes a
compressed-tensors checkpoint: `vllm serve ./M-NVFP4` (Rubin: the vllm/vllm-openai:cu134-nightly image).
`--plan-only` needs nothing and prints the recipe. NVFP4 is vLLM-validated; MXFP4 is experimental upstream
(flagged at runtime) and runs on Blackwell/Rubin via flashinfer_cutlass (validate before shipping). Verify every build with
`pollard-verify --lane mx --model <out> --source <hf> --eval <heldout.txt> --gate` before shipping.

MXFP6 + MEASURED MIXES (pollard_mxfp6; model-free, weight-only, CPU, no llm-compressor):
  pollard-mx --model <hf> --format mxfp6-e2m3 --out ./M-MXFP6-E2M3     # 6.25 bpw, compressed-tensors (Graft loads it)
  pollard-mx --model <hf> --format mxfp6-e3m2 --out ./M-MXFP6-E3M2     # 6.25 bpw, Quark layout (vLLM main loads it)
  pollard-mx --model <hf> --mix 5.5 --sensitivity s.json --out ./M-MIX # FP8 / MXFP6 / FP4 per fused layer, <= 5.5 bpw
The mix measures every linear at every atom and spends the bit budget where the sensitivity profile says it pays;
see tools/pollard_mxfp6.py for the formats, layouts and which runtimes load each.

STREAMED NVFP4 / MXFP4 / FP8 (--engine pollard, what `auto` picks wherever llm-compressor is missing or cannot load
the model whole): one calibration pass streamed block by block (pollard_stream) measures every linear's input
maximum -- NVFP4's input_global_scale, static FP8's input_scale -- and Pollard's own encoder writes the same
compressed-tensors checkpoint llm-compressor's preset does (W4A4 NVFP4 / MXFP4, FP8 W8A8, FP8_DYNAMIC), hot layers at
FP8, routers / lm_head / embeddings / towers at source precision. A source tensor that is ALREADY NVFP4 (NVIDIA's
QAT checkpoints) is kept byte for byte (codes + E4M3 block scales; the tensor scale converted from modelopt's
weight_scale_2), every other linear is quantized from the dequantized source, and pollard_mx.json records which.
"""
import argparse, json, os, sys
import pollard_workspace as ws


def allocate(sens, n_layers, hot_frac, focus=None):
    """Rank layers by measured sensitivity; the hottest `hot_frac` stay FP8, the rest go FP4.
    Mirrors pollard-export's hot/cold split so the FP4 lane inherits the same measured allocation.
    `focus` (layer indices) are forced HIGH regardless of the profile (steer the budget)."""
    if sens:
        order = sorted(sens.items(), key=lambda kv: -float(kv[1]))
        hot = set(k for k, _ in order[:max(1, int(round(hot_frac * len(order))))])
    else:
        # no profile -> protect the ends (embeddings-adjacent + final layers empirically most sensitive)
        n_hot = max(1, int(round(hot_frac * n_layers)))
        hot = set(str(i) for i in list(range(n_hot // 2)) + list(range(n_layers - (n_hot - n_hot // 2), n_layers)))
    hot |= {str(i) for i in (focus or ())}         # user-steered layers, always HIGH
    return hot


# A vision or audio tower lives inside the checkpoint on this lane. The body modifier is
# targets="Linear", which matches EVERY nn.Linear in the model -- so on a vision-language
# checkpoint the whole tower and the modality projector were going to FP4 alongside the language
# stack. Worse, the per-layer protect globs below are `.*layers\.<n>\..*proj`, which also matches
# `vision_tower.encoder.layers.<n>.self_attn.out_proj` -- so a vision block inherited whichever
# TEXT layer happened to share its index.
# Both go in `ignore`, which keeps them at the checkpoint's original precision. The projector is
# never negotiable; the tower can be included deliberately with --quantize-vision.
VISION_GLOBS = [r"re:.*(visual|vision_tower|vision_model|vision_encoder|image_encoder|patch_embed"
                r"|audio_tower|audio_encoder|speech_encoder).*"]
PROJECTOR_GLOBS = [r"re:.*(multi_modal_projector|mm_projector|mm_proj|merger|modality_project"
                   r"|resampler|perceiver|connector).*"]


# MoE routers are nn.Linear too, so targets="Linear" was sending them to FP4 with the experts. A router
# is a tiny matrix whose output picks the experts -- FP4 error there reorders the top-k, i.e. changes
# WHICH weights run, not just how precisely (pollard-export pins it at 8-bit for the same reason). It is
# also not a layer vLLM's FP4 MoE kernels consume: llm-compressor's own MoE examples ignore it
# (Qwen3-MoE / Qwen3-Next `re:.*mlp.gate$` + `re:.*shared_expert_gate$`, Mixtral `block_sparse_moe.gate`,
# Llama 4 / gpt-oss `router`). Anchored with `$` so an expert's `gate_proj` / a dense `gate_up_proj`
# never matches; on a dense model these match nothing.
ROUTER_GLOBS = [r"re:.*(mlp|block_sparse_moe|feed_forward|moe|mixer)\.gate$", r"re:.*shared_expert_gate$",
                r"re:.*(mlp|block_sparse_moe|feed_forward|moe)\.router$"]
# Attention projections of a hot layer, for --moe-protect-attn-only: standard q/k/v/o, MLA (q_a/q_b,
# kv_a_proj_with_mqa, kv_b), and linear-attention blocks (Qwen3-Next `linear_attn.in_proj_qkvz`).
ATTN_PARENTS = r"(self_attn|attn|attention|linear_attn)"
# FP8 KV cache (E4M3, one static scale per tensor, calibrated): what vLLM's `--kv-cache-dtype fp8` loads
# the k/v scales from. Static, so the calibration pass measures them -- hence --calib is required.
FP8_KV_SCHEME = {"num_bits": 8, "type": "float", "strategy": "tensor", "dynamic": False, "symmetric": True}


def build_recipe(hot_layers, scheme, protect_scheme, protect_down, quantize_vision=False,
                 moe_attn_only=False, kv_fp8=False):
    """compressed-tensors config: body at FP4 `scheme`; hot layers (+ optionally all down_proj, the
    residual writers) kept at `protect_scheme` (FP8); lm_head always ignored (kept high-precision).
    The modality projector and MoE routers are always ignored, and the vision/audio tower unless
    --quantize-vision.

    `moe_attn_only`: protect only the hot layers' ATTENTION and never an expert, so every expert in the
    model is the body scheme. vLLM picks one fused-MoE kernel per quant scheme; a hot layer whose experts
    went FP8 put a second MoE backend in the same model (and --protect-down did that to every layer's
    expert down_proj), so the FP4 MoE path -- e.g. `--moe-backend flashinfer_cutedsl` -- could not cover
    the whole model. `kv_fp8`: add an FP8 kv_cache_scheme for `vllm serve --kv-cache-dtype fp8`."""
    ignore = ["lm_head"] + PROJECTOR_GLOBS + ROUTER_GLOBS + ([] if quantize_vision else VISION_GLOBS)
    hot = sorted(hot_layers, key=lambda s: (len(s), s))
    if moe_attn_only:
        protect_globs = [f"re:.*layers\\.{L}\\.{ATTN_PARENTS}\\..*proj\\w*" for L in hot]
    else:
        # per-module protection: hot layers' linears + (optionally) every down_proj -> FP8 group
        protect_globs = [f"re:.*layers\\.{L}\\..*proj" for L in hot]
        if protect_down:
            protect_globs.append("re:.*down_proj")
    return {
        "scheme_body": scheme,
        "scheme_protect": protect_scheme,
        "ignore": ignore,
        "protect": protect_globs,
        "kv_cache_scheme": dict(FP8_KV_SCHEME) if kv_fp8 else None,
    }


def is_moe_config(cfg):
    """True when a HF config (dict or PretrainedConfig) describes a mixture-of-experts model -- any of
    the usual expert-count keys, on the config or on its text_config (multimodal wrappers)."""
    def get(c, k):
        return c.get(k) if isinstance(c, dict) else getattr(c, k, None)
    if cfg is None:
        return False
    if any(isinstance(get(cfg, k), int) and get(cfg, k) > 1
           for k in ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")):
        return True
    sub = get(cfg, "text_config")
    return sub is not None and sub is not cfg and is_moe_config(sub)


RUBIN_VLLM_IMAGE = "vllm/vllm-openai:cu134-nightly"


def serve_hints(out, scheme, kv_fp8=False, moe=False):
    """How to serve the checkpoint, Blackwell and Rubin. Flags only, never a speed claim: none of this
    has been run on Rubin hardware by Pollard -- pollard-verify on the target box is what counts."""
    flags = ["--kv-cache-dtype fp8"] if kv_fp8 else []          # load the calibrated FP8 k/v scales
    rubin_flags = list(flags)
    if moe and scheme == "NVFP4":
        rubin_flags.append("--moe-backend flashinfer_cutedsl")   # the NVFP4 MoE CuTe-DSL path is opt-in
    lines = [f"  run (Blackwell):  vllm serve {out}" + "".join(" " + f for f in flags),
             f"  run (Vera Rubin): docker run --gpus all --ipc=host -p 8000:8000 -v {out}:/model "
             f"{RUBIN_VLLM_IMAGE} /model" + "".join(" " + f for f in rubin_flags)]
    if scheme == "NVFP4":
        lines.append("    Rubin: NVFP4 dense GEMMs take the CuTe-DSL path by default on that image"
                     + ("; the NVFP4 MoE path is the --moe-backend flag above" if moe else ""))
    elif scheme == "MXFP4":
        lines.append("    MXFP4 runs on Blackwell/Rubin via flashinfer_cutlass (validate)")
    lines.append("    Rubin support is upstream-nightly and untested by Pollard -- verify on the box before shipping")
    return lines


# ----------------------------------------------------------------- Four Over Six (NVFP4 block scales)
#
# NVFP4 = E2M1 values {0, .5, 1, 1.5, 2, 3, 4, 6} x an FP8 E4M3 scale per block of 16 x an FP32 scale per tensor.
# The usual block scale maps the block's max to 6 -- and E2M1 has nothing between 4 and 6, so values near 5/6 of
# the max round badly. "Four Over Six" (Cook et al., arXiv 2512.02010, MIT HAN Lab, github.com/mit-han-lab/
# fouroversix, MIT licence) ALSO tries mapping the block max to 4 (a finer grid near the top, 6 left unused) and
# keeps whichever reconstructs the block better. To leave room for the x1.5 larger scale that "4" needs, the tensor
# scale is computed against an E4M3 max of 256 instead of 448 (6/4 * 256 = 384 <= 448). Clean-room from the
# paper/reference; the output is STANDARD NVFP4 -- same packed FP4, same E4M3 block scales, same FP32 global
# scale -- so vLLM / compressed-tensors load it unchanged.
#
# Pollard's version picks per block by the error WEIGHTED by each input channel's activation energy (the
# imatrix: --imatrix), not plain MSE: a block's error matters as much as the activations that multiply it.

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX, E4M3_MAX_FOUROVERSIX = 448.0, 256.0
FP4_SCALE_RULES = ("6", "4over6")
NVFP4_BLOCK = 16


def round_e4m3(x):
    """Round float32 to the nearest FP8 E4M3 (fn) value, as float32 (numpy; subnormals, saturating at 448)."""
    import numpy as np
    x = np.asarray(x, np.float32)
    a = np.minimum(np.abs(x), E4M3_MAX)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -9)))
    e = np.maximum(e, -6.0)                                     # below 2^-6: subnormal step 2^-9
    step = 2.0 ** (e - 3)
    r = np.rint(a / step) * step
    return (np.sign(x) * np.minimum(r, E4M3_MAX)).astype(np.float32)


def fp4_codes(y):
    """E2M1 code (sign bit 3, magnitude index 0..7) of the nearest representable value, ties to even index."""
    import numpy as np
    vals = np.asarray(E2M1_VALUES, np.float32)
    a = np.minimum(np.abs(y), 6.0)
    d = np.abs(a[..., None] - vals)
    idx = np.argmin(d + (np.arange(8) % 2) * 1e-7, axis=-1)     # exact ties -> the even (mantissa 0) code
    sign = ((y < 0) & (idx > 0)).astype(np.uint8) << 3
    return (idx.astype(np.uint8) | sign)


def fp4_values(codes):
    import numpy as np
    vals = np.asarray(E2M1_VALUES, np.float32)
    c = np.asarray(codes, np.uint8)
    v = vals[c & 7]
    return np.where(c & 8, -v, v)


def pack_fp4(codes):
    """[N, K] codes -> [N, K/2] uint8, element 2i in the LOW nibble (compressed-tensors / vLLM layout)."""
    import numpy as np
    c = np.asarray(codes, np.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)


def unpack_fp4(packed):
    import numpy as np
    p = np.asarray(packed, np.uint8)
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), np.uint8)
    out[..., 0::2], out[..., 1::2] = p & 0xF, p >> 4
    return out


def nvfp4_decode(packed, scales, global_scale):
    """compressed-tensors convention: w = fp4 * (weight_scale / weight_global_scale)."""
    import numpy as np
    v = fp4_values(unpack_fp4(packed))
    N, K = v.shape
    s = np.asarray(scales, np.float32) / np.float32(global_scale)
    return (v.reshape(N, K // NVFP4_BLOCK, NVFP4_BLOCK) * s[..., None]).reshape(N, K)


def nvfp4_encode(w, rule="6", imp=None, global_scale=None):
    """Weight [N, K] -> standard NVFP4 (packed uint8 [N, K/2], E4M3-exact scales [N, K/16] as float32,
    global_scale float32, mask of blocks that took the "4" scale).

    rule "6": every block max -> 6 (plain NVFP4, global = 448*6/amax). rule "4over6": per block, max -> 6 or
    -> 4, whichever has the lower sum(imp * err^2) (global = 256*6/amax). `global_scale` overrides the tensor
    scale (a fused q/k/v or gate/up group must share one: vLLM reads one per fused layer)."""
    import numpy as np
    w = np.asarray(w, np.float32)
    N, K = w.shape
    if K % NVFP4_BLOCK:
        raise ValueError(f"K={K} is not a multiple of the NVFP4 block ({NVFP4_BLOCK})")
    if rule not in FP4_SCALE_RULES:
        raise ValueError(f"--fp4-scale is one of {FP4_SCALE_RULES}")
    amax = float(np.abs(w).max()) or 1.0
    g = np.float32(global_scale if global_scale is not None else
                   (E4M3_MAX if rule == "6" else E4M3_MAX_FOUROVERSIX) * 6.0 / amax)
    blk = w.reshape(N, K // NVFP4_BLOCK, NVFP4_BLOCK)
    bmax = np.abs(blk).max(-1)
    I = None
    if imp is not None:
        I = np.asarray(imp, np.float32).reshape(1, K // NVFP4_BLOCK, NVFP4_BLOCK)

    def at(top):
        sc = round_e4m3(g * bmax / top)                           # stored E4M3 block scale
        eff = np.where(sc > 0, sc / g, 1.0)[..., None]
        codes = fp4_codes(np.where(sc[..., None] > 0, blk / eff, 0.0))
        err = (blk - fp4_values(codes) * np.where(sc > 0, sc / g, 0.0)[..., None]) ** 2
        if I is not None:
            err = err * I
        return codes, sc, err.sum(-1)
    c6, s6, e6 = at(6.0)
    four = np.zeros_like(e6, dtype=bool)
    codes, sc = c6, s6
    if rule == "4over6":
        c4, s4, e4 = at(4.0)
        four = e4 < e6
        codes = np.where(four[..., None], c4, c6)
        sc = np.where(four, s4, s6)
    return pack_fp4(codes.reshape(N, K)), sc.astype(np.float32), g, four


def nvfp4_weighted_error(w, packed, scales, global_scale, imp=None):
    import numpy as np
    w = np.asarray(w, np.float32)
    e = (w - nvfp4_decode(packed, scales, global_scale)) ** 2
    if imp is not None:
        e = e * np.asarray(imp, np.float32)[None, :]
    return float(e.sum())


def _source_tensor(handles, key):
    """The source weight for a compressed key, exact name first, then by suffix (VLM prefixes differ)."""
    for h in handles:
        if key in h.keys():
            return h.get_tensor(key)
    tail = key.split(".", 1)[-1]
    for h in handles:
        for k in h.keys():
            if k.endswith(tail) and k.split(".")[-1] == "weight":
                return h.get_tensor(k)
    return None


def apply_four_over_six(out_dir, source_dir, importance=None, log=print):
    """Re-encode every NVFP4 weight of a compressed-tensors checkpoint with Four Over Six block scales.

    Fused groups keep sharing one global scale: each tensor's global scale is llm-compressor's times 256/448,
    which is exactly 256*6/amax_group for the group amax llm-compressor already shared. Activation scales are
    untouched. Rewrites the shards in place; returns stats."""
    import glob, numpy as np, torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    importance = importance or {}
    src = [safe_open(f, framework="pt") for f in sorted(glob.glob(os.path.join(source_dir, "*.safetensors")))]
    if not src:
        raise SystemExit(f"Four Over Six: no source safetensors in {source_dir}")
    st = {"tensors": 0, "blocks": 0, "four": 0, "weighted": 0, "skipped": [], "err_before": 0.0, "err_after": 0.0}
    for shard in sorted(glob.glob(os.path.join(out_dir, "*.safetensors"))):
        sd = load_file(shard)
        changed = False
        for key in [k for k in sd if k.endswith(".weight_packed")]:
            base = key[:-len(".weight_packed")]
            sk, gk = base + ".weight_scale", base + ".weight_global_scale"
            if sk not in sd or gk not in sd or sd[sk].dtype != torch.float8_e4m3fn:
                continue                                          # not NVFP4 (MXFP4 / INT packs)
            w = _source_tensor(src, base + ".weight")
            if w is None:
                st["skipped"].append(base)
                continue
            w = w.float().numpy()
            if w.shape != (sd[key].shape[0], sd[key].shape[1] * 2):
                st["skipped"].append(base)
                continue
            g_old = float(sd[gk].float().reshape(-1)[0])
            imp = importance.get(base)
            if imp is not None and np.asarray(imp).size != w.shape[1]:
                imp = None
            before = nvfp4_weighted_error(w, sd[key].numpy(), sd[sk].float().numpy(), g_old, imp)
            packed, sc, g, four = nvfp4_encode(w, "4over6", imp, global_scale=g_old * E4M3_MAX_FOUROVERSIX / E4M3_MAX)
            after = nvfp4_weighted_error(w, packed, sc, g, imp)
            sd[key] = torch.from_numpy(packed)
            sd[sk] = torch.from_numpy(sc).to(torch.float8_e4m3fn)
            sd[gk] = torch.full_like(sd[gk], float(g), dtype=sd[gk].dtype)
            st["tensors"] += 1
            st["blocks"] += four.size
            st["four"] += int(four.sum())
            st["weighted"] += imp is not None
            st["err_before"] += before
            st["err_after"] += after
            changed = True
        if changed:
            save_file(sd, shard, metadata={"format": "pt"})
    if st["tensors"]:
        log(f"   Four Over Six: {st['tensors']} NVFP4 tensors re-scaled ({st['weighted']} imatrix-weighted), "
            f"{100 * st['four'] / max(1, st['blocks']):.1f}% of blocks took the 4-scale; weighted error "
            f"{st['err_after'] / max(st['err_before'], 1e-30):.3f}x of plain NVFP4")
    if st["skipped"]:
        log(f"   Four Over Six: {len(st['skipped'])} tensors left as llm-compressor wrote them (no source match), "
            f"e.g. {st['skipped'][:3]}")
    return st


#: pure-FP8 body schemes (W8A8): everything but routers / lm_head / modality towers at FP8
PURE_FP8 = ("FP8", "FP8_DYNAMIC")


def scheme_bits(scheme):
    """Nominal weight bits for a scheme name (FP4/NVFP4/MXFP4/W4A16 -> 4; FP8/W8A16 -> 8)."""
    return 4.0 if "4" in scheme else 8.0


def avg_bits(n_layers, hot_frac, protect_down, scheme="NVFP4", protect_scheme="FP8"):
    body = scheme_bits(scheme)
    frac_protected = min(1.0, hot_frac + (0.14 if protect_down else 0.0))   # down_proj ~1/7 of linears
    return round(body * (1 - frac_protected) + scheme_bits(protect_scheme) * frac_protected, 2)


def _snapshot(model_id):
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])


def mx6_main(a):
    """--format mxfp6-* / --mix: the model-free MX encoder (pollard_mxfp6)."""
    import pollard_mxfp6 as M6
    if a.format and a.mix is not None:
        sys.exit("ERROR: --format and --mix are separate rungs; pass one")
    src = a.model if os.path.isdir(a.model) else _snapshot(a.model)
    sens = {}
    if a.sensitivity and os.path.exists(a.sensitivity):
        sens = json.load(open(a.sensitivity))
        sens = sens.get("layers", sens) if isinstance(sens, dict) else {}
    imp = {}
    if a.imatrix:
        from pollard_mlx import load_importance
        imp = load_importance(a.imatrix)
    tag = (a.format or f"mix-{a.mix:g}bpw").upper()
    out = a.out or ws.resolve_out(a.model, "mx", tag=tag)
    print(f"== pollard-mx :: {a.model}  {a.format or f'measured mix <= {a.mix:g} bpw'}  (model-free, weight-only)")
    meta = M6.build(src, out, fmt=a.format, mix=a.mix, layout=a.layout, fp6=a.fp6, fp4=a.fp4, rule=a.scale_rule,
                    importance=imp, sens=sens, focus=ws.parse_layers(a.focus_layers),
                    quantize_vision=a.quantize_vision, plan_only=a.plan_only)
    if a.plan_only:
        return
    ws.record_build(a.model, "mx", out, tag=tag, bpw=meta["bpw"])
    print(f"wrote MX checkpoint -> {out}  ({meta['layout']}, {meta['bpw']} bpw)")
    for rt, status, how in meta["runtimes"]:
        print(f"   {rt:18s} {status:8s} {how}")
    for line in M6.serve_lines(meta, out):
        print(f"  run:  {line}")
    print(f"  VERIFY:  pollard-verify --lane mx --model {out} --source {a.model} --eval <heldout.txt> --gate")


STREAM_SCHEMES = ("NVFP4", "MXFP4", "FP8", "FP8_DYNAMIC")


def pick_engine(model, want="auto", scheme="NVFP4"):
    """('pollard' | 'llmcompressor', why) -- auto keeps llm-compressor for what it always handled (a plain causal
    LM that fits in RAM, no custom code, not already quantized) and streams everything else."""
    if want != "auto":
        return want, "asked for"
    if scheme not in STREAM_SCHEMES:
        return "llmcompressor", f"{scheme} is llm-compressor's"
    try:
        import llmcompressor  # noqa: F401
    except Exception:                                                     # noqa: BLE001
        return "pollard", "llm-compressor is not installed here"
    if not os.path.isdir(model):
        return "llmcompressor", ""
    import pollard_stream as S
    if S.is_overlay(model):
        return "pollard", "a smoothing overlay -- only the stream reader applies it"
    if S.needs_remote_code(model):
        return "pollard", "custom code"
    cfg = ws.config_dict(model)
    if cfg.get("quantization_config"):
        return "pollard", "the source is already quantized (NVFP4 tensors pass through)"
    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as LM
        if cfg.get("model_type") not in LM:
            return "pollard", f"transformers has no causal-LM class for {cfg.get('model_type')!r} (VLM / omni)"
    except Exception:                                                     # noqa: BLE001
        return "pollard", "transformers is not importable"
    w = sum(os.path.getsize(p) for p in S.weight_files(model)) / 1e9
    free = S.free_ram_gb() or 0
    if w * 1.2 > free:
        return "pollard", f"{w:.1f} GB of weights vs {free:.1f} GB free RAM: streamed"
    return "llmcompressor", ""


def stream_main(a, hot, protect_down):
    """--engine pollard for NVFP4 / MXFP4 / FP8 / FP8_DYNAMIC: one streamed calibration pass for the activation
    scales (when the scheme needs them), then the model-free encoder writes the compressed-tensors rung."""
    import pollard_mxfp6 as M6
    import pollard_stream as S
    if a.scheme not in STREAM_SCHEMES:
        sys.exit(f"ERROR: --engine pollard writes {', '.join(STREAM_SCHEMES)}; {a.scheme} needs llm-compressor")
    if a.kv_fp8:
        sys.exit("ERROR: --kv-fp8 (static FP8 k/v scales) needs llm-compressor; drop it for the stream engine")
    if a.protect_scheme not in M6.SCHEME_WEIGHTS:
        sys.exit(f"ERROR: --protect-scheme {a.protect_scheme} is llm-compressor's; the stream engine protects at "
                 "FP8 / FP8_DYNAMIC")
    if protect_down:
        print("   note: --protect-down is llm-compressor's recipe; the stream engine protects whole hot layers")
    need = {a.scheme, a.protect_scheme if hot else a.scheme} & set(M6.NEEDS_INPUT_SCALE)
    amax, cal = None, None
    if need:
        if not a.calib:
            sys.exit(f"ERROR: --calib required ({', '.join(sorted(need))} activation scales need calibration)")
        dev = S.stream_device(a.device)
        print(f"   calibration pass (streamed, {dev}) for the {', '.join(sorted(need))} activation scales ...")
        amax, cal = S.collect_input_amax(a.model, a.calib, a.nsamples, a.seqlen, dev, a.trust_remote_code,
                                         a.max_ram_gb)
        print(f"   {cal['linears']} linears calibrated on {cal['rows']} rows; peak RSS {cal['peak_rss_gb']} GB")
    meta = M6.build_scheme(a.model, a.out, a.scheme, hot={int(h) for h in hot}, protect=a.protect_scheme,
                           input_amax=amax, moe_attn_only=a.moe_protect_attn_only,
                           quantize_vision=a.quantize_vision)
    if cal:
        meta["calibration"] = cal
        json.dump(meta, open(os.path.join(a.out, M6.SIDECAR), "w"), indent=1)
    if meta.get("passthrough"):
        print(f"   NVFP4 passthrough: {len(meta['passthrough'])} source NVFP4 linears kept byte for byte")
    if meta.get("dequantized_source"):
        print(f"   quantized from the dequantized source: {meta['dequantized_source']}")
    ws.record_build(a.model, "mx", a.out, tag=f"{a.scheme}-{meta['bpw']}bpw", bpw=meta["bpw"])
    print(f"wrote MX checkpoint -> {a.out}  (compressed-tensors, {meta['bpw']} bpw)")
    print("\n".join(serve_hints(a.out, a.scheme, moe=bool(meta.get('modules') and any('.experts.' in m for m in meta['modules'])))))
    print(f"  VERIFY:  pollard-verify --lane mx --model {a.out} --source {a.model} --eval <heldout.txt> --gate")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF model dir or id (FP16/BF16 source)")
    ap.add_argument("--out", help="output compressed-tensors dir (required unless --plan-only)")
    ap.add_argument("--sensitivity", help="Pollard sensitivity.json (pollard-probe/-sensitivity)")
    ap.add_argument("--calib", help="calibration text (required for real emit; NVFP4 activations need it)")
    ap.add_argument("--quantize-vision", action="store_true",
                    help="also quantize a vision/audio tower inside the checkpoint (default: left "
                         "at source precision). The modality projector is never quantized. "
                         "Measure the cost with pollard-mmeval before trusting this.")
    ap.add_argument("--scheme", default="NVFP4", choices=["NVFP4", "MXFP4", "W4A16", "W8A16", "FP8", "FP8_DYNAMIC"],
                    help="body scheme: NVFP4 (Blackwell FP4, vLLM-validated default) / MXFP4 (OCP MX, "
                         "experimental) / W4A16 / W8A16 (INT weight-only compressed-tensors -- runs on any "
                         "vLLM GPU, not just Blackwell)")
    ap.add_argument("--gptq", action="store_true",
                    help="for INT schemes (W4A16/W8A16): use GPTQ error-feedback for the body (more "
                         "accurate than RTN); needs --calib. Ignored for the FP4 schemes.")
    ap.add_argument("--protect-scheme", default="FP8", choices=["FP8", "FP8_DYNAMIC", "W8A16"],
                    help="precision for the protected (hot) layers")
    ap.add_argument("--hot-frac", type=float, default=0.25, help="fraction of layers kept at FP8 (not FP4)")
    ap.add_argument("--focus-layers", help="force these layers to HIGH (FP8) regardless of the profile, "
                    "e.g. '3,4,8' or '3-8,16' -- steer the budget to layers you care about")
    ap.add_argument("--protect-down", action="store_true",
                    help="also keep every down_proj (residual writer) at FP8 -- usually worth it at 4-bit")
    ap.add_argument("--moe-protect-attn-only", action="store_true",
                    help="MoE: protect only the hot layers' attention, never an expert -- every expert stays "
                         "the body scheme so one MoE backend covers the whole model (ignores --protect-down)")
    ap.add_argument("--kv-fp8", action="store_true",
                    help="also calibrate an FP8 KV cache (kv_cache_scheme) -- serve with "
                         "`vllm serve --kv-cache-dtype fp8`. Off by default; needs --calib")
    ap.add_argument("--fp4-scale", default="6", choices=FP4_SCALE_RULES,
                    help="NVFP4 block scales: '6' = plain (block max -> 6); '4over6' = Four Over Six (MIT, arXiv "
                         "2512.02010): per block max -> 4 or 6, whichever has the lower imatrix-weighted error. "
                         "Output stays standard NVFP4")
    ap.add_argument("--imatrix", help="activation importance for --fp4-scale 4over6 (llama.cpp imatrix, llama-family "
                    "names mapped, or .npz/.safetensors keyed by module path); without it, plain MSE")
    ap.add_argument("--layers", type=int, default=0, help="n decoder layers (else read from config)")
    ap.add_argument("--trust-remote-code", default="auto", choices=["auto", "on", "off"],
                    help="run a model's own modeling code (custom archs like Spark2_5); 'auto' = only if "
                         "config.json has an auto_map")
    ap.add_argument("--plan-only", action="store_true", help="print the recipe + avg bits, build nothing")
    ap.add_argument("--engine", default="auto", choices=["auto", "pollard", "llmcompressor"],
                    help="NVFP4 / MXFP4 / FP8 rungs: 'pollard' = Pollard's own encoder streamed from disk (any size, "
                         "VLM / omni text backbones, custom code with consent, already-quantized sources -- an NVFP4 "
                         "source tensor passes through byte for byte; activation scales from one streamed "
                         "calibration pass); 'llmcompressor' = llm-compressor's oneshot (loads through "
                         "transformers); 'auto' = llm-compressor for a plain causal LM that fits in RAM when it is "
                         "installed, Pollard's engine for everything else")
    ap.add_argument("--device", default="auto", help="pollard engine: auto (CUDA when present) | cpu | cuda[:N]")
    ap.add_argument("--nsamples", type=int, default=128, help="pollard engine: calibration rows")
    ap.add_argument("--seqlen", type=int, default=2048, help="pollard engine: tokens per calibration row")
    ap.add_argument("--max-ram-gb", type=float, help="pollard engine: cap the run's RAM (default 85%% of free)")
    ap.add_argument("--format", choices=["mxfp6-e3m2", "mxfp6-e2m3"],
                    help="a uniform OCP MXFP6 rung (every decoder linear at 6.25 bpw; routers / lm_head / embeddings / "
                         "towers kept at source precision). Model-free, CPU, no calibration")
    ap.add_argument("--mix", type=float, metavar="BPW",
                    help="a measured MX mix under this average bits/weight (scales included): each fused layer gets "
                         "MXFP8 / MXFP6 / FP4 by measured error x the --sensitivity profile")
    ap.add_argument("--layout", default="auto", choices=["auto", "quark", "compressed-tensors"],
                    help="--format/--mix on-disk layout: auto = compressed-tensors for uniform E2M3 (Graft, "
                         "compressed-tensors), quark otherwise (vLLM main; the only layout with E3M2)")
    ap.add_argument("--fp6", default="auto", choices=["auto", "e2m3", "e3m2"],
                    help="--mix: the FP6 element; auto = whichever reconstructs each layer better")
    ap.add_argument("--fp4", choices=["mxfp4", "nvfp4"],
                    help="--mix: the FP4 atom (default: nvfp4 for compressed-tensors, mxfp4 for quark)")
    ap.add_argument("--scale-rule", default="search", choices=["ocp", "search"],
                    help="--format/--mix E8M0 scale: 'ocp' = the spec's floor(log2 amax) - emax; 'search' = that or "
                         "one step either side, lowest (imatrix-weighted) error. Output is standard MX either way")
    a = ap.parse_args()
    if a.format or a.mix is not None:
        return mx6_main(a)

    n_layers, moe = a.layers, False
    cfg = ws.text_config_dict(ws.config_dict(a.model))        # plain JSON: no checkpoint code runs to read it
    if cfg:
        n_layers = n_layers or int(cfg.get("num_hidden_layers") or 0)
        moe = is_moe_config(cfg) or any(isinstance(b, dict) and b.get("block_type") == "moe"
                                        for b in (cfg.get("block_configs") or []))
    sens = {}
    if a.sensitivity and os.path.exists(a.sensitivity):
        sens = json.load(open(a.sensitivity))
        sens = sens.get("layers", sens) if isinstance(sens, dict) else {}

    focus = ws.parse_layers(a.focus_layers)
    if focus:
        print(f"   focus-layers: forcing layers {sorted(focus)} to FP8 (steered budget)")
    pure_fp8 = a.scheme in PURE_FP8
    hot = set() if pure_fp8 else allocate(sens, n_layers or 32, a.hot_frac, focus=focus)
    if pure_fp8 and (sens or focus):
        print("   pure FP8: every linear is already 8-bit -- the sensitivity profile / focus layers are not used")
    if a.moe_protect_attn_only and a.protect_down:
        print("   --moe-protect-attn-only: ignoring --protect-down (it would put every expert down_proj at "
              f"{a.protect_scheme} and split the MoE backend)")
    if a.moe_protect_attn_only and not moe:
        print("   note: --moe-protect-attn-only on a model whose config shows no experts -- only attention "
              "of hot layers is protected")
    protect_down = a.protect_down and not a.moe_protect_attn_only and not pure_fp8
    rec = build_recipe(hot, a.scheme, a.protect_scheme, protect_down, a.quantize_vision,
                       moe_attn_only=a.moe_protect_attn_only, kv_fp8=a.kv_fp8)
    ab = 8.0 if pure_fp8 else avg_bits(n_layers or 32, a.hot_frac, protect_down, a.scheme, a.protect_scheme)
    is_int = a.scheme in ("W4A16", "W8A16")

    if a.fp4_scale == "4over6" and a.scheme != "NVFP4":
        print(f"   --fp4-scale 4over6 applies to NVFP4 (E4M3 block scales); {a.scheme} keeps its own scales")
    if a.scheme == "MXFP4":
        print(" !! MXFP4 is EXPERIMENTAL upstream (MXFP4PackedCompressor); it runs on Blackwell/Rubin via "
              "flashinfer_cutlass (validate). NVFP4 is the vLLM-validated default.")
    # attention-only protection covers far less than whole hot layers, so the layer-fraction estimate
    # is an upper bound there
    what = "hot layers' attention" if a.moe_protect_attn_only else "hot layers"
    print(f"== pollard-mx :: {a.model}  scheme={a.scheme}  "
          f"{'<=' if a.moe_protect_attn_only else '~'}{ab} bpw avg  "
          f"(body {a.scheme}  |  {len(hot)} {what} + {'down_proj ' if protect_down else ''}@ {a.protect_scheme})"
          + ("  KV fp8" if a.kv_fp8 else ""))
    print(("   pure FP8 W8A8 (Cohere's recipe): every linear at FP8; MoE routers, lm_head and modality towers "
           "left at the checkpoint's precision." if pure_fp8 else
           f"   Pollard intent -> compressed-tensors: crush the cold bulk to {a.scheme}, keep measured-hot "
           f"layers at {a.protect_scheme}, lm_head high-precision.")
          + ("  [INT weight-only: runs on any vLLM GPU, not just Blackwell]" if is_int else ""))
    if a.fp4_scale == "4over6" and a.scheme == "NVFP4":
        print("   block scales: Four Over Six (arXiv 2512.02010, MIT) -- per block max -> 4 or 6 by "
              + ("imatrix-weighted error" if a.imatrix else "MSE (pass --imatrix for the weighted choice)")
              + "; applied after llm-compressor writes the checkpoint")
    print("   recipe:")
    print(f"     body:    QuantizationModifier(targets='Linear', scheme='{a.scheme}', ignore={rec['ignore'] + rec['protect']})")
    if rec["protect"]:
        print(f"     protect: QuantizationModifier(targets={rec['protect'][:3]}{'...' if len(rec['protect'])>3 else ''}, scheme='{a.protect_scheme}')")
    if rec["kv_cache_scheme"]:
        print(f"     kv:      kv_cache_scheme={rec['kv_cache_scheme']}  (serve with --kv-cache-dtype fp8)")
    if a.plan_only:
        return
    if not a.out:
        a.out = ws.resolve_out(a.model, "mx", tag=f"{a.scheme}-{ab}bpw")
        print(f"   (no --out) -> workspace: {a.out}")
    engine, why = pick_engine(a.model, a.engine, a.scheme)
    print(f"   engine: {engine}" + (f"  ({why})" if why else ""))
    if engine == "pollard":
        return stream_main(a, hot, protect_down)
    if not a.calib and not is_int and a.scheme != "FP8_DYNAMIC":
        sys.exit("ERROR: --calib required for real emit (FP4 activation scales need calibration).")
    if a.gptq and not a.calib:
        sys.exit("ERROR: --gptq needs --calib (error feedback is measured on the calibration set).")
    if a.kv_fp8 and not a.calib:
        sys.exit("ERROR: --kv-fp8 needs --calib (the static FP8 k/v scales are measured on the calibration set).")
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except Exception:
        sys.exit("ERROR: llm-compressor not installed here. On the target CUDA box: "
                 "pip install llmcompressor ; then rerun. It writes a compressed-tensors checkpoint "
                 "that `vllm serve` loads (FP4 on Blackwell tensor cores; W4A16/W8A16 on any vLLM GPU).")
    # body (ignore lm_head + protected globs) + a second modifier pinning the protected globs high.
    # For INT schemes, --gptq swaps the body to GPTQ error-feedback (more accurate than RTN at W4A16).
    BodyMod = QuantizationModifier
    if a.gptq and is_int:
        try:
            from llmcompressor.modifiers.quantization import GPTQModifier as BodyMod
        except Exception:
            print("   (GPTQModifier unavailable -- falling back to RTN QuantizationModifier for the body)")
    kv = {"kv_cache_scheme": rec["kv_cache_scheme"]} if rec["kv_cache_scheme"] else {}
    mods = [BodyMod(targets="Linear", scheme=a.scheme, ignore=rec["ignore"] + rec["protect"], **kv)]
    if rec["protect"]:
        # same ignore on the protect modifier: otherwise a vision block whose index matches a hot
        # TEXT layer is still captured here, at FP8 -- wrong, just more quietly
        mods.append(QuantizationModifier(targets=rec["protect"], scheme=a.protect_scheme,
                                         ignore=rec["ignore"]))
    cal = ws.calib_samples(a.calib) if a.calib else None    # whole samples (tool calls intact), not lines
    print(f"   emitting via llm-compressor ({len(cal) if cal else 'no'} calib rows) -> {a.out}")
    trc = ws.resolve_trust_remote_code(a.model, a.trust_remote_code)
    def _oneshot(**kw):
        # llm-compressor exposes trust_remote_code as trust_remote_code_model; tolerate older versions
        try:
            oneshot(trust_remote_code_model=trc, **kw)
        except TypeError:
            if trc:
                print("   (this llm-compressor lacks trust_remote_code_model; custom arch may not load)")
            oneshot(**kw)
    if cal:                                        # FP4 activation scales / GPTQ error-feedback need it
        _oneshot(model=a.model, recipe=mods, dataset=cal, output_dir=a.out)
    else:                                          # INT weight-only RTN: no calibration set required
        _oneshot(model=a.model, recipe=mods, output_dir=a.out)
    if a.fp4_scale == "4over6" and a.scheme == "NVFP4":
        imp = {}
        if a.imatrix:
            from pollard_mlx import load_importance
            imp = load_importance(a.imatrix)
        st = apply_four_over_six(a.out, a.model if os.path.isdir(a.model) else _snapshot(a.model), imp)
        try:                                              # say so in the checkpoint, for the card
            cfgp = os.path.join(a.out, "config.json")
            cfg = json.load(open(cfgp))
            cfg.setdefault("quantization_config", {})["pollard_fp4_scale"] = {
                "rule": "4over6", "tensors": st["tensors"], "four_share": round(st["four"] / max(1, st["blocks"]), 4),
                "reference": "Four Over Six, arXiv 2512.02010 (MIT HAN Lab, MIT licence)"}
            json.dump(cfg, open(cfgp, "w"), indent=2)
        except Exception as e:                            # noqa: BLE001
            print(f"   (could not note 4over6 in config.json: {e})")
    ws.record_build(a.model, "mx", a.out, tag=f"{a.scheme}-{ab}bpw" + ("-4o6" if a.fp4_scale == "4over6" else ""),
                    bpw=ab)
    print(f"wrote MX checkpoint -> {a.out}")
    print("\n".join(serve_hints(a.out, a.scheme, kv_fp8=a.kv_fp8, moe=moe)))
    print(f"  VERIFY:  pollard-verify --lane mx --model {a.out} --source {a.model} --eval <heldout.txt> --gate")


if __name__ == "__main__":
    main()
