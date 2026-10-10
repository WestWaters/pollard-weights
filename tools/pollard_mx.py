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

Real emit needs `llm-compressor` on a CUDA box (`pip install llmcompressor`); it writes a
compressed-tensors checkpoint: `vllm serve ./M-NVFP4` (Rubin: the vllm/vllm-openai:cu134-nightly image).
`--plan-only` needs nothing and prints the recipe. NVFP4 is vLLM-validated; MXFP4 is experimental upstream
(flagged at runtime) and runs on Blackwell/Rubin via flashinfer_cutlass (validate before shipping). Verify every build with
`pollard-verify --model <out> --source <hf> --end-to-end` before shipping.
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
ROUTER_GLOBS = [r"re:.*(mlp|block_sparse_moe|feed_forward|moe)\.gate$", r"re:.*shared_expert_gate$",
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


def scheme_bits(scheme):
    """Nominal weight bits for a scheme name (FP4/NVFP4/MXFP4/W4A16 -> 4; FP8/W8A16 -> 8)."""
    return 4.0 if "4" in scheme else 8.0


def avg_bits(n_layers, hot_frac, protect_down, scheme="NVFP4", protect_scheme="FP8"):
    body = scheme_bits(scheme)
    frac_protected = min(1.0, hot_frac + (0.14 if protect_down else 0.0))   # down_proj ~1/7 of linears
    return round(body * (1 - frac_protected) + scheme_bits(protect_scheme) * frac_protected, 2)


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
    ap.add_argument("--scheme", default="NVFP4", choices=["NVFP4", "MXFP4", "W4A16", "W8A16"],
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
    ap.add_argument("--layers", type=int, default=0, help="n decoder layers (else read from config)")
    ap.add_argument("--trust-remote-code", default="auto", choices=["auto", "on", "off"],
                    help="run a model's own modeling code (custom archs like Spark2_5); 'auto' = only if "
                         "config.json has an auto_map")
    ap.add_argument("--plan-only", action="store_true", help="print the recipe + avg bits, build nothing")
    a = ap.parse_args()

    n_layers, moe = a.layers, False
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
        n_layers = n_layers or getattr(cfg, "num_hidden_layers", 0) or getattr(
            getattr(cfg, "text_config", None), "num_hidden_layers", 0)
        moe = is_moe_config(cfg)
    except Exception:
        pass
    sens = {}
    if a.sensitivity and os.path.exists(a.sensitivity):
        sens = json.load(open(a.sensitivity))
        sens = sens.get("layers", sens) if isinstance(sens, dict) else {}

    focus = ws.parse_layers(a.focus_layers)
    if focus:
        print(f"   focus-layers: forcing layers {sorted(focus)} to FP8 (steered budget)")
    hot = allocate(sens, n_layers or 32, a.hot_frac, focus=focus)
    if a.moe_protect_attn_only and a.protect_down:
        print("   --moe-protect-attn-only: ignoring --protect-down (it would put every expert down_proj at "
              f"{a.protect_scheme} and split the MoE backend)")
    if a.moe_protect_attn_only and not moe:
        print("   note: --moe-protect-attn-only on a model whose config shows no experts -- only attention "
              "of hot layers is protected")
    protect_down = a.protect_down and not a.moe_protect_attn_only
    rec = build_recipe(hot, a.scheme, a.protect_scheme, protect_down, a.quantize_vision,
                       moe_attn_only=a.moe_protect_attn_only, kv_fp8=a.kv_fp8)
    ab = avg_bits(n_layers or 32, a.hot_frac, protect_down, a.scheme, a.protect_scheme)
    is_int = a.scheme in ("W4A16", "W8A16")

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
    print(f"   Pollard intent -> compressed-tensors: crush the cold bulk to {a.scheme}, keep measured-hot "
          f"layers at {a.protect_scheme}, lm_head high-precision."
          + ("  [INT weight-only: runs on any vLLM GPU, not just Blackwell]" if is_int else ""))
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
    if not a.calib and not is_int:
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
    cal = ([l for l in open(a.calib, encoding="utf-8", errors="ignore").read().splitlines() if l.strip()]
           if a.calib else None)
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
    ws.record_build(a.model, "mx", a.out, tag=f"{a.scheme}-{ab}bpw", bpw=ab)
    print(f"wrote MX checkpoint -> {a.out}")
    print("\n".join(serve_hints(a.out, a.scheme, kv_fp8=a.kv_fp8, moe=moe)))
    print(f"  VERIFY:  pollard-verify --model {a.out} --source {a.model} --end-to-end")


if __name__ == "__main__":
    main()
