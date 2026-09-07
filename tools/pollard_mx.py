#!/usr/bin/env python3
"""pollard-mx — emit an FP4/FP8 checkpoint for Blackwell (NVFP4 default, MXFP4 experimental), carrying
Pollard's measured-sensitivity allocation. The 5th output lane: GGUF / GPTQ / MLX / EXL3 / **MX (FP4)**.

NVFP4 is Blackwell's native 4-bit float (E2M1, per-group-16 local scale + per-tensor global scale); MXFP4
is the OCP microscaling variant (E2M1, shared E8M0 scale per 32). Both run on Blackwell FP4 tensor cores
via vLLM (compressed-tensors format). This lane produces a **vLLM-loadable** checkpoint — same idea as the
GPTQ lane, targeting the FP4 cores instead of INT4.

Pollard's edge (vs a uniform NVFP4 dump): the measured-sensitivity profile decides **which layers/roles
stay high-precision (FP8) and which drop to FP4** — protect the residual writers / salient layers, crush
the cold bulk to 4-bit — instead of flattening everything to FP4. Same "one allocator, many emitters".

LOW-BIT: precondition first with `pollard-hf-smooth`. FP4's tiny dynamic range is exactly what massive-
activation channels blow out; smoothing the fp16 model migrates the outliers so the FP4 groups aren't
wrecked (llm-compressor pairs SmoothQuant with NVFP4 for the same reason). Or `pollard-doctor --repair`.

  pollard-mx --model Qwen/Qwen3-30B-A3B --sensitivity sens.json --calib cal.txt --out ./M-NVFP4   # emit
  pollard-mx --model <hf> --sensitivity sens.json --plan-only                                       # recipe only
  pollard-mx --model <hf> --scheme MXFP4 --hot-frac 0.25 --out ./M-MXFP4                            # OCP MX

Real emit needs `llm-compressor` on a Blackwell/CUDA box (`pip install llmcompressor`); it writes a
compressed-tensors checkpoint: `vllm serve ./M-NVFP4`. `--plan-only` needs nothing and prints the recipe.
NVFP4 is vLLM-validated; MXFP4 is experimental upstream (flagged at runtime). Verify every build with
`pollard-verify --model <out> --source <hf> --end-to-end` before shipping.
"""
import argparse, json, os, sys
import pollard_workspace as ws


def allocate(sens, n_layers, hot_frac):
    """Rank layers by measured sensitivity; the hottest `hot_frac` stay FP8, the rest go FP4.
    Mirrors pollard-export's hot/cold split so the FP4 lane inherits the same measured allocation."""
    if sens:
        order = sorted(sens.items(), key=lambda kv: -float(kv[1]))
        hot = set(k for k, _ in order[:max(1, int(round(hot_frac * len(order))))])
    else:
        # no profile -> protect the ends (embeddings-adjacent + final layers empirically most sensitive)
        n_hot = max(1, int(round(hot_frac * n_layers)))
        hot = set(str(i) for i in list(range(n_hot // 2)) + list(range(n_layers - (n_hot - n_hot // 2), n_layers)))
    return hot


def build_recipe(hot_layers, scheme, protect_scheme, protect_down):
    """compressed-tensors config: body at FP4 `scheme`; hot layers (+ optionally all down_proj, the
    residual writers) kept at `protect_scheme` (FP8); lm_head always ignored (kept high-precision)."""
    ignore = ["lm_head"]
    # per-module protection: hot layers' linears + (optionally) every down_proj -> FP8 group
    protect_globs = [f"re:.*layers\\.{L}\\..*proj" for L in sorted(hot_layers, key=lambda s: (len(s), s))]
    if protect_down:
        protect_globs.append("re:.*down_proj")
    return {
        "scheme_body": scheme,
        "scheme_protect": protect_scheme,
        "ignore": ignore,
        "protect": protect_globs,
    }


def avg_bits(n_layers, hot_frac, protect_down):
    body = 4.0
    frac_protected = min(1.0, hot_frac + (0.14 if protect_down else 0.0))   # down_proj ~1/7 of linears
    return round(body * (1 - frac_protected) + 8.0 * frac_protected, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF model dir or id (FP16/BF16 source)")
    ap.add_argument("--out", help="output compressed-tensors dir (required unless --plan-only)")
    ap.add_argument("--sensitivity", help="Pollard sensitivity.json (pollard-probe/-sensitivity)")
    ap.add_argument("--calib", help="calibration text (required for real emit; NVFP4 activations need it)")
    ap.add_argument("--scheme", default="NVFP4", choices=["NVFP4", "MXFP4"],
                    help="FP4 scheme: NVFP4 (vLLM-validated, default) or MXFP4 (OCP MX, experimental)")
    ap.add_argument("--protect-scheme", default="FP8", choices=["FP8", "FP8_DYNAMIC"],
                    help="precision for the protected (hot) layers")
    ap.add_argument("--hot-frac", type=float, default=0.25, help="fraction of layers kept at FP8 (not FP4)")
    ap.add_argument("--protect-down", action="store_true",
                    help="also keep every down_proj (residual writer) at FP8 — usually worth it at 4-bit")
    ap.add_argument("--layers", type=int, default=0, help="n decoder layers (else read from config)")
    ap.add_argument("--plan-only", action="store_true", help="print the recipe + avg bits, build nothing")
    a = ap.parse_args()

    n_layers = a.layers
    if not n_layers:
        try:
            from transformers import AutoConfig
            n_layers = getattr(AutoConfig.from_pretrained(a.model), "num_hidden_layers", 0)
        except Exception:
            n_layers = 0
    sens = {}
    if a.sensitivity and os.path.exists(a.sensitivity):
        sens = json.load(open(a.sensitivity))
        sens = sens.get("layers", sens) if isinstance(sens, dict) else {}

    hot = allocate(sens, n_layers or 32, a.hot_frac)
    rec = build_recipe(hot, a.scheme, a.protect_scheme, a.protect_down)
    ab = avg_bits(n_layers or 32, a.hot_frac, a.protect_down)

    if a.scheme == "MXFP4":
        print(" !! MXFP4 is EXPERIMENTAL upstream (MXFP4PackedCompressor; vLLM validation pending). "
              "NVFP4 is the vLLM-validated default.")
    print(f"== pollard-mx :: {a.model}  scheme={a.scheme}  ~{ab} bpw avg  "
          f"(body FP4 · {len(hot)} hot layers + {'down_proj ' if a.protect_down else ''}@ {a.protect_scheme})")
    print("   Pollard intent -> compressed-tensors: crush the cold bulk to FP4, keep measured-hot layers "
          "at FP8, lm_head high-precision.")
    print("   recipe:")
    print(f"     body:    QuantizationModifier(targets='Linear', scheme='{a.scheme}', ignore={rec['ignore'] + rec['protect']})")
    if rec["protect"]:
        print(f"     protect: QuantizationModifier(targets={rec['protect'][:3]}{'...' if len(rec['protect'])>3 else ''}, scheme='{a.protect_scheme}')")
    if a.plan_only:
        return
    if not a.out:
        a.out = ws.resolve_out(a.model, "mx", tag=f"{a.scheme}-{ab}bpw")
        print(f"   (no --out) -> workspace: {a.out}")
    if not a.calib:
        sys.exit("ERROR: --calib required for real emit (FP4 activation scales need calibration).")
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except Exception:
        sys.exit("ERROR: llm-compressor not installed here. On the Blackwell/CUDA box: "
                 "pip install llmcompressor ; then rerun. It writes a compressed-tensors checkpoint "
                 "that `vllm serve` loads on the FP4 tensor cores.")
    # body FP4 (ignore lm_head + protected globs) + a second modifier pinning protected globs to FP8
    mods = [QuantizationModifier(targets="Linear", scheme=a.scheme, ignore=rec["ignore"] + rec["protect"])]
    if rec["protect"]:
        mods.append(QuantizationModifier(targets=rec["protect"], scheme=a.protect_scheme))
    cal = [l for l in open(a.calib, encoding="utf-8", errors="ignore").read().splitlines() if l.strip()]
    print(f"   emitting via llm-compressor ({len(cal)} calib rows) -> {a.out}")
    oneshot(model=a.model, recipe=mods, dataset=cal, output_dir=a.out)
    ws.record_build(a.model, "mx", a.out, tag=f"{a.scheme}-{ab}bpw", bpw=ab)
    print(f"wrote MX checkpoint -> {a.out}\n  run:  vllm serve {a.out}"
          f"\n  VERIFY:  pollard-verify --model {a.out} --source {a.model} --end-to-end")


if __name__ == "__main__":
    main()
