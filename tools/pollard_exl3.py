#!/usr/bin/env python3
"""pollard-exl3 — emit an EXL3 (exllamav3) model with Pollard's allocation, same as the other lanes.

The GPU desktop lane: GGUF (llama.cpp), GPTQ (vLLM/SGLang), MLX (Apple), and THIS — EXL3 for the
exllamav3 runtime. Same Pollard brain (detect arch, decide where the bits go), a different emitter.
EXL3's trellis encode is the HEAVY lane by nature (a single low-bit output can take many hours — that
cost is the FORMAT, not Pollard), so this is the option for people who specifically run exllama; the
cheap lanes (GGUF/GPTQ/MLX) are the default. Pollard still feeds EXL3 the allocation so it skips its own
budgeted search.

Two ways to carry the allocation into exllamav3's convert_model:
  * budgeted (default, robust): map Pollard's intent to EXL3's knobs — target --bits + protected
    --head_bits / --mtp_bits + --hq (bump select layers on MoE). No tensor-key matching, can't misfire.
  * recipe (--recipe file): pass an explicit per-tensor bitrate recipe (exllamav3 applies it verbatim,
    'in place of the budgeted allocation') — the exact Pollard plan, once you have exllamav3 enumerating
    the model's tensor keys (needs the runtime loaded).

  pollard-exl3 --model Qwen/Qwen3-8B --out ./Qwen3-8B-Pollard-EXL3 --bpw 3.0
  pollard-exl3 --model <hf> --out <dir> --plan-only          # print the exllamav3 command, build nothing
  pollard-exl3 --model <hf> --out <dir> --recipe plan.yaml   # explicit per-tensor plan

Needs exllamav3 installed AND its CUDA ext loadable (a bleeding-edge GPU like Blackwell may need a
source build — the pip wheel's prebuilt ext can fail with 'DLL load failed importing exllamav3_ext')."""
import argparse, os, subprocess, sys


def detect_moe(model_id, layers_hint=0):
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)
        n = getattr(cfg, "num_hidden_layers", 0) or layers_hint
        moe = any(getattr(cfg, k, 0) for k in
                  ("num_experts", "num_local_experts", "n_routed_experts"))
        return bool(moe), int(n)
    except Exception:
        return False, layers_hint


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF model dir or id (FP16/BF16 source)")
    ap.add_argument("--out", help="output EXL3 dir (required unless --plan-only)")
    ap.add_argument("--bpw", type=float, default=3.0, help="target average bits/weight (EXL3 budgeted)")
    ap.add_argument("--head-bits", type=int, default=6, help="output/head layer bits (protected)")
    ap.add_argument("--mtp-bits", type=int, default=4, help="MTP layer bits (GLM/DeepSeek MTP head)")
    ap.add_argument("--recipe", help="explicit per-tensor bitrate recipe (YAML) — overrides budgeted")
    ap.add_argument("--hq", dest="hq", action="store_true", default=None,
                    help="bump bitrate of select layers (MoE) — default: auto-on for MoE")
    ap.add_argument("--no-hq", dest="hq", action="store_false")
    ap.add_argument("--cal-data", help="calibration data (safetensors token rows); else EXL3's bundled mix")
    ap.add_argument("--devices", default="0", help="CUDA device list for the convert, e.g. 0,1")
    ap.add_argument("--plan-only", action="store_true", help="print the exllamav3 command, build nothing")
    a = ap.parse_args()

    is_moe, _ = detect_moe(a.model)
    hq = is_moe if a.hq is None else a.hq
    kind = "MoE" if is_moe else "dense"
    # exllamav3 conversion entry: python -m exllamav3.conversion.convert_model
    cmd = [sys.executable, "-m", "exllamav3.conversion.convert_model", "-i", a.model]
    if a.out:
        cmd += ["-o", a.out]
    if a.recipe:                                            # explicit Pollard per-tensor plan
        cmd += ["-rcp", a.recipe]
    else:                                                   # budgeted: Pollard intent -> EXL3 knobs
        cmd += ["-b", str(a.bpw), "-hb", str(a.head_bits), "-mb", str(a.mtp_bits)]
        if hq:
            cmd += ["-hq"]                                  # protect select (MoE) layers
    if a.cal_data:
        cmd += ["-cd", a.cal_data]
    cmd += ["-d", a.devices]

    print(f"== pollard-exl3 :: {a.model}  [{kind}]  "
          + (f"recipe {os.path.basename(a.recipe)}" if a.recipe
             else f"{a.bpw} bpw · head {a.head_bits} · mtp {a.mtp_bits}"
                  + (" · hq(MoE)" if hq else "")))
    print("   Pollard intent -> EXL3: crush the body to target bpw, protect head/MTP high"
          + (", bump select MoE layers (--hq)" if hq else "") + ".")
    print("   NOTE: EXL3 trellis is the HEAVY lane (hours for a low-bit output) — that's the format, "
          "not Pollard. Prefer GGUF/GPTQ/MLX unless you need the exllama runtime.")
    print("   $ " + " ".join(cmd))
    if a.plan_only:
        return
    if not a.out:
        sys.exit("ERROR: --out required unless --plan-only.")
    try:
        import exllamav3  # noqa: F401
    except Exception as e:
        sys.exit(f"ERROR: exllamav3 not usable here ({repr(e)[:80]}). Install it AND make sure its CUDA "
                 f"ext loads (a bleeding-edge GPU may need a source build). Then rerun.")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"exllamav3 convert exited {r.returncode}")
    print(f"wrote EXL3 model -> {a.out}\n  run:  exllamav3 / TabbyAPI loads {a.out}")


if __name__ == "__main__":
    main()
