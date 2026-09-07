#!/usr/bin/env python3
"""pollard-exl3 — emit an EXL3 (exllamav3) model with Pollard's allocation, same as the other lanes.

🔒 LOCKED GOLD RECIPE (EXL3) — THE WIN: **the Pollard method (its smoothing preconditioner + Calib 3.0)
beats EXL3 on EXL3's OWN allocator, trellis atoms, and custom kernel.** MEASURED (Qwen2.5-3B @4bpw):
broken 3090 → smoothed 8.699 → smoothed + Calib 3.0 **8.670**, vs EXL3 out-of-box **8.699** — a win on
their own turf, untuned and with no extra calibration. And smoothing alone takes unusable low-bit (PPL
3090) to 4bpw ≈ 8bpw quality at HALF the size. First and only tool that does this.
`pollard --format exl3` runs the gold recipe by DEFAULT (--no-smooth to skip). proxy_err is banned (legacy/).

LOW-BIT mechanics: precondition first with `pollard-hf-smooth` (SmoothQuant folded into the RMSNorms).
Massive-activation input channels otherwise collapse the trellis global scale and silently wreck a layer
(measured: layer sqnr -14 -> +18 after smoothing). One-shot: `pollard-doctor --source <fp16> --repair
--lane exl3` (smooth → reconvert → verify), or `pollard --format exl3 --run`.

The GPU desktop lane: GGUF (llama.cpp), GPTQ (vLLM/SGLang), MLX (Apple), and THIS — EXL3 for the
exllamav3 runtime. It's the compatibility lane: same one command into a 4th runtime.

ALLOCATION (gold path): keep **EXL3's native allocator** and win through the Pollard method's
preconditioning + calibration — that's where the win comes from, and it lands on EXL3's own atoms. The
allocator is already strong on its trellis atoms, so don't spend your bits reallocating it: the GGUF
K-quant role-map is a DEAD LEVER on EXL3 (K-quant priors don't map to trellis atoms — measured worse, not
worth the ~50 min). The gold recipe already beats EXL3 out-of-box without touching allocation. A finer
per-tensor-KL recipe measured directly in EXL3's atom space is an open R&D lever — until one is shown to
beat the gold recipe, the gold path is smoothing + Calib 3.0 + native allocator. Judge by pollard-verify.

Two allocation modes:
  * budgeted (DEFAULT, the gold path): EXL3's allocator via --bits + --head_bits/--mtp_bits + --hq.
  * recipe (--recipe): EXPERIMENTAL per-tensor bitrate YAML — R&D only. Don't port a GGUF role map here;
    only use a recipe measured in EXL3's atom space and shown to beat the gold recipe.

  pollard-exl3 --model Qwen/Qwen3-8B --out ./Qwen3-8B-Pollard-EXL3 --bpw 3.0   # budgeted (recommended)
  pollard-exl3 --model <hf> --out <dir> --plan-only          # print the exllamav3 command, build nothing
  pollard-exl3 --model <hf> --out <dir> --recipe plan.yaml   # EXPERIMENTAL explicit per-tensor plan

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


def pack_calib(model, text_path, out_path, rows=256, cols=2048):
    """Tokenize a TEXT corpus with the model's own tokenizer and pack it into the -cd safetensors
    EXL3 wants: {input_ids: int64[rows, cols]}. Real tokens only — NEVER tile; if the corpus is short,
    emit fewer rows. 256 rows is the measured sweet spot (more shifts the Hessian → worse allocation)."""
    from transformers import AutoTokenizer
    import torch
    from safetensors.torch import save_file
    tok = AutoTokenizer.from_pretrained(model)
    ids = tok(open(text_path, encoding="utf-8", errors="ignore").read(), return_tensors="pt").input_ids[0]
    if ids.numel() < rows * cols:                              # not enough real tokens -> fewer rows, no tiling
        rows = max(1, ids.numel() // cols)
    ids = ids[:rows * cols].reshape(rows, cols).contiguous().to(torch.long)
    save_file({"input_ids": ids}, out_path)
    return out_path, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF model dir or id (FP16/BF16 source)")
    ap.add_argument("--out", help="output EXL3 dir (required unless --plan-only)")
    ap.add_argument("--work-dir", help="conversion work dir (checkpoints/resume; default: <out>_work)")
    ap.add_argument("--bpw", type=float, default=3.0, help="target average bits/weight (EXL3 budgeted)")
    ap.add_argument("--head-bits", type=int, default=6, help="output/head layer bits (protected)")
    ap.add_argument("--mtp-bits", type=int, default=4, help="MTP layer bits (GLM/DeepSeek MTP head)")
    ap.add_argument("--recipe", help="EXPERIMENTAL per-tensor bitrate recipe (YAML) — overrides budgeted. "
                    "A GGUF-style role map LOSES to budgeted on EXL3 (measured); use only a recipe measured "
                    "in EXL3's atom space that beats budgeted at <= same bpw.")
    ap.add_argument("--hq", dest="hq", action="store_true", default=None,
                    help="bump bitrate of select layers (MoE) — default: auto-on for MoE")
    ap.add_argument("--no-hq", dest="hq", action="store_false")
    ap.add_argument("--cal-data", help="calibration data (safetensors token rows, -cd); else EXL3's bundled mix")
    ap.add_argument("--calib-text", help="a TEXT corpus (e.g. Calib 3.0) — tokenized+packed to the -cd "
                    "safetensors here, so the locked gold recipe (smoothing + Calib 3.0) runs one-shot")
    ap.add_argument("--cal-rows", type=int, default=256, help="rows to pack from --calib-text (256 = the "
                    "measured EXL3 sweet spot; more is NON-monotonic and can hurt)")
    ap.add_argument("--cal-cols", type=int, default=2048, help="tokens per row for --calib-text packing")
    ap.add_argument("--devices", default="0", help="CUDA device list for the convert, e.g. 0,1")
    ap.add_argument("--plan-only", action="store_true", help="print the exllamav3 command, build nothing")
    a = ap.parse_args()

    if not a.out and not a.plan_only:                       # no --out -> organized workspace path
        import pollard_workspace as ws
        a.out = ws.resolve_out(a.model, "exl3", tag=f"{a.bpw}bpw")
        print(f"   (no --out) -> workspace: {a.out}")
    # gold recipe: pack a TEXT calib (Calib 3.0) into the -cd safetensors, unless one was given directly
    if a.calib_text and not a.cal_data:
        packed = (a.out.rstrip("/\\") + ".cal.safetensors") if a.out else "pollard_exl3_cal.safetensors"
        if a.plan_only:
            print(f"   [cal] would pack {a.calib_text} -> {packed} ({a.cal_rows}x{a.cal_cols} tokens) for -cd")
            a.cal_data = packed
        else:
            try:
                a.cal_data, rows = pack_calib(a.model, a.calib_text, packed, a.cal_rows, a.cal_cols)
                print(f"   [cal] packed Calib 3.0 -> {a.cal_data} ({rows}x{a.cal_cols} real tokens, -cd)")
            except Exception as e:
                print(f"   [cal] could not pack --calib-text ({repr(e)[:70]}); using EXL3's bundled cal")
    is_moe, _ = detect_moe(a.model)
    hq = is_moe if a.hq is None else a.hq
    kind = "MoE" if is_moe else "dense"
    # exllamav3 conversion entry: python -m exllamav3.conversion.convert_model
    cmd = [sys.executable, "-m", "exllamav3.conversion.convert_model", "-i", a.model]
    if a.out:
        cmd += ["-o", a.out]
    work = a.work_dir or ((a.out.rstrip("/\\") + "_work") if a.out else None)
    if work:
        cmd += ["-w", work]                                 # exllamav3 REQUIRES a work dir (checkpoints/resume)
    if a.recipe:                                            # explicit Pollard per-tensor plan
        cmd += ["-rcp", a.recipe]
    else:                                                   # budgeted: Pollard intent -> EXL3 knobs
        cmd += ["-b", str(a.bpw), "-hb", str(a.head_bits), "-mb", str(a.mtp_bits)]
        if hq:
            cmd += ["-hq"]                                  # protect select (MoE) layers
    if a.cal_data:
        cmd += ["-cd", a.cal_data]
    cmd += ["-d", a.devices]

    if a.recipe:
        print(" !! EXPERIMENTAL --recipe: a GGUF/K-quant role map LOSES to EXL3-budgeted (measured: "
              "17.15 vs 14.30 PPL at more bits). Only use a recipe measured in EXL3's atom space.")
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
    if work:
        os.makedirs(work, exist_ok=True)
    r = subprocess.run(cmd)
    # verify REAL output (convert_model can print an arg error yet exit 0) — trust files, not returncode
    made = a.out and os.path.isdir(a.out) and any(
        f.endswith(".safetensors") for f in os.listdir(a.out)) and \
        os.path.exists(os.path.join(a.out, "config.json"))
    if r.returncode != 0 or not made:
        sys.exit(f"exllamav3 convert failed (exit {r.returncode}; output {'present' if made else 'MISSING'}) "
                 f"— see console above.")
    try:
        import pollard_workspace as ws
        ws.record_build(a.model, "exl3", a.out, tag=f"{a.bpw}bpw", bpw=a.bpw)
    except Exception:
        pass
    print(f"wrote EXL3 model -> {a.out}\n  run:  exllamav3 / TabbyAPI loads {a.out}"
          f"\n  VERIFY:  pollard-verify --model {a.out} --source {a.model} --end-to-end")


if __name__ == "__main__":
    main()
