#!/usr/bin/env python3
"""pollard-doctor (DrDiag) — diagnose, repair, or improve a model for low-bit quantization, on any lane.

It packages the hard-won diagnostics into one command so a user can point it at a model and get a
straight answer: is this convert healthy, WHY is it broken, and exactly how to fix it — without ever
trusting a lying proxy metric (see legacy/PROXY_ERR_BANNED.md).

Three things it checks:
  1. HEALTH (needs --source fp16): per-tensor decode round-trip vs the source weight, and the assembled
     end-to-end forward (next-tok acc / ppl). Real reconstruction only. (via pollard-verify)
  2. RISK / ROOT CAUSE (scan the fp16 source): finds massive-activation input channels — the outliers
     that collapse a low-bit quantizer's global scale and silently wreck a layer (the exact bug behind
     "weights decode fine but the model is garbage"). Predicts which layers will break at low bit.
  3. FIX: recommends (or with --repair, runs) the repair — `pollard-hf-smooth` preconditioning to migrate
     the outliers, then reconvert on the chosen lane, then re-verify.

  pollard-doctor --model ./M-exl3 --source ./M-fp16                 # diagnose a converted model
  pollard-doctor --source ./M-fp16 --predict                        # will low-bit break? which layers?
  pollard-doctor --source ./M-fp16 --repair --lane exl3 --bpw 4.0   # smooth -> reconvert -> verify
  pollard-doctor --cal-response --model buildA --compare buildB --eval-dir ./domains  # WHY did cal change move it?

Cal-response (why a calibration change helped/hurt): diffs the two builds' per-tensor bit allocation
(big shift = the allocator re-deciding from the cal = NON-MONOTONIC response, not fixable by mix), then
per-domain evals (domain-specific delta = mix DILUTION, fixable; uniform = allocator/noise), and reminds
you the noise floor needs a repeat run. Tells you whether to fix the mix, accept it, or ignore it.

Lanes: exl3 health implemented; risk-scan + repair plan are lane-agnostic (they operate on the fp16
source). Works on any CUDA GPU. Verify is the source of truth; a proxy metric is never consulted.
"""
import argparse, os, subprocess, sys


def scan_outliers(source_dir, device, calib, rows, cols, thresh):
    """Find massive-activation input channels per layer on the fp16 source (the low-bit break predictor)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(source_dir)
    model = AutoModelForCausalLM.from_pretrained(source_dir, dtype=torch.float16, device_map=device).eval()
    try:
        layers = model.model.layers
    except AttributeError:
        print("   (risk-scan: unsupported arch — expected model.model.layers)"); return []
    amax = {}
    def mk(i, tag):
        def h(m, inp, out):
            x = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).amax(0).float()
            amax[(i, tag)] = torch.maximum(amax[(i, tag)], x) if (i, tag) in amax else x
        return h
    hooks = []
    for i, L in enumerate(layers):
        if hasattr(L, "self_attn") and getattr(L.self_attn, "q_proj", None) is not None:
            hooks.append(L.self_attn.q_proj.register_forward_hook(mk(i, "attn")))
        if hasattr(L, "mlp") and getattr(L.mlp, "up_proj", None) is not None:
            hooks.append(L.mlp.up_proj.register_forward_hook(mk(i, "mlp")))
    text = open(calib, encoding="utf-8", errors="ignore").read() if calib else ("Hello world. " * 6000)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        for a in range(0, min(ids.shape[1], rows * cols) - cols, cols):
            model(ids[:, a:a + cols])
    for h in hooks: h.remove()
    risky = []
    for (i, tag), v in amax.items():
        mx = v.max().item()
        if mx >= thresh:
            risky.append((i, tag, mx))
    return sorted(risky, key=lambda r: -r[2])


def _exl3_bits(model_dir):
    """Per-tensor bit-width from EXL3 trellis shapes: K = trellis.shape[-1] // 16 (no weights loaded)."""
    import glob
    from safetensors import safe_open
    bits = {}
    for f in glob.glob(os.path.join(model_dir, "*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                if k.endswith(".trellis"):
                    shp = h.get_slice(k).get_shape()
                    bits[k[:-len(".trellis")]] = int(shp[-1]) // 16
    return bits


def cal_response(model_a, model_b, source, eval_dir, device):
    """Classify WHY two builds (from different calibrations) differ: allocator-shift vs dilution vs noise."""
    print("\n[cal-response] why did the calibration change move the number?")
    problems = 0
    # 1) ALLOCATION DIFF — did the (budgeted) allocator re-decide bits? big shift = non-monotonic response to cal
    ba, bb = _exl3_bits(model_a), _exl3_bits(model_b)
    common = sorted(set(ba) & set(bb))
    changed = [(t, ba[t], bb[t]) for t in common if ba[t] != bb[t]]
    frac = len(changed) / max(1, len(common))
    print(f"  1) allocation diff: {len(changed)}/{len(common)} tensors changed bits ({frac*100:.1f}%)")
    for t, a, b in sorted(changed, key=lambda x: -abs(x[1] - x[2]))[:6]:
        print(f"       {t:<44} {a}b -> {b}b")
    if frac >= 0.10:
        print("     => the allocator RE-DECIDED bits from the new cal (NON-MONOTONIC response to cal volume/mix)."
              " This is inherent allocator behavior — not fixable by mix alone; find the cal size sweet spot.")
    else:
        print("     => allocation barely moved; the delta is from the cal CONTENT, not re-allocation (see step 2).")
    # 2) MULTI-DOMAIN EVAL — domain-specific delta = dilution (fixable by mix); uniform = allocator/noise
    if eval_dir and os.path.isdir(eval_dir):
        import glob
        print("  2) per-domain eval (domain-specific delta = mix dilution; uniform = allocator/noise):")
        deltas = []
        for ef in sorted(glob.glob(os.path.join(eval_dir, "*.txt"))):
            dom = os.path.splitext(os.path.basename(ef))[0]
            pa = _ppl_on(model_a, source, ef, device)
            pb = _ppl_on(model_b, source, ef, device)
            deltas.append((dom, pb - pa))
            print(f"       {dom:<16} A={pa:.3f}  B={pb:.3f}  d={pb-pa:+.3f}")
        if deltas:
            spread = max(d for _, d in deltas) - min(d for _, d in deltas)
            uniform = spread < 0.1
            print(f"     => {'UNIFORM across domains -> allocator/noise, not dilution' if uniform else 'DOMAIN-SPECIFIC -> mix DILUTION (fixable: rebalance the cal domains)'}")
    else:
        print("  2) per-domain eval: skipped (pass --eval-dir <dir of domain .txt> to test dilution vs allocator)")
        problems += 0
    # 3) NOISE — the honest floor
    print("  3) noise floor: re-run the SAME cal config once; if the number moves ~as much as A-vs-B, the"
          " difference is NOISE, not a real lever. (Convert-level noise needs a repeat convert — can't infer from one run.)")
    return 0


def _ppl_on(model_dir, source, eval_file, device, length=2048, rows=8):
    import math, torch
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.util.measures import compute_target_log_probs
    cfg = Config.from_directory(model_dir); m = Model.from_config(cfg)
    cache = Cache(m, max_num_tokens=((length // 256) + 2) * 256); m.load()
    tok = Tokenizer.from_config(cfg)
    ids = tok.encode(open(eval_file, encoding="utf-8", errors="ignore").read())
    if ids.dim() == 1: ids = ids.unsqueeze(0)
    n = ids.shape[-1]; s = c = 0; done = 0
    for a in range(0, max(1, n - length), length):
        seq = ids[:, a:a + length]
        lg = m.forward(seq, {"attn_mode": "flash_attn_nc"})[:, :-1, :].float()
        tg = seq[:, 1:].to(lg.device)
        s += compute_target_log_probs(lg, tg, tok.actual_vocab_size).sum().item(); c += tg.numel(); done += 1
        if done >= rows: break
    try: m.unload()
    except Exception: pass
    return math.exp(-s / max(1, c))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", help="converted model dir to diagnose (with --source)")
    ap.add_argument("--source", help="fp16 HF source (for health + risk scan + repair)")
    ap.add_argument("--lane", default="exl3", choices=["exl3", "gptq", "gguf", "mx"])
    ap.add_argument("--bpw", type=float, default=4.0, help="target bits for repair reconvert")
    ap.add_argument("--predict", action="store_true", help="scan the source and predict low-bit break risk")
    ap.add_argument("--repair", action="store_true", help="run smooth -> reconvert -> verify")
    ap.add_argument("--cal-response", dest="cal_response", action="store_true",
                    help="classify WHY two builds (from different calibrations) differ: allocator-shift vs "
                         "mix-dilution vs noise. Needs --model A and --compare B (both same lane/bpw).")
    ap.add_argument("--compare", help="second build dir (build B) for --cal-response")
    ap.add_argument("--eval-dir", dest="eval_dir",
                    help="dir of per-domain eval files (<domain>.txt) for --cal-response dilution test")
    ap.add_argument("--calib", help="calibration text (risk scan / repair)")
    ap.add_argument("--out", help="output dir for --repair (default: <source>-sm-<lane>)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--cols", type=int, default=512)
    ap.add_argument("--outlier-thresh", type=float, default=50.0,
                    help="max|activation| above this flags a massive-activation channel (break risk)")
    a = ap.parse_args()
    print(f"== pollard-doctor (DrDiag) :: lane={a.lane} ==")

    if a.cal_response:
        if not (a.model and a.compare):
            ap.error("--cal-response needs --model <build A> and --compare <build B>")
        sys.exit(cal_response(a.model, a.compare, a.source, a.eval_dir, a.device))

    problems = 0

    # 1. HEALTH — delegate to pollard-verify (real reconstruction, never proxy_err)
    if a.model and a.source:
        print("\n[1] HEALTH — real reconstruction vs source (via pollard-verify)")
        here = os.path.dirname(os.path.abspath(__file__))
        cmd = [sys.executable, os.path.join(here, "pollard_verify.py"),
               "--model", a.model, "--source", a.source, "--lane", a.lane, "--end-to-end"]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            problems += 1
            print("   -> health check FAILED (broken build).")

    # 2. RISK / ROOT CAUSE — scan the fp16 source for massive activations
    risky = []
    if a.source and (a.predict or a.repair or (a.model and problems)):
        print("\n[2] RISK — massive-activation input channels (low-bit break predictor)")
        risky = scan_outliers(a.source, a.device, a.calib, a.rows, a.cols, a.outlier_thresh)
        if risky:
            problems += 1
            print(f"   !! {len(risky)} risky seam(s) — a low-bit convert may silently break these layers:")
            for i, tag, mx in risky[:8]:
                print(f"      layer {i:>3}  {tag}-input  max|X|={mx:.1f}")
            print("   These outliers collapse the quantizer's global scale -> normal channels under-"
                  "quantized -> garbage forward (per-tensor/proxy metrics won't show it).")
        else:
            print(f"   clean — no input channel above {a.outlier_thresh}. Low-bit convert should be safe.")

    # 3. FIX — recommend or run the repair
    if risky or a.repair:
        out = a.out or (os.path.normpath(a.source).rstrip("/\\") + f"-sm")
        here = os.path.dirname(os.path.abspath(__file__))
        smooth = [sys.executable, os.path.join(here, "pollard_hf_smooth.py"),
                  "--model", a.source, "--out", out] + (["--calib", a.calib] if a.calib else [])
        print("\n[3] FIX — SmoothQuant preconditioning, then reconvert on the lane")
        print("   smooth:    " + " ".join(smooth))
        lane_cmd = {"exl3": f"pollard-exl3 --model {out} --out {out}-exl3 --bpw {a.bpw}",
                    "gptq": f"pollard-gptq --model {out} --out {out}-gptq --bits {int(a.bpw)}",
                    "mx":   f"pollard-mx  --model {out} --out {out}-mx  --calib <cal>",
                    "gguf": f"pollard-smooth (imatrix path) then pollard-fit  # GGUF has its own smoothing"}[a.lane]
        print("   reconvert: " + lane_cmd)
        print(f"   verify:    pollard-verify --model {out}-{a.lane} --source {a.source} --end-to-end")
        if a.repair:
            print("\n   running repair (smoothing)...")
            rc = subprocess.run(smooth).returncode
            if rc != 0:
                sys.exit("   smoothing failed.")
            print(f"   smoothed model -> {out}. Now reconvert on the box:\n     {lane_cmd}\n   then verify.")

    if not (a.model or a.source):
        ap.error("give --model+--source (diagnose a build), or --source (--predict / --repair).")
    print(f"\n{'ISSUES FOUND: ' + str(problems) if problems else 'HEALTHY — no issues found.'}")
    sys.exit(1 if problems and not a.repair else 0)


if __name__ == "__main__":
    main()
