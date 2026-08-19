#!/usr/bin/env python3
"""pollard-sensitivity — measure each tensor group's TRUE KL cost, per model.

This is the calibration that lets pollard-fit BEAT uniform quants. Instead of
guessing which tensors matter (or trusting imatrix magnitude, which lies — big
activations are NOT the same as high KL sensitivity), it CRUSHES one group at a
time to a probe quant and measures the actual KL hit vs a reference build. The
layers that barely move can be compressed hard; the ones that spike must be kept
high. Emits a profile that `pollard-fit --sensitivity` allocates on.

It is not free — one build + one KL eval per group (≈ 2·num_layers passes). But
it is one-time per model, and it is exactly the "calibration" the good quantizers
(Unsloth Dynamic, etc.) pay for. Run it once, reuse the profile for every build.

    pollard-sensitivity --gguf model-f16.gguf --imatrix imatrix.dat \\
        --eval held-out.txt --out model.sensitivity.json

Then:  pollard-fit --gguf model-f16.gguf --ram 16 --imatrix imatrix.dat \\
           --sensitivity model.sensitivity.json

Measurement notes: use a held-out eval corpus (disjoint from the imatrix
calibration) so the sensitivity is not overfit. GPU strongly recommended — this
is many perplexity passes.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

from pollard_calc import (read_gguf_meta, gguf_to_config, analyse,
                          read_gguf_tensor_names)
from pollard_fit import LADDER, PRESET, IMATRIX_REQUIRED_PRESETS


def _kl(ppl, model, eval_f, base, rpc=None):
    """Mean KL-divergence of `model` vs the base logits, or None on failure.
    `rpc` (host:port[,host:port…]) pools RPC nodes so a model too big for one box
    can run the forward pass — the only way to profile a 300B+ MoE on one Spark."""
    cmd = [ppl, "-m", model, "-f", eval_f, "--kl-divergence",
           "--kl-divergence-base", base, "-ngl", "99", "-c", "512"]
    if rpc:
        cmd += ["--rpc", rpc]
    r = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"Mean\s+KLD:\s*([0-9.]+)", r.stdout + r.stderr)
    return float(m.group(1)) if m else None


# standard per-layer matmuls the imatrix always covers; ANYTHING else (routers,
# norms, exotic per-model tensors like DeepSeek's compressors) must not be crushed
# to an imatrix-only IQ2 type in a uniform noise build or llama-quantize hard-fails.
_STD_MATMUL = re.compile(
    r"blk\.\d+\.(ffn_(up|down|gate)(_exps)?|attn_(q|k|v|qkv|output))\.weight$")


def _uncoverable_pins(gguf):
    """--tensor-type '<name>=q6_K' args for every tensor that isn't a standard
    per-layer matmul, so a uniform IQ2 noise build can't crash on an imatrix-
    uncovered tensor (token_embd/output are handled by their own flags)."""
    args = []
    for nm in read_gguf_tensor_names(gguf):
        if nm in ("token_embd.weight", "output.weight"):
            continue
        if not _STD_MATMUL.search(nm):
            args += ["--tensor-type", f"{re.escape(nm)}=q6_K"]
    return args


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True, help="source GGUF (f16/bf16)")
    ap.add_argument("--imatrix", required=True, help="importance matrix (llama-imatrix)")
    ap.add_argument("--eval", required=True, help="held-out eval text (disjoint from imatrix)")
    ap.add_argument("--out", help="profile path (default: <gguf>.sensitivity.json)")
    ap.add_argument("--groups", default="ffn,attn",
                    help="tensor groups to probe, comma-separated (default ffn,attn)")
    ap.add_argument("--probe", default="iq2_xxs", help="crush type for the probe (default iq2_xxs)")
    ap.add_argument("--ref", default="IQ4_XS", help="reference uniform preset (default IQ4_XS)")
    ap.add_argument("--llama-quantize", default="llama-quantize")
    ap.add_argument("--llama-perplexity", default="llama-perplexity")
    ap.add_argument("--rpc", help="RPC servers to pool for the forward pass, "
                                  "'host:port[,host:port…]' (run ggml-rpc-server on each "
                                  "peer). REQUIRED to profile a model too big for one node "
                                  "— the quantize step streams and needs no RPC.")
    a = ap.parse_args()

    for tool in (a.llama_quantize, a.llama_perplexity):
        import shutil
        if shutil.which(tool) is None and not os.path.exists(tool):
            sys.exit(f"ERROR: '{tool}' not found — build llama.cpp (install.sh) or pass the path.")

    arch = analyse(gguf_to_config(read_gguf_meta(a.gguf), a.gguf))
    layers = arch["layers"]
    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    tmp = tempfile.mkdtemp(prefix="pollard_sens_")
    base = os.path.join(tmp, "base.dat")
    ref = os.path.join(tmp, "ref.gguf")
    probe = os.path.join(tmp, "probe.gguf")

    print(f"== pollard-sensitivity :: {a.gguf}  ({layers} layers, groups={groups})")
    print(f"reference={a.ref}  probe={a.probe}  — {len(groups)*layers} measured passes")

    if a.rpc:
        print(f"RPC pool: {a.rpc}  (forward passes span these nodes; quantize stays local)")
    # 1. base logits from the full-precision source (forward pass -> may need RPC)
    base_cmd = [a.llama_perplexity, "-m", a.gguf, "-f", a.eval,
                "--kl-divergence-base", base, "-ngl", "99", "-c", "512"]
    if a.rpc:
        base_cmd += ["--rpc", a.rpc]
    subprocess.run(base_cmd, capture_output=True)
    # 2. reference build (quantize = local, streams from disk, no RPC) + its KL
    subprocess.run([a.llama_quantize, "--imatrix", a.imatrix, a.gguf, ref, a.ref],
                   capture_output=True)
    kl_ref = _kl(a.llama_perplexity, ref, a.eval, base, a.rpc)
    if kl_ref is None:
        sys.exit("ERROR: could not measure reference KL — check the eval file and tools.")
    print(f"reference KL = {kl_ref:.5f}")

    # 2b. the NOISE curve for THIS model — uniform KL at each ladder rung. This is
    # the per-type KL cost the allocator needs, MEASURED per model (it shifts a bit
    # model to model), so nothing is baked in. Cheap next to the sensitivity sweep.
    noise = {}
    pins = _uncoverable_pins(a.gguf)                    # exotic/router/norm tensors
    print("noise curve (uniform KL per type):")
    for t in LADDER:
        uni = os.path.join(tmp, "uni.gguf")
        cmd = [a.llama_quantize, "--imatrix", a.imatrix,
               "--token-embedding-type", t, "--output-tensor-type", t]
        if PRESET[t] in IMATRIX_REQUIRED_PRESETS:      # pin uncoverable tensors so
            cmd += pins                                # the IQ2 build doesn't crash
        cmd += [a.gguf, uni, PRESET[t]]
        subprocess.run(cmd, capture_output=True)
        k = _kl(a.llama_perplexity, uni, a.eval, base, a.rpc)
        noise[t] = k if k is not None else None
        os.path.exists(uni) and os.remove(uni)
        print(f"  {t:8} {noise[t]}{'  (uncovered-tensor build failed)' if noise[t] is None else ''}")

    # 3. crush each group in each layer, measure the KL cost over the reference
    profile = {g: {} for g in groups}
    failed = []
    for i in range(layers):
        for g in groups:
            pat = rf"blk\.{i}\.{g}_.*={a.probe}"
            subprocess.run([a.llama_quantize, "--imatrix", a.imatrix,
                            "--tensor-type", pat, a.gguf, probe, a.ref],
                           capture_output=True)
            k = _kl(a.llama_perplexity, probe, a.eval, base, a.rpc)
            # a FAILED probe (build/eval crashed) is UNMEASURED, not zero-cost —
            # recording 0 would tell the allocator this is the LEAST important group
            # and crush it hardest. Mark None; protect it after the sweep.
            profile[g][str(i)] = (max(0.0, k - kl_ref) if k is not None else None)
            if k is None:
                failed.append(f"{g}.{i}")
            os.path.exists(probe) and os.remove(probe)
        done = (i + 1) / layers
        print(f"  layer {i:>3}/{layers}  " +
              "  ".join(f"{g}={profile[g][str(i)]}" for g in groups) +
              f"   [{done*100:4.0f}%]")

    # failed probes -> PROTECT (max sensitivity), never crush. Loud about it.
    if failed:
        for g in groups:
            vals = [v for v in profile[g].values() if v is not None]
            hi = max(vals) if vals else 1.0
            for k2, v in profile[g].items():
                if v is None:
                    profile[g][k2] = hi
        print(f"\nWARNING: {len(failed)} probe build(s) failed (likely imatrix-uncovered "
              f"tensors) — those groups were set to MAX sensitivity (PROTECTED), not 0, "
              f"so the allocator keeps their bits. Groups: "
              f"{', '.join(failed[:8])}{' …' if len(failed) > 8 else ''}")

    for f in (base, ref):
        os.path.exists(f) and os.remove(f)
    os.rmdir(tmp) if not os.listdir(tmp) else None

    out = a.out or a.gguf.rsplit(".gguf", 1)[0] + ".sensitivity.json"
    payload = {**profile, "noise": noise, "ref": a.ref, "probe": a.probe,
               "layers": layers, "source": os.path.basename(a.gguf)}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    # a quick read on the signal we found
    for g in groups:
        vals = list(profile[g].values())
        lo, hi = min(vals), max(vals)
        print(f"  {g}: spread {hi/max(lo,1e-6):.0f}x  (min {lo:.4f}  max {hi:.4f})")
    print(f"\ndone: {out}\nuse it:  pollard-fit --gguf {a.gguf} --ram <GB> "
          f"--imatrix {a.imatrix} --sensitivity {out}")


if __name__ == "__main__":
    main()
