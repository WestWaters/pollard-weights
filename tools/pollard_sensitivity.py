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
                          read_gguf_tensor_names, find_llama_bin,
                          detect_available_ram_gb, imatrix_covered_tensors)
from pollard_fit import LADDER, PRESET, IMATRIX_REQUIRED_PRESETS, BPW


def _kl(ppl, model, eval_f, base, rpc=None):
    """Mean KL-divergence of `model` vs the base logits, or None on failure.
    `rpc` (host:port[,host:port…]) pools RPC nodes so a model too big for one box
    can run the forward pass — the only way to profile a 300B+ MoE on one Spark."""
    cmd = [ppl, "-m", model, "-f", eval_f, "--kl-divergence",
           "--kl-divergence-base", base, "-ngl", "99", "-c", "512"]
    if rpc:
        cmd += ["--rpc", rpc]
    r = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"Mean\s+KLD:\s*(-?[0-9.]+)", r.stdout + r.stderr)
    return float(m.group(1)) if m else None


def _valid_gguf(path):
    """A quantize output is valid only if it starts with the GGUF magic — a crashed
    build leaves an all-zeros file that 'exists' but won't load, turning into a
    cryptic downstream error. Check the magic and fail loud at the source instead."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except Exception:
        return False


# GPU/Metal out-of-memory signatures — a forward pass at -ngl 99 that doesn't fit
# the GPU crashes HERE, not with a quant error. Detect it and say the actual fix
# instead of a misleading "imatrix-uncovered tensor" guess.
_OOM = re.compile(
    r"Insufficient Memory|kIOGPUCommandBufferCallbackErrorOutOfMemory|out of memory"
    r"|cudaMalloc|failed to allocate|command buffer .* failed with status 5"
    r"|recommendedMaxWorkingSetSize", re.I)


def _diagnose(stderr, what):
    """Turn a swallowed subprocess failure into an actionable message. OOM at -ngl 99
    is the common footgun: an f16/bf16 model too big for the GPU. Give the fix."""
    s = stderr or ""
    if _OOM.search(s):
        return (f"{what} ran OUT OF GPU MEMORY — the model is too big to run at "
                f"-ngl 99 on this GPU/Metal. Fix: lower --ram so the reference fits, "
                f"offload fewer layers, or build the imatrix on a smaller quantized "
                f"host (e.g. Q8_0) instead of f16/bf16. (This is memory, not a bug.)")
    tail = "\n".join(l for l in s.strip().splitlines() if l.strip())[-400:]
    return f"{what} failed. Last output:\n{tail}" if tail else f"{what} failed (no output)."


def _run(cmd):
    """Run a llama.cpp step, keep its output so failures can be diagnosed."""
    return subprocess.run(cmd, capture_output=True, text=True)


# standard per-layer matmuls the imatrix always covers; ANYTHING else (routers,
# norms, exotic per-model tensors like DeepSeek's compressors) must not be crushed
# to an imatrix-only IQ2 type in a uniform noise build or llama-quantize hard-fails.
_STD_MATMUL = re.compile(
    r"blk\.\d+\.(ffn_(up|down|gate)(_exps)?|attn_(q|k|v|qkv|output))\.weight$")


def _uncoverable_pins(gguf, imatrix):
    """--tensor-type '<name>=q6_K' for every weight the IMATRIX DOESN'T COVER, so an
    aggressive IQ2/IQ1 build can't hard-fail on it. This catches the ones a name
    heuristic misses — MTP/`nextn` layer tensors look like standard attention
    (`blk.64.attn_k.weight`) but are never calibrated, so llama-quantize refuses
    them at low bit. Falls back to the name heuristic if the imatrix can't be read.
    (token_embd/output are handled by their own flags.)"""
    covered = imatrix_covered_tensors(imatrix)
    lines = []
    for nm in read_gguf_tensor_names(gguf):
        if nm in ("token_embd.weight", "output.weight"):
            continue
        uncov = (nm not in covered) if covered is not None else (not _STD_MATMUL.search(nm))
        if uncov:
            lines.append(f"{re.escape(nm)}=q6_K")
    return lines                                           # written to a --tensor-type-file


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
    ap.add_argument("--ram", help="usable RAM/VRAM in GB (or 'auto') to CALIBRATE ON A "
                                  "SMALL BOX: if f16 won't fit the forward pass, the base "
                                  "reference drops to the highest quant that fits — Pollard "
                                  "MAKES big models on small hardware, not just runs them.")
    a = ap.parse_args()

    a.llama_quantize = find_llama_bin(a.llama_quantize)
    a.llama_perplexity = find_llama_bin(a.llama_perplexity)
    missing = [n for n, v in (("llama-quantize", a.llama_quantize),
                              ("llama-perplexity", a.llama_perplexity)) if v is None]
    if missing:
        sys.exit(f"ERROR: {', '.join(missing)} not found. install.sh builds these into "
                 f"runtime/llama.cpp/build/bin — re-run install.sh, or pass the path.")

    # fail loud NOW if the imatrix is missing/empty — otherwise every quantize below
    # silently no-ops and you sit staring at a dead run. (A too-big f16 imatrix job
    # that OOM'd on the GPU leaves no file — that's the usual cause; build it on a Q8.)
    if not os.path.exists(a.imatrix) or os.path.getsize(a.imatrix) < 1024:
        sys.exit(f"ERROR: imatrix not found or empty: {a.imatrix}\n"
                 f"  build it first: llama-imatrix -m <Q8-or-smaller-host>.gguf -f calib.txt "
                 f"-o {a.imatrix} -ngl 99\n"
                 f"  (use a Q8_0 host, NOT f16/bf16, if the full model won't fit your GPU.)")
    if not os.path.exists(a.eval) or os.path.getsize(a.eval) == 0:
        sys.exit(f"ERROR: eval corpus not found or empty: {a.eval}")

    arch = analyse(gguf_to_config(read_gguf_meta(a.gguf), a.gguf))
    layers = arch["layers"]
    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    tmp = tempfile.mkdtemp(prefix="pollard_sens_")
    base = os.path.join(tmp, "base.dat")
    ref = os.path.join(tmp, "ref.gguf")
    probe = os.path.join(tmp, "probe.gguf")

    print(f"== pollard-sensitivity :: {a.gguf}  ({layers} layers, groups={groups})")
    pins = _uncoverable_pins(a.gguf, a.imatrix)         # imatrix-uncovered tensors (MTP/norm/exotic)
    pin_file = os.path.join(tmp, "pins.txt")
    if pins:                                            # a file, not hundreds of CLI args
        open(pin_file, "w").write("\n".join(pins) + "\n")
        print(f"  {len(pins)} imatrix-uncovered tensors pinned to q6_K (MTP/norm/exotic)")
    PINARG = ["--tensor-type-file", pin_file] if pins else []

    # --- memory-adaptive reference: Pollard MAKES big models on small hardware ---
    # The sweep runs FORWARD passes, so the base must fit RAM. f16 is the ideal ground
    # truth but rarely fits a big model on a small box, so drop the base to the highest
    # ladder type that fits and measure against THAT. It can't see the f16->that-type
    # loss, but it measures the crush-from-here regime — exactly the allocation decision.
    base_src, base_note, mem_budget = a.gguf, "f16 (ground truth)", None
    if a.ram:
        ram = detect_available_ram_gb() if str(a.ram).lower() == "auto" else float(a.ram)
        mem_budget = (ram or 0) * 0.72                  # usable for a forward pass
        f16_gb = arch["total"] * 16 / 8 / 1e9
        if mem_budget and f16_gb > mem_budget:
            fit = next((t for t in LADDER if arch["total"] * BPW[t] / 8 / 1e9 <= mem_budget),
                       LADDER[-1])
            fit_gb = arch["total"] * BPW[fit] / 8 / 1e9
            base_src = os.path.join(tmp, "membase.gguf")
            a.ref = PRESET[fit]                         # probes' baseline must also fit
            base_note = f"{fit} (memory-fit; f16 too big for ~{mem_budget:.0f} GB)"
            print(f"[--ram {ram:.0f}] f16 is {f16_gb:.0f} GB > ~{mem_budget:.0f} GB usable — "
                  f"basing on {PRESET[fit]} ({fit_gb:.0f} GB). Signal is vs {fit}, not f16 "
                  f"(a touch weaker, but it fits YOUR box).")
            subprocess.run([a.llama_quantize, "--imatrix", a.imatrix]
                           + (PINARG if PRESET[fit] in IMATRIX_REQUIRED_PRESETS else [])
                           + [a.gguf, base_src, PRESET[fit]], capture_output=True)
    print(f"reference={a.ref}  probe={a.probe}  base={base_note}  — {len(groups)*layers} passes")

    if a.rpc:
        print(f"RPC pool: {a.rpc}  (forward passes span these nodes; quantize stays local)")
    # 1. base logits from the reference (f16, or the memory-fit quant)
    base_cmd = [a.llama_perplexity, "-m", base_src, "-f", a.eval,
                "--kl-divergence-base", base, "-ngl", "99", "-c", "512"]
    if a.rpc:
        base_cmd += ["--rpc", a.rpc]
    r = _run(base_cmd)
    if not os.path.exists(base) or os.path.getsize(base) == 0:
        sys.exit(f"ERROR: could not build the reference logits — {_diagnose(r.stderr, 'the base forward pass')}")
    if base_src != a.gguf and not _valid_gguf(base_src):
        sys.exit(f"ERROR: the memory-fit base build ({a.ref}) came out invalid — "
                 f"{_diagnose(r.stderr, 'llama-quantize')}")
    # 2. reference build — SAME pins as the base (an IQ2 ref crashes on uncovered tensors too)
    r = _run([a.llama_quantize, "--imatrix", a.imatrix]
             + (PINARG if a.ref in IMATRIX_REQUIRED_PRESETS else [])
             + [a.gguf, ref, a.ref])
    if not _valid_gguf(ref):
        sys.exit(f"ERROR: the reference build ({a.ref}) came out invalid — {_diagnose(r.stderr, 'llama-quantize')}")
    kl_ref = _kl(a.llama_perplexity, ref, a.eval, base, a.rpc)
    if kl_ref is None:
        sys.exit("ERROR: reference GGUF built fine but perplexity couldn't score it — "
                 "check the eval file, the tools, and free memory (the forward pass runs here).")
    print(f"reference KL = {kl_ref:.5f}")

    # 2b. the NOISE curve for THIS model — uniform KL at each ladder rung. This is
    # the per-type KL cost the allocator needs, MEASURED per model (it shifts a bit
    # model to model), so nothing is baked in. Cheap next to the sensitivity sweep.
    noise = {}
    print("noise curve (uniform KL per type):")
    for t in LADDER:
        if mem_budget and arch["total"] * BPW[t] / 8 / 1e9 > mem_budget:
            noise[t] = None                            # too big to eval here; interpolated
            print(f"  {t:8} (skipped — {arch['total']*BPW[t]/8/1e9:.0f} GB > budget; interpolated)")
            continue
        uni = os.path.join(tmp, "uni.gguf")
        cmd = [a.llama_quantize, "--imatrix", a.imatrix,
               "--token-embedding-type", t, "--output-tensor-type", t]
        if PRESET[t] in IMATRIX_REQUIRED_PRESETS:      # pin uncoverable tensors so
            cmd += PINARG                              # the IQ2 build doesn't crash
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
            # crush this group, keep the a.ref base — but the base needs the SAME pins
            # (an IQ2 base crashes on uncovered tensors), so PINARG goes too.
            subprocess.run([a.llama_quantize, "--imatrix", a.imatrix, "--tensor-type", pat]
                           + (PINARG if a.ref in IMATRIX_REQUIRED_PRESETS else [])
                           + [a.gguf, probe, a.ref], capture_output=True)
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
               "layers": layers, "source": os.path.basename(a.gguf), "base": base_note}
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
