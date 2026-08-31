#!/usr/bin/env python3
"""pollard — the autoaware entry point. Point it at ANY model; it detects dense vs MoE
and drives the RIGHT path, so a user who doesn't know (or care) which their model is
never runs the wrong tool or wastes a run:

  DENSE  ->  imatrix-guided K-quants          (dispatches to `pollard-fit`)
  MoE    ->  automap measured expert-allocation (dry-run -> `pollard-automap` -> build)

Plans by default (prints the exact commands for THIS model); `--run` executes them.

    pollard --gguf model-f16.gguf --ram 16 --imatrix model.imatrix           # plan
    pollard --gguf moe-f16.gguf   --imatrix moe.imatrix --run                # detect + build
"""
import argparse, os, subprocess, sys

from pollard_calc import read_gguf_meta, gguf_to_config, analyse, find_llama_bin

# reference bits-per-weight for the common formats, so a user can see where a Pollard
# build lands vs f16 / NVFP4 / the usual GGUF tiers ("half the size of NVFP4" etc.).
_REF_BPW = [("f16", 16.0), ("Q8_0", 8.5), ("Q6_K", 6.6), ("Q5_K_M", 5.5),
            ("NVFP4", 4.25), ("Q4_K_M", 4.85), ("IQ2 (~2-bit)", 2.1)]


def size_ladder(params_b, pollard_gb=None, pollard_label="Pollard"):
    """Print the shrink story: f16 size, where Pollard's build lands, and the same model
    at each reference format — so people SEE what Pollard did and can compare to NVFP4."""
    if not params_b:
        return
    gb = lambda bpw: params_b * 1e9 * bpw / 8 / 1e9
    f16 = gb(16.0)
    print("   --- what Pollard did (size) ---")
    if pollard_gb:
        pct = 100 * (1 - pollard_gb / f16)
        ratio = f16 / max(pollard_gb, 1e-9)
        bpw = pollard_gb * 8 * 1e9 / (params_b * 1e9)
        print(f"   {pollard_label}: {f16:.1f} GB (f16) -> {pollard_gb:.2f} GB  "
              f"(-{pct:.0f}%, {ratio:.1f}x smaller, {bpw:.2f} bpw)")
    print("   same model at each format:  " +
          "  ".join(f"{n} {gb(b):.1f}GB" for n, b in _REF_BPW) +
          (f"  ->  {pollard_label} {pollard_gb:.2f}GB" if pollard_gb else ""))


def _run(cmd, do_run, cwd=None):
    print("   $ " + " ".join(str(c) for c in cmd))
    if do_run:
        r = subprocess.run(cmd, cwd=cwd)
        if r.returncode != 0:
            sys.exit(f"   step failed (exit {r.returncode}).")
    return not do_run


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", required=True, help="f16/bf16 source GGUF (convert from HF first)")
    ap.add_argument("--imatrix", help="importance matrix (required for the low-bit build)")
    ap.add_argument("--ram", default="16", help="RAM budget in GB for the dense memory-fit build")
    ap.add_argument("--out", help="output path (dense build)")
    ap.add_argument("--eval", default="wikitext2_test.txt")
    ap.add_argument("--bin", help="llama.cpp bin dir (for the MoE dry-run/build)")
    ap.add_argument("--run", action="store_true", help="execute the path (default: plan/print it)")
    ap.add_argument("--benchmark", "--reproduce", dest="benchmark", action="store_true",
                    help="ALSO run the gold-card benchmark (3-bar comparison + PPL) — the "
                         "hour-long validation. OFF by default: a normal build makes ONE model "
                         "fast and skips the eval. Use this only to reproduce our published numbers.")
    ap.add_argument("--force-dense", action="store_true", help="override detection -> dense path")
    ap.add_argument("--force-moe", action="store_true", help="override detection -> MoE path")
    a = ap.parse_args()

    arch = analyse(gguf_to_config(read_gguf_meta(a.gguf), a.gguf))
    is_moe = (arch.get("n_experts") or 0) > 0 or "moe" in str(arch.get("kind", "")).lower()
    if a.force_dense: is_moe = False
    if a.force_moe: is_moe = True
    tag = "MoE" if is_moe else "DENSE"
    print(f"pollard :: {os.path.basename(a.gguf)}")
    print(f"   detected {tag}  ({(arch.get('total') or 0)/1e9:.1f}B total, "
          f"{(arch.get('active') or arch.get('total') or 0)/1e9:.1f}B active, {arch.get('layers')}L, "
          f"{arch.get('n_experts') or 0} experts)")
    # show the size ladder so the user sees the shrink + can compare to NVFP4/Q4/etc.
    # (actual built size is reported by the build step; this is where Pollard will land)
    size_ladder((arch.get("total") or 0) / 1e9)

    if not is_moe:
        # DENSE -> imatrix-guided K-quants (pollard-fit auto-sizes to RAM; NO sensitivity sweep).
        print("   path: DENSE -> imatrix-guided K-quants via pollard-fit (no sensitivity sweep — "
              "it doesn't pay off on dense)")
        cmd = ["pollard-fit", "--gguf", a.gguf, "--ram", str(a.ram)]
        if a.imatrix: cmd += ["--imatrix", a.imatrix]
        if a.out: cmd += ["--out", a.out]
        if not a.run: cmd += ["--plan-only"]
        _run(cmd, a.run)
        return

    # MoE -> automap measured expert-allocation (crush cold experts, protect the hot set).
    # With an imatrix -> the extreme-low trellis mix; WITHOUT one -> the imatrix-FREE K-quant
    # mix (robust default: no 6-hour, coverage-hungry imatrix step). The user need not know.
    kfree = not a.imatrix
    if kfree:
        print("   path: MoE -> automap IMATRIX-FREE K-quant mix (no imatrix supplied -> the "
              "robust default: builds straight off the F16, no 6-hour imatrix, kill-proof)")
        print("        (pass --imatrix <covered diverse imatrix> for the extreme-low trellis mix.)")
    else:
        print("   path: MoE -> automap trellis mix (imatrix supplied; auto-pins uncovered experts)")
    binq = find_llama_bin("llama-quantize") if not a.bin else os.path.join(a.bin, "llama-quantize")
    here = os.path.dirname(os.path.abspath(a.gguf)) or "."
    tensors = os.path.join(here, "pollard_auto_tensors.txt")
    print("   1) full tensor list (Q6_K dry-run — can't abort mid-list):")
    print(f"   $ {binq} --dry-run {os.path.basename(a.gguf)} x.gguf Q6_K > pollard_auto_tensors.txt")
    if a.run:
        with open(tensors, "w") as f:
            subprocess.run([binq, "--dry-run", a.gguf, "x.gguf", "Q6_K"],
                           stdout=f, stderr=subprocess.STDOUT)
    mode = "gold-card BENCHMARK (3 bars + PPL)" if a.benchmark else "fast build (PollardMix only, no eval)"
    print(f"   2) automap emits the MoE build recipe -> {mode}:")
    am = ["pollard-automap", "--tensors", tensors, "--model", a.gguf,
          "--out", os.path.join(here, "pollard_auto_build.bat")]
    am += ["--no-imatrix"] if kfree else ["--imatrix", a.imatrix]
    if not a.benchmark:
        am += ["--mix-only", "--no-eval"]   # a plain build = ONE model, no benchmark
    if a.bin: am += ["--bin", a.bin]
    _run(am, a.run, cwd=here)
    if a.benchmark:
        bars = "uniform-Q2_K / PollardMix / uniform-Q3_K_M" if kfree else "uniform-IQ1 / PollardMix / uniform-IQ2"
        print(f"   3) run the emitted build script -> {bars} + PPL (the benchmark).")
    else:
        print("   3) run the emitted build script -> ONE PollardMix model, no eval (fast).")
        print("      (want the published gold-card numbers? re-run with --benchmark.)")
    print("      (pollard_auto_build.bat — run it on the box with a llama.cpp build.)")
    if not a.run:
        print("\n   plan only — re-run with --run to execute. (dense would run pollard-fit directly.)")


if __name__ == "__main__":
    main()
