#!/usr/bin/env python3
"""pollard — the autoaware entry point. Point it at ANY model; it detects dense vs MoE
and drives the WINNING method, so a user never runs a losing path or wastes hours.

WINNING PATH — the SAME for dense AND MoE (no losing fallback):
  (1) the imatrix K-quant ladder (pollard-fit) — the honest fit-your-RAM baseline, plus
  (2) the mixed-precision FLAGSHIP mix (automap trellis) — the hand-coded winner (crush body,
      protect attn/down/first-last; MoE = expert-allocation) — WHEN an imatrix is present.
  No imatrix -> just the ladder (stock K-quants). There is NO "imatrix-free mix": it loses
  to stock Q2_K, so it's deprecated, not a path here.

⛔ LOSING / TRAP paths — never the default, opt-in only:
  - sensitivity SWEEP on DENSE = loses (no expert redundancy) -> pollard-sensitivity refuses dense.
  - sensitivity SWEEP on a BIG MoE = ~2*layers full-model quantizes = many HOURS; opt-in R&D only.
  - the 3-bar comparison + KL/PPL = the BENCHMARK, opt-in via --benchmark (see benchmarks/).

Plans by default (prints the exact commands for THIS model); `--run` executes them.

    pollard --gguf model-f16.gguf --ram 16 --imatrix model.imatrix           # plan
    pollard --gguf moe-f16.gguf   --imatrix moe.imatrix --run                # detect + build (winning path)
    pollard --gguf model-f16.gguf --imatrix m.imatrix --benchmark --run      # + the gold-card board (slow)
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


def _automap_mix(a, is_moe):
    """Emit + (with --run) build the automap mix — the hand-coded mixed-precision flagship.
    MoE: expert-allocation (crush cold experts, protect router/down/shared/attn). DENSE: the
    IQ1_KT crush-body / protect-attn+down+edges flagship (automap-on-dense, --allow-dense).
    Fast by default (--mix-only --no-eval); --benchmark emits the 3-bar + PPL board instead.
    This is what keeps the winning hand-coded mix a first-class BUILD, not benchmark-only."""
    binq = find_llama_bin("llama-quantize") if not a.bin else os.path.join(a.bin, "llama-quantize")
    here = os.path.dirname(os.path.abspath(a.gguf)) or "."
    tensors = os.path.join(here, "pollard_auto_tensors.txt")
    print(f"   tensor list (Q6_K dry-run): {binq} --dry-run {os.path.basename(a.gguf)} x.gguf Q6_K > tensors")
    if a.run:
        with open(tensors, "w") as f:
            subprocess.run([binq, "--dry-run", a.gguf, "x.gguf", "Q6_K"],
                           stdout=f, stderr=subprocess.STDOUT)
    out = os.path.join(here, "pollard_auto_build.bat" if is_moe else "pollard_auto_flagship.bat")
    # the flagship is the TRELLIS mix (winner) — always imatrix-guided. No K-quant fallback.
    am = ["pollard-automap", "--tensors", tensors, "--model", a.gguf, "--out", out,
          "--imatrix", a.imatrix]
    if not is_moe:
        am += ["--allow-dense"]                             # dense flagship = the hand-coded mix
    if not a.benchmark:
        am += ["--mix-only", "--no-eval"]                   # plain build = ONE model, no benchmark
    if a.bin:
        am += ["--bin", a.bin]
    _run(am, a.run, cwd=here)
    mode = "3-bar BENCHMARK + PPL" if a.benchmark else "ONE mix model, no eval (fast)"
    print(f"   -> run {os.path.basename(out)} on the box -> {mode}"
          + ("" if a.benchmark else "  (add --benchmark for the gold-card numbers)"))
    return out


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

    # WINNING PATH — SAME shape for dense AND MoE (no losing fallback):
    #   (1) the imatrix K-quant ladder (pollard-fit) — the honest, fit-your-RAM baseline, and
    #   (2) the mixed-precision FLAGSHIP mix (automap trellis) — the hand-coded winner —
    #       WHEN an imatrix is given. No imatrix -> just the ladder (stock K-quants), which
    #       is the correct imatrix-free build; there is NO "imatrix-free mix" (it loses to Q2_K).
    flagship = "PollardMix expert-allocation" if is_moe else "IQ1_KT"
    print(f"   path: {tag} -> K-quant ladder (pollard-fit) + the {flagship} mixed-precision "
          f"flagship (the hand-coded winner) when an imatrix is present")
    cmd = ["pollard-fit", "--gguf", a.gguf, "--ram", str(a.ram)]
    if a.imatrix: cmd += ["--imatrix", a.imatrix]
    if a.out: cmd += ["--out", a.out]
    if not a.run: cmd += ["--plan-only"]
    print("   1) the K-quant ladder (fits your RAM budget):")
    _run(cmd, a.run)
    if a.imatrix:
        print(f"   2) the {flagship} mixed-precision flagship (automap trellis mix):")
        _automap_mix(a, is_moe=is_moe)
    else:
        print(f"   2) (supply --imatrix for the {flagship} mixed-precision flagship — the winning "
              f"build; without one you get the stock K-quant ladder above, no losing mix)")
    if not a.run:
        print("\n   plan only — re-run with --run to execute.")


if __name__ == "__main__":
    main()
