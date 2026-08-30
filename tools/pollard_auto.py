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

    # MoE -> automap measured expert-allocation (crush cold experts, protect the hot set,
    # auto-pin imatrix-uncovered experts). Needs an imatrix and a llama.cpp bin dir.
    print("   path: MoE -> automap measured expert-allocation (auto-pins uncovered experts)")
    if not a.imatrix:
        sys.exit("   MoE build needs --imatrix (build one from a DIVERSE calib: llama-imatrix "
                 "-m f16.gguf -f calib_diverse.txt -o m.imatrix --chunks 200).")
    binq = find_llama_bin("llama-quantize") if not a.bin else os.path.join(a.bin, "llama-quantize")
    here = os.path.dirname(os.path.abspath(a.gguf)) or "."
    tensors = os.path.join(here, "pollard_auto_tensors.txt")
    print("   1) full tensor list (Q6_K dry-run — can't abort mid-list):")
    print(f"   $ {binq} --dry-run {os.path.basename(a.gguf)} x.gguf Q6_K > pollard_auto_tensors.txt")
    if a.run:
        with open(tensors, "w") as f:
            subprocess.run([binq, "--dry-run", a.gguf, "x.gguf", "Q6_K"],
                           stdout=f, stderr=subprocess.STDOUT)
    print("   2) automap emits the MoE build recipe (+ pins uncovered experts):")
    am = ["pollard-automap", "--tensors", tensors, "--model", a.gguf, "--imatrix", a.imatrix,
          "--out", os.path.join(here, "pollard_auto_build.bat")]
    if a.bin: am += ["--bin", a.bin]
    _run(am, a.run, cwd=here)
    print("   3) run the emitted build script -> uniform-IQ1 / PollardMix / uniform-IQ2 + PPL.")
    print("      (pollard_auto_build.bat — run it on the box with a llama.cpp build.)")
    if not a.run:
        print("\n   plan only — re-run with --run to execute. (dense would run pollard-fit directly.)")


if __name__ == "__main__":
    main()
