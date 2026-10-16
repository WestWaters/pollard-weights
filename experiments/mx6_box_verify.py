#!/usr/bin/env python3
"""CUDA verification of the MX lane's MXFP6 / mix rungs on the box: KLD + top-1 vs the bf16 source (pollard-verify
--lane mx --gate) and the tool-call gate (pollard-toolcall, vLLM backend; records not-measured where vLLM cannot
serve the rung). Encodes any rung that is not built yet (CPU, model-free) first.

REFUSES TO START if nvidia-smi shows ANY non-shell process on the GPU (a game counts): the box's GPU is shared, and
this only runs when nobody else is on it. It re-checks before every GPU step and stops if someone arrived.

    python experiments/mx6_box_verify.py --model C:\\pollard\\models\\Qwen2.5-0.5B-Instruct \\
        --eval heldout.txt --out C:\\pollard\\mx6 [--rungs mxfp6-e2m3,mxfp6-e3m2,mix5.5,mix5,mix4.5]

Writes <out>/results-mx.json (the release results format) and prints the gate table per rung.
"""
from __future__ import annotations

import argparse, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
sys.path.insert(0, TOOLS)


def gpu_busy():
    """None when the GPU is free for us; else why not."""
    if not shutil.which("nvidia-smi"):
        return "no nvidia-smi: this is not the CUDA box"
    import pollard_toolcall as T
    users = T.gpu_users()
    if users:
        return "GPU in use by " + ", ".join(users) + " -- not starting (the box GPU is shared)"
    free = T.gpu_free_gb()
    if free is None or free < 4:
        return f"only {free} GB VRAM free"
    return None


def tool(name, *args, ok=(0,)):
    cmd = [sys.executable, os.path.join(TOOLS, name)] + [str(a) for a in args]
    print("$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode not in ok:
        raise SystemExit(f"{name} exited {r.returncode}")


def rung_args(r):
    r = r.strip().lower()
    if r.startswith("mix"):
        bpw = float(r[3:].lstrip("-_").removesuffix("bpw"))
        return f"MIX-{bpw:g}bpw", ["--mix", f"{bpw:g}"]
    if r in ("mxfp6-e2m3", "mxfp6-e3m2"):
        return r.upper(), ["--format", r]
    raise SystemExit(f"unknown rung {r!r} (mxfp6-e2m3 | mxfp6-e3m2 | mix<bpw>)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="the bf16 HF source dir")
    ap.add_argument("--eval", required=True, help="held-out text for the KLD")
    ap.add_argument("--out", required=True, help="where the rungs and results-mx.json go")
    ap.add_argument("--rungs", default="mxfp6-e2m3,mxfp6-e3m2,mix5.5,mix5,mix4.5")
    ap.add_argument("--sensitivity", help="pollard-probe sensitivity.json for the mixes")
    ap.add_argument("--gates", default="", help="gates.yaml (default: the built-in gates)")
    ap.add_argument("--no-toolcall", action="store_true")
    a = ap.parse_args()

    why = gpu_busy()
    if why:
        sys.exit(f"REFUSING: {why}")
    os.makedirs(a.out, exist_ok=True)
    res = os.path.join(a.out, "results-mx.json")
    base = os.path.basename(a.model.rstrip("/\\"))
    for r in a.rungs.split(","):
        tag, build = rung_args(r)
        out = os.path.join(a.out, f"{base}-Pollard-MX-{tag}")
        if not os.path.exists(os.path.join(out, "pollard_mx.json")):
            extra = ["--sensitivity", a.sensitivity] if a.sensitivity else []
            tool("pollard_mx.py", "--model", a.model, "--out", out, *build, *extra)     # CPU, no GPU needed
        why = gpu_busy()
        if why:
            sys.exit(f"STOPPING before {tag}: {why}")
        tool("pollard_verify.py", "--lane", "mx", "--model", out, "--source", a.model, "--eval", a.eval,
             "--device", "cuda:0", "--rung", os.path.basename(out), "--results", res, "--gate", a.gates, ok=(0, 1))
        if not a.no_toolcall:
            why = gpu_busy()
            if why:
                sys.exit(f"STOPPING before the {tag} tool gate: {why}")
            tool("pollard_toolcall.py", "--model", out, "--backend", "vllm", "--ref", a.model, "--results", res,
                 "--rung", os.path.basename(out), ok=(0, 1))
    print(f"\nresults: {res}")


if __name__ == "__main__":
    main()
