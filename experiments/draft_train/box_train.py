#!/usr/bin/env python3
"""box_train.py -- Pollard's own quant-aware EAGLE-3 head for Qwen3-4B, trained on the Windows box (RTX 5070 Ti 16GB).

What it runs (pollard-draft train, one stage after another, each resumable from what is on disk):
  1. answers   8000 pollard-calib prompts (chat / tool-call / code / reasoning) answered by the Q4_K_M rung in
               llama-server on the CPU (-ngl 0, --threads 19: the 60/40 rule -- the GPU is not taken for this)
  2. capture   taps [2, 18, 33] + the final layer from the Q4_K_M rung's own weights (GGUF dequantized into the HF
               model, bf16, ~9GB of VRAM) -> data/hidden_states (~8000 x ~700 tokens x 4 x 2560 x 2 B ~ 115GB)
  3. train     speculators (Apache-2.0) eagle3, packed sequence 2048 (fits 16GB: ~13GB estimated), sdpa + eager
               loss (no Triton on Windows), 10 epochs; a watchdog stops it the moment anything else lands on the GPU
               (rerun the same task: it resumes from the last checkpoint)
  4. convert   GGUF ladder (Q8_0, Q4_K_M) + the HF folder, recorded in the MANIFEST as Pollard-trained, quant-aware
  5. measure   acceptance vs the Q4_K_M rung AND vs the BF16 GGUF (llama-server -ngl 0)

Why EAGLE-3 and not DFlash on this card: a 5-layer DFlash head on a 4B target estimates at ~18GB even at 2 layers and a
1024-token sequence -- `pollard-draft train --kind dflash` refuses on 16GB rather than OOM mid-run. EAGLE-3 converts to
GGUF (llama.cpp --spec-type draft-eagle3) and serves in vLLM / SGLang as-is.

It REFUSES to start (and pollard-draft refuses each GPU stage again) while any process other than the desktop shell
and the POLLARD_GPU_IGNORE apps is on nvidia-smi -- a game counts. Disk: refuses when C: cannot hold the hidden states.

ONE-TIME SETUP (over SSH as jwate):
    C:\\pollard\\venv312\\Scripts\\python.exe -m venv C:\\pollard\\venv-spec
    C:\\pollard\\venv-spec\\Scripts\\pip install "speculators>=0.8.1" torch --index-url https://download.pytorch.org/whl/cu128 ^
        --extra-index-url https://pypi.org/simple
  (speculators pins transformers < 5.17: its own venv, so Pollard's stays as it is)
  The rung + BF16 GGUF: C:\\pollard\\home\\models\\Qwen__Qwen3-4B\\gguf\\ (or pass --rung / --bf16).

RUN (only when the GPU is free -- the son is not gaming -- and C: has ~150GB free), detached so it survives SSH:
    schtasks /create /tn PollardDraftTrain /ru SYSTEM /sc once /sd 01/01/2026 /st 00:00 /f ^
      /tr "cmd /c C:\\pollard\\venv312\\Scripts\\python.exe C:\\pollard\\pw\\experiments\\draft_train\\box_train.py --run >> C:\\pollard\\draft-train.log 2>&1"
    schtasks /run /tn PollardDraftTrain
  (the start date is in the PAST on purpose: a future /st fires on its own -- the 10-09 incident; /run starts it)
  check:   type C:\\pollard\\draft-train.log
  done:    C:\\pollard\\home\\models\\Qwen__Qwen3-4B\\draft\\train-eagle3\\results-draft.json exists
           -> schtasks /delete /tn PollardDraftTrain /f
  `--print-schtasks` prints these lines; nothing here creates or runs a task.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools")
sys.path.insert(0, TOOLS)

TASK = "PollardDraftTrain"
DEFAULTS = {
    "home": r"C:\pollard\home",
    "target": "Qwen/Qwen3-4B",
    "trainer_python": r"C:\pollard\venv-spec\Scripts\python.exe",
    "llama_server": r"C:\pollard\llama.cpp\build\bin\Release\llama-server.exe",
    "threads": 19,                      # 60/40: 19 of 32 CPU threads
    "samples": 8000,
}


def schtasks_lines(py=r"C:\pollard\venv312\Scripts\python.exe", script=r"C:\pollard\pw\experiments\draft_train\box_train.py",
                   log=r"C:\pollard\draft-train.log"):
    return [f'schtasks /create /tn {TASK} /ru SYSTEM /sc once /sd 01/01/2026 /st 00:00 /f '
            f'/tr "cmd /c {py} {script} --run >> {log} 2>&1"',
            f"schtasks /run /tn {TASK}",
            f"schtasks /delete /tn {TASK} /f     (after results-draft.json exists)"]


def gguf_rung(home, target, tag):
    d = os.path.join(home, "models", target.replace("/", "__"), "gguf")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".gguf") and tag.lower() in f.lower():
                return os.path.join(d, f)
    return None


def argv_for(a):
    """The pollard-draft train command line this box runs."""
    import pollard_draft_train as DT  # noqa: F401  (import check: the tool is there)
    cmd = ["train", "--target", a.target, "--kind", "eagle3", "--rung", a.rung, "--rung-tag", "Q4_K_M",
           "--taps", "rung", "--samples", str(a.samples), "--device", "cuda", "--ngl", "0",
           "--threads", str(a.threads), "--trainer-python", a.trainer_python,
           "--measure-target", f"BF16={a.bf16}", "--rungs", "Q8_0,Q4_K_M"]
    if a.llama_server:
        cmd += ["--llama-server", a.llama_server]
    if a.max_steps:
        cmd += ["--max-steps", str(a.max_steps)]
    if a.run:
        cmd += ["--run"]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--home", default=os.environ.get("POLLARD_HOME") or DEFAULTS["home"])
    ap.add_argument("--target", default=DEFAULTS["target"])
    ap.add_argument("--rung", help="the Q4_K_M GGUF (default: found in <home>/models/<target>/gguf)")
    ap.add_argument("--bf16", help="the BF16/F16 GGUF to measure against (default: found beside the rung)")
    ap.add_argument("--trainer-python", default=DEFAULTS["trainer_python"])
    ap.add_argument("--llama-server", default=DEFAULTS["llama_server"])
    ap.add_argument("--threads", type=int, default=DEFAULTS["threads"])
    ap.add_argument("--samples", type=int, default=DEFAULTS["samples"])
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--run", action="store_true", help="train (otherwise: checks + the plan)")
    ap.add_argument("--print-schtasks", action="store_true")
    a = ap.parse_args(argv)
    if a.print_schtasks:
        print("\n".join(schtasks_lines()))
        return 0
    if not shutil.which("nvidia-smi"):
        raise SystemExit("no nvidia-smi: this is the CUDA box recipe")
    os.environ["POLLARD_HOME"] = a.home                        # SYSTEM has no user profile: say where things live
    os.environ.setdefault("HF_HOME", os.path.join(a.home, "hf"))
    import pollard_draft_train as DT
    DT.gpu_guard()                                             # a game on the GPU -> refuse, before anything starts
    a.rung = a.rung or gguf_rung(a.home, a.target, "Q4_K_M")
    a.bf16 = a.bf16 or gguf_rung(a.home, a.target, "BF16") or gguf_rung(a.home, a.target, "F16")
    if not a.rung or not os.path.exists(a.rung):
        raise SystemExit(f"no Q4_K_M rung under {a.home} -- pass --rung")
    if not a.bf16:
        raise SystemExit("no BF16/F16 GGUF to measure against -- pass --bf16")
    if a.llama_server and not os.path.exists(a.llama_server):
        a.llama_server = None
    import pollard_draft as D
    cmd = argv_for(a)
    print("== box_train :: pollard-draft " + " ".join(cmd), flush=True)
    return D.main(cmd)


if __name__ == "__main__":
    sys.exit(main())
