#!/usr/bin/env python3
"""pollard-probes — task-accuracy probes on a Pollard'd GGUF, via llama.cpp's OWN MCQ modes.

PPL + KL + top-1 (pollard-bench) say the distribution is close; they don't catch a capability the
crush quietly broke. This measures real multiple-choice accuracy (HellaSwag / Winogrande / MMLU-style)
so you SEE reasoning regression, and it pairs with pollard-calib (calibrate on code -> measure on code).

Why NOT lm-eval-harness here (verified 2026-09-04): lm-eval's server backends can't do this for a
Pollard GGUF. Its `gguf` backend needs OpenAI-legacy `logprobs.token_logprobs` with working `echo`, but
llama-server returns `logprobs.content` and does NOT honor echo (no prompt-token logprobs). And
llama-cpp-python (which would) bundles STOCK llama.cpp — it can't even LOAD a non-standard type like
STQ1_0. So we use the ONE tool that both reads every Pollard type AND scores MCQ internally: our patched
`llama-perplexity` (`--hellaswag` / `--winogrande` / `--multiple-choice`) — no server, no echo, no
logprobs plumbing. Verified: STQ1_0 1.5B HellaSwag 38.5% vs its f16 53.0%.

  pollard-probes --gguf model.gguf --label pollard                       # HellaSwag (auto-prep 400 tasks)
  pollard-probes --gguf model.gguf --label pollard --tasks 1000
  pollard-probes --gguf a.gguf --label q ; pollard-probes --gguf f16.gguf --label ref   # compare
  pollard-probes --gguf model.gguf --winogrande wino.csv                 # BYO Winogrande/MMLU datafile

Needs our patched llama-perplexity (reads STQ1_0/IQ*_KT). HellaSwag auto-prep needs `datasets`."""
import argparse, os, re, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pollard_calc import find_llama_bin


def prep_hellaswag(out, n):
    """Write llama.cpp's 6-lines-per-task HellaSwag format (context, gold-idx, 4 endings)."""
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    w = 0
    with open(out, "w", encoding="utf-8") as f:
        for ex in ds:
            if w >= n:
                break
            ctx = (ex.get("ctx") or "").replace("\n", " ").strip()
            label, endings = ex.get("label", ""), ex.get("endings") or []
            if ctx == "" or label == "" or len(endings) != 4:
                continue
            f.write(ctx + "\n" + str(label) + "\n")
            for e in endings:
                f.write(e.replace("\n", " ").strip() + "\n")
            w += 1
    return w


def parse_score(log_path):
    """Last '<n>\\t<score>%\\t[ci]' line llama-perplexity prints = the running/final accuracy."""
    last = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"\s*(\d+)\s+(\d+\.\d+)\s*%", line)
            if m:
                last = (int(m.group(1)), float(m.group(2)))
    return last


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True, help="the model to probe (Pollard'd, f16 ref, or a rival)")
    ap.add_argument("--label", default="model", help="tag for the output log")
    ap.add_argument("--tasks", type=int, default=400, help="number of MCQ tasks")
    ap.add_argument("--hellaswag-data", help="pre-made HellaSwag datafile (else auto-prep via datasets)")
    ap.add_argument("--winogrande", help="Winogrande datafile -> runs --winogrande instead of HellaSwag")
    ap.add_argument("--multiple-choice", dest="mc", help="MMLU/ARC-style datafile -> --multiple-choice")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--perplexity-bin", help="path to llama-perplexity (else auto-find)")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    binp = a.perplexity_bin or find_llama_bin("llama-perplexity")
    if not binp:
        sys.exit("ERROR: llama-perplexity not found (needs the PATCHED build for STQ1_0/IQ*_KT) — "
                 "pass --perplexity-bin.")

    if a.winogrande:
        mode, data, flag, tflag = "winogrande", a.winogrande, "--winogrande", "--winogrande-tasks"
    elif a.mc:
        mode, data, flag, tflag = "multiple-choice", a.mc, "--multiple-choice", "--multiple-choice-tasks"
    else:
        mode, flag, tflag = "hellaswag", "--hellaswag", "--hellaswag-tasks"
        data = a.hellaswag_data or os.path.join(a.out_dir, "hellaswag_val.txt")
        if not a.hellaswag_data:
            print(f"== pollard-probes :: auto-prep HellaSwag ({a.tasks} tasks) -> {data}")
            got = prep_hellaswag(data, a.tasks)
            print(f"   wrote {got} tasks")

    log = os.path.join(a.out_dir, f"probes_{a.label}_{mode}.log")
    cmd = [binp, "-m", a.gguf, "-f", data, flag, tflag, str(a.tasks),
           "-ngl", str(a.ngl), "-c", str(a.ctx)]
    print(f"== pollard-probes :: {a.gguf}  [{a.label}]  {mode}  {a.tasks} tasks")
    print("   $ " + " ".join(cmd))
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    sc = parse_score(log)
    if r.returncode != 0 or not sc:
        sys.exit(f"   run failed or no score parsed — see {log}")
    print(f"   {mode} accuracy: {sc[1]:.2f}%  ({sc[0]} tasks)   [{a.label}]")
    print(f"   log: {log}")


if __name__ == "__main__":
    main()
