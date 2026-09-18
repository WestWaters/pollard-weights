#!/usr/bin/env python3
"""pollard-taskeval -- what a quantization costs on real TASKS, not just on the logits.

Every other Pollard measurement is intrinsic: KL to f16, top-1 agreement, perplexity, trajectory
divergence. Those are the right things to ALLOCATE against -- they are cheap, dense, and sensitive.
They are not what anyone else quotes. Competing releases report task scores and a retention figure
("98.2% of the full-precision baseline"), and a reader holding a KL number next to that has no basis
for comparison.

This closes that gap: run the same public benchmarks against a quantized build and its f16 source,
and report per-task scores plus RETENTION -- quantized / reference, in percent.

  pollard-taskeval --model ./M-Q6_K.gguf --ref ./M-f16.gguf --suite quick
  pollard-taskeval --model <hf-dir> --tasks gsm8k,ifeval --limit 200

Two things worth knowing before reading any number it prints:

  * INTRINSIC AND TASK METRICS CAN DISAGREE. A build with slightly worse KL can score better on a
    task, and the reverse. The allocator optimises KL because that is what it can measure per tensor.
    If a rung's retention does not track its KL, that is a finding about the allocation, not noise --
    and it is invisible without running both.
  * A retention figure is only meaningful against a reference measured THE SAME WAY, on the same
    tasks, the same shots, the same limit. Comparing our retention to someone else's published
    retention is not valid unless the suites match; comparing OUR rungs to OUR f16 always is.

Suites, chosen for signal per GPU-hour:

  quick   gsm8k, ifeval                              -- minutes; a smoke test that a rung is sane
  core    + mmlu_redux_generative, gpqa_diamond_zeroshot, humaneval_plus, mbpp_plus, minerva_math500
  vision  charxiv, realworldqa, ok_vqa, ocrbench -- runs through lmms-eval, a separate harness,
          because lm-eval has none of these. `pip install lmms-eval`.
  agentic tau2-bench and BFCL. NOT wired: both need a live multi-turn tool environment and a user
          simulator, which is a service to stand up rather than a task name to pass. The suite is
          declared here with the exact repos so the gap is explicit and someone can close it, and
          asking for it prints those instructions rather than a confusing failure.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

SUITES = {
    # minutes, for checking a rung is sane before spending hours on it
    "quick": ["gsm8k", "ifeval"],
    # the runnable part of the suite competing releases quote, so a retention figure here is
    # comparable to theirs rather than merely similar-looking
    "core":  ["gsm8k", "minerva_math500", "ifeval", "mmlu_redux_generative",
              "gpqa_diamond_zeroshot", "humaneval_plus", "mbpp_plus"],
    # lmms-eval, not lm-eval -- different harness, different runner, same reporting here
    "vision": ["charxiv", "realworldqa", "ok_vqa", "ocrbench"],
}
VISION_SUITE = "vision"

# Not runnable from a task name. Both need a live environment: tau2-bench stands up a dual-control
# customer-service simulator, BFCL a function-calling executor. Declared so the gap is visible.
AGENTIC = {
    "tau2_bench": "https://github.com/sierra-research/tau2-bench",
    "bfcl":       "https://github.com/ShishirPatil/gorilla (berkeley-function-call-leaderboard)",
}

# The category each task reports under, so the summary lines up with how these are usually quoted.
CATEGORY = {
    "gsm8k": "Math", "minerva_math500": "Math",
    "ifeval": "Instruction Following",
    "mmlu_redux_generative": "Knowledge & Reasoning", "gpqa_diamond_zeroshot": "Knowledge & Reasoning",
    "humaneval_plus": "Coding", "mbpp_plus": "Coding",
    "charxiv": "Vision", "realworldqa": "Vision", "ok_vqa": "Vision",
    "ocrbench": "Vision",
}


def model_args(path: str) -> tuple[str, str]:
    """(backend, model_args) for lm-eval, for a GGUF file or an HF directory/id."""
    if path.endswith(".gguf"):
        if not os.path.isfile(path):
            raise SystemExit(f"no such GGUF: {path}")
        d, f = os.path.split(os.path.abspath(path))
        # lm-eval loads a GGUF through the HF backend; the tokenizer comes from the source repo,
        # because a GGUF's own tokenizer is not one transformers can construct.
        return "hf", f"pretrained={d},gguf_file={f}"
    return "hf", f"pretrained={path}"


def run(path: str, tasks: list[str], limit: int, device: str, batch: str, out_dir: str,
        harness: str = "lm_eval") -> dict:
    """Run one harness over one model. lmms-eval takes the same shape of arguments as lm-eval,
    which is why a single runner covers both: the difference is the module and the model wrapper."""
    backend, margs = model_args(path)
    if harness == "lmms_eval":
        # lmms-eval drives a vision-language model, so the wrapper differs from lm-eval's plain hf
        backend = "hf-multimodal" if not path.endswith(".gguf") else "hf-multimodal"
    cmd = [sys.executable, "-m", harness, "--model", backend, "--model_args", margs,
           "--tasks", ",".join(tasks), "--device", device, "--batch_size", batch,
           "--output_path", out_dir]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"   $ {' '.join(cmd[2:])}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        # The last N lines of a harness that logs progress are all INFO, so the real cause scrolls
        # past. Pull the lines that look like a failure first, and only fall back to the tail.
        blob = ((r.stderr or "") + "\n" + (r.stdout or "")).splitlines()
        keys = ("Error", "error", "Exception", "Traceback", "raise ", "not supported",
                "No module", "out of memory", "Killed")
        hits = [l for l in blob if any(k in l for k in keys)][-10:]
        detail = "\n".join(hits or blob[-10:]) or "(no output at all -- likely killed by the OS)"
        raise SystemExit(f"{harness} failed on {os.path.basename(path)} (exit {r.returncode}):\n"
                         f"{detail}")
    return collect(out_dir)


def collect(out_dir: str) -> dict:
    """Pull the headline metric per task out of whatever lm-eval wrote."""
    best, newest = {}, None
    for root, _d, files in os.walk(out_dir):
        for fn in files:
            if fn.endswith(".json"):
                p = os.path.join(root, fn)
                if newest is None or os.path.getmtime(p) > os.path.getmtime(newest):
                    newest = p
    if not newest:
        raise SystemExit(f"lm-eval wrote no results under {out_dir}")
    res = json.load(open(newest, encoding="utf-8")).get("results", {})
    for task, metrics in res.items():
        for key in ("exact_match,strict-match", "exact_match,none", "acc_norm,none", "acc,none",
                    "pass@1,none", "prompt_level_strict_acc,none"):
            if key in metrics and isinstance(metrics[key], (int, float)):
                best[task] = float(metrics[key])
                break
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--model", required=True, help="the build to score: a .gguf or an HF dir/id")
    ap.add_argument("--ref", help="f16 source, to report RETENTION against (strongly recommended: a "
                                  "task score alone says nothing about what quantization cost)")
    ap.add_argument("--suite", default="quick", choices=sorted(SUITES) + ["agentic"],
                    help="which task set. 'vision' runs through lmms-eval; 'agentic' is declared "
                         "but not wired -- it needs a live tool environment, and asking for it "
                         "prints what to stand up rather than failing obscurely.")
    ap.add_argument("--tasks", help="explicit comma-separated lm-eval task names, overrides --suite")
    ap.add_argument("--limit", type=int, default=0, help="samples per task (0 = the whole set). Use "
                                                         "a limit to compare rungs cheaply, but "
                                                         "compare only against a ref run at the SAME limit")
    ap.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    ap.add_argument("--batch-size", default="1",
                    help="lm-eval batch size. Default 1 because 'auto' probes for a batch that "
                         "fits and dies on CPU without a useful message; raise it on a GPU.")
    ap.add_argument("--out", default="taskeval", help="directory for lm-eval's raw output")
    a = ap.parse_args()

    if a.suite == "agentic" and not a.tasks:
        print("\n  The agentic category is not runnable from this tool, and pretending otherwise "
              "would be worse\n  than saying so. Both benchmarks need a live environment rather "
              "than a dataset:\n")
        for name, url in AGENTIC.items():
            print(f"    {name:12s} {url}")
        print("\n  tau2-bench stands up a dual-control customer-service simulator; BFCL needs a "
              "function-\n  calling executor. Once either is running, point this tool at its own "
              "harness.\n")
        raise SystemExit(2)

    harness = "lmms_eval" if a.suite == VISION_SUITE else "lm_eval"
    try:
        __import__(harness)
    except ImportError:
        pkg = "lmms-eval" if harness == "lmms_eval" else "lm-eval"
        raise SystemExit(f"the {a.suite} suite needs {pkg}: pip install {pkg}") from None

    tasks = [t.strip() for t in a.tasks.split(",")] if a.tasks else SUITES[a.suite]
    print(f"\n== pollard-taskeval :: {len(tasks)} task(s), limit={a.limit or 'full'}\n")

    got = run(a.model, tasks, a.limit, a.device, a.batch_size,
              os.path.join(a.out, "model"), harness)
    ref = run(a.ref, tasks, a.limit, a.device, a.batch_size,
              os.path.join(a.out, "ref"), harness) if a.ref else {}

    print(f"\n  {'task':30s} {'score':>8s}" + (f" {'ref':>8s} {'retention':>10s}" if ref else ""))
    cats: dict[str, list[float]] = {}
    for t in tasks:
        s = got.get(t)
        if s is None:
            print(f"  {t:30s} {'--':>8s}   (no metric reported)")
            continue
        line = f"  {t:30s} {100*s:7.2f}%"
        if ref and ref.get(t):
            keep = 100 * s / ref[t]
            line += f" {100*ref[t]:7.2f}% {keep:9.1f}%"
            cats.setdefault(CATEGORY.get(t, "Other"), []).append(keep)
        print(line)

    if cats:
        print(f"\n  {'category':30s} {'retention':>10s}")
        for c, vals in sorted(cats.items()):
            print(f"  {c:30s} {sum(vals)/len(vals):9.1f}%")
        allv = [v for vals in cats.values() for v in vals]
        print(f"  {'OVERALL':30s} {sum(allv)/len(allv):9.1f}%")
        print("\n  Retention is against YOUR f16 on these exact tasks and settings. It is not "
              "comparable\n  to a retention figure published against a different suite.")


if __name__ == "__main__":
    main()
