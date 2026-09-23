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
import contextlib
import json
import os
import subprocess
import sys
import time

#: lm-eval marks these UNSAFE_CODE: scoring them means EXECUTING code the model wrote. It
#: refuses to run them without an explicit opt-in, and it aborts the WHOLE invocation rather than
#: skipping them -- so one code task in a nine-task suite costs all nine. `bonsai` and `core` both
#: contain two, which is why --allow-code-exec exists rather than the flag being buried.
UNSAFE_CODE_TASKS = {"humaneval", "humaneval_plus", "mbpp", "mbpp_plus"}

#: lm-eval grades those two with a pass@k metric that forks and uses POSIX signals for its
#: execution timeout, so on Windows it raises `NotImplementedError: This metric is currently not
#: supported on Windows` -- after generation, which is the expensive half. Nothing we pass fixes
#: it; the tasks have to be scored on a POSIX box.
CODE_METRIC_POSIX_ONLY = sys.platform == "win32"


SUITES = {
    # minutes, for checking a rung is sane before spending hours on it
    "quick": ["gsm8k", "ifeval"],
    # the runnable part of the suite competing releases quote, so a retention figure here is
    # comparable to theirs rather than merely similar-looking
    "core":  ["gsm8k", "minerva_math500", "ifeval", "mmlu_redux_generative",
              "gpqa_diamond_zeroshot", "humaneval_plus", "mbpp_plus"],
    # Mirrors the task list a competing low-bit 27B release quotes, category for category, so our
    # per-category and overall figures answer theirs directly instead of being merely adjacent.
    # Three of their tasks have no lm-eval implementation (see UNAVAILABLE) -- the suite says so
    # rather than quietly averaging over a smaller set and calling it the same number.
    "bonsai": ["mmlu_redux_generative", "leaderboard_musr",          # Knowledge & reasoning
               "gsm8k", "minerva_math500", "aime25", "aime26",       # Math
               "humaneval_plus", "mbpp_plus",                        # Coding
               "ifeval"],                                            # Instruction following
    # lmms-eval, not lm-eval -- different harness, different runner, same reporting here
    "vision": ["charxiv", "realworldqa", "ok_vqa", "ocrbench"],
}
VISION_SUITE = "vision"

# Tasks Pollard ships itself, because the releases we are answering quote them and lm-eval has no
# implementation. Loaded via --include_path so they register like any built-in.
TASK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks")

# Still quoted by that release with nothing to run them. Named so a reader knows the overall covers
# 9 of their 11 rather than silently being a different quantity.
UNAVAILABLE = {"LiveCodeBench": "needs its own execution harness (sandboxed run + date filtering)",
               "IFBench": "needs its per-constraint verifiers"}

# Not runnable from a task name. Both need a live environment: tau2-bench stands up a dual-control
# customer-service simulator, BFCL a function-calling executor. Declared so the gap is visible.
AGENTIC = {
    "tau2_bench": "https://github.com/sierra-research/tau2-bench",
    "bfcl":       "https://github.com/ShishirPatil/gorilla (berkeley-function-call-leaderboard)",
}

# The category each task reports under, so the summary lines up with how these are usually quoted.
CATEGORY = {
    "gsm8k": "Math", "minerva_math500": "Math", "aime25": "Math",
    "ifeval": "Instruction Following",
    "mmlu_redux_generative": "Knowledge & Reasoning", "gpqa_diamond_zeroshot": "Knowledge & Reasoning",
    "leaderboard_musr": "Knowledge & Reasoning",
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


def chat_model(path: str) -> bool:
    """Does this model have a chat template? Then it must be scored through a chat turn.

    Pollard already reads this (pollard-modelkind says "instruct" from the same field); taskeval
    simply never asked.
    """
    try:
        from pollard_modelkind import _template_and_arch
        return bool(_template_and_arch(path)[0])
    except Exception:
        return False


#: the served model's own stdout+stderr. Written next to wherever taskeval was run.
_SERVER_LOG = "taskeval-server.log"


def _server_tail(n: int = 12) -> str:
    """The last few lines the server managed to say, for an error message that can be acted on."""
    try:
        with open(_SERVER_LOG, encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip() for l in f if l.strip()]
    except OSError:
        return f"  (no {_SERVER_LOG})"
    if not lines:
        return f"  ({_SERVER_LOG} is empty)"
    return f"  last {min(n, len(lines))} line(s) of {_SERVER_LOG}:\n" + \
           "\n".join("    " + l for l in lines[-n:])


@contextlib.contextmanager
def served(gguf: str, ngl: str, port: int, ctx: int, server: str | None = None):
    """Host a GGUF on llama-server for the duration, and yield its base_url.

    lm-eval's HF backend loads a GGUF by DEQUANTIZING it, so a 5.9GB build becomes ~55GB of fp32 in
    RAM -- it scores the small models fine and cannot open the ones Pollard exists for. Worse, it
    would not be measuring what ships: the point of the number is the quantized build on the
    quantized kernel. llama-server keeps it quantized and on the GPU, and lm-eval's `gguf` backend
    talks to it over HTTP."""
    import urllib.request
    from pollard_calc import find_llama_bin
    # An explicit --llama-server wins outright. It has to: find_llama_bin prefers
    # $POLLARD_HOME/bin over PATH, so on a box that keeps a stock llama.cpp there, a trellis
    # build (IQ*_KT is ggml type >= 43, which stock caps out below) can never be scored -- the
    # stock server refuses the file and no ordering of PATH changes which binary is chosen.
    binsrv = server or find_llama_bin("llama-server") or "llama-server"
    # --jinja: apply the model's OWN chat template. Without it an instruct model is scored as a
    # base model -- it never reaches its end-of-turn token, runs past the answer, and the harness
    # scrapes a stop string out of the overrun. Every generative task loses points to that, and
    # the release being answered quoted numbers measured WITH their template, so the comparison
    # was biased against us rather than merely noisy.
    # --reasoning-format deepseek: a thinking model's <think> block lands in reasoning_content
    # instead of being parsed as the answer.
    cmd = [binsrv, "-m", gguf, "--host", "127.0.0.1", "--port", str(port),
           "-ngl", str(ngl), "-c", str(ctx), "--jinja",
           "--reasoning-format", "deepseek"]
    print(f"   serving: {binsrv} ... -ngl {ngl}", flush=True)
    # NOT DEVNULL. A server that dies on load says exactly why ("invalid ggml type 153",
    # "failed to allocate", a missing DLL); discarding it leaves only "exited (1)" and turns a
    # one-line diagnosis into an afternoon.
    log = open(_SERVER_LOG, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(600):                                   # a big model takes a while to load
            if proc.poll() is not None:
                raise SystemExit(f"llama-server exited ({proc.returncode}) before serving "
                                 f"{gguf}\n{_server_tail()}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit(f"llama-server never became ready for {gguf}\n{_server_tail()}")
        yield base
    finally:
        log.close()
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


def run(path: str, tasks: list[str], limit: int, device: str, batch: str, out_dir: str,
        harness: str = "lm_eval", serve: bool = True, ngl: str = "99", port: int = 8080,
        ctx: int = 4096, server: str | None = None, allow_code: bool = False) -> dict:
    """Run one harness over one model. lmms-eval takes the same shape of arguments as lm-eval,
    which is why a single runner covers both: the difference is the module and the model wrapper."""
    backend, margs = model_args(path)
    if harness == "lmms_eval":
        # lmms-eval drives a vision-language model, so the wrapper differs from lm-eval's plain hf
        backend = "hf-multimodal" if not path.endswith(".gguf") else "hf-multimodal"

    # lm-eval evaluates a --tasks list in ONE process and aborts the lot if any single task will
    # not run -- an unsafe-code gate, a POSIX-only metric, a missing extra. A nine-task suite then
    # reports nothing because of one task. So the coding tasks are invoked SEPARATELY from the
    # rest: whatever happens to one group, the other still produces its numbers.
    code_tasks = [x for x in tasks if x in UNSAFE_CODE_TASKS]
    other_tasks = [x for x in tasks if x not in UNSAFE_CODE_TASKS]
    skipped: list[str] = []
    if code_tasks and not allow_code:
        skipped = code_tasks
        print(f"   (!) skipping {', '.join(code_tasks)}: they EXECUTE code the model wrote. "
              f"Pass --allow-code-exec to score them.", file=sys.stderr, flush=True)
    elif code_tasks and CODE_METRIC_POSIX_ONLY:
        skipped = code_tasks
        print(f"   (!) skipping {', '.join(code_tasks)}: lm-eval's pass@k grader is POSIX-only "
              f"and raises NotImplementedError on Windows, after generation. Score these on a "
              f"Linux or macOS box.", file=sys.stderr, flush=True)
    if skipped:
        code_tasks = []

    groups = [g for g in (other_tasks, code_tasks) if g]

    def _invoke(backend, margs, extra=(), subset=None):
        subset = subset or tasks
        is_code = bool(UNSAFE_CODE_TASKS.intersection(subset))
        cmd = [sys.executable, "-m", harness, "--model", backend, "--model_args", margs,
               "--tasks", ",".join(subset), "--batch_size", batch, "--output_path", out_dir]
        if harness == "lm_eval" and os.path.isdir(TASK_DIR):
            cmd += ["--include_path", TASK_DIR]      # Pollard's own tasks (e.g. aime26)
        if is_code and allow_code and harness == "lm_eval":
            cmd += ["--confirm_run_unsafe_code"]
        cmd += list(extra)
        if limit:
            cmd += ["--limit", str(limit)]
        print(f"   $ {' '.join(cmd[2:])}", flush=True)
        env = dict(os.environ)
        if is_code and allow_code:
            # the CLI flag gets the task past lm-eval's gate; the grader itself checks this
            env["HF_ALLOW_CODE_EVAL"] = "1"
        return subprocess.run(cmd, capture_output=True, text=True, env=env)

    # A GGUF is scored ON the quantized kernel, served, not dequantized into RAM through the HF
    # backend -- otherwise the number describes an fp32 copy of the build rather than the build.
    results = []                                 # (subset, CompletedProcess) per group
    if serve and harness == "lm_eval" and path.endswith(".gguf"):
        with served(path, ngl, port, ctx, server) as base:
            for subset in groups:
                if chat_model(path):
                    # lm-eval's `gguf` backend posts to /v1/completions -- the RAW endpoint, no
                    # template. local-chat-completions posts to /v1/chat/completions, where the
                    # server applies the model's own template and the model stops on its own
                    # end-of-turn token.
                    results.append((subset, _invoke(
                        "local-chat-completions",
                        f"base_url={base}/v1/chat/completions,"
                        f"model=pollard,num_concurrent=1,tokenized_requests=False",
                        ("--apply_chat_template",), subset)))
                else:
                    results.append((subset, _invoke("gguf", f"base_url={base}", (), subset)))
    else:
        for subset in groups:
            results.append((subset, _invoke(backend, margs, ("--device", device), subset)))

    for subset, r in results:
        if r.returncode == 0:
            continue
        # A group that failed is reported and survived -- unless it is the only group, in which
        # case there is nothing to report and the old hard exit is still the right answer.
        fatal = len(results) == 1
        # The last N lines of a harness that logs progress are all INFO, so the real cause scrolls
        # past. Pull the lines that look like a failure first, and only fall back to the tail.
        blob = ((r.stderr or "") + "\n" + (r.stdout or "")).splitlines()
        keys = ("Error", "error", "Exception", "Traceback", "raise ", "not supported",
                "No module", "out of memory", "Killed")
        hits = [l for l in blob if any(k in l for k in keys)][-10:]
        detail = "\n".join(hits or blob[-10:]) or "(no output at all -- likely killed by the OS)"
        msg = (f"{harness} failed on {os.path.basename(path)} for "
               f"{', '.join(subset)} (exit {r.returncode}):\n{detail}")
        if fatal:
            raise SystemExit(msg)
        print(f"   (!) {msg}\n       the other task group(s) still ran; their numbers stand.",
              file=sys.stderr, flush=True)
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
    ap.add_argument("--no-serve", dest="serve", action="store_false",
                    help="score a GGUF through the HF backend, which DEQUANTIZES it into RAM "
                         "(~9x its file size) and measures an fp32 copy rather than the build. "
                         "Only sensible for a small model on a machine with room.")
    ap.set_defaults(serve=True)
    ap.add_argument("--ngl", default="99", help="GPU layers for the served GGUF (default all)")
    ap.add_argument("--port", type=int, default=8080, help="llama-server port (--ref uses port+1)")
    ap.add_argument("--allow-code-exec", dest="allow_code", action="store_true",
                    help="run the coding tasks (humaneval*/mbpp*), which EXECUTE code the model "
                         "wrote, on this machine. Off by default. Without it lm-eval refuses the "
                         "entire invocation -- not just those tasks -- so a suite containing one "
                         "reports nothing at all.")
    ap.add_argument("--llama-server", dest="llama_server", default=None,
                    help="llama-server binary to host the GGUF with. Point this at an ik_llama "
                         "build to score a trellis (IQ*_KT) build: those types are outside stock "
                         "llama.cpp's range, and a stock server on $POLLARD_HOME/bin would "
                         "otherwise be picked and refuse the file.")
    ap.add_argument("--ctx", type=int, default=4096, help="server context size")
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
              os.path.join(a.out, "model"), harness, a.serve, a.ngl, a.port, a.ctx,
              a.llama_server, a.allow_code)
    # a second server would collide on the port, so the reference gets its own
    ref = run(a.ref, tasks, a.limit, a.device, a.batch_size,
              os.path.join(a.out, "ref"), harness, a.serve, a.ngl, a.port + 1, a.ctx,
              a.llama_server, a.allow_code) if a.ref else {}

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
