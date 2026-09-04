#!/usr/bin/env python3
"""pollard-probes — run TASK PROBES on a Pollard'd GGUF (reasoning AND code) via lm-eval-harness.

PPL + KL + top-1 (pollard-bench) say the distribution is close; they do NOT catch a capability that
the crush quietly broke. A coding model quantized with a prose-only calib regresses on CODE first —
and a prose PPL board never sees it. This runs lm-evaluation-harness against a llama.cpp server on the
GGUF: reasoning probes by default, and coding probes (HumanEval / MBPP pass@1) with --coding — the
missing "did the quant hurt reasoning/code?" cell. Pairs with pollard-calib (calibrate on code -> you
must measure on code).

  pollard-probes --gguf model.gguf --label pollard                       # reasoning set
  pollard-probes --gguf model.gguf --label pollard --coding --limit 100  # + HumanEval/MBPP
  pollard-probes --gguf pollard.gguf --label p ; pollard-probes --gguf f16.gguf --label ref  # compare

Cross-platform (no Windows `timeout` — that errors under a redirected/SSH stdin, the bug that bit the
box). Needs lm_eval + a built llama-server (auto-found, or --server-bin). The GGUF must load on this
box's GPU/CPU (it's the eval runtime).

BACKEND NOTE (verified 2026-09-04, e2e — the reason this is not yet auto-run): lm-eval's server
backends are version-fragile against current llama-server. `--model gguf` (this tool's default) hits
llama-server's native /completion, but recent llama-server returns `completion_probabilities` while
lm-eval's gguf.py still expects OpenAI-style `logprobs.token_logprobs` -> "Invalid logprobs data" on
every request (results invalid). `--model local-completions` (needs `pip install lm-eval[api]`) hits
/v1/completions but errored "Session is closed" against this build. Until a compatible lm-eval /
llama-server pairing is pinned, run task probes with a matched pair (e.g. llama-cpp-python's in-process
`--model gguf`, or a llama-server build known to match your lm-eval), or read the numbers off whichever
backend validates. The board's decision-table verdict does NOT depend on this cell (size/PPL/KLD/top-1/
chat already close it); this probe is EXTRA capability validation."""
import argparse, json, os, shutil, subprocess, sys, time, urllib.request

REASONING = ["arc_challenge", "arc_easy", "winogrande", "piqa", "hellaswag"]
CODING = ["humaneval", "mbpp"]      # pass@1 — the code-regression signal


def find_server(explicit):
    if explicit:
        return explicit
    for c in ("llama-server", "llama-server.exe"):
        p = shutil.which(c)
        if p:
            return p
    for root in ("llama.cpp/build/bin", "llama-stq/build/bin/Release",
                 "build/bin", "build/bin/Release"):
        for c in ("llama-server", "llama-server.exe"):
            cand = os.path.join(root, c)
            if os.path.exists(cand):
                return cand
    sys.exit("ERROR: llama-server not found — build it (cmake --build build --target llama-server) "
             "or pass --server-bin.")


def wait_health(port, timeout_s=180):
    url = f"http://localhost:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)                                   # stdin-safe sleep (not Windows `timeout`)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True, help="the model to probe (Pollard'd, f16 ref, or a rival)")
    ap.add_argument("--label", required=True, help="tag for the output json/log")
    ap.add_argument("--tasks", help="comma list (default: the reasoning set; --coding adds code tasks)")
    ap.add_argument("--coding", action="store_true", help="add HumanEval + MBPP pass@1 (code regression)")
    ap.add_argument("--limit", type=int, default=200, help="examples per task (0 = full)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--server-bin", help="path to llama-server (else auto-find)")
    ap.add_argument("--out-dir", default=".", help="where to write probes_<label>.json")
    a = ap.parse_args()

    tasks = [t.strip() for t in a.tasks.split(",")] if a.tasks else list(REASONING)
    if a.coding:
        tasks += CODING
    server = find_server(a.server_bin)
    out_json = os.path.join(a.out_dir, f"probes_{a.label}.json")

    print(f"== pollard-probes :: {a.gguf}  [{a.label}]  tasks={','.join(tasks)}  limit={a.limit or 'full'}")
    srv = subprocess.Popen([server, "-m", a.gguf, "-ngl", str(a.ngl), "-c", str(a.ctx),
                            "--port", str(a.port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_health(a.port):
            sys.exit("ERROR: llama-server did not become healthy in time.")
        print("   server healthy — running lm-eval ...")
        env = dict(os.environ, HF_ALLOW_CODE_EVAL="1")   # HumanEval/MBPP execute generated code
        cmd = [sys.executable, "-m", "lm_eval", "--model", "gguf",
               "--model_args", f"base_url=http://localhost:{a.port}",
               "--tasks", ",".join(tasks), "--output_path", out_json]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        if a.coding:
            cmd += ["--confirm_run_unsafe_code"]         # required for code-exec tasks
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            sys.exit(f"lm-eval exited {r.returncode}")
        print(f"   wrote {out_json}")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    main()
