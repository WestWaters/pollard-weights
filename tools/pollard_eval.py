#!/usr/bin/env python3
"""pollard-eval — trajectory-divergence eval, comparable to the strong quantizers.

Single-token mean-KL says a quant's NEXT token is close to the reference's — but a
quant can look great on mean-KL and still drift over a paragraph. This measures the
stronger, publishable thing: **top-1 token agreement with the f16/bf16 reference
over a held-out set**, plus mean KL, per quant. That is the metric family the strong
quantizers report (e.g. Unsloth's "Divergence-300 @32" / top-1 agreement), so a
Pollard number measured this way is directly comparable to theirs — no more guessing
whether we win.

Two ways to build the held-out set:
  * default: score against a held-out TEXT corpus (--eval held.txt) — simple, robust.
  * --trajectory: first greedy-generate N tokens from the REFERENCE on each prompt
    in --prompts, then score the quants on those reference trajectories. This is the
    "does the quant follow BF16's own path" variant; closest to Divergence-300 @32.

Point it at OUR builds AND a competitor's GGUFs in one run to get an apples-to-apples
table (top-1 agreement + KL vs size) — and a CSV the chart reads.

    pollard-eval --ref model-f16.gguf --eval held.txt \\
        --quants pollard-q3.gguf unsloth-UD-Q3_K_XL.gguf --out results.csv

    pollard-eval --ref model-f16.gguf --prompts prompts.txt --trajectory --gen-tokens 32 \\
        --quants pollard-q3.gguf unsloth-UD-Q3_K_XL.gguf --out results.csv

GPU strongly recommended; --rpc pools nodes for a model too big for one box.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

from pollard_calc import _shard_paths, find_llama_bin

_KLD = re.compile(r"Mean\s+KLD\s*[:=]\s*([0-9.eE+-]+)")
# llama-perplexity --kl-divergence prints a top-1 agreement line; the wording has
# shifted across versions ("Same top p", "top-1 …", "maximum top token …"), so match
# the common shapes and take the first percentage on that line.
_TOP1 = re.compile(
    r"(?im)^(?:.*\bsame\s+top|.*\btop[-\s]?1|.*\btop[-\s]?token).*?([0-9.]+)\s*%")


def _size_gb(path):
    return sum(os.path.getsize(p) for p in _shard_files(path)) / 1e9


def _shard_files(path):
    return _shard_paths(path)


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def _score(ppl, quant, corpus, base, rpc):
    """top-1 agreement (%) and mean KL of `quant` vs the base logits, or (None, None)."""
    cmd = [ppl, "-m", quant, "-f", corpus, "--kl-divergence",
           "--kl-divergence-base", base, "-ngl", "99", "-c", "512"]
    if rpc:
        cmd += ["--rpc", rpc]
    out = _run(cmd)
    txt = out.stdout + out.stderr
    kld = _KLD.search(txt)
    top1 = _TOP1.search(txt)
    return (float(top1.group(1)) if top1 else None,
            float(kld.group(1)) if kld else None)


def _greedy(cli, ref, prompt, n, rpc):
    """N greedy tokens continued from `prompt` by the reference model (temp 0)."""
    cmd = [cli, "-m", ref, "-p", prompt, "-n", str(n), "--temp", "0",
           "-ngl", "99", "--no-display-prompt", "-no-cnv", "--simple-io"]
    if rpc:
        cmd += ["--rpc", rpc]
    out = _run(cmd)
    # strip the llama.cpp end-of-generation marker and any trailing banner noise
    return out.stdout.split("[end of text]")[0].strip()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--ref", required=True, help="reference GGUF (f16/bf16) — the ground truth")
    ap.add_argument("--quants", required=True, nargs="+",
                    help="quant GGUFs to score (ours AND a competitor's, in one run)")
    ap.add_argument("--eval", help="held-out text corpus to score on")
    ap.add_argument("--prompts", help="prompts file (one per line) for --trajectory mode")
    ap.add_argument("--trajectory", action="store_true",
                    help="generate reference continuations first, then score on them "
                         "(Divergence-300 @N style); needs --prompts")
    ap.add_argument("--gen-tokens", type=int, default=32, help="tokens to generate per prompt")
    ap.add_argument("--out", help="CSV out (label,size_gb,top1_agree,mean_kl) for the chart")
    ap.add_argument("--rpc", help="RPC servers to pool, 'host:port[,host:port…]'")
    ap.add_argument("--llama-cli", default="llama-cli")
    ap.add_argument("--llama-perplexity", default="llama-perplexity")
    a = ap.parse_args()

    # auto-detect the llama.cpp binaries (PATH, or the runtime build install.sh made)
    a.llama_perplexity = find_llama_bin(a.llama_perplexity)
    if a.trajectory:
        a.llama_cli = find_llama_bin(a.llama_cli)
    need = {"llama-perplexity": a.llama_perplexity}
    if a.trajectory:
        need["llama-cli"] = a.llama_cli
    missing = [n for n, v in need.items() if v is None]
    if missing:
        sys.exit(f"ERROR: {', '.join(missing)} not found. install.sh builds these into "
                 f"runtime/llama.cpp/build/bin — re-run install.sh, or pass the path "
                 f"(e.g. --llama-perplexity /path/to/llama-perplexity).")
    if a.trajectory and not a.prompts:
        sys.exit("ERROR: --trajectory needs --prompts (the seeds to continue from f16).")
    if not a.trajectory and not a.eval:
        sys.exit("ERROR: pass --eval <held-out.txt>, or --trajectory --prompts <prompts.txt>.")

    tmp = tempfile.mkdtemp(prefix="pollard_eval_")
    corpus = a.eval
    if a.trajectory:
        corpus = os.path.join(tmp, "trajectories.txt")
        prompts = [p for p in open(a.prompts).read().splitlines() if p.strip()]
        print(f"== building reference trajectories :: {len(prompts)} prompts x "
              f"{a.gen_tokens} tokens from {os.path.basename(a.ref)}")
        with open(corpus, "w") as f:
            for i, p in enumerate(prompts):
                cont = _greedy(a.llama_cli, a.ref, p, a.gen_tokens, a.rpc)
                f.write(p + " " + cont + "\n")
                print(f"  [{i+1}/{len(prompts)}] {len(cont)} chars")

    # reference logits (the ground truth the quants are compared against)
    base = os.path.join(tmp, "base.dat")
    print(f"== reference logits on the held-out set ({os.path.basename(a.ref)})")
    bcmd = [a.llama_perplexity, "-m", a.ref, "-f", corpus,
            "--kl-divergence-base", base, "-ngl", "99", "-c", "512"]
    if a.rpc:
        bcmd += ["--rpc", a.rpc]
    _run(bcmd)

    rows = []
    print(f"\n{'quant':<34} {'size':>8} {'top-1 agree':>12} {'mean KL':>10}")
    for q in a.quants:
        top1, kld = _score(a.llama_perplexity, q, corpus, base, a.rpc)
        gb = _size_gb(q)
        rows.append((os.path.basename(q), gb, top1, kld))
        t1 = f"{top1:.2f}%" if top1 is not None else "FAIL"
        kk = f"{kld:.5f}" if kld is not None else "FAIL"
        print(f"{os.path.basename(q):<34} {gb:7.2f}G {t1:>12} {kk:>10}")

    for f in (base,):
        os.path.exists(f) and os.remove(f)

    if a.out:
        with open(a.out, "w") as f:
            f.write("label,size_gb,top1_agree,mean_kl\n")
            for name, gb, top1, kld in rows:
                f.write(f"{name},{gb:.3f},{top1 if top1 is not None else ''},"
                        f"{kld if kld is not None else ''}\n")
        print(f"\nwrote {a.out} — feed it to experiments/plot_kl_win.py or plot_eval.py")
    if any(t is None for _, _, t, _ in rows):
        print("\nNOTE: some 'top-1 agree' came back blank — this llama-perplexity build "
              "may word the agreement line differently; mean KL still ranks them.")


if __name__ == "__main__":
    main()
