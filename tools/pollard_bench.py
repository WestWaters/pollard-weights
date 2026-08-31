#!/usr/bin/env python3
"""pollard-bench — the drop-in BENCHMARK. Point it at a GGUF (or two) and get the gold-card
board: PPL + Mean/Median KLD + top-1, all from the same harness at matched size. This is the
symmetric half of `pollard` (which BUILDS): `pollard` makes the model, `pollard-bench` scores it.

    pollard-bench --gguf model.gguf --ref f16.gguf --eval held.txt          # one model, full board
    pollard-bench --gguf pollard.gguf --vs rival.gguf --ref f16.gguf         # HEAD-TO-HEAD (Pareto verdict)
    pollard-bench --gguf model.gguf --eval held.txt                         # PPL only (no --ref -> no KLD)

--ref is the KL reference (f16 ideally; a Q8_0/Q6_K host if f16 won't load — KLD vs a near-lossless
ref is what the 14B/30B cards use). --vs runs the SAME eval on a competitor's file (AWQ/GPTQ/unsloth/
bartowski GGUF) so the comparison is honest: read it as Pareto — Pollard wins its size class (same
quality for fewer GB, or more quality at the same GB), not as a single number.

Reuses llama-perplexity; no rebuild. This is the opt-in benchmark — a plain `pollard` build never
runs it (that's the split that stopped a minutes-long shrink from taking hours).
"""
import argparse, os, re, subprocess, sys

from pollard_calc import find_llama_bin


def _size_gb(p):
    try:
        return os.path.getsize(p) / 1e9
    except OSError:
        return None


def _ppl_kl(ppl_bin, model, eval_f, base, ngl):
    """Run llama-perplexity and parse PPL (+ Mean/Median KLD + top-1 when a base is given)."""
    cmd = [ppl_bin, "-m", model, "-f", eval_f, "-c", "2048", "-ngl", str(ngl)]
    if base:
        cmd += ["--kl-divergence", "--kl-divergence-base", base]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    def g(pat):
        m = re.search(pat, out)
        return float(m.group(1)) if m else None
    return {
        "ppl":    g(r"Final estimate:\s*PPL[^=]*=\s*([0-9.]+)"),
        "mean_kld":   g(r"Mean\s+KLD:\s*([0-9.]+)"),
        "median_kld": g(r"Median\s+KLD:\s*([0-9.]+)"),
        "top1":   g(r"Same top[^:]*:\s*([0-9.]+)"),      # top-1 agreement %
    }


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, float) else "—"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", required=True, help="the model to score (a Pollard build, or any GGUF)")
    ap.add_argument("--vs", dest="rival", help="a competitor GGUF to score the same way (head-to-head)")
    ap.add_argument("--ref", help="KL reference GGUF (f16, or a near-lossless Q8_0/Q6_K host). "
                                  "Omit for PPL-only (no KLD/top-1).")
    ap.add_argument("--eval", default="wikitext2_test.txt", help="held-out eval text")
    ap.add_argument("--ngl", type=int, default=99, help="GPU layers (lower for a model bigger than the GPU)")
    ap.add_argument("--out", help="write a results.json (feeds pollard-scorecard)")
    ap.add_argument("--llama-perplexity", default="llama-perplexity")
    a = ap.parse_args()

    ppl_bin = find_llama_bin(a.llama_perplexity)
    if not ppl_bin:
        sys.exit("llama-perplexity not found — build llama.cpp/ik_llama.cpp or pass --llama-perplexity.")
    if not os.path.exists(a.eval) or os.path.getsize(a.eval) == 0:
        sys.exit(f"eval corpus not found or empty: {a.eval}")
    for f in [a.gguf, a.rival, a.ref]:
        if f and not os.path.exists(f):
            sys.exit(f"file not found: {f}")

    base = None
    if a.ref:
        base = os.path.splitext(a.gguf)[0] + ".klbase.dat"
        print(f"[1] KL base logits from {os.path.basename(a.ref)} (ngl {a.ngl}) ...")
        r = subprocess.run([ppl_bin, "-m", a.ref, "-f", a.eval, "-c", "2048",
                            "-ngl", str(a.ngl), "--kl-divergence-base", base],
                           capture_output=True, text=True)
        if not os.path.exists(base) or os.path.getsize(base) == 0:
            sys.exit("could not build KL base logits (ref too big for the GPU? lower --ngl, or use "
                     "a smaller near-lossless --ref like Q6_K).")
    else:
        print("[1] no --ref -> PPL only (pass --ref f16/Q8/Q6 for Mean/Median KLD + top-1).")

    targets = [("model", a.gguf)] + ([("rival", a.rival)] if a.rival else [])
    rows = []
    for i, (tag, m) in enumerate(targets, 2):
        print(f"[{i}] scoring {os.path.basename(m)} ...")
        r = _ppl_kl(ppl_bin, m, a.eval, base, a.ngl)
        r.update({"tag": tag, "name": os.path.basename(m), "gb": _size_gb(m)})
        rows.append(r)

    # board
    print("\n=== pollard-bench ===")
    print(f"{'model':<40} {'size(GB)':>9} {'PPL':>7} {'MeanKLD':>9} {'MedKLD':>8} {'top-1':>7}")
    for r in rows:
        print(f"{r['name']:<40} {_fmt(r['gb'],2):>9} {_fmt(r['ppl'],2):>7} "
              f"{_fmt(r['mean_kld']):>9} {_fmt(r['median_kld']):>8} {_fmt(r['top1'],1):>7}")

    # head-to-head Pareto verdict
    if a.rival and len(rows) == 2:
        m, v = rows
        if m["gb"] and v["gb"] and m["ppl"] and v["ppl"]:
            smaller = m["gb"] <= v["gb"]
            better = m["ppl"] <= v["ppl"]
            if smaller and better:
                verdict = "Pollard WINS outright (smaller AND lower PPL)."
            elif not smaller and not better:
                verdict = "competitor wins outright (smaller AND lower PPL)."
            else:
                verdict = ("Pareto trade — read the size/quality curve: "
                           + ("Pollard is smaller, competitor lower PPL." if smaller
                              else "Pollard is lower PPL, competitor smaller."))
            print(f"\nHead-to-head: {verdict}")

    if a.out:
        import json
        json.dump({"eval": a.eval, "ref": a.ref, "rows": rows}, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}  (feed pollard-scorecard for the card)")


if __name__ == "__main__":
    main()
