#!/usr/bin/env python3
"""pollard-bench — the drop-in BENCHMARK. Point it at a GGUF (or two) and get the gold-card
board: PPL + Mean/Median KLD + top-1, all from the same harness at matched size. This is the
symmetric half of `pollard` (which BUILDS): `pollard` makes the model, `pollard-bench` scores it.

    pollard-bench --gguf model.gguf --ref f16.gguf --eval held.txt          # one model, full board
    pollard-bench --gguf pollard.gguf --vs rival.gguf --ref f16.gguf         # HEAD-TO-HEAD (Pareto verdict)
    pollard-bench --gguf model.gguf --eval held.txt                         # PPL only (no --ref -> no KLD)
    pollard-bench --gguf model.gguf --coherence                             # COHERENCE GATE (loop check + sampling sweep)
    pollard-bench --gguf model.gguf --coherence --quick                     # fast one-prompt post-build sanity

--ref is the KL reference (f16 ideally; a Q8_0/Q6_K host if f16 won't load — KLD vs a near-lossless
ref is what the 14B/30B cards use). --vs runs the SAME eval on a competitor's file (AWQ/GPTQ/unsloth/
bartowski GGUF) so the comparison is honest: read it as Pareto — Pollard wins its size class (same
quality for fewer GB, or more quality at the same GB), not as a single number.

Reuses llama-perplexity; no rebuild. This is the opt-in benchmark — a plain `pollard` build never
runs it (that's the split that stopped a minutes-long shrink from taking hours).
"""
import argparse, os, re, subprocess, sys, zlib
from collections import Counter

from pollard_calc import find_llama_bin


# ---- coherence gate: separate a sampling-fixable loop from a below-the-floor build -----------
# The mix sets QUALITY-at-size; sampling is a RUNTIME knob it can't reach. So a build can loop
# two ways: (a) good build, bad sampling -> a sweep finds coherent settings (ship them); (b) the
# bit tier is below the model's coherence floor -> it loops under EVERY sampling -> bump a tier.
# This gate runs the model, detects loops, sweeps sampling, and returns which case it is.

GATE_PROMPTS = [
    "Paris is the capital of France. The largest planet in our solar system is",
    "Here is a short explanation of how photosynthesis works:",
    "# Python function to compute the nth Fibonacci number\ndef fib(n):",
]
# tried in order; first config where ALL prompts are loop-free wins. Escalating anti-repetition.
SAMPLING_CONFIGS = [
    ("temp0.7/rp1.15", ["--temp", "0.7", "--repeat-penalty", "1.15", "--repeat-last-n", "256",
                        "--top-k", "40", "--top-p", "0.9"]),
    ("temp0.6/rp1.18/freq0.6", ["--temp", "0.6", "--repeat-penalty", "1.18", "--repeat-last-n", "320",
                                "--top-k", "40", "--frequency-penalty", "0.6"]),
    ("temp0.5/rp1.20/pres0.5", ["--temp", "0.5", "--repeat-penalty", "1.2", "--repeat-last-n", "384",
                                "--top-k", "30", "--min-p", "0.1", "--presence-penalty", "0.5"]),
]


def detect_loop(text, min_chars=80):
    """Pure heuristic loop detector. Returns (is_loop, metric, reason). Two stdlib signals:
      - zlib compression ratio: degenerate repetition (word- OR char-level: 'as big as the ...',
        'SgSgSg') compresses to almost nothing; a coherent paragraph sits ~0.35-0.6.
      - distinct-word ratio: unique/total words; collapses toward 0 on a phrase loop.
    Short outputs (< min_chars) are UNDECIDED -> not a loop (can't judge). Thresholds picked
    against the real failures this project hit (mix loops) vs the Q8 coherent baselines."""
    t = (text or "").strip()
    b = t.encode("utf-8", "ignore")
    if len(b) < min_chars:
        return False, 0.0, "too short to judge"
    comp = len(zlib.compress(b, 6)) / len(b)
    if comp < 0.18:
        return True, round(comp, 3), f"compression ratio {comp:.2f} < 0.18 (degenerate repetition)"
    words = re.findall(r"\S+", t.lower())
    if len(words) >= 12:
        distinct = len(set(words)) / len(words)
        if distinct < 0.35:
            return True, round(distinct, 3), f"distinct-word ratio {distinct:.2f} < 0.35 (phrase loop)"
    return False, round(comp, 3), "coherent"


def _generate(cli_bin, model, prompt, sampling, ngl, n_predict=80):
    cmd = ([cli_bin, "-m", model, "-ngl", str(ngl), "-c", "2048", "-n", str(n_predict), "-p", prompt]
           + sampling)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = r.stdout or ""
    # llama-cli echoes the prompt then the continuation; keep only the continuation
    return out.split(prompt, 1)[-1] if prompt in out else out


def coherence_gate(cli_bin, model, ngl, quick=False):
    """Run the model over the gate prompts, sweeping sampling. A config PASSES only if EVERY
    prompt is loop-free. Returns a verdict dict: PASS (+the winning sampling) or BELOW_FLOOR.
    quick=True: one prompt, default sampling only (a fast post-build sanity, not the full gate)."""
    prompts = GATE_PROMPTS[:1] if quick else GATE_PROMPTS
    configs = SAMPLING_CONFIGS[:1] if quick else SAMPLING_CONFIGS
    last = []
    for cfg_name, sampling in configs:
        rows, looped = [], False
        for p in prompts:
            gen = _generate(cli_bin, model, p, sampling, ngl)
            is_loop, metric, reason = detect_loop(gen)
            rows.append({"prompt": p.splitlines()[0][:48], "loop": is_loop, "reason": reason,
                         "sample": gen.strip().replace("\n", " ")[:120]})
            looped = looped or is_loop
        last = rows
        if not looped:
            return {"verdict": "PASS", "config": cfg_name, "sampling": sampling, "rows": rows}
    return {"verdict": "BELOW_FLOOR", "config": None, "rows": last}


def print_gate(res):
    print("\n=== coherence gate ===")
    for r in res["rows"]:
        print(f"  [{'LOOP' if r['loop'] else 'ok  '}] {r['prompt']:<50} {r['reason']}")
        if r["loop"]:
            print(f"         -> {r['sample']}")
    if res["verdict"] == "PASS":
        s = " ".join(res["sampling"])
        print(f"\nVERDICT: PASS — coherent. Ship these sampling defaults on the card:\n  {s}")
    else:
        print("\nVERDICT: BELOW FLOOR — loops under EVERY sampling config. This is NOT a sampling\n"
              "  problem; the bit tier is below the model's coherence floor. Bump the crush one\n"
              "  tier and rebuild (e.g. --body iq1_kt -> iq2_kt), then re-gate. (Small/sparse\n"
              "  models hit this; big models clear 1-bit fine — it's a size property.)")
    return res["verdict"] == "PASS"


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
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
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
    ap.add_argument("--llama-cli", default="llama-cli", help="generation binary for the coherence gate")
    ap.add_argument("--coherence", action="store_true",
                    help="run the COHERENCE GATE: generate over fixed prompts, detect loops, sweep "
                         "sampling, and report PASS (+the sampling to ship) or BELOW-FLOOR (bump a tier). "
                         "Runs alone (no --ref/--eval needed); add --ref for the full board too.")
    ap.add_argument("--quick", action="store_true",
                    help="with --coherence: fast one-prompt / default-sampling sanity instead of the full sweep.")
    a = ap.parse_args()

    # --- coherence gate (can run standalone: no perplexity bin / eval corpus required) ---
    if a.coherence or a.quick:
        if not os.path.exists(a.gguf):
            sys.exit(f"file not found: {a.gguf}")
        cli_bin = find_llama_bin(a.llama_cli)
        if not cli_bin:
            sys.exit("llama-cli not found — build llama.cpp/ik_llama.cpp or pass --llama-cli.")
        res = coherence_gate(cli_bin, a.gguf, a.ngl, quick=a.quick)
        passed = print_gate(res)
        if not a.ref:                      # gate-only invocation -> done (exit code reflects verdict)
            sys.exit(0 if passed else 2)

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
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
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
