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


# A distinct sentinel for "the model produced nothing". It must never be confused with a short but
# valid generation: empty text reads as "too short to judge", which the gate treats as not-a-loop,
# which would PASS a build that never spoke. A silent false pass is worse than a visible hang.
class _NoOutput(str):
    __slots__ = ()


_NO_OUTPUT = _NoOutput()
_NO_CNV = {}          # cli_bin -> does this build accept -no-cnv? (probed once)


def _one_shot_flag(cli_bin):
    """The flag that makes THIS build generate once and exit, or None if it needs none.

    Builds disagree and the name changed upstream:
      * older llama.cpp   -no-cnv           (conversation on by default)
      * current llama.cpp -st/--single-turn ("run conversation for a single turn only, then exit")
      * ik_llama.cpp      none needed       (conversation is opt-in via -cnv)

    Closing stdin is NOT always sufficient: a current build was observed loaded, 15 GB resident, at
    0% GPU, waiting at its prompt with stdin already at EOF. So probe --help and pass what the binary
    says it takes. Passing an unknown flag is a hard error, which is why this cannot be unconditional."""
    if cli_bin not in _NO_CNV:
        flag = None
        try:
            h = subprocess.run([cli_bin, "--help"], capture_output=True, text=True,
                               errors="replace", timeout=60, stdin=subprocess.DEVNULL)
            help_text = (h.stdout or "") + (h.stderr or "")
            for cand in ("-no-cnv", "--single-turn"):
                if cand in help_text:
                    flag = cand
                    break
        except Exception:
            pass
        _NO_CNV[cli_bin] = flag
    return _NO_CNV[cli_bin]


def _generate(cli_bin, model, prompt, sampling, ngl, n_predict=80):
    cmd = ([cli_bin, "-m", model, "-ngl", str(ngl), "-c", "2048", "-n", str(n_predict), "-p", prompt]
           + ([_one_shot_flag(cli_bin)] if _one_shot_flag(cli_bin) else [])
           + sampling)
    # On Windows, put the child in its own process group so console control events (Ctrl+C /
    # Ctrl+Break, and the close event a SYSTEM scheduled task generates when it has no interactive
    # desktop) are not delivered to it. Without this, adding `timeout=` below is enough to break a
    # gate that used to work: a plain communicate() blocks uninterruptibly, but
    # communicate(timeout=...) waits on a lock Windows CAN interrupt, so a stray console event
    # surfaces as KeyboardInterrupt and takes the whole run down.
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    # stdin=DEVNULL is the whole fix, and it is one line: some llama-cli builds (the MBZUAI-IFM
    # fork, b1-35999d1) open an interactive chat after generating and never exit on their own, so
    # subprocess.run waits on a process that is finished working but still sitting at a prompt.
    # Closing stdin ends that. NO timeout: adding one made communicate() wait on a lock Windows can
    # interrupt, which turned stray console events into fatal KeyboardInterrupts and -- worse -- gave
    # an empty result that the loop detector scored as PASS.
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       stdin=subprocess.DEVNULL, **kw)
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
            if isinstance(gen, _NoOutput) or not gen.strip():
                rows.append({"prompt": p.splitlines()[0][:48], "loop": None,
                             "reason": "NO OUTPUT (timeout or the run produced nothing)",
                             "sample": ""})
                return {"verdict": "NO_OUTPUT", "config": cfg_name, "rows": rows}
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
        mark = "----" if r["loop"] is None else ("LOOP" if r["loop"] else "ok  ")
        print(f"  [{mark}] {r['prompt']:<50} {r['reason']}")
        if r["loop"]:
            print(f"         -> {r['sample']}")
    if res["verdict"] == "NO_OUTPUT":
        print("\nVERDICT: NO OUTPUT -- the model produced nothing before the timeout, so this build\n"
              "  is UNGATED, not passed. Usually the run is simply slow (a big model with no GPU\n"
              "  offload): re-run with more time, or with -ngl set, before reading anything into it.")
        return False
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


_SPEED = re.compile(r"Generation:\s*([\d.]+)\s*t/s")
_PROMPT_SPEED = re.compile(r"Prompt:\s*([\d.]+)\s*t/s")
# ik_llama and older llama.cpp print the classic timing block instead:
#   main: prompt eval time = 217.23 ms / 1 tokens ( 217.23 ms per token, 4.60 tokens per second)
#   main:        eval time = 156.00 ms / 32 tokens (   4.88 ms per token, 205.12 tokens per second)
# Matching only "Generation: t/s" meant every trellis build reported "no speed line", which is why
# IQ*_KT rungs have never carried a tok/s figure on a card.
_CLASSIC = re.compile(r"([\d.]+)\s*tokens per second")


def _classic_speeds(text):
    """(generation, prompt) from the classic timing block, either may be None."""
    gen = pro = None
    for line in text.splitlines():
        if "tokens per second" not in line or "eval time" not in line:
            continue
        m = _CLASSIC.search(line)
        if not m:
            continue
        if "prompt eval time" in line:
            pro = float(m.group(1))
        else:
            gen = float(m.group(1))
    return gen, pro


def measure_speed(cli_bin, model, ngl, n_predict=128,
                  prompt="Explain why the sky is blue."):
    """Decode speed for one build: (generation_tps, prompt_tps), either may be None.

    This lives here rather than in a shell loop because the invocation needs the same hardening
    _generate has -- own process group so console events cannot kill it mid-load, stdin closed so a
    conversation-mode build exits. A raw llama-cli in a .bat on Windows gets killed part way through
    loading ("Loading model... ^C") and silently produces nothing.

    Speed is hardware-specific, so whatever consumes this has to name the machine beside it."""
    cmd = ([cli_bin, "-m", model, "-ngl", str(ngl), "-n", str(n_predict),
            "--no-warmup", "-p", prompt]
           + ([_one_shot_flag(cli_bin)] if _one_shot_flag(cli_bin) else []))
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           stdin=subprocess.DEVNULL, **kw)
    except KeyboardInterrupt:
        print("   ! interrupted by a console event during the speed run (not a real Ctrl+C)")
        return None, None
    out = (r.stdout or "") + (r.stderr or "")
    g, p = _SPEED.search(out), _PROMPT_SPEED.search(out)
    gen = float(g.group(1)) if g else None
    pro = float(p.group(1)) if p else None
    if gen is None:                        # older/ik_llama builds print the classic timing block
        gen, pro2 = _classic_speeds(out)
        pro = pro if pro is not None else pro2
    return gen, pro


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
    ap.add_argument("--speed", action="store_true",
                    help="also measure decode tok/s for --gguf (and --vs), on the same flags so the "
                         "two are comparable. Needs --llama-cli. tok/s is hardware-specific: state "
                         "the machine wherever you publish it.")
    ap.add_argument("--quick", action="store_true",
                    help="with --coherence: fast one-prompt / default-sampling sanity instead of the full sweep.")
    a = ap.parse_args()

    # --- coherence gate (can run standalone: no perplexity bin / eval corpus required) ---
    # The gate used to exit here as soon as it had a verdict, which silently swallowed --speed: ask
    # for both and you got the gate, exit 0, and no tok/s anywhere. A measuring tool must not drop a
    # measurement you asked for without saying so, so the verdict is held and the exit happens after
    # every requested measurement has run.
    gate_passed = None
    if a.coherence or a.quick:
        if not os.path.exists(a.gguf):
            sys.exit(f"file not found: {a.gguf}")
        cli_bin = find_llama_bin(a.llama_cli)
        if not cli_bin:
            sys.exit("llama-cli not found — build llama.cpp/ik_llama.cpp or pass --llama-cli.")
        res = coherence_gate(cli_bin, a.gguf, a.ngl, quick=a.quick)
        gate_passed = print_gate(res)
        if not a.ref and not a.speed:      # nothing else was asked for -> exit on the verdict
            sys.exit(0 if gate_passed else 2)

    if a.speed:
        cli_bin = find_llama_bin(a.llama_cli)
        if not cli_bin:
            sys.exit("--speed needs llama-cli — build it or pass --llama-cli.")
        print("\n=== decode speed ===")
        for label, m in (("this build", a.gguf), ("rival", a.rival)):
            if not m:
                continue
            gen, pro = measure_speed(cli_bin, m, a.ngl)
            name = os.path.basename(m)
            if gen is None:
                print(f"  {name}: no speed line (the run produced none)")
            else:
                print(f"  {name}: {gen:.1f} tok/s generation"
                      + (f"  ({pro:.0f} prompt)" if pro else ""))
        print("  tok/s is hardware-specific -- name the machine wherever you publish it.")
        if not a.ref and not os.path.exists(a.eval):
            # nothing left to score. If a gate also ran, its verdict is what the exit code means.
            sys.exit(0 if gate_passed is None else (0 if gate_passed else 2))

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

    out = a.out
    if not out:
        try:
            import pollard_workspace as ws, os as _os, time as _t
            out = _os.path.join(ws.reports_dir(a.gguf, create=True), f"bench-{_t.strftime('%Y%m%d-%H%M%S')}.json")
        except Exception:
            out = None
    if out:
        import json
        json.dump({"eval": a.eval, "ref": a.ref, "rows": rows}, open(out, "w"), indent=2)
        print(f"\nwrote {out}  (feed pollard-scorecard for the card)")


if __name__ == "__main__":
    main()
