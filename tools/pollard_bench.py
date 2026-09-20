#!/usr/bin/env python3
"""pollard-bench -- the drop-in BENCHMARK. Point it at a GGUF (or two) and get the gold-card
board: PPL + Mean/Median KLD + top-1, all from the same harness at matched size. This is the
symmetric half of `pollard` (which BUILDS): `pollard` makes the model, `pollard-bench` scores it.

    pollard-bench --gguf model.gguf --ref f16.gguf --eval held.txt          # one model, full board
    pollard-bench --gguf pollard.gguf --vs rival.gguf --ref f16.gguf         # HEAD-TO-HEAD (Pareto verdict)
    pollard-bench --gguf model.gguf --eval held.txt                         # PPL only (no --ref -> no KLD)
    pollard-bench --gguf model.gguf --coherence                             # COHERENCE GATE (loop check + sampling sweep)
    pollard-bench --gguf model.gguf --coherence --quick                     # fast one-prompt post-build sanity

--ref is the KL reference (f16 ideally; a Q8_0/Q6_K host if f16 won't load -- KLD vs a near-lossless
ref is what the 14B/30B cards use). --vs runs the SAME eval on a competitor's file (AWQ/GPTQ/unsloth/
bartowski GGUF) so the comparison is honest: read it as Pareto -- Pollard wins its size class (same
quality for fewer GB, or more quality at the same GB), not as a single number.

Reuses llama-perplexity; no rebuild. This is the opt-in benchmark -- a plain `pollard` build never
runs it (that's the split that stopped a minutes-long shrink from taking hours).
"""
import argparse, os, re, shutil, subprocess, sys, zlib
from collections import Counter

from pollard_calc import find_llama_bin


# ---- coherence gate: separate a sampling-fixable loop from a below-the-floor build -----------
# The mix sets QUALITY-at-size; sampling is a RUNTIME knob it can't reach. So a build can loop
# two ways: (a) good build, bad sampling -> a sweep finds coherent settings (ship them); (b) the
# bit tier is below the model's coherence floor -> it loops under EVERY sampling -> bump a tier.
# This gate runs the model, detects loops, sweeps sampling, and returns which case it is.

# (prompt, any-of expected in the CONTINUATION). The gate used to check only that output did not
# LOOP, then call the result "coherent" -- so a build emitting fluent-looking token salad
# ("isletedGESarz Svensri--st IC himself1 andict zichzelf") passed, because salad does not repeat.
# A known answer is the cheapest real check: a model that cannot finish these is broken, whatever
# its perplexity says.
GATE_PROMPTS = [
    ("Paris is the capital of France. The largest planet in our solar system is",
     ("jupiter",)),
    ("Here is a short explanation of how photosynthesis works:",
     ("light", "sun", "water", "carbon", "energy", "plant", "chloroph")),
    ("# Python function to compute the nth Fibonacci number\ndef fib(n):",
     ("return", "fib", "n-1", "n - 1", "if n")),
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


THREADS = None          # set once from --threads in main(); the subprocess builders read it


def _t():
    """`-t N` for the llama.cpp binaries, or nothing when the user has not asked."""
    return ["-t", str(THREADS)] if THREADS else []


def _env_threads():
    """POLLARD_THREADS, so a shared box can be configured once instead of per-command."""
    try:
        v = int(os.environ.get("POLLARD_THREADS", "") or 0)
        return v if v > 0 else None
    except ValueError:
        return None


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


OPENERS = {"<think>": "</think>", "<reasoning>": "</reasoning>",
           "<scratchpad>": "</scratchpad>", "<answer>": "</answer>"}
ANSWER_RE = re.compile(r"(?:\\boxed\{|final answer|the answer is|therefore,?\s|answer:|"
                       r"^\s*(?:so|thus)\b)", re.I | re.M)


def detect_process_failures(text, budget=None, emitted=None):
    """The ways a low-bit build fails that are NOT a phrase loop.

    detect_loop above catches degenerate repetition. Below about three bits three more failures
    show up, and each inflates token count while leaving the text locally coherent enough that
    compression ratio and distinct-word ratio both look fine:

      budget-exhaustion   generation never halts; it stops because it hit the cap
      delayed-commitment  an answer exists, but only after most of the budget is spent
      unclosed-segment    a reasoning block or code fence opened and never closed

    They matter because the token saving from smaller weights is given straight back when a trace
    balloons -- a 2-bit build that needs four times the tokens is not faster, whatever its
    per-token cost. Returns a list of (name, detail); empty means none of these fired.
    """
    t = (text or "").strip()
    out = []
    if not t:
        return out

    words = re.findall(r"\S+", t)
    # Hit the cap rather than choosing to stop. `emitted` is authoritative when the runtime
    # reports it; the word count is a fallback and is deliberately conservative. Both the
    # exhaustion check and the no-answer check below read the SAME number -- using the word count
    # for one and the token count for the other reports a build that ran to the cap as having
    # simply not answered yet.
    used = emitted if emitted is not None else len(words)
    exhausted = bool(budget and used >= budget * 0.98)
    if exhausted:
        out.append(("budget-exhaustion",
                    f"stopped at the {budget}-token cap rather than finishing"))

    unclosed = [tag for tag, close in OPENERS.items() if t.count(tag) > t.count(close)]
    if t.count("```") % 2:
        unclosed.append("```")
    if unclosed:
        out.append(("unclosed-segment", "never closed: " + ", ".join(unclosed)))

    m = ANSWER_RE.search(t)
    if m and len(t) > 200 and m.start() / len(t) > 0.85:
        out.append(("delayed-commitment",
                    f"first commits to an answer {m.start() / len(t):.0%} of the way through"))
    elif not m and exhausted:
        out.append(("no-answer", "ran to the cap without ever committing to an answer"))
    return out


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


def _generate(cli_bin, model, prompt, sampling, ngl, n_predict=220):
    # 220, not 80: a reasoning-tuned model spends its first hundred-odd tokens inside a
    # thinking block, so a short budget cuts it off mid-thought and the known-answer check
    # fails a build that was about to answer correctly.
    cmd = ([cli_bin, "-m", model, "-ngl", str(ngl), "-c", "2048", "-n", str(n_predict), "-p", prompt]
           + _t()
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
    # A thinking model reasons before it answers, so a fixed budget cuts it off mid-thought
    # and the known-answer check fails a build that was about to be right. Ask what it is.
    try:
        from pollard_modelkind import classify, describe
        kind = classify(model)
        budget = kind["gate_tokens"]
        print(f"  model kind: {describe(kind)}  (gate budget {budget} tokens)")
    except Exception:
        budget = 220
    last = []
    for cfg_name, sampling in configs:
        rows, looped = [], False
        for p, expect in prompts:
            gen = _generate(cli_bin, model, p, sampling, ngl, budget)
            if isinstance(gen, _NoOutput) or not gen.strip():
                rows.append({"prompt": p.splitlines()[0][:48], "loop": None,
                             "reason": "NO OUTPUT (timeout or the run produced nothing)",
                             "sample": ""})
                return {"verdict": "NO_OUTPUT", "config": cfg_name, "rows": rows}
            is_loop, metric, reason = detect_loop(gen)
            # A CONTROL token repeating (<|channel|>, <|im_start|>, <pad>) is not the body of the
            # model collapsing -- it is the token embedding losing resolution, and the fix is to
            # protect that tensor, not to spend size on every layer.
            if is_loop and re.search(r"<\|[^|>]{1,32}\|?>|<[a-z_]{2,16}>", gen):
                reason += "  [control tokens repeating -> token-embedding precision]"
            # Not looping is not the same as coherent. Check the model actually produced the
            # known answer; salad passes a loop test and fails this.
            knows = any(e.lower() in gen.lower() for e in expect)
            if not is_loop and not knows:
                reason = f"INCOHERENT (no {'/'.join(expect[:3])} in the continuation)"
            # Loops are not the only way a low-bit build fails. A trace that never halts, commits
            # only at the very end, or leaves a reasoning block open stays locally coherent --
            # compression ratio and distinct-word ratio both look fine -- while the token count
            # balloons and gives back the saving the smaller weights bought.
            proc = detect_process_failures(gen, budget=budget)
            if proc:
                reason += "  [" + "; ".join(f"{n}: {d}" for n, d in proc) + "]"
            rows.append({"prompt": p.splitlines()[0][:48], "loop": is_loop or not knows,
                         "reason": reason, "process": [n for n, _ in proc],
                         "sample": gen.strip().replace("\n", " ")[:120]})
            looped = looped or is_loop or not knows or bool(proc)
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
        print(f"\nVERDICT: PASS -- coherent. Ship these sampling defaults on the card:\n  {s}")
    else:
        print("\nVERDICT: BELOW FLOOR -- every sampling config was tried and none held together, so\n"
              "  this is not a sampling problem. It is also not the end of the road: a build lands\n"
              "  here when the bits it was given cannot carry the model, and Pollard has a lever for\n"
              "  every part of that. Work them in cost order -- each is cheaper than the one after:\n"
              "\n"
              "  -- protect what the failure points at ------------------------------------------\n"
              "   1. TOKEN EMBEDDING + OUTPUT TENSOR. First when a control token repeats\n"
              "      (<|channel|>, <|im_start|>): a big vocabulary at Q4_K loses the special tokens\n"
              "      first, and the model cannot stop emitting them. A few hundred MB, not a tier.\n"
              "        pollard-automap ... --output-tensor-type Q6_K --token-embedding-type Q6_K\n"
              "\n"
              "  -- measure before you spend --------------------------------------------------\n"
              "   2. MEASURED ALLOCATION. Put the bits where this model actually needs them\n"
              "      instead of crushing evenly. pollard-probe is the cheap any-box profile;\n"
              "      pollard-sensitivity is the ground truth (heavier, GGUF sweep).\n"
              "        pollard-probe --model <hf> --eval held.txt --out m.sensitivity.json\n"
              "        pollard-fit --gguf <f16> --ram N --sensitivity m.sensitivity.json\n"
              "\n"
              "   3. CALIBRATION. A low-bit build leans on the imatrix harder than any other rung,\n"
              "      and a short or single-domain corpus is the usual reason a tier looks\n"
              "      impossible. Calib 3.0 is multi-domain on purpose -- do not trim it.\n"
              "        pollard-calib --out calib.txt --held-out eval.txt\n"
              "        llama-imatrix -m <f16> -f calib.txt -o m.dat --output-format dat\n"
              "      (dat, not the gguf default: ik_llama builds the trellis flagship and reads\n"
              "       only the legacy format.)\n"
              "\n"
              "  -- precondition the weights --------------------------------------------------\n"
              "   4. Let Pollard pick the preconditioner for THIS model and bit-width -- the best\n"
              "      one is bit-width dependent, and guessing wastes a build:\n"
              "        pollard-precondition --model <hf>          # measures which lever wins\n"
              "      or drive one directly:\n"
              "        pollard-rotate     # incoherence (QuIP#/QuaRot) -- pays at IQ low bit\n"
              "        pollard-smooth     # activation-aware, AWQ-style\n"
              "        pollard-hf-smooth  # SmoothQuant, in place on the HF weights\n"
              "\n"
              "  -- change the alphabet, not the size ------------------------------------------\n"
              "   5. BELOW 2 BITS, a different alphabet beats a smaller number of bits:\n"
              "        pollard-palette    # measured mixed-ALPHABET allocation under the 2-bit floor\n"
              "        pollard-lowbit     # extreme-low-bit R&D prototype\n"
              "      MoE only: drop whole cold experts rather than crushing every one of them:\n"
              "        pollard-prune      # REAP-style expert pruning\n"
              "\n"
              "  -- last, because it costs size -------------------------------------------------\n"
              "   6. Widen protection (--protect iq3_kt), then bump the body tier\n"
              "      (--body iq1_kt -> iq2_kt) and re-gate. Last because the levers above often\n"
              "      make it unnecessary.\n"
              "\n"
              "  Re-gate after each: `pollard-bench --gguf <build> --coherence`. Sampling is already\n"
              "  swept here, so a PASS reports the defaults to ship on the card.\n"
              "  (Small/sparse models hit the floor sooner; big models clear 1-bit fine -- a size\n"
              "   property, not a defect in the build.)")
    return res["verdict"] == "PASS"


def _size_gb(p):
    try:
        return os.path.getsize(p) / 1e9
    except OSError:
        return None


def _ppl_kl(ppl_bin, model, eval_f, base, ngl, chunks=0):
    """Run llama-perplexity and parse PPL (+ Mean/Median KLD + top-1 when a base is given)."""
    cmd = [ppl_bin, "-m", model, "-f", eval_f, "-c", "2048", "-ngl", str(ngl)] + _t()
    if chunks:
        cmd += ["--chunks", str(chunks)]
    if base:
        cmd += ["--kl-divergence", "--kl-divergence-base", base]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr
    def g(pat):
        m = re.search(pat, out, re.M)
        return float(m.group(1)) if m else None
    # llama-perplexity prints a different report under --kl-divergence: no "Final estimate", but
    # BOTH perplexities, which is what the card wants anyway. Every pattern is anchored to its own
    # line: a negated-colon class matches NEWLINES too, so the old top-1 pattern latched onto the
    # table header ("... Same top p") and ran on to the next colon anywhere below, reporting a
    # number that was not a percentage and looked like a result.
    return {
        "ppl":    g(r"^Mean PPL\(Q\)\s*:\s*([0-9.]+)")
                  or g(r"Final estimate:\s*PPL[^=\n]*=\s*([0-9.]+)"),
        "ref_ppl":    g(r"^Mean PPL\(base\)\s*:\s*([0-9.]+)"),
        "mean_kld":   g(r"^Mean\s+KLD:\s*([0-9.]+)"),
        "median_kld": g(r"^Median\s+KLD:\s*([0-9.]+)"),
        "top1":   g(r"^Same top p:\s*([0-9.]+)"),        # top-1 agreement %
    }


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, float) else "--"


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
    cmd = ([cli_bin, "-m", model, "-ngl", str(ngl), "-n", str(n_predict)] + _t() + [
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



def _eval_overlaps_calib(imatrix_path, eval_path):
    """Is the eval text the same text that calibrated the build? Returns a reason, or None.

    THIS tool's own check -- pollard-bench does not depend on another tool to know whether the
    number it is about to print is a real measurement.

    An imatrix records the dataset(s) it was built from, so the cheap answer is the recorded path.
    When the names differ, compare content: a held-out split that shares most of its lines with the
    calibration corpus is held out in name only. That is how gemma-4-12B-it came to publish
    perplexity measured on its own calibration text -- 1366 of 1366 lines of `eval_heldout.txt`
    are inside `gemma4_calib.txt`.
    """
    ds = []                                               # os/re are module-level; a local import
    try:                                                  # of either would shadow them body-wide
        blob = open(imatrix_path, "rb").read()
        ds = [m.decode("utf-8", "ignore")
              for m in re.findall(rb"[A-Za-z]:\\[^\x00]{3,200}?\.txt", blob)]
    except Exception:
        pass
    try:
        from gguf import GGUFReader                       # GGUF imatrix records it in the KV
        for f in GGUFReader(imatrix_path).fields.values():
            if "dataset" in f.name.lower():
                ds += [str(f.parts[i].tobytes().decode("utf-8", "ignore")) for i in f.data]
    except Exception:
        pass
    ev = os.path.abspath(eval_path)
    for d in ds:
        if os.path.abspath(d) == ev or os.path.basename(d) == os.path.basename(ev):
            return f"the imatrix records this exact file as its calibration corpus: {d}"
    try:
        with open(ev, encoding="utf-8", errors="ignore") as fh:
            ev_lines = {ln.strip() for ln in fh if len(ln.strip()) > 40}
        for d in set(ds):
            if not os.path.exists(d):
                continue
            with open(d, encoding="utf-8", errors="ignore") as fh:
                cal = {ln.strip() for ln in fh if len(ln.strip()) > 40}
            if ev_lines:
                frac = len(ev_lines & cal) / len(ev_lines)
                if frac > 0.10:
                    return (f"{frac*100:.1f}% of the eval's lines are in the calibration corpus "
                            f"{os.path.basename(d)}")
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", required=True, help="the model to score (a Pollard build, or any GGUF)")
    ap.add_argument("--vs", dest="rival", help="a competitor GGUF to score the same way (head-to-head)")
    ap.add_argument("--ref", help="KL reference GGUF (f16, or a near-lossless Q8_0/Q6_K host). "
                                  "Omit for PPL-only (no KLD/top-1).")
    ap.add_argument("--eval", default="wikitext2_test.txt", help="held-out eval text")
    ap.add_argument("--imatrix", help="the imatrix the build used. Given, the eval is CHECKED "
                                      "against the corpus that calibrated the model -- numbers "
                                      "measured on the calibration text are not a quality result.")
    ap.add_argument("--allow-eval-overlap", action="store_true",
                    help="score anyway when the eval overlaps the calibration corpus (research)")
    ap.add_argument("--chunks", type=int, default=0,
                    help="cap the eval at N chunks. The KL base holds FULL logits, so a large "
                         "vocab over a long eval runs to tens of GB; 200 gives the same "
                         "comparison at a fraction of the size. 0 = whole file.")
    ap.add_argument("--ngl", type=int, default=99, help="GPU layers (lower for a model bigger than the GPU)")
    ap.add_argument("--out", help="write a results.json (feeds pollard-scorecard)")
    ap.add_argument("--llama-perplexity", default="llama-perplexity")
    ap.add_argument("--llama-cli", default="llama-cli", help="generation binary for the coherence gate")
    ap.add_argument("--threads", type=int, default=_env_threads(),
                    help="number of threads for the heavy step. Default: the tool's own choice, which is usually every core. Set it lower to leave the machine usable -- a quantize that takes the whole box is a quantize you cannot run while anything else matters. POLLARD_THREADS sets it for every tool.")
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
    global THREADS
    THREADS = a.threads

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
            sys.exit("llama-cli not found -- build llama.cpp/ik_llama.cpp or pass --llama-cli.")
        res = coherence_gate(cli_bin, a.gguf, a.ngl, quick=a.quick)
        gate_passed = print_gate(res)
        if not a.ref and not a.speed:      # nothing else was asked for -> exit on the verdict
            sys.exit(0 if gate_passed else 2)

    if a.speed:
        cli_bin = find_llama_bin(a.llama_cli)
        if not cli_bin:
            sys.exit("--speed needs llama-cli -- build it or pass --llama-cli.")
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
        sys.exit("llama-perplexity not found -- build llama.cpp/ik_llama.cpp or pass --llama-perplexity.")
    # The corpus that BUILT the quant must not also MEASURE it. If the same text drives the imatrix
    # and the eval, a bad allocation scores well because it is graded on the lines it was tuned on --
    # and these numbers go on a public card. This is not hypothetical: gemma-4-12B-it shipped with
    # PPL measured on `eval_heldout.txt`, every line of which is inside `gemma4_calib.txt`.
    if a.imatrix and os.path.exists(a.imatrix) and os.path.exists(a.eval):
        hit = _eval_overlaps_calib(a.imatrix, a.eval)
        if hit and not a.allow_eval_overlap:
            sys.exit(f"REFUSING to score on the calibration corpus.\n  {hit}\n"
                     "  Perplexity measured on the text the imatrix was built from is flattered,\n"
                     "  not held out. Build a disjoint eval (pollard-calib --out train.txt\n"
                     "  --held-out eval.txt writes both from ONE run), or pass --allow-eval-overlap.")
    elif not a.imatrix:
        print("  (no --imatrix given: eval/calibration disjointness NOT verified)")
    if not os.path.exists(a.eval) or os.path.getsize(a.eval) == 0:
        sys.exit(f"eval corpus not found or empty: {a.eval}")
    for f in [a.gguf, a.rival, a.ref]:
        if f and not os.path.exists(f):
            sys.exit(f"file not found: {f}")

    base = None
    if a.ref:
        # Named from the REFERENCE, not the model being scored: the base logits depend only on
        # the reference, so one file serves every rung. Naming it per-model wrote a full copy
        # each time -- 76GB apiece for a 152k vocab -- and filled the disk, after which the
        # remaining rungs silently reported "--".
        # ...and written into the WORKSPACE cache, not beside the reference. Every Pollard artifact
        # belongs under POLLARD_HOME; dropping a 30GB intermediate next to whatever file happened to
        # be passed as --ref scatters them across downloads/, source trees and working directories,
        # so nobody can find them, account for the space, or clean them up.
        try:
            import pollard_workspace as _ws
            base = os.path.join(_ws.cache_dir(create=True),
                                os.path.splitext(os.path.basename(a.ref))[0] + ".klbase.dat")
        except Exception:                                   # no workspace: keep the old behaviour
            base = os.path.splitext(a.ref)[0] + ".klbase.dat"
        if os.path.exists(base) and os.path.getsize(base) > 0:
            print(f"[1] reusing KL base {os.path.basename(base)} "
                  f"({os.path.getsize(base)/1e9:.1f} GB)")
        else:
            print(f"[1] KL base logits from {os.path.basename(a.ref)} (ngl {a.ngl}) ...")
            cmd = [ppl_bin, "-m", a.ref, "-f", a.eval, "-c", "2048",
                   "-ngl", str(a.ngl), "--kl-divergence-base", base]
            if a.chunks:
                cmd += ["--chunks", str(a.chunks)]
            subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            free = shutil.disk_usage(os.path.dirname(os.path.abspath(base)) or ".").free
            if not os.path.exists(base) or os.path.getsize(base) == 0:
                sys.exit(f"could not build KL base logits ({free/1e9:.1f} GB free). The base holds "
                         "FULL logits -- tokens x vocab x 4 bytes -- so a large vocab over a long "
                         "eval runs to tens of GB. Cap it with --chunks, lower --ngl, or use a "
                         "smaller near-lossless --ref like Q6_K.")
    else:
        print("[1] no --ref -> PPL only (pass --ref f16/Q8/Q6 for Mean/Median KLD + top-1).")

    try:
        from pollard_modelkind import classify, describe
        k = classify(a.gguf)
        if k["eval"] == "in-domain":
            print(f"  NOTE: this is a {describe(k)} model. Perplexity on RAW text measures the")
            print("        mismatch, not the build -- gemma-4-12B-it reads ~664 on WikiText where a")
            print("        plain 7B reads 5.4 on the same corpus and binary. Score it on text it was")
            print("        tuned for:  pollard-calib --out train.txt --held-out eval.txt")
    except Exception:
        pass
    targets = [("model", a.gguf)] + ([("rival", a.rival)] if a.rival else [])
    rows = []
    for i, (tag, m) in enumerate(targets, 2):
        print(f"[{i}] scoring {os.path.basename(m)} ...")
        r = _ppl_kl(ppl_bin, m, a.eval, base, a.ngl, a.chunks)
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
                verdict = ("Pareto trade -- read the size/quality curve: "
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
