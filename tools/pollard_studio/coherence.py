#!/usr/bin/env python3
"""Detect the ways a low-bit model fails that perplexity does not see.

Perplexity is a blunt instrument here. The literature is blunt about it too: a model can improve
on perplexity while its behaviour degrades, and reasoning accuracy falls several times faster than
perplexity suggests. What actually goes wrong below ~3 bits is process-level, and it has a shape:

    repetitive loops      the same n-gram cycling until the token budget runs out
    budget exhaustion     generation never halts; it stops because it hit the cap
    delayed commitment    an answer exists, but only after most of the budget is spent
    unclosed segments     a reasoning block is opened and never closed

These are cheap to detect from the generated text alone -- no reference model, no logits -- which
makes them usable as a gate on every build rather than an occasional audit.

Named after the failure taxonomy in "Extreme Low-Bit Inference in Reasoning Models"
(arXiv:2606.02011), where detecting loops and falling back was worth more than any amount of
extra calibration.

    python -m pollard_studio.coherence transcript.txt
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

OPENERS = {"<think>": "</think>", "<reasoning>": "</reasoning>",
           "<scratchpad>": "</scratchpad>", "<answer>": "</answer>", "```": "```"}
ANSWER_RE = re.compile(
    r"(?:\\boxed\{|final answer|the answer is|therefore,?\s|answer:)", re.I)


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def repetition(text: str, n: int = 8, threshold: int = 3) -> dict:
    """Longest repeated n-gram and how many times it cycles.

    A loop is not 'some repetition' -- natural text repeats. It is the SAME window recurring
    enough times that the model is clearly not advancing.
    """
    w = _words(text)
    if len(w) < n * 2:
        return {"looping": False, "repeats": 0, "phrase": "", "n": n}
    grams = Counter(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))
    phrase, count = grams.most_common(1)[0]
    return {"looping": count >= threshold, "repeats": count, "phrase": phrase[:120], "n": n}


def tail_repetition(text: str, window: int = 60) -> dict:
    """Is the END of the generation cycling? A loop that starts late is still a loop."""
    w = _words(text)[-window * 3:]
    return repetition(" ".join(w), n=6, threshold=3)


def unclosed(text: str) -> dict:
    """Reasoning blocks opened and never closed."""
    open_tags = []
    for tag, close in OPENERS.items():
        if tag == "```":
            if text.count("```") % 2:
                open_tags.append("```")
            continue
        if text.count(tag) > text.count(close):
            open_tags.append(tag)
    return {"unclosed": bool(open_tags), "tags": open_tags}


def commitment(text: str) -> dict:
    """Where in the generation an answer first appears, as a fraction of the whole."""
    m = ANSWER_RE.search(text)
    if not m:
        return {"committed": False, "at": None}
    return {"committed": True, "at": round(m.start() / max(len(text), 1), 3)}


def analyse(text: str, max_tokens: int | None = None,
            stop_reason: str | None = None) -> dict:
    """Every check, plus a single verdict the gate can act on."""
    words = _words(text)
    rep = repetition(text)
    tail = tail_repetition(text)
    unc = unclosed(text)
    com = commitment(text)
    exhausted = bool(stop_reason == "length"
                     or (max_tokens and len(words) >= max_tokens * 0.98))

    failures = []
    if rep["looping"] or tail["looping"]:
        failures.append("repetitive-loop")
    if exhausted:
        failures.append("budget-exhaustion")
    if com["committed"] and com["at"] is not None and com["at"] > 0.85:
        failures.append("delayed-commitment")
    if unc["unclosed"]:
        failures.append("unclosed-segment")
    if not com["committed"] and exhausted:
        failures.append("no-answer")

    return {
        "ok": not failures, "failures": failures,
        "words": len(words), "repetition": rep, "tail_repetition": tail,
        "unclosed": unc, "commitment": com, "budget_exhausted": exhausted,
        "verdict": "coherent" if not failures else "incoherent: " + ", ".join(failures),
    }


def gate(samples: list[dict]) -> dict:
    """Roll several generations into a pass/fail a build can be held to."""
    if not samples:
        return {"pass": False, "reason": "no samples", "rate": 0.0, "n": 0}
    results = [analyse(s.get("text", ""), s.get("max_tokens"), s.get("stop_reason"))
               for s in samples]
    bad = [r for r in results if not r["ok"]]
    rate = 1 - len(bad) / len(results)
    modes = Counter(f for r in bad for f in r["failures"])
    return {"pass": not bad, "rate": round(rate, 3), "n": len(results),
            "failed": len(bad), "modes": dict(modes), "results": results,
            "reason": "all coherent" if not bad
                      else f"{len(bad)}/{len(results)} incoherent: {dict(modes)}"}


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    text = Path(sys.argv[1]).read_text(errors="replace")
    r = analyse(text)
    print(json.dumps(r, indent=2))
    raise SystemExit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
