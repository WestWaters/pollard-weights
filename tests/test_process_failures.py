"""The ways a low-bit build fails that are not a phrase loop.

detect_loop catches degenerate repetition. Below about three bits three more failures appear, and
each leaves the text locally coherent -- compression ratio and distinct-word ratio both look fine
-- while the trace balloons. A build that needs four times the tokens is not faster, whatever its
per-token cost is.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _detector():
    """Lift the detector out of pollard_bench without importing it (it needs runtime deps)."""
    src = (ROOT / "tools/pollard_bench.py").read_text()
    seg = src[src.index("OPENERS = {"):src.index("# A distinct sentinel")]
    ns = {"re": re}
    exec(seg, ns)
    return ns["detect_process_failures"]


F = _detector()


def names(*a, **kw):
    return [n for n, _ in F(*a, **kw)]


def test_a_finished_answer_trips_nothing():
    assert names("The derivative is 2x. Therefore, the answer is 2x.", budget=100, emitted=12) == []


def test_running_to_the_cap_is_budget_exhaustion():
    assert "budget-exhaustion" in names("thinking " * 50, budget=100, emitted=100)


def test_the_runtime_token_count_wins_over_the_word_count():
    """A runtime that reports tokens is authoritative; words are a conservative fallback."""
    assert "budget-exhaustion" in names("a b c", budget=100, emitted=100)
    assert "budget-exhaustion" not in names("word " * 200, budget=100, emitted=10)


def test_an_unclosed_reasoning_block_is_caught():
    assert "unclosed-segment" in names("<think>still working", budget=100, emitted=5)


def test_an_unclosed_code_fence_is_caught():
    assert "unclosed-segment" in names("here:\n```python\nx = 1", budget=100, emitted=6)


def test_a_closed_block_is_fine():
    assert "unclosed-segment" not in names("<think>ok</think> the answer is 4", budget=100, emitted=9)


def test_a_balanced_fence_is_fine():
    assert "unclosed-segment" not in names("```py\nx=1\n```\nthe answer is 1", budget=100, emitted=9)


def test_committing_only_at_the_end_is_delayed_commitment():
    assert "delayed-commitment" in names("padding here. " * 40 + " the answer is 7")


def test_committing_early_is_not():
    assert "delayed-commitment" not in names("The answer is 7. " + "then some working. " * 30)


def test_never_committing_within_the_budget_is_reported():
    assert "no-answer" in names("musing " * 60, budget=100, emitted=100)


def test_empty_output_reports_nothing_rather_than_guessing():
    """Empty is handled elsewhere by the NO_OUTPUT sentinel; this must not invent a failure."""
    assert names("", budget=100, emitted=0) == []


def test_the_gate_fails_a_build_with_a_process_failure():
    src = (ROOT / "tools/pollard_bench.py").read_text()
    assert "detect_process_failures(gen, budget=budget)" in src
    assert "or bool(proc)" in src, "a process failure must fail the gate, not just be printed"


def test_the_detector_stays_ascii():
    """Docstrings become --help output, and a legacy Windows codepage cannot encode arrows."""
    src = (ROOT / "tools/pollard_bench.py").read_text()
    seg = src[src.index("OPENERS = {"):src.index("# A distinct sentinel")]
    assert all(ord(c) < 128 for c in seg)
