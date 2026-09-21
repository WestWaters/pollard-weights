"""KV cache precision: pollard-calc could size it, nothing measured what it costs.

At long context the cache, not the weights, is what fills the machine -- so the trade gets made
blind. And keys and values do not want the same precision: their outlier structure differs, which
is why the sweep includes asymmetric rows. A symmetric-only sweep never tries the row that usually
wins.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "tools/pollard_bench.py").read_text()


def _ns():
    seg = SRC[SRC.index("KV_SWEEP = ["):SRC.index("def detect_loop")]
    ns = {"re": re, "subprocess": None, "_t": lambda: []}
    exec(seg, ns)
    return ns


NS = _ns()


def test_the_flag_exists():
    assert '"--kv-sweep"' in SRC and '"--kv-ctx"' in SRC


def test_it_passes_the_cache_type_flags_llamacpp_actually_takes():
    assert '"-ctk", k, "-ctv", v' in SRC


def test_the_sweep_tries_asymmetric_combinations():
    """Keys and values are not symmetric; a sweep that only tries k==v misses the useful row."""
    combos = [(k, v) for k, v, _ in NS["KV_SWEEP"]]
    assert any(k != v for k, v in combos), "no asymmetric row"
    assert ("q8_0", "q4_0") in combos, "keys-high/values-low is the row that usually wins"


def test_the_baseline_is_first_so_deltas_mean_something():
    assert NS["KV_SWEEP"][0][:2] == ("f16", "f16")


def test_every_row_explains_itself():
    for k, v, note in NS["KV_SWEEP"]:
        assert note and len(note) > 10, f"{k}/{v} has no explanation"


def test_the_sweep_honours_the_thread_flag():
    seg = SRC[SRC.index("def kv_sweep("):SRC.index("def print_kv_sweep")]
    assert "+ _t()" in seg


def test_an_unsupported_cache_type_is_reported_not_crashed():
    """Not every llama.cpp build supports every cache type; that is a row, not a failure."""
    seg = SRC[SRC.index("def kv_sweep("):SRC.index("def print_kv_sweep")]
    assert '"ppl": None' in seg and "unsupported cache type" in seg


def test_the_recommendation_requires_both_cheap_and_smaller():
    """f16/f16 costs 0% and saves nothing; it must never be recommended as the win."""
    seg = SRC[SRC.index("def print_kv_sweep("):]
    seg = seg[:seg.index("\n\ndef ")]
    assert 'abs(r["pct"]) < 1.0 and r["cache_frac"] < 0.9' in seg


def test_it_says_perplexity_understates_kv_damage():
    """Short-chunk perplexity is the least sensitive way to see this, and the user should know."""
    seg = SRC[SRC.index("def print_kv_sweep("):]
    assert "LEAST sensitive" in seg[:2000]


def test_it_can_run_standalone():
    assert 'if not (a.gguf and a.eval)' in SRC
    assert "if not a.coherence:\n            return" in SRC


def test_the_new_code_stays_ascii():
    seg = SRC[SRC.index("KV_SWEEP = ["):SRC.index("def detect_loop")]
    assert all(ord(c) < 128 for c in seg)
