"""pollard-bench must never turn a failed measurement into a quiet row of "--".

FrogNano, 2026-10-05: the box's disk filled while writing the KL base, so the base stopped at chunk 109.
Every rung then scored 109 chunks, died reading chunk 109, and llama-perplexity -- which prints its
summary only after the last chunk -- printed none. The bench recorded null for PPL, KLD and top-1 on
three rungs, exited 0, and nothing said why.
"""
from __future__ import annotations

import io
import os
import re
import sys
import types
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _ppl_kl(stdout, returncode=0):
    """Lift _ppl_kl out of pollard_bench (it needs runtime deps to import) and feed it a canned run."""
    src = (ROOT / "tools/pollard_bench.py").read_text()
    seg = src[src.index("def _ppl_kl("):src.index("def _fmt(")]
    fake = types.SimpleNamespace(run=lambda *a, **k: types.SimpleNamespace(stdout=stdout, stderr="", returncode=returncode))
    ns = {"re": re, "os": os, "sys": sys, "subprocess": fake, "_t": lambda: []}
    exec(seg, ns)
    err = io.StringIO()
    with redirect_stderr(err):
        res = ns["_ppl_kl"]("llama-perplexity", "/m/FrogNano-Q6_K.gguf", "eval.txt", "base.dat", 0)
    return res, err.getvalue()


SHORT_BASE = """\
 108       9.8906 ±    0.0851       0.01065 ±    0.00064       0.01163 ±    0.00066     2.673 ±  0.079 %    96.790 ±  0.053 %
 109       9.8982 ±    0.0848       0.01062 ±    0.00064       0.01155 ±    0.00066     2.664 ±  0.079 %    96.798 ±  0.053 %
kl_divergence: failed reading log-probs for chunk 109
"""

GOOD = """\
====== Perplexity statistics ======
Mean PPL(Q)                   :   9.912345 ±   0.084000
Mean PPL(base)                :   9.880001 ±   0.083000
====== KL divergence statistics ======
Mean    KLD:   0.010650 ±   0.000640
Median  KLD:   0.004321
====== Token probability statistics ======
Same top p: 96.790 ± 0.053 %
"""


def test_short_kl_base_is_reported_not_silent():
    res, err = _ppl_kl(SHORT_BASE, returncode=1)
    assert all(v is None for v in res.values())
    assert "llama-perplexity failed on FrogNano-Q6_K.gguf" in err
    assert "failed reading log-probs for chunk 109" in err


def test_failure_is_reported_even_when_exit_code_is_zero():
    res, err = _ppl_kl(SHORT_BASE, returncode=0)
    assert all(v is None for v in res.values())
    assert "failed reading log-probs" in err


def test_a_good_run_parses_and_stays_quiet():
    res, err = _ppl_kl(GOOD)
    assert res == {"ppl": 9.912345, "ref_ppl": 9.880001, "mean_kld": 0.01065, "median_kld": 0.004321, "top1": 96.79}
    assert err == ""


def test_partial_kl_base_is_never_kept():
    src = (ROOT / "tools/pollard_bench.py").read_text()
    seg = src[src.index("[1] KL base logits from"):src.index("could not build KL base logits")]
    assert "os.remove(base)" in seg and "free < 1e9" in seg
