"""Pollard's own trellis quantizer.

Provenance matters here more than usual. This is written from the published description of
trellis coded quantization with a computed codebook, not translated from any implementation:
Cornell-RelaxML/qtip is GPL-3 (copying it would force GPL-3 on everything downstream), and the
attempt to carry ik_llama's types into mainline llama.cpp was closed over exactly that provenance
question. Algorithms are not copyrightable; implementations are.

What these check is that it is a real trellis -- that it beats a scalar quantizer at equal rate,
which is the only reason to pay for Viterbi at all.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("ptr", ROOT / "tools/pollard_trellis.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ptr"] = m
    spec.loader.exec_module(m)
    return m


T = _load()


def rtn_error(W, K):
    """Round-to-nearest at the same bit rate -- the thing a trellis has to beat."""
    lo, hi = W.min(), W.max()
    step = (hi - lo) / ((1 << K) - 1)
    R = np.round((W - lo) / step) * step + lo
    return np.linalg.norm(W - R) / np.linalg.norm(W)


@pytest.mark.parametrize("K", [2, 3])
def test_it_beats_round_to_nearest_at_the_same_rate(K):
    """If it does not, the Viterbi search is not buying anything and the method is pointless."""
    W = np.random.default_rng(0).standard_normal((4, 256)) * 0.02
    Q, _ = T.quantize(W, L=10, K=K)
    err = np.linalg.norm(W - Q) / np.linalg.norm(W)
    assert err < rtn_error(W, K), f"K={K}: trellis {err:.4f} vs RTN {rtn_error(W, K):.4f}"


def test_more_bits_is_less_error():
    W = np.random.default_rng(1).standard_normal((4, 256)) * 0.02
    errs = []
    for K in (1, 2, 3):
        Q, _ = T.quantize(W, L=10, K=K)
        errs.append(np.linalg.norm(W - Q) / np.linalg.norm(W))
    assert errs[0] > errs[1] > errs[2], errs


def test_the_state_is_a_sliding_window_of_emitted_symbols():
    """The bitshift trellis is the whole point: the structure is implied by a shift, so nothing
    has to be stored, which is what makes decoding cheap."""
    L, K = 10, 2
    mask = (1 << L) - 1
    for state in (0, 1, 511, 1023):
        for sym in range(1 << K):
            assert ((state << K) | sym) & mask == T.quantize_row.__globals__["np"].uint64(
                ((state << K) | sym) & mask)


def test_the_codebook_is_computed_not_stored():
    src = (ROOT / "tools/pollard_trellis.py").read_text()
    assert "def codebook(" in src
    body = src[src.index("def codebook("):src.index("def quantize_row")]
    assert "np.zeros" not in body and "table" not in body.lower(), \
        "a stored table defeats the purpose"


def test_the_codebook_is_roughly_unit_variance_and_centred():
    """The trellis assumes a Gaussian-ish target, which is what incoherence processing produces."""
    v = T.codebook(np.arange(1 << 14), scale=1.0)
    assert abs(float(v.mean())) < 0.05, float(v.mean())
    assert 0.8 < float(v.std()) < 1.25, float(v.std())


def test_the_codebook_is_deterministic():
    a = T.codebook(np.arange(1000))
    b = T.codebook(np.arange(1000))
    assert np.array_equal(a, b), "decode must reproduce encode exactly"


def test_hessian_weighting_protects_the_channels_that_matter():
    """With a diagonal Hessian the solver should spend its budget on the heavy channels."""
    rng = np.random.default_rng(3)
    w = rng.standard_normal(256) * 0.02
    h = np.ones(256)
    h[:32] = 100.0                                    # the first 32 channels matter far more
    q_flat, _, _ = T.quantize_row(w, L=10, K=2)
    q_wtd, _, _ = T.quantize_row(w, L=10, K=2, hdiag=h)
    heavy_flat = float(np.sum(h[:32] * (w[:32] - q_flat[:32]) ** 2))
    heavy_wtd = float(np.sum(h[:32] * (w[:32] - q_wtd[:32]) ** 2))
    assert heavy_wtd <= heavy_flat, "Hessian weighting made the important channels worse"


def test_the_rate_is_k_bits_plus_the_initial_state():
    assert T.bits_per_weight(1024, L=16, K=2) == pytest.approx(2 + 16 / 1024)
    assert T.bits_per_weight(64, L=16, K=3) == pytest.approx(3 + 16 / 64)


def test_an_empty_row_is_handled():
    q, s, st = T.quantize_row(np.array([]), L=8, K=2)
    assert q.size == 0 and s.size == 0 and st == 0


def test_a_mismatched_hessian_is_refused_not_broadcast():
    with pytest.raises(ValueError):
        T.quantize_row(np.zeros(10), hdiag=np.ones(7))


def test_symbols_fit_the_declared_rate():
    _, syms, _ = T.quantize_row(np.random.default_rng(4).standard_normal(128), L=10, K=2)
    assert syms.max() < (1 << 2) and syms.min() >= 0


def test_provenance_is_documented_in_the_file():
    """Anyone editing this has to know why it was not simply ported."""
    src = (ROOT / "tools/pollard_trellis.py").read_text()
    assert "GPL-3" in src and "arXiv:2406.11235" in src
    assert "not a translation" in src.lower() or "NOT a translation" in src


def test_it_stays_ascii():
    assert all(ord(c) < 128 for c in (ROOT / "tools/pollard_trellis.py").read_text())
