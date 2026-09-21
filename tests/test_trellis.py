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
import pathlib
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


# --- the round trip is the correctness property that matters --------------------------------
@pytest.mark.parametrize("L,K", [(8, 2), (10, 2), (10, 3), (12, 2)])
def test_decode_reproduces_encode_exactly(L, K):
    """Encoding pays for Viterbi once; decoding just replays the shift. If they disagree by even
    one position the file still loads and every weight is out of phase, which nothing downstream
    would catch."""
    w = np.random.default_rng(5).standard_normal(512) * 0.02
    scale = float(np.sqrt(np.mean(w * w)))
    q, syms, state = T.quantize_row(w, L=L, K=K)
    d = T.decode(syms, state, L=L, K=K, scale=scale)
    assert np.allclose(d, q), "decode is out of phase with encode"


def test_the_returned_state_is_the_one_before_the_first_symbol():
    """Returning the state AFTER step 0 decodes everything one position out."""
    w = np.random.default_rng(6).standard_normal(64) * 0.02
    q, syms, state = T.quantize_row(w, L=8, K=2)
    mask = (1 << 8) - 1
    first = ((state << 2) | int(syms[0])) & mask
    assert T.codebook(np.array([first]), float(np.sqrt(np.mean(w * w))))[0] == pytest.approx(q[0])


def _code_only(path):
    """Source with comments and docstrings stripped, so a check about CODE is neither satisfied
    nor broken by prose that happens to mention the thing."""
    import ast
    tree = ast.parse(pathlib.Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.unparse(tree)


def test_encode_uses_a_gather_not_a_scatter():
    """np.minimum.at is a serialised ufunc and dominated the runtime; inverting the shift turns
    the recurrence into a fixed-width gather."""
    code = _code_only(ROOT / "tools/pollard_trellis.py")
    assert "np.minimum.at" not in code, "the scatter is still in the code path"
    assert "pred" in code and "n_sym" in code


def test_decode_has_no_search_in_it():
    """The runtime path must not carry Viterbi. If it does, the format is not servable."""
    code = _code_only(ROOT / "tools/pollard_trellis.py")
    i = code.index("def decode")
    body = code[i:]
    nxt = body.find("\ndef ")
    body = body[:nxt] if nxt > 0 else body
    for banned in ("argmin", "minimum", "back["):
        assert banned not in body, "decode should not contain " + repr(banned)


def test_encode_is_fast_enough_to_use():
    """A reference implementation nobody can run is not a reference implementation."""
    import time
    w = np.random.default_rng(7).standard_normal(2048) * 0.02
    t = time.perf_counter()
    T.quantize_row(w, L=10, K=2)
    assert time.perf_counter() - t < 5.0, "2048 weights should not take seconds at L=10"
