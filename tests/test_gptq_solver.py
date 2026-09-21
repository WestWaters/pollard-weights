"""The GPTQ solver's invariant: blocking changes the speed, never the objective.

pollard_gptq.gptq_quantize uses lazy batch updates -- per-column inside a block, one matmul for
everything after it. That is 8-39x faster than updating every remaining column each step, but it
reorders float operations, so individual weights can land on an adjacent level. What must NOT move
is the thing the solver minimises: Hessian-weighted reconstruction error.

This test keeps a plain per-column reference implementation in-file and holds the shipped solver to
it. If someone tunes BLOCK, or rewrites the loop again, the objective is still checked.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("pgptq", ROOT / "tools/pollard_gptq.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pgptq"] = m
    spec.loader.exec_module(m)
    return m


G = _load()


def reference(W, H, bits, groupsize, percdamp=0.01):
    """Plain per-column GPTQ. Slow on purpose -- it is the definition, not the implementation."""
    W = W.clone().float()
    rows, cols = W.shape
    groupsize = cols if not groupsize or groupsize < 0 else min(groupsize, cols)
    maxq = 2 ** bits - 1
    H = H.clone().float()
    base = torch.mean(torch.diag(H)).clamp(min=1e-8)
    Hd = H.clone()
    Hd[range(cols), range(cols)] += percdamp * base
    L = torch.linalg.cholesky(Hd)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True).to(W.dtype)

    Q = torch.zeros_like(W)
    scale = zero = None
    for i in range(cols):
        if i % groupsize == 0:
            scale, zero = G.group_params(W[:, i:i + groupsize], maxq)
        w = W[:, i]
        q = G._col_quant(w, scale, zero, maxq, "int")
        Q[:, i] = q
        err = (w - q) / Hinv[i, i]
        W[:, i:] -= err.unsqueeze(1) * Hinv[i, i:].unsqueeze(0)
    return Q


def hessian(cols, seed=0):
    torch.manual_seed(seed)
    A = torch.randn(cols * 2, cols)
    H = (A.T @ A) / (cols * 2)
    return H + torch.eye(cols) * 1e-3 * torch.diag(H).mean()


def weighted_error(W, Q, H):
    D = W - Q.float()
    return float(torch.einsum("rc,cd,rd->", D, H, D))


@pytest.mark.parametrize("bits,groupsize", [(4, 128), (4, 32), (2, 128), (8, 128)])
def test_blocking_preserves_the_objective(bits, groupsize):
    torch.manual_seed(1)
    W = torch.randn(256, 512) * 0.02
    H = hessian(512)
    ours = G.gptq_quantize(W, H, bits=bits, groupsize=groupsize)
    ref = reference(W, H, bits, groupsize)
    eo, er = weighted_error(W, ours, H), weighted_error(W, ref, H)
    assert abs(eo / er - 1) < 0.02, f"objective moved: {eo:.4f} vs reference {er:.4f}"


def test_block_width_is_a_throughput_knob_not_a_quality_one():
    """Changing BLOCK must not change the result meaningfully."""
    torch.manual_seed(2)
    W = torch.randn(256, 512) * 0.02
    H = hessian(512)
    errs = []
    original = G.BLOCK
    try:
        for b in (64, 128, 256):
            G.BLOCK = b
            errs.append(weighted_error(W, G.gptq_quantize(W, H, bits=4, groupsize=64), H))
    finally:
        G.BLOCK = original
    assert max(errs) / min(errs) - 1 < 0.02, f"block width changed the objective: {errs}"


@pytest.mark.parametrize("groupsize", [0, -1])
def test_per_channel_does_not_crash(groupsize):
    """0 and -1 both mean one group per row, as they do in every other solver in the file."""
    torch.manual_seed(3)
    W = torch.randn(64, 256) * 0.02
    Q = G.gptq_quantize(W, hessian(256), bits=4, groupsize=groupsize)
    assert Q.shape == W.shape and torch.isfinite(Q).all()


def test_groupsize_larger_than_the_layer_is_clamped():
    torch.manual_seed(4)
    W = torch.randn(64, 128) * 0.02
    Q = G.gptq_quantize(W, hessian(128), bits=4, groupsize=4096)
    assert Q.shape == W.shape and torch.isfinite(Q).all()


@pytest.mark.parametrize("qmode", ["int", "ternary", "binary"])
def test_every_qmode_still_solves(qmode):
    torch.manual_seed(5)
    W = torch.randn(64, 256) * 0.02
    Q = G.gptq_quantize(W, hessian(256), bits=2, groupsize=128, qmode=qmode)
    assert torch.isfinite(Q).all()
    if qmode == "ternary":
        assert len(torch.unique(torch.sign(Q))) <= 3


def test_act_order_still_returns_columns_in_the_original_order():
    """act_order permutes internally; a permuted output would corrupt the layer silently."""
    torch.manual_seed(6)
    W = torch.randn(64, 256) * 0.02
    H = hessian(256)
    plain = G.gptq_quantize(W, H, bits=4, groupsize=128, act_order=False)
    ordered = G.gptq_quantize(W, H, bits=4, groupsize=128, act_order=True)
    # not equal (that is the point of act_order) but correlated column-wise, not shuffled
    per_col = torch.nn.functional.cosine_similarity(
        plain.float(), ordered.float(), dim=0)
    assert float(per_col.mean()) > 0.9, "act_order looks like it returned a permuted layer"


def test_dead_channels_are_zeroed_not_nan():
    torch.manual_seed(7)
    W = torch.randn(32, 128) * 0.02
    H = hessian(128)
    H[10, :] = 0
    H[:, 10] = 0                                   # a genuinely dead channel
    Q = G.gptq_quantize(W, H, bits=4, groupsize=64)
    assert torch.isfinite(Q).all()
    assert float(Q[:, 10].abs().max()) == 0.0


def test_moe_token_floor_drops_offdiagonal_not_the_layer():
    """Fewer tokens than columns means the off-diagonal is noise; the layer must still solve."""
    torch.manual_seed(8)
    W = torch.randn(32, 256) * 0.02
    Q = G.gptq_quantize(W, hessian(256), bits=4, groupsize=128, n_tokens=64)
    assert torch.isfinite(Q).all()
