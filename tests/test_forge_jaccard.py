"""pollard-forge's weighted-Jaccard distillation loss (NVIDIA Nemotron-3.5-Bolt tech report section 5):
1 - sum(min(ps, pt)) / sum(max(ps, pt)) at T=1. Identity -> 0, bounded in [0, 1], finite gradients, and
equal to an independent reference (numpy, via the L1 identity J = 2*L1 / (2 + L1)). Needs torch."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_forge as G  # noqa: E402


def _logits(seed, shape=(2, 5, 11), scale=2.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def _ref(s, t):
    """numpy reference, written differently: sum(min) = (2 - L1)/2 and sum(max) = (2 + L1)/2 for two
    distributions, so the weighted Jaccard distance is 2*L1 / (2 + L1)."""
    def sm(x):
        x = x.astype(np.float64)
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    ps, pt = sm(s.numpy()).reshape(-1, s.shape[-1]), sm(t.numpy()).reshape(-1, t.shape[-1])
    l1 = np.abs(ps - pt).sum(-1)
    return float((2 * l1 / (2 + l1)).mean())


def test_identity_is_zero():
    s = _logits(0)
    assert float(G.kd_loss(s, s.clone(), 1.0, "jaccard")) == pytest.approx(0.0, abs=1e-7)
    assert float(G.kd_loss(s, s + 3.0, 1.0, "jaccard")) == pytest.approx(0.0, abs=1e-6), "shift-invariant logits"


def test_bounds_and_disjoint_supports():
    for seed in range(20):
        v = float(G.kd_loss(_logits(seed, scale=1 + seed), _logits(100 + seed, scale=1 + seed), 1.0, "jaccard"))
        assert 0.0 <= v <= 1.0
    a = torch.full((1, 1, 4), -80.0); a[..., 0] = 80.0
    b = torch.full((1, 1, 4), -80.0); b[..., 3] = 80.0
    assert float(G.kd_loss(a, b, 1.0, "jaccard")) == pytest.approx(1.0, abs=1e-6), "disjoint mass -> 1"


def test_matches_reference_and_ignores_temperature():
    for seed in range(5):
        s, t = _logits(2 * seed), _logits(2 * seed + 1)
        got = float(G.kd_loss(s, t, 1.0, "jaccard"))
        assert got == pytest.approx(_ref(s, t), abs=1e-6)
        assert float(G.kd_loss(s, t, 5.0, "jaccard")) == pytest.approx(got, abs=1e-7), "T=1 always (Bolt)"


def test_symmetric_and_padded_vocab():
    s, t = _logits(3), _logits(4, (2, 5, 13))
    assert float(G.kd_loss(s, t, 1.0, "jaccard")) == pytest.approx(_ref(s, t[..., :11]), abs=1e-6)
    assert float(G.kd_loss(s, t[..., :11], 1.0, "jaccard")) == pytest.approx(
        float(G.kd_loss(t[..., :11], s, 1.0, "jaccard")), abs=1e-7)


def test_gradient_finite_and_descends():
    t = _logits(5)
    s = _logits(6).requires_grad_(True)
    opt = torch.optim.SGD([s], lr=50.0)
    first = None
    for _ in range(30):
        opt.zero_grad()
        loss = G.kd_loss(s, t, 1.0, "jaccard")
        loss.backward()
        assert torch.isfinite(s.grad).all()
        first = float(loss) if first is None else first
        opt.step()
    assert float(G.kd_loss(s, t, 1.0, "jaccard")) < first


def test_cli_accepts_jaccard():
    assert G._loss_name("Jaccard") == "jaccard" and "jaccard" in G.LOSSES
