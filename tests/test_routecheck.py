"""Routing-consistency metrics for MoE quantization.

A MoE has no fixed computation graph: the router picks a subset of experts per token. If
quantization changes that choice, the model runs different weights than the ones that were
measured, and perplexity absorbs a surprising amount of it.

The subtlety these tests pin down: matching router LOGITS is not sufficient. Top-k depends only on
ORDER, so a shift too small to move a loss can still swap an expert at the selection boundary.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("prc", ROOT / "tools/pollard_routecheck.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["prc"] = m
    spec.loader.exec_module(m)
    return m


RC = _load()


def logits(seed=0, tokens=500, experts=8):
    return np.random.default_rng(seed).normal(size=(tokens, experts))


def test_identical_routers_swap_nothing():
    ref = logits()
    st = RC.route_stats(ref, ref.copy(), top_k=2)
    assert st["top1_swap"] == 0.0 and st["topk_swap"] == 0.0
    assert st["experts_changed_per_token"] == 0.0


def test_noise_swaps_more_experts_as_it_grows():
    ref = logits()
    rng = np.random.default_rng(1)
    swaps = [RC.route_stats(ref, ref + rng.normal(scale=s, size=ref.shape), 2)["topk_swap"]
             for s in (0.001, 0.05, 0.5)]
    assert swaps[0] < swaps[1] < swaps[2], swaps


def test_a_uniform_shift_changes_no_routing():
    """Top-k is order-only, so adding a constant to every logit must swap nothing."""
    ref = logits()
    st = RC.route_stats(ref, ref + 7.5, top_k=2)
    assert st["topk_swap"] == 0.0


def test_scaling_all_logits_changes_no_routing():
    ref = logits()
    assert RC.route_stats(ref, ref * 3.0, top_k=2)["topk_swap"] == 0.0


def test_reordering_inside_the_selection_is_not_a_swap():
    """Swapping rank 1 and rank 2 keeps the same experts running. It is not a routing change."""
    ref = np.array([[5.0, 4.0, 1.0, 0.0]])
    qnt = np.array([[4.0, 5.0, 1.0, 0.0]])          # top-2 set identical, order flipped
    st = RC.route_stats(ref, qnt, top_k=2)
    assert st["topk_swap"] == 0.0
    assert st["top1_swap"] == 1.0                    # but the top-1 did change, and that is reported


def test_a_boundary_flip_is_caught_even_though_the_logits_barely_moved():
    """The failure this tool exists for: a tiny shift at the k/k+1 boundary swaps an expert."""
    ref = np.array([[5.0, 4.0, 3.99, 0.0]])
    qnt = np.array([[5.0, 3.99, 4.0, 0.0]])          # 0.01 apart; expert 1 replaced by expert 2
    st = RC.route_stats(ref, qnt, top_k=2)
    assert st["topk_swap"] == 1.0
    assert st["experts_changed_per_token"] == 1.0
    assert abs(ref - qnt).max() <= 0.011, "the point is that the logits hardly moved"


def test_margin_is_measured_at_the_selection_boundary():
    ref = np.array([[5.0, 4.0, 1.0, 0.0]])           # boundary gap = 4.0 - 1.0 = 3.0
    st = RC.route_stats(ref, ref.copy(), top_k=2)
    assert st["margin_median"] == pytest.approx(3.0)


def test_margin_drop_reports_erosion_not_improvement():
    ref = np.array([[5.0, 4.0, 1.0, 0.0]])
    qnt = np.array([[5.0, 4.0, 3.0, 0.0]])           # the runner-up closed in: margin 3.0 -> 1.0
    st = RC.route_stats(ref, qnt, top_k=2)
    assert st["margin_drop"] == pytest.approx(2.0)


def test_at_risk_counts_tokens_sitting_on_the_boundary():
    tight = np.tile(np.array([[5.0, 4.0, 3.999, 0.0]]), (100, 1))
    loose = np.tile(np.array([[5.0, 4.0, 0.0, -1.0]]), (100, 1))
    assert RC.route_stats(tight, tight.copy(), 2)["at_risk"] == 1.0
    assert RC.route_stats(loose, loose.copy(), 2)["at_risk"] == 0.0


def test_top_k_is_clamped_to_the_expert_count():
    ref = logits(experts=4)
    st = RC.route_stats(ref, ref.copy(), top_k=99)
    assert st["top_k"] == 4 and st["topk_swap"] == 0.0


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError):
        RC.route_stats(logits(tokens=10), logits(tokens=11), top_k=2)


# ── the gate ────────────────────────────────────────────────────────────────────────────────────
def test_verdict_passes_when_every_layer_is_inside_budget():
    rows = [{"layer": f"l{i}", "topk_swap": 0.001} for i in range(4)]
    v = RC.verdict(rows, swap_budget=0.02)
    assert v["pass"] and v["over_budget"] == []


def test_verdict_names_the_layers_that_failed():
    rows = [{"layer": "l0", "topk_swap": 0.001}, {"layer": "l1", "topk_swap": 0.4}]
    v = RC.verdict(rows, swap_budget=0.02)
    assert not v["pass"] and v["over_budget"] == ["l1"] and v["worst_layer"] == "l1"


def test_no_layers_is_a_failure_not_a_pass():
    """A dense model measuring nothing must never read as a clean bill of health."""
    assert RC.verdict([], 0.02)["pass"] is False
