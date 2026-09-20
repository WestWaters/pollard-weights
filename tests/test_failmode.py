"""Signal degradation vs computation collapse: repair fixes one and not the other.

Preconditioning migrates difficulty off the weights, which recovers error that ACCUMULATED. It
does not bring back a component that stopped working. Spending a full reconvert to discover that
is the expensive way to learn it, and a build that looks partially repaired is worse than one that
plainly failed.

The discriminator is cheap: per-layer agreement between the reference and the build. Degradation
climbs. Collapse is a cliff, and it happens early.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("pfm", ROOT / "tools/pollard_failmode.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pfm"] = m
    spec.loader.exec_module(m)
    return m


FM = _load()


def test_a_healthy_build_is_healthy():
    v = FM.classify([0.99 - 0.0005 * i for i in range(32)])
    assert v["mode"] == "healthy" and v["repairable"] is True


def test_gradual_erosion_is_signal_degradation():
    v = FM.classify([0.99 - 0.012 * i for i in range(32)])
    assert v["mode"] == "signal-degradation" and v["repairable"] is True


def test_an_early_cliff_is_computation_collapse():
    v = FM.classify([0.99, 0.98, 0.97, 0.35] + [0.30] * 28)
    assert v["mode"] == "computation-collapse" and v["repairable"] is False
    assert v["worst_drop_layer"] == 3


def test_a_late_cliff_is_not_collapse():
    """By then the model has already done most of its work -- error, not a broken component."""
    v = FM.classify([0.99] * 26 + [0.95, 0.60, 0.40, 0.35, 0.33, 0.30])
    assert v["mode"] == "signal-degradation"


def test_early_breakage_without_a_single_cliff_is_still_collapse():
    """Several small drops that destroy the signal early are a collapse, not erosion."""
    v = FM.classify([0.9, 0.82, 0.74, 0.66, 0.60] + [0.55] * 27)
    assert v["mode"] == "computation-collapse"


def test_no_layers_is_unknown_not_a_pass():
    v = FM.classify([])
    assert v["mode"] == "unknown" and v["repairable"] is None


def test_every_verdict_carries_a_next_step():
    """A verdict without a next step is half an answer."""
    for cos in ([0.99] * 8, [0.99 - 0.02 * i for i in range(32)], [0.9, 0.3] + [0.2] * 20, []):
        v = FM.classify(cos)
        assert v["advice"], v["mode"]
        assert all(isinstance(line, str) and line for line in v["advice"])


def test_collapse_advice_says_repair_will_not_help_and_what_will():
    v = FM.classify([0.99, 0.3] + [0.25] * 20)
    joined = " ".join(v["advice"]).lower()
    assert "not recover" in joined or "will not" in joined
    assert "pollard-errsrc" in joined, "should point at the tool that finds WHICH tensor broke"


def test_degradation_advice_points_at_the_levers():
    v = FM.classify([0.99 - 0.012 * i for i in range(32)])
    joined = " ".join(v["advice"]).lower()
    assert "precondition" in joined or "smooth" in joined or "rotate" in joined


def test_the_reason_is_specific_enough_to_check():
    v = FM.classify([0.99, 0.98, 0.97, 0.35] + [0.30] * 28)
    assert "layer 3" in v["reason"] and "%" in v["reason"]


def test_doctor_consults_the_classifier_before_repairing():
    src = (ROOT / "tools/pollard_doctor.py").read_text()
    assert "_failmode(" in src, "repair should classify first"
    assert "--force-repair" in src, "and the user should be able to override it"
    assert "will not recover this" in src.lower()


def test_doctor_falls_back_rather_than_blocking_on_the_diagnostic():
    """If the classifier cannot run, repair proceeds -- a diagnostic must not become a gate."""
    src = (ROOT / "tools/pollard_failmode.py").read_text()
    assert "best-effort" in (ROOT / "tools/pollard_doctor.py").read_text().lower()
    assert "does not refuse" in src.lower()
