"""A rung whose tool calls WERE measured but whose reference could not be served on this machine (a bf16 bigger than
RAM) is gated on its absolute tool-call verdict -- what pollard-toolcall itself reports -- not failed as "missing"."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_verify as V  # noqa: E402

NOTE = "llama-server --jinja reference not measured: 15.1GB model needs ~18.3GB of RAM, 10.8GB free"


def _results(passed, retention=None, status="measured", note=NOTE):
    tc = {"status": status, "pass": passed, "rate": 1.0 if passed else 0.4, "n": 60}
    if note:
        tc["ref_note"] = note
    if retention is not None:
        tc["retention"] = retention
    return {"Q6_K": {"toolcall": tc, "mean_kld": 0.013, "top1": 97.1}}


def _row(gate, name="tool-call retention %"):
    return next(c for c in gate["checks"] if c["check"] == name)


def test_ref_not_measured_uses_the_absolute_verdict():
    m = V.metrics_from_results(_results(True), "Q6_K")
    g = V.evaluate_gate(m, {"toolcall_retention_min": 90}, missing="fail")
    r = _row(g)
    assert g["pass"] and r["ok"] and "reference not measured" in r["why"] and "PASS" in r["why"]


def test_ref_not_measured_and_rung_failing_still_fails():
    m = V.metrics_from_results(_results(False), "Q6_K")
    g = V.evaluate_gate(m, {"toolcall_retention_min": 90}, missing="fail")
    assert not g["pass"] and "FAIL" in _row(g)["why"]


def test_a_measured_retention_is_still_the_number():
    m = V.metrics_from_results(_results(True, retention=80.0), "Q6_K")
    assert not V.evaluate_gate(m, {"toolcall_retention_min": 90}, missing="fail")["pass"]


def test_no_toolcall_record_at_all_is_still_missing():
    m = V.metrics_from_results({"Q6_K": {"mean_kld": 0.01}}, "Q6_K")
    assert not V.evaluate_gate(m, {"toolcall_retention_min": 90}, missing="fail")["pass"]
