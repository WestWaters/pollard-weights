"""pollard-verify --gate: per-rung thresholds decide whether a rung may be published, every lane is measured
on the same KLD/top-1 scale, an unmeasured threshold fails, and the verdict reaches the MANIFEST, the
results file and the card. Fakes only: no model, no llama.cpp, no torch."""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_card as C  # noqa: E402
import pollard_verify as V  # noqa: E402
import pollard_workspace as ws  # noqa: E402


def test_kld_top1_matches_definition():
    r = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 0.0]])
    same = V.kld_top1(r, r)
    assert same["kld"] == pytest.approx(0.0, abs=1e-12) and same["top1"] == 100.0
    q = np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 0.0]])
    out = V.kld_top1(r, q)
    p = np.exp(r[0]) / np.exp(r[0]).sum(); pq = np.exp(q[0]) / np.exp(q[0]).sum()
    assert out["kld"] == pytest.approx((p * np.log(p / pq)).sum() / 2)
    assert out["top1"] == 50.0
    # a padded vocab on one side does not break it
    assert V.kld_top1(np.c_[r, [9, 9]], r)["n"] == 2


def test_default_gates_parse_and_resolve_per_rung():
    g = V.load_gates(None)
    assert g["missing"] == "fail"
    q6 = V.thresholds_for(g, "Q6_K", "gguf")
    assert q6["kld_max"] == 0.02 and q6["runtime"] == "stock" and q6["coherence"] is True
    kt = V.thresholds_for(g, "IQ1_KT", "gguf")
    assert kt["runtime"] == "ik_llama"
    mlx = V.thresholds_for(g, "MLX-4bit", "mlx")
    assert mlx["kld_max"] == 0.08 and "runtime" not in mlx
    assert V.thresholds_for(g, "MX-FP8", "mx")["top1_min"] == 97
    assert V.thresholds_for(g, "GPTQ-int2", "gptq")["kld_max"] == 0.5


def test_gate_fails_unmeasured_thresholds_and_wrong_runtime():
    th = {"kld_max": 0.08, "top1_min": 90, "retention_min": 95, "toolcall_pass": True, "runtime": "stock"}
    res = V.evaluate_gate({"kld": 0.05, "top1": 92.0, "runtime": "ik_llama", "toolcall_pass": True}, th)
    bad = {c["check"] for c in res["checks"] if not c["ok"]}
    assert not res["pass"] and bad == {"task retention %", "runtime"}
    # the IQ4_XS-that-is-really-ik_llama case: named stock, needs ik -> blocked
    assert any("card and the runtime would disagree" in c["why"] for c in res["checks"])
    relaxed = V.evaluate_gate({"kld": 0.05, "top1": 92.0, "runtime": "stock", "toolcall_pass": True}, th, "skip")
    assert relaxed["pass"]
    assert not V.evaluate_gate({"kld": 0.2}, {"kld_max": 0.08})["pass"]
    assert V.evaluate_gate({"runtime": "stock"}, {"runtime": "ik_llama"})["pass"], "stock also runs in ik"


def test_metrics_from_results_speak_the_gate_vocabulary():
    res = {"Q4": {"kld": 0.07, "top1": 91.0, "retention": {"overall": 97.5},
                  "mt": {"a": {"retention": 99.0}, "b": {"retention": 97.0}},
                  "toolcall": {"pass": True}, "decision": {"agreement": 0.95}}}
    m = V.metrics_from_results(res, "Q4")
    assert m == {"kld": 0.07, "top1": 91.0, "retention": 97.5, "mt_retention": 98.0,
                 "toolcall_pass": True, "decision_agreement": 95.0}


FAKE_PPL = r'''import sys
a = sys.argv[1:]
base = a[a.index("--kl-divergence-base") + 1]
if "--kl-divergence" not in a:
    open(base, "wb").write(b"logits")
    print("saved base")
else:
    m = a[a.index("-m") + 1]
    kld = "0.150000" if "IQ2" in m else "0.012000"
    print("====== Perplexity statistics ======")
    print("Mean PPL(Q)                   :  10.4142 +/-   0.06733")
    print("Mean PPL(base)                :  10.3478 +/-   0.06657")
    print("====== KL divergence statistics ======")
    print(f"Mean    KLD:   {kld} +/-   0.000104")
    print("Median  KLD:   0.004000")
    print("Same top p: 96.533 +/- 0.047 %")
'''


def _fake_bin(tmp_path):
    py = tmp_path / "fake_ppl.py"
    py.write_text(FAKE_PPL)
    sh = tmp_path / "llama-perplexity"
    sh.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{py}' \"$@\"\n")
    sh.chmod(sh.stat().st_mode | stat.S_IEXEC)
    return str(sh)


@pytest.mark.skipif(os.name == "nt", reason="shell-script fake binary")
def test_gguf_lane_measures_with_llama_perplexity_and_caches_the_base(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    ev = tmp_path / "held.txt"; ev.write_text("text " * 100)
    binp = _fake_bin(tmp_path)
    m = V.measure_gguf(str(tmp_path / "M-Q6_K.gguf"), str(tmp_path / "M-f16.gguf"), str(ev), "0", 4, binp)
    assert m["kld"] == 0.012 and m["top1"] == 96.533 and m["ppl"] == 10.4142
    bases = list((tmp_path / "home" / "cache").glob("*.klbase.dat"))
    assert len(bases) == 1, "one base per reference, reused by every rung"


def _manifest_build(tmp_path, tag):
    key = str(tmp_path / "Org__M")
    p = tmp_path / f"M-Pollard-{tag}.gguf"
    p.write_bytes(b"x")
    ws.record_build(key, "gguf", str(p), tag=tag)
    return key, str(p)


def test_gate_cli_blocks_publish_stamps_manifest_results_and_card(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    key, q4 = _manifest_build(tmp_path, "IQ4_XS")
    res = tmp_path / "results.json"
    ws.merge_results(str(res), "IQ4_XS", {"kld": 0.05, "top1": 91.0, "coherence": "PASS"})
    monkeypatch.setattr(V, "gguf_runtime", lambda p: "ik_llama")      # attn_v got IQ4_K: ik-only
    monkeypatch.setattr(sys, "argv", ["pollard-verify", "--model", q4, "--gate", "--rung", "IQ4_XS",
                                      "--results", str(res), "--no-measure"])
    with pytest.raises(SystemExit) as e:
        V.main()
    assert e.value.code == 1, "a failed gate must exit non-zero so release never uploads it"
    b = ws.read_manifest(key)["builds"][0]
    assert b["verified"] is False and b["gate"]["rung"] == "IQ4_XS"
    assert json.load(open(res))["IQ4_XS"]["verified"]["failed"] == ["runtime"]

    monkeypatch.setattr(V, "gguf_runtime", lambda p: "stock")
    V.main()                                                            # passes: no SystemExit
    b = ws.read_manifest(key)["builds"][0]
    assert b["verified"] is True and b["gate"]["pass"] is True
    builds = [dict(b, name="M-Pollard-IQ4_XS.gguf")]
    assert C.verified_cell(builds[0], {}) == " yes |"
    assert C.gate_verdict({"name": "x"}, {"x": {"verified": {"pass": False}}}) is False
    assert C.gate_verdict({"name": "x"}, {}) is None


def test_shipped_gates_file_is_the_builtin_default():
    shipped = ROOT / "recipes" / "gates.yaml"
    if not shipped.exists():                       # the nix check builds from .py files only
        pytest.skip("recipes/ not in this source tree")
    assert shipped.read_text(encoding="utf-8") == V.DEFAULT_GATES, \
        "recipes/gates.yaml drifted: regenerate with pollard-verify --print-gates > recipes/gates.yaml"


def test_exl3_lane_is_gated_on_the_forward_not_on_a_kld_it_cannot_measure():
    th = V.thresholds_for(V.load_gates(None), "M-Pollard-EXL3-4.0bpw", "exl3")
    assert "kld_max" not in th and "top1_min" not in th and th["e2e_acc_min"] == 0.3
    assert V.evaluate_gate({"e2e_acc": 0.6}, th)["pass"]


def test_tool_calls_not_measured_publish_by_default_and_fail_stays_available():
    """Owner decision (2026-10-16): an agentic rung whose tool-call gate is NOT MEASURED (no vLLM on Windows, too big
    for in-process) publishes, and the card says 'not measured (<reason>)'. `fail` is still there to hold it back."""
    import pollard_verify as V
    import pollard_card as C
    g = V.load_gates(None)
    assert g["default"]["toolcall_not_measured"] == "skip"
    th = V.thresholds_for(g, "int4", lane="gptq", kinds=("agentic",))
    assert th.get("toolcall_retention_min") == 90 and th.get("toolcall_not_measured") == "skip"
    m = {"kld": 0.01, "top1": 99.0, "toolcall_status": "not-measured", "toolcall_reason": "vLLM is not installed here"}
    res = V.evaluate_gate(m, th, missing=g.get("missing", "fail"))
    tc = [r for r in res["checks"] if "tool" in r["check"]]
    assert res["pass"] is True and tc and all(r["ok"] and "not measured: vLLM" in r["why"] for r in tc)
    held = V.evaluate_gate(m, dict(th, toolcall_not_measured="fail"), missing="fail")
    assert held["pass"] is False
    assert C.tool_ready_cell({"status": "not-measured", "reason": "vLLM is not installed here"}) == \
        "not measured (vLLM is not installed here)"
    shipped = (Path(V.__file__).resolve().parents[1] / "recipes" / "gates.yaml")
    if shipped.exists():
        assert V.load_gates(str(shipped))["default"]["toolcall_not_measured"] == "skip"
