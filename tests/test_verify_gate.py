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


def test_exl3_lane_is_gated_on_its_kld_and_the_forward():
    """pollard-verify measures an EXL3 rung's KLD through exllamav3 now (measure_exl3, the shared protocol): the lane
    takes the same per-bit-width KLD / top-1 thresholds as every other, plus the end-to-end forward check."""
    th = V.thresholds_for(V.load_gates(None), "M-Pollard-EXL3-4.0bpw", "exl3")
    assert th["kld_max"] == 0.08 and th["top1_min"] == 90 and th["e2e_acc_min"] == 0.3
    assert V.evaluate_gate({"e2e_acc": 0.6, "kld": 0.031, "top1": 91.9}, th)["pass"]
    assert not V.evaluate_gate({"e2e_acc": 0.6}, th)["pass"], "an unmeasured KLD is not a verified rung"


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


# ------------------------------------------------- quality numbers are report-only by default (owner, 2026-10-25)

def _gate_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["pollard-verify", *args])
    try:
        V.main()
    except SystemExit as e:
        return e.code
    return 0


def test_a_rung_over_the_kld_band_publishes_by_default_and_blocks_under_strict(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    key, q4 = _manifest_build(tmp_path, "IQ4_XS")
    res = tmp_path / "results.json"
    # IQ4_XS's band: KLD <= 0.08, top-1 >= 90 -- this rung misses both, tool calls too; it loads, is coherent, stock
    ws.merge_results(str(res), "IQ4_XS", {"kld": 0.31, "top1": 81.0, "coherence": "PASS",
                                          "toolcall": {"status": "measured", "pass": False, "retention": 40.0}})
    monkeypatch.setattr(V, "gguf_runtime", lambda p: "stock")
    common = ["--model", q4, "--rung", "IQ4_XS", "--results", str(res), "--no-measure"]
    assert _gate_cli(monkeypatch, *common, "--gate") == 0, "report-only: measured, carded, not blocking"
    b = ws.read_manifest(key)["builds"][0]
    assert b["verified"] is True and b["gate"]["strict"] is False
    v = json.load(open(res))["IQ4_XS"]["verified"]
    assert v["pass"] is True and v["failed"] == []
    assert {"mean KLD", "top-1 agreement %", "tool-call retention %"} <= set(v["report_only_failed"])
    assert "report-only, does not block" in capsys.readouterr().out
    cell = C.verified_cell(dict(b, name="M-Pollard-IQ4_XS.gguf"), {})
    assert cell.startswith(" yes (below band:") and "mean KLD" in cell, "the card keeps the pass / fail label"

    assert _gate_cli(monkeypatch, *common, "--gate", "--strict") == 1, "--strict: the thresholds block"
    b = ws.read_manifest(key)["builds"][0]
    assert b["verified"] is False
    assert "mean KLD" in json.load(open(res))["IQ4_XS"]["verified"]["failed"]

    strict = tmp_path / "strict-gates.yaml"
    strict.write_text(V.DEFAULT_GATES.replace("\nstrict: false\n", "\nstrict: true\n"))
    assert _gate_cli(monkeypatch, *common, "--gate", str(strict)) == 1, "strict: true in the gates file blocks too"
    assert ws.read_manifest(key)["builds"][0]["verified"] is False


def test_what_still_blocks_by_default_complete_coherence_runtime(tmp_path, monkeypatch):
    g = V.load_gates(None)
    ro, strict = V.gate_policy(g)
    assert strict is False and set(ro) == {"kld_max", "top1_min", "toolcall_retention_min", "toolcall_pass",
                                           "retention_min", "e2e_acc_min"}
    th = V.thresholds_for(g, "IQ4_XS", "gguf")
    good = {"kld": 0.5, "top1": 70.0, "coherence": "PASS", "runtime": "stock"}
    assert V.evaluate_gate(good, th, report_only=ro)["pass"]
    assert not V.evaluate_gate(dict(good, complete=False, complete_missing=["mmproj"]), th, report_only=ro)["pass"], \
        "an incomplete model still blocks"
    assert not V.evaluate_gate(dict(good, coherence="FAIL"), th, report_only=ro)["pass"]
    assert not V.evaluate_gate(dict(good, runtime="ik_llama"), th, report_only=ro)["pass"]
    assert not V.evaluate_gate(good, th, report_only=ro, strict=True)["pass"]
    exl3 = V.thresholds_for(g, "EXL3-4.0bpw", "exl3")
    assert V.evaluate_gate({"e2e_acc": 0.1, "kld": 0.9, "top1": 50.0}, exl3, report_only=ro)["pass"], \
        "EXL3's end-to-end agreement is report-only too"


def test_an_incomplete_model_still_blocks_at_the_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    src, out = tmp_path / "M", tmp_path / "M-Pollard-MLX-4bit"
    src.mkdir(); out.mkdir()
    res = tmp_path / "results.json"
    ws.merge_results(str(res), "MLX-4bit", {"kld": 0.01, "top1": 99.0})
    monkeypatch.setattr(V, "complete_check", lambda m, s, lane: {"ok": False, "missing": ["vision tower"]})
    code = _gate_cli(monkeypatch, "--lane", "mlx", "--model", str(out), "--source", str(src), "--rung", "MLX-4bit",
                     "--results", str(res), "--no-measure", "--gate")
    assert code == 1
    v = json.load(open(res))["MLX-4bit"]["verified"]
    assert v["pass"] is False and v["failed"] == ["complete model"]
