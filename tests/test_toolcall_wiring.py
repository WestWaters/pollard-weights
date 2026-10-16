"""The tool-call gate wired in: an AGENTIC model (pollard-modelkind finds a tool-call syntax in its template) gets
pollard-toolcall by default in `pollard --ladder --run` and `pollard release`, and pollard-verify --gate holds its rungs
to a tool-call retention threshold (gates.yaml `kinds: agentic:`). A model with no tool syntax is untouched.
Fakes only: no model, no llama.cpp."""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_auto as pa  # noqa: E402
import pollard_modelkind as MK  # noqa: E402
import pollard_release as R  # noqa: E402
import pollard_verify as V  # noqa: E402
import pollard_workspace as ws  # noqa: E402


def _kind(family):
    return lambda m: {"tools": {"family": family}, "thinking": False, "decision": False, "vision": False}


# ------------------------------------------------------------------ verify: kinds + retention threshold

def test_agentic_rungs_get_a_retention_threshold_by_bit_width():
    g = V.load_gates(None)
    assert "toolcall_retention_min" not in V.thresholds_for(g, "Q4_K_M", "gguf")          # not agentic: untouched
    assert V.thresholds_for(g, "Q6_K", "gguf", ["agentic"])["toolcall_retention_min"] == 90
    assert V.thresholds_for(g, "IQ3_S", "gguf", ["agentic"])["toolcall_retention_min"] == 85
    assert V.thresholds_for(g, "IQ2_XXS", "gguf", ["agentic"])["toolcall_retention_min"] == 75
    # the per-rung KLD thresholds are still there underneath
    assert V.thresholds_for(g, "IQ3_S", "gguf", ["agentic"])["kld_max"] == 0.2


def test_retention_gate_passes_fails_and_fails_when_unmeasured():
    th = {"toolcall_retention_min": 90}
    assert V.evaluate_gate({"toolcall_retention": 93.0}, th)["pass"] is True
    assert V.evaluate_gate({"toolcall_retention": 80.0}, th)["pass"] is False
    res = V.evaluate_gate({}, th, "fail")
    assert res["pass"] is False and res["checks"][0]["why"] == "not measured"


def test_metrics_read_full_and_quick_toolcall_results():
    full = {"r": {"toolcall": {"pass": True, "rate": 0.5, "ref_rate": 0.55, "retention": 90.9}}}
    assert V.metrics_from_results(full, "r")["toolcall_retention"] == 90.9
    quick = {"r": {"toolcall": {"pass": True, "rate": 0.75, "ref_rate": 1.0}}}
    assert V.metrics_from_results(quick, "r")["toolcall_retention"] == 75.0


def test_model_kinds_reads_the_tool_family(monkeypatch):
    monkeypatch.setattr(MK, "classify", _kind("hermes"))
    assert V.model_kinds("x.gguf") == ["agentic"]
    monkeypatch.setattr(MK, "classify", _kind("none"))
    assert V.model_kinds("x.gguf") == []

    def boom(m):
        raise SystemExit("not a GGUF")
    monkeypatch.setattr(MK, "classify", boom)
    assert V.model_kinds("x.gguf") == []


def test_shipped_gates_yaml_matches_the_builtin():
    assert (ROOT / "recipes" / "gates.yaml").read_text(encoding="utf-8") == V.DEFAULT_GATES


# ------------------------------------------------------------------ release: toolcall auto

def test_release_toolcall_auto_follows_the_template(monkeypatch, tmp_path):
    f16 = tmp_path / "m-f16.gguf"
    f16.write_bytes(b"GGUF")
    c = {"f16": str(f16), "hf_dir": str(tmp_path / "nope")}
    cfg = R.load_plan()
    assert cfg["measure"]["toolcall"] == "auto"
    monkeypatch.setattr(MK, "classify", _kind("qwen3_coder"))
    assert R.toolcall_wanted(cfg, c) is True
    monkeypatch.setattr(MK, "classify", _kind("none"))
    assert R.toolcall_wanted(cfg, c) is False
    cfg["measure"]["toolcall"] = True
    assert R.toolcall_wanted(cfg, c) is True
    assert R.toolcall_wanted(R.load_plan(preset="decision"), c) is False


def test_release_adds_the_toolcall_step_for_an_agentic_model(monkeypatch, tmp_path):
    f16 = tmp_path / "m-f16.gguf"
    f16.write_bytes(b"GGUF")
    c = {"f16": str(f16), "hf_dir": str(tmp_path), "mdir": str(tmp_path), "ngl": "0", "heldout": "h.txt",
         "gates": None}
    builds = [{"tag": "Q4_K_M", "path": str(tmp_path / "m-Q4_K_M.gguf")}]
    monkeypatch.setattr(MK, "classify", _kind("hermes"))
    ids = [s.id for s in R.gguf_rung_steps(R.load_plan(), c, builds)]
    assert "gguf/Q4_K_M/toolcall" in ids and ids.index("gguf/Q4_K_M/toolcall") < ids.index("gguf/Q4_K_M/verify")
    monkeypatch.setattr(MK, "classify", _kind("none"))
    assert "gguf/Q4_K_M/toolcall" not in [s.id for s in R.gguf_rung_steps(R.load_plan(), c, builds)]


# ------------------------------------------------------------------ pollard --run: the gate after the build

def _auto_args(tmp_path, **kw):
    f16 = tmp_path / "m-f16.gguf"
    f16.write_bytes(b"GGUF")
    a = types.SimpleNamespace(run=True, toolcall=True, gguf=str(f16), ngl="auto", results=None)
    a.__dict__.update(kw)
    return a


def test_auto_runs_the_gate_on_this_runs_rungs_only(monkeypatch, tmp_path):
    a = _auto_args(tmp_path)
    new, old = tmp_path / "m-Q4.gguf", tmp_path / "m-old.gguf"
    new.write_bytes(b"x")
    old.write_bytes(b"x")
    os.utime(old, (1_000_000, 1_000_000))                     # built long before this run
    monkeypatch.setattr(MK, "classify", _kind("hermes"))
    monkeypatch.setattr(ws, "read_manifest", lambda m: {"builds": [
        {"lane": "gguf", "name": "m-Q4.gguf", "path": str(new)},
        {"lane": "gguf", "name": "m-old.gguf", "path": str(old)}]})
    calls = []
    monkeypatch.setattr(pa.subprocess, "run", lambda cmd, **k: calls.append(cmd) or types.SimpleNamespace(returncode=1))
    pa._toolcall_gate(a, since=os.path.getmtime(new))
    assert len(calls) == 1
    cmd = calls[0]
    assert "--gguf" in cmd and str(new) in cmd and cmd[cmd.index("--ref") + 1] == a.gguf
    assert cmd[cmd.index("--ngl") + 1] == "0"                  # 'auto' never becomes a GPU claim here
    assert a.results and cmd[cmd.index("--results") + 1] == a.results


def test_auto_skips_a_model_with_no_tool_syntax_and_honours_no_toolcall(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pa.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    monkeypatch.setattr(ws, "read_manifest", lambda m: {"builds": [{"lane": "gguf", "path": __file__}]})
    monkeypatch.setattr(MK, "classify", _kind("none"))
    pa._toolcall_gate(_auto_args(tmp_path), since=0)
    monkeypatch.setattr(MK, "classify", _kind("hermes"))
    pa._toolcall_gate(_auto_args(tmp_path, toolcall=False), since=0)
    pa._toolcall_gate(_auto_args(tmp_path, run=False), since=0)
    assert calls == []
