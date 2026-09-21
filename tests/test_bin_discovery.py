"""Finding the binaries is Pollard's job, not the user's.

--bin defaulted to the RELATIVE path ik_llama.cpp\\build\\bin, so the same install worked from one
directory and failed from another with "could not list the tensors" -- and the backslash made the
default Windows-only. A tool that cannot find a binary sitting in an obvious place should look.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import pollard_automap as A  # noqa: E402

EXE = ".exe" if sys.platform == "win32" else ""


def _mk(d):
    d.mkdir(parents=True, exist_ok=True)
    f = d / ("llama-quantize" + EXE)
    f.write_text("")
    f.chmod(0o755)
    return d


def test_an_explicit_directory_wins(tmp_path):
    d = _mk(tmp_path / "mine" / "build" / "bin")
    assert A.find_ik_bin(str(d)) == str(d)


def test_the_environment_is_honoured(tmp_path, monkeypatch):
    d = _mk(tmp_path / "env" / "build" / "bin")
    monkeypatch.setenv("POLLARD_IK_BIN", str(d))
    assert A.find_ik_bin(None) == str(d)


def test_it_looks_beside_the_workspace(tmp_path, monkeypatch):
    """The real case that failed: POLLARD_HOME was C:\\pollard\\phome and the build was at
    C:\\pollard\\ik_llama.cpp\\build\\bin -- one level up, never searched."""
    monkeypatch.delenv("POLLARD_IK_BIN", raising=False)
    home = tmp_path / "pollard" / "phome"
    home.mkdir(parents=True)
    monkeypatch.setenv("POLLARD_HOME", str(home))
    d = _mk(tmp_path / "pollard" / "ik_llama.cpp" / "build" / "bin")
    assert A.find_ik_bin(None) == str(d)


def test_it_looks_in_the_workspaces_own_bin(tmp_path, monkeypatch):
    monkeypatch.delenv("POLLARD_IK_BIN", raising=False)
    home = tmp_path / "pollard"
    monkeypatch.setenv("POLLARD_HOME", str(home))
    d = _mk(home / "bin")
    assert A.find_ik_bin(None) == str(d)


def test_a_directory_without_the_binary_is_not_accepted(tmp_path, monkeypatch):
    """An empty ik_llama.cpp/build/bin must not shadow a real one elsewhere."""
    monkeypatch.delenv("POLLARD_IK_BIN", raising=False)
    home = tmp_path / "pollard"
    (home / "ik_llama.cpp" / "build" / "bin").mkdir(parents=True)   # exists, but empty
    monkeypatch.setenv("POLLARD_HOME", str(home))
    real = _mk(home / "bin")
    assert A.find_ik_bin(None) == str(real)


def test_the_default_is_no_longer_a_relative_windows_path():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "pollard_automap.py"), encoding="utf-8").read()
    assert r"ik_llama.cpp\build\bin" not in src, \
        "the relative default is back; it resolves against the current directory"


def test_nothing_found_returns_something_the_caller_can_report(tmp_path, monkeypatch):
    monkeypatch.delenv("POLLARD_IK_BIN", raising=False)
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "nowhere"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    got = A.find_ik_bin(None)
    assert isinstance(got, str)          # "" is fine; None would crash os.path.join downstream


# ── automap BUILDS; it does not hand back homework ──────────────────────────────────────────────

class _A:
    """Just enough of the parsed args for the planner."""
    def __init__(self, **kw):
        self.model, self.imatrix, self.eval = "m-f16.gguf", "m.imatrix", "e.txt"
        self.bin, self.log, self.ngl = "/opt/ik/bin", "automap.log", 0
        self.mix_only, self.no_eval, self.rival, self.gate = True, True, None, False
        self.body, self.protect, self.fragile = "iq1_s", "iq2_kt", None
        self.no_imatrix = False
        self.__dict__.update(kw)


def _names(nl=2):
    out = ["token_embd.weight", "output.weight", "output_norm.weight"]
    for i in range(nl):
        for t in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
            out.append(f"blk.{i}.{t}.weight")
    return out


def _plan(**kw):
    return A.emit_bat(_A(**kw), 2, False, _names())


def test_the_plan_is_argv_not_shell_text():
    """It was assembled as Windows batch -- @echo off, %BIN%\\llama-quantize.exe, backslashes --
    so automap on a Mac or Linux box wrote a file nothing could run."""
    plan = _plan()
    assert isinstance(plan, dict) and plan["steps"], "the plan is not a list of steps"
    for label, argv in plan["steps"]:
        assert isinstance(argv, list) and argv, f"{label} is not argv"
        assert not any("%" in str(x) for x in argv), f"{label} still carries a batch variable"


def test_it_renders_for_either_platform():
    plan = _plan()
    win, nix = A.render_script(plan, for_windows=True), A.render_script(plan, for_windows=False)
    assert win.startswith("@echo off") and "1>>" in win
    assert nix.startswith("#!/bin/sh") and ">>" in nix
    assert "@echo off" not in nix


def test_the_binary_name_matches_the_platform():
    argv = _plan()["steps"][0][1]
    exe = str(argv[0])
    assert exe.endswith(".exe") == (sys.platform == "win32"), \
        f"the quantizer name does not match this platform: {exe}"


def test_building_is_the_default_and_plan_only_is_the_opt_out():
    """automap exists to produce a model. Emitting a script and stopping left the user to re-run
    the thing they had already asked for."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "pollard_automap.py"), encoding="utf-8").read()
    assert "def run_plan(" in src, "automap cannot run what it plans"
    main = src[src.index("def main("):]
    assert "run_plan(plan)" in main, "main never runs the plan"
    assert "--plan-only" in src, "there is no way to just look at the plan"
    assert 'ap.add_argument("--out", default=""' in src, \
        "--out still defaults to writing a script nobody asked for"


def test_a_failed_step_stops_and_says_which(tmp_path, monkeypatch):
    """A build that fails halfway must not report success, and must name the step."""
    plan = {"header": "h", "log": str(tmp_path / "a.log"), "mix": "m.gguf",
            "steps": [("first", [sys.executable, "-c", "raise SystemExit(3)"]),
                      ("second", [sys.executable, "-c", "print('should not run')"])]}
    assert A.run_plan(plan) == 3
    blob = (tmp_path / "a.log").read_text()
    assert "first" in blob and "should not run" not in blob


def test_a_clean_run_reports_zero(tmp_path):
    plan = {"header": "h", "log": str(tmp_path / "b.log"), "mix": "m.gguf",
            "steps": [("only", [sys.executable, "-c", "print('ok')"])]}
    assert A.run_plan(plan) == 0
    assert "ok" in (tmp_path / "b.log").read_text()
