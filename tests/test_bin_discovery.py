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


# ── an uncovered tensor is repaired, not reported ───────────────────────────────────────────────

def test_the_mtp_head_is_known_to_need_an_imatrix():
    """nextn.eh_proj is a real matmul and imatrix runs routinely skip it, so a low-bit build
    reached it unpinned and llama-quantize bailed -- after an hour of quantizing. It is also the
    tensor pollard-fragile ranks MOST heavy-tailed on this family."""
    for n in ("blk.64.nextn.eh_proj.weight", "blk.64.nextn.shared_head_head.weight"):
        assert A._NEEDS_IMATRIX.search(n), f"{n} would not be pinned"
    assert not A._NEEDS_IMATRIX.search("blk.64.nextn.enorm.weight"), "a norm is not a matmul"


FAKE_QUANTIZER = """import sys, pathlib
# argv[-1], not argv[1]: the pin is inserted just after argv[0] and shifts everything along
p = pathlib.Path(sys.argv[-1])
p.write_text(str(int(p.read_text() or 0) + 1))
if "--custom-q" in sys.argv and "eh_proj" in sys.argv[sys.argv.index("--custom-q") + 1]:
    print("ok")
    sys.exit(0)
print("Missing importance matrix for tensor blk.64.nextn.eh_proj.weight in a very low-bit quantization")
sys.exit(1)
"""

ALWAYS_FAILS = """import sys
print("Missing importance matrix for tensor blk.1.ffn_up.weight in a very low-bit quantization")
sys.exit(1)
"""


def _fake_quantizer(tmp_path, body, name="fake-quantize"):
    """An executable stand-in for llama-quantize, so argv[0] is the BINARY the way it is in a
    real run -- which is what makes the inserted pin land in the right position."""
    f = tmp_path / name
    f.write_text("#!" + sys.executable + "\n" + body)
    f.chmod(0o755)
    return f


@pytest.mark.skipif(sys.platform == "win32", reason="needs an executable shebang")
def test_a_missing_imatrix_tensor_is_pinned_and_the_build_retried(tmp_path):
    """The pin list is built from a pattern, and a pattern is the tensors someone thought of.
    llama-quantize names the tensor it wanted, which is enough to pin it and carry on."""
    marker = tmp_path / "attempts"
    marker.write_text("0")
    q = _fake_quantizer(tmp_path, FAKE_QUANTIZER)
    plan = {"header": "h", "log": str(tmp_path / "l.log"), "mix": "m.gguf",
            "steps": [("build", [str(q), str(marker)])]}
    assert A.run_plan(plan) == 0, "the repair did not take"
    assert marker.read_text() == "2", "it should have run twice: once failing, once pinned"
    assert "pinned blk.64.nextn.eh_proj.weight" in (tmp_path / "l.log").read_text()


@pytest.mark.skipif(sys.platform == "win32", reason="needs an executable shebang")
def test_the_same_tensor_is_not_pinned_forever(tmp_path):
    """If pinning does not help, stop -- do not loop."""
    q = _fake_quantizer(tmp_path, ALWAYS_FAILS, name="always-fails")
    plan = {"header": "h", "log": str(tmp_path / "l2.log"), "mix": "m.gguf",
            "steps": [("build", [str(q)])]}
    assert A.run_plan(plan) == 1


def test_the_pin_goes_first_because_custom_q_is_first_match_wins():
    got = A._pin_and_retry(["quantize", "--custom-q", "attn_q=iq2_kt", "src", "dst", "IQ1_S"],
                           "blk.64.nextn.eh_proj.weight")
    rules = got[got.index("--custom-q") + 1]
    assert rules.startswith("blk\\.64\\.nextn\\.eh_proj\\.weight=q6_K,"), \
        "a later rule would lose to the recipe's own entry for that tensor"
    assert "attn_q=iq2_kt" in rules, "the existing rules must survive"


def test_a_build_with_no_custom_q_still_gets_the_pin():
    got = A._pin_and_retry(["quantize", "src", "dst", "IQ1_S"], "blk.1.ffn_up.weight")
    assert "--custom-q" in got and got[-3:] == ["src", "dst", "IQ1_S"]


def test_repair_can_be_turned_off():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "pollard_automap.py"), encoding="utf-8").read()
    assert "def run_plan(plan, repair=True)" in src
