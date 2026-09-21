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
