"""pollard-runtime --update on macOS (2026-10-28, llama.cpp 2e0d86beb): the tree was built as <tree>.next with an
absolute build rpath, then renamed live -- every binary then failed with dyld 'Library not loaded' looking for
.../llama.cpp.next/build/bin/libllama-common.0.dylib. The pre-swap smoke passed, and the smoke's own 'usage|version|
build' test would have passed the dyld error too (the error names a build/bin path)."""
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import pollard_runtime_update as U  # noqa: E402

pytestmark = pytest.mark.skipif(U.WIN, reason="shell-script stand-ins for the binaries")

DYLD = ("dyld[123]: Library not loaded: @rpath/libllama-common.0.dylib\\n  Reason: tried: "
        "'/Users/x/runtime/llama.cpp.next/build/bin/libllama-common.0.dylib' (no such file)")


def _tree(tmp_path, body):
    d = tmp_path / "llama.cpp" / "build" / "bin"
    d.mkdir(parents=True)
    for b in U.KEY_BINS:
        f = d / b
        f.write_text(f"#!/bin/sh\n{body}\n")
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return str(tmp_path / "llama.cpp")


def test_a_binary_that_cannot_load_its_libraries_fails_the_smoke(tmp_path):
    ok, msg = U.smoke(_tree(tmp_path, f'printf "{DYLD}" >&2; exit 134'))
    assert not ok and "did not start" in msg


def test_binaries_that_start_pass_the_smoke(tmp_path):
    ok, _ = U.smoke(_tree(tmp_path, 'echo "version: 0.4.0-dev (build 999, commit abc)"'))
    assert ok


def test_the_build_uses_relative_rpaths_so_the_rename_to_live_keeps_them_valid(tmp_path, monkeypatch):
    seen = []

    class R:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(U, "_run", lambda s, **k: (seen.append(s), R())[1])
    U.build(str(tmp_path), {"GGML_METAL": "ON"}, 2)
    assert "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON" in seen[0]                 # Linux ($ORIGIN)
    if sys.platform == "darwin":                                           # macOS ignores the above
        assert "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON" in seen[0]
        assert "-DCMAKE_INSTALL_RPATH=@loader_path" in seen[0]
