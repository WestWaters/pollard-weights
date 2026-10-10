"""--bin <dir> must name the tool the way the OS can find it. On Windows a bare join gave pollard-fit
"...\\build\\bin\\llama-quantize" -- no such file -- and the Clef 27B ladder died after a 2-hour imatrix."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_auto as pa  # noqa: E402


def test_windows_gets_exe(tmp_path, monkeypatch):
    (tmp_path / "llama-quantize.exe").write_text("")
    monkeypatch.setattr(pa.os, "name", "nt")
    assert pa._bin_in(str(tmp_path), "llama-quantize").endswith("llama-quantize.exe")


def test_posix_unchanged(tmp_path, monkeypatch):
    (tmp_path / "llama-quantize").write_text("")
    monkeypatch.setattr(pa.os, "name", "posix")
    assert pa._bin_in(str(tmp_path), "llama-quantize") == str(tmp_path / "llama-quantize")


def test_windows_without_exe_file_falls_back_to_bare(tmp_path, monkeypatch):
    monkeypatch.setattr(pa.os, "name", "nt")
    assert pa._bin_in(str(tmp_path), "llama-imatrix") == str(tmp_path / "llama-imatrix")
