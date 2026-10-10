"""Converter discovery must survive a search path Windows refuses to traverse.

The box, 2026-10-09: ~/pollard/llama.cpp was a junction to C:/pollard/q35-llama that Windows reports as an
"untrusted mount point" (WinError 448). Path.is_file() RAISES on it instead of answering False, so the very
first stat in the candidate list crashed find_converter -- even with POLLARD_CONVERTER set, because every
candidate is checked before any is used. pollard-smoke then failed its converter check for Cloudflare/clef-flash.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_convert as pc  # noqa: E402


def test_untraversable_candidate_is_skipped(tmp_path, monkeypatch):
    good = tmp_path / "llama.cpp" / "convert_hf_to_gguf.py"
    good.parent.mkdir()
    good.write_text("class ClefModel: pass\n")
    bad = tmp_path / "junction" / "convert_hf_to_gguf.py"
    real_is_file = Path.is_file

    def is_file(self):
        if self == bad:
            raise OSError(448, "The path cannot be traversed because it contains an untrusted mount point")
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", is_file)
    monkeypatch.setattr(pc, "_search_paths", lambda: [bad, good])
    path, note = pc.find_converter()
    assert path == good, note


def test_is_file_never_raises(tmp_path, monkeypatch):
    def boom(self):
        raise OSError(448, "untrusted mount point")
    monkeypatch.setattr(Path, "is_file", boom)
    assert pc._is_file(tmp_path / "x") is False
