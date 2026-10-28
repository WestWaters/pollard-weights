"""pollard --hf ... --output <new dir> (2026-10-28): the converter only writes into an existing folder, so the first
release of a new model (Qwen3-ASR-1.7B, granite-speech-5.0-2b) wrote neither its f16 nor its mmproj and the build
died on the missing file. _hf_to_gguf creates the output folder before it converts."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import pollard_auto as A  # noqa: E402


def test_the_output_folder_exists_before_the_converter_runs(tmp_path, monkeypatch):
    out_dir = tmp_path / "models" / "New-Model" / "gguf"            # does not exist yet
    hf = tmp_path / "downloads" / "New-Model"
    hf.mkdir(parents=True)
    seen = {}
    monkeypatch.setattr(A, "_resolve_hf", lambda a: str(hf))
    monkeypatch.setattr(A, "_find_convert", lambda d: "/x/convert_hf_to_gguf.py")
    monkeypatch.setattr(A, "_emit_mmproj", lambda *a, **k: seen.setdefault("mmproj_dir_exists", out_dir.is_dir()))
    monkeypatch.setattr(A.subprocess, "run", lambda cmd, *a, **k: seen.setdefault("convert_dir_exists", out_dir.is_dir()))
    out = A._hf_to_gguf(argparse.Namespace(output=str(out_dir), run=True))
    assert out == str(out_dir / "New-Model-f16.gguf")
    assert seen == {"mmproj_dir_exists": True, "convert_dir_exists": True}


def test_a_plan_only_run_creates_nothing(tmp_path, monkeypatch):
    out_dir = tmp_path / "gguf"
    hf = tmp_path / "M"
    hf.mkdir()
    monkeypatch.setattr(A, "_resolve_hf", lambda a: str(hf))
    monkeypatch.setattr(A, "_find_convert", lambda d: "/x/convert_hf_to_gguf.py")
    monkeypatch.setattr(A, "_emit_mmproj", lambda *a, **k: None)
    A._hf_to_gguf(argparse.Namespace(output=str(out_dir), run=False))
    assert not out_dir.exists()
