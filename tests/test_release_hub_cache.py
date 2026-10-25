"""A rung folder holding huggingface_hub's own local cache (`.cache/huggingface/`, left by `hf download --local-dir`)
is still intact on the Hub when every real file is there: upload_folder never sends that cache (its
DEFAULT_IGNORE_PATTERNS), so the intact check, the free step and the relay staging must not ask for it either.
E4B MLX/2bit (2026-10-25): every file on the Hub, publish said "not intact (.cache/huggingface/.gitignore: not on the
Hub)" and kept 3.3 GB. Fakes only."""
from __future__ import annotations

import sys
import types
from fnmatch import fnmatch
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_release as R  # noqa: E402


def _rung(tmp_path):
    d = tmp_path / "Tiny-1B-Pollard-MLX-2bit"
    (d / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (d / ".cache" / "huggingface" / ".gitignore").write_text("*")
    (d / ".cache" / "huggingface" / "download" / "config.json.metadata").write_text("x")
    (d / ".git").mkdir()
    (d / ".git" / "HEAD").write_text("ref")
    (d / "model.safetensors").write_bytes(b"w" * 64)
    (d / "config.json").write_text("{}")
    (d / "sub").mkdir()
    (d / "sub" / "tokenizer.json").write_text("{}")
    return d


REAL = {"model.safetensors", "config.json", "sub/tokenizer.json"}


def test_hub_skipped_matches_upload_folder_defaults():
    hf = pytest.importorskip("huggingface_hub.hf_api")
    pats = hf.DEFAULT_IGNORE_PATTERNS
    for rel in (".cache/huggingface/.gitignore", ".cache/huggingface/download/a.metadata", ".git/HEAD",
                "sub/.cache/huggingface/x", "model.safetensors", "config.json", ".cache/other/file",
                "cache/huggingface/x", "sub/tokenizer.json"):
        assert R._hub_skipped(rel) == any(fnmatch(rel, p) for p in pats), rel


def test_intact_check_pairs_leave_out_the_hub_cache(tmp_path):
    d = _rung(tmp_path)
    pairs = R._local_pairs(str(d), "MLX/2bit")
    assert {rp for rp, _ in pairs} == {f"MLX/2bit/{r}" for r in REAL}


def test_free_check_files_leave_out_the_hub_cache(tmp_path):
    assert set(R._local_files(str(_rung(tmp_path)))) == REAL


def test_relay_staging_leaves_out_the_hub_cache(tmp_path):
    plan = R.RelayApi._local_plan(str(_rung(tmp_path)), "MLX/2bit")
    assert {rel for rel, _, _ in plan} == REAL


def test_rung_with_a_hub_cache_is_intact_when_every_real_file_is_there(tmp_path):
    d = _rung(tmp_path)
    pairs = R._local_pairs(str(d), "MLX/2bit")
    rows = [{"path": rp, "size": Path(lp).stat().st_size, "lfs": {"sha256": R._sha256(lp)}, "blob_id": None}
            for rp, lp in pairs]

    class Api:
        def get_paths_info(self, repo, paths):
            return [types.SimpleNamespace(**r) for r in rows if r["path"] in paths]
    assert R.remote_mismatches(Api(), "r", pairs) == []
