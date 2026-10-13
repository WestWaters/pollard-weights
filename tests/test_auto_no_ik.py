"""pollard --ladder without ik_llama: no ik-only atoms are planned, and the run completes.

`pollard --hf <model> --run --ladder` printed "ik_llama build not found" and then planned the IQ1_KT
automap mix anyway: the flagship step was gated on having an imatrix, not on having ik_llama. Without
--bin, pollard-automap fell back to the stock llama-quantize on PATH, which rejects every ik type, so
the run died after the ladder. A stock llama-quantize is not an ik build -- only ik's ggml names iq1_kt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_auto as pa  # noqa: E402
import pollard_automap as A  # noqa: E402

POINTS = [(9.5, "Q6_K", 7.5), (6.6, "IQ4_XS", 5.61), (5.7, "IQ3_S", 4.83), (4.3, "IQ2_XXS", 3.65)]


def _bin(d, lib=b"llama\0iq2_xxs\0"):
    d.mkdir(parents=True)
    (d / "llama-quantize").write_bytes(b"\x7fELF quantize")
    (d / "libggml-base.dylib").write_bytes(b"\0" * 64 + lib)
    return d


def _ik(d):
    return _bin(d, lib=b"llama\0iq2_xxs\0iq1_kt\0iq2_kt\0")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """No ik anywhere unless a test puts one there; every command recorded, nothing executed."""
    monkeypatch.delenv("IK_LLAMA_BIN", raising=False)
    monkeypatch.delenv("POLLARD_IK_BIN", raising=False)
    monkeypatch.setattr(pa, "_MANAGED_IK", str(tmp_path / "no-managed"))
    stock = _bin(tmp_path / "homebrew" / "bin")
    monkeypatch.setattr(A, "find_ik_bin", lambda explicit=None: str(stock))  # PATH fallback = stock
    ran = []
    monkeypatch.setattr(pa, "_run", lambda cmd, do_run, cwd=None: ran.append([str(c) for c in cmd]))
    monkeypatch.setattr(pa, "read_gguf_meta", lambda p: {"general.architecture": "llama"})
    monkeypatch.setattr(pa, "gguf_to_config", lambda meta, p: {})
    monkeypatch.setattr(pa, "analyse", lambda cfg: {"total": 7e9, "layers": 32, "n_experts": 0})
    monkeypatch.setattr(pa, "size_ladder", lambda *a, **k: None)
    monkeypatch.setattr(pa, "ladder_points", lambda *a, **k: POINTS)
    monkeypatch.setattr(pa, "find_llama_bin", lambda name: str(stock / name))
    monkeypatch.setattr(pa.subprocess, "run", lambda *a, **k: argparse.Namespace(returncode=0))
    gguf = tmp_path / "model-f16.gguf"
    gguf.write_bytes(b"GGUF" + b"\0" * 64)
    imat = tmp_path / "model-f16.imatrix"
    imat.write_bytes(b"\0")
    return argparse.Namespace(tmp=tmp_path, stock=stock, ran=ran, gguf=str(gguf), imatrix=str(imat))


def _args(env, **kw):
    a = dict(gguf=env.gguf, hf=None, format="gguf", smooth=False, run=True, card=False,
             imatrix=env.imatrix, auto_imatrix=True, sensitivity=None, measure=False, ladder=True,
             bin=None, benchmark=False, gate=True, force_dense=False, force_moe=False, ram="16",
             out=None, calib=None, ngl="auto")
    a.update(kw)
    return argparse.Namespace(**a)


def _automap(ran):
    return [c for c in ran if c and c[0].endswith("pollard-automap")]


def test_stock_quantize_is_not_an_ik_build(tmp_path):
    assert pa._is_ik_build(str(_bin(tmp_path / "stock"))) is False
    assert pa._is_ik_build(str(_ik(tmp_path / "ik"))) is True
    assert pa._is_ik_build(str(tmp_path / "missing")) is False
    libs_only = tmp_path / "libs-only"
    libs_only.mkdir()
    (libs_only / "libggml-base.dylib").write_bytes(b"iq1_kt")
    assert pa._is_ik_build(str(libs_only)) is False          # no llama-quantize, nothing to build with


def test_stock_on_path_is_not_taken_for_ik(env):
    assert pa.find_ik_build(None) is None
    assert pa.find_ik_build(str(env.stock)) is None          # --bin pointing at a stock build


def test_missing_ik_plans_no_ik_atoms_and_completes(env, capsys):
    pa._dispatch(_args(env))
    out = capsys.readouterr().out
    assert not _automap(env.ran), "planned the ik-only automap mix with no ik_llama build"
    fits = [c for c in env.ran if c[0].endswith("pollard-fit")]
    assert len(fits) == 4                                     # Q6_K / IQ4_XS / IQ3_S / stock IQ2
    assert "IQ2" in out and "IQ2_XXS" in out
    assert "ik_llama.cpp build not found" in out
    assert "skipped the IQ1_KT flagship" in out and "IK_LLAMA_BIN" in out
    assert not any("_kt" in " ".join(c).lower() or "iq1_bn" in " ".join(c).lower() for c in env.ran)


def test_missing_ik_without_ladder_also_skips_the_flagship(env, capsys):
    pa._dispatch(_args(env, ladder=False))
    assert not _automap(env.ran)
    assert "skipped the IQ1_KT flagship" in capsys.readouterr().out


def test_ik_present_plan_is_unchanged(env, capsys):
    ik = _ik(env.tmp / "ik_llama.cpp" / "build" / "bin")
    pa._dispatch(_args(env, bin=str(ik)))
    out = capsys.readouterr().out
    assert "trellis flagship: available" in out
    fits = [c for c in env.ran if c[0].endswith("pollard-fit")]
    assert len(fits) == 3                                     # the 2-bit rung is the flagship's
    am = _automap(env.ran)
    assert len(am) == 1 and am[0][am[0].index("--bin") + 1] == str(ik)
    assert "--allow-dense" in am[0] and am[0][am[0].index("--imatrix") + 1] == env.imatrix


def test_ik_found_via_env_is_what_automap_builds_with(env, monkeypatch):
    ik = _ik(env.tmp / "ik")
    monkeypatch.setenv("IK_LLAMA_BIN", str(ik))
    pa._dispatch(_args(env))
    am = _automap(env.ran)
    assert len(am) == 1 and am[0][am[0].index("--bin") + 1] == str(ik)
