"""The measured MoE rung's own imatrix gets the decision-head shape, and pollard-moefit resolves its engine by arch.

clef-omni release, [gguf/MOE]: the main imatrix (built with _imatrix_shape's -c/-b/-ub 512 --no-ppl) was fine, then
the MoE rung built its per-expert GGUF imatrix with only `-c 512` and llama-imatrix died on
GGML_ASSERT(params.n_ctx == n_seq * n_ctx). Every imatrix now takes its shape from the one helper.
"""
from __future__ import annotations

import os
import sys
import types as _t
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_auto as A  # noqa: E402
import pollard_moefit as M  # noqa: E402

DECISION_META = {"general.architecture": "clef-omni", "clef-omni.decision.block_count": 1,
                 "clef-omni.expert_count": 4}


def _plan_moe_rung(tmp_path, monkeypatch, meta):
    f16 = tmp_path / "m-f16.gguf"
    f16.write_bytes(b"GGUF" + b"\0" * 4096)
    calib = tmp_path / "release-calib.txt"
    calib.write_text("x\n" * 10)
    (tmp_path / "release-calib.heldout.txt").write_text("held\n" * 10)
    ran = []
    monkeypatch.setattr(A, "_run", lambda cmd, do_run, cwd=None: ran.append([str(c) for c in cmd]) or True)
    monkeypatch.setattr(A, "find_llama_bin", lambda n: f"/bin/{n}")
    monkeypatch.setattr(A, "_fit_ngl", lambda g, n: 99)
    monkeypatch.setattr(A, "_moe_budget_gb", lambda a, f16_gb: 0.36)
    monkeypatch.setattr(A, "read_gguf_meta", lambda p: dict(meta))
    a = _t.SimpleNamespace(gguf=str(f16), calib=str(calib), heldout=None, bin=None, ngl="auto", run=False,
                           imatrix=str(tmp_path / "m.imatrix"), sensitivity=None, moe_gb=None)
    A._moe_rung(a)
    imx = [c for c in ran if os.path.basename(c[0]) == "llama-imatrix"]
    assert len(imx) == 1
    return imx[0]


def test_moe_rung_imatrix_on_a_decision_head_gets_the_embedding_shape(tmp_path, monkeypatch):
    cmd = _plan_moe_rung(tmp_path, monkeypatch, DECISION_META)
    for flag in ("-b", "-ub"):
        assert cmd[cmd.index(flag) + 1] == "512"
    assert "--no-ppl" in cmd
    assert cmd.count("-c") == 1 and cmd[cmd.index("-c") + 1] == "512"
    assert cmd[cmd.index("--chunks") + 1] == "128"
    assert cmd[cmd.index("-o") + 1].endswith(".moe.imatrix.gguf")        # still GGUF: per-expert counts


def test_moe_rung_imatrix_matches_the_main_imatrix_shape(tmp_path, monkeypatch):
    cmd = _plan_moe_rung(tmp_path, monkeypatch, DECISION_META)
    shape = A._imatrix_shape("any.gguf")                                  # same patched metadata
    assert shape and all(x in cmd for x in shape)


def test_moe_rung_imatrix_on_a_plain_moe_is_unchanged(tmp_path, monkeypatch):
    cmd = _plan_moe_rung(tmp_path, monkeypatch, {"general.architecture": "qwen3moe", "qwen3moe.expert_count": 8})
    assert cmd[cmd.index("-c") + 1] == "512"
    assert "--no-ppl" not in cmd and "-b" not in cmd and "-ub" not in cmd


def test_moefit_engine_lookup_is_arch_aware(tmp_path, monkeypatch):
    """pollard-moefit run on its own sets the model's arch before resolving llama-quantize / llama-perplexity, so
    find_llama_bin picks the engine tree carrying that arch's patches (as pollard / release / verify do)."""
    import pollard_calc as C
    g = tmp_path / "clef-omni-f16.gguf"
    g.write_bytes(b"GGUF")
    monkeypatch.setattr(C, "read_gguf_meta", lambda p: dict(DECISION_META))
    seen = []
    monkeypatch.setattr(C, "find_llama_bin", lambda n, arch=None: seen.append((n, os.environ.get("POLLARD_ARCH")))
                        or f"/engines/bolt/{n}")
    q, ppl = M._engine(_t.SimpleNamespace(gguf=str(g)), "llama-quantize", "llama-perplexity")
    assert (q, ppl) == ("/engines/bolt/llama-quantize", "/engines/bolt/llama-perplexity")
    assert seen == [("llama-quantize", "clef-omni"), ("llama-perplexity", "clef-omni")]


def test_moefit_engine_passes_an_unresolved_name_through(monkeypatch):
    import pollard_calc as C
    monkeypatch.setattr(C, "find_llama_bin", lambda n, arch=None: None)
    assert M._engine(_t.SimpleNamespace(), "llama-quantize") == ("llama-quantize",)


def test_moefit_errmap_uses_the_arch_aware_engine(tmp_path, monkeypatch):
    got = {}
    monkeypatch.setattr(M, "_engine", lambda a, *n: tuple(f"/resolved/{x}" for x in n))
    monkeypatch.setattr(M, "errmap", lambda g, imx, out, types, q, *rest, **kw: got.setdefault("q", q))
    M.main(["errmap", "--gguf", str(tmp_path / "m.gguf"), "--imatrix", "m.imatrix", "--out", str(tmp_path / "e.json")])
    assert got["q"] == "/resolved/llama-quantize"
