"""Each lane emits a LADDER of rungs: MLX 2/3/4/5/6/8-bit, GPTQ int2/3/4/8, MX NVFP4/MXFP4/pure FP8 W8A8.
Plan-level only -- no mlx_lm, gptqmodel or llm-compressor."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_export as E  # noqa: E402
import pollard_mlx as M  # noqa: E402
import pollard_mx as X  # noqa: E402

SENS = {"ffn": {"0": 1.0, "1": 9.0, "2": 2.0, "3": 3.0}, "attn": {"0": 5.0, "1": 1.0, "2": 2.0, "3": 3.0}}


def test_mlx_rungs_keep_hot_layers_above_the_body():
    assert [M.rung_bits(b) for b in M.MLX_BITS] == [(2, 4), (3, 6), (4, 8), (5, 8), (6, 8), (8, 8)]
    with pytest.raises(ValueError):
        M.rung_bits(7)
    al = M.allocate(SENS, 4, 0.25, low=3, high=6)
    assert al[1]["ffn"] == 6 and al[0]["attn"] == 6 and al[2]["ffn"] == 3
    assert M.bits_for("model.layers.2.mlp.up_proj", al, False, 64, low=3, high=6)["bits"] == 3
    # the modality projector never drops with the rung
    assert M.bits_for("model.visual.merger.mlp.0", al, False, 64, low=2, high=4)["bits"] == 8
    # the default (no --bits) is still the 4/8 mix
    assert M.allocate(SENS, 4, 0.25)[1]["ffn"] == 8


def test_gptq_rungs_and_dynamic_map_follow_the_high_bits():
    al = E.allocate(SENS, 4, 0.25, low=2, high=4)
    assert al[1]["ffn"] == 4 and al[3]["ffn"] == 2
    dyn = E.dynamic_config(al, high=4)
    assert any(v == {"bits": 4} for v in dyn.values()) and not any(v == {"bits": 8} for v in dyn.values())
    assert E.GPTQ_BITS == (2, 3, 4, 8)


def test_mx_pure_fp8_protects_only_routers_head_and_towers():
    rec = X.build_recipe(set(), "FP8", "FP8", False)
    assert rec["protect"] == [] and "lm_head" in rec["ignore"]
    assert any("gate$" in g for g in rec["ignore"]), "MoE routers stay at checkpoint precision"
    assert "FP8" in X.PURE_FP8 and "FP8_DYNAMIC" in X.PURE_FP8
    assert X.scheme_bits("FP8") == 8.0 and X.scheme_bits("NVFP4") == 4.0


def test_mx_cli_plans_pure_fp8(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pollard-mx", "--model", "/nonexistent", "--layers", "4", "--scheme", "FP8",
                                      "--plan-only"])
    X.main()
    out = capsys.readouterr().out
    assert "pure FP8 W8A8" in out and "8.0 bpw" in out
