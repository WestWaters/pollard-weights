"""The forward probe scores a hybrid model's linear-attention layers instead of handing them back at 0.0.

clef-flash (Qwen3.5), 2026-10-09: 24 of 32 layers mix with `linear_attn` (Gated DeltaNet), not `self_attn`. The
probe's attn group listed only self_attn, so those layers scored 0.0 and pollard-fit planned every one of their
attn_qkv / attn_gate at iq2_xxs in the IQ3_S rung -- an unmeasured tensor crushed as if measured "free".
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_probe as P  # noqa: E402


def _layer(kind):
    lin = lambda: torch.nn.Linear(4, 4)
    blk = types.SimpleNamespace(mlp=types.SimpleNamespace(gate_proj=lin(), up_proj=lin(), down_proj=lin()))
    if kind == "full":
        blk.self_attn = types.SimpleNamespace(q_proj=lin(), k_proj=lin(), v_proj=lin(), o_proj=lin())
    else:
        blk.linear_attn = types.SimpleNamespace(in_proj_qkv=lin(), in_proj_z=lin(), in_proj_a=lin(),
                                                in_proj_b=lin(), out_proj=lin(), conv1d=lin())
    return blk


def _model():
    return types.SimpleNamespace(model=types.SimpleNamespace(layers=[_layer("linear"), _layer("full")]))


def test_linear_attention_layer_has_attn_linears():
    got = P._linears(_model(), 0, "attn")
    assert len(got) == 5                                  # in_proj_qkv/z/a/b + out_proj, not conv1d


def test_full_attention_layer_unchanged():
    assert len(P._linears(_model(), 1, "attn")) == 4
    assert len(P._linears(_model(), 0, "ffn")) == 3 and len(P._linears(_model(), 1, "ffn")) == 3
