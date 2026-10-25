"""Low MLX rungs: the per-layer embedding side paths held high, attention v / o floored, a per-layer token table at
the rung's measured choice, and per-rung pollard-mlx flags from the release plan. Measured on
NVIDIA-Nemotron-3.5-Bolt-E4B 2-bit: KLD 7.61 -> 0.964, top-1 12.9% -> 71.2%, tool calls 15/60 -> 55/60."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import pollard_mlx as M  # noqa: E402

ALLOC = {i: {"attn": 2, "ffn": 2} for i in range(4)}


def _b(path, **k):
    r = M.bits_for(path, ALLOC, False, 64, None, 2, 4, **k)
    return r["bits"] if r else r


def test_ple_side_paths_are_held_high():
    assert _b("language_model.model.layers.1.per_layer_input_gate") >= 6
    assert _b("language_model.model.layers.1.per_layer_projection") >= 6
    assert _b("language_model.model.per_layer_model_projection") >= 6
    assert _b("language_model.model.layers.1.mlp.gate_proj") == 2, "the body keeps the rung's bits"


def test_table_bits_only_moves_the_per_layer_table():
    assert _b("language_model.model.embed_tokens_per_layer") == 4, "default: with the embeddings, HIGH"
    assert _b("language_model.model.embed_tokens_per_layer", table_bits=2) == 2
    assert _b("language_model.model.embed_tokens", table_bits=2) == 4, "the input embedding stays HIGH"


def test_attention_floor_lifts_v_and_o_only():
    assert _b("language_model.model.layers.2.self_attn.v_proj", attn_min=4) == 4
    assert _b("language_model.model.layers.2.self_attn.o_proj", attn_min=4) == 4
    assert _b("language_model.model.layers.2.self_attn.q_proj", attn_min=4) == 2
    assert _b("language_model.model.layers.2.self_attn.v_proj") == 2, "no floor unless asked"


def test_release_passes_per_rung_args(tmp_path, monkeypatch):
    import json
    import pollard_release as R
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    hf = tmp_path / "M"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    cfg = R.load_plan(None)
    cfg["lanes"]["mlx"] = {"enabled": True, "rungs": [2, 4], "rung_args": {2: ["--table-bits", "2"], "4": ["--x"]}}
    c = R._ctx(cfg, str(hf))
    steps = R.lane_steps(cfg, c, "mlx")
    cmds = {s.rung: s.cmd for s in steps if s.id.endswith("/build")}
    assert cmds["2bit"][-2:] == ["--table-bits", "2"] and cmds["4bit"][-1:] == ["--x"]
