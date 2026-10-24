"""pollard_calc on a GGUF whose per-layer metadata arrays are longer than 64 entries.

What it protects: the GGUF reader dropped EVERY list over 64 entries to skip tokenizer blobs, so Nemotron-3.5-Bolt
(88 layers; expert_used_count stored per layer, 0 on its Mamba and attention layers) lost its experts-per-token and
every ladder rung died with "MoE config but no experts-per-token field found" (2026-10-23).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import pollard_calc as C

LAYERS = 88


def write(path):
    w = gguf.GGUFWriter(str(path), "nemotron_h_moe")
    w.add_block_count(LAYERS)
    w.add_embedding_length(64)
    w.add_expert_count(496)
    used = [9 if i % 2 else 0 for i in range(LAYERS)]
    w.add_array("nemotron_h_moe.expert_used_count", used)
    w.add_array("nemotron_h_moe.expert_feed_forward_length", [32 if u else 0 for u in used])
    w.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(100)])
    w.add_tensor("blk.1.ffn_up_exps.weight", np.zeros((4, 2, 64), np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def test_long_per_layer_arrays_survive_and_reduce(tmp_path):
    p = tmp_path / "m.gguf"
    write(p)
    meta = C._read_one_gguf(str(p))[0]
    assert len(meta["nemotron_h_moe.expert_used_count"]) == LAYERS
    assert "tokenizer.ggml.tokens" not in meta                       # tokenizer blobs still skipped
    cfg = C.gguf_to_config(meta, str(p))
    assert cfg["num_experts_per_tok"] == 9
    assert cfg["moe_intermediate_size"] == 32
    assert cfg["num_experts"] == 496
