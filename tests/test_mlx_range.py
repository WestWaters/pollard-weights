"""pollard-mlx --range search: the imatrix-weighted per-group range search (GPQ's encoder idea) emitting
STANDARD MLX affine -- same packed layout, same scale/bias tensors, only the values chosen differ.

numpy-only tests run everywhere; the MLX layout cross-checks need mlx; the end-to-end convert (tiny llama ->
pollard-mlx -> mlx_lm.load) needs mlx_lm + transformers + torch + tokenizers. The real KL A/B is a box/Mac run
(notes/mlx-range-search.md)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_mlx as M  # noqa: E402


def _weights(n=128, k=512, seed=0):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((n, k)) * 0.02).astype(np.float32)
    w[:, rng.integers(0, k, 6)] *= 10                      # outlier input channels, as real layers have
    imp = rng.gamma(0.5, 1.0, k).astype(np.float32)
    imp[rng.integers(0, k, 12)] *= 40                      # a few channels carry most activation energy
    return w, imp


@pytest.mark.parametrize("bits", M.MLX_BITS)
def test_pack_is_mlx_little_endian_bitstream_and_round_trips(bits):
    rng = np.random.default_rng(bits)
    q = rng.integers(0, 2 ** bits, (3, 128)).astype(np.uint8)
    p = M.pack_mlx(q, bits)
    assert p.dtype == np.uint32 and p.shape == (3, 128 * bits // 32)
    assert np.array_equal(M.unpack_mlx(p, bits, 128), q)
    if bits == 4:                                          # element 0 in the lowest nibble of word 0
        assert int(p[0, 0]) & 0xF == int(q[0, 0]) and (int(p[0, 0]) >> 4) & 0xF == int(q[0, 1])


@pytest.mark.parametrize("bits", (2, 3, 4, 6))
def test_search_is_standard_layout_and_beats_minmax_on_weighted_error(bits):
    w, imp = _weights()
    p, s, b = M.encode_affine(w, bits, 64, imp=imp)
    assert p.dtype == np.uint32 and p.shape == (128, 512 * bits // 32)
    assert s.shape == b.shape == (128, 512 // 64)
    e_search = M.weighted_error(w, p, s, b, bits, 64, imp)
    pm, sm, bm = M.encode_affine(w, bits, 64, method="minmax")
    e_minmax = M.weighted_error(w, pm, sm, bm, bits, 64, imp)
    assert e_search < 0.8 * e_minmax, (e_search, e_minmax)
    # unweighted search still lowers plain weight error
    pu, su, bu = M.encode_affine(w, bits, 64)
    assert M.weighted_error(w, pu, su, bu, bits, 64) <= M.weighted_error(w, pm, sm, bm, bits, 64)


def test_search_never_loses_a_group_to_minmax():
    w, imp = _weights(16, 256, seed=3)
    g = w.reshape(16, 4, 64)
    I = imp.reshape(1, 4, 64)
    q, s, b, e = M.search_groups(g, 4, I)
    _, _, _, e0 = M.search_groups(g, 4, I, grid=(0.0,), refits=0)
    assert np.all(e <= e0 + 1e-9)


def test_bf16_storage_is_what_gets_scored():
    w, imp = _weights(8, 128)
    _, s, b = M.encode_affine(w, 4, 64, imp=imp, store="bfloat16")
    for x in (s, b):
        assert np.all((np.ascontiguousarray(x, np.float32).view(np.uint32) & 0xFFFF) == 0), "not bf16-exact"


def test_importance_shapes_and_errors():
    w, imp = _weights(8, 128)
    M.encode_affine(w, 4, 64, imp=np.tile(imp[:128], (8, 1)))          # per-row importance works too
    with pytest.raises(ValueError):
        M.encode_affine(w, 4, 64, imp=np.ones(100, np.float32))
    with pytest.raises(ValueError):
        M.encode_affine(w[:, :100], 4, 64)
    with pytest.raises(ValueError):
        M.encode_affine(w, 7, 64)


def test_gguf_imatrix_names_map_to_module_paths(tmp_path):
    assert M.gguf_name_to_path("blk.3.attn_q.weight") == "model.layers.3.self_attn.q_proj"
    assert M.gguf_name_to_path("blk.0.ffn_down.weight") == "model.layers.0.mlp.down_proj"
    assert M.gguf_name_to_path("output.weight") == "lm_head"
    assert M.gguf_name_to_path("token_embd.weight") is None
    p = tmp_path / "imp.npz"
    np.savez(p, **{"model.layers.0.mlp.up_proj": np.ones(64, np.float32)})
    assert set(M.load_importance(str(p))) == {"model.layers.0.mlp.up_proj"}


def test_legacy_imatrix_is_read_without_torch(tmp_path):
    import struct
    vals = np.arange(1, 65, dtype=np.float32)
    name = b"blk.1.attn_k.weight"
    blob = struct.pack("<i", 1) + struct.pack("<i", len(name)) + name + struct.pack("<ii", 4, 64) + vals.tobytes()
    blob += struct.pack("<i", 10) + struct.pack("<i", 4) + b"calb"        # trailing chunk count + dataset name
    p = tmp_path / "m.imatrix"
    p.write_bytes(blob)
    imp = M.load_importance(str(p))
    assert np.allclose(imp["model.layers.1.self_attn.k_proj"], vals / 4)


# ----------------------------------------------------------------- against MLX itself

def test_output_is_what_mlx_dequantizes_and_beats_mx_quantize():
    mx = pytest.importorskip("mlx.core")
    w, imp = _weights()
    for bits in (3, 4):
        wq, s0, b0 = mx.quantize(mx.array(w), 64, bits)
        # same packing as MLX for MLX's own codes
        q0 = np.clip(np.rint((w.reshape(128, 8, 64) - np.array(b0)[..., None]) / np.array(s0)[..., None]),
                     0, 2 ** bits - 1).reshape(128, 512)
        assert np.array_equal(M.pack_mlx(q0, bits), np.array(wq))
        p, s, b = M.encode_affine(mx.array(w), bits, 64, imp=imp, xp=mx, init=(s0, b0))
        assert p.dtype == np.uint32 and tuple(s.shape) == tuple(s0.shape) and s.dtype == s0.dtype
        deq = np.array(mx.dequantize(mx.array(p), s, b, group_size=64, bits=bits))
        ours = (M.unpack_mlx(p, bits, 512).reshape(128, 8, 64) * np.array(s)[..., None]
                + np.array(b)[..., None]).reshape(128, 512)
        assert np.allclose(deq, ours, atol=1e-6)
        e_ours = M.weighted_error(w, p, np.array(s), np.array(b), bits, 64, imp)
        e_mlx = M.weighted_error(w, np.array(wq), np.array(s0), np.array(b0), bits, 64, imp)
        assert e_ours < 0.8 * e_mlx, (bits, e_ours, e_mlx)


def test_context_manager_routes_mlx_quantization_and_restores_it():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    mx.random.seed(0)
    model = nn.Sequential(nn.Linear(128, 64), nn.Linear(64, 128))
    orig = mx.quantize
    imp = {"layers.0": np.linspace(0.1, 10, 128).astype(np.float32)}
    rs = M.RangeSearchQuantize(imp)

    def pred(path, module):
        rs.note(path, module)
        return isinstance(module, nn.Linear)
    with rs:
        nn.quantize(model, 64, 4, class_predicate=pred)
    assert mx.quantize is orig
    assert type(model.layers[0]).__name__ == "QuantizedLinear"
    assert rs.stats["searched"] == 2 and rs.stats["weighted"] == 1
    y = model(mx.ones((1, 128)))
    assert y.shape == (1, 128)


def test_end_to_end_tiny_llama_converts_and_loads_in_mlx_lm(tmp_path):
    pytest.importorskip("mlx_lm")
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    tk = pytest.importorskip("tokenizers")
    vocab = {f"t{i}": i for i in range(256)}
    tok = tk.Tokenizer(tk.models.WordLevel(vocab=vocab, unk_token="t0"))
    tok.pre_tokenizer = tk.pre_tokenizers.Whitespace()
    src = tmp_path / "tiny"
    cfg = tr.LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(0)
    tr.LlamaForCausalLM(cfg).to(torch.float16).save_pretrained(src)
    tr.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="t0", bos_token="t1",
                               eos_token="t2").save_pretrained(src)
    np.savez(tmp_path / "imp.npz", **{"model.layers.0.self_attn.q_proj": np.ones(128, np.float32)})
    out = tmp_path / "mlx"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "tools"), HF_HUB_OFFLINE="1")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "pollard_mlx.py"), "--model", str(src),
                        "--bits", "4", "--imatrix", str(tmp_path / "imp.npz"), "--out", str(out)],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "1 imatrix-weighted" in r.stdout
    q = json.loads((out / "config.json").read_text())["quantization"]
    assert q["group_size"] == 64
    from mlx_lm import load
    import mlx.core as mx
    m, _ = load(str(out))
    assert m(mx.array([[1, 2, 3]])).shape == (1, 3, 256)
