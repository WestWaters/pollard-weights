"""The streamed MLX verify never holds a big lookup table whole (Gemma 4's per-layer embedding: 5.6 GB of bf16):
only the rows of the token ids it feeds are read from the file, and the lookups are remapped onto them -- the
result is the full table's, exactly, for plain and quantized tables."""
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import pollard_mlx as M  # noqa: E402


class Toy(nn.Module):
    def __init__(self, V=512, D=128):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, 64)
        self.embed_tokens_per_layer = nn.Embedding(V, D)


def _saved(tmp_path, quantized):
    m = Toy()
    rng = np.random.default_rng(0)
    m.embed_tokens_per_layer.weight = mx.array(rng.standard_normal((512, 128)).astype(np.float32)).astype(mx.bfloat16)
    if quantized:
        nn.quantize(m, group_size=64, bits=4, class_predicate=lambda p, mod: "per_layer" in p)
    from mlx.utils import tree_flatten
    p = str(tmp_path / "m.safetensors")
    mx.save_safetensors(p, dict(tree_flatten(m.parameters())))
    return p, quantized


def _lazy(p, quantized):
    m = Toy()
    if quantized:
        nn.quantize(m, group_size=64, bits=4, class_predicate=lambda p_, mod: "per_layer" in p_)
    w = mx.load(p)
    m.load_weights(list(w.items()))
    loads = []
    M.register_loads(loads, p, w)
    return m, loads


@pytest.mark.parametrize("quantized", [False, True])
def test_gathered_table_gives_the_full_tables_rows(tmp_path, monkeypatch, quantized):
    p, q = _saved(tmp_path, quantized)
    full, _ = _lazy(p, q)
    ids = [[5, 9, 9, 400], [1, 5, 511]]
    want = [np.array(full.embed_tokens_per_layer(mx.array([r])).astype(mx.float32)) for r in ids]
    monkeypatch.setattr(M, "GATHER_MIN_BYTES", 0)
    m, loads = _lazy(p, q)
    swapped = M.gather_lookup_tables(m, loads, ids, log=lambda *a: None)
    assert [a for _, a, _ in swapped] == ["embed_tokens_per_layer"], "the per-layer table, not the input embedding"
    g = m.embed_tokens_per_layer
    assert isinstance(g, M._GatheredTable) and g.module["weight"].shape[0] == 5, "only the 5 distinct ids' rows"
    for r, w in zip(ids, want):
        got = np.array(m.embed_tokens_per_layer(mx.array([r])).astype(mx.float32))
        assert np.array_equal(got, w), "identical to the full table's lookup"


def test_a_computed_table_is_left_whole(tmp_path, monkeypatch):
    p, q = _saved(tmp_path, False)
    m, loads = _lazy(p, q)
    m.embed_tokens_per_layer.weight = m.embed_tokens_per_layer.weight * 1
    monkeypatch.setattr(M, "GATHER_MIN_BYTES", 0)
    assert M.gather_lookup_tables(m, loads, [[1, 2]], log=lambda *a: None) == []


def test_only_big_non_input_embeddings_are_gathered():
    assert M._is_gathered_table("language_model.model.embed_tokens_per_layer", 6 << 30)
    assert not M._is_gathered_table("language_model.model.embed_tokens", 6 << 30), "the (tied) input embedding stays"
    assert not M._is_gathered_table("language_model.model.embed_tokens_per_layer", 1 << 20), "small tables stay"
