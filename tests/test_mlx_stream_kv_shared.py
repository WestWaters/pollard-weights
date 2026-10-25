"""The layer-major streamed forward handles blocks that return (h, shared_kv, offset) and read an EARLIER block's
K/V (Gemma 4's KV-shared layers): the streamed logits equal the model's own forward, exactly."""
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import pollard_mlx as M  # noqa: E402


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Linear(d, d)
        self.k = nn.Linear(d, d)

    def __call__(self, h, mask=None, cache=None, per_layer_input=None, shared_kv=None, offset=None):
        kv = shared_kv if shared_kv is not None else self.k(h)
        return h + mx.tanh(self.a(h) + kv * 0.5), kv, (offset or 0) + h.shape[1]


class Inner(nn.Module):
    def __init__(self, V=64, d=16):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, d)
        self.layers = [Block(d) for _ in range(4)]
        self.previous = [0, 1, 0, 1]                       # blocks 2 and 3 read blocks 0 and 1's K/V

    def __call__(self, ids):
        h = self.embed_tokens(ids)
        inter = [(None, None)] * len(self.layers)
        for i, layer in enumerate(self.layers):
            kvs, off = inter[self.previous[i]] if i >= 2 else (None, None)
            h, kvs, off = layer(h, None, None, per_layer_input=None, shared_kv=kvs, offset=off)
            inter[i] = (kvs, off)
        return h @ self.embed_tokens.weight.T


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = Inner()


def test_streamed_equals_full_with_kv_sharing(monkeypatch):
    mx.random.seed(0)
    model = Toy()
    mx.eval(model.parameters())
    rows = [[1, 5, 9, 2, 7], [3, 3, 8]]
    want = [np.array(model.language_model(mx.array([r]))[0]) for r in rows]
    monkeypatch.setattr(M, "_lazy_load", lambda path, engine: model)
    s = M.StreamedLM("toy", "mlx_lm", log=lambda *a, **k: None)
    assert s.stateful, "a block taking shared_kv is run as (h, kvs, offset)"
    s.run(rows)
    for b, w in enumerate(want):
        assert np.allclose(s.logits(b), w, atol=1e-6)
    assert s.kv == {}, "every K/V no later block reads is dropped"


def test_refs_swap_and_collect():
    st = {(0, 1): ("KV", 7)}
    r = M._KVRef(0, 1, "kvs")
    assert M._swap_refs({"shared_kv": r, "offset": M._KVRef(0, 1, "offset")}, st) == {"shared_kv": "KV", "offset": 7}
    assert M._refs_in([1, {"x": r}], set()) == {(0, 1)}
    assert (r + 3) is r, "offset arithmetic on a record pass keeps the reference"
