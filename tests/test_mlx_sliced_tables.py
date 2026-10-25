"""A big stored table (Gemma 4's 2.8B-param per-layer embedding: 5.6 GB of bf16) is encoded slice by slice, read
from the safetensors file, never evaluated whole -- and the bytes are the one-shot encode's, exactly."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import pollard_mlx as M  # noqa: E402


def _table(tmp_path, rows=600, K=256, seed=0):
    rng = np.random.default_rng(seed)
    w = mx.array(rng.standard_normal((rows, K)).astype(np.float32) * 0.05).astype(mx.bfloat16)
    p = str(tmp_path / "t.safetensors")
    mx.save_safetensors(p, {"table": w, "other": mx.ones((4, 64))})
    return p


def _job(w, bits, imp=None):
    return {"w": w, "bits": bits, "gs": 64, "imp": imp, "path": "t", "out": None, "left": 3}


@pytest.mark.parametrize("bits,method,with_imp", [(4, "search", False), (3, "search", True), (8, "minmax", False),
                                                  (2, "search", False)])
def test_sliced_encode_is_byte_identical_to_the_one_shot(tmp_path, monkeypatch, bits, method, with_imp):
    p = _table(tmp_path)
    monkeypatch.setattr(M, "SLICE_MIN_BYTES", 0)                 # every tensor counts as big
    monkeypatch.setattr(M, "ENCODE_CHUNK_ELEMS", 256 * 32)       # encode_affine works 32 rows at a time
    monkeypatch.setattr(M, "SLICE_BYTES", 256 * 2 * 96)          # 96 rows (3 inner chunks) per slice -> 7 slices
    imp = np.abs(np.random.default_rng(1).standard_normal(256)).astype(np.float32) if with_imp else None
    q = M.StreamedQuantize({}, method)
    with q:
        w = mx.load(p)["table"]
        src = q.file_source(w)
        assert src is not None and src[1] == "table", "a plain read of one stored tensor is found in its file"
        sliced = q.encode_sliced(_job(w, bits, imp), src)
        assert q.stats["sliced"] == 1
    q2 = M.StreamedQuantize({}, method)
    with q2:
        w2 = mx.load(p)["table"]
        monkeypatch.setattr(M, "SLICE_MIN_BYTES", 1 << 62)       # the one-shot path
        whole = q2.encode(_job(w2, bits, imp))
    assert isinstance(sliced[0], np.ndarray), "packed weights come back as numpy (no second copy)"
    assert np.array_equal(sliced[0], np.array(whole[0])), "packed weights identical"
    for a, b in ((sliced[1], whole[1]), (sliced[2], whole[2])):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(np.array(a.view(mx.uint16)), np.array(b.view(mx.uint16))), "scales / biases identical"


def test_only_plain_reads_are_sliced(tmp_path, monkeypatch):
    p = _table(tmp_path)
    monkeypatch.setattr(M, "SLICE_MIN_BYTES", 0)
    q = M.StreamedQuantize({}, "search")
    with q:
        w = mx.load(p)["table"]
        assert q.file_source(w) is not None
        assert q.file_source(w * 2) is None, "a computed tensor is not a file read"
        assert q.file_source(w.T) is None, "a transposed one neither"
        assert q.file_source(w.astype(mx.bfloat16)) is not None, "a same-dtype cast still reads the file as stored"
    assert mx.load is not None and q.loads == [], "mx.load restored, registry dropped"


def test_the_registry_never_holds_the_arrays(tmp_path, monkeypatch):
    p = _table(tmp_path)
    monkeypatch.setattr(M, "SLICE_MIN_BYTES", 0)
    q = M.StreamedQuantize({}, "search")
    with q:
        mx.load(p)
        assert q.loads and all(not isinstance(x, mx.array) for e in q.loads for x in e), \
            "file, key, shape, Load ids -- an evaluated tensor is never kept alive by the registry"


def test_encode_routes_big_plain_tables_to_the_sliced_path(tmp_path, monkeypatch):
    p = _table(tmp_path)
    monkeypatch.setattr(M, "SLICE_MIN_BYTES", 0)
    q = M.StreamedQuantize({}, "search")
    with q:
        out = q.encode(_job(mx.load(p)["table"], 4))
    assert q.stats["sliced"] == 1 and isinstance(out[0], np.ndarray)


def test_plan_counts_a_slice_not_the_whole_table(tmp_path, monkeypatch):
    p = _table(tmp_path, rows=4096, K=256)
    import json
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    monkeypatch.setattr(M, "SLICE_MIN_BYTES", 1 << 20)          # the 2 MB table counts as big
    monkeypatch.setattr(M, "SLICE_BYTES", 1 << 16)
    plan = M.plan_build(str(tmp_path), low=4, high=8)
    n = 4096 * 256
    assert plan["largest_bytes"] == pytest.approx((1 << 16) * 4 + n * 8 / 8)
