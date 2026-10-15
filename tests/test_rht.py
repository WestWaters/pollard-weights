"""pollard-rht: online block-RHT folds that stock llama.cpp (>= b11556, PR #29600) un-does at run time.

The contract being pinned here is the runtime's, read from llama.cpp src/llama-model.cpp
(load_hparams_hadamard / load_tensors_hadamard) and src/llama-graph.cpp (build_hadamard_input):

  * the graph feeds a folded weight H.(D.x): signs first, then the normalized Sylvester WHT applied
    block-wise along the input (last) dimension, entry (r, c) = (-1)^popcount(r & c) / sqrt(B);
  * so the stored weight must be W_f = W.D.H, and W_f.(H.D.x) == W.x exactly (H.H = I, D.D = I);
  * one global block size, one +-1 sign vector per INPUT WIDTH (INT32 arrays), weight names restricted
    to the build_lora_mm kinds, archs restricted to llama/qwen3/qwen3moe/qwen35/qwen35moe/qwen3next;
  * a tied model (no output.weight) uses manifest v2: tied_output=true and token_embd.weight listed as
    an inverse-after-lookup table.

An end-to-end KLD against the real runtime is in notes/rht-rungs.md (Qwen3-0.6B: mean KLD 0.000005).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


R = _load("pollard_rht", "tools/pollard_rht.py")


def runtime_hadamard(B):
    """The matrix load_tensors_hadamard writes, transcribed literally (parity of row & col)."""
    H = np.empty((B, B))
    for r in range(B):
        for c in range(B):
            H[r, c] = -1.0 if bin(r & c).count("1") & 1 else 1.0
    return H / np.sqrt(B)


@pytest.mark.parametrize("B", [32, 64, 128, 256])
def test_sylvester_is_the_runtime_matrix(B):
    H = R.sylvester(B)
    assert np.array_equal(H, runtime_hadamard(B))
    assert np.allclose(H, H.T)
    assert np.allclose(H @ H, np.eye(B), atol=1e-12)


@pytest.mark.parametrize("B,width", [(64, 256), (128, 384), (256, 768)])
def test_fold_is_undone_by_the_runtime_input_transform(B, width):
    rng = np.random.default_rng(3)
    W = rng.standard_normal((5, width)).astype(np.float32)
    x = rng.standard_normal(width).astype(np.float32)
    d = R.signs_for(width, 7).astype(np.float32)
    H = R.sylvester(B)
    Wf = R.fold_rows(W, d, H)
    got = Wf @ R.runtime_input(x, d, H.astype(np.float32))
    assert np.allclose(got, W @ x, rtol=1e-4, atol=1e-4)
    # the fold must actually change the weights (identity signs + identity H would also pass above)
    assert not np.allclose(Wf, W)


def test_fold_handles_expert_stacks():
    rng = np.random.default_rng(4)
    W = rng.standard_normal((3, 4, 256)).astype(np.float32)        # [experts, out, in]
    d = R.signs_for(256, 1).astype(np.float32)
    H = R.sylvester(128)
    Wf = R.fold_rows(W, d, H)
    x = rng.standard_normal(256).astype(np.float32)
    for e in range(3):
        assert np.allclose(Wf[e] @ R.runtime_input(x, d, H.astype(np.float32)), W[e] @ x, atol=1e-4)


def test_signs_are_deterministic_and_per_width():
    a, b = R.signs_for(1024, 1), R.signs_for(1024, 1)
    assert np.array_equal(a, b) and set(np.unique(a)) == {-1, 1}
    assert not np.array_equal(R.signs_for(1024, 1), R.signs_for(1024, 2))
    assert not np.array_equal(R.signs_for(2048, 1)[:1024], a)


def _tiny(path, arch="qwen3", tied=False, hidden=256, ff=384):
    rng = np.random.default_rng(0)
    w = gguf.GGUFWriter(str(path), arch)
    w.add_block_count(1)
    t = {
        "token_embd.weight": rng.standard_normal((64, hidden)).astype(np.float16),
        "blk.0.attn_norm.weight": np.ones(hidden, dtype=np.float32),
        "blk.0.attn_q.weight": rng.standard_normal((hidden, hidden)).astype(np.float16),
        "blk.0.attn_output.weight": rng.standard_normal((hidden, hidden)).astype(np.float16),
        "blk.0.ffn_up.weight": rng.standard_normal((ff, hidden)).astype(np.float16),
        "blk.0.ffn_down.weight": rng.standard_normal((hidden, ff)).astype(np.float16),
        "output_norm.weight": np.ones(hidden, dtype=np.float32),
    }
    if not tied:
        t["output.weight"] = rng.standard_normal((64, hidden)).astype(np.float16)
    for k, v in t.items():
        w.add_tensor(k, v)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return t


def _fields(path):
    r = gguf.GGUFReader(str(path))
    return r, {k: f for k, f in r.fields.items()}


def test_fold_writes_the_manifest_the_loader_reads(tmp_path):
    from gguf.constants import GGUFValueType as VT
    src, out = tmp_path / "m.gguf", tmp_path / "m-rht.gguf"
    orig = _tiny(src)
    p = R.fold(str(src), str(out), block=128, scope="all", seed=5, quiet=True)
    r, f = _fields(out)
    key = "prism.hadamard."
    assert f[key + "version"].types == [VT.UINT32] and f[key + "version"].contents() == 1
    assert f[key + "block_size"].contents() == 128
    assert f[key + "transform"].contents() == "normalized-sylvester-walsh-hadamard"
    assert f[key + "axis"].contents() == "input-last-dimension"
    assert f[key + "sign_mode"].contents() == "explicit"
    assert key + "tied_output" not in f, "v1 must not carry tied_output (the loader rejects v1+tied)"
    names = f[key + "weight_names"].contents()
    assert set(names) == {"blk.0.attn_q.weight", "blk.0.attn_output.weight", "blk.0.ffn_up.weight",
                          "blk.0.ffn_down.weight", "output.weight"}
    assert f[key + "sign_widths"].types == [VT.ARRAY, VT.INT32], "loader's get_arr<int32_t> needs INT32"
    widths = f[key + "sign_widths"].contents()
    vals = f[key + "sign_values"].contents()
    assert widths == [256, 384] and len(vals) == 256 + 384
    assert vals[:256] == [int(v) for v in R.signs_for(256, 5)]
    assert all(w % 128 == 0 for w in widths)

    ten = {t.name: t for t in r.tensors}
    # unfolded tensors are byte-identical; folded ones are W.D.H stored in the source dtype
    assert np.array_equal(np.asarray(ten["token_embd.weight"].data), orig["token_embd.weight"])
    assert np.array_equal(np.asarray(ten["blk.0.attn_norm.weight"].data), orig["blk.0.attn_norm.weight"])
    H = R.sylvester(128)
    for n in names:
        W = orig[n].astype(np.float32)
        d = R.signs_for(W.shape[1], 5).astype(np.float32)
        want = R.fold_rows(W, d, H).astype(np.float16)
        got = np.asarray(ten[n].data)
        assert got.dtype == np.float16 and np.array_equal(got, want), n
    assert p["canary_worst"] < 5e-3


def test_tied_model_gets_v2_latent_embedding(tmp_path):
    src, out = tmp_path / "t.gguf", tmp_path / "t-rht.gguf"
    orig = _tiny(src, tied=True)
    R.fold(str(src), str(out), block=64, quiet=True)
    r, f = _fields(out)
    k = "prism.hadamard."
    assert f[k + "version"].contents() == 2 and f[k + "tied_output"].contents() is True
    assert f[k + "inverse_weight_names"].contents() == ["token_embd.weight"]
    assert "token_embd.weight" not in f[k + "weight_names"].contents(), \
        "the loader adds token_embd itself for tied_output; listing it would be a duplicate"
    E = np.asarray({t.name: t for t in r.tensors}["token_embd.weight"].data).astype(np.float32)
    # a lookup row e_f, put back through the runtime inverse (h = s * (H z)), must give the original row
    d = R.signs_for(256, 1).astype(np.float32)
    H = R.sylvester(64).astype(np.float32)
    back = (E.reshape(64, -1, 64) @ H).reshape(64, 256) * d[None, :]
    assert np.allclose(back, orig["token_embd.weight"].astype(np.float32), atol=2e-2)


def test_scope_down_only(tmp_path):
    src, out = tmp_path / "m.gguf", tmp_path / "m-down.gguf"
    _tiny(src)
    R.fold(str(src), str(out), scope="down", quiet=True)
    _, f = _fields(out)
    assert f["prism.hadamard.weight_names"].contents() == ["blk.0.ffn_down.weight"]
    assert f["prism.hadamard.sign_widths"].contents() == [384]


def test_refuses_archs_the_runtime_would_refuse(tmp_path):
    src = tmp_path / "q2.gguf"
    _tiny(src, arch="qwen2")
    with pytest.raises(ValueError, match="qwen2"):
        R.fold(str(src), str(tmp_path / "x.gguf"), quiet=True)


def test_refuses_to_fold_twice(tmp_path):
    src, out = tmp_path / "m.gguf", tmp_path / "m-rht.gguf"
    _tiny(src)
    R.fold(str(src), str(out), quiet=True)
    with pytest.raises(ValueError, match="already"):
        R.fold(str(out), str(tmp_path / "again.gguf"), quiet=True)


def test_binary_detection(tmp_path):
    new, old = tmp_path / "llama-perplexity", tmp_path / "old" / "llama-perplexity"
    old.parent.mkdir()
    new.write_bytes(b"\0junk\0prism.hadamard.version\0more")
    old.write_bytes(b"\0junk\0general.architecture\0")
    assert R.binary_supports_prism(str(new))
    assert not R.binary_supports_prism(str(old))
    assert not R.binary_supports_prism(str(tmp_path / "missing"))


def test_ggufcheck_reports_the_version_requirement(tmp_path):
    import pollard_ggufcompat as gc
    src, out = tmp_path / "m.gguf", tmp_path / "m-rht.gguf"
    _tiny(src)
    R.fold(str(src), str(out), block=128, quiet=True)
    assert gc.runtime_of(str(src), offline=True)[0] == "stock"
    rt, why = gc.runtime_of(str(out), offline=True)
    assert rt == "prism", (rt, why)
    ph = why["prism_hadamard"]
    assert ph["min_build"] == "b11556" and ph["block_size"] == 128 and ph["weights"] == 5
    assert "ik_llama" in ph["note"]
    # read_header keeps its old two-value shape for every existing caller
    arch, counts = gc.read_header(str(out))
    assert arch == "qwen3" and counts


def test_precondition_candidate_names():
    pc = _load("pollard_precondition", "tools/pollard_precondition.py")
    assert pc.rht_spec("rht-B128") == ("all", 128)
    assert pc.rht_spec("rht-B64") == ("all", 64)
    assert pc.rht_spec("rht-B256") == ("all", 256)
    assert pc.rht_spec("rht-down-only") == ("down", 128)
    assert pc.rht_spec("rht-down-B256") == ("down", 256)
    assert pc.rht_spec("rht-attn-B64") == ("attn", 64)
    assert pc.rht_spec("rot-block32") is None and pc.rht_spec("none") is None
