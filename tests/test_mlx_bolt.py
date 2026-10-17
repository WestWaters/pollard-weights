"""The MLX lane ships Nemotron-3.5-Bolt whole through mlx-vlm (nemotron_h_omni over a nemotron_h_puzzle decoder):
the hybrid Mamba2 / attention / latent-MoE decoder with its per-layer expert counts, the RADIO tower and the mlp1
projector, from either release -- the -BF16 one, or the 24GB one whose experts are ggml IQ2_XXS inside safetensors
(decoded to BF16 first: mlx-vlm reads float weights).

The first half is numpy-only (every CI shape, mlx absent): the engine decision against a stand-in mlx-vlm, the bit
policy on mlx-vlm's module names, the layer count of a config that only lists its blocks. The second half builds the
tiny Bolt fixture (torch + transformers) and, on Apple Silicon with an mlx-vlm that has Bolt, runs the lane."""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

import pollard_mlx as M  # noqa: E402


def _st(path, tensors):
    """A safetensors file from {name: (dtype, shape)} with zero bytes, written by hand."""
    size = {"BF16": 2, "F32": 4, "I64": 8}
    hdr, off = {}, 0
    for k, (dt, shape) in tensors.items():
        n = int(np.prod(shape)) * size[dt]
        hdr[k] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(hdr).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + bytes(off))


BLOCKS = [{"block_type": "mamba"},
          {"block_type": "moe", "moe_intermediate_size": 64, "n_routed_experts": 4, "num_experts_per_tok": 2},
          {"block_type": "attention"}]


def _bolt_dir(tmp_path):
    """config.json shaped like the release (no num_hidden_layers: the block list is the layer list) + a shard."""
    d = tmp_path / "bolt"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "nemotron_h_omni", "architectures": ["NemotronH_Omni_Reasoning_V3"],
        "auto_map": {"AutoConfig": "configuration_nemotron_h_omni.NemotronH_Omni_Reasoning_V3_Config"},
        "llm_config": {"model_type": "nemotron_h_puzzle", "num_experts": 512, "block_configs": BLOCKS,
                       "layers_block_type": [b["block_type"] for b in BLOCKS]},
        "vision_config": {"model_type": "radio"}}))
    _st(d / "model.safetensors", {"language_model.backbone.layers.0.mixer.in_proj.weight": ("BF16", (8, 8)),
                                  "vision_model.radio_model.model.blocks.0.attn.qkv.weight": ("BF16", (24, 8)),
                                  "mlp1.1.weight": ("BF16", (8, 32))})
    return d


def _fake_mlx_vlm(monkeypatch, per_layer):
    """Stand-in mlx_vlm modules: a Bolt-capable one (TextConfig.for_layer) or one that predates it."""
    class TextConfig:
        pass
    if per_layer:
        TextConfig.for_layer = lambda self, i: self
    pkg = types.ModuleType("mlx_vlm")
    pkg.__path__ = []
    utils = types.ModuleType("mlx_vlm.utils")
    utils.get_model_type = lambda cfg: {"nemotron_h_omni": "nemotron_h_nano_omni"}.get(cfg["model_type"],
                                                                                        cfg["model_type"])
    models = types.ModuleType("mlx_vlm.models")
    models.__path__ = []
    omni = types.ModuleType("mlx_vlm.models.nemotron_h_nano_omni")
    omni.TextConfig = TextConfig
    for name, mod in (("mlx_vlm", pkg), ("mlx_vlm.utils", utils), ("mlx_vlm.models", models),
                      ("mlx_vlm.models.nemotron_h_nano_omni", omni)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_bolt_goes_through_an_mlx_vlm_with_per_layer_block_configs(tmp_path, monkeypatch):
    d = _bolt_dir(tmp_path)
    _fake_mlx_vlm(monkeypatch, per_layer=True)
    assert M.mlx_engine(str(d)) == ("mlx_vlm", "nemotron_h_nano_omni")


def test_bolt_holds_on_an_mlx_vlm_that_builds_every_layer_from_one_config(tmp_path, monkeypatch):
    d = _bolt_dir(tmp_path)
    _fake_mlx_vlm(monkeypatch, per_layer=False)
    eng, why = M.mlx_engine(str(d))
    assert eng == "hold" and "block config" in why and "mlx-vlm" in why and "git+https" in why


def test_bolt_holds_on_an_mlx_vlm_without_nemotron_omni_and_says_which_to_install(tmp_path, monkeypatch):
    d = _bolt_dir(tmp_path)
    _fake_mlx_vlm(monkeypatch, per_layer=True)
    monkeypatch.delitem(sys.modules, "mlx_vlm.models.nemotron_h_nano_omni")
    sys.modules["mlx_vlm.utils"].get_model_type = lambda cfg: cfg["model_type"]     # 0.7.8: no remap
    eng, why = M.mlx_engine(str(d))
    assert eng == "hold" and "'nemotron_h_omni'" in why and "newer than 0.7.8" in why


def test_bolt_layer_count_comes_from_its_block_list(tmp_path):
    assert M.detect_moe(str(_bolt_dir(tmp_path))) == (True, 3)


def test_bolt_bit_policy_on_mlx_vlm_module_names():
    alloc = {i: {"attn": 6, "ffn": 3} for i in range(4)}
    bits = lambda p: M.bits_for(p, alloc, True, 64, None, 3, 8)["bits"]   # noqa: E731
    L = "language_model.backbone.layers."
    assert bits(L + "0.mixer.in_proj") == 6 and bits(L + "0.mixer.out_proj") == 6       # Mamba2: token mixer
    assert bits(L + "2.mixer.q_proj") == 6 and bits(L + "2.mixer.o_proj") == 6          # attention
    assert bits(L + "1.mixer.switch_mlp.fc1") == 3 and bits(L + "1.mixer.fc1_latent_proj") == 3
    assert bits(L + "1.mixer.shared_experts.up_proj") == 8                                # pinned high
    assert bits("language_model.backbone.embeddings") == 8 and bits("language_model.lm_head") == 8
    assert bits("mlp1.layers.1") == 8 and bits("mlp1.layers.3") == 8                      # the projector
    assert bits("vision_model.radio_model.model.blocks.0.attn.qkv") == 8
    assert M.bits_for("vision_model.radio_model.model.blocks.0.attn.qkv", alloc, True, 64, 6)["bits"] == 6


# ------------------------------------------------------------------------------------------------ tiny Bolt

def _need_fixture():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    pytest.importorskip("gguf")


@pytest.fixture(scope="module")
def bolt(tmp_path_factory):
    _need_fixture()
    import make_tiny_bolt as B
    root = tmp_path_factory.mktemp("mlxbolt")
    ref = B.make(str(root / "bolt-24gb"))
    B.make(str(root / "bolt-bf16"), encoding="bf16")
    return {"root": root, "q": str(root / "bolt-24gb"), "bf16": str(root / "bolt-bf16"), "ref": ref["float"]}


def test_the_24gb_release_decodes_to_the_bf16_release(bolt):
    """Every encoded linear (NVFP4 mamba projections, IQ2_XXS experts, Q8_0 latents / embeddings) decodes to the
    reference values; the tower, projector and summary_idxs come through; no quantization config is left."""
    import pollard_stream as S
    assert M.encoded_source(bolt["q"]) and not M.encoded_source(bolt["bf16"])
    out = M.decode_to_float(bolt["q"], str(bolt["root"] / "decoded"), log=lambda *_: None)
    got = S.Source(out)
    assert not got.modules
    assert set(got.logical) == set(S.Source(bolt["bf16"]).logical)
    for k, v in bolt["ref"].items():
        want = np.asarray(v, np.float32)
        if got.dtype(k) == "BF16":                                           # the reference, rounded to BF16
            want = (S.to_storage(want, "BF16").astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(got.read(k).reshape(want.shape), want, err_msg=k)
    dt, idx = got.read_raw("vision_model.summary_idxs")
    assert dt == "I64" and idx.tolist() == [0, 1]
    cfg = json.loads((Path(out) / "config.json").read_text())
    assert "quantization_config" not in cfg and cfg["llm_config"]["block_configs"]
    assert (Path(out) / "processor_config.json").exists() and (Path(out) / "tokenizer.json").exists()


def _bolt_mlx():
    if not (importlib.util.find_spec("mlx") and importlib.util.find_spec("mlx_vlm")):
        pytest.skip("mlx / mlx-vlm not installed (Apple Silicon lane)")
    try:
        from mlx_vlm.models.nemotron_h_nano_omni import TextConfig
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"this mlx-vlm has no Nemotron omni ({e})")
    if not hasattr(TextConfig, "for_layer"):
        pytest.skip("this mlx-vlm predates Nemotron-3.5-Bolt (per-layer block configs)")
    from mlx_vlm.models.nemotron_h_nano_omni import VisionConfig
    if "mlp_ratio" not in VisionConfig.__dataclass_fields__:
        # the released tower is 1280 wide x4 = the 5120 such an mlx-vlm assumes; the tiny one is not
        pytest.skip("this mlx-vlm sizes the RADIO MLP at 5120 whatever the config's mlp_ratio (tiny tower needs it)")


def _reference_logits(src, ids):
    import torch
    import pollard_stream as S
    bb = S.Backbone(src, "off", log=lambda *_: None)
    bb.load(list(bb.kmap), "cpu", torch.float32)
    with torch.no_grad():                                      # transformers' NemotronH, one config per layer
        return bb.root(input_ids=torch.tensor(ids), use_cache=False).logits.float().numpy()


@pytest.mark.parametrize("release", ["q", "bf16"])
def test_mlx_lane_ships_bolt_whole_and_matches_the_reference(bolt, release, monkeypatch):
    _bolt_mlx()
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm.utils import load_model
    out = bolt["root"] / f"mlx-{release}-8bit"
    monkeypatch.setattr(sys, "argv", ["pollard-mlx", "--model", bolt[release], "--out", str(out), "--bits", "8",
                                      "--range", "minmax"])
    M.main()                                                   # exits on an incomplete build (pollard-complete)
    assert not Path(str(out) + ".float-src").exists(), "the BF16 staging copy is removed"
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "nemotron_h_omni" and cfg["quantization"]["bits"] == 8
    assert "quant_method" not in cfg.get("quantization_config", {}), "the modelopt config is not carried over"
    model = load_model(out)
    params = dict(tree_flatten(model.parameters()))
    assert any(k.startswith("vision_model.radio_model.model.blocks.0.") for k in params)
    assert "mlp1.layers.1.scales" in params, "the projector is quantized (held at 8 bits), not dropped"
    layers = model.language_model.layers
    assert [layers[i].mixer.switch_mlp.fc1.weight.shape[0] for i in (1, 3)] == [4, 6]   # per-layer expert counts
    rng = np.random.default_rng(0)
    ids = rng.integers(3, 250, size=(1, 16))
    expected = _reference_logits(bolt[release], ids)
    got = np.array(model.language_model(mx.array(ids)).logits.astype(mx.float32))
    rel = np.abs(got - expected).max(-1)[0] / np.abs(expected).max(-1)[0]     # per position
    # 8-bit + bf16 noise ~3%; the fixture's random router is near-tied, so a position may pick another expert
    assert np.median(rel) < 0.05 and (rel > 0.1).sum() <= 2, rel
    assert (got.argmax(-1) == expected.argmax(-1)).mean() >= 0.75
    pix = mx.array(rng.standard_normal((1, 3, 64, 64)).astype(np.float32))
    feats = model.extract_feature(pix)
    assert feats.shape == (1, 4, 256) and bool(mx.all(mx.isfinite(feats)))


def test_mlx_lane_4bit_bolt_generates(bolt, monkeypatch):
    _bolt_mlx()
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.models.nemotron_h_nano_omni import NemotronHNanoOmniProcessor
    out = bolt["root"] / "mlx-q-4bit"
    monkeypatch.setattr(sys, "argv", ["pollard-mlx", "--model", bolt["q"], "--out", str(out), "--bits", "4"])
    M.main()
    model, processor = load(str(out))                          # mlx-vlm's own processor, no checkpoint code
    assert isinstance(processor, NemotronHNanoOmniProcessor)
    # greedy decode with the hybrid cache (Mamba2 state + KV) == greedy without a cache. The fixture's byte-level
    # tokenizer is smaller than its vocab, so this decodes token ids rather than text.
    lm = model.language_model
    ids = processor.tokenizer("hello there")["input_ids"]
    cache, seq = lm.make_cache(), list(ids)
    logits = lm(mx.array([seq]), cache=cache).logits
    for _ in range(6):
        seq.append(int(mx.argmax(logits[0, -1]).item()))
        logits = lm(mx.array([seq[-1:]]), cache=cache).logits
    nocache = list(ids)
    for _ in range(6):
        nocache.append(int(mx.argmax(lm(mx.array([nocache])).logits[0, -1]).item()))
    assert seq == nocache
