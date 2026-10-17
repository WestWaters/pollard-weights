"""The streamed HF lanes (pollard_stream): every transformers lane -- SmoothQuant, GPTQ, MX, EXLR8, the verify gate --
runs one decoder block at a time read from disk, resolves the text backbone of VLM / omni / custom-code / *_puzzle
checkpoints from config.json, dequantizes an already-quantized source on read, passes NVIDIA's NVFP4 tensors through
byte for byte, and runs a checkpoint's own code only with consent. Born 2026-10-16: LFM2.5 (48 GB on a 32 GB box),
Nemotron-3.5-Bolt (custom code, NVFP4 + ggml IQ2_XXS QAT) and clef-omni (Qwen3-Omni-MoE) were HELD on every HF lane.

The numpy-only checks run in every CI shape (packing, consent, the source reader, the MX passthrough, the memory
planner, the EXL3 decision); the rest need torch + transformers and build tiny random checkpoints (no download)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

import pollard_gptqpack as G  # noqa: E402
import pollard_stream as S  # noqa: E402


# ------------------------------------------------------------------------------------------------ GPTQ packing

def _autogptq_pack3(intweight):
    """AutoGPTQ's own 3-bit pack loop (qlinear pack, bits == 3), restated as the reference."""
    intweight = intweight.astype(np.uint32)
    q = np.zeros((intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32)
    i = row = 0
    while row < q.shape[0]:
        for j in range(i, i + 10):
            q[row] |= intweight[j] << (3 * (j - i))
        i += 10
        q[row] |= intweight[i] << 30
        row += 1
        q[row] |= (intweight[i] >> 2) & 1
        i += 1
        for j in range(i, i + 10):
            q[row] |= intweight[j] << (3 * (j - i) + 1)
        i += 10
        q[row] |= intweight[i] << 31
        row += 1
        q[row] |= (intweight[i] >> 1) & 0x3
        i += 1
        for j in range(i, i + 10):
            q[row] |= intweight[j] << (3 * (j - i) + 2)
        i += 10
        row += 1
    return q


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_gptq_bit_stream_round_trips(bits):
    x = np.random.default_rng(bits).integers(0, 2 ** bits, (128, 24))
    words = G.pack_stream(x, bits)
    assert words.dtype == np.int32 and words.shape == (128 * bits // 32, 24)
    assert np.array_equal(G.unpack_stream(words, bits, 128), x)
    if bits in (2, 4, 8):                         # 32/bits codes per word, code k at bit bits*k
        per = 32 // bits
        want = sum(x[k::per].astype(np.uint64) << np.uint64(bits * k) for k in range(per)).astype(np.uint32)
        assert np.array_equal(words.view(np.uint32), want)


def test_gptq_3bit_is_autogptqs_layout():
    x = np.random.default_rng(3).integers(0, 8, (96, 16))
    assert np.array_equal(G.pack_stream(x, 3).view(np.uint32), _autogptq_pack3(x))


def test_gptq_module_tensors_are_v1_sym():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 16, (64, 256))
    scales = rng.uniform(0.01, 0.1, (64, 2)).astype(np.float32)
    t = G.pack_module(codes, scales, 4, 128)
    assert t["qweight"].shape == (32, 64) and t["qzeros"].shape == (2, 8) and t["scales"].shape == (2, 64)
    assert t["scales"].dtype == np.float16 and np.array_equal(t["g_idx"], np.arange(256) // 128)
    assert np.all(G.unpack_stream(t["qzeros"].T, 4, 64) == 7), "v1 stores zero - 1 (sym 4-bit zero = 8)"
    w = G.dequant_module(t, 4, 256)
    s16 = scales.astype(np.float16).astype(np.float32)
    assert np.allclose(w, (codes - 8) * np.repeat(s16, 128, 1))


def test_gptq_dynamic_rules_put_exclusions_first():
    kb = {"model.layers.3.self_attn.q_proj.weight": 8, "model.layers.3.mlp.experts.0.up_proj.weight": 8,
          "model.layers.4.self_attn.q_proj.weight": 4}
    dyn = G.dynamic_rules(kb, {"model.layers.3.mlp.gate"}, 4)
    keys = list(dyn)
    assert keys[0] == r"-:^model\.layers\.3\.mlp\.gate$", "vLLM takes the first match: the router exclusion first"
    import re
    plus = [k for k in keys if k.startswith("+:")]
    assert any(re.match(k[2:], "model.layers.3.self_attn.q_proj") for k in plus)
    assert any(re.match(k[2:], "model.layers.3.mlp.experts.0.up_proj") for k in plus)
    assert not any(re.match(k[2:], "model.layers.4.self_attn.q_proj") for k in plus)
    assert G.side_of("language_model.backbone.layers.1.mixer.experts.3.up_proj.weight") == "ffn"
    assert G.side_of("language_model.backbone.layers.0.mixer.in_proj.weight") == "attn"
    assert G.side_of("model.layers.2.feed_forward.experts.1.w1.weight") == "ffn"


def test_gptq_never_quantizes_a_router_or_a_norm():
    assert G.quantizable("m.layers.0.mlp.gate.weight", (8, 128), 128)[0] is False
    assert G.quantizable("m.layers.0.mixer.gate.weight", (8, 128), 128)[0] is False
    assert G.quantizable("m.layers.0.input_layernorm.weight", (128,), 128)[0] is False
    assert G.quantizable("m.layers.0.self_attn.q_proj.weight", (128, 96), 128)[0] is False
    assert G.quantizable("m.layers.0.self_attn.q_proj.weight", (128, 256), 128)[0] is True


# ------------------------------------------------------------------------------------------------ consent

def _custom_dir(tmp):
    d = tmp / "custom"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "my_arch", "auto_map": {
        "AutoConfig": "configuration_my.MyConfig", "AutoModelForCausalLM": "modeling_my.MyForCausalLM"}}))
    (d / "configuration_my.py").write_text("x = 1\n")
    (d / "modeling_my.py").write_text("x = 2\n")
    (d / "helper_my.py").write_text("x = 3\n")
    return d


def test_custom_code_never_runs_silently(tmp_path, monkeypatch):
    d = _custom_dir(tmp_path)
    monkeypatch.delenv("POLLARD_TRUST_REMOTE_CODE", raising=False)
    files, missing = S.remote_code_files(str(d))
    assert files == ["configuration_my.py", "helper_my.py", "modeling_my.py"] and not missing
    with pytest.raises(S.RemoteCodeRefused) as e:                      # pytest's stdin is no terminal
        S.consent(str(d), "auto")
    assert "modeling_my.py" in str(e.value) and "--trust-remote-code on" in str(e.value)
    said = []
    assert S.consent(str(d), "on", log=said.append) is True and "helper_my.py" in said[0]
    assert S.consent(str(d), "off", log=lambda *_: None) is False
    assert S.consent(str(d), "auto", ask=lambda q: "y", log=lambda *_: None) is True
    with pytest.raises(S.RemoteCodeRefused):
        S.consent(str(d), "auto", ask=lambda q: "")                    # the default answer is No
    monkeypatch.setenv("POLLARD_TRUST_REMOTE_CODE", "1")
    assert S.consent(str(d), "auto", log=lambda *_: None) is True
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert S.consent(str(plain), "auto") is False


def test_workspace_resolver_follows_the_consent_rules(tmp_path, monkeypatch):
    import pollard_workspace as ws
    d = _custom_dir(tmp_path)
    monkeypatch.delenv("POLLARD_TRUST_REMOTE_CODE", raising=False)
    with pytest.raises(SystemExit):
        ws.resolve_trust_remote_code(str(d), "auto")
    assert ws.resolve_trust_remote_code(str(d), "on") is True
    assert ws.resolve_trust_remote_code(str(d), "off") is False


def test_config_is_read_as_json_never_by_running_it(tmp_path):
    import pollard_workspace as ws
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "nemotron_h_omni", "auto_map": {"A": "x.Y"},
                                               "llm_config": {"num_hidden_layers": 88, "n_routed_experts": 240,
                                                              "model_type": "nemotron_h_puzzle"}}))
    t = ws.text_config_dict(ws.config_dict(str(d)))
    assert t["num_hidden_layers"] == 88 and t["n_routed_experts"] == 240
    import pollard_export as E
    assert E.detect_moe(str(d)) == (True, 88)


# ------------------------------------------------------------------------------------------------ the source reader

def _bf16(a):
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def quantized_source(d, rng=None):
    """A checkpoint the way NVIDIA's QAT release stores it, numpy only: a modelopt NVFP4 linear, ggml Q8_0 / IQ2_XXS
    linears (U8 blocks inside the safetensors), plain bf16 tensors, a vision tower. Returns {key: float32 truth}."""
    import pollard_mx as X
    from pollard_mxfp6 import _f32_to_e4m3_bytes
    from gguf import quants, GGMLQuantizationType as Q
    rng = rng or np.random.default_rng(0)
    os.makedirs(d, exist_ok=True)
    t, ref, ql = {}, {}, {}
    w = (rng.standard_normal((64, 128)) * 0.05).astype(np.float32)
    packed, sc, g, _ = X.nvfp4_encode(w, "6")
    m = "model.layers.0.mixer.in_proj"
    t[m + ".weight"] = ("U8", packed)
    t[m + ".weight_scale"] = ("F8_E4M3", _f32_to_e4m3_bytes(sc))
    t[m + ".weight_scale_2"] = ("F32", np.array(1.0 / g, np.float32))
    ref[m + ".weight"] = X.nvfp4_decode(packed, sc, g)
    ql[m] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    q8 = quants.quantize((rng.standard_normal((32, 128)) * 0.05).astype(np.float32), Q.Q8_0)
    m = "model.layers.0.mixer.out_proj"
    t[m + ".weight"] = ("U8", q8.reshape(32, 4, 34))
    ref[m + ".weight"] = quants.dequantize(q8, Q.Q8_0)
    ql[m] = {"quant_algo": "Q8_0", "packing": "ggml"}
    raw = rng.integers(0, 256, (32, 1, 66), dtype=np.uint8)
    raw[..., :2] = np.full((32, 1), 0.004, np.float16).view(np.uint8).reshape(32, 1, 2)
    m = "model.layers.0.mixer.experts.0.up_proj"
    t[m + ".weight"] = ("U8", raw)
    ref[m + ".weight"] = quants.dequantize(raw.reshape(32, -1), Q.IQ2_XXS)
    ql[m] = {"quant_algo": "IQ2_XXS", "packing": "ggml"}
    for k, shp in (("model.layers.0.norm.weight", (128,)), ("model.layers.0.mixer.q_proj.weight", (64, 128)),
                   ("vision_model.blocks.0.qkv.weight", (48, 32)), ("lm_head.weight", (100, 128))):
        a = rng.standard_normal(shp).astype(np.float32)
        t[k] = ("BF16", _bf16(a))
        ref[k] = (_bf16(a).astype(np.uint32) << 16).view(np.float32)
    S._st_write(os.path.join(d, "model.safetensors"), t, {"format": "pt"})
    json.dump({"model_type": "x", "num_hidden_layers": 1, "hidden_size": 128,
               "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                       "quantized_layers": ql}}, open(os.path.join(d, "config.json"), "w"))
    return ref


def test_reader_dequantizes_nvfp4_and_ggml_blocks_on_read(tmp_path):
    pytest.importorskip("gguf")
    ref = quantized_source(str(tmp_path / "src"))
    src = S.Source(str(tmp_path / "src"))
    assert {m["algo"] for m in src.modules.values()} == {"nvfp4", "ggml:Q8_0", "ggml:IQ2_XXS"}
    assert "model.layers.0.mixer.in_proj.weight_scale" not in src.logical, "aux tensors hide behind the module"
    for k, v in ref.items():
        assert src.shape(k) == v.shape, k
        assert np.allclose(src.read(k), v, atol=1e-6), k


def test_mx_nvfp4_rung_passes_nvidias_nvfp4_through_byte_for_byte(tmp_path):
    """The MX lane's NVFP4 rung on an already-NVFP4 source: the codes and E4M3 block scales are copied, the tensor
    scale converted from modelopt's weight_scale_2; every other linear quantized from its dequantized values and
    recorded; the tower copied; the ggml lm_head written as bf16 (compressed-tensors holds no ggml)."""
    pytest.importorskip("gguf")
    import pollard_mxfp6 as M6
    ref = quantized_source(str(tmp_path / "src"))
    src = S.Source(str(tmp_path / "src"))
    amax = {k: 4.0 for k in src.logical if ".layers." in k and k.endswith(".weight") and len(src.shape(k)) == 2}
    out = str(tmp_path / "nvfp4")
    meta = M6.build_scheme(str(tmp_path / "src"), out, "NVFP4", input_amax=amax, log=lambda *_: None)
    assert meta["passthrough"] == ["model.layers.0.mixer.in_proj"]
    assert meta["dequantized_source"] == {"ggml:Q8_0": 1, "ggml:IQ2_XXS": 1}
    o = S.Source(out)
    m = "model.layers.0.mixer.in_proj"
    assert np.array_equal(o.read_raw(m + ".weight_packed")[1], src.read_raw(m + ".weight")[1])
    assert np.array_equal(o.read_raw(m + ".weight_scale")[1], src.read_raw(m + ".weight_scale")[1])
    assert np.allclose(o.read(m + ".weight"), ref[m + ".weight"], rtol=1e-6, atol=1e-7)
    assert np.array_equal(o.read_raw("vision_model.blocks.0.qkv.weight")[1],
                          src.read_raw("vision_model.blocks.0.qkv.weight")[1])
    cfg = json.load(open(os.path.join(out, "config.json")))
    q = cfg["quantization_config"]
    assert q["quant_method"] == "compressed-tensors" and q["format"] == "nvfp4-pack-quantized"
    assert q["config_groups"]["group_0"]["input_activations"]["dynamic"] == "local"
    assert o.read_raw(m + ".input_global_scale")[1][0] == np.float32(448.0 * 6.0 / 4.0)
    dec = M6.dequantized_weights(out)
    for k in ("model.layers.0.mixer.out_proj.weight", "model.layers.0.mixer.experts.0.up_proj.weight"):
        rel = np.linalg.norm(dec[k] - ref[k]) / np.linalg.norm(ref[k])
        assert rel < 0.2, (k, rel)


def test_the_card_says_what_each_rung_was_built_from(tmp_path):
    pytest.importorskip("gguf")
    import pollard_mxfp6 as M6
    import pollard_card as C
    ref = quantized_source(str(tmp_path / "src"))
    out = str(tmp_path / "M-Pollard-MX-NVFP4")
    M6.build_scheme(str(tmp_path / "src"), out, "NVFP4", input_amax={k: 1.0 for k in ref}, log=lambda *_: None)
    lines = C.provenance_lines("mx", [{"path": out}])
    assert lines[0] == "How these rungs were made:"
    text = " ".join(lines)
    assert "`nvfp4/`" in text and "1 linears kept byte for byte" in text and "Q8_0" in text and "IQ2_XXS" in text
    ex = tmp_path / "M-Pollard-EXL3-4.0bpw"
    ex.mkdir()
    (ex / "pollard-exl3.json").write_text(json.dumps({"input": "text", "why": "its language model is Qwen3MoE"}))
    assert "language model only" in " ".join(C.provenance_lines("exl3", [{"path": str(ex)}]))
    assert C.provenance_lines("gptq", [{"path": str(tmp_path / "nope")}]) == []


@pytest.mark.parametrize("scheme", ["MXFP4", "FP8", "FP8_DYNAMIC"])
def test_mx_scheme_rungs_decode_close_to_the_source(tmp_path, scheme):
    pytest.importorskip("gguf")
    import pollard_mxfp6 as M6
    ref = quantized_source(str(tmp_path / "src"))
    amax = {k: 3.0 for k in ref if ".layers." in k and len(ref[k].shape) == 2}
    out = str(tmp_path / scheme)
    meta = M6.build_scheme(str(tmp_path / "src"), out, scheme, input_amax=amax, log=lambda *_: None)
    assert meta["scheme"] == scheme and not meta["passthrough"]
    for k, v in M6.dequantized_weights(out).items():
        assert np.linalg.norm(v - ref[k]) / np.linalg.norm(ref[k]) < (0.15 if scheme == "MXFP4" else 0.05), k


def test_a_scheme_that_needs_activation_scales_refuses_without_them(tmp_path):
    pytest.importorskip("gguf")
    import pollard_mxfp6 as M6
    quantized_source(str(tmp_path / "src"))
    with pytest.raises(SystemExit, match="activation scales"):
        M6.build_scheme(str(tmp_path / "src"), str(tmp_path / "o"), "NVFP4", log=lambda *_: None)


# ------------------------------------------------------------------------------------------------ memory planner

class _FakeSrc:
    def __init__(self, shapes):
        self.s = shapes
        self.logical = {k: ("raw", k) for k in shapes}

    def shape(self, k):
        return self.s[k]


class _FakeBB:
    """A 4-block MoE backbone's shapes, nothing else (the planner reads shapes only)."""

    def __init__(self, E=64, H=2048, I=1536, L=4, V=65536):
        sh = {"model.embed_tokens.weight": (V, H), "model.norm.weight": (H,)}
        self.layer_keys = []
        for i in range(L):
            ks = []
            for n, s in ((f"model.layers.{i}.self_attn.q_proj.weight", (H, H)),
                         (f"model.layers.{i}.self_attn.k_proj.weight", (H // 8, H))):
                sh[n] = s
                ks.append(n)
            for e in range(E):
                for p, s in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
                    n = f"model.layers.{i}.mlp.experts.{e}.{p}.weight"
                    sh[n] = s
                    ks.append(n)
            self.layer_keys.append(ks)
        self.src = _FakeSrc(sh)
        self.n_layers = L
        self.outer_keys = ["model.embed_tokens.weight", "model.norm.weight"]

    def hidden_size(self):
        return 2048

    def text_config(self):
        return type("C", (), {"head_dim": 128, "num_attention_heads": 16})()


def test_memory_plan_counts_one_block_and_shrinks_to_the_budget():
    bb = _FakeBB()
    p = S.plan_memory(bb, 128, 2048)
    assert p["layer_weights_gb"] > 2 and p["statistics_gb"] > 1.5 and p["activations_gb"] > 2
    fit = S.fit_memory(bb, 128, 2048, budget_gb=8.0, log=lambda *_: None)
    assert fit["peak_gb"] <= 8.0 and (fit["nsamples"] < 128 or fit["expert_chunks"] > 1)
    with pytest.raises(S.OutOfMemoryBudget, match="budget is 2.0 GB"):
        S.fit_memory(bb, 128, 2048, budget_gb=2.0, log=lambda *_: None)


def test_free_ram_guard_stops_cleanly(monkeypatch):
    monkeypatch.setattr(S, "free_ram_gb", lambda: 3.0)
    with pytest.raises(S.OutOfMemoryBudget, match="stopping before layer 7"):
        S.free_ram_guard(2.0, what="layer 7")
    S.free_ram_guard(0.5, what="layer 7")
    assert S.peak_rss_bytes() >= 0


# ------------------------------------------------------------------------------------------------ EXL3 decision

def _cfg(tmp, name, cfg):
    d = tmp / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


def test_exl3_input_plan_from_config_alone(tmp_path):
    import pollard_exl3 as E
    assert E.exl3_input_plan(_cfg(tmp_path, "lfm", {"architectures": ["Lfm2MoeForCausalLM"],
                                                    "model_type": "lfm2_moe"}))["mode"] == "direct"
    om = E.exl3_input_plan(_cfg(tmp_path, "omni", {"architectures": ["Qwen3OmniMoeForConditionalGeneration"],
                                                   "model_type": "qwen3_omni_moe"}))
    assert om["mode"] == "text" and om["arch"] == "Qwen3MoeForCausalLM"
    bolt = E.exl3_input_plan(_cfg(tmp_path, "bolt", {
        "architectures": ["NemotronH_Omni_Reasoning_V3"], "model_type": "nemotron_h_omni",
        "llm_config": {"architectures": ["NemotronHPuzzleForCausalLM"], "model_type": "nemotron_h_puzzle",
                       "moe_latent_size": 1024, "block_configs": [{"block_type": "mamba"}]}}))
    assert bolt["mode"] == "hold"
    assert "NemotronH_Omni_Reasoning_V3" in bolt["why"] and "latent MoE" in bolt["why"] \
        and "per-layer block configs" in bolt["why"]


# ------------------------------------------------------------------------------------------------ torch + transformers

def _need_torch():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    pytest.importorskip("gguf")


ROWS = [list(np.random.default_rng(i).integers(0, 250, 40)) for i in range(4)]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    _need_torch()
    import tiny_models as TM
    import make_tiny_bolt as B
    root = tmp_path_factory.mktemp("stream")
    out = {"llama": TM.llama(str(root / "llama")), "moe": TM.qwen3_moe(str(root / "moe")),
           "lfm2": TM.lfm2_moe(str(root / "lfm2")), "omni": TM.omni(str(root / "omni")),
           "custom": TM.custom_code(str(root / "custom")), "root": root}
    out["bolt_ref"] = B.make(str(root / "bolt"))
    out["bolt"] = str(root / "bolt")
    return out


@pytest.mark.parametrize("kind", ["llama", "moe", "lfm2", "omni", "bolt"])
def test_one_block_at_a_time_matches_the_whole_model(tiny, kind):
    """The engine's recorded block calls replayed one block at a time give the whole model's last hidden state --
    dense, MoE, hybrid conv/attention, the omni thinker, and Bolt's puzzle NemotronH over a dequantized source --
    and never more than one block is in memory."""
    import torch
    bb = S.Backbone(tiny[kind], "off", log=lambda *_: None)
    eng = S.Engine(bb, act_dtype=torch.float32, log=lambda *_: None).record(ROWS[:2])
    for i in range(bb.n_layers):
        eng.load_layer(i, guard_gb=False)
        live = [j for j, L in enumerate(bb.layers) if any(not p.is_meta for p in L.parameters())]
        assert live == [i], f"blocks in memory: {live}"
        eng.hidden = eng.run_layer(i)
        bb.free_layer(i)
    ref = S.Backbone(tiny[kind], "off", log=lambda *_: None)
    ref.load(list(ref.kmap), "cpu", torch.float32)
    got = {}
    norm, _ = S._final_head(ref)
    h = norm.register_forward_pre_hook(lambda m, a: got.setdefault("x", a[0]))
    with torch.no_grad():
        ref.text_root(input_ids=torch.tensor([ROWS[0]]), use_cache=False)
    h.remove()
    assert torch.allclose(eng.hidden[0], got["x"].float(), atol=2e-4, rtol=1e-3)


def test_backbones_resolve_without_running_checkpoint_code(tiny):
    d = S.Backbone(tiny["omni"], "off", log=lambda *_: None).describe()
    assert d["root"] == "Qwen3OmniMoeThinkerForConditionalGeneration" and d["layer_prefix"] == "thinker.model.layers."
    b = S.Backbone(tiny["bolt"], "off", log=lambda *_: None)
    assert b.describe()["root"] == "NemotronHForCausalLM" and b.Q == "language_model." and not b.remote
    assert "one block config per layer" in b.how
    assert len(b.layers[1].mixer.experts.up_proj) == 4 and len(b.layers[3].mixer.experts.up_proj) == 6
    assert any(k.startswith("vision_model.") for k in b.other_keys) and "mlp1.1.weight" in b.other_keys
    st = S.static_backbone(tiny["bolt"])
    assert st["route"] == "puzzle"


def test_custom_code_needs_consent_and_then_streams(tiny, monkeypatch):
    monkeypatch.delenv("POLLARD_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.delenv("POLLARD_TEST_CUSTOM_CODE_RAN", raising=False)
    with pytest.raises(S.RemoteCodeRefused):
        S.Backbone(tiny["custom"], "auto", log=lambda *_: None)
    assert "POLLARD_TEST_CUSTOM_CODE_RAN" not in os.environ, "refused = the checkpoint's code never ran"
    bb = S.Backbone(tiny["custom"], "on", log=lambda *_: None)
    assert bb.remote and os.environ.get("POLLARD_TEST_CUSTOM_CODE_RAN") == "1"
    out = str(tiny["root"] / "custom-gptq")
    side = G.build_stream(tiny["custom"], out, None, low=4, group=32, rows=ROWS, nsamples=4, seqlen=40,
                          trust_remote_code="on", log=lambda *_: None)
    assert side["quantized"] == 14 and side["backbone"]["remote_code"] is True


def test_streamed_smoothing_is_an_identity_and_the_overlay_is_the_same_model(tiny):
    import torch, transformers as T
    import pollard_hf_smooth as H
    cal = tiny["root"] / "cal.txt"
    cal.write_text("the quick brown fox jumps over the lazy dog\n" * 80)
    full = str(tiny["root"] / "llama-sm")
    sm = H.smooth_stream(tiny["llama"], full, str(cal), rows=4, cols=48, log=lambda *_: None)
    assert sm["seams"] == 6 and sm["layout"] == "checkpoint"
    a = T.AutoModelForCausalLM.from_pretrained(tiny["llama"], dtype=torch.float32).eval()
    b = T.AutoModelForCausalLM.from_pretrained(full, dtype=torch.float32).eval()
    x = torch.tensor([ROWS[0]])
    with torch.no_grad():
        assert torch.allclose(a(x).logits, b(x).logits, atol=1e-4)
    assert not torch.equal(a.model.layers[0].self_attn.q_proj.weight, b.model.layers[0].self_attn.q_proj.weight)
    ov = str(tiny["root"] / "llama-ov")
    H.smooth_stream(tiny["llama"], ov, str(cal), rows=4, cols=48, overlay=True, log=lambda *_: None)
    assert sum(os.path.getsize(os.path.join(ov, f)) for f in os.listdir(ov) if f.endswith(".safetensors")) < 20000
    s_ov, s_full = S.Source(ov), S.Source(full)
    for k in s_full.logical:
        assert np.array_equal(s_ov.read(k), s_full.read(k)), k
    mat = H.materialize(ov, str(tiny["root"] / "llama-mat"), log=lambda *_: None)
    s_mat = S.Source(mat)
    assert all(np.array_equal(s_mat.read_raw(k)[1], s_full.read_raw(k)[1]) for k in s_full.logical)


def test_streamed_gptq_on_omni_keeps_the_towers_and_decodes_to_the_rung(tiny):
    out = str(tiny["root"] / "omni-gptq")
    side = G.build_stream(tiny["omni"], out, None, alloc={0: {"attn": 8, "ffn": 8}}, low=4, group=32, rows=ROWS,
                          nsamples=4, seqlen=40, trust_remote_code="off", log=lambda *_: None)
    assert side["bits_hist"] == {8: 16, 4: 16} and side["hessians"] == {"hessian": 32}
    assert set(side["skipped"]) == {"thinker.model.layers.0.mlp.gate", "thinker.model.layers.1.mlp.gate"}
    src, o = S.Source(tiny["omni"]), S.Source(out)
    for k in ("thinker.audio_tower.layers.0.self_attn.q_proj.weight", "thinker.visual.blocks.0.attn.qkv.weight",
              "talker.model.layers.0.self_attn.q_proj.weight", "thinker.lm_head.weight"):
        assert np.array_equal(o.read_raw(k)[1], src.read_raw(k)[1]), f"{k} must be copied through untouched"
    q = json.load(open(os.path.join(out, "config.json")))["quantization_config"]
    assert q["quant_method"] == "gptq" and q["sym"] and not q["desc_act"] and list(q["dynamic"])[0].startswith("-:")
    r = S.stream_fidelity(tiny["omni"], out, ROWS[2:], trust_remote_code="off", log=lambda *_: None)
    assert r["n"] == 80 and 0 < r["kld"] < 0.05
    same = S.stream_fidelity(tiny["omni"], tiny["omni"], ROWS[2:], trust_remote_code="off", log=lambda *_: None)
    assert same["kld"] < 1e-9 and same["top1"] == 100.0


def test_bolt_lanes_quantize_from_the_dequantized_source_and_pass_nvfp4_through(tiny):
    import pollard_mxfp6 as M6
    bolt = tiny["bolt"]
    gq = str(tiny["root"] / "bolt-gptq")
    side = G.build_stream(bolt, gq, None, low=4, group=128, rows=ROWS, nsamples=4, seqlen=40,
                          trust_remote_code="off", log=lambda *_: None)
    assert side["dequantized_source"] == {"nvfp4": 2, "ggml:IQ2_XXS": 24, "ggml:Q8_0": 4}
    o = S.Source(gq)
    assert o.dtype("language_model.lm_head.weight") == "BF16", "the Q8_0 head is written as bf16 in a GPTQ rung"
    amax, cal = S.collect_input_amax(bolt, rows=ROWS, nsamples=4, seqlen=40, trust_remote_code="off",
                                     log=lambda *_: None)
    assert cal["linears"] >= 34
    mx = str(tiny["root"] / "bolt-nvfp4")
    meta = M6.build_scheme(bolt, mx, "NVFP4", input_amax=amax, log=lambda *_: None)
    assert meta["passthrough"] == ["language_model.backbone.layers.0.mixer.in_proj",
                                   "language_model.backbone.layers.0.mixer.out_proj"]
    src, m = S.Source(bolt), S.Source(mx)
    for mod in meta["passthrough"]:
        assert np.array_equal(m.read_raw(mod + ".weight_packed")[1], src.read_raw(mod + ".weight")[1])
        assert np.array_equal(m.read_raw(mod + ".weight_scale")[1], src.read_raw(mod + ".weight_scale")[1])
    assert np.array_equal(m.read_raw("mlp1.3.weight")[1], src.read_raw("mlp1.3.weight")[1])
    r = S.stream_fidelity(bolt, mx, ROWS[:2], trust_remote_code="off", log=lambda *_: None)
    assert r["n"] == 80 and np.isfinite(r["kld"])


def test_text_backbone_of_the_omni_is_qwen3_moe_token_for_token(tiny):
    import torch, transformers as T
    bb = S.Backbone(tiny["omni"], "off", log=lambda *_: None)
    bb.load(list(bb.kmap), "cpu", torch.float32)
    out = S.extract_text(tiny["omni"], str(tiny["root"] / "omni-text"), log=lambda *_: None)
    m = T.AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    assert type(m).__name__ == "Qwen3MoeForCausalLM"
    x = torch.tensor([ROWS[1]])
    with torch.no_grad():
        assert torch.allclose(bb.root(input_ids=x).logits, m(x).logits, atol=1e-4)
    assert not any(k.startswith(("thinker.audio", "talker")) for k in S.Source(out).logical)


def test_exl3_converts_the_omni_language_model_lifted_out_of_its_smoothing_overlay(tiny):
    """EXL3 of clef-omni: exllamav3 has no Qwen3-Omni, so pollard-exl3 hands it the thinker's language model as
    Qwen3MoeForCausalLM -- lifted straight out of the smoothing overlay (one copy, text only, smoothed)."""
    import torch, transformers as T
    import pollard_exl3 as E
    import pollard_hf_smooth as H
    cal = tiny["root"] / "cal-omni.txt"
    cal.write_text("the quick brown fox jumps over the lazy dog\n" * 80)
    ov = str(tiny["root"] / "omni-ov")
    H.smooth_stream(tiny["omni"], ov, str(cal), rows=2, cols=32, overlay=True, log=lambda *_: None)
    src, plan = E.exl3_input(ov, log=lambda *_: None)
    assert plan["mode"] == "text" and src.endswith(".text")
    m = T.AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
    bb = S.Backbone(tiny["omni"], "off", log=lambda *_: None)
    bb.load(list(bb.kmap), "cpu", torch.float32)
    x = torch.tensor([ROWS[3]])
    with torch.no_grad():
        a, b = bb.root(input_ids=x).logits, m(x).logits
    # smoothing is an identity; what is left is the bf16 rounding of the folded q/k/v and norms
    assert (a - b).abs().max() < 0.02 * a.abs().max()
    q = S.Source(src).read("model.layers.0.self_attn.q_proj.weight")
    assert not np.array_equal(q, S.Source(tiny["omni"]).read("thinker.model.layers.0.self_attn.q_proj.weight"))


def test_streamed_exlr8_encode_and_reference_decode(tiny, monkeypatch):
    import torch
    import pollard_exlr8 as X
    import tiny_models as TM
    d = TM.llama(str(tiny["root"] / "llama128"), layers=1, hidden=128)
    cal = tiny["root"] / "cal8.txt"
    cal.write_text("the quick brown fox jumps over the lazy dog\n" * 40)
    store = str(tiny["root"] / "store")
    meta = X.encode_model(d, store, widths=[2], calib=str(cal), device="cpu", cal_rows=2, cal_seqlen=32,
                          cal_tokens=64, align=64, trust_remote_code="off", log=lambda *_: None)
    assert meta["complete"] and meta["engine"] == "pollard-stream" and meta["tensor_prefix"] == "model.layers."
    assert len(meta["tensors"]) == 5 and all(t["hessian_tokens"] > 0 for t in meta["tensors"].values())
    rung = str(tiny["root"] / "rung")
    X.build_rung(store, X.plan_rung(meta, 2.0, None, [2]), rung, align=64)
    m = X.load_model(rung, "cpu", torch.float32)
    with torch.no_grad():
        assert torch.isfinite(m(torch.tensor([ROWS[0][:16]])).logits).all()
    r = S.stream_fidelity(d, rung, [ROWS[0][:24]], trust_remote_code="off", log=lambda *_: None)
    assert np.isfinite(r["kld"]) and r["n"] == 24
