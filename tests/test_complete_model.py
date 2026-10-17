"""Every lane ships the WHOLE model (owner, 2026-10-16: "I don't understand why any lane is text only"): the language
model in the lane's format, and every tower / projector / talker / code2wav / side weight plus the processor files of
the source. pollard-complete is the gate; release runs it per rung, pollard-verify folds it into the publish gate.

The first half is numpy-only (every CI shape): hand-written safetensors folders shaped like Qwen3-Omni and
Nemotron-3.5-Bolt. The second half runs the real streamed lanes on the tiny omni fixture (torch + transformers)."""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

import pollard_complete as PC  # noqa: E402


def _st(path, tensors):
    """A safetensors file from {name: (dtype, shape)} with zero bytes, written by hand (no safetensors package)."""
    size = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U8": 1}
    hdr, off = {}, 0
    for k, (dt, shape) in tensors.items():
        n = int(np.prod(shape)) * size[dt]
        hdr[k] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(hdr).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + bytes(off))


OMNI_TEXT = {"thinker.model.embed_tokens.weight": ("BF16", (64, 16)),
             "thinker.model.layers.0.self_attn.q_proj.weight": ("BF16", (16, 16)),
             "thinker.model.layers.0.mlp.experts.0.up_proj.weight": ("BF16", (8, 16)),
             "thinker.model.norm.weight": ("BF16", (16,)),
             "thinker.lm_head.weight": ("BF16", (64, 16))}
OMNI_OTHER = {"thinker.visual.blocks.0.attn.qkv.weight": ("BF16", (48, 16)),
              "thinker.visual.merger.mlp.0.weight": ("BF16", (64, 64)),
              "thinker.audio_tower.layers.0.self_attn.q_proj.weight": ("BF16", (16, 16)),
              "thinker.audio_tower.proj2.weight": ("BF16", (16, 16)),
              "talker.model.layers.0.self_attn.q_proj.weight": ("BF16", (16, 16)),
              "code2wav.decoder.0.conv.weight": ("BF16", (8, 4, 7))}
OMNI_FILES = ("preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
              "chat_template.jinja")


def _omni(tmp_path, joint_head=False):
    d = tmp_path / "omni"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": ["Qwen3OmniMoeForConditionalGeneration"],
                                               "model_type": "qwen3_omni_moe",
                                               "thinker_config": {"text_config": {"num_hidden_layers": 1}}}))
    for f in OMNI_FILES:
        (d / f).write_text("{}")
    (d / "tokenizer.json").write_text("{}")
    _st(d / "model.safetensors", {**OMNI_TEXT, **OMNI_OTHER})
    if joint_head:
        _st(d / "joint_head.safetensors", {"layers.0.linear1.weight": ("F32", (8, 8))})
    return str(d)


def _bolt(tmp_path):
    d = tmp_path / "bolt"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": ["NemotronH_Omni_Reasoning_V3"],
                                               "model_type": "nemotron_h_omni", "llm_config": {}}))
    (d / "preprocessor_config.json").write_text("{}")
    (d / "image_processing_nemotron_h_omni.py").write_text("# processor code\n")
    _st(d / "model-00001-of-00002.safetensors", {
        "language_model.backbone.layers.0.mixer.in_proj.weight": ("BF16", (32, 16)),
        "language_model.lm_head.weight": ("BF16", (64, 16)),
        "vision_model.radio_model.model.blocks.0.attn.qkv.weight": ("BF16", (48, 16)),
        "mlp1.1.weight": ("BF16", (16, 64))})
    _st(d / "model-summary-idxs.safetensors", {"vision_model.summary_idxs": ("I32", (2,))})
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "language_model.backbone.layers.0.mixer.in_proj.weight": "model-00001-of-00002.safetensors",
        "language_model.lm_head.weight": "model-00001-of-00002.safetensors",
        "vision_model.radio_model.model.blocks.0.attn.qkv.weight": "model-00001-of-00002.safetensors",
        "mlp1.1.weight": "model-00001-of-00002.safetensors",
        "vision_model.summary_idxs": "model-summary-idxs.safetensors"}}))
    return str(d)


# ------------------------------------------------------------------------------------------------ inventory

def test_inventory_finds_every_non_text_component_of_the_omni(tmp_path):
    inv = PC.inventory(_omni(tmp_path, joint_head=True))
    c = inv["components"]
    assert set(c) == {"vision", "audio", "speech_out"}
    assert len(c["vision"]) == 2, "the merger inside the vision tower belongs to the tower"
    assert set(c["speech_out"]) == {"talker.model.layers.0.self_attn.q_proj.weight", "code2wav.decoder.0.conv.weight"}
    assert not any(k.startswith("thinker.model.") or k == "thinker.lm_head.weight" for v in c.values() for k in v)
    assert set(inv["processor_files"]) == set(OMNI_FILES)
    assert inv["side_weights"] == {"joint_head.safetensors": ["layers.0.linear1.weight"]}


def test_inventory_of_bolt_reads_the_index_and_its_processor_code(tmp_path):
    inv = PC.inventory(_bolt(tmp_path))
    assert set(inv["components"]) == {"vision", "projector"}
    assert "vision_model.summary_idxs" in inv["components"]["vision"], "an indexed side shard is a model shard"
    assert inv["side_weights"] == {}
    assert set(inv["processor_files"]) == {"preprocessor_config.json", "image_processing_nemotron_h_omni.py"}


def test_a_text_model_has_nothing_to_carry(tmp_path):
    d = tmp_path / "llama"
    d.mkdir()
    (d / "config.json").write_text("{}")
    _st(d / "model.safetensors", {"model.layers.0.self_attn.q_proj.weight": ("BF16", (8, 8)),
                                  "lm_head.weight": ("BF16", (8, 8))})
    assert PC.inventory(str(d))["components"] == {}
    out = tmp_path / "out"
    out.mkdir()
    _st(out / "model.safetensors", {"lm_head.weight": ("BF16", (8, 8))})
    assert PC.check(str(d), str(out), "gptq")["ok"]


# ------------------------------------------------------------------------------------------------ the gate

def _text_only_rung(tmp_path, name="rung", quantized=True):
    out = tmp_path / name
    out.mkdir()
    t = {"thinker.lm_head.weight": ("BF16", (64, 16))}
    t.update({"thinker.model.layers.0.self_attn.q_proj.qweight": ("I32", (2, 16))} if quantized else {})
    _st(out / "model.safetensors", t)
    (out / "config.json").write_text("{}")
    return out


def test_a_text_only_rung_fails_with_every_missing_part_listed(tmp_path):
    src = _omni(tmp_path, joint_head=True)
    res = PC.check(src, str(_text_only_rung(tmp_path)), "gptq")
    assert not res["ok"]
    text = "\n".join(res["missing"])
    for part in ("vision: 0/2", "audio: 0/2", "speech_out: 0/2", "file preprocessor_config.json",
                 "file processor_config.json", "file video_preprocessor_config.json", "file chat_template.jinja",
                 "side weights joint_head.safetensors"):
        assert part in text, f"{part!r} not reported in {res['missing']}"
    assert PC.main(["--source", src, "--out", str(tmp_path / "rung"), "--lane", "gptq"]) == 1


def test_quantized_tower_parts_count_as_carried(tmp_path):
    """A lane may quantize a tower deliberately (EXL3 -vb, MLX --vision-bits): `<module>.trellis` / `.qweight`
    carries `<module>.weight`."""
    src = _omni(tmp_path)
    out = _text_only_rung(tmp_path)
    keys = {k: v for k, v in OMNI_OTHER.items() if "visual" not in k}
    keys["thinker.visual.blocks.0.attn.qkv.trellis"] = ("I32", (4, 4))
    keys["thinker.visual.merger.mlp.0.suh"] = ("F16", (64,))
    _st(out / "towers.safetensors", keys)
    for f in OMNI_FILES:
        (out / f).write_text("{}")
    res = PC.check(src, str(out), "exl3")
    assert res["ok"], res["missing"]


def test_fill_writes_the_missing_parts_byte_exact_and_then_the_rung_is_complete(tmp_path):
    src = _omni(tmp_path, joint_head=True)
    out = _text_only_rung(tmp_path)
    wrote = PC.fill(src, str(out), "gptq")
    assert {"components-vision.safetensors", "components-audio.safetensors", "components-speech_out.safetensors",
            "joint_head.safetensors", *OMNI_FILES} == set(wrote)
    assert PC.check(src, str(out), "gptq")["ok"]
    assert PC.fill(src, str(out), "gptq") == [], "a complete rung is left alone"
    hdr = PC._st_keys(str(out / "components-speech_out.safetensors"))
    assert hdr["code2wav.decoder.0.conv.weight"]["shape"] == [8, 4, 7]


def test_the_mlx_lane_is_checked_by_component_presence(tmp_path):
    """mlx-vlm re-lays tensors (conv layouts, its own module names): a component counts as carried when the output has
    it at all, not name for name."""
    src = _omni(tmp_path)
    out = tmp_path / "mlx"
    out.mkdir()
    _st(out / "model.safetensors", {"thinker.vision_tower.blocks.0.attn.qkv.weight": ("BF16", (48, 16)),
                                    "thinker.audio_tower.layers.0.self_attn.q_proj.weight": ("BF16", (16, 16)),
                                    "language_model.model.layers.0.self_attn.q_proj.scales": ("F16", (16, 1))})
    for f in OMNI_FILES:
        (out / f).write_text("{}")
    res = PC.check(src, str(out), "mlx")
    assert not res["ok"] and res["missing"] == ["speech_out: source 2 tensors, output 0"]


def test_gguf_needs_an_mmproj_declaring_each_encoder_and_a_sidecar_for_the_talker(tmp_path):
    gguf = pytest.importorskip("gguf")
    src = _omni(tmp_path)
    out = tmp_path / "gguf"
    out.mkdir()
    (out / "omni-Q4_K_M.gguf").write_bytes(b"")
    res = PC.check(src, str(out), "gguf")
    assert not res["ok"] and any(m.startswith("vision") for m in res["missing"]) \
        and any(m.startswith("audio") for m in res["missing"])
    w = gguf.GGUFWriter(str(out / "mmproj-omni-BF16.gguf"), "clip")
    w.add_bool("clip.has_vision_encoder", True)
    w.add_bool("clip.has_audio_encoder", False)
    w.add_tensor("v.patch_embd.weight", np.zeros((4, 4), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    res = PC.check(src, str(out), "gguf")
    assert [m.split(":")[0] for m in res["missing"]] == ["audio", "speech_out"], res["missing"]
    assert PC.fill(src, str(out), "gguf") == ["components-speech_out.safetensors"], \
        "the talker goes beside the GGUF; the towers are the mmproj's job"
    assert [m.split(":")[0] for m in PC.check(src, str(out), "gguf")["missing"]] == ["audio"]


# ------------------------------------------------------------------------------------------------ lane policy

def test_mx_never_quantizes_the_talker_code2wav_or_bolts_projector():
    import pollard_mxfp6 as M6
    for k in ("talker.model.layers.0.self_attn.q_proj.weight", "code2wav.decoder.layers.1.fc.weight",
              "mlp1.1.weight", "language_model.model.layers.0.sound_projection.linear.weight"):
        ok, why = M6.quantizable(k, (64, 64))
        assert not ok, f"{k} would be quantized"
    assert M6.quantizable("thinker.model.layers.0.self_attn.q_proj.weight", (64, 64))[0]


def test_gptq_dynamic_excludes_every_copied_component_first():
    import pollard_gptqpack as G
    roots = G.component_roots(["thinker.visual.blocks.0.attn.qkv.weight", "thinker.audio_tower.proj2.weight",
                               "talker.model.layers.0.self_attn.q_proj.weight", "code2wav.decoder.0.conv.weight",
                               "mlp1.1.weight", "thinker.model.norm.weight"])
    assert roots == ["code2wav", "mlp1", "talker", "thinker.audio_tower", "thinker.visual"]
    rules = list(G.dynamic_rules({"thinker.model.layers.0.self_attn.q_proj.weight": 8}, {"thinker.model.layers.0.mlp.gate"},
                                 4, roots))
    assert rules[:5] == [r"-:^code2wav\..*", r"-:^mlp1\..*", r"-:^talker\..*", r"-:^thinker\.audio_tower\..*",
                         r"-:^thinker\.visual\..*"]
    assert rules[5].startswith("-:") and rules[-1].startswith("+:")


def test_mlx_keeps_the_speech_output_at_source_precision_and_holds_without_mlx_vlm(tmp_path, monkeypatch):
    import builtins
    import pollard_mlx as M
    assert M.bits_for("talker.model.layers.0.self_attn.q_proj", {0: {"attn": 4, "ffn": 4}}, True, 64) is False
    assert M.bits_for("thinker.model.layers.0.self_attn.q_proj", {0: {"attn": 4, "ffn": 4}}, True, 64)["bits"] == 4
    real = builtins.__import__

    def no_vlm(name, *a, **k):
        if name.startswith("mlx_vlm"):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_vlm)
    eng, why = M.mlx_engine(_omni(tmp_path))
    assert eng == "hold" and "mlx-vlm" in why and "speech_out" in why
    d = tmp_path / "text"
    d.mkdir()
    (d / "config.json").write_text("{}")
    _st(d / "model.safetensors", {"model.layers.0.mlp.up_proj.weight": ("BF16", (8, 8))})
    assert M.mlx_engine(str(d)) == ("mlx_lm", None)


def test_verify_gate_fails_an_incomplete_rung_whatever_the_thresholds():
    import pollard_verify as V
    r = V.evaluate_gate({"kld": 0.01, "complete": False, "complete_missing": ["audio: 0/2 tensors"]},
                        {"kld_max": 0.1})
    assert not r["pass"]
    row = [c for c in r["checks"] if c["check"] == "complete model"][0]
    assert "audio: 0/2 tensors" in row["why"]
    assert V.evaluate_gate({"kld": 0.01, "complete": True}, {"kld_max": 0.1})["pass"]


def test_release_gates_every_rung_and_the_gguf_repo_on_completeness(tmp_path, monkeypatch):
    import pollard_release as R
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    src = _omni(tmp_path)
    c, _, steps = R.plan_steps(R.load_plan(None), src)
    by = {s.id: s for s in steps}
    got = [s.id for s in steps]
    rungs = [s.id[:-len("/build")] for s in steps if s.id.endswith("/build") and s.lane != "gguf" and s.rung]
    assert {r.split("/")[0] for r in rungs} >= {"gptq", "mx", "mlx", "exl3"}
    for rid in rungs:
        st = by[f"{rid}/complete"]
        assert st.kind == "gate" and st.cmd[:2] == ["pollard-complete", "--source"]
        assert got.index(f"{rid}/build") < got.index(f"{rid}/complete") < got.index(f"{rid}/verify")
    g = by["gguf/complete"]
    assert g.cmd[-2:] == ["gguf", "--fill"] and g.blocks == ["gguf"]
    assert "exl3/recompose" in by, "the omni EXL3 is recomposed into the complete checkpoint, not shipped as text"


# ------------------------------------------------------------------------------------------------ EXL3 recompose

def test_exl3_recompose_ships_the_complete_omni_checkpoint(tmp_path):
    """exllamav3 converted the lifted-out Qwen3MoE; the recompose puts the quantized tensors back under thinker.*,
    restores the source's config (+ the EXL3 quantization_config) and adds every component at source precision --
    the layout exllamav3 itself writes when it has Qwen3OmniMoeForConditionalGeneration."""
    import pollard_exl3 as E
    src = _omni(tmp_path, joint_head=True)
    out = tmp_path / "exl3"
    out.mkdir()
    _st(out / "model.safetensors", {"model.embed_tokens.weight": ("F16", (64, 16)),
                                    "model.layers.0.self_attn.q_proj.trellis": ("I32", (1, 16)),
                                    "model.layers.0.self_attn.q_proj.suh": ("F16", (16,)),
                                    "model.norm.weight": ("F16", (16,)),
                                    "lm_head.trellis": ("I32", (4, 16))})
    q = {"quant_method": "exl3", "bits": 4.0, "tensor_storage": {
        "lm_head": {"stored_tensors": {"lm_head.trellis": {}}},
        "model.layers.0.self_attn.q_proj": {"stored_tensors": {"model.layers.0.self_attn.q_proj.trellis": {}}}}}
    (out / "config.json").write_text(json.dumps({"architectures": ["Qwen3MoeForCausalLM"], "quantization_config": q}))
    (out / "quantization_config.json").write_text(json.dumps(q))
    res = E.recompose_complete(src, str(out), log=lambda *_: None)
    assert res["ok"]
    keys = set(PC._out_tensors(str(out)))
    assert {"thinker.model.layers.0.self_attn.q_proj.trellis", "thinker.lm_head.trellis",
            "thinker.model.embed_tokens.weight", *OMNI_OTHER} <= keys
    assert not any(k.startswith(("model.", "lm_head")) for k in keys)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["architectures"] == ["Qwen3OmniMoeForConditionalGeneration"] and "thinker_config" in cfg
    ts = cfg["quantization_config"]["tensor_storage"]
    assert set(ts) == {"thinker.lm_head", "thinker.model.layers.0.self_attn.q_proj"}
    assert "thinker.lm_head.trellis" in ts["thinker.lm_head"]["stored_tensors"]
    assert set(json.loads((out / "quantization_config.json").read_text())["tensor_storage"]) == set(ts)
    for f in (*OMNI_FILES, "joint_head.safetensors", "tokenizer.json"):
        assert (out / f).exists(), f


def test_exl3_plan_says_complete_not_text_only(tmp_path):
    import pollard_exl3 as E
    d = tmp_path / "o"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": ["Qwen3OmniMoeForConditionalGeneration"],
                                               "model_type": "qwen3_omni_moe"}))
    p = E.exl3_input_plan(str(d))
    assert p["mode"] == "text" and "a text-only EXL3" not in p["why"] and "complete" in p["why"]


# ------------------------------------------------------------------------------------------------ real lanes (torch)

ROWS = [list(np.random.default_rng(i).integers(0, 250, 40)) for i in range(4)]


@pytest.fixture(scope="module")
def omni(tmp_path_factory):
    for m in ("torch", "transformers", "tokenizers", "gguf"):
        pytest.importorskip(m)
    import tiny_models as TM
    root = tmp_path_factory.mktemp("complete")
    return {"root": root, "omni": TM.omni(str(root / "omni"))}


def test_streamed_gptq_of_the_omni_is_complete(omni):
    import pollard_gptqpack as G
    out = str(omni["root"] / "omni-gptq")
    G.build_stream(omni["omni"], out, None, low=4, group=32, rows=ROWS, nsamples=4, seqlen=40,
                   trust_remote_code="off", log=lambda *_: None)
    res = PC.check(omni["omni"], out, "gptq")
    assert res["ok"], res["missing"]
    dyn = json.load(open(os.path.join(out, "config.json")))["quantization_config"]["dynamic"]
    assert r"-:^talker\..*" in dyn and r"-:^thinker\.visual\..*" in dyn


def test_streamed_mx_of_the_omni_is_complete_and_leaves_the_talker_alone(omni):
    import pollard_mxfp6 as M6
    import pollard_stream as S
    out = str(omni["root"] / "omni-mxfp4")
    M6.build_scheme(omni["omni"], out, "MXFP4", log=lambda *_: None)
    res = PC.check(omni["omni"], out, "mx")
    assert res["ok"], res["missing"]
    src, o = S.Source(omni["omni"]), S.Source(out)
    k = "talker.model.layers.0.self_attn.q_proj.weight"
    assert np.array_equal(o.read_raw(k)[1], src.read_raw(k)[1]), "the talker is copied, not quantized"
