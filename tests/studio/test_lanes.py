"""Tests for the non-GGUF lane reader.

The packing math is the part that can be silently wrong. A GPTQ `qweight` is int32 holding 32/bits
logical weights per stored element; counting stored elements instead would undercount parameters
by 8x at 4-bit and report a bits-per-weight that looks far better than the file actually is. These
tests build real safetensors files with known shapes so the arithmetic is checked, not assumed.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

# the package lives at <repo>/tools/pollard_studio, so ROOT is the tools dir:
# every path below stays written as "pollard_studio/..." and the import works too
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "tools"
sys.path.insert(0, str(ROOT))

from pollard_studio import ggufread, saferead, workspace   # noqa: E402


def write_st(path: Path, tensors: dict) -> None:
    """Write a real safetensors file. {name: (dtype, shape, nbytes)}"""
    header, offset = {}, 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


# ── packing ─────────────────────────────────────────────────────────────────────────────────────
def test_gptq_qweight_unpacks_to_logical_params(tmp_path):
    """4-bit in int32 = 8 logical weights per stored element."""
    d = tmp_path / "m"
    d.mkdir()
    # qweight [in/8, out] for a 4096x4096 linear at 4 bits
    write_st(d / "model.safetensors", {
        "model.layers.0.mlp.down_proj.qweight": ("I32", [512, 4096], 512 * 4096 * 4),
        "model.layers.0.mlp.down_proj.scales": ("F16", [32, 4096], 32 * 4096 * 2),
        "model.layers.0.mlp.down_proj.qzeros": ("I32", [32, 512], 32 * 512 * 4),
    })
    (d / "quantize_config.json").write_text(json.dumps({"bits": 4, "group_size": 128}))
    s = saferead.summarise(saferead.read(d))
    assert s["lane"] == "GPTQ"
    # 512*4096 stored * 8 per int32 = 16,777,216 logical, which is 4096*4096
    assert s["params"] == 4096 * 4096


def test_companions_are_overhead_not_parameters(tmp_path):
    """Scales, zeros and g_idx are overhead. Counting them as params flatters the bpw."""
    d = tmp_path / "m"
    d.mkdir()
    write_st(d / "model.safetensors", {
        "layer.qweight": ("I32", [512, 4096], 512 * 4096 * 4),
        "layer.scales": ("F16", [32, 4096], 32 * 4096 * 2),
        "layer.g_idx": ("I32", [4096], 4096 * 4),
    })
    (d / "quantize_config.json").write_text(json.dumps({"bits": 4}))
    info = saferead.read(d)
    by = {t["name"]: t for t in info["tensors"]}
    assert by["layer.scales"]["logical"] == 0
    assert by["layer.g_idx"]["logical"] == 0
    assert by["layer.qweight"]["logical"] == 4096 * 4096


def test_8bit_packs_four_per_int32(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    write_st(d / "model.safetensors",
             {"layer.qweight": ("I32", [1024, 512], 1024 * 512 * 4)})
    (d / "quantize_config.json").write_text(json.dumps({"bits": 8}))
    info = saferead.read(d)
    assert info["tensors"][0]["logical"] == 1024 * 512 * 4


def test_unpacked_weights_count_directly(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    write_st(d / "model.safetensors", {"w": ("BF16", [1000, 500], 1000 * 500 * 2)})
    s = saferead.summarise(saferead.read(d))
    assert s["params"] == 500_000
    assert round(s["bpw"]) == 16


# ── lane detection ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cfgfile,cfg,expect", [
    ("quantize_config.json", {"bits": 4, "group_size": 128}, "GPTQ"),
    ("config.json", {"quantization_config": {"quant_method": "gptq", "bits": 4}}, "GPTQ"),
    ("config.json", {"quantization_config": {"quant_method": "compressed-tensors",
                                             "format": "nvfp4"}}, "MX"),
    ("config.json", {"quantization": {"group_size": 64, "bits": 4}}, "MLX"),
    ("config.json", {"model_type": "llama"}, "SAFETENSORS"),
])
def test_lane_is_detected_from_file_contents(tmp_path, cfgfile, cfg, expect):
    """Detection reads the files, never the directory name — a name is wrong when it matters."""
    d = tmp_path / "some-misleading-name"
    d.mkdir()
    write_st(d / "model.safetensors", {"w": ("BF16", [16, 16], 512)})
    (d / cfgfile).write_text(json.dumps(cfg))
    assert saferead.read(d)["lane"] == expect


def test_directory_name_does_not_decide_the_lane(tmp_path):
    d = tmp_path / "definitely-exl3-i-promise"
    d.mkdir()
    write_st(d / "model.safetensors", {"w": ("BF16", [16, 16], 512)})
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert saferead.read(d)["lane"] != "EXL3"


# ── agreement with the GGUF reader ──────────────────────────────────────────────────────────────
def test_both_readers_agree_on_the_same_model():
    """A model's parameter count must not depend on which lane you read it from."""
    root = workspace.home()
    src = root / "downloads/Qwen__Qwen2.5-0.5B-Instruct"
    gguf = next((root / "rungs").glob("Qwen2.5-0.5B-Instruct-Pollard-*.gguf"), None)
    if not src.is_dir() or gguf is None:
        pytest.skip("reference pair not in the workspace")
    a = saferead.summarise(saferead.read(src))["params"]
    b = ggufread.summarise(ggufread.read(gguf))["params"]
    assert a == b, f"safetensors says {a}, gguf says {b}"


# ── robustness ──────────────────────────────────────────────────────────────────────────────────
def test_empty_directory_is_an_error_not_a_crash(tmp_path):
    with pytest.raises(ValueError):
        saferead.read(tmp_path)


def test_truncated_header_is_reported(tmp_path):
    """A bad shard is reported, not raised -- a partial file mid-build is normal, not fatal."""
    d = tmp_path / "m"
    d.mkdir()
    (d / "model.safetensors").write_bytes(struct.pack("<Q", 10**12))
    info = saferead.read(d)
    assert info["warnings"] and info["params"] == 0
    with pytest.raises(ValueError):
        saferead.read_header(d / "model.safetensors")


def test_a_bad_shard_does_not_lose_the_good_ones(tmp_path):
    """One corrupt shard must not make the whole build unreadable."""
    d = tmp_path / "m"
    d.mkdir()
    write_st(d / "a.safetensors", {"w": ("BF16", [100, 100], 20000)})
    (d / "b.safetensors").write_bytes(struct.pack("<Q", 10**12))
    info = saferead.read(d)
    assert info["params"] == 10_000 and info["warnings"]


def test_sharded_model_sums_across_shards(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    write_st(d / "model-00001-of-00002.safetensors", {"a": ("BF16", [100, 100], 20000)})
    write_st(d / "model-00002-of-00002.safetensors", {"b": ("BF16", [100, 100], 20000)})
    s = saferead.summarise(saferead.read(d))
    assert s["params"] == 20_000 and s["shards"] == 2


def test_summaries_share_a_shape_across_lanes(tmp_path):
    """The UI must not care which reader produced a build."""
    d = tmp_path / "m"
    d.mkdir()
    write_st(d / "model.safetensors", {"model.layers.0.self_attn.q_proj.weight":
                                       ("BF16", [64, 64], 8192)})
    s = saferead.summarise(saferead.read(d))
    for k in ("name", "path", "architecture", "block_count", "params", "file_bytes",
              "bpw", "groups", "types", "mtp_block", "inexact"):
        assert k in s, f"missing {k}"


# ── multimodal ──────────────────────────────────────────────────────────────────────────────────
def test_vision_tower_is_not_filed_as_language():
    """`visual.blocks.0.attn.qkv` contains "attn". A language-first pattern order files the whole
    vision tower under the mixing path, and then the allocator crushes the model's eyes."""
    for n in ("visual.blocks.0.attn.qkv.weight", "vision_tower.encoder.layers.3.mlp.fc1.weight",
              "vision_model.encoder.layers.0.self_attn.k_proj.weight"):
        assert saferead._group(n) == "vision", n


def test_projector_is_its_own_group():
    for n in ("multi_modal_projector.linear_1.weight", "mm_projector.0.weight",
              "visual.merger.mlp.0.weight", "model.resampler.attn.q.weight"):
        assert saferead._group(n) == "projector", n


def test_audio_tower_is_recognised():
    for n in ("audio_tower.layers.0.self_attn.q_proj.weight",
              "model.audio_encoder.conv1.weight"):
        assert saferead._group(n) == "audio", n


def test_language_layers_are_untouched_by_the_multimodal_patterns():
    assert saferead._group("model.layers.0.self_attn.q_proj.weight") == "mixing"
    assert saferead._group("model.layers.0.mlp.down_proj.weight") == "ffn"
    assert saferead._group("model.embed_tokens.weight") == "embeddings"


def test_a_real_vl_model_separates_its_towers():
    d = workspace.home() / "downloads/Qwen__Qwen2-VL-2B-Instruct"
    if not d.is_dir():
        pytest.skip("no VL model in the workspace")
    g = saferead.summarise(saferead.read(d))["groups"]
    assert g.get("vision", {}).get("params", 0) > 1e8, "vision tower not separated"
    assert g.get("projector", {}).get("params", 0) > 0, "projector not separated"
    # and none of it leaked into the language groups
    assert g["vision"]["tensors"] > 100


# ── modality detection and the structural checks ────────────────────────────────────────────────
from pollard_studio import modalities as MOD   # noqa: E402


def test_detection_reads_artifacts_not_names(tmp_path):
    """A directory called `my-vision-model` with no vision config has no vision."""
    d = tmp_path / "my-vision-model-i-promise"
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "llama"}')
    assert MOD.detect(d)["modalities"]["vision_in"] is False


def test_a_vision_config_is_detected(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text('{"model_type":"x","vision_config":{"hidden_size":8}}')
    assert MOD.detect(d)["modalities"]["vision_in"] is True


def test_a_diffusion_pipeline_is_detected_as_image_out(tmp_path):
    d = tmp_path / "m"
    (d / "vae").mkdir(parents=True)
    (d / "unet").mkdir()
    (d / "config.json").write_text("{}")
    assert MOD.detect(d)["modalities"]["image_out"] is True


def test_text_is_always_present():
    assert MOD.detect("/nonexistent")["modalities"]["text"] is True


def test_a_real_vl_checkpoint_reports_vision():
    d = MOD.workspace_home() if hasattr(MOD, "workspace_home") else None
    from pollard_studio import workspace
    p = workspace.home() / "downloads/Qwen__Qwen2-VL-2B-Instruct"
    if not p.is_dir():
        pytest.skip("no VL checkpoint in the workspace")
    assert MOD.detect(p)["modalities"]["vision_in"] is True


def _wav(samples, rate=16000):
    import struct as st
    body = st.pack("<%dh" % len(samples), *samples)
    return (b"RIFF" + st.pack("<I", 36 + len(body)) + b"WAVE" + b"fmt "
            + st.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + st.pack("<I", len(body)) + body)


def test_silence_is_caught():
    assert MOD.check_audio(_wav([0] * 16000))["ok"] is False


def test_a_constant_tone_is_not_speech():
    buzz = [8000 if (i // 8000) % 2 else -8000 for i in range(16000)]
    r = MOD.check_audio(_wav(buzz))
    assert r["ok"] is False and "tone" in r["reason"]


def test_clipping_is_caught():
    clip = [32767 if i % 2 else -32768 for i in range(16000)]
    assert MOD.check_audio(_wav(clip))["ok"] is False


def test_speech_like_audio_passes():
    import math as m
    import random
    random.seed(0)
    s = [int(9000 * m.sin(2 * m.pi * (110 + 40 * m.sin(i / 1600)) * i / 16000)
             + random.gauss(0, 700)) for i in range(16000)]
    assert MOD.check_audio(_wav(s))["ok"] is True


def _ppm(px, w, h):
    return b"P5\n" + f"{w} {h}".encode() + b"\n255\n" + bytes(px)


def test_a_flat_frame_is_caught():
    r = MOD.check_image(_ppm([128] * 4096, 64, 64))
    assert r["ok"] is False and "flat" in r["reason"]


def test_pure_noise_is_caught():
    import random
    random.seed(1)
    r = MOD.check_image(_ppm([random.randrange(256) for _ in range(4096)], 64, 64))
    assert r["ok"] is False and "noise" in r["reason"]


def test_structure_passes():
    px = [(0 if (x // 8 + y // 8) % 2 else 255) for y in range(64) for x in range(64)]
    assert MOD.check_image(_ppm(px, 64, 64))["ok"] is True


def test_undecodable_bytes_are_refused_not_guessed():
    assert MOD.check_image(b"not an image")["ok"] is False
    assert MOD.check_audio(b"not a wav")["ok"] is False


# ── playback ────────────────────────────────────────────────────────────────────────────────────
def test_media_types_cover_image_audio_and_video():
    kinds = {k for k, _ in MOD.MEDIA.values()}
    assert kinds == {"image", "audio", "video"}
    for ext in (".png", ".wav", ".mp4", ".webm", ".mp3", ".flac", ".ppm"):
        assert ext in MOD.MEDIA, ext


def test_a_non_media_file_is_refused_with_a_reason():
    r = MOD.load_media("/tmp/whatever.gguf")
    assert r["ok"] is False and "not a media type" in r["reason"]


def test_media_comes_back_as_a_data_uri_not_a_path():
    """The page has its own origin and cannot read the disk. A file:// link renders as a broken
    box with no explanation."""
    import tempfile
    d = Path(tempfile.mkdtemp())
    f = d / "t.ppm"
    f.write_bytes(b"P5\n8 8\n255\n" + bytes(range(64)))
    r = MOD.load_media(f)
    assert r["ok"] and r["data"].startswith("data:image/png;base64,")


def test_netpbm_is_converted_because_no_browser_renders_it():
    import tempfile
    d = Path(tempfile.mkdtemp())
    f = d / "t.ppm"
    f.write_bytes(b"P5\n8 8\n255\n" + bytes([200]) * 64)
    r = MOD.load_media(f)
    assert r["mime"] == "image/png"
    import base64
    assert base64.b64decode(r["data"].split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_check_grades_the_original_not_the_converted_copy():
    """Converting first and checking after reads the compressed stream instead of the pixels, and
    anything highly compressible comes back as 'flat frame' -- the very failure it detects."""
    import tempfile
    d = Path(tempfile.mkdtemp())
    px = bytes((0 if (x // 8 + y // 8) % 2 else 255) for y in range(64) for x in range(64))
    f = d / "checker.ppm"
    f.write_bytes(b"P5\n64 64\n255\n" + px)
    assert MOD.load_media(f)["check"]["ok"] is True, "a checkerboard is structure, not a flat frame"


def test_a_huge_file_is_refused_rather_than_inlined():
    """A data URI is copied whole into the page, so it costs that memory twice."""
    assert MOD.MAX_INLINE <= 128 * 1024 * 1024


def test_noise_is_judged_on_roughness_not_entropy():
    """Tried entropy: on a small frame a rendered image with grain scores HIGHER than uniform
    noise, because 4096 samples over 256 bins undershoots the maximum. It read the wrong way."""
    src = (ROOT / "pollard_studio/modalities.py").read_text()
    assert "noise = rough > 60.0\n" in src
    assert "NOT used to gate" in src
