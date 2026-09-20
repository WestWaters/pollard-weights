"""Vision towers and modality projectors must not be quantized as if they were language layers.

On GGUF this cannot happen: the projector ships as a separate mmproj and pollard-fit says outright
not to quantize it. On every other lane the tower lives INSIDE the checkpoint, and the per-module
policies name only language modules -- so a vision block either matched nothing and took the low
bits, or matched `.layers.<n>.` and was handed a TEXT layer's sensitivity allocation.

These pin the policy so the modality path cannot silently regress.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _policy():
    """bits_for + its patterns, without importing mlx (which is not installed everywhere)."""
    src = (ROOT / "tools/pollard_mlx.py").read_text()
    ns: dict = {"re": re}
    for pat in (r"^HIGH, LOW = .*$", r"^VISION_RE = .*?\n(?:\s+r?\".*\n)*",
                r"^PROJ_RE = .*?\n(?:\s+r?\".*\n)*",
                r"^def layer_of.*?(?=\n\n)", r"^def bits_for.*?(?=\n\ndef )"):
        m = re.search(pat, src, re.M | re.S)
        assert m, f"could not lift {pat!r} out of pollard_mlx.py"
        exec(m.group(0), ns)
    return ns


P = _policy()
HIGH, LOW = P["HIGH"], P["LOW"]
ALLOC = {i: {"attn": LOW, "ffn": LOW} for i in range(8)}


def bits(path, vision_bits=None):
    return P["bits_for"](path, ALLOC, False, 64, vision_bits)["bits"]


@pytest.mark.parametrize("path", [
    "visual.blocks.0.attn.qkv",
    "visual.blocks.11.mlp.fc1",
    "vision_tower.encoder.layers.2.mlp.fc1",
    "vision_model.encoder.layers.0.self_attn.k_proj",
    "model.vision_encoder.blocks.4.attn.proj",
    "audio_tower.layers.3.self_attn.q_proj",
])
def test_vision_and_audio_towers_are_held_high(path):
    assert bits(path) == HIGH, f"{path} would be quantized as a language layer"


@pytest.mark.parametrize("path", [
    "multi_modal_projector.linear_1",
    "mm_projector.0",
    "visual.merger.mlp.0",
    "model.resampler.attn.q",
])
def test_the_projector_is_always_high(path):
    """Modality alignment is the one thing that cannot be traded away."""
    assert bits(path) == HIGH
    assert bits(path, vision_bits=2) == HIGH, "--vision-bits must not reach the projector"


def test_a_vision_layer_does_not_inherit_a_text_layers_allocation():
    """`vision_tower.encoder.layers.2.` also matches the language `.layers.<n>.` pattern."""
    alloc = {2: {"attn": 2, "ffn": 2}}                      # a very low text allocation
    got = P["bits_for"]("vision_tower.encoder.layers.2.self_attn.q_proj", alloc, False, 64, None)
    assert got["bits"] == HIGH, "vision block took the text layer's bits"


def test_vision_bits_is_honoured_when_asked_for():
    """Compressing the tower is a legitimate choice -- it just must not happen silently."""
    assert bits("visual.blocks.0.attn.qkv", vision_bits=4) == 4


def test_language_modules_are_unaffected():
    assert bits("model.layers.3.self_attn.q_proj") == ALLOC[3]["attn"]
    assert bits("model.layers.3.mlp.down_proj") == ALLOC[3]["ffn"]
    assert bits("model.embed_tokens") == HIGH
    assert bits("lm_head") == HIGH


def test_the_flag_exists_on_the_cli():
    assert "--vision-bits" in (ROOT / "tools/pollard_mlx.py").read_text()


def test_gguf_lane_still_says_not_to_quantize_the_projector():
    """The GGUF policy is stronger than the other lanes and must stay that way."""
    src = (ROOT / "tools/pollard_fit.py").read_text()
    assert re.search(r"not\s+quantize\s+the\s+mmproj", src, re.I)


# ── MX / compressed-tensors lane ────────────────────────────────────────────────────────────────
def _mx():
    """build_recipe and its globs, without importing llmcompressor."""
    src = (ROOT / "tools/pollard_mx.py").read_text()
    ns: dict = {"re": re}
    for pat in (r"^VISION_GLOBS = .*?\n(?:\s+r\".*\n)*", r"^PROJECTOR_GLOBS = .*?\n(?:\s+r\".*\n)*",
                r"^def build_recipe.*?(?=\n\ndef )"):
        m = re.search(pat, src, re.M | re.S)
        assert m, f"could not lift {pat!r} out of pollard_mx.py"
        exec(m.group(0), ns)
    return ns


MX = _mx()


def _matches(globs, name):
    """compressed-tensors 're:' globs, evaluated the way it evaluates them."""
    return any(re.fullmatch(g[3:], name) for g in globs if g.startswith("re:"))


@pytest.mark.parametrize("name", [
    "visual.blocks.0.attn.qkv",
    "vision_tower.encoder.layers.2.self_attn.out_proj",
    "model.vision_model.encoder.layers.5.mlp.fc1",
    "audio_tower.layers.1.self_attn.k_proj",
])
def test_mx_ignores_the_vision_tower_by_default(name):
    """targets='Linear' matches every Linear in the model, so the tower must be ignored by name."""
    rec = MX["build_recipe"](["3"], "NVFP4", "FP8", False)
    assert _matches(rec["ignore"], name), f"{name} would be quantized to FP4"


@pytest.mark.parametrize("name", [
    "multi_modal_projector.linear_1",
    "visual.merger.mlp.0",
    "model.mm_projector.2",
])
def test_mx_always_ignores_the_projector(name):
    for quantize_vision in (False, True):
        rec = MX["build_recipe"](["3"], "NVFP4", "FP8", False, quantize_vision)
        assert _matches(rec["ignore"], name), "--quantize-vision must not reach the projector"


def test_mx_quantize_vision_releases_the_tower_only():
    rec = MX["build_recipe"](["3"], "NVFP4", "FP8", False, True)
    assert not _matches(rec["ignore"], "visual.blocks.0.attn.qkv")
    assert _matches(rec["ignore"], "visual.merger.mlp.0")


def test_mx_language_modules_are_still_quantized():
    rec = MX["build_recipe"](["3"], "NVFP4", "FP8", False)
    for name in ("model.layers.0.self_attn.q_proj", "model.layers.7.mlp.down_proj"):
        assert not _matches(rec["ignore"], name), f"{name} should still be quantized"


def test_mx_lm_head_stays_ignored():
    assert "lm_head" in MX["build_recipe"](["3"], "NVFP4", "FP8", False)["ignore"]


def test_mx_protect_globs_cannot_capture_a_vision_layer():
    """`.*layers\\.3\\..*proj` also matches vision_tower.encoder.layers.3.* — the ignore list,
    applied to BOTH modifiers, is what stops a vision block inheriting a text layer's hotness."""
    rec = MX["build_recipe"](["3"], "NVFP4", "FP8", False)
    vision = "vision_tower.encoder.layers.3.self_attn.out_proj"
    assert _matches(rec["protect"], vision), "precondition: the glob does reach it"
    assert _matches(rec["ignore"], vision), "so ignore must override it"
