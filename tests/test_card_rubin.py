"""pollard-card: the MX lane names Rubin; nothing else does, and no card claims a Rubin speed.

The MX (compressed-tensors NVFP4/FP8) checkpoints run on Vera Rubin through vLLM with the same formats as
Blackwell, but the card said "for Blackwell/vLLM", carried no rubin / fp8 tags and gave only `vllm serve`,
which on Rubin needs the cu134-nightly image. GGUF is the opposite case: llama.cpp has no FP4 tensor-core
path on sm_100/107, so a Rubin tag or line on a GGUF card would advertise an optimized build that does
not exist. Pollard has not run on Rubin, so no card may carry a Rubin throughput number.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_card as C

RUBIN_TAGS = {"rubin", "vera-rubin", "nvfp4", "fp8"}


def _tags(fm):
    return [l[2:] for l in fm.splitlines() if l.startswith("- ")]


def test_mx_lane_carries_rubin_tags():
    t = _tags(C.frontmatter("x/y", "mit", ["mx"], "qwen3", None))
    assert RUBIN_TAGS <= set(t), t
    assert "blackwell" in t


def test_rubin_tags_are_mx_only():
    for lane in ("gguf", "gptq", "mlx", "exl3"):
        t = set(_tags(C.frontmatter("x/y", "mit", [lane], "qwen3", None)))
        assert not (t & RUBIN_TAGS), f"{lane} card tagged {t & RUBIN_TAGS}"


def test_mx_wording():
    assert C.LANE_WORD["mx"] == " for Blackwell/Rubin (vLLM)"
    assert "Rubin" not in C.LANE_WORD["gguf"]


def test_mx_serve_block_has_the_rubin_image_and_flags():
    text = "\n".join(C.vllm_serve_block("acme/M-Pollard-NVFP4", ["mx"]))
    assert "vllm serve acme/M-Pollard-NVFP4" in text
    assert "vllm/vllm-openai:cu134-nightly acme/M-Pollard-NVFP4" in text
    assert "--moe-backend flashinfer_cutedsl" in text and "--kv-cache-dtype fp8" in text


def test_gptq_serve_block_has_no_rubin_line():
    text = "\n".join(C.vllm_serve_block("acme/M", ["gptq"]))
    assert "vllm serve acme/M" in text and "Rubin" not in text and "cu134" not in text


def test_gguf_gets_no_vllm_block():
    assert C.vllm_serve_block("acme/M", ["gguf"]) == []


def test_no_rubin_speed_claim_anywhere_in_the_template():
    """No tok/s, 'x faster' or similar next to Rubin, in the serve block or the template source."""
    block = "\n".join(C.vllm_serve_block("acme/M", ["mx", "gptq"]))
    src = open(os.path.join(os.path.dirname(__file__), "..", "tools", "pollard_card.py"), encoding="utf-8").read()
    speed = re.compile(r"tok/s|tokens/s|\d+(\.\d+)?\s*x\s+faster|faster than", re.I)
    assert not speed.search(block)
    for line in src.splitlines():
        if "Rubin" in line:
            assert not speed.search(line), line
