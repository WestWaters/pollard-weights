"""MX (compressed-tensors FP4/FP8) lane: MoE routers, one MoE backend, FP8 KV, Blackwell/Rubin serve lines.

What broke: build_recipe's body modifier is targets="Linear", and a MoE router (`mlp.gate`,
`shared_expert_gate`, `router`) is an nn.Linear -- so routers went to FP4 with the experts. FP4 error in a
router reorders the top-k and changes which experts run; llm-compressor's own MoE examples ignore them.
And protecting a hot layer's `.*proj` (or every `down_proj` with --protect-down) put that layer's experts at
FP8, so one model needed two fused-MoE backends in vLLM. These pin the router ignore, the attention-only
protection, the FP8 kv_cache_scheme, and the serve hints (flags only -- never a speed claim).
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_mx as M


def _matches(globs, name):
    """A 're:' glob that covers the WHOLE name -- the strict reading, for 'must match' checks."""
    return any(re.fullmatch(g[3:], name) for g in globs if g.startswith("re:"))


def _reaches(globs, name):
    """compressed-tensors 0.19 match_name: `re.match` (anchored at the start only) -- the loose reading,
    for 'must NOT match' checks."""
    return any(re.match(g[3:], name) for g in globs if g.startswith("re:"))


@pytest.mark.parametrize("name", [
    "model.layers.3.mlp.gate",                         # Qwen2/3-MoE, Qwen3-Next
    "model.layers.0.mlp.shared_expert_gate",           # Qwen MoE shared-expert gate
    "model.layers.1.block_sparse_moe.gate",            # Mixtral
    "language_model.model.layers.2.feed_forward.router",  # Llama 4
    "model.layers.5.mlp.router",                       # gpt-oss
])
def test_moe_routers_are_never_quantized(name):
    for attn_only in (False, True):
        rec = M.build_recipe(["3"], "NVFP4", "FP8", False, moe_attn_only=attn_only)
        assert _matches(rec["ignore"], name), f"router {name} would go to FP4"


@pytest.mark.parametrize("name", [
    "model.layers.3.mlp.experts.7.gate_proj",
    "model.layers.3.mlp.gate_proj",
    "model.layers.0.mlp.gate_up_proj",
    "model.layers.3.mlp.shared_expert.gate_proj",
    "model.layers.3.self_attn.q_proj",
])
def test_router_globs_do_not_catch_projections(name):
    rec = M.build_recipe(["3"], "NVFP4", "FP8", False)
    assert not _reaches(rec["ignore"], name), f"{name} is a projection, not a router"


@pytest.mark.parametrize("name", [
    "model.layers.3.self_attn.q_proj",
    "model.layers.3.self_attn.o_proj",
    "model.layers.3.self_attn.kv_a_proj_with_mqa",     # MLA (DeepSeek / GLM)
    "model.layers.3.linear_attn.in_proj_qkvz",         # Qwen3-Next linear attention
    "model.layers.3.linear_attn.out_proj",
])
def test_attn_only_protects_hot_attention(name):
    rec = M.build_recipe(["3"], "NVFP4", "FP8", False, moe_attn_only=True)
    assert _matches(rec["protect"], name)


@pytest.mark.parametrize("name", [
    "model.layers.3.mlp.experts.0.down_proj",
    "model.layers.3.mlp.experts.12.gate_proj",
    "model.layers.3.mlp.shared_expert.down_proj",
    "model.layers.7.mlp.experts.0.down_proj",
    "model.layers.4.self_attn.q_proj",                 # not a hot layer
])
def test_attn_only_keeps_every_expert_in_the_body_scheme(name):
    """--protect-down is passed too: attention-only must still never reach an expert."""
    rec = M.build_recipe(["3"], "NVFP4", "FP8", True, moe_attn_only=True)
    assert not _reaches(rec["protect"], name), f"{name} would leave the body scheme"
    assert "re:.*down_proj" not in rec["protect"]


def test_default_protection_is_unchanged():
    """Without the MoE option, a hot layer's projections (experts included) are protected as before."""
    rec = M.build_recipe(["3"], "NVFP4", "FP8", True)
    assert rec["protect"] == [r"re:.*layers\.3\..*proj", "re:.*down_proj"]
    assert _matches(rec["protect"], "model.layers.3.mlp.experts.0.up_proj")


def test_kv_cache_scheme_is_off_by_default():
    assert M.build_recipe(["3"], "NVFP4", "FP8", False)["kv_cache_scheme"] is None


def test_kv_fp8_scheme_is_static_per_tensor_float8():
    kv = M.build_recipe(["3"], "NVFP4", "FP8", False, kv_fp8=True)["kv_cache_scheme"]
    assert kv == {"num_bits": 8, "type": "float", "strategy": "tensor", "dynamic": False, "symmetric": True}
    kv["num_bits"] = 4                                   # a caller's edit must not leak into the next recipe
    assert M.build_recipe(["3"], "NVFP4", "FP8", False, kv_fp8=True)["kv_cache_scheme"]["num_bits"] == 8


def test_is_moe_config():
    assert M.is_moe_config({"num_experts": 128})
    assert M.is_moe_config({"num_local_experts": 8})
    assert M.is_moe_config({"n_routed_experts": 256})
    assert M.is_moe_config({"text_config": {"num_local_experts": 16}})
    assert not M.is_moe_config({"num_hidden_layers": 28})
    assert not M.is_moe_config({"num_experts": 0})
    assert not M.is_moe_config(None)


def test_serve_hints_name_the_rubin_image_and_flags():
    text = "\n".join(M.serve_hints("/m", "NVFP4", kv_fp8=True, moe=True))
    assert "vllm/vllm-openai:cu134-nightly" in text
    assert "--kv-cache-dtype fp8" in text
    assert "--moe-backend flashinfer_cutedsl" in text
    assert "vllm serve /m --kv-cache-dtype fp8" in text


def test_serve_hints_moe_backend_only_for_nvfp4_moe():
    assert "--moe-backend" not in "\n".join(M.serve_hints("/m", "NVFP4", moe=False))
    assert "--moe-backend" not in "\n".join(M.serve_hints("/m", "MXFP4", moe=True))
    assert "--kv-cache-dtype" not in "\n".join(M.serve_hints("/m", "NVFP4"))


def test_serve_hints_mxfp4_wording_and_no_speed_claims():
    for scheme in ("NVFP4", "MXFP4", "W4A16"):
        text = "\n".join(M.serve_hints("/m", scheme, kv_fp8=True, moe=True))
        assert not re.search(r"tok/s|tokens/s|\d+(\.\d+)?x faster|faster than", text, re.I), text
        assert "untested" in text
    assert "flashinfer_cutlass (validate)" in "\n".join(M.serve_hints("/m", "MXFP4"))
