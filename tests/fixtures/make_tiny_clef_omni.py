"""Tiny random clef-omni checkpoint for tests: a 2-layer Qwen3-Omni MoE thinker (hidden 64, 4 experts, top-2,
interleaved M-RoPE) plus a random joint decision head (1 routing block, 1 joint block, 4 heads, width 32).

Built to check the clef-omni runtime patch after an upstream rebase: convert it, serve one /v1/systemone request,
compare with the recorded answers (see tests/test_clef_omni_tiny.py). Seeded, so the weights are reproducible.
"""
import json, os, shutil

import torch
from safetensors.torch import save_file
from transformers import Qwen3OmniMoeConfig


def make(out, tokenizer_dir):
    """Write the checkpoint to `out`, with the Qwen3 tokenizer files copied from `tokenizer_dir`.
    Returns the thinker's parameter count."""
    torch.manual_seed(0)
    os.makedirs(out, exist_ok=True)
    tok = tokenizer_dir.rstrip("/") + "/"
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy(tok + f, out)
    V, H, L, NH, NKV, HD, E, K, FF = 151936, 64, 2, 4, 2, 16, 4, 2, 32
    c = Qwen3OmniMoeConfig().to_dict()
    t = c["thinker_config"]["text_config"]
    t.update(vocab_size=V, hidden_size=H, num_hidden_layers=L, num_attention_heads=NH, num_key_value_heads=NKV,
             head_dim=HD, num_experts=E, num_experts_per_tok=K, moe_intermediate_size=FF, intermediate_size=FF*2,
             max_position_embeddings=4096)
    # Interleaved M-RoPE, set explicitly: transformers >= 5.15's Qwen3OmniMoeConfig() no longer carries
    # mrope_section, and without it the GGUF has no clef-omni.rope.dimension_sections and llama-server refuses
    # it. The sections count rope pairs and must sum to head_dim/2 = 8 (the full-size [24, 20, 20] trips a
    # GGML_ASSERT). Text-only positions are equal on all three axes, so the recorded answers do not change.
    for key in ("rope_parameters", "rope_scaling"):
        if key == "rope_parameters" or isinstance(t.get(key), dict):
            t[key] = dict(t.get(key) or {}, mrope_section=[3, 3, 2], mrope_interleaved=True)
    c["thinker_config"]["vision_config"]["deepstack_visual_indexes"] = [0]
    c["architectures"] = ["Qwen3OmniMoeForConditionalGeneration"]
    json.dump(c, open(f"{out}/config.json", "w"), indent=1)
    r = lambda *s: (torch.randn(*s) * 0.08).to(torch.bfloat16)
    one = lambda n: (1 + 0.1 * torch.randn(n)).to(torch.bfloat16)
    W = {"thinker.model.embed_tokens.weight": r(V, H), "thinker.model.norm.weight": one(H), "thinker.lm_head.weight": r(V, H)}
    for i in range(L):
        p = f"thinker.model.layers.{i}."
        W[p+"input_layernorm.weight"] = one(H); W[p+"post_attention_layernorm.weight"] = one(H)
        W[p+"self_attn.q_proj.weight"] = r(NH*HD, H); W[p+"self_attn.k_proj.weight"] = r(NKV*HD, H)
        W[p+"self_attn.v_proj.weight"] = r(NKV*HD, H); W[p+"self_attn.o_proj.weight"] = r(H, NH*HD)
        W[p+"self_attn.q_norm.weight"] = one(HD); W[p+"self_attn.k_norm.weight"] = one(HD)
        W[p+"mlp.gate.weight"] = r(E, H)
        for e in range(E):
            W[p+f"mlp.experts.{e}.gate_proj.weight"] = r(FF, H); W[p+f"mlp.experts.{e}.up_proj.weight"] = r(FF, H)
            W[p+f"mlp.experts.{e}.down_proj.weight"] = r(H, FF)
    save_file(W, f"{out}/model.safetensors", metadata={"format": "pt"})
    # joint head: h=32, ffn 64, 1 routing (evidence) layer + 1 joint layer, 4 heads
    h, fh, NR, NJ = 32, 64, 1, 1
    json.dump({"routing_layers": NR, "layers": NJ, "heads": 4}, open(f"{out}/joint_head_config.json", "w"))
    f32 = lambda *s: torch.randn(*s) * 0.1
    J = {"type_embedding.weight": f32(3, h)}
    def ln(n, size): J[n+".weight"] = 1 + 0.1*torch.randn(size); J[n+".bias"] = 0.05*torch.randn(size)
    def mha(n): J[n+".in_proj_weight"] = f32(3*h, h); J[n+".in_proj_bias"] = 0.05*torch.randn(3*h); J[n+".out_proj.weight"] = f32(h, h); J[n+".out_proj.bias"] = 0.05*torch.randn(h)
    def lin(n, o, i): J[n+".weight"] = f32(o, i); J[n+".bias"] = 0.05*torch.randn(o)
    for i in range(NR):
        p = f"evidence_layers.{i}."
        ln(p+"query_norm", h); ln(p+"memory_norm", h); mha(p+"attention"); ln(p+"feedforward_norm", h)
        lin(p+"feedforward.0", fh, h); lin(p+"feedforward.3", h, fh)
    for i in range(NJ):
        p = f"layers.{i}."
        ln(p+"norm1", h); mha(p+"self_attn"); ln(p+"norm2", h); mha(p+"multihead_attn"); ln(p+"norm3", h)
        lin(p+"linear1", fh, h); lin(p+"linear2", h, fh)
    ln("hidden_norm", H)
    for n in ("memory_projection", "question_projection", "option_question_projection", "global_projection",
              "option_context_projection", "option_lexical_projection"):
        J[n+".weight"] = f32(h, H)
    for n in ("option_summary_norm", "field_norm", "option_norm"): ln(n, h)
    lin("residual_scorer.0", h, 4*h); lin("residual_scorer.3", 1, h)
    J["prior_logit_scale"] = torch.tensor(0.3); J["joint_logit_scale"] = torch.tensor(0.5); J["residual_gate"] = torch.tensor(0.2)
    save_file(J, f"{out}/joint_head.safetensors")
    return sum(v.numel() for v in W.values())
