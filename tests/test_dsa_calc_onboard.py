#!/usr/bin/env python3
"""DeepSeek-Sparse-Attention-family regressions (GLM-5.3 `glm_moe_dsa`, Tencent Hy4 `hy_v4`), anchored to measurements
taken 2026-09 on a 744B GLM-5.3 deployment:

  * pollard-calc must count the DSA indexer key cache — measured 41 KB/token (NVFP4 latent) vs 22 KB latent-only;
    Hy4 caches only on its 21 "full" indexer layers.
  * pollard-calc build-time rates must be within 2x of the measured 744B cook times (GPTQ ~20 node-h, EXL3 ~66 GPU-h).
  * pollard-onboard must match per-expert tensors after the layer/expert index is collapsed to N (it never did), and
    recognise fused expert tensors, the indexer, the attention output gate, shared experts and the MTP side model.

    python tests/test_dsa_calc_onboard.py   |   pytest tests/test_dsa_calc_onboard.py
"""
import json, os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_calc as C
import pollard_onboard as O

GLM53 = {"model_type": "glm_moe_dsa", "hidden_size": 6144, "num_hidden_layers": 78, "num_attention_heads": 64,
         "num_key_value_heads": 64, "head_dim": 256, "intermediate_size": 12288, "moe_intermediate_size": 2048,
         "n_routed_experts": 256, "num_experts_per_tok": 8, "n_shared_experts": 1, "vocab_size": 154880,
         "kv_lora_rank": 512, "qk_rope_head_dim": 64, "q_lora_rank": 2048, "index_head_dim": 128, "index_n_heads": 32,
         "index_topk": 2048, "first_k_dense_replace": 3, "num_nextn_predict_layers": 1}
HY4 = dict(GLM53, model_type="hy_v4", vocab_size=120832, intermediate_size=18432,
           indexer_types=["full", "full"] + ["shared"] * 57 + ["full"] * 19, mlp_layer_types=["dense"] + ["sparse"] * 77)


def test_calc_counts_the_indexer_cache():
    a = C.analyse(GLM53)
    assert a["index_head_dim"] == 128 and a["n_indexer"] == 78
    per_tok_nvfp4 = C.kv_cache_bytes(a, 1, 0.5)
    per_tok_fp8 = C.kv_cache_bytes(a, 1, 1.0)
    # measured on the deployment: 41 KB/token nvfp4, 57 KB fp8 (latent-only would be 22.5 / 44.9)
    assert 36_000 < per_tok_nvfp4 < 46_000, per_tok_nvfp4
    assert 50_000 < per_tok_fp8 < 62_000, per_tok_fp8


def test_calc_shared_indexer_layers_keep_no_cache():
    a = C.analyse(HY4)
    assert a["n_indexer"] == 21
    a_glm = C.analyse(GLM53)
    assert C.kv_cache_bytes(a, 1, 0.5) < C.kv_cache_bytes(a_glm, 1, 0.5)


def test_calc_without_indexer_is_unchanged():
    cfg = {k: v for k, v in GLM53.items() if k not in ("index_head_dim", "index_n_heads", "index_topk")}
    a = C.analyse(cfg)
    assert a["n_indexer"] == 0 and C.kv_cache_bytes(a, 1, 0.5) == 78 * 576 * 0.5


def test_build_time_within_2x_of_measured_744b():
    a = C.analyse(GLM53)
    a["total"] = 744e9
    bt = C.estimate_build_time(a)
    assert 10 < bt["gptq"] < 40, bt["gptq"]      # measured ~20 node-hours
    assert 33 < bt["exl3"] < 132, bt["exl3"]     # measured ~66 GPU-hours


def _audit(cfg, keys):
    d = tempfile.mkdtemp()
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in keys}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    return O.audit(d)


def test_onboard_matches_per_expert_tensors_after_index_collapse():
    keys = [f"model.layers.{l}.mlp.experts.{e}.{m}_proj.weight" for l in (3, 40) for e in (0, 255) for m in ("gate", "up", "down")]
    keys += ["model.layers.3.self_attn.q_a_proj.weight", "model.layers.3.input_layernorm.weight", "model.embed_tokens.weight", "lm_head.weight"]
    f = _audit(GLM53, keys)
    assert not f["unmatched"], f["unmatched"]


def test_onboard_recognises_the_dsa_family():
    keys = ["model.layers.5.self_attn.indexer.wq_b.weight", "model.layers.5.self_attn.indexer.wk.weight",
            "model.layers.5.self_attn.indexer.weights_proj.weight", "model.layers.5.self_attn.linear_gate.weight",
            "model.layers.5.mlp.gate.weight", "model.layers.5.mlp.shared_experts.gate_proj.weight",
            "model.layers.5.mlp.experts.gate_up_proj", "model.layers.5.mlp.experts.down_proj",
            "model.layers.5.hc_attn_layer.hc_pre.hc_fn", "model.mtp_layers.0.eh_proj.weight", "model.mtp_layers.0.enorm.weight",
            "model.embed_tokens.weight", "lm_head.weight"]
    f = _audit(HY4, keys)
    assert not f["unmatched"], f["unmatched"]
    flags = " ".join(f["flags"])
    for word in ("FUSED EXPERTS", "indexer", "GATE", "mtp_layers", "hyperconnections"):
        assert word in flags, (word, flags)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    sys.exit(1 if fails else 0)
