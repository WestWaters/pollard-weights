"""Tiny random Nemotron-3.5-Bolt-shaped checkpoint for the streaming lanes: what NVIDIA's 24GB release looks like on
disk, at toy size.

  * config.json: an omni wrapper (model_type nemotron_h_omni, an auto_map naming custom modelling files) whose
    `llm_config` is a `nemotron_h_puzzle` -- per-layer block configs (mamba / moe / attention / moe) with a
    different expert count and width per MoE layer, latent MoE (fc1/fc2_latent_proj), relu2 experts.
  * weights under `language_model.` (backbone.* + lm_head) plus a vision tower (`vision_model.*`) and a projector
    (`mlp1.*`) the text lanes must copy through untouched.
  * modelopt MIXED_PRECISION encoding, as NVIDIA ships it: the mamba in_proj / out_proj in W4A16 NVFP4 (packed U8 +
    F8_E4M3 block scales + F32 weight_scale_2), every routed and shared expert ggml-packed IQ2_XXS inside the
    safetensors (U8 [rows, blocks, 66]), the latent projections, embeddings and lm_head ggml Q8_0.
  * the custom modelling files are stubs that raise if imported: Pollard must build the backbone from transformers'
    native NemotronH, never from them.

make(out) returns {"float": {key: float32 numpy}} -- the DEQUANTIZED reference of every tensor -- so a test can
check what the reader decodes. Seeded.
"""
import json, os

import numpy as np

H, LAT, V = 256, 256, 512
BLOCKS = [{"block_type": "mamba"},
          {"block_type": "moe", "moe_intermediate_size": 256, "n_routed_experts": 4, "num_experts_per_tok": 2},
          {"block_type": "attention"},
          {"block_type": "moe", "moe_intermediate_size": 512, "n_routed_experts": 6, "num_experts_per_tok": 2}]


def llm_config():
    return {"model_type": "nemotron_h_puzzle", "architectures": ["NemotronHPuzzleForCausalLM"],
            "auto_map": {"AutoConfig": "configuration_nemotron_h_puzzle.NemotronHPuzzleConfig",
                         "AutoModelForCausalLM": "modeling_nemotron_h_puzzle.NemotronHPuzzleForCausalLM"},
            "vocab_size": V, "hidden_size": H, "intermediate_size": 512, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 64, "mamba_num_heads": 8, "mamba_head_dim": 32, "ssm_state_size": 16,
            "n_groups": 2, "conv_kernel": 4, "expand": 2, "chunk_size": 16, "moe_latent_size": LAT,
            "moe_shared_expert_intermediate_size": 256, "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
            "routed_scaling_factor": 2.5, "mlp_hidden_act": "relu2",
            "num_experts": 8, "num_local_experts": 8,          # global values the per-layer blocks must override "hidden_act": "relu2",
            "mamba_hidden_act": "silu", "layer_norm_epsilon": 1e-5, "tie_word_embeddings": False,
            "time_step_limit": [0.0, {"__float__": "Infinity"}], "use_mamba_kernels": False,
            "block_configs": BLOCKS, "num_hidden_layers": len(BLOCKS), "max_position_embeddings": 4096,
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}


STUB = 'raise ImportError("Pollard must not import the checkpoint\'s own modelling code here")\n'


def _q(arr, qtype):
    from gguf import quants, GGMLQuantizationType as Q
    return quants.quantize(np.asarray(arr, np.float32), Q[qtype])


def _iq2xxs(rows, cols, rng):
    """Random but well-formed IQ2_XXS blocks [rows, cols/256, 66] (a small fp16 scale + random grid / sign bits)."""
    nb = cols // 256
    raw = rng.integers(0, 256, size=(rows, nb, 66), dtype=np.uint8)
    d = (rng.uniform(0.002, 0.01, size=(rows, nb))).astype(np.float16).view(np.uint8).reshape(rows, nb, 2)
    raw[..., :2] = d
    return raw


def _deq(raw2d, qtype):
    from gguf import quants, GGMLQuantizationType as Q
    return quants.dequantize(raw2d, Q[qtype]).astype(np.float32)


def make(out):
    import torch
    import pollard_stream as S
    import pollard_mx as X
    from pollard_mxfp6 import _f32_to_e4m3_bytes
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    got, why = S._puzzle_build(llm_config())
    assert got, why
    cls, conf, per, modeling, blocks = got
    import transformers
    conf._experts_implementation = "eager"
    orig = {b.__name__: b for b in blocks}
    try:
        for b in blocks:
            class Per(b):
                def __init__(self, config, layer_idx=None, *a, _b=b, **k):
                    c = per[layer_idx]
                    c._experts_implementation = "eager"
                    _b.__init__(self, c, layer_idx, *a, **k)
            Per.__name__ = b.__name__
            setattr(modeling, b.__name__, Per)
        model = cls(conf)
    finally:
        for n, b in orig.items():
            setattr(modeling, n, b)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.05)
            elif n.endswith(("A_log",)):
                p.uniform_(0, 1)
            elif "norm" in n:
                p.fill_(1.0)
    from transformers.core_model_loading import revert_weight_conversion
    sd = revert_weight_conversion(model, {k: v.detach().float().contiguous() for k, v in model.state_dict().items()})
    tensors, ref = {}, {}

    def put(name, dt, arr):
        tensors[name] = (dt, np.ascontiguousarray(arr))

    qlayers = {}
    for k, v in sorted(sd.items()):
        key = "language_model." + k
        a = v.numpy().astype(np.float32)
        mod = key[:-len(".weight")] if key.endswith(".weight") else None
        if mod and (mod.endswith("mixer.in_proj") or mod.endswith("mixer.out_proj")):
            packed, sc, g, _ = X.nvfp4_encode(a, "6")
            put(key, "U8", packed)
            put(mod + ".weight_scale", "F8_E4M3", _f32_to_e4m3_bytes(sc))
            put(mod + ".weight_scale_2", "F32", np.array(1.0 / g, np.float32))
            ref[key] = X.nvfp4_decode(packed, sc, g)
            qlayers[mod] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        elif mod and (".experts." in mod or ".shared_experts." in mod) and a.ndim == 2:
            raw = _iq2xxs(a.shape[0], a.shape[1], rng)
            put(key, "U8", raw)
            ref[key] = _deq(raw.reshape(a.shape[0], -1), "IQ2_XXS")
            qlayers[mod] = {"quant_algo": "IQ2_XXS", "group_size": 256, "effective_bits": 2.0625,
                            "block_payload_bytes": 66, "packing": "ggml"}
        elif mod and (mod.endswith("latent_proj") or mod.endswith("embeddings") or mod.endswith("lm_head")):
            q = _q(a, "Q8_0")
            put(key, "U8", q.reshape(a.shape[0], -1, 34))
            ref[key] = _deq(q, "Q8_0")
            qlayers[mod] = {"quant_algo": "Q8_0", "group_size": 32, "effective_bits": 8.5,
                            "block_payload_bytes": 34, "packing": "ggml"}
        else:
            bf = torch.from_numpy(a).to(torch.bfloat16)
            put(key, "BF16", bf.view(torch.int16).numpy().view(np.uint16))
            ref[key] = bf.float().numpy()
    # a vision tower + projector that no text lane may touch
    for n, shp in (("vision_model.radio_model.model.blocks.0.attn.qkv.weight", (96, 32)),
                   ("vision_model.radio_model.model.blocks.0.attn.qkv.bias", (96,)),
                   ("vision_model.radio_model.model.blocks.0.mlp.fc1.weight", (64, 32)),
                   ("mlp1.0.weight", (128,)), ("mlp1.1.weight", (H, 128)), ("mlp1.3.weight", (H, H))):
        bf = torch.randn(*shp).to(torch.bfloat16)
        put(n, "BF16", bf.view(torch.int16).numpy().view(np.uint16))
        ref[n] = bf.float().numpy()
    names = sorted(tensors)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    for fn, ks in shards.items():
        S._st_write(os.path.join(out, fn), {k: tensors[k] for k in ks}, {"format": "pt", "native_iq2_layers": "{}"})
    json.dump({"metadata": {}, "weight_map": {k: fn for fn, ks in shards.items() for k in ks}},
              open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
    cfg = {"architectures": ["NemotronH_Omni_Reasoning_V3"], "model_type": "nemotron_h_omni",
           "auto_map": {"AutoConfig": "configuration_nemotron_h_omni.NemotronH_Omni_Reasoning_V3_Config",
                        "AutoModelForCausalLM": "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3"},
           "llm_config": llm_config(), "vision_config": {"model_type": "radio"},
           "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                   "producer": {"name": "modelopt", "version": "0.37.0"},
                                   "quantized_layers": qlayers, "ignore": []}}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    import tiny_models
    tiny_models.tokenizer(out)
    for f in ("configuration_nemotron_h_omni.py", "modeling_nemotron_h_omni.py", "configuration_nemotron_h_puzzle.py",
              "modeling_nemotron_h_puzzle.py", "modeling_nemotron_h.py"):
        open(os.path.join(out, f), "w").write(STUB)
    return {"float": ref, "quantized_layers": qlayers}
