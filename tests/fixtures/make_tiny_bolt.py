"""Tiny random Nemotron-3.5-Bolt-shaped checkpoint for the streaming lanes: what NVIDIA's 24GB release looks like on
disk, at toy size.

  * config.json: an omni wrapper (model_type nemotron_h_omni, an auto_map naming custom modelling files) whose
    `llm_config` is a `nemotron_h_puzzle` -- per-layer block configs (mamba / moe / attention / moe) with a
    different expert count and width per MoE layer, latent MoE (fc1/fc2_latent_proj), relu2 experts.
  * weights under `language_model.` (backbone.* + lm_head) plus a whole tiny RADIO vision tower (`vision_model.*`,
    the released key names) and the projector (`mlp1.0/1/3`) the text lanes must copy through untouched and the MLX
    lane builds with mlx-vlm. Processor configs as NVIDIA ships them.
  * modelopt MIXED_PRECISION encoding, as NVIDIA ships it: the mamba in_proj / out_proj in W4A16 NVFP4 (packed U8 +
    F8_E4M3 block scales + F32 weight_scale_2), every routed and shared expert ggml-packed IQ2_XXS inside the
    safetensors (U8 [rows, blocks, 66]), the latent projections, embeddings and lm_head ggml Q8_0.
  * the custom modelling / processing files are stubs that raise if imported: Pollard must build the backbone from
    transformers' native NemotronH (and MLX from mlx-vlm's own model), never from them.

make(out, encoding="bf16") writes the same model unquantized, as the -BF16 release ships it.

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
            "num_key_value_heads": 2, "head_dim": 64, "mamba_num_heads": 8, "mamba_head_dim": 32, "ssm_state_size": 32,
            "n_groups": 2, "conv_kernel": 4, "expand": 2, "chunk_size": 16, "moe_latent_size": LAT,
            "moe_shared_expert_intermediate_size": 256, "n_shared_experts": 1, "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
            "routed_scaling_factor": 2.5, "mlp_hidden_act": "relu2",
            "num_experts": 8, "num_local_experts": 8,          # global values the per-layer blocks must override "hidden_act": "relu2",
            "mamba_hidden_act": "silu", "layer_norm_epsilon": 1e-5, "tie_word_embeddings": False,
            "attention_bias": False, "mamba_proj_bias": False, "mlp_bias": False, "use_bias": False,
            "use_conv_bias": True,
            "time_step_limit": [0.0, {"__float__": "Infinity"}], "use_mamba_kernels": False,
            "block_configs": BLOCKS, "num_hidden_layers": len(BLOCKS), "max_position_embeddings": 4096,
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}


STUB = 'raise ImportError("Pollard must not import the checkpoint\'s own modelling code here")\n'

VIT, PROJ, CPE = 32, H, 64          # RADIO width, projector width, CPE grid (pixels): 4x4 position embeddings


def vision_config():
    return {"model_type": "radio", "hidden_size": VIT, "num_hidden_layers": 1, "num_attention_heads": 4,
            "mlp_ratio": 4.0, "hidden_act": "gelu", "layer_norm_eps": 1e-6, "image_size": 224, "patch_size": 16,
            "max_img_size": CPE, "num_cls_tokens": 3, "num_registers": 7, "summary_idxs": [0, 1],
            "qkv_bias": True, "video_temporal_patch_size": 2, "min_num_patches": 4, "max_num_patches": 64,
            "args": {"model": "vit_huge_patch16_224", "cls_token_per_teacher": True, "cpe_max_size": CPE,
                     "register_multiple": 10, "teachers": [{"name": 0, "use_summary": True},
                                                           {"name": 1, "use_summary": True},
                                                           {"name": 2, "use_summary": False}]}}


def vision_tensors():
    """The released RADIO + projector key names and shapes, for the tiny tower."""
    b, pg = "vision_model.radio_model.model.blocks.0.", "vision_model.radio_model.model.patch_generator."
    return {pg + "cls_token.token": (10, VIT), pg + "embedder.weight": (VIT, 3 * 16 * 16),
            pg + "video_embedder.weight": (VIT, 2 * 3 * 16 * 16), pg + "pos_embed": (1, (CPE // 16) ** 2, VIT),
            b + "norm1.weight": (VIT,), b + "norm1.bias": (VIT,), b + "norm2.weight": (VIT,), b + "norm2.bias": (VIT,),
            b + "attn.qkv.weight": (3 * VIT, VIT), b + "attn.qkv.bias": (3 * VIT,),
            b + "attn.proj.weight": (VIT, VIT), b + "attn.proj.bias": (VIT,),
            b + "mlp.fc1.weight": (4 * VIT, VIT), b + "mlp.fc1.bias": (4 * VIT,),
            b + "mlp.fc2.weight": (VIT, 4 * VIT), b + "mlp.fc2.bias": (VIT,),
            "mlp1.0.weight": (4 * VIT,), "mlp1.1.weight": (PROJ, 4 * VIT), "mlp1.3.weight": (H, PROJ)}


IMAGE_PROCESSOR = {"_downsample_factor": 2, "downsample_ratio": 0.5, "patch_size": 16, "max_model_len": 16384,
                   "image_processor_type": "NemotronH_Omni_Reasoning_V3ImageProcessor",
                   "auto_map": {"AutoImageProcessor":
                                "image_processing_nemotron_h_omni.NemotronH_Omni_Reasoning_V3ImageProcessor",
                                "AutoProcessor": "processing_nemotron_h_omni.NemotronH_Omni_Reasoning_V3Processor"},
                   "max_num_patches": 64, "min_num_patches": 4, "norm_mean": [0.48145466, 0.4578275, 0.40821073],
                   "norm_std": [0.26862954, 0.26130258, 0.27577711], "rescale_factor": 1 / 255,
                   "video_maintain_aspect_ratio": True, "video_target_num_patches": 16}


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


def make(out, encoding="modelopt"):
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
        if encoding == "bf16":
            mod = None                                   # the -BF16 release: every tensor plain BF16
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
    # the vision tower + projector no text lane may touch (the released RADIO key names, a whole tiny tower)
    for n, shp in vision_tensors().items():
        scale = len(shp) == 1 and "bias" not in n                      # a norm's scale: near 1
        bf = ((1 if scale else 0) + (0.1 if scale else 0.05) * torch.randn(*shp)).to(torch.bfloat16)
        put(n, "BF16", bf.view(torch.int16).numpy().view(np.uint16))
        ref[n] = bf.float().numpy()
    for n, v in (("vision_model.radio_model.input_conditioner.norm_mean", IMAGE_PROCESSOR["norm_mean"]),
                 ("vision_model.radio_model.input_conditioner.norm_std", IMAGE_PROCESSOR["norm_std"])):
        put(n, "F32", np.asarray(v, np.float32).reshape(3, 1, 1))
        ref[n] = np.asarray(v, np.float32).reshape(3, 1, 1)
    put("vision_model.summary_idxs", "I64", np.array([0, 1], np.int64))
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
           "downsample_ratio": 0.5, "force_image_size": 64, "patch_size": 16, "img_context_token": "<image>",
           "img_context_token_id": 18, "image_tag_type": "internvl", "ps_version": "v2",
           "projector_hidden_size": PROJ, "vit_hidden_size": VIT, "vision_final_layernorm": False,
           "template": "n5h_5p5_nanov2", "video_pruning_rate": 0.7, "video_temporal_patch_size": 2,
           "sound_config": None, "eos_token_id": 2,
           "llm_config": llm_config(), "vision_config": vision_config()}
    if encoding != "bf16":
        cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                      "producer": {"name": "modelopt", "version": "0.37.0"},
                                      "quantized_layers": qlayers, "ignore": []}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    json.dump(IMAGE_PROCESSOR, open(os.path.join(out, "preprocessor_config.json"), "w"), indent=1)
    json.dump({"auto_map": {"AutoProcessor": "processing_nemotron_h_omni.NemotronH_Omni_Reasoning_V3Processor"},
               "image_processor": IMAGE_PROCESSOR, "processor_class": "NemotronH_Omni_Reasoning_V3Processor",
               "video_temporal_patch_dim": 2}, open(os.path.join(out, "processor_config.json"), "w"), indent=1)
    import tiny_models
    tiny_models.tokenizer(out)
    for f in ("configuration_nemotron_h_omni.py", "modeling_nemotron_h_omni.py", "configuration_nemotron_h_puzzle.py",
              "modeling_nemotron_h_puzzle.py", "modeling_nemotron_h.py", "processing_nemotron_h_omni.py",
              "image_processing_nemotron_h_omni.py", "configuration_radio.py", "modeling_radio.py"):
        open(os.path.join(out, f), "w").write(STUB)
    return {"float": ref, "quantized_layers": qlayers}
