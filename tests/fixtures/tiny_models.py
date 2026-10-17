"""Tiny random checkpoints for the streamed lanes (pollard_stream): a byte-level tokenizer that needs no download,
and dense / MoE / hybrid / omni / custom-code models of a few hundred KB. Seeded; every builder returns the folder."""
import json, os


#: the named special tokens Qwen3-Omni's processor reads off its tokenizer (clef-omni's tokenizer_config.json)
OMNI_TOKENS = {"audio_bos_token": "<|audio_start|>", "audio_eos_token": "<|audio_end|>", "audio_token": "<|audio_pad|>",
               "image_token": "<|image_pad|>", "video_token": "<|video_pad|>", "vision_bos_token": "<|vision_start|>",
               "vision_eos_token": "<|vision_end|>"}


def tokenizer(out, vocab=256, named=None):
    """A byte-level BPE tokenizer with no merges (ids 0..255), saved as a transformers fast tokenizer. `named`:
    extra named special tokens (ids from 256 up), e.g. OMNI_TOKENS."""
    from tokenizers import Tokenizer, decoders, models
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast
    alphabet = sorted(ByteLevel.alphabet())
    tok = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet[:vocab])}, merges=[]))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    kw = {"extra_special_tokens": dict(named)} if named else {}
    PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=alphabet[0], pad_token=alphabet[0], **kw).save_pretrained(out)
    return out


def _save(model, out):
    model.save_pretrained(out)
    tokenizer(out)
    return out


def llama(out, layers=3, outlier=True, hidden=64):
    import torch, transformers as T
    torch.manual_seed(0)
    c = T.LlamaConfig(vocab_size=300, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    m = T.LlamaForCausalLM(c)
    if outlier:                                   # a massive-activation channel for the smoothing to find
        with torch.no_grad():
            m.model.embed_tokens.weight[:, 3] *= 30
    return _save(m, out)


def qwen3_moe(out, hidden=64, expert=32):
    import torch, transformers as T
    torch.manual_seed(0)
    c = T.Qwen3MoeConfig(vocab_size=300, hidden_size=hidden, intermediate_size=2 * hidden, moe_intermediate_size=expert,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=hidden // 4,
                         num_experts=4, num_experts_per_tok=2, max_position_embeddings=512)
    return _save(T.Qwen3MoeForCausalLM(c), out)


def lfm2_moe(out):
    import torch, transformers as T
    torch.manual_seed(0)
    c = T.Lfm2MoeConfig(vocab_size=300, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, num_experts=4,
                        num_experts_per_tok=2, num_dense_layers=1,
                        layer_types=["conv", "full_attention", "conv", "full_attention"], max_position_embeddings=512)
    return _save(T.Lfm2MoeForCausalLM(c), out)


CUSTOM_CONFIG = '''from transformers import LlamaConfig


class PollardCustomConfig(LlamaConfig):
    model_type = "pollard_custom_llama"
'''
CUSTOM_MODELING = '''import os
from transformers import LlamaForCausalLM
from .configuration_pollard_custom import PollardCustomConfig

# a side effect the test can see: the checkpoint's code ran
os.environ["POLLARD_TEST_CUSTOM_CODE_RAN"] = "1"


class PollardCustomForCausalLM(LlamaForCausalLM):
    config_class = PollardCustomConfig
'''


def custom_code(out):
    """A checkpoint whose config names an auto_map into its own .py files (model_type unknown to transformers)."""
    import torch, transformers as T
    torch.manual_seed(0)
    c = T.LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    _save(T.LlamaForCausalLM(c), out)
    cfg = json.load(open(os.path.join(out, "config.json")))
    cfg.update(model_type="pollard_custom_llama", architectures=["PollardCustomForCausalLM"],
               auto_map={"AutoConfig": "configuration_pollard_custom.PollardCustomConfig",
                         "AutoModelForCausalLM": "modeling_pollard_custom.PollardCustomForCausalLM"})
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    open(os.path.join(out, "configuration_pollard_custom.py"), "w").write(CUSTOM_CONFIG)
    open(os.path.join(out, "modeling_pollard_custom.py"), "w").write(CUSTOM_MODELING)
    return out


def omni(out):
    """A 2-layer Qwen3-Omni-MoE thinker (as clef-omni) with a token audio tower / visual tensor, a talker and a
    code2wav tensor and the release's processor files -- everything a lane must ship next to the language model."""
    import torch
    from safetensors.torch import load_file, save_file
    import make_tiny_clef_omni as M
    tok = tokenizer(out + "-tok")
    for f in ("vocab.json", "merges.txt"):                # make_tiny_clef_omni copies a BPE tokenizer's files
        if not os.path.exists(os.path.join(tok, f)):
            open(os.path.join(tok, f), "w").write("{}" if f.endswith(".json") else "#version: 0.2\n")
    M.make(out, tok)
    W = load_file(os.path.join(out, "model.safetensors"))
    g = torch.Generator().manual_seed(1)
    W["thinker.audio_tower.layers.0.self_attn.q_proj.weight"] = torch.randn(32, 32, generator=g).to(torch.bfloat16)
    W["thinker.visual.blocks.0.attn.qkv.weight"] = torch.randn(96, 32, generator=g).to(torch.bfloat16)
    W["talker.model.layers.0.self_attn.q_proj.weight"] = torch.randn(32, 32, generator=g).to(torch.bfloat16)
    W["code2wav.decoder.0.conv.weight"] = torch.randn(16, 8, 7, generator=g).to(torch.bfloat16)
    save_file(W, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    write_omni_processor_files(out)
    return out


#: what the real Qwen3-Omni release ships next to its weights for the audio / image / video inputs
OMNI_PREPROCESSOR = {"feature_extractor_type": "WhisperFeatureExtractor", "feature_size": 128, "hop_length": 160,
                     "n_fft": 400, "sampling_rate": 16000, "image_processor_type": "Qwen2VLImageProcessor",
                     "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "min_pixels": 3136,
                     "max_pixels": 65536, "merge_size": 2, "patch_size": 16, "temporal_patch_size": 2,
                     "processor_class": "Qwen3OmniMoeProcessor"}


def write_omni_processor_files(out):
    json.dump(OMNI_PREPROCESSOR, open(os.path.join(out, "preprocessor_config.json"), "w"), indent=1)
    json.dump({"processor_class": "Qwen3OmniMoeProcessor"}, open(os.path.join(out, "processor_config.json"), "w"))
    json.dump({"video_processor_type": "Qwen2VLVideoProcessor", "merge_size": 2},
              open(os.path.join(out, "video_preprocessor_config.json"), "w"))
    return out


def omni_full(out):
    """A COMPLETE tiny Qwen3-Omni-MoE (clef-omni's architecture) as transformers writes it: the 2-layer MoE thinker
    with its audio tower and vision tower, the talker (MoE + code predictor) and code2wav -- every tensor mlx-vlm's
    qwen3_omni_moe builds -- plus the release's processor files and a clef-style side weight (joint_head) next to the
    index. ~6M parameters, seeded."""
    import torch
    from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration
    torch.manual_seed(0)
    c = Qwen3OmniMoeConfig().to_dict()
    th = c["thinker_config"]
    th["audio_config"].update(encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=64, d_model=32,
                              output_dim=64, downsample_hidden_size=16)
    th["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                               deepstack_visual_indexes=[0])
    t = th["text_config"]
    t.update(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
             num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, intermediate_size=64,
             max_position_embeddings=4096, rope_theta=1000000.0)
    for key in ("rope_parameters", "rope_scaling"):          # interleaved M-RoPE, sections sum to head_dim / 2
        if key == "rope_parameters" or isinstance(t.get(key), dict):
            t[key] = dict(t.get(key) or {}, mrope_section=[3, 3, 2], mrope_interleaved=True, interleaved=True)
    tk = c["talker_config"]
    tk["text_config"].update(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=16, intermediate_size=64, moe_intermediate_size=16, num_local_experts=4,
                             num_experts_per_tok=2, shared_expert_intermediate_size=32, rope_theta=1000000.0)
    tk["code_predictor_config"].update(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                       num_key_value_heads=1, head_dim=16, intermediate_size=64, rope_theta=1000000.0)
    tk.update(thinker_hidden_size=64, accept_hidden_layer=1, spatial_merge_size=2)
    c["code2wav_config"].update(hidden_size=32, num_attention_heads=2, num_key_value_heads=2, intermediate_size=64,
                                num_hidden_layers=1, decoder_dim=32)

    def one_type_per_layer(d):
        if isinstance(d, dict):
            if d.get("layer_types") and isinstance(d.get("num_hidden_layers"), int):
                d["layer_types"] = d["layer_types"][:1] * d["num_hidden_layers"]
            for v in d.values():
                one_type_per_layer(v)
    one_type_per_layer(c)
    c.update(architectures=["Qwen3OmniMoeForConditionalGeneration"], enable_audio_output=True)
    m = Qwen3OmniMoeForConditionalGeneration(Qwen3OmniMoeConfig(**c)).to(torch.bfloat16)
    m.save_pretrained(out)
    cfg = json.load(open(os.path.join(out, "config.json")))

    def release_spelling(d):                                # the release spells these out (mlx-vlm reads them)
        if isinstance(d, dict):
            if isinstance(d.get("rope_parameters"), dict) and "rope_theta" in d["rope_parameters"]:
                d.setdefault("rope_theta", d["rope_parameters"]["rope_theta"])
            if "num_local_experts" in d:
                d.setdefault("num_experts", d["num_local_experts"])
            for v in d.values():
                release_spelling(v)
    release_spelling(cfg)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    from safetensors.torch import load_file, save_file
    W = load_file(os.path.join(out, "model.safetensors"))
    json.dump({"metadata": {"total_size": sum(v.numel() * v.element_size() for v in W.values())},
               "weight_map": {k: "model.safetensors" for k in sorted(W)}},
              open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
    g = torch.Generator().manual_seed(2)
    save_file({"hidden_norm.weight": torch.randn(64, generator=g), "type_embedding.weight": torch.randn(3, 32, generator=g)},
              os.path.join(out, "joint_head.safetensors"))
    json.dump({"routing_layers": 1, "layers": 1, "heads": 4}, open(os.path.join(out, "joint_head_config.json"), "w"))
    tokenizer(out, named=OMNI_TOKENS)
    write_omni_processor_files(out)
    return out
