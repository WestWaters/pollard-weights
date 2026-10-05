"""Model loader shim referenced by the tPD config (`pretrained_model_class: tpd_models.Qwen3ForTPD`).

tPD's lm_decomposition.py calls `resolve_class(cfg.pretrained_model_class).from_pretrained(name)`.
A plain `transformers.AutoModelForCausalLM` would load Qwen3 in its checkpoint dtype (bf16 under
transformers>=5), and tPD then builds its U/V components in that dtype -> the decomposition would
be *trained* in bf16 params. This shim pins fp32 master weights (tPD's own bf16 autocast still
does the matmuls in bf16) and turns the KV cache off (it is pure waste during decomposition).

This directory must be on PYTHONPATH (run_tpd.py / quant_arms.py do that for you).
"""

from __future__ import annotations

import torch


def _load_causal_lm(name_or_path: str, dtype: torch.dtype = torch.float32):
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, dtype=dtype, attn_implementation="sdpa"
        )
    except TypeError:  # transformers < 4.56 spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, torch_dtype=dtype, attn_implementation="sdpa"
        )
    model.config.use_cache = False
    return model


class Qwen3ForTPD:
    """Factory with the `from_pretrained` classmethod tPD expects. Works for any HF causal LM."""

    @classmethod
    def from_pretrained(cls, name_or_path: str, **_: object):
        return _load_causal_lm(name_or_path, torch.float32)
