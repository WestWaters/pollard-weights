#!/usr/bin/env python3
"""Load any backbone a Pollard tool needs -- text-only or vision-language.

AutoModelForCausalLM refuses a vision-language config outright:

    ValueError: Unrecognized configuration class Qwen2VLConfig for this kind of
    AutoModel: AutoModelForCausalLM

Eleven tools called it directly, so every one of them died on a VL model even though nothing they do
afterwards cares -- probing, smoothing, abliterating and KL all work on the text stack, which a VL
model keeps under `model.language_model`. The failure reads as an unsupported model rather than an
unasked question, and it sent a whole quant ladder into the ditch.

One loader, used everywhere: try the causal LM, fall back to the vision-language auto classes, and
report BOTH failures if neither works rather than only the last one.
"""
from __future__ import annotations


def load_backbone(model_id: str, dtype=None, device: str = "cpu", eval_mode: bool = True,
                  **kw):
    """Load `model_id` under whichever auto class accepts it.

    dtype defaults to float32; pass torch.float16 etc for the lanes that want it. Extra kwargs
    (trust_remote_code, attn_implementation, ...) pass straight through.
    """
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    if dtype is None:
        dtype = torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kw)
    except ValueError as text_only_err:
        model = None
        errs = [f"AutoModelForCausalLM: {text_only_err}"]
        for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(model_id, dtype=dtype, **kw)
                break
            except Exception as e:
                errs.append(f"{name}: {e}")
        if model is None:
            raise SystemExit(f"could not load {model_id!r} as a causal LM or a vision-language "
                             "model:\n  " + "\n  ".join(str(e)[:160] for e in errs)) from None
    model = model.to(device)
    return model.eval() if eval_mode else model


def text_layers(model):
    """The decoder layer list, wherever this family keeps it.

    A VL model's text stack lives under `model.language_model`; a text model's is `model.model`.
    Tools that count layers or iterate them need the same answer for both.
    """
    for path in ("model.language_model", "language_model.model", "model"):
        o = model
        try:
            for part in path.split("."):
                o = getattr(o, part)
            if hasattr(o, "layers"):
                return o.layers
        except AttributeError:
            continue
    raise ValueError("could not locate a decoder stack on this model")
