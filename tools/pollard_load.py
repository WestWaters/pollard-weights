#!/usr/bin/env python3
"""pollard_load -- how Pollard loads a model, including a vision-language one.

This is MODEL infrastructure. Pollard quantizes vision-language models like any other, so reaching a
VL backbone is a Pollard capability and belongs here, owned by the model path -- not inherited from
some other feature that happened to need it first.

Two things every tool needs and each was reimplementing:

  load_backbone()  AutoModelForCausalLM refuses a vision-language config outright --
                   "Unrecognized configuration class Qwen2VLConfig for this kind of AutoModel" --
                   so a VL checkpoint could not be opened at all, even though probing, smoothing,
                   abliterating and KL all work on the text stack a VL model keeps inside it.
                   Try the causal LM, fall back to the vision-language auto classes, and report
                   BOTH failures if neither works rather than only the last one.

  text_layers()    The decoder layer list, wherever this family keeps it. A VL model's text stack
                   lives under `model.language_model`; a text model's is `model.model`. Tools that
                   count layers or iterate them need the same answer for both, and reaching for
                   `model.model.layers` directly is how a VL model turns into an AttributeError
                   three steps into a build.

Anything else may import this. It imports nothing of Pollard's in return, so it cannot drag another
feature's concerns into the build path.
"""
from __future__ import annotations


def load_backbone(model_id: str, dtype=None, device: str = "cpu", eval_mode: bool = True, **kw):
    """Load `model_id` under whichever auto class accepts it, text-only or vision-language.

    dtype defaults to float32; pass torch.float16 etc for the lanes that want it. Extra kwargs
    (trust_remote_code, attn_implementation, ...) pass straight through.

    Pass `device_map=` (plus the usual `max_memory=` / `offload_folder=`) to shard a model bigger
    than the accelerator across GPU+CPU+disk. accelerate owns placement once a model is dispatched
    that way and moving it afterwards raises, so `device` is ignored in that case.
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
    if kw.get("device_map") is None:            # accelerate already placed a dispatched model
        model = model.to(device)
    return model.eval() if eval_mode else model


def text_layers(model):
    """The decoder layer list, wherever this family keeps it."""
    for path in ("model.language_model", "language_model.model", "model"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        layers = getattr(node, "layers", None) if node is not None else None
        if layers is not None:
            return layers
    layers = getattr(model, "layers", None)
    if layers is not None:
        return layers
    raise SystemExit(f"could not find the decoder layers on {type(model).__name__}; "
                     "pollard_load.text_layers needs a path for this architecture")
