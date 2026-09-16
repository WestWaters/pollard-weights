#!/usr/bin/env python3
"""Runtimes a brain can attach to.

A FlyBrain needs exactly four things from a backbone, and nothing else:

  embed(ids)             token ids -> embeddings, so memory tokens can be prepended
  forward(embeds)        run from EMBEDDINGS and hand back (logits, last hidden state)
  forward_ids(ids)       run from ids, last hidden state only -- for the probe bridge
  out_weight()           the output embedding, for the brain's vote and its token codes

Everything else about the brain is runtime-independent. The reason brains ran only under
transformers was not the memory, the connectome or the payload -- it was that those four calls were
written inline against one library. They are a protocol here instead.

The hard requirement is `forward(embeds)`. A runtime that cannot be fed embeddings cannot host a
brain at all, because the memory is delivered by prepending vectors to the sequence. That rules out
plain text-in/text-out server APIs and nothing else: llama.cpp, MLX and exllamav3 all accept
embeddings, they just each spell it differently.
"""
from __future__ import annotations

import os


class Backend:
    """What a brain needs from whatever is running the model."""
    name = "?"
    hidden_size = 0

    def embed(self, ids):              raise NotImplementedError
    def forward(self, embeds):         raise NotImplementedError
    def forward_ids(self, ids):        raise NotImplementedError
    def out_weight(self):              raise NotImplementedError


class Transformers(Backend):
    """HF transformers. The reference path -- every published number was measured here."""
    name = "transformers"

    def __init__(self, model, stack, hidden):
        self.model, self.stack, self.hidden_size = model, stack, hidden

    def embed(self, ids):
        return self.stack.embed_tokens(ids)

    def forward(self, embeds, k_skip=0):
        o = self.model(inputs_embeds=embeds, output_hidden_states=True)
        h = o.hidden_states[-1]
        return (o.logits[:, k_skip:], h[:, k_skip:].float())

    def forward_ids(self, ids):
        return self.model(input_ids=ids, output_hidden_states=True).hidden_states[-1].float()

    def out_weight(self):
        return (self.model.get_output_embeddings() or self.stack.embed_tokens).weight


class MLX(Backend):
    """Apple MLX via mlx-lm.

    An mlx_lm model is a callable module, so embeddings go in directly and hidden states come back
    the same way as under transformers -- this is the cheapest lane to support, not the hardest.
    MLX arrays are converted at the boundary so the brain's own arithmetic stays in torch; the
    memory is 11 MB and the conversion is not what costs time here.
    """
    name = "mlx"

    def __init__(self, model, tok):
        import mlx.core as mx
        import torch
        self.mx, self.torch, self.model, self.tok = mx, torch, model, tok
        self.hidden_size = int(model.args.hidden_size)

    def _t(self, a):
        import numpy as np
        return self.torch.from_numpy(np.array(a, copy=False)).float()

    def _m(self, t):
        return self.mx.array(t.detach().cpu().numpy())

    def embed(self, ids):
        return self._t(self.model.model.embed_tokens(self._m(ids)))

    def forward(self, embeds, k_skip=0):
        # mlx-lm spells it input_embeddings, and `inputs` is positional-but-ignored when it is given.
        # Tied-embedding models (most small Qwens) have no lm_head at all; the embedding matrix is
        # reused as the output projection, which mlx exposes as .as_linear().
        h = self.model.model(None, input_embeddings=self._m(embeds))
        logits = self.model.lm_head(h) if hasattr(self.model, "lm_head") else \
            self.model.model.embed_tokens.as_linear(h)
        return (self._t(logits)[:, k_skip:], self._t(h)[:, k_skip:])

    def forward_ids(self, ids):
        return self._t(self.model.model(self._m(ids)))

    def out_weight(self):
        """The output embedding, DEQUANTIZED.

        In a quantized MLX model this tensor is packed -- a 4-bit Qwen reports (151936, 112) where
        the real matrix is (151936, 896). The brain derives its token codes from this, so handing it
        packed bytes builds a codebook out of bit-patterns and every stored token decodes to noise,
        with nothing anywhere raising an error.
        """
        mod = getattr(self.model, "lm_head", None) or self.model.model.embed_tokens
        w = mod.weight
        if hasattr(mod, "scales"):                      # quantized: unpack before use
            w = self.mx.dequantize(w, mod.scales, mod.biases,
                                   group_size=getattr(mod, "group_size", 64),
                                   bits=getattr(mod, "bits", 4))
        return self._t(w)


class ExLlamaV3(Backend):
    """exllamav3. Accepts input embeddings and can return hidden states per layer."""
    name = "exl3"

    def __init__(self, model, cache, tok):
        import torch
        self.torch, self.model, self.cache, self.tok = torch, model, cache, tok
        self.hidden_size = int(model.config.hidden_size)

    def embed(self, ids):
        return self.model.modules[0].forward(ids)

    def forward(self, embeds, k_skip=0):
        out = self.model.forward(input_embeddings=embeds, cache=self.cache,
                                 return_last_state=True)
        logits = out["logits"] if isinstance(out, dict) else out
        h = out.get("last_state") if isinstance(out, dict) else None
        if h is None:
            raise RuntimeError("this exllamav3 build does not return hidden states; "
                               "a brain needs them to address memory")
        return (logits[:, k_skip:], h[:, k_skip:].float())

    def forward_ids(self, ids):
        return self.forward(self.embed(ids))[1]

    def out_weight(self):
        return self.model.modules[-1].get_weight_tensor()


class LlamaCpp(Backend):
    """llama.cpp through llama-cpp-python.

    llama.cpp takes embeddings (llama_batch carries an `embd` field) and exposes the final hidden
    state, so a GGUF CAN host a brain -- it is a binding job, not a wall. What it will not do is
    stream hidden states from an arbitrary layer, so the brain reads the last one, which is what it
    uses anyway.
    """
    name = "gguf"

    def __init__(self, llama):
        import torch
        self.torch, self.llama = torch, llama
        self.hidden_size = int(llama.n_embd())

    def embed(self, ids):
        import numpy as np
        tbl = self.llama.token_get_embeddings() if hasattr(self.llama, "token_get_embeddings") else None
        if tbl is None:
            raise RuntimeError("this llama-cpp-python build does not expose the embedding table; "
                               "build with LLAMA_CPP_EXPOSE_EMBD=1 or use the transformers copy")
        return self.torch.from_numpy(np.asarray(tbl)[ids.cpu().numpy()]).float()

    def forward(self, embeds, k_skip=0):
        raise NotImplementedError(
            "llama-cpp-python does not yet expose embedding input plus hidden-state output through "
            "its Python API. The C API supports both (llama_batch.embd, llama_get_embeddings), so "
            "this is a binding to write, not a limitation of the format.")

    def forward_ids(self, ids):
        raise NotImplementedError(self.forward.__doc__)

    def out_weight(self):
        raise NotImplementedError


def open_backend(kind: str, model_id: str, device: str = "cpu", **kw) -> Backend:
    """Open a backbone under the named runtime."""
    if kind in ("auto", "transformers", "hf"):
        from pollard_flybrain import FlyBrain, load_backbone
        m = load_backbone(model_id, device=device)
        stack = FlyBrain._find_stack(m)
        return Transformers(m, stack, FlyBrain._hidden_size(m, stack))
    if kind == "mlx":
        try:
            from mlx_lm import load
        except ImportError:
            raise SystemExit("the MLX lane needs mlx-lm: pip install mlx-lm") from None
        model, tok = load(model_id)
        return MLX(model, tok)
    if kind == "exl3":
        try:
            from exllamav3 import Config, Model, Cache, Tokenizer
        except ImportError:
            raise SystemExit("the EXL3 lane needs exllamav3") from None
        cfg = Config.from_directory(model_id)
        model = Model.from_config(cfg); model.load()
        return ExLlamaV3(model, Cache(model, max_num_tokens=4096), Tokenizer.from_config(cfg))
    if kind == "gguf":
        try:
            from llama_cpp import Llama
        except ImportError:
            raise SystemExit("the GGUF lane needs llama-cpp-python") from None
        return LlamaCpp(Llama(model_path=model_id, embedding=True, logits_all=True, verbose=False))
    raise SystemExit(f"unknown runtime {kind!r}: auto, mlx, exl3, gguf")
