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
        # The caller does not know where this model lives, so move the ids rather than making every
        # caller guess -- "Passed CPU tensor to MPS op" is otherwise the first thing a Mac user sees.
        return self.stack.embed_tokens(ids.to(self.stack.embed_tokens.weight.device))

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
        """mlx array -> torch. Cast to float32 FIRST.

        numpy has no bfloat16, and most quantized MLX models compute in it, so viewing the buffer
        directly dies with "'bfloat16' is not a valid PEP 3118 buffer format string" -- which reads
        as a broken model rather than a dtype that cannot cross the boundary.
        """
        import numpy as np
        return self.torch.from_numpy(np.array(a.astype(self.mx.float32), copy=False)).float()

    def _m(self, t):
        """torch -> mlx, preserving integer dtype.

        This carries both token IDS and EMBEDDINGS. Casting everything to float breaks the id path --
        gather refuses non-integral indices -- and casting nothing breaks the embedding path, because
        numpy cannot view bfloat16. So the dtype decides.
        """
        t = t.detach().cpu()
        if not t.is_floating_point():
            return self.mx.array(t.numpy())
        return self.mx.array(t.float().numpy())

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
    """exllamav3.

    Model.forward() takes ids only, but the layer walk underneath it does not: forward_ls() iterates
    `fwd_modules` and hands each module's output to the next. The first module is the embedding and
    the last is the head, so skipping the first lets embeddings IN and stopping before the last lets
    the hidden state OUT -- which is precisely the pair a brain needs. No fork, no patch; the seam
    was already there.
    """
    name = "exl3"

    def __init__(self, model, cache, tok):
        import torch
        self.torch, self.model, self.cache, self.tok = torch, model, cache, tok
        self.hidden_size = int(model.config.hidden_size)

    def _mods(self):
        return list(self.model.fwd_modules)

    def embed(self, ids):
        module, instance, _ = self._mods()[0]
        params = {"layer_instance": instance}
        return module.forward(module.prepare_for_device(ids, params), params)

    def forward(self, embeds, k_skip=0):
        mods = self._mods()
        params = {}
        x = embeds
        for module, instance, _ in mods[1:-1]:      # past the embedding, up to the head
            params["layer_instance"] = instance
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        h = x                                        # the last hidden state, before the head
        module, instance, _ = mods[-1]
        params["layer_instance"] = instance
        logits = module.forward(module.prepare_for_device(h, params), params)
        return (logits[:, k_skip:], h[:, k_skip:].float())

    def forward_ids(self, ids):
        return self.forward(self.embed(ids))[1]

    def out_weight(self):
        """The head's weight, dequantized.

        EXL3 stores trellis-quantized weights, so whatever the head holds is not a plain matrix. Any
        packed tensor handed to the brain becomes a codebook of bit-patterns and every stored token
        decodes to noise -- the same failure the MLX lane had -- so this insists on a real one.
        """
        head = self._mods()[-1][0]
        for attr in ("get_weight_tensor", "get_weight", "unpack"):
            fn = getattr(head, attr, None)
            if callable(fn):
                w = fn()
                if hasattr(w, "shape") and len(w.shape) == 2 and w.shape[-1] == self.hidden_size:
                    return w.float()
        w = getattr(head, "weight", None)
        if w is not None and w.shape[-1] == self.hidden_size:
            return w.float()
        raise RuntimeError(
            "could not get an unpacked output embedding from this exllamav3 head. The brain derives "
            "its token codes from it, and a packed tensor would build a codebook of bit-patterns "
            "that decodes to noise with no error raised.")


class LlamaCpp(Backend):
    """llama.cpp, through the low-level bindings rather than the convenience wrapper.

    The high-level Llama.eval() takes tokens only, which is why this lane looked closed. The C API
    underneath takes both halves a brain needs and llama-cpp-python exposes all of it:
    llama_batch_init(n_tokens, embd, n_seq) allocates a batch whose `embd` field carries EMBEDDINGS
    instead of ids, and llama_get_embeddings_ith() returns the final hidden state per token once the
    context is put in embeddings mode. So a GGUF can host a brain; it needed a binding, not a fork.

    Two things that decide whether this works at all: the context must be created with embeddings
    enabled AND pooling set to NONE, or llama.cpp returns one pooled vector for the whole sequence
    instead of one per token, and a brain addressing on a pooled mean has nothing to key on.
    """
    name = "gguf"

    def __init__(self, llama, path=None):
        import ctypes
        import torch
        import llama_cpp.llama_cpp as C
        self.C, self.ct, self.torch = C, ctypes, torch
        self.llama = llama
        self.path = path or getattr(llama, 'model_path', None)
        self._ow = None
        self.ctx = llama._ctx.ctx
        self.model = llama._model.model
        self.hidden_size = int(C.llama_model_n_embd(self.model))
        C.llama_set_embeddings(self.ctx, True)

    def embed(self, ids):
        """Token ids -> embeddings, by running the input layer alone.

        llama.cpp does not hand out its embedding matrix, so this decodes the ids and reads the
        per-token states back. It costs a forward pass, which is why the brain embeds a window once
        and reuses it rather than calling this per token.
        """
        import numpy as np
        toks = ids.flatten().tolist()
        self.C.llama_memory_clear(self.C.llama_get_memory(self.ctx), True)
        batch = self.C.llama_batch_init(len(toks), 0, 1)
        try:
            for k, t in enumerate(toks):
                batch.token[k] = t
                batch.pos[k] = k
                batch.n_seq_id[k] = 1
                batch.seq_id[k][0] = 0
                batch.logits[k] = 1
            batch.n_tokens = len(toks)
            if self.C.llama_decode(self.ctx, batch) != 0:
                raise RuntimeError("llama_decode failed while embedding")
            rows = [np.ctypeslib.as_array(self.C.llama_get_embeddings_ith(self.ctx, k),
                                          (self.hidden_size,)).copy() for k in range(len(toks))]
        finally:
            self.C.llama_batch_free(batch)
        return self.torch.from_numpy(np.stack(rows)).float().unsqueeze(0)

    def forward(self, embeds, k_skip=0):
        """Run FROM embeddings: the batch carries `embd`, not ids."""
        import numpy as np
        x = embeds[0].detach().cpu().float().numpy()
        n = x.shape[0]
        self.C.llama_memory_clear(self.C.llama_get_memory(self.ctx), True)
        batch = self.C.llama_batch_init(n, self.hidden_size, 1)
        try:
            flat = x.reshape(-1)
            for k in range(flat.size):
                batch.embd[k] = float(flat[k])
            for k in range(n):
                batch.pos[k] = k
                batch.n_seq_id[k] = 1
                batch.seq_id[k][0] = 0
                batch.logits[k] = 1
            batch.n_tokens = n
            if self.C.llama_decode(self.ctx, batch) != 0:
                raise RuntimeError("llama_decode failed on an embedding batch")
            h = np.stack([np.ctypeslib.as_array(
                self.C.llama_get_embeddings_ith(self.ctx, k), (self.hidden_size,)).copy()
                for k in range(n)])
            lg = self.C.llama_get_logits(self.ctx)
            V = int(self.C.llama_vocab_n_tokens(self.C.llama_model_get_vocab(self.model)))
            logits = np.ctypeslib.as_array(lg, (n, V)).copy()
        finally:
            self.C.llama_batch_free(batch)
        t = self.torch
        return (t.from_numpy(logits).float().unsqueeze(0)[:, k_skip:],
                t.from_numpy(h).float().unsqueeze(0)[:, k_skip:])

    def forward_ids(self, ids):
        return self.forward(self.embed(ids))[1]

    def out_weight(self):
        """The output embedding, read from the GGUF FILE and dequantized.

        llama.cpp does not hand this matrix to Python, which looked like it needed an upstream patch.
        It does not: the tensor is in the file. A GGUF carries `output.weight`, or `token_embd.weight`
        when the model ties them, and the gguf package reads and dequantizes either. So the lane is
        self-sufficient with no fork and no binding change -- the brain's token codes come from the
        vocabulary, and the vocabulary is in the artifact.

        Read once and cached: dequantizing a 151936 x 896 matrix is not something to repeat per call.
        """
        if getattr(self, "_ow", None) is not None:
            return self._ow
        import numpy as np
        from gguf import GGUFReader, quants
        r = GGUFReader(self.path)
        by = {t.name: t for t in r.tensors}
        t = by.get("output.weight") or by.get("token_embd.weight")
        if t is None:
            raise RuntimeError(f"{self.path} has neither output.weight nor token_embd.weight")
        w = np.array(t.data)
        if str(t.tensor_type) not in ("GGMLQuantizationType.F32", "GGMLQuantizationType.F16"):
            w = quants.dequantize(w, t.tensor_type)
        w = w.astype(np.float32).reshape(int(t.shape[1]), int(t.shape[0]))
        self._ow = self.torch.from_numpy(w)
        return self._ow


class VLLM(Backend):
    """vLLM -- the serving lane.

    vLLM takes embeddings: `--enable-prompt-embeds`, then a `prompt_embeds` tensor of shape
    (seq_len, hidden_size). That is the half that looked hardest and it is already there.

    The other half is narrower than under transformers. Per-token hidden states come from vLLM's
    POOLING runner, not from generate(), so one engine cannot hand back logits and per-token states
    in a single call through the public API. That costs less than it sounds: recall() reads the
    memory and decodes signs, and never touches logits at all. So a brain SERVES under vLLM for
    retrieval, and training still belongs on the transformers path where both are free.

    Pooling must be NONE. The default pools a sequence to one vector, and a brain addressing on a
    pooled mean has nothing per-token to key on -- the same trap as llama.cpp, for the same reason.
    """
    name = "vllm"

    def __init__(self, llm, tok, pooling_llm=None):
        import torch
        self.torch, self.llm, self.tok = torch, llm, tok
        self.pool = pooling_llm or llm
        cfg = llm.llm_engine.model_config.hf_config
        self.hidden_size = int(getattr(cfg, "hidden_size", 0) or cfg.text_config.hidden_size)

    def embed(self, ids):
        emb = self.llm.llm_engine.model_executor.driver_worker.model_runner.model \
            .get_input_embeddings(ids.to("cuda"))
        return emb.float().cpu()

    def forward(self, embeds, k_skip=0):
        raise NotImplementedError(
            "vLLM does not return logits and per-token hidden states from one call. Use "
            "forward_ids()/hidden states for retrieval -- recall() needs no logits -- and train on "
            "the transformers backend, where both come back together.")

    def forward_ids(self, ids):
        """Per-token hidden states, through the pooling runner with pooling disabled."""
        out = self.pool.encode(self.tok.decode(ids[0].tolist()), pooling_task="token")
        h = out[0].outputs.data
        return self.torch.as_tensor(h).float().unsqueeze(0)

    def out_weight(self):
        m = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        w = getattr(getattr(m, "lm_head", None), "weight", None)
        if w is None:
            w = m.get_input_embeddings().weight
        return w.float().cpu()


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
            import llama_cpp.llama_cpp as C
        except ImportError:
            raise SystemExit("the GGUF lane needs llama-cpp-python") from None
        # pooling NONE is not optional: the default pools the sequence into ONE vector, and a brain
        # addressing on a pooled mean has nothing per-token to key on.
        if not os.path.isfile(model_id):
            raise SystemExit(
                f"the GGUF lane needs a path to a .gguf FILE, not {model_id!r}. "
                "Quantize one first (`pollard --hf <id> --format gguf --run`) and pass that file.")
        return LlamaCpp(Llama(model_path=model_id, n_ctx=kw.get("n_ctx", 4096),
                              embedding=True, pooling_type=C.LLAMA_POOLING_TYPE_NONE,
                              verbose=False), path=model_id)
    if kind == "vllm":
        try:
            from vllm import LLM
        except ImportError:
            raise SystemExit("the vLLM lane needs vllm") from None
        from transformers import AutoTokenizer
        # enable_prompt_embeds is what lets memory tokens in at all; without it vLLM accepts
        # text and ids only and a brain has no way to deliver anything.
        llm = LLM(model=model_id, enable_prompt_embeds=True,
                  runner=kw.get("runner", "pooling"), **kw.get("engine", {}))
        return VLLM(llm, AutoTokenizer.from_pretrained(model_id))
    raise SystemExit(f"unknown runtime {kind!r}: auto, mlx, exl3, gguf, vllm")
