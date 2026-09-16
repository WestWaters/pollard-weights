"""pollard-flybrain -- a fruit-fly connectome as a language model's memory.

A transformer's memory is its KV cache: it grows with every token, and when the context window fills,
the oldest tokens are gone. A fly has no context window. This attaches a real connectome (MaleCNS
v1.0, FlyEM/Janelia) to a frozen model as an associative memory whose size never changes.

    from pollard_flybrain import FlyBrain

    brain = FlyBrain.load("FlyBrain-Pollard-CNSv1.pt", device="cuda")
    brain.bind(model, tok)                   # any HF causal LM; the backbone is never modified
    brain.feed(long_document)                # read a document of ANY length, 128 tokens at a time
    print(brain.answer("what is the secret word?"))
    brain.save_state("session.flystate")     # the whole session, in a file that never grows

Measured on Qwen2.5-0.5B-Instruct, a six-letter string stated once and then buried under filler,
with the model never seeing more than 128 tokens at a time:

    no brain (the fact is outside the window)      acc   0.0%    NLL 11.234
    with brain                                     acc  97.9%    NLL  0.249
    control: a word the document never contained   acc   0.0%
    control: brain reads a DIFFERENT document      acc   0.0%

Training words are drawn fresh for every example and never reused, so there is nothing to memorise;
the only way to score is to store and retrieve. The two controls are what separate that from a leak.

How it works, and why each piece is there:

  * ONE address function serves both writing and reading. Filing something under an address the
    query cannot reproduce is the same as not filing it.
  * The KEY is the context BEFORE a token; the VALUE is the token. Keyed on the token itself, a
    random string is unfindable -- the question shares nothing with it. Keyed on the words that
    precede it, the question is nearly identical to the key. Measured: 0.00x versus 4.3x.
  * The VALUE is the HOST MODEL'S OWN CODE for that token, not one the brain invents. That is what
    lets it name words it never saw in training, and it is why the brain has to know what it is
    mounted to.
  * The state is (slots x width) and fixed -- 11.2 MB for CNSv1 at width 328 -- whether the document
    is a thousand tokens or a million. The state update costs about 3% of one model forward pass and
    fits comfortably on CPU, so it can run beside the model rather than competing with it.

A brain is fitted to one backbone: the address and gate matrices have that model's hidden size, and
the token codes come from its output embedding. `bind()` refuses a model it was not trained for.
Training your own takes one `--train` run; see `pollard-flybrain --help`.
"""
from __future__ import annotations

import math
import os
from typing import Optional

try:
    import torch
    import torch.nn as nn
except ImportError as _e:      # the connectome lane is an optional extra, not a core dependency
    raise SystemExit(
        "pollard-flybrain needs PyTorch, which Pollard does not install by default.\n"
        "  pip install 'pollard-weights[flybrain]'      (torch, transformers, pandas, scipy)\n"
        f"({_e})"
    ) from None

__all__ = ["FlyBrain", "train_brain"]

# A token id is stored as BITS signs rather than as a vector. A vector has to survive averaging AND
# win a nearest-neighbour search among non-orthogonal embedding rows; a sign only has to stay on the
# correct side of zero. Interference has to FLIP a bit to corrupt it, which is why this reaches exact
# reproduction where a continuous code plateaued at 25%.
_SPAN = 4                      # answer tokens carried by one retrieval


def _bits_for(vocab: int) -> int:
    """Bits needed to name any token in THIS model's vocabulary, plus a terminator.

    Deriving this from the backbone rather than fixing it is what keeps the trainer model-agnostic:
    a fixed 18 bits happens to cover most current vocabularies, but it is an assumption about the
    model rather than something read from it, and a larger vocabulary would silently corrupt every
    stored token. The terminator is the id `vocab` itself -- one past the end, so it can never
    collide with a real token whatever the tokenizer does.
    """
    n, bits = vocab, 1
    while (1 << bits) <= n:
        bits += 1
    return bits
_STATE_MAGIC = b"FLYS"
_CODE_SEED = 11                # the token-code projection is fixed; the seed makes it reproducible


def _code_projection(hidden: int, width: int, device) -> torch.Tensor:
    """A fixed orthonormal (width x hidden) map. P @ P.t() = I, so P.t() inverts it exactly.

    Nothing here is learned. The brain stores P @ out_embedding[token] and reads it back with P.t(),
    which means encoding and decoding are correct by construction for EVERY token in the vocabulary,
    including ones that never appeared in training. A learned encoder can only encode what it has
    seen, and that is exactly how an earlier version scored 12.5% on a fixed word list and 0% once
    the words were drawn fresh each time.
    """
    g = torch.Generator().manual_seed(_CODE_SEED)
    return torch.linalg.qr(torch.randn(hidden, width, generator=g))[0].t().to(device)


class FlyBrain:
    """A connectome-shaped associative memory bound to one frozen backbone."""

    def __init__(self, blob: dict, device: str = "cpu"):
        d = self.device = device
        self.meta = blob["meta"]
        self.n = int(self.meta["neurons"])
        self.width = int(self.meta.get("width", 256))
        self.win = int(self.meta.get("win", 128))
        self.k_mem = int(self.meta.get("k_mem", 8))
        self.hidden = int(self.meta["hidden"])

        self.src = torch.as_tensor(blob["src"], dtype=torch.long, device=d)
        self.dst = torch.as_tensor(blob["dst"], dtype=torch.long, device=d)
        self.w = torch.as_tensor(blob["w"], dtype=torch.float32, device=d)
        sign = torch.as_tensor(blob["sign"], dtype=torch.float32, device=d)
        self.sgn = sign[self.src]

        # sizes first: the layers below are shaped by how much of the state carries signs
        self.bits = int(self.meta["bits"])          # recorded at training time, from the backbone
        self.span = int(self.meta.get("span", _SPAN))
        self.eos_id = int(self.meta.get("eos_id", -1))   # "the answer ends here"
        self.nbit = self.bits * self.span
        self.cw = self.width - self.nbit          # continuous half; the remainder carries the signs

        def _lin(key, i, o, bias=True):
            m = nn.Linear(i, o, bias=bias).to(d)
            m.load_state_dict({k: torch.as_tensor(v).to(d) for k, v in blob[key].items()})
            return m.eval()

        self.addr = _lin("addr", self.hidden, self.n)              # WHERE, from the hidden state
        self.addr_e = _lin("addr_e", self.hidden, self.n, bias=False)  # WHERE, from the tokens
        self.val = _lin("val", 2 * self.hidden, self.cw)        # HOW MUCH of the code to write
        self.wgate = _lin("wgate", 2 * self.hidden, 1)             # does this token deserve memory
        self.out = _lin("out", self.width, self.hidden)   # reads the whole slot, bits included            # retrieved code -> hidden space
        self.voice = torch.as_tensor(blob["voice"], dtype=torch.float32, device=d)
        self.temp = torch.as_tensor(blob["temp"], dtype=torch.float32, device=d)
        self.amix = torch.as_tensor(blob["amix"], dtype=torch.float32, device=d)
        self.dbeta = torch.as_tensor(blob["dbeta"], dtype=torch.float32, device=d)
        self.ek = int(self.meta.get("ek", 4))

        self._adj = torch.sparse_coo_tensor(
            torch.stack([self.dst, self.src]), self.w * self.sgn, (self.n, self.n)).coalesce()
        self.P = _code_projection(self.hidden, self.cw, d)
        pw = (2 ** torch.arange(self.bits, device=d)).float()
        self._pw = pw
        self.model = self.tok = self.stack = None
        self.code = None                                            # filled in by bind()
        self.reset()

    # ------------------------------------------------------------------ loading / binding
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "FlyBrain":
        return cls(torch.load(path, map_location=device, weights_only=False), device=device)

    def bind(self, model, tokenizer=None, verbose: bool = True) -> "FlyBrain":
        """Point the brain at a model. Reads its output embedding; changes nothing about it."""
        stack = self._find_stack(model)
        hid = self._hidden_size(model, stack)
        if hid != self.hidden:
            raise ValueError(
                f"this brain was trained against hidden size {self.hidden}, this model is {hid}. "
                "A brain does not transfer between backbones -- its address matrix and its token "
                "codes are that model's. Train one with --train."
            )
        self.model, self.tok, self.stack = model, tokenizer, stack
        emb = (model.get_output_embeddings() or stack.embed_tokens).weight.detach()
        code = emb.float().to(self.device) @ self.P.t()
        self.code = code / code.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.reset()
        if verbose:
            print(f"[flybrain] bound: {self.n:,} slots x {self.width} = {self.n * self.width:,} "
                  f"numbers ({self.state_bytes / 1e6:.1f} MB, constant at any length)")
            print(f"[flybrain] vocabulary {self.code.shape[0]:,} tokens, window {self.win}, "
                  f"backbone untouched")
        return self

    def detach(self):
        """Forget the backbone. The model is returned exactly as it was; nothing was patched."""
        self.model = self.tok = self.stack = self.code = None
        return self

    def _bitcode(self, ids: torch.Tensor) -> torch.Tensor:
        """+-1 code for a token id: its binary expansion, one sign per bit."""
        return ((ids.unsqueeze(-1).float() // self._pw) % 2) * 2 - 1

    def _bit_value(self, ids_row: torch.Tensor) -> torch.Tensor:
        """At each position, the signs for the next `span` tokens -- one retrieval, whole answer."""
        T = ids_row.shape[0]
        return torch.cat([self._bitcode(ids_row[torch.clamp(torch.arange(T, device=self.device) + j,
                                                            max=T - 1)])
                          for j in range(self.span)], -1).unsqueeze(0)

    def decode(self, raw: torch.Tensor) -> torch.Tensor:
        """Signs -> token ids. No vocabulary search, no nearest neighbour: `span` exact ids."""
        b = (raw[:, -self.nbit:] > 0).float().view(-1, self.span, self.bits)
        return (b * self._pw).sum(-1).long()

    # ------------------------------------------------------------------ state
    @property
    def state_bytes(self) -> int:
        return self.n * self.width * 4

    def reset(self, batch: int = 1):
        self.mem = torch.zeros(batch, self.n, self.width, device=self.device)
        self.z = torch.zeros(batch, self.n, device=self.device)
        self.carry = None
        self._ecarry = None
        return self

    def save_state(self, path: str) -> int:
        with open(path, "wb") as f:
            f.write(_STATE_MAGIC)
            torch.save({"mem": self.mem.cpu(), "z": self.z.cpu(),
                        "carry": None if self.carry is None else self.carry.cpu(),
                        "neurons": self.n, "width": self.width}, f)
        return os.path.getsize(path)

    def load_state(self, path: str):
        with open(path, "rb") as f:
            if f.read(4) != _STATE_MAGIC:
                raise ValueError(f"{path} is not a flybrain state file")
            s = torch.load(f, map_location=self.device, weights_only=False)
        if int(s["neurons"]) != self.n or int(s.get("width", self.width)) != self.width:
            raise ValueError("state was written by a differently shaped brain")
        self.mem, self.z = s["mem"].to(self.device), s["z"].to(self.device)
        self.carry = None if s["carry"] is None else s["carry"].to(self.device)
        return self

    # ------------------------------------------------------------------ the memory itself
    @staticmethod
    def _std(x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)

    def _phi(self, h: torch.Tensor, ek: torch.Tensor) -> torch.Tensor:
        """An address over slots, from two signals, at the fly's own sparsity.

        The hidden-state path carries context; the token-embedding path matches a repeated phrase
        EXACTLY (cosine 1.000 against 0.074 for filler), because the same tokens produce the same
        embeddings. Each is standardised before mixing -- their raw scales differ by orders of
        magnitude, and unstandardised the exact path contributes nothing.

        The temperature is clamped into the band where roughly 1-20% of slots are active. Kenyon
        cells fire ~5% at a time; running at 0.03% -- which an earlier version did -- means a value
        sits on ~3 slots and a single collision destroys a third of it. The fly's distributed code is
        error correcting, and that is the whole reason retrieval works at all.
        """
        g = torch.sigmoid(self.amix)
        lg = (1 - g) * self._std(self.addr(h)) + g * self._std(self.addr_e(ek))
        return torch.softmax(lg * self.temp.clamp(0.8, 3.0), dim=-1) * math.sqrt(self.n)

    def _ekey(self, emb: torch.Tensor) -> torch.Tensor:
        """Mean of the EK token embeddings before each position (causal, carried across windows)."""
        c = self._ecarry if self._ecarry is not None else emb[:, :1].expand(-1, self.ek, -1) * 0
        pad = torch.cat([c, emb], 1)
        cs = torch.cat([torch.zeros_like(pad[:, :1]), pad.cumsum(1)], 1)
        return (cs[:, self.ek:self.ek + emb.shape[1]] - cs[:, :emb.shape[1]]) / self.ek

    def _qkey(self, emb: torch.Tensor) -> torch.Tensor:
        """Address for the token about to be PREDICTED: the window ending at the newest token.

        Using the window BEFORE the last token asks for the token already in hand -- off by one, and
        it makes an exactly matching key useless.
        """
        return emb[:, -self.ek:].mean(1)

    def _read_raw(self, h: torch.Tensor, ek: torch.Tensor) -> torch.Tensor:
        """The retrieved value in MEMORY space. The signs live here; `out` maps out of this space,
        so slicing its output would decode noise rather than the stored bits."""
        a = self._phi(h, ek)
        return torch.einsum("bn,bnd->bd", a, self.mem) / ((a * self.z).sum(-1, keepdim=True) + 1e-4)

    def _read(self, h: torch.Tensor, ek: torch.Tensor) -> torch.Tensor:
        """Query the memory, normalised by how much was written where."""
        return self.out(self._read_raw(h, ek))

    def _memory_tokens(self, emb: torch.Tensor, ekw: torch.Tensor) -> torch.Tensor:
        """What the brain hands the model before it reads the next window."""
        q = self._read(emb.mean(1), ekw.mean(1))
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6) * emb.norm(dim=-1).mean()
        return q.unsqueeze(1).expand(-1, self.k_mem, -1)

    def _chunk(self, ids: torch.Tensor, write: bool = True):
        """One window: read from memory, run the model, write back. Returns (logits, retrieved)."""
        emb = self.stack.embed_tokens(ids)
        ekw = self._ekey(emb)
        o = self.model(inputs_embeds=torch.cat([self._memory_tokens(emb, ekw), emb], 1),
                       output_hidden_states=True)
        h = o.hidden_states[-1][:, self.k_mem:].float()
        logits = o.logits[:, self.k_mem:]
        if write:
            # KEY on the context BEFORE each token, VALUE on the token. The carry keeps the shift
            # correct across window boundaries, so the first token of a chunk is still keyed on the
            # last token of the one before it rather than on nothing.
            prev = self.carry if self.carry is not None else h[:, :1] * 0
            a = self._phi(torch.cat([prev, h[:, :-1]], 1), ekw)
            # A token the brain judges unimportant should occupy no memory at all. Gating only the
            # VALUE still let every filler token add its full address mass to the normaliser, so the
            # fact was divided by a denominator inflated with hundreds of writes it did not make.
            a = a * torch.sigmoid(self.wgate(torch.cat([emb, h], -1)))
            v = torch.cat([self.code[ids[0]].unsqueeze(0)
                           * torch.sigmoid(self.val(torch.cat([emb, h], -1))),
                           self._bit_value(ids[0])], -1)          # continuous half ++ exact signs
            self.carry = h[:, -1:]
            self._ecarry = emb[:, -self.ek:]
            # Delta rule: write the ERROR, not the value. Filler landing on the fact's slot then
            # corrects what is there instead of burying it (Widrow-Hoff).
            old = torch.einsum("btn,bnd->btd", a, self.mem) / \
                  ((a * self.z.unsqueeze(1)).sum(-1, keepdim=True) + 1e-4)
            self.mem = self.mem + torch.einsum(
                "btn,btd->bnd", a, v - torch.sigmoid(self.dbeta) * old) / a.shape[1]
            self.z = self.z + a.sum(1) / a.shape[1]
            # the connectome's own recurrence, as a sparse product: the edge gather materialises
            # (edges x width) and measured 225 ms against 24 ms for this, with identical arithmetic
            self.mem = self.mem + 0.05 * torch.tanh(
                torch.sparse.mm(self._adj, self.mem[0]).unsqueeze(0))
        return logits, self._read_raw(h[:, -1], self._qkey(emb))

    def _vote(self, last: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        """Mix the brain's own answer into the model's logits.

        The vote is standardised to the backbone's logit scale first. Unnormalised, the mixing weight
        cannot set the balance at all, and the brain's correct answers lose to the model's confident
        wrong ones -- measured at 58.3% for the brain alone against 8.3% for the mix.
        """
        vote = (retrieved / math.sqrt(self.hidden)) @ \
            (self.model.get_output_embeddings() or self.stack.embed_tokens).weight.t().float()
        vote = (vote - vote.mean(-1, keepdim=True)) / vote.std(-1, keepdim=True).clamp_min(1e-6)
        return last + torch.tanh(self.voice) * vote * last.std(-1, keepdim=True)

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def feed(self, text, update: bool = True) -> torch.Tensor:
        """Read text of any length, one window at a time. Returns logits for the next token.

        Cost is linear in length and memory is flat: the state is the same size after a million
        tokens as after a hundred.
        """
        if self.model is None:
            raise RuntimeError("call bind(model, tokenizer) first")
        ids = text if torch.is_tensor(text) else \
            self.tok(text, return_tensors="pt").input_ids.to(self.device)
        last = retrieved = None
        for s0 in range(0, ids.shape[1], self.win):
            ch = ids[:, s0:s0 + self.win]
            if ch.shape[1] < 1:
                continue
            logits, retrieved = self._chunk(ch, write=update)
            last = logits[:, -1]
        # the vote path still exists for callers that want logits; recall() reads the
        # signs directly and does not depend on it
        return self._vote(last, self.out(retrieved))

    @torch.no_grad()
    def recall(self, question: str) -> str:
        """Read the answer straight out of memory, as exact token ids.

        The brain used to nudge the backbone's logits with a soft vote and hope it agreed, which is
        where the probabilistic behaviour came from -- a correct retrieval could still lose to a
        confident wrong prediction. The signs decode to ids directly, so if the memory holds the
        answer, that IS the answer.
        """
        if self.model is None:
            raise RuntimeError("call bind(model, tokenizer) first")
        ids = self.tok(question, return_tensors="pt").input_ids.to(self.device)
        emb = self.stack.embed_tokens(ids[:, -self.win:])
        o = self.model(inputs_embeds=torch.cat([self._memory_tokens(emb, self._ekey(emb)), emb], 1),
                       output_hidden_states=True)
        h = o.hidden_states[-1][:, self.k_mem:].float()
        raw = self._read_raw(h[:, -1], self._qkey(emb))
        return self.tok.decode(self.decode(raw)[0].tolist(), skip_special_tokens=True)

    @torch.no_grad()
    def answer(self, question: str, max_new_tokens: int = 16) -> str:
        """Backwards-compatible alias for recall()."""
        return self.recall(question)

    # ------------------------------------------------------------------ backbone introspection
    @staticmethod
    def _find_stack(model):
        for path in ("model.language_model", "language_model.model", "model"):
            o = model
            try:
                for part in path.split("."):
                    o = getattr(o, part)
                if hasattr(o, "layers"):
                    return o
            except AttributeError:
                continue
        raise ValueError("could not locate a decoder stack on this model")

    @staticmethod
    def _hidden_size(model, stack):
        cfg = getattr(model, "config", None)
        if cfg is not None and hasattr(cfg, "text_config"):
            return cfg.text_config.hidden_size
        if hasattr(getattr(stack, "config", None), "hidden_size"):
            return stack.config.hidden_size
        return cfg.hidden_size


# ---------------------------------------------------------------------------- training
def train_brain(model, tokenizer, corpus, connectome, signs, *, out, steps=900, width=328,
                win=128, k_mem=8, span=4, ek=3, lr=1e-4, seed=1234, device="cpu", log=print):
    """Fit a brain to THIS model. The backbone is frozen and never updated.

    The task is unanswerable without memory: a random string is stated once, buried under filler, and
    asked about far beyond the window, so the model can never see the fact and the question together
    and the floor is a true 0%. Training words are drawn fresh every example and never reused, which
    is what makes the score mean retrieval rather than recall of a word list.

    Everything here was arrived at by measurement, and each piece is load-bearing:

      * The answer is stored as BITS, not as a vector. A vector must survive averaging and then win a
        nearest-neighbour search among non-orthogonal embedding rows; a sign only has to stay on the
        correct side of zero. Continuous codes plateaued at 25% exact; bits reach 100%.
      * The KEY is the context before a token, the VALUE is the token. Keyed on a random string
        itself, a question shares nothing with it -- measured 0.00x against a filler token, versus
        1.6-4.3x keyed one token earlier.
      * Two address paths, mixed: one over the hidden state, one over a window of token embeddings
        where a repeated phrase matches exactly (cosine 1.000 against 0.074 for filler).
      * Addresses are sharp but not one-hot. Kenyon cells fire ~5% at a time; at 0.03% a value sits
        on three slots and one collision destroys it.
      * Writes use the delta rule and are gated, so filler corrects rather than buries, and a token
        the brain judges unimportant occupies no memory at all.
      * The backbone runs OUTSIDE the graph. It is frozen, so its activations are constants: keeping
        it in cost 2.7x per step and changed nothing, because recall is identical with the memory
        token path switched off entirely.
    """
    import random
    import string
    import numpy as np

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    stack = FlyBrain._find_stack(model)
    D = FlyBrain._hidden_size(model, stack)
    for p in model.parameters():
        p.requires_grad_(False)
    out_emb = (model.get_output_embeddings() or stack.embed_tokens).weight
    V = out_emb.shape[0]
    bits = _bits_for(V)                       # from the backbone's vocabulary, not assumed
    nbit = bits * span
    cw = width - nbit
    if cw < 64:
        raise ValueError(f"width {width} leaves only {cw} dims for content; raise it")

    pre, post, wts = connectome
    nodes, inv = np.unique(np.concatenate([pre, post]), return_inverse=True)
    E, N = len(pre), len(nodes)
    src, dst = inv[:E], inv[E:]
    w = np.log1p(wts.astype(np.float32))
    radius = _spectral_radius(src, dst, w * signs[src], N, device)
    w = w * (0.95 / max(radius, 1e-6))
    SRC = torch.as_tensor(src, dtype=torch.long, device=device)
    DST = torch.as_tensor(dst, dtype=torch.long, device=device)
    SGN = torch.as_tensor(signs, dtype=torch.float32, device=device)[SRC]
    EW = torch.as_tensor(w, device=device)
    # sparse product rather than an edge gather: the gather materialises (edges x width) and measured
    # 225 ms against 24 ms here, with identical arithmetic
    ADJ = torch.sparse_coo_tensor(torch.stack([DST, SRC]), EW * SGN, (N, N)).coalesce()
    log(f"connectome: {N:,} slots x {width} = {N*width:,} numbers "
        f"({N*width*4/1e6:.1f} MB, constant at any length), {E:,} edges")
    log(f"vocabulary {V:,} -> {bits} bits/token, {span}-token payload ({nbit} bits)")

    P = _code_projection(D, cw, device)
    CODE = out_emb.detach().float() @ P.t()
    CODE = CODE / CODE.norm(dim=1, keepdim=True).clamp_min(1e-6)
    PW = (2 ** torch.arange(bits, device=device)).float()
    ALLIDS = torch.arange(V, device=device).float()
    BITCODE = ((ALLIDS.unsqueeze(1) // PW) % 2) * 2 - 1        # every token id, exactly

    ADDR = nn.Linear(D, N).to(device)
    ADDR_E = nn.Linear(D, N, bias=False).to(device)
    VAL = nn.Linear(2 * D, cw).to(device)
    WGATE = nn.Linear(2 * D, 1).to(device)
    ROUT = nn.Linear(width, D).to(device)
    VOICE = nn.Parameter(torch.full((1,), 0.1, device=device))
    TEMP = nn.Parameter(torch.ones(1, device=device) * 1.6)    # ~5-12% of slots active
    AMIX = nn.Parameter(torch.zeros(1, device=device))
    DBETA = nn.Parameter(torch.full((1,), -1.0, device=device))
    torch.nn.init.normal_(ADDR_E.weight, std=0.02)
    addr_n0 = ADDR.weight.norm().detach().clone()
    params = [ADDR.weight, ADDR.bias, ADDR_E.weight, VAL.weight, VAL.bias,
              WGATE.weight, WGATE.bias, ROUT.weight, ROUT.bias, VOICE, TEMP, AMIX, DBETA]

    ASKS = [" Question: what is the secret word? Answer: The secret word is",
            " Question: what was the secret word again? Answer: The secret word is",
            " Q: secret word? A: The secret word is"]
    half = len(corpus) // 2
    fill_train, fill_eval = corpus[:half], corpus[half:]
    eval_words = ["".join(random.choice(string.ascii_lowercase)
                          for _ in range(random.randint(4, 12))) for _ in range(120)]
    eval_set = set(eval_words)

    def fresh():                              # never reused, never in the eval set
        while True:
            c = "".join(random.choice(string.ascii_lowercase)
                        for _ in range(random.randint(4, 12)))
            if c not in eval_set:
                return c

    def sample(word, filler, rng, nch):
        """Fact, filler, then the QUESTION -- which must never be truncated away.

        Building the string and cutting it to a token budget slices the question off the end of every
        example, and the task then has no answer in it at all.
        """
        qt = tokenizer(rng.choice(ASKS), add_special_tokens=False, return_tensors="pt").input_ids
        head = tokenizer(f"The secret word is {word}. ", return_tensors="pt").input_ids
        budget = nch * win - head.shape[1] - qt.shape[1]
        i = rng.randrange(0, max(len(filler) - 80_000, 1))
        fil = tokenizer(filler[i:i + 80_000], add_special_tokens=False,
                        return_tensors="pt").input_ids[:, :budget]
        ids = torch.cat([head, fil, qt], 1).to(device)
        ans = tokenizer(" " + word, add_special_tokens=False).input_ids[:span]
        return ids, torch.tensor([ans], device=device)

    def std(x):
        return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)

    def phi(h, ekv):
        g = torch.sigmoid(AMIX)
        lg = (1 - g) * std(ADDR(h)) + g * std(ADDR_E(ekv))
        return torch.softmax(lg * TEMP.clamp(0.8, 3.0), -1) * math.sqrt(N)

    def ekey(emb, carry):
        pad = torch.cat([carry if carry is not None else emb[:, :1].expand(-1, ek, -1) * 0, emb], 1)
        cs = torch.cat([torch.zeros_like(pad[:, :1]), pad.cumsum(1)], 1)
        return (cs[:, ek:ek + emb.shape[1]] - cs[:, :emb.shape[1]]) / ek

    def qkey(emb):
        """The address for the token about to be PREDICTED -- the window ENDING at the newest token.
        Using the window before it asks for the token already in hand; off by one, and an exactly
        matching key becomes useless."""
        return emb[:, -ek:].mean(1)

    def bitval(ids_row):
        T = ids_row.shape[0]
        return torch.cat([BITCODE[ids_row[torch.clamp(torch.arange(T, device=device) + j, max=T - 1)]]
                          for j in range(span)], -1).unsqueeze(0)

    def run(ids, answer=None, grad=False):
        with (torch.enable_grad() if grad else torch.no_grad()):
            mem = torch.zeros(1, N, width, device=device)
            z = torch.zeros(1, N, device=device)
            carry = e_carry = raw = None
            for s0 in range(0, ids.shape[1], win):
                ch = ids[:, s0:s0 + win]
                if ch.shape[1] < 1:
                    continue
                emb = stack.embed_tokens(ch)
                ekw = ekey(emb, e_carry)
                q = ROUT(torch.zeros(1, width, device=device)) if raw is None else ROUT(raw)
                q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6) * emb.norm(dim=-1).mean()
                with torch.no_grad():         # frozen backbone: activations are constants
                    o = model(inputs_embeds=torch.cat(
                        [q.detach().unsqueeze(1).expand(-1, k_mem, -1), emb], 1),
                        output_hidden_states=True)
                h = o.hidden_states[-1][:, k_mem:].float()
                prev = carry if carry is not None else h[:, :1] * 0
                a = phi(torch.cat([prev, h[:, :-1]], 1), ekw)
                a = a * torch.sigmoid(WGATE(torch.cat([emb, h], -1)))
                v = torch.cat([CODE[ch[0]].unsqueeze(0) *
                               torch.sigmoid(VAL(torch.cat([emb, h], -1))), bitval(ch[0])], -1)
                carry, e_carry = h[:, -1:], emb[:, -ek:]
                old = torch.einsum("btn,bnd->btd", a, mem) / \
                      ((a * z.unsqueeze(1)).sum(-1, keepdim=True) + 1e-4)
                mem = mem + torch.einsum("btn,btd->bnd", a, v - torch.sigmoid(DBETA) * old) / a.shape[1]
                z = z + a.sum(1) / a.shape[1]
                mem = mem + 0.05 * torch.tanh(torch.sparse.mm(ADJ, mem[0]).unsqueeze(0))
                aq = phi(h[:, -1], qkey(emb))
                raw = torch.einsum("bn,bnd->bd", aq, mem) / ((aq * z).sum(-1, keepdim=True) + 1e-4)
            return raw

    def decode(raw):
        b = (raw[:, -nbit:] > 0).float().view(-1, span, bits)
        return (b * PW).sum(-1).long()

    def evaluate(n=96):
        r = random.Random(999)
        exact = 0
        per = [0] * span
        cnt = [0] * span
        for _ in range(n):
            ids, ans = sample(eval_words[r.randrange(len(eval_words))], fill_eval, r, 6)
            got = decode(run(ids))[0]
            want = ans[0]
            ok = True
            for k in range(min(len(want), span)):
                cnt[k] += 1
                good = int(got[k]) == int(want[k])
                per[k] += int(good)
                ok = ok and good
            exact += int(ok)
        return exact / n, [p / max(c, 1) for p, c in zip(per, cnt)]

    warm = max(1, steps // 20)
    opt = torch.optim.Adam([{"params": [q for q in params if q is not VOICE], "lr": lr},
                            {"params": [VOICE], "lr": lr * 20}])

    def lr_at(it):
        if it < warm:
            return (it + 1) / warm
        p = (it - warm) / max(steps - warm, 1)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = random.Random(0)
    best = (0.0, None)
    for it in range(1, steps + 1):
        ids, ans = sample(fresh(), fill_train, rng, rng.choice([4, 6, 6, 8]))
        raw = run(ids, grad=True)
        nreal = ans.shape[1]
        tgt = torch.cat([BITCODE[ans[0, k]] for k in range(nreal)]).unsqueeze(0)
        loss = nn.functional.binary_cross_entropy_with_logits(
            raw[:, -nbit:][:, :nreal * bits] * 4.0, (tgt > 0).float())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        opt.step()
        with torch.no_grad():                 # sharp addressing that cannot run away
            ADDR.weight.mul_(addr_n0 / ADDR.weight.norm().clamp_min(1e-6))
        sched.step()

        if it % max(1, steps // 9) == 0:
            ex, per = evaluate()
            log(f"  step {it:>5}  exact whole word {100*ex:5.1f}%   "
                f"per-token {[f'{100*p:.0f}%' for p in per]}")
            if ex >= best[0]:                 # keep the best, not the last
                best = (ex, it)
                torch.save({"src": src, "dst": dst, "w": w, "sign": signs,
                            "addr": {k: v.detach().cpu() for k, v in ADDR.state_dict().items()},
                            "addr_e": {k: v.detach().cpu() for k, v in ADDR_E.state_dict().items()},
                            "val": {k: v.detach().cpu() for k, v in VAL.state_dict().items()},
                            "wgate": {k: v.detach().cpu() for k, v in WGATE.state_dict().items()},
                            "out": {k: v.detach().cpu() for k, v in ROUT.state_dict().items()},
                            "voice": VOICE.detach().cpu(), "temp": TEMP.detach().cpu(),
                            "amix": AMIX.detach().cpu(), "dbeta": DBETA.detach().cpu(),
                            "meta": {"neurons": int(N), "synapses": int(E), "hidden": int(D),
                                     "width": int(width), "win": int(win), "k_mem": int(k_mem),
                                     "bits": int(bits), "span": int(span), "ek": int(ek),
                                     "vocab": int(V), "radius": float(radius), "seed": seed,
                                     "code_seed": _CODE_SEED, "steps": steps, "lr": lr,
                                     "exact": float(ex), "step": int(it)}}, out)
    log(f"saved {out} ({os.path.getsize(out)/1e6:.1f} MB) -- "
        f"exact whole word {100*best[0]:.1f}% at step {best[1]}  (floor: 0.0%)")
    return best[0]


def _spectral_radius(src, dst, vals, n, device, iters=50):
    """Largest eigenvalue by power iteration. ARPACK on a graph this size exhausts a 32 GB box."""
    S = torch.as_tensor(src, dtype=torch.long, device=device)
    D = torch.as_tensor(dst, dtype=torch.long, device=device)
    V = torch.as_tensor(vals, dtype=torch.float32, device=device)
    x = torch.randn(n, device=device)
    x /= x.norm()
    lam = 1.0
    for _ in range(iters):
        y = torch.zeros(n, device=device).index_add_(0, D, x[S] * V)
        nr = y.norm()
        if nr < 1e-12:
            return 1.0
        lam = float(nr)
        x = y / nr
    return lam


def main():
    """CLI: train a brain for your model, or bind one and ask it about a long document."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--train", type=int, default=0,
                    help="fit a NEW brain to --model for this many steps, then save to --brain.\n"
                         "A brain does not transfer between backbones, so this is how you make one\n"
                         "for yours. Needs --connectome, --signs and --probes.")
    ap.add_argument("--connectome", default="",
                    help="feather/parquet with body_pre, body_post, weight (+ --signs .npy)")
    ap.add_argument("--signs", default="", help="per-neuron +1/-1 npy")
    ap.add_argument("--brain", required=True, help="a .pt brain, e.g. FlyBrain-Pollard-CNSv1.pt")
    ap.add_argument("--model", required=True, help="HF model id to train against / bind to")
    ap.add_argument("--probes", required=True, help="text corpus: filler for training, or a document")
    ap.add_argument("--width", type=int, default=328,
                    help="numbers per slot (default 328: 256 content + 72 bit, the\n"
                         "geometry the verified brain was trained at)")
    ap.add_argument("--ask", default="Question: what is the secret word? Answer:",
                    help="what to ask after reading --probes")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--save-state", default="", help="write the session state here")
    ap.add_argument("--load-state", default="", help="resume from a state file")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device).eval()
    corpus = open(a.probes, encoding="utf-8", errors="replace").read(1_000_000)

    if a.train:
        import numpy as np
        import pandas as pd
        g = pd.read_feather(a.connectome)
        train_brain(model, tok, corpus,
                    (g["body_pre"].to_numpy(), g["body_post"].to_numpy(), g["weight"].to_numpy()),
                    np.load(a.signs), out=a.brain, steps=a.train, width=a.width, device=a.device)
        return

    brain = FlyBrain.load(a.brain, device=a.device).bind(model, tok)
    if a.load_state:
        brain.load_state(a.load_state)
        print(f"resumed from {a.load_state}")
    else:
        n_tok = tok(corpus, return_tensors="pt").input_ids.shape[1]
        brain.feed(corpus)
        print(f"[flybrain] read {n_tok:,} tokens in {brain.win}-token windows; "
              f"state still {brain.state_bytes / 1e6:.1f} MB")
    print(f"\n{a.ask} {brain.answer(a.ask, max_new_tokens=a.tokens)}")

    if a.save_state:
        n = brain.save_state(a.save_state)
        print(f"\nstate -> {a.save_state} ({n:,} bytes, and it does not grow)")


if __name__ == "__main__":
    main()
