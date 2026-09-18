"""pollard-flybrain -- a fruit-fly connectome as a language model's memory.

A transformer's memory is its KV cache: it grows with every token, and when the context window fills,
the oldest tokens are gone. A fly has no context window. This attaches a real connectome (MaleCNS
v1.0, FlyEM/Janelia) to a frozen model as an associative memory whose size never changes.

    from pollard_flybrain import FlyBrain

    brain = FlyBrain.load("FlyBrain-Pollard-CNSv1.pt", device="cuda")
    brain.bind(model, tok)                   # any HF causal LM; the backbone is never modified
    brain.feed(long_document)                # read a document of ANY length, 128 tokens at a time
    print(brain.recall("The secret word is"))   # CONTINUE the text -- see below

A query is a CONTINUATION, not a question. The brain files each token under the words immediately
before it, so a document saying "The secret word is swordfish" stores that token under the context
"The secret word is" -- and that is what retrieves it. Asking "Question: what is the secret word?
Answer:" addresses a context the document never contained, so it returns confident noise from an
otherwise perfect memory. If recall looks like garbage, check this first.
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
from pollard_backbone import load_backbone as _lb

try:
    import torch
    import torch.nn as nn
except ImportError as _e:      # the connectome lane is an optional extra, not a core dependency
    raise SystemExit(
        "pollard-flybrain needs PyTorch, which Pollard does not install by default.\n"
        "  pip install 'pollard-weights[flybrain]'      (torch, transformers, pandas, scipy)\n"
        f"({_e})"
    ) from None

__all__ = ["FlyBrain", "train_brain", "load_backbone"]

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


def _progress(*args, **kw):
    """print(), but flushed.

    Python block-buffers stdout when it is not a terminal, so `pollard-flybrain --train ... > log`
    writes nothing until the buffer fills or the process exits. Training runs for minutes and the
    per-step lines are the only way to see it converging; an empty log reads as a hung job, and the
    natural response is to kill a run that was working.
    """
    kw.setdefault("flush", True)
    print(*args, **kw)


def load_backbone(model_id: str, dtype=None, device: str = "cpu", **kw):
    """Re-exported from pollard_backbone -- Pollard's model-side loader. The brain uses the
    model tooling, never the other way round."""
    return _lb(model_id, dtype, device, **kw)


def probe_basis(model, tok, corpus: str, n_probe: int, device, plen: int = 160):
    """One direction per probe: this model's mean hidden state while reading that probe text.

    This is what lets ONE brain attach to models of different widths. The basis is built from
    responses to the SAME probe texts on every backbone, so direction i means the same thing
    everywhere -- measured at +0.53 correspondence across model families, against -0.02 for a random
    per-model projection. Nothing here is trained; it is derived at attach time from the backbone
    itself, which is why a canonical brain needs no retraining to move.
    """
    rows = []
    with torch.no_grad():
        for i in range(n_probe):
            txt = corpus[i * plen:(i + 1) * plen]
            if not txt.strip():
                continue
            ids = tok(txt, return_tensors="pt").input_ids[:, :48].to(device)
            if ids.shape[1] < 2:
                continue
            h = model(input_ids=ids, output_hidden_states=True).hidden_states[-1][0].float()
            rows.append(h.mean(0))
    if len(rows) < n_probe:
        raise SystemExit(f"probe corpus too short: got {len(rows)} of {n_probe} probes "
                         f"({n_probe * plen} characters needed)")
    A = torch.stack(rows)
    return A / A.norm(dim=1, keepdim=True).clamp_min(1e-6)          # (C, D)


def bridge(A: torch.Tensor):
    """(into canonical, back out). The return path MUST be a pseudo-inverse.

    The probe basis is not orthonormal, so A.t() is not its inverse. Using the transpose measured a
    round-trip gain of 7.5 and the brain could not learn through it at all -- 400 steps stuck between
    -5% and -60% of the gap. pinv costs one decomposition at attach time and makes the round trip
    an identity.
    """
    return A, torch.linalg.pinv(A)                                   # (C, D), (D, C)


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
        self.codec = str(self.meta.get("codec", "tokens"))   # "tokens" | "bytes"
        self.canon = int(self.meta.get("canon", 0))   # 0 = locked to this hidden size
        self.dim = self.canon or self.hidden          # the width the brain lives in
        self.eos_id = int(self.meta.get("eos_id", -1))   # "the answer ends here"
        self.nbit = self.bits * self.span
        self.cw = self.width - self.nbit          # continuous half; the remainder carries the signs

        def _lin(key, i, o, bias=True):
            m = nn.Linear(i, o, bias=bias).to(d)
            m.load_state_dict({k: torch.as_tensor(v).to(d) for k, v in blob[key].items()})
            return m.eval()

        self.addr = _lin("addr", self.dim, self.n)              # WHERE, from the hidden state
        self.addr_e = _lin("addr_e", self.dim, self.n, bias=False)  # WHERE, from the tokens
        self.val = _lin("val", 2 * self.dim, self.cw)        # HOW MUCH of the code to write
        self.wgate = _lin("wgate", 2 * self.dim, 1)             # does this token deserve memory
        self.out = _lin("out", self.width, self.dim)   # reads the whole slot, bits included            # retrieved code -> hidden space
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

    def bind(self, model, tokenizer=None, verbose: bool = True,
             probe_text: Optional[str] = None) -> "FlyBrain":
        """Point the brain at a model. Reads its output embedding; changes nothing about it."""
        stack = self._find_stack(model)
        hid = self._hidden_size(model, stack)
        if not self.canon and hid != self.hidden:
            raise ValueError(
                f"this brain was trained against hidden size {self.hidden}, this model is {hid}. "
                "A brain trained without --canon is locked to that backbone: its address matrix is "
                "that model's width. Train with --canon to make one that attaches anywhere, or "
                "carry this one over with --continue-from."
            )
        self.model, self.tok, self.stack = model, tokenizer, stack
        emb = (model.get_output_embeddings() or stack.embed_tokens).weight.detach()
        code = emb.float().to(self.device) @ self.P.t()
        self.code = code / code.norm(dim=1, keepdim=True).clamp_min(1e-6)
        if self.codec == "bytes":
            self._btbl, self._blen = self._byte_table()
        if self.canon:
            if probe_text is None:
                raise ValueError("a canonical brain needs probe_text= at bind(): the bridge to this "
                                 "backbone is derived from its own responses, not shipped with the "
                                 f"brain. Any text works; it needs about {self.canon * 160:,} chars.")
            A = probe_basis(model, tokenizer, probe_text, self.canon, self.device)
            self._pin, self._pout = bridge(A)
            if verbose:
                x = torch.randn(64, A.shape[1], device=A.device)
                g = ((x @ self._pin.t()) @ self._pout.t()).norm(dim=1).mean() / x.norm(dim=1).mean()
                print(f"[flybrain] canonical {self.canon}d bridge to hidden {hid}, "
                      f"round-trip gain {float(g):.3f} (1.000 is exact)")
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

    def _bitcode(self, vals: torch.Tensor) -> torch.Tensor:
        """+-1 code for an integer: its binary expansion, one sign per bit."""
        return ((vals.unsqueeze(-1).float() // self._pw) % 2) * 2 - 1

    def _byte_table(self) -> torch.Tensor:
        """token id -> its UTF-8 bytes, padded. Built once, from the bound tokenizer.

        This is what makes a byte-payload brain tokenizer-independent: the MEMORY stores text, and
        the tokenizer is only how this particular backbone happens to spell it. Two models with
        different vocabularies store the same bytes for the same words, so a brain carried between
        them is still readable -- which a vocabulary index never is.
        """
        return _byte_table_for(self.tok, self.code.shape[0], self.span, self.device,
                               log=lambda *a, **k: None)

    def _bit_value(self, ids_row: torch.Tensor) -> torch.Tensor:
        """At each position, the signs for the next `span` units -- one retrieval, whole answer.

        A unit is a token id (codec "tokens") or a text byte (codec "bytes").
        """
        T = ids_row.shape[0]
        if self.codec == "bytes":
            flat, off = [], []
            for t in range(T):
                off.append(len(flat))
                flat.extend(self._btbl[int(ids_row[t])][:int(self._blen[int(ids_row[t])])].tolist())
            flat = flat + [0] * self.span
            vals = torch.tensor([flat[off[t]:off[t] + self.span] for t in range(T)],
                                device=self.device, dtype=torch.long)
            return self._bitcode(vals).flatten(1).unsqueeze(0)
        return torch.cat([self._bitcode(ids_row[torch.clamp(torch.arange(T, device=self.device) + j,
                                                            max=T - 1)])
                          for j in range(self.span)], -1).unsqueeze(0)

    def decode(self, raw: torch.Tensor) -> torch.Tensor:
        """Signs -> exact integers. No vocabulary search, no nearest neighbour: `span` of them.

        Token ids under codec "tokens"; UTF-8 byte values under codec "bytes" (use decode_text()).
        """
        b = (raw[:, -self.nbit:] > 0).float().view(-1, self.span, self.bits)
        return (b * self._pw).sum(-1).long()

    def decode_text(self, raw: torch.Tensor) -> str:
        """Signs -> the stored text, whichever codec this brain uses."""
        v = self.decode(raw)[0].tolist()
        if self.codec == "bytes":
            return bytes(x for x in v if 0 < x < 256).decode("utf-8", "ignore")
        return self.tok.decode(v, skip_special_tokens=True)

    # ------------------------------------------------------------------ state
    @property
    def state_bytes(self) -> int:
        return self.n * self.width * 4

    def reset(self, batch: int = 1):
        self.mem = torch.zeros(batch, self.n, self.width, device=self.device)
        self.z = torch.zeros(batch, self.n, device=self.device)
        self.carry = None
        self._ecarry = None
        self._last_ids = None
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

    def _to_canon(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone space -> the width the brain lives in. Identity for a locked brain."""
        return x if not self.canon else x @ self._pin.t()

    def _from_canon(self, x: torch.Tensor) -> torch.Tensor:
        """Back out to backbone space, through the pseudo-inverse."""
        return x if not self.canon else x @ self._pout.t()

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
        q = self._from_canon(self._read(self._to_canon(emb.float()).mean(1), ekw.mean(1)))
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6) * emb.norm(dim=-1).mean()
        return q.unsqueeze(1).expand(-1, self.k_mem, -1)

    def _chunk(self, ids: torch.Tensor, write: bool = True):
        """One window: read from memory, run the model, write back. Returns (logits, retrieved)."""
        emb = self.stack.embed_tokens(ids)
        self._last_ids = ids
        ekw = self._to_canon(self._ekey(emb).float())
        o = self.model(inputs_embeds=torch.cat([self._memory_tokens(emb, ekw), emb], 1),
                       output_hidden_states=True)
        h = self._to_canon(o.hidden_states[-1][:, self.k_mem:].float())
        embc = self._to_canon(emb.float())
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
            a = a * torch.sigmoid(self.wgate(torch.cat([embc, h], -1)))
            v = torch.cat([self.code[ids[0]].unsqueeze(0)
                           * torch.sigmoid(self.val(torch.cat([embc, h], -1))),
                           self._bit_value(ids[0])], -1)          # continuous half ++ exact signs
            self.carry = h[:, -1:]
            # The carry must be EXACTLY ek long. A final window shorter than ek leaves a short
            # carry, and the next _ekey() slices two different lengths and dies with a shape error
            # -- on a document whose length happens not to divide the window, which is most of them.
            car = emb[:, -self.ek:]
            if car.shape[1] < self.ek:
                car = torch.cat([car[:, :1].expand(-1, self.ek - car.shape[1], -1) * 0, car], 1)
            self._ecarry = car
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
        return logits, self._read_raw(h[:, -1], self._to_canon(self._qkey(emb).float()))

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
        return self._vote(last, self._from_canon(self.out(retrieved)))

    @torch.no_grad()
    def recall(self, question: str) -> str:
        """Read the answer straight out of memory, as exact token ids.

        The query is CONTINUED FROM WHAT WAS READ, not run on its own. A token is filed under the
        words immediately before it, so the address for the answer is built from the document's own
        wording -- and a question processed in a fresh pass carries none of that context. Scored
        standalone, the same brain that measures 100% in-stream measures roughly 46%: the memory is
        intact, the query simply arrives at the wrong address.

        So the tail of the last window fed is prepended to the question here, which is exactly the
        sequence training and evaluation both saw. Nothing is written; the memory is only read.
        """
        if self.model is None:
            raise RuntimeError("call bind(model, tokenizer) first")
        q = self.tok(question, return_tensors="pt").input_ids.to(self.device)
        if self._last_ids is not None:
            q = torch.cat([self._last_ids, q], 1)
        ctx = q[:, -self.win:]
        _logits, raw = self._chunk(ctx, write=False)
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
def _byte_table_for(tokenizer, vocab: int, span: int, device, log=print):
    """token id -> its UTF-8 bytes, for a whole vocabulary.

    Built with batch_decode. Calling decode() once per id is the obvious way and it is unusably slow:
    151,936 separate calls stall for minutes before a single line of training output appears, which
    reads as a hung job. Batched, it is seconds.
    """
    import numpy as np
    tbl = np.zeros((vocab, span), dtype=np.int64)
    ln = np.zeros(vocab, dtype=np.int64)
    CH = 8192
    for start in range(0, vocab, CH):
        ids = list(range(start, min(start + CH, vocab)))
        for k, txt in zip(ids, tokenizer.batch_decode([[i] for i in ids])):
            b = txt.encode("utf-8", "ignore")[:span]
            ln[k] = len(b)
            tbl[k, :len(b)] = list(b)
    log(f"byte codec: {vocab:,} token->bytes entries built")
    return (torch.as_tensor(tbl, device=device), torch.as_tensor(ln, device=device))


def _carry_over(ck: dict, parts: dict, log=print) -> int:
    """Load what still fits from an existing brain; keep a fresh init for what cannot.

    A brain is partly BACKBONE-SHAPED and partly not. The connectome, the slot count, the bit
    codebook and the write/decay behaviour are properties of the memory and travel anywhere. The
    address, value, gate and decoder matrices are sized by the backbone's hidden dimension, so moving
    to a model with a different width leaves those unusable.

    Refusing the whole transfer over that is too blunt: it throws away everything that DID carry.
    Loading it blindly is worse -- torch will happily accept a mismatched tensor in some paths and
    then produce confident nonsense. So each part is matched on shape, carried when it fits, and
    reported when it does not. What is re-initialised converges again in tens of steps because the
    memory it is learning to address is already organised.
    """
    kept = 0
    for name, mod in parts.items():
        if name not in ck:
            continue
        try:
            want = mod.state_dict()
            have = {k: torch.as_tensor(v) for k, v in ck[name].items()}
            if all(k in have and have[k].shape == v.shape for k, v in want.items()):
                mod.load_state_dict({k: have[k].to(v.device) for k, v in want.items()})
                kept += 1
            else:
                shapes = ", ".join(f"{k} {tuple(have[k].shape)}->{tuple(v.shape)}"
                                   for k, v in want.items() if k in have and have[k].shape != v.shape)
                log(f"  re-init {name}: {shapes}")
        except Exception as e:
            log(f"  re-init {name}: {e}")
    return kept


def train_brain(model, tokenizer, corpus, connectome, signs, *, out, steps=900, width=328,
                win=128, k_mem=8, span=4, ek=3, lr=1e-4, seed=1234, device="cpu",
                continue_from="", codec="tokens", canon=0, log=_progress):
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
    # Canonical space: train in a fixed width and bridge to whatever this backbone happens to be, so
    # the resulting brain attaches to ANY model instead of only this one. The bridge is derived from
    # the backbone's own responses to fixed probe texts -- nothing about it is trained -- and the
    # canonical directions correspond across families at +0.95 (measured, Qwen2.5-0.5B hidden 896
    # against SmolLM2-135M hidden 576; a random per-model projection scores ~0.00).
    #
    # 512 probes is the default because fidelity is measured on REAL hidden states, not random
    # vectors: 64 probes round-trips a hidden state at 0.879, 512 at 0.962, while random vectors
    # score 0.265 and 0.758. Hidden states occupy a low-dimensional subspace, which is the whole
    # reason a projection this aggressive costs so little.
    PIN = POUT = None
    DIM = D
    if canon:
        _A = probe_basis(model, tokenizer, corpus, canon, device)
        PIN, POUT = bridge(_A)
        DIM = canon
        with torch.no_grad():
            _ids = tokenizer(corpus[:4000], return_tensors="pt").input_ids[:, :256].to(device)
            _H = model(input_ids=_ids, output_hidden_states=True).hidden_states[-1][0].float()
            _g = ((_H @ PIN.t()) @ POUT.t()).norm(dim=1).mean() / _H.norm(dim=1).mean()
        log(f"canonical {canon}d: bridge from hidden {D}, round-trip {float(_g):.3f} on real states")
    for p in model.parameters():
        p.requires_grad_(False)
    out_emb = (model.get_output_embeddings() or stack.embed_tokens).weight
    V = out_emb.shape[0]
    if codec not in ("tokens", "bytes"):
        raise SystemExit(f"--codec must be tokens or bytes, not {codec!r}")
    # "tokens": the payload is this backbone's vocabulary indices -- smallest, but only that
    # tokenizer can read it back. "bytes": the payload is UTF-8 text bytes, so ANY tokenizer can,
    # and a six-letter word costs 6x8=48 bits against 4x18=72. Portability is the point; the size
    # is a bonus.
    bits = 8 if codec == "bytes" else _bits_for(V)   # from the backbone's vocabulary, not assumed
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

    ADDR = nn.Linear(DIM, N).to(device)
    ADDR_E = nn.Linear(DIM, N, bias=False).to(device)
    VAL = nn.Linear(2 * DIM, cw).to(device)
    WGATE = nn.Linear(2 * DIM, 1).to(device)
    ROUT = nn.Linear(width, DIM).to(device)
    VOICE = nn.Parameter(torch.full((1,), 0.1, device=device))
    TEMP = nn.Parameter(torch.ones(1, device=device) * 1.6)    # ~5-12% of slots active
    AMIX = nn.Parameter(torch.zeros(1, device=device))
    DBETA = nn.Parameter(torch.full((1,), -1.0, device=device))
    torch.nn.init.normal_(ADDR_E.weight, std=0.02)
    if continue_from:
        # Pick up where a previous run stopped -- on this backbone, or on a different one.
        ck = torch.load(continue_from, map_location=device, weights_only=False)
        prev = ck.get("meta", {})
        # A payload change is allowed. --continue-from carries trained WEIGHTS, and the written
        # memory is runtime state that lives in a .flystate file, not in the checkpoint -- so there
        # is nothing stored here for a new codebook to corrupt. Changing bits, span or codec resizes
        # the value and decoder layers, and _carry_over re-initialises exactly those while the
        # address path, gate and connectome carry across. People swap backbones and payloads
        # constantly; refusing the whole transfer over a resizable layer threw away everything that
        # did transfer. (A .flystate written by a differently shaped brain IS still refused, in
        # load_state, because that file holds real written memory.)
        pb, ps = int(prev.get("bits", bits)), int(prev.get("span", span))
        pc = str(prev.get("codec", "tokens"))
        if (pb, ps, pc) != (bits, span, codec):
            log(f"  payload change {pb}x{ps} {pc} -> {bits}x{span} {codec}: "
                "value and decoder layers re-initialise, addressing carries over")
        kept = _carry_over(ck, {"addr": ADDR, "addr_e": ADDR_E, "val": VAL,
                                "wgate": WGATE, "out": ROUT}, log=log)
        with torch.no_grad():
            for key, prm in (("voice", VOICE), ("temp", TEMP), ("amix", AMIX), ("dbeta", DBETA)):
                if key in ck:
                    prm.copy_(torch.as_tensor(ck[key]).to(device).reshape(prm.shape))
        same = int(prev.get("hidden", -1)) == D
        where = "same backbone" if same else f"NEW backbone: hidden {prev.get('hidden')} -> {D}"
        log(f"continuing from {os.path.basename(continue_from)} "
            f"(step {prev.get('step', '?')}, {kept}/5 layers carried, {where})")
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

    if codec == "bytes":
        # token id -> its UTF-8 bytes. Built once: the memory stores TEXT, and the tokenizer is only
        # how this backbone spells it, so the same words land as the same bytes on any model.
        BTBL, BLEN = _byte_table_for(tokenizer, V, span, device, log)

    def bitval(ids_row):
        T = ids_row.shape[0]
        if codec == "bytes":
            flat, off = [], []
            for t in range(T):
                i = int(ids_row[t]); off.append(len(flat))
                flat.extend(BTBL[i][:int(BLEN[i])].tolist())
            flat = flat + [0] * span
            vals = torch.tensor([flat[off[t]:off[t] + span] for t in range(T)],
                                device=device, dtype=torch.long)
            return (((vals.unsqueeze(-1).float() // PW) % 2) * 2 - 1).flatten(1).unsqueeze(0)
        return torch.cat([BITCODE[ids_row[torch.clamp(torch.arange(T, device=device) + j, max=T - 1)]]
                          for j in range(span)], -1).unsqueeze(0)

    def tocan(x):
        """Backbone space -> the width the brain trains in. Identity when not canonical."""
        return x if PIN is None else x @ PIN.t()

    def fromcan(x):
        """Back out, through the pseudo-inverse -- a transpose measured a round-trip gain of 7.5."""
        return x if POUT is None else x @ POUT.t()

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
                ekw = tocan(ekey(emb, e_carry).float())
                q = fromcan(ROUT(torch.zeros(1, width, device=device)) if raw is None else ROUT(raw))
                q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6) * emb.norm(dim=-1).mean()
                with torch.no_grad():         # frozen backbone: activations are constants
                    o = model(inputs_embeds=torch.cat(
                        [q.detach().unsqueeze(1).expand(-1, k_mem, -1), emb], 1),
                        output_hidden_states=True)
                h = tocan(o.hidden_states[-1][:, k_mem:].float())
                embc = tocan(emb.float())
                prev = carry if carry is not None else h[:, :1] * 0
                a = phi(torch.cat([prev, h[:, :-1]], 1), ekw)
                a = a * torch.sigmoid(WGATE(torch.cat([embc, h], -1)))
                v = torch.cat([CODE[ch[0]].unsqueeze(0) *
                               torch.sigmoid(VAL(torch.cat([embc, h], -1))), bitval(ch[0])], -1)
                carry, e_carry = h[:, -1:], emb[:, -ek:]
                old = torch.einsum("btn,bnd->btd", a, mem) / \
                      ((a * z.unsqueeze(1)).sum(-1, keepdim=True) + 1e-4)
                mem = mem + torch.einsum("btn,btd->bnd", a, v - torch.sigmoid(DBETA) * old) / a.shape[1]
                z = z + a.sum(1) / a.shape[1]
                mem = mem + 0.05 * torch.tanh(torch.sparse.mm(ADJ, mem[0]).unsqueeze(0))
                aq = phi(h[:, -1], tocan(qkey(emb).float()))
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
            w = eval_words[r.randrange(len(eval_words))]
            ids, ans = sample(w, fill_eval, r, 6)
            got = decode(run(ids))[0]
            # Same disease in mirror image: comparing decoded BYTES against expected TOKEN IDS
            # scores 0% by construction however well the brain learned, and nothing raises.
            want = (torch.tensor([b for b in (" " + w).encode("utf-8")[:span]], device=device)
                    if codec == "bytes" else ans[0])
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
        # The target must be in the SAME units as the payload. BITCODE indexes token ids, so using
        # it under --codec bytes trains against the id's low 8 bits while the memory stores the
        # token's UTF-8 bytes. It does not look like a failure: byte 0 is almost always 32 -- a
        # leading space -- so position 0 learns the constant and scores 98% while the rest sit at
        # exactly 0%. One position high and the others at zero is the signature of a wrong target.
        if codec == "bytes":
            flat = []
            for k in range(ans.shape[1]):
                i = int(ans[0, k])
                flat.extend(BTBL[i][:int(BLEN[i])].tolist())
            flat = (flat + [0] * span)[:span]
            nreal = span
            tgt = (((torch.tensor(flat, device=device).unsqueeze(-1).float() // PW) % 2) * 2 - 1) \
                .flatten().unsqueeze(0)
        else:
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
                                     "codec": codec,
                                     "vocab": int(V), "radius": float(radius), "seed": seed,
                                     "code_seed": _CODE_SEED, "steps": steps, "lr": lr,
                                     "canon": int(canon),
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
    ap.add_argument("--canon", type=int, default=0, metavar="N",
                    help="train in an N-dim canonical space instead of this backbone's hidden size,\n"
                         "so the brain attaches to ANY model. The bridge is derived at attach time\n"
                         "from the backbone's own responses to probe texts and is never trained;\n"
                         "canonical directions correspond across model families at +0.95. 512 is a\n"
                         "good value (round-trips a real hidden state at 0.962; 64 manages 0.879).\n"
                         "0, the default, keeps the direct path locked to this backbone.")
    ap.add_argument("--codec", default="tokens", choices=["tokens", "bytes"],
                    help="what the memory stores. 'tokens' (default) is this backbone's vocabulary\n"
                         "indices -- compact, but only this tokenizer can read them back. 'bytes'\n"
                         "stores UTF-8 text, so a brain is readable by ANY model's tokenizer and a\n"
                         "six-letter word costs 48 bits instead of 72.")
    ap.add_argument("--continue-from", default="",
                    help="carry on training an existing brain instead of starting over. On the SAME\n"
                         "backbone this is cumulative training -- run 900 steps, look at it, run 900\n"
                         "more. On a DIFFERENT backbone the memory-shaped parts (connectome, slots,\n"
                         "bit codebook, decay) carry over and only the hidden-size-shaped layers are\n"
                         "re-initialised, so the brain moves to a new model instead of starting from\n"
                         "nothing.")
    ap.add_argument("--width", type=int, default=328,
                    help="numbers per slot (default 328: 256 content + 72 bit, the\n"
                         "geometry the verified brain was trained at)")
    ap.add_argument("--ask",
                    default=" Question: what is the secret word? Answer: The secret word is",
                    help="how to CONTINUE the text, not a question to answer. The brain files a\n"
                         "token under the words that came before it, so retrieval works by\n"
                         "reproducing that context: a document saying 'The secret word is X' is\n"
                         "queried with 'The secret word is'. Phrase this as a question ending in\n"
                         "'Answer:' and you address a place nothing was ever written, and get\n"
                         "confident noise back.")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--save-state", default="", help="write the session state here")
    ap.add_argument("--load-state", default="", help="resume from a state file")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_backbone(a.model, torch.float32, a.device)
    corpus = open(a.probes, encoding="utf-8", errors="replace").read(1_000_000)

    if a.train:
        import numpy as np
        import pandas as pd
        g = pd.read_feather(a.connectome)
        train_brain(model, tok, corpus,
                    (g["body_pre"].to_numpy(), g["body_post"].to_numpy(), g["weight"].to_numpy()),
                    np.load(a.signs), out=a.brain, steps=a.train, width=a.width,
                    continue_from=a.continue_from, codec=a.codec, canon=a.canon,
                    device=a.device)
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
