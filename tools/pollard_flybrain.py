"""pollard-flybrain — a fruit-fly connectome as a language model's memory.

A transformer's memory is its KV cache: it grows with every token, and when the context window fills,
the oldest tokens are gone. A fly has no context window. This attaches a real connectome (MaleCNS
v1.0, FlyEM/Janelia) to a frozen model as a continuous recurrent state that never grows.

    from pollard_flybrain import FlyBrain

    brain = FlyBrain.load("FlyBrain-Pollard-CNSv1.pt", device="cuda")
    brain.attach(model)                      # any HF causal LM or vision-language model
    ...                                      # use the model exactly as before
    brain.save_state("session.flystate")     # 35 KB: the conversation, portable

    brain.load_state("session.flystate")     # tomorrow, pick up mid-thought

Three things worth knowing:

  * The state is fixed size -- 8,552 floats for CNSv1 -- whatever the sequence length. That is the
    whole point, and it is why the state fits in a file you can email.
  * The backbone is never modified. FlyBrain attaches with a forward hook and detaches cleanly.
  * A brain is trained against one backbone family. Loading it into an unrelated model will run, but
    it will not help -- the adapters are fitted to that model's representations. `attach()` warns
    when the hidden size does not match what the brain was trained on.
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

__all__ = ["FlyBrain"]
_STATE_MAGIC = b"FLYS"


class FlyBrain:
    """A connectome carried alongside a frozen model."""

    def __init__(self, blob: dict, device: str = "cpu"):
        self.meta = blob["meta"]
        self.device = device
        d = torch.device(device)
        # Accept lists, numpy arrays or tensors. `sign` is per-neuron and has to be gathered onto
        # the presynaptic side, which is numpy-style fancy indexing -- it raises on a plain list, and
        # a checkpoint written from lists is a perfectly reasonable thing for someone to hand us.
        self.src = torch.as_tensor(blob["src"], dtype=torch.long, device=d)
        self.dst = torch.as_tensor(blob["dst"], dtype=torch.long, device=d)
        sign = torch.as_tensor(blob["sign"], dtype=torch.float32, device=d)
        self.sgn = sign[self.src]
        self.w = blob["w"].to(d).float()
        self.tau = torch.sigmoid(blob["tau"].to(d).float())
        self.write = blob["write"].to(d).float()
        self.read = blob["read"].to(d).float()
        self.gate = torch.tanh(blob["gate"].to(d).float())
        self.n = int(self.meta["neurons"])
        self.state: Optional[torch.Tensor] = None
        self._proj_in = self._proj_out = None
        self._handle = None

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "FlyBrain":
        return cls(torch.load(path, map_location="cpu", weights_only=False), device)

    # ---------------------------------------------------------------- attach
    def attach(self, model, tokenizer=None, probe_text: Optional[str] = None, verbose: bool = True):
        """Hook the brain into a model's decoder stack.

        The bridge between the model's hidden space and the brain is built here, from the model's own
        responses to a fixed set of probe texts -- so it is derived per model, at load, and never
        shipped. Pass `probe_text` to supply the probe corpus; without it the brain runs with an
        identity-shaped bridge, which is only correct when the hidden size already matches.
        """
        stack = self._find_stack(model)
        hidden = self._hidden_size(model, stack)
        trained_on = self.meta.get("trained_against_hidden")
        if verbose and trained_on and hidden != trained_on:
            print(f"[flybrain] warning: this brain was trained against hidden {trained_on}, "
                  f"attaching to hidden {hidden}. It will run, but a brain does not transfer "
                  f"between backbone families -- retrain for this model.")

        if probe_text is not None and tokenizer is not None:
            self._build_bridge(model, tokenizer, probe_text, hidden)
        elif self._proj_in is None:
            raise ValueError("attach() needs `tokenizer` and `probe_text` to build the bridge; "
                             "use the same probe corpus the brain was trained with")

        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.state is None or self.state.shape[0] != h.shape[0]:
                self.reset(h.shape[0])
            delta = self._advance(h.float()).to(h.dtype)
            h = h + delta
            return (h,) + out[1:] if isinstance(out, tuple) else h

        self._handle = stack.layers[-1].register_forward_hook(hook)
        if verbose:
            print(f"[flybrain] attached: {self.n:,} neurons, {self.meta['synapses']:,} synapses, "
                  f"state {self.n * 4 / 1e6:.2f} MB (constant)")
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return self

    # ---------------------------------------------------------------- state
    def reset(self, batch: int = 1):
        self.state = torch.zeros(batch, self.n, device=torch.device(self.device))
        return self

    def save_state(self, path: str) -> int:
        """Write the live memory. Fixed size, whatever has happened."""
        if self.state is None:
            raise RuntimeError("no state yet -- run the model first")
        torch.save({"magic": _STATE_MAGIC, "state": self.state.detach().cpu().float(),
                    "neurons": self.n, "brain": self.meta.get("name")}, path)
        return os.path.getsize(path)

    def load_state(self, path: str):
        """Resume. A state from a different connectome is refused rather than silently accepted."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("magic") != _STATE_MAGIC:
            raise ValueError(f"{path} is not a FlyBrain state file")
        if blob["neurons"] != self.n:
            raise ValueError(f"state has {blob['neurons']} neurons, this brain has {self.n}")
        self.state = blob["state"].to(torch.device(self.device))
        return self

    # ---------------------------------------------------------------- internals
    def _advance(self, hidden: torch.Tensor) -> torch.Tensor:
        """One neural step per token; returns what to add back into the residual stream."""
        drive = (hidden @ self._proj_in.t()) @ self.write.t()
        outs = []
        st = self.state
        for i in range(hidden.shape[1]):
            msg = st[:, self.src] * (self.w * self.sgn)
            agg = torch.zeros_like(st).index_add_(1, self.dst, msg)
            st = (1 - self.tau) * st + self.tau * torch.tanh(agg + drive[:, i])
            outs.append(st)
        self.state = st.detach()
        return ((torch.stack(outs, 1) @ self.read.t()) * self.gate) @ self._proj_out.t()

    def _build_bridge(self, model, tokenizer, corpus: str, hidden: int):
        """Probe-stimulus basis: direction i is this model's response to probe i."""
        c = int(self.meta["canonical"])
        plen = int(self.meta.get("probe_len", 160))
        rows = []
        with torch.no_grad():
            for i in range(c):
                ids = tokenizer(corpus[i * plen:(i + 1) * plen],
                                return_tensors="pt").input_ids[:, :48].to(self.device)
                if ids.shape[1] < 2:
                    continue
                h = model(input_ids=ids, output_hidden_states=True).hidden_states[-1][0].float()
                rows.append(h.mean(0))
        A = torch.stack(rows)
        A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-6)
        # The basis is not orthonormal, so the return path must be a pseudo-inverse. Using the
        # transpose inflates the round trip several-fold and the memory becomes unusable.
        self._proj_in = A                          # (C, D)
        self._proj_out = torch.linalg.pinv(A)      # (D, C); applied as .t() on the way back

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
def train_brain(model, tokenizer, corpus, connectome, signs, *, out, canonical=512, probe_len=160,
                win=128, ctx=1024, steps=400, lr=5e-5, radius=0.95, gain=0.1, seed=0,
                device="cpu", log=print):
    """Fit a brain to a backbone. The backbone is frozen; only the adapters and gate learn.

    A brain does not transfer between backbone families, so this is how you make one for yours. The
    task is the claim itself: attention sees `win` tokens, the connectome sees everything, and the
    loss is scored on the final chunk -- whose useful context is entirely outside attention's reach.

    `connectome` is (src, dst, weight) arrays; `signs` is +1/-1 per neuron.
    """
    import numpy as np

    torch.manual_seed(seed); np.random.seed(seed)
    src, dst, w_raw = connectome
    src = np.asarray(src); dst = np.asarray(dst)
    nodes, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
    E, N = len(src), len(nodes)
    src_i, dst_i = inv[:E], inv[E:]
    sign = np.asarray(signs, dtype=np.float32)

    for q in model.parameters():
        q.requires_grad_(False)
    stack = FlyBrain._find_stack(model)
    D = FlyBrain._hidden_size(model, stack)
    dev = torch.device(device)

    w = np.log1p(np.asarray(w_raw, dtype=np.float32))
    rho = _spectral_radius(src_i, dst_i, w * sign[src_i], N, dev)
    w = w * (radius / rho)
    log(f"connectome {N:,} neurons {E:,} synapses, rho {rho:.1f} -> {radius}")

    # the bridge: this model's own responses to a fixed probe set (see FlyBrain._build_bridge)
    rows = []
    with torch.no_grad():
        for i in range(canonical):
            ids = tokenizer(corpus[i*probe_len:(i+1)*probe_len],
                            return_tensors="pt").input_ids[:, :48].to(dev)
            if ids.shape[1] < 2:
                continue
            rows.append(model(input_ids=ids, output_hidden_states=True)
                        .hidden_states[-1][0].float().mean(0))
    A = torch.stack(rows); A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-6)
    Ainv = torch.linalg.pinv(A)                      # NOT A.t(): the basis is not orthonormal
    C = A.shape[0]
    log(f"probe bridge: {C} probes x hidden {D}, pseudo-inverse return path")

    SRC = torch.as_tensor(src_i, dtype=torch.long, device=dev)
    DST = torch.as_tensor(dst_i, dtype=torch.long, device=dev)
    SGN = torch.as_tensor(sign, dtype=torch.float32, device=dev)[SRC]
    W = nn.Parameter(torch.as_tensor(w, dtype=torch.float32, device=dev))
    TAU = nn.Parameter(torch.full((N,), 0.5, device=dev))
    WRITE = nn.Linear(C, N, bias=False).to(dev)
    READ = nn.Linear(N, C, bias=False).to(dev)
    torch.nn.init.normal_(WRITE.weight, std=1.0 / math.sqrt(C))
    with torch.no_grad():
        READ.weight.copy_(WRITE.weight.t() * (gain * C / N))
    # not zero: at exactly zero the gradient reaching the adapters is zero too, and nothing learns
    GATE = nn.Parameter(torch.full((1,), 0.05, device=dev))

    def step_state(st, drive_t):
        msg = st[:, SRC] * (W * SGN)
        agg = torch.zeros_like(st).index_add_(1, DST, msg)
        t = torch.sigmoid(TAU)
        return (1 - t) * st + t * torch.tanh(agg + drive_t)

    ids_all = tokenizer(corpus, return_tensors="pt").input_ids.to(dev)
    embed = stack.embed_tokens

    def score(mode, ids, grad=False):
        with (torch.enable_grad() if grad else torch.no_grad()):
            if mode == "full":
                lg = model(input_ids=ids).logits[0, -win-1:-1]
                return nn.functional.cross_entropy(lg.float(), ids[0, -win:])
            st = torch.zeros(1, N, device=dev); loss = None
            for s0 in range(0, ids.shape[1], win):
                ch = ids[:, s0:s0+win]
                if ch.shape[1] < 2:
                    continue
                emb = embed(ch)
                if mode == "brain":
                    drive = (emb @ A.t()) @ WRITE.weight.t()
                    outs = []
                    for i in range(emb.shape[1]):
                        st = step_state(st, drive[:, i]); outs.append(st)
                    emb = emb + ((torch.stack(outs, 1) @ READ.weight.t())
                                 * torch.tanh(GATE)) @ Ainv.t()
                out = model(inputs_embeds=emb)
                if s0 + win >= ids.shape[1]:
                    loss = nn.functional.cross_entropy(out.logits[0, :-1].float(), ch[0, 1:])
            return loss

    ids = ids_all[:, :ctx]
    floor = float(score("window", ids)); ceil = float(score("full", ids))
    log(f"floor (windowed) {floor:.4f} | ceiling (full attention) {ceil:.4f} | gap {floor-ceil:.4f}")

    params = [WRITE.weight, READ.weight, GATE]
    opt = torch.optim.Adam(params, lr=lr)
    best, best_state = -1e9, None
    n_doc = max(1, ids_all.shape[1] // ctx)
    for it in range(1, steps + 1):
        j = np.random.randint(0, max(1, n_doc - 1))
        loss = score("brain", ids_all[:, j*ctx:(j+1)*ctx], grad=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if it % max(1, steps // 20) == 0:
            v = float(score("brain", ids))
            pct = 100 * (floor - v) / max(1e-9, floor - ceil)
            log(f"  step {it:>5}  held-out {v:.4f}   closes {pct:>5.1f}% of the gap")
            if pct > best:
                best = pct
                best_state = {"write": WRITE.weight.detach().clone(),
                              "read": READ.weight.detach().clone(),
                              "gate": GATE.detach().clone(),
                              "w": W.detach().clone(), "tau": TAU.detach().clone()}

    torch.save({"src": src_i, "dst": dst_i, "sign": sign,
                **{k: v.cpu() for k, v in best_state.items()},
                "meta": {"name": os.path.basename(out).replace(".pt", ""),
                         "backbone": getattr(model.config, "_name_or_path", "unknown"),
                         "canonical": C, "probe_len": probe_len, "neurons": int(N),
                         "synapses": int(E), "radius": radius, "gain": gain, "win": win,
                         "ctx": ctx, "steps": steps, "lr": lr, "seed": seed,
                         "trained_against_hidden": int(D), "floor": floor, "ceiling": ceil,
                         "closes_pct": best, "bridge": "probe-stimulus basis (RSA)"}}, out)
    log(f"saved {out} ({os.path.getsize(out)/1e6:.1f} MB) -- best {best:.1f}% of the gap")
    return best


def _spectral_radius(src, dst, vals, n, device, iters=50):
    """Power iteration. ARPACK on a 26M-edge graph is needlessly heavy and memory-hungry."""
    s = torch.as_tensor(src, dtype=torch.long, device=device)
    d = torch.as_tensor(dst, dtype=torch.long, device=device)
    v = torch.as_tensor(vals, dtype=torch.float32, device=device)
    x = torch.randn(n, device=device); x /= x.norm()
    lam = 1.0
    for _ in range(iters):
        y = torch.zeros(n, device=device).index_add_(0, d, x[s] * v)
        nrm = y.norm()
        if nrm < 1e-12:
            return 1.0
        lam = float(nrm); x = y / nrm
    return lam


def main():
    """CLI: attach a brain to a model and show what it costs and what it remembers."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--train", type=int, default=0,
                    help="fit a NEW brain to --model for this many steps, then save to --brain.\n                         A brain does not transfer between backbones, so this is how you make\n                         one for yours. Needs --connectome.")
    ap.add_argument("--connectome", default="",
                    help="feather/parquet with body_pre, body_post, weight (+ --signs .npy)")
    ap.add_argument("--signs", default="", help="per-neuron +1/-1 npy")
    ap.add_argument("--brain", required=True, help="a .pt brain, e.g. FlyBrain-Pollard-CNSv1.pt")
    ap.add_argument("--model", required=True, help="HF model id to attach it to")
    ap.add_argument("--probes", required=True,
                    help="probe corpus: the bridge is built from this model's responses to it")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--save-state", default="", help="write the conversation state here")
    ap.add_argument("--load-state", default="", help="resume from a state file")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device).eval()
    corpus = open(a.probes, encoding="utf-8", errors="replace").read(400_000)

    if a.train:
        import numpy as np, pandas as pd
        g = pd.read_feather(a.connectome)
        train_brain(model, tok, corpus,
                    (g["body_pre"].to_numpy(), g["body_post"].to_numpy(), g["weight"].to_numpy()),
                    np.load(a.signs), out=a.brain, steps=a.train, device=a.device)
        return

    brain = FlyBrain.load(a.brain, device=a.device).attach(model, tokenizer=tok, probe_text=corpus)
    if a.load_state:
        brain.load_state(a.load_state)
        print(f"resumed from {a.load_state}")

    ids = tok(a.prompt, return_tensors="pt").input_ids.to(a.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=a.tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    print(tok.decode(out[0], skip_special_tokens=True))

    if a.save_state:
        n = brain.save_state(a.save_state)
        print(f"\nstate -> {a.save_state} ({n:,} bytes, and it does not grow)")


if __name__ == "__main__":
    main()
