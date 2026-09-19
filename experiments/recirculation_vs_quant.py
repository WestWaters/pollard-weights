#!/usr/bin/env python3
"""Does recirculation buy back what quantization costs?

Recirculation (arXiv 2608.17981) is a training-free inference change: mix a DEEP layer's hidden
state from the previous token into a SHALLOW layer of the current one, renormalised to the
destination's scale.

    z_{t+1,d} = a * f(z_{t,s}) + (1-a) * z_{t,d}
    f(z|d)    = (||z_d||_2 / ||z||_2) * z          a = 0.15

The paper reports it on FULL-PRECISION models. The question for Pollard is the one nobody asked:
a quantized model has lost information, and recirculation gives the residual stream another pass
at what is left. If it recovers some of the loss, it is a quality lever that costs no bits -- and
the whole low-bit fight is quality at a fixed size.

This measures four points on the SAME text, in-process, no GGUF and no CUDA needed:

    fp16            fp16 + recirc
    RTN w-bit       RTN w-bit + recirc

Run:  python experiments/recirculation_vs_quant.py --model Qwen/Qwen2.5-0.5B-Instruct --bits 3
"""
from __future__ import annotations

import argparse, math, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import torch


def rtn(w: torch.Tensor, bits: int, group: int = 64) -> torch.Tensor:
    """Round-to-nearest with GROUP-WISE scales, which is what a real quantizer does.

    Per-ROW scales (one scale for a whole 3840-wide row) are not a 3-bit model, they are rubble:
    measured here at PPL 525847 against fp16's 22.4. Every shipping format -- K-quants, GPTQ, AWQ --
    scales in blocks of 32-128, and comparing against the broken version would flatter anything."""
    if bits >= 16:
        return w
    out, in_ = w.shape
    g = min(group, in_)
    if in_ % g:
        g = in_
    wv = w.reshape(out, in_ // g, g)
    q = 2 ** (bits - 1) - 1
    sc = wv.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / q
    return ((wv / sc).round().clamp(-q - 1, q) * sc).reshape(out, in_)


class Recirculator:
    """Recirculation: run the model TWICE over the same input, and seed the second pass's shallow
    layer with the first pass's deep state, renormalised to the destination's scale.

    The `t` in the paper's equation indexes the RECURRENCE STEP, not the token -- "two input stacks
    run in parallel at each recurrence step". Reading it as the previous token and rolling the
    sequence by one position is a different operation entirely, and it made every pair worse by
    0.8-2.4% on both fp16 and 3-bit. The monotonic wrongness was the tell: a real mechanism does not
    degrade that tidily.

    Nothing is trained and no weight is touched -- two hooks and one buffer."""

    def __init__(self, layers, src: int, dst: int, alpha: float = 0.15):
        self.src, self.dst, self.alpha = src, dst, alpha
        self.capture, self.inject, self.buf = False, False, None
        self.handles = [layers[src].register_forward_hook(self._grab),
                        layers[dst].register_forward_hook(self._blend)]

    @staticmethod
    def _hs(out):
        return out[0] if isinstance(out, tuple) else out

    def _grab(self, _m, _i, out):
        if self.capture:
            self.buf = self._hs(out).detach()
        return out

    def _blend(self, _m, _i, out):
        if not (self.inject and self.buf is not None):
            return out
        z_d, z_s = self._hs(out), self.buf
        if z_s.shape != z_d.shape:
            return out
        nd = z_d.norm(dim=-1, keepdim=True)
        ns = z_s.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        mixed = self.alpha * (nd / ns) * z_s + (1.0 - self.alpha) * z_d
        return (mixed,) + tuple(out[1:]) if isinstance(out, tuple) else mixed

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def perplexity(model, ids, ctx=512, rc=None):
    """Teacher-forced PPL over fixed windows. With a Recirculator, each window costs TWO passes:
    one to capture the deep state, one scored with it injected."""
    nll, n = 0.0, 0
    for i in range(0, ids.numel() - ctx, ctx):
        win = ids[i:i + ctx].unsqueeze(0)
        if rc is not None:
            rc.capture, rc.inject = True, False
            model(win)                                   # pass 1: capture
            rc.capture, rc.inject = False, True          # pass 2: inject and score
        out = model(win).logits[0, :-1].float().log_softmax(-1)
        tgt = win[0, 1:]
        nll += -out.gather(-1, tgt.unsqueeze(-1)).sum().item()
        n += tgt.numel()
        if rc is not None:
            rc.capture = rc.inject = False
    return math.exp(nll / max(n, 1))


def find_pair(model, layers, ids, ctx, alpha, baseline, quiet=False):
    """MEASURE the best source->destination pair for this model instead of assuming one.

    The paper's pairs are per-model -- {11,4} for Gemma3 1B, {18,9} for 4B, {35,16} for 12B, which
    are 0.42/0.15, 0.53/0.26 and 0.73/0.33 of depth. There is no single ratio to carry over, and
    picking one by eye is how you conclude a technique does not work when you simply chose badly.

    Returns (src, dst, ppl) for the best pair, or None if NONE of them beat the baseline -- which is
    a real answer about this model, not a failure to search."""
    best = None
    grid = [(0.35, 0.10), (0.42, 0.15), (0.53, 0.26), (0.60, 0.20), (0.73, 0.33), (0.80, 0.40)]
    n = len(layers)
    for fs, fd in grid:
        s_, d_ = max(1, int(n * fs)), max(0, int(n * fd))
        if s_ <= d_:
            continue
        r = Recirculator(layers, s_, d_, alpha)
        ppl = perplexity(model, ids, ctx, rc=r)
        r.remove()
        if not quiet:
            print(f"      {s_:>3} -> {d_:<3} ({fs:.2f}/{fd:.2f})  PPL {ppl:9.4f}  "
                  f"{(baseline - ppl) / baseline * 100:+6.2f}%")
        if ppl < baseline and (best is None or ppl < best[2]):
            best = (s_, d_, ppl)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--bits", type=int, default=3, help="RTN bits for the quantized arm")
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--src", type=int, help="source layer (default: 3/4 depth, as the paper's pairs sit)")
    ap.add_argument("--dst", type=int, help="destination layer (default: 1/3 depth)")
    ap.add_argument("--tokens", type=int, default=20000)
    ap.add_argument("--ctx", type=int, default=512)
    a = ap.parse_args()

    from pollard_backbone import load_backbone, text_layers
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_backbone(a.model, torch.float32, "cpu")
    layers = text_layers(model)
    n = len(layers)
    src = a.src if a.src is not None else int(n * 0.75)
    dst = a.dst if a.dst is not None else int(n * 0.34)
    print(f"== recirculation vs quantization :: {a.model}")
    print(f"   {n} layers, recirc {src} -> {dst}, alpha {a.alpha}, RTN {a.bits}-bit\n")

    text = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0][:a.tokens]

    orig = {id(l): {n_: p.detach().clone() for n_, p in l.named_parameters() if p.dim() == 2}
            for l in layers}

    def set_bits(bits):
        for l in layers:
            for n_, p in l.named_parameters():
                if p.dim() == 2:
                    p.data.copy_(orig[id(l)][n_] if bits >= 16 else rtn(orig[id(l)][n_], bits))

    rows = []
    for bits in (16, a.bits):
        set_bits(bits)
        base = perplexity(model, ids, a.ctx)
        label = "fp16" if bits >= 16 else f"{bits}-bit"
        print(f"   {label}  baseline PPL {base:.4f} -- searching for a pair that beats it:")
        found = find_pair(model, layers, ids, a.ctx, a.alpha, base)
        if found is None:
            print(f"      no pair beat the baseline on {label}\n")
            rows.append((bits, base, None))
            continue
        s_, d_, with_r = found
        print(f"      best {s_} -> {d_}: PPL {with_r:.4f}  ({(base - with_r) / base * 100:+.2f}%)\n")
        rows.append((bits, base, with_r))

    (_b16, p16, r16), (_bq, pq, rq) = rows
    print(f"   quantization cost   : {(pq - p16) / p16 * 100:+.1f}%")
    if rq is None:
        print("   recirculation       : did not help this model at any pair searched.")
        print("   Worth knowing before building on it. Validate the implementation against a model\n"
              "   the paper itself reports (Gemma3 family) before concluding the technique does not\n"
              "   transfer -- a null result on one family is not a null result.")
    else:
        print(f"   recovered by recirc : {(pq - rq) / (pq - p16) * 100:+.1f}% of the quantization loss")


if __name__ == "__main__":
    main()
