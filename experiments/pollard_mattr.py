#!/usr/bin/env python3
"""pollard-mattr (EXPERIMENT) -- learn the whole bit-allocation ordering in ONE run.

pollard-sensitivity / pollard-probe find what to protect by crushing one group at a time and reading
the damage: 2*layers passes, group granularity. Matryoshka Attribution (Arora et al. 2026, arXiv
2609.25518) learns one score per component and trains a differentiable sigmoid top-k mask at a RANDOM
sparsity every step, so the scores come out as a nested ordering over every sparsity at once. Their
weight experiment already interpolates per ROW of each weight matrix between two checkpoints:

    W_eff = W_base + m * (W_other - W_base),   m in [0,1] per row

Here W_base is the f16 tensor and W_other is its low-bit crush (the same RTN-2 proxy pollard-probe
uses), so m=1 means "this row takes the crush" and m=0 means "this row stays f16". Loss is the
KL(f16 || mixed) next-token distribution on calibration text. After training, sorting rows by score
gives the crush order at every budget -- the protect list at row granularity, from one training run.

What this measures (the experiment, not a shipped tool):
  1. at equal budgets (fraction of rows crushed), KL and top-1 of: learned row ordering, the same
     ordering coarsened to tensor / attn-ffn-group level (like-for-like with the probe), pollard-probe's
     group ordering, random -- does one run of MAttr beat the 2*layers sweep, and by how much?
  2. rank agreement between the learned tensor-level order and the probe's group costs.
  3. an automap-style protect list (GGUF tensor names) from the learned order.

    python experiments/pollard_mattr.py --model ~/pollard/downloads/Qwen__Qwen2.5-0.5B-Instruct \
        --calib ~/pollard/downloads/pollard_calib.txt --steps 200 --device mps \
        --probe ~/pollard/downloads/Qwen__Qwen2.5-0.5B-Instruct.sensitivity.json --out mattr.json

Everything here is model-side torch; reuses pollard_gptq's loader / RTN / text chunking.
"""
from __future__ import annotations

import argparse, json, math, os, random, re, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from pollard_gptq import load_backbone, get_wikitext, rtn_quantize, text_layers  # noqa: E402


# ── the differentiable top-k mask (MAttr eq.: m = sigma((s + c_k)/T), c_k by bisection) ────────────
class SigmoidTopK(Function):
    """Soft top-k with implicit-differentiation backward: sum_i sigma((s_i - tau)/T) = k."""

    @staticmethod
    def forward(ctx, scores, k, T, iters):
        lo = scores.min() - 10 * T
        hi = scores.max() + 10 * T
        with torch.no_grad():
            for _ in range(iters):
                mid = (lo + hi) / 2
                f = torch.sigmoid((scores - mid) / T).sum()
                lo, hi = (mid, hi) if f > k else (lo, mid)
        tau = (lo + hi) / 2
        m = torch.sigmoid((scores - tau) / T)
        ctx.save_for_backward(m)
        ctx.T = T
        return m

    @staticmethod
    def backward(ctx, g):
        (m,) = ctx.saved_tensors
        p = m * (1 - m) / ctx.T                       # d m_i / d s_i  (holding tau)
        # tau moves to keep the sum at k:  d tau / d s_i = p_i / sum(p);  d m_j / d s_i = p_j (delta_ij - p_i/sum p)
        gs = p * (g - (g * p).sum() / p.sum().clamp_min(1e-12))
        return gs, None, None, None


def sig_topk(scores, k, T=1.0, iters=30):
    return SigmoidTopK.apply(scores, float(k), float(T), int(iters))


# ── name mapping so the protect list can feed automap ───────────────────────────────────────────────
_HF2GGUF = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
            "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
            "mlp.down_proj": "ffn_down"}


def gguf_name(layer_idx: int, hf_suffix: str) -> str:
    return f"blk.{layer_idx}.{_HF2GGUF.get(hf_suffix, hf_suffix.replace('.', '_'))}.weight"


def group_of(hf_suffix: str) -> str:
    return "attn" if hf_suffix.startswith("self_attn") else "ffn"


# ── the model with row-interpolated linears ─────────────────────────────────────────────────────────
class RowMix:
    """Every decoder Linear computes F.linear(x, W + m[:,None] * D) with m from ONE score vector."""

    def __init__(self, model, bits: int, groupsize: int, device):
        self.items = []            # (layer_idx, suffix, module, W(f16), D(=Q-W), row_slice)
        rows = 0
        model.requires_grad_(False)                 # only the score vector learns
        for li, layer in enumerate(text_layers(model)):
            for name, mod in layer.named_modules():
                if not isinstance(mod, nn.Linear):
                    continue
                W = mod.weight.detach()
                Q = rtn_quantize(W.float(), bits, groupsize).to(W.dtype).to(device)
                D = (Q - W).contiguous()
                n = W.shape[0]
                self.items.append((li, name, mod, W, D, slice(rows, rows + n)))
                rows += n
                mod.weight.requires_grad_(False)
        self.n_rows = rows
        self.scores = nn.Parameter(torch.zeros(rows, device=device))
        self._mask = None
        for (_, _, mod, W, D, sl) in self.items:
            mod.forward = self._make_forward(mod, W, D, sl)

    def _make_forward(self, mod, W, D, sl):
        def fwd(x):
            if self._mask is None:
                return F.linear(x, W, mod.bias)
            m = self._mask[sl].to(W.dtype).unsqueeze(1)
            return F.linear(x, W + m * D, mod.bias)
        return fwd

    def set_mask(self, m):                # None = pure f16
        self._mask = m

    # hard masks for evaluation
    def hard_mask_from_order(self, order_scores: torch.Tensor, frac_crushed: float):
        k = int(round(frac_crushed * self.n_rows))
        m = torch.zeros(self.n_rows, device=order_scores.device)
        if k > 0:
            idx = torch.topk(order_scores, k).indices    # highest score = crushed first
            m[idx] = 1.0
        return m

    def tensor_table(self, scores: torch.Tensor):
        """Per tensor: mean score (higher = crush earlier), rows, group, gguf name."""
        out = []
        for (li, name, mod, W, D, sl) in self.items:
            s = scores[sl]
            out.append({"layer": li, "hf": name, "gguf": gguf_name(li, name), "group": group_of(name),
                        "rows": int(W.shape[0]), "cols": int(W.shape[1]),
                        "score_mean": float(s.mean()), "score_min": float(s.min()),
                        "frac_protect_at_50": float((s < scores.median()).float().mean())})
        return out


# ── KL against the f16 reference ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def ref_logprobs(model, mix, chunks, dev):
    mix.set_mask(None)
    out = []
    for c in chunks:
        lp = F.log_softmax(model(c.unsqueeze(0).to(dev)).logits[0, :-1].float(), -1)
        out.append(lp)
    return out


def kl_top1(model, mix, chunks, ref_lps, dev, mask):
    mix.set_mask(mask)
    kl_sum = t1 = n = 0.0
    for c, lpr in zip(chunks, ref_lps):
        lq = F.log_softmax(model(c.unsqueeze(0).to(dev)).logits[0, :-1].float(), -1)
        pr = lpr.exp()
        kl = (pr * (lpr - lq)).sum(-1)
        kl_sum += kl.sum(); t1 += (lpr.argmax(-1) == lq.argmax(-1)).float().sum(); n += kl.numel()
    return kl_sum / n, 100.0 * t1 / n


def probe_order(probe_path: str, mix: RowMix, dev) -> torch.Tensor | None:
    """pollard-probe's ordering lifted to rows: a group's cost is its KL when crushed; LOW cost = crush
    first, so score = -cost, the same sign convention as the learned scores."""
    if not probe_path or not os.path.isfile(probe_path):
        return None
    d = json.load(open(probe_path))
    s = torch.zeros(mix.n_rows, device=dev)
    for (li, name, mod, W, D, sl) in mix.items:
        cost = float((d.get(group_of(name)) or {}).get(str(li), 0.0))
        s[sl] = -cost
    # tiny deterministic jitter so top-k within an equal-cost group is well defined
    g = torch.Generator(device="cpu").manual_seed(0)
    s += torch.rand(mix.n_rows, generator=g).to(dev) * 1e-6
    return s


def coarsen(mix: RowMix, scores: torch.Tensor, level: str) -> torch.Tensor:
    """The learned row order re-expressed at coarser granularity so the probe comparison is
    like-for-like: 'tensor' = every row takes its tensor's mean score; 'group' = its layer's
    attn-or-ffn mean (pollard-probe's 2*layers groups)."""
    out = torch.empty_like(scores)
    if level == "tensor":
        for (li, name, mod, W, D, sl) in mix.items:
            out[sl] = scores[sl].mean()
    else:
        acc = {}
        for (li, name, mod, W, D, sl) in mix.items:
            acc.setdefault((li, group_of(name)), []).append(scores[sl])
        means = {k: torch.cat(v).mean() for k, v in acc.items()}
        for (li, name, mod, W, D, sl) in mix.items:
            out[sl] = means[(li, group_of(name))]
    g = torch.Generator(device="cpu").manual_seed(0)
    return out + torch.rand(mix.n_rows, generator=g).to(scores.device) * 1e-6


def spearman(a, b):
    ra = torch.argsort(torch.argsort(a)).float(); rb = torch.argsort(torch.argsort(b)).float()
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True, help="calibration text file (train side)")
    ap.add_argument("--eval-split", default="test", help="wikitext-2 split for held-out KL (offline cache)")
    ap.add_argument("--probe", help="pollard-probe / sensitivity.json to compare against")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--groupsize", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--temp", type=float, default=1.0, help="sigmoid temperature T")
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--calib-chunks", type=int, default=16)
    ap.add_argument("--eval-chunks", type=int, default=6)
    ap.add_argument("--budgets", default="0.3,0.5,0.7,0.85", help="fractions of rows crushed to evaluate")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="mattr.json")
    ap.add_argument("--protect-top", type=int, default=12, help="how many tensors to name in the protect list")
    a = ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = torch.device(a.device)
    t0 = time.time()
    dtype = torch.float16 if dev.type != "cpu" else torch.float32
    model = load_backbone(a.model, dtype=dtype, device=str(dev))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    calib = get_wikitext(tok, "train", a.seqlen, n=a.calib_chunks, path=a.calib)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    evalc = get_wikitext(tok, a.eval_split, a.seqlen, n=a.eval_chunks)
    print(f"model loaded {time.time()-t0:.0f}s · calib {len(calib)} chunks · eval {len(evalc)} chunks · {dev}")

    mix = RowMix(model, a.bits, a.groupsize, dev)
    print(f"components: {mix.n_rows:,} rows over {len(mix.items)} tensors (crush = RTN {a.bits}-bit, g{a.groupsize})")
    ref_calib = ref_logprobs(model, mix, calib, dev)
    ref_eval = ref_logprobs(model, mix, evalc, dev)

    # ── train the ordering: random k every step, KL to f16 ──
    opt = torch.optim.Adam([mix.scores], lr=a.lr)
    hist = []
    for step in range(a.steps):
        k = random.randint(1, mix.n_rows - 1)
        m = sig_topk(mix.scores, k, a.temp)
        mix.set_mask(m)
        i = step % len(calib)
        lq = F.log_softmax(model(calib[i].unsqueeze(0).to(dev)).logits[0, :-1].float(), -1)
        lpr = ref_calib[i]
        loss = (lpr.exp() * (lpr - lq)).sum(-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        hist.append(float(loss))
        if step % 20 == 0 or step == a.steps - 1:
            print(f"  step {step:4d}  k={k/mix.n_rows:5.2f}  KL={float(loss):.4f}  {time.time()-t0:.0f}s")
    learned = mix.scores.detach()

    # ── evaluate orderings at equal budgets on HELD-OUT text ──
    orders = {"mattr": learned, "mattr_tensor": coarsen(mix, learned, "tensor"),
              "mattr_group": coarsen(mix, learned, "group")}
    po = probe_order(a.probe, mix, dev)
    if po is not None:
        orders["probe"] = po
    g = torch.Generator(device="cpu").manual_seed(1)
    orders["random"] = torch.rand(mix.n_rows, generator=g).to(dev)
    results = {}
    with torch.no_grad():
        kl0, t10 = kl_top1(model, mix, evalc, ref_eval, dev, torch.zeros(mix.n_rows, device=dev))
        klA, t1A = kl_top1(model, mix, evalc, ref_eval, dev, torch.ones(mix.n_rows, device=dev))
        results["f16"] = {"kl": float(kl0), "top1": float(t10)}
        results["all_crushed"] = {"kl": float(klA), "top1": float(t1A)}
        print(f"f16 sanity KL={float(kl0):.5f}  all-crushed KL={float(klA):.4f} top1={float(t1A):.1f}%")
        for frac in [float(x) for x in a.budgets.split(",")]:
            row = {}
            for name, s in orders.items():
                kl, t1 = kl_top1(model, mix, evalc, ref_eval, dev, mix.hard_mask_from_order(s, frac))
                row[name] = {"kl": float(kl), "top1": float(t1)}
            results[f"crush_{frac:.2f}"] = row
            print(f"crush {frac:.0%} of rows:")
            for n, v in row.items():
                print(f"    {n:13s} KL {v['kl']:.4f}   top1 {v['top1']:.1f}%")
    mix.set_mask(None)

    # ── tensor-level view + protect list + agreement with the probe ──
    table = mix.tensor_table(learned)
    table.sort(key=lambda r: r["score_mean"])                 # lowest score = most protected
    protect = [r["gguf"] for r in table[:a.protect_top]]
    agree = None
    if po is not None:
        tl = torch.tensor([r["score_mean"] for r in table]); tp = torch.tensor(
            [float(-(json.load(open(a.probe)).get(r["group"]) or {}).get(str(r["layer"]), 0.0)) for r in table])
        agree = spearman(tl, tp)
    out = {"model": a.model, "bits": a.bits, "groupsize": a.groupsize, "steps": a.steps, "rows": mix.n_rows,
           "seconds": round(time.time() - t0, 1), "loss_curve": hist[::max(1, len(hist) // 50)],
           "results": results, "spearman_vs_probe_tensor_level": agree,
           "protect_first": protect, "tensors": table}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nlearned order vs probe (tensor level, Spearman): {agree}")
    print("protect first:", ", ".join(protect[:8]), "...")
    print(f"wrote {a.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
