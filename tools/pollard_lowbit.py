#!/usr/bin/env python3
"""pollard-lowbit — the extreme-low-bit R&D prototype. Test whether "catching the
collapse" keeps 1-2 bit coherent where plain low-bit face-plants.

Two levers everyone documents but nobody stacks for GGUF:
  1. OUTLIER-CATCH (dense+sparse, SpQR/SqueezeLLM): a tiny set of high-sensitivity
     weights carry most of the damage at 1-bit — keep THOSE in fp16, crush the rest.
  2. RESIDUAL CAROUSEL (RVQ/additive, AQLM): quantize, take the leftover error,
     quantize that, stack corrections — the error keeps refining, never fully dies.

Measures WikiText-2 PPL for: fp16, plain RTN (the collapse baseline), outlier-catch,
residual, and the combination — at a target bit-width, on a small model (fast).

Usage:
  pollard-lowbit --model hf_qwen05 --bits 1 --keep 0.01 --levels 3 \
      --calib-file wiki_train.txt --eval-file wikitext2_test.txt --device cpu
"""
import argparse, time, sys
import torch, torch.nn as nn


def _chunks(tok, text, seqlen, n=None):
    enc = tok(text, return_tensors="pt").input_ids[0]
    ch = [enc[i:i + seqlen] for i in range(0, enc.numel() - seqlen, seqlen)]
    return torch.stack(ch[:n] if n else ch)


@torch.no_grad()
def eval_ppl(model, test):
    model.eval(); dev = next(model.parameters()).device; nll = ntok = 0
    for c in test:
        ids = c.unsqueeze(0).to(dev)
        nll += model(ids, labels=ids).loss.float().item() * (ids.numel() - 1)
        ntok += ids.numel() - 1
    return torch.exp(torch.tensor(nll / ntok)).item()


def q_group(W, bits, groupsize):
    """Symmetric absmax quant to `bits` levels, per group-of-`groupsize` along input."""
    W = W.float(); rows, cols = W.shape; lv = 2 ** (bits - 1) - 1 if bits > 1 else 1
    Q = torch.zeros_like(W); gs = groupsize or cols
    for c0 in range(0, cols, gs):
        g = W[:, c0:c0 + gs]
        s = g.abs().amax(1, keepdim=True).clamp(min=1e-8) / lv
        Q[:, c0:c0 + g.shape[1]] = torch.clamp(torch.round(g / s), -lv, lv) * s
    return Q.to(torch.float16)


def q_ternary(W, groupsize):
    """BitNet b1.58 abs-MEAN ternary {-1,0,+1}. Scale = mean(|W|) per group (NOT absmax).
    This is the lever the collapse baseline skipped: absmax rounds everything but the
    group's single biggest weight to zero; abs-mean keeps the mass alive. ~1.58 bpw."""
    W = W.float(); cols = W.shape[1]; Q = torch.zeros_like(W); gs = groupsize or cols
    for c0 in range(0, cols, gs):
        g = W[:, c0:c0 + gs]
        s = g.abs().mean(1, keepdim=True).clamp(min=1e-8)          # abs-mean, per row-group
        Q[:, c0:c0 + g.shape[1]] = torch.clamp(torch.round(g / s), -1, 1) * s
    return Q.to(torch.float16)


def q_binary(W, groupsize):
    """True signed 1-bit {-1,+1} with per-group abs-mean scale (sign(W)*mean(|W|)).
    The hard constraint — no zeros. ~1.0 bpw + the group scales."""
    W = W.float(); cols = W.shape[1]; Q = torch.zeros_like(W); gs = groupsize or cols
    for c0 in range(0, cols, gs):
        g = W[:, c0:c0 + gs]
        s = g.abs().mean(1, keepdim=True).clamp(min=1e-8)
        Q[:, c0:c0 + g.shape[1]] = torch.sign(g) * s
    return Q.to(torch.float16)


def eff_bpw(nbits_sym, groupsize, keep):
    """Honest effective bits/weight: symbol entropy + per-group fp16 scale + fp16 outliers.
    nbits_sym = log2(#symbols): binary=1.0, ternary=log2(3)=1.585. Scale=16b/group.
    Each kept outlier ~ 16b value + ~16b index."""
    import math
    return nbits_sym + 16.0 / (groupsize or 1) + keep * 32.0


def q_residual(W, bits, groupsize, levels):
    """RVQ carousel: stack `levels` low-bit codes of the running residual."""
    Q = torch.zeros_like(W); R = W.clone().float()
    for _ in range(max(1, levels)):
        q = q_group(R, bits, groupsize); Q = Q + q; R = R - q
    return Q


def q_outlier(W, sens, bits, groupsize, keep, levels=1, dense_fn=None):
    """Keep the top `keep` fraction of weights (by sensitivity) in fp16 (the sparse
    escape hatch); quantize the dense remainder with `dense_fn` (default: absmax RTN)."""
    flat = sens.flatten(); k = max(1, int(flat.numel() * keep))
    thr = torch.kthvalue(flat, flat.numel() - k).values
    mask = sens >= thr                                   # the outliers to protect
    dense = W.clone().float(); dense[mask] = 0.0         # pull outliers out of the dense part
    if dense_fn is not None:
        Q = dense_fn(dense)
    else:
        Q = q_residual(dense, bits, groupsize, levels) if levels > 1 else q_group(dense, bits, groupsize)
    Q = Q.to(torch.float16); Q[mask] = W[mask].to(torch.float16)   # restore outliers exactly
    return Q


_ORTH = {}
def rand_orth(n, seed=0):
    """Random orthogonal matrix (incoherence rotation), cached per size."""
    if n not in _ORTH:
        g = torch.Generator().manual_seed(seed + n)
        q, r = torch.linalg.qr(torch.randn(n, n, generator=g))
        _ORTH[n] = (q * torch.sign(torch.diag(r))).float()
    return _ORTH[n]


def q_rotate(W, bits, groupsize, base_fn):
    """Rotate input columns by an orthogonal H (Gaussianize / kill outliers),
    quantize in the rotated space, un-rotate. Reduces the raw quant error."""
    H = rand_orth(W.shape[1]).to(W.device)
    Wr = W.float() @ H                                   # rotate cols
    Wrq = base_fn(Wr).float()
    return (Wrq @ H.t()).to(torch.float16)               # un-rotate back to original space


def linears(module):
    return {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)}


@torch.no_grad()
def per_channel_importance(model, calib, lins):
    """Diagonal activation importance (the imatrix signal): mean x^2 per input channel."""
    imp = {n: torch.zeros(m.in_features) for n, m in lins.items()}
    cnt = {n: 0 for n in lins}; hooks = []
    def mk(n):
        def h(mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            imp[n] += (x * x).sum(0).cpu(); cnt[n] += x.shape[0]
        return h
    for n, m in lins.items(): hooks.append(m.register_forward_hook(mk(n)))
    dev = next(model.parameters()).device
    for c in calib: model(c.unsqueeze(0).to(dev))
    for h in hooks: h.remove()
    return {n: (imp[n] / max(cnt[n], 1)) for n in lins}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=1)
    ap.add_argument("--groupsize", type=int, default=64)
    ap.add_argument("--keep", type=float, default=0.01, help="fraction kept in fp16 (outlier-catch)")
    ap.add_argument("--levels", type=int, default=3, help="residual-carousel depth")
    ap.add_argument("--nsamples", type=int, default=24)
    ap.add_argument("--eval-chunks", type=int, default=10)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib-file", required=True)
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"== pollard-lowbit :: {a.model}  bits={a.bits} keep={a.keep} levels={a.levels}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to(a.device)
    calib = _chunks(tok, open(a.calib_file, encoding="utf-8").read(), a.seqlen, a.nsamples)
    test = _chunks(tok, open(a.eval_file, encoding="utf-8").read(), a.seqlen, a.eval_chunks)

    ppl_fp16 = eval_ppl(model, test); print(f"fp16 PPL: {ppl_fp16:.4f}", flush=True)
    import copy; state = copy.deepcopy(model.state_dict())
    lins = linears(model.model.layers)
    print("collecting per-channel importance ...", flush=True)
    imp = per_channel_importance(model, calib, lins)

    def sens_of(n, m):
        # per-weight sensitivity ~ W^2 * input-channel importance (contribution to output error)
        return (m.weight.data.float() ** 2) * imp[n].to(m.weight.device)[None, :]

    def run(label, fn):
        model.load_state_dict(state); t0 = time.time()
        for n, m in lins.items():
            m.weight.data = fn(n, m).to(m.weight.device)
        p = eval_ppl(model, test)
        print(f"{label:34s} PPL: {p:10.4f}   (+{p-ppl_fp16:.4f})   [{int(time.time()-t0)}s]", flush=True)
        return p

    gs = a.groupsize; keep = a.keep
    bpw = {
        "rtn_absmax":   eff_bpw(1.585, gs, 0.0),
        "ternary":      eff_bpw(1.585, gs, 0.0),
        "ternary_out":  eff_bpw(1.585, gs, keep),
        "binary":       eff_bpw(1.0,   gs, 0.0),
        "binary_out":   eff_bpw(1.0,   gs, keep),
        "ternary_rot":  eff_bpw(1.585, gs, keep),
    }
    res = {}
    # --- the collapse baseline: absmax ternary (the KNOWN-BAD scale) ---
    res["rtn_absmax"] = run("absmax ternary (collapse baseline)",
                            lambda n, m: q_group(m.weight.data, 2, gs))
    # --- Grok lever #1: abs-MEAN ternary (BitNet b1.58) ---
    res["ternary"] = run("abs-mean ternary {-1,0,1}",
                         lambda n, m: q_ternary(m.weight.data, gs))
    # --- ternary + outlier-catch (salient weights kept fp16) ---
    res["ternary_out"] = run(f"abs-mean ternary + outlier ({keep:.1%})",
                             lambda n, m: q_outlier(m.weight.data, sens_of(n, m), 2, gs, keep,
                                                    dense_fn=lambda W: q_ternary(W, gs)))
    # --- ternary in ROTATED space + outlier (incoherence, done right: error stays small) ---
    res["ternary_rot"] = run(f"rotated abs-mean ternary + outlier ({keep:.1%})",
                             lambda n, m: q_outlier(m.weight.data, sens_of(n, m), 2, gs, keep,
                                                    dense_fn=lambda W: q_rotate(W, 2, gs,
                                                                               lambda X: q_ternary(X, gs))))
    # --- true signed 1-bit {-1,+1} (the hard constraint) ---
    res["binary"] = run("true 1-bit signed {-1,+1}",
                        lambda n, m: q_binary(m.weight.data, gs))
    res["binary_out"] = run(f"true 1-bit + outlier ({keep:.1%})",
                            lambda n, m: q_outlier(m.weight.data, sens_of(n, m), 1, gs, keep,
                                                   dense_fn=lambda W: q_binary(W, gs)))

    print("\n== verdict (fp16 = {:.3f}) ==".format(ppl_fp16), flush=True)
    order = ["rtn_absmax", "binary", "binary_out", "ternary", "ternary_out", "ternary_rot"]
    for k in order:
        tag = "<-- USABLE (<2x fp16)" if res[k] < ppl_fp16 * 2 else \
              ("<-- coherent" if res[k] < 100 else "")
        print(f"  {k:14s}: {res[k]:10.3f} PPL   @ ~{bpw[k]:.2f} bpw   {tag}", flush=True)
    print(f"\n  Grok's bar: strong 1-bit/ternary sits at PPL 6-15. fp16 here = {ppl_fp16:.2f}.", flush=True)


if __name__ == "__main__":
    main()
