#!/usr/bin/env python3
"""pollard-recovery (EXPERIMENT) -- does a LIFT-style "recovery vector" buy back quality on a low rung?

LIFT (arXiv 2609.31140) recovers reasoning in a degraded model with one constant vector per layer:
r = mean over answer tokens of (hidden state on the strong path - hidden state on the degraded path),
injected as x + mu * r at one layer. The strong/degraded pair there is two prompting regimes; here it is
f16 vs the low-bit crush of the SAME model, on ordinary calibration text -- so r is a constant the model
can carry for free: a bias on the residual writer that feeds that layer (down_proj of block L-1 puts it
straight into the residual stream, exactly where hidden_states[L] is read). llama.cpp's Qwen2 graph
already passes an attn_output bias slot (wo_b) into build_attn; the loader just never creates it, so
baking the vector into a GGUF is a one-line TENSOR_NOT_REQUIRED patch plus one extra tensor per layer.

What this measures on held-out text, all against the f16 reference:
  1. KL / top-1 of the crushed model (baseline, the rung's damage)
  2. the same model with r_L injected at each candidate layer L, and a mu sweep at the best layer
  3. the best 2-3 single layers injected together, and every block carrying its own r (sequential
     mean-shift correction), with and without the final block
Kill criterion: if nothing takes >= 10% off the baseline KL, the vector is not worth a runtime patch.

    python experiments/pollard_recovery.py --model ~/pollard/downloads/Qwen__Qwen2.5-0.5B-Instruct \
        --calib ~/pollard/downloads/pollard_calib.txt --bits 3 --device mps --out recovery.json
"""
from __future__ import annotations

import argparse, copy, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from pollard_gptq import load_backbone, get_wikitext, rtn_quantize, text_layers  # noqa: E402


def crush(model, bits, groupsize):
    """RTN every decoder Linear in place -- the same proxy pollard-probe and pollard_mattr use."""
    n = 0
    for layer in text_layers(model):
        for _, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                mod.weight.data = rtn_quantize(mod.weight.data.float(), bits, groupsize).to(mod.weight.dtype).to(mod.weight.device)
                n += 1
    return n


@torch.no_grad()
def hidden_means(model, chunks, dev):
    """Mean residual-stream vector per hidden_states index over every token of every chunk: [L+1, D]."""
    acc, n = None, 0
    for c in chunks:
        out = model(c.unsqueeze(0).to(dev), output_hidden_states=True)
        hs = torch.stack([h[0].float() for h in out.hidden_states])       # [L+1, T, D]
        s = hs.sum(1)
        acc = s if acc is None else acc + s
        n += hs.shape[1]
    return acc / n


@torch.no_grad()
def ref_logprobs(model, chunks, dev):
    return [F.log_softmax(model(c.unsqueeze(0).to(dev)).logits[0, :-1].float(), -1) for c in chunks]


@torch.no_grad()
def kl_top1(model, chunks, ref_lps, dev):
    kl_sum = t1 = n = 0.0
    for c, lpr in zip(chunks, ref_lps):
        lq = F.log_softmax(model(c.unsqueeze(0).to(dev)).logits[0, :-1].float(), -1)
        kl = (lpr.exp() * (lpr - lq)).sum(-1)
        kl_sum += kl.sum(); t1 += (lpr.argmax(-1) == lq.argmax(-1)).float().sum(); n += kl.numel()
    return float(kl_sum / n), float(100.0 * t1 / n)


def set_bias(model, block_idx, vec):
    """Carry `vec` (or None to clear) as the down_proj bias of block `block_idx`: adds to the residual
    stream right after that block, i.e. into hidden_states[block_idx + 1]."""
    lin = text_layers(model)[block_idx].mlp.down_proj
    lin.bias = None if vec is None else nn.Parameter(vec.to(lin.weight.dtype).to(lin.weight.device), requires_grad=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--groupsize", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--calib-chunks", type=int, default=16)
    ap.add_argument("--eval-chunks", type=int, default=6)
    ap.add_argument("--layers", default="", help="candidate hidden_states indices, e.g. 4,8,12,16,20 (default: every 2nd)")
    ap.add_argument("--mus", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="recovery.json")
    a = ap.parse_args()

    dev = torch.device(a.device)
    dtype = torch.float16 if dev.type != "cpu" else torch.float32
    t0 = time.time()
    ref = load_backbone(a.model, dtype=dtype, device=str(dev))
    ref.requires_grad_(False)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    calib = get_wikitext(tok, "train", a.seqlen, n=a.calib_chunks, path=a.calib)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    evalc = get_wikitext(tok, a.eval_split, a.seqlen, n=a.eval_chunks)
    nblk = len(text_layers(ref))
    print(f"loaded {time.time()-t0:.0f}s · {nblk} blocks · calib {len(calib)} · eval {len(evalc)} · {dev}")

    q = copy.deepcopy(ref)
    n = crush(q, a.bits, a.groupsize)
    print(f"crushed {n} linears to RTN {a.bits}-bit g{a.groupsize}")

    ref_eval = ref_logprobs(ref, evalc, dev)
    base_kl, base_t1 = kl_top1(q, evalc, ref_eval, dev)
    print(f"baseline  KL {base_kl:.4f}  top1 {base_t1:.1f}%")

    # the recovery vectors: strong (f16) minus degraded (crushed), per hidden_states index
    hm_ref = hidden_means(ref, calib, dev)
    hm_q = hidden_means(q, calib, dev)
    R = hm_ref - hm_q                                        # [L+1, D]; R[L] is read at hidden_states[L]
    norms = R.norm(dim=1)
    print("||r_L||: " + " ".join(f"{i}:{v:.2f}" for i, v in enumerate(norms.tolist())))

    cands = [int(x) for x in a.layers.split(",") if x] or list(range(2, nblk + 1, 2))
    cands = [L for L in cands if 1 <= L <= nblk]
    res = {"baseline": {"kl": base_kl, "top1": base_t1}, "single": {}, "mu_sweep": {}, "all_layers": {}}
    best = (base_kl, None)
    for L in cands:
        set_bias(q, L - 1, R[L])                            # block L-1 writes hidden_states[L]
        kl, t1 = kl_top1(q, evalc, ref_eval, dev)
        set_bias(q, L - 1, None)
        res["single"][L] = {"kl": kl, "top1": t1, "norm": float(norms[L])}
        mark = " <-" if kl < best[0] else ""
        print(f"  r@{L:2d}  KL {kl:.4f} ({(kl/base_kl-1)*100:+.1f}%)  top1 {t1:.1f}%{mark}")
        if kl < best[0]:
            best = (kl, L)
    if best[1] is not None:
        L = best[1]
        for mu in [float(x) for x in a.mus.split(",")]:
            set_bias(q, L - 1, mu * R[L])
            kl, t1 = kl_top1(q, evalc, ref_eval, dev)
            set_bias(q, L - 1, None)
            res["mu_sweep"][mu] = {"kl": kl, "top1": t1}
            print(f"  r@{L} mu={mu:.1f}  KL {kl:.4f} ({(kl/base_kl-1)*100:+.1f}%)  top1 {t1:.1f}%")

    # the best few single layers injected TOGETHER (independent vectors from the same R)
    ranked = sorted(res["single"], key=lambda L: res["single"][L]["kl"])
    res["joint"] = {}
    for k in (2, 3):
        Ls = ranked[:k]
        for L in Ls:
            set_bias(q, L - 1, R[L])
        kl, t1 = kl_top1(q, evalc, ref_eval, dev)
        for L in Ls:
            set_bias(q, L - 1, None)
        res["joint"][",".join(map(str, Ls))] = {"kl": kl, "top1": t1}
        print(f"  joint r@{Ls}  KL {kl:.4f} ({(kl/base_kl-1)*100:+.1f}%)  top1 {t1:.1f}%")

    # every block carries its own mean-shift correction, measured SEQUENTIALLY so each vector is the
    # residual mismatch still there after the earlier blocks were corrected. The LAST block is left
    # alone: hidden_states[n] sits before the final norm with a norm ~10x the rest, and its vector
    # alone wrecks the output (see r@n above) -- correcting it is the wrong lever.
    for last in (nblk - 1, nblk):
        for b in range(last):
            hm_q = hidden_means(q, calib, dev)
            set_bias(q, b, hm_ref[b + 1] - hm_q[b + 1])
        kl, t1 = kl_top1(q, evalc, ref_eval, dev)
        res["all_layers"][f"blocks_0_to_{last-1}"] = {"kl": kl, "top1": t1}
        print(f"all blocks 0..{last-1} (sequential mean-shift)  KL {kl:.4f} ({(kl/base_kl-1)*100:+.1f}%)  top1 {t1:.1f}%")
        for b in range(nblk):
            set_bias(q, b, None)
    kl = min(v["kl"] for v in res["all_layers"].values())
    kl = min(kl, min(v["kl"] for v in res["joint"].values()))

    res.update({"model": a.model, "bits": a.bits, "groupsize": a.groupsize, "best_single_layer": best[1],
                "seconds": round(time.time() - t0, 1)})
    json.dump(res, open(a.out, "w"), indent=1)
    verdict = ("WORTH A RUNTIME PATCH" if best[0] <= 0.9 * base_kl or kl <= 0.9 * base_kl else "NOT WORTH IT (<10% KL)")
    print(f"\n{verdict}: baseline {base_kl:.4f} -> best single {best[0]:.4f} (layer {best[1]}) -> best multi {kl:.4f}")
    print(f"wrote {a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
