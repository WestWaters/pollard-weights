#!/usr/bin/env python3
"""pollard-kl — KL-to-f16 + top-1 agreement for a quantization, on the SAME eval text as
the PPL runs. The metric the 1-bit argument needs: matched-size KL says how much of the
f16 next-token distribution survived, not just whether PPL looks fine. (PPL can look ok
while the tail is peaky and generation loops — KL/top-1 catches that.)

Torch path (Mac): keep the f16 model as the reference, quantize a COPY with any method/
recipe from pollard_gptq, compare their distributions token-by-token. The 7B GGUFs use
ik_llama.cpp's own --kl-divergence workflow on the PC; this validates the metric and
gives the 0.5B/1.5B numbers now. Emits a JSON row the scorecard can consume.

Usage:
  pollard-kl --model Qwen/Qwen2.5-0.5B-Instruct --method gptq-seq-ao --recipe aggr \
      --calib-file wiki_train.txt --eval-file wikitext2_test.txt --kl-chunks 4 --device mps
"""
import argparse, copy, json, time
import torch, torch.nn as nn, torch.nn.functional as F

from pollard_gptq import _chunks, sequential_gptq, make_recipe, linear_layers


@torch.no_grad()
def kl_top1(ref, q, chunks, dev):
    """Mean KL(f16 || quant) in nats and top-1 agreement over next-token positions."""
    kl_sum = t1_sum = ntok = 0.0
    for c in chunks:
        ids = c.unsqueeze(0).to(dev)
        lr = ref(ids).logits[0, :-1].float()               # [T, V] predict-next
        lq = q(ids).logits[0, :-1].float()
        pr = F.softmax(lr, dim=-1)
        lpr = F.log_softmax(lr, dim=-1); lpq = F.log_softmax(lq, dim=-1)
        kl = (pr * (lpr - lpq)).sum(-1)                     # [T] per-token KL
        t1 = (lr.argmax(-1) == lq.argmax(-1)).float()
        kl_sum += kl.sum().item(); t1_sum += t1.sum().item(); ntok += kl.numel()
        del lr, lq, pr, lpr, lpq
    return kl_sum / ntok, 100.0 * t1_sum / ntok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", default="gptq-seq-ao")
    ap.add_argument("--recipe", default="none", choices=["none", "handmix", "aggr"])
    ap.add_argument("--ablate", default="none")
    ap.add_argument("--qmode", default="int", choices=["int", "ternary", "binary"])
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--groupsize", type=int, default=64)
    ap.add_argument("--nsamples", type=int, default=16)
    ap.add_argument("--kl-chunks", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib-file", required=True)
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    print(f"== pollard-kl :: {a.model}  recipe={a.recipe} qmode={a.qmode}  dev={dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    ref = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to(dev).eval()
    calib = _chunks(tok, open(a.calib_file, encoding="utf-8").read(), a.seqlen, a.nsamples)
    ev = _chunks(tok, open(a.eval_file, encoding="utf-8").read(), a.seqlen, a.kl_chunks)

    q = copy.deepcopy(ref)                                  # the copy we quantize
    rec = make_recipe("aggr" if a.recipe == "aggr" else "handmix", a.ablate) if a.recipe != "none" else None
    t0 = time.time()
    if a.method in ("gptq-seq", "gptq-seq-ao"):
        sequential_gptq(q, calib, dev, a.bits, a.groupsize, act_order=a.method.endswith("-ao"),
                        qmode=a.qmode, recipe=rec, nlayers=len(q.model.layers))
    else:                                                   # rtn fallback (alphabet-aware)
        from pollard_lowbit import q_ternary, q_binary
        for n, m in linear_layers(q.model.layers).items():
            if a.qmode == "ternary": m.weight.data = q_ternary(m.weight.data, a.groupsize).to(dev)
            elif a.qmode == "binary": m.weight.data = q_binary(m.weight.data, a.groupsize).to(dev)
    q.eval()

    kl, t1 = kl_top1(ref, q, ev, dev)
    tag = f"{a.recipe if a.recipe!='none' else a.qmode}"
    print(f"\n{tag}: KL(f16||q) = {kl:.4f} nats   top-1 agree = {t1:.1f}%   "
          f"[{int(time.time()-t0)}s, {a.kl_chunks} chunks]", flush=True)
    if a.out:
        json.dump({"model": a.model, "recipe": tag, "kl_nats": kl, "top1_pct": t1,
                   "kl_chunks": a.kl_chunks}, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
