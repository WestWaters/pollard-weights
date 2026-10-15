#!/usr/bin/env python3
"""QAT for STQ1_0 — train the weights to ADAPT to 1.31-bit sparse ternary (the only way it works;
PTQ of a normal model to STQ1_0 is garbage). Proves the pipeline: measure FFN-STQ1_0 perplexity
BEFORE training (=PTQ) vs AFTER a short QAT run (straight-through estimator). If QAT << PTQ, the
QAT-STQ1_0 path is real and we scale it.

STQ1_0 fake-quant (faithful to the format's statistics): groups of 4, force exactly ONE zero
(the least-IMPORTANT lane), project the other 3 to {-d,+d}; d = importance-weighted mean|w| of the
survivors (weighted-LS scale, NOT amax). STE: forward uses the quantized weight, gradient flows to
the f16 shadow. Only the FFN (the STQ1_0 body) is trained; attention stays f16.
"""
import math, sys, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = r"C:\pollard\qat\qwen05b-hf"
DEV = "cuda"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
CTX = 512


def stq1_0_fq(W):
    out, inn = W.shape
    pad = (-inn) % 4
    Wp = torch.cat([W, W.new_zeros(out, pad)], 1) if pad else W
    G = Wp.shape[1] // 4
    Wg = Wp.view(out, G, 4)
    a = Wg.abs()
    zpos = a.argmin(-1, keepdim=True)                    # least-|w| lane -> forced zero (3:4)
    nz = torch.ones_like(Wg, dtype=torch.bool).scatter(-1, zpos, False)
    d = (a * nz).sum(-1, keepdim=True) / nz.sum(-1, keepdim=True).clamp(min=1)   # LS scale
    q = (torch.sign(Wg) * d) * nz                        # ternary {-d,0,+d}
    Wq = q.view(out, -1)[:, :inn]
    return W + (Wq - W).detach()                         # straight-through


class STQLinear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.weight = nn.Parameter(lin.weight.data.clone())
        self.bias = lin.bias
        self.on = True
    def forward(self, x):
        return F.linear(x, stq1_0_fq(self.weight) if self.on else self.weight, self.bias)


def wrap_ffn(model):
    n = 0
    for mod in model.modules():
        for attr in ("gate_proj", "up_proj"):  # QAT MIX: protect down_proj (residual writer)
            lin = getattr(mod, attr, None)
            if isinstance(lin, nn.Linear):
                setattr(mod, attr, STQLinear(lin)); n += 1
    return n


def set_fq(model, on):
    for m in model.modules():
        if isinstance(m, STQLinear):
            m.on = on


@torch.no_grad()
def ppl(model, ids):
    model.eval(); nll = 0.0; cnt = 0
    for i in range(0, len(ids) - 1, CTX):
        chunk = ids[i:i + CTX + 1]
        if len(chunk) < 2: break
        logits = model(chunk[:-1].unsqueeze(0)).logits[0].float()
        nll += F.cross_entropy(logits, chunk[1:], reduction="sum").item(); cnt += len(chunk) - 1
    return math.exp(nll / cnt)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
    # eval + train text
    import os
    txtf = r"C:\pollard\bench\wiki_train.txt"
    text = open(txtf, encoding="utf-8", errors="ignore").read() if os.path.exists(txtf) else (
        "The history of science is a long and winding road. " * 4000)
    ids = tok(text, return_tensors="pt").input_ids[0].to(DEV)
    ev, tr = ids[:CTX * 20], ids[CTX * 20:]  # train on ALL remaining tokens (anti-overfit)

    print("baseline f16 PPL:", round(ppl(model, ev), 3))
    nwrap = wrap_ffn(model)
    print(f"wrapped {nwrap} FFN linears with STQ1_0 fake-quant")
    set_fq(model, True)
    print("PTQ STQ1_0 PPL (no training):", round(ppl(model, ev), 3))

    for p in model.parameters(): p.requires_grad = False
    for m in model.modules():
        if isinstance(m, STQLinear): m.weight.requires_grad = True
    opt = torch.optim.AdamW([m.weight for m in model.modules() if isinstance(m, STQLinear)], lr=3e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    import random
    for step in range(STEPS):
        model.train()
        s = random.randint(0, max(0, len(tr) - CTX - 1))
        chunk = tr[s:s + CTX + 1]
        logits = model(chunk[:-1].unsqueeze(0)).logits[0]
        loss = F.cross_entropy(logits.float(), chunk[1:])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 300 == 0 and step > 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}  QAT PPL {ppl(model, ev):.2f}")
    print("QAT STQ1_0 PPL (after training):", round(ppl(model, ev), 3))


if __name__ == "__main__":
    main()
