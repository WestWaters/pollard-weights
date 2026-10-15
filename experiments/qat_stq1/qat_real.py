#!/usr/bin/env python3
"""QAT-STQ1_0, faithful to the REAL encoder, with real export. The fake-quant EXACTLY mirrors
our patched ggml `quantize_row_stq1_0_imatrix` (uniform-importance path): stride-16 groups of 4
within 64-chunks over 256-blocks, per-BLOCK scale = mean|x| over the surviving lanes, per-group
zero = argmin|x|. So the trained weights deploy to a REAL STQ1_0 GGUF with matching quality.

Usage:
  qat_real.py train  <steps>   # QAT (gate/up STQ1_0, protect down), save folded HF to <MODEL>-qat
  qat_real.py ptqppl           # PTQ fake-quant PPL (sanity)
Real GGUF board is a SEPARATE step (convert_hf_to_gguf + llama-quantize STQ1_0) — the number that counts.
"""
import math, sys, os, random, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = r"C:\pollard\qat\qwen15b-hf"
OUT = MODEL + "-qat"
DEV = "cuda"
CTX = 512
QK = 256

_IDX = None
def _gidx(device):
    global _IDX
    if _IDX is None or _IDX.device != device:
        idx = torch.empty(64, 4, dtype=torch.long)
        for g in range(64):
            c, gl = g // 16, g % 16
            for p in range(4):
                idx[g, p] = c * 64 + gl + p * 16
        _IDX = idx.reshape(-1).to(device)          # [256] block-position permutation
    return _IDX

def stq1_0_fq(W):
    out, inn = W.shape
    nb = inn // QK
    idx = _gidx(W.device)
    body = W[:, :nb * QK]
    Wg = body.view(out, nb, QK)[:, :, idx].view(out, nb, 64, 4)     # gather into stride-16 groups
    a = Wg.abs()
    zpos = a.argmin(-1, keepdim=True)                              # per-group forced zero
    nz = torch.ones_like(Wg, dtype=torch.bool).scatter(-1, zpos, False)
    d = (a * nz).sum((2, 3), keepdim=True) / nz.sum((2, 3), keepdim=True).clamp(min=1)  # per-BLOCK scale
    q = (torch.sign(Wg) * d) * nz
    outb = torch.empty(out, nb, QK, device=W.device, dtype=W.dtype)
    outb[:, :, idx] = q.reshape(out, nb, QK)                       # scatter back
    Wq = outb.reshape(out, nb * QK)
    if inn > nb * QK:
        Wq = torch.cat([Wq, W[:, nb * QK:]], 1)
    return W + (Wq - W).detach()

class STQLinear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.weight = nn.Parameter(lin.weight.data.clone()); self.bias = lin.bias; self.on = True
    def forward(self, x):
        return F.linear(x, stq1_0_fq(self.weight) if self.on else self.weight, self.bias)
    def fold(self):
        lin = nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=self.bias is not None)
        lin.weight.data = self.weight.data.clone()
        if self.bias is not None: lin.bias.data = self.bias.data.clone()
        return lin

def wrap(model):                                                  # MIX: gate/up only, protect down
    n = 0
    for mod in model.modules():
        for attr in ("gate_proj", "up_proj"):
            lin = getattr(mod, attr, None)
            if isinstance(lin, nn.Linear):
                setattr(mod, attr, STQLinear(lin)); n += 1
    return n

def fold_back(model):
    for mod in model.modules():
        for attr in ("gate_proj", "up_proj"):
            m = getattr(mod, attr, None)
            if isinstance(m, STQLinear):
                setattr(mod, attr, m.fold())

@torch.no_grad()
def ppl(model, ids):
    model.eval(); nll = 0.0; cnt = 0
    for i in range(0, len(ids) - 1, CTX):
        c = ids[i:i + CTX + 1]
        if len(c) < 2: break
        lg = model(c[:-1].unsqueeze(0)).logits[0].float()
        nll += F.cross_entropy(lg, c[1:], reduction="sum").item(); cnt += len(c) - 1
    return math.exp(nll / cnt)

def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    txtf = r"C:\pollard\bench\wiki_train.txt"
    text = open(txtf, encoding="utf-8", errors="ignore").read() if os.path.exists(txtf) else "science. " * 40000
    ids = tok(text, return_tensors="pt").input_ids[0].to(DEV)
    return tok, model, ids

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    tok, model, ids = load()
    ev, tr = ids[:CTX * 20], ids[CTX * 20:]
    print("baseline f16 PPL:", round(ppl(model, ev), 3))
    print("wrapped", wrap(model), "gate/up linears (down protected)")
    print("PTQ STQ1_0 (real-encoder-faithful) PPL:", round(ppl(model, ev), 3))
    if cmd == "ptqppl":
        return
    for p in model.parameters(): p.requires_grad = False
    for m in model.modules():
        if isinstance(m, STQLinear): m.weight.requires_grad = True
    opt = torch.optim.AdamW([m.weight for m in model.modules() if isinstance(m, STQLinear)], lr=3e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for step in range(steps):
        model.train()
        s = random.randint(0, max(0, len(tr) - CTX - 1))
        c = tr[s:s + CTX + 1]
        loss = F.cross_entropy(model(c[:-1].unsqueeze(0)).logits[0].float(), c[1:])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % 300 == 0 and step > 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}  QAT PPL {ppl(model, ev):.2f}")
    print("QAT STQ1_0 PPL (fake-quant):", round(ppl(model, ev), 3))
    fold_back(model)
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print("saved folded QAT model ->", OUT, "(convert->GGUF->llama-quantize STQ1_0 next)")

if __name__ == "__main__":
    main()
