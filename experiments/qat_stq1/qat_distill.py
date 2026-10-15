#!/usr/bin/env python3
"""QAT-STQ1_0 with DISTILLATION — the gold-card lever. Frozen f16 TEACHER + STQ1_0 STUDENT
(gate/up STQ1_0, protect down). Loss = KL(student||teacher) + small LM CE. Faithful stride-16
fake-quant (matches the real encoder), so the trained weights export to a real STQ1_0 GGUF.
Adafactor (low VRAM) + gradient checkpointing so it shares the GPU nicely.

  qat_distill.py <steps>
Saves folded student to <MODEL>-qatd; then convert->llama-quantize STQ1_0 boards the REAL GGUF.
"""
import math, sys, os, random, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor

MODEL = r"C:\pollard\qat\qwen15b-hf"
OUT = MODEL + "-qatd"
DEV = "cuda"; CTX = 512; QK = 256
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500

_IDX = None
def _gidx(dev):
    global _IDX
    if _IDX is None or _IDX.device != dev:
        idx = torch.empty(64, 4, dtype=torch.long)
        for g in range(64):
            c, gl = g // 16, g % 16
            for p in range(4): idx[g, p] = c * 64 + gl + p * 16
        _IDX = idx.reshape(-1).to(dev)
    return _IDX

def stq1_0_fq(W):
    out, inn = W.shape; nb = inn // QK; idx = _gidx(W.device)
    Wg = W[:, :nb * QK].view(out, nb, QK)[:, :, idx].view(out, nb, 64, 4)
    a = Wg.abs(); zpos = a.argmin(-1, keepdim=True)
    nz = torch.ones_like(Wg, dtype=torch.bool).scatter(-1, zpos, False)
    d = (a * nz).sum((2, 3), keepdim=True) / nz.sum((2, 3), keepdim=True).clamp(min=1)
    q = (torch.sign(Wg) * d) * nz
    outb = torch.empty(out, nb, QK, device=W.device, dtype=W.dtype)
    outb[:, :, idx] = q.reshape(out, nb, QK)
    Wq = outb.reshape(out, nb * QK)
    if inn > nb * QK: Wq = torch.cat([Wq, W[:, nb * QK:]], 1)
    return W + (Wq - W).detach()

class STQLinear(nn.Module):
    def __init__(self, lin):
        super().__init__(); self.weight = nn.Parameter(lin.weight.data.clone()); self.bias = lin.bias; self.on = True
    def forward(self, x): return F.linear(x, stq1_0_fq(self.weight) if self.on else self.weight, self.bias)
    def fold(self):
        lin = nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=self.bias is not None)
        lin.weight.data = self.weight.data.clone()
        if self.bias is not None: lin.bias.data = self.bias.data.clone()
        return lin

def wrap(m):
    n = 0
    for mod in m.modules():
        for a in ("gate_proj", "up_proj"):
            lin = getattr(mod, a, None)
            if isinstance(lin, nn.Linear): setattr(mod, a, STQLinear(lin)); n += 1
    return n

def fold_back(m):
    for mod in m.modules():
        for a in ("gate_proj", "up_proj"):
            x = getattr(mod, a, None)
            if isinstance(x, STQLinear): setattr(mod, a, x.fold())

@torch.no_grad()
def ppl(m, ids):
    m.eval(); nll = 0.0; cnt = 0
    for i in range(0, len(ids) - 1, CTX):
        c = ids[i:i + CTX + 1]
        if len(c) < 2: break
        lg = m(c[:-1].unsqueeze(0)).logits[0].float()
        nll += F.cross_entropy(lg, c[1:], reduction="sum").item(); cnt += len(c) - 1
    return math.exp(nll / cnt)

def mk(dtype=torch.bfloat16):
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(DEV)
    m.config.use_cache = False; m.gradient_checkpointing_enable(); return m

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = open(r"C:\pollard\bench\wiki_train.txt", encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt").input_ids[0].to(DEV)
    ev, tr = ids[:CTX * 20], ids[CTX * 20:]

    teacher = mk(); teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    student = mk()
    print("baseline f16 PPL:", round(ppl(teacher, ev), 3))
    print("wrapped", wrap(student), "gate/up (down protected)")
    print("PTQ STQ1_0 PPL:", round(ppl(student, ev), 3))
    for p in student.parameters(): p.requires_grad = False
    for m in student.modules():
        if isinstance(m, STQLinear): m.weight.requires_grad = True
    opt = Adafactor([m.weight for m in student.modules() if isinstance(m, STQLinear)],
                    lr=1e-4, scale_parameter=False, relative_step=False, warmup_init=False)
    for step in range(STEPS):
        student.train()
        s = random.randint(0, max(0, len(tr) - CTX - 1)); c = tr[s:s + CTX + 1]
        x = c[:-1].unsqueeze(0)
        with torch.no_grad():
            t = teacher(x).logits[0].float()
        sl = student(x).logits[0].float()
        distill = F.kl_div(F.log_softmax(sl, -1), F.softmax(t, -1), reduction="batchmean")
        lm = F.cross_entropy(sl, c[1:])
        loss = 0.9 * distill + 0.1 * lm
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0 and step > 0:
            print(f"  step {step:4d}  KL {distill.item():.4f}  LM {lm.item():.3f}  PPL {ppl(student, ev):.2f}")
    print("QAT+distill STQ1_0 PPL (fake-quant):", round(ppl(student, ev), 3))
    fold_back(student); student.save_pretrained(OUT); tok.save_pretrained(OUT)
    print("saved ->", OUT)

if __name__ == "__main__":
    main()
