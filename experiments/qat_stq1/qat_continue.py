#!/usr/bin/env python3
"""Session-2 probe #1: MORE DISTILLATION (warm-start). Continue from the already-adapted -qatd student
(not from scratch), teacher = ORIGINAL instruct f16, wiki+chat mix. The fake-quant wiki-PPL curve was
still DESCENDING at 3000 steps (46->34), so push further and watch for plateau. Goal: drive real-GGUF
PPL from 2.12x f16 toward the <=1.5x f16 gold tier. Saves -> <ORIG>-qatd2."""
import math, sys, os, random, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor

ORIG = r"C:\pollard\qat\qwen15b-hf"          # teacher (original instruct)
START = ORIG + "-qatd"                          # warm-start student (already adapted)
OUT = ORIG + "-qatd2"
DEV = "cuda"; CTX = 512; QK = 256
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
CHAT_FRAC = 0.5

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

def mk(path, dtype=torch.bfloat16):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(DEV)
    m.config.use_cache = False; m.gradient_checkpointing_enable(); return m

def load_ids(tok, path):
    return tok(open(path, encoding="utf-8", errors="ignore").read(), return_tensors="pt").input_ids[0].to(DEV)

def win(stream, n):
    s = random.randint(0, max(0, len(stream) - n - 1)); return stream[s:s + n + 1]

def main():
    tok = AutoTokenizer.from_pretrained(ORIG)
    wiki = load_ids(tok, r"C:\pollard\bench\wiki_train.txt")
    chat = load_ids(tok, r"C:\pollard\bench\chat_train.txt")
    ev, wtr = wiki[:CTX * 20], wiki[CTX * 20:]
    teacher = mk(ORIG); teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    student = mk(START)
    print("warm-start student wiki-PPL:", round(ppl(student, ev), 3))
    print("wrapped", wrap(student), "gate/up (down protected)")
    print("re-wrapped student STQ1_0 wiki-PPL:", round(ppl(student, ev), 3))
    for p in student.parameters(): p.requires_grad = False
    for m in student.modules():
        if isinstance(m, STQLinear): m.weight.requires_grad = True
    opt = Adafactor([m.weight for m in student.modules() if isinstance(m, STQLinear)],
                    lr=1e-4, scale_parameter=False, relative_step=False, warmup_init=False)
    best = 1e9
    for step in range(STEPS):
        student.train()
        c = win(chat, CTX) if random.random() < CHAT_FRAC else win(wtr, CTX)
        x = c[:-1].unsqueeze(0)
        with torch.no_grad(): t = teacher(x).logits[0].float()
        sl = student(x).logits[0].float()
        distill = F.kl_div(F.log_softmax(sl, -1), F.softmax(t, -1), reduction="batchmean")
        loss = 0.9 * distill + 0.1 * F.cross_entropy(sl, c[1:])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 400 == 0 and step > 0:
            p = ppl(student, ev); best = min(best, p)
            print(f"  step {step:4d}  KL {distill.item():.4f}  wiki-PPL {p:.2f}  best {best:.2f}", flush=True)
    print("FINAL continuation wiki-PPL (fake-quant):", round(ppl(student, ev), 3))
    fold_back(student); student.save_pretrained(OUT); tok.save_pretrained(OUT)
    print("saved ->", OUT)

if __name__ == "__main__":
    main()
