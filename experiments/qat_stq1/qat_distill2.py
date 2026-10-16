#!/usr/bin/env python3
"""QAT-STQ1_0 + DISTILL on a WIKITEXT+CHAT mix. The prior run trained on raw wikitext only,
which preserved wiki-PPL but DESTROYED chat (the QAT'd f16 itself couldn't answer 'Paris').
Fix: distill the STQ1_0 student against the frozen ORIGINAL instruct teacher over a 50/50 mix
of wikitext windows and ChatML instruct windows, so chat behavior is retained while gate/up
adapt to the 1.31-bit format. Faithful stride-16 fake-quant -> exports to a REAL STQ1_0 GGUF.

  qat_distill2.py <steps>
Saves folded student -> <MODEL>-qatd (convert -> llama-quantize STQ1_0 -> board next)."""
import math, sys, os, random, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor

MODEL = r"C:\pollard\qat\qwen15b-hf"
OUT = MODEL + "-qatd"
DEV = "cuda"; CTX = 512; QK = 256
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
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

def mk(dtype=torch.bfloat16):
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(DEV)
    m.config.use_cache = False; m.gradient_checkpointing_enable(); return m

def load_ids(tok, path, fallback=None):
    if os.path.exists(path):
        text = open(path, encoding="utf-8", errors="ignore").read()
        return tok(text, return_tensors="pt").input_ids[0].to(DEV)
    return fallback

def win(stream, n):
    s = random.randint(0, max(0, len(stream) - n - 1))
    return stream[s:s + n + 1]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    wiki = load_ids(tok, r"C:\pollard\bench\wiki_train.txt")
    chat = load_ids(tok, r"C:\pollard\bench\chat_train.txt")
    assert wiki is not None and chat is not None, "need wiki_train.txt AND chat_train.txt"
    ev, wtr = wiki[:CTX * 20], wiki[CTX * 20:]
    print(f"wiki train toks {len(wtr)}  chat toks {len(chat)}  chat_frac {CHAT_FRAC}")

    teacher = mk(); teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    student = mk()
    print("baseline f16 wiki-PPL:", round(ppl(teacher, ev), 3))
    print("wrapped", wrap(student), "gate/up (down protected)")
    print("PTQ STQ1_0 wiki-PPL:", round(ppl(student, ev), 3))
    for p in student.parameters(): p.requires_grad = False
    for m in student.modules():
        if isinstance(m, STQLinear): m.weight.requires_grad = True
    opt = Adafactor([m.weight for m in student.modules() if isinstance(m, STQLinear)],
                    lr=1e-4, scale_parameter=False, relative_step=False, warmup_init=False)
    for step in range(STEPS):
        student.train()
        c = win(chat, CTX) if random.random() < CHAT_FRAC else win(wtr, CTX)
        x = c[:-1].unsqueeze(0)
        with torch.no_grad():
            t = teacher(x).logits[0].float()
        sl = student(x).logits[0].float()
        distill = F.kl_div(F.log_softmax(sl, -1), F.softmax(t, -1), reduction="batchmean")
        lm = F.cross_entropy(sl, c[1:])
        loss = 0.9 * distill + 0.1 * lm
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0 and step > 0:
            print(f"  step {step:4d}  KL {distill.item():.4f}  LM {lm.item():.3f}  wiki-PPL {ppl(student, ev):.2f}")
    print("QAT+distill(mix) STQ1_0 wiki-PPL (fake-quant):", round(ppl(student, ev), 3))
    fold_back(student); student.save_pretrained(OUT); tok.save_pretrained(OUT)
    print("saved ->", OUT)

if __name__ == "__main__":
    main()
