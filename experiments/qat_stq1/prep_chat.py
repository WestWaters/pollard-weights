#!/usr/bin/env python3
"""Build a ChatML-formatted instruct corpus for QAT/distill so chat behavior is PRESERVED.
Prefers a real instruct dataset (alpaca); falls back to teacher-generated on-policy data
(the Qwen instruct teacher answers a prompt bank) if HF is unreachable. Writes chat_train.txt
already wrapped in the model's OWN chat template (apply_chat_template)."""
import os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = r"C:\pollard\qat\qwen15b-hf"
OUT = r"C:\pollard\bench\chat_train.txt"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

tok = AutoTokenizer.from_pretrained(MODEL)

def fmt(user, asst):
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": asst}]
    return tok.apply_chat_template(msgs, tokenize=False)

texts = []
try:
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    for ex in ds:
        if len(texts) >= N: break
        u = ex["instruction"] + (("\n" + ex["input"]) if ex["input"] else "")
        a = ex["output"]
        if u.strip() and a.strip(): texts.append(fmt(u, a))
    print(f"REAL alpaca examples: {len(texts)}")
except Exception as e:
    print("HF datasets unreachable -> teacher-generated on-policy:", repr(e)[:120])
    prompts = ["What is the capital of France?", "Explain photosynthesis simply.",
               "Write a haiku about the ocean.", "What is 17 times 24?",
               "Give three tips for better sleep.", "Summarize the water cycle.",
               "What causes seasons on Earth?", "Define recursion in one sentence.",
               "List four primary colors.", "How do vaccines work?",
               "Translate 'good morning' to Spanish.", "What is the boiling point of water?",
               "Name three planets in our solar system.", "Explain gravity to a child.",
               "What is the largest ocean?"]
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    i = 0
    while len(texts) < N:
        p = prompts[i % len(prompts)]; i += 1
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9)
        a = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        texts.append(fmt(p, a))
    print(f"teacher-generated examples: {len(texts)}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(texts))
print("wrote", OUT, os.path.getsize(OUT), "bytes")
