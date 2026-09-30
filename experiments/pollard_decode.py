#!/usr/bin/env python3
"""pollard-decode (EXPERIMENT) -- does composed decoding (CompoSimplex) buy back accuracy on a LOW RUNG?

"Composable Decoding on the Probability Simplex" (arXiv 2609.34992) writes every sampler as
    q* = argmax_q <q, s> - lambda * sum_i alpha_i Omega_i(q)   over a support C_t
and shows that KL-to-base + coverage (their Best-of-K) beats greedy / top-p / min-p by up to +10.6pp
pass@1 on f16 models. A crushed rung has flatter, noisier logits, so the question for Pollard is not
"does it work" but "does it work MORE on a low rung than on f16" -- if the gap it closes is larger at
3 bits than at 16, it is a free quality lever to ship with every low rung (a sampler, no retraining),
and worth porting into llama.cpp's sampler chain. If the gain is the same or smaller, it is a generic
sampler and not ours to carry.

Same body twice: f16, then RTN-crushed in place (the proxy pollard-probe / pollard_mattr use).
GSM8K test subset, K samples per question in lockstep (the paper's own loop), pass@1 = mean over
samples, pass@K = any. Decoders: greedy, top-p, min-p, KL+coverage, KL+diversity.

    python experiments/pollard_decode.py --model ~/pollard/downloads/Qwen__Qwen2.5-0.5B-Instruct \
        --n 40 --k 4 --bits 3 --device mps --composimplex <clone dir> --out decode.json
"""
from __future__ import annotations

import argparse, copy, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import torch
import torch.nn as nn

from pollard_gptq import load_backbone, rtn_quantize, text_layers  # noqa: E402

DECODERS = {
    "greedy":      {"temperature": 1.0, "support": {"type": "full"}, "chooser": {"type": "argmax"},
                    "lambda": 1.0, "regularizers": []},
    # plain sampling is solver "softmax" on the support: no regularizer at all would mean argmax <q,s> = one-hot
    "top_p":       {"temperature": 0.7, "support": {"type": "topp", "value": 0.95}, "solver": "softmax", "regularizers": []},
    "min_p":       {"temperature": 0.7, "support": {"type": "minp", "value": 0.1}, "solver": "softmax", "regularizers": []},
    "kl+coverage": {"temperature": 0.5, "support": {"type": "topm", "value": 200},
                    "base_distribution": {"type": "softmax", "temperature": 1.0}, "lambda": 1.0, "solver": "auto",
                    "optimizer": {"name": "mirror_ascent", "steps": 10, "lr": 0.1, "tol": 0.0},
                    "regularizers": [{"type": "kl_to_base", "alpha": 0.5}, {"type": "coverage", "alpha": 0.5, "K": 4}]},
    "kl+diversity": {"temperature": 0.5, "support": {"type": "topm", "value": 200},
                     "base_distribution": {"type": "softmax", "temperature": 1.0}, "lambda": 1.0, "solver": "auto",
                     "optimizer": {"name": "mirror_ascent", "steps": 10, "lr": 0.1, "tol": 0.0},
                     "regularizers": [{"type": "kl_to_base", "alpha": 0.5},
                                      {"type": "diversity_gap", "alpha": 0.5, "K": 4, "gap_tau": 1.0}]},
}


def crush(model, bits, groupsize):
    n = 0
    for layer in text_layers(model):
        for _, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                mod.weight.data = rtn_quantize(mod.weight.data.float(), bits, groupsize).to(mod.weight.dtype).to(mod.weight.device)
                n += 1
    return n


def gold_of(answer_field: str) -> str:
    return answer_field.split("####")[-1].strip().replace(",", "")


def norm_num(s: str) -> str:
    s = (s or "").strip().replace(",", "").replace("$", "").rstrip(".")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return s.lower()
    v = m[-1]
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--composimplex", required=True, help="clone dir (its benchmark/ loop and grader are used)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--groupsize", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--decoders", default=",".join(DECODERS))
    ap.add_argument("--states", default="f16,crushed", help="f16,crushed (default) or crushed alone when f16 is already measured")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="decode.json")
    a = ap.parse_args()

    sys.path.insert(0, a.composimplex)
    from composimplex.config import normalize_sampler_config
    from composimplex.sampler import CompoSimplexSampler
    from benchmark.run import generate_samples, resolve_eos_token_ids
    from benchmark.grader import extract_math_answer

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(a.n))
    items = [(r["question"], gold_of(r["answer"])) for r in ds]

    dev = torch.device(a.device)
    dtype = torch.float16 if dev.type != "cpu" else torch.float32
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_backbone(a.model, dtype=dtype, device=str(dev))
    model.requires_grad_(False)
    eos_id, eos_ids = resolve_eos_token_ids(model, tok)
    names = [d for d in a.decoders.split(",") if d in DECODERS]
    results = {"model": a.model, "n": a.n, "k": a.k, "bits": a.bits, "states": {}}
    t0 = time.time()

    states = [("f16" if x == "f16" else f"rtn{a.bits}") for x in a.states.split(",") if x]
    for state in states:
        if state != "f16":
            n = crush(model, a.bits, a.groupsize)
            print(f"crushed {n} linears to RTN {a.bits}-bit g{a.groupsize}")
        results["states"][state] = {}
        for name in names:
            sampler = CompoSimplexSampler(normalize_sampler_config(dict(DECODERS[name], max_new_tokens=a.max_new)))
            p1 = pk = toks = 0.0
            ts = time.time()
            for qi, (q, gold) in enumerate(items):
                prompt = tok.apply_chat_template(
                    [{"role": "user", "content": q + "\nSolve step by step, then give the final answer as \\boxed{number}."}],
                    tokenize=False, add_generation_prompt=True)
                ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
                k = 1 if name == "greedy" else a.k
                gens = generate_samples(model=model, tokenizer=tok, input_ids=ids, sampler=sampler,
                                        max_new_tokens=a.max_new, num_samples=k, seed=a.seed + 1000 * qi,
                                        eos_token_id=eos_id, eos_token_ids=eos_ids)
                oks = []
                for g in gens:
                    text = tok.decode(g.token_ids, skip_special_tokens=True)
                    oks.append(norm_num(extract_math_answer(text)) == norm_num(gold))
                    toks += len(g.token_ids)
                p1 += sum(oks) / len(oks); pk += float(any(oks))
            row = {"pass1": p1 / a.n, "passk": pk / a.n, "tokens_per_sample": toks / (a.n * (1 if name == "greedy" else a.k)),
                   "seconds": round(time.time() - ts)}
            results["states"][state][name] = row
            print(f"  {state:5s} {name:13s} pass@1 {row['pass1']:.3f}  pass@{a.k} {row['passk']:.3f}  "
                  f"{row['tokens_per_sample']:.0f} tok/sample  {row['seconds']}s", flush=True)
            json.dump(results, open(a.out, "w"), indent=1)
    results["seconds"] = round(time.time() - t0)
    json.dump(results, open(a.out, "w"), indent=1)
    # the number that matters: how much of the crush's loss each decoder buys back vs greedy
    if "f16" in results["states"] and f"rtn{a.bits}" in results["states"] and "greedy" in names:
        f, r = results["states"]["f16"], results["states"][f"rtn{a.bits}"]
        print("\ndecoder gain over greedy (pass@1):  f16 -> crushed")
        for name in names:
            print(f"  {name:13s} {f[name]['pass1']-f['greedy']['pass1']:+.3f} -> {r[name]['pass1']-r['greedy']['pass1']:+.3f}")
    print(f"wrote {a.out}  ({results['seconds']}s)")


if __name__ == "__main__":
    main()
