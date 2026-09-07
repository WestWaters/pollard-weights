#!/usr/bin/env python3
"""pollard-serve-eval — A/B a QUANTIZED served model against its baseline, on the real serving stack.

The offline PPL you measured while quantizing is not the number that ships — the served model runs a
different kernel path (vLLM/SGLang paged attention, fused Marlin, a KV quant), so the honest check is
on the endpoint. This talks to any OpenAI-compatible server (vLLM `vllm serve`, SGLang, TGI-compat)
over plain HTTP — no torch, no transformers, stdlib only — and reports:

  * perplexity      — teacher-forced NLL over a text corpus, via echo+prompt_logprobs
  * top-1 agreement — how often the quantized model's greedy next token matches the baseline's
                      (the metric that actually predicts "does it still behave like the original")
  * KL (optional)   — mean KL(baseline || quantized) over next-token logprobs, if both serve logprobs

  # one endpoint, absolute perplexity:
  pollard-serve-eval --base http://localhost:8000/v1 --model my-int4 --text calib.txt

  # A/B: quantized vs the fp16 baseline on another port (agreement + KL):
  pollard-serve-eval --base http://localhost:8000/v1 --model fp16 \\
                     --cand http://localhost:8001/v1 --cand-model int4 --text held_out.txt

Start the servers first, e.g.:  vllm serve <path> --port 8000   (add --quantization gptq for the int4).
Runs anywhere with Python; the servers are where the GPUs are.
"""
import argparse
import json
import math
import sys
import urllib.request


def _post(url, payload, timeout=120, key=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def completion(base, model, prompt, key=None, echo=False, max_tokens=0, logprobs=0, temperature=0.0):
    """One /completions call. echo+max_tokens=0 returns the prompt's own token logprobs (teacher forcing)."""
    payload = {"model": model, "prompt": prompt, "temperature": temperature,
               "max_tokens": max_tokens, "echo": echo}
    if logprobs:
        payload["logprobs"] = logprobs
    return _post(base.rstrip("/") + "/completions", payload, key=key)


def corpus_ppl(base, model, texts, key=None):
    """Teacher-forced perplexity: sum the per-token logprobs the server returns for each text
    (echo=True, max_tokens=0) and exponentiate the mean NLL. Skips the first token (no context)."""
    total_nll, total_tok = 0.0, 0
    for t in texts:
        r = completion(base, model, t, key=key, echo=True, max_tokens=0, logprobs=1)
        lp = r["choices"][0].get("logprobs") or {}
        toklp = lp.get("token_logprobs") or []
        vals = [x for x in toklp if isinstance(x, (int, float))]  # first token's logprob is null
        total_nll += -sum(vals); total_tok += len(vals)
    if not total_tok:
        return float("nan"), 0
    return math.exp(total_nll / total_tok), total_tok


def _greedy_next(base, model, prompt, key=None):
    """The server's greedy next token + its top logprobs dict (for KL)."""
    r = completion(base, model, prompt, key=key, echo=False, max_tokens=1, logprobs=20, temperature=0.0)
    ch = r["choices"][0]
    lp = ch.get("logprobs") or {}
    tokens = lp.get("tokens") or [ch.get("text", "")]
    top = (lp.get("top_logprobs") or [{}])
    return (tokens[0] if tokens else ch.get("text", "")), (top[0] if top else {})


def ab_agreement(base, bmodel, cand, cmodel, texts, key=None, stride=8):
    """Top-1 agreement + optional KL between baseline and candidate, at each position (strided)."""
    match = total = 0
    kl_sum = 0.0
    kl_n = 0
    for t in texts:
        words = t.split()
        for i in range(4, len(words), stride):                 # a few real prefixes per text
            prefix = " ".join(words[:i])
            bt, btop = _greedy_next(base, bmodel, prefix, key=key)
            ct, ctop = _greedy_next(cand, cmodel, prefix, key=key)
            total += 1
            match += int(bt == ct)
            if btop and ctop:                                  # KL(base || cand) over the shared support
                for tokn, blp in btop.items():
                    if tokn in ctop:
                        p = math.exp(blp)
                        kl_sum += p * (blp - ctop[tokn]); kl_n += 1
    agree = (match / total) if total else float("nan")
    kl = (kl_sum / kl_n) if kl_n else None
    return agree, total, kl


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--base", required=True, help="baseline endpoint, e.g. http://localhost:8000/v1")
    ap.add_argument("--model", required=True, help="baseline model name as the server advertises it")
    ap.add_argument("--cand", help="candidate (quantized) endpoint; omit for a single-model PPL run")
    ap.add_argument("--cand-model", help="candidate model name (defaults to --model)")
    ap.add_argument("--text", required=True, help="eval corpus: a text file, one sample per line")
    ap.add_argument("--max-samples", type=int, default=50, help="cap on lines used (keeps it quick)")
    ap.add_argument("--stride", type=int, default=8, help="token stride for the A/B agreement probes")
    ap.add_argument("--api-key", default=None, help="bearer token if the endpoint needs one")
    a = ap.parse_args()

    texts = [ln.strip() for ln in open(a.text, encoding="utf-8") if ln.strip()][:a.max_samples]
    if not texts:
        sys.exit("ERROR: --text file has no non-empty lines.")
    print(f"== pollard-serve-eval :: {len(texts)} samples")

    try:
        ppl, ntok = corpus_ppl(a.base, a.model, texts, key=a.api_key)
        print(f"baseline  [{a.model}]  perplexity {ppl:.4f}  ({ntok} tokens)")
    except Exception as e:
        sys.exit(f"ERROR talking to baseline endpoint {a.base}: {e}\n"
                 "Is the server up and does it serve echo+logprobs? (vLLM: --max-logprobs >0)")

    if a.cand:
        cmodel = a.cand_model or a.model
        try:
            cppl, _ = corpus_ppl(a.cand, cmodel, texts, key=a.api_key)
            print(f"candidate [{cmodel}]  perplexity {cppl:.4f}   (Δ {cppl - ppl:+.4f} vs baseline)")
        except Exception as e:
            print(f"(candidate PPL skipped: {e})")
        agree, n, kl = ab_agreement(a.base, a.model, a.cand, cmodel, texts, key=a.api_key, stride=a.stride)
        print(f"top-1 agreement: {agree*100:.2f}%  over {n} positions")
        if kl is not None:
            print(f"mean KL(base||cand): {kl:.4f} nats  (lower = closer to the original distribution)")
        print("\nverdict: >99% top-1 agreement and KL < ~0.05 means the quant behaves like the original;"
              "\n         a big PPL gap with high agreement usually means a KV-quant or a kernel path, not"
              " the weights.")


if __name__ == "__main__":
    main()
