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
  * spec-decode acceptance (optional, vLLM) — with --metrics <url>/metrics and --accept-gen N: generates N tokens per
                      sample and reads the delta of vLLM's spec_decode counters → accepted draft tokens per step and
                      per-position acceptance. Teacher-forced PPL never decodes, so this is the only way to see the
                      speculative head's contribution on the real stack (and it dominates single-stream speed).

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


def spec_counters(metrics_url):
    """Snapshot vLLM's speculative-decoding Prometheus counters (returns None if the server has none)."""
    import re, urllib.request
    txt = urllib.request.urlopen(metrics_url, timeout=20).read().decode()
    def val(name):
        tot = 0.0; seen = False
        for ln in txt.splitlines():
            if ln.startswith(name + "{") or ln.startswith(name + " "):
                try: tot += float(ln.rsplit(" ", 1)[1]); seen = True
                except ValueError: pass
        return tot if seen else None
    c = {"drafts": val("vllm:spec_decode_num_drafts_total"), "draft_tokens": val("vllm:spec_decode_num_draft_tokens_total"),
         "accepted": val("vllm:spec_decode_num_accepted_tokens_total"), "per_pos": {}}
    for ln in txt.splitlines():
        m = re.match(r'vllm:spec_decode_num_accepted_tokens_per_pos\{.*?position="(\d+)".*?\} ([0-9.e+]+)', ln)
        if m: c["per_pos"][int(m.group(1))] = c["per_pos"].get(int(m.group(1)), 0.0) + float(m.group(2))
    return None if c["drafts"] is None else c


def spec_acceptance(base, model, texts, metrics_url, gen_tokens, key=None):
    """Generate gen_tokens per sample (real decode) and return the acceptance derived from the counter deltas."""
    before = spec_counters(metrics_url)
    if before is None:
        return None
    for t in texts:
        completion(base, model, t[:2000], key=key, max_tokens=gen_tokens, temperature=0.0)
    after = spec_counters(metrics_url)
    d = (after["drafts"] or 0) - (before["drafts"] or 0); a = (after["accepted"] or 0) - (before["accepted"] or 0)
    dt = (after["draft_tokens"] or 0) - (before["draft_tokens"] or 0)
    if d <= 0:
        return {"drafts": 0}
    pos = {k: (after["per_pos"].get(k, 0) - before["per_pos"].get(k, 0)) / d for k in sorted(after["per_pos"])}
    return {"drafts": d, "draft_tokens": dt, "accepted": a, "accepted_per_step": a / d, "tokens_per_step": 1 + a / d,
            "draft_accept_rate": (a / dt) if dt else float("nan"), "per_position": pos}


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
    ap.add_argument("--calib", help="calibration corpus to EXCLUDE from --text (gate hygiene): any eval "
                    "line that also appears in the calib set is dropped so the score isn't inflated by overlap")
    ap.add_argument("--max-samples", type=int, default=50, help="cap on lines used (keeps it quick)")
    ap.add_argument("--stride", type=int, default=8, help="token stride for the A/B agreement probes")
    ap.add_argument("--api-key", default=None, help="bearer token if the endpoint needs one")
    ap.add_argument("--metrics", help="vLLM Prometheus endpoint of the model under test, e.g. http://host:8000/metrics — "
                    "enables the speculative-decoding acceptance read (counter deltas around real generations)")
    ap.add_argument("--accept-gen", type=int, default=128, help="tokens to generate per sample for the acceptance read")
    a = ap.parse_args()

    texts = [ln.strip() for ln in open(a.text, encoding="utf-8") if ln.strip()]
    if a.calib:                                            # gate hygiene: drop eval lines seen in calib
        import hashlib
        norm = lambda s: hashlib.sha1(" ".join(s.split()).lower().encode()).hexdigest()
        seen = {norm(ln) for ln in open(a.calib, encoding="utf-8") if ln.strip()}
        before = len(texts)
        texts = [t for t in texts if norm(t) not in seen]
        dropped = before - len(texts)
        if dropped:
            print(f"gate hygiene: dropped {dropped}/{before} eval lines that overlapped the calib set")
    texts = texts[:a.max_samples]
    if not texts:
        sys.exit("ERROR: --text file has no non-empty lines (after calib-overlap exclusion).")
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
    if a.metrics:
        tgt_base = a.cand or a.base; tgt_model = (a.cand_model or a.model) if a.cand else a.model
        try:
            acc = spec_acceptance(tgt_base, tgt_model, texts, a.metrics, a.accept_gen, key=a.api_key)
            if acc is None:
                print("spec-decode: no spec_decode counters at --metrics (server runs without a drafter?)")
            elif not acc["drafts"]:
                print("spec-decode: counters did not move — is the drafter enabled on this model?")
            else:
                pp = " ".join(f"p{k}={v:.3f}" for k, v in acc["per_position"].items())
                print(f"spec-decode [{tgt_model}]: {acc['drafts']} drafts, accepted {acc['accepted_per_step']:.3f} draft tokens/step "
                      f"(=> {acc['tokens_per_step']:.2f} tokens/step), draft accept rate {acc['draft_accept_rate']*100:.1f}%  {pp}")
        except Exception as e:  # noqa: BLE001 — report, don't abort the other metrics
            print(f"(spec-decode acceptance skipped: {e})")
        print("\nverdict: >99% top-1 agreement and KL < ~0.05 means the quant behaves like the original;"
              "\n         a big PPL gap with high agreement usually means a KV-quant or a kernel path, not"
              " the weights.")


if __name__ == "__main__":
    main()
