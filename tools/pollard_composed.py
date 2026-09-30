#!/usr/bin/env python3
"""pollard-composed -- composed decoding on the probability simplex, as a llama.cpp sampler.

"Composable Decoding on the Probability Simplex" (Khamis et al. 2026, arXiv 2609.34992) writes every
sampler as one optimisation over the next-token distribution:

    q* = argmax_q  <q, s>  -  lambda * sum_i alpha_i * Omega_i(q)      q on the simplex over a support

s = logits / T over whatever support the earlier samplers left (top-k / top-p / min-p), p = the base
distribution softmax(logits / base_temp) over that support, Omega_i from {kl_to_base, js_to_base,
entropy, coverage, diversity_gap}. Their Best-of-K decoder is kl:0.5 + coverage:0.5. On f16 models
it lifts pass@1 by up to +10.6pp and multi-sample pass@K well beyond top-p; Pollard's own measurement on
a 0.5B (experiments/pollard_decode.py) saw the f16 gain and a smaller one on a crushed rung, and the
7B+ rungs are where the question is still open -- so the sampler ships in the runtime for everyone to
use on any rung, and the measurements follow.

The runtime side is runtime-patches/llama-composed-sampler.patch (common/composed-sampler.cpp): a
sampler named `composed` placed after `temperature`, configured by `--composed SPEC` on any llama.cpp
binary or `"composed": "SPEC"` / an object in a llama-server request. This module is the SAME math in
numpy (the reference the C++ is checked against), the spec presets, and the check itself:

    pollard-composed --presets                                    # the specs worth shipping on a card
    pollard-composed --check --server http://127.0.0.1:8080       # server's q* vs this reference, live
    pollard-composed --solve '{"logits":[...]}' --spec "kl:0.5,coverage:0.5:K=4"

Spec syntax: regularizers as type[:alpha[:key=val...]] joined by ',': kl | js | entropy | coverage |
diversity; keys K, top_m, weight_mode (topm_uniform | topm | topm_l2 | reference), gap_tau; globals
lambda=, base_temp=, steps=, lr=, tol=.
"""
from __future__ import annotations

import argparse, json, math, sys, urllib.request

import numpy as np

EPS = 1e-8

#: the paper's named decoders, as specs. K is the number of samples the objective is written for.
PRESETS = {
    "best-of-k":  "kl:0.5,coverage:0.5:K=4:top_m=8",
    "diverse":    "kl:0.5,diversity:0.5:K=4:gap_tau=1.0",
    "js-entropy": "js:0.5,entropy:0.5",
    "kl-only":    "kl:1.0",           # closed form: softmax(log p + s / lambda)  -- a temperature blend
}

_TYPES = {"kl": "kl_to_base", "kl_to_base": "kl_to_base", "js": "js_to_base", "js_to_base": "js_to_base",
          "entropy": "entropy", "coverage": "coverage", "cov": "coverage",
          "diversity": "diversity_gap", "div": "diversity_gap", "diversity_gap": "diversity_gap"}


def parse_spec(spec: str) -> dict:
    """Same grammar as common_composed_parse in the runtime."""
    out = {"lambda": 1.0, "base_temp": 1.0, "steps": 10, "lr": 0.1, "tol": 0.0, "regularizers": []}
    for item in filter(None, spec.split(",")):
        if "=" in item and ":" not in item:
            k, v = item.split("=", 1)
            if k not in ("lambda", "base_temp", "steps", "lr", "tol"):
                raise ValueError(f"unknown composed key: {k}")
            out[k] = int(v) if k == "steps" else float(v)
            continue
        parts = item.split(":")
        if parts[0] not in _TYPES:
            raise ValueError(f"unknown regularizer: {parts[0]}")
        r = {"type": _TYPES[parts[0]], "alpha": 1.0, "K": 16, "top_m": 8, "weight_mode": "topm_uniform", "gap_tau": 1.0}
        for i, p in enumerate(parts[1:]):
            if i == 0 and "=" not in p:
                r["alpha"] = float(p); continue
            k, v = p.split("=", 1)
            if k not in r:
                raise ValueError(f"unknown regularizer key: {k}")
            r[k] = v if k == "weight_mode" else (int(v) if k in ("K", "top_m") else float(v))
        out["regularizers"].append(r)
    if not out["regularizers"]:
        raise ValueError("composed spec names no regularizer")
    return out


def softmax(x):
    x = np.asarray(x, dtype=np.float64); x = x - x.max()
    e = np.exp(x); s = e.sum()
    return e / s if np.isfinite(s) and s > 0 else np.full_like(x, 1.0 / x.size)


def _topm_weights(p, top_m, K, mode):
    n = p.size; m = max(1, min(top_m, n))
    w = np.zeros(n)
    if mode == "reference":
        return p.copy()
    idx = np.argsort(-p)[:m]
    if mode == "topm":
        w[idx] = 1.0
    elif mode == "topm_l2":
        w[idx] = 1.0 / math.sqrt(m)
    else:                                            # topm_uniform: uniform coverage of one
        max_hit = 1.0 - (1.0 - 1.0 / m) ** K
        w[idx] = 1.0 / (m * max_hit)
    return w


def grad(q, s, p, raw, cfg):
    """gradient of <q,s> - lambda * sum alpha Omega(q); mirrors composimplex's regularizers."""
    g = s.copy()
    for r in cfg["regularizers"]:
        coef = cfg["lambda"] * r["alpha"]
        qs = np.clip(q, EPS, 1.0); ps = np.clip(p, EPS, 1.0)
        t = r["type"]
        if t == "kl_to_base":
            g -= coef * (np.log(qs / ps) + 1.0)
        elif t == "js_to_base":
            m = np.clip(0.5 * (ps + qs), EPS, 1.0)
            g -= coef * (0.5 * np.log(qs / m))
        elif t == "entropy":
            g -= coef * (np.log(qs) + 1.0)
        elif t in ("coverage", "diversity_gap"):
            if t == "coverage":
                w = _topm_weights(p, r["top_m"], r["K"], r["weight_mode"])
                qc = qs
            else:
                gap = raw.max() - raw; tau = max(r["gap_tau"], EPS)
                w = gap * np.exp(-gap / tau); tot = w.sum()
                if np.isfinite(tot) and tot > 0: w = w / tot
                qc = q
            cov = r["K"] * np.clip(1.0 - qc, 0.0, 1.0) ** (r["K"] - 1)
            g += coef * w * cov                      # Omega = -coverage
    return g


def solve(logits, cfg, temp=1.0):
    """q* over the given (already supported) logits. temp = the chain's temperature (s = logits/T)."""
    raw = np.asarray(logits, dtype=np.float64)
    T = temp if temp > 0 else 1.0
    s = raw / T
    p = softmax(raw / max(cfg["base_temp"], 1e-6))
    R = cfg["regularizers"]
    types = {r["type"] for r in R}
    if not R or cfg["lambda"] == 0.0:
        q = np.zeros_like(s); q[int(np.argmax(s))] = 1.0; return q
    if types == {"entropy"}:
        return softmax(s / max(cfg["lambda"] * sum(r["alpha"] for r in R), EPS))
    if types == {"kl_to_base"}:
        return softmax(np.log(np.clip(p, EPS, 1.0)) + s / max(cfg["lambda"] * sum(r["alpha"] for r in R), EPS))
    q = np.clip(p, EPS, None); q = q / q.sum(); q0 = q.copy()
    for _ in range(cfg["steps"]):
        g = grad(q, s, p, raw, cfg)
        if not np.isfinite(g).all():
            return q0
        if cfg["tol"] > 0 and (g.max() - float(q @ g)) <= cfg["tol"]:
            break
        q = np.clip(q * np.exp(cfg["lr"] * (g - g.max())), EPS, None)
        tot = q.sum()
        if not np.isfinite(tot) or tot <= 0:
            return q0
        q = q / tot
    return q


# ------------------------------------------------------------------ live check against a server
def _post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def check(base: str, spec: str, prompt: str, temp: float, n_probs: int = 40) -> dict:
    """Ask the server for the pre-sampling top-n logprobs (the support the chain sees, with
    top_k = n_probs and no other support sampler) and for its post-sampling probabilities under the
    composed spec; solve the same support here; report the max abs difference. What this proves is
    that the C++ sampler and this reference agree on the same candidates -- the paper's math, in the
    runtime."""
    cfg = parse_spec(spec)
    common = {"prompt": prompt, "n_predict": 1, "temperature": temp, "top_k": n_probs, "top_p": 1.0, "min_p": 0.0,
              "seed": 0, "cache_prompt": False, "samplers": ["top_k", "temperature"]}
    pre = _post(base, "/completion", dict(common, n_probs=n_probs, post_sampling_probs=False))
    tok_pre = (pre.get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
    post = _post(base, "/completion", dict(common, n_probs=n_probs, post_sampling_probs=True, composed=spec))
    tok_post = (post.get("completion_probabilities") or [{}])[0].get("top_probs") or []
    if not tok_pre or not tok_post:
        return {"ok": False, "error": "server returned no probabilities (needs n_probs support)"}
    # pre-sampling logprobs are of the softmax over the FULL vocab at raw logits; the support is top-n by id
    ids = [t["id"] for t in tok_pre]
    raw = np.array([t["logprob"] for t in tok_pre])            # log p_full(i) = logit_i - logZ: a constant shift, softmax-invariant
    q_ref = solve(raw, cfg, temp)
    q_srv = {t["id"]: t["prob"] for t in tok_post}
    diffs = [abs(q_ref[i] - q_srv.get(tid, 0.0)) for i, tid in enumerate(ids)]
    return {"ok": bool(max(diffs) < 2e-3), "max_abs_diff": float(max(diffs)), "n_support": int(len(ids)),
            "argmax_agree": bool(ids[int(np.argmax(q_ref))] == max(q_srv, key=q_srv.get)),
            "top": [(int(tid), round(float(q_ref[i]), 4), round(float(q_srv.get(tid, 0.0)), 4)) for i, tid in enumerate(ids[:5])]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--presets", action="store_true", help="print the named specs")
    ap.add_argument("--spec", default=PRESETS["best-of-k"], help="composed spec (or a preset name)")
    ap.add_argument("--solve", help='JSON {"logits":[...]} to solve here')
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--check", action="store_true", help="compare a llama-server's composed sampling to this reference")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--prompt", default="The capital of France is")
    a = ap.parse_args()
    spec = PRESETS.get(a.spec, a.spec)
    if a.presets:
        for k, v in PRESETS.items():
            print(f"  {k:12s} {v}")
        return
    if a.solve:
        q = solve(json.loads(a.solve)["logits"], parse_spec(spec), a.temp)
        print(json.dumps([round(float(x), 6) for x in q])); return
    if a.check:
        r = check(a.server, spec, a.prompt, a.temp)
        print(json.dumps(r, indent=1))
        sys.exit(0 if r.get("ok") else 1)
    print(json.dumps(parse_spec(spec), indent=1))


if __name__ == "__main__":
    main()
