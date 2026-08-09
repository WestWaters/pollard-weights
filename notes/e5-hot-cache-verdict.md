# E5 — does a bounded hot-expert cache work? Measured: no.

> **Scope note (2026-08-09):** this experiment measured **prefill** routing.
> Decode-phase behavior differs fundamentally — see `e10-the-hot-set-lives-in-decode.md`
> and the measured results in `e12-both-legs-measured.md`.


_2026-07-29. Simulated on the validated 39-layer trace from the fresh Q4 model._

This is the experiment that should have run first: not "can routing be predicted" (E3) but the design
as actually described — **hold a few GB of experts hot, stream the misses on demand, run in 3–6 GB
instead of 19.**

Method: replay the real routing trace token by token; keep a fixed-size LRU of `(layer, expert)`
pairs; count how many of each token's 8-per-layer expert loads are already resident. Both a cold
cache and a *warm* one (pre-loaded with the most-used experts, i.e. best-case steady state).

## Result — single-domain, which was the fair test

266 tokens of pure Python (one continuous code session), all 39 layers:

| cache | = % of expert pool | warm hit% | flash GiB/token | tok/s |
|---|---|---|---|---|
| 3 GB | 16% | 0.4% | 0.590 | 5.1 |
| **6 GB** | **32%** | **1.9%** | 0.581 | **5.2** |
| 8 GB | 42% | 4.6% | 0.565 | 5.3 |
| 12 GB | 63% | 15.7% | 0.499 | 6.0 |

Mixed-domain (462 tokens, code+prose+math) gives 3.0% at 6 GB and 17.2% at 12 GB — **statistically
the same.** Restricting to one domain does not concentrate demand.

## The one number that explains it

> **266 tokens of pure code touch 9,741 of 9,984 possible (layer, expert) pairs — 97.6%.**

There is no hot set. Within a few hundred tokens of a single domain, the model has used essentially
every expert in every layer. That is not a property of mixed workloads or of a cold start; it is what
a load-balanced router does by design, and it matches E2's near-uniform usage (skew 1.13×) and E3's
below-chance token-to-token overlap. Three independent measurements, one fact.

## Why the cache underperforms even its own size

A 12 GB cache holds **63%** of all experts but serves only **16%** of demand. That gap is **reuse
distance**, not skew:

- each token touches 312 distinct experts (8 × 39 layers), never repeating within a token
- for one layer, a given expert recurs roughly every 32 tokens
- in that window the model has touched ~10,000 expert-slots — more than the cache holds

So every expert is evicted before it is reused. Enlarging the cache moves the threshold but does not
change the shape until the cache approaches the whole pool.

## The arithmetic this leaves

Active weights are 0.593 GiB/token. At 3.0 GB/s flash:

| target | required hit rate | best measured (6 GB) |
|---|---|---|
| 30 tok/s | 83.1% | 1.9% |
| 70 tok/s | 92.8% | 1.9% |
| 170 tok/s | 97.0% | 1.9% |

A 6 GB cache is 32% of the pool, so even a *clairvoyant* cache of that size could not exceed ~32%
hit against uniform demand. The targets need 93–97%. The gap is structural, not a tuning problem.

## Verdict

**The hot-cache / stream-on-demand design does not work on this model.** Not because of the index,
the predictor, or the eviction policy — but because the working set is the entire expert pool within
a few hundred tokens. Nothing bounded can hold it.

What *did* work, measured the same evening: **making the weights fit.** 11.46 GiB resident on Metal
gave **41 tok/s vs 4.56** on the flash path — a 9× win from bytes, not foresight. That is the live
path, and its open problem is quantisation *quality*, not speed.

## Honest limits of this result

1. 266 tokens single-domain / 462 mixed. Enough to show the working set is ~the whole pool (97.6%
   coverage arrives almost immediately), not enough to characterise a long session's tail.
2. One model. Laguna-XS.2's router is aggressively load-balanced; a model with weaker balancing
   could behave differently. The method here transfers directly — point it at another MoE.
3. LRU only. Better policies exist, but with a reuse distance larger than the cache no policy helps;
   the ceiling is set by pool coverage, which no policy changes.
4. The larger single-domain capture (~1,750 tokens) was attempted five times and never completed —
   see `notes/rig-failures.md`. The conclusion rests on the smaller validated trace.
