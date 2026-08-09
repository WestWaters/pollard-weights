# E10 — the hot set lives in decode

_2026-08-09. Supersedes the scope — not the data — of `e5-hot-cache-verdict.md`._

## The error every prior verdict shared

Every "no hot set" measurement in this project (e2, e5) — and every external
critique built on them — measured routing during **prefill**: batch-reading
foreign text. Users don't live there. They live in **decode**: the model
generating, settling into its own topic, looping on its own task. E5's verdict
was correct *for the regime it measured*. It measured the wrong regime.

## Measurement

Capture tool extended with a decode phase (`--gen N`, greedy or temperature
sampling, records tagged by phase). Model: the 3B/40-expert testbed, f16.
Four decode regimes vs the prefill baseline, same instrument:

| regime | decode records | pattern reuse | top-25% expert share |
|---|---:|---:|---:|
| prefill baseline (e5's regime) | 22,446 | 4.4% | 25.4% (flat) |
| greedy, long | 5,792 | 12.1% | **51.2%** |
| temp 0.8, seed 1 | 8,608 | 18.4% | **51.1%** |
| temp 0.8, seed 2 | 13,920 | 26.7% | **51.9%** |
| task prompt, temp 0.8 | 23,040 | 40.0% | **52.3%** |

Two findings:

1. **Decode routing concentrates ~2× over prefill** — top-quarter experts carry
   half the traffic — and the share is *stable across sampling regimes* (greedy
   vs temperature vs seeds vs prompt style: 51.1–52.3%). Not a repetition
   artifact: temperature sampling shows it as strongly as greedy.
2. **Pattern reuse grows with generation length** (12% → 40% as generations run
   longer): sustained work concentrates further. Agent-style workloads — long
   generations on one task — are the *best* case for residency, not the worst.

## The e5 simulator, re-run on decode traffic

Same `e5_cache_sim.py`, unchanged, fed the merged decode trace:

- 2 GB cache (25% of this model's expert pool): **97.2% hit rate**
- 6 GB cache: **100.0%**, implied ceiling 202 tok/s
- The tool's own verdict line: *"THE DESIGN WORKS."*

The same code printed "Measured: no" on prefill traffic. The regime was the
variable.

## What this does NOT yet establish

- The testbed's expert pool (~8 GB) is small relative to the caches simulated —
  hit rates are flattered. The decisive test is a model whose pool dwarfs the
  cache (30B-class or larger).
- ~1,600–2,000 decode tokens per run; longer traces needed for churn.
- One model family. Concentration may vary with load-balancing recipe.
- 202 tok/s is bandwidth-ceiling arithmetic, not a stopwatch measurement.

## Gates: PASSED (2026-08-09)

1. ~~Long traces~~ — 12,000 decode tokens across 6 regimes on the 30B gate.
2. ~~30B-class where cache << pool~~ — Qwen3-30B-A3B (128e/48L): top-25%
   experts carry **70–78%** of decode traffic per prompt (68.7% merged);
   concentration *grew* with model scale. 6 GB cache serves 90.3% of traffic,
   8 GB serves 96.7%.
3. ~~Measured tok/s~~ — done: see `e12-both-legs-measured.md`. Measured
   placement beat blind placement in every paired run, +21% mean.
