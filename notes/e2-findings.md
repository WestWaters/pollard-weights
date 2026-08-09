# E2 — routing reuse and the hot-expert cache, MEASURED

> **Scope note (2026-08-09):** this experiment measured **prefill** routing.
> Decode-phase behavior differs fundamentally — see `e10-the-hot-set-lives-in-decode.md`
> and the measured results in `e12-both-legs-measured.md`.


> ## ⚠️ CORRECTION (same day) — read this before the tables below
>
> The sensitivity numbers further down used **6.29 MB per expert**, which is README §3's
> **fp16/bf16** figure. We are running a **Q4 file**. Measured from the GGUF's own tensor table:
>
> | | fp16 (what I wrongly used) | Q4_K_M (actual) |
> |---|---|---|
> | per expert | 6.29 MB | **1.945 MB** |
> | active per token | 1.92 GiB | **0.593 GiB** |
> | all expert tensors | 64.4 GB | **18.96 GiB** |
>
> (per expert = `ffn_down_exps` 0.820 MB at **Q6_K** + `ffn_gate_exps` 0.5625 + `ffn_up_exps`
> 0.5625, all at Q4_K. Note down-proj is kept at Q6_K — it is 42% of an expert's bytes.)
>
> Corrected requirements at 3.0 GB/s flash: **70 tok/s needs 92.8% hit** (not 97.76%), 170 tok/s
> needs 97.0%. **RAM-bandwidth ceiling is ~202 tok/s**, so the targets are not physically excluded.
>
> **And the real headline this exposes:** the entire expert set is **18.96 GiB**, i.e. only **~3 GiB
> more than a 16 GB machine**. Under the measured (near-uniform) usage no cache can reach 92%, but
> *nothing needs to be predicted if everything is resident*. The lever is therefore **fitting the
> weights, not indexing them** — see `notes/next-steps.md`.


_Run 2026-07-29 on the real `Laguna-XS-2.1-Q4_K_M.gguf` (19 GB), 16 GB M4, `ngl=0`, mmap._

Captured with `e2_capture_routing.cpp`: a backend eval-callback reads llama.cpp's own
`ffn_moe_topk-<layer>` tensor, so these are the router's **actual** top-8 selections, not a
reimplementation that could disagree. 462 tokens, **17,561 routing records**, 5 prompts across
code / prose / math.

---

## The headline: expert usage is ~UNIFORM, not Zipf

README §4 marks the ~126 tok/s figure as resting on *"Zipf-skewed expert reuse — an assumption, not
a measurement. This is the number the whole design rests on."*

Measured, the skew is barely there:

| cache holds | = share of pool | serves | skew ratio |
|---|---|---|---|
| 1 expert/layer | 0.4% | 0.5% | 1.18 |
| 8 | 3.1% | 3.7% | 1.18 |
| 32 | 12.5% | 14.2% | 1.14 |
| 64 | 25.0% | 27.4% | 1.09 |
| 128 | 50.0% | 53.0% | 1.06 |

A skew ratio of ~1.1 is a **flat distribution**. Zipf would put 3–10× here. The hottest 12.9% of
experts carry 14.6% of traffic — you cannot cache your way out of that.

**The statistics point the same way, not the other way.** At 462 tokens each expert is drawn ~14
times on average, so individual frequencies are noisy — but sampling noise *manufactures* spurious
hot experts, it does not hide real ones. Measuring near-uniform *despite* noise is stronger evidence
of true uniformity than a clean large-sample measurement would need to be.

## Why that is fatal to the 70 tok/s target

Per-token active weight is 1.92 GB. At the measured 3.0 GB/s flash rate, required hit rates are:

| target | required hit rate |
|---|---|
| 10 tok/s | 84.35% |
| 30 tok/s | 94.78% |
| **70 tok/s** | **97.76%** |
| 126 tok/s | 98.76% |

An 8 GB cache (33 experts/layer, the most a 16 GB Mac can spare) buys **14.6%** → **1.8 tok/s**.
The gap is not a tuning problem; it is two orders of magnitude of required skew that isn't present.

## Routing-pattern reuse (open Q1): weak, and NOT converging

- Mean reuse **28.7%** (334 distinct patterns per layer out of 462 occurrences).
- New-pattern rate by decile: `100 100 100 30 97 76 54 29 50 70` — noisy, no clean decay.
  Ratio last/first = 0.70, which is *not* convergence on this sample.
- Depth-independent: first third 27.7%, middle 29.0%, last 29.5%. **E1's depth-sparsity structure
  does not reappear here** — evidence that §2's "6.9% active in the deepest layer" was a 0.6B dense
  artifact, as it was flagged to be (open Q3).

## Distribution shift is real (open Q4)

Cross-domain pattern overlap (Jaccard): code↔math **12.4%**, code↔prose **11.5%**, math↔prose
**27.9%**. "Converged" was never going to be one global state — each domain routes differently, so a
fixed-size table is refilled on every domain switch.

## Measured speed

Prefill wall clock **1.4 tok/s** at ngl=0 — against README §4's flash-only prediction of 7.9 tok/s.
Prefill is batched and compute-bound so it is not the generation number, but it is 5.6× *worse* than
the optimistic path, not better.

## What survives

The MoE reframe itself is still right and still the interesting part: **1.68B active of ~32B, 3.1%
structural sparsity, deterministic routing, 6.29 MB experts already in the fast-read regime.** What
does not survive is the *caching* story built on top: with flat expert usage there is no hot set to
hold, and an online index of routing patterns neither converges nor transfers across domains.

## Caveats — do not over-read this

1. **462 tokens, 5 prompts, one model.** Enough to falsify a strong skew claim; not enough to
   characterise a converged table.
2. **Prefill, not generation.** Batched prefill can activate more experts per layer-pass than
   single-token decode does. A decode-time capture could show more locality — this is the one
   result that could genuinely move, and it is the obvious next run.
3. **39 layers seen, not the expected 40**, and layer 39 produced only 5 records vs 462 for every
   other layer. Layer 0 is likely dense (common in MoE), but the layer-39 shortfall is unexplained
   and is a capture-fidelity question worth closing before building anything on this data.

## Next, in order

1. Re-run capture during **generation** (not prefill) — the one caveat that could change the verdict.
2. Explain the layer-39/40 anomaly.
3. Longer, more diverse traffic (10k+ tokens) before trusting any reuse number.
4. Only if decode-time skew appears: revisit the cache. Otherwise the honest conclusion is that
   **flash-resident MoE on 16 GB is bandwidth-bound at single-digit tok/s**, and the Pollard index
   was solving the wrong half.
