# The lever is FITTING, not predicting

_2026-07-29, after correcting the fp16/Q4 mix-up in E2._

## The reframe

Measured on the real Q4 file:

| | |
|---|---|
| all expert tensors | **18.96 GiB** |
| machine | 16 GB |
| **shortfall** | **~3 GiB** |
| active per token | 0.593 GiB |
| flash-only (3 GB/s, no cache) | ~5.1 tok/s |
| **everything resident** | **RAM-bound, ~202 tok/s ceiling / ~70 realistic** |

E2 measured expert usage as near-uniform (top 12.9% of experts serve 14.6% of traffic, skew ratio
1.13). If that holds at decode time, **no cache reaches the 92.8% needed for 70 tok/s** — you can't
concentrate demand that isn't concentrated.

But uniform demand has an inverse implication that is *good* news: **if the whole set is resident,
there is nothing to predict, no index, no rotation, no cold start.** And the whole set misses by
only ~3 GiB.

So the project's decisive question changes from "can we predict routing?" to **"can we get the
expert tensors under ~13 GiB with acceptable quality?"** (13, not 16 — leave room for attention,
embeddings, KV cache, and the OS.) That is a 31% reduction, and it is a well-defined, measurable
engineering target rather than a research bet.

## Why the ballooning happened, and why this fixes it too

The 19 GB mmap did **not** swap the weights: they are clean, file-backed, read-only pages, always
re-readable from disk. What ballooned swap to ~25 GB was macOS evicting *other processes'* anonymous
memory (the agent, ollama's pinned 5.6 GB, the editor) to make room for a 19 GB page cache.

That means bounded, explicit reads of individual experts — rather than mmap-the-world — inherently
avoid it. README §1 already measured the mechanism: `F_NOCACHE` 4 MB reads at 2.4–4.9 GB/s, which
bypass the page cache entirely. So the Pollard-weights *architecture* was already the answer to the
ballooning, independent of whether the routing index pans out.

## Concrete next steps, cheapest first

1. **Requantize `ffn_down_exps` Q6_K → Q4_K.** It is currently the only Q6 tensor in an otherwise Q4
   file and accounts for 42% of an expert's bytes. Saving: (0.820 − 0.5625) MB × 256 × 39 =
   **~2.51 GiB**, taking experts to ~16.45 GiB. Not sufficient alone, but it is the single largest
   cheap win and it tests the pipeline.
2. **Try a genuinely smaller quant** (Q3_K_M, or llama.cpp's imatrix IQ3 family). ~3.4 effective bits
   would put experts near **14.3 GiB**; IQ2/IQ3_XXS clears 13 GiB comfortably. Quality must be
   measured, not assumed — perplexity plus a task check.
3. **Decode-time routing capture.** Still worth doing: it is the one measurement that could show
   real locality (E2 measured batched *prefill*, where wide expert coverage is expected). If decode
   locality turns out strong, a cache becomes viable and step 2 can be less aggressive.
4. **Only then** revisit the routing index. Under uniform decode-time usage it is the wrong lever;
   under skewed usage it is worth building.

## Honest ceiling

170 tok/s requires a 97% hit rate *and* near-peak memory-bandwidth utilisation. The 202 tok/s figure
is a theoretical bandwidth ceiling; real implementations land well under it, which is why README §4's
own "~70 realistic" is the number to aim at. **70 tok/s for a 19 GB model on a 16 GB Mac would still
be a genuinely good result** — it is ~14× the flash-only path.
