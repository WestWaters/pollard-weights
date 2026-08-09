# E3 — can routing be known AHEAD of time? (the real Pollard-weights question)

> **Scope note (2026-08-09):** this experiment measured **prefill** routing.
> Decode-phase behavior differs fundamentally — see `e10-the-hot-set-lives-in-decode.md`
> and the measured results in `e12-both-legs-measured.md`.


_2026-07-29, measured on `Laguna-XS-2.1-Q4_K_M.gguf`, 17,561 real routing records._

E2 asked the wrong question. It measured whether expert-set *patterns repeat*. The idea never needed
that — README §3 already says the router is deterministic and **"the problem is not accuracy. It is
TIMING."** A prefetch index is only worth anything if it buys **lookahead**: knowing layer L's
experts before layer L's router runs, so the read can start early and the weights stream on demand.

E3 measures lookahead three ways. Chance baseline for two independent top-8 draws from 256 experts is
**3.1%** overlap.

| signal | measured | vs chance |
|---|---|---|
| **cross-layer** (layer L → L+1, same token) | **4.1%** | 1.3× |
| **temporal** (token t → t+1, same layer) | **0.03%** | 0.01× |

## The decisive number: routing is ANTI-correlated across tokens

Consecutive tokens at the same layer share essentially no experts:

```
|overlap| histogram over 17,366 pairs:   0:17336   1:24   2:5   3:1
expected zeros under independence: ~13,545      observed: 17,336 (99.83%)
```

That is **below** chance, so it is not merely "no locality" — consecutive tokens are actively routed
*apart*. Which is exactly what a load-balanced MoE router is built to do: auxiliary balancing loss
spreads similar inputs across different experts to keep utilisation even. It also explains E2's
near-uniform expert usage — the two measurements are the same fact seen twice.

A result below chance is usually a bug, so this was checked rather than believed: all 256 expert ids
appear, 17,366 joins land, sets look uniformly spread. The data is sound.

## The DP table, scored honestly

Building the table on the first 70% of each prompt's tokens and scoring the last 30%:

| | |
|---|---|
| keys in table | 7,022 (81.4% seen only once) |
| held-out lookups | 5,180 |
| **key present at all** | **1.5%** |
| exact 8-of-8 prefetch | 0.5% |
| **mean experts prefetched** | **1.3% of 8** |

⚠️ **Scoring the same table on its own rows reports 98.8%.** That number is memorisation of
mostly-unique keys, and it is the trap this experiment nearly fell into — a self-scored lookup table
always looks like a breakthrough. Held out, the index prefetches ~1.3% of what is needed.

## What this closes, and what it opens

**Closed:** prediction. No index, no DP/kangaroo collision structure, no nearest-neighbour scheme over
router inputs can prefetch this routing, because there is no lookahead signal to exploit — across
layers (1.3× chance) or tokens (below chance). TurboVec/ANN does not rescue it either: an ANN index
only helps if similar router *inputs* recur, and consecutive hidden states demonstrably route apart.

**Open, and now the whole game:** the expert tensors are **18.96 GiB** against a 16 GB machine —
about **3 GiB short**. Under uniform, unpredictable routing no cache can win, but **nothing needs
predicting if everything is resident.** Get experts under ~13 GiB and flash reads go to zero, leaving
a RAM-bandwidth ceiling of ~202 tok/s (README §4's "~70 realistic" being the honest target — still
~14× the ~5 tok/s flash-only path).

Cheapest concrete win, already located: `ffn_down_exps` is the sole **Q6_K** tensor in an otherwise
Q4 file and is 42% of an expert's bytes. Requantising it to Q4_K saves **~2.51 GiB**. See
`notes/next-steps.md`.

## What survives of the original idea

The *architecture* does, even though the index does not:

- **MoE is the right target** — 1.68B active of ~32B, 3.1% structural sparsity, 1.945 MB experts
  already in the fast-read regime. All confirmed.
- **Stream on demand rather than mmap the world** — confirmed as the right call for a different
  reason than expected: the 19 GB mmap's page cache is what evicted every other process to swap
  (~25 GB). Bounded `F_NOCACHE` reads, measured in README §1 at 2.4–4.9 GB/s, avoid that entirely.
- **Rotation as a fixed-size table** — still the correct shape for anything resident; it just holds
  weights rather than predictions.

What does not survive is the premise that routing has exploitable structure. It is deterministic but
effectively incompressible in time: a hash, not a trajectory. The kangaroo framing needed collisions,
and a load-balanced router is engineered to avoid exactly those.
