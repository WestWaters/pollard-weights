# E13 — measured sensitivity beats uniform IQ, dense AND MoE, model-agnostic

_2026-08-15. The allocator E12 pointed at, built and measured. `pollard-sensitivity`
(new) measures the signal; `pollard-fit --sensitivity` allocates on it. Firm 48K-token
eval; `experiments/plot_kl_win.py` regenerates the chart from `experiments/data/`.
Conditions inline; replication invited._

## The claim

At matched size, measured-sensitivity allocation beats **uniform imatrix-IQ** (the
strong baseline, not just uniform Q_K) on both dense and MoE:

| model | vs uniform imatrix-IQ | notes |
|---|---|---|
| dense Qwen2.5-1.5B | **+6% to +27%** lower KLD | 5/5 practical sizes |
| MoE granite-3B-a800m (40 experts) | **+21% to +43%** lower KLD | 4/5; loses only at the IQ2_S floor |

MoE wins bigger because expert-importance variance is larger — the allocation has
more to exploit. This is the mechanism that scales to the 27B/2.4T tier.

## The baseline — stated once, used everywhere

"Uniform at size S" = **linear** interpolation between the two adjacent measured
uniform quants. That is the honest *naive-mix* baseline: dithering some layers to
the next quant type down lands you on that line, and pollard is itself a mix, so the
fair question is "does our measured mix beat a naive mix at the same size." Do **not**
interpolate the baseline in log-log — that invents a single-type quant that cannot
exist at an intermediate size, understates the baseline, and manufactures fake losses
at the tight end. (This footgun cost a full round of circles; it's now nailed down in
`plot_kl_win.py` with a comment so it can't drift again.)

## Three things E12 got directionally right and E13 nails

1. **"Bits need a sensitivity signal" (E12) — but the imatrix is the wrong signal.**
   E12 ranked layers by the imatrix (cold layers 4,7,2). We MEASURED it instead —
   crush each tensor group to a probe quant, read the KL cost — and the imatrix
   *magnitude lies*: big activations ≠ high KL sensitivity. The sharpest example:
   **attention is only 0.48× as sensitive as FFN.** The whole "protect attention /
   compress the FFN bulk" instinct (and the old `PROTECTED` list) was **backwards**.
   Allocate on the measured cost, not the proxy.

2. **The victim-layer spikes (E12) are avoidable, not a cost of doing business.**
   E12 saw "rare-token spikes in the q2_K victim layers" and accepted them. The fix
   is to allocate against a MEASURED noise curve — the uniform-quant KL per type.
   It is steeply convex: crushing any tensor to iq2_xxs costs ~10× iq3_s. A marginal
   KL-per-byte knapsack therefore *never* creates a victim layer unless the budget
   truly forces it — the catastrophic move carries a catastrophic marginal cost.
   Greedy "crush the coldest layers first" (blind to type cost) is what produced the
   spikes; the knapsack replaces it.

3. **The numbers differ per model — so measure them per model.** Granite's noise
   curve runs ~2× Qwen's at every rung. Bake in one model's and you mis-allocate the
   next. `pollard-sensitivity` measures both the sensitivity AND the noise curve for
   the model in front of it; the calibrated defaults are only a fallback (and a
   point that fails to measure is log-interpolated from the model's *own* curve).

## Why naive per-layer mixing loses (the trap we fell into first)

Mixed precision only beats uniform when importance *variance* is large enough to
overcome the convexity of the noise curve (Jensen: at fixed average bit-width, a
spread allocation has higher average error than uniform). Early attempts — allocating
on the imatrix proxy, or crushing "cold" layers greedily — landed *above* the uniform
curve. Only when we (a) measured the true sensitivity and (b) minimized importance-
weighted error against the measured noise curve did it cross below uniform, dense and
MoE alike. MoE clears the bar by more; its expert variance is huge.

## Method (the objective, stated)

Minimize `Σ_group importance(group) · noise(quant_type)` subject to
`Σ size ≤ budget`, group = a per-layer FFN/expert block or a per-layer attention
block; embeddings/output held high and counted (they carry the vocab). Solved as a
marginal KL-per-byte knapsack (a heap of moves), everything starting at the top of
the ladder and stepping down the cheapest move until it fits. Ladder
`q6_K → q5_K → iq4_xs → iq3_s → iq2_s → iq2_xxs` (iq4_xs replaces q4_K — IQ dominates
the K-quant per byte, measured).

## Honest limits
- At the IQ2_S-class floor, where nothing smaller exists to compare against and
  nothing is left to allocate, pollard *loses* — on granite at ~1.06 GB (that's the
  one red point on the MoE chart). Above the floor it wins across the board.
- Firm 48K-token eval; `experiments/plot_kl_win.py` + `experiments/data/*.csv`
  regenerate `assets/kl_win.png` end-to-end from the raw numbers.
- Per-EXPERT (not per-layer) allocation is still the open lever (E12's open item);
  fused expert tensors in GGUF block it — a runtime/kernel project.

## Tooling
`pollard-sensitivity --gguf f16 --imatrix imat --eval held-out.txt --out prof.json`,
then `pollard-fit --gguf f16 --ram N --imatrix imat --sensitivity prof.json`.
