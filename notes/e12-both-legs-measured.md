# E12 — both legs measured: placement wins throughput, fitting wins quality

_2026-08-09. The results the method was built to produce. Conditions disclosed
inline; single-machine, replication invited._

## Leg 1 — measured placement (`pollard-run`): +21% throughput

Qwen3-30B-A3B Q3_K_M (14.7 GB — does not fit the GPU), 16 GB Apple Silicon,
identical GPU budget and stream count per arm, `llama-cli` greedy n=128, n=3:

| placement | gen tok/s (reps) | mean |
|---|---|---:|
| all experts on CPU (common default) | 9.6 (n=1) | 9.6 |
| blind first-N layers (`--n-cpu-moe`) | 12.9 / 11.5 / 8.8 | 11.1 |
| **measured split (`pollard-run`)** | **15.6 / 11.9 / 12.8** | **13.4** |

Measured placement won **every paired rep**. The split it chose is
non-contiguous (hot layers deep AND shallow) — a shape the blind option cannot
express. Variance is real (shared machine); treat magnitudes as directional,
the ordering as solid.

**Budget dose-response (full-ceiling rerun, 8 GB budget / 23 streamed, 3
discarded warmups + 4 reps, median-scored):** all-CPU 6.9 · blind 11.7 ·
measured 12.0 (3/4 pairs). With a looser budget the blind first-N choice
*accidentally overlaps* the measured cold set (this model's heat is deep), so
the gap narrows from +21% to +3% — i.e. **measurement's advantage grows as
memory gets scarcer**, which is the thesis itself, observed as a curve.

## Leg 2 — measured fitting (`pollard-fit` IQ ladder): −25% KL at smaller size

granite-3B MoE, builds from f16 with the same imatrix, KL-divergence against
the f16 reference on held-out WikiText (25K tokens):

| 1.6 GB-class build | mean KLD | median | 99.9% tail | size |
|---|---:|---:|---:|---:|
| uniform Q3_K_M preset | 0.1432 | 0.0474 | 3.91 | 1.64 GB |
| **pollard IQ-mix** | **0.1070** | **0.0330** | **3.75** | **1.61 GB** |

Every axis, ~14 standard errors apart, in a smaller file.

**Replicated on a second, fully out-of-domain corpus** (literary prose):
−21% mean with the first-pass imatrix. **Upgrading the calibration corpus**
(60 KB single-domain → 1.3 MB mixed encyclopedic/narrative/dialogue/code,
`experiments/make_calibration.py`) widened the gap, most strongly out of
domain: WikiText-test **0.1100 vs 0.1500 (−27%)**, fresh literary eval
**0.1038 vs 0.1678 (−38%)**, both corpora calibration-disjoint, sensitivity
ranking unchanged (cold layers 4,7,2 under both matrices). Measured
allocation *generalizes*; the uniform preset degrades off-domain.

## How the fitting win was found (the method, working)

1. First allocator used the *reuse* signal for bits — lost by 15% perplexity.
   Lesson: reuse is a **residency** signal; bits need a **sensitivity** signal.
2. Sensitivity-driven victims reached near-parity; KL-divergence then localized
   the whole remaining gap to rare-token spikes in the q2_K victim layers.
3. `e11_sign_asymmetry.py` **refuted** the imported hypothesis that 2-bit
   failure is sign-bias: GGUF's Q2_K residuals are already symmetric (49.5%
   negative, bias ≈ 0.001). The failure is plain magnitude error. A
   sign-symmetric kernel would have fixed nothing here.
4. Direct KL comparison: IQ lattice types dominate Q_K at aggressive bits
   (IQ2_S: −23% mean KLD than Q2_K *and* 16% smaller).
5. Ladder switched to `q4_K → iq3_s → iq2_s → iq2_xxs`. At the 1.6 GB budget
   the allocator no longer needs sacrificial layers at all.

## Open
- Per-expert (not per-layer) placement — blocked on fused expert tensors in
  GGUF; a runtime/kernel project.
