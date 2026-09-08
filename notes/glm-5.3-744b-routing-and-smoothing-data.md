# GLM-5.3 (744B MoE) — routing concentration and norm-seam outlier data from a 10× DGX Spark cluster

Contributed data, not a tool change. Two measured datasets on a model Pollard does not have yet, captured during an EXL3 3.2 bpw
cook (SmoothQuant-folded, EXL3 budgeted allocator, per the v1.3.0 EXL3 lane) on ten DGX Spark (GB10) nodes.

- `experiments/data/glm53_routing_concentration.json` — per-layer, per-expert routing frequency and mass for all 75 MoE layers.
- `experiments/data/glm53_norm_seam_absmax.csv` — per-layer activation-outlier summary at both RMSNorm→linear seams (the
  quantity `pollard-hf-smooth` folds).

Model: GLM-5.3, 78 layers (3 dense + 75 MoE), 256 routed experts + 1 shared per MoE layer, top-8, `n_group = 1`, DSA sparse
attention with a per-layer indexer, MTP head. 744B total parameters, ~40B active.

## 1. Routing concentration (prefill phase)

**Capture.** Router forward hook (top-8 indices and weights per token) during a full-sequence bf16 forward, one layer streamed
at a time; 96 rows × 2048 tokens of real assistant traffic (chat, code, agent/tool calls, long documents) = 196,608 tokens per
layer. This is **prefill-phase** routing; Pollard e10 found decode concentrates ~2× vs prefill on other MoEs, so read these as
the prefill side. It is not a `pollard-run` profile (no token-level reuse trace — the serving stack's fused MoE path does not
export per-token routing at decode time).

| layer | n_eff (of 256) | top expert share | dead experts | picks covered by top-64 | top-128 | top-192 |
|---|---|---|---|---|---|---|
| 3 | 18 | 12.1 % | 174 | 100 % | 100 % | 100 % |
| 4 | 38 | 11.6 % | 46 | 96 % | 100 % | 100 % |
| 8 | 129 | 7.5 % | 2 | 65 % | 86 % | 97 % |
| 9 | 219 | 1.6 % | 0 | 43 % | 71 % | 90 % |
| 16 | 182 | 3.0 % | 0 | 55 % | 81 % | 95 % |
| 24 | 223 | 2.5 % | 0 | 43 % | 69 % | 89 % |
| 40 | 200 | 3.7 % | 0 | 49 % | 76 % | 92 % |
| 56 | 204 | 3.5 % | 0 | 47 % | 73 % | 91 % |
| 71 | 219 | 1.6 % | 0 | 45 % | 71 % | 90 % |
| 77 | 195 | 1.4 % | 0 | 50 % | 79 % | 95 % |

n_eff = exp(entropy of the pick distribution). Over the 75 MoE layers: median 204, min 18 (layer 3), max 223. Median pick
coverage: top-64 49 %, top-128 74 %, top-152 81 %, top-192 91 %. Mass coverage tracks pick coverage within 1–3 points.

**Readings.**
- Routing is **near-uniform from layer 9 onward** (n_eff 180–223 of 256). A hot set exists only in layers 3, 4 and 8. For a
  `pollard-export`-style `hot_frac` policy this model gives leverage in three layers; elsewhere each tail expert carries ~0.4 %
  of picks.
- Consistent with that, pruning 40 % of experts by routing saliency (K = 152 kept, retaining 84 % of routing mass) moved
  perplexity only +9 % but dropped HumanEval+ by 12–16 points on this model. Perplexity understates pruning damage on a flat
  router; gate pruned MoEs on task benchmarks.
- Frequency-based and saliency-based (routing-weight mass) keep-sets overlap only ~79 % at K = 152.

## 2. Norm-seam activation outliers (the SmoothQuant case) — all 78 layers

**Capture.** Per-channel |x| maximum at the two RMSNorm outputs, same streaming forward, 384 rows × 2048 tokens (786,432 tokens
per layer). Seam A = `input_layernorm` output → `q_a_proj`, `kv_a_proj_with_mqa`, indexer `wk`/`weights_proj`; seam B =
`post_attention_layernorm` output → router `gate`, routed experts' `gate_proj`/`up_proj`, shared experts' `gate_proj`/`up_proj`.

| layer | attn seam max/median | ch >8× median | MoE seam max/median | ch >8× median | MoE-seam median |
|---|---|---|---|---|---|
| 0 (dense) | 14 | 2 | 24 | 6 | 0.08 |
| 3 | 22 | 5 | 16 | 2 | 0.04 |
| 5 | 16 | 8 | 10 | 1 | 0.07 |
| 8 | 27 | 13 | **70** | 3 | 0.16 |
| 9 | 10 | 2 | **34** | 1 | 0.18 |
| 10 | 13 | 2 | 2 | 0 | 0.18 |
| 16 | 11 | 3 | 3 | 0 | 0.33 |
| 24 | 9 | 1 | 2 | 0 | 0.60 |
| 32 | 7 | 0 | 2 | 0 | 0.88 |
| 48 | 7 | 0 | 2 | 0 | 1.51 |
| 64 | 7 | 0 | 2 | 0 | 2.09 |
| 71 | 6 | 0 | 1 | 0 | 2.47 |
| 76 | 12 | 1 | 4 | 0 | 2.41 |
| 77 | 9 | 1 | 5 | 0 | 1.77 |

Across 78 layers: attention-seam max/median median 8 (range 5–27); MoE-seam median 2 (range 1–70); the MoE seam exceeds 8× only
in layers 0–9. Fold scales at α = 0.5 (`s_j = max|X_j|^½ / max_i|W_ij|^½`, clamp [1e-2, 1e2]): attention seam s ∈ [0.35, 2.7],
MoE seam s ∈ [0.07, 6.1]; no channel hit the clamp in any layer.

**Readings for `pollard-hf-smooth`.**
- Outliers on this model are **localized**: modest at the attention seam throughout, and at the MoE seam only in layers 0–9
  (8 and 9 are the hot ones). From layer 10 on the MoE seam is flat (≤ 5×), so the fold there is close to a no-op. A per-layer
  need gate (skip a seam below a max/median threshold) or per-seam α would spend the transform only where it changes the
  quantizer's view. A smoothed-vs-unsmoothed per-layer control (layer 8 vs layers 24/48/71 at identical converter settings) is
  running; numbers will follow in this note.
- The MoE-seam median grows monotonically with depth (0.04 → 2.6): the residual stream scales up ~60× through the stack while
  the relative outlier structure disappears.
- At 744B the single-device forward in `pollard-hf-smooth` cannot run. What worked: one process per layer, hooks on the two
  norms, residual stream streamed layer to layer (~90 s per MoE layer for 786K tokens on one GB10, load-bound), stats to JSON;
  the fold rewrites only the shards holding a layer's tensors (idempotent via per-layer markers) and is exactly invertible at
  α = 0.5 (`s = x / max|W_folded|`). Happy to upstream a streaming mode if in scope.

## 3. Hardware profile — DGX Spark (GB10) cluster

- Node: 121.6 GB unified memory, one GPU; usable serving budget ≈ 0.81 × RAM (higher wedges long prefills); NVMe ~2–3 GB/s.
- bf16 streaming forward at 786K tokens/layer: dense layer ~45 s; MoE layer ~80–95 s, of which 20–47 s is loading ~19 GB of
  expert weights.
- exllamav3 conversion at 384 × 2048 calibration rows, `-b 3.2 -hb 6 -mb 8 -hq`: dense layer ~4 min (rfn 0.010–0.012, sqnr
  39–41 dB at the allocator's 5.05 bpw); MoE layer (256 experts × 3 × 6144×2048 + attention) ~50 min; peak RSS ~25 GB.
- Band-parallel conversion (ten bands, each node resumes the converter from the full-precision residual stream at its band start,
  calibrated on 384 rows): ~7 h wall for 78 layers plus ~1 h merge/compile, versus ~3 days sequential on one node.
- KV bytes/token for this model (MLA + DSA indexer): NVFP4 KV ≈ 41 KB, fp8 ≈ 57 KB. At TP4 and 3.2 bpw: weights ~77.5 GB/node;
  an 8-bit MTP draft replicated per rank costs ~12 GB/node (4-bit ~6.5 GB) — the draft's precision, not the body's, decides the
  KV pool (9 GB → ~220K ctx at NVFP4 KV; 14.5 GB → ~350K).
- exllamav3 needed an aarch64 build patch (exclude the x86-only CPU inference sources + stub TU; `__builtin_ia32_pause` → `yield`;
  `cusparse.h` from the CUDA-13 pip layout). Available on request.

*Disclosure per CONTRIBUTING: measurements were produced with the help of an AI assistant operating the cluster; the numbers
above were read back from the raw capture files by the contributor and the harness code is ours, not Pollard's `e2` shim (the
model does not run under llama.cpp on this cluster).*
