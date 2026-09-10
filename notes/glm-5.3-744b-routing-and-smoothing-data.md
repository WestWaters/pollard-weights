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

## 4. EXL3 allocator depth heuristic vs measured per-layer error — and a self-QC fix (new)

exllamav3's budgeted allocator (`-b 3.2 -hq`, no recipe) assigned bits from config alone: routed experts **4 bits in layers 3–7
and 70–77, 3 bits in the other 62 MoE layers**; attention and shared experts 5 bits; dense MLP 4; overall 3.21 bpw. The converter
then reports each module's output error on the calibration rows as it goes. Measured at 384 × 2048 rows (first two layers of
every band; the full 78-layer profile follows when the cook completes):

| layers | allocator bpw | sqnr (dB) | rfn | wall per layer (one GB10) |
|---|---|---|---|---|
| 0–2 (dense) | 5.05 | 39–41 | 0.010–0.012 | ~4 min |
| 3–4 | 4.06 | 52.4 / 45.9 | 0.003 / 0.007 | 42 min |
| 8–9 | 3.07 | 37.8 / 37.7 | 0.026 / 0.005 | 53 min |
| 16–17 | 3.07 | 35.0 / 34.7 | 0.003 / 0.004 | 53 min |
| 24–25 | 3.07 | 35.2 / 34.6 | 0.007 / 0.008 | 52 min |
| **32–33** | 3.07 | **27.9 / 27.7** | 0.014 / 0.016 | 53 min |
| **40–41** | 3.07 | **27.9 / 28.4** | 0.025 / 0.025 | 53 min |
| **48–49** | 3.07 | **29.5 / 30.2** | 0.029 / 0.027 | 52 min |
| 56–57 | 3.07 | 32.7 / 33.4 | 0.024 / 0.022 | 53 min |
| 64–65 | 3.07 | 32.8 / 32.6 | 0.025 / 0.026 | 53 min |
| 71–73 | 4.06 | 36.3 / 35.8 / 36.3 | 0.017–0.019 | 42 min |

(rfn is relative to the layer's *full* output and shrinks with depth as the residual stream grows — compare within a depth
region; sqnr is the depth-independent column.)

**Reading.** At fixed bits the error is not flat with depth: layers 32–49 sit 7–10 dB below layers 8–25, then recover
partially by 56–65. The allocator spent its extra bits at the ends, where the error is already lowest. The same shape exists in
our GPTQ int4/int8 cook of the same model (Hessian-weighted int4 weight error, attention: 5.9e-4 in layers 3–15 → 2.6e-3 in
32–49 → 2.7e-3 in 50–65, peak layer 64; shared experts peak 32–49), so it is a property of GLM-5.3's weights — the mid-stack is
less compressible — not of the EXL3 converter. At 4.25 bpw (GPTQ int4 g128) it was invisible at the output; at 3.07 bpw it is
the dominant error term.

**Fix method, and why band-parallel makes it cheap.** `experiments/exl3_depth_recipe.py` reads the converter's per-layer lines,
flags base-bit layers whose sqnr is > 4 dB below the median of the base class (or rfn > 2.5× the class median), and writes an explicit
per-tensor recipe (`--recipe` for exllamav3) that lifts the flagged layers' routed experts by one bit and drops the same number
of the allocator's easiest above-base layers by one bit — budget-neutral (validated with exllamav3's own recipe loader: 3.211 →
3.211 bpw for the 13-up/13-down move; the 18-up/8-down variant is 3.341). Because every band restarts from the exact bf16
residual stream, only the bands containing changed layers are re-cooked (the budget-neutral move touches 5 of 10 bands here — the lifted mid-stack plus the end bands that give up a bit; a lift-only move touches 3 — ~7 h on as many nodes), then re-merged;
the two artifacts differ only in where the bits went. That is the experiment your EXL3 note asks for (does a measured allocation
beat the allocator?) at the depth axis, and it runs as a self-QC step at the end of the cook. Results will be appended here.

**Open question for the allocator:** it is a pure function of config + flags. A cheap per-layer sensitivity pre-pass (our
weight-space Hessian proxy from the GPTQ cook already ranks the layers correctly) could steer the depth allocation before the
first bit is spent.

## 5. Tooling details worth upstreaming

- **Exactly invertible fold** for `pollard-hf-smooth` at α = 0.5: `s_j = max|X_j| / max_i|W_folded[i,j]|`; we verified the
  unfold reproduces the recorded fold scales on a live layer (0.201–3.01 attention seam, 0.147–3.57 MoE seam, layer 8). This is
  what makes a smoothed-vs-unsmoothed control possible without keeping a second 1.5 TB copy.
- **Band-parallel exllamav3 conversion** (the converter's `ckpt/state.safetensors` is one F32 `[1, cols, hidden]` tensor per
  calibration row; inject the band-start residual stream, set `next_module_idx`, `--resume --max_module`): 78 layers in ~7 h on
  ten single-GPU nodes; smoke-tested by resuming through layer 0 from a fabricated 16-row checkpoint.
- **MTP draft precision lane**: the MTP module is quantized uncalibrated after the body, so 8-bit and 4-bit draft variants come
  from one cook; at TP4 the draft, not the body, decides the KV pool (12 GB vs 6.5 GB per node).
- **aarch64 build patch** for exllamav3 (DGX Spark / GB10, torch 2.13 + cu130): `notes/exllamav3-aarch64.patch` — excludes the
  x86-only CPU inference sources and adds a stub TU for their bound symbols; `__builtin_ia32_pause()` → `yield`; `cusparse.h`
  from the CUDA-13 pip layout via `CPATH`/`LIBRARY_PATH`.

## 6. Hardware profile — DGX Spark (GB10) cluster

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

## 7. Live-serving routing capture (TP4, 388K real tokens) — addendum

Section 1 measured routing on the calibration set. This is the same model routing **live traffic**: a sitecustomize hook on the
router (per rank, TP4 EXL3 3.2 bpw serving of GLM-5.3 on 4 × GB10, spec decode off, `--enforce-eager`) recorded per-layer expert
counts over 388,257 served tokens (agentic/tool, code, prose, long-context mix from our assistant's real sessions).
Raw per-rank JSON + the analysis table are in `experiments/data/glm53_live_routing/`.

Effective expert count `n_eff = 1/Σp²` and the share of routed mass held by the top-64 / top-128 experts, per router:

```
layer | decode tokens n_eff top64 top128 | prefill tokens n_eff top64 top128 | n_eff ratio decode/prefill
                     router0 |         — |   388257   20.9  99.9% 100.0% | nan
                     router1 |         — |   388257   36.3  99.7% 100.0% | nan
                     router2 |         — |   388257   37.2  99.5% 100.0% | nan
                     router3 |         — |   388257   38.2  96.4%  99.6% | nan
                     router4 |         — |   388257   67.3  86.5%  97.7% | nan
                     router5 |         — |   388257   52.5  87.1%  96.1% | nan
                     router6 |         — |   388257   66.2  82.7%  94.7% | nan
                     router7 |         — |   388257   60.6  87.0%  95.9% | nan
                     router8 |         — |   388257   85.9  79.3%  92.6% | nan
                     router9 |         — |   388257   49.4  87.7%  96.1% | nan
                    router10 |         — |   388257   87.8  74.7%  87.8% | nan
                    router11 |         — |   388257   81.0  78.4%  90.6% | nan
                    router12 |         — |   388257   69.2  83.3%  94.3% | nan
                    router13 |         — |   388257   72.9  82.0%  94.0% | nan
                    router14 |         — |   388257   76.0  82.6%  93.6% | nan
                    router15 |         — |   388257   81.8  82.0%  94.1% | nan
                    router16 |         — |   388257   51.4  90.4%  97.2% | nan
                    router17 |         — |   388257   46.0  90.4%  97.6% | nan
                    router18 |         — |   388257   70.4  84.0%  94.9% | nan
                    router19 |         — |   388257  101.6  75.2%  90.8% | nan
                    router20 |         — |   388257  101.0  70.9%  86.7% | nan
```

Reading: early routers are extremely concentrated on live traffic (router0 n_eff ≈ 21 of 256, top-64 = 99.9 %), loosening to
n_eff ≈ 70–90 by routers 8–15 (top-64 ≈ 75–83 %). The calibration-set picture in §1 under-states the early-layer concentration
seen in production. **Caveat:** the decode/prefill split column is empty — on the sparse-MLA backend the attention metadata lacks
`max_query_len`, so the hook could not classify steps; the hook has been fixed to use `num_decode_tokens` and the split will be
re-captured at the next TP4 window.
