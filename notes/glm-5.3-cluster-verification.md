# GLM-5.3 on a 10× DGX Spark cluster — verification of the cluster-scale tools, and the gaps we filled

Companion to `glm-5.3-744b-routing-and-smoothing-data.md`. Your roadmap says cluster verification is the contributor's job;
this is the first instalment, measured on the deployment that produced the data in that note (GLM-5.3, TP8, GB10, int4/int8
GPTQ mix in production; EXL3 3.2 bpw cook in flight).

## 1. `pollard-calc` against the measured deployment

Run: `pollard-calc --config <GLM-5.3 config.json> --ram 121.6 --device unified --rambw 273 --kv-quant nvfp4 --ctx 900000`

| quantity | pollard-calc | measured | note |
|---|---|---|---|
| KV cache @ 900K, NVFP4 | 20.2 GB (22.4 KB/token) | ~37 GB (41 KB/token) | calc counts the MLA latent only; **the DSA indexer's per-layer key cache is missing** — about half the KV bytes on this model. fp8: calc ≈ 45 KB/token vs 57 measured |
| weights, q4 / q3 presets | 433.5 GB @ 4.60 bpw / 329.8 GB @ 3.50 | int4/int8 GPTQ mix 396 GB (experts 4.25 bpw) / EXL3 3.21 bpw ≈ 310 GB | GGUF presets vs our formats — fine as a ladder |
| fit tier @ 900K | 512 GB (4× Spark) YES for q3 | our TP4 EXL3 plan fits 4 nodes at ~250–350K ctx, not 900K | the KV undercount flips the verdict at long context |
| RAM-bandwidth ceiling | 9.2 tok/s (q4, one 273 GB/s node) | 58 tok/s single-stream without speculation on 8 nodes (≈ 7.3 per node-equivalent) | consistent once the TP pool is 8× the bandwidth — show the pool figure |
| build-time estimate (GB10) | GPTQ ~377 h · EXL3 ~2262 h | GPTQ: ~2 h wall on 10 nodes (~20 node-h); EXL3: 53 min per MoE layer at 384 × 2048 rows → ~66 GPU-h serial, ~7 h on 10 nodes | **20–35× pessimistic** for this class |

Suggestions: add the indexer key cache to `kv_cache_bytes` when the config carries `index_head_dim` / `indexer_types` (DSA
models: GLM-5.3, DeepSeek-V3.2 class); model build time as per-MoE-layer time × layers ÷ nodes with the GB10 constants above;
print the tensor-parallel pool's aggregate bandwidth ceiling beside the single-node one.

Earlier agreement worth keeping: on the int4/int8 production quant, `pollard-calc`'s byte-accounting ceiling was 73.4 tok/s
vs 73.7 measured with the n-gram drafter — the bytes-per-token model is right; it is the KV and build-time sub-models that need
the DSA/cluster inputs.

## 2. `pollard-export --shard-plan` contract vs the bands we ran

Our ten bands (8/8/8/8/8/8/8/8/7/7 layers) followed exactly the contract the plan prints: each node owns a contiguous band, the
last hidden state of band k is the input of band k+1, storage egress is the wall-clock wall (the 1.5 TB staging took longer
than the quantization). Two things the contract should say explicitly, both measured here:

- The hand-off state must be the **full-precision** residual stream captured in one streaming pass (not the quantized prefix's
  output) if bands are to run concurrently; the cost is exactness at n_bands−1 boundaries, which non-sequential GPTQ already
  pays at every layer. Per-layer output error at those boundaries was indistinguishable from interior layers in our EXL3 run.
- HF shard order is by tensor-name **string** sort (`layers.5` sits after `layers.49`), so a node's band touches shards spread
  across the whole file list; a pipelined stats pass has to wait per layer, and a band's shard set is ~30 files of 282, not a
  contiguous range.

## 3. Gaps filled in this PR

- `experiments/exl3_band.py` — the executing side of the contract for the **EXL3 lane**: `inject` a band-start residual
  stream into exllamav3's checkpoint format, `band` (template → inject → resume), `merge` (gather → final norm/head/MTP →
  compile). The GLM-5.3 run used these exact steps as per-node shell scripts (78 layers, 10 nodes, ~7 h + 1 h merge; a
  fabricated 16-row checkpoint resumed through layer 0 before the full run); this file is the consolidated, node-agnostic
  version of them — the `inject` code path is the one that ran, `band`/`merge` wrap the same converter invocations.
- `pollard-serve-eval --metrics <url>/metrics --accept-gen N` — speculative-decoding acceptance from vLLM's counters around
  real generations (accepted draft tokens per step, per-position acceptance). Teacher-forced PPL never decodes. On this model
  the head's acceptance moved single-stream speed by +16 %, and offline head metrics ranked six retrained heads *opposite* to
  serving — the served number is the only one that counts.
- `experiments/vllm_decode_routing_hook.py` — decode-vs-prefill routing capture from a live vLLM server (wraps the router's
  expert selection; CUDA graphs must be off during capture). Written against 0.28; import-tested inside the 0.28 image (installs, patches `FusedMoERouter.select_experts`);
  **not yet exercised on a live server** — our next fleet window runs it on GLM-5.3 and the e10 decode/prefill ratio for a 256-expert router goes in the data note.

## 4. Still open (cluster-only, on our list)

- Task-benchmark gates next to PPL/top-1/KL: pruning 40 % of experts cost +9 % PPL and −12…−16 HumanEval+ points here.
- A cluster hardware-profile schema (nodes × RAM × fabric bandwidth × per-node bandwidth) so `pollard-calc` can take a pool
  as first-class input rather than a summed `--ram`.
- Kimi-K3 routing profile: needs exactly this kind of cluster; on the list after the current cook.
