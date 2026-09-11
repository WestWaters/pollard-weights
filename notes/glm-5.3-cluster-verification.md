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

- `pollard-exl3-band` — the executing side of the contract for the **EXL3 lane**: `inject` a band-start residual
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
- Kimi-K3 routing profile: out of reach even for this cluster (5.5T parameters; a 2-bit build exceeds the pool's 1.2 TB, so
  only NVMe-streamed capture at a fraction of a token/s). The next same-class dataset we can produce is Hy4-preview (770B/49B
  active, 256 experts, DSA + MLA, hyperconnections) — a second 750B-class family for the routing/outlier tables.

## 6. The EXL3 lane, measured end to end on GLM-5.3 (2026-09-08)

The 3.2 bpw EXL3 body from the band-parallel run (`-b 3.2 -hb 6 -mb 8 -hq`, exllamav3's budgeted allocator, SmoothQuant folded,
our 384 × 2048 in-domain rows) served on **four** GB10 nodes at TP4, against the same model's Pollard-method GPTQ int4/int8 build
on **eight** nodes. Same probes, same day, same fleet. (The serving side needed the runtime fixes described in the companion
serving-gotchas note before any of this could be read.)

| | EXL3 3.2 bpw, 4 nodes | GPTQ int4 experts / int8 attention, 8 nodes |
|---|---|---|
| artifact | 292 GB, 3.21 bpw | 396 GB, experts 4.25 bpw |
| live perplexity, 6 fixed held-out texts (4,079 tokens) | **4.831** | 4.82 – 4.84 |
| HumanEval / HumanEval+ pass@1 (greedy, EvalPlus) | 0.957 / 0.927 | 0.963 / 0.945 |
| MBPP / MBPP+ pass@1 | **0.979 / 0.841** | 0.971 / 0.828 |
| correctness probe (5 checks incl. 4- and 8-way concurrent) | all passed | all passed |
| speculative draft | in-checkpoint MTP layer, EXL3 8-bit, k=3: **2.23 accepted/step**, 74 % draft accept | separate GPTQ int8 layer-78 draft, k=5: 1.58 – 1.84 |
| single-stream decode, mixed workload | 24.0 tok/s | 40.0 tok/s |
| 4-stream aggregate | 58.2 tok/s (14.6 per node) | 85.2 tok/s (10.7 per node) |
| KV | fp8, 396K tokens at gmu 0.84 (nvfp4 unsupported by the EXL3 sparse-MLA runtime) | nvfp4, 900K at gmu 0.81 |

Readings for the lane:
- **Quality parity at 25 % fewer bits per expert weight and half the nodes.** Perplexity equal to the second decimal; HumanEval+
  −1.8 points; MBPP+ +1.3 points. This is the allocator's own allocation with no hand-tiered recipe, i.e. the configuration your
  measurement said wins; on this model it holds at 744B.
- **The allocator's depth heuristic is still wrong here, and it did not matter for these gates.** §F3 of the data note showed the
  measured per-layer error peaking mid-stack (28 dB at layers 32–49 vs 35–38 dB elsewhere at 3 bits) while the allocator spends its
  extra bit at the ends. The budget-neutral second pass (`exl3_depth_recipe.py plan`: 19 moves, 0 bytes, predicted −25 % output
  noise) is therefore optional for quality on GLM-5.3. It remains the cleanest experiment on measured-vs-heuristic allocation at
  fixed size; we will run it if a fleet window opens, and report either way.
- **The in-checkpoint MTP layer at 8 bits is the best draft we have measured on any body** (2.23 accepted/step at k=3). Quantizing the
  draft with the body, by the same method, beats a separately quantized draft — consistent with §5.
- **Per-node efficiency favours the 3-bit lane** (+37 % aggregate tokens per node); absolute single-stream speed favours more nodes.
- Practical: exllamav3 pads `out_features` to multiples of 128 and stores the MTP side model with an index that can point at the
  wrong shard; both cost us boots and are written up, with the fixes, in the serving-gotchas note and `tools/exl3_fix_mtp_ehproj.py`.

## 7. Serving the Pollard body on vLLM 0.29.0 + b12x, TP8 × 8 GB10 — deploy numbers (2026-09-09)

The Int4/Int8-mix GPTQ body from this method (experts int4 g128, attention/shared int8, 369 GB text-only repack) plus the
int8-quantized MTP layer as a separate drafter, served on our vLLM 0.29.0 + b12x merge (fork `local-inference-lab/vllm` glue,
RoCE one-shot all-reduce, sparse-MLA/indexer kernels). Same mixed-workload bench as our production gate (300 s at concurrency 1
and 4); correctness probes passed on every leg.

| config | C1 tok/s | C4 tok/s | MTP accepted/step | notes |
|---|---|---|---|---|
| spec off, fp8 KV 600K | 24.1 | 58.4 | — | engine baseline |
| + RoCE all-reduce | 32.4 | 67.5 | — | +34 % / +16 % |
| + MTP k3 (int8 layer-78 drafter) | 49.8 | 85.3 | 1.82 (60.6 %) | the int8 drafter works natively |
| + NVFP4 KV 900K + k5 schedule | 51.0 | 85.1 | 1.73 (63.6 %) | production shape; pool 1.04M tokens |
| fp8 KV 600K variant | 43.8 prose | 86.1 | 1.74 | **prefill +11 %** (1007/942/902 tok/s at 42K/92K/132K vs 906/861/830 on NVFP4 KV) |

Prose single-stream (10 prompts, 250 out, wall incl. TTFT): 42.8–43.8 tok/s; concurrency-8 aggregate 121–129 tok/s. Prefill is
the remaining gap to the best community number on this hardware (~1.3–1.4K tok/s); it is kernel-bound (sparse-MLA/indexer prefill),
not scheduler-bound — Marlin beats the Triton WNA16 MoE kernel on GB10 for both prefill and decode, and NCCL channel count,
indexer budget, AOT compile and chunk size were all inert. Relevant to the method: the deploy math holds — the 3.2–3.4 bpw-class
body with an int8 drafter gives a 51 tok/s single-stream assistant at 900K context on eight 128 GB nodes.
