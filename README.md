# Pollard Weights

**Frontier models. Small hardware. No compromise.**

![License](https://img.shields.io/badge/license-Apache--2.0-blue) ![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Dependencies](https://img.shields.io/badge/dependencies-stdlib_only-brightgreen) ![GPU required](https://img.shields.io/badge/GPU_required-none-brightgreen)

![pollard-calc prediction vs an independent public benchmark](assets/k3_validation.png)

**Pollard Weights are models built for your machine's memory, not for a
bit-width chart.** Attention, routers and embeddings keep high precision; expert
FFNs carry the compression; hot layers keep more bits when you feed the builder
a measured routing profile — and the whole build is sized to your actual RAM
minus a working reserve. The output is a normal GGUF: it runs in stock
llama.cpp, Ollama, or LM Studio the minute it's built.

```bash
./install.sh                                     # tools + the llama.cpp runtime, one shot
pollard-calc --model Qwen/Qwen3-30B-A3B --ram 16 # what CAN this machine do
pollard-fit  --gguf model-f16.gguf --ram 16      # build the Pollard Weights for it
llama-cli    -m model-pollard.gguf               # run them
```

DRAM is provisioned today as if every weight deserves the same bits and every
byte must be resident. Neither is true, and the difference is measurable —
for AI models, and for the memory tiers under them (`notes/beyond-models.md`).

## What you get

- **`pollard-fit`** — the builder. Any GGUF in, a memory-fit build out:
  role-aware and depth-aware bit allocation computed for YOUR RAM budget,
  executed through llama.cpp's per-tensor quantization. `--plan-only` shows the
  full allocation and the exact command before anything is built.
- **`pollard-fit-dit`** — the builder for everything llama-quantize rejects:
  diffusion/video/image models in GGUF. Pure-Python per-tensor quantization
  (F32/F16/Q8_0/Q5_0/Q4_0 ladder) with the same protection-policy + byte-budget
  planning — validated end-to-end (build → load → coherent generation).
- **`pollard-calc`** — the planner. Any Hugging Face id, config.json, or GGUF
  on disk; computes the memory economics and the verdict for your hardware
  before you download a byte. No GPU required. Knows MoE, dense, and the newer
  hybrid **linear-attention / SSM** stacks (Qwen3.5-class) — it flags where an
  arch makes the estimate approximate or conservative (multimodal towers, MTP
  multi-token decode) instead of silently reporting a plain-dense number.
- **`pollard-experts`** — the routing report. Point it at an `experiments/e2`
  capture and it lists the experts your workload actually runs hot, per layer,
  with an honest coverage read: a load-balanced router touches nearly the whole
  pool, so you can't prune experts by topic — "hot" is a live residency signal,
  not a skip list. Emits a keep-list for a residency planner.
- **The runtime** — `install.sh` builds llama.cpp (Metal on macOS, plus the RPC
  backend for multi-machine runs) so the chain runs end-to-end from a fresh
  clone. Pollard builds are standard GGUFs by design: the entire llama.cpp
  ecosystem is their runtime.
- **The instruments** — harnesses that capture expert routing, measure reuse on
  your workload, and simulate hot-cache residency (`experiments/`). These are
  what turn the builder's allocation from heuristic to measured.
- **The design, with receipts** — every claim in this README carries its
  experiment in `notes/`, failures and retractions included.

## The planner in action

Hardware profiles are just numbers — it works the same for an NVIDIA DGX
Spark (`--ram 128 --rambw 273`), an RTX box, Apple Silicon, or a bare CPU
server.

**Worked example — bracketing the public "Kimi K3 on a CPU" demo from the
config alone** (the demo's precision and SSD speed were unpublished; at
q8→f16 assumptions the corrected floor brackets the measured 32.7 s/token):

```
architecture        : MOE  (896 experts/layer, top-16 + 2 shared)
total params        : 2,751.4B  -> 1,582.1 GB @ 4.6bpw
active per token    : 77.3B  -> 44.5 GB reads/cold-token
flash-stream floor  :   0.08 tok/s  (@ 3.5 GB/s sequential)
RAM-bandwidth ceil  :   2.70 tok/s  (@ 120 GB/s)
VERDICT: NEEDS BIGGER TIER — ~1,861 GB RAM for residency
```

The widely-shared K3-on-CPU benchmark measured **32.7 s/token (0.031 tok/s)**;
the corrected floor band at plausible precisions (f16→q8, 23–44 s/token)
brackets it. Treat calculator outputs as assumption-stated estimates, not
oracles — and check them, like the community checked ours (see Errata).

## Proof of concept: a 66 GB video model on a 16 GB Mac Mini

![H3 speed campaign on a 16GB Mac Mini](assets/h3_campaign.png)

MiniMax-H3 (33B video+audio DiT, ~66 GB native) running locally on an M4 Mac
Mini with 16 GB unified memory — 20-step, upscaled 1664×960 output:

| Milestone | Wall time | What changed |
|---|---:|---|
| First light (480²) | 61:31 | it runs at all |
| Step + resolution tuning | 38:30 | schedule economics |
| + compile + conditioning cache | 27:57 | fused Metal kernels, encode-once |
| 20-step + FirstBlockCache | 1:28:03 | quality mode: 7/20 steps skipped, lossless |
| **Full stack, pruned 4-bit, 2× upscale** | **49:46** | **faster than the first light at ~7× the pixels** |

Every row: same scene, same seed, A/B-comparable. Cross-architecture bonus:
our FirstBlockCache skip-ratio curve on Apple Silicon reproduces NVIDIA's
GB10 curve — first cross-arch datapoint for the technique.

## Not just for models that don't fit

Local inference is **bandwidth-bound**: tokens/sec ≈ memory bandwidth ÷ bytes
read per token. So every byte the quant mix doesn't read is time you don't
spend — which means a measured, machine-fit build makes models that *already
fit* **faster**, not just possible. Measured here: a pruned 4-bit build that
was both 4 GB *smaller* and ~25% *faster* per step than the naive 3-bit it
replaced. Smaller and faster are the same axis when bytes are the bottleneck.

And measured allocation buys **quality**, not just fit — at matched size
against the standard preset, on two very different corpora:

![quality at matched size](assets/kl_quality.png)

Details and the full evidence chain, negative results included, in
`notes/e12-both-legs-measured.md`.

## For GPU users (RTX / CUDA): measured expert placement

Big MoEs on consumer GPUs run with experts offloaded to system RAM
(`--cpu-moe` / `--n-cpu-moe N`) — but llama.cpp's built-ins choose *blindly*
(all experts, or the first N layers). `pollard-run` chooses by **measurement**:
which layers' experts actually run hot in the decode regime, from a routing
profile captured during real generation. The streams stay at full checkpoint
precision — placement is lossless by construction.

```bash
pollard-run --gguf Qwen3-30B-A3B-Q3_K_M.gguf --profile qwen3-30b-a3b --vram auto --launch
```

`profiles/` ships measured heat for supported models (contribute yours —
`experiments/README.md` shows the capture). Measured results, 16 GB Apple
Silicon — and note the shape: **measurement's edge grows as memory gets
scarcer** (at looser budgets, blind first-N accidentally overlaps the
measured split; at tight budgets, knowing wins big):

![placement benchmark](assets/placement_bench.png)

At the tight budget, measured placement won **every paired run** (+21% mean
vs blind). Variance is real (shared machine); replication on your hardware is
exactly what `profiles/` wants.

## Across machines (pool their RAM)

A build too big for one box runs across several — **any number, not just two.**
llama.cpp has its own clustering (the RPC backend), so this needs no vLLM (and
keeps the GGUF you built). `install.sh` compiles it in (`-DGGML_RPC=ON` +
`rpc-server`).

```bash
# on every OTHER machine (as many as you have):
rpc-server -H 0.0.0.0 -p 50052
# on the main machine — comma-separate every peer; layers split across all, RAM pooled:
llama-cli -m model-pollard.gguf --rpc host2:50052,host3:50052,host4:50052
```

The `--rpc` list takes as many peers as you add; total usable RAM is the sum
across all of them, so you scale by adding boxes. It is pipeline-parallel (each
machine holds a slice of the layers, activations hop between them at layer
boundaries) — simpler than vLLM's tensor-parallel and a touch slower per token,
but it pools the memory, which is the point when the model doesn't fit one box.
Two DGX Sparks (128 GB each) hold a 167 GB Q4 build this way that neither could
alone; add a third and a ~250 GB build comes into reach. A fast link between
them (their ConnectX/QSFP, or plain 10GbE to start) carries the activations.

**Every vendor works** — an RTX box, a GB10 / DGX Spark, an AMD Radeon, an Intel
Arc, and an Apple Silicon Mac can all join the same cluster. Each peer runs
`rpc-server` built for its own accelerator, and `install.sh` auto-detects which:
Metal (Apple), CUDA (NVIDIA — RTX, GB10), HIP/ROCm (AMD), SYCL (Intel), or
Vulkan as a cross-vendor fallback that runs off the graphics driver alone; CPU
otherwise. Force one with `POLLARD_GPU=-DGGML_VULKAN=ON ./install.sh`. Throughput
tracks the slowest peer and the link, but the RAM adds up regardless of who made
the chips.

## Use it with your agent

The repo is written to be agent-executable: hand this README plus
`experiments/README.md` to Claude Code, Hermes Agent, Codex, Cursor, or any
coding agent, and ask it to run the calculator on a model you're considering
or to reproduce the routing measurements on one you already have.
Everything is argparse'd, stdlib-first, and states its expected inputs. Example:
"clone this repo and tell me what Kimi-K3 would do on my machine" is a complete
instruction.

## The method (what's in `notes/`)

1. **Measure the hardware floor** — cache-bypassed flash curves; block size is
   a 20–40× lever before any ML begins (`notes/e0-hardware-floor.md`).
2. **Compute the model's byte-economics** — total vs active params, expert
   pool, cold-token reads (`pollard-calc` automates this).
3. **Find the reuse** — routing concentration (MoE), step redundancy
   (diffusion), depth redundancy (both). This is where "impossible" becomes
   "viable": the working set your workload actually keeps hot is far smaller
   than the file, and the difference is the RAM you don't need to buy.
4. **Fit the quant to the machine** — measured per-layer sensitivity → mixed
   precision summed to your RAM budget. Not "Q4 because Q4 exists."
5. **Verify or it didn't happen** — seed-matched A/B for every optimization;
   a null test for every cache.

Experiment log: `notes/` — from the dense-sparsity verdict that killed the
naive version through the fitting reframe that became `pollard-fit`.

## Roadmap

- **Routing-reuse index** — the online measurement of expert-reuse locality;
  turns the calculator's floor/ceiling band into a point estimate for YOUR
  workload. The number nobody publishes.
- **Per-expert residency** — today the builder allocates bits per layer and
  role; the next runtime step pins and streams at individual-expert
  granularity, with custom Metal/GPU kernels for the hot path.
- **Depth-collapse** — post-training layer skipping priced in expert-fetches
  saved (E9); a depth-exited pass doubles as a free speculative drafter.
- ~~Video-model harnesses~~ — shipped: `recipes/minimax-h3-16gb.md`, the
  full H3-on-16GB campaign as a reproducible, agent-executable recipe.

## What this is not

- Not re-hosted weights: harnesses + method + measurements only.
- Not benchmarketing: contended runs are reported separately, approximations
  are labeled, and retractions stay in the log.

## Acknowledgements

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (ggml-org) — the runtime
  and quantization machinery `install.sh` builds and `pollard-fit` drives.
- **[MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-H3)** — open weights for the
  H3 video model used in the proof-of-concept campaign, and
  **[molbal](https://huggingface.co/molbal/MiniMax-H3-GGUF)** for the pruned GGUF
  builds its final row runs on.
- **[NVIDIA Sol-Engine](https://github.com/NVlabs/Sana/tree/sol-engine)** — the
  FirstBlockCache technique in the campaign's quality mode, via
  **[drowzeys](https://github.com/drowzeys)**' single-GPU ComfyUI ports.
- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** — the pipeline the video
  campaign ran on.
- Intellectual lineage: Apple's *LLM in a Flash* (arXiv 2312.11514) for
  flash-resident weights on constrained devices, and P. J. Denning's working-set
  theory (1968) — this project builds the model-weight-specific instruments those
  ideas point toward.

## Errata

- **2026-08-08 — K3 expert dimensions were 2× too large.** The formula ignored
  `routed_expert_hidden_size` (K3 runs experts in a half-width latent space:
  3584 vs hidden 7168), doubling total params (5.48T → correct **2.75T**),
  active bytes, and the residency tier (3.7TB → correct **~1.9TB**). Found by
  community review within a day of launch — exactly the culture this repo asks
  for. The original README also overstated the demo comparison as "validated";
  it is a worked example with stated assumptions, and is now labeled as one.

## License

Apache-2.0.
