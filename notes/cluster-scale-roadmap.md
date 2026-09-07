# Cluster-scale & any-hardware roadmap (from the GLM-5.3 744B run)

Source: an outside expert quantized **GLM-5.3 (744B MoE)** on a **DGX Spark / GB10 cluster** via the
vLLM/GPTQ path (gist bot-lab-21/aa22501...). Goal: Pollard works on **any cluster/box/device** — we build
generic + robust, and add hardware-specific handling as reports come in (can't pre-buy every box).

**Praise (validation):** SKILL.md let an outside agent apply the method in ONE pass; `pollard-calc`
predicted 73.4 tok/s vs 73.7 measured; `pollard_gptq` core "clean, textbook-faithful"; `pollard-experts`
processed 307K routing rows fine.

## Phase 1 — buildable + verifiable now (robustness / small gaps)  — DONE (2026-09-07)
- [x] **pollard-health**: GB10/aarch64 crash fixed — `_fmt()` guards every N/A field (nvidia-smi returns N/A
      for power/temp on GB10); `_aarch64_temp()` reads SoC thermal zones (EC hard-powers off ~98°C).
- [x] **pollard-calc**: the user's own `--ram` now appears as a tier in the fit ladder (marked "your rig",
      deduped against listed tiers); over-tier note says N boxes/Sparks are ONE pool via TP (vLLM) / --rpc
      (llama.cpp); `--kv-quant nvfp4` added (0.5 B/elem) alongside f16/q8/q4.
- [x] **pollard-export**: projection matchers are now arch-agnostic (`ATTN_PROJ`/`FFN_PROJ`) — cover MLA
      (`q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`, layernorms excluded) and DeepSeek/GLM plural
      `shared_experts`; `detect_mla()` prints a note. Self-tested against all the name variants.
- [x] **pollard-gptq**: damped-Cholesky robustness — dead channels now `diag(H) ≤ 1e-10·mean` (not just ==0);
      Cholesky escalates damping ×10→×100 then falls back to fp64, raising a clear error only if all fail.
      Tested on rank-deficient + massive-activation Hessians. (Was crashing the 16384-dim o_proj layer.)

## Phase 2 — the big cluster frontier  — DONE (2026-09-07; generic+robust, cluster verification is theirs)
- [x] **Band-parallel / layer-streaming export** — `pollard-export --shard-plan N --bf16-gb G` prints the
      contiguous band each node owns + byte budget + the boundary-handoff contract (offload path per band,
      last hidden state handed to the next node). Storage egress flagged as the wall-clock wall.
- [x] **GPU-side layer-streaming sensitivity** — `pollard-probe --stream`: ONE forward pass, Hessian-diagonal
      estimator (Σ ΔW²·E[x²]) scores every group at once. Same ranking as perturb+KL, no layers×groups sweep.
      Tolerates unused/pruned modules. Unit-tested (flags the amplified layer; monotone noise curve).
- [x] **MoE-aware per-expert Hessians** — per-module hooks already give per-expert H; added a **token floor**
      in `gptq_quantize` (n_tokens < cols → diagonal H, drops unreliable off-diagonal). Threaded through both
      the sequential and collect-Hessians paths. Tested cold vs warm expert.
- [x] **Allocation-as-config** — `pollard-export` writes `pollard-allocation.json` beside the checkpoint
      (full dynamic bit map + metadata), so re-A/B'ing an allocation is a re-pack, not a re-cook.
- [x] **compressed-tensors output for the INT path** — `pollard-mx` gained `W4A16`/`W8A16` schemes (+`--gptq`
      body via GPTQModifier, `W8A16` protect); per-target mixed widths natively, runs on any vLLM GPU.
- [x] **Served-model A/B eval harness** — `pollard-serve-eval` (stdlib only): teacher-forced PPL, top-1
      agreement, KL(base‖cand) against any OpenAI-compatible endpoint (vLLM/SGLang). Tested vs a mock server.

## Phase 3 — docs & hygiene  — DONE (2026-09-07)
All consolidated into **[unified-memory-playbook.md](unified-memory-playbook.md)** (one doc, not scattered):
memory-pressure modeling + page-cache drop, one-GPU-job-per-node, meta-device dtype, damped-Cholesky (now
implemented — §4), CUDA free-memory under-report + one-process-per-band workaround, band-parallel export,
byte-accounting as the speed model, served-model A/B, calibration/held-out overlap (gate hygiene), and the
spec-decode "ship the stock MTP head" note.

Rule: generic + robust first; hardware-specific quirks (like the GB10 None-guard) added as reports arrive.
