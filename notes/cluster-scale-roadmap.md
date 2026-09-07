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

## Phase 2 — the big cluster frontier (buildable; needs THEIR hardware to verify end-to-end)
- [ ] **Band-parallel / layer-streaming export** — models too big for one box (744B = 1.5TB BF16).
      `pollard-export` loads the whole model via `GPTQModel.load`; need each node owning a contiguous layer
      range, boundary hidden states handed off. (Storage egress was the wall-clock wall: ~118 MB/s/box.)
- [ ] **GPU-side layer-streaming sensitivity** — the GGUF sensitivity path needs `len(groups)×layers` full
      requantize+KL passes (156 at 744B over 1.5TB → its own guard refuses). Need a Hessian-weighted /
      shaped-noise estimator, one pass over ~1M calib tokens.
- [ ] **MoE-aware per-expert Hessians** in production GPTQ (router-replicated + token-floor/identity
      fallback + batched). Memory: 256×6144² fp32 = 38.7 GB/layer → two-set trick (+3 min/layer).
- [ ] **Both-width candidate emit + allocation-as-config** — write int4 AND int8 once, allocation is a
      config.json + re-pack, so A/B on the cluster is "a boot, not a cook."
- [ ] **compressed-tensors output for the GPTQ lane** (per-target mixed widths natively; pollard-mx
      already does this for FP4 — extend to the INT path).
- [ ] **Served-model A/B eval harness** — vLLM echo+logprobs perplexity, top-1, HumanEval+/MBPP+,
      long-context needle, spec-decode acceptance, mixed-workload throughput.

## Phase 3 — docs & hygiene
- [ ] **Unified-memory playbook** (`notes/`): page-cache drop (`posix_fadvise DONTNEED`), one GPU job per
      node, meta-device dtype (cast on meta before `to_empty` to avoid fp32 reserve), damped-Cholesky
      escalation. Covers DGX Spark/GB10, Grace-Blackwell, Apple.
- [ ] **Byte-accounting as a first-class report** — bytes-per-decoded-token IS the speed model on
      bandwidth-bound hosts (on GLM-5.3, int8 attention reads more bytes/token than routed experts).
- [ ] **Gate hygiene**: held-out/calibration overlap detection (158 of 162 held-out texts were in replay
      corpora) — exclude records whose reference completion hashes to a held-out text.
- [ ] **CUDA leak note**: on unified-memory hosts, `empty_cache()` under-reports; `cudaMemGetInfo` free
      falls — one process per layer (fresh context) is the workaround.
- [ ] **Spec-decode head note**: ship the model authors' STOCK MTP head with the new quantized body;
      retraining the head on quantized-body captures did NOT close the acceptance gap (1.80→1.36).

Rule: generic + robust first; hardware-specific quirks (like the GB10 None-guard) added as reports arrive.
