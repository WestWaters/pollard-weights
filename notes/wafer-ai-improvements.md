# Pollard improvements mined from wafer-ai/gpu-perf-engineering-resources

Source: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources)
(credited in README Acknowledgements). These improve **Pollard itself** — they are NOT the Cerebras
wafer-scale planner (`pollard-pack`), which is a separate, unrelated thing.

Five workstreams, in build order:

## 3. Correctness/perf validation GATE  *(BUILT — cheapest, highest reliability ROI)*
The days-long EXL3 debugging happened because we had no gate that catches "weights decode ~0.99 but the
assembled model forwards to garbage." Fix: a standing verify gate for every convert, on every lane.
- **Decode round-trip vs source** (per tensor): reconstruct each quantized tensor, `cos` vs the fp16
  source weight; **fail loud** if any tensor < threshold. Judge quality by real reconstruction only —
  the convert's Hessian-weighted proxy metric is BANNED (see legacy/PROXY_ERR_BANNED.md; it stayed ~0
  while models were garbage). Verify against the ORIGINAL weight, not the in-place-mutated one.
- **End-to-end model check:** logits/next-tok-acc vs fp16 source on a fixed prompt — the assembled
  forward, not just per-tensor (per-tensor correct ≠ forward correct — hard lesson).
- **Compute-Sanitizer** (`--sanitizer`): wrap a forward in memcheck/racecheck/synccheck/initcheck to
  catch memory/race/uninit issues in kernels. (Source: NVIDIA Compute Sanitizer.)
- **Nsight roofline + CUTLASS measurement methodology** for kernel perf SLOs.
- Tool: `tools/pollard_verify.py` (standalone + importable gate).

## 1. MXFP4 / FP8 output lane for Blackwell  *(biggest capability add — BUILT)*
Pollard = "one allocator, many emitters" (GGUF/GPTQ/MLX/EXL3). Added an **MXFP4/FP8 emitter** targeting
Blackwell's native FP4/FP8 tensor cores (our sm_120 hardware). Tool: `tools/pollard_mx.py` (`pollard-mx`).
- NVFP4 (default, vLLM-validated) + MXFP4 (OCP MX, experimental — flagged at runtime).
- Reuses the measured-sensitivity allocator: cold bulk → FP4, measured-hot layers (+ optional every
  `down_proj`, the residual writers) → FP8, `lm_head` kept high-precision. Same hot/cold split as
  `pollard-export`, so the FP4 lane inherits the exact allocation the other lanes use.
- Emits a **compressed-tensors** checkpoint via `llm-compressor` (`QuantizationModifier(scheme=NVFP4|MXFP4)`)
  → `vllm serve` on the FP4 cores. `--plan-only` prints the recipe + avg bpw with zero deps; real emit is
  gated behind `llm-compressor` on the CUDA box (clean error if absent). Every build → `pollard-verify`.
- Smoke-tested: plan-only for both schemes, sensitivity-driven and ends-fallback allocation, on Mac (no CUDA).
- Still open / references for the runtime side (not the emitter): OCP MX/FP8 specs, NVIDIA Transformer
  Engine, Blackwell `tcgen05` / tensor memory MMA — for when we hand-write the FP4 GEMM (see #5).

## 2. AWQ + SmoothQuant into the allocator/preconditioning  *(ALREADY PRESENT — audited, no build)*
`pollard_smooth.py` (`pollard-smooth`) already does BOTH, and better than stock:
- **AWQ** saliency — reuses the imatrix (`<w>.in_sum2 / .counts` = mean-sq activation per input channel,
  the exact AWQ signal) for free; no second calibration pass.
- **SmoothQuant** migration — the per-channel diagonal scale that moves outlier magnitude off salient
  input channels into the adjacent tensor, folded as a mathematical identity (inverse folded into the
  paired weight/RMSNorm gain) → output is a normal GGUF that runs unchanged on CPU/CUDA/Metal/Vulkan.
- **Beyond stock:** α is grid-searched **per tensor** against a REAL block quantizer (not one global α),
  self-skips tensors that show no real gain, and every fold is canary-verified (random-input identity to
  fp tolerance) before it's kept. Seams D/B/A, dimension-checked (mismatches skipped, never forced).
- Verdict: winning path, do NOT reinvent. Only future lever = feed AWQ saliency into the *allocator's*
  bit budget too (not just preconditioning), measured, accept only on a win.

## 4. KV-cache quantization  *(calculator side ALREADY PRESENT; runtime side belongs in the engine)*
- `pollard-calc` already models the KV cache arch-aware (`kv_cache_bytes`): MLA (DeepSeek/GLM single
  latent) vs GQA/MHA (K+V), with a `kv_bytes` lever = 2.0 / 1.0 / 0.56 (f16 / q8_0 / q4). So Pollard
  already answers "what does an N-bit KV cache cost me at ctx C" — its actual lane (byte-economics).
- **KIVI**'s new idea = asymmetric per-token-K / per-channel-V *runtime* quant. That's an inference-engine
  concern (llama.cpp `--cache-type-k/-v`, vLLM `--kv-cache-dtype fp8` already ship it) — NOT Pollard's to
  reimplement. Pollard's contribution stays the calculator lever above + emitting the right engine flag.
- Only worth building if we ever ship our OWN runtime; until then, cite KIVI/CacheGen, don't rebuild.

## 5. Custom GEMM kernel references  *(when we hand-write kernels)*
- **CuTe / CUTLASS 3.x** (tiling/layouts/atoms), **DeepGEMM** (compact FP8 GEMM), **DeepEP** (MoE
  dispatch/combine). Reference for Pollard's own low-precision GEMM/MoE kernels.

Rule (Mario): regression/version checks are for the **repo as a whole, pre-push CI** — NOT folded into
any single lane/path. Do not bury validation logic inside a lane where it can break the convert.
