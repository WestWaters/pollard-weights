# E7 — the method works; Laguna hits a llama.cpp Metal bug

_2026-07-30. llama.cpp b2f2216 (latest master), 16 GB M4._

## The isolating experiment

Laguna-XS-2.1 refused to compute on Metal with `Q3_K` or `IQ3_XXS` experts, at every offload level
from `ngl 10` to `ngl 99`. Two explanations were possible: llama.cpp's Metal MoE path can't handle
those types at all, or something about Laguna specifically. Testing a *different* MoE separates them.

`ibm-granite/granite-3.0-3b-a800m-instruct` — 3.37B total, 800M active, 40 experts:

| model | expert quant | size | Metal | gen tok/s | coherent |
|---|---|---|---|---|---|
| granitemoe 3B | Q4_K (baseline) | 1.92 GiB | ✅ | 118.1 | ✅ |
| **granitemoe 3B** | **Q3_K (experts only)** | **1.42 GiB** | **✅** | **106.6** | **✅** |
| Laguna-XS-2.1 | Q2_K (experts only) | 10.80 GiB | ✅ | 46.6 | ❌ empty |
| Laguna-XS-2.1 | Q3_K (down) + Q2_K | 11.79 GiB | ❌ compute error | — | — |
| Laguna-XS-2.1 | IQ3_XXS (experts only) | 12.40 GiB | ❌ compute error | — | ✅ (CPU only) |

**Conclusion: the Metal MoE path handles `Q3_K` experts fine. Laguna is the outlier.**

## The method is validated

Expert-only quantisation on granitemoe: **1.92 → 1.42 GiB, a 26% cut, with no quality loss and a
~10% speed cost.** Output comparison on `def fib(n):`:

- Q4 baseline: *"The Fibonacci sequence is a series of numbers in which each number is the sum of the
  two preceding ones…"* (prose only)
- Expert-only Q3_K: *"Here is a Python function that generates the nth Fibonacci number:
  ```python def fib(n): if n <= 0: return "Input should be a positive integer…"* (actual code)

So the technique — quantise `ffn_{down,gate,up}_exps` hard, leave attention, the router
(`ffn_gate_inp`), shared experts, layer-0 dense and the output head alone — transfers to a second
architecture and delivers most of the size saving, because experts are ~98% of the bytes.

## The bug, isolated well enough to report

- **Model:** `poolside/Laguna-XS-2.1` (arch `laguna`), GGUF
- **Symptom:** `llama_decode` → `Error: Compute error`, `res = -3`, on Metal
- **Trigger:** expert tensors (`ffn_*_exps`) at `Q3_K` or `IQ3_XXS`
- **Not the trigger:** memory (fails at `ngl 10`, ~3 GB), build age (fails on latest master, 86
  commits newer), residency sets (`GGML_METAL_NO_RESIDENCY=1` doesn't help), or the MoE path in
  general (granitemoe with `Q3_K` experts works)
- **Works:** same model with `Q2_K` experts; the original `Q4_K_M`
- Laguna has unusual features that are plausible suspects: sigmoid gating with an `exp_probs_b`
  bias tensor, mixed SWA/global attention in a 3:1 ratio, per-layer rotary scales, 256 experts with
  `moe_intermediate_size` 512.

## Practical upshot for a 16 GB Mac

- Laguna-XS-2.1: no coherent + Metal-capable configuration exists today. The coherent
  `IQ3XXS-experts` build (12.40 GiB, smaller than bartowski's 13.30 GiB IQ3_XXS) runs correctly on
  CPU at ~1.6 tok/s, and would run fine on hardware with more GPU headroom.
- Other MoEs: the method works. Apply it to Qwen3-30B-A3B, GLM, MiniMax, etc.
