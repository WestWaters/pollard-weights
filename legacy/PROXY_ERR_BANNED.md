# BANNED: `proxy_err` (exllamav3 convert metric) — do NOT use, do NOT re-introduce

**Decision (Mario, 2026-09-06):** stop using `proxy_err` for anything. It is not a correctness
signal and it cost us days. This file is the record so we never circle back to it.

## What it is
`proxy_err` is **upstream exllamav3's** internal convert metric, printed per tensor during
`exllamav3.conversion.convert_model`. It is defined as the Hessian-weighted relative error:

```
proxy_err = trace(E · H · Eᵀ) / trace(W · H · Wᵀ)      # E = W - W_quant,  H = captured Hessian
```

It is **not Pollard code** — we consume the pip package, so we cannot delete it from their source.
This ban means: **Pollard never reads it, never gates on it, never allocates from it, never reports it
as a quality signal.**

## Why it is banned — it lies (measured)
- It stayed ~0 while whole models were garbage. Concrete case: the deterministic **layer-1** break at
  4bpw (Qwen2.5-3B). Layer 1's MLP-input Hessian had a massive-activation channel (H diag max
  **125,987**). That one outlier dominates the `trace(W·H·Wᵀ)` denominator, so `proxy_err` reported
  **0.0001** (perfect) while the layer's real forward was garbage (**sqnr −14**, model PPL ~3090). The
  outlier is preserved; every normal channel is under-quantized; the H-weighted metric can't see it.
- As an **allocation** signal it also loses: EXL3 `proxy_err` allocation measured **14.83–14.97** vs
  Pollard budgeted **14.36** (see `benchmarks/Ref-pipeline.md`). Magnitude/H-proxy misranks tensors.

## What to use instead (real reconstruction, activation-truthful)
- **`pollard-verify`** — per-tensor decode-vs-fp16-**source** cos + assembled **end-to-end** forward
  (next-tok-acc / PPL). This is the gate. It never uses `proxy_err`.
- **Per-layer `sqnr`** from the convert's real forward on calibration activations (the number that
  actually caught layer 1: +28 good vs −14 broken).
- **Official `eval/ppl.py`** on wiki2 for the whole-model number.

## Rule
If you ever see `proxy_err` in a decision, an allocation, or a "looks fine" claim — it's wrong by
construction. Delete that path. Judge convert quality by real reconstruction only.
