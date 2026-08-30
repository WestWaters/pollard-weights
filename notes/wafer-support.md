# Wafer-scale (Cerebras) support — `pollard-pack`

Pollard adds **wafer-aware capacity planning** for Cerebras-class SRAM machines (WSE-3 / CS-3, WSE-3 Turbo / CS-4). It is an **offline planner**, not a runtime — see *Scope* below.

## What it does

Reuses Pollard's hot-set/cold-bulk ranking to answer the two questions that decide cost on a wafer:

- **How many wafers does my model need?** (resident footprint ÷ 44 GB SRAM/wafer)
- **How many wafers does sensitivity-ranked expert pruning save?** (REAP-style — drop the least-important experts, ~50% at quality parity per Cerebras)

It emits an **actionable per-layer expert-prune plan** you feed into your own REAP/compression pipeline.

## Usage

```bash
# forecast wafers for a model, and what pruning 50% of experts saves
pollard-pack --gguf your-moe-f16.gguf --target wse3t --prune-experts 0.5

# write the actionable prune plan (per-layer drop counts + method)
pollard-pack --gguf your-moe-f16.gguf --emit-plan plan.json

# resolve WHICH experts to drop, from a router-usage profile you collect on your calib set
#   router-usage.json = {"<layer>": [per-expert activation score, ...], ...}
pollard-pack --gguf your-moe-f16.gguf --expert-usage router-usage.json --emit-plan plan.json
```

Targets: `wse3` (CS-3, 21 PB/s) · `wse3t` (CS-4 / Turbo, 43.2 PB/s). Both 44 GB SRAM/wafer.

## Example (Qwen3-30B-A3B, CS-4, prune 50%)

```
resident:  61.0 GB -> 33.0 GB   (-46%)
WAFERS:    2 -> 1   (save 1)          <- the cost lever
```

## The model (honest)

- On-wafer weights are **16-bit resident** (FP16/BF16/cbfloat16 = 2 B/param) — source quantization does **not** reduce the wafer footprint. The only lever is **sparsity/pruning** (the cores skip zeros).
- **Footprint ∝ total params** → wafers. **Throughput ∝ active params/token** (a MoE reads only its routed experts). Verified by published numbers: gpt-oss-120B (5.1 B active) runs *faster* than dense Llama-70B despite 117 B total.
- The one quantified wafer lever is **expert pruning** (whole experts, REAP-style). Pollard's contribution is **ranking which experts to drop**. **MoE-only** — dense models get no wafer win, and the tool says so.
- Single-stream tokens/s is **layer-depth bound**, not bandwidth bound, so the tool deliberately forecasts **capacity, not a t/s number**.

## Scope

This is a **capacity forecast + prune plan**, not a benchmark. Cerebras inference is a **closed 16-bit stack** with no user low-bit path; a Pollard-quantized GGUF does **not** run on a wafer. Real deployment (applying the drop-list, measuring t/s) is a **Cerebras partnership track**. The planner lets you size and plan that conversation with real numbers.
