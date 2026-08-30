---
name: pollard-weights
description: >-
  Quantize any LLM to a memory-fit GGUF (and vLLM/SGLang GPTQ) the RIGHT way with
  Pollard Weights. Use this whenever asked to quantize / shrink / "make a Pollard
  build" of a model, or to plan its fit on a machine or wafer. It routes dense vs
  MoE models down different paths — using the wrong one wastes hours.
version: "0.3"
---

# Pollard Weights — agent skill

Pollard makes a frontier model fit a given machine's memory by **deciding where the
bits go** instead of painting every weight the same. This skill tells an agent the
correct path for *this* model so results are good and no time is wasted.

## STEP 0 — detect the model type FIRST (this decides everything)

```bash
pollard-calc --model <hf-id-or-gguf>      # prints arch: "dense" or "moe", params, layers, KV
```
- **dense** (Llama, Qwen2.5, Gemma, Mistral, …)  → the **imatrix** path.
- **MoE**   (Qwen3-A3B, Mixtral, DeepSeek, Ling, gpt-oss, …) → the **automap** path.

> **Rule: `imatrix = dense`, `automap = MoE`.** The tools ENFORCE this — `pollard-automap`
> and `pollard-sensitivity` refuse a dense model (pass `--allow-dense` only for research).
> Running the measured-KL sweep on a dense model is a multi-hour no-op; don't.

---

## DENSE path — imatrix-guided K-quants

The win on a dense model is the **importance matrix**, not per-layer reallocation
(measured: the KL knapsack does NOT beat uniform on dense — no expert redundancy).

```bash
# 1. f16 GGUF (if starting from HF)
python convert_hf_to_gguf.py <hf-dir> --outtype f16 --outfile model-f16.gguf
# 2. imatrix from a MIXED-domain calib (prose + code; not one register)
llama-imatrix -m model-f16.gguf -f calib.txt -o model.imatrix -ngl 99 --chunks 100
# 3. build the ladder you want to ship (imatrix-guided). Pick sizes for the audience:
llama-quantize --imatrix model.imatrix model-f16.gguf model-IQ4_XS.gguf IQ4_XS
llama-quantize --imatrix model.imatrix model-f16.gguf model-Q6_K.gguf    Q6_K
#    ...or let pollard-fit pick the best quant for a RAM budget:
pollard-fit --gguf model-f16.gguf --ram 16 --imatrix model.imatrix --out model-pollard.gguf
```
Ship a **ladder** (≥3 files, e.g. IQ3_S / IQ4_XS / Q6_K) so the HF repo's quant +
hardware-compatibility sidebar populates (it is one row PER GGUF FILE, not from the card).
Do **not** run `pollard-sensitivity` on a dense model.

Optional flagship extreme rung: an **automap 1-bit trellis mix** (the "gold-card")
*is* automap-on-dense — allowed only with `--allow-dense` (see below).

---

## MoE path — automap measured expert-allocation

This is where measured allocation genuinely beats uniform (crush cold experts, protect
the hot set). `pollard-automap` emits the 3-bar build (uniform-low / PollardMix / uniform-high).

```bash
# 1. f16 GGUF + imatrix — MoE NEEDS a DIVERSE calib (prose+code+varied) or rare experts
#    never route and get NO importance data. Use --chunks 200+.
llama-imatrix -m moe-f16.gguf -f calib_diverse.txt -o moe.imatrix -ngl 99 --chunks 200
# 2. FULL tensor list (dry-run at a non-low type so it can't abort mid-list):
llama-quantize --dry-run moe-f16.gguf x.gguf Q6_K > tensors.txt
# 3. automap: emits the MoE recipe (crush ffn_*_exps, protect router/ffn_down_exps/shared/attn),
#    AUTO-PINS any imatrix-uncovered expert to q6_K so the build can't hard-fail.
pollard-automap --tensors tensors.txt --model moe-f16.gguf --imatrix moe.imatrix --out build.bat
# 4. run build.bat  -> uniform-IQ1 / PollardMix / uniform-IQ2 + PPL each
```
**Watch automap's output:** if it reports "PINNING <many> uncovered tensor(s)", the
imatrix under-covered experts — the mix will BLOAT. Re-run the imatrix with a larger,
more diverse calib (more experts routed) before trusting the build.

**Accept-the-Mix gate:** keep it only if PollardMix ≤ uniform-low size + ~8–10% AND it
beats uniform-low on PPL/KL/top-1. If it drifts toward uniform-high size, reject and
tighten the crush.

---

## Runtime targets (where the build will actually run)

- **llama.cpp / ik_llama.cpp / Ollama / LM Studio** — the GGUF above runs as-is. The
  trellis `*_KT` quants need `ik_llama.cpp`; K-quants run in any recent llama.cpp.
- **vLLM / SGLang** (GPU servers, DGX-Spark crowd) — GGUF/trellis does NOT run there.
  Export a **GPTQ** checkpoint instead:
  ```bash
  pollard-export --model <hf> --sensitivity model.sensitivity.json --calib calib.txt --out ./out
  vllm serve ./out --quantization gptq          # 4/8-bit dynamic mix, Marlin-accelerated
  ```
  Constraints (enforced): bits ∈ {4,8} only (Marlin), never split bits inside a fused
  group (q/k/v share; gate/up share), `group_size 128`, `desc_act False`. SGLang mixed-bit
  is fragile → `--uniform` for SGLang. **vLLM reserves a large KV pool up front** — a model
  that fits in GGUF can OOM in vLLM; budget weights + KV + activations, run one load test.
- **Cerebras wafer** — `pollard-pack --gguf model.gguf --target wse3t` forecasts
  wafers-per-model + a sensitivity-ranked expert-prune plan. Capacity PLANNER only
  (16-bit resident, forecast not benchmark); MoE-only lever.

## Optional — abliteration (opt-in, off by default)

`pollard-abliterate --model <hf> --harmful A.txt --harmless B.txt --out ./abl` removes the
refusal direction on the FP16 weights BEFORE quantizing (composes with any build). It costs
some coherence — measure the PPL/KL delta vs the un-ablated build before shipping.

## Guards & gotchas (why runs fail or waste time)

- **Dense + sensitivity/automap = REFUSED** (multi-hour no-op / wrong tool). Use imatrix.
- **MoE + short/prose-only imatrix** = uncovered experts → build hard-fails or bloats.
  Diverse calib, `--chunks 200+`. automap auto-pins the uncovered ones as a safety net.
- **HF sidebar empty** = the repo has too few GGUF files. Ship the ladder (1 row/file).
- **Start from f16/bf16**, never a re-quantized file (compounds loss).
- **New/newer arch fails to quantize** = rebuild the runtime llama.cpp (`git pull && cmake --build`).
- **Bench alone** — a parallel eval/build contaminates tok/s.

## Full feature index (keep this list in sync with the tools)

| tool | use it for | model type |
|---|---|---|
| `pollard-calc` | pre-flight: arch, params, "will it fit?" (KV-aware) | any |
| `pollard-fit` | memory-fit build for a RAM budget (K-quant ladder / MoE knapsack) | any |
| `pollard-automap` | MoE expert-allocation mix + auto-pin uncovered experts | **MoE only** |
| `pollard-sensitivity` | measured per-group KL profile (the knapsack input) | **MoE only** |
| `pollard-probe` | CHEAP in-process sensitivity (no GGUF/imatrix, any box) | any (research) |
| `pollard-pack` | Cerebras wafer capacity + expert-prune plan (forecast) | MoE (lever) |
| `pollard-export` | vLLM/SGLang GPTQ (4/8 dynamic) checkpoint | any |
| `pollard-abliterate` | optional refusal-direction ablation (pre-quant) | any |
| `pollard-eval` | trajectory-divergence quality metric | any |
| `pollard-scorecard` | the standardized publish scorecard | any |
| `pollard-health` | cross-vendor accelerator degradation check | — |

> **MAINTENANCE:** this skill must track the tools. When a feature is added/changed,
> update the decision tree AND this index in the same PR, or agents will run Pollard
> wrong. Verify the tool list against `pyproject.toml [project.scripts]`.
