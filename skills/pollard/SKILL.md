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

## Don't know the model type? Use the autoaware entry point

```bash
pollard --gguf model-f16.gguf --ram 16 --imatrix model.imatrix        # plan (prints the right path)
pollard --gguf model-f16.gguf --imatrix model.imatrix --run           # detect + build automatically
```
`pollard` reads the arch, decides **dense vs MoE**, and dispatches to the correct path
(dense → the imatrix K-quant ladder via `pollard-fit` **plus the IQ1_KT mixed-precision
flagship** — the hand-coded mix, the dense repos' headline build; MoE → `pollard-automap`
expert-allocation) —
the user never picks a tool. The manual steps below are exactly what it runs; read on to
drive a path yourself or understand what `pollard` chose.

> ### Build ≠ benchmark — the default is FAST
> Making a Pollard model is just the **build** (quantize): **one model, minutes, no eval.**
> The PPL / KLD / top-1 / 3-bar comparison / sensitivity sweep are the **benchmark** — they
> only *measure* the model and take **hours**. They are **opt-in**, not part of a normal build.
> - **Just make the model:** `pollard --gguf model.gguf --run` (one model, no eval).
> - **Reproduce our gold-card numbers:** add `--benchmark`, or score an existing model with the
>   drop-in **`pollard-bench --gguf model.gguf --ref f16.gguf`** (PPL + KLD + top-1; `--vs` for head-to-head).
>
> Never run the benchmark or the sensitivity sweep as part of a plain build — that's the
> mistake that turned a minutes-long shrink into a 3-hour run. See `benchmarks/README.md`.

## Winning default paths vs losing/fallback (so nobody re-hits the 3-hour trap)

| model | ✅ WINNING default (proven, minutes) | ⚠️ fallback / ⛔ losing (never the default) |
|---|---|---|
| **DENSE** | imatrix K-quant ladder (`pollard-fit`) **+ the IQ1_KT mixed-precision flagship** (the hand-coded mix — won 7B/14B) | ⛔ sensitivity SWEEP on dense = loses (no expert redundancy) → `pollard-sensitivity` refuses it |
| **MoE** | the **trellis mixed-precision mix** (`automap` WITH an imatrix) | ⛔ `automap --no-imatrix` K-quant mix = **DEPRECATED**, does NOT beat stock `Q2_K` (imatrix-free build → just use a stock K-quant ladder) · ⛔ sensitivity SWEEP on a big MoE = ~2·layers full-model quantizes = HOURS → refused unless `--allow-slow` |
| **HYV4** (Tencent Hy4-preview / hy_v4 — MLA + DSA-indexer + hyper-connection MoE) | its OWN recipe: crush `ffn_gate/up_exps`; protect `ffn_down_exps` + shared + dense-blk0 + router + MLA attn (`q_a/q_b/k_b/v_b/kv_a_mqa/gate/output`, `k_b` a tier up) + hyper-connections (`hc_*_fn`) + DSA indexer; norms/base/scale/`exp_probs_b`/`sinks` stay F32. `automap` auto-detects it. **v1 — measure & tighten like any card** | ⛔ the MoE recipe (wrong tensor names → falls to defaults → BLOAT, as Frank's 762B run hit: 381 GB) · needs a COVERED imatrix (256 experts — the coverage lesson at scale) |

The **mixed-precision hand-coded mix** (crush body → protect attn/down/first-last) is the BUILD,
proven by palette/lowbit and shipped across GGUF (`automap`), torch/GPU (`gptq --recipe`), and
vLLM (`export`). The **KL/PPL/scorecard/3-bar board** is the BENCHMARK — opt-in, `benchmarks/`.

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

## MoE path

### MoE QUALITY win — the MEASURED allocator (this is the proven one; do NOT reinvent it)

The quality win on a MoE (and on dense) is **`pollard-sensitivity` → `pollard-fit --sensitivity`**:
it CRUSHES one tensor group at a time and measures the *true KL cost*, then a knapsack spends
bits where the measurement — not a guess — says they matter. **Measured, shipped, and proven
to beat uniform imatrix-IQ at matched size on both a MoE (granite-3B-a800m, 40 experts) AND a
dense (Qwen2.5-1.5B)** — see `assets/kl_win.png`. Do not build a second allocator; use this one.
```bash
pollard-sensitivity --gguf moe-f16.gguf --imatrix moe.imatrix --eval held.txt --out moe.sens.json
pollard-fit --gguf moe-f16.gguf --ram 16 --imatrix moe.imatrix --sensitivity moe.sens.json --out moe-pollard.gguf
```
Cost: the sweep is ~2·num_layers passes (one-time per model). Note: `imatrix magnitude LIES`
(big activations ≠ high KL) — that's exactly why the measured sweep exists; don't allocate on
raw imatrix importance.

### MoE trellis (imatrix) — the extreme-low 1-bit-class variant (only with a covered imatrix)

`pollard-automap` emits the 3-bar 1-bit-class build (uniform-IQ1 / PollardMix / uniform-IQ2).
This is the extreme-low showcase (the 7B/14B gold-cards), NOT the general allocator above.

### MoE trellis (imatrix) — the extreme-low variant (only with a covered imatrix)

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
more diverse calib (more experts routed). (Build it on a Q6_K host + copy `up_exps`→`gate_exps`.)

**Accept-the-Mix gate:** keep it only if PollardMix ≤ uniform-low size + ~8–10% AND it
beats uniform-low on PPL/KL/top-1. If it drifts toward uniform-high size, reject and
tighten the crush.

### MoE with NO imatrix — just use a stock K-quant ladder (`--no-imatrix` is DEPRECATED)

`pollard-automap --no-imatrix` (a K-quant "mix") is **DEPRECATED** — measured on Qwen3-30B-A3B
it did **NOT** beat the stock `Q2_K` preset, so it's a losing path with no reason to exist.
The imatrix is no longer a swamp (build it on a **Q6_K** host + copy `up_exps`→`gate_exps` for
SwiGLU coverage → minutes). So: **make an imatrix and use the trellis mix (the winner); if you
truly won't, just build a stock K-quant ladder (`pollard-fit`)** — that's the honest imatrix-free
build, not a fake mix that loses to it.

**MoE measured K-quant ladder (alternative to the trellis mix)** — the KL knapsack that
beats uniform, for shipping IQ3/IQ4/Q6 sizes:
```bash
pollard-sensitivity --gguf moe-f16.gguf --imatrix moe.imatrix --eval held.txt --out moe.sens.json
pollard-fit --gguf moe-f16.gguf --ram 16 --imatrix moe.imatrix --sensitivity moe.sens.json --out moe-pollard.gguf
pollard-experts --gguf moe-f16.gguf --imatrix moe.imatrix          # measured hot-expert report
```

---

### Giant MoE that won't fit even quantized (e.g. Kimi K2, ~1T params) — PRUNE first

Quant has a floor: params x 1 bit. A 1T MoE is ~125 GB at 1-bit and unusable that low — you
**cannot quantize a trillion-param MoE onto one small box.** The lever is **expert pruning**:
a big MoE has many cold/redundant experts (Kimi K2: 384 experts, 8 active). Drop the cold
third-to-half FIRST (structurally smaller model), THEN quantize:
```bash
pollard-pack  --gguf kimi-f16.gguf --emit-plan plan.json      # rank experts (forecast + which to drop)
pollard-prune --gguf kimi-f16.gguf --keep 0.5 --out kimi-pruned.gguf   # execute: rewrite a smaller GGUF
pollard-automap --tensors <dry-run kimi-pruned> --model kimi-pruned.gguf --imatrix kimi.imatrix --out build.bat
```
`--score imatrix` (with a diverse-calib imatrix) keeps the experts the calib actually routed
to (REAP-correct); `magnitude` is the zero-calib default. Keep >= the model's active-expert
count. **Producing the artifact needs the source weights + storage + a big box** — the tool
runs on the box that holds the model; a user without the disk downloads the finished small
build instead of making it.

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

## Preconditioning levers (optional — apply to the f16 BEFORE the imatrix/quantize)

These rotate/smooth the weights so low-bit quant lands better. Bit-width-dependent (they
help at some tiers, hurt at others) — measure with `pollard-eval`/`pollard-kl` before shipping.
```bash
pollard-smooth   --gguf model-f16.gguf --out model-smooth.gguf --alpha 0.5     # AWQ-style channel smoothing
pollard-rotate   --gguf model-f16.gguf --out model-rot.gguf --kind orthogonal  # QuIP#/QuaRot rotation (also --kind block --block 32)
pollard-precondition --gguf model-f16.gguf --out model-pre.gguf                # dynamic selector: pick the best lever per bit-width
pollard-gptq     --model <hf-dir> --bits 4 --groupsize 128 --out ./gptq        # full-Hessian error-feedback GPTQ (beats RTN)
```

## Utilities & metrics

```bash
pollard-calc --model <hf-or-gguf> --ram 16 --ctx 8192   # will it fit? KV-cache + total RAM pre-flight
pollard-run  --gguf model-pollard.gguf --vram auto --launch   # launch a build (llama-server), VRAM-aware
pollard-eval --gguf model-pollard.gguf --ref model-f16.gguf --eval held.txt   # trajectory-divergence quality
pollard-kl   --model <hf> --eval-file held.txt --calib-file calib.txt         # KL-to-f16 + top-1 (small models, torch)
pollard-probe --model <hf> --eval held.txt --out model.sens.json              # = pollard-sensitivity, CHEAP mode (same profile output, no GGUF/imatrix sweep — any box)
pollard-scorecard --results results.json --out scorecard.md                   # the standardized publish scorecard
pollard-health                                                                # cross-vendor accelerator degradation check
pollard-fit-dit --gguf dit-model-f16.gguf --ram 16 --out dit-pollard.gguf     # memory-fit builds for DiT (image/video) archs
pollard-lowbit  --model <hf> --bits 1 --keep 0.01 --levels 3 --eval-file held.txt   # PROVEN low-bit levers: outlier-catch + residual carousel
pollard-palette --model <hf> --calib-file calib.txt --eval-file held.txt --target-bpw 1.58 1.3  # PROVEN measured mixed-alphabet allocation (beat uniform ternary -22%/-35%)
```

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
| `pollard` | **autoaware entry** — detect dense/MoE + run the right BUILD | any |
| `pollard-bench` | **drop-in benchmark** — PPL + Mean/Median KLD + top-1 for a model, `--vs` for head-to-head | any |
| `pollard-calc` | pre-flight: arch, params, "will it fit?" (KV-aware) | any |
| `pollard-fit` | memory-fit build for a RAM budget (K-quant ladder / MoE knapsack) | any |
| `pollard-automap` | MoE expert-allocation mix + auto-pin uncovered experts | **MoE only** |
| `pollard-sensitivity` | measured per-group KL profile (the knapsack input) | **MoE only** |
| `pollard-probe` | `pollard-sensitivity` **cheap mode** — same profile, in-process, no GGUF/imatrix sweep | any |
| `pollard-pack` | Cerebras wafer capacity + expert-prune plan (forecast) | MoE (lever) |
| `pollard-prune` | REAP-style expert pruning — drop cold experts, rewrite a SMALLER GGUF | **MoE only** |
| `pollard-export` | vLLM/SGLang GPTQ (4/8 dynamic) checkpoint | any |
| `pollard-abliterate` | optional refusal-direction ablation (pre-quant) | any |
| `pollard-experts` | measured hot-expert report | **MoE only** |
| `pollard-run` | launch a build (llama-server), VRAM-aware | any |
| `pollard-fit-dit` | memory-fit builds for DiT (image/video) archs | any (DiT) |
| `pollard-gptq` | full-Hessian error-feedback GPTQ (beats RTN) | any |
| `pollard-smooth` | AWQ-style channel smoothing (preconditioning) | any |
| `pollard-rotate` | QuIP#/QuaRot rotation (preconditioning) | any |
| `pollard-precondition` | dynamic selector for the best lever per bit-width | any |
| `pollard-eval` | trajectory-divergence quality metric | any |
| `pollard-kl` | KL-to-f16 + top-1 metric (small models, torch) | any |
| `pollard-scorecard` | the standardized publish scorecard | any |
| `pollard-lowbit` | PROVEN low-bit levers: outlier-catch (SpQR) + residual carousel (AQLM) | any (torch, small model) |
| `pollard-palette` | PROVEN measured mixed-alphabet allocator (prune/binary/ternary/2b) — beat uniform ternary −22%/−35% | any (torch, small model) |
| `pollard-health` | cross-vendor accelerator degradation check | — |

> **MAINTENANCE:** this skill must track the tools. When a feature is added/changed,
> update the decision tree AND this index in the same PR, or agents will run Pollard
> wrong. Verify the tool list against `pyproject.toml [project.scripts]`.
