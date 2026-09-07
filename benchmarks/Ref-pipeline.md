# Ref-pipeline — the Grok-way process for onboarding a NEW architecture to gold-card

This is **the** procedure for adding Pollard support for a new model architecture and taking it
to gold-card **without touching the recipes that already win**. Follow it every time a new arch
shows up (a new MoE flavor, MLA/hyper-connection variant, a non-transformer, etc.). It is the
generalized version of exactly what got the 7B / 14B / 30B green.

> **The one rule above all:** the **locked** recipes are locked. Dense and MoE are proven and
> shipped (gold cards on 7B, 14B, 30B). You **do not** change them to make a new arch work — you
> verify the new arch *against* the same process, and only when it's golden do you lock it too.
> The regression suite (`tests/test_recipes.py`) is what enforces this — a change that breaks a
> locked recipe fails a test before it ships.

---

## What "locked" means

| class | status | recipe |
|---|---|---|
| **dense** | 🔒 LOCKED (7B/14B gold cards) | crush ffn body; protect attn q/output/down + first/last; embed Q4_K, output Q6_K |
| **MoE** | 🔒 LOCKED (30B gold card) | crush cold experts (`ffn_gate/up_exps`); protect `ffn_down_exps` + router (`ffn_gate_inp` Q6) + shared + **all attention** + edges. Covers MLA/hyper-conn/DSA MoEs too (added rules are no-ops on Qwen-MoE) |
| a new arch | 🔬 UNVERIFIED → run THIS pipeline → 🔒 when golden | starts by *routing through the closest locked recipe*; only extend if measurement demands it |

`automap` routes by **detected features** (dense / MoE / `+MLA` / `+hyper-conn` / `+DSA-indexer`),
never by model name. A new arch first tries the existing recipe for its class. **You may get
lucky:** if it's a MoE and the MoE recipe already produces golden numbers, there is nothing to
change — you just lock the calib and move on.

---

## The pipeline (do these in order, ONE change at a time)

### Step 0 — detect & route
`pollard-calc --model <hf-or-gguf>` and `automap` on the dry-run tensor list. Confirm the
detected class (`dense` / `MoE +features`). Route through the **existing** recipe for that class.
Do **not** hand-write a new recipe up front — the whole point is to test whether the locked one
already works.

### Step 1 — get the f16 GGUF (the source of truth)
- Pre-made f16/bf16 GGUF if one exists (skip conversion), else download the safetensors and
  `convert_hf_to_gguf.py` (a new arch may need a llama.cpp conversion patch — e.g. Hy4 needed
  `hyv4.py` + arch registration). Build the mix **from f16**, never from an already-quantized file.

### Step 2 — the imatrix (calibration) — the coverage lesson
- MoE needs a **DIVERSE calib** (prose + code + varied) with enough chunks that the imatrix
  actually routes to the experts. Undercovered experts → `IQ1/IQ2` hard-fail ("Missing importance
  matrix … will be garbage, bailing out") → the build bloats to a safe fallback. This bit both the
  30B and Frank's Hy4.
- **Big model / small box:** compute the imatrix on a near-lossless **Q6_K host**, not the f16
  (f16 won't fit RAM for the forward pass), at a **partial `-ngl`** that fits the GPU. `pollard-
  sensitivity --ngl` / `llama-imatrix -ngl N` exist for exactly this.
- **Arch quirk:** some archs skip a tensor in the imatrix (Hy4 skipped `ffn_gate_exps`). If two
  tensors share an input (SwiGLU gate/up), copy one's imatrix entries to the other — same input =
  same importance. `automap` also auto-pins any still-uncovered tensor to `q6_K` so the build
  can't crash. **Calibration is the most likely thing a new arch needs modified — note it and,
  when golden, lock that calib recipe just like the dense/MoE calibs.**

### Step 3 — automap emits the recipe
`pollard-automap --tensors <dry-run> --model f16.gguf --imatrix cov.imatrix --out build.bat`.
It prints the detected arch and which recipe it used. If the arch has tensor names the recipe's
rules don't match (new attention layout, new module), that's the **only** legitimate reason to
extend the recipe — and you extend the **class** recipe with **additive, no-op-on-others** rules
(as the MoE recipe was extended to cover MLA/hc/indexer), never a per-model copy, never values
lifted from a buggy run.

### Step 4 — build the 3-bar board (the BENCHMARK, opt-in)
Uniform-low (`IQ1_KT`) / **PollardMix** / uniform-high (`IQ2_KT`), all from the same f16, same
eval. (A plain user build is `--mix-only --no-eval` = one model; the 3 bars are ours, here in
`benchmarks/`.) Apply the **size-band gate**: the mix must land **≤ uniform-low + ~8–10%**. If it
drifts toward uniform-high's size, it's out of its class — reject and tighten (this is the Mix-v3
lesson: raise the anchor too far and it just becomes a small IQ2 and loses).

### Step 5 — the full metric board
`pollard-bench --gguf mix.gguf --ref f16-or-Q6.gguf` (or the raw `llama-perplexity --kl-divergence`
against an f16 / Q8_0 / Q6_K base) → **PPL + Mean KLD + Median KLD + top-1** for all three bars,
same corpus/ctx. KLD vs a near-lossless ref is fine (note it in errata) when f16 won't load.

### Step 6 — read against the DECISION TABLE (Grok's)

| result | action |
|---|---|
| Mix beats uniform-low on **PPL + KLD + top-1** AND is in the size band | ✅ green on the quantitative board → go to the chat gate |
| Mix **loses a column** | **ONE attributable change** (Session 2): protect what carries the distribution, re-measure. Do NOT abandon. |
| Mix left the size band / lost to uniform-high at ~same GB | ❌ reject (Mix-v3): tighten the crush, don't chase the ceiling |
| KLD rank flips vs PPL | corpus mismatch/overfit — re-KL on a 2nd held-out set before changing the map |

**Session 2 (the fix loop):** change ONE thing, freeze the rest, re-score the whole board, accept
only if it improves ≥1 column without breaking the size band. That is exactly how the 30B went
from losing KLD (crushed `attn_v`) to winning every column (protected `attn_v`) — one change.

### Step 7 — chat gate
Fixed prompts (explanation / math / recursion / a loop-trap), fixed sampling (`--repeat-penalty
1.15 --temp 0.7`, ChatML), `-c 2048` (cap context so the KV cache doesn't OOM the card). Must be
coherent, no loops. Loops are usually **sampling**, not the model — the penalty note ships with
the card.

### Step 8 — LOCK it
When the board is green across **PPL + KLD + top-1 + chat**, in the size band:
1. Mark the recipe/calib canonical in code (the class it belongs to).
2. **Add a regression test** asserting its key assignments (like `test_moe_recipe_protects_attn_v`)
   so it can never silently drift.
3. Publish the full-ladder repo + gold-card scorecard.
4. ⛔ **Do not re-open it.** No new palette/codebook, no more small-model probes, no raising the mix
   to beat the 2-bit ceiling. It's done.

To make the claim travel (method, not fluke): the same green board on **≥2 models of that class**.

---

## Locked calibrations (extend this table as new arches lock)

| class | calib | notes |
|---|---|---|
| dense | mixed prose+code imatrix | the win IS the imatrix; no per-layer sweep (doesn't beat uniform on dense) |
| MoE | **diverse** prose+code+varied, 200+ chunks | must route to the experts or low-bit fails; auto-pin covers the tail |
| **EXL3 lane (trellis) — 🔒 LOCKED 2026-09-07** | ✅ = **smoothing + Calib 3.0 (256 rows, -cd) + exl3-native allocator** | **WIN: the Pollard method beats EXL3 on EXL3's own allocator, atoms, and kernel** — Qwen2.5-3B @4bpw: broken 3090 → smoothed 8.699 → smoothed+Calib3.0 **8.670** vs EXL3 out-of-box **8.699**, untuned. `pollard --format exl3` runs the gold path by default. Keep the native allocator (the win doesn't need reallocation); the GGUF role-map port is a DEAD LEVER on trellis atoms (8.794/8.862 — worse, don't waste the ~50 min). Cal rows non-monotonic (256 sweet spot, 512=8.995). Widen later via cal-mix (post-publish), validated on a 2nd eval to avoid overfit. |
| MLA/hyper-conn/DSA MoE (Deepseek, Hy4) | ✅ = MoE calib, unchanged recipe | Verified on DeepSeek-V2-Lite: automap self-routes `MoE +MLA` → the MoE recipe, **quant win proven (−57% PPL / −61% KLD / +14.8 pt top-1 vs uniform-low @ +1.5% size), zero recipe changes.** Three MECHANICAL onboarding fixes, all now automatic + regression-tested (not recipe changes): (a) convert bails on an unknown tokenizer hash → add the hash→pre-tokenizer entry; (b) ik's imatrix skips the SwiGLU **gate** (`ffn_gate_exps/shexp`) → `automap` now **auto-copies `up→gate`** (`ensure_gate_coverage`, writes `*.gatefix.imatrix`) — no manual step; (c) ik's imatrix structurally skips MLA `attn_k_b/v_b/kv_b` (can't copy-cover) → `_NEEDS_IMATRIX` flags them so they pin to q6_K. ⚠️ **Size floor:** a *shippable* 1-bit card needs the model big enough that its coherence floor is ≤ 1-bit — small/sparse models (DeepSeek-V2-Lite = 2.4B active) loop on free-gen at 1-bit and need ≥2-bit (where the mix ≈ uniform). Big MLA-MoEs (Hy4) are above the floor. This is a per-model SIZE property, not a recipe/class limit. |

---

## ⛔ PROVEN-LOSING METHODS — NEVER RUN (skip straight to what wins)
Pollard is a set of WINNING methods. If a method is on this list we already measured it losing — do
NOT run it "to check," do NOT put it in the pipeline, do NOT spend a user's (or our) cycles/tokens on
it. Skip ahead to the winner. Users want a fast Pollard build, not our whole R&D process.

| ⛔ losing method | ✅ run this instead | proof |
|---|---|---|
| **local/proxy error for anything** (imatrix MAGNITUDE, EXL3 `proxy_err` — **BANNED**, see `legacy/PROXY_ERR_BANNED.md`) | **measured MODEL-LEVEL KL** sensitivity → knapsack; gate builds with **`pollard-verify`** (real reconstruction) | magnitude misranks (e13); proxy_err stayed ~0 while a layer was garbage (cost days) — never read it |
| **sensitivity SWEEP on a DENSE model** | dense imatrix K-quant ladder + IQ1_KT flagship | no expert redundancy → loses; tools refuse it |
| **`--no-imatrix` K-quant "mix" on MoE** | automap trellis mix WITH a covered imatrix | loses to stock Q2_K |
| **PTQ of STQ1_0** (uniform or recipe) | QAT+distill on wiki+CHAT mix (extreme lane only) | uniform ~1e6; iq1_s 1.56 beats STQ1_0-PTQ |
| **porting the GGUF role-map onto EXL3** (or any lane) — instead of the gold path | keep EXL3's native allocator + do the CALIB/smooth levers (the gold path already wins, 8.670 vs 8.699) | K-quant priors don't map to trellis atoms: role-port 17.15 vs native 14.30 (0.5B); smoothed Qwen2.5-3B @4bpw native 8.699 vs sqnr-recipe 8.794 vs role-recipe 8.862 — a DEAD lever, ~50min each wasted |
| **per-model recipe from a buggy run** | route the class recipe; extend additively only if measured | the weekend loop |
| **raising the mix tier to chase a 2-bit ceiling** | reject (Mix-v3); tighten the crush | leaves the size class, becomes a small IQ2 |

**The rule:** a new arch/format tries the CLASS's WINNING method first; only a *measured* win gets
locked as a new gold path; anything we proved loses is deleted from the flow, not retried.

## 🆕 NEW LANE (new emit backend / atom class, e.g. EXL3) — TUNE lane-native, do NOT port-and-lock
A new BACKEND (EXL3/MLX/GPTQ) is a **new atom class**. Porting the GGUF/IQ recipe onto it is only a
first PROBE — it usually LOSES (K-quant priors ≠ trellis/other atoms), and **a port-failure NEVER
justifies locking the lane compatibility-only.** You must run the lane-native Session-2 loop FIRST:
- **Baseline to beat** = the lane's own default allocator (e.g. EXL3-budgeted), boarded at a target bpw.
- **Model** = a MID model (**≥3B–8B**), NOT a 0.5B, and NOT Wiki-PPL-only — add the chat gate.
- **⚠️ PREREQUISITE for low-bit trellis/error-feedback lanes (EXL3/GPTQ) — PRECONDITION FIRST.** These
  lanes have NO input-outlier protection: one massive-activation input channel collapses the encode's
  global scale and **silently produces a broken layer** (both the convert's per-tensor error AND the
  banned `proxy_err` miss it — the layer's real-activation `sqnr` goes negative). Run `pollard-hf-smooth`
  (SmoothQuant → RMSNorms, exact identity) on the fp16 model BEFORE the convert; confirm with
  `pollard-verify` (real reconstruction) / per-layer sqnr. MEASURED (EXL3, Qwen2.5-3B @4bpw): unsmoothed
  **PPL 3090 (broken)** → smoothed **8.699 (works, ≈8bpw quality)**. This is separate from allocation —
  do it before you even compare allocators, or you're tuning on a broken build.
- **⚠️⚠️ ORDER IS MANDATORY — START AT CALIBRATION (Step 1). DO NOT run an allocation/recipe port first.**
  A ported protect/crush recipe (GGUF/IQ priors → the lane's atoms) **LOSES** and wastes a ~50-min convert
  each: MEASURED on EXL3 @4bpw — budgeted (default) **8.699**, ported sqnr recipe (v1) **8.794**, ported
  role recipe (v2) **8.862** — both *worse*. The native allocator is strong on ITS OWN atoms; your first
  real lever that's actually YOURS is **your calibration**, not a recipe port. Only after the calib run do
  you touch allocation — and only with a MEASURED probe in the lane's atoms, NEVER ported priors (Step 2).
- **Loop (ONE attributable change per run, re-board each; gate every build with `pollard-verify`, NEVER proxy_err):**
  1. **CALIB for the lane — DO THIS FIRST** — Calib 3.0 (`pollard-calib` → tokenize → `-cd` packed safetensors
     `{input_ids:[rows,cols]}`, real tokens only, NEVER tiled) as the encode calibration; board vs default at same bpw.
     ⚠️ **Cal row-count is NON-MONOTONIC — do NOT just crank it.** MEASURED (EXL3, Qwen2.5-3B @4bpw): 256 rows
     = **8.670** (beats exl3-default 8.699) but 512 rows = **8.995** (WORSE than both — more rows shift the
     Hessian → worse allocation). Find the sweet spot (~250-256 default is a good start); more is not better.
  2. **RETUNE protect/body in the LANE's atoms** — SOFT priors only (protect residual writers / attn-out /
     embeddings / head; crush the coldest FFN body), with targets from a SHORT measured probe in the
     lane's bpw atoms — NOT copied IQ gate/up rules.
  3. **Attributable tuning** — one lever per run (protect bump | body floor | calib mix); reject if bpw
     creeps or PPL worsens.
  4. **Recovery** — one light repair pass only if coherent-but-soft AND the stack supports it; else skip
     (no multi-day QAT side quest).
  5. **Ship gate** — Wiki PPL + chat (fixed sampling: rep-pen 1.15, temp ≤0.7). **WIN = ≤ default bpw AND
     (PPL ≤ default OR a clear chat win with PPL within ~5%).**
- **Lock rule:** lock as a new gold lane ONLY on a matched-bpw win. Lock **compatibility-only** ONLY
  after **≥3 consecutive lane-native TUNED attempts lose at matched bpw** on the mid model — never after
  port failures alone. **Goal B (finish faster) stays weak** where the lane's default has no expensive
  search (EXL3-budgeted is search-light) — don't chase it; the win is match-quality, and never reimplement
  the lane's own optimizer.
- **EXL3 status — ✅ THE WIN (2026-09-07):** the Pollard method **beats EXL3 on EXL3's own allocator,
  trellis atoms, and custom kernel.** MEASURED (Qwen2.5-3B @4bpw, smoothed): **smoothing + OUR Calib 3.0 =
  8.670 vs EXL3 out-of-box 8.699** — same allocator/bpw, our method wins UNTUNED. Smoothing alone takes
  unusable low-bit (3090) to 8.699. The lockable Pollard EXL3 edge = **smoothing + our calibration**, on
  EXL3's native allocator. Keep the native allocator — the win doesn't need reallocation, and the GGUF
  role-map port is a DEAD LEVER (K-quant priors don't map to trellis atoms: role-recipe 8.862 / sqnr 8.794
  — worse, don't waste the ~50 min). **Open R&D to widen the win:** TUNE the
  calibration (domain mix/size/model-matched) + a finer per-tensor **measured-KL** recipe in EXL3's atom space
  (recipe YAML `{tensors:{key:int_bits}}`, bits 1-8|16, must cover every quantizable tensor; pass `-b/-hb`
  for reporting) + Calib 3.0 + the tuning loop above. Recipe/verify tooling: `pollard-exl3 --recipe`,
  `pollard-hf-smooth`, `pollard-verify`, `pollard-doctor`.

## Traps (the weekend lessons — do not repeat)
- **Don't build a per-model recipe** — especially not from a **buggy run's** numbers. Route through
  the class recipe; extend the class recipe additively only if measurement demands it.
- **Never run a losing path as the default.** Sensitivity sweep on dense = loses (refused). On a
  big MoE = ~2·layers full quantizes = HOURS (refused, use the trellis mix). `--no-imatrix` K-quant
  "mix" = deprecated (loses to stock Q2_K).
- **Build ≠ benchmark.** A plain build is ONE model, minutes. The 3-bar / KLD / sensitivity are
  here in `benchmarks/`, opt-in. Shipping them in the build is what caused the "3-hour" runs.
- **Measure, don't assume.** `imatrix magnitude LIES` (big activations ≠ high KL). Protect what the
  measurement (KL / the decision table) says, not a guess. And **never trust the convert's own printed
  error** — `proxy_err`/per-tensor nmse can read perfect while the assembled model is garbage (a massive
  outlier dominates those metrics). Gate EVERY build with `pollard-verify` (real reconstruction: decode-
  vs-source + end-to-end forward), never a proxy. proxy_err is BANNED (`legacy/PROXY_ERR_BANNED.md`).
- **Low-bit lanes need preconditioning FIRST.** Trellis/error-feedback encoders (EXL3/GPTQ) have no
  input-outlier protection — a massive-activation channel (emerges in the first couple of layers)
  collapses the global scale and silently breaks a layer. `pollard-hf-smooth` before the convert fixes
  it (measured: EXL3 3090→8.699). It composes across all lanes; abliterate is the same FP16-pre-convert
  pattern. `pollard-doctor --predict` flags at-risk layers on the fp16 model before you build.
- **Infra bites:** disk-full truncates KL base logits mid-run; f16 won't fit RAM (use a Q6/Q8 host);
  `-ngl 99` OOMs a big model on a small card (lower it — KL is offload-invariant); the case-sensitive
  `q3_K` in `--custom-q`; `set VAR=x &&` in cmd captures a trailing space; **ik_llama REFUSES a q8_0
  source** for quantize → `--allow-requantize` (or build from f16). Check the machine first.
- **Don't over-generalize from one model.** A too-small model failing the *chat* gate at 1-bit is a
  per-model SIZE property (its coherence floor > 1-bit), NOT a recipe or class failure — do NOT write
  class-level "no win" conclusions from it, and do NOT keep elevating the bit tier chasing a win.
  Bank the two facts (recipe generalizes; this model is above the floor) and move to a properly-sized
  model. And **sampling-first:** a low-bit mix that loops is usually sampling (rep-pen 1.15–1.18, temp
  ≤0.7, correct stops) — exhaust that before ANY recipe/tier change.
- **Compaction regresses this.** The winning method + this pipeline are banked to memory; read the
  anti-regression anchor before touching Pollard.

---

*Run this for every new arch. Green across the board → lock it → it joins dense & MoE as a recipe
that just works, forever.*
