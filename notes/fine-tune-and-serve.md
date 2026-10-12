# fine-tune and serve (`pollard tune`) — design note

## What Coinbase did (and which parts mattered)

Coinbase "Owning Intelligence" series, 2026-10-06..08:
[part 1](https://www.coinbase.com/blog/part-one-how-coinbase-built-evaluated-and-post-trained-a-fraud-agent) ·
[part 2](https://www.coinbase.com/blog/part-two-owning-intelligence-series) ·
[part 3](https://www.coinbase.com/blog/part-three-owning-intelligence-series).
The coinbase.com pages sit behind a Cloudflare challenge. We read parts 2 and 3 through a reader proxy.
Part 1 is known only from secondary coverage, so its numbers are marked as such.

- **Decision-agent framing (part 1).** The LLM is not the fraud system. It is a contextual review
  layered *downstream* of the existing ML model and rules, for selected transactions. It emits a risk
  classification, and a fixed policy turns that classification into a decision. Online A/B (secondary
  coverage): 30% fewer fraudulent transactions, 22% less fraud value, with projected savings about 3x
  the inference cost.
- **Benchmark (part 2).** 16,140 transactions from 7,293 users over nine weeks before rollout. It holds
  all matured fraud cases (813) plus sampled legitimate traffic, and it comes from history so the agent's
  own blocks cannot hide outcomes. Every model got the same guidance and the same
  classification-to-decision policy, so the model was the only variable.
- **Four metrics, one of them weighted.** Precision, recall, F1, and **dollar-weighted recall**: the share
  of fraudulent *value* caught. Newer frontier versions regressed on their benchmark. Sonnet 5 lost
  22.2 pts of recall against 4.6. GPT-5.6 gained 11.5 pts precision but lost 20.7 pts recall and 21.8 pts
  dollar-weighted recall. Opus 5 slipped on all four. Their lesson: read the precision/recall trade-off
  as a whole, separate model changes from prompt/policy changes, and check latency, cost and reliability.
- **Post-training (part 3).** Qwen3.5-9B, chosen to serve on one GPU. They used **RL, not SFT**: GRPO
  with verifiable deterministic rewards from historical fraud outcomes, with the reward **balanced across
  fraud and legitimate rows** so the model could not win with a constant answer. LoRA, on SkyRL (Megatron
  + vLLM rollouts), one RTX PRO 6000 96 GB, under $100. The untuned Qwen was poor, so the gain came from
  domain adaptation, not scale. Against Opus 4.5: +7.5 precision, +12.0 recall, +9.6 F1, +35.4
  dollar-weighted recall.
- **Serving.** Ray Serve LLM + vLLM on L40S with 2-8 autoscaled replicas. The output contract was cut to
  the compact classification only (no explanation), for latency. P50 0.683 s vs 1.515 s (-55%), P99
  1.036 s vs 3.312 s.
- **Operations.** "Owning intelligence means more than owning weights": domain evals, **version the whole
  agent (weights + prompt + decision policy)**, and monitoring with tested rollback.

## How the lane maps that

| Coinbase | `pollard tune` |
|---|---|
| matured-label benchmark from a past window | `--time-field` split (oldest train, newest test); held-out split leak-checked and **refused** on overlap |
| fixed guidance + policy for all models | one system prompt + label set per run, used for base, tuned and `--compare-url` alike; greedy decoding |
| precision / recall / F1 / dollar-weighted recall | positive-class P/R/F1 + `--weight-field` weighted recall, confusion matrix, unparsed rate |
| latency P50/P99 | measured per row in every column |
| reward balanced across classes | `--balance auto` oversamples the rare class in TRAIN only (to 1:2); with `--method grpo` the reward is weighted by inverse class frequency (trl/unsloth) or the GRPO prompts are sampled 1:1 (mlx) |
| GRPO, LoRA, verifiable reward from historical outcomes | `--method grpo` / `sft+grpo`: reward = eval's label parser vs the gold label, class-balanced, + a small format term |
| compact classification output | default answer = the label only; `--reason-field` opts into a reason line |
| LoRA on a single GPU | LoRA/QLoRA on mlx (Mac), TRL+PEFT, Unsloth; merged to a full HF checkpoint |
| version weights + prompt + policy | `agent.json` ships in the merged dir; lineage (`forge.json`) records data hash, splits, losses, eval |
| serve fast | `--ship`: `pollard --hf <merged> --run --ladder` → card with the eval table → `pollard serve --harness` |

**RL stage (`--method grpo`, `sft+grpo`).** The reward is the eval's own label parser compared with the gold
label, so what is rewarded is what is measured. Class balance: weight w_c = N / (K * n_c), which makes the
expected correctness reward over the prompts equal to the balanced accuracy. Any constant answer and a uniform
random guess both earn exactly 1/K; a perfect policy earns 1. The format term (+0.1 for a bare label, -0.2
for an unparseable reply or `<think>`) is the same for every label, so it cannot favour a class. Chat data is
refused: there is no verifiable reward.

Backends differ in one way that matters. GRPO normalises rewards within each prompt's group. Dividing by the
group's std (mlx-lm-lora always does; TRL does by default) cancels a per-prompt weight, because every sample
of a prompt carries the same class weight. TRL is run with `scale_rewards="batch"` so the weights act. On mlx
the GRPO prompts are oversampled to 1:1 instead, which gives each class the same share of the gradient. The
lineage records which (`reward.balance_via`). `--value-reward` (value over class mean, capped 10x) works
the same way: weighted on trl/unsloth, a no-op on mlx.

mlx-lm-lora (checked against 5.8.6): `python -m mlx_lm_lora.train --train-mode grpo` takes
`{prompt, answer, system, type}` rows. Custom rewards come from `--reward-functions-file`, a Python file that
registers functions with `register_reward_function()`; each gets `(prompts, completions, answer, types)`. We
put the row's label / options / gold / value multiplier in `type` as JSON. It fuses into the adapter dir by
default (`fuse: true`), so the config sets `fuse: false` and we run `mlx_lm fuse` as on the SFT path.

**Smoke run (M-series Mac, 16 GB, 2026-10-11).** Synthetic data: 400 card-payment descriptions, 45 fraud
(11%), with 12% of rows given the other class's look. Model: Qwen2.5-0.5B-Instruct. Held-out split: 60 rows,
7 fraud. Batch 4 prompts x 4 samples, LoRA r16; mlx-lm-lora 5.8.6 / mlx-lm 0.32.0.

| run | precision | recall | F1 | weighted recall | accuracy |
|---|---:|---:|---:|---:|---:|
| base (always `fraud`) | 11.7% | 100.0% | 20.9% | 100.0% | 11.7% |
| `grpo`, 40 steps, lr 5e-5 | 0.0% | 0.0% | 0.0% | 0.0% | 88.3% (always `legit`) |
| `grpo`, 60 steps, lr 1e-5 | 11.7% | 100.0% | 20.9% | 100.0% | 11.7% (unchanged) |
| SFT stage of `sft+grpo` (60 steps, lr 1e-4) | 37.5% | 42.9% | 40.0% | 34.7% | 85.0% |
| `sft+grpo` (+ 60 GRPO steps, lr 1e-5) | 50.0% | 100.0% | 66.7% | 100.0% | 88.3% |

The base model was close to certain (p(fraud) 0.92-0.99 on every held-out case), so most 4-sample groups
were all `fraud` and carried no advantage. lr 5e-5 drove the KL to 2.3 and the group spread to 0 by step 30.
lr 1e-5 drifted back to the reference (KL near 0, spread 0 from step 25). After SFT, the GRPO stage kept a
non-zero group spread for most of the run. Training reward went 0.775 -> 1.013 and KL ended at 0.075. Seven
test positives is a small sample; the numbers show the mechanism working, not a production estimate. The
trl path ran 10 GRPO steps on MPS with trl 1.14.0 (24 s); trl 1.15.0 fails without Triton.

## Decisions

- **Classification by generation, parsed.** We serve what we evaluate. The parser takes the first line if
  it is a label, else the earliest whole-word label (longest wins a tie, so `not fraud` beats `fraud`).
  An unparsed reply counts as wrong and is reported.
- **Typed decisions.** Unsloth `FastDecisionModel` (clef head + calibrated temperatures) when Unsloth and
  CUDA are present. Otherwise the generation path, with option probabilities from the LM's likelihoods,
  scored by `pollard_decision.score_typed` plus ECE. Ship warns when a clef model's GGUF lost its
  `.decision.temperature.*` keys.
- **Privacy.** Nothing leaves the machine unless `--compare-url`/`--judge-url` is off-box, and then only
  with `--allow-remote`. Cards carry file basenames, never paths.
- **`pollard serve`** lives on the unmerged pw-harness branch. `ship` detects `pollard_harness` and
  otherwise prints the `llama-server` line it would have run.
