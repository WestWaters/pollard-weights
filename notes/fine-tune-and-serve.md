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
| reward balanced across classes | `--balance auto` oversamples the rare class in TRAIN only (to 1:2) |
| compact classification output | default answer = the label only; `--reason-field` opts into a reason line |
| LoRA on a single GPU | LoRA/QLoRA on mlx (Mac), TRL+PEFT, Unsloth; merged to a full HF checkpoint |
| version weights + prompt + policy | `agent.json` ships in the merged dir; lineage (`forge.json`) records data hash, splits, losses, eval |
| serve fast | `--ship`: `pollard --hf <merged> --run --ladder` → card with the eval table → `pollard serve --harness` |

**Not built:** the RL stage (GRPO with an outcome reward). The SFT lane gives the baseline and, more
importantly, the benchmark. The same held-out split and metric code would be the reward and the gate
for an RL step.

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
