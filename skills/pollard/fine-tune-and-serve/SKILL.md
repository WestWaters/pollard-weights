---
name: fine-tune-and-serve
description: >-
  Fine-tune an open model (Qwen, Llama, ...) on the user's OWN data with Pollard, prove on a held-out set
  that it beats the base model (and optionally a frontier model), then quantize, card and serve it for an
  agent harness. Use when asked to "fine-tune X for <task>", "train a model on our data", "make a
  classifier / reviewer / decision model", or "/fine-tune-and-serve-llm I want to fine-tune Qwen for
  payments risk review". Covers the data checklist, `pollard tune`, reading the eval, `--ship`, and
  `pollard serve`.
version: "0.2"
---

# Fine-tune and serve — agent skill

The lane: **data checklist → `pollard tune` (prep → LoRA SFT and/or GRPO → merge) → eval against base (+ frontier) →
`--ship` (GGUF ladder + card with the eval) → `pollard serve --harness`**. Everything runs on the user's
machine. Rows leave it only if you point `--compare-url` / `--judge-url` at another host, and that needs
`--allow-remote`.

Why it is built this way (Coinbase, "Owning Intelligence" series, 2026-10): a post-trained Qwen3.5-9B
beat Opus 4.5 on their onramp-fraud benchmark (+7.5 precision, +12.0 recall, +9.6 F1, +35.4
dollar-weighted recall; P50 0.683 s vs 1.515 s). What carried it was the **evaluation**: a benchmark built
from matured labels, one fixed prompt and decision policy for every model, a metric weighted by what a
miss costs, and weights + prompt + policy versioned as one unit. Newer frontier models did *worse* on
their benchmark. Only a domain eval shows that, so never skip it.

## STEP 0 — ask the user (unless the prompt already says)

1. **The decision.** What does the model output? A label set (`fraud` / `legit`, `approve` / `review` /
   `block`), typed decisions (choice / yes-no / score), or free text? A label set gives the best evals.
2. **Which class matters.** Which label is the expensive one to miss (it becomes `--positive`)? Is there a
   per-case cost, such as the amount (`--weight-field`, giving dollar-weighted recall)?
3. **Where it runs.** The target machine and the harness (omp, claude-code, codex, opencode, ...), for
   `--ship` / `--harness`.
4. **What to beat.** Base only, or also a frontier / bigger model at an OpenAI-compatible URL? If the URL is
   off-box, confirm the user is cleared to send held-out rows there.

## Data checklist (what to collect)

For a **classifier / reviewer** (payments risk, fraud, moderation, triage), one row per case:

- **The case as the reviewer sees it *at decision time*.** Amount, account age, device/IP/country
  signals, velocity, history counts, free-text notes. CSV columns or a JSON `input` object both work.
- **⚠️ No post-outcome fields.** `chargeback_filed`, `refund_status`, `final_review_note` and the like leak
  the label. The model then learns the answer, scores 99%, and fails in production. Drop them.
- **The label, from MATURED outcomes.** Chargebacks and fraud reports arrive weeks later. Use a window old
  enough that its labels have settled (Coinbase replayed a 9-week window with all matured fraud and
  sampled legitimate traffic).
- **The cost field** (amount) for weighted recall, and a **timestamp** for `--time-field`: oldest go to
  train, newest to test. A random split lets the model see the future.
- **Volume.** Hundreds of positives is a start; more helps. Rare classes are oversampled in train only
  (`--balance auto`, up to 1:2). Val and test keep the real base rate.
- Optional: a one-line **reason** column (`--reason-field`). It teaches the model to explain itself, at
  the cost of latency. Coinbase served labels only.

For **chat SFT**: jsonl rows `{"messages": [...]}` ending with the assistant turn to learn.
For **typed decisions** (LocalLLaMA/typed-decisions format): rows `{state, questions, gold}`.

**Privacy:** the data, adapter, merged model and eval all stay in `--out`. The card names the data file by
basename only and never carries a local path. Before using `--compare-url` with a hosted API, say exactly
how many rows will be sent and to whom.

## Run it

```bash
# preflight the base on THIS box first (seconds, no training)
pollard-smoke --model Qwen/Qwen3.5-9B

# one-shot: prep -> LoRA -> merge -> eval base vs tuned
pollard tune --base Qwen/Qwen3.5-9B --data cases.csv --task classify \
    --label-field outcome --positive fraud --weight-field amount_usd --time-field created_at \
    --id-field case_id --instructions "You review card payments for fraud risk." --out runs/payments-risk

# + a frontier/bigger model as a third column (off-box needs --allow-remote; key from an env var)
pollard-tune eval --out runs/payments-risk --compare-url https://api.example.com \
    --compare-model frontier-model --api-key-env FRONTIER_API_KEY --allow-remote

# ship: pollard-smoke, `pollard --hf <merged> --run --ladder`, card with the eval table, then the serve line
pollard-tune ship --out runs/payments-risk --harness claude-code
pollard serve <printed rung>.gguf --harness claude-code
```

Steps run alone too: `pollard-tune prep|train|eval|ship --out RUN`. The `--resume` flag picks training
up from the last checkpoint and never retrains finished steps. Backends are detected: `mlx` on Apple
Silicon (`pip install mlx-lm`), `trl` (TRL + PEFT) on CUDA/CPU, and `unsloth` on CUDA. Force one with
`--backend`. Add `--qlora` for a 4-bit base when memory is tight (on mlx the merge then dequantizes the
4-bit base, which lineage records).

Defaults: LoRA r16 / alpha 32, lr 1e-4, 2 epochs, batch 4, max length 1024, loss on the answer only.
On a 16 GB Mac a 0.5B model trains at ~3 it/s. For 7-9B use `--qlora`, and `--grad-checkpoint` if needed.

## Method: `sft`, `grpo` or `sft+grpo`

`--method` (on `train` and `run`, default `sft`) picks the training objective.

- **`sft`**: supervised LoRA on the answers in the data. The quickest to run, and the right first run: it
  gives a baseline and the benchmark. Works for every task, including chat.
- **`grpo`**: RL (Group Relative Policy Optimization), as in Coinbase's part 3. For each prompt the model
  samples `--num-generations` answers. Each answer gets a deterministic reward from the historical label, and
  the model is pushed toward the answers that scored above their group's mean. The reward:
  - correctness: the label parsed from the answer (same parser as eval) equals the gold label, times an
    inverse-class-frequency weight. The weights are scaled so that any constant answer, and a random guess,
    earn exactly 1/K of a perfect score. At a 10% fraud rate, "always legit" earns 0.5, not 0.9.
  - format: `+--format-bonus` (0.1) for exactly a label, 0 for a label inside extra text, -0.2 for an
    unparseable reply or leaked `<think>`.
  - `--value-reward` (with `--weight-field`): scales correctness by the row's value over its class mean
    (capped at 10x). trl/unsloth only; on mlx it has no effect (see below).
  - decision sets: the gold probability of the chosen option when the row has soft gold, else 1/0.
  - **chat is refused**: a free-text reply has no verifiable reward. Use `sft`.
- **`sft+grpo`**: SFT first (teaches the output format), then GRPO starting from the SFT checkpoint.
  `--resume` does not redo a finished SFT stage (`RUN/stages.json`). Use this when the base model often
  answers outside the label set. GRPO cannot learn from a group where every sample scores the same.

When to use which: start with `sft`. Try `grpo` when SFT buys accuracy by under-flagging the rare class
(recall or weighted recall falls), or when the label alone is the training signal you trust. Use `sft+grpo`
when the base model does not yet answer with a bare label, or answers one label almost always. Compare them
on the same held-out split; the eval does not change.

GRPO learns only from prompts whose sampled answers disagree. In the Mac smoke run, Qwen2.5-0.5B-Instruct
gave p(fraud) 0.92-0.99 on every case, legitimate ones included. Plain `grpo` then had almost no signal. At
`--grpo-lr` 1e-5 it ended where it started (always `fraud`). At 5e-5 it collapsed to always `legit` (KL 2.3).
`sft+grpo` on the same data reached F1 66.7% (recall 7/7) against the SFT stage's 40.0% (recall 3/7). The
run prints a WARNING when every group's rewards are identical at the end (`group_std` 0 in
`RUN/grpo-log.jsonl`). Read the eval before trusting that checkpoint.

GRPO flags: `--num-generations` 4, `--max-completion-length` (16 for a label, 64 with a reason line, 8 for
decisions), `--beta` 0.04 (KL to the starting model), `--grpo-lr` 1e-5, `--grpo-iters` (default `--iters`
with `--method grpo`, else one pass over the train prompts), `--temperature` 1.0. `--batch` is prompts per step.

GRPO backends:
- **mlx** (Apple Silicon) needs `pip install mlx-lm-lora`. mlx-lm itself has no GRPO trainer. Our reward is
  loaded through its `--reward-functions-file`. Its trainer always scales each group by that group's std,
  which cancels a per-prompt weight. On mlx the classes are therefore balanced by sampling (the GRPO prompts
  are oversampled to 1:1), and the class weights come out at 1.0.
- **trl** (`pip install trl peft`; CUDA, CPU or MPS) uses `GRPOTrainer` + PEFT LoRA with
  `scale_rewards="batch"`, so the class weights are applied as weights. `--qlora` needs CUDA. trl 1.15
  computes GRPO log-probs with a Triton kernel and fails without Triton (macOS, most CPU boxes). The run
  stops with that message. On CPU/MPS use `pip install 'trl<1.15'` (1.14.0 ran on MPS).
- **unsloth** (CUDA) uses the same GRPOTrainer, loading through `FastLanguageModel`.
- On a Mac without mlx-lm-lora but with TRL installed, auto picks trl (MPS) and says so.

The lineage `tune` step records `train_method`, the reward config (class weights, format bonus, value field,
how balance was applied) and the reward at the start and end of training. `RUN/grpo-log.jsonl` holds the full
reward curve. The eval report shows the method.

**Typed decisions:** on a CUDA box with Unsloth installed, the run trains Unsloth's `FastDecisionModel`
(a clef joint head) and calibrates its temperatures. Ship then warns if the GGUF converter dropped the
`<arch>.decision.temperature.*` keys. Anywhere else it trains the **generation path** (each question is
one chat turn answered with the option name) and says so. Eval reads option probabilities from the
model's likelihoods.

## Read the eval before shipping

`RUN/eval/report.md` (also the card section) and `summary.json` hold one column per model:

- classify: precision / recall / F1 **on the positive class**, weighted recall, macro F1, accuracy,
  unparsed replies, latency P50/P99, and a confusion matrix
- chat: exact / contains (plus judge pass with `--judge-url`)
- decision: accuracy against the gold label, KL from the gold distribution, p(gold), and ECE

Ship only if the tuned column beats the base on the metric the user named, and on **weighted recall**
when there is a cost field. A model that buys precision by never flagging (recall falls) is a regression
even when accuracy rises; at a 5-18% base rate, "always legit" scores 82-95% accuracy. Report the numbers
as measured, including the losing ones, and include latency. If the tuned model loses, change the data
(more positives, matured labels, no leaks) or train longer before retrying. Don't change the metric.

## Serving: the agent is weights + prompt + policy

`RUN/<name>/agent.json` holds the exact system prompt and label set the model was trained on, and it
ships beside the weights. Serve with **that** prompt. A different prompt makes a different agent, and the
eval no longer describes it. Re-run `pollard-tune eval` after any prompt or policy change.

`pollard serve` (the harness lane) may not be installed. In that case ship prints the equivalent
`llama-server` line, and serving still works.

## Not in this lane (say so if asked)

- **Large-scale RL infrastructure.** Coinbase ran GRPO on SkyRL (Megatron + vLLM rollouts). `--method grpo`
  runs on one machine with in-process generation (mlx-lm-lora, or TRL without vLLM). That fits small and
  mid-size models and short label outputs, not long-reasoning rollouts.
