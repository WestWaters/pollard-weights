# Pollard benchmarks — OPTIONAL

**You do not need anything in this folder to make a Pollard model.** The build (quantize)
produces the model; everything here only *measures* how good it already is. These are the
gold-card reproduction and R&D tools — they were how *we* proved Pollard works and tuned the
allocation. Run them only if you want to verify or reproduce our published numbers.

## Just want the model? (the fast path — minutes)

```bash
pollard --gguf model-f16.gguf --run          # detect dense/MoE, build ONE model, no eval
```
Dense → imatrix-guided K-quant ladder. MoE → the automap PollardMix. No comparison bars, no
perplexity, no sensitivity sweep. That's the whole job.

## Want to reproduce the gold-card numbers? (opt-in — hours)

```bash
pollard --gguf model-f16.gguf --benchmark --run     # ALSO build the 3-bar comparison + PPL
```
This adds the **benchmark**: the 3-bar board (uniform-low / PollardMix / uniform-high) and the
full metric suite, so you can confirm PollardMix beats uniform at matched size.

### The pieces (run individually if you want)

| tool | measures | notes |
|---|---|---|
| `pollard-eval`  | trajectory-divergence quality | quick sanity |
| `pollard-kl`    | KL-to-f16 + top-1 (torch) | small models |
| `llama-perplexity --kl-divergence` | Mean/Median KLD + top-1 | big models, vs an f16/Q8/Q6 base |
| `pollard-scorecard` | the standardized gold-card scorecard | from a results.json |
| `pollard-sensitivity` | measured per-group KL profile (the MoE knapsack input) | **hours** (2·layers passes), MoE-only |

## Drop-in benchmark: `pollard-bench`

`pollard` builds; **`pollard-bench` scores** — the symmetric half. Point it at a model (or two)
and get the whole board (PPL + Mean/Median KLD + top-1) from one command, no hand-run flags:

```bash
pollard-bench --gguf model.gguf --ref f16.gguf --eval held.txt          # one model, full board
pollard-bench --gguf pollard.gguf --vs rival.gguf --ref f16.gguf        # HEAD-TO-HEAD (Pareto verdict)
pollard-bench --gguf model.gguf --out results.json                     # -> pollard-scorecard for the card
```
`--ref` is the KL reference (f16, or a near-lossless Q8_0/Q6_K host if f16 won't load). `--vs`
scores a competitor's file (AWQ/GPTQ/unsloth/bartowski GGUF) the **same** way, so a "Pollard beats X"
comparison isn't a special mode — it's the same harness at matched size. Read it **Pareto:
Pollard wins its size class** (same quality for fewer GB, or more at the same GB), not one number.

## Bring your own eval

Pollard's job ends at the **GGUF/GPTQ** — the model. After that it's a standard file: point
**any** framework at it (lm-eval-harness, your own suite, a downstream task). Most people will
just `pollard --gguf model.gguf --run` to get the model, then run whatever they already run.
That's the design: **make the model first, benchmark however you want.**

## Why this is separated

Shipping the benchmark *inside* the default build made a simple "shrink my model" run take
hours (three builds + evals instead of one). The model was never the slow part — the
measurement was. So: **build = default and fast; benchmark = here and opt-in.**
