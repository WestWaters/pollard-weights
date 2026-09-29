# Harnesses — reproduce our numbers, or produce your own

Each experiment is a standalone measurement you can run against **your** model
and **your** machine. They are numbered in the order the questions arose; each
docstring states what it measures, why, and what result would kill the idea.
Raw captures/logs from our runs stay out of git — the scripts regenerate them.

| Harness | Question it answers | Needs |
|---|---|---|
| `e1_measure_sparsity.py` | Is there a small active set in a **dense** FFN at all? (Killed the naive idea for SwiGLU-era models — run this first on any dense model.) | transformers + torch, small model |
| `e2_capture_routing.cpp` + `e2_build.sh` | Records which experts actually fire per token in a llama.cpp run (the raw routing trace everything below replays) | llama.cpp checkout |
| `e2_analyse_routing.py` | Does routing **reuse**? Pattern-repeat rate, per-expert frequency curve, depth dependence | a routing trace |
| `e3_predictability.py` | Can layer L's experts be known **before** L's router runs (lookahead for prefetch)? | a routing trace |
| `e5_cache_sim.py` | Replay the trace against a bounded hot-expert cache: hit rate and bytes-from-flash per token at each cache size — the residency curve `pollard-calc`'s verdict points at | a routing trace |
| `pollard_mattr.py` | Can ONE training run (Matryoshka Attribution, arXiv 2609.25518: sigmoid top-k mask at a random k every step) learn the whole bit-allocation ordering that `pollard-probe` finds by crushing 2×layers groups one at a time? Interpolates every weight row between f16 and its RTN-2 crush, KL-to-f16 loss; evaluates the learned order vs the probe's vs random at equal budgets on held-out text. | transformers + torch (MPS ok), small model, a `pollard-probe` sensitivity.json |

Typical flow on a new MoE model:

```bash
./e2_build.sh                       # builds the capture shim against llama.cpp
# run your model with the shim to produce routing.jsonl (see e2_build.sh header)
python3 e2_analyse_routing.py routing.jsonl
python3 e3_predictability.py routing.jsonl
python3 e5_cache_sim.py routing.jsonl --expert-mb 38 --cache-gb 4 8 12
```

If you run these on a model we haven't measured, open an issue with the
concentration curve — that's the dataset this project exists to build.

### Learned allocation (`pollard_mattr.py`) — first number, Qwen2.5-0.5B-Instruct, 2026-09-28

300 steps · 16×512-token calib chunks · 6 held-out wikitext-2 chunks · RTN-2 g64 crush · M4 MPS · 5 min total.
304,128 rows over 168 decoder tensors. f16 sanity KL = 0.0000; everything crushed KL = 8.68.

| rows crushed | learned (row) | learned → tensor | learned → attn/ffn group | `pollard-probe` (group) | random |
|---|---|---|---|---|---|
| 20 % | **0.288** | 0.592 | 1.241 | 1.390 | 0.799 |
| 50 % | **0.925** | 1.534 | 3.342 | 2.918 | 4.033 |
| 85 % | **2.513** | 3.069 | 6.408 | 7.079 | 7.788 |

Reading: at the probe's own granularity the learned order and the sweep tie (both ≈ random at low budgets — crushing whole
groups is what hurts). The gain is granularity, from one run: tensor-level (what a GGUF can express) halves the KL versus
the group sweep at every budget; row-level halves it again (an upper bound — GGUF/EXL3 pick one type per tensor).
The learned tensor ranking recovers the allocator's rule of thumb from data: protect `attn_v` > `ffn_down` > `attn_k`,
crush `attn_q` > `ffn_gate` > `ffn_up` first. Spearman vs probe at tensor level ≈ 0 — the two orderings genuinely differ.
Next: same run on a 3B, then feed `protect_first` into `pollard-automap` and measure a real rung's KLD against the probe-built one.
