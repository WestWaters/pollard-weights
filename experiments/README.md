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
| `pollard_recovery.py` | Does a LIFT-style recovery vector (arXiv 2609.31140: mean hidden-state difference, strong minus degraded, injected at one layer) buy back quality on a low rung? Here strong = f16, degraded = the RTN crush of the same model, so the vector is a constant — a bias the GGUF can carry for free. | transformers + torch (MPS ok), small model |

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

**Qwen2.5-1.5B-Instruct, same setup (645,120 rows over 196 tensors, 12 calib / 4 eval chunks, 300 steps):**

| rows crushed | learned (row) | learned → tensor | learned → group | `pollard-probe` (group) | random |
|---|---|---|---|---|---|
| 20 % | **0.427** | 0.558 | 1.547 | 0.841 | 0.584 |
| 50 % | **0.988** | 1.072 | 3.269 | 2.995 | 2.884 |
| 85 % | **2.458** | 2.568 | 8.645 | 6.890 | 8.651 |

Holds at 3× the size: tensor-level learned order is 2.8× lower KL than the probe sweep at 50 % crushed, and at 1.5B the group
sweep is no better than random at any budget — whole-group crushing is the wrong unit, whichever order you crush in. Same type
ranking as the 0.5B, from data: protect `attn_v` > `ffn_down` > `attn_output`, crush `attn_q` > `ffn_gate` > `ffn_up` first.
Next: feed `protect_first` into `pollard-automap` and measure a real rung's KLD against the probe-built one.

### Recovery vectors (`pollard_recovery.py`) — first number, Qwen2.5-0.5B-Instruct, 2026-09-29

RTN-3 g64 on all 168 decoder linears · 12×512 calib chunks · 6 held-out wikitext-2 chunks · M4 MPS · 96 s.
Vector r_L = mean over calib tokens of (f16 hidden_states[L] − crushed hidden_states[L]), carried as the `down_proj` bias of block L−1.

| what | KL to f16 | Δ | top-1 |
|---|---|---|---|
| crushed baseline | 1.035 | — | 51.7 % |
| r @ layer 22 (μ=1) | 0.923 | −10.8 % | 54.5 % |
| r @ layer 4 | 0.924 | −10.7 % | 53.8 % |
| **r @ 22 + r @ 4 jointly** | **0.815** | **−21.2 %** | **56.8 %** |
| r @ 22 + 4 + 20 | 0.861 | −16.8 % | 56.8 % |
| every block 0..22, sequential | 0.989 | −4.5 % | 52.5 % |
| every block 0..23 (incl. final) | 1.259 | +21.7 % | 48.8 % |

Reading: two constant vectors (2 × 896 floats, zero runtime cost) recover a fifth of a 3-bit crush's KL. μ=1 is the right scale
(0.5 and 1.5 both worse). The final hidden state (norm 47 vs 2–12 elsewhere) must be left alone — its vector alone is +83 %.
More vectors is not better: the mean shift is a lever for a couple of layers, not a per-block correction.

**Qwen2.5-1.5B-Instruct, same setup (28 blocks, 196 linears, 4 eval chunks, 440 s):** baseline KL 0.693 → best single vector
(layer 26) 0.634, **−8.4 %**, top-1 60.5 → 63.2 %; layers 10–26 all help (−2 to −8 %), layers 2–8 hurt (+5 to +23 %). Joint 26+24 is
*worse* than 26 alone (−3.0 %); every-block mean-shift is catastrophic (+300 %). The lever shrinks with scale — a fifth of the KL at
0.5B, under a tenth at 1.5B — and stops stacking. Below the 10 % bar; not worth a runtime patch on its own at this size. Where it
could still pay: a 2-bit rung (larger residual mismatch to correct) or as a free add-on to a build that is already protected by
the learned allocation. Baking path, if it comes back: llama.cpp's Qwen2 graph already passes `wo_b`, the loader just never
creates it (one `TENSOR_NOT_REQUIRED` line), and gguf-py `add_tensor` writes the extra bias tensors.
