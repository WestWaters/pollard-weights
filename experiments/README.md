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
| `pollard_decode.py` | Does composed decoding (CompoSimplex, arXiv 2609.34992: KL/coverage/diversity regularisers solved on the simplex) buy back MORE accuracy on a low rung than on f16? Same body at f16 and RTN-crushed, GSM8K subset, K lockstep samples, the paper's own loop and grader. | transformers + torch, the composimplex clone, small model |
| `mattr_to_tensor_types.py` | Turns a learned (MAttr) tensor ordering into a `llama-quantize --tensor-type-file` at a reference rung's exact per-atom byte budget, shape-aware (tensors no IQ atom can take are fixed at iq4_nl on both sides) — the like-for-like GGUF test of a learned allocation. | gguf-py, a reference `.tensor-types.txt`, a `pollard_mattr` result |

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

**On real GGUF rungs (2026-09-30).** Same bytes, same atoms, only the placement differs (`mattr_to_tensor_types.py`;
allocations in `experiments/allocations/`). Held-out wikitext-2, `pollard-bench --ref f16`:

| model | rung | placed by | size | PPL | MeanKLD | MedKLD | top-1 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | IQ3_S | probe (layer groups) | 337,957,376 | 14.03 | 0.1002 | 0.0685 | 83.8 % |
| | | **learned (MAttr)** | 337,957,376 | 13.96 | **0.0988** | **0.0655** | **84.1 %** |
| Qwen2.5-1.5B-Instruct | IQ3_S | **probe (layer groups)** | 730,086,848 | **10.66** | **0.2154** | **0.1398** | **77.8 %** |
| | | learned (MAttr) | 729,705,920 | 11.13 | 0.2742 | 0.1769 | 74.3 % |

Reading: the 0.5B is a tiny action space — 240 of its 264 block tensors have a first dimension of 896 or 128, which no IQ
atom accepts, so `llama-quantize` makes them iq4_nl whatever the allocator says and only the 24 `ffn_down` tensors are in
play; the learned order edges it there (−1.4 % KLD). The 1.5B is fully allocatable and the learned order **loses by 27 %
KLD**. In the torch proxy the same order was 2.8× *better* than the probe's. The proxy crushed rows with uniform RTN-2;
the build crushes tensors with imatrix-weighted IQ atoms, and those damage different things — the probe's layer plan is
measured closer to what the quantizer actually does. Build ≠ proxy, in numbers. What would make the learned order
real: score against the actual atom crush (dequantized iq2/iq3 with the imatrix as the "other" checkpoint) and learn
at tensor granularity directly, since that is what a GGUF can express. Until then the probe/sensitivity path stays the
allocator.

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

### Composed decoding on a low rung (`pollard_decode.py`) — Qwen2.5-0.5B-Instruct, 2026-09-29

30 GSM8K test questions · K=3 lockstep samples (greedy K=1) · 320-token budget · paper's grader · M4 MPS. pass@1 = mean over samples, pass@3 = any.

| decoder | f16 pass@1 / pass@3 | RTN-4 pass@1 / pass@3 | RTN-3 pass@1 / pass@3 |
|---|---|---|---|
| greedy | 0.233 / 0.233 | **0.167** / 0.167 | 0.000 / 0.000 |
| top-p 0.95, T 0.7 | 0.267 / 0.467 | 0.144 / 0.300 | 0.000 / 0.000 |
| KL + coverage (paper's Best-of-K) | 0.244 / 0.533 | 0.144 / **0.367** | 0.000 / 0.000 |
| KL + diversity | **0.278** / **0.600** | 0.122 / 0.333 | 0.022 / 0.067 |

Reading: on f16 the paper holds — every sampler beats greedy on pass@1 (+1 to +4.5pp) and the composed ones lead pass@3 (0.53–0.60 vs
0.47 top-p, 0.23 greedy). On the crushed rung the gain **inverts**: greedy is the best pass@1 and all three samplers sit below it;
composed decoding still leads pass@3 (0.367 vs 0.300) but by less than at f16. A crushed model's distribution is noisier, and a
regulariser that spreads mass across the reference's top tokens spreads it over more wrong paths. So this is a generic multi-sample
lever, not a low-rung lever: nothing here justifies carrying a sampler port in llama.cpp for Pollard's sake. n=30 puts ±8pp on
pass@1, so the individual gaps are inside noise — the *direction* (sampler gain shrinking under crush) is consistent across all three.
Open at scale: the paper's gains were at 1.2B–26B; a real 7B GGUF rung would need the sampler in llama.cpp first, and that is only
worth writing if some other reason appears. Parked.
