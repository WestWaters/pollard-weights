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
