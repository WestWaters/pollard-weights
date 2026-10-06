# Controlled routing workloads

`experiments/vllm_routing_profiles.py` measures expert-selection coverage and
routing-weight coverage separately for code, arithmetic reasoning and general
text. Each domain has four original synthetic prompts at 256 and 1536 tokens.
These are controlled fixtures, not a representative sample of production
traffic. They are deliberately truncated token streams, not scored tasks.

The harness uses the same worker extension as `vllm_routing_check.py`. Run with
the hook and importing `sitecustomize.py` on the worker PYTHONPATH:

```sh
python experiments/vllm_routing_profiles.py \
  --model /model --capture-dir /capture --output /capture/profiles.json
```

Add `--moe-backend marlin` only if that is the supported backend for your model
and runtime. Capture runs use eager execution, disabled prefix caching,
disabled FlashInfer autotuning, chunked prefill and a 1024-token scheduler
budget. The default is one request at a time. `--batched` allows four concurrent
requests; it does not by itself prove a mixed prefill/decode step occurred.
`mixed_router_calls` counts individual router calls containing both phases,
not scheduler steps. Null means the recorder did not provide this counter.
This harness supports a single worker, not multi-node TP or EP.

Each interval excludes initialization and preceding profiles. The harness
checks every layer's prompt and decode-feedback totals and rejects errors or
unknown observations. The last generated token is not a routing observation.
It saves each raw snapshot before the next reset can replace it.

The public summary contains per-expert histograms, coverage curves and hashes,
not text or token IDs. Router names identify model modules, not host paths.
Layer numbers are ordinals in sorted router-name order, not the architecture's
layer numbers. Keep the raw snapshot directory and runtime logs
private unless separately reviewed. Review the aggregate file before sharing.

After capture, the original router is restored and all profiles are repeated
twice in the same engine. The report distinguishes control repeatability from
capture/control agreement. A failed comparison remains a failed comparison;
it should not be described as recorder transparency.

Cross-profile comparisons report Jensen-Shannon divergence in bits and the
target demand covered by the source profile's top-K experts. Source-selected
coverage is not the target's own top-K coverage. This distinction can show how
an expert cache fitted to one workload loses coverage on another.

These measurements do not establish task accuracy, expert-pruning safety,
quantization quality, full expert-combination reuse, or throughput. The Python
recorder synchronizes tensors and can affect scheduling. Measure speed
separately without instrumentation. NVFP4 checkpoint routing is not an original
BF16 reference for a matched-size quantization comparison.

## GB10 results

The [serial result](vllm-routing-profiles-gb10-serial.json) and
[batched result](vllm-routing-profiles-gb10-batched.json) use a cached Nemotron-H
NVFP4 checkpoint and a local vLLM 0.29 development image, not a stock release.
The 52 weight shards, tokenizer and other inference files were content-hashed
and checked against public revision
`e8f3c7c4de75ad84fe1bcef95d38eca76214480b`; the report includes config and
tokenizer hashes. This is not an original BF16 reference.
MoE execution uses Marlin weight-only FP4, not a native
FP4 speed comparison.

Each run captures 21,504 prompt tokens and 744 decode-feedback tokens across
the six profiles. All 23 router layers match the request totals, with no
unknown tokens or recording errors. Published routing mass is rounded to eight
decimal places; the private raw captures retain the original precision.

The batched run records 414 router calls containing both prefill and decode.
An earlier run counted a one-token prompt continuation as decode in all 23
layers: 6,143 prefill and 125 decode tokens instead of 6,144 and 124. Preserving
the scheduler phase flags corrects that failure without changing the backend's
kernel dispatch.

Five batched profiles match both unhooked controls. The short-code profile
matches the first control but the two controls differ. The aggregate control
flags therefore remain false. Correct accounting is not proof of unchanged
batched output or a resolved cause for that output difference.

These are small synthetic samples. The per-expert vectors support checking
cache coverage across workloads, not concluding that experts can be deleted
or that a quantized model has preserved quality. No multi-node run is included.

All six serial profiles match both unhooked repeats with the final recorder.
The following values are median selection coverage across the 23 routers,
choosing each router's top 32 experts from the same profile and phase:

| Profile | Prefill | Decode |
|---|---:|---:|
| Code, short | 57.5% | 61.6% |
| Code, long | 59.0% | 70.2% |
| Reasoning, short | 71.0% | 73.1% |
| Reasoning, long | 66.7% | 73.7% |
| General text, short | 66.4% | 76.6% |
| General text, long | 70.3% | 75.3% |

Using long-general-text decode to choose the top-32 cache, then applying it to
short-code decode, covers only 26.6% of selections at the median router. Choosing
the set from short code itself covers 61.6%. This demonstrates workload shift
in these fixtures, not expected coverage on real traffic. These are single
captures with four prompt variants per profile; uncertainty has not been
estimated. The short and long fixtures share text and are not disjoint
calibration and held-out evaluation sets for a quantization study.
