# Checking the vLLM routing recorder

A [GB10 accounting and control result](vllm-routing-check-gb10.json)
is included for a Nemotron-H MoE on a local vLLM 0.29 development image. All 23
layers matched 68 prompt and 188 decode-feedback tokens; serial same-engine
greedy outputs matched two repeats after removing the hook. Fresh-process
four-request outputs did not match, so this is not batch-invariance or model
quality evidence. The checkpoint revision is unknown and the image is not a
stock release. Use the supplied check to validate your own pinned runtime.

Use an isolated measurement job with CUDA graphs and speculative decoding off.
The Python hook does not observe CUDA-graph replay, and multi-token speculative
verification cannot be classified by query length alone. Disable prefix caching
for the initial comparison; retain the runtime's supported chunked-prefill
setting and record it.

For FlashInfer attention, the recorder preserves the common attention metadata's
`is_prefilling` flags while building the backend metadata. Its decode-kernel
token count is not a request-phase count: a one-token chunked-prefill
continuation can use a decode kernel. If the bridge has no scheduler flags,
observations stay unknown. Removing the recorder also restores the original
metadata builder.

Other supported attention metadata uses scheduler flags where exposed, otherwise
query boundaries and sequence lengths. The recorder keeps
unavailable/inconsistent metadata and padding in an
`unknown` bucket. Unknown observations are excluded from the decode/prefill
comparison, not silently counted as prefill. One-token queries are classified
using sequence lengths where available. That fallback remains a query-length heuristic:
one-token chunked-prefill continuations can still look like decode. Confirm the
split against controlled request schedules before drawing a routing conclusion.

Pass `worker_extension_cls="vllm_decode_routing_hook.RoutingWorkerExtension"`
to `LLM` (or `--worker-extension-cls` to the server). After initialization, call
`llm.collective_rpc("pollard_start_capture")`. This labels routers from model
module paths and resets warmup observations. Call
`llm.collective_rpc("pollard_flush_capture")` at the end. These are named worker
methods, so enabling insecure callable serialization is unnecessary. Server
users need their controller to invoke these worker methods at the capture
boundaries; importing sitecustomize alone does not reset warmup.

Without layer-name assignment, fallback `router0` keys are observation-order
labels, not established architectural layer numbers.

The installation path is unchanged: put the hook and a `sitecustomize.py` that
imports it on the workers' PYTHONPATH. Each process writes its own PID-qualified
file with a random instance suffix, including across forks and containers.
Short runs flush at normal interpreter exit; SIGKILL cannot flush. For a
controlled offline job, use `collective_rpc` to call each worker capture's
`flush()` before shutting down. A small dump interval limits losses on failures.

For several dumps, choose a rank interpretation explicitly:

```sh
python experiments/vllm_decode_routing_hook.py --analyse captures/*.json --rank-mode replicated
```

`replicated` verifies matching layer keys, counts and routing mass, then counts
one TP replica. `disjoint` sums observations from different token sets. Do not
merge old periodic snapshots of the same cumulative capture. Expert parallelism,
pipeline parallelism, load-balanced expert IDs and changing rank layouts need
separate validation; the hook does not infer them.

For the first live check, match per-layer totals against the request token
counts. Check a prompt-only batch, a decode-only batch and a mixed batch. Check
that router outputs are unchanged with and without recording. Capture throughput
with instrumentation is not a serving benchmark: use an uninstrumented run for
speed comparisons.

## Reproduce the accounting check

Use a fresh capture directory shared by the driver and workers. Put the hook
and the importing `sitecustomize.py` in a separate directory on worker PYTHONPATH.
Set `VLLM_ROUTING_DUMP_DIR` to that shared capture directory, then run:

```sh
python experiments/vllm_routing_check.py --model /path/to/moe-model \
  --capture-dir /path/to/capture --output capture-summary.json
```

The check disables CUDA graphs, speculation, prefix caching and FlashInfer
kernel autotuning, uses TP1, and
resets recording after warmup and before each batch. Four fixed synthetic
requests first generate one token each (no decode feedback), then 48 tokens each.
Every recorded layer must match the prompt-token count and the generated-token
count minus one final token per request. Unknown tokens or recording errors fail
the check. Backend-specific overrides can be passed with `--moe-backend`.

Start a separate process with the hook directory removed from PYTHONPATH (and
without the importing sitecustomize), omit `--capture-dir`, and run again:

```sh
python experiments/vllm_routing_check.py --model /path/to/moe-model \
  --output control-summary.json
```

Compare the `output_sha256` values in corresponding batches. A match verifies
greedy output on these requests only. It is not a general quality test, does not
prove a mixed prefill/decode scheduler batch occurred, and does not validate TP
replication or expert parallelism. Run those separately before publishing a
multi-node routing conclusion. Only these fixed synthetic outputs are hashed;
do not use output fingerprints to publish private evaluation data.

If separate processes disagree, do not attribute that to the recorder without
checking repeatability of the unhooked runtime. For a more controlled comparison,
add `--serial --same-engine-control` to the capture command. The check processes
one request at a time, flushes and restores the original router through
`pollard_remove_capture`, then repeats both batches twice without the hook in
the same engine. `control_repeats_match` checks baseline repeatability and
`capture_matches_controls` checks the captured results against both controls.
Removal is one-way for that worker; start a new engine to install the hook again.
Neither flag should be read as proof that a four-request or multi-node batch is
unchanged, or that capture overhead is negligible.

Only aggregate expert counts and routing mass are written, not prompts or
generated text. Review model names, layer names, timestamps and metadata before
publication. Keep raw server logs private.
