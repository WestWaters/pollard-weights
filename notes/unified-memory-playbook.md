# Unified-memory & cluster playbook (DGX Spark / GB10, Grace-Blackwell, Apple)

Running a big quantization job on a **unified-memory** host (the GPU and the CPU share one pool)
breaks assumptions that hold on a discrete-VRAM card. These are the field-tested rules from the
GLM-5.3 744B run (see [cluster-scale-roadmap.md](cluster-scale-roadmap.md)). Generic first; add
hardware-specific quirks as reports arrive.

## 1. Memory pressure is the #1 killer — model it before you run

On unified memory there is no separate VRAM to "spill" into: the model, the KV cache, and the OS
page cache all fight for the *same* bytes. Over-commit and the box pages to disk (Linux) or swaps
(macOS) and throughput collapses to a fraction while `nvidia-smi` still says 96% util (the wedge).

- **Predict first:** `pollard-calc --model <hf> --ram <your GB> --ctx <len> --kv-quant nvfp4 --gpu <rig>`
  puts *your* rig on the fit ladder and tells you whether weights + KV + activations fit before you
  download anything.
- **Diagnose live:** `pollard-health` reads the signals that actually matter (real SM clock vs the
  card's own max, power vs limit, page-outs, SoC thermal zones on aarch64) and calls the wedge
  plainly. `pollard-health --fix` prints a no-reboot recovery plan.
- **Drop the page cache between jobs** so a finished job's file cache doesn't starve the next:
  `sync && echo 3 > /proc/sys/vm/drop_caches` (Linux), `sync && sudo purge` (macOS). When streaming
  a model off disk, `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` after reading each shard keeps the
  cache from ballooning.

## 2. One GPU job per node

On a shared unified pool, two concurrent GPU processes each reserve context and the second one tips
the box into page-out thrash. Run **one** quantize/serve process per node; use the band-parallel
plan (below) to spread a too-big model across nodes instead of over-subscribing one.

## 3. Meta-device loading — cast dtype on `meta` BEFORE materializing

Loading a big model with `device_map` and then casting to fp16/bf16 reserves a transient fp32 copy
that can OOM the pool. Cast the dtype on the **meta** device first, then `to_empty()` onto the real
device and load weights — the fp32 reserve never happens. (transformers: `from_pretrained(..., dtype=...)`
already does the right thing; watch for manual `.float()` calls in custom loaders.)

## 4. Damped-Cholesky robustness — already handled

A near-singular Hessian (huge `o_proj`, massive-activation channels, or a cold MoE expert that saw
few tokens) makes GPTQ's Cholesky non-positive-definite. `pollard-gptq` now:
- treats a channel as dead when `diag(H) ≤ 1e-10·mean` (not just exactly 0),
- escalates damping ×10 → ×100, then falls back to **fp64**, and only then errors — with a message
  telling you to raise `--percdamp` or widen the calibration set,
- applies a **token floor**: an expert with fewer tokens than input channels keeps only the diagonal
  Hessian (per-channel importance) and drops the unreliable off-diagonal, rather than trusting noise.

Nothing to configure — it just no longer crashes a layer.

## 5. CUDA free-memory under-reports on unified hosts

`torch.cuda.empty_cache()` + `cudaMemGetInfo` report free memory that keeps falling across a long
run even when you free tensors — the allocator's view of a shared pool drifts. The reliable
workaround is **one process per layer/band** (a fresh CUDA context reclaims everything on exit); the
band-parallel path does exactly this. Don't trust `mem_get_info()` as your fit gate on these boxes —
trust `pollard-calc`'s byte accounting.

## 6. Band-parallel export for models too big for one box

A 744B model is ~1.5 TB in BF16 — no single box holds it. Plan the split:

    pollard-export --model <hf> --layers <N> --shard-plan <n_nodes> --bf16-gb <total>

This prints the contiguous layer band each node owns, its byte budget, and the **handoff contract**:
each node quantizes its band with the offload path, re-runs the quantized band to produce the hidden
state at its last layer, and passes *that* to the next node (node 0 starts from the embedded calib
tokens). Contiguous bands mean only one boundary hidden state crosses the wire between neighbours.

**The wall is usually storage egress, not compute** — ~100 MB/s/box was the field number; budget the
read/write of 1.5 TB accordingly, and keep shards local to the node that owns them.

## 7. Byte-accounting is the speed model on bandwidth-bound hosts

On a unified/bandwidth-bound host, decode speed is set by **bytes read per decoded token**, not
FLOPs. `pollard-calc` reports this directly. A non-obvious consequence seen on GLM-5.3: INT8
attention can read *more* bytes/token than the routed experts do, so quantizing attention harder buys
more speed than crushing more experts. Read the bytes/token line, not just the total size.

## 8. Served-model A/B — verify on the real stack, not just offline

Offline PPL is not what ships: the served model runs a different kernel path (paged attention, fused
Marlin, a KV quant). Confirm on the endpoint:

    pollard-serve-eval --base http://baseline:8000/v1 --model fp16 \
                       --cand http://cand:8001/v1 --cand-model int4 --text held_out.txt

Reports teacher-forced perplexity, top-1 agreement, and KL(base‖cand). >99% top-1 agreement with
KL < ~0.05 means the quant behaves like the original. A large PPL gap *with* high agreement is
usually a KV-quant or kernel-path artifact, not the weights.

## 9. Gate hygiene — exclude calibration/held-out overlap

A held-out "eval" set that overlaps the calibration/replay corpus inflates every number (158 of 162
held-out texts turned up in the replay corpora on one run). `pollard-serve-eval --calib <file>` does this
for you: it normalizes (whitespace + case) and hashes every calib line and **drops any eval line that
matches**, printing how many it excluded — so an accidental overlap can't quietly inflate the score. Pass
your calibration corpus with `--calib` whenever the eval set might share text with it.

## 10. Spec-decode head — ship the authors' stock MTP head

When you quantize the body of a model that has a multi-token-prediction / speculative-decode head,
**ship the model authors' stock MTP head with the new quantized body.** Retraining the head on
quantized-body captures did *not* close the acceptance gap in the field (1.80 → 1.36 tokens/step) —
it made it worse. Leave the head as published.
