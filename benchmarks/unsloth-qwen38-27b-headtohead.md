# Head-to-head: Pollard vs Unsloth Dynamic 3.0 — Qwen3.8-27B

Unsloth's `Qwen3.8-27B-GGUF` is the most-liked GGUF on Hugging Face: 10.7M downloads and 3.7K likes
in 24 days. It is the right thing to measure against, because it is what people actually run, and
because "Dynamic 3.0" is a mixed-allocation method too — the honest question is not whether mixed
beats uniform, it is whose allocation is better **at the same size**.

## Why this comparison can be fair

Both repos publish the same base model, and two rungs land within a few percent of each other. Size
is the variable that has to be controlled: a quant that is 10% larger should win, and reporting that
as a method win is the standard way this comparison gets faked.

| pair | Pollard | Unsloth | size gap |
|---|---|---|---|
| **headline** | `Qwen3.8-27B-Pollard-IQ3_S` 12.08 GB | `Qwen3.8-27B-UD-IQ3_S` 12.04 GB | **+0.3%** |
| secondary | `Qwen3.8-27B-Pollard-Q6_K` 22.43 GB | `Qwen3.8-27B-UD-Q6_K` 21.98 GB | +2.0% |
| loose | `Qwen3.8-27B-Pollard-IQ4_XS` 15.72 GB | `Qwen3.8-27B-UD-Q4_K_S` 15.36 GB | +2.3% |

The **IQ3_S pair is the experiment**: 0.3% apart in bytes, same base model, same file format, same
runtime. Whatever separates them is allocation. Report the others as secondary, and state their size
gap next to every number, because at 2% they are suggestive rather than decisive.

## Protocol

Everything below is held identical across both sides. Any difference here invalidates the result.

- **Runtime**: one `llama.cpp` build, one binary, run back to back on one machine.
- **Eval**: `wikitext-2` raw test, ctx 512, all chunks. The same file for both.
- **Reference for KL**: Unsloth's `Q8_0` (29.05 GB) as the near-lossless host, not BF16 — the BF16
  is 54.7 GB across two shards and the box has ~75 GB free, which does not leave room for BF16 plus
  both candidates. `pollard-bench --ref` documents Q8_0/Q6_K as an accepted host. Say so in the
  result; a Q8_0 reference measures agreement-with-Q8_0, not agreement-with-BF16.
- **Metrics**, in the order they matter:
  1. **Mean KL vs the reference** and **top-1 agreement** (`pollard-kl`) — the metric that survives.
     PPL can tie while behaviour diverges.
  2. **PPL** — the number everyone quotes, kept for comparability.
  3. **tok/s**, single stream, same `-ngl`, same prompt.
- **Sampling** for any generation: fixed seed, temp 0.7, top-p 0.9, repeat-penalty 1.15, ChatML.
- **Coherence**: `pollard-bench --coherence` on both. A build that loops is not competitive at any
  perplexity.

## Reporting rules

- Publish the losses too. If Unsloth wins a rung, that is the result, and it tells us where the
  allocator needs work.
- Every number carries its size gap. A win under +2% size is not a clean win.
- Single machine, single run — state it. Invite replication.
- ⚠️ Do NOT quote Unsloth's own published quality claims as the comparison. Measure their artifact
  on our bench, or it is not a head-to-head.

## Cost

The two IQ3_S files are ~24 GB of download plus the ~29 GB reference. Three full-wikitext perplexity
passes and one KL pass over a 27B on a 16 GB-VRAM box means partial offload; budget hours, not
minutes, and run it when the GPU is free (see the 60/40 rule — this is a GPU job).

## Status

Not yet run. `benchmarks/run_unsloth_h2h.sh` executes it end to end and writes `results.json` in the
shape `pollard-scorecard` consumes. The Pollard 27B card currently ships placeholder rows
("(see repo)", "—") for PPL and tok/s; this is what fills them.
