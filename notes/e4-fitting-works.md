# E4 — fitting the weights WORKS: 4.56 → 41.28 tok/s

_2026-07-29, `Laguna-XS-2.1`, 16 GB M4, measured with `llama-bench`._

## The result

| model | size | backend | prompt tok/s | **generation tok/s** |
|---|---|---|---|---|
| Q3_K_M | 14.95 GiB | CPU / flash (`ngl 0`) | 3.30 | **4.56** |
| Q3_K_M | 14.95 GiB | Metal (`ngl 99`) | — | **failed to allocate** |
| **Q2_K** | **11.46 GiB** | **Metal (`ngl 99`)** | **157.91** | **41.28** |

**9.1× faster generation**, from the same model, purely by making it fit. A 33.44B-parameter MoE
generating at 41 tok/s on a 16 GB Mac.

## The real constraint was never RAM — it's Metal's working set

```
recommendedMaxWorkingSetSize = 12713.12 MB     (on a 16 GB machine)
```

That is the number that matters, not 16 GB. Q3_K_M at 14.95 GiB doesn't just run slowly on the GPU,
it **cannot allocate at all** (`failed to decode prompt batch, res = -3`) and silently falls back to
the 4.56 tok/s flash path. The whole game is landing under ~12.7 GB with room for the KV cache and
compute buffers.

Sizes measured, requantised from the Q4_K_M file (4.85 bpw source):

| quant | bpw | size | fits Metal? |
|---|---|---|---|
| Q4_K_M (original) | 4.85 | 18.88 GiB | no |
| Q3_K_M | 3.84 | 14.95 GiB | no |
| **Q2_K** | **2.94** | **11.46 GiB** | **yes** |
| IQ3_XXS | 3.06 | ~11.9 GiB | untested — **requires an imatrix** |

## This is what the Pollard-weights work was actually pointing at

E2 and E3 killed the *prediction* half of the idea: expert usage is near-uniform (skew 1.13×) and
routing has no lookahead (cross-layer 4.1% vs 3.1% chance, temporal *below* chance, DP table 1.5%
held-out key hit). No index can prefetch a load-balanced router.

But the same measurements implied the fix. If demand is uniform and unpredictable, **stop trying to
predict and make everything resident** — and the gap was only ~3 GiB. Closing it with quantisation
produced the 9× directly. The guiding instinct in the session ("keep it from ballooning… compress further
and faster") was the correct read; the lever was bytes, not foresight.

## Quality: Q2_K is BROKEN — controlled, not assumed

Same flags (`-c 512 -b 64 -ub 64 -n 48 -st --temp 0.1`), same prompt (`def fib(n):`):

| model | output |
|---|---|
| **Q2_K** (11.46 GiB, Metal) | **nothing.** Empty generation, immediate exit. |
| **Q4_K_M** (control, CPU) | `[Start thinking]  Okay, I need to write a function called fib that calculates the nth Fibonacci number. Let me think about how to approach this. The Fibonacci sequence starts with 0 and 1, and each subsequent number is the sum of the previous two` |

The control matters: `llama-bench`'s `tg64` *does* emit 64 tokens from the Q2_K file, so generation
works mechanically and the empty output could have been a chat-template/EOS artifact. It isn't — the
identical invocation on Q4_K_M produces coherent reasoning. **Q2_K at 2.94 bpw, requantised from an
already-quantised Q4 file, destroyed the model.** MoE routers and shared experts are known to degrade
badly at 2 bits, and this is that.

Also note the model is a reasoning model (`[Start thinking]`), which makes low-bit damage worse: a
broken chain of thought wrecks the whole answer, not just a token.

### And it doesn't fit with any headroom

`llama-cli` OOMs **during model load** at `n_batch` 2048 *and* at 128 — the batch was never the
issue. Only `-c 512 -b 64` loads at all. 11.46 GiB of weights leaves ~1.2 GB under the 12.7 GB cap
for KV cache and compute buffers, which is not a usable configuration. (Verified not to be
contention: ollama held nothing, 84% of memory free.)

**So: the mechanism is proven, the artifact is not usable.** Both halves are true and neither cancels
the other.

## The actual target now

Need a quant that is simultaneously **≤ ~11.5 GiB** (or with the Metal cap raised) **and** not
damaged. Q3_K_M (14.95 GiB) is very likely fine on quality but 3.5 GiB too big; Q2_K fits and is
dead. The gap is bridged by *better* low-bit quantisation, not more aggressive round-to-nearest:

1. **Generate an imatrix from the Q4 weights, then quantise to IQ3_XXS (~11.9 GiB) or IQ3_S
   (~13.4 GiB).** The IQ family with a calibration matrix is dramatically better than Q2_K at
   comparable size — this is the single highest-value next experiment.
   Practical note: imatrix generation on the Q4 model at `ngl 0` is ~3.3 t/s prompt (hours). Use
   PARTIAL offload (`-ngl ~20`) to make it minutes.
2. `sudo sysctl iogpu.wired_limit_mb=14336` buys ~1.6 GB — enough to make loading comfortable and to
   admit IQ3_S. Needs care: other live workloads share this 16 GB machine.
3. Only then re-measure speed AND quality together. Speed alone was never the deliverable.

## ⚠️ Original caveats — this is a SPEED proof, not a usable model

1. **Q2_K requantised from Q4_K_M is doubly lossy.** ~2.9 bpw round-to-nearest, on top of an already
   quantised file. Expect materially degraded output. Coherence must be checked before anyone calls
   this "working" in the sense that matters.
2. **41.28 tok/s is ~12% of the theoretical bandwidth ceiling** (0.336 GiB active/token at 2.94 bpw
   against 120 GB/s → ~357 tok/s). MoE gather + many small matmuls account for the gap. So the
   remaining headroom is real but not free.
3. **170 tok/s generation was not reached.** Prompt processing hit 157.9 tok/s, but that is prefill —
   batched and compute-bound. For chat, generation is the number, and it is 41.
4. `-r 2` with ±4.06 variance: this is two runs, not a rigorous benchmark.

## Next, in priority order

1. **Quality, not speed, is now the bottleneck.** Do a proper quant from the ORIGINAL weights with an
   imatrix (IQ3_S / IQ3_XXS) rather than requantising a Q4 file. Better quality *and* likely still
   under the cap.
2. **Raise the Metal working set.** `sudo sysctl iogpu.wired_limit_mb=14000` on a 16 GB machine would
   admit a ~13.4 GiB IQ3_S — better quality at the same speed class. Needs care: too aggressive
   destabilises the system, and other live workloads share this machine.
3. **Measure quality honestly** — perplexity against the Q4 baseline, plus a real task check, before
   this replaces anything.
4. From the same session: **train/distil into a smaller footprint** (QAT) is the
   principled version and would beat any post-hoc quant at 3 bits.

---

## E6 — the Metal wall (2026-07-30)

Four expert-only variants, all with the same imatrix, tested on llama.cpp **b2f2216** (latest master,
86 commits newer than the original build — updating did NOT change the outcome):

| expert quant | size | Metal | gen tok/s | coherent |
|---|---|---|---|---|
| Q2_K | 10.80 GiB | ✅ | 46.6 | ❌ empty |
| Q3_K (down) + Q2_K (gate/up) | 11.79 GiB | ❌ compute error | — | — |
| IQ3_XXS | 12.40 GiB | ❌ compute error | — | ✅ (on CPU) |
| blanket Q2_K (all tensors) | 11.46 GiB | ✅ | 41.3 | ❌ empty |

Two independent facts, each measured:

1. **Quality floor: experts die below ~3 bpw.** Expert-only Q2_K is empty just like blanket Q2_K, so
   the earlier hypothesis — that crushing attention/router was to blame — is WRONG. Sparing them
   costs 1.2 GiB and buys nothing. IQ3_XXS at 3.06 bpw is coherent. The threshold is the expert
   bit-rate, nothing else.
2. **Runtime ceiling: Metal can't compute Q3_K or IQ3_XXS in `mul_mat_id` for this model.** Fails at
   ANY offload level (tested ngl 10/20/30/36/38/99), so it is not memory — `iogpu.wired_limit_mb`
   would not have helped. Q2_K works on the same build, isolating the cause to the quant type.

**Therefore: no llama.cpp+Metal configuration of Laguna-XS-2.1 is both coherent and fast on 16 GB.**
The viable window (3-3.5 bpw) is exactly where the Metal MoE kernels fail.

### Remaining options
- **MLX** (`mlx-community/Laguna-XS-2.1-3bit`, 13.65 GiB) — different runtime, Apple-native, manages
  unified memory directly rather than honouring `recommendedMaxWorkingSetSize`. 3-bit sits above the
  measured damage threshold. Best remaining bet.
- **CPU + IQ3_XXS** — coherent but ~1.6 tok/s. Correct, unusable for chat.
- **Report the Metal MoE failure upstream** — a reproducible compute error on a public model.
