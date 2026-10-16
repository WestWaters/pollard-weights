# pollard-qat foundation: passthrough, Jaccard KD, bit-exact fake quant (2026-10-15)

Three pieces the allocation-conditioned QAD plan (probe -> allocate -> QAD on the exact tensor-type map ->
emit -> verify) needs before any training run, and that NVIDIA's Nemotron-3.5-Bolt release makes urgent.

## 1. Quantized-GGUF passthrough (`tools/pollard_passthrough.py`)

Bolt "Lane A": start from NVIDIA's QAT GGUF (experts IQ2_XXS trained in, Mamba ssm_in/out NVFP4, attention /
embeddings Q8_0) and let Pollard's measured allocation decide only which expert layers drop to IQ1_M / IQ1_S.

- **Can llama-quantize copy as-is?** Yes, at the pinned runtime: `llama_model_quantize_impl` sets
  `quantize = cur_type != new_type` and, when false, writes the source bytes in row slabs. There is no
  `COPY`-per-tensor keyword (`COPY` is a whole-file ftype), so passthrough names every quantizable tensor
  with an anchored exact-name `--tensor-type-file` line (`^blk\.1\.ffn_up_exps\.weight$=iq2_xxs`) giving its
  current type. With exact anchors no two lines can match one tensor, so the first-match-wins rule and the
  base preset (Q8_0, chosen because it never demands an imatrix) decide nothing. No direct GGUF writer was
  needed.
- **Planner:** every tensor starts at its current type; greedy cheapest-KL-per-byte downgrades inside
  `--passthrough-classes` (default `bulk`: routed experts + dense FFN) until the budget fits. Upgrades are
  refused (plan or `--tier`). IQ2/1-bit candidates only where the imatrix covers the tensor; with no imatrix
  an IQ2_XXS source has nothing lower.
- **Verify:** after the build, every tensor planned at its current type is re-hashed against the source
  (sha256 of the data range); the build fails on any difference.
- **Requant source caveat:** a downgraded tensor is quantized from dequant(current), not from bf16 (two
  roundings). Correct for QAT codes; for a plain PTQ file a rung from bf16 usually wins.
- **Measuring on the quantized host:** `pollard-sensitivity --passthrough` uses the file as its own reference
  (base logits from it; a passthrough build of the whole file is the file, KL 0 by construction), crushes one
  layer's changeable tensors at a time to the lowest legal type, and measures the noise curve relative to
  the file. `pollard --gguf <quantized> --ladder` auto-builds the imatrix on the quantized host, runs that
  sweep (`--allow-slow` for a big model; `--no-measure` = uniform), and plans rungs PT25/50/75/100 between
  the source size and its floor. No trellis flagship on this path.
- Tested on a synthetic llama GGUF with real ggml blocks (IQ2_XXS / Q4_K / Q6_K / Q8_0 / F16 / F32): untouched
  tensors byte-identical, requantized ones equal ggml's `quantize(dequant(source), imatrix)` exactly
  (IQ2_XXS -> IQ1_S / IQ1_M, Q4_K -> Q2_K).
- For Bolt itself the runtime must load `nemotron_h_moe` (llama.cpp >= cd0a5b6, #30520; the pinned runtime
  df03399b is too old) — passthrough inherits whatever the runtime can load.

## 2. Weighted-Jaccard distillation loss (`pollard-forge train --loss jaccard`)

Bolt tech report section 5: `1 - sum(min(ps, pt)) / sum(max(ps, pt))` at T=1, no CE term (`--kd 1`), beat
KL / rKL / JSD for their 2-bit QAD. Bounded [0, 1], symmetric, 0 iff identical; tested against an
independent L1 form (`J = 2 L1 / (2 + L1)`), gradients finite and descending.

## 3. Bit-exact fake quant (`tools/pollard_fq.py`)

Parity = `fq(W, imatrix)` vs `dequant(quantize(W, imatrix))` from the runtime libggml, compared as raw float32
bits. Stress run: 10 tensor families (gaussian, t2 heavy tails, lognormal, 70 % sparse, constant, exact ties,
large, 1e-6 tiny, all-zero blocks, per-channel scales) x 3 imatrices (uniform, lognormal, 20 % dead
channels), 512 x 1024 each, two seeds:

| type | encoder ported | result |
|---|---|---|
| q2_K | quantize_row_q2_K_impl (imatrix) | 0 / 122880 blocks differ |
| q3_K | quantize_row_q3_K_impl (imatrix) | 0 / 122880 |
| tq2_0 | quantize_row_tq2_0_ref | 0 / 20480 (+ no-imatrix runs) |
| iq2_xxs | quantize_row_iq2_xxs_impl | 0 / 122880 |
| iq2_xs | quantize_row_iq2_xs_impl | 0 / 122880 |
| iq2_s | quantize_row_iq2_s_impl, with and without imatrix | 0 / 122880 (+ no-imatrix runs) |
| stq1_0 | Pollard's quantize_row_stq1_0_imatrix (patched tree) | 0 / 81920 incl. no-imatrix |

Not ported (raise, never approximate): the imatrix-free `quantize_row_q{2,3}_K_ref` encoders.

**What it took.** The C is float32 and sequential, but its rounding is whatever clang made of it. With the
default `-ffp-contract=on` an `a*b + c` in one expression may be fused, and the vectoriser then decides per
loop: element-wise contractions stay fused (`fmla`), reductions it vectorises become rounded products
added in order. Read off probe builds of ggml-quants.c that reproduce the runtime lib byte for byte:

- all `sumx2 += x*x` over 256: unfused (vectorised);
- `sigma2 + x*x` inside the weight formula: fused;
- `make_qkx3_quants` (inlined into the Q2_K impl, n = 16): fused except the search-loop sums
  `sum_l / sum_l2 / sum_xl`; and `min` moves when a step wins, so the range and offset follow it;
- `make_qx_quants` (inlined twice into Q3_K): first sums loop unfused, search loops fused;
- `make_qp_quants` (called out of line, n = 16 / 32): error and final sums unfused, the itry refinement fused;
- IQ2_XXS's 32-wide `sumqx / sumq2`: unfused; IQ2_XS / IQ2_S's 16-wide ones: fused.

A runtime built without contraction (MSVC default) fuses nothing: `POLLARD_FQ_CONTRACT=0`, then re-run the
gate on that box. A compiler that vectorises differently is caught by the same gate.

**Grids** come from the pinned runtime (`ggml-common.h` dequant grids, `ggml-quants.c` kgrid encoder tables,
cross-checked); the neighbour lists replay `iq2xs_init_impl` (qsort by (d2, index), nwant shells).

**Gate:** `pytest tests/test_fq_parity.py` — live vs runtime libggml (skips without it), STQ1_0 vs a reference
compiled from the patched llama-stq tree (`~/pollard-stq/llama.cpp`, `C:/pollard/llama-stq` or
`POLLARD_STQ_SRC`; the tree's prebuilt libggml predates its own imatrix encoder, so it is compiled from source),
plus `tests/fixtures/fq_golden.json` (176 sha256 of ggml's own output) that CI checks with no ggml present.

**Cost** (CPU, 4M weights): q3_K 0.6 s, q2_K 2 s, tq2_0 / stq1_0 < 0.1 s, iq2_s 10 s, iq2_xxs 21 s,
iq2_xs 45 s (the 292-candidate neighbour search). Peak RSS ~1.1 GB on a 1024 x 4096 matrix (encoder
chunked by 2048 super-blocks). For QAT use `FakeQuantLinear(refresh_every=N)`; the STE gradient reaches the
live weight every step.
