# BF16 and merged MoE probe checks on GB10

The imatrix probe needs decoded weights, not the bytes used to store BF16.
Merged expert tensors also need separate importance values for each expert.
The tests in `tests/test_probe_gguf_weights.py` cover both with real GGUF files
and costs calculated directly from small matrices.

```sh
python -m pytest -q tests/test_probe_gguf_weights.py
```

All 22 cases passed on macOS ARM64 and in a Linux ARM64 environment using
Torch 2.13.0+cu130 and the pinned quantizer's GGUF reader. GPU access is not
needed for these tests.

## Original-model check

Source: [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B), revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` (Apache-2.0). The original BF16
safetensors passed content verification against the pinned Hub file hashes.

Converter and native runtime:
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), commit
`5f89bfc81268b4d56d2af63ccbed59de17c64c09`. Conversion used `--outtype bf16`.
The resulting GGUF was 61,095,802,816 bytes, SHA256
`1b2114d05447fa0c99ac9a4cce02c6e88b38466746d7b8356b746f0e2729baa9`.

The probe's decoder reconstructed 26 sampled matrices exactly against the
original safetensors: all four attention projections in layers 0 and 47, plus
gate/up/down projections for experts 0, 17 and 127 in those layers.
That is 66,060,288 values. This checks those matrices, not every model weight.

A separate prediction check on one GB10 compared the original HF BF16 model
with the native BF16 GGUF. HF used Transformers 5.17.0, eager attention and
Torch 2.13.0+cu130. Native used FlashAttention and F16 KV. The same token IDs
were checked before comparing predictions: two 256-token contexts per domain,
scoring 254 next-token predictions per domain after excluding the first half
of each context and its final token.
The exact input rows, public dataset revisions, formatting and file hashes
are listed in [imatrix-probe-corpus.json](imatrix-probe-corpus.json). No raw
dataset text or model outputs are included in that manifest.

| Domain | Top-1 agreement | Approximate HF-to-GGUF KL, nats | Signed mean NLL difference, nats |
| --- | ---: | ---: | ---: |
| Code | 98.43% | 0.004875 | -0.004136 |
| Reasoning | 98.82% | 0.004340 | 0.019706 |
| Prose | 96.85% | 0.003958 | -0.010874 |

The acceptance limits were fixed before the comparison: at least 90% top-1
agreement, approximate KL at most 0.03 nats and absolute signed mean NLL
difference at most 0.1 nats, separately for each domain. All passed. Native
saved probabilities use 16-bit storage and a 24-logit floor; they were
renormalized for approximate KL. No scored target probability was clipped.
This is a small reference check, not bit-exact inference equivalence or a
held-out model-quality benchmark.

## Router accounting and incomplete calibration

The native collector produced entries for `ffn_gate_inp.weight` in all 48
layers. These are MoE routers, not expert gate projections. The native
quantizer keeps them at their source type unless its separate router-type
option is used. The probe records them under `excluded_tensors` and does not
score them as expert FFNs. If a recipe quantizes routers separately, it needs
a separate measurement. Unknown tensor families still stop the probe.

Importance collection used allocation-disjoint public calibration text. The
first 64 x 512-token run skipped eight entries and filled in missing expert
statistics in other entries. Processing the whole file (129 complete contexts)
removed the skipped entries, but still produced partial-expert warnings.
Expanding to 923 unique samples (123 code, 400 reasoning, 400 prose) processed
299 complete contexts, or 153,088 tokens. All 337 collected entries had valid
dimensions and finite nonnegative values, but 28 up/down expert tensors still
had partial-data warnings, down from 36 in the 129-context run. The expanded
run took 584 seconds. Valid stored values do not prove observed coverage:
the native collector substituted statistics for unseen experts.

The exact expanded input is reconstructible from
[imatrix-calibration-corpus.json](imatrix-calibration-corpus.json). It uses
public training data and MBPP validation examples, excluding our frozen
allocation-selection and held-out samples. Normalized whole-sample overlap
and MBPP task-ID overlap were zero; near-duplicates and pretraining overlap
were not assessed. Held-out evaluation was not run or used to tune this corpus.

None of these matrices establishes complete observed expert coverage. Missing gate
entries from a fused gate/up operation need explicit accounting too.

Inspect collector warnings, not just the presence of a matrix file. This
pinned collector can substitute values of one for a few unseen experts. Its
periodic-save path also modifies those accumulated statistics. These runs
used `--output-frequency 1000000 --save-frequency 0` to defer saving until
collection finished. The expanded matrix is diagnostic evidence, not a
completed allocation-versus-uniform comparison. Its SHA256 is
`01f0526299538bd11992576f67b6023fd7f8a653982f643b2b5111ef26cc976e`.

The score remains a Hessian-weighted RTN proxy. Expert-specific activation
averages in the saved matrix do not themselves record expert-selection
frequency. These checks do not demonstrate measured quantizer KL, a quality
gain over uniform quantization, throughput improvement or multi-node scaling.

## Native build

The unmodified source was built with CMake 3.31.6, GCC 13.3.0 and CUDA 13.0.88,
targeting `121-real`. GCC's native ARM detection omitted supported DOTPROD and
FP16 features on this machine. The successful build used `GGML_NATIVE=OFF`
and `GGML_ARCH_FLAGS=-march=armv8.2-a+dotprod+fp16`. All five native tools
passed driver-enabled library and help checks. This validates this build,
not the quantizer's generic ARM fallback.

Raw corpus text, predictions and machine-specific logs are not included here.
Prepared with AI assistance and tested locally.
