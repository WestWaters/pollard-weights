# Serving an EXL3 cook that quantizes attention — three artifact/runtime gaps, measured on GLM-5.3 744B

Companion to `glm-5.3-cluster-verification.md`. The EXL3 3.2 bpw GLM-5.3 body from the band-parallel run (`experiments/exl3_band.py`)
went to a TP4 serving gate on vLLM 0.28 with the `cuda-exl3` plugin (its sparse-MLA attention backend is the only one that runs a
DSA model on GB10). It loaded, every counter was healthy, throughput was fine — and the output was gibberish. Four boots and one
afternoon later, three distinct gaps. None is a quantization-quality problem; all three bite anyone who follows the Pollard EXL3
lane *past* what the public GLM-5.3 EXL3 builds do (they keep attention bf16 and quantize only the routed experts).

## 1. exllamav3 ≥ 1.4 pads every linear's `out_features` to a multiple of 128 — decode the stored width, then trim

exllamav3 1.4.x (`modules/linear.py`, `pad_to = 128`) rounds `out_features` up to a 128-multiple before quantizing, and the EXL3
forward applies a 128-block Hadamard on the **output** side before the per-channel `svh` scale. GLM-5.3's `kv_a_proj_with_mqa`
is 512 (latent) + 64 (RoPE key) = 576 wide → stored as 640 columns (trellis `(384, 40, ·)`, `svh` 640). Every other attention
linear happens to be a 128-multiple; the routed experts (2048) are too, which is why expert-only builds never see this.

vLLM fuses `q_a_proj ‖ kv_a_proj_with_mqa` into one 2624-wide layer. `cuda-exl3` 1.0.0 sized the fused trellis at that declared
width and vLLM's merged-column loader narrowed the 40-tile checkpoint shard to 36 tiles — so the last Hadamard block (columns
512–639 of the shard) was decoded from half its columns. Measured on the real layer-5 tensors with the plugin's own kernel, random
inputs, full-width decode-then-trim vs the truncated decode:

| columns of `kv_a_proj_with_mqa` | relative error, truncated vs full |
|---|---|
| 0–511 (KV latent) | 1.3 × 10⁻⁶ |
| 512–575 (RoPE key) | **1.12** |

i.e. the RoPE key of every one of the 78 layers was noise. Symptoms at the API: coherent-looking token rate, no errors, live
perplexity on our held-out texts 3.7 × 10⁶ (the int4 body: 4.8), MTP draft acceptance 1.02 tokens per step (the draft was fine;
nothing can predict noise).

**Fix (runtime side, what we shipped in our `cuda-exl3` build, 1.0.1):** when a checkpoint shard is wider than the shard the model
declares, and the excess is `< 128` with the stored width a 128-multiple, allocate the fused trellis and `svh` at the *stored*
widths, copy each shard whole at its own offset (ignoring the offsets vLLM computed from the declared widths), run the kernel at
the padded width, and slice the output back to the declared widths (a free `narrow` when only the last shard is padded). Verified
on the real fused layer: RoPE-key columns now match the reference decode exactly. The general rule for any EXL3 runtime: a trellis
cannot be cut inside a 128-column block — never narrow it to a non-multiple of 128; decode and trim instead.

**Fix (artifact side, if the runtime cannot be changed):** keep every linear whose width is not a 128-multiple in bf16 — for
GLM-5.3 that is `kv_a_proj_with_mqa` (7 MB per layer). It has to be the whole fused group in bf16 though (vLLM cannot mix
methods inside one fused layer), so `q_a_proj` goes with it: ≈ 2.5 GB per node at TP4. Cheaper than 78 layers of noise.

Where it belongs in Pollard: `pollard-exl3` (and the export template for exllamav3 targets) could print a **serving-compatibility
line** listing every quantized linear whose `out_features % 128 != 0`, with the sentence above. It is a static config check.

## 2. vLLM's MTP drafter builds `eh_proj` as a plain linear — leave it bf16

vLLM's `DeepSeekMTP` (used for GLM-5.3's MTP layer) constructs `model.layers.<N>.eh_proj` as an unquantized `nn.Linear`. exllamav3
quantizes it with the side model (`mtp_bits`) → the loader dies with `KeyError: 'model.layers.78.eh_proj.mul1'`. Fix in the
artifact: drop the EXL3 tensors of `eh_proj` (trellis/suh/svh/mul1), add the source bf16 `eh_proj.weight` (6144 × 12288) in a
small extra shard, rewrite the index. `tools/exl3_fix_mtp_ehproj.py` does this.

## 3. Do not trust `model.safetensors.index.json` for the MTP side model — walk the shard headers

exllamav3's index pointed `eh_proj` at shard 45; the tensors were physically in shard 44. Three small norm tensors of the MTP layer
(`enorm`, `hnorm`, `shared_head.norm`) are written twice across shard boundaries (byte-identical, harmless). vLLM's loader walks
the *files*, so an index-driven fixer rewrote the wrong shard and did nothing — two failed boots before we noticed. Any post-cook
surgery on an exllamav3 artifact must resolve tensors by reading safetensors headers, not the index (the tool above does).

## Also measured on this line

- The `cuda-exl3` sparse-MLA decode kernel reads the KV cache directly and accepts **bf16 / fp8-e4m3 only**; `nvfp4` KV is
  rejected at backend selection. For fit math on this runtime use the fp8 line (GLM-5.3: ≈ 57 KB/token incl. the DSA indexer
  cache). At TP4 on 4 × 128 GB unified nodes, fp8 KV + the 8-bit in-checkpoint MTP draft: 19.8 GiB KV per node = 369K tokens.
- Throughput numbers taken on a body that fails a plain correctness read are worthless — read the words. Our gate now keeps the
  full text of every correctness probe, and treats live perplexity > 20 as "do not run the long battery".
