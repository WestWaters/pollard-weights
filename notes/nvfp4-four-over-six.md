# NVFP4 Four Over Six (`pollard-mx --fp4-scale 4over6`)

**Source.** Four Over Six (Cook et al., arXiv 2512.02010; MIT HAN Lab, github.com/mit-han-lab/fouroversix,
MIT licence). Clean-room from the paper and the reference (`quantize/pytorch/reference.py`: E4M3 max 256
for the 4/6 rules, 448 for static-6; select by MSE / abs-max).

**What Pollard does.** NVFP4 = E2M1 x E4M3 block scale (16 values) x FP32 tensor scale. Plain NVFP4 maps
each block max to 6; E2M1 has no value between 4 and 6. Four Over Six also tries max -> 4 and keeps the better
block. Pollard scores "better" by **imatrix-weighted** squared error (`--imatrix`: llama.cpp imatrix with
llama-family names mapped, or .npz/.safetensors keyed by module path), not plain MSE; without an imatrix it is
the paper's MSE rule.

**Format.** Standard NVFP4 in compressed-tensors' `nvfp4-pack-quantized` layout: `weight_packed` uint8
[out, in/2] (element 2i in the low nibble), `weight_scale` float8_e4m3fn [out, in/16], `weight_global_scale`
float32 (w = fp4 * weight_scale / weight_global_scale). The only change is values: tensor scale computed
against 256 instead of 448 (so a x1.5 "4" block scale still fits E4M3), and each block's scale/codes.
Fused q/k/v and gate/up keep ONE shared global scale (llm-compressor's x 256/448) -- vLLM reads one per fused
layer. Applied as a post-pass over llm-compressor's checkpoint (weights only; activation scales untouched);
`config.json` gets `quantization_config.pollard_fp4_scale` for the card. Default stays `--fp4-scale 6` until
the A/B below.

**Unit-tested (tests/test_mx_fouroversix.py).** E4M3 rounding == torch float8_e4m3fn; E2M1 nearest rounding;
valid layout; per-block never worse than 6 at the same tensor scale; weighted error lower; fused global scale
stays shared through the checkpoint post-pass. Synthetic (outlier channels): ~0.86x MSE, ~0.71-0.77x
imatrix-weighted error vs plain NVFP4, ~42% of blocks take the 4-scale.

## Pending: real A/B on the box (Blackwell 5070 Ti; CPU calibration per the 60/40 rule, vLLM serve when the GPU is free)

```bash
M=Qwen/Qwen3-8B
pollard-calib --out cal.txt --held-out held.txt --model $M
llama-imatrix -m Qwen3-8B-f16.gguf -f cal.txt -o q8b.imatrix -ngl 0 --threads 19 --output-format dat
pollard-probe --model $M --eval held.txt --out sens.json
pollard-mx --model $M --sensitivity sens.json --calib cal.txt --out M-NVFP4-6
pollard-mx --model $M --sensitivity sens.json --calib cal.txt --fp4-scale 4over6 --imatrix q8b.imatrix --out M-NVFP4-4o6
pollard-mx --model $M --sensitivity sens.json --calib cal.txt --fp4-scale 4over6 --out M-NVFP4-4o6-mse
for D in M-NVFP4-6 M-NVFP4-4o6 M-NVFP4-4o6-mse; do
  pollard-verify --lane mx --model $D --source $M --eval held.txt > verify-$D.txt
done
grep -H -i "kld\|top-1" verify-*.txt
# then: vllm serve M-NVFP4-4o6 (loads unchanged?) + pollard-taskeval quick on both
```

Flip the default to `4over6` if KL(4o6, imatrix) < KL(6) and vLLM serves it unchanged.
