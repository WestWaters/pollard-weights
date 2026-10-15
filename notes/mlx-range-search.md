# MLX lane: imatrix-weighted range search (`pollard-mlx --range search`)

**What.** MLX affine group quant stores one scale and one bias per group of 64 weights; `mx.quantize` takes
them from the group's min/max. `--range search` (now the default) picks them by a per-group search instead:
every (lo, hi) pair on GPQ's shrink grid (0..25% off each end), MLX's own min/max answer as one more
candidate, then two weighted least-squares refits of (scale, bias) at the chosen codes -- the pair with the
least error **weighted by the input channel's activation energy** (an imatrix row) wins. Scales/biases are
rounded to the model dtype before scoring, so the error is the error of what ships.

**Format is unchanged.** Codes are packed exactly as `mx.quantize` packs them (little-endian bitstream in
uint32, every width 2/3/4/5/6/8 -- verified against MLX 0.32.2), scales/biases stay `[..., K/64]` in the
model dtype, `config.json` is mlx-lm's own. `mlx_lm.load` / `mlx_lm.generate` / LM Studio load it unchanged.
Hooked in via the quant_predicate -> `to_quantized` -> `mx.quantize` path, so per-module bits, the vision
tower rules and the MoE pins are untouched (QuantizedLinear's random placeholder quantize is skipped).

**Where the idea came from.** Graft GPQ experiment #1 (2026-10-15, Qwen3-0.6B): the GPQ format itself lost,
but its encoder (imatrix-weighted range search) measured r4 KL 0.147-0.178 vs MLX 4-bit min/max 0.322. Clean-room
here; GPQ's planar format is NOT used.

**Unit-tested (tests/test_mlx_range.py).** Layout identical to MLX's (pack == mx.quantize's for MLX's own
codes; `mx.dequantize` of our encoding == our dequant), per-group error never above min/max, weighted error
< 0.8x `mx.quantize` on synthetic outlier tensors at 3/4 bit (observed ~0.33-0.38x), end-to-end tiny llama
-> pollard-mlx -> `mlx_lm.load` forward.

**Cost.** ~66 candidates per group on the MLX GPU: roughly 0.1 us/weight on an M4 -> ~10 min extra for an 8B.

## Pending: the real KL A/B (Mac, overnight -- NOT run yet)

Same model, same Pollard allocation, same rung, only `--range` differs; KL vs bf16 on a held-out corpus.

```bash
M=Qwen/Qwen3-4B                                  # then Qwen3-8B / the LFM2.5 build if it holds
pollard-calib --out cal.txt --held-out held.txt --model $M            # reasoning share included automatically
# imatrix from the f16 GGUF of the same model (llama-family names map to module paths)
llama-imatrix -m $M-f16.gguf -f cal.txt -o m.imatrix -ngl 0 --output-format dat
pollard-probe --model $M --eval held.txt --out sens.json
for R in minmax search; do
  for B in 3 4; do
    pollard-mlx --model $M --sensitivity sens.json --bits $B --range $R --imatrix m.imatrix --out mlx-$R-$B
    pollard-verify --lane mlx --model mlx-$R-$B --source $M --eval held.txt > verify-$R-$B.txt
  done
done
grep -H -i "kld\|top-1" verify-*.txt
```

Decision: keep `search` as the default if KL(search) < KL(minmax) at both rungs (GPQ predicts ~2x at 4-bit
with an imatrix). Also record `--range search` WITHOUT `--imatrix` (weight-error only) -- the release lane
passes no imatrix to pollard-mlx yet, so that is what `pollard release --lanes mlx` currently ships.
