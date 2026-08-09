# E0 — the hardware floor, measured

The two measurements everything downstream stands on. Both reproducible with
the harnesses in `experiments/`.

## Flash read throughput vs block size

Cache-bypassed reads (`F_NOCACHE`), 4 GB file, Apple M4 internal SSD:

| read size | throughput |
|---|---|
| 4 KB | 0.12–0.19 GB/s |
| 64 KB | 0.6–1.3 GB/s |
| 1 MB | 2.1–4.1 GB/s |
| **4 MB** | **2.4–4.9 GB/s** |

**A 20–40× swing on block size alone.** This is the entire economics of
flash-resident weights: stream in large aligned blocks or don't bother.
`pollard-calc`'s default `--flash 3.5` is the mid-band of the 4 MB row —
override it with your own machine's number.

## E1 — dense models have no small active set

`Qwen/Qwen3-0.6B` (SwiGLU), FFN down-projection inputs, 256 tokens of text
(`experiments/e1_measure_sparsity.py`):

```
MEAN active 55.5%   consecutive-token overlap 49.4%   ->   DENSE
```

More than half of FFN neurons are meaningfully active per token, so no
predictor of any kind can find a small hot set — ReLU-era sparsity results do
not transfer to modern activations. This is the measurement that killed the
naive version of this project and redirected it at MoE models, where the
active set is architectural (top-k routing) rather than statistical.
