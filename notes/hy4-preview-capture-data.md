# Hy4-preview (Tencent, 770B/49B MoE, `hy_v4`) — first per-layer capture data for the Pollard method

Measured 2026-09-08/09 on two DGX Spark (GB10) nodes by bot-lab-21. Raw per-layer JSON in `experiments/data/hy4_preview/`
(233 files: `smooth_L##.json`, `route_L##.json`, `extra_L##.json`), merged from a row-split run (rows 0-191 on one node,
192-383 on the other; `tools`-style merge = token-weighted). Calibration set: 384 × 2048 tokens, Hy4 tokenizer, same
sources/quotas/held-out exclusion as our GLM-5.3 set. Source checkpoint: `tencent/Hy4-preview` bf16 (1.56 TB), byte-verified.

## What is in the data

| file | per layer | fields |
|---|---|---|
| `smooth_L##.json` | 78 | `attn_in_absmax`, `mlp_in_absmax` (per-channel absmax at the two norm seams — the SmoothQuant fold inputs), `tokens` |
| `route_L##.json` | 77 (L0 is dense) | `count[256]`, `wmass[256]` per expert (routing concentration) |
| `extra_L##.json` | 78 | `q_a_norm_absmax`, `kv_a_norm_absmax` (MLA latent seams), hyper-connection gates `hc_{attn,ffn}_{pre,post}_mean` (4 streams), `attn_out_gate_mean` + `attn_out_gate_frac_lt_0.1`, `sinks` (min/mean/max), `stream_rms_after[4]`, per-tensor `weights` stats (incl. per-expert rms spread), `bytes_by_category`, `indexer` mode (`full`/`shared`), `seconds` |

786,432 calibration tokens per layer (384 rows × 2048).

## Headline numbers

- **Norm-seam absmax (SmoothQuant inputs), per-layer max across 78 layers:** attention input 0.78 / 1.29 / 5.28
  (min / median / max), MLP input 0.48 / 1.02 / 6.78. Compared with GLM-5.3 (see `glm-5.3-744b-routing-and-smoothing-data.md`)
  Hy4's seams are flatter: no layer reaches the double-digit outliers GLM-5.3 shows in its early dense layers. The smoothing case
  is weaker here; the allocator should not expect a large gain from folding.
- **Routing concentration (256 experts):** the top-32 experts carry 32 % / 62 % / 81 % (min / median / max) of routed weight mass
  per layer; layers 4–24 are all above 50 %. Concentration is strongest in the lower-middle stack, not at the top — the mirror
  image of GLM-5.3's profile. Decode-vs-prefill split was not captured (single-pass calibration).
- **Hyper-connections (`hc_mult` = 4 residual streams):** post-gates concentrate on stream 4; `stream_rms_after` grows from
  ~0.025 (streams 1–3) / 0.10 (stream 4) at L1 to 17.7 / 11.6 / 35.3 / 9.2 at L77 — stream 3 dominates the late residual.
  Any per-stream quantization or KV-scale decision has to be per stream.
- **Attention output gate:** mean 0.10 / 0.22 / 0.49 per layer; the fraction of gate values below 0.1 ranges 0.1 % to 66 %
  (median 20 %) — a large share of heads are effectively off in many layers, which is a prune/skip signal rather than a
  quantization one.
- **Sinks:** learnable sink params span −4.96 … +1.01 (L1).
- **Bytes per MoE layer (bf16):** experts 19.3 GB, attention 0.55 GB, shared experts 75 MB, router 3 MB, hc 1.6 MB, norms 24 KB —
  experts are >97 % of the bytes, so the allocator's expert-vs-attention split matters even more than on GLM-5.3.
- **Indexer:** 77 MoE layers alternate `full` and `shared` indexer modes (recorded per layer).

## Not included yet

- `ppl_bf16_calib.json` (bf16 reference NLL on the calibration rows): the capture's final head step crashed on a checkpoint-vs-
  transformers naming mismatch (`model.hc_head.hc_head_{fn,base,scale}` vs `HYV4HyperHead.hc_{fn,base,scale}`); fixed in the kit,
  re-run pending (needs one GB10 for ~4 h with the layer-77 stream state saved).

## Reproduction notes (kit gotchas that cost us a day)

- transformers main (5.17.0.dev0, `hy_v4` added 2026-09-07) is required; `pip --target` it next to the runtime image and set
  `PYTHONPATH`. Add a `KEY_RENAMES` rule for the head: `(r"\.hc_head\.hc_head_", ".hc_head.hc_")`.
- Build layers on the `meta` device with **bf16 as the default dtype** — fp32 default makes `masked_fill` overflow in the
  sparse-attention mask path.
- `create_causal_mask(inputs_embeds=…, allow_is_causal_skip=False)`; eager attention supports the sinks + top-k mask.
- Whole-tensor `.float()` on an expert stack is 26 GB — compute weight stats chunked over experts.
- Run the container with `--user $(id -u)`; root-owned outputs bit us.
- Stream shards over HTTP with a hard stage budget (`--stage-budget-gb`); two Hy4 shards are ~24 GB, so a node with <40 GB free
  cannot host the run — check `df` first (we lost two attempts to ENOSPC).
- Save the post-last-layer stream state (`--state-out`) so a head-step failure is resumable instead of a 4-hour re-run.
- Load-bound: ~190 s per MoE layer per node (fetch ~60 s + compute ~130 s), so ~4 h for 78 layers on one GB10.
