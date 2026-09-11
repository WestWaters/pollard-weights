# Hy4-preview (Tencent, 770B/49B MoE, `hy_v4`) — first per-layer capture data for the Pollard method

Measured 2026-09-10 on two DGX Spark (GB10) nodes by bot-lab-21 (**re-capture**: the 09-08/09 numbers were taken through a
transformers RoPE defect, see "Correction" below). Raw per-layer JSON in `experiments/data/hy4_preview/` (234 files: `smooth_L##.json`,
`route_L##.json`, `extra_L##.json`, `ppl_bf16_calib.json`), merged from a row-split run (rows 0-191 on one node, 192-383 on the other;
`tools`-style merge = token-weighted). Calibration set: 384 × 2048 tokens, Hy4 tokenizer, same
sources/quotas/held-out exclusion as our GLM-5.3 set. Source checkpoint: `tencent/Hy4-preview` bf16 (1.56 TB), byte-verified.

## What is in the data

| file | per layer | fields |
|---|---|---|
| `smooth_L##.json` | 78 | `attn_in_absmax`, `mlp_in_absmax` (per-channel absmax at the two norm seams — the SmoothQuant fold inputs), `tokens` |
| `route_L##.json` | 77 (L0 is dense) | `count[256]`, `wmass[256]` per expert (routing concentration) |
| `extra_L##.json` | 78 | `q_a_norm_absmax`, `kv_a_norm_absmax` (MLA latent seams), hyper-connection gates `hc_{attn,ffn}_{pre,post}_mean` (4 streams), `attn_out_gate_mean` + `attn_out_gate_frac_lt_0.1`, `sinks` (min/mean/max), `stream_rms_after[4]`, per-tensor `weights` stats (incl. per-expert rms spread), `bytes_by_category`, `indexer` mode (`full`/`shared`), `seconds` |
| `ppl_bf16_calib.json` | 1 | bf16 reference NLL/ppl on the 384 calibration rows, per-row NLL (token-weighted merge of the two nodes) |

786,432 calibration tokens per layer (384 rows × 2048).

## Headline numbers

- **bf16 reference on the calibration rows:** mean NLL **1.855** (ppl 6.39) over 786,048 tokens (per node: 1.850 / 1.860). This is the
  anchor any Hy4 quant should be gated against on the same rows.
- **Norm-seam absmax (SmoothQuant inputs), per-layer max across 78 layers:** attention input 0.78 / 1.33 / 4.94 (min / median / max),
  MLP input 0.49 / 1.00 / 6.59. Layers with attention-input absmax above 2: 6-8, 11, 13-14, 25, 27-29, 67, 71, 75-77. Compared with
  GLM-5.3 (see `glm-5.3-744b-routing-and-smoothing-data.md`) Hy4's seams are flat: nothing reaches GLM-5.3's double-digit early-layer
  outliers, so the smoothing fold has little to gain here.
- **Routing (256 experts, 786K tokens):** the top-32 experts carry 27 % / 38 % / 49 % (min / median / max) of routed weight mass per
  layer; **no layer exceeds 50 %**. Top-64: 44 / 55 / 65 %; top-128: 70 / 79 / 84 %. Reaching 90 % of mass takes 155-196 experts
  (median 172 of 256). Concentration is nearly flat along the stack (band medians L1-19 0.34, L20-39 0.39, L40-59 0.39, L60-77 0.36);
  the most concentrated layers are 33, 51, 25, 52, 42 (46-49 %). Hy4 routes far more evenly than GLM-5.3, so expert-pruning (REAP-style)
  has less headroom here and the allocator should not expect a small hot set. Decode-vs-prefill split was not captured (single pass).
- **Hyper-connections (`hc_mult` = 4 residual streams):** at L1 the post-gates are essentially off on streams 1-3 (0.00 / 0.00 / 0.06)
  and open on stream 4 (0.26); by L77 all four are open (1.5 / 1.6 / 1.2 / 2.0). `stream_rms_after` grows from 0.025 / 0.023 / 0.026 /
  0.106 at L1 to 24.7 / 19.4 / 30.3 / 27.7 at L77 — the four streams end up within a factor of 1.6 of each other (the broken capture
  showed a 4x spread). Per-stream quantization or KV-scale decisions should still be per stream, but the late stack is balanced.
- **Attention output gate:** mean 0.10 / 0.24 / 0.39 per layer; the fraction of gate values below 0.1 ranges 0.2 % to 64 % (median 19 %)
  — a large share of heads are effectively off in many layers, a prune/skip signal rather than a quantization one.
- **Sinks:** learnable sink params span −4.96 … +1.01 at L1 (−6.55 … +3.18 across all layers; weights, unaffected by the fix).
- **Bytes per MoE layer (bf16):** experts 19.3 GB, attention 0.55 GB, shared experts 75 MB, router 3 MB, hc 1.6 MB, norms 24 KB —
  experts are >97 % of the bytes, so the allocator's expert-vs-attention split matters even more than on GLM-5.3.
- **Indexer:** pattern F F S S S F S S S … F (full at L0, L1, then every 4th layer; 77 MoE layers), recorded per layer.

## Correction (2026-09-10) — what changed versus the 09-08/09 capture and why

The first capture (PR #57) ran through transformers main's `hy_v4` port, which applies rotate-half (NeoX) RoPE while the checkpoint
stores `q_pe`/`k_pe` in interleaved (Megatron/PTM) layout (vLLM's `hy_v4` builds its rotary with `is_neox_style=False` for both
attention and the DSA indexer, and says so in a comment). The conversion mapping has only key renames, no RoPE permutation, so
relative positions were encoded wrong in every layer. Symptoms that caught it: bf16 reference NLL 5.02 (ppl 151) with NLL *worsening*
along the sequence and 19 % top-1; a 2-layer kit-vs-`from_pretrained` equivalence test matched to 0.002 nats, so the kit was faithful to
the port and the defect was upstream. With interleaved RoPE (numerically identical to vLLM's non-NeoX rotary, 4e-7) the same rows give
NLL 1.94 on a 4-row probe and 1.855 on all 384 rows. (Upstream report to transformers pending.)

| quantity | 09-08/09 (broken RoPE) | 09-10 (fixed) | verdict |
|---|---|---|---|
| bf16 reference NLL / ppl | 5.02 / 151 | **1.855 / 6.39** | was meaningless |
| top-32 routed mass, min / med / max | 32 / 62 / 81 % | 27 / 38 / 49 % | **concentration was an artifact** (per-layer routing-mass correlation old-vs-new: median 0.56, min 0.16) |
| layers with top-32 > 50 % | 72 of 77 | 0 | " |
| attention-input absmax, min / med / max | 0.78 / 1.29 / 5.28 | 0.78 / 1.33 / 4.94 | robust (per-layer ratio new/old 0.84-1.19, median 1.00) |
| MLP-input absmax | 0.48 / 1.02 / 6.78 | 0.49 / 1.00 / 6.59 | robust |
| `stream_rms_after` L77 | 17.7 / 11.6 / 35.3 / 9.2 | 24.7 / 19.4 / 30.3 / 27.7 | late-stack stream balance was wrong |
| attention out-gate mean, min / med / max | 0.10 / 0.22 / 0.49 | 0.10 / 0.24 / 0.39 | mostly robust |
| sinks, weight stats, bytes, indexer pattern | — | — | unchanged (do not depend on activations) |

Takeaway for the method: the norm-seam statistics are insensitive to the positional defect, the routing statistics are not. A
reference-NLL step on the capture rows is cheap and is exactly what exposed this; keep it in the protocol.

## Not included

- Decode-vs-prefill routing split (single-pass capture) and any held-out perplexity (the reference is on the calibration rows).

## Reproduction notes (kit gotchas that cost us a day)

- **RoPE pairing:** transformers main's `hy_v4` applies rotate-half RoPE to a checkpoint whose `q_pe`/`k_pe` are interleaved; until that
  is fixed upstream, monkeypatch `apply_rotary_pos_emb` to interleaved pairing (pairs `(x[2i], x[2i+1])`, cos/sin repeated per pair) —
  see "Correction" above. Check with the reference NLL: ~1.9 is right, ~5 means the positions are scrambled.
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
