# Route B: EXL3 routed experts for a model no framework can load (DeepSeek-V4.1-Flash), reference-forward capture → per-expert `quantize_exl3` → hybrid vLLM checkpoint

Experimental, contributed as-is (Apache-2.0, AI assistance used; every number in the companion note was measured on our hardware). Paths and hosts are placeholders (`/data/...`, `STORE_HOST`); the shell orchestration for our 10-node cluster (claims scheduler, hub servers, in-place distribution) is deliberately omitted — it is site-specific.

Pipeline (each step is one script; all read/write safetensors + JSON, no framework model class needed):

| step | script | what it does |
|---|---|---|
| 0 | `dsv41_dequant.py` | exact upscale of the shipped checkpoint to bf16 (fp8·2^(ue8m0−127) 32×32 blocks; MXFP4 e2m1 LUT × per-32 ue8m0 scale), resumable, round-trip `--check`; Engram shards copied verbatim |
| 1 | `dsv41_build_calib.py` | in-domain calibration rows rendered through DeepSeek's DSML encoder (no Jinja template exists), held-out sets excluded by content hash |
| 2 | `dsv41_capture.py` | DeepSeek's reference `model.py`/`engram.py` as the calibration forward, layer by layer with shard staging + on-the-fly dequant and pure-torch kernel shims; emits per-layer activation statistics, the FFN input + routing dump per layer (`--hessian-expert-mode dump`), reference NLL; Engram rows gathered by hash id, never the whole table; `--engram-fq mxfp4|nvfp4` (fake-quant probe) and `--exl3-recipe` (pre-splice NLL gate: every routed expert replaced by its EXL3 reconstruction) |
| 2b | `dsv41_engram_extract.py` | extracts exactly the Engram rows the capture needs from the tables on the store host (sequential scan) |
| 3 | `dsv41_exl3_probe.py` | Gate-0: EXL3 error on the FP4-grid weights vs a Gaussian control |
| 4 | `dsv41_exl3_experts.py` | the quantizer: loads a layer's dump shards, per-expert Hessians (H_gu shared by gate/up, H_down from silu(xW1)·xW3), exllamav3 `quantize_exl3`/`_batch` with its own out-scale/seed rules, `--bits`, `--recipe` (per layer gu/down K), `--mats w2|w1,w3|all`, `--w2-weighting none|route`, `--verify-rows`; parts per 48 experts, resumable; JSON ledger (proxy_err, bpw, tokens, rfn_out) |
| 5 | `dsv41_ab_compare.py` | fair A/B of two passes on the same rows with both output-error metrics (the quantizer's own rfn_out is not comparable across weightings) |
| 6 | `dsv41_exl3_alloc.py` | budgeted allocator: cost(L, m, K) = gain_L² · Σ_e tokens_e · proxy_err_e with gain_L = residual write gain from the capture stats; greedy K=3→4 to a bpw target; missing K=4 ledgers estimated from the measured K4/K3 ratio and emitted as cook lists |
| 7 | `dsv41_exl3_mix.py` | per-layer assembly of the recipe (gate/up from one pass, down from another), raw byte copies |
| 8 | `dsv41_exl3_splice.py` | pure-Python splice into the vLLM checkpoint: expert tensors rewritten per body shard, everything else hardlinked; index + `quantization_config` (per-module `tensor_storage`) for the cuda-exl3 plugin; `--verify-only`, `--selftest`, `--allow-partial` |
| 9 | `dsv41_engram_hotset.py` | Engram row-frequency ledger over a corpus (reference `NgramHashState`, no forward): top-N row ids per layer + in-sample/held-out coverage |

Order of operations we used: 0 → 1 → 2 (10 nodes, one row range each) → 3 → 4 (K=3 all layers; then `--mats w2 --bits 4` and `--mats w1,w3 --bits 4` on the layers 6 picked) → 6 (final, measured) → 7 → 8 → serve. Timings and results: see the companion note.
