# Errata

Corrections and retractions, kept in the open — this is the culture the repo asks for.

- **2026-08-17 — robustness fixes from field testing on a DGX Spark (GB10).**
  Real runs on DeepSeek-V4 (284B MoE, 5 shards) and Qwen3-30B surfaced three bugs,
  now fixed: (1) `pollard-calc` read only the **first shard** of a multi-shard model
  → reported 0.00 bpw / absurd tok/s; it now sums params and bytes across every
  shard. (2) Aggressive IQ2 builds could **crash partway** when the base preset hit
  a tensor the imatrix doesn't cover (e.g. DeepSeek's `output_hc_fn`,
  `indexer_compressor`); pollard-fit now auto-pins those to `q6_K`. (3) With no
  calibration signal, a build could come out **larger and slower than the source**
  silently; pollard-fit now warns loudly (no signal = no benefit) and refuses to
  build larger than an already-quantized source. (4) `pollard-run --vram auto`
  read **free** VRAM, which is ~0 when a model is already resident → a useless 0
  budget; it now plans against **total** VRAM when the GPU is occupied (placement
  runs once the resident model is unloaded anyway). (5) `pollard-sensitivity` (the
  measured signal itself) was hardened the same way: a **failed probe build now
  PROTECTS that group (max sensitivity), never records it as 0** — a crash used to
  silently read as "least important, crush hardest" — and its uniform IQ2 noise
  builds pin uncoverable tensors so the aggressive end of the curve actually gets
  measured on exotic models. Thanks to the tester who ran it on real hardware and
  sent the logs.
- **2026-08-08 — K3 expert dimensions were 2× too large.** The formula ignored
  `routed_expert_hidden_size` (K3 runs experts in a half-width latent space:
  3584 vs hidden 7168), doubling total params (5.48T → correct **2.75T**),
  active bytes, and the residency tier (3.7TB → correct **~1.9TB**). Found by
  community review within a day of launch. The original README also overstated the
  demo comparison as "validated"; it is a worked example with stated assumptions,
  and is now labeled as one.
