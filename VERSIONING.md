# Versioning

Pollard follows **semantic versioning**: `MAJOR.MINOR.PATCH`.

- **MAJOR** — breaking changes / big new capability you'd stand behind.
- **MINOR** — new features, backward-compatible.
- **PATCH** — fixes, including the ones you'd rather not talk about.

(a.k.a. the "Pride versioning" meme — Proud.Default.Shame — same thing, more honest.)

Tag every release on GitHub as `vMAJOR.MINOR.PATCH` (e.g. `v1.3.0`), matching `version` in
`pyproject.toml`. Feature work lands on its own branch → PR → merge; cut a tag when a set of changes
is green and ready.

## History
- **1.3.0** — EXL3 lane fixed for low-bit: root-caused the deterministic massive-activation break
  (a single outlier input channel collapsing the trellis global scale) and fixed it with
  `pollard-hf-smooth` (SmoothQuant preconditioning folded into the RMSNorms, exact identity) — measured
  smoothed 4bpw PPL 8.70 ≈ 8bpw 8.28 (was 3090 broken). New tools: `pollard-mx` (MXFP4/FP8 Blackwell
  lane), `pollard-mlx` (Apple lane — was referenced but missing), `pollard-verify` (correctness gate),
  `pollard-doctor` (DrDiag — diagnose/predict/repair any model any lane), plus the **workspace**
  (`pollard-ls` + `pollard_workspace`): all builds auto-organized under `$POLLARD_HOME` (default
  `~/pollard`) with HF-card names + `MANIFEST.json`. Banned `proxy_err` (`legacy/PROXY_ERR_BANNED.md`).
  Packaging: every tool installs (fixed `imatrix_fix_gate` + 11 missing modules), declared
  dependencies + per-lane extras (`[convert]`/`[exl3]`/`[mlx]`/`[mx]`/`[hf]`/`[all]`), removed
  box-specific path locks (generic on any CUDA GPU). 32 CLI tools.
