# Versioning

Pollard follows **semantic versioning**: `MAJOR.MINOR.PATCH`.

- **MAJOR** — breaking changes / big new capability you'd stand behind.
- **MINOR** — new features, backward-compatible.
- **PATCH** — fixes, including the ones you'd rather not talk about.

(a.k.a. the "Pride versioning" meme — Proud.Default.Shame — same thing, more honest.)

Tag every release on GitHub as `vMAJOR.MINOR.PATCH` (e.g. `v1.3.0`), matching `version` in
`pyproject.toml`. Feature work lands on its own branch → PR → merge; cut a tag when a set of changes
is green and ready.

**This file records what shipped, not why or how.** Root causes, measurements and design belong in
`notes/`; a history entry says what changed and links to the note.

## Unreleased (on `main`, not yet tagged)

Tools added since `v1.3.0` — **42 CLI tools** now:

- `pollard-archfp` — fingerprint an architecture's layout and name the known family it is a twin of
  (dense / MoE / MLA / DSA / hybrid linear-attention), from a GGUF or HF weights.
- `pollard-recard` — bring already-published repos onto the master card template without losing the
  numbers or analysis their current cards carry.
- `pollard-errtype` — classify why a tensor will quantize badly (outlier scale spread vs concentration)
  before building any candidate.
- `pollard-errsrc` — attribute measured KL cost to a tensor *and* the trigger behind it, so budget goes
  where it buys something.
- `pollard-ggufcheck` — report which runtime a GGUF needs from its tensor types rather than its name,
  locally or against a published repo; exits non-zero so it works as a pre-publish gate.
- `pollard-card` / `pollard-onboard` / `pollard-envmatch` / `pollard-serve-eval` — card generation,
  custom-arch onboarding, environment matching, served-side evaluation.

Also on `main`: `pollard-fit --tier` (pin a whole component class and spend the remaining budget),
`pollard-calc --gpu auto` (read the installed card instead of a name table), and the one master card
template every repo is generated from — which now reads each rung's license and runtime need from the
base model's card and the built files, instead of defaulting either.

## History

- **1.3.0** — EXL3 lane fixed for low-bit (root cause and measurements:
  [`notes/exl3-massive-activation-fix.md`](notes/exl3-massive-activation-fix.md)). New tools:
  `pollard-mx` (MXFP4/FP8 Blackwell lane), `pollard-mlx` (Apple lane), `pollard-verify` (correctness
  gate), `pollard-doctor` (DrDiag), plus the **workspace** (`pollard-ls` + `pollard_workspace`): builds
  auto-organized under `$POLLARD_HOME` with HF-card names + `MANIFEST.json`. Banned `proxy_err`
  ([`legacy/PROXY_ERR_BANNED.md`](legacy/PROXY_ERR_BANNED.md)). Packaging: every tool installs,
  declared dependencies + per-lane extras (`[convert]`/`[exl3]`/`[mlx]`/`[mx]`/`[hf]`/`[all]`), no
  box-specific path locks.
