# runtime-patches

Local modifications to the llama.cpp trees Pollard builds against, captured as tracked artifacts.

**Why this directory exists.** Spark-X2.5-4B was built with `ik_llama.cpp` — its converter wrote the
`spark2_5` header, its `llama-imatrix` computed the matrix, its `llama-quantize` cut the ladder — which
means the `spark2_5` support in that tree was ours, because ik_llama does not carry it. That work lived
only as uncommitted edits in the working tree. A day later the same tree was edited for the K2 trellis
port, the spark support was overwritten, and a published model lost every runtime that could open it.

A runtime change a published model depends on is an artifact, not a working-tree edit.

## Using it

```bash
pollard-runtime --dirty                                    # runtime work not yet captured
pollard-runtime --scan <tree> --capture <name>             # export it here
pollard-runtime --verify                                   # is every captured patch still applied?
pollard-runtime --scan <tree> --apply <name>               # re-apply after a clone or reset
```

`--verify` exits non-zero when a patch has been lost, so it belongs in front of a build.

## What is here

Each patch has a `.json` manifest recording the tree's remote, the base commit it was captured
against, the files it touches, and any architecture-like strings it adds.

| patch | tree | base | what it is |
|---|---|---|---|
| `ifm-llama-k2-msvc-regex` | MBZUAI-IFM/llama.cpp | `35999d1` 2026-09-01 | K2 pre-tokenizer splitter; MSVC `std::regex` rejects `\p{L}`, which libc++ tolerates. Without it the box loads no K2 GGUF at all. Merged upstream as 69d3a4e (edited there in e78bd94), so the updater now reports it as carried by upstream and skips it. |
| `ik_llama.cpp-msvc-sse2` | ikawrakow/ik_llama.cpp | `5f89bfc81` 2026-10-02 | Prism ternary helpers `ik_hsum_i32_4_sse2`/`ik_decode_trits_sse2` sit under `#if defined(__SSE2__)`, which MSVC never defines, while their callers are under `__SSSE3__`, which `ggml-impl.h` defines for MSVC+AVX. Widens the guard so `ggml-quants.c` compiles on Windows. Carried upstream since f3647513 "Fix MSVC build" (#2595, 2026-10-05), the same guard without our comment, so it is no longer in the ik patch list (`--verify` showed 0/3: our two comment lines were never upstream). |
| `ik_llama.cpp-k2-horizon-arch` | ikawrakow/ik_llama.cpp | `850320b` 2026-08-26 | `k2-horizon` architecture support across converter, arch tables, hparams, tensor loading and vocab — what makes the trellis lane reachable for K2. |
| `llama-stq-stq1_0-quants` | ggml-org/llama.cpp | `1e411d8f5` 2026-08-10 | STQ1_0 quant kernels in `ggml-quants.c`. |
| `llama.cpp-composed-sampler` | ggml-org/llama.cpp | `69f201a20` 2026-10-10 (rebased from `df03399b8`) | `composed` sampler (common/composed-sampler.{h,cpp}): composed decoding on the simplex (arXiv 2609.34992) — KL/JS/entropy/coverage/diversity regularisers, closed forms + mirror ascent — after `temperature` in the chain; `--composed SPEC` on every binary, `"composed"` field on llama-server. Checked against `pollard-composed` to 1e-7. |
| `llama.cpp-qwen3omni-deepstack` | ggml-org/llama.cpp | `8e5a58d` 2026-10-24 (rebased from `1623d8c`) | Qwen3-Omni deepstack: the converter wrote `n_deepstack_layers = 0` and mtmd refused the wider vision projector, so any Omni model with deepstack (the real one) failed to load images. Also the mmproj `thinker.visual.` tensor-name fix. Rebased 2026-10-23, context only: upstream rewrote the line after the removed comment (`n_embd_clip` gained an `n_embd_text` fallback for a gen-only mmproj, Soprano TTS), which stopped the 3-way merge; the added and removed lines are identical to the `1623d8c` patch and to the box's live tree. None of the feature is upstream yet, and `clip_encode` now refuses an output buffer of the wrong size, which the patch already satisfies by encoding audio at its own width before the zero-padding. Tiny random Qwen3-Omni (vision 192 = 64 x 3 with two deepstack layers, audio 64): stock 8e5a58d refuses it (`mismatch between vision and audio mmproj`), patched it encodes an image and a 2 s clip. Upstream PR pending. |
| `llama.cpp-qwen3omni-audio` | ggml-org/llama.cpp | `1623d8c` 2026-10-10 | Qwen3-Omni audio: one STFT frame too many (+13 tokens per clip vs HF) and the last chunk's padding leaking into attention. Token counts now equal HF for 1-9.3 s clips. Upstream PR pending. |
| `llama.cpp-clef-omni` | ggml-org/llama.cpp | `8e5a58d` 2026-10-24 (context-only rebase from `b76b133`: upstream 19ab5a5 added `<algorithm>` to clef.cpp, so the patch applied only with `--3way`; the changed lines are identical and `tests/test_clef_omni_tiny.py` still matches the recorded `/v1/systemone` answers) | `clef-omni` architecture: Cloudflare's decision head on the Qwen3-Omni MoE body (head shared with `clef`), converter, `/v1/systemone`. Applies after the two Omni patches. Parity vs Cloudflare's reference: text 1.6e-5, image 1.4e-4, audio 4.5e-5..9.4e-5. Rebased 2026-10-15 onto master (after `vllm-decision2` and the no-cache KV padding, #30416): same tensors from the converter, and `/v1/systemone` answers equal to the 1623d8c build within 2.6e-9 (CPU) on a tiny random clef-omni. Upstream PR pending. |
| `llama.cpp-quantize-dec-blk` | ggml-org/llama.cpp | `b76b133` 2026-10-15 | `llama-quantize` on an MoE model reads every FFN tensor's layer from its name and knew only `blk.N.`, so clef-omni died on `Failed to determine layer for tensor dec.blk.0.ffn_down.weight` -- no Q8_0 imatrix host, no ftype rung that left a `dec.blk` tensor to the preset. The decision head's `dec.blk.N` is now its own stack (block count from the tensor names). One file (`src/llama-quant.cpp`), applies after `clef-omni`; also applies to the box's newer staged tree. Checked on a tiny clef-omni (b76b133 + the Omni patches, CPU): Q4_K_M / Q2_K / IQ4_XS / IQ2_XXS all build where every one failed before. |

## Architectures only our patches implement (`pollard`)

`pollard-ggufcheck` (and so `pollard-verify`'s runtime gate and the card) reports an architecture as **`pollard`**
when a patch here adds it to `LLM_ARCH_NAMES` in `src/llama-arch.cpp` (read from the patch content; a manifest may
also list it under `adds_archs`), the patch's tree is ggml-org/llama.cpp, and an engine in `engines.json` applies it.
Today that is `clef-omni` (`llama.cpp-clef-omni`). Order: stock > newer > pollard > fork > unknown -- once upstream
merges the architecture it is `newer`/`stock` again, and a port of a third-party fork's architecture (`k2-horizon` on
ik_llama) stays `fork`. The vendored `runtime/llama.cpp` is read from the checkout, never the current directory, and
the architectures our patches add to it are not counted as stock.

A `pollard` rung passes the `runtime: stock` gate only when the release built/verified it with Pollard's runtime
(`runtime_engine: pollard` in results.json), and its card says, plainly, "Needs llama.cpp with Pollard's `<arch>`
patch (runtime-patches/<patch>; upstream PR pending)" with `pollard-runtime --update` and the pinned-base build.
When the upstream PR exists, record it in the manifest as `"upstream": {"pr": "<url>"}` (or `{"merged": "<tag>"}`).

## Which engine runs a model

Every llama.cpp step (rung builds, imatrix, quantize, pollard-decision, pollard-verify) finds its binary through
one resolver, `pollard_calc.find_llama_bin`, steered by the model's architecture (`POLLARD_ARCH`, which the release,
`pollard`, `pollard-decision` and `pollard-verify` set from the GGUF). A manifest's `archs` list names architectures
that need the patch beyond loading at all: `llama.cpp-quantize-dec-blk` lists `clef-omni`. For such an architecture
the engine trees (`engines.json` plus any `runtime/<name>.lock.json` that `pollard-runtime --update` built, e.g. a
staged `llama.cpp.bolt`) are ranked by how many of those patches their source carries (by content, as below), so a
clef-omni ladder quantizes, imatrixes and serves from the tree that has the dec.blk fix even when the live
`runtime/llama.cpp` loads clef-omni without it. Architectures with no `archs` entry resolve exactly as before.

## Verification is by content, not by `git apply`

A patch is confirmed applied by checking its added lines against the tree's own diff. Reverse-applying
the patch is tried first but cannot be trusted alone: these trees are CRLF on the Windows box and the
patches are read on a Mac, and one patch would not reverse-apply even with `--ignore-whitespace` while
every line of it was plainly still in the file.

## Updating a runtime safely

`git pull` in a tree carrying uncommitted support is how `spark2_5` was lost. The order that does not
lose work:

```bash
pollard-runtime --dirty                                    # what is uncaptured right now
pollard-runtime --scan <tree> --capture pre-update         # make it an artifact first
cd <tree> && git pull && cmake --build build -j
pollard-runtime --scan <tree> --apply pre-update           # put it back
pollard-runtime --verify                                   # prove it is back
```

`pollard-fit` prints exactly this sequence when it hits a build too old for a model's architecture. It
used to print a bare `git pull`, which was advice that could destroy a published model's runtime.

**A patched tree is not updated in place.** The trees carrying local support (`ifm-llama`,
`ik_llama.cpp`, `llama-stq`) are pinned to the bases their patches were written against. A current
upstream runtime is a SEPARATE clone, so nothing a pull does can reach the patched trees.

## Daily drift check (CI)

`.github/workflows/runtime-patch-drift.yml` runs every day (and on demand): it shallow-fetches each engine in
`engines.json` from its public remote and runs `pollard-runtime --check-patches all`, which is `--update`'s own
re-apply (same function, same order, every declared patch required) with no build. Each patch is reported plain
(`git apply --check`) and 3-way, with the upstream commit; a patch that needs a rebase fails the job and opens (or
comments on) one issue. Locally: `pollard-runtime --check-patches llama.cpp [--tree <an upstream checkout>]`.

## After a sync that adds a tool

Syncing the repo updates the code but not the commands. An editable install resolves modules straight
into the checkout, so `import pollard_card` is always current, but pip only writes command launchers
when it runs — so a newly added tool exists and its command does not.

```bash
pollard-runtime --install        # does the install run what the repo declares?
```

Non-zero means a declared command has no launcher, and it prints the exact `pip install -e` to fix it.
Run that wherever Pollard is installed. Every tool added in one session was in that state on both
machines, and nothing surfaced it until someone typed a name that did not exist.
