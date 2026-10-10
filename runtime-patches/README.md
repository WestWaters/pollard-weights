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
| `ik_llama.cpp-msvc-sse2` | ikawrakow/ik_llama.cpp | `5f89bfc81` 2026-10-02 | Prism ternary helpers `ik_hsum_i32_4_sse2`/`ik_decode_trits_sse2` sit under `#if defined(__SSE2__)`, which MSVC never defines, while their callers are under `__SSSE3__`, which `ggml-impl.h` defines for MSVC+AVX. Widens the guard so `ggml-quants.c` compiles on Windows. |
| `ik_llama.cpp-k2-horizon-arch` | ikawrakow/ik_llama.cpp | `850320b` 2026-08-26 | `k2-horizon` architecture support across converter, arch tables, hparams, tensor loading and vocab — what makes the trellis lane reachable for K2. |
| `llama-stq-stq1_0-quants` | ggml-org/llama.cpp | `1e411d8f5` 2026-08-10 | STQ1_0 quant kernels in `ggml-quants.c`. |
| `llama.cpp-composed-sampler` | ggml-org/llama.cpp | `69f201a20` 2026-10-10 (rebased from `df03399b8`) | `composed` sampler (common/composed-sampler.{h,cpp}): composed decoding on the simplex (arXiv 2609.34992) — KL/JS/entropy/coverage/diversity regularisers, closed forms + mirror ascent — after `temperature` in the chain; `--composed SPEC` on every binary, `"composed"` field on llama-server. Checked against `pollard-composed` to 1e-7. |
| `llama.cpp-qwen3omni-deepstack` | ggml-org/llama.cpp | `1623d8c` 2026-10-10 | Qwen3-Omni deepstack: the converter wrote `n_deepstack_layers = 0` and mtmd refused the wider vision projector, so any Omni model with deepstack (the real one) failed to load images. Also the mmproj `thinker.visual.` tensor-name fix. Upstream PR pending. |
| `llama.cpp-qwen3omni-audio` | ggml-org/llama.cpp | `1623d8c` 2026-10-10 | Qwen3-Omni audio: one STFT frame too many (+13 tokens per clip vs HF) and the last chunk's padding leaking into attention. Token counts now equal HF for 1-9.3 s clips. Upstream PR pending. |
| `llama.cpp-clef-omni` | ggml-org/llama.cpp | `1623d8c` 2026-10-10 | `clef-omni` architecture: Cloudflare's decision head on the Qwen3-Omni MoE body (head shared with `clef`), converter, `/v1/systemone`. Applies after the two Omni patches. Parity vs Cloudflare's reference: text 1.6e-5, image 1.4e-4, audio 4.5e-5..9.4e-5. Upstream PR pending. |

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
