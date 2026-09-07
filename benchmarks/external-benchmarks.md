# External benchmarks — run them yourself on a Pollard'd model

Pollard's job ends at the **GGUF / GPTQ** file. After that it's a standard model — point any eval
framework at it. This is a curated menu of what's worth running (2026), split by what it actually tells
you. **For judging a QUANT, the top section is what matters; the rest measure model capability** and are
here so you can put your Pollard'd model on the same boards everyone else uses.

> One rule: always compare **at matched size** (Pollard vs f16/Q8, or Pollard vs a rival's file of
> similar GB). A number without a size class is marketing, not measurement.

## 1. Quant-quality metrics (this is how you grade a Pollard model)

These come with Pollard — no external tool needed. See [README.md](README.md).

| metric | what it says | tool |
|---|---|---|
| **PPL** (perplexity) | raw language-model loss vs a reference corpus | `llama-perplexity`, `pollard-bench` |
| **KL-divergence vs f16** | how far the quant's full distribution drifts from the source model | `llama-perplexity --kl-divergence`, `pollard-kl` |
| **top-1 agreement** ("Same top p") | how often the quant picks the *same next token* as f16 | same |
| **chat gate** | coherence under real sampling (loop/drift traps) | `pollard-bench --coherence` |

⚠️ **Reference = the ORIGINAL f16**, not a QAT-folded f16 (a QAT model's folded f16 is off-distribution —
see the STQ1_0 section in [Ref-pipeline.md](Ref-pipeline.md)). Cap `--chunks` (~24) — full-corpus KLD-base
writes full-vocab logits per token and balloons to tens of GB / hours.

## 2. Task probes (add 1–2 when claiming SOTA — cheap capability checks)

Runnable on a GGUF via **lm-evaluation-harness** (EleutherAI) or the benchmark's own repo.

| benchmark | measures | source |
|---|---|---|
| **GPQA-Diamond** | graduate-level science reasoning | `idavidrein/gpqa` |
| **AIME** / **FrontierMath (Tier 4)** | hard math reasoning | AIME (competition sets); FrontierMath = Epoch AI |
| **ARC-AGI-3** | abstraction/reasoning | ARC Prize |
| **MMLU-Pro**, **HellaSwag**, **wikitext** | broad knowledge / everyday quant-regression checks | lm-eval-harness built-ins |

These are the right "did the quant hurt reasoning?" probes beyond PPL — a low-bit model can hold PPL but
lose multi-step reasoning first.

## 3. Agentic / coding / long-context boards (model-capability; need a harness beyond a raw GGUF)

Useful if you want to place a Pollard'd model on the boards people cite in 2026. Most need an agent
scaffold or a hosted runner — the GGUF is just the engine.

| board | measures | notes (2026) |
|---|---|---|
| **FrontierSWE v2** (Proximal) | ultra-long-horizon coding (34 tasks, 20-h budget) | leaderboard release; find Proximal's official repo |
| **Harness-of-Harness (HoH)** | multi-day autonomous dev w/ continual improvement | arXiv 2609.01481 · GitHub `Flesymeb/HarnessOfHarness` |
| **DeepSWE v1.1** | long-horizon agentic coding | — |
| **SWEAtlas / SWE-bench** | codebase understanding & repo fixes | `princeton-nlp/SWE-bench` lineage |
| **Terminal-Bench 2.1** | agentic terminal use | — |
| **OSWorld 2.0** | agentic computer use | `xlang-ai/OSWorld` |
| **AutomationBench / JobBench / GDPVal-AA** | end-to-end business workflows / professional tool use | 2026 releases |
| **MRCR (256K–1M)** | long-context retrieval | long-context probe |
| **DeepSearchQA** | agentic browsing | 2026 release |

## 4. Speed / deployment references (adjacent, not quant-quality)

- **Voz** — ANE-optimized rearchitected NVIDIA Parakeet; ~5–20× faster on-device STT on Mac/iPhone.
- **zg** (Alibaba Zvec) — local-first BM25 search for agents (`rg`→`zg`); agent-memory/retrieval, not eval.

## Caveats

- **"GPT-6 Astra" benchmark table** circulating on X is an **unverified rumor** — treat as fake until an
  official source is cited. Don't put it in a comparison.
- Vendor self-reported tables (Muse Spark, etc.) are marketing until independently reproduced — reproduce
  before citing in a Pollard head-to-head.
