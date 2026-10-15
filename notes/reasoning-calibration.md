# Reasoning models in calibration (pollard-modelkind + pollard-calib)

**Why.** The LFM2.5-24B-A2B build calibrated a model that always reasons on a corpus with ZERO `<think>` blocks:
pollard-calib never acted on "thinking", and the reference answers it renders carry no thinking. The imatrix
then protects the tensors of a model that never thinks, while the shipped model spends most of its tokens
thinking.

**Detection (pollard_modelkind.reasoning_info).** Renders the chat template with jinja2 (no tokenizer):
an assistant turn carrying reasoning_content/thinking/reasoning shows the model's own opener, closer and
separators; rendering the generation prompt with each switch on/off (enable_thinking, thinking, thinking_mode,
reasoning, reasoning_effort no_think, ...) shows toggles and their default; a prompt that already ends in the
opener means always-on. Static fallbacks (no jinja2): known delimiter pairs, tested template variables. Then
names/tags/config (R1-distill strips the block; Qwen3 *-Instruct-2507 / non-thinking excluded). Checked against
all 75 real templates in llama.cpp's models/templates: Qwen3/3.5, QwQ, R1-distill, DeepSeek V3.1-V4, GLM 4.6/4.7,
LFM2.5, Kimi K2-Thinking/K3, MiniMax M2/M3, Seed-OSS, Cohere reasoning, Gemma 4, gpt-oss/llm-jp harmony,
Ministral reasoning, Hy3, Solar Open, Nemotron, Ling, SmolLM3, Granite 3.3, Apriel all detected with their own
delimiters; Llama/Mistral/Qwen2.5/Qwen3-Coder/Kimi-K2-Instruct/Phi-3.5 not.

**Calibration.** Reasoning share (default 0.35 of the corpus) from open-thoughts/OpenThoughts-114k `metadata`
(DeepSeek-R1 traces, Apache-2.0, rev bd093c3) with open-r1/OpenR1-Math-220k fallback (Apache-2.0, rev e4e141e),
bundled seeds offline; traces trimmed to head+tail (6000 chars). Rendered exactly as the model emits: the
template's own rendering of the final turn when it renders thinking, else generation prompt + delimiters +
end-of-turn. Toggle models: thinking ON for the reasoning share, OFF (template kwargs) for the rest.
`--self-generate` keeps the model's own thinking. The corpus is NOT written if a reasoning model ends up with zero
thinking blocks; `<out>.sources.json` carries `reasoning.verdict` + `reasoning.report` (block counts).

## Needs a real run (Mac, when the box/Mac is free)

```bash
# 1. audit the corpus the LFM2.5-24B build used (expect FAIL: zero thinking blocks)
pollard-calib --check <lfm-build>/calibration/pollard_calib.txt --model <LFM2.5-24B-A2B dir>
# 2. rebuild it (reasoning share automatic) and the imatrix, then the same rungs; compare KL on a held-out
#    split that ALSO carries thinking (the new --held-out does) -- old imatrix vs new imatrix, same rung types
pollard-calib --model <LFM2.5-24B-A2B dir> --out cal-r.txt --held-out held-r.txt
llama-imatrix -m LFM2.5-24B-A2B-f16.gguf -f cal-r.txt -o lfm-r.imatrix -ngl 0 --output-format dat
```
