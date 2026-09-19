#!/usr/bin/env python3
"""pollard-modelkind -- what KIND of model is this, so the rest of Pollard stops guessing.

A build is measured and gated as though every model were a plain text model. It is not. A
reasoning-tuned model opens with a thinking block and only answers afterwards; an agentic model is
trained to emit tool calls; an instruct model was tuned away from raw-text modelling entirely. Score
one of those on raw Wikipedia and the numbers are meaningless -- gemma-4-12B-it reports perplexity
~664 there while a plain 7B reports 5.4 on the same corpus and binary -- and gate it with a short
token budget and it gets cut off mid-thought and fails a build that was about to answer correctly.

None of that needs guessing: the chat template says so. It is shipped in the GGUF and in the HF
checkout, and it names the turn structure the model was trained on.

    pollard-modelkind --model <gguf-or-hf-dir>            # what it is, and what to measure it with

Everything here is model-side and self-contained.
"""
from __future__ import annotations

import argparse, json, os, re

# How many tokens the coherence gate must allow. A thinking model spends its first hundred-odd
# tokens reasoning, so a short budget reads as incoherence.
TOKENS_PLAIN, TOKENS_THINKING = 96, 220


def _template_and_arch(model):
    """(chat_template, architectures, config) from a GGUF or an HF checkout."""
    if str(model).endswith(".gguf") and os.path.isfile(model):
        try:
            from pollard_calc import read_gguf_meta
            m = read_gguf_meta(model)
            arch = m.get("general.architecture", "")
            return (m.get("tokenizer.chat_template", "") or "",
                    [arch] if arch else [], {k: v for k, v in m.items() if isinstance(k, str)})
        except Exception:
            return "", [], {}
    tpl = ""
    # chat_template.jinja is a plain file, not JSON -- newer checkouts ship it that way and reading
    # only the JSON ones reports a chat model as a base model.
    jinja = os.path.join(model, "chat_template.jinja")
    if os.path.isfile(jinja):
        try:
            tpl = open(jinja, encoding="utf-8").read()
        except OSError:
            pass
    for name in ("tokenizer_config.json", "chat_template.json"):
        q = os.path.join(model, name)
        if os.path.isfile(q):
            try:
                d = json.load(open(q, encoding="utf-8"))
                tpl = d.get("chat_template") or tpl
            except (ValueError, OSError):
                pass
    archs, conf = [], {}
    cfg = os.path.join(model, "config.json")
    if os.path.isfile(cfg):
        try:
            conf = json.load(open(cfg, encoding="utf-8"))
            archs = list(conf.get("architectures") or [])
            archs += list((conf.get("text_config") or {}).get("architectures") or [])
        except (ValueError, OSError):
            pass
    return tpl or "", archs, conf


def classify(model) -> dict:
    """What this model is, and what that implies for measuring it."""
    tpl, archs, conf = _template_and_arch(model)
    low, joined = tpl.lower(), " ".join(archs)
    why = []

    instruct = bool(tpl)
    if instruct:
        why.append("ships a chat template -> instruct-tuned, not a raw-text model")
    # A passing mention is not a capability; require the template to work with it repeatedly.
    thinking = len(re.findall(r"think|reasoning_content|<analysis", low)) >= 3
    if thinking:
        why.append(f"template drives a thinking block ({len(re.findall('think', low))} mentions)")
    agentic = len(re.findall(r"tool_call|tool_response|function_call|\btools\b", low)) >= 3
    if agentic:
        why.append(f"template drives tool calls ({len(re.findall('tool', low))} mentions)")
    # Modalities. HF puts each one in its own sub-config (vision_config, audio_config, ...), the
    # template carries its placeholder tokens, and the architecture name usually says so too. A
    # model that takes or emits something other than text cannot be scored like text at all.
    blob = (low + " " + joined.lower() + " " +
            " ".join(k.lower() for k in conf) + " " +
            " ".join(str(v).lower()[:200] for k, v in conf.items() if "config" in str(k).lower()))
    MODALITY = {
        "image":  (r"vision_config|image_token|<image>|pixel_values|mm_?proj|imagetext|vision2seq|"
                   r"visiontext|image_size"),
        "audio":  r"audio_config|audio_token|<audio>|whisper|wav2vec|speech_encoder|mel_bins|qwen2audio",
        "video":  r"video_config|video_token|<video>|num_frames|frame_rate|videollava|video_encoder",
        "speech_out": r"vocoder|codec_config|snac|speech_decoder|tts|audio_head",
    }
    modalities = [m for m, pat in MODALITY.items() if re.search(pat, blob)]
    vision = "image" in modalities
    for m in modalities:
        why.append(f"{m} signals in the template/config/architecture")
    if not instruct:
        why.append("no chat template -> base model; raw-text perplexity is meaningful here")

    # Anything beyond text changes what a "score" even means: a text corpus exercises none of the
    # encoder that makes the model what it is, so a text-only number is partial by construction.
    multimodal = [m for m in modalities if m != "speech_out"]
    return {
        "instruct": instruct, "thinking": thinking, "agentic": agentic, "vision": vision,
        "modalities": modalities, "base": not instruct,
        # A model tuned away from raw text must be scored on text it was tuned FOR, or the number
        # describes the mismatch rather than the build.
        "eval": ("multimodal" if multimodal else
                 "raw-text" if not instruct else "in-domain"),
        "gate_tokens": TOKENS_THINKING if thinking else TOKENS_PLAIN,
        "why": why,
    }


def describe(k: dict) -> str:
    tags = [n for n in ("base", "instruct", "thinking", "agentic") if k.get(n)]
    tags += k.get("modalities", [])
    return "+".join(tags) or "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="a .gguf or an HF checkout")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args()
    k = classify(a.model)
    if a.json:
        print(json.dumps(k, indent=1)); return
    print(f"== pollard-modelkind :: {a.model}")
    print(f"   kind        : {describe(k)}")
    for w in k["why"]:
        print(f"                 - {w}")
    note = {"in-domain": "   (raw Wikipedia would measure the mismatch, not the build)",
            "multimodal": "   (a TEXT corpus exercises none of the " +
                          "/".join(m for m in k["modalities"] if m != "speech_out") +
                          " encoder -- any text-only score is partial by construction)",
            "raw-text": ""}
    print(f"   eval corpus : {k['eval']}{note.get(k['eval'], '')}")
    print(f"   gate budget : {k['gate_tokens']} tokens"
          + ("   (thinking block runs first)" if k["thinking"] else ""))


if __name__ == "__main__":
    main()
