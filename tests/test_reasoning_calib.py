"""Reasoning-model detection (pollard-modelkind) and reasoning calibration (pollard-calib).

The LFM2.5-24B build calibrated a model that ALWAYS reasons on a corpus with zero `<think>` blocks. These pin:
every family style is detected from its template (synthetic templates, one per style), the thinking is
rendered with the model's OWN delimiters (on and off for a switchable model), the corpus sanity gate fails a
reasoning model's corpus with no thinking, and the eval side strips thinking for any delimiter style.

Template rendering needs jinja2 (skipped without it); the static fallbacks and the splitters do not."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_calib as C  # noqa: E402
import pollard_modelkind as K  # noqa: E402

HAS_JINJA = True
try:
    import jinja2  # noqa: F401
except ImportError:
    HAS_JINJA = False
needs_jinja = pytest.mark.skipif(not HAS_JINJA, reason="template rendering needs jinja2")

# ----------------------------------------------------------------- synthetic templates, one per family style

CHATML_TURNS = (
    "{%- for m in messages %}"
    "{%- if m.role == 'assistant' %}{{ '<|im_start|>assistant\\n' }}"
    "{%- if m.reasoning_content is defined and m.reasoning_content and loop.last %}"
    "{{ '<think>\\n' + m.reasoning_content + '\\n</think>\\n\\n' }}{%- endif %}"
    "{{ m.content.split('</think>')[-1] + '<|im_end|>\\n' }}"
    "{%- else %}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endif %}"
    "{%- endfor %}")

# Qwen3 hybrid: enable_thinking switch; thinking-off prefills an empty block
QWEN3 = CHATML_TURNS + (
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{ '<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}{%- endif %}")
# Qwen3-Thinking-2507 / R1-0528: the generation prompt opens the block itself (always-on)
ALWAYS_PREFILL = CHATML_TURNS + "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}"
# R1-distill: strips everything before </think>, never renders reasoning
R1_DISTILL = (
    "{{ bos_token }}{%- for m in messages %}{%- if m.role == 'user' %}<|User|>{{ m.content }}"
    "{%- elif m.role == 'assistant' %}<|Assistant|>{{ m.content.split('</think>')[-1] }}<|end_of_sentence|>"
    "{%- endif %}{%- endfor %}{%- if add_generation_prompt %}<|Assistant|>{%- endif %}")
# LFM2.5: only strips past thinking, no switch, no opener anywhere -- the model opens it itself
LFM25 = (
    "{%- set keep_past_thinking = keep_past_thinking | default(false) -%}"
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n{%- set c = m.content %}"
    "{%- if m.role == 'assistant' and not keep_past_thinking and not loop.last and '</think>' in c %}"
    "{%- set c = c.split('</think>')[-1] | trim %}{%- endif %}{{ c }}<|im_end|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}")
# gpt-oss harmony: analysis channel then final, graded by reasoning_effort
HARMONY = (
    "<|start|>system<|message|>Reasoning: {{ reasoning_effort | default('medium') }}<|end|>"
    "{%- for m in messages %}{%- if m.role == 'user' %}<|start|>user<|message|>{{ m.content }}<|end|>"
    "{%- elif m.role == 'assistant' %}{%- if m.thinking is defined and m.thinking and loop.last %}"
    "<|start|>assistant<|channel|>analysis<|message|>{{ m.thinking }}<|end|>{%- endif %}"
    "<|start|>assistant<|channel|>final<|message|>{{ m.content }}<|return|>{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}<|start|>assistant{%- endif %}")
# Seed-OSS
SEED = (
    "{%- for m in messages %}<seed:bos>{{ m.role }}\n"
    "{%- if m.role == 'assistant' and m.reasoning_content is defined and m.reasoning_content %}"
    "<seed:think>{{ m.reasoning_content }}</seed:think>\n{%- endif %}{{ m.content }}<seed:eos>{%- endfor %}"
    "{%- if thinking_budget is defined and thinking_budget == 0 %}<seed:bos>system\nno thinking<seed:eos>{%- endif %}"
    "{%- if add_generation_prompt %}<seed:bos>assistant\n{%- endif %}")
# Magistral / Ministral reasoning
MAGISTRAL = (
    "<s>{%- for m in messages %}{%- if m.role == 'user' %}[INST]{{ m.content }}[/INST]"
    "{%- else %}{%- if m.reasoning_content is defined and m.reasoning_content %}"
    "[THINK]{{ m.reasoning_content }}[/THINK]{%- endif %}{{ m.content }}</s>{%- endif %}{%- endfor %}")
# DeepSeek-V3.1: `thinking` switch, prefill <think> on / </think> off
DSV31 = (
    "{%- for m in messages %}{%- if m.role == 'user' %}<|User|>{{ m.content }}"
    "{%- else %}<|Assistant|></think>{{ m.content }}<|end_of_sentence|>{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}<|Assistant|>{%- if thinking is defined and thinking %}<think>"
    "{%- else %}</think>{%- endif %}{%- endif %}")
PLAIN = ("{%- for m in messages %}<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content }}<|eot_id|>"
         "{%- endfor %}{%- if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{%- endif %}")


def _model(tmp_path, name, tpl, extra=None):
    d = tmp_path / name
    d.mkdir()
    (d / "chat_template.jinja").write_text(tpl, encoding="utf-8")
    (d / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<bos>", "eos_token": "<eos>"}),
                                             encoding="utf-8")
    (d / "config.json").write_text(json.dumps({"architectures": ["FooForCausalLM"], **(extra or {})}),
                                   encoding="utf-8")
    return str(d)


# ----------------------------------------------------------------- detection

@needs_jinja
@pytest.mark.parametrize("name,tpl,mode,open_,close", [
    ("Qwen3-8B", QWEN3, "toggle", "<think>", "</think>"),
    ("Qwen3-4B-Thinking-2507", ALWAYS_PREFILL, "always", "<think>", "</think>"),
    ("LFM2.5-24B-A2B", LFM25, "always", "<think>", "</think>"),
    ("gpt-oss-20b", HARMONY, "always", "<|channel|>analysis<|message|>", "<|end|>"),
    ("Seed-OSS-36B", SEED, "always", "<seed:think>", "</seed:think>"),
    ("Magistral-Small-2509", MAGISTRAL, "always", "[THINK]", "[/THINK]"),
    ("DeepSeek-V3.1", DSV31, "toggle", "<think>", "</think>"),
])
def test_every_family_style_is_detected_with_its_own_delimiters(tmp_path, name, tpl, mode, open_, close):
    info = K.reasoning_info(_model(tmp_path, name, tpl))
    assert info["reasoning"], (name, info["why"])
    assert info["mode"] == mode, (name, info)
    assert info["open"] == open_ and info["close"] == close, (name, info)


@needs_jinja
def test_toggle_reports_the_switch_and_its_default(tmp_path):
    info = K.reasoning_info(_model(tmp_path, "Qwen3-8B", QWEN3))
    assert info["toggle"]["var"] == "enable_thinking"
    assert info["toggle"]["on"] == {"enable_thinking": True}
    assert info["toggle"]["off"] == {"enable_thinking": False}
    ds = K.reasoning_info(_model(tmp_path, "DeepSeek-V3.1", DSV31))
    assert ds["toggle"]["var"] == "thinking" and ds["toggle"]["default_on"] is False


@needs_jinja
def test_always_on_prefill_is_seen_in_the_generation_prompt(tmp_path):
    info = K.reasoning_info(_model(tmp_path, "Qwen3-4B-Thinking-2507", ALWAYS_PREFILL))
    assert info["prefill"] == "<think>"


def test_r1_distill_is_reasoning_although_its_template_strips_the_block(tmp_path):
    info = K.reasoning_info(_model(tmp_path, "DeepSeek-R1-Distill-Qwen-7B", R1_DISTILL))
    assert info["reasoning"] and info["mode"] == "always"
    assert info["close"] == "</think>"


def test_non_thinking_releases_and_plain_models_are_not_reasoning(tmp_path):
    # Qwen3-*-Instruct-2507 keeps the history-stripping, but the release does not think
    assert not K.reasoning_info(_model(tmp_path, "Qwen3-4B-Instruct-2507", LFM25))["reasoning"]
    assert not K.reasoning_info(_model(tmp_path, "Llama-3.1-8B-Instruct", PLAIN))["reasoning"]


def test_names_and_card_tags_catch_a_reasoning_model_with_no_template(tmp_path):
    d = tmp_path / "models--LiquidAI--LFM2.5-24B-A2B" / "snapshots" / "abc123"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"architectures": ["Lfm2MoeForCausalLM"]}), encoding="utf-8")
    assert K.reasoning_info(str(d))["reasoning"], "an HF-cache snapshot is a hash -- the repo name must count"
    t = tmp_path / "my-model"
    t.mkdir()
    (t / "README.md").write_text("---\ntags:\n- reasoning\n---\n# card\n", encoding="utf-8")
    assert K.reasoning_info(str(t))["reasoning"]


def test_static_fallback_without_jinja(tmp_path, monkeypatch):
    def no_jinja():
        raise ImportError("no jinja2")
    monkeypatch.setattr(K, "_jinja_env", no_jinja)
    q = K.reasoning_info(_model(tmp_path, "Qwen3-8B", QWEN3))
    assert q["reasoning"] and q["mode"] == "toggle" and q["toggle"]["var"] == "enable_thinking"
    lfm = K.reasoning_info(_model(tmp_path, "LFM2.5-24B-A2B", LFM25))
    assert lfm["reasoning"] and lfm["close"] == "</think>"
    h = K.reasoning_info(_model(tmp_path, "gpt-oss-20b", HARMONY))
    assert h["reasoning"] and h["style"] == "harmony"
    # an end-of-turn `<|end|>` alone (Phi) is not a reasoning channel
    phi = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}"
    assert not K.reasoning_info(_model(tmp_path, "Phi-3.5-mini-instruct", phi))["reasoning"]


def test_classify_carries_the_reasoning_verdict(tmp_path):
    k = K.classify(_model(tmp_path, "DeepSeek-R1-Distill-Qwen-7B", R1_DISTILL))
    assert k["thinking"] and k["reasoning"]["close"] == "</think>"
    assert k["gate_tokens"] == K.TOKENS_THINKING


# ----------------------------------------------------------------- splitting generated text (eval side)

@pytest.mark.parametrize("text,think,answer,unclosed", [
    ("<think>\nadd them\n</think>\n\n42", "add them", "42", False),
    ("add them\n</think>\n\n42", "add them", "42", False),                      # opener was prefilled
    ("[THINK]hmm[/THINK]Paris", "hmm", "Paris", False),
    ("<seed:think>x</seed:think>\ny", "x", "y", False),
    ("<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>ans<|return|>",
     "think", "ans", False),
    ("<think>still going and going", "still going and going", "", True),
    ("plain answer", "", "plain answer", False),
])
def test_split_reasoning_handles_every_style(text, think, answer, unclosed):
    assert K.split_reasoning(text) == (think, answer, unclosed)


def test_taskeval_and_toolcall_strip_thinking_for_any_style():
    import pollard_taskeval as T
    import pollard_toolcall as TC
    assert T._strip_thinking("[THINK]plan[/THINK]Bonjour") == "Bonjour"
    assert T._strip_thinking("<think>never closed") == ""
    tool = TC.PROBES[0]["tool"]
    good = '<think>I could call {"name": "wrong"}</think>{"name": "get_weather", "arguments": {"location": "Paris"}}'
    assert TC.check_call(good, tool, {"location": "paris"})[0], "JSON sketched while thinking is not the call"
    ok, failure, _ = TC.check_call('<think>{"name": "get_weather", "arguments": {"location": "Paris"}}', tool)
    assert not ok and failure == "unclosed-think"


def test_bench_reads_the_answer_past_any_thinking_delimiter():
    import pollard_bench as B
    got = {"choices": [{"message": {"content": "[THINK]weighing[/THINK]Jupiter."}, "finish_reason": "stop"}]}
    assert B._read_chat(got) == ("Jupiter.", "weighing", "stop")


# ----------------------------------------------------------------- calibration rendering

def _rec(q="Is 221 prime?", r="13 * 17 = 221, so no.", a="No: 221 = 13 * 17."):
    return C.reasoning_record(q, r, a)


@needs_jinja
def test_reasoning_record_renders_exactly_as_qwen3_emits_it(tmp_path):
    R = C.Renderer(_model(tmp_path, "Qwen3-8B", QWEN3))
    assert R.rinfo["mode"] == "toggle"
    out = C.render(_rec(), R)
    assert "<|im_start|>assistant\n<think>\n13 * 17 = 221, so no.\n</think>\n\nNo: 221 = 13 * 17.<|im_end|>" in out
    off = C.render({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
                    "thinking": False}, R)
    assert off.endswith("<|im_start|>assistant\nhello<|im_end|>\n")


@needs_jinja
def test_template_that_strips_thinking_still_gets_it_in_the_models_delimiters(tmp_path):
    R = C.Renderer(_model(tmp_path, "DeepSeek-R1-Distill-Qwen-7B", R1_DISTILL))
    out = C.render(_rec(), R)
    assert out == ("<bos><|User|>Is 221 prime?<|Assistant|><think>\n13 * 17 = 221, so no.\n</think>\n\n"
                   "No: 221 = 13 * 17.<|end_of_sentence|>")


@needs_jinja
def test_prefilled_opener_is_not_doubled(tmp_path):
    R = C.Renderer(_model(tmp_path, "Qwen3-4B-Thinking-2507", ALWAYS_PREFILL))
    out = C.render(_rec(), R)
    assert out.count("<think>") == 1 and "</think>" in out


@needs_jinja
@pytest.mark.parametrize("name,tpl,open_,close", [
    ("gpt-oss-20b", HARMONY, "<|channel|>analysis<|message|>", "<|start|>assistant<|channel|>final<|message|>"),
    ("Seed-OSS-36B", SEED, "<seed:think>", "</seed:think>"),
    ("Magistral-Small-2509", MAGISTRAL, "[THINK]", "[/THINK]"),
    ("LFM2.5-24B-A2B", LFM25, "<think>", "</think>"),
])
def test_other_styles_render_their_own_delimiters(tmp_path, name, tpl, open_, close):
    R = C.Renderer(_model(tmp_path, name, tpl))
    out = C.render(_rec(), R)
    assert open_ in out and close in out and out.index(open_) < out.index("13 * 17") < out.index(close)
    assert "No: 221 = 13 * 17." in out[out.index(close):]


def test_generic_chatml_without_a_template_still_thinks():
    R = C.Renderer(None, reasoning="on")
    out = C.render(_rec(), R)
    assert "<|im_start|>assistant\n<think>\n13 * 17 = 221, so no.\n</think>\n\nNo: 221" in out


def test_reasoning_record_trims_long_traces_keeping_head_and_tail():
    r = "\n".join(f"step {i}" for i in range(3000))
    rec = C.reasoning_record("q", r, "a", max_chars=600)
    t = rec["messages"][1]["reasoning_content"]
    assert len(t) <= 620 and t.startswith("step 0") and t.endswith("step 2999")


def test_reasoning_share_is_added_for_a_reasoning_model():
    c = C.domain_counts("general", 100, reasoning_share=0.35)
    total = sum(c.values())
    assert abs(c["reasoning"] / total - 0.35) < 0.01
    assert "reasoning" not in C.domain_counts("general", 100)


def test_think_report_counts_blocks_and_the_gate_fails_an_empty_corpus():
    info = {"reasoning": True, "mode": "always", "open": "<think>", "close": "</think>"}
    rep = C.think_report(["<think>\nx\n</think>\n\ny", "<think>\n\n</think>\n\nz", "plain"], info)
    assert rep["blocks"] == 1 and rep["empty_blocks"] == 1 and rep["samples_with_thinking"] == 1
    assert C.think_problem(rep) is None
    none = C.think_report(["<think>\n\n</think>\n\nz", "plain"], info)
    assert "ZERO thinking blocks" in C.think_problem(none)
    assert C.think_problem(C.think_report(["plain"], {"reasoning": False})) is None


def test_self_generate_keeps_the_models_thinking_and_rejects_a_cut_off_one():
    info = {"reasoning": True, "open": "<think>", "close": "</think>"}
    rec = _rec()

    def inline(messages, tools=None, thinking=None):
        return {"role": "assistant", "content": "<think>\nmy own thought\n</think>\n\nmy answer", "_finish": "stop"}
    out, ok, bad = C.self_generate([rec], inline, info=info)
    m = out[0]["messages"][-1]
    assert ok == 1 and m["reasoning_content"] == "my own thought" and m["content"] == "my answer"

    def cut(messages, tools=None, thinking=None):
        return {"role": "assistant", "content": "<think>\nstill thinking", "_finish": "length"}
    out, ok, bad = C.self_generate([rec], cut, info=info)
    assert bad == 1 and out[0] is rec, "a thinking block cut off by the budget keeps the reference"


def _run_calib(*args, env_extra=None):
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONPATH=str(ROOT / "tools"))
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(ROOT / "tools" / "pollard_calib.py"), *args],
                          capture_output=True, text=True, env=env, timeout=300)


@needs_jinja
def test_cli_writes_reasoning_for_a_reasoning_model_and_fails_without_it(tmp_path):
    model = _model(tmp_path, "LFM2.5-24B-A2B", LFM25)
    out = tmp_path / "c.txt"
    r = _run_calib("--out", str(out), "--model", model, "--domains", "chat,prose", "--per-domain", "4")
    assert r.returncode == 0, r.stderr[-800:]
    text = out.read_text(encoding="utf-8")
    assert "<think>" in text and "</think>" in text
    src = json.loads(Path(str(out) + ".sources.json").read_text(encoding="utf-8"))
    assert src["reasoning"]["report"]["blocks"] > 0 and "reasoning" in src["domains"]

    # no reasoning share -> the gate refuses to write the corpus
    bad = tmp_path / "bad.txt"
    r = _run_calib("--out", str(bad), "--model", model, "--domains", "chat", "--per-domain", "4",
                   "--reasoning-share", "0")
    assert r.returncode != 0 and "ZERO thinking blocks" in (r.stderr + r.stdout)
    assert not bad.exists(), "a think-less corpus must not be left on disk for a resumable pipeline"

    # --check audits an existing corpus
    (tmp_path / "old.txt").write_text("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>",
                                      encoding="utf-8")
    r = _run_calib("--check", str(tmp_path / "old.txt"), "--model", model)
    assert r.returncode != 0
    r = _run_calib("--check", str(out), "--model", model)
    assert r.returncode == 0, r.stdout + r.stderr
