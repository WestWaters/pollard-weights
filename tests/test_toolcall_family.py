"""Tool-call FORMAT detection (pollard-modelkind toolcall_info): which parser family a chat template writes calls in.

A model that calls tools writes the call in its own syntax and the server must be told which parser reads it. The
template says so: these tests grade the sniff (llama.cpp's specialised-handler markers, same order), the render-and-
classify path (a dummy tool call rendered through the template), the vLLM/llama.cpp serving advice, and the family
writers calibration falls back on when a template does not render tool_calls back.

Rendering needs jinja2 (skipped without it); the static sniff does not. When the llama.cpp checkout is present
(runtime/llama.cpp/models/templates), every shipped template is classified against a hand-labelled answer key.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load():
    spec = importlib.util.spec_from_file_location("pmk_tools", ROOT / "tools/pollard_modelkind.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pmk_tools"] = m
    spec.loader.exec_module(m)
    return m


MK = _load()
from toolcall_templates import TEMPLATES  # noqa: E402

try:
    import jinja2  # noqa: F401
    HAS_JINJA = True
except ImportError:                                                  # pragma: no cover
    HAS_JINJA = False
needs_jinja = pytest.mark.skipif(not HAS_JINJA, reason="template rendering needs jinja2")


# ------------------------------------------------------------------ the shipped templates, one family each

@needs_jinja
@pytest.mark.parametrize("key,family,source", [
    ("QWEN25", "hermes", "render"),
    ("LLAMA32", "llama3_json", "render"),
    ("MISTRAL_NEMO", "mistral", "static+render"),
    ("GLM47", "glm47", "render"),
    ("LFM2", "lfm2", "static"),                     # the template drops tool_calls; the markers carry it
])
def test_real_templates_classify_to_their_family(key, family, source):
    _, src = TEMPLATES[key]
    info = MK.toolcall_info(None, src)
    assert info["family"] == family
    assert info["source"] == source
    assert info["supports_tool_calls"] is True
    assert info["supports_tools"] is True


@needs_jinja
def test_a_template_with_no_tools_is_none():
    info = MK.toolcall_info(None, TEMPLATES["GEMMA2"][1])
    assert info["family"] == "none"
    assert info["supports_tools"] is False and info["supports_tool_calls"] is False
    assert info["vllm_parser"] is None and info["llama_cpp"] is None


@needs_jinja
def test_hermes_serving_advice_and_example():
    info = MK.toolcall_info(None, TEMPLATES["QWEN25"][1])
    assert info["vllm_parser"] == "hermes" and info["vllm_parser_confirmed"] is True
    assert info["vllm_flags"] == "--enable-auto-tool-choice --tool-call-parser hermes"
    assert info["llama_cpp"]["flags"] == "--jinja"
    assert info["llama_cpp"]["handler"] == "autoparser"
    assert "JSON_NATIVE" in info["llama_cpp"]["format"]
    # the example IS the model's own call, starting at the call and ending at its closer
    assert info["example"].startswith("<tool_call>") and info["example"].endswith("</tool_call>")
    assert '"name": "get_weather"' in info["example"]
    assert info["tool_list"] == {"placement": "system prompt", "container": "<tools>"}


@needs_jinja
def test_llama3_puts_the_tool_list_in_the_user_turn():
    info = MK.toolcall_info(None, TEMPLATES["LLAMA32"][1])
    assert info["tool_list"]["placement"] == "user turn"


def test_serve_with_line():
    t = {"family": "hermes", "vllm_parser": "hermes", "vllm_parser_confirmed": True,
         "vllm_flags": "--enable-auto-tool-choice --tool-call-parser hermes",
         "llama_cpp": {"flags": "--jinja", "handler": "autoparser", "format": "autoparser (JSON_NATIVE)"}}
    line = MK.serve_with_line(t, "org/m")
    assert "llama-server --jinja" in line and "vllm serve org/m --enable-auto-tool-choice --tool-call-parser hermes" in line
    assert "no tool calling" in MK.serve_with_line({"family": "none"})
    t2 = dict(t, family="apriel", vllm_parser=None, vllm_flags=None)
    assert "no tool parser" in MK.serve_with_line(t2)


# ------------------------------------------------------------------ the static sniff mirrors llama.cpp

@pytest.mark.parametrize("src,handler,family", [
    ("[SYSTEM_PROMPT] [TOOL_CALLS] [ARGS]", "Ministral/Magistral 3", "mistral"),
    ("[SYSTEM_PROMPT] [TOOL_CALLS] [ARGS] [CALL_ID]", "autoparser", None),          # Mistral Small 3.2
    ("<|channel|>analysis", "GPT-OSS", "harmony"),
    ("<|tool_calls_section_begin|><|tool_call_begin|>", "Kimi K2", "kimi_k2"),
    ("<|START_TEXT|><|START_ACTION|>", "Cohere2 MoE", "cohere"),
    ("<|tool_list_start|>[...]<|tool_list_end|>", "LFM2", "lfm2"),
    ("List of tools: [", "LFM2.5", "lfm2"),
    ("dsml_token DSML function_calls", "DeepSeek V3.2/V4", "deepseek_v32"),
    ("'<|tool_call>call:'", "Gemma4", "gemma4"),
    ("<tool_call><function=x><parameter=y>", "Qwen3-Coder", "qwen3_coder"),
    # first match wins, as in llama.cpp: harmony's channel marker beats a Qwen3-Coder body
    ("<|channel|> <tool_call><function=x><parameter=y>", "GPT-OSS", "harmony"),
])
def test_llama_cpp_handler_order(src, handler, family):
    assert MK.llama_cpp_handler(src) == (handler, family)


@pytest.mark.parametrize("src,family,strong", [
    ("<seed:tool_call>", "seed_oss", True),
    ("<|tool_call|>", "granite", True),
    ("<｜tool▁calls▁begin｜>function<｜tool▁sep｜>x ```json", "deepseek_v3", True),
    ("<｜tool▁calls▁begin｜>x<｜tool▁sep｜>{}", "deepseek_v31", True),
    ("[TOOL_CALLS]", "mistral", True),
    ("<tool_call>{{ tool_call | tojson }}</tool_call> tools", "hermes", False),
    ("plain chat", None, False),
])
def test_static_sniff(src, family, strong):
    assert MK._sniff_tool_family(src) == (family, strong)


def test_without_jinja_the_sniff_still_answers(monkeypatch):
    """No jinja2 -> no rendering; the markers alone still name the family llama.cpp would pick."""
    def boom(*a, **k):
        raise ImportError("no jinja2")
    monkeypatch.setattr(MK, "ChatTemplate", boom)
    info = MK.toolcall_info(None, TEMPLATES["MISTRAL_NEMO"][1])
    assert info["family"] == "mistral" and info["source"] == "static"
    info = MK.toolcall_info(None, "{% for m in messages %}{{ m.content }}{% endfor %}")
    assert info["family"] == "none"


# ------------------------------------------------------------------ the call-text classifier

@pytest.mark.parametrize("text,family", [
    ('<tool_call>\n{"name": "pzq_lookup", "arguments": {}}\n</tool_call>', "hermes"),
    ("<tool_call>\n<function=pzq_lookup>\n<parameter=city>\nx\n</parameter>\n</function>\n</tool_call>", "qwen3_coder"),
    ("<|tool_call_start|>[pzq_lookup(city='x')]<|tool_call_end|>", "lfm2"),
    ("<|start|>assistant to=functions.pzq_lookup<|channel|>commentary json<|message|>{}", "harmony"),
    ('[TOOL_CALLS]pzq_lookup[ARGS]{"city": "x"}', "mistral"),
    ('{"name": "pzq_lookup", "parameters": {"city": "x"}}', "llama3_json"),
    ("[pzq_lookup(city='x')]", "pythonic"),
    ('[{"name": "pzq_lookup", "arguments": {}}]', "xlam"),
    ("<tool_call>pzq_lookup<arg_key>city</arg_key><arg_value>x</arg_value></tool_call>", "glm47"),
    ("<tool_call>pzq_lookup\n<arg_key>city</arg_key>\n<arg_value>x</arg_value>\n</tool_call>", "glm45"),
    ("<start_function_call>call:pzq_lookup{city:<escape>x<escape>}<end_function_call>", "functiongemma"),
    ('<|tool_call|>[{"name": "pzq_lookup", "arguments": {}}]', "granite"),
    ("nothing here", None),
])
def test_classify_call_text(text, family):
    assert MK._classify_call_text(text) == family


@pytest.mark.parametrize("family", ["hermes", "qwen3_coder", "lfm2", "pythonic", "mistral", "llama3_json", "xlam",
                                    "granite", "deepseek_v3", "deepseek_v31", "glm45", "glm47", "kimi_k2",
                                    "minimax_m2", "seed_oss", "cohere", "functiongemma", "gemma4"])
def test_format_tool_call_round_trips_through_the_classifier(family):
    """What calibration writes for a template that drops tool_calls must read back as the same family."""
    txt = MK.format_tool_call(family, MK._TC_NAME, {"city": "Paris", "days": 3})
    if txt is None:
        pytest.skip(f"no writer for {family}")
    assert MK._classify_call_text(txt) == family, txt


def test_format_tool_call_unknown_family_is_none():
    assert MK.format_tool_call("apertus", "f", {}) is None or isinstance(MK.format_tool_call("apertus", "f", {}), str)
    assert MK.format_tool_call("no-such-family", "f", {}) is None


def test_every_family_has_serving_metadata():
    for fam, meta in MK.TOOL_FAMILIES.items():
        assert {"vllm", "vllm_ok", "shape", "fmt"} <= set(meta), fam
        if meta["vllm"] is None:
            assert meta["vllm_ok"] is None, fam


# ------------------------------------------------------------------ every template llama.cpp ships

TEMPLATE_DIR = Path(os.environ.get("POLLARD_LLAMA_TEMPLATES", "")) if os.environ.get("POLLARD_LLAMA_TEMPLATES") \
    else ROOT / "runtime" / "llama.cpp" / "models" / "templates"

#: hand-labelled from each template's rendered call + llama.cpp's handler for it
EXPECTED = {
    "Apertus-8B-Instruct": "apertus", "Apriel-1.6-15b-Thinker-fixed": "apriel", "Bielik-11B-v3.0-Instruct": "hermes",
    "ByteDance-Seed-OSS": "seed_oss", "Cohere2MoE": "cohere", "CohereForAI-c4ai-command-r-plus-tool_use": "cohere_cmdr",
    "CohereForAI-c4ai-command-r7b-12-2024-tool_use": "cohere", "GLM-4.6": "glm45", "GLM-4.7-Flash": "glm47",
    "GigaChat3-10B-A1.8B": "gigachat", "GigaChat3.1-10B-A1.8B": "gigachat", "HuggingFaceTB-SmolLM3-3B": "none",
    "Kimi-K2-Instruct": "kimi_k2", "Kimi-K2-Thinking": "kimi_k2", "Kimi-K3": "kimi_k3", "LFM2-8B-A1B": "lfm2",
    "LFM2.5-8B-A1B": "lfm2", "LFM2.5-Instruct": "lfm2", "MiMo-VL": "hermes", "MiniMax-M1": "minimax_m1",
    "MiniMax-M2": "minimax_m2", "MiniMax-M3": "minimax_m3", "Mistral-Small-3.2-24B-Instruct-2506": "mistral",
    "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "qwen3_coder", "NVIDIA-Nemotron-Nano-v2": "nemotron_v2",
    "NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use": "hermes", "NousResearch-Hermes-3-Llama-3.1-8B-tool_use": "hermes",
    "Qwen-QwQ-32B": "hermes", "Qwen-Qwen2.5-7B-Instruct": "hermes", "Qwen-Qwen3-0.6B": "hermes",
    "Qwen3-Coder": "qwen3_coder", "Qwen3.5-4B": "qwen3_coder", "Reka-Edge": "hermes", "Spark2.5": "glm47",
    "StepFun3.5-Flash": "qwen3_coder", "deepseek-ai-DeepSeek-R1-Distill-Llama-8B": "deepseek_v3",
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B": "deepseek_v3", "deepseek-ai-DeepSeek-V3.1": "deepseek_v31",
    "deepseek-ai-DeepSeek-V3.2": "deepseek_v32", "deepseek-ai-DeepSeek-V4-Flash-0731": "deepseek_v32",
    "deepseek-ai-DeepSeek-V4": "deepseek_v32", "fireworks-ai-llama-3-firefunction-v2": "firefunction",
    "google-gemma-2-2b-it": "none", "google-gemma-4-31B-it-interleaved": "gemma4", "google-gemma-4-31B-it": "gemma4",
    "ibm-granite-granite-3.3-2B-Instruct": "granite", "ibm-granite-granite-4.0": "hermes",
    "ibm-granite-granite-4.1": "hermes", "llama-cpp-deepseek-r1": "deepseek_v3", "llama-cpp-rwkv-world": "none",
    "meetkai-functionary-medium-v3.1": "functionary_v31", "meetkai-functionary-medium-v3.2": "functionary_v32",
    "meta-llama-Llama-3.1-8B-Instruct": "llama3_json", "meta-llama-Llama-3.2-3B-Instruct": "llama3_json",
    "meta-llama-Llama-3.3-70B-Instruct": "llama3_json", "microsoft-Phi-3.5-mini-instruct": "none",
    "mistralai-Ministral-3-14B-Reasoning-2512": "mistral", "mistralai-Mistral-Nemo-Instruct-2407": "mistral",
    "moonshotai-Kimi-K2": "kimi_k2", "muse-glimmer": "muse_glimmer", "openai-gpt-oss-120b": "harmony",
    "openbmb-MiniCPM5-1B": "minicpm5", "poolside-Laguna-S-2.1": "glm47", "poolside-Laguna-XS-2.1": "glm45",
    "poolside-Laguna-XS.2": "glm45", "tencent-Hy3": "hunyuan", "unsloth-Apriel-1.5": "apriel",
    "unsloth-mistral-Devstral-Small-2507": "mistral", "upstage-Solar-Open-100B": "solar_open",
}


@needs_jinja
@pytest.mark.skipif(not TEMPLATE_DIR.is_dir(), reason="no llama.cpp checkout with models/templates")
def test_every_llama_cpp_template_classifies():
    wrong = []
    files = sorted(TEMPLATE_DIR.glob("*.jinja"))
    for f in files:
        want = EXPECTED.get(f.stem)
        if want is None:
            continue                                            # a template newer than this key: not graded
        got = MK.toolcall_info(None, f.read_text(encoding="utf-8"))["family"]
        if got != want:
            wrong.append((f.stem, got, want))
    graded = sum(1 for f in files if f.stem in EXPECTED)
    assert graded >= 60
    assert not wrong, wrong
