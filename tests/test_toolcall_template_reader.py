"""Tool calls for EVERY model: the call reader derived from the model's own chat template.

Pollard knows ~40 call syntaxes by name (pollard-modelkind CALL_READERS). A model whose syntax has no name here -- or a
known family whose reader finds nothing -- is still read: its template is asked to render a known call (one call, then
two in one turn), the wrapper / separators / call shape are read off that rendering (derive_reader), and the model's
output is parsed with exactly those markers (parse_with_reader). The SAME reader runs on every path: a served reply
whose server parser returned no tool_calls is read again by Pollard (pollard_read: llama-server / mlx_lm.server / vLLM),
and an in-process generation goes through reply_from_text.

Also here: Granite's plain `<tool_call>[{...}]` (granite_tc), which llama-server's parser does not read -- the
granite-3.1-1b-a400m tool gate was 0% parsed for reference and rung alike before Pollard read the content itself.

Rendering needs jinja2 (skipped without it); the llama.cpp sweep needs models/templates (POLLARD_LLAMA_TEMPLATES)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_modelkind as MK  # noqa: E402
import pollard_toolcall as TC  # noqa: E402

try:
    import jinja2  # noqa: F401
    HAS_JINJA = True
except ImportError:                                                  # pragma: no cover
    HAS_JINJA = False
needs_jinja = pytest.mark.skipif(not HAS_JINJA, reason="template rendering needs jinja2")

TEMPLATE_DIR = Path(os.environ.get("POLLARD_LLAMA_TEMPLATES", "")) if os.environ.get("POLLARD_LLAMA_TEMPLATES") \
    else ROOT / "runtime" / "llama.cpp" / "models" / "templates"

A1 = {"city": MK._RD_A1, "unit": MK._RD_B1}
A2 = {"city": MK._RD_A2, "unit": MK._RD_B2}
PROSE = ["The weather in Paris is sunny, city: nice.", "I can't look that up. unit = none", "",
         "Sure! Here is a summary of the article without any tool."]

PROBE = {p["id"]: p for p in TC.SUITE}


def _tools(pid):
    return [TC.TOOLBOX[t] for t in PROBE[pid]["tools"]]


def _two_calls(src):
    for mutable in (False, True):
        ct = MK.ChatTemplate(src, {}, mutable=mutable)
        two = MK._safe(MK._rd_call_text, ct, [(MK._RD_NAME, A1), (MK._RD_NAME2, A2)])
        if two and MK._RD_NAME2 in two:
            return two
    return None


# ------------------------------------------------------------------ every template llama.cpp ships

@needs_jinja
@pytest.mark.skipif(not TEMPLATE_DIR.is_dir(), reason="no llama.cpp checkout with models/templates")
def test_the_template_reader_round_trips_every_llama_cpp_template():
    """For every shipped template that renders a call: the reader derived from it reads back the ONE call and the
    TWO parallel calls the template itself writes, exactly (name + every argument), and reads no call out of prose."""
    files = sorted(TEMPLATE_DIR.glob("*.jinja"))
    derived, wrong = 0, []
    for f in files:
        src = f.read_text(encoding="utf-8")
        info = MK.toolcall_info(None, src)
        rd = info.get("reader")
        if not rd:
            continue
        derived += 1
        one = MK.parse_with_reader(rd["example"], rd, MK._RD_TOOLS)
        if one != [(MK._RD_NAME, A1)]:
            wrong.append((f.stem, "one", one))
        two = _two_calls(src)
        if two:
            got = MK.parse_with_reader(two, rd, MK._RD_TOOLS)
            if got != [(MK._RD_NAME, A1), (MK._RD_NAME2, A2)]:
                wrong.append((f.stem, "two", got))
        for p in PROSE:
            if MK.parse_with_reader(p, rd, MK._RD_TOOLS):
                wrong.append((f.stem, "prose", p))
    assert derived >= 55, derived
    assert not wrong, wrong


@needs_jinja
@pytest.mark.skipif(not TEMPLATE_DIR.is_dir(), reason="no llama.cpp checkout with models/templates")
def test_a_template_that_renders_no_call_has_no_reader_and_keeps_its_family():
    for stem, fam in (("ibm-granite-granite-3.3-2B-Instruct", "granite"), ("LFM2-8B-A1B", "lfm2"),
                      ("google-gemma-2-2b-it", "none")):
        f = TEMPLATE_DIR / f"{stem}.jinja"
        if not f.exists():
            continue
        info = MK.toolcall_info(None, f.read_text(encoding="utf-8"))
        assert info["family"] == fam and info["reader"] is None


# ------------------------------------------------------------------ a syntax Pollard has no name for

#: an invented call syntax no family reader knows: `<<act:NAME>>{json}<</act>>` -- the reader must come from the template
EXOTIC = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}<|u|>{{ m.content }}<|e|>"
    "{%- elif m.role == 'assistant' and m.tool_calls -%}<|a|>"
    "{%- for c in m.tool_calls -%}<<act:{{ c.function.name }}>>{{ c.function.arguments | tojson }}<</act>>"
    "{%- endfor -%}<|e|>"
    "{%- elif m.role == 'assistant' -%}<|a|>{{ m.content }}<|e|>"
    "{%- elif m.role == 'tool' -%}<|t|>{{ m.content }}<|e|>{%- endif -%}{%- endfor -%}"
    "{%- if tools -%}<|tools|>{{ tools | tojson }}<|e|>{%- endif -%}"
    "{%- if add_generation_prompt -%}<|a|>{%- endif -%}")

#: invented tagged syntax: `[[call NAME]] key := value ;; ... [[/call]]`
EXOTIC_TAGGED = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}<|u|>{{ m.content }}<|e|>"
    "{%- elif m.role == 'assistant' and m.tool_calls -%}<|a|>"
    "{%- for c in m.tool_calls -%}[[call {{ c.function.name }}]]"
    "{%- for k, v in c.function.arguments.items() -%} {{ k }} := {{ v }} ;;{%- endfor -%}[[/call]]"
    "{%- endfor -%}<|e|>"
    "{%- elif m.role == 'assistant' -%}<|a|>{{ m.content }}<|e|>"
    "{%- elif m.role == 'tool' -%}<|t|>{{ m.content }}<|e|>{%- endif -%}{%- endfor -%}"
    "{%- if tools -%}<|tools|>{{ tools | tojson }}<|e|>{%- endif -%}"
    "{%- if add_generation_prompt -%}<|a|>{%- endif -%}")


@needs_jinja
def test_an_unknown_syntax_is_read_through_its_own_template():
    info = MK.toolcall_info(None, EXOTIC)
    rd = info["reader"]
    assert rd and rd["shape"] == "name+json" and rd["call_open"] == "<<act:"
    raw = 'Checking.<<act:get_weather>>{"city": "Tokyo", "unit": "celsius"}<</act>>'
    assert TC.reader_family({"tools": info})[1] in ("template", info["family"])
    reply = TC.reply_from_text(raw, info["family"], _tools("s01"), reader=rd)
    assert reply["read_by"] == "template"
    row = TC.grade_reply(reply, PROBE["s01"])
    assert row["ok"] and row["read_by"] == "template", row
    # prose is prose
    row = TC.grade_reply(TC.reply_from_text("It is sunny in Tokyo.", info["family"], _tools("s01"), reader=rd),
                         PROBE["s01"])
    assert row["failure"] == "no-call"


@needs_jinja
def test_an_unknown_tagged_syntax_types_its_arguments_by_schema():
    rd = MK.toolcall_info(None, EXOTIC_TAGGED)["reader"]
    assert rd and rd["shape"] == "tagged"
    calls = MK.parse_with_reader("[[call set_timer]] minutes := 25 ;; label := bread ;;[[/call]]", rd, _tools("s11"))
    assert calls == [("set_timer", {"minutes": 25, "label": "bread"})]


def test_a_family_without_a_reader_uses_the_template_reader_when_there_is_one():
    spec = {"shape": "name+json", "call_open": "<<act:", "mid": ">>"}
    assert TC.reader_family({"tools": {"family": "kimi_k3", "reader": spec}}) == ("kimi_k3", "template")
    calls, _, attempted, how = MK.read_tool_calls('<<act:f>>{"a": 1}', "kimi_k3", None, spec)
    assert calls == [{"name": "f", "arguments": {"a": 1}}] and attempted and how == "template"
    with pytest.raises(KeyError):
        MK.read_tool_calls("x", "kimi_k3", None, None)


def test_the_family_reader_wins_and_the_template_reader_only_fills_in():
    spec = {"shape": "json", "open": "<tool_call>", "close": "</tool_call>"}
    text = '<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>'
    assert MK.read_tool_calls(text, "hermes", None, spec)[3] == "family:hermes"
    # the family reader finds nothing (strict about its own markers); the template's markers are there
    assert MK.read_tool_calls('<<act:f>>{"a": 1}', "hermes", None,
                              {"shape": "name+json", "call_open": "<<act:", "mid": ">>"})[3] == "template"


# ------------------------------------------------------------------ Granite

GRANITE_RAW = '<tool_call>[{"arguments": {"city": "Tokyo", "unit": "celsius"}, "name": "get_weather"}]'


@pytest.mark.parametrize("family", ["granite", "granite_tc"])
def test_granite_reads_both_spellings(family):
    for raw in (GRANITE_RAW, GRANITE_RAW.replace("<tool_call>", "<|tool_call|>")):
        calls, _, attempted = MK.parse_tool_calls(raw, family)
        assert attempted and calls == [{"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}}]
    # several calls in the one list
    two = '<tool_call>[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}]'
    assert [c["name"] for c in MK.parse_tool_calls(two, family)[0]] == ["a", "b"]


def test_granite_tc_is_its_own_family_and_not_hermes():
    assert MK.format_tool_call("granite_tc", "f", {"a": 1}) == '<tool_call>[{"name": "f", "arguments": {"a": 1}}]'
    assert MK._classify_call_text('<tool_call>[{"name": "pzq_lookup", "arguments": {}}]') == "granite_tc"
    assert MK._classify_call_text('<tool_call>\n{"name": "pzq_lookup", "arguments": {}}\n</tool_call>') == "hermes"
    # a Hermes block is not read as a Granite list
    assert MK.parse_tool_calls('<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>', "granite_tc")[0] == []
    meta = MK.TOOL_FAMILIES["granite_tc"]
    assert meta["vllm"] == "granite" and meta["shape"] == "json"


GRANITE31_TEMPLATE = (
    "{%- if tools %}{%- set sm = 'You are Granite. When a tool is required to answer the user\\'s query, respond with "
    "<|tool_call|> followed by a JSON list of tools used.' %}{%- else %}{%- set sm = 'You are Granite.' %}{%- endif %}"
    "{{- '<|start_of_role|>system<|end_of_role|>' + sm + '<|end_of_text|>\\n' }}"
    "{%- if tools %}{{- '<|start_of_role|>tools<|end_of_role|>' }}{{- tools | tojson(indent=4) }}"
    "{{- '<|end_of_text|>\\n' }}{%- endif %}"
    "{%- for message in messages %}{{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content']"
    " + '<|end_of_text|>\\n' }}{%- if loop.last and add_generation_prompt %}"
    "{{- '<|start_of_role|>assistant<|end_of_role|>' }}{%- endif %}{%- endfor %}")


@needs_jinja
def test_a_granite_3_1_checkpoint_is_granite_tc(tmp_path):
    d = tmp_path / "granite-3.1-1b-a400m-instruct"
    d.mkdir()
    (d / "chat_template.jinja").write_text(GRANITE31_TEMPLATE, encoding="utf-8")
    (d / "config.json").write_text("{}", encoding="utf-8")
    assert MK.toolcall_info(None, GRANITE31_TEMPLATE)["family"] == "granite"
    info = MK.toolcall_info(str(d))
    assert info["family"] == "granite_tc" and info["vllm_parser"] == "granite"


# ------------------------------------------------------------------ the SERVED path reads with the same reader

def _served(content, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg, "finish_reason": "stop"}]}


def test_a_served_reply_the_server_did_not_parse_is_read_by_pollard():
    reply = TC.pollard_read(_served(GRANITE_RAW), "granite", None, _tools("s01"))
    assert reply["read_by"] == "family:granite"
    row = TC.grade_reply(reply, PROBE["s01"])
    assert row["ok"] and row["parsed"] and row["read_by"] == "family:granite", row
    # unread before: the same reply without Pollard's reader is `unparsed`
    assert TC.grade_reply(_served(GRANITE_RAW), PROBE["s01"])["failure"] == "unparsed"


def test_a_served_reply_the_server_parsed_is_left_alone():
    tc = [{"id": "0", "type": "function", "function": {"name": "get_weather",
                                                       "arguments": json.dumps({"city": "Tokyo"})}}]
    reply = TC.pollard_read(_served("", tc), "granite", None, _tools("s01"))
    assert reply["read_by"] == "server"
    assert reply["choices"][0]["message"]["tool_calls"] == tc


def test_served_thinking_is_kept_out_and_prose_stays_prose():
    raw = "<think>the user wants weather</think>" + GRANITE_RAW
    reply = TC.pollard_read(_served(raw), "granite", None, _tools("s01"))
    row = TC.grade_reply(reply, PROBE["s01"])
    assert row["ok"], row
    reply = TC.pollard_read(_served("Hello! How can I help?"), "granite", None, _tools("a01")
                            if "a01" in PROBE else None)
    assert "read_by" not in reply and not reply["choices"][0]["message"].get("tool_calls")


def test_the_served_path_reads_unknown_syntax_through_the_template_reader():
    spec = {"shape": "name+json", "call_open": "<<act:", "mid": ">>"}
    raw = '<<act:get_weather>>{"city": "Tokyo", "unit": "celsius"}<</act>>'
    reply = TC.pollard_read(_served(raw), "unknown", spec, _tools("s01"))
    assert reply["read_by"] == "template" and TC.grade_reply(reply, PROBE["s01"])["ok"]


def test_run_suite_applies_the_reader_to_every_served_reply(monkeypatch):
    seen = []

    def fake_post(base, body, timeout=600):
        seen.append(body)
        return _served(GRANITE_RAW)
    monkeypatch.setattr(TC, "_post", fake_post)
    rows = TC.run_suite("http://x", [PROBE["s01"]], 64, family="granite")
    assert rows[0]["ok"] and rows[0]["read_by"] == "family:granite"
    assert TC._read_by(rows) == {"family:granite": 1}


def test_the_reference_cache_key_moves_with_the_reader():
    assert TC.READER_VERSION in json.dumps([TC.READER_VERSION])
    a = TC.suite_hash([PROBE["s01"]], "x")
    old = TC.READER_VERSION
    try:
        TC.READER_VERSION = "older"
        assert TC.suite_hash([PROBE["s01"]], "x") != a
    finally:
        TC.READER_VERSION = old
