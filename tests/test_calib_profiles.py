"""pollard-calib profiles: what a model is FOR decides its calibration mix, chat samples go through the
model's own template, tool traces keep their structure, self-generation swaps in the model's own answers,
and every corpus says which open-licensed, pinned datasets it came from. All offline: no datasets, no
transformers, no server."""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_calib as C  # noqa: E402
import pollard_modelkind as MK  # noqa: E402


def test_every_profile_names_only_real_domains_and_general_matches_calib30():
    for prof, weights in C.PROFILES.items():
        assert set(weights) <= set(C.DOMAINS), prof
    assert set(C.PROFILES["general"]) == {"prose", "code", "math", "chat", "multilingual"}
    assert C.domain_counts("translation", 100)["translation"] == 400
    assert C.domain_counts("general", 50, ["code"]) == {"code": 50}


def test_sources_are_open_licensed_and_pinned():
    bad_licences = {"cc-by-nc-4.0", "cc-by-nc-sa-4.0", "unknown", None, ""}
    for dom, srcs in C._HF.items():
        for s in srcs:
            assert s["license"] not in bad_licences, (dom, s["id"])
            assert len(s.get("revision") or "") == 40, f"{s['id']} is not pinned to a commit"
    ids = {s["id"] for srcs in C._HF.values() for s in srcs}
    assert "tatsu-lab/alpaca" not in ids, "alpaca is CC-BY-NC"
    # the bug this replaced: 'multilingual' fell back to English wikitext
    assert all("wikitext" not in s["id"] for s in C._HF["multilingual"])
    # never calibrate on the MT test set the retention table scores
    assert all(s["id"] != "google/wmt24pp" for s in C._HF["translation"])
    assert all(s["split"] != "devtest" for s in C._HF["translation"])


def test_multilingual_seeds_are_actually_not_english():
    non_latin = [s for s in C._SEED["multilingual"] if any(ord(c) > 0x0400 for c in s)]
    assert len(non_latin) >= 6
    langs = {line.split("\t")[0] for line in C._SEED_PAIRS}
    assert len(langs) >= 10


def test_chatml_renders_tools_and_calls():
    rec = C._seed_records("agent")[0]
    txt = C.render(rec, C.Renderer(None))
    assert "<tools>" in txt and "get_weather" in txt and "<tool_call>" in txt and "<tool_response>" in txt


class FakeTok:
    chat_template = "{{x}}"

    def __init__(self, tools_ok=True):
        self.tools_ok, self.calls = tools_ok, []

    def apply_chat_template(self, messages, tokenize=False, tools=None):
        self.calls.append((messages, tools))
        if not self.tools_ok and (tools or any(m.get("role") == "tool" or m.get("tool_calls") for m in messages)):
            raise ValueError("template has no tool role")
        return "|".join(f"{m['role']}:{m.get('content')}" for m in messages) + (f"|TOOLS{len(tools)}" if tools else "")


def test_renderer_uses_the_models_template_and_degrades_for_toolless_templates():
    r = C.Renderer(None)
    r.tok = FakeTok()
    rec = C._seed_records("agent")[0]
    out = r(rec["messages"], rec["tools"])
    assert out.endswith("|TOOLS3") and "tool:" in out
    # arguments arrive as a dict even when the source had them JSON-encoded
    assert isinstance(r.tok.calls[0][0][1]["tool_calls"][0]["function"]["arguments"], dict)
    r.tok = FakeTok(tools_ok=False)
    out = r(rec["messages"], rec["tools"])
    assert out.startswith("<|im_start|>system")       # last resort: generic ChatML, never a crash


def test_hermes_rows_become_structured_tool_calls():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    conv = [{"from": "system", "value": "You are a function calling AI. <tools>[]</tools>"},
            {"from": "human", "value": "call f with x=1"},
            {"from": "gpt", "value": "<tool_call>\n{'name': 'f', 'arguments': {'x': 1}}\n</tool_call>"},
            {"from": "tool", "value": "<tool_response>\n{\"ok\": true}\n</tool_response>"},
            {"from": "gpt", "value": "Done."}]
    rec = C._hermes_record(conv, json.dumps(tools))
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"], "Hermes' own system prompt must not leak in"
    assert rec["messages"][1]["tool_calls"][0]["function"] == {"name": "f", "arguments": {"x": 1}}
    assert rec["tools"] == tools
    assert C._hermes_record([{"from": "human", "value": "hi"}]) is None


def test_self_generate_replaces_only_the_last_answer_and_survives_failures():
    recs = ["raw prose stays",
            {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "ref"}]},
            {"messages": [{"role": "user", "content": "boom"}, {"role": "assistant", "content": "kept"}]}]
    seen = []

    def complete(msgs, tools=None):
        seen.append(msgs)
        if msgs[-1]["content"] == "boom":
            raise RuntimeError("server down")
        return {"role": "assistant", "content": "MODEL:" + msgs[-1]["content"]}
    out, ok, bad = C.self_generate(recs, complete)
    assert out[0] == "raw prose stays"
    assert out[1]["messages"][-1]["content"] == "MODEL:q1"
    assert out[2]["messages"][-1]["content"] == "kept"
    assert (ok, bad) == (1, 1)
    assert all(m["role"] != "assistant" for m in seen[0]), "the reference answer was shown to the model"


def test_auto_profile_follows_modelkind(tmp_path):
    agent = tmp_path / "Some-Agent-8B"
    agent.mkdir()
    (agent / "tokenizer_config.json").write_text(json.dumps(
        {"chat_template": "{% for tool in tools %}{{ tool }}{% endfor %} tool_call tool_call tool_response"}))
    assert C.resolve_profile("auto", str(agent))[0] == "agent"
    coder = tmp_path / "Qwen2.5-Coder-7B-Instruct"
    coder.mkdir()
    assert C.resolve_profile("auto", str(coder))[0] == "code"
    assert MK.calib_profile({}, "Hy-MT1.5-1.8B")[0] == "translation"
    assert MK.calib_profile({}, "tencent/HY-MT1.5-7B")[0] == "translation"
    assert MK.calib_profile({}, "Qwen3-8B")[0] == "general"
    assert C.resolve_profile("auto", None)[0] == "general"
    assert C.resolve_profile("code", None) == ("code", "requested")


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["pollard-calib", *args])
    C.main()


def test_profile_run_writes_sidecar_and_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(C, "_from_hf", lambda *args: [])
    out = tmp_path / "calib.txt"
    run(monkeypatch, "--out", str(out), "--profile", "agent", "--per-domain", "4",
        "--record-model", "Org/Agent-1B")
    text = out.read_text(encoding="utf-8")
    assert "<tool_call>" in text and "get_weather" in text
    side = json.loads((tmp_path / "calib.txt.sources.json").read_text())
    assert side["profile"] == "agent" and side["domains"]["agent"]["requested"] == 16
    assert side["domains"]["agent"]["source"]["id"] == "pollard-calib bundled seeds"
    import pollard_workspace as ws
    man = ws.read_manifest("Org/Agent-1B")
    assert man["calibration"][0]["profile"] == "agent"
    assert man["calibration"][0]["path"] == str(out.resolve())


def test_self_generate_needs_a_model_to_ask(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run(monkeypatch, "--out", str(tmp_path / "x"), "--self-generate")
    assert not (tmp_path / "x").exists()
