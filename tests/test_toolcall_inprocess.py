"""pollard-toolcall in-process: the EXL3 lane through exllamav3 and the GPTQ / AWQ / FP8 lanes through transformers (where
no usable vLLM is), read back by Pollard's own per-family call reader (pollard-modelkind parse_tool_calls) and graded by
the SAME grader as a served reply (grade_reply).

Every fixture is embedded here as canned raw model output (nix ships tests/*.py only). Nothing here loads a model or
needs torch / transformers / exllamav3: the engines are faked at their API boundary; the exllamav3 fake pins the calls
the real 1.4.x API takes (Config / Model / Cache / Tokenizer / Generator, ArgmaxSampler, stop_conditions)."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_modelkind as MK  # noqa: E402
import pollard_toolcall as TC  # noqa: E402
import pollard_workspace as ws  # noqa: E402

PROBE = {p["id"]: p for p in TC.SUITE}


def tools_for(pid):
    return [TC.TOOLBOX[t] for t in PROBE[pid]["tools"]]


def grade(raw, family, pid, finish="stop", reasoning=None, prefill=""):
    reply = TC.reply_from_text(raw, family, tools_for(pid), finish, reasoning, prefill)
    return TC.grade_reply(reply, PROBE[pid])


# ------------------------------------------------------------------ the reader inverts every family's writer

ARGS = {"city": "Paris, FR", "days": 3, "metric": True, "tags": ["a", "b"], "opts": {"x": 1.5}}
SCHEMA = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"},
    "tags": {"type": "array"}, "opts": {"type": "object"}}}}}]


@pytest.mark.parametrize("family", sorted(f for f in MK.TOOL_FAMILIES if MK.format_tool_call(f, "x", {})))
def test_every_written_family_reads_back(family):
    text = MK.format_tool_call(family, "get_weather", ARGS)
    calls, _, attempted = MK.parse_tool_calls(text, family, SCHEMA)
    assert attempted and calls == [{"name": "get_weather", "arguments": ARGS}], text


def test_every_family_with_a_writer_has_a_reader():
    written = {f for f in MK.TOOL_FAMILIES if MK.format_tool_call(f, "x", {})}
    assert written <= set(MK.CALL_READERS)


def test_a_family_without_a_reader_is_reported_not_guessed():
    with pytest.raises(KeyError):
        MK.parse_tool_calls("<|open|>call tool=\"x\"", "kimi_k3")
    with pytest.raises(TC.NotMeasured, match="no in-process reader for the 'kimi_k3'"):
        TC.reader_family({"tools": {"family": "kimi_k3"}})
    assert TC.reader_family({"tools": {"family": "none"}}) == ("none", "json")
    assert TC.reader_family({"tools": {"family": "hermes"}}) == ("hermes", "hermes")


# ------------------------------------------------------------------ canned raw outputs, one per family, graded end to end

# (family, probe, raw text as the model emits it, special tokens kept) -- every one is a correct call
CANNED = [
    ("hermes", "s01", '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}}\n'
                      '</tool_call>'),
    ("hermes", "s02", 'Sure, starting it now.\n<tool_call>\n{"name": "set_timer", "arguments": {"minutes": 25}}'),
    ("qwen3_coder", "s01", "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n"
                           "<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>"),
    ("qwen3_coder", "n02", "<tool_call>\n<function=update_settings>\n<parameter=settings>\n"
                           '{"theme": "dark", "notifications": false}\n</parameter>\n</function>\n</tool_call>'),
    ("seed_oss", "s02", "<seed:tool_call>\n<function=set_timer>\n<parameter=minutes>25</parameter>\n</function>\n"
                        "</seed:tool_call>"),
    ("mistral", "s01", '[TOOL_CALLS]get_weather[ARGS]{"city": "Tokyo", "unit": "celsius"}'),
    ("mistral", "s02", '[TOOL_CALLS][{"name": "set_timer", "arguments": {"minutes": 25}}]'),
    ("llama3_json", "s01", '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Tokyo", '
                           '"unit": "celsius"}}'),
    ("llama3_json", "s02", '{"name": "set_timer", "parameters": {"minutes": 25}}'),
    ("lfm2", "s01", '<|tool_call_start|>[get_weather(city="Tokyo", unit="celsius")]<|tool_call_end|>'),
    ("lfm2", "n02", '<|tool_call_start|>[update_settings(settings={"theme": "dark", "notifications": false})]'
                    '<|tool_call_end|>Done.'),
    ("pythonic", "s02", "[set_timer(minutes=25)]"),
    ("gemma4", "s01", '<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>,unit:<|"|>celsius<|"|>}<tool_call|>'),
    ("functiongemma", "s02", "<start_function_call>call:set_timer{minutes:25}<end_function_call>"),
    ("harmony", "s01", "<|channel|>analysis<|message|>The user wants Tokyo weather in celsius.<|end|>"
                       "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>"
                       '{"city": "Tokyo", "unit": "celsius"}<|call|>'),
    ("glm45", "s01", "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Tokyo</arg_value>\n"
                     "<arg_key>unit</arg_key>\n<arg_value>celsius</arg_value>\n</tool_call>"),
    ("glm47", "s02", "<tool_call>set_timer<arg_key>minutes</arg_key><arg_value>25</arg_value></tool_call>"),
    ("kimi_k2", "s02", "<|tool_calls_section_begin|><|tool_call_begin|>functions.set_timer:0"
                       '<|tool_call_argument_begin|>{"minutes": 25}<|tool_call_end|><|tool_calls_section_end|>'),
    ("deepseek_v31", "s02", "<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>"
                            'set_timer<\uff5ctool\u2581sep\uff5c>{"minutes": 25}<\uff5ctool\u2581call\u2581end\uff5c>'
                            "<\uff5ctool\u2581calls\u2581end\uff5c>"),
    ("granite", "s02", '<|tool_call|>[{"name": "set_timer", "arguments": {"minutes": 25}}]'),
    ("minimax_m2", "s01", '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="city">Tokyo</parameter>'
                          '\n<parameter name="unit">celsius</parameter>\n</invoke>\n</minimax:tool_call>'),
    ("cohere", "s02", '<|START_ACTION|>[{"tool_call_id": "0", "tool_name": "set_timer", "parameters": '
                      '{"minutes": 25}}]<|END_ACTION|>'),
    ("nemotron_v2", "s02", '<TOOLCALL>[{"name": "set_timer", "arguments": {"minutes": 25}}]</TOOLCALL>'),
    ("xlam", "s02", '[{"name": "set_timer", "arguments": {"minutes": 25}}]'),
    ("json", "s02", 'Calling: {"name": "set_timer", "arguments": {"minutes": 25}}'),
]


@pytest.mark.parametrize("family, pid, raw", CANNED, ids=[f"{c[0]}-{c[1]}" for c in CANNED])
def test_canned_raw_output_grades_as_a_correct_call(family, pid, raw):
    row = grade(raw, family, pid)
    assert row["ok"] and row["parsed"] and row["exact"], row


def test_tagged_arguments_are_typed_by_the_schema():
    """vLLM's tagged readers type a value by the tool schema: an integer argument is 25, a string one stays text."""
    raw = "<tool_call>\n<function=set_timer>\n<parameter=minutes>\n25\n</parameter>\n<parameter=label>\n90\n" \
          "</parameter>\n</function>\n</tool_call>"
    calls, _, _ = MK.parse_tool_calls(raw, "qwen3_coder", tools_for("s11"))
    assert calls[0]["arguments"] == {"minutes": 25, "label": "90"}


# ------------------------------------------------------------------ the failures the served path reports, in-process too

def test_call_shaped_text_in_another_syntax_is_unparsed_not_prose():
    # a qwen3_coder model writing Hermes JSON: its server's parser would reject it, so the reader does too
    row = grade('<tool_call>\n{"name": "set_timer", "arguments": {"minutes": 25}}\n</tool_call>', "qwen3_coder", "s02")
    assert row["failure"] == "unparsed"
    assert grade('<tool_call>\n{"name": "set_timer", "arguments": {"minutes": 25\n</tool_call>', "hermes",
                 "s02")["failure"] == "unparsed"                              # broken JSON


def test_prose_is_no_call_and_a_call_where_none_was_needed_is_false_call():
    assert grade("You can set a timer on your phone.", "hermes", "s02")["failure"] == "no-call"
    assert grade("The capital of France is Paris.", "hermes", "a01")["ok"]
    row = grade('<tool_call>\n{"name": "search_web", "arguments": {"query": "capital of France"}}\n</tool_call>',
                "hermes", "a01")
    assert row["failure"] == "false-call"


def test_wrong_value_and_wrong_tool_survive_the_round_trip():
    assert grade('<tool_call>{"name": "set_timer", "arguments": {"minutes": 30}}</tool_call>', "hermes",
                 "s02")["failure"] == "wrong-value"
    row = grade('<tool_call>{"name": "send_email", "arguments": {"to": "x"}}</tool_call>', "hermes", "c01")
    assert row["failure"] == "wrong-tool"


QWEN3_REASONING = {"open": "<think>", "close": "</think>", "style": "tag"}


def test_thinking_is_stripped_and_a_call_sketched_while_thinking_is_not_the_call():
    raw = ('<think>\nMaybe <tool_call>{"name": "calculate"}</tool_call>? No -- a timer.\n</think>\n\n'
           '<tool_call>\n{"name": "set_timer", "arguments": {"minutes": 25}}\n</tool_call>')
    row = grade(raw, "hermes", "s02", reasoning=QWEN3_REASONING)
    assert row["ok"], row
    reply = TC.reply_from_text(raw, "hermes", tools_for("s02"), "stop", QWEN3_REASONING)
    assert "a timer" in reply["choices"][0]["message"]["reasoning_content"]


def test_unclosed_thinking_is_a_budget_failure():
    row = grade("<think>\nThe user wants a timer. Let me consider the units", "hermes", "s02", finish="length",
                reasoning=QWEN3_REASONING)
    assert row["failure"] == "unclosed-think"


def test_a_template_that_opens_the_thinking_is_split_as_a_continuation():
    prompt = "<|im_start|>assistant\n<think>\n"
    pre = TC._prefill_for(prompt, QWEN3_REASONING)
    assert pre == "<think>"
    raw = 'A timer.\n</think>\n\n<tool_call>\n{"name": "set_timer", "arguments": {"minutes": 25}}\n</tool_call>'
    assert grade(raw, "hermes", "s02", reasoning=QWEN3_REASONING, prefill=pre)["ok"]
    assert grade("still thinking about it", "hermes", "s02", finish="length", reasoning=QWEN3_REASONING,
                 prefill=pre)["failure"] == "unclosed-think"
    assert TC._prefill_for("<|im_start|>assistant\n", QWEN3_REASONING) == ""


# ------------------------------------------------------------------ ONE grader: served and in-process agree

@pytest.mark.parametrize("raw, pid", [
    ('<tool_call>\n{"name": "set_timer", "arguments": {"minutes": 25}}\n</tool_call>', "s02"),
    ('<tool_call>\n{"name": "set_timer", "arguments": {"minutes": "25"}}\n</tool_call>', "s02"),
    ("The capital of France is Paris.", "a01"),
    ('<tool_call>\n{"name": "lookup_contact", "arguments": {"name": "Priya Raman", "x": 1}}\n</tool_call>', "s04"),
])
def test_a_served_reply_and_the_same_text_in_process_grade_identically(raw, pid):
    """llama-server --jinja returns the parsed call in tool_calls; in-process the reader builds the same reply."""
    calls, content, _ = MK.parse_tool_calls(raw, "hermes", tools_for(pid))
    msg = {"role": "assistant", "content": content if calls else raw}
    if calls:
        msg["tool_calls"] = [{"id": "s", "type": "function", "function": {"name": c["name"],
                                                                          "arguments": json.dumps(c["arguments"])}}
                             for c in calls]
    served = {"choices": [{"message": msg, "finish_reason": "tool_calls" if calls else "stop"}]}
    a, b = TC.grade_reply(served, PROBE[pid]), grade(raw, "hermes", pid)
    keys = ("ok", "failure", "parsed", "name_ok", "args_valid", "exact", "called")
    assert {k: a[k] for k in keys} == {k: b[k] for k in keys}


def test_the_grader_flags_every_familys_markers_as_call_shaped():
    for fam in ("hermes", "qwen3_coder", "mistral", "lfm2", "gemma4", "kimi_k2", "granite", "harmony", "glm45",
                "functiongemma", "cohere", "minimax_m2", "seed_oss", "deepseek_v31", "nemotron_v2"):
        text = MK.format_tool_call(fam, "set_timer", {"minutes": 25}) or \
            '<|start|>assistant to=functions.set_timer<|channel|>commentary json<|message|>{"minutes": 25}<|call|>'
        assert TC._call_shaped(text), fam
    assert not TC._call_shaped("The capital of France is Paris.")


# ------------------------------------------------------------------ the in-process suite, with a fake engine

class FakeEngine:
    """Renders nothing real; answers every probe with the canned Hermes call for its expected tool."""

    def __init__(self):
        self.prompts = []

    def render(self, messages, tools, template_kwargs=None):
        self.prompts.append((messages, [t["function"]["name"] for t in tools], template_kwargs))
        return "PROMPT:" + messages[-1]["content"]

    def generate(self, prompt, max_tokens):
        p = next(q for q in TC.SUITE if prompt == "PROMPT:" + q["prompt"])
        if p["expect"] is None:
            return "Sure -- no tool needed.", "stop"
        return MK.format_tool_call("hermes", p["expect"]["name"], {}), "stop"


def test_run_suite_inprocess_sends_the_same_request_as_the_served_path():
    eng = FakeEngine()
    probes = TC._balanced(2)
    rows = TC.run_suite_inprocess(eng, probes, 64, {"enable_thinking": False}, family="hermes")
    assert len(rows) == len(probes) and [r["id"] for r in rows] == [p["id"] for p in probes]
    body = TC.request_for(probes[0], 64)
    msgs, names, tkw = eng.prompts[0]
    assert msgs == body["messages"] and names == [t["function"]["name"] for t in body["tools"]]
    assert tkw == {"enable_thinking": False}
    # the canned calls carry no arguments: the abstentions pass, the calls fail on their (missing) arguments
    assert all(r["ok"] for r in rows if r["category"] == "abstain")
    assert all(r["parsed"] and r["name_ok"] for r in rows if r["expected"])


def test_an_engine_error_on_one_probe_is_recorded_not_fatal():
    class Boom(FakeEngine):
        def generate(self, prompt, max_tokens):
            raise RuntimeError("CUDA out of memory")
    rows = TC.run_suite_inprocess(Boom(), TC._balanced(1), 32, family="hermes")
    assert all(r["failure"] == "server-error" and "out of memory" in r["detail"] for r in rows)


def test_run_model_runs_an_in_process_backend(monkeypatch, tmp_path):
    d = tmp_path / "M-EXL3-4.0bpw"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "exl3"}}))
    monkeypatch.setattr(MK, "classify", lambda m: {"tools": {"family": "hermes"}, "thinking": False})
    monkeypatch.setattr(TC, "preflight", lambda *a, **k: {"device": "cuda"})
    seen = {}

    import contextlib

    @contextlib.contextmanager
    def engine(backend, model, device="cuda", threads=None, ctx=8192):
        seen.update(backend=backend, device=device)
        yield FakeEngine()
    monkeypatch.setattr(TC, "inprocess_engine", engine)
    rows, meta = TC.run_model(str(d), "0", backend="exl3", probes=TC._balanced(1), quiet=True)
    assert seen == {"backend": "exl3", "device": "cuda"}
    assert meta["backend"] == "exl3" and meta["reader"] == "hermes" and len(rows) == 4


def test_a_vllm_rung_without_a_usable_vllm_is_measured_in_process(monkeypatch, tmp_path):
    d = tmp_path / "M-GPTQ"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "gptq"}}))
    monkeypatch.setattr(MK, "classify", lambda m: {"tools": {"family": "hermes"}})
    monkeypatch.setattr(TC, "vllm_status", lambda binary=None: (False, "the vLLM installed here has no compiled CUDA "
                                                                       "kernels"))
    monkeypatch.setattr(TC, "preflight", lambda backend, *a, **k: {"device": "cuda"} if backend == "hf" else
                        pytest.fail("vLLM must not be tried"))
    import contextlib

    @contextlib.contextmanager
    def engine(backend, model, device="cuda", threads=None, ctx=8192):
        assert backend == "hf"
        yield FakeEngine()
    monkeypatch.setattr(TC, "inprocess_engine", engine)
    rows, meta = TC.run_model(str(d), "0", backend="vllm", probes=TC._balanced(1), quiet=True)
    assert meta["backend"] == "hf" and "compiled CUDA kernels" in meta["backend_note"]


def test_two_in_process_backends_are_not_cross_backend():
    assert TC.cross_backend("exl3", "hf") is False and TC.cross_backend("hf", "hf") is False
    assert TC.cross_backend("exl3", "gguf") is True and TC.cross_backend("vllm", "hf") is True
    assert TC.cross_backend("gguf", None) is False


# ------------------------------------------------------------------ preflight: what this machine can run in-process

def _qdir(tmp_path, meth):
    d = tmp_path / f"m-{meth or 'bf16'}"
    d.mkdir()
    cfg = {"architectures": ["Qwen2ForCausalLM"]}
    if meth:
        cfg["quantization_config"] = {"quant_method": meth}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "model.safetensors").write_bytes(b"\0" * 1000)
    return str(d)


HERMES = {"tools": {"family": "hermes"}}


def test_hf_preflight_per_quant_format(monkeypatch, tmp_path):
    have = {"transformers", "torch", "gptqmodel"}
    monkeypatch.setattr(TC, "_has_module", lambda n: n in have)
    monkeypatch.setattr(TC, "gpu_users", lambda *a: [])
    monkeypatch.setattr(TC, "gpu_free_gb", lambda *a: 15.0)
    assert TC.preflight("hf", _qdir(tmp_path, "gptq"), kind=HERMES) == {"device": "cuda"}
    with pytest.raises(TC.NotMeasured, match="needs compressed_tensors"):
        TC.preflight("hf", _qdir(tmp_path, "compressed-tensors"), kind=HERMES)
    with pytest.raises(TC.NotMeasured, match="ModelOpt"):
        TC.preflight("hf", _qdir(tmp_path, "modelopt"), kind=HERMES)
    with pytest.raises(TC.NotMeasured, match="no transformers loader"):
        TC.preflight("hf", _qdir(tmp_path, "nvfp4"), kind=HERMES)
    assert TC.preflight("hf", _qdir(tmp_path, "mxfp4"), kind=HERMES) == {"device": "cuda"}
    # a game on the GPU: a quantized rung is not measured, a bf16 reference stays off the GPU
    monkeypatch.setattr(TC, "gpu_users", lambda *a: ["Game.exe"])
    with pytest.raises(TC.NotMeasured, match="resident"):
        TC.preflight("hf", _qdir(tmp_path, "fp8"), kind=HERMES)
    monkeypatch.setattr(TC, "_apple", lambda: False)
    monkeypatch.setattr(TC, "_free_ram_gb", lambda: 32.0)
    assert TC.preflight("hf", _qdir(tmp_path, ""), kind=HERMES) == {"device": "cpu"}
    # no NVIDIA GPU (a Mac): bf16 on mps, a GPTQ rung not measured
    monkeypatch.setattr(TC, "gpu_free_gb", lambda *a: None)
    monkeypatch.setattr(TC, "_apple", lambda: True)
    assert TC.preflight("hf", str(tmp_path / "m-bf16"), kind=HERMES) == {"device": "mps"}
    with pytest.raises(TC.NotMeasured, match="NVIDIA GPU"):
        TC.preflight("hf", str(tmp_path / "m-gptq"), kind=HERMES)


def test_exl3_preflight(monkeypatch, tmp_path):
    d = _qdir(tmp_path, "exl3")
    monkeypatch.setattr(TC, "gpu_free_gb", lambda *a: None)
    with pytest.raises(TC.NotMeasured, match="needs an NVIDIA GPU"):
        TC.preflight("exl3", d, kind=HERMES)
    monkeypatch.setattr(TC, "gpu_free_gb", lambda *a: 15.0)
    monkeypatch.setattr(TC, "_has_module", lambda n: n in ("exllamav3", "jinja2"))
    monkeypatch.setattr(TC, "gpu_users", lambda *a: ["Game.exe"])
    with pytest.raises(TC.NotMeasured, match="resident"):
        TC.preflight("exl3", d, kind=HERMES)
    monkeypatch.setattr(TC, "gpu_users", lambda *a: [])
    assert TC.preflight("exl3", d, kind=HERMES) == {"device": "cuda"}


# ------------------------------------------------------------------ the exllamav3 engine against its 1.4.x API (faked)

def _fake_exllamav3(monkeypatch, eos_reason="stop_token", text='<tool_call>\n{"name": "set_timer", '
                                                                 '"arguments": {"minutes": 25}}\n</tool_call>'):
    calls = {}
    m = types.ModuleType("exllamav3")

    class Config:
        @staticmethod
        def from_directory(d):
            c = Config()
            c.directory, c.eos_token_id_list = d, [151645, 151643]
            calls["config"] = d
            return c

    class Model:
        @staticmethod
        def from_config(cfg):
            return Model()

        def load(self, **kw):
            calls["loaded_with_cache"] = "cache" in calls

        def unload(self):
            calls["unloaded"] = True

    class Cache:
        def __init__(self, model, max_num_tokens, **kw):
            calls["cache"] = max_num_tokens

    class Tokenizer:
        eos_token_id, eos_token = 151645, "<|im_end|>"

        @staticmethod
        def from_config(cfg):
            return Tokenizer()

    class Generator:
        def __init__(self, model, cache, tokenizer, **kw):
            pass

        def generate(self, **kw):
            calls["generate"] = kw
            return text, {"eos_reason": eos_reason}

    class ArgmaxSampler:
        pass
    m.Config, m.Model, m.Cache, m.Tokenizer, m.Generator = Config, Model, Cache, Tokenizer, Generator
    gen = types.ModuleType("exllamav3.generator")
    smp = types.ModuleType("exllamav3.generator.sampler")
    smp.ArgmaxSampler = ArgmaxSampler
    monkeypatch.setitem(sys.modules, "exllamav3", m)
    monkeypatch.setitem(sys.modules, "exllamav3.generator", gen)
    monkeypatch.setitem(sys.modules, "exllamav3.generator.sampler", smp)
    monkeypatch.setattr(TC, "_Renderer", lambda model, tokenizer=None: (lambda msgs, tools, tkw=None: "PROMPT"))
    return calls


def test_exl3_engine_drives_the_exllamav3_api(monkeypatch):
    calls = _fake_exllamav3(monkeypatch)
    with TC.inprocess_engine("exl3", "/m/M-EXL3", ctx=5000) as eng:
        text, finish = eng.generate("PROMPT", 384)
    assert calls["config"] == "/m/M-EXL3" and calls["cache"] == 5120            # a multiple of 256
    assert calls["loaded_with_cache"] and calls["unloaded"]                       # Cache before load; unloaded after
    g = calls["generate"]
    assert g["stop_conditions"] == [151645, 151643] and g["decode_special_tokens"] and g["encode_special_tokens"]
    assert g["add_bos"] is False and g["completion_only"] and g["max_new_tokens"] == 384
    assert type(g["sampler"]).__name__ == "ArgmaxSampler"
    assert finish == "stop" and text.startswith("<tool_call>")
    row = TC.grade_reply(TC.reply_from_text(text, "hermes", tools_for("s02"), finish), PROBE["s02"])
    assert row["ok"]


def test_a_model_that_will_not_load_is_not_measured(monkeypatch):
    calls = _fake_exllamav3(monkeypatch)

    def boom(d):
        raise RuntimeError("no kernel image is available for execution on the device\nmore")
    monkeypatch.setattr(sys.modules["exllamav3"].Config, "from_directory", staticmethod(boom))
    with pytest.raises(TC.NotMeasured, match="could not load M-EXL3: RuntimeError: no kernel image"):
        with TC.inprocess_engine("exl3", "/m/M-EXL3"):
            pass
    assert "generate" not in calls


def test_exl3_engine_reports_a_budget_cut(monkeypatch):
    _fake_exllamav3(monkeypatch, eos_reason="max_new_tokens", text="<think>\nhmm")
    with TC.inprocess_engine("exl3", "/m/M-EXL3") as eng:
        assert eng.generate("PROMPT", 8)[1] == "length"


# ------------------------------------------------------------------ the renderer: the model's own template + tools

TPL = ("{%- if tools %}<|im_start|>system\n{{ messages[0]['content'] }}\n<tools>\n{%- for t in tools %}\n"
       "{{ t | tojson }}{%- endfor %}\n</tools><|im_end|>\n{%- endif %}"
       "{%- for m in messages[1:] %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
       "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}")


def test_renderer_falls_back_to_the_jinja2_template_without_transformers(monkeypatch, tmp_path):
    pytest.importorskip("jinja2")
    d = tmp_path / "M"
    d.mkdir()
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": TPL}))
    monkeypatch.setattr(TC, "_has_module", lambda n: n != "transformers")
    r = TC._Renderer(str(d))
    body = TC.request_for(PROBE["c01"], 64)
    out = r(body["messages"], body["tools"])
    assert "<tools>" in out and '"get_directions"' in out and out.endswith("<|im_start|>assistant")
    assert PROBE["c01"]["prompt"] in out


def test_renderer_folds_the_system_turn_for_a_template_that_refuses_one(monkeypatch, tmp_path):
    pytest.importorskip("jinja2")
    d = tmp_path / "M"
    d.mkdir()
    tpl = ("{%- if messages[0].role == 'system' %}{{ raise_exception('System role not supported') }}{%- endif %}"
           "{%- for m in messages %}<start_of_turn>{{ m.role }}\n{{ m.content }}<end_of_turn>\n{%- endfor %}")
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": tpl}))
    monkeypatch.setattr(TC, "_has_module", lambda n: n != "transformers")
    out = TC._Renderer(str(d))(TC.request_for(PROBE["s02"], 8)["messages"], [])
    assert TC.SYSTEM in out and PROBE["s02"]["prompt"] in out


@pytest.fixture(autouse=True)
def _no_workspace(monkeypatch):
    monkeypatch.setattr(ws, "find_build", lambda p: (None, None))
