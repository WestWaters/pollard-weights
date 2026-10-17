#!/usr/bin/env python3
"""pollard-toolcall -- can this build still call tools the way it was trained to?

Standard benchmarks report 4-bit as near-lossless, and that verdict rests almost entirely on
single-turn scoring. Multi-turn tool use is where the same error amplifies, and the capability
that breaks first is narrow: emitting a syntactically correct call with the right name and the
right argument types. It is a formatting skill concentrated in a small band of layers, which makes
it unusually fragile to quantization and unusually cheap to protect once you know it went.

The full gate (default) runs the REAL serving path, on every lane: the rung is asked with OpenAI-style `tools`, so the
model's own chat template renders the tool list, the model answers in its own call syntax (Hermes, Qwen3-Coder XML,
LFM2 pythonic, harmony, Mistral... -- pollard-modelkind names it) and the call is read back the way that family's
server parser reads it:

    served (the server's own parser reads the call):
      gguf  llama-server --jinja                                   GGUF
      mlx   mlx_lm.server                                          MLX
      vllm  vllm serve --enable-auto-tool-choice --tool-call-parser <family>
                                                                   GPTQ / AWQ / FP8 / MXFP4 / NVFP4 (a CUDA vLLM)
      url   an OpenAI-compatible server already running            TabbyAPI, a vLLM elsewhere, ...
    in this process (the model's HF chat template renders the tools, greedy decoding, and the raw text is read by
    Pollard's reader for the family -- pollard-modelkind parse_tool_calls -- with thinking stripped):
      exl3  exllamav3                                              EXL3
      hf    transformers                                           GPTQ / AWQ / FP8 / compressed-tensors where no
                                                                   usable vLLM is (Windows, not installed); the bf16
                                                                   reference of an EXL3 / in-process rung; an
                                                                   EXLR8 rung (its reference decode, pollard-exlr8)

Every path ends in the same grader (grade_reply): a served reply and an in-process one are graded by one function.
--backend auto (default) picks the backend from the file, the workspace MANIFEST lane or config.json; a GPTQ / MX
rung goes to vLLM when a vLLM with compiled CUDA kernels is installed and to the in-process transformers backend
otherwise. Sixty probes in four categories:

    single    one tool on offer, typed arguments (string / integer / number / boolean / enum)
    choose    eight tools on offer: pick the right one and fill it
    abstain   tools on offer, none needed: the right answer is prose
    nested    several arguments, nested objects and arrays

scored as parse rate, right function, arguments schema-valid, arguments exact and abstention, with
Wilson CIs. Against the reference the gate number is RETENTION (rung success / reference success) with a paired
bootstrap CI; the reference's run is cached, so a ladder runs it once. The reference is run the rung's OWN way where it
can be (the bf16 checkout through mlx_lm.server for an MLX rung, through vLLM for a vLLM-served rung, in-process
through transformers for an EXL3 or in-process rung -- same template rendering, same reader); when it cannot (no GPU
free, does not fit), --ref-fallback (the f16 GGUF through llama-server) stands in and the result says cross-backend.
A reasoning model's thinking is kept out of the grade (pollard-modelkind split_reasoning).

A backend this machine cannot run -- vLLM / exllamav3 without an NVIDIA GPU, a GPU with another process resident (a
shared box's GPU is not ours to take), a model that does not fit the free memory, a quantization format nothing here
loads -- is recorded as {"status": "not-measured", "reason": ...} and exits 0: the build goes on, and the card says
"not measured (reason)". llama-server runs with -ngl 0 unless --ngl says otherwise.

    pollard-toolcall --gguf build.gguf --ref f16.gguf                     # GGUF, retention vs f16
    pollard-toolcall --model ./M-Pollard-GPTQ-int4 --ref ./M --ref-fallback M-f16.gguf   # vLLM, else in-process
    pollard-toolcall --model ./M-Pollard-EXL3-4.0bpw --ref ./M            # exllamav3 in-process, vs bf16 in-process
    pollard-toolcall --backend url --url http://127.0.0.1:5000 --rung my-rung            # TabbyAPI / any server
    pollard-toolcall --model ./M-Pollard-MLX-4bit --ref ./M               # MLX, vs the bf16 through mlx_lm.server
    pollard-toolcall --gguf build.gguf --ref f16.gguf --quick             # the old 4-probe plain-prompt check

What --quick counts as a failure, in the order they actually happen:

    unclosed-think a reasoning model was still thinking when the budget ran out
    no-call        the model answered in prose instead of calling anything
    not-json       it emitted something call-shaped that does not parse
    wrong-tool     valid JSON, wrong function name
    missing-arg    a required argument absent
    bad-type       an argument present with the wrong type
    hallucinated   an argument the schema does not define

The full gate adds: unparsed (call-shaped text the parser -- the server's, or the in-process reader --
rejected), bad-enum, bad-args, hallucinated-tool, wrong-value and false-call (a tool called where none was needed).

A drop in tool-call success between the reference and the build is a direct measurement of the
fragile band, and pollard-sensitivity can then say which layers carry it.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ---- --quick: the original 4-probe plain-prompt check

# Deliberately ordinary tools. The point is whether FORMAT survives quantization, not whether the
# model is clever, so an exotic schema would confound the two.
PROBES = [
    {"prompt": "What is the weather in Paris? Use the tool.",
     "tool": {"name": "get_weather",
              "parameters": {"location": {"type": "string", "required": True},
                             "unit": {"type": "string", "required": False}}},
     "expect": {"location": "paris"}},
    {"prompt": "Add 17 and 25 using the calculator tool.",
     "tool": {"name": "calculate",
              "parameters": {"a": {"type": "number", "required": True},
                             "b": {"type": "number", "required": True},
                             "op": {"type": "string", "required": False}}},
     "expect": {"a": 17, "b": 25}},
    {"prompt": "Search the web for the tallest mountain. Use the tool.",
     "tool": {"name": "web_search",
              "parameters": {"query": {"type": "string", "required": True},
                             "max_results": {"type": "number", "required": False}}},
     "expect": {}},
    {"prompt": "Send an email to sam@example.com saying the build finished.",
     "tool": {"name": "send_email",
              "parameters": {"to": {"type": "string", "required": True},
                             "subject": {"type": "string", "required": False},
                             "body": {"type": "string", "required": True}}},
     "expect": {"to": "sam@example.com"}},
]

_TYPES = {"string": str, "number": (int, float), "boolean": bool,
          "array": list, "object": dict}


def extract_call(text):
    """Find a JSON object in a generation, however the model chose to wrap it.

    Models emit calls inside fences, inside <tool_call> tags, or bare. None of that is the
    capability under test, so all of it is accepted -- what is under test is whether the object
    itself is well formed.
    """
    t = (text or "").strip()
    if not t:
        return None
    for pat in (r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
                r"```(?:json)?\s*(\{.*?\})\s*```",
                r"(\{(?:[^{}]|\{[^{}]*\})*\})"):
        for m in re.finditer(pat, t, re.S):
            try:
                obj = json.loads(m.group(1))
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict):
                return obj
    return None


#: generation budget: a reasoning model thinks before it calls, and a call cut off mid-thought is a
#: budget failure, not a format failure
N_PREDICT_PLAIN, N_PREDICT_REASONING = 160, 1024


def _split_think(text):
    """(thinking, answer, unclosed) with any model's delimiters (pollard-modelkind)."""
    try:
        from pollard_modelkind import split_reasoning
        return split_reasoning(text)
    except ImportError:                                          # pragma: no cover
        return "", text or "", False


def _call_shaped(text):
    """Call-shaped text in any family's syntax (pollard-modelkind call_shaped) -- one rule for both paths."""
    try:
        from pollard_modelkind import call_shaped
    except ImportError:                                          # pragma: no cover
        return bool(re.search(r"<tool_call>|\[TOOL_CALLS\]|<\|tool_call|\"name\"\s*:|<function=", text or ""))
    return call_shaped(text)


def check_call(text, tool, expect=None):
    """Grade one generation against one tool schema. Returns (ok, failure, detail).

    A reasoning model's thinking is removed first: JSON it sketches while thinking is not the call, and a
    thinking block that never closed means the budget ran out before any call was made."""
    think, answer, unclosed = _split_think(text)
    if unclosed:
        return False, "unclosed-think", "thinking block never closed within the budget -- no call emitted"
    text = answer if think else text
    looks_like = bool(re.search(r"[{}]|tool_call|function", text or "", re.I))
    obj = extract_call(text)
    if obj is None:
        return (False, "not-json", "call-shaped output that does not parse") if looks_like \
            else (False, "no-call", "answered in prose without calling anything")

    # name and arguments live under several spellings; the wrapper is not the capability
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    if isinstance(name, dict):
        name = name.get("name")
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters") or obj.get("args") or obj.get("input")
    if isinstance(args, str):
        try:
            args = json.loads(args)          # some models double-encode the arguments
        except ValueError:
            return False, "not-json", "arguments are a string that does not parse as JSON"
    if not isinstance(args, dict):
        args = {k: v for k, v in obj.items()
                if k not in ("name", "tool", "function", "arguments", "parameters")}

    if name != tool["name"]:
        return False, "wrong-tool", f"called {name!r}, schema is {tool['name']!r}"

    spec = tool["parameters"]
    for key, meta in spec.items():
        if meta.get("required") and key not in args:
            return False, "missing-arg", f"required argument {key!r} absent"
    for key, val in args.items():
        if key not in spec:
            return False, "hallucinated", f"argument {key!r} is not in the schema"
        want = _TYPES.get(spec[key]["type"])
        if want and not isinstance(val, want):
            # a number arriving as a numeric string is a formatting slip, not a wrong answer,
            # but it still breaks a strict caller, so it is reported as its own failure
            if spec[key]["type"] == "number" and isinstance(val, str) and val.strip().lstrip("-").replace(".", "", 1).isdigit():
                return False, "bad-type", f"{key!r} is the string {val!r}, schema says number"
            return False, "bad-type", f"{key!r} is {type(val).__name__}, schema says {spec[key]['type']}"

    for key, want in (expect or {}).items():
        got = args.get(key)
        if isinstance(want, str) and isinstance(got, str):
            if want.lower() not in got.lower():
                return False, "wrong-value", f"{key!r} is {got!r}, expected something like {want!r}"
        elif got != want:
            return False, "wrong-value", f"{key!r} is {got!r}, expected {want!r}"
    return True, None, "valid call"


def score(results):
    """Format validity, and which failure dominates."""
    n = len(results)
    ok = sum(1 for r in results if r["ok"])
    modes = {}
    for r in results:
        if not r["ok"]:
            modes[r["failure"]] = modes.get(r["failure"], 0) + 1
    return {"n": n, "valid": ok, "rate": round(ok / n, 4) if n else 0.0, "modes": modes}


def _prompt_for(probe):
    """A schema the model can see. Kept plain so no chat template is required."""
    return (f"You have one tool available:\n{json.dumps(probe['tool'], indent=2)}\n\n"
            f"User: {probe['prompt']}\n\n"
            f"Reply with ONLY a JSON object of the form "
            f'{{"name": "...", "arguments": {{...}}}}.\n')


def generate(cli_bin, model, prompt, ngl, n_predict=160, threads=None):
    cmd = [cli_bin, "-m", model, "-ngl", str(ngl), "-c", "2048", "-n", str(n_predict),
           "-p", prompt, "--temp", "0", "-no-cnv", "-st", "--simple-io",
           "--no-display-prompt", "--log-disable"]
    if threads:
        cmd += ["-t", str(threads)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                           stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return ""
    return (r.stdout or "").strip()


def n_predict_for(model):
    """The generation budget for this model: room to think first when it is a reasoning model."""
    try:
        from pollard_modelkind import classify
        return N_PREDICT_REASONING if classify(model).get("thinking") else N_PREDICT_PLAIN
    except Exception:                                            # noqa: BLE001
        return N_PREDICT_PLAIN


def run(cli_bin, model, ngl, threads=None, n_predict=0):
    out = []
    n_predict = n_predict or n_predict_for(model)
    for probe in PROBES:
        gen = generate(cli_bin, model, _prompt_for(probe), ngl, n_predict=n_predict, threads=threads)
        ok, failure, detail = check_call(gen, probe["tool"], probe.get("expect"))
        out.append({"tool": probe["tool"]["name"], "ok": ok, "failure": failure,
                    "detail": detail, "sample": gen.replace("\n", " ")[:110]})
    return out


# =====================================================================================================================
# The full gate: the REAL serving path.
#
# The quick mode above asks for a bare JSON object in a plain prompt. That tests whether the model can write JSON,
# not whether it can call a tool the way it was trained to and the way a user's harness will ask it to. The full gate
# serves the rung with `llama-server --jinja` and sends OpenAI-style `tools` to /v1/chat/completions, so:
#   - the model's OWN chat template renders the tool list (Hermes <tools>, LFM2 <|tool_list_start|>, Llama 3 user turn),
#   - the model answers in its OWN call syntax (pollard-modelkind toolcall_info names it),
#   - llama.cpp's own parser (and its lazy grammar, armed at the call trigger) extracts the call.
# What comes back in `tool_calls` is what an agent harness would receive. Sixty probes in four categories:
#   single   one tool on offer, typed arguments (string / integer / number / boolean / enum)
#   choose   eight tools on offer, pick the right one and fill it
#   abstain  tools on offer, none needed: the right answer is prose
#   nested   multi-argument calls with nested objects and arrays
# Scored per probe and then pooled: parse rate, right function, arguments schema-valid, arguments exact, abstention.
# Against the f16 the number that matters is RETENTION (rung success / f16 success) with a paired bootstrap CI.

def _tool(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


_S = {"type": "string"}
_I = {"type": "integer"}
_N = {"type": "number"}
_B = {"type": "boolean"}


def _e(*vals):
    return {"type": "string", "enum": list(vals)}


def _d(spec, desc):
    return dict(spec, description=desc)


TOOLBOX = {t["function"]["name"]: t for t in [
    _tool("get_weather", "Current weather for a city.",
          {"city": _d(_S, "city name"), "unit": _d(_e("celsius", "fahrenheit"), "temperature unit")}, ["city"]),
    _tool("convert_units", "Convert a quantity between units.",
          {"value": _d(_N, "the amount"), "from_unit": _d(_S, "unit to convert from"),
           "to_unit": _d(_S, "unit to convert to")}, ["value", "from_unit", "to_unit"]),
    _tool("send_email", "Send an email.",
          {"to": _d(_S, "recipient address"), "subject": _d(_S, "subject line"), "body": _d(_S, "message text")},
          ["to", "body"]),
    _tool("set_timer", "Start a countdown timer.",
          {"minutes": _d(_I, "length in minutes"), "label": _d(_S, "optional name for the timer")}, ["minutes"]),
    _tool("search_web", "Search the web.",
          {"query": _d(_S, "search terms"), "max_results": _d(_I, "number of results")}, ["query"]),
    _tool("translate_text", "Translate text into another language.",
          {"text": _d(_S, "text to translate"), "target_language": _d(_S, "language to translate into")},
          ["text", "target_language"]),
    _tool("create_event", "Create a calendar event.",
          {"title": _d(_S, "event title"), "date": _d(_S, "date as YYYY-MM-DD"),
           "duration_minutes": _d(_I, "length in minutes"),
           "attendees": {"type": "array", "items": _S, "description": "attendee email addresses"}},
          ["title", "date"]),
    _tool("book_table", "Reserve a restaurant table.",
          {"restaurant": _d(_S, "restaurant name"), "party_size": _d(_I, "number of people"),
           "time": _d(_S, "time as HH:MM, 24-hour")}, ["restaurant", "party_size", "time"]),
    _tool("set_thermostat", "Set a room's target temperature in degrees.",
          {"temperature": _d(_N, "target temperature"), "room": _d(_S, "room name")}, ["temperature"]),
    _tool("play_music", "Play music by an artist.",
          {"artist": _d(_S, "artist name"), "shuffle": _d(_B, "shuffle the tracks")}, ["artist"]),
    _tool("get_directions", "Route between two places.",
          {"origin": _d(_S, "start"), "destination": _d(_S, "end"),
           "mode": _d(_e("driving", "walking", "transit", "cycling"), "travel mode")}, ["origin", "destination"]),
    _tool("create_ticket", "Open an issue-tracker ticket.",
          {"title": _d(_S, "short summary"), "priority": _d(_e("low", "medium", "high", "urgent"), "priority"),
           "tags": {"type": "array", "items": _S, "description": "labels"}}, ["title", "priority"]),
    _tool("add_todo", "Add an item to the to-do list.",
          {"item": _d(_S, "what to do"), "due_date": _d(_S, "due date as YYYY-MM-DD")}, ["item"]),
    _tool("lookup_contact", "Find a person's contact details.", {"name": _d(_S, "person's name")}, ["name"]),
    _tool("calculate", "Evaluate an arithmetic expression.",
          {"expression": _d(_S, "the expression, e.g. 3*(4+5)")}, ["expression"]),
    _tool("resize_image", "Resize an image file.",
          {"path": _d(_S, "image file"), "width": _d(_I, "pixels"), "height": _d(_I, "pixels"),
           "keep_aspect": _d(_B, "preserve the aspect ratio")}, ["path", "width", "height"]),
    _tool("create_user", "Create a user account.",
          {"user": {"type": "object", "description": "the new user",
                    "properties": {"name": _S, "email": _S, "age": _I}, "required": ["name", "email"]},
           "roles": {"type": "array", "items": _e("admin", "editor", "viewer"), "description": "granted roles"}},
          ["user"]),
    _tool("update_settings", "Change app settings.",
          {"settings": {"type": "object", "description": "settings to change",
                        "properties": {"theme": _e("light", "dark"), "notifications": _B, "language": _S}}},
          ["settings"]),
]}

MENU_A = ["get_weather", "send_email", "set_timer", "search_web", "translate_text", "get_directions", "calculate",
          "add_todo"]
MENU_B = ["create_event", "book_table", "set_thermostat", "play_music", "lookup_contact", "create_ticket",
          "convert_units", "resize_image"]
MENU_C = ["create_user", "update_settings", "create_event", "create_ticket", "resize_image", "send_email",
          "get_directions", "book_table", "convert_units", "play_music"]

#: the system turn every probe gets: what an agent harness says, nothing that hints at the answer
SYSTEM = "You are a helpful assistant. Use a tool only when the request needs one."


def _p(pid, cat, prompt, tools, call=None, args=None):
    return {"id": pid, "category": cat, "prompt": prompt, "tools": list(tools),
            "expect": None if call is None else {"name": call, "args": args or {}}}


# Expected arguments: an int / float / bool is exact; "~a|b" = a case-insensitive substring, any alternative;
# "=x" = case-insensitive equality; {"has": [...]} = an array holding these; {"set": [...]} = exactly these;
# a nested dict = the same rules inside an object argument. Only what the request pins down is checked.
SUITE = [
    # ---- single: one tool on offer, typed arguments
    _p("s01", "single", "What's the weather in Tokyo right now, in celsius?", ["get_weather"],
       "get_weather", {"city": "~tokyo", "unit": "=celsius"}),
    _p("s02", "single", "Set a timer for 25 minutes.", ["set_timer"], "set_timer", {"minutes": 25}),
    _p("s03", "single", "Convert 12.5 miles to kilometers.", ["convert_units"], "convert_units",
       {"value": 12.5, "from_unit": "~mi", "to_unit": "~k"}),
    _p("s04", "single", "Look up the contact details for Priya Raman.", ["lookup_contact"], "lookup_contact",
       {"name": "~priya"}),
    _p("s05", "single", "Turn the bedroom thermostat to 21.5 degrees.", ["set_thermostat"], "set_thermostat",
       {"temperature": 21.5, "room": "~bedroom"}),
    _p("s06", "single", "Play some Miles Davis, shuffled.", ["play_music"], "play_music",
       {"artist": "~miles davis", "shuffle": True}),
    _p("s07", "single", "Translate 'good morning' into Spanish.", ["translate_text"], "translate_text",
       {"text": "~good morning", "target_language": "~spanish|es"}),
    _p("s08", "single", "Search the web for the boiling point of ethanol, just 3 results.", ["search_web"],
       "search_web", {"query": "~ethanol", "max_results": 3}),
    _p("s09", "single", "Calculate 17 * 23 with the calculator.", ["calculate"], "calculate",
       {"expression": "~17*23|17 * 23|17x23|17 x 23"}),
    _p("s10", "single", "Add 'renew passport' to my to-do list.", ["add_todo"], "add_todo", {"item": "~passport"}),
    _p("s11", "single", "Set a 90 minute timer labelled bread.", ["set_timer"], "set_timer",
       {"minutes": 90, "label": "~bread"}),
    _p("s12", "single", "What's the weather like in Nairobi?", ["get_weather"], "get_weather", {"city": "~nairobi"}),
    _p("s13", "single", "Email lee@example.org and tell him the build is green.", ["send_email"], "send_email",
       {"to": "=lee@example.org", "body": "~green"}),
    _p("s14", "single", "Turn on shuffle and play Radiohead.", ["play_music"], "play_music",
       {"artist": "~radiohead", "shuffle": True}),
    _p("s15", "single", "Resize logo.png to 256 by 256 pixels.", ["resize_image"], "resize_image",
       {"path": "~logo.png", "width": 256, "height": 256}),
    # ---- choose: eight tools on offer
    _p("c01", "choose", "How do I get from the Louvre to the Eiffel Tower on foot?", MENU_A, "get_directions",
       {"origin": "~louvre", "destination": "~eiffel", "mode": "=walking"}),
    _p("c02", "choose", "Put 'call the plumber' on my to-do list, due 2026-11-03.", MENU_A, "add_todo",
       {"item": "~plumber", "due_date": "=2026-11-03"}),
    _p("c03", "choose", "Use the calculator: what is 2 to the power of 16?", MENU_A, "calculate",
       {"expression": "~16"}),
    _p("c04", "choose", "Is it raining in Lisbon?", MENU_A, "get_weather", {"city": "~lisbon"}),
    _p("c05", "choose", "Translate 'where is the station' to German.", MENU_A, "translate_text",
       {"text": "~station", "target_language": "~german|de"}),
    _p("c06", "choose", "Find recent articles about solid-state batteries.", MENU_A, "search_web",
       {"query": "~solid-state|solid state"}),
    _p("c07", "choose", "Email ana@example.com that the meeting moved to 3pm.", MENU_A, "send_email",
       {"to": "=ana@example.com", "body": "~3"}),
    _p("c08", "choose", "Start a 10 minute tea timer.", MENU_A, "set_timer", {"minutes": 10}),
    _p("c09", "choose", "Book a table for 4 at Nopa at 19:30.", MENU_B, "book_table",
       {"restaurant": "~nopa", "party_size": 4, "time": "~19:30"}),
    _p("c10", "choose", "Schedule 'Design review' on 2026-11-12 for 45 minutes.", MENU_B, "create_event",
       {"title": "~design review", "date": "=2026-11-12", "duration_minutes": 45}),
    _p("c11", "choose", "File an urgent ticket: the login page returns a 500 error.", MENU_B, "create_ticket",
       {"title": "~login", "priority": "=urgent"}),
    _p("c12", "choose", "Set the living room to 19 degrees.", MENU_B, "set_thermostat",
       {"temperature": 19, "room": "~living"}),
    _p("c13", "choose", "Resize photo.png to 800 by 600.", MENU_B, "resize_image",
       {"path": "~photo.png", "width": 800, "height": 600}),
    _p("c14", "choose", "What's Dana Okafor's phone number?", MENU_B, "lookup_contact", {"name": "~dana"}),
    _p("c15", "choose", "How many ounces is 250 grams?", MENU_B, "convert_units",
       {"value": 250, "from_unit": "~g", "to_unit": "~oz|ounce"}),
    # ---- abstain: tools on offer, none fits the request
    _p("a01", "abstain", "What is the capital of France?", MENU_B),
    _p("a02", "abstain", "Thanks, that's all for now!", MENU_A),
    _p("a03", "abstain", "Write a haiku about autumn leaves.", MENU_A),
    _p("a04", "abstain", "Explain what a hash map is in one sentence.", MENU_B),
    _p("a05", "abstain", "Hi! How are you today?", MENU_B),
    _p("a06", "abstain", "What's 2 + 2?", MENU_B),
    _p("a07", "abstain", "Tell me a short joke about cats.", MENU_A),
    _p("a08", "abstain", "Summarize in five words: 'The meeting covered the roadmap and hiring.'", MENU_B),
    _p("a09", "abstain", "What does the word 'ephemeral' mean?", ["get_weather"]),
    _p("a10", "abstain", "Who wrote Pride and Prejudice?", ["set_timer"]),
    _p("a11", "abstain", "Rephrase 'send it asap' more politely. Just give me the sentence.", ["send_email"]),
    _p("a12", "abstain", "What's the difference between a violin and a viola?", ["play_music"]),
    _p("a13", "abstain", "What is a good ratio of rice to water when cooking rice?", ["book_table"]),
    _p("a14", "abstain", "Spell the word 'necessary' backwards.", ["create_ticket"]),
    _p("a15", "abstain", "I already checked the weather and it is sunny. Just reply OK.", MENU_A),
    # ---- nested: several arguments, objects and arrays
    _p("n01", "nested", "Create a user named Ada Lovelace, email ada@example.com, age 36, with the admin and "
                        "editor roles.", MENU_C, "create_user",
       {"user": {"name": "~ada", "email": "=ada@example.com", "age": 36}, "roles": {"set": ["admin", "editor"]}}),
    _p("n02", "nested", "Switch the app to the dark theme and turn notifications off.", MENU_C, "update_settings",
       {"settings": {"theme": "=dark", "notifications": False}}),
    _p("n03", "nested", "Set up 'Sprint planning' on 2026-12-01 for 60 minutes and invite sam@example.com and "
                        "kim@example.com.", MENU_C, "create_event",
       {"title": "~sprint planning", "date": "=2026-12-01", "duration_minutes": 60,
        "attendees": {"set": ["sam@example.com", "kim@example.com"]}}),
    _p("n04", "nested", "Open a high priority ticket 'Checkout timeout' tagged api and backend.", MENU_C,
       "create_ticket", {"title": "~checkout timeout", "priority": "=high", "tags": {"set": ["api", "backend"]}}),
    _p("n05", "nested", "Resize banner.jpg to exactly 1200x300 and do not keep the aspect ratio.", MENU_C,
       "resize_image", {"path": "~banner.jpg", "width": 1200, "height": 300, "keep_aspect": False}),
    _p("n06", "nested", "Give me transit directions from Union Square to the airport.", MENU_C, "get_directions",
       {"origin": "~union square", "destination": "~airport", "mode": "=transit"}),
    _p("n07", "nested", "Reserve Chez Panisse for 2 people at 20:00.", MENU_C, "book_table",
       {"restaurant": "~panisse", "party_size": 2, "time": "~20:00"}),
    _p("n08", "nested", "Email ops@example.com with the subject 'Disk alert' saying the disk is 91% full.", MENU_C,
       "send_email", {"to": "=ops@example.com", "subject": "~disk alert", "body": "~91"}),
    _p("n09", "nested", "Add a viewer account for Bo Chen, bo@example.com.", MENU_C, "create_user",
       {"user": {"name": "~bo", "email": "=bo@example.com"}, "roles": {"set": ["viewer"]}}),
    _p("n10", "nested", "Set my language to French and use the light theme.", MENU_C, "update_settings",
       {"settings": {"language": "~fr", "theme": "=light"}}),
    _p("n11", "nested", "Convert 3.75 liters to gallons.", MENU_C, "convert_units",
       {"value": 3.75, "from_unit": "~l", "to_unit": "~gal"}),
    _p("n12", "nested", "Book 'Coffee chat' on 2026-11-20 for 30 minutes with lee@example.org.", MENU_C,
       "create_event", {"title": "~coffee chat", "date": "=2026-11-20", "duration_minutes": 30,
                        "attendees": {"has": ["lee@example.org"]}}),
    _p("n13", "nested", "Low priority ticket: typo on the about page, tags docs and frontend.", MENU_C,
       "create_ticket", {"title": "~typo", "priority": "=low", "tags": {"set": ["docs", "frontend"]}}),
    _p("n14", "nested", "Play Nina Simone in album order, no shuffle.", MENU_C, "play_music",
       {"artist": "~nina simone", "shuffle": False}),
    _p("n15", "nested", "Create user Kai Ito, kai@example.com, 29 years old, with the editor role.", MENU_C,
       "create_user", {"user": {"name": "~kai", "email": "=kai@example.com", "age": 29},
                       "roles": {"set": ["editor"]}}),
]
CATEGORIES = ("single", "choose", "abstain", "nested")


# ---------------------------------------------------------------------------------------------- grading one answer

def _type_ok(val, spec):
    t = (spec or {}).get("type")
    if t == "string":
        return isinstance(val, str)
    if t == "integer":
        return isinstance(val, int) and not isinstance(val, bool)
    if t == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if t == "boolean":
        return isinstance(val, bool)
    if t == "array":
        return isinstance(val, list)
    if t == "object":
        return isinstance(val, dict)
    return True


def validate_args(args, schema, where=""):
    """(ok, failure, detail) of arguments against a JSON schema: required present, no undeclared key, types, enums,
    array items, nested objects. Never executes anything -- the point is whether a strict caller would accept it."""
    props = schema.get("properties") or {}
    for key in schema.get("required") or []:
        if key not in args:
            return False, "missing-arg", f"required argument {where + key!r} absent"
    for key, val in args.items():
        spec = props.get(key)
        if spec is None:
            return False, "hallucinated", f"argument {where + key!r} is not in the schema"
        if not _type_ok(val, spec):
            return False, "bad-type", f"{where + key!r} is {type(val).__name__}, schema says {spec.get('type')}"
        if "enum" in spec and val not in spec["enum"]:
            return False, "bad-enum", f"{where + key!r} is {val!r}, schema allows {spec['enum']}"
        if spec.get("type") == "array":
            item = spec.get("items") or {}
            for v in val:
                if not _type_ok(v, item) or ("enum" in item and v not in item["enum"]):
                    return False, "bad-type", f"{where + key!r} holds {v!r}, schema says {item}"
        if spec.get("type") == "object":
            ok, fail, det = validate_args(val, spec, where + key + ".")
            if not ok:
                return ok, fail, det
    return True, None, "schema-valid"


def _match(got, want):
    """Does one argument value satisfy its expectation (see SUITE)?"""
    if isinstance(want, bool):
        return got is want
    if isinstance(want, (int, float)):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(float(got) - want) < 1e-6
    if isinstance(want, str):
        if not isinstance(got, str):
            return False
        g = got.strip().lower()
        if want.startswith("="):
            return g == want[1:].lower()
        return any(alt.lower() in g for alt in want.lstrip("~").split("|"))
    if isinstance(want, dict) and ("set" in want or "has" in want):
        if not isinstance(got, list):
            return False
        have = {str(x).strip().lower() for x in got}
        need = {str(x).lower() for x in want.get("set", want.get("has", []))}
        return have == need if "set" in want else need <= have
    if isinstance(want, dict):
        return isinstance(got, dict) and all(k in got and _match(got[k], v) for k, v in want.items())
    return got == want


def _mismatch(args, want):
    for k, v in (want or {}).items():
        if k not in args or not _match(args[k], v):
            return k
    return None


def grade_reply(reply, probe):
    """Grade one /v1/chat/completions reply against one probe -- THE grader: a served reply goes straight in, an
    in-process generation goes in through reply_from_text (the same reply shape). Returns a row:
    {id, category, expected, called, parsed, name_ok, args_valid, exact, ok, failure, detail, sample}."""
    msg = ((reply or {}).get("choices") or [{}])[0].get("message") or {}
    finish = ((reply or {}).get("choices") or [{}])[0].get("finish_reason")
    calls = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    think, answer, unclosed = _split_think(content)
    text = answer if think else content
    exp = probe.get("expect")
    row = {"id": probe["id"], "category": probe["category"], "expected": exp is not None, "called": bool(calls),
           "parsed": bool(calls), "name_ok": False, "args_valid": False, "exact": False, "ok": False,
           "failure": None, "detail": "", "finish": finish,
           "sample": (json.dumps(calls[0].get("function", {}))[:160] if calls else text.replace("\n", " ")[:160])}
    if reply is None or "error" in (reply or {}):
        err = (reply or {}).get("error") if reply else "no reply"
        row.update(failure="server-error", detail=str(err)[:160])
        return row
    looks_like_call = _call_shaped(text)
    if exp is None:                                              # abstention: prose is the right answer
        if calls:
            row.update(failure="false-call", detail=f"called {calls[0].get('function', {}).get('name')!r} "
                                                    "although no tool was needed")
        elif unclosed and not text.strip():
            row.update(failure="unclosed-think", detail="thinking never closed within the budget")
        else:
            row.update(ok=True, detail="answered without a tool")
        return row
    if not calls:
        if unclosed or (finish == "length" and not text.strip()):
            row.update(failure="unclosed-think", detail="the budget ran out before any call was made")
        elif looks_like_call:
            row.update(failure="unparsed", detail="call-shaped text the parser did not accept")
        elif finish == "tool_calls":
            # mlx_lm.server: the model opened a call, the parser rejected its text and dropped it from the reply
            row.update(failure="unparsed", detail="the server's parser dropped the call (finish_reason tool_calls, "
                                                  "no tool_calls)")
        else:
            row.update(failure="no-call", detail="answered in prose instead of calling")
        return row
    fn = calls[0].get("function") or {}
    name, raw = fn.get("name"), fn.get("arguments")
    tool = TOOLBOX.get(exp["name"])
    row["name_ok"] = name == exp["name"]
    if not row["name_ok"]:
        row.update(failure="wrong-tool" if name in probe["tools"] else "hallucinated-tool",
                   detail=f"called {name!r}, wanted {exp['name']!r}")
        return row
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, dict) else None)
    except ValueError:
        args = None
    if not isinstance(args, dict):
        row.update(failure="bad-args", detail="arguments are not a JSON object")
        return row
    ok, fail, det = validate_args(args, tool["function"]["parameters"])
    row["args_valid"] = ok
    if not ok:
        row.update(failure=fail, detail=det)
        return row
    bad = _mismatch(args, exp["args"])
    if bad is not None:
        row.update(failure="wrong-value", detail=f"{bad!r} is {args.get(bad)!r}, wanted {exp['args'][bad]!r}")
        return row
    row.update(exact=True, ok=True, detail="right tool, valid and exact arguments")
    return row


# ---------------------------------------------------------------------------------------------- statistics

def wilson(k, n, z=1.96):
    """Wilson score interval for k successes of n -- honest at small n and near 0 or 1, where +-1.96*SE is not."""
    if not n:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def retention_ci(rows, ref_rows, b=2000, seed=0, alpha=0.05):
    """(retention, lo, hi): rung success / f16 success, CI by a PAIRED bootstrap over probes -- each probe keeps its
    rung and f16 outcome together, so the CI is of the ratio, not of two independent rates."""
    ref = {r["id"]: r for r in ref_rows}
    pairs = [(1.0 if r["ok"] else 0.0, 1.0 if ref[r["id"]]["ok"] else 0.0) for r in rows if r["id"] in ref]
    if not pairs or not sum(p[1] for p in pairs):
        return None, None, None
    point = sum(p[0] for p in pairs) / sum(p[1] for p in pairs)
    rng = random.Random(seed)
    n, stats = len(pairs), []
    for _ in range(b):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        den = sum(x[1] for x in s)
        if den:
            stats.append(sum(x[0] for x in s) / den)
    stats.sort()
    return point, stats[int(alpha / 2 * len(stats))], stats[min(int((1 - alpha / 2) * len(stats)), len(stats) - 1)]


def score_suite(rows):
    """The board for one model: success with a Wilson CI, the four component rates, abstention, per category."""
    n = len(rows)
    call = [r for r in rows if r["expected"]]
    abst = [r for r in rows if not r["expected"]]
    ok = sum(r["ok"] for r in rows)

    def frac(k, m):
        return round(k / m, 4) if m else None
    modes = {}
    for r in rows:
        if not r["ok"]:
            modes[r["failure"]] = modes.get(r["failure"], 0) + 1
    lo, hi = wilson(ok, n)
    return {
        "n": n, "valid": ok, "rate": frac(ok, n), "rate_ci": [round(lo, 4), round(hi, 4)],
        "parse_rate": frac(sum(r["parsed"] for r in call), len(call)),
        "function_acc": frac(sum(r["name_ok"] for r in call), len(call)),
        "args_valid": frac(sum(r["args_valid"] for r in call), len(call)),
        "exact": frac(sum(r["exact"] for r in call), len(call)),
        "abstention": frac(sum(r["ok"] for r in abst), len(abst)),
        "categories": {c: frac(sum(r["ok"] for r in rows if r["category"] == c),
                               sum(1 for r in rows if r["category"] == c)) for c in CATEGORIES
                       if any(r["category"] == c for r in rows)},
        "modes": modes,
    }


# ---------------------------------------------------------------------------------------------- serving

#: the served model's own stdout+stderr, for a server that dies mid-gate
_SERVER_LOG = "toolcall-server.log"


# ---------------------------------------------------------------------------------------------- backends
#
# Every lane is gated the way its users will run it, because the chat-template rendering and the call parser decide
# what an agent harness receives:
#
#   gguf  llama-server --jinja                                   the GGUF lane                       (served)
#   mlx   mlx_lm.server                                          the MLX lane                        (served)
#   vllm  vllm serve --enable-auto-tool-choice --tool-call-parser <family>
#                                                                GPTQ / AWQ / FP8 / MXFP4 / NVFP4    (served)
#   exl3  exllamav3 in this process                              the EXL3 lane                       (in-process)
#   hf    transformers in this process                           GPTQ / AWQ / FP8 / compressed-tensors without a
#                                                                usable vLLM; bf16 references        (in-process)
#   url   an OpenAI-compatible server that is already running    TabbyAPI, a remote vLLM, ...        (served)
#
# In-process: the model's own HF chat template renders the probe's `tools` (transformers apply_chat_template, or
# pollard-modelkind's jinja2 renderer without transformers), the model decodes greedily, and the raw text is read
# back by the reader for its family (pollard-modelkind parse_tool_calls -- the same strictness as that family's
# server parser) into the same reply shape a server returns, so ONE grader (grade_reply) grades every path.
#
# A backend this machine cannot run -- vLLM on a Mac, a GPU with someone else's process resident, a model that will
# not fit the memory that is free -- is never faked: the rung is recorded as {"status": "not-measured", "reason": ...},
# the build carries on, and the card says "not measured (reason)".

BACKENDS = ("gguf", "mlx", "vllm", "exl3", "hf", "url")
#: the backends that run the model inside this process (no server, Pollard's own reader)
INPROCESS = ("exl3", "hf")
#: build lane -> the backend that serves it (a vllm lane falls back to hf where no usable vLLM is installed)
LANE_BACKEND = {"gguf": "gguf", "mlx": "mlx", "gptq": "vllm", "awq": "vllm", "mx": "vllm", "exl3": "exl3",
                # an EXLR8 rung runs in transformers on its reference decode (GR8 serves GLM-5.3 only; vLLM cannot
                # load EXLR8 files): the hf engine loads it through pollard_exlr8.load_model
                "exlr8": "hf"}
#: the backend -> how the card and the logs name it
BACKEND_SERVER = {"gguf": "llama-server --jinja", "mlx": "mlx_lm.server", "vllm": "vllm serve",
                  "exl3": "exllamav3 (in-process)", "hf": "transformers (in-process)",
                  "url": "an OpenAI-compatible server"}
#: name the vLLM server answers to (the requests carry it)
SERVED_NAME = "pollard-toolcall"
#: Windows / desktop processes that sit on every NVIDIA GPU and do not mean someone is using it
GPU_SHELL = ("dwm.exe", "explorer.exe", "shellhost", "searchhost", "startmenuexperiencehost", "textinputhost",
             "shellexperiencehost", "crossdeviceresume", "systemsettings", "applicationframehost", "ctfmon",
             "lockapp", "widgets", "windowsterminal", "nvidia overlay", "nvcontainer", "msedgewebview2", "dcv2",
             "xorg", "gnome-shell", "xwayland", "kwin")


class NotMeasured(Exception):
    """This backend cannot measure this model on this machine. str(e) is the reason the card prints."""


def _apple():
    import platform
    return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")


def _has_module(name):
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _config(path):
    for name in ("config.json",):
        try:
            with open(os.path.join(path, name), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            continue
    return {}


def lane_of(path):
    """gguf | mlx | gptq | mx | exl3 | hf (an unquantized checkpoint) | None -- which lane a build belongs to: the
    workspace MANIFEST's word when the build is recorded there, else what its files say."""
    p = str(path or "").rstrip("/\\")
    if p.lower().endswith(".gguf"):
        return "gguf"
    if not p or not os.path.isdir(p):
        return None
    if os.path.isfile(os.path.join(p, "pollard-exlr8.json")):     # an EXLR8 rung carries its marker
        return "exlr8"
    try:
        import pollard_workspace as ws
        _, b = ws.find_build(p)
        if b and b.get("lane"):
            return b["lane"]
    except Exception:                                            # noqa: BLE001 -- no workspace: read the files
        pass
    cfg = _config(p)
    qc = cfg.get("quantization_config") or {}
    meth = str(qc.get("quant_method") or "").lower()
    if not meth and os.path.isfile(os.path.join(p, "quantization_config.json")):
        try:
            meth = str(json.load(open(os.path.join(p, "quantization_config.json"))).get("quant_method", "")).lower()
        except (OSError, ValueError):
            meth = ""
    if meth.startswith("exl"):
        return "exl3"
    if meth in ("compressed-tensors", "fp8", "mxfp4", "nvfp4", "modelopt", "modelopt_fp4", "quark"):
        return "mx"
    if meth:
        return "gptq"                                            # gptq / awq / bitsandbytes ...: vLLM serves it
    if isinstance(cfg.get("quantization"), dict) and "bits" in cfg["quantization"]:
        return "mlx"                                             # mlx-lm writes {"quantization": {group_size, bits}}
    if cfg:
        return "hf"
    return None


def quant_method(path):
    """The checkpoint's quant_method ("gptq", "awq", "compressed-tensors", "fp8", "mxfp4", "exl3", ...) or ""."""
    p = str(path or "").rstrip("/\\")
    qc = _config(p).get("quantization_config") or {}
    meth = str(qc.get("quant_method") or "").lower() if isinstance(qc, dict) else ""
    if not meth and os.path.isfile(os.path.join(p, "quantization_config.json")):
        try:
            meth = str(json.load(open(os.path.join(p, "quantization_config.json"))).get("quant_method", "")).lower()
        except (OSError, ValueError, AttributeError):
            meth = ""
    return meth


def pick_backend(path, prefer=None, url=None, vllm_ok=None):
    """The backend for this model. A GPTQ / AWQ / MX build goes to vLLM where a usable vLLM is installed, else to the
    in-process transformers backend. An unquantized checkpoint (a reference) goes to the rung's own backend when that
    backend loads HF weights (mlx, vllm, hf), so the reference is run the SAME way as the rung; an EXL3 rung's
    reference goes to hf (exllamav3 loads only EXL3 weights; hf renders and reads exactly the way exl3 does)."""
    if url:
        return "url"
    lane = lane_of(path)

    def vllm_or_hf():
        ok = vllm_status()[0] if vllm_ok is None else vllm_ok
        return "vllm" if ok else "hf"
    if lane in LANE_BACKEND:
        b = LANE_BACKEND[lane]
        return vllm_or_hf() if b == "vllm" else b
    if lane == "hf":
        if prefer in ("mlx", "hf"):
            return prefer
        if prefer == "exl3":
            return "hf"
        if prefer == "vllm":
            return vllm_or_hf()
        return "mlx" if _apple() else vllm_or_hf()
    return prefer if prefer in BACKENDS else "gguf"


def model_gb(path):
    """Weight GB on disk (a GGUF file, or every weight file in a checkpoint dir). 0 = unknown."""
    p = str(path or "")
    try:
        if os.path.isfile(p):
            return os.path.getsize(p) / 1e9
        tot = 0
        for root, _, files in os.walk(p):
            for f in files:
                if f.endswith((".safetensors", ".bin", ".gguf", ".npz", ".pt")):
                    tot += os.path.getsize(os.path.join(root, f))
        return tot / 1e9
    except OSError:
        return 0.0


def _nvsmi(args):
    """nvidia-smi's stdout, or None when there is no NVIDIA driver here."""
    import shutil
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(["nvidia-smi"] + list(args), capture_output=True, text=True, timeout=20)
    except Exception:                                            # noqa: BLE001
        return None
    return r.stdout if r.returncode == 0 else None


def gpu_users(table=None, apps=None):
    """Processes resident on the NVIDIA GPU that are not the desktop shell -- graphics (a game, G / C+G) as well as
    compute. `--query-compute-apps` alone misses a game, which is exactly the process the shared box must not
    disturb, so the plain process table is read too."""
    table = _nvsmi([]) if table is None else table
    apps = _nvsmi(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]) \
        if apps is None else apps
    names = []
    in_procs = False
    for line in (table or "").splitlines():
        if "Processes:" in line:
            in_procs = True
            continue
        if not in_procs:
            continue
        m = re.match(r"^\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(C\+G|G|C|M|M\+C)\s+(.+?)\s+(\d+MiB|N/A)\s*\|", line)
        if m:
            names.append(m.group(3).strip())
    for line in (apps or "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[1]:
            names.append(parts[1])
    # a shared box's always-on desktop apps (chat overlays, recorders, browsers) hold a graphics context all day; the
    # owner of the box may name them in POLLARD_GPU_IGNORE (comma list of name fragments) -- a game still counts
    extra = tuple(x.strip().lower() for x in os.environ.get("POLLARD_GPU_IGNORE", "").split(",") if x.strip())
    out = []
    for n in names:
        base = re.split(r"[\\/]", n)[-1]
        if not any(s in base.lower() for s in GPU_SHELL + extra) and base not in out:
            out.append(base)
    return out


def gpu_free_gb(text=None):
    """Free VRAM on the emptiest card (vLLM / exllamav3 load onto one), or None when there is no NVIDIA GPU."""
    text = _nvsmi(["--query-gpu=memory.free", "--format=csv,noheader,nounits"]) if text is None else text
    vals = [int(x) for x in re.findall(r"\d+", text or "")]
    return max(vals) / 1024 if vals else None


def _free_ram_gb():
    try:
        from pollard_calc import detect_available_ram_gb
        return detect_available_ram_gb() or 0.0
    except Exception:                                            # noqa: BLE001
        return 0.0


def check_gpu(backend, need_gb=0.0, users=None, free=None):
    """A GPU-only backend (vLLM, exllamav3) runs only on an NVIDIA GPU that nobody else is using and that has room.
    On a shared box the GPU belongs to whoever is on it: a resident process means not-measured, never a launch."""
    users = gpu_users() if users is None else users
    free = gpu_free_gb() if free is None else free
    if free is None:
        raise NotMeasured(f"{BACKEND_SERVER[backend]} needs an NVIDIA GPU and this machine has none")
    if users:
        raise NotMeasured("the GPU has another process resident (" + ", ".join(users[:3]) + ") -- not ours to take")
    want = need_gb * 1.2 + 1.5                                   # weights + a 4K KV cache + the runtime's own
    if need_gb and want > free:
        raise NotMeasured(f"{need_gb:.1f}GB model needs ~{want:.1f}GB of VRAM, {free:.1f}GB free")


def check_ram(need_gb, avail=None):
    """A CPU / unified-memory backend (llama-server -ngl 0, mlx_lm) must fit the RAM that is free right now."""
    avail = _free_ram_gb() if avail is None else avail
    want = need_gb * 1.15 + 1.0
    if need_gb and avail and want > avail:
        raise NotMeasured(f"{need_gb:.1f}GB model needs ~{want:.1f}GB of RAM, {avail:.1f}GB free")


_VLLM_STATUS = {}


def _python_beside(binary):
    """The interpreter of the environment a console script (vllm / vllm.exe) was installed into."""
    d = os.path.dirname(os.path.abspath(binary))
    for name in ("python.exe", "python", "python3"):
        if os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    return None


def vllm_status(binary=None):
    """(usable, reason): is there a vLLM here that can actually serve on CUDA? Installed is not enough -- `pip install
    vllm` on Windows builds a source distribution with no compiled kernels (no vllm/_C), which imports and then
    cannot run anything. So the package directory is checked for its compiled _C extension, without importing vLLM
    (a fork's Windows wheel -- SystemPanic/vllm-windows, aivrar/vllm-windows-build -- ships one; so does Linux)."""
    import shutil
    binary = binary or os.environ.get("POLLARD_VLLM") or shutil.which("vllm")
    key = binary or "<this python>"
    if key in _VLLM_STATUS:
        return _VLLM_STATUS[key]
    py = (_python_beside(binary) if binary else None) or sys.executable
    if not binary and not _has_module("vllm"):
        res = (False, "vLLM is not installed here")
    else:
        # the compiled extension by exact name (_C.abi3.so / _C.cp312-win_amd64.pyd ...): Windows' glob is
        # case-blind, and `_C*` there matches vllm/_custom_ops.py
        probe = ("import importlib.util as u, os\n"
                 "s = u.find_spec('vllm')\n"
                 "d = list(s.submodule_search_locations or [])[0] if s else ''\n"
                 "ok = d and any(n.startswith('_C.') and n.endswith(('.so', '.pyd')) for n in os.listdir(d))\n"
                 "print('NOPKG' if not d else ('OK' if ok else 'NOC'))")
        try:
            out = subprocess.run([py, "-c", probe], capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e:                                   # noqa: BLE001
            out = f"ERR {e}"
        if out.endswith("OK"):
            res = (True, f"vLLM with compiled kernels ({py})")
        elif out.endswith("NOC"):
            res = (False, "the vLLM installed here has no compiled CUDA kernels (vllm/_C missing -- a source build "
                          "with no GPU target, e.g. `pip install vllm` on Windows); install a CUDA build (a Windows "
                          "fork wheel or WSL2) or let the in-process transformers backend measure it")
        else:
            res = (False, "vLLM is not installed here")
    _VLLM_STATUS[key] = res
    return res


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def gguf_cmd(binary, model, port, ngl, ctx=4096, threads=None):
    """llama-server --jinja: the model's own template renders the tools, llama.cpp's own parser (with its lazy
    grammar) reads the call."""
    cmd = [binary, "-m", model, "--host", "127.0.0.1", "--port", str(port), "-ngl", str(ngl), "-c", str(ctx),
           "-np", "1",                       # one slot: deterministic, and no KV pool shared four ways
           "--jinja",
           "--reasoning-format", "deepseek"]  # thinking goes to reasoning_content, not into the call parser
    if threads:
        cmd += ["-t", str(threads)]
    return cmd


def mlx_cmd(python, model, port, trust_remote_code=False):
    """mlx_lm.server: the tokenizer's chat template renders `tools`, mlx-lm's tool parser fills `tool_calls`."""
    cmd = [python, "-m", "mlx_lm.server", "--model", model, "--host", "127.0.0.1", "--port", str(port),
           "--temp", "0"]
    if trust_remote_code:
        cmd += ["--trust-remote-code"]
    return cmd


def vllm_cmd(binary, model, port, parser, ctx=4096, trust_remote_code=False, gpu_util=0.85):
    """vllm serve with the tool parser for this model's call syntax (pollard-modelkind toolcall_info)."""
    cmd = [binary, "serve", model, "--host", "127.0.0.1", "--port", str(port), "--served-model-name", SERVED_NAME,
           "--enable-auto-tool-choice", "--tool-call-parser", parser, "--max-model-len", str(ctx),
           "--gpu-memory-utilization", f"{gpu_util:.2f}"]
    if trust_remote_code:
        cmd += ["--trust-remote-code"]
    return cmd


def _trust(model):
    try:
        return bool(_config(model).get("auto_map"))
    except Exception:                                            # noqa: BLE001
        return False


@contextlib.contextmanager
def _launch(cmd, what, log_path=None, health="/health", port=None, env=None, cwd=None, wait_s=900):
    """Start one server, wait for `health` to answer 200, yield its base url, always stop it afterwards."""
    log = open(log_path or _SERVER_LOG, "w", encoding="utf-8", errors="replace")
    try:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, cwd=cwd)
    except OSError as e:
        log.close()
        raise NotMeasured(f"could not start {what}: {e}") from None
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(wait_s):
            if proc.poll() is not None:
                raise SystemExit(f"{what} exited ({proc.returncode}) before serving; see {log.name}")
            try:
                with urllib.request.urlopen(base + health, timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:                                    # noqa: BLE001 -- still loading
                time.sleep(1)
        else:
            raise SystemExit(f"{what} never became ready; see {log.name}")
        yield base
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:                                    # noqa: BLE001
                proc.kill()
        log.close()


@contextlib.contextmanager
def served(model, ngl, ctx=4096, threads=None, server=None, log_path=None):
    """llama-server --jinja for the duration; yields its base url."""
    from pollard_calc import find_llama_bin
    import shutil
    import pollard_llamaserve as LS
    binary = server or LS.bin_for(model, "llama-server") or shutil.which("llama-server")
    if not binary:
        raise NotMeasured("llama-server not found (install llama.cpp or pass --llama-server)")
    # pollard_llamaserve: -ngl from free VRAM (a bare `auto` is never handed to the server), one slot, a wait scaled
    # to the file, the server's own log tail on failure, one retry on the CPU
    import pollard_llamaserve as LS
    extra = gguf_cmd(binary, model, 0, ngl, ctx, threads)
    extra = extra[extra.index("--jinja"):]                       # the template + reasoning flags (+ -t)
    try:
        with LS.serve(model, ngl, ctx, binary=binary, extra=extra,
                      log_path=log_path or os.path.abspath(_SERVER_LOG)) as srv:
            yield srv.base
    except LS.ServerFailed as e:
        raise SystemExit(str(e)) from None


def preflight(backend, model, ngl="0", kind=None, server=None):
    """Raise NotMeasured when this machine cannot serve `model` with `backend`; else the extra request fields the
    backend wants. Asked BEFORE a server starts: memory first (never start what will not fit), then the GPU rule."""
    kind = kind or {}
    need = model_gb(model)
    if backend == "gguf":
        if str(ngl) in ("0", ""):
            check_ram(need)
        return {}
    if backend == "mlx":
        if not _apple():
            raise NotMeasured("mlx_lm.server needs Apple Silicon")
        if not (os.environ.get("POLLARD_MLX_PYTHON") or _has_module("mlx_lm")):
            raise NotMeasured("mlx-lm is not installed here (pip install mlx-lm)")
        check_ram(need)
        return {}
    if backend in INPROCESS:
        return preflight_inprocess(backend, model, kind)
    if backend == "vllm" and gpu_free_gb() is None:
        raise NotMeasured(f"{BACKEND_SERVER[backend]} needs an NVIDIA GPU and this machine has none")
    if backend == "vllm":
        ok, why = vllm_status(server)
        if not ok:
            raise NotMeasured(why)
        parser = (kind.get("tools") or {}).get("vllm_parser")
        if not parser:
            fam = (kind.get("tools") or {}).get("family", "none")
            raise NotMeasured(f"no vLLM --tool-call-parser reads the '{fam}' call syntax")
        check_gpu("vllm", need)
        return {"model": SERVED_NAME}
    return {}


# ---------------------------------------------------------------------------------------------- in-process backends

#: quant_method -> the modules transformers needs to load it (any one of each tuple); None = nothing loads it here
HF_QUANT_NEEDS = {
    "gptq": [("gptqmodel", "auto_gptq")], "awq": [("gptqmodel", "awq")],
    "compressed-tensors": [("compressed_tensors",)], "fp8": [], "mxfp4": [],
    "bitsandbytes": [("bitsandbytes",)],
    "modelopt": None, "modelopt_fp4": None, "nvfp4": None, "quark": None,
}
_HF_NO_LOADER = {
    "modelopt": "a ModelOpt (TensorRT-LLM) checkpoint has no transformers loader -- serve it with vLLM / TensorRT-LLM",
    "modelopt_fp4": "a ModelOpt NVFP4 checkpoint has no transformers loader -- serve it with vLLM / TensorRT-LLM",
    "nvfp4": "a native NVFP4 checkpoint has no transformers loader (compressed-tensors NVFP4 does) -- serve it with vLLM",
    "quark": "an AMD Quark checkpoint has no transformers loader on this stack -- serve it with vLLM",
}


def reader_family(kind):
    """(family, reader key) for an in-process run: the template's family when Pollard has a reader for it; a template
    with no call syntax of its own is read as plain JSON calls; a family with no reader cannot be measured here."""
    from pollard_modelkind import CALL_READERS
    fam = ((kind or {}).get("tools") or {}).get("family") or "none"
    if fam in CALL_READERS:
        return fam, fam
    if fam == "none":
        return fam, "json"
    raise NotMeasured(f"no in-process reader for the '{fam}' call syntax here (serve it with --backend url)")


def exlr8_decoded_gb(rung):
    """bf16 GB of an EXLR8 rung once decoded (what the engine will hold), from the safetensors headers alone."""
    try:
        import pollard_exlr8 as X
        return X.decoded_bytes(rung) / 1e9
    except Exception:                                            # noqa: BLE001 -- unreadable: size on disk
        return model_gb(rung)


def _gpu_or_none():
    try:
        return gpu_free_gb()
    except Exception:                                            # noqa: BLE001
        return None


def preflight_inprocess(backend, model, kind=None):
    """exl3 / hf: what this machine can run in this process, and on which device. Returns {"device": ...} (the run
    reads it, the request does not). Memory first, then the shared-GPU rule, as for the served backends."""
    need = exlr8_decoded_gb(model) if lane_of(model) == "exlr8" else model_gb(model)
    reader_family(kind)
    if backend == "exl3":
        if _gpu_or_none() is None:
            raise NotMeasured("exllamav3 needs an NVIDIA GPU and this machine has none")
        if not _has_module("exllamav3"):
            raise NotMeasured("exllamav3 is not installed here (pip install exllamav3, with a CUDA torch)")
        if not (_has_module("transformers") or _has_module("jinja2")):
            raise NotMeasured("no chat-template renderer here (transformers or jinja2)")
        check_gpu("exl3", need)
        return {"device": "cuda"}
    # hf
    if not (_has_module("transformers") and _has_module("torch")):
        raise NotMeasured("transformers + torch are not installed here")
    import pollard_mxfp6                                         # a pollard-mx MXFP6 / mix rung: decoded, no loader
    meth = "" if pollard_mxfp6.read_meta(model) else quant_method(model)
    if meth:
        if meth not in HF_QUANT_NEEDS or HF_QUANT_NEEDS[meth] is None:
            raise NotMeasured(_HF_NO_LOADER.get(meth, f"transformers has no loader for a '{meth}' checkpoint here"))
        for alts in HF_QUANT_NEEDS[meth]:
            if not any(_has_module(m) for m in alts):
                raise NotMeasured(f"transformers needs {' or '.join(alts)} to load a {meth} checkpoint "
                                  f"(pip install {alts[0]})")
        if _gpu_or_none() is None:
            raise NotMeasured(f"a {meth} checkpoint runs in transformers on an NVIDIA GPU and this machine has none")
        check_gpu("hf", need)
        return {"device": "cuda"}
    if _gpu_or_none() is not None:                               # unquantized: the GPU when it is free and fits
        try:
            check_gpu("hf", need)
            return {"device": "cuda"}
        except NotMeasured:
            pass                                                 # someone is on it / too big: stay off it
    if _apple():
        check_ram(need)
        return {"device": "mps"}
    check_ram(need)
    return {"device": "cpu"}


def _windows_cuda_dlls():
    """Python 3.8+ ignores PATH for extension DLLs on Windows: torch's own CUDA runtime (and the pip nvidia-* wheels)
    must be on the DLL search path before exllamav3's extension loads."""
    if sys.platform != "win32":
        return
    import glob
    cands = []
    try:
        import torch
        cands.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
        cands += glob.glob(os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "nvidia", "*", "bin"))
    except Exception:                                            # noqa: BLE001
        pass
    for v in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(v):
            cands.append(os.path.join(os.environ[v], "bin"))
    for d in cands:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass


class _Renderer:
    """The model's own chat template with the probe's `tools`: transformers' apply_chat_template when it is here (the
    renderer every HF harness uses), else pollard-modelkind's jinja2 renderer of the same template."""

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tok = tokenizer
        if self.tok is None and _has_module("transformers"):
            try:
                from transformers import AutoTokenizer
                self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=_trust(model))
            except Exception:                                    # noqa: BLE001 -- jinja2 below
                self.tok = None
        self.ct = None
        if self.tok is None or not getattr(self.tok, "chat_template", None):
            from pollard_modelkind import ChatTemplate, _template_and_arch, _special_tokens
            tpl, _, conf = _template_and_arch(model)
            if isinstance(tpl, list):
                tpl = next((t.get("template") for t in tpl if t.get("name") == "tool_use"), tpl[0].get("template"))
            if not tpl:
                raise NotMeasured("the checkpoint ships no chat template")
            self.ct = ChatTemplate(tpl, _special_tokens(model, conf))

    def __call__(self, messages, tools, template_kwargs=None):
        kw = dict(template_kwargs or {})
        for msgs in (messages, _fold_system(messages)):            # a template that refuses a system turn
            try:
                if self.ct is not None:
                    return self.ct.render(msgs, add_generation_prompt=True, tools=tools, **kw)
                return self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False,
                                                    **kw)
            except Exception as e:                               # noqa: BLE001
                err = e
        raise err


def _fold_system(messages):
    if not messages or messages[0].get("role") != "system":
        return messages
    sys_txt, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
    if rest and rest[0].get("role") == "user":
        rest[0]["content"] = f"{sys_txt}\n\n{rest[0]['content']}"
    return rest


class HFEngine:
    """transformers in this process: the checkpoint (bf16, or GPTQ / AWQ / FP8 / compressed-tensors through the
    loaders transformers dispatches to), greedy decoding, special tokens kept in the text (they ARE the call syntax)."""

    def __init__(self, model, device="cpu", threads=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(int(threads))
        trc = _trust(model)
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=trc)
        self.render = _Renderer(model, self.tok)
        kw = {"trust_remote_code": trc}
        if device == "cuda":
            kw["device_map"] = "cuda"
        import pollard_mxfp6
        if lane_of(model) == "exlr8":                            # EXLR8: the weights the stored codes decode to
            import pollard_exlr8 as X
            self.model = X.load_model(model, device, trust_remote_code=trc)
        elif pollard_mxfp6.read_meta(model):                     # pollard-mx MXFP6 / mix: its own bytes, decoded
            self.model = pollard_mxfp6.load_hf_model(model, "cuda" if device == "cuda" else "cpu", trc)
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model, dtype="auto", **kw)
            except TypeError:                                    # transformers 4.x spelling
                self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", **kw)
        if device in ("cpu", "mps"):
            self.model = self.model.to(device)
        self.model.eval()
        gc = getattr(self.model, "generation_config", None)
        eos = getattr(gc, "eos_token_id", None)
        eos = list(eos) if isinstance(eos, (list, tuple)) else ([eos] if eos is not None else [])
        if self.tok.eos_token_id is not None and self.tok.eos_token_id not in eos:
            eos.append(self.tok.eos_token_id)
        self.eos = eos
        self.eos_text = [self.tok.decode([e], skip_special_tokens=False) for e in eos]
        self.device = device

    def generate(self, prompt, max_tokens):
        import torch
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_tokens,
                                      do_sample=False, temperature=None, top_p=None, top_k=None,
                                      eos_token_id=self.eos or None,
                                      pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None
                                      else (self.eos[0] if self.eos else None))
        new = out[0, ids.shape[1]:].tolist()
        stopped = bool(new) and new[-1] in self.eos
        if stopped:
            new = new[:-1]
        text = _cut_at(self.tok.decode(new, skip_special_tokens=False), self.eos_text)
        return text, ("stop" if stopped or len(new) < max_tokens else "length")

    def close(self):
        import gc
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                        # noqa: BLE001
            pass


class EXL3Engine:
    """exllamav3 in this process (Config / Model / Cache / Tokenizer / Generator, as of exllamav3 1.4.x): the EXL3
    rung on one CUDA device, greedy (ArgmaxSampler), stopping on every EOS id the checkpoint declares (the generator
    stops on none by default), special tokens decoded (they ARE the call syntax)."""

    def __init__(self, model, ctx=8192):
        _windows_cuda_dlls()
        from exllamav3 import Cache, Config, Generator, Model, Tokenizer
        from exllamav3.generator.sampler import ArgmaxSampler
        self.config = Config.from_directory(model)
        self.model = Model.from_config(self.config)
        self.cache = Cache(self.model, max_num_tokens=max(256, (int(ctx) + 255) // 256 * 256))   # keep referenced
        self.model.load()
        self.tokenizer = Tokenizer.from_config(self.config)
        self.generator = Generator(model=self.model, cache=self.cache, tokenizer=self.tokenizer)
        self.sampler = ArgmaxSampler()
        stops = list(getattr(self.config, "eos_token_id_list", None) or [])
        if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id not in stops:
            stops.append(self.tokenizer.eos_token_id)
        self.stops = stops
        self.render = _Renderer(model)
        self.eos_text = [t for t in (getattr(self.tokenizer, "eos_token", None),) if t]

    def generate(self, prompt, max_tokens):
        out = self.generator.generate(prompt=prompt, max_new_tokens=max_tokens, sampler=self.sampler,
                                      encode_special_tokens=True, decode_special_tokens=True,
                                      stop_conditions=self.stops, add_bos=False, completion_only=True,
                                      return_last_results=True, seed=0)
        text, last = out if isinstance(out, tuple) else (out, {})
        reason = (last or {}).get("eos_reason")
        return _cut_at(text or "", self.eos_text), ("length" if reason == "max_new_tokens" else "stop")

    def close(self):
        try:
            self.model.unload()
        except Exception:                                        # noqa: BLE001
            pass
        self.generator = self.cache = self.model = None
        import gc
        gc.collect()


def _cut_at(text, stops):
    """Text up to the first end-of-turn string the decoder left in."""
    cut = min([text.find(s) for s in stops if s and s in text] or [len(text)])
    return text[:cut]


@contextlib.contextmanager
def inprocess_engine(backend, model, device="cuda", threads=None, ctx=8192):
    """The model loaded in this process for the duration, always released afterwards (one model at a time). A load
    that fails -- a loader missing a kernel for this GPU, out of memory -- is not-measured with its reason."""
    try:
        eng = EXL3Engine(model, ctx) if backend == "exl3" else HFEngine(model, device, threads)
    except NotMeasured:
        raise
    except Exception as e:                                       # noqa: BLE001
        raise NotMeasured(f"{BACKEND_SERVER[backend]} could not load {os.path.basename(str(model).rstrip('/'))}: "
                          f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}") from None
    try:
        yield eng
    finally:
        eng.close()


def reply_from_text(raw, family, tools=None, finish="stop", reasoning=None, prefill=""):
    """An in-process generation as the /v1/chat/completions reply a server would have returned: thinking split off
    (reasoning_content), calls read by the family's reader (tool_calls, arguments as a JSON string), the prose left as
    content -- so grade_reply grades it exactly like a served reply. `prefill` = the reasoning opener the prompt
    already ended with (the model's text then starts mid-thought)."""
    from pollard_modelkind import CALL_READERS, call_shaped, parse_tool_calls, split_reasoning
    text = (prefill or "") + (raw or "")
    rf = family if family in CALL_READERS else "json"
    if rf == "harmony":                                          # analysis channel = thinking; calls ride commentary
        think = " ".join(re.findall(r"<\|channel\|>\s*analysis\s*<\|message\|>(.*?)(?:<\|end\|>|$)", text, re.S))
        body = re.sub(r"<\|channel\|>\s*analysis\s*<\|message\|>.*?(?:<\|end\|>|$)", "", text, flags=re.S)
        fin = re.search(r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", body, re.S)
        calls, _, attempted = parse_tool_calls(body, rf, tools)
        answer = fin.group(1).strip() if fin else ("" if calls or think else body.strip())
        unclosed = bool(think) and not fin and not calls and "<|channel|>" not in body
    else:
        think, answer, unclosed = split_reasoning(text, reasoning)
        calls, content, attempted = ([], answer, call_shaped(answer)) if unclosed else \
            parse_tool_calls(answer, rf, tools)
        answer = content if calls else answer
    msg = {"role": "assistant", "content": text if unclosed else answer}
    if think and not unclosed:
        msg["reasoning_content"] = think
    if calls:
        msg["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                              "function": {"name": c["name"],
                                           "arguments": c["arguments"] if isinstance(c["arguments"], str)
                                           else json.dumps(c["arguments"], ensure_ascii=False)}}
                             for i, c in enumerate(calls)]
    return {"choices": [{"message": msg, "finish_reason": "tool_calls" if calls else finish}],
            "raw": raw}


def _prefill_for(prompt, reasoning):
    """The reasoning opener a rendered prompt already ends with (Qwen3.5 / R1-style templates open the thinking for the
    model), so the generated text is split as the continuation it is."""
    op = ((reasoning or {}).get("open") or "").strip()
    return op if op and (prompt or "").rstrip().endswith(op) else ""


def run_suite_inprocess(engine, probes=None, max_tokens=384, template_kwargs=None, progress=None, family=None,
                        reasoning=None):
    """The suite through an in-process engine: the SAME request a served backend gets (request_for), rendered by the
    model's template, decoded greedily, read back by the family's reader, graded by grade_reply."""
    rows = []
    for i, p in enumerate(probes or SUITE):
        body = request_for(p, max_tokens, template_kwargs)
        try:
            prompt = engine.render(body["messages"], body["tools"], template_kwargs)
            raw, finish = engine.generate(prompt, max_tokens)
            reply = reply_from_text(raw, family, body["tools"], finish, reasoning, _prefill_for(prompt, reasoning))
        except NotMeasured:
            raise
        except Exception as e:                                   # noqa: BLE001 -- one probe, recorded
            reply = {"error": f"{type(e).__name__}: {e}"[:200]}
        rows.append(grade_reply(reply, p))
        if progress:
            progress(i + 1, len(probes or SUITE), rows[-1])
    return rows


@contextlib.contextmanager
def backend_server(backend, model, ngl="0", threads=None, server=None, url=None, kind=None, ctx=4096):
    """The rung served by `backend` for the duration; yields its base url."""
    if backend == "url":
        if not url:
            raise NotMeasured("--backend url needs --url")
        yield re.sub(r"/v1/?$", "", url.rstrip("/"))
        return
    if backend == "gguf":
        with served(model, ngl, ctx, threads=threads, server=server) as base:
            yield base
        return
    port = _free_port()
    if backend == "mlx":
        py = os.environ.get("POLLARD_MLX_PYTHON") or sys.executable
        cmd = mlx_cmd(py, model, port, _trust(model))
    elif backend == "vllm":
        import shutil
        parser = ((kind or {}).get("tools") or {}).get("vllm_parser")
        cmd = vllm_cmd(server or shutil.which("vllm"), model, port, parser, ctx, _trust(model))
    else:
        raise NotMeasured(f"unknown backend {backend!r}")
    with _launch(cmd, f"{BACKEND_SERVER[backend]} on {model}", port=port) as base:
        yield base


def _post(base, payload, timeout=600):
    req = urllib.request.Request(base + "/v1/chat/completions", method="POST", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Connection": "close"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:                          # a parse failure surfaces as a 500 with a body
        try:
            return {"error": json.loads(e.read().decode("utf-8", "replace")).get("error", str(e))}
        except Exception:                                        # noqa: BLE001
            return {"error": str(e)}
    except Exception as e:                                       # noqa: BLE001
        return {"error": repr(e)[:200]}


def request_for(probe, max_tokens, template_kwargs=None, extra=None):
    """The OpenAI request one probe sends: the user turn, the tools on offer, greedy decoding."""
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": probe["prompt"]}],
            "tools": [TOOLBOX[t] for t in probe["tools"]], "tool_choice": "auto", "parallel_tool_calls": False,
            "temperature": 0, "top_k": 1, "seed": 0, "max_tokens": max_tokens}
    if template_kwargs:
        body["chat_template_kwargs"] = template_kwargs
    if extra:
        body.update(extra)
    return body


def run_suite(base, probes=None, max_tokens=384, template_kwargs=None, progress=None, extra=None):
    rows = []
    for i, p in enumerate(probes or SUITE):
        reply = _post(base, request_for(p, max_tokens, template_kwargs, extra))
        rows.append(grade_reply(reply, p))
        if progress:
            progress(i + 1, len(probes or SUITE), rows[-1])
    return rows


def suite_hash(probes=None, extra=""):
    blob = json.dumps([probes or SUITE, TOOLBOX, SYSTEM, extra], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def fingerprint(path):
    """size:mtime of a file; of a checkpoint dir, its weight bytes and newest weight mtime."""
    p = str(path or "")
    if os.path.isfile(p):
        st = os.stat(p)
        return f"{st.st_size}:{int(st.st_mtime)}"
    size, newest = 0, 0
    for root, _, files in os.walk(p):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".gguf", ".json")):
                st = os.stat(os.path.join(root, f))
                size, newest = size + st.st_size, max(newest, int(st.st_mtime))
    return f"{size}:{newest}"


def _ref_cache_path(ref, key):
    ref = str(ref).rstrip("/\\")
    if os.path.isdir(ref):                                       # never write into an HF cache snapshot
        try:
            import pollard_workspace as ws
            d = os.path.join(ws.cache_dir(create=True), "toolcall")
        except Exception:                                        # noqa: BLE001
            d = os.path.join(os.path.dirname(os.path.abspath(ref)), ".pollard-cache")
        tag = hashlib.sha256(os.path.abspath(ref).encode()).hexdigest()[:8]
        return os.path.join(d, f"{os.path.basename(ref)}-{tag}.toolcall-{key}.json")
    d = os.path.join(os.path.dirname(os.path.abspath(ref)), ".pollard-cache")
    return os.path.join(d, f"{os.path.basename(ref)}.toolcall-{key}.json")


def effective_backend(backend, server=None, model=None):
    """(backend, note): vLLM where a usable one is installed; where it is not (Windows' kernel-less source build, not
    installed) the in-process transformers backend measures the same checkpoint instead, and the note says why. A
    pollard-mx MXFP6 / mix rung this vLLM cannot load goes in-process too (decoded from its own bytes)."""
    if backend != "vllm":
        return backend, None
    ok, why = vllm_status(server)
    if ok and model and os.path.isdir(str(model)):
        import pollard_mxfp6
        why = pollard_mxfp6.vllm_load_problem(model)
        ok = not why
    if ok:
        return backend, None
    return "hf", f"{why} -- measured in-process with transformers instead"


def run_model(model, ngl, threads=None, server=None, max_tokens=0, no_think=False, probes=None, quiet=False,
              backend="gguf", url=None):
    """Run the suite on one model with `backend` (served, or in this process). Returns (rows, meta); NotMeasured when
    this machine cannot run it that way. meta["backend"] is the backend that actually ran (vllm -> hf fallback)."""
    kind = {}
    if model and os.path.exists(str(model)):
        try:
            from pollard_modelkind import classify
            kind = classify(model)
        except (Exception, SystemExit):                          # noqa: BLE001
            kind = {}
    backend, note = effective_backend(backend, server, model)
    extra = preflight(backend, model, ngl, kind, server) if backend != "url" else {}
    thinking = bool(kind.get("thinking"))
    budget = max_tokens or (N_PREDICT_REASONING if thinking and not no_think else 384)
    tkw = None
    if no_think and thinking:
        tog = ((kind.get("reasoning") or {}).get("toggle") or {})
        tkw = dict(tog.get("off") or {}) or None
    fam = (kind.get("tools") or {}).get("family")
    if not quiet:
        if note:
            print(f"   ({note})", flush=True)
        where = f" on {extra['device']}" if backend in INPROCESS and extra.get("device") else ""
        print(f"   {os.path.basename(str(model or url).rstrip('/'))}: {BACKEND_SERVER.get(backend, backend)}{where}, "
              f"tool syntax {fam or '?'}, budget {budget} tokens" + (", thinking off" if tkw else ""), flush=True)

    def prog(i, n, row):
        if not quiet and (i % 10 == 0 or i == n):
            print(f"     {i}/{n}", flush=True)
    meta = {"family": fam, "budget": budget, "thinking": thinking, "template_kwargs": tkw, "backend": backend}
    if note:
        meta["backend_note"] = note
    if backend in INPROCESS:
        rfam, rkey = reader_family(kind)
        meta.update(reader=rkey, device=extra.get("device"))
        with inprocess_engine(backend, model, extra.get("device", "cuda"), threads) as eng:
            rows = run_suite_inprocess(eng, probes, budget, tkw, prog, rkey, kind.get("reasoning"))
        return rows, meta
    with backend_server(backend, model, ngl, threads, server, url, kind) as base:
        rows = run_suite(base, probes, budget, tkw, prog, extra)
    return rows, meta


def run_ref_cached(ref, ngl, threads=None, server=None, max_tokens=0, no_think=False, probes=None, refresh=False,
                   backend="gguf", url=None):
    """The reference's rows, cached: every rung of a ladder is compared to the same reference, so it runs once."""
    if backend == "url":
        return run_model(ref, ngl, threads, server, max_tokens, no_think, probes, backend=backend, url=url)
    backend, _ = effective_backend(backend, server, ref)        # key the cache on what actually runs
    key = suite_hash(probes, f"{fingerprint(ref)}:{max_tokens}:{no_think}:{backend}")
    path = _ref_cache_path(ref, key)
    if not refresh and os.path.isfile(path):
        try:
            d = json.load(open(path, encoding="utf-8"))
            print(f"   reference: cached ({path})")
            return d["rows"], d.get("meta", {})
        except (ValueError, OSError, KeyError):
            pass
    rows, meta = run_model(ref, ngl, threads, server, max_tokens, no_think, probes, backend=backend)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"rows": rows, "meta": meta}, open(path, "w", encoding="utf-8"), indent=1)
    except OSError:
        pass
    return rows, meta


def _pct(x):
    return "--" if x is None else f"{100 * x:.0f}%"


def print_board(name, s, rows=None, verbose=False):
    lo, hi = s["rate_ci"]
    print(f"\n  {name}")
    print(f"    success      {s['valid']}/{s['n']}  {_pct(s['rate'])}  (95% CI {_pct(lo)}-{_pct(hi)})")
    print(f"    parse {_pct(s['parse_rate'])}  function {_pct(s['function_acc'])}  args valid "
          f"{_pct(s['args_valid'])}  exact {_pct(s['exact'])}  abstention {_pct(s['abstention'])}")
    print("    by category  " + "  ".join(f"{c} {_pct(v)}" for c, v in s["categories"].items()))
    if s["modes"]:
        print(f"    failures     {dict(sorted(s['modes'].items(), key=lambda kv: -kv[1]))}")
    if verbose and rows:
        for r in rows:
            if not r["ok"]:
                print(f"      FAIL {r['id']:<4} {r['failure']:<14} {r['detail'][:70]}  | {r['sample'][:60]}")


def _balanced(limit):
    """The first `limit` probes of each category -- a smaller suite that still covers all four."""
    if not limit:
        return list(SUITE)
    out = []
    for c in CATEGORIES:
        out += [p for p in SUITE if p["category"] == c][:limit]
    return out


def _advice():
    print("\n  Tool-call formatting is concentrated in a narrow band of layers, so a drop this size\n"
          "  is usually a few tensors rather than the whole model. Run pollard-sensitivity on\n"
          "  agent-profile text to find which, then protect them with pollard-automap instead of\n"
          "  raising the bit-width everywhere.")


def main_quick(a):
    """The original check: 4 probes, plain prompt, llama-cli. Fast, template-free, base-model friendly."""
    cli = a.llama_cli
    budget = {"n_predict": a.n_predict} if a.n_predict else {}
    rows = run(cli, a.gguf, a.ngl, a.threads, **budget)
    s = score(rows)
    print(f"== pollard-toolcall --quick :: {os.path.basename(a.gguf)}")
    for r in rows:
        mark = "ok  " if r["ok"] else "FAIL"
        print(f"  {mark} {r['tool']:<14} {r['detail']}")
        if not r["ok"]:
            print(f"       got: {r['sample']}")
    print(f"\n  format validity {s['valid']}/{s['n']}  ({s['rate']:.0%})")
    if s["modes"]:
        print(f"  failures: {s['modes']}")
    verdict = s["rate"] >= a.min_rate
    rs = None
    if a.ref:
        ref_rows = run(cli, a.ref, a.ngl, a.threads, **budget)
        rs = score(ref_rows)
        drop = rs["rate"] - s["rate"]
        print(f"  reference       {rs['valid']}/{rs['n']}  ({rs['rate']:.0%})"
              f"   DROP {drop:+.0%}")
        if drop > 0.1:
            _advice()
        verdict = verdict and drop <= 0.1
    if a.out:
        json.dump({"score": s, "rows": rows, "ref_score": rs}, open(a.out, "w"), indent=2)
    if a.results:
        import pollard_workspace as ws
        tc = {"pass": bool(verdict), "rate": s["rate"], "n": s["n"], "modes": s["modes"], "mode": "quick"}
        if rs is not None:
            tc.update({"ref_rate": rs["rate"], "drop": round(rs["rate"] - s["rate"], 4)})
        ws.merge_results(a.results, a.rung or os.path.basename(a.gguf), {"toolcall": tc})
    print(f"\n{'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


def cross_backend(backend, ref_backend):
    """Was the reference read a different way from the rung? Two in-process backends (exl3 rung, hf reference) render
    with the same template code and read with the same reader: same path, not cross-backend."""
    if not ref_backend or ref_backend == backend:
        return False
    return not (backend in INPROCESS and ref_backend in INPROCESS)


def _same(rows, ref_rows):
    """Probes where the rung and the f16 had the same outcome: retention can sit at 100% while the rung trades
    false calls for missed ones, and this is the number that shows it."""
    ref = {q["id"]: q["ok"] for q in ref_rows}
    return sum(1 for r in rows if r["id"] in ref and ref[r["id"]] == r["ok"])


def verdict_for(s, rs=None, ret=None, min_rate=0.75, min_retention=90.0):
    """PASS/FAIL. With a reference the gate is retention (the rung against its own f16 -- a small model that was
    never good at tools is not failed for that); without one it is the absolute success rate."""
    if rs is not None and ret is not None:
        return 100.0 * ret >= min_retention
    if rs is not None and not rs["valid"]:
        return True                                        # the f16 passes nothing: there is nothing to retain
    return (s["rate"] or 0.0) >= min_rate


def _server_for(backend, a):
    """The server binary override that applies to this backend (--llama-server / --vllm)."""
    return {"gguf": a.llama_server, "vllm": getattr(a, "vllm", None)}.get(backend)


def record_not_measured(a, rung, backend, reason, lane=None):
    """A rung this machine could not measure: said so in results.json, never a fake number."""
    if a.results:
        import pollard_workspace as ws
        tc = {"status": "not-measured", "reason": str(reason), "backend": backend, "mode": "full"}
        if lane:
            tc["lane"] = lane
        ws.merge_results(a.results, rung, {"toolcall": tc})


def _resolve_ref(a, backend, kw):
    """(ref_rows, ref_meta, ref_backend, note) -- the reference served by the rung's own backend when it can be; else
    the --ref-fallback (the f16 GGUF through llama-server), labelled cross-backend; else no reference at all."""
    if not (a.ref or a.ref_url):
        return None, None, None, None
    rb = a.ref_backend if a.ref_backend != "auto" else pick_backend(a.ref, prefer=backend, url=a.ref_url)
    try:
        rows, meta = run_ref_cached(a.ref, a.ngl, refresh=a.refresh_ref, backend=rb, url=a.ref_url,
                                    **dict(kw, server=_server_for(rb, a)))
        return rows, meta, meta.get("backend", rb), None
    except NotMeasured as e:
        why = f"{BACKEND_SERVER.get(rb, rb)} reference not measured: {e}"
        print(f"   ({why})")
    if a.ref_fallback and os.path.exists(a.ref_fallback):
        fb = pick_backend(a.ref_fallback)
        try:
            rows, meta = run_ref_cached(a.ref_fallback, a.ngl, refresh=a.refresh_ref, backend=fb,
                                        **dict(kw, server=_server_for(fb, a)))
            fb = meta.get("backend", fb)
            return rows, meta, fb, f"{why}; reference = {os.path.basename(a.ref_fallback)} via {BACKEND_SERVER[fb]}"
        except NotMeasured as e:
            why += f"; fallback {os.path.basename(a.ref_fallback)} not measured either: {e}"
            print(f"   ({why})")
    return None, None, None, why


def main_full(a):
    probes = _balanced(a.limit)
    model = a.model or a.gguf
    backend = a.backend if a.backend != "auto" else pick_backend(model, url=a.url)
    rung = a.rung or os.path.basename(str(model or "served").rstrip("/\\"))
    lane = lane_of(model) if model else None
    print(f"== pollard-toolcall :: {rung}  ({len(probes)} probes, {BACKEND_SERVER.get(backend, backend)})")
    kw = dict(threads=a.threads, max_tokens=a.n_predict, no_think=a.no_think, probes=probes)
    try:
        rows, meta = run_model(model, a.ngl, backend=backend, url=a.url, server=_server_for(backend, a), **kw)
    except NotMeasured as e:
        print(f"\n  NOT MEASURED ({backend}): {e}\n  -- recorded as such; nothing here is a tool-call result.")
        record_not_measured(a, rung, backend, e, lane)
        return 0
    backend = meta.get("backend", backend)
    s = score_suite(rows)
    print_board(rung, s, rows, a.verbose)
    ref_rows, ref_meta, ref_backend, ref_note = _resolve_ref(a, backend, kw)
    rs, ret = None, (None, None, None)
    if ref_rows is not None:
        rs = score_suite(ref_rows)
        ref_name = os.path.basename(str((a.ref_fallback if ref_note else None) or a.ref or a.ref_url).rstrip("/\\"))
        print_board(f"{ref_name}  (reference, {BACKEND_SERVER.get(ref_backend, ref_backend)})", rs, ref_rows,
                    a.verbose)
        ret = retention_ci(rows, ref_rows)
        if ret[0] is not None:
            print(f"\n  RETENTION {100 * ret[0]:.1f}%  (95% paired-bootstrap CI {100 * ret[1]:.1f}-{100 * ret[2]:.1f}%)"
                  f"   success {_pct(s['rate'])} vs {_pct(rs['rate'])}")
            print(f"  same outcome as the reference on {_same(rows, ref_rows)}/{len(rows)} probes")
            if cross_backend(backend, ref_backend):
                print(f"  (cross-backend: rung via {BACKEND_SERVER.get(backend, backend)}, reference via "
                      f"{BACKEND_SERVER.get(ref_backend, ref_backend)} -- part of any gap is the server, not the quant)")
        else:
            print("\n  (the reference passes no probe: nothing to retain -- retention is undefined)")
    elif ref_note:
        print(f"\n  no reference measured ({ref_note}): the gate is the absolute success rate")
    ok = verdict_for(s, rs, ret[0], a.min_rate, a.min_retention)
    if not ok and ret[0] is not None:
        _advice()
    if a.out:
        json.dump({"score": s, "rows": rows, "ref_score": rs, "ref_rows": ref_rows, "meta": meta,
                   "retention": ret[0], "retention_ci": [ret[1], ret[2]], "backend": backend,
                   "ref_backend": ref_backend, "ref_note": ref_note}, open(a.out, "w"), indent=2)
    if a.results:
        import pollard_workspace as ws
        tc = {"status": "measured", "pass": bool(ok), "mode": "full", "backend": backend,
              "rate": s["rate"], "rate_ci": s["rate_ci"], "n": s["n"],
              "modes": s["modes"], "parse_rate": s["parse_rate"], "function_acc": s["function_acc"],
              "args_valid": s["args_valid"], "exact": s["exact"], "abstention": s["abstention"],
              "categories": s["categories"], "family": meta.get("family")}
        if meta.get("backend_note"):
            tc["backend_note"] = meta["backend_note"]
        if meta.get("reader"):
            tc["reader"] = meta["reader"]
        if lane:
            tc["lane"] = lane
        if rs is not None:
            tc.update({"ref_rate": rs["rate"], "ref_rate_ci": rs["rate_ci"], "ref_backend": ref_backend,
                       "cross_backend": cross_backend(backend, ref_backend),
                       "drop": round((rs["rate"] or 0) - (s["rate"] or 0), 4)})
        if ref_note:
            tc["ref_note"] = ref_note
        if ref_rows is not None:
            tc["agreement"] = round(_same(rows, ref_rows) / max(len(rows), 1), 4)
        if ret[0] is not None:
            tc.update({"retention": round(100 * ret[0], 2),
                       "retention_ci": [round(100 * ret[1], 2), round(100 * ret[2], 2)]})
        ws.merge_results(a.results, rung, {"toolcall": tc})
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--model", help="the build to check: a GGUF file or a lane's checkpoint dir (MLX / GPTQ / AWQ / "
                                    "MX / EXL3)")
    ap.add_argument("--gguf", help="the GGUF build to check (same as --model; --quick needs a GGUF)")
    ap.add_argument("--backend", default="auto", choices=("auto",) + BACKENDS,
                    help="how the build is run: gguf (llama-server --jinja), mlx (mlx_lm.server), vllm (vllm serve "
                         "--tool-call-parser <family>; falls back to hf where no usable vLLM is installed), exl3 "
                         "(exllamav3 in-process), hf (transformers in-process), url (already running: TabbyAPI, ...). "
                         "auto (default) = from the file / the MANIFEST lane / config.json")
    ap.add_argument("--url", help="--backend url: the OpenAI-compatible server's base url")
    ap.add_argument("--ref", help="reference (the f16 GGUF, or the bf16 HF checkout for a directory lane), to report "
                                  "RETENTION rather than a bare rate")
    ap.add_argument("--ref-backend", default="auto", choices=("auto",) + BACKENDS,
                    help="the reference's backend (auto: the rung's own backend whenever it can load the reference; "
                         "hf for an EXL3 rung)")
    ap.add_argument("--ref-url", help="--ref-backend url: where the reference is already served")
    ap.add_argument("--ref-fallback", help="when the reference cannot be served by the rung's backend here (no GPU, "
                                           "too big): use this one instead (the f16 GGUF) -- labelled cross-backend")
    ap.add_argument("--quick", action="store_true",
                    help="the old 4-probe plain-prompt check through llama-cli (no chat template, no server)")
    ap.add_argument("--ngl", default="0", help="GPU layers for llama-server (a number, or auto; default 0 -- shared "
                                               "boxes: the GPU is not ours unless this says so)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--llama-cli", default="llama-cli", help="--quick: the llama-cli binary")
    ap.add_argument("--llama-server", default=None, help="the llama-server binary (default: the runtime build/PATH)")
    ap.add_argument("--vllm", default=None, help="the vllm binary (default: vllm on PATH)")
    ap.add_argument("--min-rate", type=float, default=0.75,
                    help="without --ref: fail below this success rate (default 0.75)")
    ap.add_argument("--min-retention", type=float, default=90.0,
                    help="with --ref: fail below this retention vs the f16, percent (default 90)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N probes of each category (a smaller balanced suite)")
    ap.add_argument("--no-think", action="store_true",
                    help="a reasoning model with a thinking switch: run it with thinking off")
    ap.add_argument("--refresh-ref", action="store_true", help="re-run the reference instead of its cached rows")
    ap.add_argument("--verbose", "-v", action="store_true", help="list every failed probe")
    ap.add_argument("--out", help="write the full result here as JSON")
    ap.add_argument("--results", help="merge this rung's numbers into this results.json "
                                      "(what pollard-card --results and pollard-verify --gate read)")
    ap.add_argument("--rung", help="the rung's name/tag in --results (default: the build's file / folder name)")
    ap.add_argument("--n-predict", type=int, default=0,
                    help=f"generation budget (default: 384, or {N_PREDICT_REASONING} for a reasoning model; "
                         f"--quick: {N_PREDICT_PLAIN} / {N_PREDICT_REASONING})")
    a = ap.parse_args(argv)
    if a.quick:
        a.gguf = a.gguf or a.model
        if not a.gguf:
            ap.error("--quick needs --gguf")
        raise SystemExit(main_quick(a))
    if not (a.model or a.gguf or a.url):
        ap.error("pass --model (or --gguf) for the build to check, or --url for one already served")
    if a.ref_backend == "url" and not a.ref_url:
        ap.error("--ref-backend url needs --ref-url")
    raise SystemExit(main_full(a))

if __name__ == "__main__":
    main()
