#!/usr/bin/env python3
"""pollard-toolcall -- can this build still call tools the way it was trained to?

Standard benchmarks report 4-bit as near-lossless, and that verdict rests almost entirely on
single-turn scoring. Multi-turn tool use is where the same error amplifies, and the capability
that breaks first is narrow: emitting a syntactically correct call with the right name and the
right argument types. It is a formatting skill concentrated in a small band of layers, which makes
it unusually fragile to quantization and unusually cheap to protect once you know it went.

The full gate (default) runs the REAL serving path: the rung is served with `llama-server --jinja`
and asked through /v1/chat/completions with OpenAI-style `tools`, so the model's own chat template
renders the tool list, the model answers in its own call syntax (Hermes, Qwen3-Coder XML, LFM2
pythonic, harmony, Mistral... -- pollard-modelkind names it) and llama.cpp's own parser extracts the
call. Sixty probes in four categories:

    single    one tool on offer, typed arguments (string / integer / number / boolean / enum)
    choose    eight tools on offer: pick the right one and fill it
    abstain   tools on offer, none needed: the right answer is prose
    nested    several arguments, nested objects and arrays

scored as parse rate, right function, arguments schema-valid, arguments exact and abstention, with
Wilson CIs. Against the f16 the gate number is RETENTION (rung success / f16 success) with a paired
bootstrap CI; the f16's run is cached beside it, so a ladder runs it once. A reasoning model's
thinking is kept out of the grade (pollard-modelkind split_reasoning).

    pollard-toolcall --gguf build.gguf --ref f16.gguf              # full gate, retention vs f16
    pollard-toolcall --gguf build.gguf --ref f16.gguf --quick      # the old 4-probe plain-prompt check

What --quick counts as a failure, in the order they actually happen:

    unclosed-think a reasoning model was still thinking when the budget ran out
    no-call        the model answered in prose instead of calling anything
    not-json       it emitted something call-shaped that does not parse
    wrong-tool     valid JSON, wrong function name
    missing-arg    a required argument absent
    bad-type       an argument present with the wrong type
    hallucinated   an argument the schema does not define

The full gate adds: unparsed (call-shaped text the server's parser rejected), bad-enum, bad-args,
hallucinated-tool, wrong-value and false-call (a tool called where none was needed).

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
    """Grade one /v1/chat/completions reply against one probe. Returns a row:
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
    looks_like_call = bool(re.search(r"<tool_call>|\[TOOL_CALLS\]|<\|tool_call|\"name\"\s*:|<function=|"
                                     r"\w+\([^()]*=", text))
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
            row.update(failure="unparsed", detail="call-shaped text the server's parser did not accept")
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


@contextlib.contextmanager
def served(model, ngl, ctx=4096, threads=None, server=None, log_path=None):
    """llama-server --jinja for the duration; yields its base url. --jinja is what makes this the real path: the
    model's own template renders the tools, and llama.cpp's own parser (with its lazy grammar) reads the call."""
    from pollard_calc import find_llama_bin
    binary = server or find_llama_bin("llama-server") or "llama-server"
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    cmd = [binary, "-m", model, "--host", "127.0.0.1", "--port", str(port), "-ngl", str(ngl), "-c", str(ctx),
           "-np", "1",                       # one slot: deterministic, and no KV pool shared four ways
           "--jinja",
           "--reasoning-format", "deepseek"]  # thinking goes to reasoning_content, not into the call parser
    if threads:
        cmd += ["-t", str(threads)]
    log = open(log_path or _SERVER_LOG, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(900):
            if proc.poll() is not None:
                raise SystemExit(f"llama-server exited ({proc.returncode}) before serving {model}; "
                                 f"see {log.name}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:                                    # noqa: BLE001 -- still loading
                time.sleep(1)
        else:
            raise SystemExit(f"llama-server never became ready for {model}; see {log.name}")
        yield base
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:                                    # noqa: BLE001
                proc.kill()
        log.close()


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


def request_for(probe, max_tokens, template_kwargs=None):
    """The OpenAI request one probe sends: the user turn, the tools on offer, greedy decoding."""
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": probe["prompt"]}],
            "tools": [TOOLBOX[t] for t in probe["tools"]], "tool_choice": "auto", "parallel_tool_calls": False,
            "temperature": 0, "top_k": 1, "seed": 0, "max_tokens": max_tokens}
    if template_kwargs:
        body["chat_template_kwargs"] = template_kwargs
    return body


def run_suite(base, probes=None, max_tokens=384, template_kwargs=None, progress=None):
    rows = []
    for i, p in enumerate(probes or SUITE):
        reply = _post(base, request_for(p, max_tokens, template_kwargs))
        rows.append(grade_reply(reply, p))
        if progress:
            progress(i + 1, len(probes or SUITE), rows[-1])
    return rows


def suite_hash(probes=None, extra=""):
    blob = json.dumps([probes or SUITE, TOOLBOX, SYSTEM, extra], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _ref_cache_path(ref, key):
    d = os.path.join(os.path.dirname(os.path.abspath(ref)), ".pollard-cache")
    return os.path.join(d, f"{os.path.basename(ref)}.toolcall-{key}.json")


def run_model(model, ngl, threads=None, server=None, max_tokens=0, no_think=False, probes=None, quiet=False):
    """Serve one GGUF and run the suite on it. Returns (rows, meta)."""
    kind = {}
    try:
        from pollard_modelkind import classify
        kind = classify(model)
    except Exception:                                            # noqa: BLE001
        kind = {}
    thinking = bool(kind.get("thinking"))
    budget = max_tokens or (N_PREDICT_REASONING if thinking and not no_think else 384)
    tkw = None
    if no_think and thinking:
        tog = ((kind.get("reasoning") or {}).get("toggle") or {})
        tkw = dict(tog.get("off") or {}) or None
    fam = (kind.get("tools") or {}).get("family")
    if not quiet:
        print(f"   {os.path.basename(model)}: tool syntax {fam or '?'}, budget {budget} tokens"
              + (", thinking off" if tkw else ""), flush=True)

    def prog(i, n, row):
        if not quiet and (i % 10 == 0 or i == n):
            print(f"     {i}/{n}", flush=True)
    with served(model, ngl, threads=threads, server=server) as base:
        rows = run_suite(base, probes, budget, tkw, prog)
    return rows, {"family": fam, "budget": budget, "thinking": thinking, "template_kwargs": tkw}


def run_ref_cached(ref, ngl, threads=None, server=None, max_tokens=0, no_think=False, probes=None, refresh=False):
    """The f16's rows, cached beside it: every rung of a ladder is compared to the same f16, so it runs once."""
    st = os.stat(ref)
    key = suite_hash(probes, f"{st.st_size}:{int(st.st_mtime)}:{max_tokens}:{no_think}")
    path = _ref_cache_path(ref, key)
    if not refresh and os.path.isfile(path):
        try:
            d = json.load(open(path, encoding="utf-8"))
            print(f"   reference: cached ({path})")
            return d["rows"], d.get("meta", {})
        except (ValueError, OSError, KeyError):
            pass
    rows, meta = run_model(ref, ngl, threads, server, max_tokens, no_think, probes)
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


def verdict_for(s, rs=None, ret=None, min_rate=0.75, min_retention=90.0):
    """PASS/FAIL. With a reference the gate is retention (the rung against its own f16 -- a small model that was
    never good at tools is not failed for that); without one it is the absolute success rate."""
    if rs is not None and ret is not None:
        return 100.0 * ret >= min_retention
    if rs is not None and not rs["valid"]:
        return True                                        # the f16 passes nothing: there is nothing to retain
    return (s["rate"] or 0.0) >= min_rate


def main_full(a):
    probes = _balanced(a.limit)
    print(f"== pollard-toolcall :: {os.path.basename(a.gguf)}  ({len(probes)} probes, llama-server --jinja)")
    kw = dict(threads=a.threads, server=a.llama_server, max_tokens=a.n_predict, no_think=a.no_think, probes=probes)
    rows, meta = run_model(a.gguf, a.ngl, **kw)
    s = score_suite(rows)
    print_board(os.path.basename(a.gguf), s, rows, a.verbose)
    rs, ref_rows, ret = None, None, (None, None, None)
    if a.ref:
        ref_rows, _ = run_ref_cached(a.ref, a.ngl, refresh=a.refresh_ref, **kw)
        rs = score_suite(ref_rows)
        print_board(os.path.basename(a.ref) + "  (reference)", rs, ref_rows, a.verbose)
        ret = retention_ci(rows, ref_rows)
        if ret[0] is not None:
            print(f"\n  RETENTION {100 * ret[0]:.1f}%  (95% paired-bootstrap CI {100 * ret[1]:.1f}-{100 * ret[2]:.1f}%)"
                  f"   success {_pct(s['rate'])} vs {_pct(rs['rate'])}")
            same = sum(1 for r in rows for q in ref_rows if q["id"] == r["id"] and q["ok"] == r["ok"])
            print(f"  same outcome as the f16 on {same}/{len(rows)} probes")
        else:
            print("\n  (the reference passes no probe: nothing to retain -- retention is undefined)")
    ok = verdict_for(s, rs, ret[0], a.min_rate, a.min_retention)
    if not ok and ret[0] is not None:
        _advice()
    if a.out:
        json.dump({"score": s, "rows": rows, "ref_score": rs, "ref_rows": ref_rows, "meta": meta,
                   "retention": ret[0], "retention_ci": [ret[1], ret[2]]}, open(a.out, "w"), indent=2)
    if a.results:
        import pollard_workspace as ws
        tc = {"pass": bool(ok), "mode": "full", "rate": s["rate"], "rate_ci": s["rate_ci"], "n": s["n"],
              "modes": s["modes"], "parse_rate": s["parse_rate"], "function_acc": s["function_acc"],
              "args_valid": s["args_valid"], "exact": s["exact"], "abstention": s["abstention"],
              "categories": s["categories"], "family": meta.get("family")}
        if rs is not None:
            tc.update({"ref_rate": rs["rate"], "ref_rate_ci": rs["rate_ci"],
                       "drop": round((rs["rate"] or 0) - (s["rate"] or 0), 4)})
        if ret[0] is not None:
            tc.update({"retention": round(100 * ret[0], 2),
                       "retention_ci": [round(100 * ret[1], 2), round(100 * ret[2], 2)]})
        ws.merge_results(a.results, a.rung or os.path.basename(a.gguf), {"toolcall": tc})
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--gguf", required=True, help="the build to check")
    ap.add_argument("--ref", help="reference build (the f16), to report RETENTION rather than a bare rate")
    ap.add_argument("--quick", action="store_true",
                    help="the old 4-probe plain-prompt check through llama-cli (no chat template, no server)")
    ap.add_argument("--ngl", default="0", help="GPU layers (a number, or auto; default 0 -- shared boxes)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--llama-cli", default="llama-cli", help="--quick: the llama-cli binary")
    ap.add_argument("--llama-server", default=None, help="the llama-server binary (default: the runtime build/PATH)")
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
    ap.add_argument("--rung", help="the rung's name/tag in --results (default: the GGUF file name)")
    ap.add_argument("--n-predict", type=int, default=0,
                    help=f"generation budget (default: 384, or {N_PREDICT_REASONING} for a reasoning model; "
                         f"--quick: {N_PREDICT_PLAIN} / {N_PREDICT_REASONING})")
    a = ap.parse_args(argv)
    raise SystemExit(main_quick(a) if a.quick else main_full(a))


if __name__ == "__main__":
    main()
