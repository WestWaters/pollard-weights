#!/usr/bin/env python3
"""pollard-toolcall -- can this build still emit a valid tool call?

Standard benchmarks report 4-bit as near-lossless, and that verdict rests almost entirely on
single-turn scoring. Multi-turn tool use is where the same error amplifies, and the capability
that breaks first is narrow: emitting a syntactically correct call with the right name and the
right argument types. It is a formatting skill concentrated in a small band of layers, which makes
it unusually fragile to quantization and unusually cheap to protect once you know it went.

Full agentic evaluation needs a live environment and a user simulator (see pollard-taskeval
--suite agentic, which says so rather than pretending). This measures the part that does NOT need
one, and it is the part that fails first.

What counts as a failure here, in the order they actually happen:

    no-call        the model answered in prose instead of calling anything
    not-json       it emitted something call-shaped that does not parse
    wrong-tool     valid JSON, wrong function name
    missing-arg    a required argument absent
    bad-type       an argument present with the wrong type
    hallucinated   an argument the schema does not define

    pollard-toolcall --gguf build.gguf --ref f16.gguf

A drop in format validity between the reference and the build is a direct measurement of the
fragile band, and pollard-sensitivity can then say which layers carry it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

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


def check_call(text, tool, expect=None):
    """Grade one generation against one tool schema. Returns (ok, failure, detail)."""
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


def run(cli_bin, model, ngl, threads=None):
    out = []
    for probe in PROBES:
        gen = generate(cli_bin, model, _prompt_for(probe), ngl, threads=threads)
        ok, failure, detail = check_call(gen, probe["tool"], probe.get("expect"))
        out.append({"tool": probe["tool"]["name"], "ok": ok, "failure": failure,
                    "detail": detail, "sample": gen.replace("\n", " ")[:110]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--gguf", required=True, help="the build to check")
    ap.add_argument("--ref", help="reference build, to report the DROP rather than a bare rate")
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--llama-cli", default="llama-cli")
    ap.add_argument("--min-rate", type=float, default=0.75,
                    help="fail below this format-validity rate (default 0.75)")
    ap.add_argument("--out", help="write the full result here as JSON")
    a = ap.parse_args()

    cli = a.llama_cli
    rows = run(cli, a.gguf, a.ngl, a.threads)
    s = score(rows)
    print(f"== pollard-toolcall :: {os.path.basename(a.gguf)}")
    for r in rows:
        mark = "ok  " if r["ok"] else "FAIL"
        print(f"  {mark} {r['tool']:<14} {r['detail']}")
        if not r["ok"]:
            print(f"       got: {r['sample']}")
    print(f"\n  format validity {s['valid']}/{s['n']}  ({s['rate']:.0%})")
    if s["modes"]:
        print(f"  failures: {s['modes']}")

    verdict = s["rate"] >= a.min_rate
    if a.ref:
        ref_rows = run(cli, a.ref, a.ngl, a.threads)
        rs = score(ref_rows)
        drop = rs["rate"] - s["rate"]
        print(f"  reference       {rs['valid']}/{rs['n']}  ({rs['rate']:.0%})"
              f"   DROP {drop:+.0%}")
        if drop > 0.1:
            print("\n  Tool-call formatting is concentrated in a narrow band of layers, so a drop "
                  "this size\n  is usually a few tensors rather than the whole model. Run "
                  "pollard-sensitivity to find\n  which, then protect them with pollard-automap "
                  "instead of raising the bit-width everywhere.")
        verdict = verdict and drop <= 0.1
    if a.out:
        json.dump({"score": s, "rows": rows}, open(a.out, "w"), indent=2)
    print(f"\n{'PASS' if verdict else 'FAIL'}")
    raise SystemExit(0 if verdict else 1)


if __name__ == "__main__":
    main()
