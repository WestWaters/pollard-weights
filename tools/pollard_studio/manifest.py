#!/usr/bin/env python3
"""Build a manifest of every Pollard tool and every flag, by READING the tools.

Hand-writing a screen per tool does not survive contact with ~60 tools: the first flag anyone adds
makes the UI a liar, and nothing fails when it happens. So the UI never hardcodes a control -- it
renders this, and this is derived from the source.

Derived by parsing the AST, not by importing. Importing a tool runs its module-level code, which
for several of these means pulling in torch; a manifest builder that needs the heavy extras
installed to describe a tool is a manifest builder nobody runs.

    python manifest.py ~/Desktop/Pollard-Weights/tools > manifest.json
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys


def _const(node):
    """Best-effort literal, so defaults and choices survive into the manifest."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def parse_tool(path: pathlib.Path):
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    doc = (ast.get_docstring(tree) or "").strip()
    summary = doc.splitlines()[0] if doc else ""
    flags = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        names = [a.value for a in node.args if isinstance(a, ast.Constant)
                 and isinstance(a.value, str)]
        if not names:
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        entry = {
            "names": names,
            "help": _const(kw["help"]) if "help" in kw else "",
            "default": _const(kw["default"]) if "default" in kw else None,
            "choices": _const(kw["choices"]) if "choices" in kw else None,
            "required": bool(_const(kw["required"])) if "required" in kw else False,
            "is_flag": (_const(kw["action"]) if "action" in kw else None) in
                       ("store_true", "store_false"),
            "type": getattr(kw.get("type"), "id", None) if "type" in kw else None,
        }
        flags.append(entry)
    return {"module": path.stem, "summary": summary, "doc": doc, "flags": flags}


def build(tools_dir: pathlib.Path):
    out = []
    for p in sorted(tools_dir.glob("pollard_*.py")):
        t = parse_tool(p)
        if t and t["flags"]:                      # a module with no CLI is a library, not a tool
            out.append(t)
    return out


def main():
    if len(sys.argv) > 1:
        d = pathlib.Path(sys.argv[1])
    else:                                    # no argument: find the tools beside this package
        from .runner import tools_dir
        d = tools_dir()
    tools = build(d)
    print(json.dumps({"tools": tools}, indent=2))


if __name__ == "__main__":
    main()
