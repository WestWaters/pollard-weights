#!/usr/bin/env python3
"""Validate flags against what a tool actually declares, before anything runs.

The Tools screen exposes 60 tools and 562 flags. Passing those through unchecked is how you get a
six-hour build that dies on argument 3, or worse, one that succeeds with a silently wrong value.

So every value is checked against the tool's own argparse declaration -- type, choices, required,
and whether a path that must exist does. The manifest is parsed from source (manifest.py), so this
cannot drift from the tools.

    python -m pollard_studio.validate pollard_fit '{"--ram": "16"}'
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# flags whose value names a file that must already exist. Being wrong here is cheap to catch and
# expensive to discover an hour into a build.
_MUST_EXIST = {"--gguf", "--model", "--imatrix", "--source", "--eval", "--calib-file",
               "--eval-file", "--ref", "--sensitivity", "--config", "--recipe", "--fragile"}
_TRUTHY = {"1", "true", "yes", "on", "y"}
_FALSY = {"0", "false", "no", "off", "n", ""}


class Invalid(ValueError):
    """Raised with every problem found, not just the first."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def load_manifest(path: Path | None = None) -> dict:
    """{tool: {flag: spec}} from the parsed tools."""
    from . import runner
    tools_dir = runner.REPO / "tools"
    data = None
    # The REPO WINS. A snapshot on disk goes stale the moment a tool changes, and a validator
    # working from yesterday's contract rejects a flag the tool now accepts -- which looks to the
    # user like Pollard refusing a legitimate build. The file is only a fallback for an install
    # that has no repo beside it.
    if path is None and tools_dir.is_dir():
        out = subprocess.run([sys.executable, str(HERE / "manifest.py"), str(tools_dir)],
                             capture_output=True, text=True, timeout=90)
        if out.returncode == 0:
            data = json.loads(out.stdout)["tools"]
    if data is None:
        snapshot = path or HERE.parent / "manifest.json"
        if not snapshot.exists():
            return {}
        data = json.loads(snapshot.read_text())["tools"]
    return {t["module"]: {n: f for f in t["flags"] for n in f["names"]} for t in data}


def _coerce(spec: dict, flag: str, raw) -> object:
    t = spec.get("type")
    try:
        if t == "int":
            return int(str(raw).strip())
        if t == "float":
            return float(str(raw).strip())
    except ValueError:
        raise Invalid([f"{flag} expects {t}, got {raw!r}"])
    return raw


def validate(tool: str, values: dict, manifest: dict | None = None,
             check_paths: bool = True) -> list[str]:
    """Turn {flag: value} into argv for `tool`, raising Invalid with every problem at once.

    A flag set to None/"" is dropped rather than emitted bare -- a dangling flag eats the next
    argument and quietly shifts everything after it.
    """
    manifest = manifest if manifest is not None else load_manifest()
    declared = manifest.get(tool)
    if declared is None:
        raise Invalid([f"unknown tool {tool!r}"])

    problems: list[str] = []
    argv: list[str] = []

    for flag, raw in (values or {}).items():
        if not str(flag).startswith("-"):
            argv.append(str(raw))                     # positional
            continue
        spec = declared.get(flag)
        if spec is None:
            problems.append(f"{tool} does not accept {flag}")
            continue
        if spec.get("is_flag"):
            on = raw if isinstance(raw, bool) else str(raw).lower() in _TRUTHY
            if on:
                argv.append(flag)
            continue
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue                                   # unset, not an error
        try:
            value = _coerce(spec, flag, raw)
        except Invalid as e:
            problems += e.problems
            continue
        choices = spec.get("choices")
        if choices and str(value) not in {str(c) for c in choices}:
            problems.append(f"{flag} must be one of {list(choices)}, got {value!r}")
            continue
        if check_paths and flag in _MUST_EXIST and not Path(str(value)).expanduser().exists():
            problems.append(f"{flag}: no such file — {value}")
            continue
        argv += [flag, str(value)]

    for flag, spec in declared.items():
        if spec.get("required") and flag not in values and spec["names"][0] == flag:
            problems.append(f"{flag} is required")

    if problems:
        raise Invalid(problems)
    return argv


def check(tool: str, values: dict, **kw) -> dict:
    """Non-raising form, for the UI."""
    try:
        return {"ok": True, "argv": validate(tool, values, **kw), "problems": []}
    except Invalid as e:
        return {"ok": False, "argv": [], "problems": e.problems}


def describe(tool: str, manifest: dict | None = None) -> list[dict]:
    """Flag specs for one tool, for rendering a form."""
    manifest = manifest if manifest is not None else load_manifest()
    seen, out = set(), []
    for flag, spec in (manifest.get(tool) or {}).items():
        key = tuple(spec["names"])
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    r = check(sys.argv[1], json.loads(sys.argv[2]))
    print(json.dumps(r, indent=2))
    raise SystemExit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()


# ── lane fit ────────────────────────────────────────────────────────────────────────────────────
#: a tool declaring --gguf reads a llama.cpp container and nothing else. Handing it an MLX or
#: EXL3 directory builds a command that parses and then fails on a confusing read error.
def lane_of(path: str) -> str:
    """Which lane a source on disk belongs to, from what it IS -- never from its name."""
    p = Path(str(path)).expanduser()
    if p.suffix == ".gguf":
        return "GGUF"
    if p.is_dir():
        names = {c.name for c in p.iterdir()} if p.exists() else set()
        if any(n.endswith(".gguf") for n in names):
            return "GGUF"
        if "quantize_config.json" in names or "quant_config.json" in names:
            return "GPTQ"
        if any(n.startswith("out_tensor") for n in names) or "exl3_config.json" in names:
            return "EXL3"
        if "model.safetensors.index.json" in names or any(n.endswith(".safetensors") for n in names):
            return "HF"          # an f16/bf16 checkout, or an MLX/MX export
    return "?"


def lane_fit(tool: str, values: dict, manifest: dict | None = None) -> list[str]:
    """Problems that are about the LANE rather than the flag.

    Pollard's answer to a mismatch is a route, not a refusal -- so this names the conversion
    that makes the run possible instead of just saying no.
    """
    manifest = manifest if manifest is not None else load_manifest()
    declared = manifest.get(tool) or {}
    out: list[str] = []
    if "--gguf" not in declared:
        return out                                   # takes --model: lane-agnostic
    src = values.get("--gguf")
    if not src or not str(src).strip():
        return out
    got = lane_of(str(src))
    if got in ("GGUF", "?"):                         # unknown stays quiet; it may not exist yet
        return out
    out.append(
        f"{tool.replace('_', '-')} reads a GGUF, and this source is {got}. "
        f"Convert it on the Convert screen (target GGUF) and point this at the result — "
        f"the allocation carries across, so nothing is re-solved.")
    return out
