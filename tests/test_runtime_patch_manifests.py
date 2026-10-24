"""Every runtime patch's manifest still describes its patch.

A rebase rewrites the .patch (new context, new hunk offsets, sometimes new lines) and the manifest has to follow:
`files` is what `--verify` diffs, `adds_strings` is what it looks for in the source and the binaries, and
`base_commit` is how far back `--update` fetches history for the 3-way merge. `pollard-runtime --capture` writes all
of them from the diff; this re-derives them the same way, so a patch rebased by hand without re-capturing fails here
instead of on the box (the qwen3omni-deepstack rebase onto 8e5a58df8, 2026-10-23, regenerated both files together).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parent.parent / "runtime-patches"


def _manifests():
    if not DIR.is_dir():
        return []
    return sorted(p for p in DIR.glob("*.json") if p.name != "engines.json")


@pytest.mark.parametrize("mp", _manifests(), ids=lambda p: p.stem)
def test_manifest_matches_its_patch(mp):
    m = json.loads(mp.read_text(encoding="utf-8"))
    pp = mp.with_suffix(".patch")
    assert pp.is_file(), f"{mp.name} has no {pp.name}"
    diff = pp.read_text(encoding="utf-8", errors="replace")
    # the same derivations as pollard_runtime.capture()
    assert m["files"] == sorted(re.findall(r"^\+\+\+ b/(.+)$", diff, re.M)), "files"
    assert m["adds_strings"] == sorted(set(re.findall(r'^\+.*?"\s*([a-z][a-z0-9._\-]{2,31})\s*"', diff, re.M))), \
        "adds_strings"
    assert m["diff_bytes"] == len(diff), "diff_bytes"
    assert re.fullmatch(r"[0-9a-f]{7,40}", m.get("base_commit") or ""), "base_commit"
    assert m["tree_slug"] + "-" + m["name"] == mp.stem, "the updater finds a patch by <tree_slug>-<name>"


def test_every_declared_patch_is_captured_from_its_engine():
    if not (DIR / "engines.json").is_file():
        pytest.skip("no runtime-patches/engines.json")
    engines = {k: v for k, v in json.loads((DIR / "engines.json").read_text()).items() if not k.startswith("_")}
    for name, e in engines.items():
        for p in e.get("patches", []):
            mp = DIR / f"{p}.json"
            assert mp.is_file() and (DIR / f"{p}.patch").is_file(), f"{name} declares {p}, which is not captured"
            assert json.loads(mp.read_text())["remote"].rstrip("/") == e["url"].rstrip("/"), \
                f"{p} was captured from another tree than {name}"
