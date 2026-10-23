"""CI workflows run with a read-only token and third-party actions pinned to a commit (security audit)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

WF = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def _workflows():
    if not WF.is_dir():
        pytest.skip(".github/ not in this source tree (the nix check builds from .py files only)")
    return sorted(WF.glob("*.yml")) + sorted(WF.glob("*.yaml"))


def test_every_workflow_has_a_read_only_top_level_token():
    for p in _workflows():
        text = p.read_text(encoding="utf-8")
        assert re.search(r"^permissions:\s*\n\s+contents:\s*read\s*$", text, re.M), p.name


def test_every_action_is_pinned_to_a_full_commit_sha_with_its_tag():
    for p in _workflows():
        for ln in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*-?\s*uses:\s*(\S+)(.*)$", ln)
            if not m or m.group(1).startswith("./"):
                continue
            ref = m.group(1).split("@", 1)
            assert len(ref) == 2 and re.fullmatch(r"[0-9a-f]{40}", ref[1]), f"{p.name}: {ln.strip()}"
            assert re.search(r"#\s*v\d", m.group(2)), f"{p.name}: tag comment missing: {ln.strip()}"
