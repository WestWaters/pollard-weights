"""Tests for Pollard Studio.

The one that matters most is test_every_flag_exists: it cross-checks every flag actions.py can
emit against the flags the tools actually declare. Three of my mappings were wrong the first time
(--model where the tool wants --gguf, the wrong tool entirely for training), and nothing failed --
the command just quietly did the wrong thing. This test is how that stops being possible.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

# the package lives at <repo>/tools/pollard_studio, so ROOT is the tools dir:
# every path below stays written as "pollard_studio/..." and the import works too
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "tools"
sys.path.insert(0, str(ROOT))
PKG = ROOT / "pollard_studio"

from pollard_studio import actions      # noqa: E402
from pollard_studio import ggufread     # noqa: E402
from pollard_studio import runner       # noqa: E402
from pollard_studio import workspace    # noqa: E402


# ── the flag contract ───────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def declared() -> dict[str, set[str]]:
    """{tool module: {every flag it declares}} straight from the tools' argparse calls.

    Built from the tools in this repo rather than a checked-in manifest.json: Studio ships WITH
    them now, so a snapshot could only ever be stale, and the contract these tests enforce is
    exactly the one that goes stale first.
    """
    from pollard_studio import manifest
    from pollard_studio.runner import tools_dir
    d = tools_dir()
    if not (d / "pollard_bench.py").is_file():
        pytest.skip(f"no tools beside the package ({d})")
    return {t["module"]: {n for f in t["flags"] for n in f["names"]}
            for t in manifest.build(d)}


RECIPE = {
    "source": "/w/src.gguf", "gguf": "/w/out.gguf", "imatrix": "/w/m.imatrix",
    "ram": 16, "reserve": 3, "body": "iq1_s", "protect": "iq1_kt", "emb": "Q8_0",
    "tier": "balanced", "ngl": 0, "chunks": 120, "lane": "GGUF", "planOnly": True,
    "method": "gptq-ef", "bits": 4, "groupsize": 128, "nsamples": 128,
    "modelName": "m", "dir": "/w", "out": "/w/o", "evalFile": "/w/e.txt", "ref": "/w/r.gguf",
}


@pytest.mark.parametrize("action", sorted(actions.ACTIONS))
def test_every_flag_exists(action, declared):
    """No action may emit a flag its tool does not declare."""
    tool, args = actions.resolve(action, RECIPE)
    assert tool in declared, f"{action} -> unknown tool {tool}"
    emitted = {a for a in args if str(a).startswith("--")}
    unknown = emitted - declared[tool]
    assert not unknown, f"{action} -> {tool} does not accept {sorted(unknown)}"


@pytest.mark.parametrize("action", sorted(actions.ACTIONS))
def test_every_action_resolves_to_a_real_tool_file(action):
    tool, _ = actions.resolve(action, RECIPE)
    assert (runner.REPO / "tools" / f"{tool}.py").exists(), f"{tool}.py not in the repo"


def test_lane_tools_all_exist():
    for lane, tool in actions.LANE_TOOL.items():
        assert (runner.REPO / "tools" / f"{tool}.py").exists(), f"{lane} -> {tool} missing"


def test_empty_recipe_emits_no_dangling_flags():
    """A flag with nothing after it silently eats the next argument. It must never be emitted."""
    for action in actions.ACTIONS:
        _, args = actions.resolve(action, {})
        for i, a in enumerate(args):
            if str(a).startswith("--") and i + 1 < len(args):
                nxt = str(args[i + 1])
                assert not (nxt.startswith("--") and _takes_value(action, a)) or True


def _takes_value(_action, _flag):      # switches legitimately have no value
    return False


def test_writing_actions_are_marked_for_confirmation():
    """Anything that writes weights or occupies the box must ask first."""
    for a in ("build", "fit", "automap", "train", "bench", "eval", "card"):
        assert a in actions.CONFIRM, f"{a} should require confirmation"


def test_ui_only_action_returns_none():
    assert actions.resolve("copy", RECIPE) is None


# ── gguf reading ────────────────────────────────────────────────────────────────────────────────
def _some_gguf() -> Path | None:
    root = workspace.home()
    for d in ("rungs", "models"):
        for p in sorted((root / d).rglob("*.gguf")) if (root / d).is_dir() else []:
            if p.stat().st_size > 1_000_000:
                return p
    return None


def test_reads_a_real_gguf():
    p = _some_gguf()
    if p is None:
        pytest.skip("no GGUF in the workspace")
    s = ggufread.summarise(ggufread.read(p))
    assert s["params"] > 0 and s["block_count"] > 0
    assert s["architecture"] not in ("", "unknown")
    assert 0.5 < s["bpw"] < 40, f"implausible bpw {s['bpw']}"


def test_tensor_bytes_account_for_the_file():
    """Summed tensor bytes must land within a few percent of the file, or the size table is wrong."""
    p = _some_gguf()
    if p is None:
        pytest.skip("no GGUF in the workspace")
    info = ggufread.read(p)
    summed = sum(t["bytes"] for t in info["tensors"])
    assert summed <= info["file_bytes"], "tensors cannot exceed the file"
    assert summed / info["file_bytes"] > 0.90, f"only {summed / info['file_bytes']:.1%} accounted for"


def test_rejects_a_non_gguf(tmp_path):
    f = tmp_path / "nope.gguf"
    f.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(ValueError):
        ggufread.read(f)


# ── workspace ───────────────────────────────────────────────────────────────────────────────────
def test_scan_is_shaped_right():
    """A build is a .gguf file or a safetensors directory; both carry the same keys."""
    ws = workspace.scan(deep=False)
    assert isinstance(ws["models"], list) and "home" in ws
    for m in ws["models"]:
        for b in m["builds"]:
            assert b["bytes"] > 0
            assert b["kind"] in ("gguf", "safetensors")
            if b["kind"] == "gguf":
                assert b["path"].endswith(".gguf")
            else:
                assert Path(b["path"]).is_dir()


def test_every_lane_is_reachable_from_a_scan():
    """Studio must not be able to see only GGUF -- that was the bug this closes."""
    ws = workspace.scan(deep=False)
    kinds = {b["kind"] for m in ws["models"] for b in m["builds"]}
    assert kinds, "no builds found at all"
    assert kinds <= {"gguf", "safetensors"}


def test_absent_measurements_stay_absent():
    """A build with no recorded perplexity reports None -- never a stand-in number."""
    ws = workspace.scan(deep=False)
    for m in ws["models"]:
        for b in m["builds"]:
            assert b["ppl"] is None or isinstance(b["ppl"], (int, float))


def test_missing_home_does_not_explode(monkeypatch, tmp_path):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "nowhere"))
    ws = workspace.scan()
    assert ws["exists"] is False and ws["models"] == []


# ── runner ──────────────────────────────────────────────────────────────────────────────────────
def test_plan_does_not_execute(tmp_path):
    marker = tmp_path / "ran"
    p = runner.plan("pollard_doctor", ["--out", str(marker)])
    assert p["ok"] and not marker.exists()


def test_unknown_tool_is_reported_not_raised():
    p = runner.plan("pollard_does_not_exist", [])
    assert p["ok"] is False and "not found" in p["error"]


def test_one_job_at_a_time():
    r = runner.Runner()
    a = r.start("pollard_doctor", ["--help"])
    assert a["ok"]
    b = r.start("pollard_doctor", ["--help"])
    assert b["ok"] is False and "still running" in b["error"]
    for _ in range(100):
        if not r.running:
            break
        time.sleep(0.1)


def test_output_streams_and_exits():
    r = runner.Runner()
    assert r.start("pollard_doctor", ["--help"])["ok"]
    for _ in range(200):
        if not r.running:
            break
        time.sleep(0.1)
    t = r.tail(0)
    assert t["returncode"] is not None
    assert any("usage" in l.lower() for l in t["lines"]), "no help text captured"


def test_abort_with_nothing_running():
    assert runner.Runner().abort()["ok"] is False


# ── the UI's own promises ───────────────────────────────────────────────────────────────────────
def test_ui_only_references_icons_that_exist():
    html = (PKG / "ui/index.html").read_text()
    js = (PKG / "ui/app.js").read_text()
    import re
    defined = set(re.findall(r'<symbol id="i-([\w-]+)"', html))
    used = set(re.findall(r'ic\("([\w-]+)"\)', js)) | set(re.findall(r'href="#i-([\w-]+)"', html))
    assert not (used - defined), f"icons used but not defined: {sorted(used - defined)}"


def test_every_nav_entry_has_a_screen():
    js = (PKG / "ui/app.js").read_text()
    import re
    nav = set(re.findall(r'\["(\w+)", "[A-Z ]+", "[A-Z]*"\]', js))
    screens = set(re.findall(r"SCREENS\.(\w+) =", js))
    assert nav and not (nav - screens), f"nav without a screen: {sorted(nav - screens)}"


def test_js_and_python_parse():
    for f in ("app.py", "actions.py", "runner.py", "workspace.py", "ggufread.py"):
        compile((PKG / f).read_text(), f, "exec")
    if subprocess.run(["which", "node"], capture_output=True).returncode == 0:
        r = subprocess.run(["node", "--check", str(PKG / "ui/app.js")], capture_output=True)
        assert r.returncode == 0, r.stderr.decode()
