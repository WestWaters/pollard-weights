"""Every lane builds and publishes its rungs SMALLEST FIRST (owner, 2026-10-24/25: "we always do the smallest first",
for ALL lanes and formats) -- unless the plan file lists that lane's `rungs:` / `ladder:` itself, which is then the
order. Fakes only: no model, no Hub."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_release as R  # noqa: E402

SHUFFLED = {"mlx": [8, 4, 2, 6, 3, 5], "gptq": [8, 4, 2, 3],
            "mx": ["fp8", "nvfp4", "mxfp6_e3m2", "mix5", "mxfp4", "mix4.5", "mxfp6_e2m3", "mix5.5"],
            "exl3": [6.0, 2.5, 4.0, 3.0, 5.0], "exlr8": [6.0, 3.25, 2.5, 5.0, 4.0, 3.0], "draft": ["Q8_0", "Q4_K_M"]}
SMALLEST_FIRST = {"mlx": [2, 3, 4, 5, 6, 8], "gptq": [2, 3, 4, 8],
                  "mx": ["mxfp4", "nvfp4", "mix4.5", "mix5", "mix5.5", "mxfp6_e3m2", "mxfp6_e2m3", "fp8"],
                  "exl3": [2.5, 3.0, 4.0, 5.0, 6.0], "exlr8": [2.5, 3.0, 3.25, 4.0, 5.0, 6.0], "draft": ["Q4_K_M", "Q8_0"]}


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    d = tmp_path / "Tiny-1B"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 2}))
    return str(d)


def _plan(tmp_path, lanes=None):
    p = tmp_path / "plan.yaml"
    p.write_text(json.dumps({"lanes": lanes or {}}))              # JSON is YAML
    return R.load_plan(str(p))


def _tags(lane, rungs):
    return [R.rung_tag(lane, r) for r in rungs]


def _builds(cfg, model, lane):
    _, _, steps = R.plan_steps(cfg, model, {lane})
    return [s.rung for s in steps if s.lane == lane and s.kind == "build" and s.rung is not None]


@pytest.mark.parametrize("lane", ["mlx", "gptq", "mx", "exlr8"])
def test_a_lanes_default_rungs_build_smallest_first(lane, model, tmp_path):
    cfg = _plan(tmp_path)
    cfg["lanes"][lane]["rungs"] = SHUFFLED[lane]                  # not from the plan file: the default order rule
    assert _builds(cfg, model, lane) == _tags(lane, SMALLEST_FIRST[lane])


@pytest.mark.parametrize("lane", ["mlx", "gptq", "mx", "exlr8"])
def test_a_plan_that_lists_the_rungs_keeps_its_order(lane, model, tmp_path):
    cfg = _plan(tmp_path, {lane: {"rungs": SHUFFLED[lane]}})
    assert lane in cfg["_explicit_order"]
    assert _builds(cfg, model, lane) == _tags(lane, SHUFFLED[lane])


def test_exl3_lowest_bpw_first_and_the_recipe_ab_rung_leads(model, tmp_path):
    cfg = _plan(tmp_path)
    cfg["lanes"]["exl3"].update(rungs=SHUFFLED["exl3"], recipe_choice="pollard")
    assert _builds(cfg, model, "exl3") == _tags("exl3", SMALLEST_FIRST["exl3"])
    # recipe_choice auto: the A/B rung is built first (every later rung builds with its winner), the rest smallest first
    cfg["lanes"]["exl3"].update(recipe_choice="auto", default=4.0)
    assert _builds(cfg, model, "exl3") == _tags("exl3", [4.0, 2.5, 3.0, 5.0, 6.0])


def test_draft_converts_q4_before_q8(model, tmp_path):
    cfg = _plan(tmp_path)
    cfg["lanes"]["draft"].update(enabled=True, rungs=SHUFFLED["draft"])
    _, _, steps = R.plan_steps(cfg, model, {"draft"})
    conv = next(s for s in steps if s.id == "draft/convert").cmd
    assert conv[conv.index("--rungs") + 1] == "Q4_K_M,Q8_0"
    cfg = _plan(tmp_path, {"draft": {"enabled": True, "rungs": ["Q8_0", "Q4_K_M"]}})
    conv = next(s for s in R.plan_steps(cfg, model, {"draft"})[2] if s.id == "draft/convert").cmd
    assert conv[conv.index("--rungs") + 1] == "Q8_0,Q4_K_M"


def test_template_defaults_are_written_smallest_first():
    cfg = R.load_plan(None)
    assert cfg["_explicit_order"] == []
    for lane in ("mlx", "gptq", "mx", "exl3", "exlr8", "draft"):
        rungs = cfg["lanes"][lane]["rungs"]
        assert rungs == R.order_rungs(cfg, lane, rungs), lane


def test_upload_sends_the_smallest_build_first(tmp_path):
    small, big = tmp_path / "a.gguf", tmp_path / "b.gguf"
    small.write_bytes(b"x" * 10)
    big.write_bytes(b"x" * 1000)
    builds = [{"path": str(big)}, {"path": str(small)}, {"path": str(tmp_path / "freed.gguf")}]
    assert [b["path"] for b in sorted(builds, key=R._build_bytes)] == [str(tmp_path / "freed.gguf"), str(small), str(big)]
