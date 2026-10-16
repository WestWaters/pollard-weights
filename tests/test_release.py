"""pollard release: one plan -> every lane as a ladder of rungs, prep first (smoke before anything is spent),
serial, resumable from the MANIFEST, gated per rung, carded per lane, uploaded only on a double opt-in.
Fakes only: commands are recorded, not run."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_card as C  # noqa: E402
import pollard_release as R  # noqa: E402
import pollard_workspace as ws  # noqa: E402


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    d = tmp_path / "Tiny-1B"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 2}))
    return str(d)


ALL = {"any": True, "apple": True, "cuda": True}


def ids(steps):
    return [s.id for s in steps]


def test_template_parses_and_presets_merge():
    cfg = R.load_plan(None)
    assert cfg["preset"] == "text" and cfg["upload"] is False and cfg["ngl"] == 0
    assert cfg["lanes"]["mlx"]["rungs"] == [2, 3, 4, 5, 6, 8]
    assert cfg["lanes"]["gptq"]["rungs"] == [2, 3, 4, 8]
    assert cfg["lanes"]["mx"]["rungs"] == ["nvfp4", "mxfp4", "fp8"]
    assert cfg["lanes"]["draft"]["enabled"] is False
    moe = R.load_plan(None, "moe")
    assert moe["lanes"]["gptq"]["rungs"] == [4, 8] and moe["lanes"]["mx"]["moe_protect_attn_only"] is True
    dec = R.load_plan(None, "decision")
    assert [l for l in R.LANES if dec["lanes"][l]["enabled"]] == ["gguf"]
    assert dec["measure"]["decision"] is True and dec["measure"]["taskeval"] is None
    assert R.load_plan(None, "vlm")["lanes"]["exl3"]["enabled"] is False


def test_plan_file_overrides_preset_and_cli_preset_overrides_file(tmp_path):
    p = tmp_path / "release.yaml"
    p.write_text("preset: moe\norg: Me\nlanes:\n  mlx: {rungs: [4]}\n")
    cfg = R.load_plan(str(p))
    assert cfg["org"] == "Me" and cfg["lanes"]["mlx"]["rungs"] == [4] and cfg["lanes"]["gptq"]["rungs"] == [4, 8]
    assert R.load_plan(str(p), "text")["lanes"]["gptq"]["rungs"] == [2, 3, 4, 8]


def test_repo_names_match_the_published_pollardweights_shelf():
    assert R.repo_name("PollardWeights", "Qwen/Qwen3-8B", "gguf") == "PollardWeights/Qwen3-8B-Pollard"
    assert R.repo_name("PollardWeights", "openbmb/MiniCPM5-2B", "mlx") == "PollardWeights/MiniCPM5-2B-Pollard-MLX"
    for lane in ("gptq", "mx", "exl3"):
        assert R.repo_name("O", "a/M", lane) == f"O/M-Pollard-{lane.upper()}"


def test_every_lane_is_a_ladder_with_build_then_verify_per_rung(model):
    c, lanes, steps = R.plan_steps(R.load_plan(None), model)
    got = ids(steps)
    assert got[0] == "prep/smoke", "preflight before anything is spent"
    assert got.index("prep/calib") < got.index("prep/probe") < got.index("gguf/ladder")
    assert got.index("prep/smooth") < got.index("gptq/int2/build")
    for lane, tags in {"mlx": ["2bit", "3bit", "4bit", "5bit", "6bit", "8bit"], "gptq": ["int2", "int3", "int4", "int8"],
                       "mx": ["NVFP4", "MXFP4", "FP8"], "exl3": ["2.5bpw", "3.0bpw", "4.0bpw", "5.0bpw", "6.0bpw"]}.items():
        for t in tags:
            assert got.index(f"{lane}/{t}/build") + 1 == got.index(f"{lane}/{t}/verify")
        assert got.index(f"{lane}/{tags[-1]}/verify") < got.index(f"{lane}/card") < got.index(f"{lane}/upload")
    by = {s.id: s for s in steps}
    assert "--bits" in by["mlx/3bit/build"].cmd and "3" in by["mlx/3bit/build"].cmd
    fp8 = by["mx/FP8/build"].cmd
    assert fp8[fp8.index("--scheme") + 1] == "FP8" and "--sensitivity" not in fp8, "pure FP8: no hot/cold split"
    # smoothed lanes build from the smoothed copy but are GATED against the original weights
    assert c["smooth_dir"] in by["gptq/int4/build"].cmd
    v = by["gptq/int4/verify"].cmd
    assert v[v.index("--source") + 1] == c["hf_dir"] and "--gate" in v
    assert by["exl3/4.0bpw/verify"].cmd[-1] == "--end-to-end"
    gg = by["gguf/ladder"].cmd
    assert gg[:2] == ["pollard", "--hf"] and "--ladder" in gg and "--sensitivity" in gg, "the way we build"
    assert "--verified-only" in by["mlx/card"].cmd
    assert "draft/note" not in got


def test_dry_run_prints_everything_and_runs_nothing(model, monkeypatch):
    ran = []
    monkeypatch.setattr(R, "run_cmd", lambda cmd, ok=(0,): ran.append(cmd))
    cfg = R.load_plan(None)
    cfg["lanes"]["draft"]["enabled"] = True
    c, _, steps = R.plan_steps(cfg, model)
    lines = []
    st = R.execute(steps, c, cfg, run=False, have=ALL, out=lines.append)
    assert not ran
    text = "\n".join(lines)
    assert "(per rung) verify: $ pollard-verify --lane gguf" in text
    assert "$ pollard-draft find --target" in text and "(per rung) measure: $ pollard-draft measure" in text, \
        "the draft lane plans find -> convert -> measure + gate per drafter rung"
    assert set(st.values()) <= {"planned", "note"}


def test_lanes_this_machine_cannot_build_are_skipped_not_failed(model):
    cfg = R.load_plan(None)
    c, _, steps = R.plan_steps(cfg, model, {"gptq", "mlx"})
    lines = []
    st = R.execute(steps, c, cfg, run=False, have={"any": True, "apple": True, "cuda": False}, out=lines.append)
    assert st["gptq/int4/build"] == "skipped: needs cuda" and st["gptq/upload"] == "skipped: needs cuda"
    assert st["mlx/4bit/build"] == "planned"


def test_run_is_serial_resumable_and_isolates_a_failed_rung(model, monkeypatch):
    cfg = R.load_plan(None)
    cfg["lanes"]["mlx"]["rungs"] = [3, 4]
    cfg["measure"]["taskeval"] = None
    c, _, steps = R.plan_steps(cfg, model, {"mlx"})
    ran = []

    def fake(cmd, ok=(0,)):
        ran.append(cmd[0] + ":" + (cmd[cmd.index("--bits") + 1] if "--bits" in cmd else ""))
        if cmd[0] == "pollard-mlx" and "3" in cmd:
            raise RuntimeError("exit 1")
        if cmd[0] == "pollard-mlx":
            out = cmd[cmd.index("--out") + 1]
            os.makedirs(out, exist_ok=True)
            ws.record_build(c["hf_dir"], "mlx", out, tag="MLX-4bit")
        if cmd[0] == "pollard-verify":
            ws.record_gate(cmd[cmd.index("--model") + 1], {"pass": True})
    monkeypatch.setattr(R, "run_cmd", fake)
    monkeypatch.setattr(R, "upload", lambda *a, **k: "held")
    st = R.execute(steps, c, cfg, run=True, have=ALL, out=lambda *_: None)
    assert st["mlx/3bit/build"] == "failed" and st["mlx/3bit/verify"] == "skipped: upstream failed"
    assert st["mlx/4bit/verify"] == "done" and st["mlx/card"] == "done"
    assert "pollard-verify:" not in ran[:ran.index("pollard-mlx:4")], "verify never runs for a failed build"
    # second run: everything done is skipped, the failed rung is retried
    ran.clear()
    st2 = R.execute(steps, c, cfg, run=True, have=ALL, out=lambda *_: None)
    assert st2["prep/calib"] == "skipped: done"
    assert st2["mlx/4bit/build"].startswith("skipped:") and st2["mlx/3bit/build"] == "failed"
    assert ran == ["pollard-mlx:3"]
    # --redo forces a step even when the state says done
    ran.clear()
    R.execute(steps, c, cfg, run=True, redo=["mlx/card"], have=ALL, out=lambda *_: None)
    assert "pollard-card:" in ran


def test_a_failed_prep_step_stops_the_release(model, monkeypatch):
    cfg = R.load_plan(None)
    c, _, steps = R.plan_steps(cfg, model, {"mlx"})
    monkeypatch.setattr(R, "run_cmd", lambda cmd, ok=(0,): (_ for _ in ()).throw(RuntimeError("no converter")))
    with pytest.raises(SystemExit, match="prep/smoke"):
        R.execute(steps, c, cfg, run=True, have=ALL, out=lambda *_: None)


def test_gguf_rungs_are_expanded_from_the_manifest_and_get_their_published_names(model, monkeypatch, tmp_path):
    cfg = R.load_plan(None)
    cfg["measure"] = {"taskeval": {"suite": "quick", "limit": 50}, "toolcall": True}
    c, _, steps = R.plan_steps(cfg, model, {"gguf"})
    os.makedirs(c["gguf_dir"], exist_ok=True)
    ran = []

    def fake(cmd, ok=(0,)):
        ran.append(cmd)
        if cmd[0] == "pollard":                         # the ladder: pollard-fit names rungs after the f16
            for tag in ("Q6_K", "IQ4_XS"):
                p = os.path.join(c["gguf_dir"], f"Tiny-1B-f16.gguf-Pollard-GGUF-{tag}.gguf")
                open(p, "wb").write(b"x")
                ws.record_build(c["f16"], "gguf", p, tag=tag)
    monkeypatch.setattr(R, "run_cmd", fake)
    monkeypatch.setattr(R, "upload", lambda *a, **k: "held")
    st = R.execute(steps, c, cfg, run=True, have=ALL, out=lambda *_: None)
    names = sorted(b["name"] for b in R.gguf_builds(c))
    assert names == ["Tiny-1B-Pollard-IQ4_XS.gguf", "Tiny-1B-Pollard-Q6_K.gguf"]
    assert all(os.path.exists(b["path"]) for b in R.gguf_builds(c))
    for tag in ("Q6_K", "IQ4_XS"):
        assert st[f"gguf/{tag}/taskeval"] == "done" and st[f"gguf/{tag}/toolcall"] == "done"
        assert st[f"gguf/{tag}/verify"] == "done"
    ver = [cmd for cmd in ran if cmd[0] == "pollard-verify"]
    assert all("--coherence" in v and "--gate" in v and v[v.index("--ref") + 1] == c["f16"] for v in ver)
    assert all(v[v.index("--model") + 1].endswith(f"Tiny-1B-Pollard-{v[v.index('--rung') + 1]}.gguf") for v in ver)


def test_upload_needs_both_opt_ins_and_only_ships_verified_rungs(model, monkeypatch, tmp_path):
    cfg = R.load_plan(None)
    c, _, steps = R.plan_steps(cfg, model, {"mlx"})
    up = next(s for s in steps if s.id == "mlx/upload")
    with pytest.raises(RuntimeError, match="no verified"):
        R.upload(up, c, cfg, run_upload=True)
    good = tmp_path / "home" / "models" / "Tiny-1B" / "Tiny-1B-Pollard-MLX-4bit"
    good.mkdir(parents=True)
    ws.record_build(c["hf_dir"], "mlx", str(good), tag="MLX-4bit")
    ws.record_gate(str(good), {"pass": True})
    os.makedirs(up.out, exist_ok=True)
    open(os.path.join(up.out, "README.md"), "w").write("card")
    assert R.upload(up, c, cfg, run_upload=False) == "held"


def test_subfolders_and_default_rung(model):
    b4 = {"name": "M-Pollard-MLX-4bit", "bytes": 4}
    b8 = {"name": "M-Pollard-MLX-8bit", "bytes": 8}
    assert C.rung_subfolder(b4, "mlx") == "4bit"
    assert C.rung_subfolder({"name": "M-Pollard-EXL3-4.0bpw"}, "exl3") == "4.0bpw"
    assert R.default_build([b4, b8], "mlx", 4) is b4
    assert R.default_build([b8], "mlx", 4) is b8, "default not verified -> the largest verified rung"
    md = "\n".join(C.rungs_section([b4, b8], "mlx", "O/M-Pollard-MLX"))
    assert "| `8bit/` |" in md and 'hf download O/M-Pollard-MLX --include "4bit/*"' in md
    assert C.rungs_section([b4], "gguf", "x") == []


def test_cli_init_and_dry_run(model, tmp_path, capsys):
    p = tmp_path / "plan.yaml"
    assert R.main(["--init", str(p), "--preset", "moe"]) == 0
    assert R.load_plan(str(p))["preset"] == "moe"
    with pytest.raises(SystemExit):
        R.main(["--init", str(p)])                   # never overwrites an edited plan
    assert R.main(["--hf", model, "--plan", str(p), "--lanes", "gguf"]) == 0
    out = capsys.readouterr().out
    assert "plan only" in out and "Tiny-1B-Pollard " in out and "upload: off" in out


def test_shipped_plan_templates_match_the_builtin_ones():
    rec = ROOT / "recipes"
    if not (rec / "release.yaml").exists():
        pytest.skip("recipes/ not in this source tree")
    assert (rec / "release.yaml").read_text(encoding="utf-8") == R.TEMPLATE
    for p in ("vlm", "decision", "moe"):
        assert R.load_plan(str(rec / f"release-{p}.yaml"))["preset"] == p
