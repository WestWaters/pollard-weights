"""pollard release --free-after-upload: one rung at a time (build -> gate -> upload -> delete the local copy), so a
box whose free disk is smaller than a whole ladder still publishes every rung; plus inputs named in the plan.
Fakes only: no model, no Hub."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_release as R  # noqa: E402
import pollard_workspace as ws  # noqa: E402

ALL = {"any": True, "apple": True, "cuda": True}


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    d = tmp_path / "Tiny-1B"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 2}))
    return str(d)


@pytest.fixture
def inputs(tmp_path):
    src = tmp_path / "inputs"
    src.mkdir()
    f16 = src / "Tiny-1B-f16.gguf"
    f16.write_bytes(b"GGUF" + b"\0" * 1000)
    for n in ("tiny.imatrix", "tiny.sens.json", "calib.txt", "heldout.txt"):
        (src / n).write_text("x")
    return {"f16": str(f16), "imatrix": str(src / "tiny.imatrix"), "sensitivity": str(src / "tiny.sens.json"),
            "calib_file": str(src / "calib.txt"), "heldout_file": str(src / "heldout.txt")}


class FakeApi:
    def __init__(self, store):
        self.store = store

    def create_repo(self, repo, **k):
        self.store.setdefault(repo, set())

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id):
        assert os.path.exists(path_or_fileobj), "uploads read the local file before it is freed"
        self.store.setdefault(repo_id, set()).add(path_in_repo)

    def upload_folder(self, folder_path, repo_id, path_in_repo=""):
        assert os.path.isdir(folder_path)
        for f in os.listdir(folder_path):
            self.store.setdefault(repo_id, set()).add(f"{path_in_repo}/{f}" if path_in_repo else f)

    def list_repo_files(self, repo):
        return sorted(self.store.get(repo, set()))


@pytest.fixture
def hub(monkeypatch):
    """A fake huggingface_hub module: the tests run where the package is not installed (the nix core build)."""
    import types
    store = {}
    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = lambda: FakeApi(store)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return store


def plan(inputs, **extra):
    cfg = R.load_plan(None, "moe")
    cfg.update(inputs)
    cfg["upload"] = True
    cfg["_free_after_upload"] = True
    cfg["measure"] = {"taskeval": None, "toolcall": False}
    cfg.update(extra)
    return cfg


def test_inputs_named_in_the_plan_are_used_not_rebuilt(model, inputs):
    cfg = plan(inputs)
    cfg.pop("_free_after_upload")
    c, _, steps = R.plan_steps(cfg, model, {"gguf"})
    by = {s.id: s for s in steps}
    assert c["f16"] == os.path.abspath(inputs["f16"]) and c["calib"] == os.path.abspath(inputs["calib_file"])
    assert by["prep/calib"].kind == "note" and by["prep/probe"].kind == "note"
    cmd = by["gguf/ladder"].cmd
    assert cmd[cmd.index("--gguf") + 1] == c["f16"] and "--hf" not in cmd
    assert cmd[cmd.index("--imatrix") + 1] == os.path.abspath(inputs["imatrix"])
    assert cmd[cmd.index("--sensitivity") + 1] == os.path.abspath(inputs["sensitivity"])


def test_free_mode_builds_the_ladder_one_rung_at_a_time(model, inputs):
    c, _, steps = R.plan_steps(plan(inputs), model, {"gguf"})
    got = [s.id for s in steps]
    builds = [s for s in steps if s.id.startswith("gguf/ladder-")]
    assert [s.rung for s in builds] == ["Q6_K", "IQ4_XS", "IQ3_S", "IQ2", "flagship"]
    for s in builds:
        assert s.cmd[s.cmd.index("--rung") + 1] == s.rung and s.disk_gb > 0
        assert got[got.index(s.id) + 1] == f"gguf/rungs-{s.rung}", "each rung is gated + published before the next"
    assert got.index("gguf/ladder-flagship") < got.index("gguf/card") < got.index("gguf/upload")
    assert "--runtime-from-repo" in next(s for s in steps if s.id == "gguf/card").cmd


def test_free_mode_needs_both_upload_opt_ins(model):
    with pytest.raises(SystemExit):
        R.main(["--hf", model, "--free-after-upload", "--lanes", "gguf"])


def test_each_rung_is_uploaded_then_freed_and_the_inputs_never_are(model, inputs, hub, monkeypatch):
    cfg = plan(inputs)
    c, _, steps = R.plan_steps(cfg, model, {"gguf"})
    rdir = ws.model_dir(c["f16"], create=True)
    peak, live = [], []

    def fake(cmd, ok=(0,)):
        if cmd[0] == "pollard":
            label = cmd[cmd.index("--rung") + 1]
            if label == "IQ2":
                return                                        # left to the flagship on an ik box: builds nothing
            if label == "flagship":
                p = os.path.join(os.path.dirname(c["f16"]), "Tiny-1B-f16-mix.gguf")
                open(p, "wb").write(b"GGUF")
                ws.record_build(c["f16"], "gguf", p, tag="IQ1_KT", extra={"flagship": True})
            else:
                p = os.path.join(rdir, f"Tiny-1B-f16.gguf-Pollard-GGUF-{label}.gguf")
                open(p, "wb").write(b"GGUF")
                ws.record_build(c["f16"], "gguf", p, tag=label)
            live.append(p)
            peak.append(sum(os.path.exists(x) for x in live))
        elif cmd[0] == "pollard-verify":
            m = cmd[cmd.index("--model") + 1]
            ws.record_gate(m, {"pass": "IQ3_S" not in m})       # IQ3_S fails its gate
            if "IQ3_S" in m:
                raise RuntimeError("exit 1")
        elif cmd[0] == "pollard-card":
            out = cmd[cmd.index("--out") + 1]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            open(out, "w").write("card")
    monkeypatch.setattr(R, "run_cmd", fake)
    monkeypatch.setattr(R, "disk_free_gb", lambda p: 1000.0)
    st = R.execute(steps, c, cfg, run=True, run_upload=True, have=ALL, out=lambda *_: None)
    assert max(peak) == 1, "never more than one rung on disk"
    repo = "PollardWeights/Tiny-1B-Pollard"
    assert {"Tiny-1B-Pollard-Q6_K.gguf", "Tiny-1B-Pollard-IQ4_XS.gguf", "Tiny-1B-Pollard-IQ1_KT.gguf",
            "README.md", "Tiny-1B.imatrix"} <= hub[repo]
    assert "Tiny-1B-Pollard-IQ3_S.gguf" not in hub[repo], "a rung that failed its gate is not published"
    assert st["gguf/IQ3_S/verify"] == "failed" and st["gguf/IQ3_S/publish"] == "done"
    assert not any(os.path.exists(p) for p in live), "every rung freed, the failed one too"
    for k in ("f16", "imatrix", "sensitivity", "calib_file", "heldout_file"):
        assert os.path.exists(inputs[k]), f"input {k} must never be deleted"
    assert st["gguf/upload"] == "done"
    man = {b["tag"]: b for b in R.gguf_builds(c)}
    assert man["Q6_K"]["uploaded"] == repo and man["Q6_K"]["freed"] == "uploaded"
    assert man["IQ3_S"].get("uploaded") is None and "gate" in man["IQ3_S"]["freed"]


def test_a_rung_that_does_not_fit_is_held_and_the_release_goes_on(model, inputs, hub, monkeypatch):
    cfg = plan(inputs)
    c, _, steps = R.plan_steps(cfg, model, {"gguf"})
    ran = []
    monkeypatch.setattr(R, "run_cmd", lambda cmd, ok=(0,): ran.append(cmd[cmd.index("--rung") + 1])
                        if cmd[0] == "pollard" else None)
    monkeypatch.setattr(R, "disk_free_gb", lambda p: 3.0)       # less than any rung + the 3 GB margin
    st = R.execute(steps, c, cfg, run=True, run_upload=True, have=ALL, out=lambda *_: None)
    assert st["gguf/ladder-Q6_K"].startswith("held: disk") and ran == []
    assert all(st[f"gguf/ladder-{l}"].startswith("held: disk") for l in ("IQ4_XS", "IQ3_S", "IQ2", "flagship"))


def test_the_smoothed_copy_not_fitting_holds_only_its_lanes(model, inputs, monkeypatch):
    cfg = plan(inputs)
    c, _, steps = R.plan_steps(cfg, model, {"gguf", "gptq"})
    (Path(model) / "model.safetensors").write_bytes(b"\0" * 2000)
    c, _, steps = R.plan_steps(cfg, model, {"gguf", "gptq"})
    monkeypatch.setattr(R, "disk_free_gb", lambda p: 0.0)
    monkeypatch.setattr(R, "run_cmd", lambda cmd, ok=(0,): None)
    cfg["disk_margin_gb"] = 0.0000001
    st = R.execute(steps, c, cfg, run=True, run_upload=True, have=ALL, out=lambda *_: None)
    assert st["prep/smooth"].startswith("held: disk")
    assert st["gptq/int4/build"] == "skipped: upstream failed"
    assert st["prep/smoke"] == "done", "the release itself goes on"


def test_flagship_and_fit_outputs_are_freeable_inputs_are_not(model, inputs):
    c, _, _ = R.plan_steps(plan(inputs), model, {"gguf"})
    mix = os.path.join(os.path.dirname(c["f16"]), "Tiny-1B-f16-mix.gguf")
    assert R._freeable(c, {"path": mix, "flagship": True})
    assert not R._freeable(c, {"path": mix}), "only a recorded flagship next to the f16"
    assert not R._freeable(c, {"path": c["f16"], "flagship": True})
    assert not R._freeable(c, {"path": inputs["imatrix"]})
    assert R._freeable(c, {"path": os.path.join(ws.model_dir(c["f16"]), "x.gguf")})


def test_pollard_auto_rung_filter():
    import argparse
    import pollard_auto as A
    assert A._wanted_rungs(argparse.Namespace(rung=None)) is None
    assert A._wanted_rungs(argparse.Namespace(rung=["iq4_xs,Flagship", "Q6_K"])) == {"IQ4_XS", "flagship", "Q6_K"}


def test_bare_pollard_resolves_to_its_module_without_a_launcher(monkeypatch):
    import shutil
    import pollard_auto as A
    monkeypatch.setattr(shutil, "which", lambda n: None)        # a SYSTEM scheduled task: no venv Scripts on PATH
    cmd = A._resolve_tool(["pollard", "--gguf", "x.gguf"])
    assert cmd[0] == sys.executable and cmd[1].endswith("pollard_auto.py") and cmd[2:] == ["--gguf", "x.gguf"]
    assert A._resolve_tool(["pollard-verify", "--x"])[1].endswith("pollard_verify.py")


def test_a_quantized_source_gets_the_passthrough_ladder_rung_by_rung(model, inputs, monkeypatch):
    import pollard_passthrough as pt
    monkeypatch.setattr(pt, "read_tensors", lambda p: [{"n_bytes": 10}])
    monkeypatch.setattr(pt, "is_quantized_source", lambda ts: True)
    monkeypatch.setattr(pt, "ladder_targets", lambda ts, imatrix_ok=True: [("PT25", 9e9), ("PT100", 6e9)])
    c, _, steps = R.plan_steps(plan(inputs), model, {"gguf"})
    builds = [s for s in steps if s.id.startswith("gguf/ladder-")]
    assert [s.rung for s in builds] == ["PT25", "PT100"]
    assert 9.0 < builds[0].disk_gb < 9.5 and builds[0].cmd[-2:] == ["--rung", "PT25"]
