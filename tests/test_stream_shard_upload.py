"""A rung bigger than the free disk (Bolt int8 GPTQ / FP8, ~75 GB) is not "held: disk": its safetensors shards are
written ONE at a time, each uploaded -> confirmed on the Hub (size + sha256) -> freed before the next; the index goes
up last; the repo stays private until the whole rung and lane check out; a shard whose upload fails or does not land
intact is KEPT and the build stops. The rung can still be read (verify) -- its tensors come back by range request.
Fakes only: no Hub, no model."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import pollard_stream as PS  # noqa: E402
import pollard_release as R  # noqa: E402
import pollard_workspace as ws  # noqa: E402


class Hub:
    """Bytes per (repo, path); get_paths_info answers size + LFS sha256 of what is stored."""
    def __init__(self):
        self.blobs, self.private, self.log = {}, {}, []
        self.fail_upload = False
        self.corrupt = False

    def api(self):
        hub = self

        class Api:
            def create_repo(self, repo, repo_type="model", exist_ok=True, private=False):
                hub.private.setdefault(repo, private)

            def upload_file(self, path_or_fileobj, path_in_repo, repo_id):
                if hub.fail_upload:
                    raise ConnectionError("upload interrupted")
                data = open(path_or_fileobj, "rb").read()
                hub.blobs[(repo_id, path_in_repo)] = data[:-3] if hub.corrupt else data
                hub.log.append(path_in_repo)

            def upload_folder(self, folder_path, repo_id, path_in_repo="", ignore_patterns=None):
                import fnmatch
                for f in sorted(os.listdir(folder_path)):
                    if any(fnmatch.fnmatch(f, p) for p in ignore_patterns or ()):
                        continue
                    self.upload_file(os.path.join(folder_path, f), f"{path_in_repo}/{f}", repo_id)

            def list_repo_files(self, repo):
                return sorted(p for (r, p) in hub.blobs if r == repo)

            def get_paths_info(self, repo, paths):
                return [types.SimpleNamespace(path=p, size=len(hub.blobs[(repo, p)]), blob_id=None,
                                              lfs={"sha256": hashlib.sha256(hub.blobs[(repo, p)]).hexdigest()})
                        for p in paths if (repo, p) in hub.blobs]

            def update_repo_settings(self, repo_id, private=None):
                hub.private[repo_id] = private
        return Api()


@pytest.fixture
def hub(monkeypatch):
    h = Hub()
    monkeypatch.setattr(PS, "_hf_api", h.api)
    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = h.api
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    def rng(repo, rp, start, n, retries=3):
        return h.blobs[(repo, rp)][start:start + n]
    monkeypatch.setattr(PS, "hub_range", rng)
    return h


def _spec(monkeypatch, out, repo="O/M-Pollard", sub="GPTQ/int8"):
    monkeypatch.setenv(PS.SHARD_UPLOAD_ENV, json.dumps({"out": str(out), "repo": repo, "path_in_repo": sub}))


def _write(out, n_tensors=5, each=1000):
    w = PS.ShardWriter(str(out), max_shard_bytes=each * 4 * 2, log=lambda *_: None)   # ~2 tensors per shard
    arrs = {}
    for i in range(n_tensors):
        a = (np.arange(each, dtype=np.float32) + i).reshape(10, -1)
        arrs[f"model.layers.{i}.mlp.weight"] = a
        w.add(f"model.layers.{i}.mlp.weight", "F32", a)
    w.close()
    return arrs


def test_each_shard_is_uploaded_confirmed_and_freed_before_the_next(tmp_path, hub, monkeypatch):
    out = tmp_path / "M-Pollard-GPTQ-int8"
    _spec(monkeypatch, out)
    seen_local = []
    real = PS.stream_shard

    def watch(o, path, spec, api=None, log=print):
        seen_local.append(sorted(f for f in os.listdir(o) if f.endswith(".safetensors")))
        return real(o, path, spec, api, log)
    monkeypatch.setattr(PS, "stream_shard", watch)
    arrs = _write(out)
    assert all(len(x) == 1 for x in seen_local), "never more than the one shard being written on disk"
    assert not [f for f in os.listdir(out) if f.endswith(".safetensors")], "every shard freed"
    reg = PS.read_streamed(str(out))
    assert sorted(reg) == ["model-00001.safetensors", "model-00002.safetensors", "model-00003.safetensors"]
    for fn, e in reg.items():
        assert e["path_in_repo"] == f"GPTQ/int8/{fn}" and len(hub.blobs[("O/M-Pollard", e["path_in_repo"])]) == e["size"]
    idx = json.load(open(out / "model.safetensors.index.json"))
    assert set(idx["weight_map"].values()) == set(reg), "the index maps the final, already-uploaded names"
    assert not any(p.endswith("index.json") for p in hub.log), "the index is NOT uploaded by the writer"
    assert hub.private["O/M-Pollard"] is True
    # the rung is still readable: tensors come back from the Hub by range
    src = PS.Source(str(out))
    for k, a in arrs.items():
        assert np.array_equal(src.read_raw(k)[1], a)


def test_an_upload_error_keeps_the_shard_and_stops_the_build(tmp_path, hub, monkeypatch):
    out = tmp_path / "rung"
    _spec(monkeypatch, out)
    hub.fail_upload = True
    with pytest.raises(ConnectionError):
        _write(out)
    assert (out / "model-00001.safetensors").exists(), "the shard is all there is: kept"
    assert PS.read_streamed(str(out)) == {}


def test_a_shard_the_hub_does_not_hold_intact_is_kept(tmp_path, hub, monkeypatch):
    out = tmp_path / "rung"
    _spec(monkeypatch, out)
    hub.corrupt = True
    with pytest.raises(RuntimeError, match="not intact"):
        _write(out)
    assert (out / "model-00001.safetensors").exists() and PS.read_streamed(str(out)) == {}


def test_only_the_named_rung_directory_streams(tmp_path, hub, monkeypatch):
    _spec(monkeypatch, tmp_path / "the-rung")
    other = tmp_path / "M-smoothed"                          # the smoothed source is written by a ShardWriter too
    _write(other)
    assert hub.blobs == {} and len([f for f in os.listdir(other) if f.endswith(".safetensors")]) == 3
    assert (other / "model-00001-of-00003.safetensors").exists(), "a normal write keeps the -of- names"


# ------------------------------------------------------------------------------- the release

from test_release_free import ALL, inputs, model, plan  # noqa: E402,F401


def _gptq_plan(inputs):
    cfg = plan(inputs, lanes={"gptq": {"enabled": True, "rungs": [8], "default": 8}},
               measure={"taskeval": None, "toolcall": True})
    return cfg


def test_a_rung_bigger_than_the_disk_streams_instead_of_holding(model, inputs, hub, monkeypatch, tmp_path):
    (Path(model) / "model.safetensors").write_bytes(b"\0" * 4000)
    cfg = _gptq_plan(inputs)
    monkeypatch.setattr(R, "transformers_loadable", lambda d: (True, ""))
    c, _, steps = R.plan_steps(cfg, model, {"gptq"})
    build = next(s for s in steps if s.id == "gptq/int8/build")
    assert build.stream_to == ("PollardWeights/Tiny-1B-Pollard", "GPTQ/int8")
    out = build.out
    ran = {}

    def fake(cmd, ok=(0,), env=None):
        tool = cmd[0]
        if tool == "pollard-export":
            assert env and json.loads(env[PS.SHARD_UPLOAD_ENV])["out"] == os.path.abspath(out)
            monkeypatch.setenv(PS.SHARD_UPLOAD_ENV, env[PS.SHARD_UPLOAD_ENV])
            _write(out)
            Path(out, "config.json").write_text("{}")
            ws.record_build(c["smooth_dir"], "gptq", out, tag="int8")
            monkeypatch.delenv(PS.SHARD_UPLOAD_ENV)
            ran["build"] = True
        elif tool == "pollard-verify":
            assert PS.Source(out).read_raw("model.layers.0.mlp.weight")[1].shape == (10, 100), "verify reads it back"
            ws.record_gate(out, {"pass": True})
        elif tool == "pollard-toolcall":
            raise AssertionError("a streamed rung is not on disk for the tool-call probe")
        elif tool == "pollard-card":
            o = cmd[cmd.index("--out") + 1]
            os.makedirs(os.path.dirname(o), exist_ok=True)
            open(o, "w").write("card")
    monkeypatch.setattr(R, "run_cmd", fake)
    monkeypatch.setattr(R, "disk_free_gb", lambda p: 10.0)    # > shard + margin, < the rung estimate? force it:
    for s in steps:
        if s.id == "gptq/int8/build":
            s["disk_gb"] = 75.0
    said = []
    st = R.execute([s for s in steps if s.lane in ("gptq",) or s.id.startswith("prep/smoke")], c, cfg, run=True,
                   run_upload=True, have=ALL, out=said.append)
    assert ran.get("build") and st["gptq/int8/build"] == "done"
    assert any("STREAMING" in x for x in said)
    assert st["gptq/int8/toolcall"] == "done"
    res = json.load(open(R.results_path(c, "gptq")))
    tc = next(v["toolcall"] for v in res.values() if isinstance(v, dict) and "toolcall" in v)
    assert tc["status"] == "not-measured" and "streamed" in tc["reason"]
    assert st["gptq/int8/publish"] == "done"
    repo = "PollardWeights/Tiny-1B-Pollard"
    files = [p for (r, p) in hub.blobs if r == repo]
    assert "GPTQ/int8/config.json" in files and "GPTQ/int8/model.safetensors.index.json" in files
    assert hub.log.index("GPTQ/int8/model.safetensors.index.json") > max(
        hub.log.index(p) for p in hub.log if p.startswith("GPTQ/int8/") and not p.endswith("index.json")), \
        "the index is uploaded LAST"
    assert "GPTQ/int8/pollard-streamed.json" not in files
    assert not os.path.exists(out), "the rest of the rung freed once everything is confirmed"


def test_a_streamed_rung_that_fails_its_gate_keeps_the_repo_private(model, inputs, hub, monkeypatch):
    cfg = _gptq_plan(inputs)
    c = R._ctx(dict(cfg, _free_after_upload=True), model)
    out = os.path.join(ws.model_dir(c["smooth_dir"], create=True), "Tiny-1B-Pollard-GPTQ-int8")
    _spec(monkeypatch, out, repo="PollardWeights/Tiny-1B-Pollard")
    _write(out)
    monkeypatch.delenv(PS.SHARD_UPLOAD_ENV)
    ws.record_build(c["smooth_dir"], "gptq", out, tag="int8")
    ws.record_gate(out, {"pass": False})
    st = R._publish_rung_step(c, "gptq", "int8", out)
    assert R.publish_rung(st, c, cfg, True, out=lambda *_: None) == "kept: not verified"
    b = ws.find_build(out)[1]
    assert b["streamed_unverified"] and "PRIVATE" in b["kept"]
    assert os.path.isdir(out), "nothing deleted -- not the local files, not the shards on the Hub"
    assert any(k[1].startswith("GPTQ/int8/model-") for k in hub.blobs)
    pub = R.public_step(cfg, c, ["gptq"])
    with pytest.raises(RuntimeError, match="stays private"):
        R.go_public(pub, c, cfg, True, {"gptq/upload": "done"}, out=lambda *_: None)
