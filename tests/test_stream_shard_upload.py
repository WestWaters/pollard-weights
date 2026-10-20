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
    """Bytes per (repo, path) on main and per (repo, path) on each branch; get_paths_info answers size + LFS sha256 of
    what a revision stores; create_commit copies files between revisions (CommitOperationCopy, server side)."""
    def __init__(self):
        self.blobs, self.private, self.log = {}, {}, []
        self.branches = {}                                   # (repo, branch) -> {path: bytes}
        self.commits, self.deleted = [], []
        self.fail_upload = False
        self.corrupt = False

    def store(self, repo, revision=None):
        if revision in (None, "main"):
            return None
        if (repo, revision) not in self.branches:
            raise FileNotFoundError(f"404: no branch {revision} in {repo}")
        return self.branches[(repo, revision)]

    def get(self, repo, path, revision=None):
        st = self.store(repo, revision)
        return self.blobs.get((repo, path)) if st is None else st.get(path)

    def api(self):
        hub = self

        class Api:
            def create_repo(self, repo, repo_type="model", exist_ok=True, private=False):
                hub.private.setdefault(repo, private)

            def create_branch(self, repo, branch, exist_ok=False):
                if (repo, branch) in hub.branches and not exist_ok:
                    raise RuntimeError("409: branch exists")
                hub.branches.setdefault((repo, branch), {})

            def delete_branch(self, repo, branch):
                if (repo, branch) not in hub.branches:
                    raise FileNotFoundError(f"404: no branch {branch}")
                del hub.branches[(repo, branch)]
                hub.deleted.append((repo, branch))

            def upload_file(self, path_or_fileobj, path_in_repo, repo_id, revision=None):
                if hub.fail_upload:
                    raise ConnectionError("upload interrupted")
                data = open(path_or_fileobj, "rb").read()
                data = data[:-3] if hub.corrupt else data
                st = hub.store(repo_id, revision)
                if st is None:
                    hub.blobs[(repo_id, path_in_repo)] = data
                else:
                    st[path_in_repo] = data
                hub.log.append(path_in_repo if st is None else f"{revision}:{path_in_repo}")

            def upload_folder(self, folder_path, repo_id, path_in_repo="", ignore_patterns=None):
                import fnmatch
                for f in sorted(os.listdir(folder_path)):
                    if any(fnmatch.fnmatch(f, p) for p in ignore_patterns or ()):
                        continue
                    self.upload_file(os.path.join(folder_path, f), f"{path_in_repo}/{f}", repo_id)

            def create_commit(self, repo_id, operations, commit_message="", revision=None):
                staged = {}
                for op in operations:                        # atomic: every source must exist or nothing lands
                    data = hub.get(repo_id, op.src_path_in_repo, op.src_revision)
                    if data is None:
                        raise FileNotFoundError(f"404: {op.src_path_in_repo}@{op.src_revision}")
                    staged[op.path_in_repo] = data
                for p, data in staged.items():
                    hub.blobs[(repo_id, p)] = data
                    hub.log.append(f"copy:{p}")
                hub.commits.append((repo_id, commit_message, sorted(staged)))

            def list_repo_files(self, repo):
                return sorted(p for (r, p) in hub.blobs if r == repo)

            def get_paths_info(self, repo, paths, revision=None):
                out = []
                for p in paths:
                    data = hub.get(repo, p, revision)
                    if data is not None:
                        out.append(types.SimpleNamespace(path=p, size=len(data), blob_id=None,
                                                         lfs={"sha256": hashlib.sha256(data).hexdigest()}))
                return out

            def update_repo_settings(self, repo_id, private=None):
                hub.private[repo_id] = private
        return Api()


@pytest.fixture
def hub(monkeypatch):
    h = Hub()
    monkeypatch.setattr(PS, "_hf_api", h.api)
    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = h.api

    class CommitOperationCopy:
        def __init__(self, src_path_in_repo, path_in_repo, src_revision=None):
            self.src_path_in_repo, self.path_in_repo, self.src_revision = src_path_in_repo, path_in_repo, src_revision
    fake.CommitOperationCopy = CommitOperationCopy
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    def rng(repo, rp, start, n, retries=3, revision=None):
        return h.get(repo, rp, revision)[start:start + n]
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
            assert not [p for (r, p) in hub.blobs if p.endswith(".safetensors")], \
                "while it is unverified, main holds none of its shards (the staging branch does)"
            assert hub.branches[(repo, "pollard-staging-GPTQ-int8")], "verify reads from the staging branch"
            ws.record_gate(out, {"pass": True})
        elif tool == "pollard-toolcall":
            raise AssertionError("a streamed rung is not on disk for the tool-call probe")
        elif tool == "pollard-card":
            o = cmd[cmd.index("--out") + 1]
            os.makedirs(os.path.dirname(o), exist_ok=True)
            open(o, "w").write("card")
    repo = "PollardWeights/Tiny-1B-Pollard"
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
    files = [p for (r, p) in hub.blobs if r == repo]
    assert "GPTQ/int8/config.json" in files and "GPTQ/int8/model.safetensors.index.json" in files
    assert {f"GPTQ/int8/model-0000{i}.safetensors" for i in (1, 2, 3)} <= set(files), "the shards reached main"
    assert len(hub.commits) == 1 and len(hub.commits[0][2]) == 3, "ONE server-side copy commit, no re-upload"
    assert not any(x.startswith("GPTQ/int8/model-") for x in hub.log), "no shard was ever uploaded to main"
    assert (repo, "pollard-staging-GPTQ-int8") not in hub.branches and hub.deleted == [
        (repo, "pollard-staging-GPTQ-int8")], "the staging branch is deleted once main holds the rung"
    main = [x[len("copy:"):] if x.startswith("copy:") else x for x in hub.log if ":" not in x or x.startswith("copy:")]
    assert main.index("GPTQ/int8/model.safetensors.index.json") > max(
        main.index(p) for p in main if p.startswith("GPTQ/int8/") and not p.endswith("index.json")), \
        "the index reaches main LAST (after every shard is copied)"
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


def test_the_complete_gate_sees_a_streamed_rungs_tensors(tmp_path, hub, monkeypatch):
    import pollard_complete as PC
    out = tmp_path / "rung"
    _spec(monkeypatch, out)
    _write(out)
    keys = PC._out_tensors(str(out))
    assert "model.layers.0.mlp.weight" in keys and "model.layers.4.mlp.weight" in keys


def test_streamed_shards_go_to_the_staging_branch_and_read_back_from_it(tmp_path, hub, monkeypatch):
    out = tmp_path / "rung"
    monkeypatch.setenv(PS.SHARD_UPLOAD_ENV, json.dumps({"out": str(out), "repo": "O/M", "path_in_repo": "MX/nvfp4",
                                                        "revision": PS.staging_branch("MX/nvfp4")}))
    arrs = _write(out)
    assert PS.staging_branch("MX/nvfp4") == "pollard-staging-MX-nvfp4"
    assert hub.blobs == {}, "main (which may be PUBLIC) never receives an unverified shard"
    assert sorted(hub.branches[("O/M", "pollard-staging-MX-nvfp4")]) == [f"MX/nvfp4/model-0000{i}.safetensors"
                                                                        for i in (1, 2, 3)]
    reg = PS.read_streamed(str(out))
    assert {e["revision"] for e in reg.values()} == {"pollard-staging-MX-nvfp4"}
    src = PS.Source(str(out))                                 # verify: range reads from the staging branch
    for k, a in arrs.items():
        assert np.array_equal(src.read_raw(k)[1], a)
    PS.promote_staging(hub.api(), reg, log=lambda *_: None)
    assert sorted(p for (_, p) in hub.blobs) == sorted(e["path_in_repo"] for e in reg.values())
    assert ("O/M", "pollard-staging-MX-nvfp4") not in hub.branches


def test_a_promotion_that_does_not_land_intact_keeps_the_branch(tmp_path, hub, monkeypatch):
    out = tmp_path / "rung"
    monkeypatch.setenv(PS.SHARD_UPLOAD_ENV, json.dumps({"out": str(out), "repo": "O/M", "path_in_repo": "GPTQ/int4",
                                                        "revision": "pollard-staging-GPTQ-int4"}))
    _write(out)
    reg = PS.read_streamed(str(out))
    first = sorted(reg)[0]
    reg[first] = dict(reg[first], sha256="0" * 64)            # main would hold other bytes than were uploaded
    with pytest.raises(RuntimeError, match="branch is kept"):
        PS.promote_staging(hub.api(), reg, log=lambda *_: None)
    assert ("O/M", "pollard-staging-GPTQ-int4") in hub.branches, "nothing lost: the verified bytes stay on the branch"
    with pytest.raises(ValueError, match="not a Pollard staging branch"):
        PS.discard_staging(hub.api(), "O/M", "main")


def test_a_staged_rung_that_fails_its_gate_deletes_its_branch_and_never_touches_main(model, inputs, hub, monkeypatch):
    cfg = _gptq_plan(inputs)
    c = R._ctx(dict(cfg, _free_after_upload=True), model)
    repo = "PollardWeights/Tiny-1B-Pollard"
    out = os.path.join(ws.model_dir(c["smooth_dir"], create=True), "Tiny-1B-Pollard-GPTQ-int8")
    monkeypatch.setenv(PS.SHARD_UPLOAD_ENV, json.dumps({"out": out, "repo": repo, "path_in_repo": "GPTQ/int8",
                                                        "revision": "pollard-staging-GPTQ-int8"}))
    _write(out)
    monkeypatch.delenv(PS.SHARD_UPLOAD_ENV)
    hub.blobs[(repo, "README.md")] = b"card"
    hub.private[repo] = False                                 # already PUBLIC (an earlier lane went out)
    before = dict(hub.blobs)
    ws.record_build(c["smooth_dir"], "gptq", out, tag="int8")
    ws.record_gate(out, {"pass": False})
    st = R._publish_rung_step(c, "gptq", "int8", out)
    assert R.publish_rung(st, c, cfg, True, out=lambda *_: None) == "kept: not verified"
    assert hub.blobs == before, "main untouched"
    assert (repo, "pollard-staging-GPTQ-int8") not in hub.branches, "the failed rung's branch is gone"
    assert PS.read_streamed(out) == {}, "and so is its registry"
    assert not ws.find_build(out)[1].get("streamed_unverified"), "nothing stranded: the repo is not blocked"


def test_a_staged_rung_whose_verify_was_held_keeps_its_branch(model, inputs, hub, monkeypatch):
    cfg = _gptq_plan(inputs)
    c = R._ctx(dict(cfg, _free_after_upload=True), model)
    repo = "PollardWeights/Tiny-1B-Pollard"
    out = os.path.join(ws.model_dir(c["smooth_dir"], create=True), "Tiny-1B-Pollard-GPTQ-int8")
    monkeypatch.setenv(PS.SHARD_UPLOAD_ENV, json.dumps({"out": out, "repo": repo, "path_in_repo": "GPTQ/int8",
                                                        "revision": "pollard-staging-GPTQ-int8"}))
    _write(out)
    monkeypatch.delenv(PS.SHARD_UPLOAD_ENV)
    ws.record_build(c["smooth_dir"], "gptq", out, tag="int8")   # no gate verdict: held / not run
    st = R._publish_rung_step(c, "gptq", "int8", out)
    said = []
    assert R.publish_rung(st, c, cfg, True, out=said.append) == "kept: not verified"
    assert hub.branches[(repo, "pollard-staging-GPTQ-int8")], "resumable: the shards stay on their branch"
    assert any("staging branch" in x for x in said)
    assert not any(k[1].endswith(".safetensors") for k in hub.blobs), "and main still has none of them"
    # a build that FAILED part-way: its staging branch is deleted when the publish step is skipped
    assert R.drop_failed_stream(st, {"gptq/int8/build": "failed"}, out=lambda *_: None)
    assert (repo, "pollard-staging-GPTQ-int8") not in hub.branches
