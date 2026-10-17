"""The MLX lane on a 16 GB Mac: the build and the verify gate STREAM (one tensor / one decoder block in memory at a
time), are preflighted against free RAM, and the release holds or labels what does not fit instead of swapping the
machine. Uploads can go through another machine's HF login (--upload-via ssh://...).

The first half is numpy-only (every CI shape: no mlx, no torch): the chunked encoder, the memory planners on
header-only checkpoints, the LFM2 bit policy, the release's memory preflight / rungs-only plan / ssh relay, the
tool-call reference measured elsewhere, the mlx-vlm setup requirement. The second half builds tiny checkpoints
(torch + transformers) and, on Apple Silicon with mlx / mlx-lm / mlx-vlm, checks that the streamed build writes the
SAME BYTES as the converter's own save (dense, MoE expert stacks, the whole Qwen3-Omni with towers + talker +
code2wav), that qwen3_omni_moe loads in mlx-vlm, and that the streamed verify gives the whole-model numbers."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

import pollard_mlx as M  # noqa: E402


def _st(path, tensors):
    """A safetensors file from {name: (dtype, shape)} with zero bytes, written by hand (no safetensors package)."""
    size = {"BF16": 2, "F32": 4, "I64": 8}
    hdr, off = {}, 0
    for k, (dt, shape) in tensors.items():
        n = int(np.prod(shape)) * size[dt]
        hdr[k] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(hdr).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + bytes(off))


def _lfm_like(tmp_path, experts=8, inter=256, hidden=128, layers=2, vocab=1000):
    """Header-only LFM2-MoE-shaped checkpoint: conv + attention mixers, per-expert w1/w2/w3, a tied embedding."""
    d = tmp_path / "lfm"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "lfm2_moe", "num_hidden_layers": layers,
                                               "num_experts": experts}))
    t = {"model.embed_tokens.weight": ("BF16", (vocab, hidden))}
    for i in range(layers):
        p = f"model.layers.{i}."
        t[p + "conv.in_proj.weight"] = ("BF16", (3 * hidden, hidden))
        t[p + "conv.out_proj.weight"] = ("BF16", (hidden, hidden))
        t[p + "feed_forward.gate.weight"] = ("BF16", (experts, hidden))
        for e in range(experts):
            for w, shp in (("w1", (inter, hidden)), ("w3", (inter, hidden)), ("w2", (hidden, inter))):
                t[p + f"feed_forward.experts.{e}.{w}.weight"] = ("BF16", shp)
    _st(d / "model.safetensors", t)
    return d


# --------------------------------------------------------------------------------------------- numpy only

@pytest.mark.parametrize("bits", [3, 4])
def test_encoder_is_chunk_invariant_and_init_may_come_per_chunk(bits):
    """Cast, search and pack one row chunk at a time (never a float32 copy of the whole tensor): the bytes do not
    depend on the chunk size, and MLX's own answer given per chunk (a callable) == given whole."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((48, 256)).astype(np.float32)
    imp = rng.random(256).astype(np.float32)
    whole = M.encode_affine(w, bits, 64, imp=imp, max_elems=1 << 30)
    small = M.encode_affine(w, bits, 64, imp=imp, max_elems=256 * 5)          # 5 rows per chunk, a ragged tail
    for a, b in zip(whole, small):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    _, s0, b0, _ = M.search_groups(w.reshape(48, 4, 64), bits, grid=(0.0,), refits=0)
    t = M.encode_affine(w, bits, 64, imp=imp, init=(s0, b0), max_elems=256 * 7)
    c = M.encode_affine(w, bits, 64, imp=imp, init=lambda r0, r1: (s0[r0:r1], b0[r0:r1]), max_elems=256 * 7)
    for a, b in zip(t, c):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert t[0].shape == (48, 256 * bits // 32) and t[1].shape == (48, 4)


def test_build_plan_counts_one_expert_stack_and_refuses_what_does_not_fit(tmp_path, monkeypatch):
    import pollard_stream as S
    d = _lfm_like(tmp_path)
    p = M.plan_build(str(d), 3, 6)
    # the unit of work is one projection of one layer for every expert (what sanitize stacks), not one expert
    assert p["largest"].endswith("feed_forward.experts.*.w1.weight") or "experts.*" in p["largest"]
    stack = 8 * 256 * 128 * 2
    assert p["largest_bytes"] == 2 * stack + 8 * 256 * 128            # reads + stack together, + its 8-bit worst case
    assert p["peak_gb"] == pytest.approx(p["largest_bytes"] / 1e9 + M.BASE_GB + M.SEARCH_GB, abs=0.01)
    assert 0 < p["out_bytes"] < p["source_bytes"] / 3                    # a 3-bit rung of a bf16 source
    monkeypatch.setattr(S, "free_ram_gb", lambda: 2.5)                 # 0.5 GB usable over the 2 GB floor
    with pytest.raises(S.OutOfMemoryBudget, match="needs ~"):
        M.preflight_build(p, str(tmp_path / "out"), log=lambda *a: None)
    monkeypatch.setattr(S, "free_ram_gb", lambda: 64.0)
    M.preflight_build(p, str(tmp_path / "out"), log=lambda *a: None)
    with pytest.raises(S.OutOfMemoryBudget, match="max-ram-gb"):
        M.preflight_build(p, str(tmp_path / "out"), max_ram_gb=0.1, log=lambda *a: None)


def test_verify_plan_is_one_block_plus_the_head(tmp_path):
    d = _lfm_like(tmp_path, layers=3)
    p = M.plan_verify(str(d), length=64)
    layer = (3 * 128 * 128 + 128 * 128 + 8 * 128 + 8 * 3 * 256 * 128) * 2
    assert p["layer_gb"] == round(layer / 1e9, 2)
    assert p["outer_gb"] == round(1000 * 128 * 2 / 1e9, 2)
    assert p["logits_gb"] == round((64 * 1000 * 4 * 3 + 64 * 1000 * 8 * 5) / 1e9, 2)   # this vocab, not a default


def test_lfm2_bit_policy_router_mixers_and_experts():
    """LFM2 names its router `feed_forward.gate`, its experts `feed_forward.switch_mlp`, its dense MLP
    `feed_forward.gate_proj` and its short-conv token mixer `conv.in_proj` -- none of which the Qwen/Llama names
    matched, so the router fell to the body bits and the mixers ignored the attention profile."""
    alloc = {0: {"attn": 8, "ffn": 3}, 3: {"attn": 3, "ffn": 6}}
    b = lambda p: M.bits_for(p, alloc, True, 64, None, 3, 6)["bits"]           # noqa: E731
    assert b("model.layers.3.feed_forward.gate") == 6                         # router: high
    assert b("model.layers.0.conv.in_proj") == 8 and b("model.layers.0.conv.out_proj") == 8   # mixer = attn profile
    assert b("model.layers.3.self_attn.out_proj") == 3
    assert b("model.layers.3.feed_forward.switch_mlp.gate_proj") == 6       # experts = ffn profile
    assert b("model.layers.0.feed_forward.gate_proj") == 3                  # dense MLP = ffn profile
    assert b("thinker.language_model.model.layers.3.mlp.gate") == 6        # Qwen3-Omni router unchanged
    assert M.bits_for("talker.model.layers.0.mlp.gate_proj", alloc, True, 64) is False


def test_mlx_vlm_is_the_lane_requirement_for_a_model_with_towers(tmp_path):
    import pollard_setup as S
    lane = S.LANES["mlx-vlm"]
    assert lane["machine"] == "apple" and lane.get("optional")
    assert {"py:mlx", "py:mlx_lm", "py:mlx_vlm", "py:torchvision"} <= set(lane["needs"])
    assert "git+https://github.com/Blaizzy/mlx-vlm" in S.PY["mlx_vlm"][0]
    assert "qwen3_omni_moe" in S._BODIES["mlx_vlm"]                         # the probe imports the omni model
    d = tmp_path / "omni"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3_omni_moe"}))
    _st(d / "model.safetensors", {"thinker.model.layers.0.self_attn.q_proj.weight": ("BF16", (8, 8)),
                                  "thinker.audio_tower.layers.0.fc1.weight": ("BF16", (8, 8)),
                                  "talker.model.layers.0.mlp.gate.weight": ("BF16", (4, 8))})
    assert M.needs_mlx_vlm(str(d))
    assert not M.needs_mlx_vlm(str(_lfm_like(tmp_path)))


# --- release: memory preflight, rungs-only, relay

def _release(tmp_path, monkeypatch):
    import pollard_release as R
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    return R


def test_release_holds_an_mlx_build_that_does_not_fit_and_labels_the_tool_gate(tmp_path, monkeypatch):
    R = _release(tmp_path, monkeypatch)
    import pollard_stream as S
    res = tmp_path / "results-mlx.json"
    steps = [R.Step(id="mlx/3bit/build", lane="mlx", rung="3bit", kind="build", requires="any",
                    cmd=[sys.executable, "-c", "raise SystemExit(9)"], out=str(tmp_path / "r3"), mem="mlx-build", bits=3),
             R.Step(id="mlx/4bit/toolcall", lane="mlx", rung="4bit", kind="measure", requires="any",
                    cmd=[sys.executable, "-c", "raise SystemExit(9)"], mem="mlx-serve", rung_dir=str(tmp_path / "r4"),
                    results=str(res), rung_name="M-MLX-4bit", ok_codes=(0, 1)),
             R.Step(id="mlx/4bit/verify", lane="mlx", rung="4bit", kind="verify", requires="any",
                    cmd=[sys.executable, "-c", "pass"], out=str(tmp_path / "r4"), mem="mlx-verify")]
    need = {"mlx-build": 9.0, "mlx-serve": 11.0, "mlx-verify": 2.0}
    monkeypatch.setattr(R, "memory_need", lambda st, c: need[st.mem])
    monkeypatch.setattr(S, "free_ram_gb", lambda: 6.0)                  # 4 GB usable over the floor
    c = {"state": str(tmp_path / "state.json"), "mdir": str(tmp_path)}
    lines = []
    st = R.execute(steps, c, {}, run=True, have={"any": True}, out=lines.append)
    assert st["mlx/3bit/build"].startswith("held: memory") and "9.0 GB" in st["mlx/3bit/build"]
    assert st["mlx/4bit/toolcall"] == "done"                             # recorded, never run, never faked
    tc = json.loads(res.read_text())["M-MLX-4bit"]["toolcall"]
    assert tc["status"] == "not-measured" and "mlx_lm.server" in tc["reason"] and tc["backend"] == "mlx"
    assert st["mlx/4bit/verify"] == "done"                              # 2 GB fits: it ran
    pub = [R.Step(id="mlx/3bit/publish", lane="mlx", rung="3bit", kind="publish", requires="any", fn="publish_rung",
                  out=str(tmp_path / "r3"))]
    st = R.execute(steps[:1] + pub, c, {}, run=True, have={"any": True}, out=lines.append)
    assert st["mlx/3bit/publish"].startswith("held: memory"), "a rung held for memory is kept, never freed"
    plan = []
    R.execute(steps[:1], c, {}, run=False, have={"any": True}, out=plan.append)
    assert any("needs ~9.0 GB of RAM" in x for x in plan)


def test_mlx_lane_steps_carry_their_memory_kind_and_rungs_only_publishes_nothing_else(tmp_path, monkeypatch):
    R = _release(tmp_path, monkeypatch)
    src = tmp_path / "LFM2.5-24B-A2B"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "lfm2_moe", "num_hidden_layers": 2}))
    cfg = R.load_plan(None, "moe")
    cfg.update(calib_file=str(tmp_path / "c.txt"), heldout_file=str(tmp_path / "h.txt"),
               sensitivity=str(tmp_path / "s.json"), toolcall_ref_rows=str(tmp_path / "rows.json"))
    cfg["measure"]["toolcall"] = True
    cfg["_free_after_upload"] = True
    cfg["_rungs_only"] = True
    c, lanes, steps = R.plan_steps(cfg, str(src), {"mlx"})
    mem = {s.id: s.mem for s in steps if s.mem}
    assert mem["mlx/3bit/build"] == "mlx-build" and mem["mlx/3bit/verify"] == "mlx-verify"
    assert mem["mlx/3bit/toolcall"] == "mlx-serve"
    assert not [s for s in steps if s.kind in ("card", "upload", "finalize")]      # no README, no visibility change
    assert [s.id for s in steps if s.kind == "publish"] == [f"mlx/{b}bit/publish" for b in (3, 4, 6, 8)]
    tc = next(s for s in steps if s.id == "mlx/3bit/toolcall")
    assert tc.cmd[tc.cmd.index("--ref-rows") + 1] == str(tmp_path / "rows.json")
    assert R.repo_name("PollardWeights", str(src)) == "PollardWeights/LFM2.5-24B-A2B-Pollard"
    with pytest.raises(SystemExit):
        R.main(["--hf", str(src), "--rungs-only"])                       # rung by rung only: needs the free flow


def test_release_mirrors_a_remote_source_instead_of_fetching_it(tmp_path, monkeypatch):
    R = _release(tmp_path, monkeypatch)
    cfg = R.load_plan(None, "vlm")
    url = "ssh://jwate@10.0.0.107/C:/pollard/downloads/clef-omni?python=C:/pollard/venv312/Scripts/python.exe"
    cfg.update(remote_source=url, calib_file=str(tmp_path / "c.txt"), heldout_file=str(tmp_path / "h.txt"),
               sensitivity=str(tmp_path / "s.json"))
    c, lanes, steps = R.plan_steps(cfg, str(tmp_path / "src" / "clef-omni"), {"mlx"})
    ids = [s.id for s in steps]
    assert "prep/mirror" in ids and "prep/fetch" not in ids
    st = steps[ids.index("prep/mirror")]
    assert st.cmd[:2] == ["pollard-mlx", "--model"] and st.cmd[-3:] == ["--remote-source", url, "--mirror-only"]
    assert R.repo_name("PollardWeights", c["src"]) == "PollardWeights/clef-omni-Pollard"
    cfg.pop("sensitivity")                         # none given: never the forward probe on a mirror's holes
    c, lanes, steps = R.plan_steps(cfg, str(tmp_path / "src" / "clef-omni"), {"mlx"})
    probe = next(s for s in steps if s.id == "prep/probe")
    assert probe.kind == "note" and "remote mirror" in probe.desc
    assert "--sensitivity" not in next(s for s in steps if s.id == "mlx/4bit/build").cmd


class _FakeRelay:
    """ssh / scp stand-in: runs the relay's python locally against a fake huggingface_hub, copies with shutil."""

    def __init__(self, tmp, hub, free_gb=500.0):
        self.tmp, self.hub, self.free, self.cmds = tmp, hub, free_gb, []

    def __call__(self, cmd, input=None, capture_output=False, text=False, **kw):
        self.cmds.append(cmd)
        if cmd[0] == "scp":
            import shutil
            src, dst = cmd[-2], cmd[-1].split(":", 1)[1]
            shutil.copytree(src, os.path.join(dst, os.path.basename(src))) if os.path.isdir(src) else \
                shutil.copy(src, dst)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        code = input.replace("from huggingface_hub import HfApi", "from fakehub import HfApi")
        code = code.replace("shutil.disk_usage(", f"(lambda p: type('U', (), {{'free': {self.free * 1e9}}}))(")
        r = subprocess.run([sys.executable, "-"], input=code, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONPATH=str(self.tmp), FAKEHUB=str(self.hub)))
        return r


FAKEHUB = '''
import os, json, shutil, hashlib
ROOT = os.environ["FAKEHUB"]
class _F:
    def __init__(self, path, size, sha):
        self.path, self.size = path, size
        self.lfs = type("L", (), {"sha256": sha})() if sha else None
class HfApi:
    def whoami(self):
        return {"name": "box"}
    def create_repo(self, repo, repo_type="model", exist_ok=True, private=True):
        os.makedirs(os.path.join(ROOT, repo), exist_ok=True)
        return repo
    def _files(self, repo):
        base = os.path.join(ROOT, repo)
        for r, _, fs in os.walk(base):
            for f in fs:
                yield os.path.relpath(os.path.join(r, f), base).replace(os.sep, "/"), os.path.join(r, f)
    def list_repo_files(self, repo):
        return sorted(p for p, _ in self._files(repo))
    def list_repo_tree(self, repo, recursive=True):
        for p, full in self._files(repo):
            sha = hashlib.sha256(open(full, "rb").read()).hexdigest()
            if os.environ.get("FAKEHUB_CORRUPT") and p.endswith(".safetensors"):
                sha = "0" * 64
            yield _F(p, os.path.getsize(full), sha if p.endswith(".safetensors") else None)
    def upload_folder(self, folder_path, path_in_repo, repo_id):
        shutil.copytree(folder_path, os.path.join(ROOT, repo_id, path_in_repo), dirs_exist_ok=True)
        return "ok"
    def upload_file(self, path_or_fileobj, path_in_repo, repo_id):
        dst = os.path.join(ROOT, repo_id, path_in_repo)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(path_or_fileobj, dst)
        return "ok"
    def update_repo_settings(self, repo_id, private):
        return "ok"
'''


def test_upload_relay_stages_uploads_checks_and_cleans_up(tmp_path, monkeypatch):
    R = _release(tmp_path, monkeypatch)
    (tmp_path / "fakehub.py").write_text(FAKEHUB)
    hub, staging = tmp_path / "hub", tmp_path / "staging"
    rung = tmp_path / "LFM-MLX-4bit"
    rung.mkdir()
    (rung / "config.json").write_text("{}")
    (rung / "model.safetensors").write_bytes(b"\x01" * 4096)
    fake = _FakeRelay(tmp_path, hub)
    url = f"ssh://jwate@box:2222{staging}?python=py&key=~/.ssh/k&min_free_gb=40"
    api = R.RelayApi(url, runner=fake, out=lambda *a: None)
    assert (api.host, api.port, api.staging, api.min_free) == ("jwate@box", "2222", str(staging), 40.0)
    win = R.RelayApi("ssh://jwate@10.0.0.107/C:/pollard/mlx-staging?python=C:/pollard/venv312/Scripts/python.exe")
    assert win.staging == "C:/pollard/mlx-staging" and win.port == "22"
    api.create_repo("Org/M-Pollard", private=True)
    api.upload_folder(str(rung), "MLX/4bit", "Org/M-Pollard")
    assert sorted(api.list_repo_files("Org/M-Pollard")) == ["MLX/4bit/config.json", "MLX/4bit/model.safetensors"]
    assert not any(staging.iterdir()), "the staging copy is deleted after the upload"
    scp = [c for c in fake.cmds if c[0] == "scp"][0]
    assert scp[:3] == ["scp", "-i", os.path.expanduser("~/.ssh/k")] and "-P" in scp and "2222" in scp
    monkeypatch.setenv("FAKEHUB_CORRUPT", "1")                      # the Hub copy differs from the local one
    with pytest.raises(RuntimeError, match="differ from the local copy"):
        api.upload_folder(str(rung), "MLX/4bit", "Org/M-Pollard")
    monkeypatch.delenv("FAKEHUB_CORRUPT")
    tight = R.RelayApi(url, runner=_FakeRelay(tmp_path, hub, free_gb=40.0), out=lambda *a: None)
    with pytest.raises(RuntimeError, match="must keep 40 GB free"):
        tight.upload_folder(str(rung), "MLX/4bit", "Org/M-Pollard")
    assert isinstance(R.hf_api({"upload_via": url}), R.RelayApi)


def test_toolcall_reference_rows_measured_elsewhere_are_labelled(tmp_path):
    import pollard_toolcall as T
    rows = [{"id": "p1", "ok": True}, {"id": "p2", "ok": False}]
    f = tmp_path / "f16.toolcall.json"
    f.write_text(json.dumps({"rows": rows, "meta": {"backend": "gguf"}}))
    a = types.SimpleNamespace(ref=None, ref_url=None, ref_rows=str(f), ref_backend="auto", ref_fallback=None)
    got, meta, rb, note = T._resolve_ref(a, "mlx", {})
    assert got == rows and rb == "gguf" and "measured elsewhere" in note
    assert T.cross_backend("mlx", rb)


# --------------------------------------------------------------------------------------------- Apple Silicon

def _has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):                       # a hidden / broken module is an absent one
        return False


def _need_mlx(vlm=False):
    for m in ("mlx", "mlx_lm") + (("mlx_vlm",) if vlm else ()):
        if not _has(m):
            pytest.skip(f"{m} not installed (Apple Silicon lane)")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    _need_mlx()
    import tiny_models as TM
    root = tmp_path_factory.mktemp("mlxstream")
    out = {"llama": TM.llama(str(root / "llama")), "lfm2": TM.lfm2_moe(str(root / "lfm2")), "root": root}
    if _has("mlx_vlm") and _has("torchvision"):
        out["omni"] = TM.omni_full(str(root / "omni"))
    return out


def _build(src, out, *extra):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "tools"), HF_HUB_OFFLINE="1")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "pollard_mlx.py"), "--model", str(src), "--out", str(out)]
                       + list(extra), capture_output=True, text=True, env=env, timeout=900)
    return r


def _tree(d):
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(d).rglob("*"))
            if p.is_file()}


@pytest.mark.parametrize("kind,extra", [("llama", ["--bits", "3", "--imatrix", "IMP"]),
                                        ("llama", ["--bits", "4", "--range", "minmax"]),
                                        ("lfm2", ["--bits", "3"]), ("omni", ["--bits", "4"])])
def test_streamed_build_writes_the_same_bytes_as_the_converters_own_save(tiny, kind, extra):
    """Every file of the rung -- weights (MLX's own header and key order), config, tokenizer, processor files, the
    towers / talker / code2wav of the omni copied through, the clef side weight -- is byte-identical to the build
    that holds the whole quantized model before mlx-lm / mlx-vlm save it."""
    if kind not in tiny:
        pytest.skip("mlx-vlm / torchvision not installed")
    root = tiny["root"]
    if "IMP" in extra:
        np.savez(root / "imp.npz", **{"model.layers.0.self_attn.q_proj": np.linspace(0.1, 3, 64, dtype=np.float32)})
        extra = [str(root / "imp.npz") if x == "IMP" else x for x in extra]
    tag = f"{kind}-{'-'.join(extra[:2])}-{len(extra)}"
    on, off = root / f"{tag}-on", root / f"{tag}-off"
    r1 = _build(tiny[kind], on, "--stream", "on", *extra)
    assert r1.returncode == 0, r1.stdout[-1500:] + r1.stderr[-1500:]
    assert "streamed build: peak footprint" in r1.stdout
    r2 = _build(tiny[kind], off, "--stream", "off", *extra)
    assert r2.returncode == 0, r2.stderr[-1500:]
    a, b = _tree(on), _tree(off)
    assert a == b, {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert any(k.endswith(".safetensors") for k in a)
    if "--imatrix" in extra:
        assert "1 imatrix-weighted" in r1.stdout


def test_qwen3_omni_moe_loads_in_mlx_vlm_whole(tiny):
    """clef-omni's architecture on mlx-vlm master: the rung loads with its towers, talker and code2wav, the thinker's
    experts are quantized, the talker is not, and the language model runs."""
    if "omni" not in tiny:
        pytest.skip("mlx-vlm / torchvision not installed")
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm.utils import load_model
    out = tiny["root"] / "omni-load"
    r = _build(tiny["omni"], out, "--bits", "4")
    assert r.returncode == 0, r.stderr[-1500:]
    model = load_model(out)
    keys = dict(tree_flatten(model.parameters()))
    assert any(k.startswith("thinker.audio_tower.") for k in keys) and any(k.startswith("thinker.vision_tower.")
                                                                           for k in keys)
    assert any(k.startswith("talker.") for k in keys) and any(k.startswith("code2wav.") for k in keys)
    assert any("switch_mlp" in k and k.endswith(".scales") for k in keys)
    assert not any(k.startswith("talker.") and k.endswith(".scales") for k in keys), "speech output kept as is"
    logits = model.language_model(mx.array([[1, 2, 3, 4]])).logits
    assert logits.shape[:2] == (1, 4) and bool(mx.all(mx.isfinite(logits)))
    assert (out / "joint_head.safetensors").exists() and (out / "preprocessor_config.json").exists()


def test_a_build_that_runs_out_of_ram_stops_held_and_leaves_nothing(tiny):
    out = tiny["root"] / "oom"
    env_floor = "1000"                                       # no machine has 1000 GB free: the preflight refuses
    env = dict(os.environ, PYTHONPATH=str(ROOT / "tools"), HF_HUB_OFFLINE="1", POLLARD_MIN_FREE_GB=env_floor)
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "pollard_mlx.py"), "--model", tiny["llama"], "--out",
                        str(out), "--bits", "4"], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == M.EXIT_HELD and "HELD:" in r.stdout and not out.exists()


@pytest.mark.parametrize("kind", ["llama", "lfm2"])
def test_streamed_verify_gives_the_whole_model_numbers(tiny, kind, tmp_path, monkeypatch):
    import pollard_verify as V
    rung = tiny["root"] / f"{kind}-verify-rung"
    if not rung.exists():
        assert _build(tiny[kind], rung, "--bits", "3").returncode == 0
    ev = tmp_path / "eval.txt"
    ev.write_text(" ".join(f"word{i % 97} token{i % 13}" for i in range(4000)))
    got = {}
    for mode in ("on", "off"):
        monkeypatch.setenv("POLLARD_HOME", str(tmp_path / f"home-{mode}"))    # each computes its own reference
        got[mode] = V.measure_mlx(str(rung), tiny[kind], str(ev), 64, 3, mode, log=lambda *a, **k: None)
    assert got["on"]["streamed"] and not got["off"]["streamed"]
    for k in ("kld", "top1", "ppl", "ref_ppl", "kld_p99"):
        assert got["on"][k] == pytest.approx(got["off"][k], rel=1e-5, abs=1e-7), k
    assert got["on"]["n"] == 3 * 64 and got["on"]["kld"] > 0


def test_streamed_verify_of_the_omni_matches_its_whole_language_model(tiny, tmp_path, monkeypatch):
    """The omni is measured through mlx-vlm's language model, block by block, and every block is emptied once it has
    run: the numbers match the whole model's logits computed directly."""
    if "omni" not in tiny:
        pytest.skip("mlx-vlm / torchvision not installed")
    import mlx.core as mx
    from mlx.utils import tree_flatten
    import pollard_verify as V
    from mlx_vlm.utils import load_model
    rung = tiny["root"] / "omni-load"
    if not rung.exists():
        assert _build(tiny["omni"], rung, "--bits", "4").returncode == 0
    rows = [list(range(5, 69)), list(range(70, 134))]
    s = M.StreamedLM(tiny["omni"], "mlx_vlm", log=lambda *a, **k: None).run(rows)
    q = M.StreamedLM(str(rung), "mlx_vlm", log=lambda *a, **k: None).run(rows)
    for blk in s.layers:                                     # nothing of any block is left in memory
        assert sum(v.size for _, v in tree_flatten(blk._inner.parameters())) == 0
    ref = load_model(Path(tiny["omni"]))
    full = load_model(rung)
    for b, ids in enumerate(rows):
        want_s = np.array(ref.language_model(mx.array([ids])).logits[0].astype(mx.float32))
        want_q = np.array(full.language_model(mx.array([ids])).logits[0].astype(mx.float32))
        np.testing.assert_allclose(s.logits(b), want_s, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(q.logits(b), want_q, rtol=1e-4, atol=1e-4)
    ev = tmp_path / "eval.txt"
    ev.write_text(" ".join(f"w{i % 50}" for i in range(3000)))
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    r = V.measure_mlx(str(rung), tiny["omni"], str(ev), 64, 2, "auto", log=lambda *a, **k: None)
    assert r["streamed"] and r["n"] == 128 and r["kld"] > 0 and r["ppl"] > 1


# --------------------------------------------------------------------------------------------- remote mirror

def _local_remote(monkeypatch):
    """The mirror's ssh, run here: the 'remote' python is this interpreter reading a local folder."""
    monkeypatch.setattr(M.RemoteMirror, "_cmd", lambda self: [sys.executable, "-"])


def test_mirror_is_sparse_headers_real_and_never_read_by_other_lanes(tmp_path, monkeypatch):
    _local_remote(monkeypatch)
    import pollard_stream as S
    src = _lfm_like(tmp_path, experts=4, inter=2048, hidden=256)
    (src / "tokenizer.json").write_text("{}")
    mir = tmp_path / "mirror"
    m = M.RemoteMirror.create(f"ssh://u@h{src}?python=py", str(mir), log=lambda *a: None)
    shard = mir / "model.safetensors"
    assert shard.stat().st_size == (src / "model.safetensors").stat().st_size
    assert shard.stat().st_blocks * 512 < shard.stat().st_size / 20, "the shard is a hole, not a copy"
    assert (mir / "tokenizer.json").read_text() == "{}" and (mir / "config.json").exists()
    assert set(S.Source(str(mir)).logical) == set(S.Source(str(src)).logical)          # headers are real
    with pytest.raises(SystemExit, match="remote mirror"):
        S.Source(str(mir)).read("model.layers.0.conv.in_proj.weight")
    assert M.plan_build(str(mir), 4, 8)["source_bytes"] == M.plan_build(str(src), 4, 8)["source_bytes"]
    assert set(m.entries) == set(S.Source(str(src)).logical) and not m.present
    key = "model.layers.1.feed_forward.experts.2.w2.weight"
    got = m.ensure({key})
    assert got == m.entries[key][2] and key in m.present
    f, off, n = m.entries[key]
    assert (mir / f).read_bytes()[off:off + n] == (src / f).read_bytes()[off:off + n]
    m2 = M.RemoteMirror(str(mir), log=lambda *a: None)                                  # the cache survives
    assert key in m2.present
    m2.free_bytes = lambda: int(3e9 + n - 1)                     # the next fetch only fits if the cache gives room
    other = "model.layers.1.feed_forward.experts.3.w2.weight"
    m2.done({key})
    m2.ensure({other})
    assert key not in m2.present and other in m2.present and m2.evicted > 0
    assert (mir / f).read_bytes()[off + 4096:off + 8192] == bytes(4096), "evicted bytes are punched out"


def test_build_preflight_counts_the_mirrors_cache_as_disk_and_evicts_it(tmp_path, monkeypatch):
    """The 8-bit LFM2.5 rung was HELD for disk (25.5 GB needed, 22.6 GB free) beside ~42 GB of evictable mirror cache:
    the release counted that cache, pollard-mlx's own preflight did not. Now the preflight counts it, and evicts what
    the rung needs before the build starts (a build whose source is all cached never fetches, so never evicts)."""
    _local_remote(monkeypatch)
    import collections
    import shutil
    import pollard_stream as S
    monkeypatch.setattr(S, "free_ram_gb", lambda: 64.0)
    src = _lfm_like(tmp_path, experts=4, inter=2048, hidden=256)
    mir = tmp_path / "mirror"
    m = M.RemoteMirror.create(f"ssh://u@h{src}?python=py", str(mir), log=lambda *a: None)
    m.ensure(set(m.entries))
    m.done(set(m.entries))
    cache = m.reclaimable_bytes()
    assert cache == sum(n for _, _, n in m.entries.values()) and cache > 0
    base = 10e9                                               # what the disk has besides the cache
    du = collections.namedtuple("du", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: du(0, 0, int(base + m.evicted)))
    m.free_bytes = lambda: int(base + m.evicted)
    plan = {"peak_gb": 1.0, "largest": "x", "largest_gb": 0.0, "out_gb": (base - 1e9 + cache * 0.5) / 1e9}
    out = str(tmp_path / "rung")
    with pytest.raises(S.OutOfMemoryBudget, match="disk"):                # without the mirror: held, as before
        M.preflight_build(plan, out, log=lambda *a: None)
    M.preflight_build(plan, out, log=lambda *a: None, mirror=m)           # with it: fits, and room is made
    assert m.evicted > 0 and base + m.evicted >= plan["out_gb"] * 1e9
    assert m.reclaimable_bytes() == cache - m.evicted
    big = dict(plan, out_gb=(base + cache * 2) / 1e9)                    # more than disk + cache: still held
    with pytest.raises(S.OutOfMemoryBudget, match="mirror's cache can give back"):
        M.preflight_build(big, out, log=lambda *a: None, mirror=m)


def test_streamed_build_from_a_mirror_is_the_same_rung(tiny, monkeypatch):
    """Every tensor MLX reads is fetched first -- the Load nodes of its graph say which -- so a build from the sparse
    mirror writes exactly the bytes of the build from the real folder."""
    _local_remote(monkeypatch)
    root = tiny["root"]
    for kind in ("lfm2",) + (("omni",) if "omni" in tiny else ()):
        mir = root / f"{kind}-mirror"
        if not mir.exists():
            M.RemoteMirror.create(f"ssh://u@h{tiny[kind]}?python=py", str(mir), log=lambda *a: None)
        out, ref = root / f"{kind}-from-mirror", root / f"{kind}-from-disk"
        for model, dest in ((str(mir), out), (tiny[kind], ref)):
            monkeypatch.setattr(sys, "argv", ["pollard-mlx", "--model", model, "--out", str(dest), "--bits", "3"])
            M.main()
        a, b = _tree(out), _tree(ref)
        assert a == b, {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        assert not any(k.startswith(".pollard-remote") for k in a), "the mirror's own files never ship"


def test_streamed_verify_from_a_mirror_on_a_tiny_disk_matches(tiny, tmp_path, monkeypatch):
    """The verify reads the source block by block through the mirror; a disk that holds about one block forces the
    cache to evict as it goes -- and the numbers are the ones from the folder itself."""
    _local_remote(monkeypatch)
    import pollard_verify as V
    root = tiny["root"]
    mir = root / "lfm2-mirror-v"
    M.RemoteMirror.create(f"ssh://u@h{tiny['lfm2']}?python=py", str(mir), log=lambda *a: None)
    rung = root / "lfm2-verify-rung"
    if not rung.exists():
        assert _build(tiny["lfm2"], rung, "--bits", "3").returncode == 0
    ent = M.RemoteMirror(str(mir)).entries
    import re as _re
    blocks = {}
    for k, (_, _, n) in ent.items():
        m = _re.search(r"layers\.(\d+)\.", k)
        blocks[m.group(1) if m else "outer"] = blocks.get(m.group(1) if m else "outer", 0) + n
    budget = blocks.pop("outer") + max(blocks.values()) + 8192       # the head (kept) + one block at a time
    assert budget < sum(n for _, _, n in ent.values()) * 0.75
    seen = []
    orig_init = M.RemoteMirror.__init__

    def tiny_disk(self, *a, **k):
        orig_init(self, *a, **k)
        seen.append(self)
        self.free_bytes = lambda: int(3e9 + budget - sum(self.entries[x][2] for x in self.present))
    monkeypatch.setattr(M.RemoteMirror, "__init__", tiny_disk)
    ev = tmp_path / "eval.txt"
    ev.write_text(" ".join(f"word{i % 97} token{i % 13}" for i in range(4000)))
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "h1"))
    got = V.measure_mlx(str(rung), str(mir), str(ev), 64, 3, "auto", log=lambda *a, **k: None)
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "h2"))
    want = V.measure_mlx(str(rung), tiny["lfm2"], str(ev), 64, 3, "on", log=lambda *a, **k: None)
    assert got["streamed"]
    for k in ("kld", "top1", "ppl", "ref_ppl"):
        assert got[k] == pytest.approx(want[k], rel=1e-6, abs=1e-9), k
    assert any(m.evicted for m in seen), "the cache had to give room back"
    st = json.loads((mir / M.MIRROR_STATE).read_text())
    assert sum(ent[x][2] for x in st) <= budget, "the cache stayed in its budget"
