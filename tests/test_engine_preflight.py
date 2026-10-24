"""The engine preflight: every engine step a release runs is resolved as the step resolves it and loads THIS file first.

The Nemotron-3.5-Bolt release died hours in on things a 30-second check catches: a runtime older than the
architecture's upstream support, a loader that disagreed with the file ("expected 496 experts, got 240"), a converter
that lived only in another engine tree. These fixtures are fake llama.cpp builds (shell scripts + a libllama that does
or does not carry the architecture string) and a tiny real GGUF header.
"""
import argparse
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_calc as PC  # noqa: E402
import pollard_preflight as PF  # noqa: E402
import pollard_release as R  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="fake binaries are shell scripts")
ARCH = "nemotron_h_moe"
OK_LOAD = 'echo "llama_context: n_ctx = 128"; echo "[1]4.1234"; exit 0'
OLD_LOAD = ('echo "llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; '
            'expected 496, got 240" >&2; echo "llama_model_load_from_file_impl: failed to load model" >&2; exit 1')


def _script(path, body):
    with open(path, "w") as f:
        f.write("#!/bin/sh\n"
                'case "$1" in --version) echo "version: 9999 (ccdbe29)"; exit 0;; esac\n' + body + "\n")
    os.chmod(path, 0o755)


def _build(root, name="llama.cpp", knows=True, load=OK_LOAD, quantize="exit 0", rev=True):
    """A fake engine tree <root>/runtime/<name> with build/bin/{llama-*}; `knows` = its libllama carries ARCH."""
    tree = os.path.join(str(root), "runtime", name)
    b = os.path.join(tree, "build", "bin")
    os.makedirs(b)
    for n in ("llama-perplexity", "llama-imatrix"):
        _script(os.path.join(b, n), load)
    _script(os.path.join(b, "llama-server"), "exit 0")
    _script(os.path.join(b, "llama-quantize"), quantize)
    _script(os.path.join(b, "llama-mtmd-cli"), 'echo " Running in chat mode, available commands:"; exit 0')
    with open(os.path.join(b, "libllama.so"), "wb") as f:
        f.write(b"\x00llama\x00qwen3\x00" + ((ARCH.encode() + b"\x00") if knows else b""))
    if rev:
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t", GIT_COMMITTER_DATE="2026-10-12T00:00:00", GIT_AUTHOR_DATE="2026-10-12T00:00:00")
        for cmd in (["init", "-q"], ["commit", "-q", "--allow-empty", "-m", "old"]):
            subprocess.run(["git", "-C", tree] + cmd, check=True, capture_output=True, env=env)
    return tree, b


def _gguf(path, arch=ARCH):
    gguf = pytest.importorskip("gguf")
    import numpy as np
    w = gguf.GGUFWriter(str(path), arch)
    w.add_block_count(1)
    w.add_tensor("token_embd.weight", np.zeros((4, 8), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_ENGINE_PREFLIGHT", "1")
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))       # the pass cache
    monkeypatch.setattr(PC, "engine_trees", lambda repo=None: [])
    import pollard_runtime
    monkeypatch.setattr(pollard_runtime, "upstream_archs", lambda: {"llama", "qwen3", ARCH})
    return tmp_path


def _use(monkeypatch, bindir):
    monkeypatch.setattr(PC, "find_llama_bin", lambda name, arch=None: (
        os.path.join(bindir, name) if os.path.isfile(os.path.join(bindir, name)) else None))


def test_an_old_binary_that_cannot_load_the_file_fails_with_the_update_hint(env, monkeypatch):
    tree, b = _build(env, load=OLD_LOAD)
    _use(monkeypatch, b)
    g = _gguf(env / "Bolt-f16.gguf")
    lines = []
    res = PF.run(["gguf"], gguf=g, out=lines.append)
    assert res["failed_lanes"] == {"gguf"}
    row = next(r for r in res["rows"] if r["step"] == "perplexity")
    assert not row["ok"] and "expected 496, got 240" in row["detail"]
    assert "predates this architecture's support (nemotron_h_moe); run `pollard-runtime --update`" in row["fix"]
    assert tree in row["fix"] and "(2026-10-12)" in row["fix"]                       # names the tree and its commit
    assert "STOP imatrix, perplexity" in lines[0] and "llama-perplexity" in lines[0]
    imx = next(r for r in res["rows"] if r["step"] == "imatrix")
    assert not imx["ok"]                                                             # same build, same verdict


def test_a_build_that_does_not_know_the_arch_fails_before_loading(env, monkeypatch):
    _tree, b = _build(env, knows=False)
    _use(monkeypatch, b)
    res = PF.run(["gguf"], gguf=_gguf(env / "m-f16.gguf"), out=lambda *_: None)
    assert res["failed_lanes"] == {"gguf"}
    for r in res["rows"]:
        assert not r["ok"] and r["detail"] == f"this build does not know {ARCH}"
        assert "run `pollard-runtime --update`" in r["fix"]


def test_upstream_lacking_the_arch_says_it_needs_a_patch(env, monkeypatch):
    import pollard_runtime
    monkeypatch.setattr(pollard_runtime, "upstream_archs", lambda: {"llama", "qwen3"})
    _tree, b = _build(env, knows=False)
    _use(monkeypatch, b)
    res = PF.run(["gguf"], gguf=_gguf(env / "m-f16.gguf"), toolcall=False, imatrix=False, out=lambda *_: None)
    assert "upstream llama.cpp does not carry it either" in res["rows"][0]["fix"]


def test_another_engine_tree_that_loads_it_is_named(env, monkeypatch):
    _old, b = _build(env, knows=False)
    bolt, _ = _build(env, name="llama.cpp.bolt", rev=False)
    monkeypatch.setattr(PC, "engine_trees", lambda repo=None: [("llama.cpp", _old), ("llama.cpp.bolt", bolt)])
    import pollard_runtime
    monkeypatch.setattr(pollard_runtime, "upstream_archs", lambda: {"llama"})
    _use(monkeypatch, b)
    res = PF.run(["gguf"], gguf=_gguf(env / "m-f16.gguf"), toolcall=False, imatrix=False, out=lambda *_: None)
    assert f"only {bolt} can" in res["rows"][0]["fix"]


def test_a_current_build_passes_every_step_and_the_pass_is_cached(env, monkeypatch, tmp_path):
    _tree, b = _build(env)
    _use(monkeypatch, b)
    g = _gguf(env / "Omni-f16.gguf")
    mm = env / "mmproj-Omni-BF16.gguf"
    mm.write_bytes(b"GGUF")
    assert PF.find_mmproj(g) == str(mm)
    res = PF.run(["gguf"], gguf=g, mmproj=str(mm), out=lambda *_: None)
    assert res["failed_lanes"] == set()
    assert [r["step"] for r in res["rows"]] == ["imatrix", "quantize", "perplexity", "server", "mtmd"]
    assert all(r["ok"] for r in res["rows"]), res["text"]
    again = PF.run(["gguf"], gguf=g, mmproj=str(mm), out=lambda *_: None)
    assert all(r["detail"].endswith("(cached)") for r in again["rows"])


def test_a_broken_quantize_dry_run_fails_the_lane(env, monkeypatch):
    _tree, b = _build(env, quantize='echo "llama_model_quantize: failed to quantize: Failed to determine layer for '
                                      'tensor dec.blk.0.ffn_down.weight" >&2; exit 1')
    _use(monkeypatch, b)
    res = PF.run(["gguf"], gguf=_gguf(env / "clef-f16.gguf"), toolcall=False, out=lambda *_: None)
    q = next(r for r in res["rows"] if r["step"] == "quantize")
    assert not q["ok"] and "dec.blk.0" in q["detail"] and res["failed_lanes"] == {"gguf"}


def test_ik_llama_failing_only_drops_the_flagship(env, monkeypatch):
    _tree, b = _build(env)
    _ik, ikb = _build(env, name="ik_llama.cpp", knows=False, rev=False)
    _use(monkeypatch, b)
    res = PF.run(["gguf"], gguf=_gguf(env / "m-f16.gguf"), flagship=True, ik_dir=ikb, out=lambda *_: None)
    ik = next(r for r in res["rows"] if r["step"] == "ik-quantize")
    assert not ik["ok"] and not ik["critical"]
    assert res["failed_lanes"] == set()                                    # the ladder still runs
    assert "note ik-quantize" in res["text"]


def test_converter_only_in_another_tree(env, monkeypatch, tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text('{"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h"}')
    bolt, _ = _build(env, name="llama.cpp.bolt", rev=False)
    with open(os.path.join(bolt, "convert_hf_to_gguf.py"), "w") as f:
        f.write('@ModelBase.register("NemotronHForCausalLM")\nclass X: pass\n')
    monkeypatch.setattr(PC, "engine_trees", lambda repo=None: [("llama.cpp.bolt", bolt)])
    import pollard_convert
    monkeypatch.setattr(pollard_convert, "find_converter",
                        lambda d=None: (None, "no converter on this machine registers NemotronHForCausalLM.\n  ..."))
    _t, b = _build(env)
    _use(monkeypatch, b)
    res = PF.run(["gguf"], hf_dir=str(hf), out=lambda *_: None)
    conv = res["rows"][0]
    assert conv["step"] == "convert" and not conv["ok"]
    assert conv["fix"].startswith(f"converter for NemotronHForCausalLM only in {bolt}")
    assert res["failed_lanes"] == {"gguf"}


def test_hf_source_checks_the_converter_then_every_binary_against_what_it_writes(env, monkeypatch, tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text('{"architectures": ["NemotronHForCausalLM"]}')
    conv = tmp_path / "convert_hf_to_gguf.py"
    conv.write_text("import sys\n")
    import pollard_convert
    monkeypatch.setattr(pollard_convert, "find_converter", lambda d=None: (str(conv), f"{conv} registers it"))
    monkeypatch.setattr(PF, "_convert", lambda c, h, w: (True, f"--vocab-only writes {ARCH}", ARCH))
    _t, b = _build(env, knows=False)                                      # converter is new, runtime is old
    _use(monkeypatch, b)
    res = PF.run(["gguf"], hf_dir=str(hf), toolcall=False, out=lambda *_: None)
    assert res["rows"][0]["ok"] and res["arch"] == ARCH
    assert all(not r["ok"] and "pollard-runtime --update" in r["fix"] for r in res["rows"][1:])


# ---- the release and `pollard --run` ---------------------------------------------------------------------------

def _ctx(tmp_path, g):
    return {"f16": g, "hf_dir": str(tmp_path / "nohf"), "src": "M", "mdir": str(tmp_path), "probe_route": "hf",
            "state": str(tmp_path / "release-state.json")}


def test_release_holds_the_failing_lane_and_the_others_carry_on(env, monkeypatch):
    _tree, b = _build(env, load=OLD_LOAD)
    _use(monkeypatch, b)
    monkeypatch.setattr(R, "toolcall_wanted", lambda cfg, c: True)
    import pollard_auto
    monkeypatch.setattr(pollard_auto, "find_ik_build", lambda *a: None)
    g = _gguf(env / "Bolt-f16.gguf")
    st = R.Step(id="prep/engines", lane="prep", kind="prep", fn="engine_preflight", lanes=["gguf", "gptq"])
    have = {"any": True, "cuda": True, "apple": False}
    held = R.engine_preflight(st, _ctx(env, g), {"lanes": {}}, have, out=lambda *_: None)
    assert held == {"gguf"}
    with pytest.raises(R.PreflightFailed, match="critical lane"):          # `critical: true` stops the release
        R.engine_preflight(st, _ctx(env, g), {"lanes": {"gguf": {"critical": True}}}, have, out=lambda *_: None)
    only = R.Step(st, lanes=["gguf"])
    with pytest.raises(R.PreflightFailed, match="no lane"):
        R.engine_preflight(only, _ctx(env, g), {"lanes": {}}, have, out=lambda *_: None)


def test_release_execute_skips_the_held_lane_before_any_long_step(env, monkeypatch):
    _tree, b = _build(env, load=OLD_LOAD)
    _use(monkeypatch, b)
    monkeypatch.setattr(R, "toolcall_wanted", lambda cfg, c: False)
    monkeypatch.setattr(R, "engine_arch", lambda c: None)
    import pollard_auto
    monkeypatch.setattr(pollard_auto, "find_ik_build", lambda *a: None)
    ran = []
    monkeypatch.setattr(R, "run_cmd", lambda cmd, ok=(0,), env=None: ran.append(cmd[0]))
    g = _gguf(env / "Bolt-f16.gguf")
    S = R.Step
    steps = [S(id="prep/engines", lane="prep", kind="prep", requires="any", fn="engine_preflight",
               lanes=["gguf", "mlx"]),
             S(id="gguf/ladder", lane="gguf", kind="build", requires="any", cmd=["pollard", "--ladder"]),
             S(id="mlx/4/build", lane="mlx", rung="4", kind="build", requires="any", cmd=["pollard-mlx"])]
    st = R.execute(steps, _ctx(env, g), {"lanes": {}}, run=True, have={"any": True, "apple": True, "cuda": False},
                   out=lambda *_: None)
    assert st["prep/engines"] == "done"
    assert st["gguf/ladder"] == "skipped: upstream failed"
    assert ran == ["pollard-mlx"]                                          # the working lane still ran


def test_release_plan_runs_the_preflight_right_after_smoke(tmp_path, monkeypatch):
    model = tmp_path / "Tiny-1B"
    model.mkdir()
    (model / "config.json").write_text('{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}')
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    cfg = R.load_plan(None) if hasattr(R, "load_plan") else {}
    c, lanes, steps = R.plan_steps(cfg, str(model), {"gguf"})
    ids = [s.id for s in steps]
    assert ids.index("prep/engines") == ids.index("prep/smoke") + 1
    assert ids.index("prep/engines") < ids.index("prep/calib")
    assert R.resume_reason(steps[ids.index("prep/engines")], {"prep/engines": {"status": "done"}}) is None


def test_pollard_run_stops_before_the_imatrix(env, monkeypatch, capsys):
    import pollard_auto as A
    _tree, b = _build(env, load=OLD_LOAD)
    _use(monkeypatch, b)
    monkeypatch.setattr(A, "find_ik_build", lambda *a: None)
    a = argparse.Namespace(gguf=_gguf(env / "Bolt-f16.gguf"), toolcall=False, imatrix=None, auto_imatrix=True,
                           bin=None, rung=None, rungs=None)
    monkeypatch.setattr(A, "_wanted_rungs", lambda a: None)
    with pytest.raises(SystemExit, match="STOPPING before the imatrix"):
        A._engine_preflight(a)
    assert "pollard-runtime --update" in capsys.readouterr().out


def test_pollard_run_drops_only_the_flagship_when_ik_cannot_take_the_file(env, monkeypatch):
    import pollard_auto as A
    _tree, b = _build(env)
    _ik, ikb = _build(env, name="ik_llama.cpp", knows=False, rev=False)
    _use(monkeypatch, b)
    monkeypatch.setattr(A, "find_ik_build", lambda *a: ikb)
    monkeypatch.setattr(A, "_wanted_rungs", lambda a: None)
    a = argparse.Namespace(gguf=_gguf(env / "m-f16.gguf"), toolcall=False, imatrix="x.imatrix", auto_imatrix=True,
                           bin=None)
    A._engine_preflight(a)
    assert "does not know" in a._ik_preflight_failed or "pollard-runtime" in a._ik_preflight_failed


def test_off_switch(monkeypatch):
    monkeypatch.setenv("POLLARD_ENGINE_PREFLIGHT", "0")
    assert PF.disabled()
    assert R.engine_preflight(R.Step(id="prep/engines", lanes=["gguf"]), {}, {}, {"any": True}, out=lambda *_: None) == set()


def test_mlx_check_loads_the_class_and_one_shard(tmp_path):
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    st = pytest.importorskip("safetensors.numpy")
    import numpy as np
    (tmp_path / "config.json").write_text('{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}')
    st.save_file({"model.embed_tokens.weight": np.zeros((4, 8), dtype=np.float16)}, str(tmp_path / "model.safetensors"))
    ok, detail = PF._mlx(str(tmp_path))
    assert ok, detail
    assert "llama" in detail and "model.safetensors loads (1 tensors)" in detail
    (tmp_path / "config.json").write_text('{"model_type": "no_such_arch_xyz"}')
    ok, detail = PF._mlx(str(tmp_path))
    assert not ok
