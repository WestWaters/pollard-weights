"""pollard release picks the probe that can actually run: the imatrix probe (no model load) when a GGUF input exists
and the HF weights cannot be used -- transformers has no class for the architecture, the checkpoint needs custom
code, the folder holds no weights, or the weights outgrow RAM + accelerator -- and builds the imatrix first when the
plan names none. Lanes that export from transformers weights are HELD (recorded) when transformers cannot load the
model, and the release carries on. Born 2026-10-16: LFM2.5 / Nemotron-3.5-Bolt / clef-omni all died at prep/probe.
Fakes only: no model, no runtime."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_release as R  # noqa: E402

ALL = {"any": True, "apple": True, "cuda": True}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    return tmp_path


def hf_dir(tmp, name, config, weights=b"\0" * 64):
    d = tmp / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    if weights:
        (d / "model.safetensors").write_bytes(weights)
    return str(d)


def plan(tmp, **extra):
    f16 = tmp / "model-f16.gguf"
    f16.write_bytes(b"GGUF" + b"\0" * 64)
    (tmp / "calib.txt").write_text("x")
    (tmp / "heldout.txt").write_text("x")
    cfg = R.load_plan(None, "moe")
    cfg.update({"f16": str(f16), "calib_file": str(tmp / "calib.txt"), "heldout_file": str(tmp / "heldout.txt")})
    cfg.update(extra)
    return cfg


def by_id(steps):
    return {s.id: s for s in steps}


def test_custom_code_checkpoint_takes_the_imatrix_probe_and_builds_the_imatrix_first(home, monkeypatch):
    # Nemotron-3.5-Bolt: config.json auto_map (custom code), 74B -- the forward probe can never load it
    src = hf_dir(home, "Bolt", {"model_type": "nemotron_h_omni", "auto_map": {"AutoModel": "x.Y"}})
    cfg = plan(home)
    c, lanes, steps = R.plan_steps(cfg, src, {"gguf", "gptq", "mx"})
    s = by_id(steps)
    got = [x.id for x in steps]
    assert got.index("prep/calib") < got.index("prep/imatrix") < got.index("prep/probe") < got.index("gguf/ladder")
    assert s["prep/imatrix"].fn == "imatrix" and s["prep/imatrix"].out == c["imatrix"]
    probe = s["prep/probe"].cmd
    assert probe[:2] == ["pollard-probe", "--from-imatrix"] and probe[probe.index("--gguf") + 1] == cfg["f16"]
    assert "--model" not in probe, "never transformers for a model it cannot load"
    lad = s["gguf/ladder"].cmd
    assert lad[lad.index("--imatrix") + 1] == c["imatrix"], "the ladder uses the imatrix the probe used"
    hold = s["prep/smooth"]
    assert hold.kind == "hold" and set(hold.blocks) == {"gptq", "mx"} and "custom modelling code" in hold.desc
    assert "prep/probe-smoothed" not in got


def test_unknown_architecture_is_routed_from_config_alone(home, monkeypatch):
    src = hf_dir(home, "clef-omni", {"model_type": "definitely_not_a_transformers_arch_xyz"})
    pytest.importorskip("transformers")
    ok, why = R.transformers_loadable(src)
    assert ok is False and "definitely_not_a_transformers_arch_xyz" in why
    route, _, loadable = R.probe_route(plan(home), R._ctx(plan(home), src))
    assert route == "imatrix" and loadable is False


def test_too_big_for_ram_and_vram_uses_the_imatrix_probe_but_keeps_the_hf_lanes(home, monkeypatch):
    src = hf_dir(home, "LFM2.5-24B-A2B", {"model_type": "lfm2_moe"})
    monkeypatch.setattr(R, "transformers_loadable", lambda d: (True, ""))
    monkeypatch.setattr(R, "probe_fits", lambda d: (False, "48.0 GB of HF weights > 10.0 GB free RAM + 14.0 GB"))
    imx = home / "given.imatrix"
    imx.write_text("x")
    c, _, steps = R.plan_steps(plan(home, imatrix=str(imx)), src, {"gguf", "gptq"})
    s = by_id(steps)
    assert "prep/imatrix" not in s, "an imatrix named in the plan is used as it is"
    assert s["prep/probe"].cmd[s["prep/probe"].cmd.index("--from-imatrix") + 1] == str(imx)
    assert s["prep/smooth"].kind == "prep", "transformers can load it: the smoothed lanes still run"


def test_a_model_that_fits_keeps_the_forward_probe(home, monkeypatch):
    src = hf_dir(home, "Tiny", {"model_type": "qwen3"})
    monkeypatch.setattr(R, "transformers_loadable", lambda d: (True, ""))
    monkeypatch.setattr(R, "probe_fits", lambda d: (True, ""))
    c, _, steps = R.plan_steps(plan(home), src, {"gguf"})
    probe = by_id(steps)["prep/probe"].cmd
    assert probe[:3] == ["pollard-probe", "--model", c["hf_dir"]] and "prep/imatrix" not in by_id(steps)
    # and with no GGUF input there is nothing to take an imatrix of: the forward probe, as before
    cfg = R.load_plan(None, "moe")
    c2, _, steps2 = R.plan_steps(cfg, src, {"gguf"})
    assert by_id(steps2)["prep/probe"].cmd[1] == "--model"


def test_plan_can_force_either_probe(home, monkeypatch):
    src = hf_dir(home, "M", {"model_type": "qwen3"})
    monkeypatch.setattr(R, "transformers_loadable", lambda d: (True, ""))
    monkeypatch.setattr(R, "probe_fits", lambda d: (False, "big"))
    _, _, st = R.plan_steps(plan(home, probe="hf"), src, {"gguf"})
    assert by_id(st)["prep/probe"].cmd[1] == "--model"
    monkeypatch.setattr(R, "probe_fits", lambda d: (True, ""))
    _, _, st = R.plan_steps(plan(home, probe="imatrix"), src, {"gguf"})
    assert by_id(st)["prep/probe"].cmd[1] == "--from-imatrix"
    with pytest.raises(SystemExit, match="probe"):
        R.plan_steps(plan(home, probe="nope"), src, {"gguf"})


def test_missing_transformers_is_not_a_verdict_on_the_model(home, monkeypatch):
    src = hf_dir(home, "M", {"model_type": "qwen3"})
    monkeypatch.setattr(R, "transformers_loadable", lambda d: (None, "transformers is not installed here"))
    monkeypatch.setattr(R, "probe_fits", lambda d: (True, ""))
    _, _, st = R.plan_steps(plan(home), src, {"gguf", "gptq"})
    assert by_id(st)["prep/smooth"].kind == "prep" and by_id(st)["prep/probe"].cmd[1] == "--model"


def test_held_lanes_are_recorded_and_the_release_carries_on(home, monkeypatch):
    src = hf_dir(home, "Bolt", {"model_type": "x", "auto_map": {"A": "b.C"}})
    cfg = plan(home)
    c, _, steps = R.plan_steps(cfg, src, {"gguf", "gptq"})
    ran = []

    def fake(cmd, ok=(0,)):
        ran.append(cmd[0])
    monkeypatch.setattr(R, "run_cmd", fake)
    def fake_imatrix(c, cfg, out=print):
        os.makedirs(os.path.dirname(c["imatrix"]), exist_ok=True)
        Path(c["imatrix"]).write_text("imx")
    monkeypatch.setattr(R, "build_imatrix", fake_imatrix)
    monkeypatch.setattr(R, "upload", lambda *a, **k: "held")
    st = R.execute(steps, c, cfg, run=True, have=ALL, out=lambda *_: None)
    assert st["prep/imatrix"] == "done" and st["prep/probe"] == "done" and st["gguf/ladder"] == "done"
    assert st["prep/smooth"].startswith("held: gptq held")
    assert st["gptq/int4/build"] == "skipped: upstream failed"
    assert "pollard-export" not in ran
    state = json.load(open(c["state"]))
    assert state["prep/smooth"]["status"].startswith("held:"), "the reason is on record"


def write_gguf(path, blocks, other=0):
    gguf = pytest.importorskip("gguf")
    import numpy as np
    w = gguf.GGUFWriter(str(path), "llama")
    for i, n in enumerate(blocks):
        w.add_tensor(f"blk.{i}.ffn_up.weight", np.zeros(n // 4, dtype=np.float32))
    if other:
        w.add_tensor("token_embd.weight", np.zeros(other // 4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def test_partial_ngl_offloads_from_the_last_block_and_never_everything(tmp_path):
    mib = 2 ** 20
    g = write_gguf(tmp_path / "m.gguf", [4 * mib] * 8, other=64 * mib)
    # budget = free - 1536 - 1536 MiB; 3072 + 10 MiB leaves room for two 4 MiB blocks
    assert R.partial_ngl(g, 3072 + 10) == 2
    assert R.partial_ngl(g, 3072) == 0 and R.partial_ngl(g, 100) == 0
    assert R.partial_ngl(g, 3072 + 40) == 7, "never the whole model while it does not fit"
    assert R.partial_ngl(g, 3072 + 200) == 99, "the whole file fits: full offload"


def test_build_imatrix_makes_a_q8_host_for_an_f16_bigger_than_ram_and_retries_on_cpu(home, monkeypatch):
    src = hf_dir(home, "Omni", {"model_type": "x", "auto_map": {"A": "b.C"}})
    cfg = plan(home, imatrix_chunks=500, ngl="auto")
    c, _, _ = R.plan_steps(cfg, src, {"gguf"})
    import pollard_calc
    monkeypatch.setattr(pollard_calc, "detect_available_ram_gb", lambda: 0.0)
    monkeypatch.setattr(pollard_calc, "find_llama_bin", lambda name, arch=None: f"/bin/{name}")
    import pollard_auto
    monkeypatch.setattr(pollard_auto, "_imatrix_shape", lambda g: [])
    monkeypatch.setattr(R, "_free_vram_mib", lambda: 16000)
    monkeypatch.setattr(R, "partial_ngl", lambda g, free: 17)
    monkeypatch.setattr(R, "disk_free_gb", lambda p: 500.0)
    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": "--parse-special"})())
    ran = []

    def fake(cmd, ok=(0,)):
        ran.append(cmd)
        if cmd[0].endswith("llama-quantize"):
            Path(cmd[2]).write_text("q8")
        elif "-ngl" in cmd and cmd[cmd.index("-ngl") + 1] == "17":
            raise RuntimeError("exit 3221226505")                 # the GPU run dies: retried on the CPU
        else:
            Path(cmd[cmd.index("-o") + 1]).write_text("imx")
    monkeypatch.setattr(R, "run_cmd", fake)
    assert R.build_imatrix(c, cfg, out=lambda *_: None) == c["imatrix"]
    host = os.path.join(c["gguf_dir"], f"{c['name']}-Q8_0-host.gguf")
    assert ran[0][0].endswith("llama-quantize") and ran[0][1:] == [cfg["f16"], host + ".partial", "Q8_0"]
    assert os.path.exists(host) and os.path.exists(c["imatrix"]) and not os.path.exists(c["imatrix"] + ".partial")
    im = ran[1]
    assert im[im.index("-m") + 1] == host and im[im.index("-f") + 1] == c["calib"]
    assert im[im.index("--chunks") + 1] == "500" and "--parse-special" in im and "dat" in im
    assert [r[r.index("-ngl") + 1] for r in ran[1:]] == ["17", "0"]


def test_a_quantized_vendor_gguf_is_its_own_imatrix_host(home, monkeypatch):
    src = hf_dir(home, "Bolt", {"model_type": "x", "auto_map": {"A": "b.C"}})
    cfg = plan(home)
    c, _, _ = R.plan_steps(cfg, src, {"gguf"})
    import pollard_passthrough as pt
    monkeypatch.setattr(pt, "read_tensors", lambda p: [{"ndim": 2, "n_bytes": 100, "type": "iq2_xxs"}])
    assert R.imatrix_host(cfg, c) == (cfg["f16"], None)
    assert R.imatrix_host(dict(cfg, imatrix_host="/x/host.gguf"), c) == (os.path.abspath("/x/host.gguf"), None)
