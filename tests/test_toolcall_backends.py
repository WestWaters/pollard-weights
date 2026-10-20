"""pollard-toolcall on every lane: the backend layer (gguf / mlx / vllm / exl3 / url), how each server is started,
which backend a build gets, the memory and shared-GPU guards, the not-measured path, the cross-backend reference,
and what the card, the gate and pollard release make of it. Fakes only: no model, no server, no GPU."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_card as C  # noqa: E402
import pollard_release as R  # noqa: E402
import pollard_toolcall as TC  # noqa: E402
import pollard_verify as V  # noqa: E402
import pollard_workspace as ws  # noqa: E402
import pollard_modelkind as MK  # noqa: E402


def _dir(tmp_path, name, cfg):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "model.safetensors").write_bytes(b"\0" * 1000)
    return str(d)


@pytest.fixture(autouse=True)
def _no_workspace(monkeypatch):
    monkeypatch.setattr(ws, "find_build", lambda p: (None, None))
    # a machine with a working vLLM unless a test says otherwise (vllm_status itself is tested below)
    monkeypatch.setattr(TC, "vllm_status", lambda binary=None: (True, "vLLM with compiled kernels (fake)"))


# ------------------------------------------------------------------ which backend a build gets

def test_lane_and_backend_from_the_files(tmp_path):
    g = tmp_path / "m-Q4_K_M.gguf"
    g.write_bytes(b"GGUF")
    mlx = _dir(tmp_path, "m-mlx", {"quantization": {"group_size": 64, "bits": 4}})
    gptq = _dir(tmp_path, "m-gptq", {"quantization_config": {"quant_method": "gptq", "bits": 4}})
    awq = _dir(tmp_path, "m-awq", {"quantization_config": {"quant_method": "awq"}})
    mx = _dir(tmp_path, "m-nvfp4", {"quantization_config": {"quant_method": "compressed-tensors"}})
    exl3 = _dir(tmp_path, "m-exl3", {"quantization_config": {"quant_method": "exl3", "bits": 4.0}})
    hf = _dir(tmp_path, "m", {"architectures": ["Qwen2ForCausalLM"]})
    assert [TC.lane_of(p) for p in (str(g), mlx, gptq, awq, mx, exl3, hf)] == \
        ["gguf", "mlx", "gptq", "gptq", "mx", "exl3", "hf"]
    assert [TC.pick_backend(p) for p in (str(g), mlx, gptq, mx, exl3)] == ["gguf", "mlx", "vllm", "vllm", "exl3"]
    # an unquantized checkpoint (a reference) is served by the rung's own backend when that backend loads HF weights
    assert TC.pick_backend(hf, prefer="mlx") == "mlx" and TC.pick_backend(hf, prefer="vllm") == "vllm"
    assert TC.pick_backend(str(g), url="http://x") == "url"
    # exllamav3 loads only EXL3 weights: an EXL3 rung's bf16 reference runs in-process in transformers
    assert TC.pick_backend(hf, prefer="exl3") == "hf" and TC.pick_backend(hf, prefer="hf") == "hf"
    # no usable vLLM here (Windows' kernel-less source build, not installed): GPTQ / MX go in-process
    assert [TC.pick_backend(p, vllm_ok=False) for p in (gptq, awq, mx, exl3)] == ["hf", "hf", "hf", "exl3"]
    assert TC.pick_backend(hf, prefer="vllm", vllm_ok=False) == "hf"


def test_the_workspace_manifest_lane_wins(tmp_path, monkeypatch):
    d = _dir(tmp_path, "m-x", {})
    monkeypatch.setattr(ws, "find_build", lambda p: ("Org/M", {"lane": "mx", "path": d}))
    assert TC.lane_of(d) == "mx" and TC.pick_backend(d) == "vllm"


# ------------------------------------------------------------------ how each server is started

def test_gguf_command_is_llama_server_jinja_on_the_cpu_by_default():
    cmd = TC.gguf_cmd("llama-server", "m.gguf", 9000, "0")
    assert "--jinja" in cmd and cmd[cmd.index("-ngl") + 1] == "0" and cmd[cmd.index("--port") + 1] == "9000"
    assert cmd[cmd.index("--reasoning-format") + 1] == "deepseek"


def test_mlx_command_is_mlx_lm_server():
    cmd = TC.mlx_cmd("/py", "./M-MLX", 9001, trust_remote_code=True)
    assert cmd[:3] == ["/py", "-m", "mlx_lm.server"] and cmd[cmd.index("--model") + 1] == "./M-MLX"
    assert "--trust-remote-code" in cmd and cmd[cmd.index("--temp") + 1] == "0"


def test_vllm_command_carries_the_familys_tool_parser():
    cmd = TC.vllm_cmd("vllm", "./M-GPTQ", 9002, "qwen3_coder", ctx=4096)
    assert cmd[:3] == ["vllm", "serve", "./M-GPTQ"] and "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert cmd[cmd.index("--max-model-len") + 1] == "4096" and cmd[cmd.index("--served-model-name") + 1] == TC.SERVED_NAME


def _fake_vllm(root, compiled):
    pkg = root / "site" / "vllm"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "_custom_ops.py").write_text("")                  # what a case-blind `_C*` glob matched on Windows
    if compiled:
        (pkg / ("_C.pyd" if os.name == "nt" else "_C.abi3.so")).write_bytes(b"")
    return root / "site"


@pytest.mark.parametrize("compiled, ok", [(True, True), (False, False)])
def test_vllm_is_usable_only_with_its_compiled_kernels(tmp_path, monkeypatch, compiled, ok):
    """`pip install vllm` on Windows builds a source dist with no vllm/_C: installed, but it serves nothing."""
    monkeypatch.undo()                                                           # the real vllm_status
    site = _fake_vllm(tmp_path, compiled)
    monkeypatch.setenv("PYTHONPATH", str(site))
    monkeypatch.setattr(TC, "_VLLM_STATUS", {})
    monkeypatch.setattr(TC, "_has_module", lambda n: n == "vllm")
    monkeypatch.setattr(__import__("shutil"), "which", lambda n: None)
    monkeypatch.delenv("POLLARD_VLLM", raising=False)
    got, why = TC.vllm_status()
    assert got is ok
    if not ok:
        assert "no compiled CUDA kernels" in why


def test_no_vllm_at_all(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(TC, "_VLLM_STATUS", {})
    monkeypatch.setattr(TC, "_has_module", lambda n: False)
    monkeypatch.setattr(__import__("shutil"), "which", lambda n: None)
    monkeypatch.delenv("POLLARD_VLLM", raising=False)
    assert TC.vllm_status() == (False, "vLLM is not installed here")


def test_an_unusable_vllm_falls_back_to_in_process_transformers(monkeypatch):
    monkeypatch.setattr(TC, "vllm_status", lambda binary=None: (False, "the vLLM installed here has no compiled CUDA "
                                                                       "kernels"))
    b, note = TC.effective_backend("vllm")
    assert b == "hf" and "in-process with transformers" in note and "compiled" in note
    assert TC.effective_backend("exl3") == ("exl3", None) and TC.effective_backend("gguf") == ("gguf", None)


def test_vllm_requests_name_the_served_model():
    p = TC.SUITE[0]
    assert TC.request_for(p, 64, extra={"model": TC.SERVED_NAME})["model"] == TC.SERVED_NAME
    assert "model" not in TC.request_for(p, 64)          # mlx_lm.server would try to LOAD a named model


# ------------------------------------------------------------------ the shared-GPU rule and the memory guards

SMI = """\
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1234    C+G   C:\\Windows\\explorer.exe                  N/A      |
|    0   N/A  N/A            2345    C+G   ...\\dwm.exe                               N/A      |
|    0   N/A  N/A            7777    C+G   ...\\Steam\\game\\Game-Win64-Shipping.exe     N/A      |
+-----------------------------------------------------------------------------------------+
"""


def test_a_game_on_the_gpu_is_a_user_and_the_desktop_shell_is_not():
    assert TC.gpu_users(SMI, "") == ["Game-Win64-Shipping.exe"]
    assert TC.gpu_users(SMI.replace("Game-Win64-Shipping.exe", "dwm.exe"), "") == []
    assert TC.gpu_users("", "4242, python.exe, 9000\n") == ["python.exe"]          # a compute job


def test_the_box_owner_can_name_always_on_desktop_apps_but_a_game_still_counts(monkeypatch):
    smi = SMI.replace("Game-Win64-Shipping.exe", "Discord.exe") + SMI
    assert TC.gpu_users(smi, "") == ["Discord.exe", "Game-Win64-Shipping.exe"]
    monkeypatch.setenv("POLLARD_GPU_IGNORE", "discord, medal")
    assert TC.gpu_users(smi, "") == ["Game-Win64-Shipping.exe"]


def test_gpu_backends_refuse_a_busy_missing_or_small_gpu():
    with pytest.raises(TC.NotMeasured, match="no NVIDIA|none"):
        TC.check_gpu("vllm", 4.0, users=[], free=None)
    with pytest.raises(TC.NotMeasured, match="another process resident"):
        TC.check_gpu("vllm", 4.0, users=["Game.exe"], free=15.0)
    with pytest.raises(TC.NotMeasured, match="VRAM"):
        TC.check_gpu("exl3", 12.0, users=[], free=14.0)
    TC.check_gpu("vllm", 4.0, users=[], free=15.0)                              # idle and roomy: fine


def test_ram_guard():
    with pytest.raises(TC.NotMeasured, match="RAM"):
        TC.check_ram(20.0, avail=12.0)
    TC.check_ram(1.0, avail=12.0)
    TC.check_ram(0.0, avail=1.0)                                                # unknown size: no guard


def test_gpu_free_is_the_emptiest_card():
    assert TC.gpu_free_gb("1024\n8192\n") == 8.0 and TC.gpu_free_gb("") is None


def test_preflight_not_measured_paths(monkeypatch, tmp_path):
    d = _dir(tmp_path, "m", {})
    monkeypatch.setattr(TC, "gpu_free_gb", lambda *a: None)
    with pytest.raises(TC.NotMeasured, match="needs an NVIDIA GPU"):             # a Mac: the first reason
        TC.preflight("vllm", d, kind={"tools": {"vllm_parser": "hermes"}}, server="/bin/vllm")
    monkeypatch.setattr(TC, "gpu_free_gb", lambda *a: 15.0)
    monkeypatch.setattr(TC, "_apple", lambda: False)
    with pytest.raises(TC.NotMeasured, match="Apple Silicon"):
        TC.preflight("mlx", d)
    monkeypatch.setattr(TC, "vllm_status", lambda binary=None: (False, "vLLM is not installed here"))
    with pytest.raises(TC.NotMeasured, match="vLLM is not installed"):
        TC.preflight("vllm", d, kind={"tools": {"family": "hermes", "vllm_parser": "hermes"}})
    monkeypatch.setattr(TC, "vllm_status", lambda binary=None: (True, "ok"))
    with pytest.raises(TC.NotMeasured, match="no vLLM --tool-call-parser"):
        TC.preflight("vllm", d, kind={"tools": {"family": "cohere_cmdr", "vllm_parser": None}}, server="/bin/vllm")
    monkeypatch.setattr(TC, "_has_module", lambda n: False)
    with pytest.raises(TC.NotMeasured, match="exllamav3 is not installed"):
        TC.preflight("exl3", d)
    with pytest.raises(TC.NotMeasured, match="transformers \\+ torch are not installed"):
        TC.preflight("hf", d)
    monkeypatch.setattr(TC, "_has_module", lambda n: True)
    with pytest.raises(TC.NotMeasured, match="no in-process reader for the 'kimi_k3'"):
        TC.preflight("exl3", d, kind={"tools": {"family": "kimi_k3"}})
    # vLLM present and a parser known: the GPU rule decides
    monkeypatch.setattr(TC, "gpu_users", lambda *a: ["Game.exe"])
    with pytest.raises(TC.NotMeasured, match="resident"):
        TC.preflight("vllm", d, kind={"tools": {"vllm_parser": "hermes"}}, server="/bin/vllm")
    monkeypatch.setattr(TC, "gpu_users", lambda *a: [])
    assert TC.preflight("vllm", d, kind={"tools": {"vllm_parser": "hermes"}}, server="/bin/vllm") == \
        {"model": TC.SERVED_NAME}


def test_url_backend_uses_the_running_server():
    with TC.backend_server("url", None, url="http://127.0.0.1:8000/v1/") as base:
        assert base == "http://127.0.0.1:8000"
    with pytest.raises(TC.NotMeasured):
        with TC.backend_server("url", None):
            pass


# ------------------------------------------------------------------ main: not measured, cross-backend reference

def _rows(n_ok):
    return [{"id": p["id"], "category": p["category"], "expected": p["expect"] is not None, "called": True,
             "parsed": True, "name_ok": True, "args_valid": True, "exact": i < n_ok, "ok": i < n_ok,
             "failure": None if i < n_ok else "wrong-value", "detail": "", "sample": ""}
            for i, p in enumerate(TC.SUITE)]


def test_a_rung_this_machine_cannot_serve_is_recorded_not_faked(monkeypatch, tmp_path):
    gptq = _dir(tmp_path, "M-Pollard-GPTQ-int4", {"quantization_config": {"quant_method": "gptq"}})

    def nope(*a, **k):
        raise TC.NotMeasured("vLLM is not installed here")
    monkeypatch.setattr(TC, "run_model", nope)
    res = tmp_path / "results.json"
    with pytest.raises(SystemExit) as e:
        TC.main(["--model", gptq, "--ref", str(tmp_path), "--results", str(res)])
    assert e.value.code == 0                                                    # the build goes on
    tc = json.load(open(res))["M-Pollard-GPTQ-int4"]["toolcall"]
    assert tc == {"status": "not-measured", "reason": "vLLM is not installed here", "backend": "vllm",
                  "mode": "full", "lane": "gptq"}
    assert C.tool_ready_cell(tc) == "not measured (vLLM is not installed here)"
    m = V.metrics_from_results(json.load(open(res)), "M-Pollard-GPTQ-int4")
    assert m["toolcall_status"] == "not-measured"
    th = {"toolcall_retention_min": 90}
    row = V.evaluate_gate(m, th)["checks"][0]
    assert row["ok"] is True and row["why"] == "not measured: vLLM is not installed here", \
        "default: a rung whose tool calls could not be measured here publishes, with the reason"
    assert V.evaluate_gate(m, th, missing="fail")["pass"] is True, "not-measured is its own policy, not `missing`"
    failed = V.evaluate_gate(m, dict(th, toolcall_not_measured="fail"))
    assert failed["pass"] is False and failed["checks"][0]["why"] == "not measured: vLLM is not installed here"


def test_reference_falls_back_to_the_f16_gguf_cross_backend(monkeypatch, tmp_path):
    gptq = _dir(tmp_path, "M-Pollard-GPTQ-int4", {"quantization_config": {"quant_method": "gptq"}})
    hf = _dir(tmp_path, "M", {"architectures": ["X"]})
    f16 = tmp_path / "M-f16.gguf"
    f16.write_bytes(b"GGUF")
    monkeypatch.setattr(TC, "run_model", lambda *a, **k: (_rows(45), {"family": "hermes", "backend": "vllm"}))
    seen = []

    def ref(path, ngl, **k):
        seen.append((path, k["backend"]))
        if k["backend"] == "vllm":
            raise TC.NotMeasured("the GPU has another process resident (Game.exe) -- not ours to take")
        return _rows(50), {"family": "hermes"}
    monkeypatch.setattr(TC, "run_ref_cached", ref)
    res = tmp_path / "results.json"
    with pytest.raises(SystemExit):
        TC.main(["--model", gptq, "--ref", hf, "--ref-fallback", str(f16), "--results", str(res)])
    assert seen == [(hf, "vllm"), (str(f16), "gguf")]                         # same backend first, then the f16
    tc = json.load(open(res))["M-Pollard-GPTQ-int4"]["toolcall"]
    assert tc["status"] == "measured" and tc["backend"] == "vllm" and tc["ref_backend"] == "gguf"
    assert tc["cross_backend"] is True and tc["retention"] == 90.0 and "resident" in tc["ref_note"]
    assert C.tool_ready_cell(tc).startswith("yes (90% of f16 GGUF, cross-backend")


def test_an_mlx_rung_is_compared_to_its_bf16_through_mlx(monkeypatch, tmp_path):
    mlx = _dir(tmp_path, "M-Pollard-MLX-4bit", {"quantization": {"bits": 4, "group_size": 64}})
    hf = _dir(tmp_path, "M", {"architectures": ["X"]})
    got = {}
    monkeypatch.setattr(TC, "run_model", lambda m, ngl, **k: got.setdefault("rung", k["backend"]) and
                        (_rows(48), {"family": "hermes"}))
    monkeypatch.setattr(TC, "run_ref_cached", lambda r, ngl, **k: got.setdefault("ref", k["backend"]) and
                        (_rows(50), {"family": "hermes"}))
    res = tmp_path / "results.json"
    with pytest.raises(SystemExit):
        TC.main(["--model", mlx, "--ref", hf, "--results", str(res)])
    assert got == {"rung": "mlx", "ref": "mlx"}
    tc = json.load(open(res))["M-Pollard-MLX-4bit"]["toolcall"]
    assert tc["cross_backend"] is False and tc["lane"] == "mlx"
    assert C.tool_ready_cell(tc).startswith("yes (96% of bf16")


def test_ref_cache_for_a_checkpoint_dir_lives_in_the_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    hf = _dir(tmp_path, "snap", {})
    p = TC._ref_cache_path(hf, "k")
    assert p.startswith(str(tmp_path / "home")) and "snap-" in os.path.basename(p)
    assert TC.fingerprint(hf).startswith("10")                                   # weight + config bytes


# ------------------------------------------------------------------ the card, per lane

TOOLS = {"family": "hermes", "supports_tools": True, "vllm_parser": "hermes", "vllm_parser_confirmed": True,
         "vllm_flags": "--enable-auto-tool-choice --tool-call-parser hermes", "llama_cpp": {"format": "autoparser"}}


@pytest.mark.parametrize("lane, want", [
    ("gguf", "llama-server -m m.gguf --jinja"),
    ("mlx", "mlx_lm.server --model Org/M-Pollard-MLX"),
    ("gptq", "vllm serve Org/M-Pollard-MLX --enable-auto-tool-choice --tool-call-parser hermes"),
    ("mx", "vllm serve Org/M-Pollard-MLX --enable-auto-tool-choice --tool-call-parser hermes"),
    ("exl3", "TabbyAPI"),
])
def test_tool_calling_section_has_the_right_server_for_each_lane(lane, want):
    md = "\n".join(C.tool_serving_section(TOOLS, [lane], "Org/M-Pollard-MLX", "m.gguf"))
    assert "## Tool calling" in md and want in md
    if lane in ("mlx", "exl3"):
        assert "llama-server" not in md and "vllm serve" not in md


def test_tool_ready_note_names_the_servers_measured():
    builds = [{"name": "a"}, {"name": "b"}]
    note = C.tool_ready_note(builds, {"a": {"toolcall": {"backend": "mlx", "rate": 1.0}},
                                      "b": {"toolcall": {"status": "not-measured", "reason": "x"}}})
    assert "`mlx_lm.server`" in note and "llama-server" not in note


def test_retention_section_says_not_measured():
    builds = [{"name": "r", "bytes": 1}]
    md = "\n".join(C.retention_section(builds, {"r": {"toolcall": {"status": "not-measured", "reason": "x"}}}))
    assert "| `r` | not measured |" in md


@pytest.mark.parametrize("reason", ["", "candidate: 1 server-error probes | unavailable"])
def test_card_distinguishes_toolcall_errors_from_model_capability(reason):
    tc = {"status": "error", "pass": False, "rate": None, "reason": reason, "backend": "vllm"}
    expected = "error (candidate: 1 server-error probes / unavailable)" if reason else "error"
    assert C.tool_ready_cell(tc) == expected
    md = "\n".join(C.retention_section([{"name": "r", "bytes": 1}], {"r": {"toolcall": tc}}))
    assert "| `r` | error |" in md


def test_tool_ready_note_excludes_failed_measurements():
    builds = [{"name": "ok"}, {"name": "error"}]
    results = {"ok": {"toolcall": {"backend": "mlx", "rate": 1.0}},
               "error": {"toolcall": {"backend": "vllm", "status": "error", "pass": False}}}
    note = C.tool_ready_note(builds, results)
    assert "`mlx_lm.server`" in note and C.TOOL_SERVERS["vllm"] not in note
    note = C.tool_ready_note(builds[1:], results)
    assert "no completed measurement" in note and "served with" not in note


# ------------------------------------------------------------------ release: every lane's rungs gated

def _ctx(tmp_path):
    return {"src": "Org/M", "hf_dir": str(tmp_path / "M"), "f16": str(tmp_path / "M-f16.gguf"),
            "mdir": str(tmp_path), "smooth_dir": str(tmp_path / "M-smoothed"), "calib": "c.txt", "heldout": "h.txt",
            "sens_raw": "s.json", "sens_smooth": "ss.json", "ngl": "0", "gates": None, "gguf_dir": str(tmp_path / "gguf")}


@pytest.mark.parametrize("lane, backend", [("mlx", "mlx"), ("gptq", "vllm"), ("mx", "vllm"), ("exl3", "exl3")])
def test_release_measures_tool_calls_on_every_lanes_rungs(monkeypatch, tmp_path, lane, backend):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(R, "toolcall_wanted", lambda cfg, c: True)
    cfg = R.load_plan()
    steps = R.lane_steps(cfg, _ctx(tmp_path), lane)
    ids = [s.id for s in steps]
    tcs = [s for s in steps if s.id.endswith("/toolcall")]
    assert tcs and len(tcs) == len([i for i in ids if i.endswith("/build")])
    for s in tcs:
        rid = s.id.rsplit("/", 1)[0]
        assert ids.index(f"{rid}/build") < ids.index(s.id) < ids.index(f"{rid}/verify")
        assert s.cmd[s.cmd.index("--backend") + 1] == backend and s.ok_codes == (0, 1)
        assert s.cmd[s.cmd.index("--ref") + 1] == str(tmp_path / "M")          # the ORIGINAL weights
        assert ("--ref-fallback" in s.cmd) == (lane != "mlx")
        assert ("--ref-backend" in s.cmd) == (lane == "exl3")
        if lane == "exl3":                                  # exllamav3 in-process: its bf16 reference in-process too
            assert s.cmd[s.cmd.index("--ref-backend") + 1] == "hf"
    monkeypatch.setattr(R, "toolcall_wanted", lambda cfg, c: False)
    assert not [s for s in R.lane_steps(cfg, _ctx(tmp_path), lane) if s.id.endswith("/toolcall")]


def test_release_gguf_ladder_leaves_the_gate_to_its_own_rung_steps(tmp_path):
    (step, _) = R.lane_steps(R.load_plan(), _ctx(tmp_path), "gguf")
    assert "--no-toolcall" in step.cmd


def test_verify_treats_a_measured_rung_as_agentic_even_when_its_dir_is_unreadable(monkeypatch):
    th = V.thresholds_for(V.load_gates(None), "M-Pollard-MLX-3bit", "mlx", ["agentic"])
    assert th["toolcall_retention_min"] == 85                                   # the 3-bit band covers MLX names
    th = V.thresholds_for(V.load_gates(None), "M-Pollard-EXL3-2.5bpw", "exl3", ["agentic"])
    assert th["toolcall_retention_min"] == 75
    assert V.thresholds_for(V.load_gates(None), "M-Pollard-MX-NVFP4", "mx", ["agentic"])["toolcall_retention_min"] == 90


# ------------------------------------------------------------------ calibration: whole samples for every lane

def test_calib_samples_keep_a_multiline_tool_trace_whole(tmp_path):
    import pollard_calib as PC
    trace = "<|im_start|>user\nweather?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"name\": \"w\"}\n</tool_call>"
    corpus = tmp_path / "calib.txt"
    corpus.write_text("\n".join([trace, "plain prose"]))
    assert len(ws.calib_samples(str(corpus))) == 7                               # no twin: lines, as before
    PC.write_samples(str(corpus), [trace, "plain prose"])
    assert ws.calib_samples(str(corpus)) == [trace, "plain prose"]
    os.utime(PC.samples_path(str(corpus)), (1, 1))                               # a stale twin is ignored
    assert len(ws.calib_samples(str(corpus))) == 7


def test_export_lanes_read_whole_samples():
    for f in ("pollard_export.py", "pollard_mx.py"):
        assert "ws.calib_samples(a.calib)" in (ROOT / "tools" / f).read_text()


def test_smoothing_calibrates_on_the_lane_corpus(monkeypatch, tmp_path):
    import pollard_auto as pa
    a = types.SimpleNamespace(format="gguf", output=None, calib=None, run=False, hf="Org/M", gguf=None,
                              calib_profile="auto")
    assert pa._smooth_calib(a, str(tmp_path)) is None
    a.format, a.output = "gptq", str(tmp_path / "out" / "M-GPTQ")
    seen = {}
    monkeypatch.setattr(pa, "_ensure_calib_text", lambda a, here: seen.update(here=here, src=a._hf_dir) or "c.txt")
    assert pa._smooth_calib(a, "/hf/M") == "c.txt"
    assert seen == {"here": str(tmp_path / "out"), "src": "/hf/M"} and getattr(a, "_hf_dir", None) is None
