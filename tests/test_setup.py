"""pollard setup: lane detection, the report, the install plan per OS/GPU, the prompts, the cached first-run hook,
and the one device helper. Everything is faked -- no torch, no GPU, no network, nothing installed."""
import json
import os
import subprocess
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_device as D  # noqa: E402
import pollard_setup as S  # noqa: E402


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "ph"))
    monkeypatch.delenv("POLLARD_NO_SETUP_HINT", raising=False)
    monkeypatch.delenv("POLLARD_SKIP_SETUP_CHECK", raising=False)
    return tmp_path / "ph"


# ------------------------------------------------------------------------------------------ fakes
def ok(version="1.0", **kw):
    return {"ok": True, "version": version, **kw}


def gone():
    return {"ok": False, "why": "not installed"}


def torch_ok(dev, cuda=None, mps=None, torch_cuda=None, version="2.9.0"):
    return {"ok": True, "version": version, "op": f"ok on {dev}",
            "detail": {"torch": version, "torch_cuda": torch_cuda, "device": dev,
                       "cuda": cuda or [dev == "cuda", "sm_120" if dev == "cuda" else "no CUDA device visible"],
                       "mps": mps or [dev == "mps", "Apple GPU (Metal)" if dev == "mps" else "not an Apple machine"]}}


def mac_facts(**mods):
    m = {"numpy": ok(), "gguf": ok(), "mlx": ok("0.30"), "mlx_lm": ok("0.31"), "huggingface_hub": ok(),
         "torch": torch_ok("mps"), "transformers": ok("5.17"), "safetensors": ok(), "datasets": ok()}
    m.update(mods)
    return {"os": "darwin", "arch": "arm64", "apple": True, "python": "/v/bin/python", "pyver": "3.12.0",
            "venv": True, "deep": True, "gpu": None,
            "toolchain": {"nvcc": None, "cmake": (4, 0, 2), "git": True, "cc": True, "xcode_clt": True, "msvc": None},
            "ram_free_gb": 20.0, "disk_free_gb": 200.0, "modules": m,
            "runtime": {"llama.cpp": {"ok": True, "detail": "Metal build", "fix": []},
                        "ik_llama.cpp": {"ok": False, "detail": "not built", "fix": ["py", "pollard_runtime.py",
                                                                                       "--update", "ik_llama.cpp"]}},
            "vllm": {"ok": False, "why": "vLLM is not installed here"}}


def box_facts(torch=None, nvcc=(12, 8), msvc="C:/vc/vcvars64.bat", driver=(12, 8), caps=("12.0",), os_="win32",
              venv=True, **mods):
    m = {"numpy": ok(), "gguf": ok(), "huggingface_hub": ok(), "transformers": ok("5.17"), "safetensors": ok(),
         "datasets": ok(), "xxhash": ok(), "compressed_tensors": ok(), "llmcompressor": ok(), "gptqmodel": ok(),
         "exllamav3": ok(ext="prebuilt"),
         "torch": torch or torch_ok("cuda", torch_cuda="12.8")}
    m.update(mods)
    return {"os": os_, "arch": "AMD64", "apple": False, "python": "C:/pollard/venv312/Scripts/python.exe",
            "pyver": "3.12.0", "venv": venv, "deep": True,
            "gpu": {"present": True, "names": ["NVIDIA GeForce RTX 5070 Ti"], "caps": list(caps), "vram_gb": 16.0,
                    "free_gb": 14.0, "driver_cuda": driver, "users": []},
            "toolchain": {"nvcc": nvcc, "cmake": (4, 0, 2), "git": True, "cc": True, "xcode_clt": None, "msvc": msvc},
            "ram_free_gb": 40.0, "disk_free_gb": 300.0, "modules": m,
            "runtime": {"llama.cpp": {"ok": True, "detail": "CUDA 120a-real build", "fix": []}},
            "vllm": {"ok": False, "why": "the vLLM installed here has no compiled CUDA kernels"}}


def rows_by_lane(facts, lanes=None):
    return {r[0]: r for r in S.evaluate(facts, lanes)}


# ------------------------------------------------------------------------------------------ torch wheel choice
@pytest.mark.parametrize("caps,driver,want", [
    (["12.0"], (12, 8), "cu128"),        # Blackwell RTX 50xx: kernels start at cu128
    (["12.0"], (13, 0), "cu130"),
    (["10.0"], (12, 9), "cu128"),        # B200
    (["8.6"], (12, 6), "cu126"),         # Ampere on an older driver
    (["8.9"], (13, 1), "cu130"),
    (["6.1"], (13, 0), "cu126"),         # Pascal: CUDA 13 dropped it, cu126 is the last wheel with it
    (["7.0"], (13, 0), "cu128"),         # Volta: no cu130 (sm < 75), cu128 still has it
    (["10.7"], (13, 0), "cu130"),        # Rubin: CUDA 13 only
])
def test_torch_cuda_tag(caps, driver, want):
    assert S.torch_cuda_tag(caps, driver)[0] == want


def test_torch_cuda_tag_refuses_a_driver_too_old_for_the_card():
    tag, why = S.torch_cuda_tag(["12.0"], (12, 4))
    assert tag is None and "update the driver" in why
    assert S.torch_cuda_tag(["12.0"], None)[0] == "cu130"      # unknown driver: newest wheel that has the arch


def test_torch_index_url_per_machine():
    assert S.torch_index_url(mac_facts()) is None                           # PyPI's macOS wheel has MPS
    assert S.torch_index_url(box_facts()).endswith("/cu128")
    assert S.torch_index_url({"os": "linux", "gpu": None}).endswith("/whl/cpu")
    assert S.torch_index_url({"os": "win32", "gpu": None}) is None          # PyPI's Windows wheel is the CPU one


# ------------------------------------------------------------------------------------------ device helper
def fake_torch(cuda=False, count=1, cap=(12, 0), arches=("sm_80", "sm_90", "sm_120"), mps=False, cuda_ver="12.8"):
    t = types.SimpleNamespace(__version__="2.9.0")
    t.version = types.SimpleNamespace(cuda=cuda_ver if cuda or cuda_ver else None)
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda, device_count=lambda: count if cuda else 0,
                                   get_device_capability=lambda i=0: cap, get_arch_list=lambda: list(arches),
                                   is_bf16_supported=lambda: True,
                                   get_device_properties=lambda i: types.SimpleNamespace(total_memory=16 * 2**30))
    t.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    t.bfloat16, t.float16, t.float32 = "bf16", "fp16", "fp32"
    return t


def test_wheel_covers():
    assert D.wheel_covers((12, 0), ["sm_90", "sm_120"])
    assert not D.wheel_covers((12, 0), ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"])  # cu124
    assert D.wheel_covers((12, 0), ["sm_80", "compute_90"])                # PTX JITs forward
    assert not D.wheel_covers((8, 6), ["sm_90", "compute_90"])             # ...never backward
    assert D.wheel_covers((8, 6), [])                                       # unknown list: no guess-refusal


def test_pick_native_everywhere():
    assert D.pick("auto", fake_torch(cuda=True)) == "cuda"
    assert D.pick("auto", fake_torch(mps=True, cuda_ver=None)) == "mps"
    assert D.pick("auto", fake_torch(cuda_ver=None)) == "cpu"
    assert D.pick("cuda:1", fake_torch(cuda=True)) == "cuda:1"
    assert D.pick("cpu", fake_torch(cuda=True)) == "cpu"


def test_pick_falls_back_with_a_note():
    notes = []
    assert D.pick("mps", fake_torch(cuda_ver=None), note=notes.append) == "cpu"
    assert D.pick("cuda", fake_torch(mps=True, cuda_ver=None), note=notes.append) == "mps"
    assert len(notes) == 2 and "not usable" in notes[0]


def test_a_wheel_without_kernels_for_the_card_is_not_cuda():
    t = fake_torch(cuda=True, arches=("sm_50", "sm_80", "sm_90"), cuda_ver="12.4")
    usable, why = D.cuda_ok(t)
    assert not usable and "no kernels for this GPU" in why and "sm_120" in why
    assert D.pick("auto", t) == "cpu"


def test_hidden_gpu_is_not_cuda():
    assert D.cuda_ok(fake_torch(cuda=True, count=0))[0] is False


def test_dtype_and_memory():
    t = fake_torch(cuda=True)
    assert D.dtype_for("cuda", t) == "bf16" and D.dtype_for("mps", t) == "fp16" and D.dtype_for("cpu", t) == "fp32"
    assert D.accel_bytes("cuda:0", t) == int(16 * 2**30 * 0.9) and D.accel_bytes("cpu", t) == 0


def test_no_torch_means_cpu(monkeypatch):
    monkeypatch.setattr(D, "_torch", lambda torch=None: None)
    assert D.pick("auto") == "cpu" and D.describe()["device"] == "cpu"


# ------------------------------------------------------------------------------------------ evaluation
def test_mac_report_marks_cuda_lanes_impossible_with_why():
    r = rows_by_lane(mac_facts())
    assert r["mlx"][1] == "ready" and r["gguf"][1] == "ready" and r["convert"][1] == "ready"
    for lane in ("gptq", "mx", "exl3", "exlr8"):
        assert r[lane][1] == "impossible" and "NVIDIA" in r[lane][3] and f"--lanes {lane}" in r[lane][3]
    assert r["ik"][1] == "missing"                                          # optional, reported, not defaulted
    assert "ik" not in S.default_lanes(mac_facts())


def test_box_report_marks_mlx_impossible():
    r = rows_by_lane(box_facts())
    assert r["mlx"][1] == "impossible" and "Apple Silicon" in r["mlx"][3]
    assert all(r[lane][1] == "ready" for lane in ("gptq", "mx", "exl3", "exlr8", "gguf"))
    assert "transformers" in r["toolcall"][3] or r["toolcall"][1] == "ready"


def test_cpu_torch_on_an_nvidia_box_is_the_wrong_torch():
    f = box_facts(torch=torch_ok("cpu", torch_cuda=None))
    r = rows_by_lane(f)
    assert r["gptq"][1] == "missing" and "CPU-only build" in r["gptq"][3]
    acts = S.plan_actions(f, ["gptq"])
    t = [a for a in acts if a.kind == "pip" and "torch" in a.cmd]
    assert t and "--force-reinstall" in t[0].cmd and t[0].cmd[-1].endswith("/cu128")


def test_wheel_without_blackwell_kernels_is_replaced():
    bad = torch_ok("cpu", torch_cuda="12.4",
                   cuda=[False, "torch 2.4 (CUDA 12.4) has no kernels for this GPU (sm_120; wheel carries sm_90)"])
    f = box_facts(torch=bad)
    assert S.requirement("torch", f)[0] == "wrong"
    cmd = next(a.cmd for a in S.plan_actions(f, ["exl3"]) if a.kind == "pip" and "torch" in a.cmd)
    assert "--force-reinstall" in cmd and "https://download.pytorch.org/whl/cu128" in cmd


def test_pip_plan_uses_no_build_isolation_for_cuda_extensions():
    f = box_facts(gptqmodel=gone(), xxhash=gone(), llmcompressor=gone())
    acts = S.plan_actions(f, ["gptq", "exlr8", "mx"])
    plain = [a for a in acts if a.kind == "pip" and "xxhash" in a.cmd]
    assert plain and "llmcompressor" in plain[0].cmd and "gptqmodel" not in plain[0].cmd
    g = next(a for a in acts if a.kind == "pip" and "gptqmodel" in a.cmd)
    assert "--no-build-isolation" in g.cmd and g.cmd[:3] == [f["python"], "-m", "pip"]


def test_exl3_jit_needs_matching_nvcc_and_msvc():
    f = box_facts(exllamav3=ok(ext="jit"), nvcc=(13, 0), msvc=None)
    state, why = S.requirement("exl3-ext", f)
    assert state == "wrong" and "13.0" in why and "12.8" in why
    acts = S.plan_actions(f, ["exl3"])
    sys_ = [a for a in acts if a.kind == "system"]
    assert [a.what for a in sys_] == ["cuda", "msvc"]
    assert "winget" in sys_[0].cmd and "cuda-12-8-0" in sys_[0].link
    assert "VCTools" in sys_[1].cmd
    f2 = box_facts(exllamav3=ok(ext="jit"), nvcc=(12, 8))
    assert S.requirement("exl3-ext", f2)[0] == "ok"
    assert S.requirement("exl3-ext", box_facts(exllamav3=ok(ext="jit"), nvcc=None))[0] == "missing"


def test_system_pieces_on_a_mac_and_linux():
    f = mac_facts()
    f["toolchain"]["xcode_clt"] = False
    f["runtime"]["llama.cpp"] = {"ok": False, "detail": "not built", "fix": ["py", "rt", "--update", "llama.cpp"]}
    acts = S.plan_actions(f, ["gguf"])
    assert acts[0].kind == "system" and acts[0].cmd == "xcode-select --install"
    assert acts[-1].kind == "runtime" and acts[-1].cmd[-2:] == ["--update", "llama.cpp"]
    lin = box_facts(os_="linux", nvcc=None)
    lin["runtime"]["llama.cpp"] = {"ok": False, "detail": "not built", "fix": []}
    lin["toolchain"]["cc"] = False
    whats = [a.what for a in S.plan_actions(lin, ["gguf"]) if a.kind == "system"]
    assert whats == ["cc", "cuda"]                        # without nvcc the runtime would build CPU-only


def test_old_driver_is_a_system_fix_not_a_pip_guess():
    f = box_facts(torch=gone(), driver=(12, 2))
    acts = S.plan_actions(f, ["gptq"])
    assert any(a.kind == "system" and a.what == "driver" for a in acts)
    assert not any(a.kind == "pip" and "torch" in a.cmd for a in acts)


def test_speculators_gets_its_own_env(home):
    f = box_facts()
    f["speculators"] = {"python": None, "ok": False, "why": "no speculators env"}
    acts = S.plan_actions(f, ["draft-train"])
    v = next(a for a in acts if a.kind == "venv")
    assert v.cmds[0][1:3] == ["-m", "venv"] and str(home) in v.dir
    assert "transformers<5.17" in v.cmds[-1] and any("cu128" in " ".join(c) for c in v.cmds)
    ran = []
    res = S.apply([v], yes=True, out=lambda m: None, runner=lambda c: ran.append(c) or subprocess.CompletedProcess(c, 0))
    assert res[0][1] and len(ran) == 4
    cfg = json.load(open(os.path.join(str(home), "config.json")))
    assert cfg["envs"]["speculators"]["python"] == v.python


def test_draft_trainer_uses_the_recorded_env(home, monkeypatch):
    py = home / "envs" / "speculators" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    S.save_config({"envs": {"speculators": {"python": str(py)}}})
    monkeypatch.delenv("POLLARD_SPECULATORS_PYTHON", raising=False)
    import pollard_draft_train as DT
    assert DT._python_for_trainer() == str(py)
    assert DT._python_for_trainer("/explicit") == "/explicit"


# ------------------------------------------------------------------------------------------ runtime build
def _tree(tmp_path, cache):
    t = tmp_path / "llama.cpp"
    (t / "build" / "bin").mkdir(parents=True)
    for b in ("llama-quantize", "llama-cli", "llama-server", "llama-imatrix", "llama-perplexity"):
        (t / "build" / "bin" / (b + (".exe" if S.WIN else ""))).write_text("")
    (t / "build" / "CMakeCache.txt").write_text("\n".join(f"{k}:STRING={v}" for k, v in cache.items()))
    return t


def test_runtime_must_match_the_accelerator(tmp_path, monkeypatch):
    import pollard_runtime_update as U
    f = box_facts()
    t = _tree(tmp_path, {"GGML_CUDA": "OFF"})
    monkeypatch.setitem(U.ENGINES, "llama.cpp", {"dir": str(t), "url": ""})
    ok_, why, fix = S.runtime_status("llama.cpp", f, deep=False)
    assert not ok_ and "without CUDA" in why
    assert fix[-2:] == ["--cuda-arch", "120a-real"] and "native" not in fix     # explicit arch, never native
    t2 = _tree(tmp_path / "b", {"GGML_CUDA": "ON", "CMAKE_CUDA_ARCHITECTURES": "86-real;89-real"})
    monkeypatch.setitem(U.ENGINES, "llama.cpp", {"dir": str(t2), "url": ""})
    ok_, why, _ = S.runtime_status("llama.cpp", f, deep=False)
    assert not ok_ and "12.0" in why
    t3 = _tree(tmp_path / "c", {"GGML_CUDA": "ON", "CMAKE_CUDA_ARCHITECTURES": "120a-real"})
    monkeypatch.setitem(U.ENGINES, "llama.cpp", {"dir": str(t3), "url": ""})
    assert S.runtime_status("llama.cpp", f, deep=False)[0]
    t4 = _tree(tmp_path / "d", {"GGML_METAL": "OFF"})
    monkeypatch.setitem(U.ENGINES, "llama.cpp", {"dir": str(t4), "url": ""})
    ok_, why, fix = S.runtime_status("llama.cpp", mac_facts(), deep=False)
    assert not ok_ and "Metal" in why and "--cuda-arch" not in fix


def test_rubin_on_an_old_toolkit_gets_no_arch_guess():
    f = box_facts(caps=("10.7",), nvcc=(12, 8))
    assert S.cuda_arch_for(f) is None
    assert S.cuda_arch_for(box_facts(caps=("10.7",), nvcc=(13, 4))).startswith("107")


# ------------------------------------------------------------------------------------------ probes
def test_probe_parses_the_marker_and_survives_a_crash():
    def good(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout='noise\n@@POLLARD@@{"ok": true, "version": "2.0"}\n',
                                           stderr="")
    assert S.probe_module("gptqmodel", runner=good) == {"ok": True, "version": "2.0"}

    def crash(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 3221225477, stdout="", stderr="access violation")
    r = S.probe_module("exllamav3", runner=crash)
    assert not r["ok"] and "crashed" in r["why"]


def test_probe_skips_the_device_op_when_the_gpu_is_taken():
    seen = {}

    def run(cmd, **kw):
        seen["code"] = cmd[-1]
        return subprocess.CompletedProcess(cmd, 0, stdout="@@POLLARD@@{}", stderr="")
    S.probe_module("torch", runner=run, skip_op="Game.exe")
    assert "Game.exe" in seen["code"] and "torch.ones" in seen["code"]
    compile(seen["code"], "<probe>", "exec")                             # the generated probe is valid Python


def test_real_probe_of_a_stdlib_module():
    r = S.probe_module("json")
    assert r["ok"] is True


def test_parse_driver_cuda():
    assert S.parse_driver_cuda("| NVIDIA-SMI 580.1   Driver Version: 580.1   CUDA Version: 13.0  |") == (13, 0)
    assert S.parse_driver_cuda("| NVIDIA-SMI 610.88   KMD Version: 610.88   CUDA UMD Version: 13.3  |") == (13, 3)
    assert S.parse_driver_cuda("") is None


def test_printed_commands_paste_into_a_shell(monkeypatch):
    cmd = ["py", "-m", "pip", "install", "speculators>=0.8.1", "transformers<5.17"]
    monkeypatch.setattr(S, "WIN", True)
    assert S.fmt_cmd(cmd).endswith('"speculators>=0.8.1" "transformers<5.17"')
    monkeypatch.setattr(S, "WIN", False)
    assert S.fmt_cmd(cmd).endswith("'speculators>=0.8.1' 'transformers<5.17'")


def test_probe_forces_utf8_output():
    seen = {}

    def run(cmd, **kw):
        seen.update(kw["env"])
        return subprocess.CompletedProcess(cmd, 0, stdout="@@POLLARD@@{}", stderr="")
    S.probe_module("gptqmodel", runner=run)
    assert seen["PYTHONIOENCODING"] == "utf-8"


# ------------------------------------------------------------------------------------------ prompts
class Runner:
    def __init__(self):
        self.cmds = []

    def __call__(self, cmd, **kw):
        self.cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)


def missing_box(**kw):
    return box_facts(gptqmodel=gone(), xxhash=gone(), **kw)


def _setup(argv, facts, inp=None, tty=True):
    out, runner, asked = [], Runner(), []

    def ask(p):
        asked.append(p)
        return inp.pop(0) if inp else ""
    rc = S.run_setup(argv, out=out.append, detect_fn=lambda lanes, deep=True: facts, runner=runner, inp=ask, tty=tty)
    return rc, "\n".join(out), runner.cmds, asked


def test_prompt_yes_installs():
    rc, text, cmds, asked = _setup(["--lanes", "gptq,exlr8"], missing_box(), inp=[""])
    assert "Install missing pieces for lanes gptq,exlr8? [Y/n]" in asked[0]
    assert any("gptqmodel" in c for c in cmds) and any("xxhash" in c for c in cmds)


def test_prompt_no_installs_nothing():
    rc, text, cmds, _ = _setup(["--lanes", "gptq"], missing_box(), inp=["n"])
    assert rc == 1 and not cmds and "nothing installed" in text


def test_yes_flag_skips_the_prompt():
    rc, text, cmds, asked = _setup(["--lanes", "gptq", "--yes"], missing_box(), tty=False)
    assert not asked and cmds


def test_dry_run_prints_exact_commands_and_runs_nothing():
    rc, text, cmds, asked = _setup(["--lanes", "gptq,exlr8", "--dry-run"], missing_box())
    assert rc == 0 and not cmds and not asked
    assert "--no-build-isolation gptqmodel" in text and "dry run" in text


def test_non_tty_without_yes_does_not_install():
    rc, text, cmds, _ = _setup(["--lanes", "gptq"], missing_box(), tty=False)
    assert rc == 1 and not cmds and "--yes" in text


def test_system_step_waits_for_ok():
    f = box_facts(exllamav3=ok(ext="jit"), nvcc=None)
    rc, text, cmds, asked = _setup(["--lanes", "exl3"], f, inp=["", ""])
    assert len(asked) == 2 and "press Enter (OK)" in asked[1]
    assert not cmds                                     # nothing for setup itself to run: the toolkit is the user's


def test_outside_a_venv_pip_is_refused():
    rc, text, cmds, _ = _setup(["--lanes", "gptq", "--yes"], missing_box(venv=False), tty=False)
    assert not cmds and "not a virtual environment" in text and "-m venv" in text


def test_impossible_lane_named_says_why():
    rc, text, _, _ = _setup(["--lanes", "gptq", "--dry-run"], mac_facts())
    assert "gptq: needs an NVIDIA GPU" in text and "nothing to install here" in text


def test_all_ready():
    rc, text, cmds, asked = _setup([], mac_facts())
    assert rc == 0 and "every requested lane is ready" in text and not asked


# ------------------------------------------------------------------------------------------ hook + cache
def test_hint_is_cached_per_environment(monkeypatch):
    calls = []

    def det(lanes, deep=False):
        calls.append(tuple(lanes))
        return missing_box()
    lines = []
    S.hint(["gptq"], out=lines.append, detect_fn=det)
    S.hint(["gptq"], out=lines.append, detect_fn=det)
    assert len(calls) == 1, "the second run reads the cache"
    assert lines[0].startswith("   [setup] gptq lane needs gptqmodel") and "pollard setup --lanes gptq" in lines[0]
    monkeypatch.setattr(S, "fingerprint", lambda: "changed")
    S.hint(["gptq"], out=lines.append, detect_fn=det)
    assert len(calls) == 2, "an environment change re-checks"


def test_hint_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("POLLARD_NO_SETUP_HINT", "1")
    assert S.hint(["gptq"], detect_fn=lambda *a, **k: 1 / 0) == []


def test_hint_never_raises():
    assert S.hint(["gptq"], out=lambda m: None, detect_fn=lambda *a, **k: 1 / 0) == []


def test_require_stops_with_the_fix():
    out = []
    bad = S.require(["gptq", "exl3"], out=out.append, detect_fn=lambda lanes, deep=True: missing_box())
    assert bad == ["gptq"]
    assert "fix: pollard setup --lanes gptq" in "\n".join(out)


def test_a_shallow_cache_never_answers_a_deep_question():
    calls = []

    def det(lanes, deep=False):
        calls.append(deep)
        return box_facts()
    S.lane_status(["gptq"], deep=False, detect_fn=det)
    S.lane_status(["gptq"], deep=True, detect_fn=det)
    S.lane_status(["gptq"], deep=False, detect_fn=det)
    assert calls == [False, True]


# ------------------------------------------------------------------------------------------ wiring
def test_release_reads_the_one_requirements_table():
    import pollard_release as R
    assert R.REQUIRES == {"gguf": "any", "mlx": "apple", "gptq": "cuda", "mx": "cuda", "exl3": "cuda",
                          "exlr8": "cuda", "draft": "any"}
    assert set(R.LANES) <= set(S.LANES)


def test_release_run_stops_before_the_first_step(monkeypatch, tmp_path):
    import pollard_release as R
    monkeypatch.setattr(R, "machine", lambda: {"any": True, "apple": False, "cuda": True})
    monkeypatch.setattr(S, "require", lambda lanes, **kw: ["gptq"] if "gptq" in lanes else [])
    monkeypatch.setattr(R, "execute", lambda *a, **k: pytest.fail("must not start"))
    src = tmp_path / "Tiny"
    src.mkdir()
    (src / "config.json").write_text("{}")
    assert R.main(["--hf", str(src), "--lanes", "gptq", "--run"]) == 2


def test_pollard_setup_subcommand(monkeypatch):
    import pollard_auto as A
    got = {}
    monkeypatch.setattr(S, "main", lambda argv: got.setdefault("argv", argv) and 0)
    monkeypatch.setattr(sys, "argv", ["pollard", "setup", "--dry-run"])
    with pytest.raises(SystemExit):
        A.main()
    assert got["argv"] == ["--dry-run"]


def test_doctor_env_is_the_same_checker(monkeypatch):
    import pollard_doctor as Doc
    got = {}
    monkeypatch.setattr(S, "run_setup", lambda argv: got.setdefault("argv", argv) and 0)
    monkeypatch.setattr(sys, "argv", ["pollard-doctor", "--env", "--lanes", "gptq"])
    with pytest.raises(SystemExit):
        Doc.main()
    assert got["argv"] == ["--lanes", "gptq"]
