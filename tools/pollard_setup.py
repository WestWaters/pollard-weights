#!/usr/bin/env python3
"""pollard setup -- is everything each lane needs on this machine, and working? If not, install it.

    pollard setup                         # check every lane, show what is missing, offer to install it
    pollard setup --lanes gguf,gptq       # just these lanes
    pollard setup --dry-run               # the report + the exact commands it would run, nothing changed
    pollard setup --yes                   # install without asking (CI, a remote box)
    pollard-doctor --env                  # the same report, from the doctor

What it checks, per lane (the REQUIREMENTS table below):
  gguf        the llama.cpp runtime, built for THIS machine (Metal on a Mac, CUDA for the detected GPU arch)
  ik          ik_llama.cpp -- only for the IQ*_KT flagship rungs (optional)
  mlx         Apple Silicon + mlx + mlx-lm
  mlx-vlm     + mlx-vlm (git master: Qwen3-Omni, Nemotron omni) + torchvision -- the MLX lane of a model with
              vision / audio towers, a projector or a talker (`pollard release` asks for it on such a model)
  gptq        torch with a working CUDA device + transformers + gguf-py (the streamed engine) + gptqmodel
  mx          torch (CUDA) + gguf-py (the streamed engine) + llm-compressor + compressed-tensors
  exl3        torch (CUDA) + exllamav3 + its CUDA extension (prebuilt, or a toolchain to JIT it: an nvcc whose
              major matches torch's CUDA, and MSVC on Windows)
  exlr8       torch (CUDA) + transformers + gguf-py + xxhash
  draft       huggingface_hub + the llama.cpp runtime (find / convert / measure a drafter)
  draft-train speculators in its OWN venv (it pins transformers<5.17); recorded in Pollard's config so
              `pollard-draft train` uses it (optional)
  toolcall    at least one serving backend: llama-server, mlx_lm, vLLM with compiled kernels, transformers
  convert     torch + transformers + safetensors + datasets (probe / smooth / sensitivity / verify)

"Installed" is not "working": every Python package is imported in a fresh interpreter, torch runs a tiny op
on the device it would use (skipped when someone else's process holds the GPU), vLLM is checked for its
compiled kernels (a Windows `pip install vllm` imports and has none), the runtime's binaries are started.

Installs go into the ACTIVE venv only (pip, with the PyTorch index that matches the GPU: cu126 / cu128 / cu130
from the card's arch and the driver, MPS wheels on a Mac). Pins that conflict get their own venv under
$POLLARD_HOME/envs/. System pieces -- CUDA toolkit, MSVC, Xcode command line tools, git -- are never installed
for you: the one-line command or link is printed and setup waits for you to say OK. Nothing runs under sudo.
Nix is optional (a packaging route for the core CLI); this covers pip / uv installs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

WIN = sys.platform == "win32"

# ------------------------------------------------------------------------------------------ requirements
#: what each lane needs. `machine`: any | apple | cuda (also what `pollard release` checks before it plans a
#: lane on this box). `needs`: requirement ids, every one must be OK. `optional`: lanes `pollard setup` reports
#: but only installs when named in --lanes.
LANES = {
    "gguf":        {"machine": "any",   "needs": ["py:numpy", "py:gguf", "runtime:llama.cpp"]},
    "ik":          {"machine": "any",   "needs": ["runtime:ik_llama.cpp"], "optional": True},
    "mlx":         {"machine": "apple", "needs": ["py:mlx", "py:mlx_lm"]},
    # a VLM / omni ships whole through mlx-vlm; its processors (Qwen2-VL video) import torchvision
    "mlx-vlm":     {"machine": "apple", "needs": ["py:mlx", "py:mlx_lm", "py:mlx_vlm", "py:torchvision"],
                    "optional": True},
    # gptq / mx / exlr8 stream through pollard_stream (torch + transformers; gguf-py dequantizes a ggml-packed
    # source on read); gptqmodel / llm-compressor are the engines `--engine auto` keeps for a plain model that fits
    "gptq":        {"machine": "cuda",  "needs": ["torch", "py:transformers", "py:safetensors", "py:datasets",
                                                  "py:gguf", "py:gptqmodel"]},
    "mx":          {"machine": "cuda",  "needs": ["torch", "py:transformers", "py:datasets", "py:gguf",
                                                  "py:compressed_tensors",
                                                  "py:llmcompressor"]},
    "exl3":        {"machine": "cuda",  "needs": ["torch", "py:transformers", "py:safetensors", "py:datasets",
                                                  "py:exllamav3", "exl3-ext"]},
    "exlr8":       {"machine": "cuda",  "needs": ["torch", "py:transformers", "py:safetensors", "py:datasets",
                                                  "py:gguf", "py:xxhash"]},
    "draft":       {"machine": "any",   "needs": ["py:huggingface_hub", "runtime:llama.cpp"]},
    "draft-train": {"machine": "cuda",  "needs": ["env:speculators"], "optional": True},
    "toolcall":    {"machine": "any",   "needs": ["backend:any"]},
    "convert":     {"machine": "any",   "needs": ["torch", "py:transformers", "py:safetensors", "py:datasets"]},
}
#: the lanes `pollard release` builds, and the machine each needs (pollard_release.REQUIRES is this)
RELEASE_LANES = ("gguf", "mlx", "gptq", "mx", "exl3", "exlr8", "draft")
MACHINE = {lane: LANES[lane]["machine"] for lane in LANES}

#: import name -> (pip spec, distribution name for the version, extra pip flags)
PY = {
    "numpy": ("numpy", "numpy", []),
    "gguf": ("gguf", "gguf", []),
    "transformers": ("transformers", "transformers", []),
    "safetensors": ("safetensors", "safetensors", []),
    "datasets": ("datasets", "datasets", []),
    "xxhash": ("xxhash", "xxhash", []),
    "huggingface_hub": ("huggingface-hub", "huggingface-hub", []),
    "mlx": ("mlx", "mlx", []),
    "mlx_lm": ("mlx-lm", "mlx-lm", []),
    # Qwen3-Omni (qwen3_omni_moe) and Nemotron-3.5-Bolt (nemotron_h_omni) are on master, not on PyPI's 0.7.8
    "mlx_vlm": ("mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm", "mlx-vlm", []),
    "torchvision": ("torchvision", "torchvision", []),
    "compressed_tensors": ("compressed-tensors", "compressed-tensors", []),
    "llmcompressor": ("llmcompressor", "llmcompressor", []),
    # these two compile against the torch already in the env: build isolation would fetch a CPU torch instead
    "gptqmodel": ("gptqmodel", "gptqmodel", ["--no-build-isolation"]),
    "exllamav3": ("exllamav3", "exllamav3", ["--no-build-isolation"]),
}
#: modules whose import pulls torch (and a CUDA context): probed one at a time, never in parallel
TORCH_MODULES = {"gptqmodel", "exllamav3", "llmcompressor", "compressed_tensors", "transformers"}
SPECULATORS = {"pip": ["speculators>=0.8.1", "transformers<5.17"], "env": "speculators"}

#: PyTorch CUDA wheel indexes, newest first: (minimum driver CUDA, tag)
TORCH_INDEX = "https://download.pytorch.org/whl/"
TORCH_CUDA = [((13, 0), "cu130"), ((12, 8), "cu128"), ((12, 6), "cu126")]


def torch_cuda_tag(caps, driver_cuda):
    """The PyTorch CUDA wheel for these GPUs and this driver -> ('cu128', note) or (None, why not).

    Blackwell (sm_100 / sm_120) has kernels from cu128 on; Rubin (sm_107) from cu130. CUDA 13 dropped every
    arch below sm_75, so a Volta/Pascal card stays on cu126/cu128. The driver caps the wheel: a cu130 wheel
    does not start on a 12.x driver."""
    nums = [int(c.split(".")[0]) * 10 + int(c.split(".")[1]) for c in caps] or [0]
    lo, hi = min(nums), max(nums)
    need = (13, 0) if 107 in nums else (12, 8) if hi >= 100 else (12, 6)
    best = None
    for ver, tag in TORCH_CUDA:
        if ver < need:
            continue
        if ver >= (13, 0) and lo and lo < 75:
            continue
        if lo and lo < 70 and ver > (12, 6):
            continue                                             # Maxwell/Pascal: cu126 is the last with them
        if driver_cuda and ver > driver_cuda:
            continue
        best = tag
        break
    if best:
        return best, ""
    if driver_cuda and driver_cuda < need:
        return None, (f"the NVIDIA driver supports CUDA {driver_cuda[0]}.{driver_cuda[1]}; this GPU needs a "
                      f"CUDA {need[0]}.{need[1]} build of torch -- update the driver (https://www.nvidia.com/drivers)")
    return "cu128", "driver CUDA version unknown -- assuming cu128"


def torch_index_url(facts):
    """The pip index for torch on this machine (None = the default PyPI wheel, which is right on a Mac)."""
    gpu = facts.get("gpu") or {}
    if gpu.get("caps") or gpu.get("present"):
        tag, _ = torch_cuda_tag(gpu.get("caps") or [], gpu.get("driver_cuda"))
        return TORCH_INDEX + tag if tag else None
    if facts.get("os") == "darwin":
        return None
    if facts.get("os", "").startswith("linux"):
        return TORCH_INDEX + "cpu"                               # PyPI's Linux torch drags in 3GB of CUDA libs
    return None


# ------------------------------------------------------------------------------------------ small helpers
def pollard_home():
    return os.path.abspath(os.environ.get("POLLARD_HOME", os.path.expanduser("~/pollard")))


def config_path():
    return os.path.join(pollard_home(), "config.json")


def load_config():
    try:
        return json.load(open(config_path(), encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    try:
        os.makedirs(os.path.dirname(config_path()), exist_ok=True)
        json.dump(cfg, open(config_path(), "w", encoding="utf-8"), indent=1)
        return True
    except OSError:
        return False


def env_python(name):
    """The interpreter of a separate env `pollard setup` made (e.g. 'speculators'), or None."""
    py = ((load_config().get("envs") or {}).get(name) or {}).get("python")
    return py if py and os.path.exists(py) else None


def _venv_python(env_dir):
    return os.path.join(env_dir, "Scripts", "python.exe") if WIN else os.path.join(env_dir, "bin", "python")


def in_venv():
    return (getattr(sys, "base_prefix", sys.prefix) != sys.prefix or bool(os.environ.get("VIRTUAL_ENV"))
            or bool(os.environ.get("CONDA_PREFIX")))


def _probe(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def fmt_cmd(cmd):
    if isinstance(cmd, str):
        return cmd
    if WIN:   # list2cmdline leaves < > | & bare, and cmd.exe would read `transformers<5.17` as a redirect
        return " ".join(f'"{c}"' if re.search(r'[<>|&^]', c) and " " not in c else subprocess.list2cmdline([c])
                        for c in cmd)
    return shlex.join(cmd)


# ------------------------------------------------------------------------------------------ detection
def parse_driver_cuda(text):
    # "CUDA Version: 12.8" (drivers up to 58x); "CUDA UMD Version: 13.3" (61x on)
    m = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def detect_gpu(probe=_probe):
    """NVIDIA GPUs from nvidia-smi: names, compute caps, VRAM, the driver's CUDA version. None = no NVIDIA."""
    if not shutil.which("nvidia-smi") and probe is _probe:
        return None
    q = probe(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,memory.free",
               "--format=csv,noheader,nounits"])
    names, caps, total, free = [], [], 0.0, 0.0
    for line in (q or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        names.append(parts[0])
        if re.match(r"^\d+\.\d+$", parts[1]) and parts[1] not in caps:
            caps.append(parts[1])
        try:
            total += float(parts[2]) / 1024
            free = max(free, float(parts[3]) / 1024)
        except ValueError:
            pass
    if not names:
        return {"present": bool(shutil.which("nvidia-smi")), "names": [], "caps": [], "vram_gb": None,
                "free_gb": None, "driver_cuda": None, "users": []}
    try:
        from pollard_toolcall import gpu_users
        users = gpu_users()
    except Exception:                                            # noqa: BLE001
        users = []
    return {"present": True, "names": names, "caps": caps, "vram_gb": round(total, 1), "free_gb": round(free, 1),
            "driver_cuda": parse_driver_cuda(probe(["nvidia-smi"])), "users": users}


def detect_toolchain(probe=_probe):
    import pollard_runtime_update as U
    nvcc = U.parse_nvcc_version(probe([U._nvcc_cmd(), "--version"]))
    cmake = U.parse_cmake_version(probe([shutil.which("cmake") or "cmake", "--version"]))
    out = {"nvcc": nvcc, "cmake": cmake, "git": bool(shutil.which("git")), "msvc": None, "xcode_clt": None,
           "cc": bool(shutil.which("cc") or shutil.which("gcc") or shutil.which("clang"))}
    if WIN:
        out["msvc"] = U._vcvars()
        out["cc"] = bool(out["msvc"] or shutil.which("cl"))
    if sys.platform == "darwin":
        out["xcode_clt"] = bool(probe(["xcode-select", "-p"]).strip())
        out["cc"] = out["cc"] and out["xcode_clt"]
    return out


def _cmake_cache(tree):
    p = os.path.join(tree, "build", "CMakeCache.txt")
    out = {}
    try:
        for line in open(p, encoding="utf-8", errors="replace"):
            m = re.match(r"^([A-Z0-9_]+):[A-Z]+=(.*)$", line.strip())
            if m:
                out[m.group(1)] = m.group(2)
    except OSError:
        return None
    return out


def _arch_covers(entries, cap):
    """Does a CMAKE_CUDA_ARCHITECTURES list give a GPU of capability `cap` ('12.0') code it can run?"""
    n = int(cap.split(".")[0]) * 10 + int(cap.split(".")[1])
    for e in entries:
        m = re.match(r"^(\d+)", e.strip())
        if not m:
            continue
        k = int(m.group(1))
        if k == n or (e.strip().endswith("-virtual") and k <= n):
            return True
    return False


def runtime_status(name, facts, deep=True):
    """(ok, detail, fix): is engine `name` built and built for THIS machine's accelerator?"""
    import pollard_runtime_update as U
    tree = U.ENGINES.get(name, {}).get("dir") or os.path.join(REPO, "runtime", name)
    d = U.bin_dir(tree) if os.path.isdir(tree) else None
    gpu = facts.get("gpu") or {}
    fix = runtime_fix_cmd(name, facts)
    if not d:
        if name == "ik_llama.cpp":
            try:
                from pollard_auto import find_ik_build
                ext = find_ik_build()
            except Exception:                                    # noqa: BLE001
                ext = None
            if ext:
                return True, f"external ik_llama build at {ext}", ""
        else:
            from pollard_calc import find_llama_bin
            ext = find_llama_bin("llama-quantize")
            if ext:
                return True, f"external build at {os.path.dirname(ext)} (backend not checked; the managed " \
                             f"runtime is built by `pollard setup`)", ""
        return False, f"not built (no binaries under {tree})", fix
    cache = _cmake_cache(tree) or {}
    if facts.get("apple") and cache.get("GGML_METAL", "ON").upper() in ("OFF", "0", "FALSE"):
        return False, "built WITHOUT Metal (GGML_METAL=OFF) -- CPU only on this Mac", fix
    if gpu.get("caps"):
        if cache and cache.get("GGML_CUDA", "OFF").upper() not in ("ON", "1", "TRUE"):
            return False, f"built without CUDA -- the {', '.join(gpu.get('names') or ['GPU'])} is unused", fix
        archs = cache.get("CMAKE_CUDA_ARCHITECTURES", "")
        if archs.lower() == "native" or not archs:
            bad = U.configured_cuda_arch_problem(tree)
            if bad:
                return False, bad, fix
        else:
            entries = [e for e in archs.split(";") if e.strip()]
            missing = [c for c in gpu["caps"] if not _arch_covers(entries, c)]
            if missing:
                return False, (f"CUDA arch list '{archs}' has no code for this GPU (compute capability "
                               f"{', '.join(missing)})"), fix
    backend = "Metal" if facts.get("apple") else ("CUDA " + cache.get("CMAKE_CUDA_ARCHITECTURES", "")
                                                  if gpu.get("caps") else "CPU")
    if deep:
        ok, msg = U.smoke(tree)
        if not ok:
            return False, f"binaries do not start: {msg}", fix
    return True, f"{backend.strip()} build at {d}", ""


def runtime_fix_cmd(name, facts):
    """The command that (re)builds engine `name` for this machine. An explicit arch from the detected GPU,
    never `native`: a hidden GPU at configure time would build kernels for no card at all."""
    cmd = [sys.executable, os.path.join(TOOLS, "pollard_runtime.py"), "--update", name]
    arch = cuda_arch_for(facts)
    if arch:
        cmd += ["--cuda-arch", arch]
    return cmd


def cuda_arch_for(facts):
    caps = (facts.get("gpu") or {}).get("caps") or []
    if not caps:
        return None
    import pollard_runtime_update as U
    tc = facts.get("toolchain") or {}
    want = []
    try:
        for cap in caps:
            want += [a for a in U.archs_for_cap(cap, tc.get("nvcc"), tc.get("cmake")) if a not in want]
    except U.CudaArchError:
        return None
    return ";".join(want) or None


_PROBE_TMPL = r"""
import json, sys, time
sys.path.insert(0, {tools!r})
res = {{"ok": False}}
try:
{body}
    res["ok"] = True
except BaseException as e:
    res["why"] = (type(e).__name__ + ": " + str(e)).strip()[:400]
try:
    import importlib.metadata as md
    res["version"] = md.version({dist!r})
except Exception:
    pass
print("@@POLLARD@@" + json.dumps(res, default=str))
"""

_TORCH_BODY = """
    import torch, pollard_device as D
    info = D.describe(torch)
    res["detail"] = info
    dev = info["device"]
    if dev != "cpu" and {skip_op!r}:
        res["op"] = "skipped: the GPU is in use by " + {skip_op!r}
    else:
        x = torch.ones(8, device=dev) * 2
        assert float(x.sum().item()) == 16.0
        res["op"] = "ok on " + dev
"""

_BODIES = {
    "mlx": "    import mlx.core as mx\n    a = mx.array([1.0, 2.0]) + 1\n    mx.eval(a)\n"
           "    assert float(a.sum().item()) == 5.0\n    res['op'] = 'ok on ' + str(mx.default_device())\n",
    "exllamav3": "    import torch, exllamav3, importlib.util as u\n"
                 "    res['ext'] = 'prebuilt' if u.find_spec('exllamav3_ext') else 'jit'\n"
                 "    if res['ext'] == 'prebuilt':\n        import exllamav3_ext\n",
    "speculators": "    import speculators, transformers\n    res['transformers'] = transformers.__version__\n",
    # PyPI's mlx-vlm predates the omni models the MLX lane ships whole: master has them
    "mlx_vlm": "    import mlx_vlm\n    from mlx_vlm.models import qwen3_omni_moe, nemotron_h_nano_omni\n"
               "    res['op'] = 'qwen3_omni_moe + nemotron_h_nano_omni'\n",
}


def probe_module(mod, python=None, skip_op="", runner=subprocess.run, timeout=600):
    """Import `mod` (and do its tiny op) in a fresh interpreter -> {"ok", "why", "version", ...}.
    A fresh process per module: a native crash (a bad CUDA DLL) takes down the probe, not setup."""
    if mod == "torch":
        body = _TORCH_BODY.format(skip_op=skip_op)
        dist = "torch"
    else:
        body = _BODIES.get(mod, f"    import {mod}\n")
        dist = PY.get(mod, (mod, mod, []))[1]
    code = _PROBE_TMPL.format(tools=TOOLS, body=body.rstrip("\n"), dist=dist)
    # UTF-8 out: a package that prints a banner with emoji at import (gptqmodel does) dies on a Windows cp1252
    # pipe with UnicodeEncodeError -- that is the probe's pipe, not the package, and must not read as broken
    env = dict(os.environ, PYTHONWARNINGS="ignore", TF_CPP_MIN_LOG_LEVEL="3", PYTHONIOENCODING="utf-8",
               PYTHONUTF8="1")
    try:
        r = runner([python or sys.executable, "-c", code], capture_output=True, text=True, errors="replace",
                   timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "why": f"import did not finish in {timeout}s"}
    except OSError as e:
        return {"ok": False, "why": str(e)}
    for line in (r.stdout or "").splitlines():
        if line.startswith("@@POLLARD@@"):
            try:
                return json.loads(line[len("@@POLLARD@@"):])
            except ValueError:
                break
    tail = ((r.stderr or "") + (r.stdout or "")).strip()[-300:]
    return {"ok": False, "why": f"interpreter crashed (exit {r.returncode}): {tail}"}


def shallow_module(mod):
    """Presence only (no import): find_spec + the installed version."""
    import importlib.util
    try:
        present = importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        present = False
    res = {"ok": present, "shallow": True}
    if not present:
        res["why"] = "not installed"
    try:
        import importlib.metadata as md
        res["version"] = md.version(PY.get(mod, (mod, mod, []))[1] if mod != "torch" else "torch")
    except Exception:                                            # noqa: BLE001
        pass
    return res


def needed_modules(lanes):
    mods = []
    for lane in lanes:
        for req in LANES[lane]["needs"]:
            m = req[3:] if req.startswith("py:") else ("torch" if req == "torch" else
                                                       "exllamav3" if req == "exl3-ext" else None)
            if m and m not in mods:
                mods.append(m)
    if "toolcall" in lanes:
        for m in ("torch", "transformers") + (("mlx_lm",) if _is_apple() else ()):
            if m not in mods:
                mods.append(m)
    return mods


def _is_apple():
    return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")


def machine_kinds(facts):
    return {"any": True, "apple": bool(facts.get("apple")), "cuda": bool((facts.get("gpu") or {}).get("present"))}


def detect(lanes=None, deep=True, probe=_probe, module_probe=None, out=None):
    """Everything setup needs to know about this machine, for `lanes` (default: all)."""
    lanes = list(lanes or LANES)
    facts = {"os": sys.platform, "arch": platform.machine(), "apple": _is_apple(), "python": sys.executable,
             "pyver": platform.python_version(), "venv": in_venv(), "deep": deep}
    facts["gpu"] = detect_gpu(probe)
    facts["toolchain"] = detect_toolchain(probe)
    try:
        from pollard_calc import detect_available_ram_gb
        facts["ram_free_gb"] = round(detect_available_ram_gb() or 0, 1)
    except Exception:                                            # noqa: BLE001
        facts["ram_free_gb"] = None
    home = pollard_home()
    base = home
    while base and not os.path.exists(base):
        nxt = os.path.dirname(base)
        base = None if nxt == base else nxt
    try:
        facts["disk_free_gb"] = round(shutil.disk_usage(base or os.getcwd()).free / 1e9, 1)
    except OSError:
        facts["disk_free_gb"] = None
    lanes = [lane for lane in lanes if not impossible_why(lane, facts)]   # no gptqmodel import on a Mac
    users = (facts["gpu"] or {}).get("users") or []
    skip = ", ".join(users[:3]) if users else ""
    mods = {}
    for m in needed_modules(lanes):
        if not deep:
            mods[m] = shallow_module(m)
            continue
        if shallow_module(m)["ok"] is False:
            mods[m] = {"ok": False, "why": "not installed"}
            continue
        if out:
            out(f"   checking {m} ...")
        mods[m] = (module_probe or probe_module)(m, skip_op=skip)
    facts["modules"] = mods
    facts["runtime"] = {}
    for lane in lanes:
        for req in LANES[lane]["needs"]:
            if req.startswith("runtime:") and req[8:] not in facts["runtime"]:
                ok, detail, fix = runtime_status(req[8:], facts, deep)
                facts["runtime"][req[8:]] = {"ok": ok, "detail": detail, "fix": fix}
    if "toolcall" in lanes and "llama.cpp" not in facts["runtime"]:
        ok, detail, fix = runtime_status("llama.cpp", facts, deep)
        facts["runtime"]["llama.cpp"] = {"ok": ok, "detail": detail, "fix": fix}
    if "toolcall" in lanes:
        facts["vllm"] = _vllm(deep)
    if "draft-train" in lanes:
        py = env_python(SPECULATORS["env"])
        if py and deep:
            r = (module_probe or probe_module)("speculators", python=py)
            facts["speculators"] = {"python": py, **r}
        else:
            facts["speculators"] = {"python": py, "ok": bool(py), "why": "" if py else "no speculators env"}
    return facts


def _vllm(deep):
    if not deep:
        return {"ok": shallow_module("vllm")["ok"], "why": "presence only"}
    try:
        from pollard_toolcall import vllm_status
        ok, why = vllm_status()
        return {"ok": ok, "why": why}
    except Exception as e:                                       # noqa: BLE001
        return {"ok": False, "why": str(e)}


# ------------------------------------------------------------------------------------------ evaluation
def torch_problem(facts):
    """Why the installed torch is the WRONG torch for this machine, or ''."""
    t = (facts.get("modules") or {}).get("torch") or {}
    if not t.get("ok"):
        return ""
    d = t.get("detail") or {}
    if t.get("shallow"):
        return ""
    gpu = facts.get("gpu") or {}
    if gpu.get("caps"):
        cuda = d.get("cuda") or [False, ""]
        if not d.get("torch_cuda"):
            return f"torch {d.get('torch')} is a CPU-only build -- this machine has an NVIDIA GPU"
        if not cuda[0] and "no kernels" in str(cuda[1]):
            return cuda[1]
    if facts.get("apple"):
        mps = d.get("mps") or [False, ""]
        if not mps[0]:
            return f"torch {d.get('torch')} cannot use the Apple GPU: {mps[1]}"
    return ""


def requirement(req, facts):
    """(state, detail) for one requirement id. state: ok | missing | broken | wrong."""
    mods = facts.get("modules") or {}
    tc = facts.get("toolchain") or {}
    if req.startswith("py:"):
        r = mods.get(req[3:]) or {"ok": False, "why": "not checked"}
        if r.get("ok"):
            return "ok", r.get("version") or "installed"
        why = r.get("why") or "not installed"
        return ("missing" if why == "not installed" else "broken"), why
    if req == "torch":
        r = mods.get("torch") or {"ok": False, "why": "not checked"}
        if not r.get("ok"):
            why = r.get("why") or "not installed"
            return ("missing" if why == "not installed" else "broken"), why
        bad = torch_problem(facts)
        if bad:
            return "wrong", bad
        d = r.get("detail") or {}
        if not r.get("shallow") and (facts.get("gpu") or {}).get("caps") and not (d.get("cuda") or [False])[0]:
            return "broken", f"no usable CUDA device: {(d.get('cuda') or [0, '?'])[1]}"
        dev = d.get("device") or "?"
        op = str(r.get("op") or "")
        op = "device op skipped: GPU busy" if op.startswith("skipped") else op
        return "ok", f"{r.get('version') or d.get('torch')} -> {dev}" + (f" ({op})" if op else "")
    if req.startswith("runtime:"):
        r = (facts.get("runtime") or {}).get(req[8:]) or {"ok": False, "detail": "not checked"}
        return ("ok" if r["ok"] else "missing"), r["detail"]
    if req == "exl3-ext":
        r = mods.get("exllamav3") or {}
        if not r.get("ok"):
            return "missing", "exllamav3 not working"
        if r.get("ext") == "prebuilt" or r.get("shallow"):
            return "ok", "prebuilt extension" if r.get("ext") == "prebuilt" else "installed (extension not checked)"
        # no prebuilt extension: the first use JIT-compiles it, which needs nvcc (major == torch's CUDA) + MSVC
        tcuda = ((mods.get("torch") or {}).get("detail") or {}).get("torch_cuda")
        nvcc = tc.get("nvcc")
        if not nvcc:
            return "missing", "no prebuilt exllamav3_ext and no CUDA toolkit (nvcc) to JIT-build it"
        if tcuda and str(nvcc[0]) != str(tcuda).split(".")[0]:
            return "wrong", (f"nvcc {nvcc[0]}.{nvcc[1]} does not match torch's CUDA {tcuda} -- the extension "
                             "build refuses a major mismatch")
        if WIN and not tc.get("msvc"):
            return "missing", "MSVC Build Tools are needed to JIT-build exllamav3_ext on Windows"
        return "ok", f"JIT-builds on first use (nvcc {nvcc[0]}.{nvcc[1]})"
    if req == "env:speculators":
        r = facts.get("speculators") or {}
        if r.get("ok"):
            return "ok", f"{r.get('python')} (transformers {r.get('transformers', '?')})"
        return "missing", r.get("why") or "no speculators env"
    if req == "backend:any":
        have = []
        rt = (facts.get("runtime") or {}).get("llama.cpp") or {}
        if rt.get("ok"):
            have.append("llama-server")
        if (mods.get("mlx_lm") or {}).get("ok"):
            have.append("mlx_lm")
        if (facts.get("vllm") or {}).get("ok"):
            have.append("vllm")
        if (mods.get("transformers") or {}).get("ok") and (mods.get("torch") or {}).get("ok"):
            have.append("transformers")
        if have:
            v = facts.get("vllm") or {}
            extra = "" if "vllm" in have or not v.get("why") else f"; vllm: {v['why']}"
            return "ok", ", ".join(have) + extra
        return "missing", "no serving backend (llama-server, mlx_lm, vllm or transformers)"
    return "missing", f"unknown requirement {req}"


def impossible_why(lane, facts):
    need = LANES[lane]["machine"]
    if need == "apple" and not facts.get("apple"):
        return "needs Apple Silicon (MLX runs on Metal only) -- build it on an Apple Silicon Mac: pollard release --lanes " + lane
    if need == "cuda" and not (facts.get("gpu") or {}).get("present"):
        where = "a Mac has no CUDA" if facts.get("os") == "darwin" else "no NVIDIA GPU / driver found (nvidia-smi)"
        how = f"pollard release --lanes {lane}" if lane in RELEASE_LANES else f"pollard setup --lanes {lane}"
        return f"needs an NVIDIA GPU ({where}) -- run it on a CUDA machine: {how}"
    return ""


def evaluate(facts, lanes=None):
    """[(lane, status, [(req, state, detail)], why)] -- status: ready | missing | impossible."""
    rows = []
    for lane in (lanes or LANES):
        why = impossible_why(lane, facts)
        if why:
            rows.append((lane, "impossible", [], why))
            continue
        reqs = [(r, *requirement(r, facts)) for r in LANES[lane]["needs"]]
        bad = [x for x in reqs if x[1] != "ok"]
        rows.append((lane, "ready" if not bad else "missing", reqs,
                     "; ".join(f"{label(r)}: {d}" for r, _, d in bad)))
    return rows


# ------------------------------------------------------------------------------------------ install plan
class Action(dict):
    """kind: pip | venv | runtime | system | note. cmd: argv list (or a string for system lines)."""
    def __getattr__(self, k):
        return self.get(k)


def system_fix(what, facts):
    """(command or None, link) for a machine-level piece setup never installs itself."""
    osn = facts.get("os", "")
    if what == "cuda":
        tcuda = (((facts.get("modules") or {}).get("torch") or {}).get("detail") or {}).get("torch_cuda")
        ver = tcuda or "12.8"
        link = f"https://developer.nvidia.com/cuda-{ver.replace('.', '-')}-0-download-archive"
        if osn == "win32":
            return "winget install --id Nvidia.CUDA -e  (pick the version matching torch's CUDA " + ver + ")", link
        return None, link
    if what == "msvc":
        return ('winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override '
                '"--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"',
                "https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    if what == "xcode":
        return "xcode-select --install", "https://developer.apple.com/xcode/resources/"
    if what == "git":
        if osn == "win32":
            return "winget install --id Git.Git -e", "https://git-scm.com/downloads"
        if osn == "darwin":
            return "xcode-select --install", "https://git-scm.com/downloads"
        return "sudo apt install git   (or your distro's package manager)", "https://git-scm.com/downloads"
    if what == "cc":
        return "sudo apt install build-essential   (or your distro's compiler package)", "https://gcc.gnu.org/"
    if what == "driver":
        return None, "https://www.nvidia.com/drivers"
    return None, ""


def plan_actions(facts, lanes, python=None):
    """The ordered fixes for `lanes` (only lanes this machine can run). System pieces first (they are the
    user's), then torch (everything else builds against it), then pip packages, separate envs, runtimes."""
    py = python or facts.get("python") or sys.executable
    rows = {r[0]: r for r in evaluate(facts, lanes)}
    todo = [lane for lane in lanes if rows[lane][1] == "missing"]
    acts, sysd = [], set()
    tc = facts.get("toolchain") or {}
    gpu = facts.get("gpu") or {}

    def sysfix(what, why, lane):
        if what in sysd:
            return
        sysd.add(what)
        cmd, link = system_fix(what, facts)
        acts.append(Action(kind="system", lane=lane, what=what, cmd=cmd, link=link, why=why))

    def bad(lane, req):
        return any(r == req and s != "ok" for r, s, _ in rows[lane][2])

    # --- system pieces
    runtime_lanes = [lane for lane in todo if any(bad(lane, r) for r in LANES[lane]["needs"]
                                                  if r.startswith("runtime:"))]
    if runtime_lanes:
        if not tc.get("git"):
            sysfix("git", "the runtime build clones llama.cpp", runtime_lanes[0])
        if facts.get("os") == "darwin" and not tc.get("xcode_clt"):
            sysfix("xcode", "a C/C++ compiler + Metal toolchain for the llama.cpp build", runtime_lanes[0])
        elif facts.get("os") == "win32" and not tc.get("msvc"):
            sysfix("msvc", "the C++ compiler the llama.cpp build uses on Windows", runtime_lanes[0])
        elif facts.get("os", "").startswith("linux") and not tc.get("cc"):
            sysfix("cc", "a C/C++ compiler for the llama.cpp build", runtime_lanes[0])
        if gpu.get("caps") and not tc.get("nvcc"):
            sysfix("cuda", "without the CUDA toolkit the runtime builds CPU-only and leaves the GPU idle",
                   runtime_lanes[0])
    if "exl3" in todo and bad("exl3", "exl3-ext"):
        state = next(s for r, s, _ in rows["exl3"][2] if r == "exl3-ext")
        if state in ("missing", "wrong") and (not tc.get("nvcc") or state == "wrong"):
            sysfix("cuda", "exllamav3_ext JIT-compiles with nvcc whose major matches torch's CUDA", "exl3")
        if facts.get("os") == "win32" and not tc.get("msvc"):
            sysfix("msvc", "exllamav3_ext JIT-compiles with MSVC on Windows", "exl3")
    if gpu.get("present") and any(bad(lane, "torch") for lane in todo):
        tag, why = torch_cuda_tag(gpu.get("caps") or [], gpu.get("driver_cuda"))
        if not tag:
            sysfix("driver", why, "torch")

    # --- torch, with the index that matches the accelerator
    torch_lanes = [lane for lane in todo if bad(lane, "torch")]
    if torch_lanes:
        state = next(s for r, s, _ in rows[torch_lanes[0]][2] if r == "torch")
        idx = torch_index_url(facts)
        if gpu.get("present") and idx is None:
            pass                                                 # driver too old: the system line above says so
        else:
            cmd = [py, "-m", "pip", "install", "--upgrade"]
            if state in ("wrong", "broken"):
                cmd += ["--force-reinstall"]
            cmd += ["torch"] + (["--index-url", idx] if idx else [])
            acts.append(Action(kind="pip", lane=",".join(torch_lanes), cmd=cmd,
                               why=("replace the wrong torch: " if state != "missing" else "torch for ")
                               + (idx.rsplit("/", 1)[1] if idx else "this Mac (MPS)" if facts.get("apple")
                                  else "this machine")))

    # --- pip packages for every lane still missing something
    plain, special = [], []
    for lane in todo:
        for r, s, _ in rows[lane][2]:
            mod = r[3:] if r.startswith("py:") else ("exllamav3" if r == "exl3-ext" and s != "ok"
                                                     and not ((facts.get("modules") or {}).get("exllamav3") or {})
                                                     .get("ok") else None)
            if not mod or s == "ok" or mod not in PY:
                continue
            spec, _, flags = PY[mod]
            bucket = special if flags else plain
            if spec not in [b[0] for b in bucket]:
                bucket.append((spec, flags, lane, s))
    if plain:
        force = any(s == "broken" for *_, s in plain)
        acts.append(Action(kind="pip", lane=",".join(sorted({b[2] for b in plain})),
                           cmd=[py, "-m", "pip", "install", "--upgrade"] + (["--force-reinstall", "--no-deps"]
                                                                            if force and len(plain) == 1 else [])
                           + [b[0] for b in plain],
                           why="python packages"))
    for spec, flags, lane, s in special:
        acts.append(Action(kind="pip", lane=lane, cmd=[py, "-m", "pip", "install", "--upgrade"] + flags
                           + (["--force-reinstall", "--no-deps"] if s == "broken" else []) + [spec],
                           why=f"{spec} builds against the torch in this env"))

    # --- the speculators env (its own venv: it pins transformers<5.17)
    if "draft-train" in todo:
        env_dir = os.path.join(pollard_home(), "envs", SPECULATORS["env"])
        epy = _venv_python(env_dir)
        idx = torch_index_url(facts)
        acts.append(Action(kind="venv", lane="draft-train", env=SPECULATORS["env"], dir=env_dir, python=epy,
                           cmds=[[py, "-m", "venv", env_dir],
                                 [epy, "-m", "pip", "install", "--upgrade", "pip"],
                                 [epy, "-m", "pip", "install", "torch"] + (["--index-url", idx] if idx else []),
                                 [epy, "-m", "pip", "install"] + SPECULATORS["pip"]],
                           why="speculators pins transformers<5.17 -- its own venv, recorded in "
                               f"{config_path()}"))

    # --- the runtimes, built for this machine's accelerator
    for name in ("llama.cpp", "ik_llama.cpp"):
        r = (facts.get("runtime") or {}).get(name)
        lanes_here = [lane for lane in todo if bad(lane, "runtime:" + name)]
        if r and not r["ok"] and lanes_here:
            acts.append(Action(kind="runtime", lane=",".join(lanes_here), engine=name, cmd=r["fix"],
                               why=f"{name}: {r['detail']}"))
    if "toolcall" in todo:
        acts.append(Action(kind="note", lane="toolcall",
                           why="no serving backend yet -- the gguf lane's llama-server (above) is the default one"))
    return acts


# ------------------------------------------------------------------------------------------ report
ICON = {"ready": "ready", "missing": "MISSING", "impossible": "n/a"}
LABEL = {"backend:any": "backends:", "exl3-ext": "exllamav3_ext", "env:speculators": "speculators env"}


def _short(text, n=90):
    text = str(text)
    return text if len(text) <= n else text[:n - 3] + "..."


def label(req):
    return LABEL.get(req) or req.split(":", 1)[-1]


def report(facts, rows, out=print):
    gpu = facts.get("gpu") or {}
    tc = facts.get("toolchain") or {}
    out(f"== pollard setup :: {platform.system()} {facts.get('arch')} | python {facts.get('pyver')} "
        f"({'venv' if facts.get('venv') else 'NOT a venv'}: {facts.get('python')})")
    if gpu.get("names"):
        dc = gpu.get("driver_cuda")
        out(f"   GPU: {', '.join(gpu['names'])} | compute {', '.join(gpu['caps'])} | {gpu.get('vram_gb')} GB VRAM "
            f"({gpu.get('free_gb')} GB free) | driver CUDA " + (f"{dc[0]}.{dc[1]}" if dc else "?"))
        if gpu.get("users"):
            out(f"        in use by: {', '.join(gpu['users'][:4])} (device ops skipped, nothing was loaded on it)")
    elif facts.get("apple"):
        out("   GPU: Apple Silicon (Metal / MPS / MLX), unified memory")
    else:
        out("   GPU: none detected (CPU only)")
    nv = tc.get("nvcc")
    out(f"   toolchain: nvcc {'%d.%d' % nv if nv else '-'} | cmake {'.'.join(map(str, tc['cmake'])) if tc.get('cmake') else '-'}"
        f" | git {'yes' if tc.get('git') else 'NO'} | compiler {'yes' if tc.get('cc') else 'NO'}"
        + (f" | MSVC {'yes' if tc.get('msvc') else 'NO'}" if facts.get("os") == "win32" else ""))
    out(f"   memory: {facts.get('ram_free_gb')} GB RAM free | disk free at {pollard_home()}: {facts.get('disk_free_gb')} GB")
    if facts.get("disk_free_gb") is not None and facts["disk_free_gb"] < 30:
        out("   !! under 30 GB free: a 7B ladder alone needs ~40 GB")
    out("")
    width = max(len(r[0]) for r in rows) if rows else 8
    for lane, status, reqs, why in rows:
        opt = " (optional)" if LANES[lane].get("optional") else ""
        out(f"   {lane:<{width}}  {ICON[status]:<8} " + (why if status != "ready" else
            "; ".join(f"{label(r)} {_short(d)}" for r, s, d in reqs if s == "ok")) + opt)
    if not facts.get("deep"):
        out("\n   (fast check: packages found, not imported -- `pollard setup` imports and runs each one)")


def show_actions(acts, out=print):
    for i, a in enumerate(acts, 1):
        if a.kind == "system":
            out(f"   {i}. [you] {a.what}: {a.why}")
            if a.cmd:
                out(f"        run: {a.cmd}")
            if a.link:
                out(f"        see: {a.link}")
        elif a.kind == "venv":
            out(f"   {i}. [{a.lane}] {a.why}")
            for c in a.cmds:
                out(f"        $ {fmt_cmd(c)}")
        elif a.kind == "note":
            out(f"   {i}. [{a.lane}] {a.why}")
        else:
            out(f"   {i}. [{a.lane}] {a.why}\n        $ {fmt_cmd(a.cmd)}")


# ------------------------------------------------------------------------------------------ cache + hooks
def fingerprint():
    """Changes when the environment does: the interpreter, any site-packages dir (pip touches it), the
    managed runtime trees, and whether an NVIDIA driver is present."""
    import site
    parts = [sys.executable, sys.version, sys.platform, str(bool(shutil.which("nvidia-smi")))]
    dirs = list(getattr(site, "getsitepackages", lambda: [])()) + [site.getusersitepackages()
                                                                    if hasattr(site, "getusersitepackages") else ""]
    for d in dirs + [os.path.join(REPO, "runtime"), os.path.join(REPO, "runtime", "llama.cpp", "build", "bin"),
                     os.path.join(REPO, "runtime", "ik_llama.cpp", "build", "bin"), config_path()]:
        try:
            parts.append(f"{d}:{os.stat(d).st_mtime_ns}")
        except (OSError, TypeError):
            parts.append(f"{d}:-")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def cache_path():
    return os.path.join(pollard_home(), "setup", "state.json")


def _load_cache():
    try:
        return json.load(open(cache_path(), encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(c):
    try:
        os.makedirs(os.path.dirname(cache_path()), exist_ok=True)
        json.dump(c, open(cache_path(), "w", encoding="utf-8"), indent=1, default=str)
    except OSError:
        pass                                                     # a read-only HOME never stops a build


def lane_status(lanes, deep=False, detect_fn=None):
    """{lane: (status, why)} for `lanes`, from the cache while the environment has not changed. A deep result
    in the cache answers a shallow question too; a shallow one never answers a deep one."""
    lanes = [lane for lane in lanes if lane in LANES]
    fp = fingerprint()
    cache = _load_cache()
    ent = cache.get(fp) or {}
    key = "deep" if deep else "any"
    have = dict(ent.get(key) or {})
    if not deep:
        have.update(ent.get("deep") or {})                       # a deep answer is the better shallow one
    need = [lane for lane in lanes if lane not in have]
    if need:
        facts = (detect_fn or detect)(need, deep=deep)
        got = {lane: [status, why] for lane, status, _, why in evaluate(facts, need)}
        have.update(got)
        ent[key] = {**(ent.get(key) or {}), **got}
        ent["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _save_cache({fp: ent})                                   # one environment's answers; older ones are stale
    return {lane: tuple(have[lane]) for lane in lanes}


def fix_command(lanes):
    return "pollard setup --lanes " + ",".join(lanes)


def hint(lanes, out=None, detect_fn=None):
    """The first-run hook: one line per requested lane that is not ready on this machine. Fast (no imports) and
    cached per environment; never raises. POLLARD_NO_SETUP_HINT=1 turns it off."""
    if os.environ.get("POLLARD_NO_SETUP_HINT") == "1":
        return []
    out = out or (lambda m: print(m, file=sys.stderr, flush=True))
    try:
        st = lane_status(lanes, deep=False, detect_fn=detect_fn)
    except Exception:                                            # noqa: BLE001
        return []
    lines = []
    for lane, (status, why) in st.items():
        if status == "missing":
            first = why.split("; ")[0]
            lines.append(f"   [setup] {lane} lane needs {first} -- run `{fix_command([lane])}`")
    for line in lines:
        out(line)
    return lines


def require(lanes, out=print, detect_fn=None):
    """For a pipeline about to run `lanes` on THIS machine: the deep check (imports + device op), cached. Returns
    the lanes that are not ready (and prints why + the one fix command); [] = go."""
    lanes = [lane for lane in lanes if lane in LANES]
    if not lanes or os.environ.get("POLLARD_SKIP_SETUP_CHECK") == "1":
        return []
    st = lane_status(lanes, deep=True, detect_fn=detect_fn)
    bad = [lane for lane in lanes if st[lane][0] == "missing"]
    if bad:
        out("   setup check: these lanes cannot run on this machine yet:")
        for lane in bad:
            out(f"     {lane}: {st[lane][1]}")
        out(f"   fix: {fix_command(bad)}   (or --skip-setup-check to try anyway)")
    return bad


# ------------------------------------------------------------------------------------------ apply
def _ask(prompt, inp=input):
    try:
        return inp(prompt).strip().lower()
    except EOFError:
        return "n"


def apply(acts, yes=False, out=print, runner=subprocess.run, inp=input, tty=None, runtime_fn=None):
    """Run the plan. System lines are shown and waited on (OK to continue / s to skip); pip, venv and runtime
    steps run. Returns the list of (action, ok)."""
    tty = sys.stdin.isatty() if tty is None else tty
    done = []
    for a in acts:
        if a.kind == "note":
            continue
        if a.kind == "system":
            out(f"\n   [you] {a.what}: {a.why}")
            if a.cmd:
                out(f"        run: {a.cmd}")
            if a.link:
                out(f"        see: {a.link}")
            if tty and not yes:
                ans = _ask("        run that yourself, then press Enter (OK) to continue, or s to skip: ", inp)
                done.append((a, ans not in ("s", "skip")))
            else:
                done.append((a, False))
            continue
        if a.kind == "venv":
            ok = True
            for c in a.cmds:
                out(f"   $ {fmt_cmd(c)}")
                if runner(c).returncode != 0:
                    ok = False
                    break
            if ok:
                cfg = load_config()
                cfg.setdefault("envs", {})[a.env] = {"python": a.python, "dir": a.dir,
                                                     "made": time.strftime("%Y-%m-%d")}
                save_config(cfg)
                out(f"   recorded the {a.env} env in {config_path()}")
            done.append((a, ok))
            continue
        if a.kind == "runtime":
            out(f"   $ {fmt_cmd(a.cmd)}")
            if runtime_fn:
                ok = runtime_fn(a)
            else:
                ok = runner(a.cmd).returncode == 0
            done.append((a, ok))
            continue
        out(f"   $ {fmt_cmd(a.cmd)}")
        done.append((a, runner(a.cmd).returncode == 0))
    return done


def default_lanes(facts):
    return [lane for lane in LANES if not LANES[lane].get("optional") and not impossible_why(lane, facts)]


def run_setup(argv=None, out=print, detect_fn=None, runner=subprocess.run, inp=input, tty=None):
    ap = argparse.ArgumentParser(prog="pollard setup", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--lanes", help="comma list (default: every lane this machine can run; optional lanes "
                                    f"-- {', '.join(k for k, v in LANES.items() if v.get('optional'))} -- only when named)")
    ap.add_argument("--yes", "-y", action="store_true", help="install without asking")
    ap.add_argument("--dry-run", action="store_true", help="print the report and the exact commands; change nothing")
    ap.add_argument("--fast", action="store_true", help="presence only (no imports / device ops)")
    ap.add_argument("--json", action="store_true", help="print the detected facts as JSON")
    ap.add_argument("--allow-system-python", action="store_true",
                    help="pip-install even though this interpreter is not a venv")
    a = ap.parse_args(argv)
    lanes = None
    if a.lanes:
        lanes = [x.strip() for x in a.lanes.split(",") if x.strip()]
        bad = [x for x in lanes if x not in LANES]
        if bad:
            ap.error(f"unknown lane(s) {bad}; choose from {list(LANES)}")
    facts = (detect_fn or detect)(list(LANES), deep=not a.fast,
                                  **({} if detect_fn else {"out": (lambda m: None) if a.json else out}))
    if a.json:
        out(json.dumps(facts, indent=1, default=str))
        return 0
    rows = evaluate(facts, list(LANES))
    report(facts, rows, out)
    want = lanes or default_lanes(facts)
    runnable = [lane for lane in want if not impossible_why(lane, facts)]
    if lanes and len(runnable) < len(lanes):
        for lane in lanes:
            if lane not in runnable:
                out(f"\n   {lane}: {impossible_why(lane, facts)}")
    acts = plan_actions(facts, runnable)
    missing = [r[0] for r in evaluate(facts, runnable) if r[1] == "missing"]
    try:
        cache = {fingerprint(): {"deep" if not a.fast else "any": {r[0]: [r[1], r[3]] for r in rows},
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%S")}}
        _save_cache(cache)
    except Exception:                                            # noqa: BLE001
        pass
    if not runnable:
        out("\n   none of the requested lanes can run on this machine -- nothing to install here.")
        return 0
    if not missing:
        out(f"\n   every requested lane is ready: {', '.join(runnable)}")
        return 0
    pip_acts = [x for x in acts if x.kind in ("pip", "venv")]
    if pip_acts and not facts.get("venv") and not a.allow_system_python:
        out("\n   !! this Python is not a virtual environment -- Pollard installs only into a venv. Make one first:")
        out(f"        {sys.executable} -m venv {os.path.join(pollard_home(), 'venv')}")
        out("        then activate it, `pip install pollard-weights` (or -e <checkout>) and rerun `pollard setup` "
            "(or pass --allow-system-python)")
        acts = [x for x in acts if x.kind not in ("pip", "venv")]
    out(f"\n   to make {', '.join(missing)} ready:")
    show_actions(acts, out)
    if a.dry_run:
        out("\n   dry run -- nothing was installed. Run without --dry-run to be asked, or with --yes.")
        return 0
    if not acts:
        return 1
    tty = sys.stdin.isatty() if tty is None else tty
    if not a.yes:
        if not tty:
            out("\n   not a terminal -- rerun with --yes to install, or --dry-run to only print.")
            return 1
        ans = _ask(f"\n   Install missing pieces for lanes {','.join(missing)}? [Y/n] ", inp)
        if ans not in ("", "y", "yes", "ok"):
            out("   nothing installed.")
            return 1
    res = apply(acts, yes=a.yes, out=out, runner=runner, inp=inp, tty=tty)
    failed = [x for x, ok in res if not ok and x.kind != "system"]
    facts = (detect_fn or detect)(list(LANES), deep=not a.fast)
    rows = evaluate(facts, list(LANES))
    out("\n== after setup ==")
    report(facts, rows, out)
    still = [r[0] for r in evaluate(facts, runnable) if r[1] == "missing"]
    if failed:
        out(f"\n   {len(failed)} step(s) failed: " + "; ".join(fmt_cmd(x.cmd or x.cmds[0]) for x in failed))
    return 1 if still else 0


def main(argv=None):
    return run_setup(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    sys.exit(main())
