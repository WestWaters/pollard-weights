#!/usr/bin/env python3
"""Keep Pollard's engines current: one managed tree per engine, updated from upstream, never backward.

Why this exists: install.sh cloned llama.cpp once and then printed "using existing llama.cpp" forever.
Every new architecture (qwen35, then 3.8, then 4) meant someone hand-building another copy beside the
last one, until the build box held eight llama.cpp trees and none of them could open Qwen3.5. This makes
the managed trees (runtime/llama.cpp, runtime/ik_llama.cpp) track upstream on their own:

  1. capture   -- any uncommitted runtime work in the live tree is exported first (pollard-runtime
                  --capture), because local support a published model depends on has been lost to a
                  pull before (spark2_5).
  2. stage     -- fresh upstream clone into <tree>.next; the live tree keeps serving builds meanwhile.
  3. re-apply  -- every captured patch for this tree, plus scripted patches in runtime-patches/scripts/.
                  A patch upstream already absorbed is fine; a patch that fails AND whose architectures
                  upstream lacks blocks the swap.
  4. build     -- every target (a single-target build relinks libllama and breaks the siblings), with
                  the live tree's own CMake flags carried over (CUDA arch, Metal, RPC...).
  5. verify    -- each binary answers --version, and no architecture the live build could load is
                  missing from the new one. Going backward is how a shipped model lost its runtime.
  6. swap      -- live -> <tree>.prev, staged -> live. `--rollback` swaps back.

  pollard-runtime --update                 # llama.cpp (default) ; --update all | ik_llama.cpp
  pollard-runtime --update --check         # only report how far behind upstream each engine is
  pollard-runtime --rollback llama.cpp
  pollard-runtime --update --cuda-arch '107-real;100f-virtual'   # force the CUDA arch list (wins over detection)
  POLLARD_CUDA_ARCH=120a-real pollard-runtime --update   # arch to use when the GPU is hidden (also engines.json
                                                          # "cuda_arch"; else the arch seen when it was visible)
  pollard-runtime --schedule daily         # launchd (macOS) / Task Scheduler (Windows, SYSTEM), low priority, -j cores/2
  pollard-runtime --check                  # live vs upstream + how the last update ended ("update blocked: ...")

The scheduled run (`--update all --scheduled daily|weekly`) never collides with a build: while the machine lock,
a POLLARD_HEAVY_LOCK directory, a release / `pollard --run` / llama-* process or (Windows) a running SYSTEM release
task is present it SKIPS and the next hourly attempt retries; it asks again right before the swap. A refused update
(a patch that needs a rebase, a failed build) is recorded for --check and leaves no half-built .next behind.

Pipelines call update_engine() themselves when a model's architecture is missing (POLLARD_AUTO_UPDATE=0
turns that off), so a new family -- Qwen 3.8, Qwen 4 -- builds the day upstream supports it.
"""
import contextlib as _contextlib
import datetime as _dt
import glob
import json
import os
import re
import shutil
import stat
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _load_engines():
    """runtime-patches/engines.json: every engine Pollard builds against, where it comes from (a branch or
    a pull ref when the work is not upstream yet), and which captured patches ride on it."""
    reg = {"llama.cpp": {"url": "https://github.com/ggml-org/llama.cpp"},
           "ik_llama.cpp": {"url": "https://github.com/ikawrakow/ik_llama.cpp"}}
    try:
        reg.update({k: v for k, v in json.load(open(os.path.join(REPO, "runtime-patches", "engines.json"))).items()
                    if not k.startswith("_")})
    except (OSError, ValueError):
        pass
    for name, e in reg.items():
        e.setdefault("dir", os.path.join(REPO, "runtime", name))
    return reg


ENGINES = _load_engines()
KEY_BINS = ("llama-quantize", "llama-imatrix", "llama-perplexity", "llama-cli", "llama-server")
CARRY_FLAGS = ("GGML_CUDA", "CMAKE_CUDA_ARCHITECTURES", "GGML_METAL", "GGML_RPC", "GGML_NATIVE", "GGML_VULKAN",
               "GGML_HIP", "GGML_CUDA_FA_ALL_QUANTS", "GGML_BLAS", "GGML_BLAS_VENDOR", "LLAMA_CURL")
WIN = sys.platform == "win32"


def _rmtree(path):
    """rmtree that works on Windows, where git marks its object files read-only."""
    def _force(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass
    if os.path.exists(path):
        shutil.rmtree(path, onerror=_force)
    return not os.path.exists(path)


def log(msg):
    print(f"[runtime {_dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


class GitOwnershipError(RuntimeError):
    """git refused a tree as "dubious ownership" -- never the same thing as "this patch is absent"."""


def _is_git(cmd):
    if isinstance(cmd, str):
        return cmd.lstrip().startswith("git ")
    return bool(cmd) and os.path.basename(str(cmd[0])).lower() in ("git", "git.exe")


def _git_target(cmd):
    if not isinstance(cmd, str):
        for i, x in enumerate(cmd[:-1]):
            if x == "-C":
                return cmd[i + 1]
    return os.getcwd()


def trust_trees(*paths):
    """Let git work in these trees whoever owns them, for this process and every git it starts.

    The box runs the nightly update as SYSTEM while the trees belong to the user, and git then answers
    every command with "detected dubious ownership". `git apply --check --reverse` failing looked exactly
    like "this patch is not in the live build", so every live patch was skipped without a word. A
    safe.directory entry in GIT_CONFIG_COUNT/KEY/VALUE is command-line scope, which git honours for it
    (a repo's own config would not be), and it reaches the git calls in pollard_runtime too."""
    env = os.environ
    try:
        n = int(env.get("GIT_CONFIG_COUNT") or 0)
    except ValueError:
        n = 0
    have = {env.get(f"GIT_CONFIG_VALUE_{i}") for i in range(n) if env.get(f"GIT_CONFIG_KEY_{i}") == "safe.directory"}
    for p in paths:
        a = os.path.abspath(p)
        for v in dict.fromkeys((a, a.replace("\\", "/"))):
            if v not in have:
                env[f"GIT_CONFIG_KEY_{n}"], env[f"GIT_CONFIG_VALUE_{n}"] = "safe.directory", v
                have.add(v)
                n += 1
    env["GIT_CONFIG_COUNT"] = str(n)


def _run(cmd, cwd=None, timeout=None, env=None):
    if WIN and isinstance(cmd, str):
        r = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, errors="replace",
                           timeout=timeout, env=env)
    else:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace", timeout=timeout, env=env)
    # Fail loudly: a refused tree must stop the update, not read as "patch absent" / "no changes".
    if r.returncode != 0 and _is_git(cmd) and "dubious ownership" in (r.stderr or ""):
        tree = _git_target(cmd) if cwd is None else cwd
        raise GitOwnershipError(f"git refuses {tree} (dubious ownership: it belongs to another account than "
                                f"this one). Run as the tree's owner, or: git config --global --add "
                                f"safe.directory {os.path.abspath(tree).replace(os.sep, '/')}")
    return r


def _git(tree, *a):
    r = _run(["git", "-C", tree, *a], timeout=120)
    return r.stdout.strip() if r.returncode == 0 else ""


def upstream_head(url, ref=None):
    """(sha, date) of the source an engine tracks: the default branch, a branch, or a pull ref."""
    m = re.match(r"https://github.com/([^/]+/[^/]+?)(?:\.git)?$", url)
    if ref:
        r = _run(["git", "ls-remote", url, ref if ref.startswith("pull/") else f"refs/heads/{ref}"], timeout=60)
        return (r.stdout.split()[0][:9] if r.returncode == 0 and r.stdout.strip() else None), None
    if m:
        try:
            import urllib.request
            req = urllib.request.Request(f"https://api.github.com/repos/{m.group(1)}/commits/HEAD",
                                         headers={"User-Agent": "pollard-runtime", "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=30) as fh:
                d = json.load(fh)
            return d["sha"][:9], d["commit"]["committer"]["date"][:10]
        except Exception:                                                  # noqa: BLE001
            pass
    r = _run(["git", "ls-remote", url, "HEAD"], timeout=60)
    return (r.stdout.split()[0][:9] if r.returncode == 0 and r.stdout else None), None


def live_state(tree):
    if not os.path.isdir(tree):
        return {"present": False}
    is_git = os.path.isdir(os.path.join(tree, ".git"))
    return {"present": True, "git": is_git, "commit": _git(tree, "rev-parse", "--short=9", "HEAD") if is_git else None,
            "date": _git(tree, "log", "-1", "--format=%cs") if is_git else None,
            "dirty": bool(_git(tree, "status", "--porcelain", "--untracked-files=no")) if is_git else False,
            "bin": bin_dir(tree)}


def bin_dir(tree):
    for d in (os.path.join(tree, "build", "bin", "Release"), os.path.join(tree, "build", "bin"), tree):
        if any(os.path.exists(os.path.join(d, b + (".exe" if WIN else ""))) for b in KEY_BINS):
            return d
    return None


class CudaArchError(RuntimeError):
    """The GPU in this box cannot be built for with the CUDA toolkit that is installed."""


# Rubin (compute capability 10.7, sm_107) is only known to nvcc from CUDA 13.4 on. An older nvcc
# fails deep inside the first .cu file with "Unsupported gpu architecture 'compute_107'", an hour
# into a build -- so it is checked before the build starts, with the fix in the message.
RUBIN_CC = "10.7"
RUBIN_MIN_NVCC = (13, 4)
# CUDA 13 removed offline compilation for everything below compute capability 7.5 (Maxwell,
# Pascal, Volta). ik_llama's non-native default list is "60;61;70;75;80" (or "50;61;70;75;80"), and
# a carried-over list from a CUDA 12 build can hold the same entries: both fail to configure on
# CUDA 13 with "Unsupported gpu architecture 'compute_60'".
CUDA13_MIN_ARCH = 75
# What a portable (non-native) CUDA 13 build gets when there is no GPU to read: llama.cpp's own
# CUDA >= 13 default list.
CUDA13_PORTABLE = "75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real;121a-real"


def _probe(cmd, timeout=20):
    """stdout of a version/query command, '' when the tool is absent or fails."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def parse_compute_caps(text):
    """`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` -> sorted distinct ['10.7', '12.0'].
    One line per GPU; a mixed box lists every capability once. Anything that is not X.Y is ignored
    (nvidia-smi prints an error line, not a number, when the driver is not loaded)."""
    caps = {m.group(1) for m in re.finditer(r"(?m)^[ \t]*(\d+\.\d+)[ \t\r]*$", text or "")}
    return sorted(caps, key=lambda c: tuple(int(x) for x in c.split(".")))


def parse_nvcc_version(text):
    """`nvcc --version` -> (13, 4), None when unreadable."""
    m = re.search(r"release (\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_cmake_version(text):
    """`cmake --version` -> (4, 0, 2), None when unreadable."""
    m = re.search(r"cmake version (\d+)\.(\d+)(?:\.(\d+))?", text or "")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)) if m else None


def cmake_parses_f_suffix(ver):
    """Can this CMake validate a family-specific arch like `100f-virtual`?

    The `f` suffix conflicted with CMake's own arch-validation regex and was fixed in 3.31.8 (3.31.x)
    and 4.0.2 -- undocumented in the release notes, see Modules/Internal/CMakeCUDAArchitecturesValidate
    .cmake (llama.cpp's ggml-cuda CMakeLists records the same versions). An older CMake rejects the
    list at configure time, so the fallback list avoids `f` entirely."""
    if not ver:
        return False
    return (3, 31, 8) <= ver < (4, 0, 0) or ver >= (4, 0, 2)


def _nvcc_cmd():
    """The nvcc CMake will use: CUDACXX wins, then the toolkit CUDA_PATH names (the Visual Studio
    generator builds with that toolkit, not whatever is first on PATH), then PATH."""
    if os.environ.get("CUDACXX"):
        return os.environ["CUDACXX"]
    cp = os.environ.get("CUDA_PATH")
    if cp:
        exe = os.path.join(cp, "bin", "nvcc" + (".exe" if WIN else ""))
        if os.path.isfile(exe):
            return exe
    return shutil.which("nvcc") or "nvcc"


def detect_cuda(probe=_probe):
    """(compute caps of every visible GPU, nvcc version, cmake version) -- each empty/None if unknown."""
    caps = parse_compute_caps(probe(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]))
    return caps, parse_nvcc_version(probe([_nvcc_cmd(), "--version"])), \
        parse_cmake_version(probe([shutil.which("cmake") or "cmake", "--version"]))


def _arch_num(cap):
    major, minor = cap.split(".")
    return int(major) * 10 + int(minor)


def archs_for_cap(cap, nvcc, cmake):
    """The arch entries that give one compute capability native code.

    10.7 (Rubin) gets real sm_107 code plus `100f` PTX: Rubin is in the Blackwell datacenter family,
    so family-portable sm_100f code runs on it -- if a kernel has no sm_107 build, the driver still
    has something better than the Hopper fallback to JIT. 12.x stays `12Xa-real`, the arch-specific
    form llama.cpp itself rewrites 12X to (the FP4 tensor-core path is not forward compatible)."""
    n = _arch_num(cap)
    if cap == RUBIN_CC:
        if nvcc is None or nvcc < RUBIN_MIN_NVCC:
            have = "no nvcc found" if nvcc is None else f"nvcc is {nvcc[0]}.{nvcc[1]}"
            raise CudaArchError(
                f"this GPU is Rubin (compute capability {RUBIN_CC}, sm_107) and {have}: sm_107 needs CUDA "
                f"toolkit >= {RUBIN_MIN_NVCC[0]}.{RUBIN_MIN_NVCC[1]}. Install it (or point CUDACXX / CUDA_PATH at "
                "it) and rerun, or pass --cuda-arch to build for an arch this toolkit knows.")
        return ["107-real", "100f-virtual"] if cmake_parses_f_suffix(cmake) else ["107a-real", "90-virtual"]
    if 120 <= n < 130:
        return [f"{n}a-real"]
    return [f"{n}-real"]


def _entry_num(entry):
    m = re.match(r"^(\d+)", entry.strip())
    return int(m.group(1)) if m else None


def _has_real_code(entries, n):
    """Does the list carry device code (not only PTX) for sm_n? `120a-real`, `120`, `120a` do."""
    return any(_entry_num(e) == n and not e.strip().endswith("-virtual") for e in entries)


def select_cuda_arch(cached, caps, nvcc, cmake, override=None, fallback=None):
    """CMAKE_CUDA_ARCHITECTURES for this build -> (value or None to leave unset, note or '').

    The update used to carry CMakeCache's value over verbatim. That is right while the box stays the
    same and wrong the day it does not: a list written for the old GPU has no code for the new one,
    and a list written under CUDA 12 holds archs CUDA 13 cannot compile. So the cached value is kept
    only while it still covers every GPU in the box under this toolkit. `native` already resolves
    against the GPU present at configure time, so it is kept -- the RTX 5070 Ti build box (cc 12.0)
    builds exactly as before -- except on Rubin, where it is replaced by the explicit list that also
    carries the Blackwell-family PTX. An `override` (--cuda-arch) wins over all of it.

    With NO GPU visible (CUDA_VISIBLE_DEVICES=-1 -- the build box hides its GPU while it is in use), `native`
    is never kept: CMake then records "No CUDA devices found.-real" and the build carries no kernels for
    the card at all. `fallback` (POLLARD_CUDA_ARCH, the engine's cuda_arch in engines.json, else the arch
    detected on an earlier run) is used instead; with none, CUDA 13 gets the portable list and anything
    older is refused (CudaArchError) rather than built for nothing."""
    if override:
        return override, f"--cuda-arch override: {override}"
    if not caps and cached and cached.strip().lower() == "native":
        if fallback:
            return fallback, f"no GPU visible, '{cached}' would build for none -- using {fallback}"
        if nvcc and nvcc >= (13, 0):
            return CUDA13_PORTABLE, f"no GPU visible, '{cached}' would build for none -- portable list {CUDA13_PORTABLE}"
        raise CudaArchError(
            f"no GPU is visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}) and the "
            f"build's arch is '{cached}', which would compile kernels for no GPU at all. Set POLLARD_CUDA_ARCH "
            "(e.g. 120a-real), \"cuda_arch\" for this engine in runtime-patches/engines.json, or pass --cuda-arch "
            "-- or run once with the GPU visible so its arch is recorded.")
    if not caps and not cached and fallback:
        return fallback, f"no GPU visible -- using configured arch {fallback}"
    want = []
    for cap in caps:                                  # raises for Rubin on a too-old toolkit
        want += [a for a in archs_for_cap(cap, nvcc, cmake) if a not in want]
    cuda13 = bool(nvcc and nvcc >= (13, 0))
    if cached and cached.strip().lower() in ("native", "all", "all-major"):
        if RUBIN_CC in caps:
            return ";".join(want), f"'{cached}' replaced on Rubin by {';'.join(want)} (adds sm_100f PTX)"
        return cached, ""
    if cached:
        entries = [e for e in cached.split(";") if e.strip()]
        kept = [e for e in entries if not (cuda13 and (_entry_num(e) or 0) < CUDA13_MIN_ARCH)]
        dropped = [e for e in entries if e not in kept]
        missing = [c for c in caps if not _has_real_code(kept, _arch_num(c))]
        if missing:
            return ";".join(want), (f"cached arch list '{cached}' has no code for this box's GPU "
                                    f"(compute capability {', '.join(missing)}) -- using {';'.join(want)}")
        if dropped:
            value = ";".join(kept) or CUDA13_PORTABLE
            return value, f"dropped {';'.join(dropped)}: CUDA 13 cannot compile below sm_75 -- using {value}"
        return cached, ""
    if want:                                          # CUDA build with no arch carried: name the GPU
        return ";".join(want), f"no arch in the cache -- built for this box's GPU: {';'.join(want)}"
    if cuda13:                                        # no GPU to read, and the engine default breaks on 13
        return CUDA13_PORTABLE, f"no GPU detected, CUDA 13 -- portable list {CUDA13_PORTABLE}"
    return None, ""


CUDA_CAPS_FILE = os.path.join(REPO, "runtime", "cuda_caps.json")


def _save_caps(caps):
    """Remember the GPU's compute capability, for runs where the GPU is hidden."""
    try:
        old = json.load(open(CUDA_CAPS_FILE)).get("caps")
    except (OSError, ValueError, AttributeError):
        old = None
    if old == caps:
        return
    try:
        os.makedirs(os.path.dirname(CUDA_CAPS_FILE), exist_ok=True)
        json.dump({"caps": caps, "detected": _dt.datetime.now().isoformat(timespec="seconds")},
                  open(CUDA_CAPS_FILE, "w"), indent=1)
    except OSError as exc:
        log(f"cuda arch: could not record detected compute capability ({exc})")


def _saved_caps():
    try:
        caps = json.load(open(CUDA_CAPS_FILE)).get("caps") or []
        return [c for c in caps if re.match(r"^\d+\.\d+$", str(c))]
    except (OSError, ValueError, AttributeError):
        return []


def configured_cuda_arch(configured=None, nvcc=None, cmake=None):
    """The arch to build for when no GPU is visible -> (value or None, where it came from).
    POLLARD_CUDA_ARCH, then the engine's engines.json `cuda_arch` (`configured`), then the compute
    capability recorded the last time the GPU was visible."""
    env = os.environ.get("POLLARD_CUDA_ARCH", "").strip()
    if env:
        return env, "POLLARD_CUDA_ARCH"
    if configured:
        return configured, "engines.json cuda_arch"
    caps = _saved_caps()
    if caps:
        want = []
        for cap in caps:
            want += [a for a in archs_for_cap(cap, nvcc, cmake) if a not in want]
        return ";".join(want), f"compute capability {', '.join(caps)} recorded in {CUDA_CAPS_FILE}"
    return None, ""


def cache_flags(tree, cuda_arch=None, detect=None, configured=None):
    """CMake options the live build was configured with, so an update builds the same engine.

    On a CUDA build the arch list is re-checked against the GPU and toolkit in the box right now
    (select_cuda_arch); raises CudaArchError when the toolkit cannot build for the GPU at all.
    A visible GPU's compute capability is recorded, so a later run with the GPU hidden still builds
    for it (`configured` = the engine's engines.json cuda_arch, see configured_cuda_arch)."""
    p = os.path.join(tree, "build", "CMakeCache.txt")
    out = {}
    if os.path.isfile(p):
        for line in open(p, encoding="utf-8", errors="replace"):
            m = re.match(r"^([A-Z0-9_]+):[A-Z]+=(.*)$", line.strip())
            if m and m.group(1) in CARRY_FLAGS and m.group(2) not in ("", "OFF"):
                out[m.group(1)] = m.group(2)
    if not out:                                     # no previous build: pick the platform's accelerator
        if sys.platform == "darwin":
            out["GGML_METAL"] = "ON"
        elif shutil.which("nvcc") or os.environ.get("CUDA_PATH") or os.environ.get("CUDACXX"):
            out["GGML_CUDA"] = "ON"
            out["CMAKE_CUDA_ARCHITECTURES"] = "native"
    if out.get("GGML_CUDA") == "ON":
        caps, nvcc, cmake = ([], None, None) if cuda_arch else (detect or detect_cuda)()
        fallback, src = (None, "")
        if caps:
            _save_caps(caps)
        elif not cuda_arch:
            fallback, src = configured_cuda_arch(configured, nvcc, cmake)
        value, note = select_cuda_arch(out.get("CMAKE_CUDA_ARCHITECTURES"), caps, nvcc, cmake, cuda_arch, fallback)
        if fallback and value == fallback:
            note += f" (from {src})"
        if value:
            out["CMAKE_CUDA_ARCHITECTURES"] = value
        if note:
            log(f"cuda arch: {note}")
    out.setdefault("LLAMA_CURL", "OFF")
    return out


def _vcvars():
    for base in (r"C:\Program Files (x86)\Microsoft Visual Studio\2022", r"C:\Program Files\Microsoft Visual Studio\2022"):
        for ed in ("BuildTools", "Community", "Professional", "Enterprise"):
            p = os.path.join(base, ed, r"VC\Auxiliary\Build\vcvars64.bat")
            if os.path.isfile(p):
                return p
    return None


def build(tree, flags, jobs):
    """Configure + build ALL targets. Binaries land in build/bin on every platform."""
    defs = [f"-D{k}={v}" for k, v in sorted(flags.items())]
    if WIN:
        out = os.path.join(tree, "build", "bin")
        defs += [f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={out}", f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={out}"]
        vc = _vcvars()
        pre = f'call "{vc}" >nul && ' if vc else ""
        cfg = pre + f'cmake -B build -G "Visual Studio 17 2022" {" ".join(defs)}'
        bld = pre + f"cmake --build build --config Release -j {jobs}"
        steps = [cfg, bld]
    else:
        # relative rpaths ($ORIGIN / @loader_path): the tree is built as <tree>.next and then RENAMED live; an
        # absolute build rpath kept pointing at .next, so on macOS every binary failed to load libllama-common
        # after the swap (2026-10-28, 2e0d86beb) while the pre-swap smoke had passed
        steps = [["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON", *defs],
                 ["cmake", "--build", "build", "--config", "Release", "-j", str(jobs)]]
    for i, s in enumerate(steps):
        r = _run(s, cwd=tree, timeout=4 * 3600)
        if r.returncode != 0:
            tail = (r.stdout + r.stderr)[-1500:]
            return False, f"build step failed: {s if isinstance(s, str) else ' '.join(s)}\n{tail}"
        if i == 0 and flags.get("GGML_CUDA") == "ON":
            bad = configured_cuda_arch_problem(tree)
            if bad:
                return False, f"configure produced a CUDA build for no GPU -- not compiling: {bad}"
    return True, "built all targets"


NO_DEVICE = "no cuda devices"


def configured_cuda_arch_problem(tree):
    """After configure: does the build actually target a GPU? -> problem text, '' when it does.

    `native` with no visible GPU makes CMake record CMAKE_CUDA_ARCHITECTURES_NATIVE "No CUDA devices
    found.-real" (build/CMakeFiles/<ver>/CMakeCUDACompiler.cmake) and the build has no kernels for the
    card. Caught here, before an hour of compiling, instead of at the first model load."""
    cache = os.path.join(tree, "build", "CMakeCache.txt")
    value, ggml_native = None, False
    try:
        for line in open(cache, encoding="utf-8", errors="replace"):
            if re.match(r"^GGML_NATIVE:BOOL=(ON|TRUE|1)$", line.strip(), re.I):
                ggml_native = True
            m = re.match(r"^(CMAKE_CUDA_ARCHITECTURES\w*):[A-Z]+=(.*)$", line.strip())
            if m:
                if NO_DEVICE in m.group(2).lower():
                    return f"{m.group(1)}={m.group(2)} in {cache}"
                if m.group(1) == "CMAKE_CUDA_ARCHITECTURES":
                    value = m.group(2).strip()
    except OSError:
        return ""
    if value is not None and not value:
        return f"CMAKE_CUDA_ARCHITECTURES is empty in {cache}"
    if value is None:
        if not ggml_native:
            return ""
        value = "native"            # llama.cpp's own default with GGML_NATIVE=ON and no arch given
    for f in glob.glob(os.path.join(tree, "build", "CMakeFiles", "*", "CMakeCUDACompiler.cmake")):
        try:
            text = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        m = re.search(r'set\(CMAKE_CUDA_ARCHITECTURES_NATIVE\s+"([^"]*)"\)', text)
        if m and "native" in value.lower() and (NO_DEVICE in m.group(1).lower() or not m.group(1).strip()):
            return (f"CMAKE_CUDA_ARCHITECTURES={value} resolved to '{m.group(1)}' ({f}) -- the GPU was not "
                    "visible at configure time; set POLLARD_CUDA_ARCH or pass --cuda-arch")
    return ""


#: a binary that cannot load its own libraries (macOS dyld / Linux ld.so / Windows): never a successful start
LOADER_ERROR = re.compile(r"Library not loaded|dyld\[|image not found|error while loading shared libraries|"
                          r"cannot open shared object file|was not found|0xc0000135", re.I)


def smoke(tree):
    d = bin_dir(tree)
    if not d:
        return False, "no binaries produced"
    bad = []
    for b in KEY_BINS:
        exe = os.path.join(d, b + (".exe" if WIN else ""))
        if not os.path.exists(exe):
            bad.append(f"{b} missing")
            continue
        r = _run([exe, "--version"], timeout=60)
        out = (r.stdout or "") + (r.stderr or "")
        # The question is "does it start", not "does it implement --version": llama-quantize has no
        # --version and answers with its usage and exit 1. A crash or a missing DLL prints neither.
        # a loader failure names the library's path (".../build/bin/...") -- that is not a start
        if LOADER_ERROR.search(out) or (r.returncode != 0 and not re.search(r"usage|version|build", out, re.I)):
            bad.append(f"{b} did not start (exit {r.returncode}): {out.strip()[-160:]}")
    return (not bad), ("; ".join(bad) if bad else f"{len(KEY_BINS)} binaries answer --version")


def _patches_for(tree):
    """The engine's declared patches (engines.json), else every capture made from a tree of this name."""
    from pollard_runtime import load_captured, _slug
    caps = load_captured(os.path.join(REPO, "runtime-patches"))
    for e in ENGINES.values():
        if os.path.abspath(e["dir"]) == os.path.abspath(tree) and e.get("patches"):
            want = set(e["patches"])
            return [m for m in caps if os.path.basename(m["_patch"])[:-6] in want]
    slug = _slug(tree)
    return [m for m in caps if m.get("tree_slug") == slug]


def _git_err(r, n=400):
    """git's stderr (else stdout) on one line, the tail that names the cause."""
    text = " | ".join(l.strip() for l in ((r.stderr or "") or (r.stdout or "")).splitlines() if l.strip())
    return text[-n:] or "(no output)"


def refresh_staging_index(staging):
    """Bring the freshly cloned STAGING tree's index back in line with its files before any patch.

    On the Windows build box (update run as SYSTEM) every patch failed right after the clone -- clef-omni
    "not in the live build", the rest BLOCKED -- and every one applied cleanly after `git reset --hard HEAD`.
    The clone's index held stat data that no longer matched the files (racy timestamps, autocrlf rewrites,
    a second account touching the tree), and `git apply --3way` refuses any file that "does not match
    index". A fresh clone has nothing of ours in it, so resetting it is safe. Never call this on the live
    tree. Returns a note when something had to be fixed or failed, '' when the index was already clean."""
    r = _run(["git", "-C", staging, "update-index", "-q", "--refresh"], timeout=300)
    stale = r.returncode != 0
    d = _run(["git", "-C", staging, "diff", "--quiet", "HEAD", "--"], timeout=300)
    if not stale and d.returncode == 0:
        return ""
    h = _run(["git", "-C", staging, "reset", "-q", "--hard", "HEAD"], timeout=600)
    if h.returncode != 0:
        return f"index refresh: reset of the staged clone failed: {_git_err(h)}"
    return ("index refresh: staged clone's index was stale" + (f" ({_git_err(r, 200)})" if stale else "")
            + " -- reset to HEAD before applying patches")


def reapply(staging, tree, fresh=False, record=None):
    """Captured patches + scripted patches. Returns (blocking_failures, notes).

    `fresh`: there is no live tree to compare against, so every DECLARED patch is part of what this
    engine is (ik_llama without k2-horizon is not the engine Pollard means) and must apply.
    `record`: a list that gets one row per captured patch -- {patch, plain, threeway, verdict, git} -- for the
    drift check (check_patches), which runs exactly this, in this order, without building."""
    from pollard_runtime import _arch_list_from_source
    up = _arch_list_from_source(staging) or set()
    block, notes = [], []
    n = refresh_staging_index(staging)
    if n:
        notes.append(n)
    for m in _patches_for(tree):
        row = {"patch": os.path.basename(m["_patch"])[:-6]}
        if record is not None:
            # exact-context apply on the tree as the earlier patches left it: does it still apply without a merge?
            p = _run(["git", "-C", staging, "apply", "--check", "--ignore-whitespace", m["_patch"]], timeout=300)
            row["plain"] = p.returncode == 0
            record.append(row)
        snap = _git(staging, "stash", "create")          # index + tree as the earlier patches left them
        r = _run(["git", "-C", staging, "apply", "--3way", "--ignore-whitespace", m["_patch"]], timeout=300)
        row["threeway"] = r.returncode == 0
        if r.returncode == 0:
            row["verdict"] = "applied"
            notes.append(f"re-applied {os.path.basename(m['_patch'])}")
            continue
        row["git"] = _git_err(r, 300)
        # Keep git's own words: a failure that is not a real conflict (a stale index, a missing base blob)
        # reads exactly like one without them.
        notes.append(f"{os.path.basename(m['_patch'])}: git apply failed (exit {r.returncode}): {_git_err(r)}")
        # A failed --3way apply does NOT leave the tree alone: the hunks that merged stay applied and the
        # rest are written as <<<<<<< conflicts. ik_llama's superseded k2-horizon-arch patch did exactly
        # that, was reported "not needed", and the build then died on conflict markers in llama-arch.h.
        _run(["git", "-C", staging, "reset", "-q", "--hard", "HEAD"], timeout=300)
        if snap:
            _run(["git", "-C", staging, "stash", "apply", "-q", "--index", snap], timeout=300)
        name = os.path.basename(m["_patch"])
        if not fresh and not patch_is_live(tree, m):
            row["verdict"] = "not live"
            notes.append(f"{name} is not in the live build -- nothing to carry over")
        elif _is_arch_patch(m, up):
            row["verdict"] = "superseded"
            notes.append(f"{name} not needed: upstream now carries the architecture it added")
        elif _absorbed(staging, m):
            row["verdict"] = "absorbed"
            notes.append(f"{name} not needed: upstream already carries its changes")
        else:
            row["verdict"] = "CONFLICT"
            block.append(f"{name} is live in this engine and does not apply to upstream -- rebase it first "
                         f"(it changes {', '.join(m.get('files', [])[:4])}{'...' if len(m.get('files', [])) > 4 else ''}; git: {_git_err(r, 300)})")
    for sc in sorted(glob.glob(os.path.join(REPO, "runtime-patches", "scripts", "*.py"))):
        r = _run([sys.executable, sc, staging], timeout=600)
        notes.append(f"{os.path.basename(sc)}: {'ok' if r.returncode == 0 else 'not applicable (' + _git_err(r, 120) + ')'}")
    return block, notes


def patch_is_live(tree, m):
    """Is this captured patch part of the build that is serving right now?

    A git tree answers exactly (the patch reverse-applies). A binary-only tree is asked for the patch's
    most distinctive strings. A patch the live engine does not carry cannot be lost by updating it."""
    if not os.path.isdir(tree):
        return False
    if os.path.isdir(os.path.join(tree, ".git")):
        r = _run(["git", "-C", tree, "apply", "--check", "--reverse", "--ignore-whitespace", m["_patch"]], timeout=120)
        return r.returncode == 0
    from pollard_runtime import archs_in_binaries
    distinct = [x for x in m.get("adds_strings", []) if len(x) >= 8 and re.search(r"[-_.]", x)]
    if not distinct:
        return False
    found, _ = archs_in_binaries(tree, distinct)
    return len(found) >= max(1, len(distinct) // 2)


_ARCH_ADDED = (re.compile(r'^\+\s*\{\s*LLM_ARCH_[A-Z0-9_]+\s*,\s*"([^"]+)"\s*\}', re.M),   # llama-arch.cpp name map
               re.compile(r'^\+\s*MODEL_ARCH\.[A-Z0-9_]+\s*:\s*"([^"]+)"', re.M))             # gguf-py MODEL_ARCH_NAMES


def patch_arch_names(diff_text):
    """The architecture names a patch registers: its added LLM_ARCH_NAMES / MODEL_ARCH_NAMES entries."""
    return {n for rx in _ARCH_ADDED for n in rx.findall(diff_text or "")}


def _is_arch_patch(m, upstream_archs):
    """A patch whose architectures upstream now ALL carries is superseded.

    Decided by the exact architecture names the patch registers, never by its other added strings: the
    clef-omni patch also adds the plain string "clef", upstream has `clef` but not `clef-omni`, and an
    any()-over-strings test called the patch "not needed" (2026-10-15, staging 620d222)."""
    try:
        text = open(m["_patch"], encoding="utf-8", errors="replace").read()
    except (OSError, KeyError):
        return False
    archs = patch_arch_names(text)
    return bool(archs) and archs <= set(upstream_archs or ())


def _absorbed(staging, m, need=0.9):
    """Did upstream merge this patch's change itself? Checked by content: the patch's substantive added
    lines are (nearly) all in the staged files already. The ifm-llama MSVC regex fix was merged upstream
    (69d3a4e) and then edited there (e78bd94), so it neither applies nor reverse-applies -- yet carrying
    it is exactly what upstream now does."""
    from pollard_runtime import _added_lines
    try:
        want = {x for x in _added_lines(open(m["_patch"], encoding="utf-8", errors="replace").read())
                if len(x) >= 12 and not x.startswith("//")}
    except OSError:
        return False
    have = set()
    for f in m.get("files") or []:
        try:
            have |= {ln.strip() for ln in open(os.path.join(staging, f), encoding="utf-8", errors="replace")}
        except OSError:
            return False                               # a file the patch touches is gone: not absorbed
    return bool(want) and len(want & have) >= need * len(want)


def lost_archs(old_tree, new_tree):
    """Architectures the live tree can load that the new one cannot -- the 'going backward' check."""
    from pollard_runtime import _arch_list_from_source, archs_in_binaries
    old = _arch_list_from_source(old_tree)
    new = _arch_list_from_source(new_tree) or set()
    if old is None:                                   # binary-only tree: ask its binaries about upstream's list
        from pollard_runtime import upstream_archs
        cands = (upstream_archs() or set()) | new
        old, _ = archs_in_binaries(old_tree, cands)
    return sorted(set(old) - set(new))


def _changed_lines(diff_text):
    """{file: sorted added/removed lines} -- what a diff changes, independent of hunk order, line numbers,
    context and line endings. Header lines are only recognised between `diff --git` and the first hunk."""
    out, cur, in_hunk = {}, None, False
    for line in diff_text.splitlines():
        line = line.rstrip("\r")
        if line.startswith("diff --git "):
            cur, in_hunk = None, False
        elif not in_hunk and line.startswith("+++ "):
            cur = line[4:].strip()
            cur = cur[2:] if cur.startswith("b/") else cur
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and cur and line[:1] in "+-":
            out.setdefault(cur, []).append(line.rstrip())
    return {f: sorted(v) for f, v in out.items() if v}


def _touched_files(diff_text):
    """Every file a diff touches, from its `diff --git a/X b/Y` headers -- including mode-only, rename and
    binary changes, which carry no +/- lines and so never show up in _changed_lines."""
    out = set()
    for line in diff_text.splitlines():
        m = re.match(r"^diff --git a/(.*) b/(.*)$", line.rstrip("\r"))
        if m:
            out.update(m.groups())
    return out


def dirty_is_declared(tree):
    """True when a tree's uncommitted changes are exactly its live declared patches.

    Those are already stored in runtime-patches/, so a pre-update capture of them only writes the same patch
    again under a new date -- every night, on every box. An untracked source file that no live patch creates,
    or any change that is not one of the declared patches, still returns False, and is captured before the
    update as before."""
    from pollard_runtime import untracked_sources
    patches = [m for m in _patches_for(tree) if patch_is_live(tree, m)]
    created = set()
    for m in patches:
        created |= set(re.findall(r"^diff --git a/.* b/(.*)\n(?:(?:old|new|deleted) mode .*\n)*new file mode",
                                  open(m["_patch"], encoding="utf-8", errors="replace").read(), re.M))
    new = untracked_sources(tree)
    # A declared patch that adds a file (common/composed-sampler.cpp) leaves it untracked when it was applied
    # directly rather than with --3way, and an untracked source file used to mean "undeclared work": the box
    # re-captured all four declared llama.cpp patches as a 73 KB pre-update patch (2026-10-10). Files the live
    # patches themselves create count as declared; intent-to-add puts their content into the diff compared below.
    if any(f not in created for f in new):
        return False
    if new:
        _run(["git", "-C", tree, "add", "-N", "--"] + new, timeout=120)
    live = _run(["git", "-C", tree, "diff", "HEAD"], timeout=600).stdout or ""
    have = _changed_lines(live)
    if not have:
        return False
    want, want_files = {}, set()
    for m in patches:
        text = open(m["_patch"], encoding="utf-8", errors="replace").read()
        want_files |= _touched_files(text)
        for f, lines in _changed_lines(text).items():
            want.setdefault(f, []).extend(lines)
    # Content alone is not enough: a declared patch plus a chmod / rename / binary edit has the same +/- lines,
    # and the extra change would skip capture (Joey's review). The touched-file sets must match too.
    return have == {f: sorted(v) for f, v in want.items()} and _touched_files(live) == want_files


# ---- update status: what the last update did, for `pollard-runtime --check` ---------------------------------

STATUS_FILE = os.path.join(REPO, "runtime", "update-status.json")


def read_status(path=None):
    try:
        with open(path or STATUS_FILE, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def write_status(name, state, reason="", path=None, **extra):
    """Record what an update of engine `name` ended as: live / current / blocked / failed / skipped / deferred.
    `pollard-runtime --check` reads it back, so a refused nightly update is visible the next morning instead of
    only in a log nobody opens."""
    path = path or STATUS_FILE
    d = read_status(path)
    d[name] = {"state": state, "reason": reason, "when": _dt.datetime.now().isoformat(timespec="seconds"), **extra}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
    except OSError as exc:
        log(f"{name}: could not record the update status ({exc})")


def status_line(name, st=None):
    """One line on the last update of `name`, '' when none is recorded."""
    s = (st if st is not None else read_status()).get(name) or {}
    state, when = s.get("state"), (s.get("when") or "").replace("T", " ")[:16]
    up = f" (upstream {s['upstream']}, {when})" if s.get("upstream") else (f" ({when})" if when else "")
    if state == "blocked":
        pats = s.get("patches") or []
        what = ", ".join(pats) if pats else "a captured patch"
        return (f"update blocked: {what} {'needs' if len(pats) <= 1 else 'need'} a rebase{up} -- "
                "rebase it onto upstream and re-capture it, then rerun pollard-runtime --update")
    if state == "failed":
        return f"last update failed{up}: {s.get('reason', '')}"
    if state in ("skipped", "deferred"):
        return f"last scheduled update {state}{up}: {s.get('reason', '')}"
    if state == "live":
        return f"last update: live at {s.get('commit', '?')}{up}"
    if state == "current":
        return f"last update: already at upstream{up}"
    return ""


def _blocked_patch(entry):
    """The patch name a reapply() block entry is about ('llama.cpp-qwen3omni-deepstack.patch is live ...')."""
    first = (entry or "").split(" ", 1)[0]
    return first[:-6] if first.endswith(".patch") else first


def update_engine(name="llama.cpp", check=False, jobs=None, allow_drop=False, extra=None, cuda_arch=None, guard=None):
    """Update one engine. `guard` (the scheduled run passes busy_reasons) is asked again right before the swap: a
    release that started during the build must not have its binaries replaced under it."""
    e = ENGINES[name]
    tree = e["dir"]
    trust_trees(tree, tree + ".next", tree + ".prev")
    ok = False
    try:
        ok = _update_engine(name, check, jobs, allow_drop, extra, cuda_arch, guard)
        return ok
    except GitOwnershipError as exc:
        log(f"{name}: NOT updating -- {exc}")
        write_status(name, "failed", str(exc))
        return False
    finally:
        # A refused or failed update leaves nothing behind: a half-built .next is gigabytes on the box's
        # already-full C: and nothing ever reuses it (every update clones fresh). .prev stays -- it is the rollback.
        if not check and not ok and os.path.isdir(tree + ".next"):
            log(f"{name}: removing the unswapped staging tree {tree}.next")
            _rmtree(tree + ".next")


def _update_engine(name, check, jobs, allow_drop, extra, cuda_arch, guard=None):
    e = ENGINES[name]
    tree, url = e["dir"], e["url"]
    st = live_state(tree)
    ref = e.get("ref")
    head, hdate = upstream_head(url, ref)
    if check:
        cur = st.get("commit") or ("binary-only tree" if st.get("present") else "not installed")
        log(f"{name}: live {cur} {st.get('date') or ''} | upstream {head} {hdate or ''}"
            + (" | UP TO DATE" if head and st.get("commit") and head.startswith(st["commit"][:7]) else ""))
        line = status_line(name)
        if line:
            log(f"{name}: {line}")
        return True
    up = {"upstream": head or "?"}
    if st.get("commit") and head and head.startswith(st["commit"][:7]) and not extra and not cuda_arch:
        log(f"{name}: already at upstream {head}")
        write_status(name, "current", commit=st["commit"], **up)
        return True
    # Flags first, before any clone: a toolkit that cannot build for this GPU (Rubin on CUDA < 13.4)
    # is known now, not an hour into the build. A fresh tree has no cache, so `tree` and the staging
    # clone give the same answer. --with CMAKE_CUDA_ARCHITECTURES=... counts as an override too.
    try:
        flags = cache_flags(tree, cuda_arch=cuda_arch or (extra or {}).get("CMAKE_CUDA_ARCHITECTURES"),
                            configured=e.get("cuda_arch"))
    except CudaArchError as exc:
        log(f"{name}: NOT updating -- {exc}")
        write_status(name, "failed", str(exc), **up)
        return False
    if st.get("dirty") and dirty_is_declared(tree):
        log(f"{name}: local changes are exactly the declared patches -- already in runtime-patches/, nothing new to capture")
    elif st.get("dirty"):
        from pollard_runtime import capture
        meta, msg = capture(tree, f"pre-update-{_dt.date.today():%Y%m%d}", os.path.join(REPO, "runtime-patches"))
        log(f"{name}: captured local runtime work first -- {msg}")
        if meta is None and "no uncommitted" not in msg:
            log(f"{name}: refusing to update: {msg}")
            write_status(name, "failed", f"could not capture local runtime work first: {msg}", **up)
            return False
    staging, prev = tree + ".next", tree + ".prev"
    _rmtree(staging)
    log(f"{name}: fetching {ref or 'upstream'} {head or ''} into {staging}")
    if ref:   # a branch or pull ref: init + fetch works for both, a plain clone --branch does not do pull refs
        steps = [["git", "init", "-q", staging], ["git", "-C", staging, "remote", "add", "origin", url],
                 ["git", "-C", staging, "fetch", "-q", "--depth", "1", "origin", ref],
                 ["git", "-C", staging, "checkout", "-q", "FETCH_HEAD"]]
    else:
        steps = [["git", "clone", "--depth", "1", url, staging]]
    for stp in steps:
        r = _run(stp, timeout=1800)
        if r.returncode != 0:
            log(f"{name}: fetch failed ({' '.join(stp[-2:])}): {r.stderr.strip()[-300:]}")
            write_status(name, "failed", f"fetch failed: {r.stderr.strip()[-200:]}", **up)
            return False
    live = [m for m in _patches_for(tree) if patch_is_live(tree, m)] if st.get("present") else _patches_for(tree)
    if live:
        # A depth-1 clone has none of the base blobs, so `git apply --3way` cannot merge and every patch
        # falls back to exact context. Fetch history back to the oldest live patch's base, and the
        # composed-sampler patch, for one, applies cleanly to a month-newer upstream.
        since = min((m.get("base_date") or "2026-01-01") for m in live)
        since = (_dt.date.fromisoformat(since) - _dt.timedelta(days=2)).isoformat()
        r = _run(["git", "-C", staging, "fetch", "-q", f"--shallow-since={since}", "origin"] + ([ref] if ref else []), timeout=1800)
        log(f"{name}: history back to {since} for 3-way patch merges" + ("" if r.returncode == 0 else f" (fetch failed: {r.stderr.strip()[-200:]})"))
    block, notes = reapply(staging, tree, fresh=not st.get("present"))
    for n in notes:
        log(f"{name}: {n}")
    if block:
        log(f"{name}: NOT updating -- " + "; ".join(block) + " (rebase the patch, then rerun)")
        write_status(name, "blocked", "; ".join(block), patches=[_blocked_patch(b) for b in block], **up)
        return False
    jobs = jobs or max(2, int((os.cpu_count() or 4) * 0.6))       # leave the machine usable (60/40)
    flags.update(extra or {})                      # --with K=V; carried into every later update via the cache
    log(f"{name}: building all targets -j{jobs} with {flags}")
    ok, msg = build(staging, flags, jobs)
    if not ok:
        log(f"{name}: {msg}")
        write_status(name, "failed", msg.splitlines()[0] if msg else "build failed", **up)
        return False
    ok, msg = smoke(staging)
    log(f"{name}: smoke -- {msg}")
    if not ok:
        write_status(name, "failed", f"smoke: {msg}", **up)
        return False
    if st.get("present"):
        lost = lost_archs(tree, staging)
        if lost and not allow_drop:
            log(f"{name}: NOT swapping -- the new build cannot load {lost[:10]} which the live one can. "
                "Capture/rebase that support or pass --allow-drop.")
            write_status(name, "failed", f"the new build cannot load {', '.join(lost[:10])} which the live one can", **up)
            return False
    if guard:
        why = guard()
        if why:
            log(f"{name}: built, but NOT swapping -- {'; '.join(why)} started meanwhile; the next attempt rebuilds")
            write_status(name, "deferred", f"built, not swapped: {'; '.join(why)}", **up)
            return False
    with _swap_hold(name, held=bool(guard)):
        _rmtree(prev)
        if st.get("present"):
            os.replace(tree, prev)
        os.replace(staging, tree)
        # the binaries must still start from where they now live (a path baked in at build time breaks here)
        ok, msg = smoke(tree)
        if not ok:
            os.replace(tree, staging)
            if st.get("present"):
                os.replace(prev, tree)
            log(f"{name}: smoke FAILED after the swap -- {msg}; the previous build is live again")
            write_status(name, "failed", f"smoke after swap: {msg}", **up)
            _rmtree(staging)
            return False
    lock = {"engine": name, "commit": _git(tree, "rev-parse", "--short=9", "HEAD"), "date": _git(tree, "log", "-1", "--format=%cs"),
            "flags": flags, "built": _dt.datetime.now().isoformat(timespec="seconds"), "platform": sys.platform,
            "previous": st.get("commit") or ("binary-only" if st.get("present") else None)}
    json.dump(lock, open(os.path.join(REPO, "runtime", f"{name}.lock.json"), "w"), indent=1)
    log(f"{name}: LIVE at {lock['commit']} ({lock['date']}); previous kept at {prev}")
    write_status(name, "live", commit=lock["commit"], previous=lock["previous"], **up)
    return True


def rollback(name):
    tree = ENGINES[name]["dir"]
    prev = tree + ".prev"
    if not os.path.isdir(prev):
        log(f"{name}: no previous build to roll back to")
        return False
    tmp = tree + ".rolled"
    _rmtree(tmp)
    os.replace(tree, tmp)
    os.replace(prev, tree)
    os.replace(tmp, prev)
    log(f"{name}: rolled back; the newer build is now {prev}")
    return True


# ---- the scheduled update: never collides with a build ------------------------------------------------------

TASK_NAME = "PollardRuntimeUpdate"
#: the scheduled run starts at 04:30 and, while the machine is busy, tries again every hour for this many hours
RETRY_HOURS = 8
#: once a scheduled update has RUN (updated, current, blocked or failed -- anything but "busy, skipped"), the hourly
#: retries are no-ops until this long has passed
DUE_AFTER = {"daily": _dt.timedelta(hours=18), "weekly": _dt.timedelta(days=6)}
#: the Windows box's own release tasks run as SYSTEM; any running one that starts something under a Pollard path
#: means a release is on the machine
SYSTEM_USERS = ("SYSTEM", "NT AUTHORITY\\SYSTEM")


#: Pollard commands that never hold the machine (reports, the app, the card writer): everything else named pollard-*
#: is a release step (toolcall, bench, calib, mlx ...) and makes the machine busy
LIGHT_TOOLS = {"pollard-ls", "pollard-calc", "pollard-card", "pollard-recard", "pollard-doctor", "pollard-smoke",
               "pollard-studio", "pollard-node", "pollard-stop", "pollard-setup", "pollard-onboard", "pollard-envmatch",
               "pollard-ggufcheck", "pollard-ggufcompat", "pollard-hubcheck", "pollard-refcheck", "pollard-modelkind",
               "pollard-archfp", "pollard-workspace"}


def _base(tok):
    return re.sub(r"\.(exe|py|bat|cmd)$", "", os.path.basename(tok.strip().strip('"').replace("\\", "/")).lower())


def heavy_process(argv):
    """What a process is when it is a Pollard heavy job (a release, `pollard --run`, a llama.cpp binary, another
    engine update), else None. Judged by the program and the script it runs, never by an argument that merely
    mentions a pollard path (`cd ~/pollard && ...`)."""
    if not argv:
        return None
    b0 = _base(argv[0])
    if b0.startswith("llama-"):
        return b0
    script, rest = argv[0], argv[1:]
    if b0.startswith("python") or b0 in ("py", "pythonw"):
        i = 0
        while i < len(rest) and rest[i].startswith("-") and rest[i] != "-m":
            i += 2 if rest[i] in ("-X", "-W") else 1      # interpreter options that take a value
        if i < len(rest) and rest[i] == "-m":
            i += 1
        if i >= len(rest):
            return None
        script, rest = rest[i], rest[i + 1:]
    b = _base(script).replace("_", "-")
    if b in ("pollard-release", "pollard-auto"):
        return b
    if b == "pollard":
        # `pollard --hf X --run`, `pollard release ...`; not "/Applications/Pollard Studio.app/..." split at its space
        return b if rest and (rest[0].startswith("-") or rest[0].isalpha()) and "/" not in rest[0] else None
    if b == "pollard-runtime":
        return "pollard-runtime --update" if any(a.startswith("--update") for a in rest) and "--check" not in rest else None
    if b.startswith("pollard-") and b not in LIGHT_TOOLS:
        return b                                        # pollard-toolcall / -bench / -calib / -mlx ...: a release step
    if "--hf" in rest and "--run" in rest:
        return f"{_base(script)} (a --hf ... --run release)"   # the box's own release wrappers (clefnight_run.py)
    if b.startswith("convert-") and b.endswith("-gguf"):
        return f"{_base(script)} (a GGUF conversion)"
    # anything else running out of an engine tree: renaming the tree under it breaks the run
    marks = {f"runtime/{n}/" for n in ENGINES} | {os.path.abspath(e['dir']).replace(os.sep, '/').lower() + '/'
                                                  for e in ENGINES.values()}
    for tok in (script, *rest):
        t = tok.strip('"').replace("\\", "/").lower()
        if any(m in t for m in marks):
            return f"{_base(script)} (runs from an engine tree)"
    return None


def _split_cmdline(cmd):
    if not WIN:
        return cmd.split()
    import shlex
    try:
        return shlex.split(cmd, posix=False)
    except ValueError:
        return cmd.split()


def list_processes():
    """[(pid, ppid, argv)] of every process on the machine ([] when the listing fails)."""
    if WIN:
        r = _run('powershell -NoProfile -Command "Get-CimInstance Win32_Process | ForEach-Object { '
                 "[string]$_.ProcessId + '|' + [string]$_.ParentProcessId + '|' + $_.CommandLine }\"", timeout=120)
        rows = [ln.split("|", 2) for ln in (r.stdout or "").splitlines()]
    else:
        r = _run(["ps", "-A", "-o", "pid=,ppid=,args="], timeout=60)
        rows = [ln.split(None, 2) for ln in (r.stdout or "").splitlines()]
    out = []
    for row in rows:
        if len(row) == 3 and row[0].strip().isdigit() and row[1].strip().isdigit():
            out.append((int(row[0]), int(row[1]), _split_cmdline(row[2].strip())))
    return out


def _ancestors(procs, pid):
    parent = {p: pp for p, pp, _ in procs}
    seen = {pid}
    while parent.get(pid) and parent[pid] not in seen:
        pid = parent[pid]
        seen.add(pid)
    return seen


def running_release_tasks(text=None):
    """Names of the box's running SYSTEM release tasks, from `schtasks /query /fo csv /v /nh` (or `text`).
    Columns: 1 TaskName, 3 Status, 8 Task To Run, 14 Run As User."""
    import csv
    import io
    if text is None:
        if not WIN:
            return []
        text = _run("schtasks /query /fo csv /v /nh", timeout=120).stdout or ""
    out = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 15 or row[3].strip() != "Running":
            continue
        tn, run, user = row[1].strip(), row[8], row[14].strip().upper()
        if tn.lstrip("\\").lower() == TASK_NAME.lower() or tn.lower().startswith("\\microsoft\\"):
            continue
        if user in SYSTEM_USERS and "pollard" in run.lower() and tn not in out:
            out.append(tn)
    return out


def heavy_lock_dirs():
    """The shared heavy-lock directories this machine's jobs agree on: POLLARD_HEAVY_LOCK (os.pathsep-separated).
    `--schedule` writes the scheduling shell's value into the task, which has no environment of its own."""
    return [p for p in (os.environ.get("POLLARD_HEAVY_LOCK") or "").split(os.pathsep) if p.strip()]


def _dir_lock_holder(path):
    """'<owner line>' of a held heavy-lock directory, '' when free or its owner is dead (a stale lock)."""
    if not os.path.isdir(path):
        return ""
    try:
        who = open(os.path.join(path, "owner"), encoding="utf-8").read().strip()
    except OSError:
        return "?"                                     # just created, owner not written yet: held
    try:
        pid = int((who.split() or ["0"])[0])
    except ValueError:
        return who or "?"
    return "" if pid > 0 and pid != os.getpid() and not _pid_alive(pid) else (who or "?")


def _pid_alive(pid):
    try:
        if WIN:
            return str(pid) in (_run(["tasklist", "/FI", f"PID eq {pid}"], timeout=10).stdout or "")
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def busy_reasons(procs=None, tasks_text=None):
    """Why this machine must not have its engines swapped right now -> [] when it is free.

    The machine lock ($POLLARD_HOME/.pollard.lock), the shared POLLARD_HEAVY_LOCK directories, a running release /
    `pollard --run` / llama-* binary / other engine update (this process and its ancestors excepted), and on
    Windows the box's running SYSTEM release tasks."""
    why = []
    try:
        from pollard_workspace import running_job, pollard_home
        job = running_job()
        if job:
            why.append(f"the machine lock {os.path.join(pollard_home(), '.pollard.lock')} is held by pid "
                       f"{job.get('pid')} ({job.get('what', '?')})")
    except Exception as exc:                                               # noqa: BLE001
        log(f"guard: could not read the machine lock ({exc})")
    for d in heavy_lock_dirs():
        who = _dir_lock_holder(d)
        if who:
            why.append(f"the heavy-job lock {d} is held ({who})")
    procs = list_processes() if procs is None else procs
    mine = _ancestors(procs, os.getpid())
    seen = set()
    for pid, _pp, argv in procs:
        if pid in mine:
            continue
        what = heavy_process(argv)
        if what and what not in seen:
            seen.add(what)
            why.append(f"{what} is running (pid {pid})")
    for tn in running_release_tasks(tasks_text):
        why.append(f"the release task {tn} is running")
    return why


@_contextlib.contextmanager
def _swap_hold(name, held=True):
    """Hold the machine lock for the seconds the swap takes, so a release starting right then waits for it
    instead of opening a half-renamed tree. Not taken (and not needed) by a manual update."""
    path = None
    if held:
        try:
            from pollard_workspace import pollard_home
            path = os.environ.get("POLLARD_MACHINE_LOCK") or os.path.join(pollard_home(), ".pollard.lock")
            if os.path.exists(path):
                path = None
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump({"pid": os.getpid(), "what": f"pollard-runtime: swapping {name}"}, fh)
        except OSError:
            path = None
    try:
        yield
    finally:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass


def scheduled_update(names, when="daily", jobs=None, busy=None, now=None, status_path=None):
    """What the scheduled task runs. Skips (and the next hourly attempt retries) while anything heavy is on the
    machine; otherwise updates every engine exactly as `--update` does, re-checking the guard before each swap."""
    busy = busy or busy_reasons
    now = now or _dt.datetime.now()
    st = read_status(status_path).get("_scheduled") or {}
    try:
        last = _dt.datetime.fromisoformat(st["ran"]) if st.get("ran") else None
    except ValueError:
        last = None
    if last and now - last < DUE_AFTER.get(when, DUE_AFTER["daily"]):
        log(f"scheduled update: already ran {last:%Y-%m-%d %H:%M} ({st.get('result', '?')}) -- next one is not due yet")
        return True
    why = busy()
    if why:
        log(f"scheduled update SKIPPED -- {'; '.join(why)}. The next hourly attempt retries "
            f"(04:30 + {RETRY_HOURS} h window), else the next scheduled day.")
        for n in names:
            write_status(n, "skipped", "; ".join(why), path=status_path)
        return True
    ok = all([update_engine(n, jobs=jobs, guard=busy) for n in names])
    states = {n: (read_status(status_path).get(n) or {}).get("state") for n in names}
    if "deferred" in states.values():
        return ok                                       # a release began mid-update: still due, retried next hour
    write_status("_scheduled", "ran", path=status_path, ran=now.isoformat(timespec="seconds"),
                 result="ok" if ok else ", ".join(f"{n} {s}" for n, s in states.items()))
    return ok


def scheduled_jobs():
    """Compile jobs for the scheduled build: about half the cores, so the machine stays usable."""
    return max(2, (os.cpu_count() or 4) // 2)


def _baked_env():
    """The environment the scheduled task needs and does not have: SYSTEM's ~ is not the user's (POLLARD_HOME), the
    shared heavy lock is a shell variable, and launchd's PATH has no git/cmake."""
    from pollard_workspace import pollard_home
    env = {"POLLARD_HOME": pollard_home()}
    for k in ("POLLARD_HEAVY_LOCK", "POLLARD_MACHINE_LOCK", "PATH"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def windows_task_bat(when, py, me, jobs, logf, env):
    """The .bat the Windows task runs: below-normal CPU (`start /low`), half the cores, appended to update.log."""
    lines = ["@echo off", "rem written by pollard-runtime --schedule; rerun that to change it"]
    lines += [f'set "{k}={v}"' for k, v in env.items()]
    lines += [f'cd /d "{REPO}"',
              f'start "" /low /b /wait "{py}" "{me}" --update all --scheduled {when} --jobs {jobs} >> "{logf}" 2>&1']
    return "\r\n".join(lines) + "\r\n"


def windows_task_command(when, bat):
    """The schtasks line. A start date in the PAST (01/01/2026): a near-future /st on a new task fires on its own
    (every box task re-ran at once on 2026-10-09). SYSTEM so it survives logoff and the son's console Ctrl+C;
    hourly retries for RETRY_HOURS so a skip while a release runs is rescheduled, not lost."""
    sc = "/sc weekly /d SUN" if when == "weekly" else "/sc daily"
    tr = f'"\\"{bat}\\""' if " " in bat else f'"{bat}"'
    return (f"schtasks /create /tn {TASK_NAME} /tr {tr} {sc} /st 04:30 /sd 01/01/2026 "
            f"/ri 60 /du {RETRY_HOURS:02d}:00 /ru SYSTEM /rl HIGHEST /f")


def launchd_plist(when, py, me, jobs, logf, env):
    """The macOS LaunchAgent: 04:30 plus an attempt every hour of the retry window (Sunday only when weekly)."""
    wd = "<key>Weekday</key><integer>0</integer>" if when == "weekly" else ""
    cal = "".join(f"<dict>{wd}<key>Hour</key><integer>{h}</integer><key>Minute</key><integer>30</integer></dict>"
                  for h in range(4, 4 + RETRY_HOURS))
    envx = "".join(f"<key>{k}</key><string>{_xml(v)}</string>" for k, v in env.items())
    args = "".join(f"<string>{_xml(x)}</string>" for x in
                   (py, me, "--update", "all", "--scheduled", when, "--jobs", str(jobs)))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.pollard.runtime-update</string>
<key>ProgramArguments</key><array>{args}</array>
<key>WorkingDirectory</key><string>{_xml(REPO)}</string>
<key>EnvironmentVariables</key><dict>{envx}</dict>
<key>StartCalendarInterval</key><array>{cal}</array>
<key>Nice</key><integer>15</integer><key>LowPriorityIO</key><true/>
<key>StandardOutPath</key><string>{_xml(logf)}</string><key>StandardErrorPath</key><string>{_xml(logf)}</string>
</dict></plist>"""


def _xml(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def schedule(when):
    me = os.path.abspath(__file__).replace("pollard_runtime_update.py", "pollard_runtime.py")
    py = sys.executable
    logf = os.path.join(REPO, "runtime", "update.log")
    jobs = scheduled_jobs()
    if sys.platform == "darwin":
        plist = os.path.expanduser("~/Library/LaunchAgents/com.pollard.runtime-update.plist")
        if when == "off":
            _run(["launchctl", "unload", plist]); os.path.exists(plist) and os.remove(plist)
            return log("schedule removed")
        os.makedirs(os.path.dirname(logf), exist_ok=True)
        open(plist, "w").write(launchd_plist(when, py, me, jobs, logf, _baked_env()))
        _run(["launchctl", "unload", plist]); r = _run(["launchctl", "load", plist])
        return log(f"scheduled {when} at 04:30 (+ hourly retries for {RETRY_HOURS} h while busy) via launchd ({plist})"
                   + ("" if r.returncode == 0 else " -- load failed: " + r.stderr.strip()))
    if WIN:
        if when == "off":
            _run(f"schtasks /delete /tn {TASK_NAME} /f"); return log("schedule removed")
        bat = os.path.join(REPO, "runtime", "runtime-update.bat")
        os.makedirs(os.path.dirname(bat), exist_ok=True)
        with open(bat, "w", encoding="utf-8", newline="") as fh:
            fh.write(windows_task_bat(when, py, me, jobs, logf, _baked_env()))
        r = _run(windows_task_command(when, bat))
        _run(f'powershell -NoProfile -Command "$t=Get-ScheduledTask {TASK_NAME}; $t.Settings.Priority=7; '
             "$t.Settings.ExecutionTimeLimit='PT6H'; Set-ScheduledTask -InputObject $t | Out-Null\"")
        return log(f"scheduled {when} at 04:30 (+ hourly retries for {RETRY_HOURS} h while busy) via Task Scheduler as "
                   f"SYSTEM, below-normal priority, -j{jobs}; run it now with: schtasks /run /tn {TASK_NAME}"
                   + ("" if r.returncode == 0 else f" -- {r.stderr.strip() or r.stdout.strip()}"))
    log("scheduling: add a cron line: 30 4-11 * * * " + f"nice -n 15 {py} {me} --update all --scheduled {when} --jobs {jobs}")


def auto_update_for(arch):
    """Called by pipelines when no local build loads `arch`. Returns True if an update ran and succeeded."""
    if os.environ.get("POLLARD_AUTO_UPDATE", "1") == "0":
        return False
    log(f"no local build loads '{arch}' -- updating llama.cpp from upstream (POLLARD_AUTO_UPDATE=0 to disable)")
    return update_engine("llama.cpp")


# ---- patch drift: does every captured patch still apply to upstream? (CI; no build) -------------------------

def stage_upstream(name, staging):
    """Fetch engine `name`'s source into `staging` as --update stages it: a depth-1 clone (or init + fetch of its
    branch / pull ref), then history back to the oldest declared patch's base so `git apply --3way` can merge.
    -> (ok, upstream commit or the error)."""
    e = ENGINES[name]
    url, ref = e["url"], e.get("ref")
    _rmtree(staging)
    if ref:
        steps = [["git", "init", "-q", staging], ["git", "-C", staging, "remote", "add", "origin", url],
                 ["git", "-C", staging, "fetch", "-q", "--depth", "1", "origin", ref],
                 ["git", "-C", staging, "checkout", "-q", "FETCH_HEAD"]]
    else:
        steps = [["git", "clone", "-q", "--depth", "1", url, staging]]
    for stp in steps:
        for attempt in range(3):                      # a dropped connection is not drift: try again before failing
            r = _run(stp, timeout=1800)
            if r.returncode == 0:
                break
            if stp[1:2] == ["clone"]:
                _rmtree(staging)
        if r.returncode != 0:
            return False, f"fetch failed ({' '.join(stp[-2:])}): {_git_err(r, 300)}"
    pats = _patches_for(e["dir"])
    if pats:
        since = min((m.get("base_date") or "2026-01-01") for m in pats)
        since = (_dt.date.fromisoformat(since) - _dt.timedelta(days=2)).isoformat()
        r = _run(["git", "-C", staging, "fetch", "-q", f"--shallow-since={since}", "origin"] + ([ref] if ref else []),
                 timeout=1800)
        if r.returncode != 0:
            log(f"{name}: history fetch back to {since} failed ({_git_err(r, 200)}) -- 3-way merges may fall back")
    return True, _git(staging, "rev-parse", "--short=9", "HEAD")


def check_patches(name, staging):
    """Apply engine `name`'s captured patches to an upstream tree exactly as --update's re-apply does (same function,
    same order, every declared patch required as on a fresh install), and build nothing.
    -> (rows, notes): one row per patch {patch, plain, threeway, verdict, git}; verdict CONFLICT = needs a rebase."""
    trust_trees(staging)
    rows = []
    _block, notes = reapply(staging, ENGINES[name]["dir"], fresh=True, record=rows)
    return rows, notes


def drift_report(results):
    """Markdown for the drift check. `results`: [(engine, upstream commit, rows, error)]."""
    lines, bad = ["# Runtime patch drift", ""], []
    for name, head, rows, err in results:
        e = ENGINES.get(name, {})
        src = e.get("url", "").replace("https://github.com/", "") + (f" {e['ref']}" if e.get("ref") else "")
        commit = (f"[`{head}`]({e['url']}/commit/{head})" if head and e.get("url", "").startswith("https://github.com/")
                  else f"`{head or '?'}`")
        lines += [f"## {name} @ {commit} ({src})", ""]
        if err:
            lines += [f"**could not stage upstream:** {err}", ""]
            bad.append(f"{name}: could not stage upstream ({err})")
            continue
        if not rows:
            lines += ["no captured patches", ""]
            continue
        lines += ["| patch | plain apply | 3-way apply | result |", "|---|---|---|---|"]
        for r in rows:
            res = r.get("verdict", "?")
            if res == "CONFLICT":
                res = "**CONFLICT -- needs a rebase**"
                bad.append(f"{r['patch']} no longer applies to {name} @ {head}"
                           + (f" (git: {r['git']})" if r.get("git") else ""))
            lines.append(f"| `{r['patch']}` | {'ok' if r.get('plain') else 'FAILS'} | "
                         f"{'ok' if r.get('threeway') else 'FAILS'} | {res} |")
        lines.append("")
    lines += (["## Needs a rebase", ""] + [f"- {b}" for b in bad] if bad else ["Every captured patch applies."])
    return "\n".join(lines) + "\n", bad


def drift_main(names, work=None, tree=None, summary=None):
    """`pollard-runtime --check-patches ENGINE|all`: 0 when every captured patch applies to its upstream, else 1."""
    import tempfile
    work = work or tempfile.mkdtemp(prefix="pollard-drift-")
    results = []
    for name in names:
        if not _patches_for(ENGINES[name]["dir"]):
            log(f"{name}: no captured patches -- nothing to check")
            continue
        staging = tree or os.path.join(work, name)
        if tree:
            ok, head = True, _git(tree, "rev-parse", "--short=9", "HEAD")
        else:
            log(f"{name}: staging {ENGINES[name]['url']} {ENGINES[name].get('ref') or ''} into {staging}")
            ok, head = stage_upstream(name, staging)
        if not ok:
            log(f"{name}: {head}")
            results.append((name, None, [], head))
            continue
        rows, notes = check_patches(name, staging)
        for n in notes:
            log(f"{name}: {n}")
        for r in rows:
            log(f"{name} @ {head}: {r['patch']:40s} plain {'ok' if r.get('plain') else 'FAILS':5s} "
                f"3-way {'ok' if r.get('threeway') else 'FAILS':5s} -> {r.get('verdict')}")
        results.append((name, head, rows, None))
    md, bad = drift_report(results)
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(md)
    for b in bad:
        log(f"DRIFT: {b}")
    if not bad:
        log("every captured patch applies to its upstream")
    return 0 if not bad else 1
