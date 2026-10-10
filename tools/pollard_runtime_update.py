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
  pollard-runtime --schedule daily         # launchd (macOS) / Task Scheduler (Windows), low priority

Pipelines call update_engine() themselves when a model's architecture is missing (POLLARD_AUTO_UPDATE=0
turns that off), so a new family -- Qwen 3.8, Qwen 4 -- builds the day upstream supports it.
"""
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


def _run(cmd, cwd=None, timeout=None, env=None):
    if WIN and isinstance(cmd, str):
        return subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, errors="replace",
                              timeout=timeout, env=env)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace", timeout=timeout, env=env)


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


def select_cuda_arch(cached, caps, nvcc, cmake, override=None):
    """CMAKE_CUDA_ARCHITECTURES for this build -> (value or None to leave unset, note or '').

    The update used to carry CMakeCache's value over verbatim. That is right while the box stays the
    same and wrong the day it does not: a list written for the old GPU has no code for the new one,
    and a list written under CUDA 12 holds archs CUDA 13 cannot compile. So the cached value is kept
    only while it still covers every GPU in the box under this toolkit. `native` already resolves
    against the GPU present at configure time, so it is kept -- the RTX 5070 Ti build box (cc 12.0)
    builds exactly as before -- except on Rubin, where it is replaced by the explicit list that also
    carries the Blackwell-family PTX. An `override` (--cuda-arch) wins over all of it."""
    if override:
        return override, f"--cuda-arch override: {override}"
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


def cache_flags(tree, cuda_arch=None, detect=None):
    """CMake options the live build was configured with, so an update builds the same engine.

    On a CUDA build the arch list is re-checked against the GPU and toolkit in the box right now
    (select_cuda_arch); raises CudaArchError when the toolkit cannot build for the GPU at all."""
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
        value, note = select_cuda_arch(out.get("CMAKE_CUDA_ARCHITECTURES"), caps, nvcc, cmake, cuda_arch)
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
        steps = [["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", *defs],
                 ["cmake", "--build", "build", "--config", "Release", "-j", str(jobs)]]
    for s in steps:
        r = _run(s, cwd=tree, timeout=4 * 3600)
        if r.returncode != 0:
            tail = (r.stdout + r.stderr)[-1500:]
            return False, f"build step failed: {s if isinstance(s, str) else ' '.join(s)}\n{tail}"
    return True, "built all targets"


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
        if r.returncode != 0 and not re.search(r"usage|version|build", out, re.I):
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


def reapply(staging, tree, fresh=False):
    """Captured patches + scripted patches. Returns (blocking_failures, notes).

    `fresh`: there is no live tree to compare against, so every DECLARED patch is part of what this
    engine is (ik_llama without k2-horizon is not the engine Pollard means) and must apply."""
    from pollard_runtime import _arch_list_from_source
    up = _arch_list_from_source(staging) or set()
    block, notes = [], []
    for m in _patches_for(tree):
        snap = _git(staging, "stash", "create")          # index + tree as the earlier patches left them
        r = _run(["git", "-C", staging, "apply", "--3way", "--ignore-whitespace", m["_patch"]], timeout=300)
        if r.returncode == 0:
            notes.append(f"re-applied {os.path.basename(m['_patch'])}")
            continue
        # A failed --3way apply does NOT leave the tree alone: the hunks that merged stay applied and the
        # rest are written as <<<<<<< conflicts. ik_llama's superseded k2-horizon-arch patch did exactly
        # that, was reported "not needed", and the build then died on conflict markers in llama-arch.h.
        _run(["git", "-C", staging, "reset", "-q", "--hard", "HEAD"], timeout=300)
        if snap:
            _run(["git", "-C", staging, "stash", "apply", "-q", "--index", snap], timeout=300)
        name = os.path.basename(m["_patch"])
        if not fresh and not patch_is_live(tree, m):
            notes.append(f"{name} is not in the live build -- nothing to carry over")
        elif _is_arch_patch(m, up):
            notes.append(f"{name} not needed: upstream now carries the architecture it added")
        elif _absorbed(staging, m):
            notes.append(f"{name} not needed: upstream already carries its changes")
        else:
            block.append(f"{name} is live in this engine and does not apply to upstream -- rebase it first "
                         f"(it changes {', '.join(m.get('files', [])[:4])}{'...' if len(m.get('files', [])) > 4 else ''})")
    for sc in sorted(glob.glob(os.path.join(REPO, "runtime-patches", "scripts", "*.py"))):
        r = _run([sys.executable, sc, staging], timeout=600)
        notes.append(f"{os.path.basename(sc)}: {'ok' if r.returncode == 0 else 'not applicable (' + (r.stdout + r.stderr).strip().splitlines()[-1][:120] + ')'}")
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


def _is_arch_patch(m, upstream_archs):
    """A patch that only added architecture registrations upstream now has is safely superseded."""
    adds = [x for x in m.get("adds_strings", []) if re.match(r"^[a-z][a-z0-9_.\-]{2,24}$", x)]
    return bool(adds) and any(x in upstream_archs for x in adds)


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
    again under a new date -- every night, on every box. Any untracked source file, or any change that is
    not one of the declared patches, still returns False, and is captured before the update as before."""
    from pollard_runtime import untracked_sources
    if untracked_sources(tree):
        return False
    live = _run(["git", "-C", tree, "diff", "HEAD"], timeout=600).stdout or ""
    have = _changed_lines(live)
    if not have:
        return False
    want, want_files = {}, set()
    for m in _patches_for(tree):
        if not patch_is_live(tree, m):
            continue
        text = open(m["_patch"], encoding="utf-8", errors="replace").read()
        want_files |= _touched_files(text)
        for f, lines in _changed_lines(text).items():
            want.setdefault(f, []).extend(lines)
    # Content alone is not enough: a declared patch plus a chmod / rename / binary edit has the same +/- lines,
    # and the extra change would skip capture (Joey's review). The touched-file sets must match too.
    return have == {f: sorted(v) for f, v in want.items()} and _touched_files(live) == want_files


def update_engine(name="llama.cpp", check=False, jobs=None, allow_drop=False, extra=None, cuda_arch=None):
    e = ENGINES[name]
    tree, url = e["dir"], e["url"]
    st = live_state(tree)
    ref = e.get("ref")
    head, hdate = upstream_head(url, ref)
    if check:
        cur = st.get("commit") or ("binary-only tree" if st.get("present") else "not installed")
        log(f"{name}: live {cur} {st.get('date') or ''} | upstream {head} {hdate or ''}"
            + (" | UP TO DATE" if head and st.get("commit") and head.startswith(st["commit"][:7]) else ""))
        return True
    if st.get("commit") and head and head.startswith(st["commit"][:7]) and not extra and not cuda_arch:
        log(f"{name}: already at upstream {head}")
        return True
    # Flags first, before any clone: a toolkit that cannot build for this GPU (Rubin on CUDA < 13.4)
    # is known now, not an hour into the build. A fresh tree has no cache, so `tree` and the staging
    # clone give the same answer. --with CMAKE_CUDA_ARCHITECTURES=... counts as an override too.
    try:
        flags = cache_flags(tree, cuda_arch=cuda_arch or (extra or {}).get("CMAKE_CUDA_ARCHITECTURES"))
    except CudaArchError as exc:
        log(f"{name}: NOT updating -- {exc}")
        return False
    if st.get("dirty") and dirty_is_declared(tree):
        log(f"{name}: local changes are exactly the declared patches -- already in runtime-patches/, nothing new to capture")
    elif st.get("dirty"):
        from pollard_runtime import capture
        meta, msg = capture(tree, f"pre-update-{_dt.date.today():%Y%m%d}", os.path.join(REPO, "runtime-patches"))
        log(f"{name}: captured local runtime work first -- {msg}")
        if meta is None and "no uncommitted" not in msg:
            log(f"{name}: refusing to update: {msg}")
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
        return False
    jobs = jobs or max(2, int((os.cpu_count() or 4) * 0.6))       # leave the machine usable (60/40)
    flags.update(extra or {})                      # --with K=V; carried into every later update via the cache
    log(f"{name}: building all targets -j{jobs} with {flags}")
    ok, msg = build(staging, flags, jobs)
    if not ok:
        log(f"{name}: {msg}")
        return False
    ok, msg = smoke(staging)
    log(f"{name}: smoke -- {msg}")
    if not ok:
        return False
    if st.get("present"):
        lost = lost_archs(tree, staging)
        if lost and not allow_drop:
            log(f"{name}: NOT swapping -- the new build cannot load {lost[:10]} which the live one can. "
                "Capture/rebase that support or pass --allow-drop.")
            return False
    _rmtree(prev)
    if st.get("present"):
        os.replace(tree, prev)
    os.replace(staging, tree)
    lock = {"engine": name, "commit": _git(tree, "rev-parse", "--short=9", "HEAD"), "date": _git(tree, "log", "-1", "--format=%cs"),
            "flags": flags, "built": _dt.datetime.now().isoformat(timespec="seconds"), "platform": sys.platform,
            "previous": st.get("commit") or ("binary-only" if st.get("present") else None)}
    json.dump(lock, open(os.path.join(REPO, "runtime", f"{name}.lock.json"), "w"), indent=1)
    log(f"{name}: LIVE at {lock['commit']} ({lock['date']}); previous kept at {prev}")
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


def schedule(when):
    me = os.path.abspath(__file__).replace("pollard_runtime_update.py", "pollard_runtime.py")
    py = sys.executable
    if sys.platform == "darwin":
        plist = os.path.expanduser("~/Library/LaunchAgents/com.pollard.runtime-update.plist")
        if when == "off":
            _run(["launchctl", "unload", plist]); os.path.exists(plist) and os.remove(plist)
            return log("schedule removed")
        cal = "<dict><key>Hour</key><integer>4</integer><key>Minute</key><integer>30</integer></dict>"
        if when == "weekly":
            cal = cal.replace("<dict>", "<dict><key>Weekday</key><integer>0</integer>")
        logf = os.path.join(REPO, "runtime", "update.log")
        open(plist, "w").write(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.pollard.runtime-update</string>
<key>ProgramArguments</key><array><string>{py}</string><string>{me}</string><string>--update</string><string>all</string></array>
<key>WorkingDirectory</key><string>{REPO}</string>
<key>StartCalendarInterval</key>{cal}
<key>Nice</key><integer>15</integer><key>LowPriorityIO</key><true/>
<key>StandardOutPath</key><string>{logf}</string><key>StandardErrorPath</key><string>{logf}</string>
</dict></plist>""")
        _run(["launchctl", "unload", plist]); r = _run(["launchctl", "load", plist])
        return log(f"scheduled {when} at 04:30 via launchd ({plist}){'' if r.returncode == 0 else ' -- load failed: ' + r.stderr.strip()}")
    if WIN:
        if when == "off":
            _run('schtasks /delete /tn PollardRuntimeUpdate /f'); return log("schedule removed")
        sc = "/sc weekly /d SUN" if when == "weekly" else "/sc daily"
        logf = os.path.join(REPO, "runtime", "update.log")
        cmd = f'cmd /c ""{py}" "{me}" --update all >> "{logf}" 2>&1"'
        r = _run(f'schtasks /create /tn PollardRuntimeUpdate /tr "{cmd}" {sc} /st 04:30 /f')
        _run('powershell -NoProfile -Command "$t=Get-ScheduledTask PollardRuntimeUpdate; $t.Settings.Priority=7; '
             "$t.Settings.ExecutionTimeLimit='PT6H'; Set-ScheduledTask -InputObject $t | Out-Null\"")
        return log(f"scheduled {when} at 04:30 via Task Scheduler (below-normal priority)"
                   + ("" if r.returncode == 0 else f" -- {r.stderr.strip() or r.stdout.strip()}"))
    log("scheduling: add a cron line: 30 4 * * * " + f"{py} {me} --update all")


def auto_update_for(arch):
    """Called by pipelines when no local build loads `arch`. Returns True if an update ran and succeeded."""
    if os.environ.get("POLLARD_AUTO_UPDATE", "1") == "0":
        return False
    log(f"no local build loads '{arch}' -- updating llama.cpp from upstream (POLLARD_AUTO_UPDATE=0 to disable)")
    return update_engine("llama.cpp")
