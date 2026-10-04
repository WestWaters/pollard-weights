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


def cache_flags(tree):
    """CMake options the live build was configured with, so an update builds the same engine."""
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
        elif shutil.which("nvcc") or os.environ.get("CUDA_PATH"):
            out["GGML_CUDA"] = "ON"
            out["CMAKE_CUDA_ARCHITECTURES"] = "native"
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


def reapply(staging, tree):
    """Captured patches + scripted patches. Returns (blocking_failures, notes)."""
    from pollard_runtime import _arch_list_from_source
    up = _arch_list_from_source(staging) or set()
    block, notes = [], []
    for m in _patches_for(tree):
        r = _run(["git", "-C", staging, "apply", "--3way", "--ignore-whitespace", m["_patch"]], timeout=300)
        if r.returncode == 0:
            notes.append(f"re-applied {os.path.basename(m['_patch'])}")
            continue
        name = os.path.basename(m["_patch"])
        if not patch_is_live(tree, m):
            notes.append(f"{name} is not in the live build -- nothing to carry over")
        elif _is_arch_patch(m, up):
            notes.append(f"{name} not needed: upstream now carries the architecture it added")
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


def update_engine(name="llama.cpp", check=False, jobs=None, allow_drop=False, extra=None):
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
    if st.get("commit") and head and head.startswith(st["commit"][:7]) and not extra:
        log(f"{name}: already at upstream {head}")
        return True
    if st.get("dirty"):
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
    live = [m for m in _patches_for(tree) if patch_is_live(tree, m)] if st.get("present") else []
    if live:
        # A depth-1 clone has none of the base blobs, so `git apply --3way` cannot merge and every patch
        # falls back to exact context. Fetch history back to the oldest live patch's base, and the
        # composed-sampler patch, for one, applies cleanly to a month-newer upstream.
        since = min((m.get("base_date") or "2026-01-01") for m in live)
        since = (_dt.date.fromisoformat(since) - _dt.timedelta(days=2)).isoformat()
        r = _run(["git", "-C", staging, "fetch", "-q", f"--shallow-since={since}", "origin"] + ([ref] if ref else []), timeout=1800)
        log(f"{name}: history back to {since} for 3-way patch merges" + ("" if r.returncode == 0 else f" (fetch failed: {r.stderr.strip()[-200:]})"))
    block, notes = reapply(staging, tree) if st.get("present") else ([], [])
    for n in notes:
        log(f"{name}: {n}")
    if block:
        log(f"{name}: NOT updating -- " + "; ".join(block) + " (rebase the patch, then rerun)")
        return False
    jobs = jobs or max(2, int((os.cpu_count() or 4) * 0.6))       # leave the machine usable (60/40)
    flags = cache_flags(tree) if st.get("present") else cache_flags(staging)
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
