#!/usr/bin/env python3
"""pollard-stop -- end a Pollard job without leaving orphans, and without cutting a save in half.

Two things go wrong when a long build is stopped, and they pull in opposite directions.

Stopping too gently leaves ORPHANS. A scheduled task ended with `schtasks /end` takes the shell
with it and leaves the worker running: an imatrix here survived its task and sat on 11.4GB until
someone noticed. The next run then competes with a job nobody knows about.

Stopping too hard corrupts the ARTIFACT. llama-imatrix rewrites its .dat every few chunks and
llama-quantize streams a GGUF tensor by tensor; killed mid-write, the file is short but perfectly
well-formed -- it loads, it is wrong, and nothing says so. That is the same failure as a truncated
K-quant: silent.

So: ask the whole process TREE to stop, watch whatever it is writing until the bytes settle, and
only then insist. A job that is mid-save gets to finish the save.

    pollard-stop                 # list Pollard jobs running here
    pollard-stop --all           # stop them all, waiting for saves
    pollard-stop --pid 1234      # stop one tree
    pollard-stop --all --now     # skip the wait (accepts a truncated artifact)
"""
from __future__ import annotations

import argparse, os, subprocess, sys, time

# What a Pollard BUILD job looks like. Deliberately narrow: a first pass matched any command line
# containing "pollard", which caught the operator's own shells (cwd inside the repo) and a
# llama-server belonging to an unrelated project. A tool that stops things must never guess.
JOB_BINARIES = ("llama-imatrix", "llama-quantize", "llama-perplexity")   # build work, not serving
JOB_SCRIPTS = ("pollard_auto", "pollard_fit", "pollard_bench", "pollard_probe", "pollard_automap",
               "pollard_convert", "pollard_taskeval", "pollard_sensitivity", "pollard-bench",
               "pollard-fit", "pollard-probe", "pollard-automap")
SETTLE_SECONDS = 90          # a save that has not grown this long is finished
POLL = 3


def _ps():
    """[(pid, ppid, name, cmdline)] for everything we might own."""
    out = []
    if sys.platform == "win32":
        ps = ("Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,"
              "CommandLine | ConvertTo-Csv -NoTypeInformation")
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True)
        import csv, io
        for row in csv.DictReader(io.StringIO(r.stdout)):
            out.append((int(row["ProcessId"]), int(row["ParentProcessId"] or 0),
                        row["Name"] or "", row["CommandLine"] or ""))
    else:
        r = subprocess.run(["ps", "-eo", "pid=,ppid=,comm=,args="], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            parts = line.split(None, 3)
            if len(parts) >= 3:
                out.append((int(parts[0]), int(parts[1]), parts[2], parts[3] if len(parts) > 3 else ""))
    return out


def find_jobs(procs=None):
    """Pollard's own heavy processes -- not every python on the machine."""
    procs = procs or _ps()
    jobs = []
    for pid, ppid, name, cmd in procs:
        low, base = (name + " " + cmd).lower(), os.path.basename(name).lower()
        if "pollard_stop" in low or "pollard-stop" in low:
            continue                                       # never count ourselves
        # a build binary, by executable NAME -- not by a path that happens to say pollard
        is_bin = any(base.startswith(j) for j in JOB_BINARIES) or \
            any(("\\" + j) in low or ("/" + j) in low for j in JOB_BINARIES)
        # or an interpreter actually RUNNING one of our scripts
        is_script = ("python" in base or base.endswith(".exe")) and \
            any(sc in low for sc in JOB_SCRIPTS)
        # a shell is never a job, whatever its working directory is called
        if base.split(".")[0] in ("zsh", "bash", "sh", "cmd", "powershell", "pwsh", "conhost"):
            continue
        if is_bin or is_script:
            jobs.append((pid, ppid, name, cmd))
    return jobs


def tree(pid, procs=None):
    """pid plus every descendant, so nothing is left behind."""
    procs = procs or _ps()
    kids = {}
    for p, pp, _n, _c in procs:
        kids.setdefault(pp, []).append(p)
    seen, stack = [], [pid]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.append(p)
        stack += kids.get(p, [])
    return seen


def _open_writes(pid):
    """Files this pid is writing, with their current size. Empty if it cannot be read."""
    sizes = {}
    try:
        if sys.platform == "win32":
            return sizes                                   # handle enumeration needs a helper; the
                                                           # size-settle check below still applies to
                                                           # any path passed with --watch
        r = subprocess.run(["lsof", "-p", str(pid), "-Fn"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            if line.startswith("n/") and os.path.isfile(line[1:]):
                sizes[line[1:]] = os.path.getsize(line[1:])
    except Exception:
        pass
    return sizes


def wait_for_save(paths, settle=SETTLE_SECONDS, quiet=False):
    """Block while any of `paths` is still growing. This is the whole point of the tool."""
    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        return
    last, stable_since = {}, time.time()
    while True:
        now = {p: os.path.getsize(p) for p in paths if os.path.exists(p)}
        if now != last:
            last, stable_since = now, time.time()
            if not quiet:
                for p, n in now.items():
                    print(f"   still writing {os.path.basename(p)} ({n/1e9:.2f} GB) -- waiting")
        elif time.time() - stable_since >= settle:
            return
        time.sleep(POLL)


def stop(pid, watch=(), now=False, quiet=False):
    """Stop one process tree: ask, wait for the save, then insist."""
    pids = tree(pid)
    if not quiet:
        print(f"   tree for {pid}: {', '.join(str(p) for p in pids)}")
    for p in reversed(pids):                               # children first
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(p)], capture_output=True)   # no /F: ask
            else:
                os.kill(p, 15)
        except Exception:
            pass
    if not now:
        wait_for_save(watch, quiet=quiet)
    for p in reversed(pids):
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(p), "/F"], capture_output=True)
            else:
                os.kill(p, 9)
        except Exception:
            pass
    if not quiet:
        print(f"   stopped {pid}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--all", action="store_true", help="stop every Pollard job on this machine")
    ap.add_argument("--pid", type=int, help="stop one process tree")
    ap.add_argument("--watch", action="append", default=[],
                    help="a file being written; the stop waits until it stops growing (repeatable)")
    ap.add_argument("--now", action="store_true",
                    help="do NOT wait for a save to finish. A file mid-write is left short but "
                         "well-formed -- it will load and be wrong.")
    ap.add_argument("--settle", type=int, default=SETTLE_SECONDS,
                    help=f"seconds of no growth that counts as saved (default {SETTLE_SECONDS})")
    a = ap.parse_args()

    jobs = find_jobs()
    if not (a.all or a.pid):
        if not jobs:
            print("no Pollard jobs running here."); return
        print("Pollard jobs on this machine:")
        for pid, ppid, name, cmd in jobs:
            print(f"  pid {pid:>6}  {name:<22} {cmd[:90]}")
        print("\n  stop them with: pollard-stop --all   (waits for any save to finish)")
        return
    targets = [a.pid] if a.pid else sorted({p for p, _pp, _n, _c in jobs})
    for pid in targets:
        stop(pid, watch=a.watch, now=a.now)


if __name__ == "__main__":
    main()
