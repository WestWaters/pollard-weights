#!/usr/bin/env python3
"""EXLR8 lane, live on the Windows box (RTX 5070 Ti, sm_120): build -> load -> KLD -> tool-gate a Qwen2.5-0.5B ladder.

It runs the product path, not a test harness: `pollard release --hf <model> --lanes exlr8 --run` (smoke, the
tool-aware pollard-calib corpus, pollard-hf-smooth, pollard-probe on the smoothed copy, ONE pollard-exlr8 encode at
K=2..6, every rung of the ladder as a manifest over it, pollard-toolcall through the in-process server, pollard-verify
--lane exlr8 --gate, pollard-card) -- then collects what it measured into one JSON.

THE GPU IS NOT OURS BY DEFAULT. Before anything starts, and every 20 s while it runs, nvidia-smi is read through
pollard-toolcall's own GPU-busy check (gpu_users): any process on the GPU that is not the desktop shell or an app named
in POLLARD_GPU_IGNORE means it REFUSES to start, or stops the run and kills its process tree. On the box, Discord,
Spotify, Opera, Medal, Voicemod and Steam's web helper (and a few tray utilities) are always resident, so they are
ignored by default (BOX_IGNORE, unless POLLARD_GPU_IGNORE is already set); a game is not, and blocks. Nothing here
passes -ngl / --device to take the GPU anyway.

    python experiments\\exlr8\\box_live_check.py                      # refuse-or-run, Qwen2.5-0.5B-Instruct
    python experiments\\exlr8\\box_live_check.py --check-only         # just the GPU / disk / env verdict
    python experiments\\exlr8\\box_live_check.py --print-bat          # the .bat + schtasks lines below, filled in

Scheduled run (SYSTEM, the way box jobs run -- a detached ssh shell dies with the session):

    REM one-time: a checkout of the branch next to the main one
    git -C C:\\pollard\\pw fetch origin feat/exlr8-lane
    git -C C:\\pollard\\pw worktree add C:\\pollard\\pw-exlr8 origin/feat/exlr8-lane
    REM write C:\\pollard\\exlr8_live.bat with --print-bat, then (a start date in the PAST + /run: never auto-fires)
    schtasks /create /tn PollardExlr8Live /tr C:\\pollard\\exlr8_live.bat /sc once /sd 01/01/2026 /st 00:00 /ru SYSTEM /rl HIGHEST /f
    schtasks /run /tn PollardExlr8Live
    REM when C:\\pollard\\exlr8_live.log ends with EXLR8_LIVE_DONE:
    schtasks /delete /tn PollardExlr8Live /f

Exit codes: 0 every rung built and measured, 1 something failed (see the JSON), 3 refused (GPU busy / no GPU /
disk short), 4 stopped mid-run because someone started using the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LADDER = [2.5, 3.0, 3.25, 4.0, 5.0, 6.0]
MIN_DISK_GB = 20.0
#: always-resident, non-game apps on the box's GPU (name fragments; pollard-toolcall's POLLARD_GPU_IGNORE)
BOX_IGNORE = "discord,spotify,opera,medal,voicemod,steamwebhelper,razer central,mscopilot,snippingtool,widgetboard"
BAT = r"""@echo off
REM EXLR8 live check (pollard-exlr8): refuses to start if the GPU is in use. Scheduled as SYSTEM.
set PYTHONUTF8=1
set POLLARD_THREADS=19
set POLLARD_HOME=C:\Users\jwate\pollard
set HF_HOME=C:\Users\jwate\.cache\huggingface
set POLLARD_GPU_IGNORE={ignore}
set LOG=C:\pollard\exlr8_live.log
cd /d {repo}
echo == exlr8 live %date% %time% > %LOG%
{python} -X utf8 experiments\exlr8\box_live_check.py --out C:\pollard\exlr8_live.json >> %LOG% 2>&1
call echo EXLR8_LIVE_EXIT %%ERRORLEVEL%% >> %LOG%
echo EXLR8_LIVE_DONE >> %LOG%
"""


def gpu_verdict():
    """(ok, why): an NVIDIA GPU with nobody on it -- pollard-toolcall's gpu_users (desktop shell and
    POLLARD_GPU_IGNORE apps excluded). Reads nvidia-smi only."""
    os.environ.setdefault("POLLARD_GPU_IGNORE", BOX_IGNORE)
    import pollard_toolcall as T
    free = T.gpu_free_gb()
    if free is None:
        return False, "no NVIDIA GPU (nvidia-smi missing or failed)"
    users = T.gpu_users()
    if users:
        return False, "GPU in use by " + ", ".join(users[:6]) + " -- not ours to take"
    return True, f"GPU free ({free:.1f} GB VRAM free)"


def env_verdict():
    out = {}
    for mod in ("torch", "transformers", "safetensors", "xxhash", "datasets"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "ok")
        except Exception as e:                                   # noqa: BLE001
            out[mod] = f"MISSING ({type(e).__name__})"
    try:
        import torch
        out["cuda"] = torch.cuda.is_available() and torch.cuda.get_device_name(0)
        out["sm"] = torch.cuda.is_available() and ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception:                                            # noqa: BLE001
        out["cuda"] = False
    out["disk_free_gb"] = round(shutil.disk_usage(os.path.expanduser("~")).free / 1e9, 1)
    return out


def _kill_tree(proc):
    if proc.poll() is not None:
        return
    if sys.platform.startswith("win"):
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.kill()


def run_release(model, ladder, plan_path, poll_s=20):
    """pollard release for the exlr8 lane, with a watchdog that stops it the moment the GPU stops being free."""
    cmd = [sys.executable, "-X", "utf8", os.path.join(TOOLS, "pollard_release.py"), "--hf", model, "--plan",
           plan_path, "--lanes", "exlr8", "--run"]
    print("   $ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO)
    stopped = {}

    def watch():
        import pollard_toolcall as T
        os.environ.setdefault("POLLARD_GPU_IGNORE", BOX_IGNORE)
        while proc.poll() is None:
            # our own steps show up as python(.exe) on the GPU; anything else (a game, an overlay) is someone's
            users = [u for u in T.gpu_users() if not u.lower().startswith("python")]
            if users:
                stopped["why"] = "GPU taken by " + ", ".join(users[:6])
                print(f"\n   !! {stopped['why']}: stopping the run", flush=True)
                _kill_tree(proc)
                return
            time.sleep(poll_s)
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    rc = proc.wait()
    th.join(timeout=1)
    return rc, stopped.get("why")


def collect(model):
    import pollard_release as R
    import pollard_workspace as ws
    cfg = R.load_plan(None)
    c = R._ctx(cfg, model)
    res_path = R.results_path(c, "exlr8")
    results = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else {}
    builds = [b for b in ws.read_manifest(R.lane_key(c, "exlr8")).get("builds", []) if b.get("lane") == "exlr8"]
    store = None
    try:
        import pollard_exlr8 as X
        store = X.read_store(c["exlr8_store"])
    except Exception:                                            # noqa: BLE001
        pass
    rungs = []
    for b in sorted(builds, key=lambda x: x.get("bpw") or 0):
        r = results.get(b["name"], {})
        tc = r.get("toolcall") or {}
        rungs.append({"rung": b["name"], "bpw": b.get("bpw"), "bytes": b.get("bytes"), "kld": r.get("kld"),
                      "top1": r.get("top1"), "toolcall": {k: tc.get(k) for k in ("status", "reason", "rate",
                                                                                     "ref_rate", "retention", "pass")},
                      "verified": b.get("verified"), "gate_failed": (r.get("verified") or {}).get("failed")})
    state = R.load_state(c["state"])
    return {"rungs": rungs, "encode_minutes": (store or {}).get("encode_minutes"),
            "store_widths": (store or {}).get("widths"), "results": res_path, "state": c["state"],
            "failed_steps": sorted(k for k, v in state.items() if (v or {}).get("status") == "failed"),
            "card": os.path.join(c["release_dir"], "exlr8", "README.md")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ladder", default=",".join(map(str, LADDER)))
    ap.add_argument("--out", default="exlr8_live.json")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--print-bat", action="store_true")
    a = ap.parse_args(argv)
    if a.print_bat:
        print(BAT.format(repo=REPO, python=sys.executable, ignore=os.environ.get("POLLARD_GPU_IGNORE") or BOX_IGNORE))
        return 0
    ok, why = gpu_verdict()
    env = env_verdict()
    report = {"model": a.model, "gpu": why, "env": env, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"== EXLR8 live check :: {a.model}\n   {why}\n   env {env}", flush=True)
    if not ok:
        report["status"] = "refused"
        json.dump(report, open(a.out, "w"), indent=1)
        print("   REFUSED -- nothing started.")
        return 3
    if env["disk_free_gb"] < MIN_DISK_GB:
        report["status"] = f"refused: {env['disk_free_gb']} GB free < {MIN_DISK_GB}"
        json.dump(report, open(a.out, "w"), indent=1)
        print("   REFUSED -- " + report["status"])
        return 3
    if a.check_only:
        print("   check only: would run.")
        return 0
    ladder = [float(x) for x in a.ladder.split(",")]
    plan = os.path.join(os.path.dirname(os.path.abspath(a.out)) or ".", "exlr8_live_plan.yaml")
    with open(plan, "w", encoding="utf-8") as f:              # JSON is valid YAML; upload stays off
        json.dump({"org": "PollardWeights", "upload": False, "ngl": 0,
                   "calib": {"profile": "auto", "per_domain": 60},
                   "measure": {"taskeval": None, "toolcall": True},
                   "lanes": {l: {"enabled": l == "exlr8"} for l in ("gguf", "mlx", "gptq", "mx", "exl3", "draft")}
                   | {"exlr8": {"enabled": True, "rungs": ladder, "default": 4.0, "widths": [2, 3, 4, 5, 6]}}},
                  f, indent=1)
    t0 = time.time()
    rc, stopped = run_release(a.model, ladder, plan)
    report.update(collect(a.model))
    report["minutes"] = round((time.time() - t0) / 60, 1)
    report["release_exit"] = rc
    report["status"] = "stopped: " + stopped if stopped else ("ok" if rc == 0 else "failed")
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\n   {report['status']} in {report['minutes']} min -> {a.out}")
    for r in report["rungs"]:
        tc = r["toolcall"] or {}
        print(f"   {r['rung']:44s} {r['bpw'] or 0:5.2f} bpw  KLD {r['kld'] if r['kld'] is not None else '--'}  "
              f"top1 {r['top1'] if r['top1'] is not None else '--'}  tools {tc.get('retention') or tc.get('status')}")
    return 4 if stopped else (0 if rc == 0 else 1)


if __name__ == "__main__":
    sys.exit(main())
