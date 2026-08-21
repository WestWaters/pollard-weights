#!/usr/bin/env python3
"""pollard-health — is your accelerator actually at full speed, or silently degraded?

96% GPU utilisation and P0 do NOT mean healthy. A wedged GB10 / DGX Spark (or a
throttling RTX, or a Mac drowning in page-outs) shows "busy" while the real clock
sits at 1/3 speed and power at 1/5 — you lose half your throughput and never know
until you benchmark by accident. This reads the signals that actually matter:

  * NVIDIA: real SM clock vs the card's OWN max, power draw vs limit, throttle reasons
  * Apple Silicon: swap/page-outs (the memory-pressure thrash) + thermal speed-limit

and calls it plainly. Cross-vendor (NVIDIA via nvidia-smi, Apple via macOS tools).

  pollard-health              # check — read-only, safe
  pollard-health --fix        # attempt a NO-REBOOT recovery (prints the plan; --yes to run)

Root cause is usually over-committed memory (page-outs) — which `pollard-calc --ctx
--gpu` lets you avoid BEFORE you run. --fix is designed from the DGX-Spark community's
data; validate it on real silicon before trusting it (a deep firmware wedge may still
need a power-cycle).
"""
import argparse
import platform
import re
import shutil
import subprocess
import sys


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception:
        return None


def _num(s):
    m = re.search(r"-?[0-9]+\.?[0-9]*", str(s))
    return float(m.group()) if m else None


# ---------------- NVIDIA ----------------

def check_nvidia():
    """Per-GPU (label, (state, why), detail). None if no nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    base = ("index,name,clocks.sm,clocks.max.sm,power.draw,power.limit,pstate,"
            "utilization.gpu,temperature.gpu")
    # The throttle field was renamed clocks_throttle_reasons.* -> clocks_event_reasons.*
    # on newer drivers (exactly the GB10/Spark case). Try each; fall back to no field.
    r = None
    for tf in (",clocks_throttle_reasons.active", ",clocks_event_reasons.active", ""):
        r = _run(["nvidia-smi", f"--query-gpu={base}{tf}",
                  "--format=csv,noheader,nounits"])
        if r and r.returncode == 0 and r.stdout.strip():
            break
    if not r or r.returncode != 0 or not r.stdout.strip():
        return None
    out = []
    for line in r.stdout.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 9:
            continue
        idx, name = f[0], f[1]
        sm, sm_max = _num(f[2]), _num(f[3])
        pdraw, plim, pstate = _num(f[4]), _num(f[5]), f[6]
        util, temp = _num(f[7]), _num(f[8])
        throttle = f[9] if len(f) > 9 else ""
        pct = (sm / sm_max * 100) if sm and sm_max else None
        active_throttle = throttle not in ("0x0000000000000000", "Not Active", "", "N/A")
        detail = (f"SM {sm:.0f}/{sm_max:.0f} MHz"
                  + (f" ({pct:.0f}% of max)" if pct is not None else "")
                  + f" · {pdraw:.0f}/{plim:.0f} W · {pstate} · util {util:.0f}% · {temp:.0f}°C")
        # THE WEDGE: high util, no throttle flag, yet the clock is far below the max.
        if (util and util > 40 and pct is not None and pct < 65 and not active_throttle):
            state = ("DEGRADED", f"stuck at {pct:.0f}% clock despite {util:.0f}% util — you're "
                     f"at ~{pct:.0f}% throughput. 96% util / P0 is lying to you (the wedge).")
        elif active_throttle:
            state = ("THROTTLING", f"active throttle reasons: {throttle} (thermal/power/hw)")
        else:
            state = ("OK", "clock at expected level for the load")
        out.append((f"GPU{idx} {name}", state, detail))
    return out


# ---------------- Apple Silicon ----------------

def check_apple():
    """(label, (state, why), detail) for the Mac. None if not macOS."""
    if platform.system() != "Darwin":
        return None
    swap_used = 0.0
    r = _run(["sysctl", "-n", "vm.swapusage"])
    if r and r.stdout:
        m = re.search(r"used\s*=\s*([0-9.]+)([MG])", r.stdout)
        if m:
            swap_used = float(m.group(1)) * (1024 if m.group(2) == "G" else 1)  # MB
    pageouts = 0
    r = _run(["vm_stat"])
    if r and r.stdout:
        m = re.search(r"Pageouts:\s*([0-9]+)", r.stdout)
        if m:
            pageouts = int(m.group(1))
    speed_limit = 100
    r = _run(["pmset", "-g", "therm"])
    if r and r.stdout:
        m = re.search(r"CPU_Speed_Limit\s*=\s*([0-9]+)", r.stdout)
        if m:
            speed_limit = int(m.group(1))
    detail = f"swap used {swap_used/1024:.1f} GB · lifetime page-outs {pageouts:,} · CPU speed cap {speed_limit}%"
    if speed_limit < 100:
        state = ("THROTTLING", f"thermal speed-limit at {speed_limit}% — the Mac is throttling under heat")
    elif swap_used > 4096:  # >4 GB swapped = you over-committed unified memory
        state = ("DEGRADED", f"{swap_used/1024:.1f} GB swapped out — memory-pressure thrash (the page-out "
                 f"wedge). A model + KV bigger than RAM pages to disk and everything crawls.")
    else:
        state = ("OK", "no swap thrash, no thermal cap")
    return [("Apple Silicon (unified memory)", state, detail)]


# ---------------- recovery ----------------

def fix_plan():
    """Escalating NO-REBOOT recovery steps per platform: (label, [shell cmds])."""
    if platform.system() == "Darwin":
        return [("free inactive memory + purge page cache", ["sync", "sudo purge"]),
                ("(then) stop the process that over-committed, and re-run within your RAM budget "
                 "— check pollard-calc --ctx --gpu first", [])]
    # NVIDIA / Linux — escalating, least invasive first
    return [
        ("drop page caches + compact memory (clears page-out pressure)",
         ["sync", "sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'",
          "sudo sh -c 'echo 1 > /proc/sys/vm/compact_memory'"]),
        ("reset GPU clocks to default", ["sudo nvidia-smi -rgc"]),
        ("reset the GPU (needs NO processes using it)", ["sudo nvidia-smi --gpu-reset"]),
        ("reload the driver module (the no-reboot nuclear option; stop all GPU procs first)",
         ["sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia",
          "sudo modprobe nvidia && sudo modprobe nvidia_uvm"]),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--fix", action="store_true",
                    help="attempt a no-reboot recovery (prints the plan; add --yes to run it)")
    ap.add_argument("--yes", action="store_true", help="actually run --fix (sudo; use with care)")
    a = ap.parse_args()

    rows = check_nvidia() or check_apple()
    if rows is None:
        sys.exit("No supported accelerator found (need nvidia-smi, or run on Apple Silicon).")

    print("== pollard-health ==")
    worst = "OK"
    for label, (state, why), detail in rows:
        mark = {"OK": "[ OK ]", "THROTTLING": "[WARN]", "DEGRADED": "[BAD ]"}[state]
        print(f"{mark} {label}\n       {detail}\n       -> {why}")
        if state == "DEGRADED":
            worst = "DEGRADED"
        elif state == "THROTTLING" and worst != "DEGRADED":
            worst = "THROTTLING"
    print()
    if worst == "OK":
        print("VERDICT: healthy — running at full speed.")
        return
    print(f"VERDICT: {worst} — you are losing throughput. "
          + ("`pollard-health --fix` attempts a no-reboot recovery." if not a.fix else ""))

    if a.fix:
        print("\n== recovery plan (escalating; least invasive first) ==")
        print("NOTE: if none of these take, it's a power-cycle — a deep firmware wedge on "
              "integrated Grace-Blackwell may not clear from software. Validate on your hardware.\n")
        for i, (label, cmds) in enumerate(fix_plan(), 1):
            print(f"  {i}. {label}")
            for c in cmds:
                print(f"       $ {c}")
        if not a.yes:
            print("\n(dry-run — re-run with --yes to execute these, one step at a time.)")
            return
        print("\n--yes given: executing step by step, re-checking after each…")
        for label, cmds in fix_plan():
            if not cmds:
                continue
            print(f"\n-> {label}")
            for c in cmds:
                subprocess.run(c, shell=True)
            after = check_nvidia() or check_apple() or []
            if all(s[0] == "OK" for _, s, _ in after):
                print("   recovered — back to full speed.")
                return
        print("\nStill degraded after all soft resets — this one needs a power-cycle "
              "(full disconnect ~10 min for a wedged Spark).")


if __name__ == "__main__":
    main()
