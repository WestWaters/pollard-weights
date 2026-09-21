#!/usr/bin/env python3
"""Run real Pollard commands and stream their output back to the UI.

Three rules this module exists to enforce:

  ONE AT A TIME. A second job is refused while one is live rather than queued behind it. Two
  quantizers on one box thrash the page cache and hand you two slow builds instead of one fast one.

  NOTHING RUNS UNANNOUNCED. `plan()` resolves and returns the exact argv without executing, so the
  UI can show what it is about to do and the user can decide.

  ABORT KILLS THE TREE. Pollard tools shell out to llama-quantize and friends; killing the parent
  alone orphans the child, which then holds the GPU and keeps writing to a file nobody is watching.

    python runner.py pollard-doctor --help
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

def tools_dir() -> Path:
    """Where Pollard's tool modules are, in a checkout OR a pip install.

    pyproject sets package-dir {"" = "tools"}, so every tool installs FLAT next to this package:
    in a checkout the parent of pollard_studio/ is tools/, and in site-packages it is
    site-packages/. One expression covers both, which is why neither needs configuring.
    """
    env = os.environ.get("POLLARD_TOOLS")
    if env and (Path(env) / "pollard_bench.py").is_file():
        return Path(env)
    beside = Path(__file__).resolve().parent.parent
    if (beside / "pollard_bench.py").is_file():
        return beside
    repo = os.environ.get("POLLARD_REPO")
    if repo and (Path(repo) / "tools" / "pollard_bench.py").is_file():
        return Path(repo) / "tools"
    return beside


def repo_root() -> Path:
    """The checkout, when there is one -- used as the working directory for a tool run."""
    env = os.environ.get("POLLARD_REPO")
    if env:
        return Path(env).expanduser()
    t = tools_dir()
    return t.parent if (t.parent / "pyproject.toml").is_file() else t


#: kept as a module attribute because callers (and tests) read runner.REPO directly
REPO = repo_root()
MAX_LINES = 4000


def resolve(tool: str) -> list[str] | None:
    """argv prefix for a Pollard tool: the installed console script, else the repo source."""
    cli = tool.replace("_", "-")
    found = shutil.which(cli)
    if found:
        return [found]
    src = REPO / "tools" / (tool.replace("-", "_") + ".py")
    if src.exists():
        return [sys.executable, str(src)]
    return None


def plan(tool: str, args: list[str]) -> dict:
    """What would run, without running it."""
    argv = resolve(tool)
    if argv is None:
        return {"ok": False, "error": f"{tool} not found on PATH or in {REPO / 'tools'}",
                "argv": [], "display": ""}
    full = argv + [str(a) for a in args]
    return {"ok": True, "argv": full, "error": None,
            "display": " ".join([tool.replace("_", "-")] + [str(a) for a in args])}


class Runner:
    """One job at a time, with a rolling output buffer the UI polls."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.lines: deque[tuple[float, str]] = deque(maxlen=MAX_LINES)
        self.seq = 0                      # monotonic line counter, so the UI can poll for deltas
        self.label = ""
        self.argv: list[str] = []
        self.started = 0.0
        self.finished: float | None = None
        self.returncode: int | None = None
        self._lock = threading.Lock()

    # -- state -----------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status(self) -> dict:
        return {"running": self.running, "label": self.label,
                "display": " ".join(self.argv[-12:]) if self.argv else "",
                "elapsed": round((self.finished or time.time()) - self.started, 1) if self.started else 0,
                "returncode": self.returncode, "seq": self.seq}

    def tail(self, since: int = 0) -> dict:
        """Lines produced after `since`. The UI polls this instead of holding a socket open."""
        with self._lock:
            have = len(self.lines)
            first = self.seq - have
            start = max(0, since - first)
            out = [t for _, t in list(self.lines)[start:]]
        return {**self.status(), "from": max(since, first), "lines": out}

    # -- control ---------------------------------------------------------------------------------
    def start(self, tool: str, args: list[str], label: str = "", cwd: str | None = None) -> dict:
        if self.running:
            return {"ok": False, "error": f"'{self.label}' is still running — abort it first"}
        p = plan(tool, args)
        if not p["ok"]:
            return p

        self.label = label or tool
        self.argv = p["argv"]
        self.started = time.time()
        self.finished = None
        self.returncode = None
        self._emit(f"$ {p['display']}")

        try:
            self.proc = subprocess.Popen(
                p["argv"], cwd=cwd or str(REPO), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                # own process group, so abort can take the whole tree including llama-quantize
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except Exception as e:
            self._emit(f"! failed to start: {e}")
            self.proc = None
            return {"ok": False, "error": str(e)}

        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        return {"ok": True, "display": p["display"]}

    def abort(self) -> dict:
        if not self.running:
            return {"ok": False, "error": "nothing running"}
        pid = self.proc.pid
        self._emit("! abort requested")
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)      # the group, not just the parent
        except Exception:
            self.proc.terminate()
        for _ in range(50):                                  # 5s to exit cleanly
            if self.proc.poll() is not None:
                break
            time.sleep(0.1)
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                self.proc.kill()
            self._emit("! escalated to SIGKILL")
        return {"ok": True}

    # -- internals -------------------------------------------------------------------------------
    def _emit(self, text: str) -> None:
        with self._lock:
            self.lines.append((time.time(), text))
            self.seq += 1

    def _pump(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stdout:
                self._emit(line.rstrip("\n"))
        except Exception as e:
            self._emit(f"! stream ended: {e}")
        finally:
            proc.wait()
            self.returncode = proc.returncode
            self.finished = time.time()
            self._emit(f"— exit {proc.returncode} after {self.finished - self.started:.1f}s")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    r = Runner()
    res = r.start(sys.argv[1], sys.argv[2:])
    if not res.get("ok"):
        print("error:", res.get("error"))
        raise SystemExit(1)
    seen = 0
    while True:
        t = r.tail(seen)
        for line in t["lines"]:
            print(line)
        seen = t["seq"]
        if not t["running"]:
            break
        time.sleep(0.2)
    raise SystemExit(r.returncode or 0)


if __name__ == "__main__":
    main()
