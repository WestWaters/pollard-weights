#!/usr/bin/env python3
"""Pollard Studio — a desktop window over the Pollard CLI.

Why a webview and not a terminal UI: the look is knobs, faders, gauges and a silhouette mark.
Those are pixels; a terminal only has character cells. So the window is the OS's own webview
(WKWebView on macOS, WebView2 on Windows, GTK on Linux) and the UI is HTML/CSS.

Why pywebview and not Tauri or Electron: Pollard installs with pip. Tauri needs a Rust toolchain
and Electron needs Node -- either one means `pip install` stops being enough to get the app.

What this file is NOT: a second implementation of Pollard. Every number comes from a real file on
disk (workspace.py / ggufread.py) and every button runs a real tool (actions.py / runner.py).
Nothing here estimates, and nothing here re-implements a solver.

    pollard-studio                 # or: python app.py
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

import webview

from . import actions
from . import cluster as _cluster
from . import coherence
from . import convert as _convert
from . import find as _find
from . import icon as _icon
from . import modalities as _mod
from . import runner
from . import runtimes
from . import validate as _validate
from . import workspace

HERE = pathlib.Path(__file__).resolve().parent


def load_manifest() -> list:
    """Every tool and every flag, parsed out of the repo -- never hand-written.

    Hand-writing a screen per tool does not survive 60 tools: the first flag anyone adds makes the
    UI a liar, and nothing fails when it happens.
    """
    mf, tools = HERE / "manifest.py", runner.REPO / "tools"
    if not (mf.exists() and tools.is_dir()):
        return []
    try:
        out = subprocess.run([sys.executable, str(mf), str(tools)],
                             capture_output=True, text=True, timeout=90)
        return json.loads(out.stdout)["tools"] if out.returncode == 0 else []
    except Exception:
        return []


class Api:
    """The JS side calls these. Deliberately thin: this is a front end over the CLI."""

    def __init__(self):
        self.window = None
        self.run = runner.Runner()
        self._tools: list | None = None
        self._flagspec: dict | None = None
        self.server = runtimes.Runtime('llama.cpp')
        self._ws: dict | None = None
        self._remote: dict = {}          # path -> the box that holds it
        self.selected: str | None = None

    # -- data ------------------------------------------------------------------------------------
    def state(self, rescan: bool = False) -> dict:
        if self._ws is None or rescan:
            self._ws = workspace.scan()
        if self._tools is None:
            self._tools = load_manifest()
        model = workspace.pick(self._ws, self.selected)
        return {
            "home": self._ws["home"], "home_exists": self._ws["exists"],
            "scanned": self._ws["scanned"],
            "models": [{"key": m["key"], "name": m["name"], "builds": len(m["builds"])}
                       for m in self._ws["models"]],
            "model": model,
            "downloads": self._ws["downloads"],
            "tools": self._tools,
            "repo": str(runner.REPO), "repo_exists": runner.REPO.is_dir(),
            "status": self.run.status(),
            "version": _version(),
        }

    def select(self, key: str) -> dict:
        self.selected = key
        return self.state()

    def rescan(self) -> dict:
        """Re-read the workspace AND the repo's tools.

        The manifest is generated from Pollard's own argparse declarations, so a fix that lands
        in the repo -- a new flag, a corrected choice list -- must show up here without restarting
        Studio. Caching it for the life of the process meant the UI kept offering yesterday's
        contract after the tool had already changed.
        """
        self._tools = None
        self._flagspec = None
        return self.state(rescan=True)

    # -- execution -------------------------------------------------------------------------------
    def plan(self, action: str, recipe: dict) -> dict:
        """What this button would run. Never executes, and never hides a bad flag."""
        got = actions.resolve(action, recipe or {})
        if got is None:
            return {"ok": False, "ui_only": True, "display": "", "confirm": False}
        p = runner.plan(*got)
        p["confirm"] = action.split(":", 1)[0] in actions.CONFIRM
        p["outbound"] = action.split(":", 1)[0] in actions.OUTBOUND
        p["problems"] = self._problems(*got)
        if p["problems"]:
            p["ok"] = False
            p["error"] = "; ".join(p["problems"])
        return p

    def _problems(self, tool: str, args: list) -> list[str]:
        """Re-check a built argv against the tool's own declaration."""
        pairs, i = {}, 0
        while i < len(args):
            a = str(args[i])
            if not a.startswith("-"):
                i += 1
                continue
            if i + 1 < len(args) and not str(args[i + 1]).startswith("-"):
                pairs[a] = args[i + 1]
                i += 2
            else:
                pairs[a] = True
                i += 1
        got = _validate.check(tool, pairs, manifest=self._manifest()).get("problems", [])
        # a lane mismatch parses fine and fails at read time, so it is reported here too
        got += _validate.lane_fit(tool, pairs, manifest=self._manifest())
        return got + self._remote_problems(pairs)

    def _remote_problems(self, pairs: dict) -> list[str]:
        """Flag a build this machine cannot currently open.

        Deliberately checks the filesystem first, because on most clusters this is a non-issue:
        shared storage (NFS, SMB, a cluster filesystem) puts the same path on every node, and
        then there is nothing to warn about. It only speaks up when the path genuinely is not
        readable from here -- and then it says so before the run rather than twenty minutes into
        one, which is when a read error would otherwise surface.
        """
        out = []
        for flag, val in (pairs or {}).items():
            m = self._remote.get(str(val))
            if not m:
                continue
            try:
                if pathlib.Path(str(val)).exists():
                    continue                 # shared storage: this node can read it, carry on
            except OSError:
                pass
            out.append(
                f"{flag} lives on {m['host']} and is not readable from this machine. Either run "
                f"it on {m['host']}, or put the build on storage both nodes share.")
        return out

    def _manifest(self) -> dict:
        if self._flagspec is None:
            self._flagspec = _validate.load_manifest()
        return self._flagspec

    def run_tool(self, tool: str, values: dict) -> dict:
        """Run any tool with arbitrary flags — validated first, so a bad value cannot start."""
        v = _validate.check(tool, values or {}, manifest=self._manifest())
        if not v["ok"]:
            return {"ok": False, "error": "; ".join(v["problems"]), "problems": v["problems"]}
        return self.run.start(tool, v["argv"], label=tool)

    def check_tool(self, tool: str, values: dict) -> dict:
        """Validate without running, so the form can show problems as they are typed."""
        v = _validate.check(tool, values or {}, manifest=self._manifest())
        v["display"] = (tool.replace("_", "-") + " " + " ".join(map(str, v["argv"]))).strip()
        return v

    def start(self, action: str, recipe: dict) -> dict:
        got = actions.resolve(action, recipe or {})
        if got is None:
            return {"ok": False, "error": f"'{action}' has nothing to run"}
        return self.run.start(got[0], got[1], label=action)

    # -- chat / coherence ------------------------------------------------------------------------
    def convert_plan(self, build: str, target: str) -> dict:
        """Ordered steps to move a build onto another lane. Always a route, never a refusal."""
        return _convert.plan(build, target)

    def lanes(self) -> dict:
        return {"lanes": {k: v["note"] for k, v in _convert.LANES.items()}}

    def modalities(self, path: str) -> dict:
        """What this build can do, read from its artifacts -- never from its name."""
        d = _mod.detect(path)
        d["labels"] = {k: {"label": v[0], "why": v[1]} for k, v in _mod.MODALITIES.items()}
        return d

    def check_artifact(self, path: str, kind: str) -> dict:
        """Structural check on something a build generated: is it a picture / is it speech?

        Deliberately blunt. Judging whether speech sounds RIGHT needs an ASR round-trip and a
        speaker model, and whether an image depicts the prompt needs CLIP -- those live in the
        Pollard tools that own them. What this catches is how low-bit builds actually fail:
        silence, a buzz, a flat frame, uniform noise.
        """
        try:
            raw = pathlib.Path(path).expanduser().read_bytes()
        except OSError as e:
            return {"ok": False, "reason": str(e)}
        return _mod.check_image(raw) if kind == "image" else _mod.check_audio(raw)

    def load_media(self, path: str) -> dict:
        """A file the build generated, ready to play. See modalities.load_media."""
        return _mod.load_media(path)

    # -- bring your own data ---------------------------------------------------------------------
    #: what each kind of "point at your own file" field should offer in the dialog
    FILE_KINDS = {
        "text":    ("Eval corpus or prompts", ("Text (*.txt;*.md;*.jsonl;*.json)", "All files (*.*)")),
        "gguf":    ("GGUF model", ("GGUF (*.gguf)", "All files (*.*)")),
        "imatrix": ("Importance matrix", ("imatrix (*.imatrix;*.dat)", "All files (*.*)")),
        "data":    ("Benchmark datafile", ("Data (*.jsonl;*.json;*.csv;*.txt;*.bin)", "All files (*.*)")),
        "any":     ("Choose a file", ("All files (*.*)",)),
        "dir":     ("Choose a folder", ()),
    }

    def pick_file(self, kind: str = "any") -> dict:
        """Open the OS file dialog so a user can point Pollard at their OWN eval, benchmark or
        model rather than typing a path. Returns {"path": ...} or {"path": None} when cancelled.
        """
        if self.window is None:                       # headless (tests) -- nothing to open
            return {"path": None}
        label, types = self.FILE_KINDS.get(kind, self.FILE_KINDS["any"])
        # webview.OPEN_DIALOG / FOLDER_DIALOG still work but are deprecated; prefer the enum and
        # fall back so this keeps working on either side of that removal.
        _fd = getattr(webview, "FileDialog", None)
        open_d = _fd.OPEN if _fd else webview.OPEN_DIALOG
        folder_d = _fd.FOLDER if _fd else webview.FOLDER_DIALOG
        try:
            if kind == "dir":
                got = self.window.create_file_dialog(folder_d)
            else:
                got = self.window.create_file_dialog(
                    open_d, allow_multiple=False, file_types=types)
        except Exception as e:                        # a platform without a dialog must not break
            return {"path": None, "error": str(e)}
        if not got:
            return {"path": None}
        path = got[0] if isinstance(got, (list, tuple)) else got
        return {"path": str(path), "label": label}

    def search_files(self, query: str = "", kind: str = "any", limit: int = 60) -> dict:
        """Search the machine for a model, eval corpus or benchmark datafile by name.

        Bounded by find.search so a big disk cannot hang the window; `truncated` says whether
        there was more than this.
        """
        try:
            return _find.search(query or "", kind or "any", limit=max(1, min(int(limit), 200)))
        except Exception as e:
            return {"results": [], "truncated": False, "error": str(e), "searched": []}

    def hardware(self) -> dict:
        """The machine this is running on, so the defaults are not a number someone typed.

        A 16 GB target is meaningful on a 16 GB laptop and meaningless everywhere else, and a
        slider that stops at 128 GB is wrong on a workstation. Read the box.
        """
        import shutil
        total = None
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, AttributeError, OSError):
            try:                                     # Windows
                import ctypes

                class _MS(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_ulong),
                                ("dwMemoryLoad", ctypes.c_ulong),
                                ("ullTotalPhys", ctypes.c_ulonglong),
                                ("ullAvailPhys", ctypes.c_ulonglong),
                                ("ullTotalPageFile", ctypes.c_ulonglong),
                                ("ullAvailPageFile", ctypes.c_ulonglong),
                                ("ullTotalVirtual", ctypes.c_ulonglong),
                                ("ullAvailVirtual", ctypes.c_ulonglong),
                                ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
                st = _MS()
                st.dwLength = ctypes.sizeof(_MS)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
                total = int(st.ullTotalPhys)
            except Exception:
                total = None
        gb = round(total / 1e9) if total else None
        free = None
        try:
            free = round(shutil.disk_usage(self._ws["home"] if self._ws else ".").free / 1e9, 1)
        except Exception:
            pass
        return {"ram_gb": gb, "cpus": os.cpu_count(),
                # a sane place to start: most of the machine, leaving the OS room
                "suggest_target_gb": max(2, int(gb * 0.75)) if gb else 16,
                "suggest_reserve_gb": max(1, round(gb * 0.12)) if gb else 3,
                "disk_free_gb": free, "platform": sys.platform}

    # -- the pool ---------------------------------------------------------------------------------
    def cluster_survey(self, endpoints: str = "", timeout: float = 1.5) -> dict:
        """This machine and every linked box, with a combined figure and each box on its own.

        Read live each time: a peer that went down between builds must show as down.
        """
        try:
            return _cluster.survey(endpoints, timeout=timeout,
                                   home=(self._ws or {}).get("home"))
        except Exception as e:                      # a panel must never take the window with it
            return {"nodes": [], "configured": 0, "online": 0, "offline": [],
                    "combined_gb": None, "pooled": False, "error": str(e)}

    def cluster_discover(self, port: int = _cluster.DEFAULT_RPC_PORT) -> dict:
        """Sweep the local subnet for ggml-rpc-servers. Only ever on an explicit request."""
        try:
            return _cluster.discover(port=int(port))
        except Exception as e:
            return {"found": [], "scanned": 0, "subnet": None, "error": str(e)}

    def devices(self, endpoints: str = "") -> dict:
        """Every device a build can place layers on, in llama.cpp's own -ts order."""
        try:
            surv = _cluster.survey(endpoints, timeout=1.5,
                                   home=(self._ws or {}).get("home"))
            return {"devices": _cluster.device_order(surv)}
        except Exception as e:
            return {"devices": [], "error": str(e)}

    def placement_args(self, shares: dict, endpoints: str = "", ngl: int | None = None) -> dict:
        """Turn a share per device into the llama.cpp arguments that express it."""
        try:
            surv = _cluster.survey(endpoints, timeout=1.5,
                                   home=(self._ws or {}).get("home"))
            order = _cluster.device_order(surv)
            args = _cluster.placement_args(shares or {}, order, ngl)
            return {"args": args, "display": " ".join(args)}
        except Exception as e:
            return {"args": [], "display": "", "error": str(e)}

    def cluster_models(self, endpoints: str = "") -> dict:
        """Every build on every linked box, so a model can be picked from the cluster.

        Remote entries carry the host that holds them: a path only means something on its own
        machine, and a list that hides which box a file is on invites a run that cannot find it.
        """
        try:
            surv = _cluster.survey(endpoints, timeout=1.5,
                                   home=(self._ws or {}).get("home"))
        except Exception as e:
            return {"models": [], "error": str(e)}
        out = []
        for n in surv.get("nodes", []):
            if n.get("role") == "this machine" or not n.get("models"):
                continue
            for m in n["models"]:
                out.append({**m, "host": n.get("host") or n.get("endpoint"),
                            "endpoint": n.get("endpoint"), "remote": True,
                            "workspace": n.get("workspace", "")})
        out.sort(key=lambda m: (m["host"], -m.get("mtime", 0)))
        self._remote = {m["path"]: m for m in out}
        return {"models": out, "hosts": sorted({m["host"] for m in out})}

    def rpc_status(self, port: int = 0) -> dict:
        """Is this box able to hold part of a split model, and if not, exactly why."""
        try:
            return _cluster.rpc_status(int(port) or _cluster.DEFAULT_RPC_PORT)
        except Exception as e:
            return {"found": False, "running": False, "problems": [str(e)]}

    def rpc_serve(self, port: int = 0) -> dict:
        """Start ggml-rpc-server here, bound so other boxes can actually reach it."""
        try:
            return _cluster.serve_local(int(port) or _cluster.DEFAULT_RPC_PORT)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def rpc_stop(self) -> dict:
        try:
            return _cluster.stop_local()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def runtimes(self) -> dict:
        """Every runtime and whether it can be used here — local servers and remote endpoints."""
        return {"available": runtimes.available(), "current": self.server.name}

    def use_runtime(self, name: str, ngl: int = 0, ctx: int = 4096) -> dict:
        """Switch runtime. The old one is stopped rather than left holding memory."""
        try:
            self.server.stop()
            self.server = runtimes.Runtime(name, ngl=int(ngl), ctx=int(ctx))
            return {"ok": True, "runtime": name}
        except ValueError as e:
            return {"ok": False, "error": str(e)}

    def runtimes_for(self, path: str, lane: str | None = None) -> list:
        kind = "gguf" if str(path).endswith(".gguf") else "safetensors"
        return runtimes.for_build(kind, lane)

    def chat(self, gguf: str, prompt: str, max_tokens: int = 256,
             temperature: float = 0.7) -> dict:
        """Generate once, and check the result for the failures metrics cannot see.

        A build can hold its perplexity and still loop forever, never halt, or open a reasoning
        block it never closes. That only shows up when you make it generate.
        """
        up = self.server.ensure(gguf)
        if not up["ok"]:
            return {"ok": False, "error": up["error"]}
        r = self.server.complete(prompt, max_tokens, temperature)
        if not r["ok"]:
            return r
        return {"ok": True, "text": r["text"], "tokens": r["tokens"],
                "coherence": coherence.analyse(r["text"], int(max_tokens), r["stop_reason"])}

    def server_status(self) -> dict:
        return self.server.status()

    def unload(self) -> dict:
        self.server.stop()
        return {"ok": True}

    def coherence_gate(self, gguf: str, prompts: list | None = None,
                       max_tokens: int = 192) -> dict:
        """Run several generations and return a pass/fail a build can be held to."""
        prompts = prompts or [
            "Explain in two sentences why the sky is blue.",
            "What is 17 * 24? Show your working, then give the answer.",
            "List three uses for a paperclip.",
        ]
        up = self.server.ensure(gguf)
        if not up["ok"]:
            return {"pass": False, "reason": up["error"]}
        samples = []
        for q in prompts:
            r = self.server.complete(q, max_tokens)
            if not r["ok"]:
                return {"pass": False, "reason": r["error"]}
            samples.append({"text": r["text"], "max_tokens": max_tokens,
                            "stop_reason": r["stop_reason"]})
        return coherence.gate(samples)

    def tail(self, since: int = 0) -> dict:
        return self.run.tail(int(since or 0))

    def abort(self) -> dict:
        return self.run.abort()

    # -- window ----------------------------------------------------------------------------------
    def win(self, what: str) -> None:
        """The window is frameless, so the panel's own buttons are the only window controls."""
        w = self.window
        if w is None:
            return
        {"minimize": w.minimize, "toggle": w.toggle_fullscreen, "close": w.destroy}.get(
            what, lambda: None)()


def _version() -> str:
    """One version, Pollard's. A separate Studio number is only a thing to forget to bump."""
    from . import __version__
    return __version__


def main() -> None:
    ap = argparse.ArgumentParser(description="Pollard Studio")
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--framed", action="store_true",
                    help="use the OS title bar instead of the panel's own window buttons")
    args = ap.parse_args()

    api = Api()
    api.window = webview.create_window(
        "Pollard Studio", str(HERE / "ui/index.html"),
        js_api=api, width=args.width, height=args.height, min_size=(1180, 800),
        background_color="#efece5", frameless=not args.framed, easy_drag=False,
    )
    # The dock/taskbar otherwise shows a generic Python rocket, which is the wrong object.
    # Applied once before the loop and once after: pywebview creates its own NSApplication on
    # start, and whichever of the two runs last is the one the dock keeps.
    print("pollard-studio:", _icon.apply(api.window), flush=True)
    webview.start(lambda: _icon.apply(api.window), private_mode=False)


if __name__ == "__main__":
    main()
