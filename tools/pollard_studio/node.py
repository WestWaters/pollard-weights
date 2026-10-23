#!/usr/bin/env python3
"""pollard-node -- report what THIS box brings to the cluster, so Studio can total it up.

ggml's RPC protocol exposes one thing: the memory of each backend device. That is what a build
places weights in, so it is the right number for placement -- but it is not what someone staring
at a cluster wants to know. They want the machine: its RAM, its cards, its free disk. There is no
RPC call for any of that, and inventing it would be worse than leaving it blank.

So each box runs this next to its ggml-rpc-server. It is deliberately tiny:

  * read-only. There is no endpoint that runs anything, writes anything, or takes a parameter.
  * one JSON document -- hostname, platform, cpus, RAM, GPUs, free disk -- and nothing else.
  * no dependencies beyond the standard library, because a build box should not need a pip
    install to be counted.

    pollard-node                      # serve on 127.0.0.1:50053 (this machine only)
    pollard-node --lan                # serve on 0.0.0.0:50053, so the cluster's Studio can count this box
    pollard-node --port 9100 --host <this box's LAN address>

It is machine specs and a model list (with paths), so it binds loopback unless told otherwise:
`--lan` (or an explicit --host) is the deliberate step that puts it on the network. It sends no CORS
header -- Studio reads it server-side, and a web page has no business reading it -- unless an origin is
named with --allow-origin. A request whose Host is a name this box does not answer to (a DNS-rebinding
page) or whose Origin is not on that list is refused. Nothing it serves can change the box.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import pathlib
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import cluster

DEFAULT_PORT = 50053
#: loopback unless --lan / --host says otherwise
DEFAULT_HOST = "127.0.0.1"
LAN_HOST = "0.0.0.0"
#: bumped when the payload shape changes, so an older Studio can tell
SCHEMA = 1


#: where a Pollard workspace keeps finished builds
_MODEL_DIRS = ("rungs", "models", "downloads", "exl3", "mlx", "mx", "gptq", ".")
#: a single-file build. Every OTHER lane is a DIRECTORY -- listing its safetensors shards would
#: show files instead of models, and there is no such thing as picking half an MLX export.
#:
#: .pt is a trained brain, fly OR human: pollard-connectome builds either graph and
#: pollard-flybrain trains on whichever it is given, so the artifact is the same shape.
#: .feather is the CONNECTOME -- the graph a brain is trained ON, not something you run. It is
#: listed because it is expensive to build and worth finding on a cluster, and named for what it
#: is so it cannot be mistaken for a model.
_FILE_LANES = {".gguf": "GGUF", ".pt": "BRAIN", ".feather": "CONNECTOME"}


def workspace_home() -> pathlib.Path:
    return pathlib.Path(os.environ.get("POLLARD_HOME") or (pathlib.Path.home() / "pollard"))


def _dir_lane(d: str, names: set) -> str | None:
    """Which lane a directory of weights is, from what is IN it -- never from its name.

    config.json is small and worth opening: MLX, MX and an unquantized HF checkout all look
    identical from the filename list alone, and calling an MLX export "HF" sends it to the wrong
    tool.
    """
    if "quantize_config.json" in names or "quant_config.json" in names:
        return "GPTQ"
    if "exl3_config.json" in names or any(n.startswith("out_tensor") for n in names):
        return "EXL3"
    if not any(n.endswith(".safetensors") for n in names):
        return None
    if "config.json" not in names:
        return "HF"
    try:
        with open(os.path.join(d, "config.json"), encoding="utf-8", errors="replace") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return "HF"
    q = cfg.get("quantization_config") or {}
    method = str(q.get("quant_method", "")).lower()
    if method in ("compressed-tensors", "modelopt", "nvfp4", "mxfp4"):
        return "MX"
    if method in ("gptq", "awq"):
        return "GPTQ"
    if "quantization" in cfg:               # mlx_lm writes this block, and nothing else does
        return "MLX"
    return "HF"


def _dir_bytes(d: str, names: list) -> int:
    tot = 0
    for n in names:
        if n.endswith((".safetensors", ".bin", ".pt", ".gguf")):
            try:
                tot += os.path.getsize(os.path.join(d, n))
            except OSError:
                pass
    return tot


def models(home: pathlib.Path | None = None, limit: int = 400) -> list:
    """Every build on THIS box, for EVERY lane.

    A GGUF is one file; an MLX, MX, GPTQ or EXL3 export is a directory. Reporting the second as
    its shards would list files nobody can select -- "model.safetensors" is not a model.

    Deliberately shallow: names, sizes and mtimes, no header parsing. Reading headers would mean
    shipping the reader to every node and paying a disk read per build just to fill a list.
    Studio reads the header when a build is actually chosen.
    """
    root = home or workspace_home()
    out: list = []
    seen: set = set()

    def add(path, name, lane, size, mtime, parent):
        if path in seen:
            return
        seen.add(path)
        out.append({"name": name, "path": path, "dir": parent, "lane": lane,
                    "bytes": size, "mtime": mtime})

    for sub in _MODEL_DIRS:
        d = root / sub if sub != "." else root
        if not d.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames[:] = [x for x in dirnames if not x.startswith(".")][:64]
            if len(pathlib.Path(dirpath).parts) - len(root.parts) > 3:
                dirnames[:] = []
            names = set(filenames)

            # a directory that IS a build: one entry, not its shards
            lane = _dir_lane(dirpath, names)
            if lane:
                try:
                    st = os.stat(dirpath)
                except OSError:
                    continue
                add(dirpath, os.path.basename(dirpath) or str(dirpath), lane,
                    _dir_bytes(dirpath, filenames), st.st_mtime, os.path.dirname(dirpath))
                dirnames[:] = []            # its subdirs are shards, not further models
                continue

            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                flane = _FILE_LANES.get(ext)
                if not flane or fn.startswith("."):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                add(full, fn, flane, st.st_size, st.st_mtime, dirpath)
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
    out.sort(key=lambda m: -m["mtime"])
    return out[:limit]


def report(home: str | None = None) -> dict:
    """What this box brings, read from the OS -- the same detection Studio uses on itself."""
    got = cluster.local(home)
    got["schema"] = SCHEMA
    got["agent"] = "pollard-node"
    got["workspace"] = str(workspace_home())
    try:
        got["models"] = models()
    except Exception:
        got["models"] = []
    return got


def _host_name(header: str | None) -> str:
    """The name part of a Host header: 'box:50053' -> 'box', '[::1]:50053' -> '::1'."""
    h = (header or "").strip().lower()
    if h.startswith("["):
        return h[1:h.find("]")] if "]" in h else h[1:]
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


def _own_names() -> set:
    names = {"localhost"}
    try:
        hn = socket.gethostname().lower()
        names |= {hn, hn.split(".")[0], hn.split(".")[0] + ".local"}
        names.add(socket.getfqdn().lower())
    except OSError:
        pass
    return {n for n in names if n}


def host_allowed(header: str | None, extra=()) -> bool:
    """A Host this box answers to: an IP literal, localhost, its own hostname, or one named with --allow-host.
    Anything else is a name someone else pointed here -- the DNS-rebinding way into a LAN service."""
    name = _host_name(header)
    if not name:
        return False
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        pass
    return name in _own_names() or name in {e.lower() for e in extra}


class _Handler(BaseHTTPRequestHandler):
    server_version = "pollard-node"
    #: origins a browser page may read /info from (--allow-origin); none by default
    allow_origins: tuple = ()
    #: extra Host names accepted beside the box's own (--allow-host)
    allow_hosts: tuple = ()

    def do_GET(self):                                   # noqa: N802  (http.server's spelling)
        if not host_allowed(self.headers.get("Host"), self.allow_hosts):
            self.send_error(403, "unknown Host")
            return
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.allow_origins:
            self.send_error(403, "origin not allowed")
            return
        if self.path.rstrip("/") not in ("", "/info"):
            self.send_error(404, "only /info")
            return
        body = json.dumps(report(), separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if origin is not None:                          # only ever an origin on the explicit list
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        """Quiet by default: this is polled every time someone opens the build screen."""
        return


def make_handler(allow_origins=(), allow_hosts=()):
    return type("_NodeHandler", (_Handler,), {"allow_origins": tuple(allow_origins),
                                              "allow_hosts": tuple(allow_hosts)})


def bind_host(host: str | None, lan: bool) -> str:
    """--host wins; else --lan means every interface; else loopback."""
    return host or (LAN_HOST if lan else DEFAULT_HOST)


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, allow_origins=(), allow_hosts=()) -> None:
    srv = ThreadingHTTPServer((host, port), make_handler(allow_origins, allow_hosts))
    where = f"{socket.gethostname()} ({host}:{port})"
    print(f"pollard-node: serving this box's specs on {where}")
    print("             read-only; nothing here can change the machine. ctrl-c to stop.")
    if host in ("127.0.0.1", "localhost", "::1"):
        print("             loopback only -- other boxes cannot see it. `pollard-node --lan` to join a cluster.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=None,
                    help=f"interface to bind (default {DEFAULT_HOST}: this machine only)")
    ap.add_argument("--lan", action="store_true",
                    help=f"bind every interface ({LAN_HOST}) so Studio on another box can count this one")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--allow-origin", action="append", default=[], metavar="ORIGIN",
                    help="a web origin allowed to read /info from a browser (e.g. http://studio.lan:8000); "
                         "repeatable. None by default -- Studio reads it server-side and needs no CORS")
    ap.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                    help="an extra Host name to answer to (a DNS alias of this box); repeatable")
    ap.add_argument("--once", action="store_true",
                    help="print the report and exit, instead of serving")
    a = ap.parse_args()
    if a.once:
        print(json.dumps(report(), indent=1))
        return
    serve(bind_host(a.host, a.lan), a.port, a.allow_origin, a.allow_host)


if __name__ == "__main__":
    main()
