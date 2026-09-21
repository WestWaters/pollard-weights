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

    pollard-node                      # serve on 0.0.0.0:50053
    pollard-node --port 9100 --host 127.0.0.1

It is machine specs on a LAN port, so treat it the way you would any other local service: bind it
to an interface your cluster is actually on. Nothing it serves can change the box.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import cluster

DEFAULT_PORT = 50053
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


class _Handler(BaseHTTPRequestHandler):
    server_version = "pollard-node"

    def do_GET(self):                                   # noqa: N802  (http.server's spelling)
        if self.path.rstrip("/") not in ("", "/info"):
            self.send_error(404, "only /info")
            return
        body = json.dumps(report(), separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        """Quiet by default: this is polled every time someone opens the build screen."""
        return


def serve(host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> None:
    srv = ThreadingHTTPServer((host, port), _Handler)
    where = f"{socket.gethostname()} ({host}:{port})"
    print(f"pollard-node: serving this box's specs on {where}")
    print("             read-only; nothing here can change the machine. ctrl-c to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="0.0.0.0",
                    help="interface to bind (default: every one)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--once", action="store_true",
                    help="print the report and exit, instead of serving")
    a = ap.parse_args()
    if a.once:
        print(json.dumps(report(), indent=1))
        return
    serve(a.host, a.port)


if __name__ == "__main__":
    main()
