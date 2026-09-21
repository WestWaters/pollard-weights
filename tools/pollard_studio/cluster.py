"""cluster -- what hardware is actually available, here and on every linked box.

Pollard is not capped on model size; the hardware is. Once boxes are pooled over RPC the budget
is the POOL, not this machine, and a build that fits nowhere on its own becomes possible. The
panel that decides a memory target therefore has to show the pool, and it has to show it the way
people actually reason about it: a combined figure, and then each box separately, because "96 GB"
means something very different as one 96 GB box than as six 16 GB boxes.

Nothing here is estimated. Local memory is read from the OS, local VRAM from nvidia-smi, and a
peer's memory from the peer itself over ggml's own RPC protocol -- the same protocol llama.cpp
uses, so if this can read it, a build can use it.

A note on unified memory: on Apple silicon the GPU has no separate pool, it shares system RAM.
Reporting a "VRAM" figure there would double-count the same bytes, so unified machines report
one number and say so.
"""

from __future__ import annotations

import concurrent.futures
import os
import pathlib
import shutil
import socket
import struct
import subprocess
import time
import sys

# ── ggml RPC wire protocol ──────────────────────────────────────────────────────────────────────
# ggml/src/ggml-rpc/ggml-rpc.cpp. A request is: uint8 cmd, uint64 input_size, input bytes.
# A response is: uint64 output_size, output bytes.
CMD_GET_DEVICE_MEMORY = 11
CMD_HELLO = 14
CMD_DEVICE_COUNT = 15
CONN_CAPS_SIZE = 24          # transport.h: RPC_CONN_CAPS_SIZE
PROTO_MAJOR = 6              # ggml-rpc.h: RPC_PROTO_MAJOR_VERSION
DEFAULT_RPC_PORT = 50052


def _round(b: float | int | None) -> float | None:
    return None if not b else round(b / 1e9, 1)


def _exchange(sock: socket.socket, cmd: int, payload: bytes, expect: int) -> bytes | None:
    """One request/response. Returns None if the peer framed it differently than we expect."""
    sock.sendall(bytes([cmd]) + struct.pack("<Q", len(payload)) + payload)
    head = _recv_exactly(sock, 8)
    if head is None:
        return None
    (out_size,) = struct.unpack("<Q", head)
    if out_size != expect:
        return None
    return _recv_exactly(sock, expect)


def _recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


def parse_endpoint(ep: str) -> tuple[str, int]:
    """'host:port' -> (host, port); a bare host gets the default RPC port.

    IPv6 in brackets is handled, because '[::1]:50052' splits wrong on a naive rsplit.
    A trailing '/agentport' is stripped here and read by `agent_port`.
    """
    ep = ep.split("/", 1)[0].strip() if "/" in ep else ep.strip()
    if ep.startswith("["):
        host, _, rest = ep[1:].partition("]")
        port = int(rest.lstrip(":")) if rest.lstrip(":") else DEFAULT_RPC_PORT
        return host, port
    if ep.count(":") == 1:
        host, _, port = ep.partition(":")
        return host, int(port) if port.isdigit() else DEFAULT_RPC_PORT
    return ep, DEFAULT_RPC_PORT


def agent_port(ep: str) -> int | None:
    """An endpoint may name its agent port explicitly: 'host:50052/50053'.

    Needed whenever the two do not sit on the same host:port pair from Studio's side -- two SSH
    tunnels to one box being the usual case. Without it the box is reached twice and counted
    twice, and a single 16 GB card reads as 32.
    """
    if "/" not in ep:
        return None
    tail = ep.rsplit("/", 1)[1].strip()
    return int(tail) if tail.isdigit() else None


def probe(endpoint: str, timeout: float = 1.5, expect: int | None = None) -> dict:
    """Ask one ggml-rpc-server what it is and how much memory it has.

    Degrades honestly: a box that answers TCP but not the protocol is reported as reachable with
    unknown memory, rather than guessed at or dropped.
    """
    host, port = parse_endpoint(endpoint)
    out = {"endpoint": f"{host}:{port}", "host": host, "port": port,
           "reachable": False, "protocol": None, "devices": [],
           "total_gb": None, "free_gb": None, "note": "", "mismatch": False}
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError as e:
        out["note"] = e.strerror or str(e)
        return out
    out["reachable"] = True
    with sock:
        sock.settimeout(timeout)
        try:
            rsp = _exchange(sock, CMD_HELLO, b"\x00" * CONN_CAPS_SIZE, 4 + CONN_CAPS_SIZE)
            if not rsp:
                out["note"] = "listening, but did not answer the ggml RPC handshake"
                return out
            major, minor, patch = rsp[0], rsp[1], rsp[2]
            out["protocol"] = f"{major}.{minor}.{patch}"
            # Compare against what the LOCAL BINARY actually speaks, not what its sources
            # declare. A build older than its own headers is normal, and warning about a
            # mismatch that exists only on paper sends people off to rebuild for nothing.
            want = expect if expect is not None else PROTO_MAJOR
            if major != want:
                # Report it, but still TRY to read memory: a peer one protocol behind usually
                # answers the memory query identically, and calling a box that is plainly up
                # "offline" is worse than a caveat.
                out["mismatch"] = True
                out["note"] = (f"speaks RPC {major}.{minor}, this machine's llama.cpp speaks "
                               f"{want}.x — rebuild whichever is older before pooling them")

            rsp = _exchange(sock, CMD_DEVICE_COUNT, b"", 4)
            count = struct.unpack("<I", rsp)[0] if rsp else 1

            total = free = 0
            for dev in range(min(count, 16)):
                rsp = _exchange(sock, CMD_GET_DEVICE_MEMORY, struct.pack("<I", dev), 16)
                if not rsp:
                    break
                f, t = struct.unpack("<QQ", rsp)
                if not t:
                    continue        # a backend with no memory of its own (the CPU device)
                out["devices"].append({"device": dev, "free_gb": _round(f), "total_gb": _round(t)})
                total += t
                free += f
            out["total_gb"] = _round(total)
            out["free_gb"] = _round(free)
            if not out["devices"]:
                out["note"] = "connected, but reported no devices"
        except (OSError, struct.error) as e:
            out["note"] = f"handshake failed: {e}"
    return out


# ── the pollard-node agent ──────────────────────────────────────────────────────────────────────
#: RPC gives device memory and nothing else. A box running pollard-node also answers for its
#: HOST -- RAM, cards, free disk -- which is what a cluster panel actually wants to show.
NODE_PORT = 50053


def probe_agent(host: str, port: int = NODE_PORT, timeout: float = 1.0) -> dict | None:
    """Ask a peer's pollard-node what the machine is. None when it is not running one."""
    import json
    import urllib.error
    import urllib.request
    url = f"http://{_bracket(host)}:{port}/info"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:   # noqa: S310 (fixed scheme)
            if r.status != 200:
                return None
            got = json.loads(r.read(64_000).decode())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    return got if isinstance(got, dict) and got.get("agent") == "pollard-node" else None


def _bracket(host: str) -> str:
    """IPv6 literals need brackets inside a URL."""
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


# ── this machine ────────────────────────────────────────────────────────────────────────────────

def _phys_ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, AttributeError, OSError):
        pass
    try:                                              # Windows
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = _MS()
        st.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return int(st.ullTotalPhys)
    except Exception:
        return None


def _nvidia_vram_gb() -> tuple[float | None, list]:
    """Total VRAM and the card names, from nvidia-smi. (None, []) when there is no NVIDIA card."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6)
    except (OSError, subprocess.SubprocessError):
        return None, []
    if out.returncode != 0:
        return None, []
    cards, total = [], 0.0
    for line in out.stdout.strip().splitlines():
        name, _, mib = line.partition(",")
        try:
            # nvidia-smi reports MiB. Everything else here is decimal GB (RPC returns bytes),
            # so convert rather than divide by 1024 -- otherwise the same card reads as 15.9 in
            # one place and 17.1 in another and neither number can be trusted.
            gb = round(float(mib.strip()) * 1048576 / 1e9, 1)
        except ValueError:
            continue
        cards.append({"name": name.strip(), "gb": gb})
        total += gb
    return (round(total, 1) if cards else None), cards


def local(home: str | None = None) -> dict:
    """This box, read from the OS."""
    ram = _phys_ram_bytes()
    ram_gb = _round(ram)
    unified = sys.platform == "darwin"                 # Apple silicon GPU shares system RAM
    vram_gb, cards = (None, []) if unified else _nvidia_vram_gb()
    try:
        disk_gb = round(shutil.disk_usage(home or ".").free / 1e9, 1)
    except OSError:
        disk_gb = None
    return {
        "endpoint": "local", "host": socket.gethostname(), "reachable": True,
        "role": "this machine", "platform": sys.platform, "cpus": os.cpu_count(),
        "ram_gb": ram_gb, "vram_gb": vram_gb, "cards": cards, "disk_free_gb": disk_gb,
        "unified": unified,
        # what a build can actually spend here: unified machines must not count RAM twice
        "usable_gb": ram_gb if unified else (vram_gb or ram_gb),
        "note": "unified memory — the GPU shares system RAM" if unified else "",
    }


# ── the pool ────────────────────────────────────────────────────────────────────────────────────

def survey(endpoints: str | list | None = None, timeout: float = 1.5,
           home: str | None = None) -> dict:
    """This machine plus every peer, with a combined figure and each box on its own.

    `endpoints` may be the comma-separated string the UI holds. When it is empty the environment
    is consulted, so a pool configured outside Studio still shows up.
    """
    if endpoints is None or endpoints == "":
        endpoints = os.environ.get("POLLARD_RPC", "")
    eps = ([e.strip() for e in endpoints.split(",")] if isinstance(endpoints, str)
           else list(endpoints))
    eps = [e for e in eps if e]

    here = local(home)
    nodes = [here]
    if eps:
        # measured once, so ten peers do not each start a server to find out
        expect = measured_proto_major() or header_proto_major(rpc_binary())

        def _one(ep: str) -> dict:
            got = probe(ep, timeout, expect=expect)
            got["role"] = "rpc peer"
            got["usable_gb"] = got.get("total_gb")
            # if the box also runs pollard-node, it can answer for the MACHINE, not just the
            # devices -- that is what turns a device list into a cluster
            # Look for the agent on its own port, and failing that on the port that was listed:
            # behind an SSH tunnel the agent is reached on whatever local port was forwarded, so
            # insisting on 50053 would never find a box linked that way.
            # an explicitly named agent port wins; otherwise look on 50053, then on the port
            # that was listed (a tunnel forwards the agent to some arbitrary local port)
            named = agent_port(ep)
            agent = probe_agent(got["host"], named, timeout) if named else None
            if agent is None and named is None:
                agent = probe_agent(got["host"], NODE_PORT, timeout)
                if agent is None and got["port"] != NODE_PORT:
                    agent = probe_agent(got["host"], got["port"], timeout)
            # Serving means RPC actually answered with devices a build can place weights in.
            # An open TCP port is not that -- the agent's own port answers a connection too.
            got["serving"] = bool(got.get("devices"))
            if agent:
                got["agent"] = True
                # A box is part of the cluster if it can SAY what it is. Requiring
                # ggml-rpc-server to be up as well meant a build box that was plainly online
                # showed as offline and contributed nothing -- the opposite of useful.
                got["reachable"] = True
                if not got["serving"]:
                    # It is a build box either way -- Pollard runs on it directly. What it is
                    # not, yet, is a peer that can hold PART of one model being split across
                    # machines. Saying "cannot place weights here" was simply wrong.
                    got["note"] = ("builds on its own; start ggml-rpc-server to also host part "
                                   "of a model too big for any single box")
                got["host"] = agent.get("host") or got["host"]
                got["platform"] = agent.get("platform")
                got["cpus"] = agent.get("cpus")
                got["ram_gb"] = agent.get("ram_gb")
                got["disk_free_gb"] = agent.get("disk_free_gb")
                got["cards"] = agent.get("cards") or []
                got["unified"] = agent.get("unified", False)
                # the builds that box holds, so a cluster is not a list of machines you cannot
                # pick a model from
                got["models"] = agent.get("models") or []
                got["workspace"] = agent.get("workspace") or ""
                # trust the peer's OWN accelerator figure when RPC could not read one
                if not got.get("total_gb"):
                    got["total_gb"] = agent.get("usable_gb")
                    got["usable_gb"] = agent.get("usable_gb")
            else:
                got["agent"] = False
            return got

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(eps), 16)) as pool:
            nodes += _merge_same_box(list(pool.map(_one, eps)))

    online = [n for n in nodes if n.get("reachable")]
    usable = [n["usable_gb"] for n in online if n.get("usable_gb")]

    # The cluster AS ONE MACHINE: what the whole pool brings, by kind of resource. A build is
    # bounded by accelerator memory, so that is the headline; RAM and disk are what the boxes
    # have to stage and hold the work.
    ram = vram = disk = 0.0
    devices = 0
    ram_known = vram_known = disk_known = False
    for n in online:
        if n.get("role") == "this machine":
            if n.get("ram_gb"):
                ram += n["ram_gb"]; ram_known = True
            if n.get("unified"):
                # one pool of bytes serving both roles -- count it once, as accelerator too
                if n.get("ram_gb"):
                    vram += n["ram_gb"]; vram_known = True
                devices += 1
            elif n.get("vram_gb"):
                vram += n["vram_gb"]; vram_known = True
                devices += len(n.get("cards") or []) or 1
            if n.get("disk_free_gb"):
                disk += n["disk_free_gb"]; disk_known = True
        else:
            # Over RPC a peer reports the memory of each BACKEND DEVICE it exposes -- precisely
            # what a build can place weights in, but nothing about the host. A box also running
            # pollard-node answers for the machine, and then it counts in every total.
            if n.get("total_gb"):
                vram += n["total_gb"]; vram_known = True
            devices += len(n.get("devices") or []) or (1 if n.get("total_gb") else 0)

            if n.get("agent"):
                if n.get("ram_gb"):
                    ram += n["ram_gb"]; ram_known = True
                if n.get("disk_free_gb"):
                    disk += n["disk_free_gb"]; disk_known = True

    return {
        "nodes": nodes,
        "configured": len(eps),
        "online": len(online) - 1,                      # peers only; this box is always here
        "boxes": len(online),
        "devices": devices,
        "offline": [n["endpoint"] for n in nodes if not n.get("reachable")],
        "combined_gb": round(sum(usable), 1) if usable else None,
        "largest_single_gb": round(max(usable), 1) if usable else None,
        # the cluster totalled up, by resource
        "total_ram_gb": round(ram, 1) if ram_known else None,
        "total_vram_gb": round(vram, 1) if vram_known else None,
        "total_disk_gb": round(disk, 1) if disk_known else None,
        # only peers WITHOUT the agent leave a hole: they answer for devices, not for the host
        "ram_partial": any(n.get("role") == "rpc peer" and not n.get("agent") for n in online),
        "disk_partial": any(n.get("role") == "rpc peer" and not n.get("agent") for n in online),
        "agents": sum(1 for n in online if n.get("agent")),
        # boxes a build can actually place weights in: this one, plus every RPC peer answering
        "serving": 1 + sum(1 for n in online if n.get("role") == "rpc peer" and n.get("serving")),
        "no_agent": [n["endpoint"] for n in online
                     if n.get("role") == "rpc peer" and not n.get("agent")],
        "pooled": len(online) > 1,
    }


def _merge_same_box(peers: list) -> list:
    """Fold entries that are the SAME machine reached by more than one endpoint.

    It happens easily: one SSH tunnel for the RPC port and another for the agent, or a box
    listed twice by address and by name. Left alone it double-counts a single 16 GB card as
    32 GB, and a memory target built on that figure does not fit anything.

    Identity comes from the agent's hostname when a box is running one; without that there is no
    way to tell two endpoints apart, so they are left as they are.
    """
    out: list = []
    by_host: dict = {}
    for n in peers:
        key = n.get("host") if n.get("agent") else None
        if not key or key not in by_host:
            if key:
                by_host[key] = n
            out.append(n)
            continue
        first = by_host[key]
        # keep the measured device memory, wherever it came from
        if n.get("devices") and not first.get("devices"):
            first["devices"] = n["devices"]
            first["total_gb"] = n.get("total_gb")
            first["free_gb"] = n.get("free_gb")
            first["usable_gb"] = n.get("total_gb")
            first["protocol"] = n.get("protocol")
            first["serving"] = True
            first["note"] = ""
        first["endpoint"] = f"{first['endpoint']} + {n['endpoint']}"
    return out


def discover(port: int = DEFAULT_RPC_PORT, timeout: float = 0.25,
             subnet: str | None = None) -> dict:
    """Sweep the local /24 for ggml-rpc-servers.

    Deliberately NOT automatic. Scanning a network is something a user asks for, not something a
    UI does on its own -- on a corporate or shared network it is the kind of traffic that gets
    noticed. Only the one RPC port is touched.
    """
    base = subnet or _local_subnet()
    if not base:
        return {"found": [], "scanned": 0, "subnet": None,
                "note": "could not work out a local subnet to scan"}
    hosts = [f"{base}.{i}" for i in range(1, 255)]

    def _open(h: str) -> str | None:
        try:
            with socket.create_connection((h, port), timeout=timeout):
                return h
        except OSError:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
        live = [h for h in pool.map(_open, hosts) if h]
    found = [probe(f"{h}:{port}") for h in live]
    return {"found": [f for f in found if f["reachable"]],
            "scanned": len(hosts), "subnet": base, "note": ""}


def _local_subnet() -> str | None:
    """The first three octets of this machine's LAN address, without sending anything."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))          # TEST-NET-1: routable-looking, never answers
        ip = s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()
    if ip.startswith("127.") or ip.count(".") != 3:
        return None
    return ip.rsplit(".", 1)[0]


# ── keeping ggml-rpc-server out of the user's way ───────────────────────────────────────────────
# Three things go wrong with it, and all three are silent until a build dies:
#
#   1. It is not running, so a box that is plainly online cannot take weights.
#   2. It binds 127.0.0.1 BY DEFAULT, so a server that looks fine locally is invisible to the
#      cluster -- the most common "why can't it see my box" there is.
#   3. Its protocol version is compiled in. Pool a peer built from older sources and llama.cpp
#      refuses on the major version, after you have waited for the model to load.
#
# Studio should answer all three before a build starts, not after.

#: where a llama.cpp build puts it, relative to a checkout
_RPC_RELATIVE = (
    "build/bin/ggml-rpc-server", "build/bin/rpc-server",
    "build/bin/Release/ggml-rpc-server.exe", "build/bin/ggml-rpc-server.exe",
    "bin/ggml-rpc-server", "ggml-rpc-server",
)


def rpc_binary(extra: str | None = None) -> pathlib.Path | None:
    """Find ggml-rpc-server without being told where it is.

    Looked for in this order: an explicit override, POLLARD_RPC_BIN, whatever is on PATH, then
    the llama.cpp checkouts Pollard actually uses.
    """
    import shutil as _sh
    cands: list[pathlib.Path] = []
    for raw in (extra, os.environ.get("POLLARD_RPC_BIN")):
        if raw:
            cands.append(pathlib.Path(raw).expanduser())
    for name in ("ggml-rpc-server", "rpc-server"):
        found = _sh.which(name)
        if found:
            cands.append(pathlib.Path(found))
    roots = [pathlib.Path(os.environ["POLLARD_REPO"]).expanduser()] if os.environ.get("POLLARD_REPO") \
        else []
    roots += [pathlib.Path.home() / "Desktop/Pollard-Weights", pathlib.Path.home() / "pollard",
              pathlib.Path("C:/pollard/pw"), pathlib.Path.home() / "llama.cpp"]
    for root in roots:
        for sub in ("runtime/llama.cpp", "llama.cpp", "."):
            for rel in _RPC_RELATIVE:
                cands.append(root / sub / rel)
    for c in cands:
        try:
            if c.is_file() and os.access(c, os.X_OK):
                return c
        except OSError:
            continue
    return None


def rpc_staleness(binary: pathlib.Path) -> dict:
    """Is this binary older than the sources it was built from?

    A stale build is how a protocol mismatch happens: the checkout moved on, the binary did not,
    and nothing says so until a peer is refused.
    """
    out = {"stale": False, "built": None, "source": None, "source_seen": False}
    try:
        built = binary.stat().st_mtime
    except OSError:
        return out
    out["built"] = built
    # walk up out of build/bin to the checkout, then look at the RPC sources themselves
    for up in binary.parents:
        src = up / "ggml/src/ggml-rpc/ggml-rpc.cpp"
        hdr = up / "ggml/include/ggml-rpc.h"
        if src.exists():
            newest = max(src.stat().st_mtime, hdr.stat().st_mtime if hdr.exists() else 0)
            out["source"] = newest
            out["source_seen"] = True
            out["stale"] = newest > built
            break
    return out


_MEASURED: dict[str, int | None] = {}


def measured_proto_major(binary: pathlib.Path | None = None, timeout: float = 25.0) -> int | None:
    """What the local binary ACTUALLY speaks, by starting it and asking.

    The header says what the SOURCES speak, and a binary older than its sources does not. Warning
    a user about a mismatch that only exists on paper -- while the two machines agree perfectly --
    would send them off to rebuild for nothing, so the number that matters is measured.
    """
    binary = binary or rpc_binary()
    if binary is None:
        return None
    key = str(binary)
    if key in _MEASURED:
        return _MEASURED[key]
    got = None
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    proc = None
    try:
        proc = subprocess.Popen([str(binary), "-H", "127.0.0.1", "-p", str(port)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            d = probe(f"127.0.0.1:{port}", timeout=0.5)
            if d["protocol"]:
                got = int(d["protocol"].split(".")[0])
                break
            if proc.poll() is not None:
                break
            time.sleep(0.4)
    except OSError:
        got = None
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    _MEASURED[key] = got
    return got


def header_proto_major(binary: pathlib.Path | None = None) -> int | None:
    """What protocol the SOURCES next to this binary declare."""
    if binary is not None:
        for up in binary.parents:
            hdr = up / "ggml/include/ggml-rpc.h"
            if hdr.exists():
                for line in hdr.read_text(errors="ignore").splitlines():
                    if "RPC_PROTO_MAJOR_VERSION" in line:
                        got = line.split()[-1].strip()
                        if got.isdigit():
                            return int(got)
                break
    return None


def rpc_status(port: int = DEFAULT_RPC_PORT, extra_bin: str | None = None) -> dict:
    """Everything Studio needs to keep the user out of rpc-server's way, on THIS box."""
    binary = rpc_binary(extra_bin)
    out: dict = {
        "binary": str(binary) if binary else None,
        "found": binary is not None,
        "running": False, "port": port, "protocol": None,
        "listening_everywhere": None, "stale": False, "expects": None,
        "problems": [], "fix": None,
    }
    if binary:
        st = rpc_staleness(binary)
        out["stale"] = st["stale"]
        out["expects"] = measured_proto_major(binary) or header_proto_major(binary)

    here = probe(f"127.0.0.1:{port}", timeout=0.6)
    out["running"] = bool(here.get("devices")) or here["protocol"] is not None
    out["protocol"] = here["protocol"]

    if out["running"]:
        # bound to loopback only? then no other box can reach it, however healthy it looks here
        out["listening_everywhere"] = _reachable_off_box(port)
        if out["listening_everywhere"] is False:
            out["problems"].append(
                "ggml-rpc-server is bound to 127.0.0.1, so no other box can reach it. "
                "Restart it with -H 0.0.0.0.")
        want = out["expects"] or PROTO_MAJOR
        if out["protocol"] and int(out["protocol"].split(".")[0]) != want:
            out["problems"].append(
                f"it speaks RPC {out['protocol']} but this llama.cpp expects {want}.x — "
                "a build will refuse to pool it.")
    else:
        out["problems"].append("ggml-rpc-server is not running on this box, so a build cannot "
                               "place weights here.")
    if out["stale"]:
        out["problems"].append("the binary is older than the llama.cpp sources beside it — "
                               "rebuild it before pooling, or peers will disagree on the "
                               "protocol.")
    if not out["found"]:
        out["problems"] = ["ggml-rpc-server was not found on this box. Build it in llama.cpp "
                           "(-DLLAMA_BUILD_RPC=ON) or set POLLARD_RPC_BIN."]
    out["fix"] = serve_command(binary, port) if binary else None
    return out


def _reachable_off_box(port: int) -> bool | None:
    """Is the server bound to something other than loopback?

    Asked by connecting to this machine's own LAN address -- the same path another box takes.
    """
    sub = _local_subnet()
    if not sub:
        return None
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))
        mine = s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()
    try:
        with socket.create_connection((mine, port), timeout=0.6):
            return True
    except OSError:
        return False


def serve_command(binary: pathlib.Path | str, port: int = DEFAULT_RPC_PORT,
                  host: str = "0.0.0.0") -> list:
    """The command to run it so the cluster can actually see it.

    -H 0.0.0.0 is the whole point: the default is loopback, which is the single most common
    reason a box that is up does not join.
    """
    return [str(binary), "-H", host, "-p", str(port)]


_SERVER: dict = {"proc": None, "port": None}


def serve_local(port: int = DEFAULT_RPC_PORT, host: str = "0.0.0.0",
                extra_bin: str | None = None, timeout: float = 30.0) -> dict:
    """Start ggml-rpc-server on THIS box so it can hold part of a split model.

    Started with -H 0.0.0.0, because the default is loopback and a loopback server is invisible
    to every other machine -- the single most common reason a box that is plainly up never joins.
    """
    if _SERVER["proc"] is not None and _SERVER["proc"].poll() is None:
        return {"ok": True, "already": True, "port": _SERVER["port"],
                "note": "already serving from Studio"}
    binary = rpc_binary(extra_bin)
    if binary is None:
        return {"ok": False, "error": "ggml-rpc-server was not found on this box. Build it in "
                                      "llama.cpp, or set POLLARD_RPC_BIN to it."}
    live = probe(f"127.0.0.1:{port}", timeout=0.6)
    if live["protocol"]:
        return {"ok": True, "already": True, "port": port,
                "note": "something is already serving on that port"}
    cmd = serve_command(binary, port, host)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        return {"ok": False, "error": f"could not start it: {e}"}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return {"ok": False, "error": f"it exited immediately (code {proc.returncode})"}
        if probe(f"127.0.0.1:{port}", timeout=0.5)["protocol"]:
            _SERVER.update(proc=proc, port=port)
            return {"ok": True, "already": False, "port": port,
                    "binary": str(binary), "command": " ".join(cmd)}
        time.sleep(0.4)
    proc.kill()
    return {"ok": False, "error": "it did not start answering in time"}


def stop_local() -> dict:
    """Stop only the server Studio started. Anything else on the box is left alone."""
    proc = _SERVER["proc"]
    if proc is None or proc.poll() is not None:
        _SERVER.update(proc=None, port=None)
        return {"ok": True, "note": "Studio was not running one"}
    proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    _SERVER.update(proc=None, port=None)
    return {"ok": True, "note": "stopped"}


# ── per-device placement ────────────────────────────────────────────────────────────────────────
# -ngl is ONE number: how many layers leave the CPU. Which device each of those layers lands on is
# -ts, a share per device -- so "a fader per device" is -ts, with -ngl as the master.
#
# The order of -ts is not obvious and getting it wrong sends the wrong share to the wrong box
# silently. llama.cpp builds its device list as:
#
#     RPC devices, in the order given to --rpc   (src/llama.cpp: "add RPC servers at the front
#     of the list to minimize network transfers")
#     then local discrete GPUs
#     then integrated GPUs, only when there are no discrete ones
#
# so that is the order used here, read out of llama.cpp rather than assumed.

def device_order(surv: dict) -> list:
    """Every device a build can place layers on, in llama.cpp's own -ts order."""
    out: list = []
    peers = [n for n in surv.get("nodes", [])
             if n.get("role") == "rpc peer" and n.get("serving")]
    for n in peers:
        for d in (n.get("devices") or []):
            out.append({
                "key": f"{n['endpoint']}#{d['device']}",
                "host": n.get("host") or n["endpoint"],
                "label": f"{n.get('host') or n['endpoint']}",
                "detail": f"rpc dev {d['device']}",
                "gb": d.get("total_gb"),
                "free_gb": d.get("free_gb"),
                "remote": True,
            })
    here = next((n for n in surv.get("nodes", []) if n.get("role") == "this machine"), None)
    if here:
        cards = here.get("cards") or []
        if cards:
            for i, c in enumerate(cards):
                out.append({"key": f"local#{i}", "host": here.get("host", "local"),
                            "label": c.get("name") or f"gpu {i}", "detail": "local",
                            "gb": c.get("gb"), "free_gb": None, "remote": False})
        elif here.get("unified"):
            out.append({"key": "local#0", "host": here.get("host", "local"),
                        "label": "unified memory", "detail": "local",
                        "gb": here.get("ram_gb"), "free_gb": None, "remote": False})
    return out


def tensor_split(shares: dict, order: list) -> str | None:
    """Build the -ts string from a share per device, in the order llama.cpp expects.

    A device left at 0 gets no layers at all -- which is how you keep a build off a machine
    someone else is using, without taking it out of the pool.
    """
    if not order:
        return None
    vals = [max(0.0, float(shares.get(d["key"], 0) or 0)) for d in order]
    if not any(vals):
        return None
    return ",".join(f"{v:g}" for v in vals)


def placement_args(shares: dict, order: list, ngl: int | None = None) -> list:
    """The llama.cpp arguments for this placement, ready to append verbatim."""
    out: list = []
    if ngl is not None and str(ngl) != "":
        out += ["-ngl", str(int(ngl))]
    ts = tensor_split(shares, order)
    if ts:
        out += ["-ts", ts]
    return out
