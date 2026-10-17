#!/usr/bin/env python3
"""One way to stand up llama-server (or pick -ngl for any llama.cpp tool) for a measurement: pollard-taskeval,
pollard-toolcall and pollard-verify all serve a rung -- and the f16 it is compared against -- through this.

What it fixes (the LFM2.5-24B-A2B release, 2026-10-16): the tools handed `-ngl auto` straight to llama-server with its
default four slots. On the 16 GB box the 48 GB f16 reference loaded with its experts on the CPU, paged off disk at
0.75 tok/s for 92 minutes and then died without a word in its log, and lm-eval saw "connection refused" -- three
rungs failed their measure that way. So:

  * -ngl is picked from FREE VRAM (`auto` -> the last N blocks that fit, the whole file when it fits; the same
    logic as the box's ngl_calc.py), never left to the server's guess.
  * one slot (-np 1): every measurement is one request at a time, and LFM2 / LFM2-MoE serve wrongly with more than
    one sequence (llama.cpp #29002).
  * the wait for /health scales with the file (a 20 GB load is minutes on a cold disk), and a server that exits or
    never answers is reported WITH the tail of its own log.
  * a GPU launch that fails is retried once on the CPU (-ngl 0).
  * POLLARD_GPU_UNTIL=HH:MM -- after that local time `auto` means the CPU (-ngl 0): a GPU borrowed for the evening
    is handed back by the clock, not by somebody remembering to.
"""
from __future__ import annotations

import contextlib
import os
import re
import shutil
import socket
import subprocess
import time

#: architectures that must run with one sequence (llama.cpp #29002: LFM2's short-conv state is per sequence)
ONE_SEQ_ARCHS = ("lfm2", "lfm2moe")
#: MiB kept free on the GPU (another process, the driver) and for the compute buffers + KV
HEADROOM_MIB, OVERHEAD_MIB = 1536, 1536


def free_vram_mib():
    """Free VRAM on the first NVIDIA GPU in MiB; 0 when there is none (or nvidia-smi is missing)."""
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return int(float(r.stdout.strip().splitlines()[0]))
    except Exception:                                             # noqa: BLE001
        return 0


def _block_bytes(gguf):
    """({block index: bytes}, total bytes) of a GGUF's tensors."""
    from gguf import GGUFReader
    blk, total = {}, 0
    for t in GGUFReader(gguf).tensors:
        total += int(t.n_bytes)
        m = re.match(r"blk\.(\d+)\.", t.name)
        if m:
            blk[int(m.group(1))] = blk.get(int(m.group(1)), 0) + int(t.n_bytes)
    return blk, total


def partial_ngl(gguf, free_mib, headroom_mib=HEADROOM_MIB, overhead_mib=OVERHEAD_MIB, blocks=None):
    """-ngl from FREE VRAM: llama.cpp offloads the LAST N blocks, so sum blk.* bytes from the end until the budget
    is spent. 99 (everything) only when the whole file fits the budget."""
    blk, total = blocks if blocks is not None else _block_bytes(gguf)
    budget = (free_mib - headroom_mib - overhead_mib) * 2 ** 20
    if budget <= 0 or not blk:
        return 0
    if total <= budget:
        return 99
    n, ngl, used = max(blk) + 1, 0, 0
    for i in range(n - 1, -1, -1):
        if used + blk.get(i, 0) > budget or ngl >= n - 1:
            break
        used += blk.get(i, 0)
        ngl += 1
    return ngl


def gpu_lent(now=None):
    """False once POLLARD_GPU_UNTIL (HH:MM, local time) has passed; True when it is unset. The window is an
    evening's: a deadline from noon on means "until then today", one before noon "until then after midnight"."""
    until = os.environ.get("POLLARD_GPU_UNTIL", "").strip()
    if not until:
        return True
    try:
        h, m = (int(x) for x in until.split(":"))
    except ValueError:
        return True
    t = time.localtime(now if now is not None else time.time())
    cur, end, noon = t.tm_hour * 60 + t.tm_min, h * 60 + m, 12 * 60
    if end >= noon:
        return noon <= cur < end
    return cur >= noon or cur < end


def resolve_ngl(gguf, ngl, free_mib=None):
    """The -ngl to pass for this GGUF: a number as given; `auto` -> partial_ngl on free VRAM (CUDA), 0 when the
    GPU is not lent any more (POLLARD_GPU_UNTIL), and `auto` left to llama.cpp where there is no nvidia-smi
    (Apple unified memory: its own fit is right there)."""
    s = str(ngl).strip().lower()
    if s != "auto":
        return str(ngl)
    if not gpu_lent():
        return "0"
    if free_mib is None:
        if not shutil.which("nvidia-smi"):
            return "auto"
        free_mib = free_vram_mib()
    if not free_mib:
        return "0"
    try:
        return str(partial_ngl(gguf, free_mib))
    except Exception:                                             # noqa: BLE001 -- unreadable: stay on the CPU
        return "0"


def gguf_arch(gguf):
    try:
        from gguf import GGUFReader
        f = GGUFReader(gguf).fields.get("general.architecture")
        return bytes(f.parts[f.data[0]]).decode("utf-8", "replace") if f else ""
    except Exception:                                             # noqa: BLE001
        return ""


def slot_args(gguf=None):
    """-np 1: measurements send one request at a time, and LFM2 / LFM2-MoE (ONE_SEQ_ARCHS) serve wrongly with more
    than one sequence (llama.cpp #29002) -- so one slot for every model, not a per-arch special case."""
    return ["-np", "1"]


def load_timeout_s(gguf, base_s=180, per_gb_s=25):
    """How long to wait for /health: a fixed start-up plus the time to read the file at a slow-disk rate."""
    try:
        gb = os.path.getsize(gguf) / 1e9
    except OSError:
        gb = 0.0
    return int(base_s + per_gb_s * gb)


def free_port(preferred=None):
    """`preferred` if nothing listens there, else any free port (a stale server on 8080 must not eat a run)."""
    if preferred:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", int(preferred)))
                return int(preferred)
            except OSError:
                pass
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def log_tail(path, n=20):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip() for l in f if l.strip()]
    except OSError:
        return f"  (no {path})"
    if not lines:
        return f"  ({path} is empty)"
    return f"  last {min(n, len(lines))} line(s) of {path}:\n" + "\n".join("    " + l for l in lines[-n:])


class ServerFailed(RuntimeError):
    """llama-server exited or never answered /health; the message carries its log tail."""


class Server:
    """A running llama-server: .base (url), .proc, .log_path, .ngl; .alive() / .why() for after-the-fact checks."""

    def __init__(self, proc, base, log_path, ngl):
        self.proc, self.base, self.log_path, self.ngl = proc, base, log_path, ngl

    def alive(self):
        return self.proc.poll() is None

    def why(self):
        rc = self.proc.poll()
        state = "still running" if rc is None else f"exited ({rc})"
        return f"llama-server ({state}, -ngl {self.ngl}) at {self.base}\n{log_tail(self.log_path)}"


def _start(cmd, log_path, port, timeout_s, what, health="/health", popen=subprocess.Popen, sleep=time.sleep,
           probe=None):
    import urllib.request
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    try:
        proc = popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    except OSError as e:
        log.close()
        raise ServerFailed(f"could not start {what}: {e}") from None
    base = f"http://127.0.0.1:{port}"

    def ready():
        if probe is not None:
            return probe(base)
        try:
            with urllib.request.urlopen(base + health, timeout=3) as r:
                return r.status == 200
        except Exception:                                         # noqa: BLE001 -- loading (503) or not up yet
            return False

    t0 = time.time()
    while True:
        if proc.poll() is not None:
            log.close()
            raise ServerFailed(f"{what} exited ({proc.returncode}) before serving\n{log_tail(log_path)}")
        if ready():
            return proc, base, log
        if time.time() - t0 > timeout_s:
            _stop(proc)
            log.close()
            raise ServerFailed(f"{what} not ready after {timeout_s}s\n{log_tail(log_path)}")
        sleep(1)


def _stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:                                         # noqa: BLE001
            proc.kill()


def thread_args(extra=()):
    """-t / -tb from POLLARD_THREADS (a shared box leaves the rest of the CPU to its owner), unless already given."""
    t = os.environ.get("POLLARD_THREADS", "").strip()
    if not t or "-t" in extra or "--threads" in extra:
        return []
    return ["-t", t, "-tb", t]


def server_cmd(binary, gguf, port, ngl, ctx, extra=()):
    return [binary, "-m", gguf, "--host", "127.0.0.1", "--port", str(port), "-ngl", str(ngl), "-c", str(ctx)] \
        + slot_args(gguf) + thread_args(extra) + list(extra)


@contextlib.contextmanager
def serve(gguf, ngl="auto", ctx=4096, port=None, binary=None, extra=(), log_path=None, timeout_s=None,
          retry_cpu=True, out=print, _start_fn=None):
    """llama-server on `gguf` for the duration; yields a Server. -ngl from resolve_ngl; a GPU start that fails is
    retried once with -ngl 0; ServerFailed (with the server's own words) when neither comes up."""
    from pollard_calc import find_llama_bin
    binary = binary or find_llama_bin("llama-server") or shutil.which("llama-server") or "llama-server"
    first = resolve_ngl(gguf, ngl)
    attempts = [first] + (["0"] if retry_cpu and first != "0" else [])
    timeout_s = timeout_s or load_timeout_s(gguf)
    log_path = log_path or os.path.abspath(f"llama-server-{os.path.basename(gguf)}.log")
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    start = _start_fn or _start
    errors = []
    for i, n in enumerate(attempts):
        p = free_port(port)
        lp = log_path if i == 0 else log_path + ".cpu"
        cmd = server_cmd(binary, gguf, p, n, ctx, extra)
        out(f"   serving: {os.path.basename(binary)} -m {os.path.basename(gguf)} -ngl {n} -np 1 -c {ctx} "
            f"(port {p}, wait <= {timeout_s}s, log {lp})")
        try:
            proc, base, log = start(cmd, lp, p, timeout_s, f"llama-server on {os.path.basename(gguf)} -ngl {n}")
        except ServerFailed as e:
            errors.append(str(e))
            out(f"   (!) {e}" + ("\n   retrying on the CPU (-ngl 0)" if i + 1 < len(attempts) else ""))
            continue
        srv = Server(proc, base, lp, n)
        try:
            yield srv
        finally:
            _stop(proc)
            try:
                log.close()
            except Exception:                                     # noqa: BLE001
                pass
        return
    raise ServerFailed("llama-server would not come up:\n" + "\n".join(errors))
