#!/usr/bin/env python3
"""Talk to any runtime, so a Pollard build is not locked to one way of running it.

Pollard emits six lanes. Tying the UI to llama.cpp would mean five of them could be built and not
tried, and would shut out anyone whose model lives behind an API rather than on their disk.

Two kinds of runtime, one interface:

    LOCAL   a server this machine starts and owns -- llama.cpp, ik_llama, vLLM, MLX.
            Started once per build and reused; loading a large checkpoint per message is the
            difference between a usable chat and a useless one.
    REMOTE  an endpoint that already exists -- OpenRouter, or anything OpenAI-compatible,
            including a vLLM someone else is running. No process, just a key.

Every local runtime here speaks an OpenAI-compatible or llama.cpp-compatible HTTP API, so one
client covers all of them and adding the next one is a table entry, not a new code path.

    python -m pollard_studio.runtimes                       # what is available here
    python -m pollard_studio.runtimes llama.cpp model.gguf "hello"
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LOCAL, REMOTE = "local", "remote"


# ── the table: adding a runtime is an entry here ────────────────────────────────────────────────
SPECS: dict[str, dict] = {
    "llama.cpp": {
        "kind": LOCAL, "exe": "llama-server", "accepts": ["gguf"],
        "note": "stock llama.cpp — loads anywhere, cannot load trellis atoms",
        "args": lambda m, port, ngl, ctx: ["-m", m, "--port", str(port), "--host", "127.0.0.1",
                                           "-c", str(ctx), "-ngl", str(ngl), "--log-disable"],
        "api": "llama",
    },
    "ik_llama": {
        "kind": LOCAL, "exe": "llama-server", "env_exe": "IK_LLAMA_SERVER", "accepts": ["gguf"],
        "path_marker": "ik_llama",
        "note": "ik_llama fork — required for IQ*_KT trellis atoms",
        "args": lambda m, port, ngl, ctx: ["-m", m, "--port", str(port), "--host", "127.0.0.1",
                                           "-c", str(ctx), "-ngl", str(ngl), "--log-disable"],
        "api": "llama",
    },
    "vllm": {
        "kind": LOCAL, "exe": "vllm", "accepts": ["safetensors"],
        "note": "serves GPTQ / MX / compressed-tensors checkpoints",
        "args": lambda m, port, ngl, ctx: ["serve", m, "--port", str(port),
                                           "--max-model-len", str(ctx)],
        "api": "openai", "health": "/v1/models", "boot": 900,
    },
    "mlx": {
        "kind": LOCAL, "exe": "mlx_lm.server", "accepts": ["safetensors"],
        "note": "Apple silicon — MLX checkpoints",
        "args": lambda m, port, ngl, ctx: ["--model", m, "--port", str(port)],
        "api": "openai", "health": "/v1/models",
    },
    "openrouter": {
        "kind": REMOTE, "base": "https://openrouter.ai/api/v1",
        "key_env": ["OPENROUTER_API_KEY", "OPENROUTER_KEY"],
        "note": "hosted — compare your build against the model it came from",
        "api": "openai",
    },
    "openai-compatible": {
        "kind": REMOTE, "base_env": ["POLLARD_OPENAI_BASE", "OPENAI_BASE_URL"],
        "key_env": ["POLLARD_OPENAI_KEY", "OPENAI_API_KEY"],
        "note": "any OpenAI-compatible endpoint — a vLLM someone else is running, LM Studio, …",
        "api": "openai",
    },
}


def _exe(spec: dict) -> str | None:
    for var in spec.get("env_exe", []) if isinstance(spec.get("env_exe"), list) else \
               ([spec["env_exe"]] if spec.get("env_exe") else []):
        p = os.environ.get(var)
        if p and Path(p).exists():
            return p
    found = shutil.which(spec["exe"]) if spec.get("exe") else None
    # A fork whose binary has the SAME NAME as the stock one cannot be resolved off PATH: the
    # stock llama-server answers to `which llama-server` and then refuses every trellis atom,
    # which reads as "the build is broken" rather than "wrong runtime". Only accept a path that
    # actually names the fork; otherwise say it is missing so the caller can set the env var.
    marker = spec.get("path_marker")
    if found and marker and marker not in Path(found).as_posix().lower():
        return None
    return found


def _env_first(names) -> str | None:
    for n in names or []:
        v = os.environ.get(n)
        if v:
            return v
    return None


def available() -> list[dict]:
    """Every runtime, and whether it can actually be used on this machine right now."""
    out = []
    for name, spec in SPECS.items():
        row = {"name": name, "kind": spec["kind"], "note": spec["note"],
               "accepts": spec.get("accepts", []), "ready": False, "why": ""}
        if spec["kind"] == LOCAL:
            path = _exe(spec)
            row["ready"] = bool(path)
            row["why"] = path or f"{spec['exe']} not on PATH"
            if name == "ik_llama" and path and not os.environ.get("IK_LLAMA_SERVER"):
                row["why"] = f"{path} — set IK_LLAMA_SERVER if the fork lives elsewhere"
        else:
            key = _env_first(spec.get("key_env"))
            base = spec.get("base") or _env_first(spec.get("base_env"))
            row["ready"] = bool(key and base)
            row["why"] = (f"{base}" if key and base else
                          "set " + " or ".join(spec.get("key_env", []))
                          + ("" if spec.get("base") else
                             " and " + " or ".join(spec.get("base_env", []))))
        out.append(row)
    return out


def for_build(kind: str, lane: str | None = None) -> list[str]:
    """Which runtimes can host this build, best first.

    A GGUF carrying trellis atoms needs the ik_llama fork; stock llama.cpp will refuse it. That is
    a property of the file, so pollard-ggufcheck stays the authority -- this only orders the
    candidates.
    """
    want = "gguf" if kind == "gguf" else "safetensors"
    local = [n for n, s in SPECS.items()
             if s["kind"] == LOCAL and want in s.get("accepts", [])]
    trellis = want == "gguf" and lane and "KT" in str(lane).upper()
    if trellis:
        local.sort(key=lambda n: n != "ik_llama")          # trellis: the fork first
    remote = [n for n, s in SPECS.items() if s["kind"] == REMOTE]
    # Rank, do not filter. Hiding a runtime the user has not installed yet hides the fact that
    # it is an option at all -- the UI marks what is missing and how to get it.
    ready = {r["name"] for r in available() if r["ready"]}
    cands = local + remote
    # REQUIRED outranks READY. For a trellis build stock llama.cpp is not a fallback -- it cannot
    # open the file at all -- so ranking it above a not-yet-installed ik_llama points the user at
    # the runtime guaranteed to fail and hides the one that works.
    def rank(n):
        return (not (trellis and n == "ik_llama"), n not in ready, cands.index(n))
    return sorted(cands, key=rank)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Runtime:
    """One runtime, local or remote, behind one interface."""

    def __init__(self, name: str = "llama.cpp", ngl: int = 0, ctx: int = 4096):
        if name not in SPECS:
            raise ValueError(f"unknown runtime {name!r}; have {sorted(SPECS)}")
        self.name, self.spec = name, SPECS[name]
        self.ngl, self.ctx = ngl, ctx
        self.proc: subprocess.Popen | None = None
        self.model: str | None = None
        self.port: int | None = None

    # -- lifecycle -------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        if self.spec["kind"] == REMOTE:
            return bool(_env_first(self.spec.get("key_env")))
        return self.proc is not None and self.proc.poll() is None

    def ensure(self, model: str, timeout: float | None = None) -> dict:
        if self.spec["kind"] == REMOTE:
            base = self.spec.get("base") or _env_first(self.spec.get("base_env"))
            key = _env_first(self.spec.get("key_env"))
            if not (base and key):
                return {"ok": False, "error": f"{self.name}: "
                        + "set " + " or ".join(self.spec.get("key_env", []))}
            self.model = model
            return {"ok": True, "remote": True, "base": base}

        if self.running and self.model == model:
            return {"ok": True, "port": self.port, "reused": True}
        self.stop()

        exe = _exe(self.spec)
        if not exe:
            return {"ok": False, "error": f"{self.spec['exe']} not on PATH"}
        if not Path(model).exists():
            return {"ok": False, "error": f"no such build: {model}"}

        self.port = _free_port()
        argv = [exe] + self.spec["args"](model, self.port, self.ngl, self.ctx)
        # NOT DEVNULL. A server that dies on load says exactly why -- "invalid ggml type 153",
        # "failed to allocate", a missing dylib -- and discarding it leaves the UI showing
        # "exited while loading" with no cause, which looks like the BUILD is broken.
        self._log = Path(tempfile.gettempdir()) / f"pollard-studio-{self.name}.log"
        logf = open(self._log, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        logf.close()
        self.model = model

        health = self.spec.get("health", "/health")
        deadline = time.time() + (timeout or self.spec.get("boot", 300))
        while time.time() < deadline:
            if not self.running:
                return {"ok": False,
                        "error": f"{self.name} exited while loading. {self._why()}"}
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}{health}", timeout=2) as r:
                    if r.status == 200:
                        return {"ok": True, "port": self.port, "reused": False}
            except Exception:
                time.sleep(0.4)
        self.stop()
        return {"ok": False, "error": f"{self.name} did not come up in time"}

    def _why(self, n: int = 6) -> str:
        """The server's own last words, with the one we can name called out.

        A trellis atom in a stock runtime is the common case and it has a specific fix, so it is
        translated rather than handed over as a raw ggml error.
        """
        try:
            lines = [l.rstrip() for l in open(self._log, encoding="utf-8",
                                              errors="replace") if l.strip()]
        except (OSError, AttributeError):
            return "(no server log)"
        blob = "\n".join(lines)
        if "invalid ggml type" in blob or "should be in [0, 43)" in blob:
            return ("This build carries ik_llama-only atoms (IQ*_KT) and stock llama.cpp caps "
                    "ggml types at 42. Switch the runtime to ik_llama, or set IK_LLAMA_SERVER "
                    "to an ik_llama llama-server.")
        return "Last lines: " + " | ".join(lines[-n:]) if lines else "(server log empty)"

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                self.proc.terminate()
            for _ in range(40):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.1)
            else:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    self.proc.kill()
        self.proc = None
        self.model = None
        self.port = None

    # -- generation ------------------------------------------------------------------------------
    def complete(self, prompt: str, n_predict: int = 256, temperature: float = 0.7,
                 timeout: float = 300) -> dict:
        """One completion. Always reports WHY it stopped, because that is a failure mode."""
        try:
            if self.spec["kind"] == REMOTE:
                return self._openai(prompt, n_predict, temperature, timeout,
                                    self.spec.get("base") or _env_first(self.spec.get("base_env")),
                                    {"Authorization": f"Bearer {_env_first(self.spec['key_env'])}"},
                                    self.model)
            if not self.running:
                return {"ok": False, "error": "runtime not running"}
            base = f"http://127.0.0.1:{self.port}"
            if self.spec["api"] == "openai":
                return self._openai(prompt, n_predict, temperature, timeout, base + "/v1",
                                    {}, Path(self.model).name)
            data = _post(f"{base}/completion",
                         {"prompt": prompt, "n_predict": int(n_predict),
                          "temperature": float(temperature), "stream": False}, {}, timeout)
            return {"ok": True, "text": (data.get("content") or "").strip(),
                    "stop_reason": "length" if data.get("stopped_limit") else "stop",
                    "tokens": data.get("tokens_predicted") or 0}
        except urllib.error.HTTPError as e:
            return {"ok": False, "error": f"{self.name} {e.code}: {e.read()[:200].decode('utf8', 'replace')}"}
        except (urllib.error.URLError, TimeoutError) as e:
            return {"ok": False, "error": f"{self.name}: {e}"}

    def _openai(self, prompt, n, temp, timeout, base, headers, model) -> dict:
        data = _post(f"{base}/chat/completions",
                     {"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": int(n), "temperature": float(temp)},
                     headers, timeout)
        ch = (data.get("choices") or [{}])[0]
        return {"ok": True,
                "text": ((ch.get("message") or {}).get("content") or "").strip(),
                # 'length' is the OpenAI spelling of budget exhaustion
                "stop_reason": "length" if ch.get("finish_reason") == "length" else "stop",
                "tokens": (data.get("usage") or {}).get("completion_tokens", 0)}

    def status(self) -> dict:
        return {"runtime": self.name, "kind": self.spec["kind"], "running": self.running,
                "model": self.model, "port": self.port,
                "name": Path(self.model).name if self.model and self.spec["kind"] == LOCAL
                        else self.model}


def main() -> None:
    if len(sys.argv) < 2:
        for r in available():
            mark = "ready" if r["ready"] else "  -  "
            print(f"  [{mark}] {r['name']:19} {r['kind']:7} {r['why'][:56]}")
            print(f"           {r['note']}")
        return
    rt = Runtime(sys.argv[1])
    up = rt.ensure(sys.argv[2])
    if not up["ok"]:
        print("error:", up["error"])
        raise SystemExit(1)
    try:
        print(json.dumps(rt.complete(sys.argv[3] if len(sys.argv) > 3 else "hello", 64), indent=2))
    finally:
        rt.stop()


if __name__ == "__main__":
    main()
