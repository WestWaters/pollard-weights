#!/usr/bin/env python3
"""pollard-preflight -- prove every engine step a release will run works on THIS model file, before the hours start.

pollard-smoke answers "is there a converter that registers this architecture" from the model's shape. That is not the
same as "will the engines this run is about to use load this file". The Nemotron-3.5-Bolt release failed four times over
two nights on things a 30-second check would have caught: the box's runtime/llama.cpp predated the architecture's
upstream support, the converter for it lived only in another engine tree, and a model loader disagreed with the file
("expected 496 experts, got 240") -- each found hours in, after calibration and an imatrix.

So before calibration / imatrix / any build, for every step the plan will run:

  1. the binary is resolved exactly as that step resolves it (pollard_calc.find_llama_bin with the model's
     architecture set, the ik_llama build the flagship uses, the converter pollard-convert picks, the MLX engine);
  2. the REAL file is loaded cheaply on the CPU (-ngl 0): llama-perplexity -c 128 --chunks 1 (and llama-imatrix, the
     server's and mtmd's build when they are another tree), llama-quantize --dry-run, mtmd-cli with the mmproj, the
     converter --vocab-only for an HF source, the MLX model class + one shard;
  3. a table step -> binary -> result, and the exact fix when one fails ("runtime/llama.cpp ccdbe29 predates this
     architecture's support; run `pollard-runtime --update`", "converter for <arch> only in <tree>").

A lane whose critical step cannot run is stopped here, before anything long; the other lanes carry on. Results that
passed are cached by (binary, model file) so a second preflight in the same run (the release, then its gguf lane's
`pollard --run`) costs nothing.

    pollard-preflight --gguf model-f16.gguf                    # the GGUF lane's engines on this file
    pollard-preflight --hf <dir> --lanes gguf,mlx              # an HF source: converter + MLX + binaries
    pollard-preflight --gguf m.gguf --mmproj mmproj.gguf --flagship

POLLARD_ENGINE_PREFLIGHT=0 turns it off (pollard --no-engine-preflight for one run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#: what each step does, which lane it belongs to, and the binary it resolves
STEPS = {
    "convert":     ("gguf", None),
    "imatrix":     ("gguf", "llama-imatrix"),
    "quantize":    ("gguf", "llama-quantize"),
    "perplexity":  ("gguf", "llama-perplexity"),
    "server":      ("gguf", "llama-server"),
    "mtmd":        ("gguf", "llama-mtmd-cli"),
    "ik-quantize": ("gguf", "llama-quantize"),
    "mlx":         ("mlx", None),
}
#: a step whose failure does not stop its lane: only the flagship rung needs ik_llama
NON_CRITICAL = {"ik-quantize"}
#: what a model loader prints when it cannot load the file (anything else after a load is not a load failure)
LOAD_FAIL = re.compile(r"error loading model|failed to load model|unable to load model|unknown model architecture|"
                       r"wrong number of tensors|missing tensor|tensor '[^']*' has wrong shape|expected \d+[^\n]*got \d+|"
                       r"GGML_ASSERT|terminate called|Segmentation fault|failed to load|unknown argument|"
                       r"invalid argument|error: ", re.I)
ARCH_FAIL = re.compile(r"unknown model architecture|expected \d+[^\n]*got \d+|wrong number of tensors|missing tensor|"
                       r"has wrong shape|done_getting_tensors", re.I)
LOAD_TIMEOUT = 900
QUICK_TIMEOUT = 120
CACHE_DAYS = 7
TEXT = ("The quick brown fox jumps over the lazy dog while the committee reviews the quarterly report. "
        "Rivers carry sediment to the sea, and over centuries the delta grows into farmland. ") * 40


def disabled():
    """POLLARD_ENGINE_PREFLIGHT=0 turns the preflight off everywhere (pollard --no-engine-preflight for one run)."""
    return os.environ.get("POLLARD_ENGINE_PREFLIGHT", "1").strip().lower() in ("0", "off", "false", "no")


def _threads():
    return str(max(1, int((os.cpu_count() or 4) * 0.6)))                      # leave the machine usable (60/40)


def _run(cmd, timeout=QUICK_TIMEOUT, stdin=None, cpu=False):
    """(exit code, combined output) -- never raises; a missing file or a timeout is a result. `cpu`: the GPU is hidden
    (CUDA_VISIBLE_DEVICES=-1) so even op offload of the 128-token batch never touches a shared GPU (the build box's
    belongs to someone else while it games); the loader is what is tested, not the kernels."""
    env = None
    if cpu and "CUDA_VISIBLE_DEVICES" not in os.environ:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    try:
        r = subprocess.run([str(x) for x in cmd], input=stdin, capture_output=True, text=True, errors="replace",
                           timeout=timeout, stdin=None if stdin is not None else subprocess.DEVNULL, env=env)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"") + (e.stderr or b"")
        return -9, (out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out)) + \
            f"\n(timed out after {timeout}s)"
    except OSError as e:
        return -1, str(e)


def _why(out, n=220):
    """The line of a tool's output that names the failure."""
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    hits = [ln for ln in lines if ARCH_FAIL.search(ln)] or [ln for ln in lines if LOAD_FAIL.search(ln)]
    return ((hits[0] if hits else (lines[-1] if lines else "(no output)")))[-n:]


def tree_of(binary):
    """The engine tree a binary was built in (<tree>/build/bin[/Release]/x), or None."""
    if not binary:
        return None
    d = os.path.dirname(os.path.abspath(binary))
    for _ in range(3):
        if os.path.basename(d).lower() == "build":
            return os.path.dirname(d)
        d = os.path.dirname(d)
    return None


def _tree_rev(tree):
    """'ccdbe29 (2026-10-12)' for a git tree, '' otherwise."""
    if not tree or not os.path.isdir(os.path.join(tree, ".git")):
        return ""
    rc, out = _run(["git", "-C", tree, "log", "-1", "--format=%h %cs"], timeout=20)
    parts = out.split()
    return f"{parts[0]} ({parts[1]})" if rc == 0 and len(parts) >= 2 else ""


def _rel(path):
    try:
        import pollard_calc as PC
        root = PC._repo_root()
        ap = os.path.abspath(str(path))
        return os.path.relpath(ap, root) if ap.startswith(root + os.sep) else str(path)
    except Exception:                                                     # noqa: BLE001
        return str(path)


def _fingerprint(*paths):
    h = hashlib.sha1()
    for p in paths:
        try:
            st = os.stat(p)
            h.update(f"{os.path.realpath(p)}|{st.st_size}|{int(st.st_mtime)}\n".encode())
        except (OSError, TypeError):
            h.update(f"{p}|-\n".encode())
    return h.hexdigest()


def _cache_path():
    home = os.environ.get("POLLARD_HOME") or os.path.expanduser("~/pollard")
    return os.path.join(home, "cache", "engine-preflight.json")


def _cache_get(key):
    """The cached PASS for this (step, binary, model file), or None. Only passes are cached: a failure is re-checked."""
    import time
    try:
        d = json.load(open(_cache_path(), encoding="utf-8"))
        e = d.get(key)
        if e and time.time() - e.get("at", 0) < CACHE_DAYS * 86400:
            return e
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _cache_put(key, detail, **extra):
    import time
    p = _cache_path()
    try:
        d = json.load(open(p, encoding="utf-8"))
        d = d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        d = {}
    d[key] = {"detail": detail, "at": time.time(), **extra}
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
    except OSError:
        pass


# ---- the fix to print ----------------------------------------------------------------------------------------

def _upstream_has(arch):
    """True / False when upstream llama.cpp master's architecture list says, None when unreachable."""
    if not arch:
        return None
    try:
        from pollard_runtime import upstream_archs
        up = upstream_archs()
    except Exception:                                                     # noqa: BLE001
        up = None
    return None if up is None else arch in up


def _trees_that_load(arch, name, skip=None):
    """Other managed engine trees (engines.json + runtime/<name>.lock.json) whose `name` binary knows `arch`."""
    try:
        import pollard_calc as PC
        out = []
        for _n, d in PC.engine_trees():
            b = PC._tree_bin(d, name)
            if b and (not skip or os.path.abspath(d) != os.path.abspath(skip)) and PC.binary_supports_arch(b, arch):
                out.append(d)
        return out
    except Exception:                                                     # noqa: BLE001
        return []


def fix_for_old_binary(binary, arch, name):
    """The exact fix for a binary that cannot load this architecture's file."""
    tree = tree_of(binary)
    where = _rel(tree) if tree else os.path.dirname(str(binary))
    rev = _tree_rev(tree)
    other = _trees_that_load(arch, name, skip=tree)
    up = _upstream_has(arch)
    head = f"{where}{' ' + rev if rev else ''}"
    if up is True or (up is None and not other):
        msg = f"{head} predates this architecture's support ({arch}); run `pollard-runtime --update`"
        if up is None:
            msg += " (upstream unreachable from here -- if upstream lacks it too, it needs a captured patch)"
    elif other:
        msg = (f"{head} cannot load {arch}; only {', '.join(_rel(t) for t in other)} can -- a runtime patch the live "
               "tree lacks: rebase/capture it into runtime-patches/ and run `pollard-runtime --update`")
    else:
        msg = (f"{head} cannot load {arch} and upstream llama.cpp does not carry it either: it needs a captured "
               "runtime patch (runtime-patches/) or the vendor's fork")
    if other and up is True:
        msg += f" (meanwhile {', '.join(_rel(t) for t in other)} loads it)"
    return msg


def fix_for_converter(hf_dir, note):
    """The fix when pollard-convert finds no converter for this model: name the engine tree that has one."""
    try:
        from pollard_convert import converter_registers, model_architectures
        import pollard_calc as PC
        archs = model_architectures(hf_dir)
        for _n, d in PC.engine_trees():
            c = os.path.join(d, "convert_hf_to_gguf.py")
            if os.path.isfile(c) and any(converter_registers(c, a) for a in archs):
                return (f"converter for {', '.join(archs)} only in {_rel(d)}: point POLLARD_CONVERTER at {c} (or bring "
                        "that support into runtime/llama.cpp with `pollard-runtime --update`)")
        if archs:
            return (f"no converter here registers {', '.join(archs)}: run `pollard-runtime --update` (or pollard-onboard "
                    f"--model {hf_dir} --contribute for a new architecture)")
    except Exception:                                                     # noqa: BLE001
        pass
    return note.splitlines()[0] if note else "no converter found"


# ---- the checks ----------------------------------------------------------------------------------------------

def _resolve(name, arch, bin_dir=None):
    """The binary the real step will run: --bin's dir when given, else find_llama_bin(name) with the model's
    architecture set (POLLARD_ARCH), exactly as pollard / pollard release / their children resolve it."""
    if bin_dir:
        exe = os.path.join(bin_dir, name + (".exe" if sys.platform == "win32" else ""))
        return exe if os.path.isfile(exe) else None
    import pollard_calc as PC
    old = os.environ.get("POLLARD_ARCH")
    if arch:
        os.environ["POLLARD_ARCH"] = arch
    try:
        return PC.find_llama_bin(name)
    finally:
        if arch:
            if old is None:
                os.environ.pop("POLLARD_ARCH", None)
            else:
                os.environ["POLLARD_ARCH"] = old


def _arch_ok(binary, arch):
    try:
        import pollard_calc as PC
        return PC.binary_supports_arch(binary, arch)
    except Exception:                                                     # noqa: BLE001
        return True


def _starts(binary):
    rc, out = _run([binary, "--version"], timeout=60)
    if rc == 0 or re.search(r"usage|version|build", out, re.I):
        return True, ""
    return False, f"does not start (exit {rc}): {_why(out)}"


def _load_gguf(binary, gguf, kind, work):
    """(ok, detail) of a real CPU load of `gguf` by llama-perplexity / llama-imatrix (-c 128 --chunks 1)."""
    text = os.path.join(work, "preflight.txt")
    if not os.path.exists(text):
        open(text, "w", encoding="utf-8").write(TEXT)
    cmd = [binary, "-m", gguf, "-f", text, "-c", "128", "--chunks", "1", "-ngl", "0", "-t", _threads(), "-b", "128"]
    if kind == "imatrix":
        cmd += ["-o", os.path.join(work, "preflight.imatrix")]
    rc, out = _run(cmd, timeout=LOAD_TIMEOUT, cpu=True)
    if rc == 0:
        return True, "loads + 1 chunk on CPU (-c 128 --chunks 1 -ngl 0)"
    if rc < 0 or ARCH_FAIL.search(out) or re.search(r"error loading model|failed to load model|unable to load model",
                                                      out, re.I):
        return False, _why(out)
    if re.search(r"^llama_context:|llama_init_from_model|calculating perplexity|compute_imatrix", out, re.M):
        return True, f"loads (exit {rc} after the load: {_why(out, 120)})"   # the file loaded; the run is not the test
    return False, _why(out)


def _quantize_dry(binary, gguf, ftype="Q4_K_M"):
    rc, out = _run([binary, "--dry-run", "--allow-requantize", gguf, ftype], timeout=QUICK_TIMEOUT * 3, cpu=True)
    if rc == 0:
        return True, f"--dry-run {ftype} reads every tensor"
    if re.search(r"unknown argument|invalid argument", out, re.I) and "dry" in out.lower():
        ok, why = _starts(binary)
        return ok, "this build has no --dry-run: starts" + ("" if ok else f" -- {why}")
    return False, _why(out)


def _mtmd(binary, gguf, mmproj):
    # --jinja: the model's own chat template, the way every Pollard llama.cpp step serves it. Without it mtmd-cli falls
    # back to llama.cpp's built-in template list, and a template it does not know (Gemma 4 / Nemotron-3.5-Bolt-E4B)
    # aborts at start-up -- "this custom template is not supported, try using --jinja" -- on a model that runs fine.
    rc, out = _run([binary, "-m", gguf, "--mmproj", mmproj, "--jinja", "-ngl", "0", "-c", "256", "-t", _threads()],
                   timeout=LOAD_TIMEOUT, stdin="/quit\n", cpu=True)
    if rc == 0 and "chat mode" in out.lower():
        return True, "loads the model + mmproj on CPU"
    return False, _why(out)


def _convert(conv, hf_dir, work):
    """Run the converter the release will use with --vocab-only: config + tokenizer + the arch's model class, no
    tensors. -> (ok, detail, gguf architecture it writes or None)."""
    out_f = os.path.join(work, "preflight-vocab.gguf")
    rc, out = _run([sys.executable, conv, hf_dir, "--vocab-only", "--outfile", out_f], timeout=QUICK_TIMEOUT * 5)
    if rc != 0 or not os.path.isfile(out_f):
        return False, _why(out), None
    arch = None
    try:
        from pollard_ggufcompat import read_header
        arch = read_header(out_f)[0]
    except Exception:                                                     # noqa: BLE001
        pass
    return True, f"--vocab-only writes {arch or 'a header'}", arch


MLX_CHECK = r"""
import glob, json, os, sys
sys.path.insert(0, sys.argv[2])
d = sys.argv[1]
import pollard_mlx as M
eng, why = M.mlx_engine(d)
if eng == "hold":
    print("HOLD:", why); sys.exit(3)
cfg = json.load(open(os.path.join(d, "config.json")))
if eng == "mlx_lm":
    import mlx_lm.utils as U
    get = getattr(U, "_get_classes", None) or getattr(U, "get_model_classes", None)
    if get is None:
        import importlib; importlib.import_module("mlx_lm.models." + cfg["model_type"])
    else:
        get(cfg)
shards = sorted(glob.glob(os.path.join(d, "*.safetensors")), key=os.path.getsize)
if sys.argv[3] == "0":
    print(f"OK {eng} {cfg.get('model_type')} (shard not read: remote mirror) 0 tensors"); sys.exit(0)
if not shards:
    print("no safetensors shard"); sys.exit(4)
import mlx.core as mx
n = len(mx.load(shards[0]))
print(f"OK {eng} {cfg.get('model_type')} {os.path.basename(shards[0])} {n} tensors")
"""


def _mlx(hf_dir, shard=True):
    here = os.path.dirname(os.path.abspath(__file__))
    rc, out = _run([sys.executable, "-c", MLX_CHECK, hf_dir, here, "1" if shard else "0"], timeout=QUICK_TIMEOUT * 3)
    line = next((ln for ln in out.splitlines() if ln.startswith(("OK ", "HOLD:"))), "")
    if rc == 0 and line.startswith("OK "):
        _, eng, mt, shard, n = line.split(" ", 4)
        return True, f"{eng} has {mt}; {shard} loads ({n})"
    return False, (line[5:].strip() if line.startswith("HOLD:") else _why(out))


# ---- the plan --------------------------------------------------------------------------------------------------

def plan_steps(lanes, gguf=None, hf_dir=None, mmproj=None, toolcall=True, flagship=False, imatrix=True, convert=True):
    """[(step, lane, critical)] for what this run will do."""
    out = []
    if "gguf" in lanes:
        if hf_dir and not gguf and convert:
            out.append(("convert", "gguf", True))
        if imatrix:
            out.append(("imatrix", "gguf", True))
        out += [("quantize", "gguf", True), ("perplexity", "gguf", True)]
        if toolcall:
            out.append(("server", "gguf", True))
        if mmproj:
            out.append(("mtmd", "gguf", True))
        if flagship:
            out.append(("ik-quantize", "gguf", False))
    if "draft" in lanes and not any(s == "server" for s, _l, _c in out):
        out.append(("server", "draft", True))
    if "mlx" in lanes and hf_dir:
        out.append(("mlx", "mlx", True))
    return out


def run(lanes, gguf=None, hf_dir=None, mmproj=None, toolcall=True, flagship=False, imatrix=True, ik_dir=None,
        bin_dir=None, arch=None, work=None, out=print, convert=True, mlx_shard=True):
    """Run the preflight. -> {"rows": [...], "failed_lanes": set, "text": table}.
    A row: {step, lane, binary, ok, detail, fix, critical}."""
    work = work or tempfile.mkdtemp(prefix="pollard-preflight-")
    os.makedirs(work, exist_ok=True)
    if gguf and not arch:
        try:
            from pollard_ggufcompat import read_header
            arch = read_header(gguf)[0]
        except Exception:                                                 # noqa: BLE001
            arch = None
    rows, loaded = [], {}                                                 # tree -> (ok, detail) of its real load
    steps = plan_steps(lanes, gguf, hf_dir, mmproj, toolcall, flagship, imatrix, convert)

    def add(step, lane, critical, binary, ok, detail, fix=""):
        rows.append({"step": step, "lane": lane, "critical": critical, "binary": binary, "ok": ok,
                     "detail": detail, "fix": fix})

    for step, lane, critical in steps:
        if step == "convert":
            from pollard_convert import find_converter
            conv, note = find_converter(hf_dir)
            if conv is None:
                add(step, lane, critical, None, False, note.splitlines()[0], fix_for_converter(hf_dir, note))
                continue
            key = "convert|" + _fingerprint(conv, os.path.join(hf_dir, "config.json"))
            hit = _cache_get(key)
            if hit:
                ok, detail, carch = True, hit["detail"] + " (cached)", hit.get("arch")
            else:
                ok, detail, carch = _convert(str(conv), hf_dir, work)
                if ok:
                    _cache_put(key, detail, arch=carch)
            arch = arch or carch                         # every binary below is checked against what it writes
            fix = ""
            if not ok:
                fix = (fix_for_converter(hf_dir, detail) if re.search(r"not supported|registers", detail, re.I)
                       else f"the converter fails on this checkpoint: {detail}")
            add(step, lane, critical, str(conv), ok, detail, fix)
            continue
        if step == "mlx":
            ok, detail = _mlx(hf_dir, mlx_shard)
            add(step, lane, critical, "mlx", ok, detail, "" if ok else
                "the MLX lane cannot load this model here: " + detail)
            continue
        name = STEPS[step][1]
        if step == "ik-quantize":
            binary = os.path.join(ik_dir, name + (".exe" if sys.platform == "win32" else "")) if ik_dir else None
            if not binary or not os.path.isfile(binary):
                add(step, lane, critical, None, False, "no ik_llama.cpp build", "build it: pollard-runtime --update "
                    "ik_llama.cpp (or set IK_LLAMA_BIN) -- the ladder ships without the flagship meanwhile")
                continue
        else:
            binary = _resolve(name, arch, bin_dir)
        if not binary:
            add(step, lane, critical, None, False, f"{name} not found",
                "build the runtime: pollard-runtime --update (or ./install.sh)")
            continue
        ok, why = _starts(binary)
        if not ok:
            add(step, lane, critical, binary, False, why, f"rebuild {_rel(tree_of(binary) or binary)}: "
                "pollard-runtime --update (a partial build relinks libllama and breaks its siblings)")
            continue
        if arch and not _arch_ok(binary, arch):
            add(step, lane, critical, binary, False, f"this build does not know {arch}",
                fix_for_old_binary(binary, arch, name))
            continue
        if not gguf:
            add(step, lane, critical, binary, True, "resolved, starts" + (f", knows {arch}" if arch else "")
                + " (no GGUF yet: loaded once it is converted)")
            continue
        files = [binary, gguf] + ([mmproj] if step == "mtmd" else [])
        key = f"{step}|" + _fingerprint(*files)
        hit = _cache_get(key)
        if hit:
            add(step, lane, critical, binary, True, hit["detail"] + " (cached)")
            continue
        tree = tree_of(binary) or os.path.dirname(binary)
        if step in ("perplexity", "imatrix"):
            if step == "imatrix" and tree in loaded and loaded[tree][0]:
                ok, detail = True, "same build as llama-perplexity (loaded)"
            else:
                ok, detail = _load_gguf(binary, gguf, step, work)
                loaded.setdefault(tree, (ok, detail))
        elif step == "server":
            if tree not in loaded:
                sib = os.path.join(os.path.dirname(binary), "llama-perplexity" + (".exe" if sys.platform == "win32" else ""))
                loaded[tree] = _load_gguf(sib, gguf, "perplexity", work) if os.path.isfile(sib) else \
                    (True, "starts (no llama-perplexity in its build to load with)")
            ok, detail = loaded[tree][0], ("its build loads the file" if loaded[tree][0] else loaded[tree][1])
        elif step == "mtmd":
            ok, detail = _mtmd(binary, gguf, mmproj)
        else:                                                             # quantize, ik-quantize
            ok, detail = _quantize_dry(binary, gguf, "Q4_K" if step == "ik-quantize" else "Q4_K_M")
        fix = ""
        if not ok:
            fix = (fix_for_old_binary(binary, arch, name) if ARCH_FAIL.search(detail)
                   else f"{_rel(binary)} fails on this file: {detail}")
        else:
            _cache_put(key, detail)
        add(step, lane, critical, binary, ok, detail, fix)

    failed = {r["lane"] for r in rows if not r["ok"] and r["critical"]}
    text = table(rows, gguf or hf_dir, arch)
    out(text)
    return {"rows": rows, "failed_lanes": failed, "text": text, "arch": arch}


def table(rows, model, arch):
    name = os.path.basename(str(model or "").rstrip("/\\"))
    lines = [f"   engine preflight :: {name}" + (f"  (arch {arch})" if arch else "")]
    if not rows:
        return lines[0] + "\n      (no engine steps planned)"
    w = max(len(_rel(r["binary"] or "-")) for r in rows)
    w = min(max(w, 6), 70)
    lines.append(f"      {'step':12s} {'lane':5s} {'binary':{w}s}  result")
    for r in rows:
        tag = "ok" if r["ok"] else ("FAIL" if r["critical"] else "warn")
        b = _rel(r["binary"] or "-")
        b = b if len(b) <= w else "..." + b[-(w - 3):]
        lines.append(f"      {r['step']:12s} {r['lane']:5s} {b:{w}s}  {tag:4s} {r['detail']}")
    fixes = {}                                        # one line per distinct fix: "STOP imatrix, perplexity: ..."
    for r in rows:
        if not r["ok"]:
            fixes.setdefault((r["critical"], r["fix"] or r["detail"]), []).append(r["step"])
    for (crit, fix), names in fixes.items():
        lines.append(f"      {'STOP' if crit else 'note'} {', '.join(names)}: {fix}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", help="the GGUF the run starts from (f16 / bf16 / a QAT source)")
    ap.add_argument("--hf", help="the HF checkout (converter + MLX checks)")
    ap.add_argument("--mmproj", help="the multimodal projector GGUF (default: mmproj-*.gguf beside --gguf)")
    ap.add_argument("--lanes", default="gguf", help="comma list of lanes this run builds (default gguf)")
    ap.add_argument("--flagship", action="store_true", help="the run builds the ik_llama trellis flagship")
    ap.add_argument("--no-toolcall", dest="toolcall", action="store_false", help="no tool-call gate (no llama-server)")
    ap.add_argument("--no-imatrix", dest="imatrix", action="store_false", help="the run computes no imatrix")
    ap.add_argument("--bin", help="a llama.cpp build/bin to use instead of the resolver (pollard --bin)")
    a = ap.parse_args(argv)
    if not (a.gguf or a.hf):
        ap.error("pass --gguf and/or --hf")
    mmproj = a.mmproj or (find_mmproj(a.gguf) if a.gguf else None)
    ik_dir = None
    if a.flagship:
        from pollard_auto import find_ik_build
        ik_dir = find_ik_build(a.bin)
    res = run([x.strip() for x in a.lanes.split(",") if x.strip()], gguf=a.gguf, hf_dir=a.hf, mmproj=mmproj,
              toolcall=a.toolcall, flagship=a.flagship, imatrix=a.imatrix, ik_dir=ik_dir, bin_dir=a.bin)
    return 1 if res["failed_lanes"] else 0


def find_mmproj(gguf):
    """The projector pollard writes beside THIS model's f16 (mmproj-<base>-BF16.gguf, pollard_auto._emit_mmproj), or
    None. Never another model's projector that happens to share the folder."""
    import glob
    if not gguf:
        return None
    d, base = os.path.dirname(os.path.abspath(gguf)), os.path.basename(gguf)
    exact = os.path.join(d, "mmproj-" + base.replace("-f16.gguf", "-BF16.gguf"))
    if base.endswith("-f16.gguf") and os.path.isfile(exact):
        return exact
    stem = re.sub(r"(?i)[-.](f16|bf16|f32)\.gguf$|\.gguf$", "", base)
    c = sorted(glob.glob(os.path.join(d, glob.escape(f"mmproj-{stem}") + "*.gguf")))
    return c[0] if c else None


if __name__ == "__main__":
    sys.exit(main())
