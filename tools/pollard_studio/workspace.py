#!/usr/bin/env python3
"""Read the real Pollard workspace, so Studio shows what is on disk and nothing else.

$POLLARD_HOME (default ~/pollard) holds:
    models/<repo-ish name>/MANIFEST.json + the built GGUFs
    rungs/<name>.gguf                    + .tensor-types.txt sidecars
    downloads/<source weights>

Everything here is measured or absent. A build with no perplexity recorded reports None and the UI
says "not measured" -- it never falls back to a plausible-looking number, because a plausible
number in a quality panel is worse than a blank one.

    python workspace.py            # what Studio will show
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from . import ggufread
from . import saferead

CACHE = Path.home() / ".cache/pollard-studio/gguf.json"
TAG_RE = re.compile(r"-(?:Pollard-)?((?:IQ|Q|BF|F)\w+|FLAGSHIP|f16)\.gguf$", re.I)


def home() -> Path:
    return Path(os.environ.get("POLLARD_HOME", Path.home() / "pollard"))


# ── gguf summary cache (header parse is cheap, but not free on a spinning list) ─────────────────
def _cache_load() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def _cache_save(c: dict) -> None:
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(c))
    except Exception:
        pass


def summarise_dir(d: Path, cache: dict | None = None) -> dict | None:
    """Header summary for a safetensors build (GPTQ / MLX / EXL3 / MX), cached on the newest shard."""
    own = cache is None
    cache = _cache_load() if own else cache
    shards = sorted(d.glob("*.safetensors"))
    if not shards:
        return None
    stamp = max(int(p.stat().st_mtime) for p in shards)
    total = sum(p.stat().st_size for p in shards)
    key = f"{d}|dir|{stamp}|{total}"
    if key not in cache:
        try:
            cache[key] = saferead.summarise(saferead.read(d))
        except Exception as e:
            cache[key] = {"error": str(e), "file_bytes": total, "name": d.name}
    if own:
        _cache_save(cache)
    return cache[key]


def summarise(path: Path, cache: dict | None = None) -> dict | None:
    """Header summary for one GGUF, cached on (path, mtime, size)."""
    own = cache is None
    cache = _cache_load() if own else cache
    try:
        st = path.stat()
    except OSError:
        return None
    key = f"{path}|{int(st.st_mtime)}|{st.st_size}"
    if key not in cache:
        try:
            cache[key] = ggufread.summarise(ggufread.read(path))
        except Exception as e:                       # a partial file mid-build is normal, not fatal
            cache[key] = {"error": str(e), "file_bytes": st.st_size, "name": path.name}
    if own:
        _cache_save(cache)
    return cache[key]


def tag_of(path: Path) -> str:
    m = TAG_RE.search(path.name)
    return m.group(1) if m else path.stem.split("-")[-1]


def _tensor_types(path: Path) -> dict | None:
    """The .tensor-types.txt sidecar: the exact per-tensor recipe the build was given."""
    side = path.with_suffix(path.suffix + ".tensor-types.txt")
    if not side.exists():
        return None
    rules = []
    for line in side.read_text(errors="replace").splitlines():
        if "=" in line:
            pat, atom = line.rsplit("=", 1)
            rules.append({"pattern": pat.strip(), "atom": atom.strip()})
    return {"path": str(side), "rules": rules}


def scan(deep: bool = True) -> dict:
    """Every model and every build under $POLLARD_HOME."""
    root = home()
    cache = _cache_load()
    models: list[dict] = []

    mdir = root / "models"
    if mdir.is_dir():
        for d in sorted(p for p in mdir.iterdir() if p.is_dir()):
            mf = d / "MANIFEST.json"
            declared = {}
            if mf.exists():
                try:
                    data = json.loads(mf.read_text())
                    declared = {b.get("name"): b for b in data.get("builds", [])}
                except Exception:
                    declared = {}
            builds = []
            for g in sorted(d.glob("*.gguf")):
                builds.append(_build(g, declared.get(g.name, {}), cache, deep))
            builds += _safetensors_builds(d, declared, cache, deep)
            if builds:
                models.append({"key": d.name, "name": d.name.replace("__", "/"),
                               "dir": str(d), "manifest": mf.exists(), "builds": builds})

    rdir = root / "rungs"
    if rdir.is_dir():
        fam: dict[str, list] = {}
        for g in sorted(rdir.glob("*.gguf")):
            base = TAG_RE.sub("", g.name) or g.stem
            fam.setdefault(base, []).append(g)
        for base, files in sorted(fam.items()):
            models.append({"key": f"rungs:{base}", "name": base, "dir": str(rdir),
                           "manifest": False,
                           "builds": [_build(g, {}, cache, deep) for g in files]})

    # downloads hold the fp16 sources; they are builds too, and the reference a lane is measured
    # against, so they belong in the same list rather than in a footnote
    ddir = root / "downloads"
    if ddir.is_dir():
        for d in sorted(p for p in ddir.iterdir() if p.is_dir()):
            got = _safetensors_builds(d, {}, cache, deep)
            if got:
                models.append({"key": f"src:{d.name}", "name": d.name.replace("__", "/"),
                               "dir": str(d), "manifest": False, "source": True,
                               "builds": got})

    _cache_save(cache)
    dl = root / "downloads"
    return {
        "home": str(root),
        "exists": root.is_dir(),
        "models": models,
        "downloads": sorted(p.name for p in dl.iterdir()) if dl.is_dir() else [],
        "scanned": time.strftime("%H:%M:%S"),
    }


def _safetensors_builds(d: Path, declared: dict, cache: dict, deep: bool) -> list[dict]:
    """A safetensors build is a DIRECTORY, not a file -- one entry per lane dir, not per shard."""
    out = []
    for cand in ([d] if list(d.glob("*.safetensors")) else []) + \
                [p for p in sorted(d.iterdir()) if p.is_dir() and list(p.glob("*.safetensors"))]:
        shards = sorted(cand.glob("*.safetensors"))
        total = sum(p.stat().st_size for p in shards)
        stamp = max(p.stat().st_mtime for p in shards)
        b = {
            "name": cand.name, "path": str(cand), "tag": cand.name.split("-")[-1],
            "bytes": total, "gb": total / 1e9, "kind": "safetensors",
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp)),
            "ppl": declared.get(cand.name, {}).get("ppl"),
            "verified": declared.get(cand.name, {}).get("verified"),
            "lane": "?", "recipe": None,
        }
        if deep:
            s = summarise_dir(cand, cache) or {}
            b.update({"architecture": s.get("architecture"), "blocks": s.get("block_count"),
                      "params": s.get("params"), "bpw": s.get("bpw"), "groups": s.get("groups"),
                      "types": s.get("types"), "lane": s.get("lane", "?"),
                      "quant": s.get("quant"), "shards": s.get("shards"),
                      "mtp_block": None, "error": s.get("error")})
            if s.get("lane"):
                b["tag"] = s["lane"]
        out.append(b)
    return out


def _build(path: Path, declared: dict, cache: dict, deep: bool) -> dict:
    st = path.stat()
    b = {
        "name": path.name, "path": str(path), "tag": tag_of(path), "kind": "gguf",
        "bytes": st.st_size, "gb": st.st_size / 1e9,
        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
        # only ever what was recorded; absent stays absent
        "ppl": declared.get("ppl"), "verified": declared.get("verified"),
        "lane": declared.get("lane", "GGUF"),
        "recipe": _tensor_types(path),
    }
    if deep:
        s = summarise(path, cache) or {}
        b.update({
            "architecture": s.get("architecture"), "blocks": s.get("block_count"),
            "params": s.get("params"), "bpw": s.get("bpw"),
            "groups": s.get("groups"), "types": s.get("types"),
            "mtp_block": s.get("mtp_block"), "error": s.get("error"),
        })
    return b


def pick(ws: dict, key: str | None = None) -> dict | None:
    """The model Studio opens on: the one asked for, else the one with the most builds."""
    if not ws["models"]:
        return None
    if key:
        for m in ws["models"]:
            if m["key"] == key:
                return m
    return max(ws["models"], key=lambda m: (len(m["builds"]),
                                            max(b["bytes"] for b in m["builds"])))


def main() -> None:
    ws = scan(deep="--shallow" not in sys.argv)
    print(f"POLLARD_HOME  {ws['home']}   ({'found' if ws['exists'] else 'MISSING'})")
    print(f"{len(ws['models'])} models, {sum(len(m['builds']) for m in ws['models'])} builds\n")
    for m in ws["models"]:
        print(f"  {m['name']}")
        for b in m["builds"]:
            ppl = f"ppl {b['ppl']:.2f}" if b.get("ppl") else "ppl not measured"
            bpw = f"{b['bpw']:.2f} bpw" if b.get("bpw") else "bpw ?"
            print(f"      {b['tag']:<10} {b['gb']:7.2f} GB  {bpw:>10}  {ppl}"
                  + ("  +recipe" if b["recipe"] else ""))
        print()


if __name__ == "__main__":
    main()
