#!/usr/bin/env python3
"""pollard_workspace — the shared, organized output home for every Pollard build. Pure stdlib so any
tool can import it. Users get a predictable, self-documenting tree instead of builds scattered wherever
they ran the command.

Layout (default ~/pollard, override with $POLLARD_HOME):

  ~/pollard/
    models/
      <Org>__<Model>/                         e.g. Qwen__Qwen2.5-3B
        <Model>-Pollard-<LANE>-<tag>/         runtime-ready build, HF-card-style name
        calibration/                          imatrix, cal data, sensitivity.json
        reports/                              pollard-verify / scorecard / ppl outputs
        MANIFEST.json                         every build here: lane, bpw, ppl, verified, size, date
    cache/                                    downloads + _work dirs (safe to delete)

Tools call `resolve_out(model, lane, tag, explicit)` — if the user passed --out it's respected; otherwise
the build lands in the workspace automatically. After a build, call `record_build(...)` to log it to the
model's MANIFEST so `pollard-ls` can show what exists and whether it verified.
"""
import json, os, time
from datetime import datetime, timezone


def pollard_home() -> str:
    return os.path.abspath(os.environ.get("POLLARD_HOME", os.path.expanduser("~/pollard")))


def model_slug(model: str) -> str:
    """HF id 'Qwen/Qwen2.5-3B' -> 'Qwen__Qwen2.5-3B'; a local path -> its basename."""
    m = model.rstrip("/\\")
    if "/" in m and not os.path.exists(m):                 # looks like an HF repo id
        return m.replace("/", "__")
    return os.path.basename(m) or m.replace("/", "__")


def model_basename(model: str) -> str:
    slug = model_slug(model)
    return slug.split("__")[-1]


def model_dir(model: str, create: bool = False) -> str:
    d = os.path.join(pollard_home(), "models", model_slug(model))
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def calibration_dir(model: str | None = None, create: bool = False) -> str:
    d = os.path.join(model_dir(model), "calibration") if model else os.path.join(pollard_home(), "calibration")
    if create: os.makedirs(d, exist_ok=True)
    return d


def reports_dir(model: str | None = None, create: bool = False) -> str:
    d = os.path.join(model_dir(model), "reports") if model else os.path.join(pollard_home(), "reports")
    if create: os.makedirs(d, exist_ok=True)
    return d


def cache_dir(create: bool = False) -> str:
    d = os.path.join(pollard_home(), "cache")
    if create: os.makedirs(d, exist_ok=True)
    return d


def charts_dir(model: str | None = None, create: bool = False) -> str:
    """Where rendered charts / eval outputs land so users don't hunt for them.
    Per-model when `model` is given (models/<slug>/charts), else a top-level ~/pollard/charts."""
    d = os.path.join(model_dir(model), "charts") if model else os.path.join(pollard_home(), "charts")
    if create: os.makedirs(d, exist_ok=True)
    return d


def build_name(model: str, lane: str, tag: str = "") -> str:
    """Canonical build dir/file name, e.g. Qwen2.5-3B-Pollard-EXL3-4.0bpw (HF-card convention)."""
    parts = [model_basename(model), "Pollard", lane.upper()]
    if tag:
        parts.append(str(tag).replace("/", "-").replace(" ", ""))
    return "-".join(parts)


def resolve_out(model: str, lane: str, tag: str = "", explicit: str | None = None,
                ext: str = "", create_parent: bool = True) -> str:
    """--out wins if given; else an organized path in the workspace. `ext` for single-file lanes (GGUF)."""
    if explicit:
        return explicit
    d = model_dir(model, create=create_parent)
    name = build_name(model, lane, tag) + (ext if ext else "")
    out = os.path.join(d, name)
    return out


def _manifest_path(model: str) -> str:
    return os.path.join(model_dir(model), "MANIFEST.json")


def read_manifest(model: str) -> dict:
    p = _manifest_path(model)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return {"model": model, "builds": []}


def record_build(model: str, lane: str, path: str, tag: str = "", bpw=None, ppl=None,
                 verified=None, extra: dict | None = None):
    """Append/replace a build entry in the model's MANIFEST.json."""
    man = read_manifest(model)
    man["model"] = model
    size = _dir_size(path)
    entry = {"name": os.path.basename(path.rstrip("/\\")), "lane": lane, "tag": str(tag),
             "path": os.path.abspath(path), "bpw": bpw, "ppl": ppl, "verified": verified,
             "bytes": size, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    man["builds"] = [b for b in man.get("builds", []) if b.get("path") != entry["path"]]
    man["builds"].append(entry)
    md = model_dir(model, create=True)
    tmp = _manifest_path(model) + ".tmp"
    json.dump(man, open(tmp, "w"), indent=2)
    os.replace(tmp, _manifest_path(model))
    return entry


def mark_verified(build_path: str, ok: bool, ppl=None):
    """Find the build with this path in any model's MANIFEST and stamp its verified flag (+ ppl).
    No-op if the build isn't in the workspace (e.g. user built to an explicit --out elsewhere)."""
    build_path = os.path.abspath(build_path.rstrip("/\\"))
    root = os.path.join(pollard_home(), "models")
    if not os.path.isdir(root):
        return False
    for slug in os.listdir(root):
        mpath = os.path.join(root, slug, "MANIFEST.json")
        if not os.path.exists(mpath):
            continue
        try:
            man = json.load(open(mpath))
        except Exception:
            continue
        changed = False
        for b in man.get("builds", []):
            if os.path.abspath(str(b.get("path", "")).rstrip("/\\")) == build_path:
                b["verified"] = bool(ok)
                if ppl is not None:
                    b["ppl"] = ppl
                changed = True
        if changed:
            tmp = mpath + ".tmp"; json.dump(man, open(tmp, "w"), indent=2); os.replace(tmp, mpath)
            return True
    return False


def _dir_size(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try: total += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    return total


def human(n) -> str:
    if not n:
        return "-"
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def list_models() -> list:
    root = os.path.join(pollard_home(), "models")
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
