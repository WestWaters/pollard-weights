#!/usr/bin/env python3
"""pollard_workspace -- the shared, organized output home for every Pollard build. Pure stdlib so any
tool can import it. Users get a predictable, self-documenting tree instead of builds scattered wherever
they ran the command.

Layout (default ~/pollard, override with $POLLARD_HOME):

  ~/pollard/
    downloads/                                pulled SOURCE models (the input side), one findable copy
      <Org>__<Model>/                         e.g. Qwen__Qwen2.5-3B (mirrors the models/ naming)
      .hf-cache/                              hub cache pinned here so a pull can't dup into ~/.cache
    models/                                   BUILT Pollard outputs (the output side)
      <Org>__<Model>/                         e.g. Qwen__Qwen2.5-3B
        <Model>-Pollard-<LANE>-<tag>/         runtime-ready build, HF-card-style name
        calibration/                          imatrix, cal data, sensitivity.json
        reports/                              pollard-verify / scorecard / ppl outputs
        MANIFEST.json                         every build here: lane, bpw, ppl, verified, size, date
    cache/                                    _work dirs: probe GGUFs, base logits (safe to delete)
    staging/                                  copies waiting to be uploaded FROM this machine (the --upload-via
      <Org>__<Repo>/<LANE>__<rung>/           relay): one fixed folder per rung, resumed by a rerun, deleted once
                                              the Hub holds it (size + sha256); `pollard-ls --clean-staging`

Tools call `resolve_out(model, lane, tag, explicit)` -- if the user passed --out it's respected; otherwise
the build lands in the workspace automatically. After a build, call `record_build(...)` to log it to the
model's MANIFEST so `pollard-ls` can show what exists and whether it verified.
"""
import json, os, sys, time, tempfile
from contextlib import contextmanager
from datetime import datetime, timezone

# Windows consoles default to cp1252 and CRASH (UnicodeEncodeError) on the progress bars / emoji that
# Pollard and its deps (gptqmodel, llm-compressor, tqdm) print. Every Pollard tool imports this module,
# so forcing UTF-8 here -- at import, before those deps load -- makes the whole toolchain safe on Windows
# with no PYTHONUTF8 env needed. errors="replace" so an odd byte degrades a glyph instead of crashing.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


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


def downloads_dir(create: bool = False) -> str:
    """Staging home for pulled SOURCE models -- the input side, distinct from models/ (built outputs) and
    cache/ (scratch). $POLLARD_HOME/downloads/<Org>__<Model>/ keeps every fetched model in one findable,
    human-named place instead of scattered under wherever the command was run."""
    d = os.path.join(pollard_home(), "downloads")
    if create: os.makedirs(d, exist_ok=True)
    return d


def source_dir(model: str, create: bool = False) -> str:
    """Per-source-model folder under downloads/, named like the models/ tree so the pair lines up."""
    d = os.path.join(downloads_dir(), model_slug(model))
    if create: os.makedirs(d, exist_ok=True)
    return d


def fetch_source(repo_id: str, filename: str | None = None, revision: str | None = None,
                 allow_patterns=None, token: str | None = None) -> str:
    """Pull an HF source model (or one file) into $POLLARD_HOME/downloads/<slug> -- ONE findable copy.

    Returns the local file path (when `filename` is given) or the model dir. A path that already exists
    locally is returned as-is (point Pollard at a repo id OR a dir). The Xet backend is disabled (its
    write token can expire mid-transfer, which stalls big pulls) and the hub cache is pinned INSIDE
    downloads/ (`.hf-cache`) so a download can't silently duplicate the model into ~/.cache and fill the
    disk -- everything the pull touches lives under one folder you can see and clean. Idempotent: a
    complete prior download is reused, not re-fetched."""
    if os.path.exists(repo_id):                       # already a local dir/file -- use it directly
        return repo_id
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    dest = source_dir(repo_id, create=True)
    hf_cache = os.path.join(downloads_dir(), ".hf-cache")
    if filename:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id, filename, revision=revision, local_dir=dest,
                               cache_dir=hf_cache, token=token)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, revision=revision, local_dir=dest, cache_dir=hf_cache,
                             allow_patterns=allow_patterns, token=token)


def clean_downloads() -> int:
    """Delete the downloads/ staging tree (source models are re-pullable). Returns bytes freed."""
    import shutil
    d = downloads_dir()
    n = _dir_size(d)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    return n


# ------------------------------------------------------------------------------------------------ staging
# A copy of a built rung waiting to be uploaded from THIS machine (the --upload-via relay copies a rung from the
# builder here, uploads it with this machine's HF login, checks it on the Hub, then deletes the copy). One fixed
# folder per rung -- $POLLARD_HOME/staging/<repo slug>/<path in repo>/ -- so a transfer that dies part way is
# resumed by the next run (files already here at the right size + sha256 are not sent again), and a folder left
# behind is always findable, sized by pollard-ls and reclaimed by `pollard-ls --clean-staging`.

#: written into every staging folder: which repo / path it is for and the size + sha256 of every file, so a sweep
#: can tell a copy whose upload is already confirmed on the Hub (safe to delete) from one still waiting
STAGING_MARK = ".pollard-staging.json"
#: where the relay staged before it used the workspace (a random folder per transfer, never reclaimed). Swept for
#: empty folders and copies already on the Hub; nothing else outside the workspace is ever deleted.
LEGACY_STAGING = "C:/pollard/mlx-staging"
#: a staging folder not confirmed on the Hub and older than this is reported (never deleted) by a sweep
STAGING_REPORT_DAYS = 7


def staging_root(home: str | None = None) -> str:
    return os.path.join(home or pollard_home(), "staging")


def staging_key(path_in_repo: str) -> str:
    """'MLX/4bit' -> 'MLX__4bit' (one flat folder per upload: a file uploaded INTO a rung folder never lands
    inside that rung's staging copy); '' -> '_root'."""
    p = str(path_in_repo or "").strip("/\\").replace("\\", "/")
    return p.replace("/", "__") if p else "_root"


def staging_dir(repo_id: str, path_in_repo: str, home: str | None = None) -> str:
    return os.path.join(staging_root(home), model_slug(repo_id) if "/" in repo_id else repo_id,
                        staging_key(path_in_repo))


def scan_staging(root, depth=2):
    """Every staging folder `depth` levels under `root` (2 = <repo slug>/<key>, 1 = the legacy <uuid>): path, bytes,
    {relative file: size} (the marker excluded), newest mtime, the marker (or None). Self-contained (stdlib only,
    imports inside; json + os are imported where the relay runs it) -- the relay ships this function's source to
    the machine it uploads from and runs it there."""
    out = []
    if not os.path.isdir(root):
        return out
    level = [root]
    for _ in range(depth):
        level = [os.path.join(d, n) for d in level for n in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, n))]
    for d in level:
        files, mark, newest = {}, None, os.path.getmtime(d)
        for r, _, fs in os.walk(d):
            for f in fs:
                p = os.path.join(r, f)
                rel = os.path.relpath(p, d).replace(os.sep, "/")
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                newest = max(newest, st.st_mtime)
                if rel == ".pollard-staging.json":
                    try:
                        with open(p, encoding="utf-8") as fh:
                            mark = json.load(fh)
                    except (OSError, ValueError):
                        mark = None
                    continue
                files[rel] = st.st_size
        out.append({"path": d.replace(os.sep, "/"), "bytes": sum(files.values()), "files": files, "mtime": newest,
                    "mark": mark, "has_mark": os.path.isfile(os.path.join(d, ".pollard-staging.json"))})
    return out


def remove_within(root, path):
    """Delete `path` (a folder strictly inside `root`) and then every folder between it and `root` left empty.
    Refuses anything else. True when removed. Self-contained -- shipped to the relay machine like scan_staging."""
    import shutil
    r = os.path.normcase(os.path.realpath(root))
    p = os.path.normcase(os.path.realpath(path))
    if p == r or not p.startswith(r.rstrip(os.sep) + os.sep):
        raise ValueError(f"refusing to delete {path}: not inside {root}")
    shutil.rmtree(path)
    d = os.path.dirname(os.path.realpath(path))
    while os.path.normcase(d) != r and os.path.normcase(d).startswith(r) and os.path.isdir(d) and not os.listdir(d):
        os.rmdir(d)
        d = os.path.dirname(d)
    return True


def sha256_files(base, rels):
    """{rel: sha256} of files under `base`. Self-contained -- shipped to the relay machine."""
    import hashlib
    out = {}
    for rel in rels:
        h = hashlib.sha256()
        try:
            with open(os.path.join(base, rel), "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
        except OSError:
            continue
        out[rel] = h.hexdigest()
    return out


def hub_holds(listing, files):
    """Does a Hub listing {path in repo: [size, lfs sha256 or None]} hold every file of `files`
    {path in repo: [size, sha256]} intact (same size, and the same sha256 wherever the Hub reports one)?"""
    if not files or listing is None:
        return False
    for rp, (size, sha) in files.items():
        got = listing.get(rp)
        if not got or int(got[0]) != int(size) or (got[1] and sha and got[1] != sha):
            return False
    return True


def _legacy_on_hub(entry, listing, shas_of):
    """A legacy staging folder (no marker: <uuid>/<rung folder or file>) is on the Hub when every file in it matches a
    Hub file of the same name and size -- and the same sha256 wherever the Hub reports one (hashed only when the
    names and sizes already line up)."""
    if not entry["files"] or not listing:
        return False
    by = {}
    for rp, (size, sha) in listing.items():
        by.setdefault((rp.rsplit("/", 1)[-1], int(size)), []).append(sha)
    cand = {rel: by.get((rel.rsplit("/", 1)[-1], int(sz))) for rel, sz in entry["files"].items()}
    if not all(cand.values()):
        return False
    need = [rel for rel, s in cand.items() if any(s)]
    got = shas_of(entry, need) if need else {}
    return all(not any(s) or got.get(rel) in s for rel, s in cand.items())


def plan_sweep(entries, listing_of, shas_of, repos=(), now=None, report_days=STAGING_REPORT_DAYS, legacy=False):
    """[(entry, action, why)] for staging folders: action 'remove' (empty, or every file already confirmed on the Hub
    by size + sha256), 'report' (not on the Hub and older than `report_days` -- kept, said out loud) or 'keep' (a
    recent partial copy: the next upload resumes it). `listing_of(repo)` -> Hub listing or None (unreadable: keep);
    `shas_of(entry, rels)` -> {rel: sha256} of files in the folder. Nothing that is not proven uploaded is removed."""
    now = time.time() if now is None else now
    plan = []
    for e in entries:
        if not e["files"] and not e.get("has_mark"):
            plan.append((e, "remove", "empty"))
            continue
        mark = e.get("mark") or {}
        on_hub = None
        if mark.get("repo_id") and mark.get("files"):
            listing = listing_of(mark["repo_id"])
            if listing is not None and hub_holds(listing, mark["files"]):
                on_hub = mark["repo_id"]
        elif legacy or not e.get("has_mark"):
            for repo in repos:
                listing = listing_of(repo)
                if listing and _legacy_on_hub(e, listing, shas_of):
                    on_hub = repo
                    break
        if on_hub:
            plan.append((e, "remove", f"already on the Hub ({on_hub}, size + sha256 confirmed)"))
        elif now - e["mtime"] > report_days * 86400:
            plan.append((e, "report", f"not confirmed on the Hub, untouched for {(now - e['mtime']) / 86400:.0f} days "
                                      f"({human(e['bytes'])}) -- kept; `pollard-ls --clean-staging --force` removes it"))
        else:
            plan.append((e, "keep", f"partial copy ({human(e['bytes'])}) -- the next upload of it resumes here"))
    return plan


def staging_usage(home: str | None = None) -> list:
    """scan_staging of this machine's workspace staging area (pollard-ls)."""
    return scan_staging(staging_root(home))


def _hub_listing(api, repo):
    try:
        d = {}
        for f in api.list_repo_tree(repo, recursive=True):
            if hasattr(f, "size") and not hasattr(f, "tree_id"):
                lfs = getattr(f, "lfs", None)
                sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
                d[f.path] = [f.size, sha]
        return d
    except Exception:                                                     # noqa: BLE001
        return None


def sweep_staging(api=None, repos=(), force=False, home=None, legacy_root=None, report_days=STAGING_REPORT_DAYS,
                  log=print, now=None):
    """Sweep THIS machine's staging area (and the legacy relay folder, when it exists here): remove empty folders and
    copies already confirmed on the Hub, report anything else older than `report_days`, keep recent partial copies.
    `force` also removes the copies that are not confirmed (a staging copy is only ever a copy: the rung itself
    stays on the machine that built it until its upload is confirmed). Returns (bytes freed, plan)."""
    cache = {}

    def listing_of(repo):
        if repo not in cache:
            cache[repo] = _hub_listing(api, repo) if api is not None else None
        return cache[repo]

    def shas_of(e, rels):
        return sha256_files(e["path"], rels)

    freed, done = 0, []
    roots = [(staging_root(home), 2, False)]
    lr = legacy_root if legacy_root is not None else LEGACY_STAGING
    if lr and os.path.isdir(lr):
        roots.append((lr, 1, True))
    for root, depth, legacy in roots:
        plan = plan_sweep(scan_staging(root, depth), listing_of, shas_of, repos, now, report_days, legacy)
        for e, act, why in plan:
            if act == "remove" or (force and not legacy):
                remove_within(root, e["path"])
                freed += e["bytes"]
                log(f"   staging: removed {e['path']} -- {why if act == 'remove' else 'forced (--force)'}")
                act = "remove"
            elif act == "report":
                log(f"   staging: KEPT {e['path']} -- {why}")
            done.append((e, act, why))
        _prune_empty(root, legacy, log)
    return freed, done


def _prune_empty(root, legacy, log=print):
    """Empty <repo slug> folders under the staging root go; the legacy root itself goes once it is empty."""
    if not os.path.isdir(root):
        return
    for n in os.listdir(root):
        p = os.path.join(root, n)
        if os.path.isdir(p) and not os.listdir(p):
            os.rmdir(p)
    if legacy and not os.listdir(root):
        os.rmdir(root)
        log(f"   staging: removed the empty legacy folder {root}")


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
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"model": model, "builds": []}


@contextmanager
def manifest_transaction(model: str):
    """Lock the model's read/modify/replace transaction across tool processes.

    Lock a stable sidecar, not MANIFEST.json itself: replacing the JSON changes its inode. Keep the
    sidecar after unlocking so a waiting writer and a new writer always lock the same file.
    """
    model_dir(model, create=True)
    with open(_manifest_path(model) + ".lock", "a+b") as lock:
        if os.name == "nt":
            import errno
            import msvcrt
            if lock.tell() == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            while True:
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as e:
                    if e.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    time.sleep(0.05)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            man = read_manifest(model)
            yield man
            _write_manifest(model, man)
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def record_build(model: str, lane: str, path: str, tag: str = "", bpw=None, ppl=None,
                 verified=None, extra: dict | None = None):
    """Append/replace a build entry in the model's MANIFEST.json."""
    size = _dir_size(path)
    entry = {"name": os.path.basename(path.rstrip("/\\")), "lane": lane, "tag": str(tag),
             "path": os.path.abspath(path), "bpw": bpw, "ppl": ppl, "verified": verified,
             "bytes": size, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    entry.update(extra or {})
    with manifest_transaction(model) as man:
        man["model"] = model
        man["builds"] = [b for b in man.get("builds", []) if b.get("path") != entry["path"]]
        man["builds"].append(entry)
    return entry


def _write_manifest(model: str, man: dict):
    """Replace a manifest while the caller holds manifest_transaction's lock."""
    write_json_atomic(_manifest_path(model), man)


def update_build(model: str, build_path: str, patch: dict, remove=()) -> bool:
    """Apply just this writer's fields (and drop `remove`) on EVERY build entry recorded for `build_path`, under the
    manifest lock, preserving the fields other writers own. True when at least one entry matched; False -- and the
    MANIFEST unchanged -- when none did."""
    build_path = os.path.abspath(build_path.rstrip("/\\"))
    found = False
    with manifest_transaction(model) as man:
        for b in man.get("builds", []):
            if os.path.abspath(str(b.get("path", "")).rstrip("/\\")) == build_path:
                b.update(patch)
                for key in remove:
                    b.pop(key, None)
                found = True
    return found


def record_calibration(model: str, corpus: str, sources: dict):
    """Log a calibration corpus and EXACTLY what it was built from (dataset ids, revisions, licences,
    profile, self-generation) in the model's MANIFEST -- a build should be able to say what it saw."""
    entry = {"path": os.path.abspath(corpus), **{k: v for k, v in sources.items() if k != "path"},
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with manifest_transaction(model) as man:
        man["model"] = model
        man["calibration"] = [c for c in man.get("calibration", []) if c.get("path") != entry["path"]] + [entry]
    return entry


def _find_build(build_path: str):
    build_path = os.path.abspath(build_path.rstrip("/\\"))
    for slug in list_models():
        mpath = os.path.join(pollard_home(), "models", slug, "MANIFEST.json")
        try:
            with open(mpath, encoding="utf-8") as f:
                man = json.load(f)
        except Exception:
            continue
        for b in man.get("builds", []):
            if os.path.abspath(str(b.get("path", "")).rstrip("/\\")) == build_path:
                return man.get("model") or slug, b, slug
    return None, None, None


def find_build(build_path: str):
    """(model key, build entry) for a build path recorded in any MANIFEST, or (None, None)."""
    return _find_build(build_path)[:2]


def record_gate(build_path: str, gate: dict) -> bool:
    """Stamp a pollard-verify --gate verdict (pass/fail + every check) onto a build in its MANIFEST.
    `verified` follows the gate, so pollard-ls, pollard-release's resume and the card all agree."""
    build_path = os.path.abspath(build_path.rstrip("/\\"))
    model, entry, slug = _find_build(build_path)
    if not slug:
        return False
    return update_build(slug, build_path, {"verified": bool(gate.get("pass")), "gate": gate})


def calib_samples(path: str) -> list:
    """Calibration SAMPLES from a corpus, for an exporter that takes one sample per item (GPTQ, llm-compressor).

    pollard-calib writes <corpus>.samples.jsonl beside the corpus with every sample whole -- a rendered agent trace
    keeps its tool list, its call and the tool's answer in one sample. Without that twin (a corpus from elsewhere, or
    an older one) the corpus is read the way it always was: one sample per non-empty line."""
    twin = str(path) + ".samples.jsonl"
    # a twin older than the corpus belongs to a corpus that has since been replaced
    if os.path.isfile(twin) and os.path.getmtime(twin) >= os.path.getmtime(path) - 2:
        out = []
        try:
            with open(twin, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        s = json.loads(line)
                        if isinstance(s, str) and s.strip():
                            out.append(s)
        except (OSError, ValueError):
            out = []
        if out:
            return out
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _process_umask() -> int:
    """This process's umask, read once at import (os.umask can only be read by setting it, which is not safe to do
    while other threads create files)."""
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


_UMASK = _process_umask()


def write_json_atomic(path: str, obj, indent=2) -> None:
    """Write `obj` as JSON to `path` through a unique temp file + fsync + rename: a full disk (ENOSPC) or a crash mid-write
    leaves the previous file exactly as it was, and the half-written temp file is removed. Raises OSError (same errno)
    naming the file."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        # mkstemp creates the temp file 0600; the file it becomes is an ordinary one (0644 under the usual umask),
        # as readable as a plain open() would have made it
        os.chmod(tmp, 0o666 & ~_UMASK)
        os.replace(tmp, path)
    except BaseException as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        if isinstance(e, OSError):
            raise OSError(e.errno, f"could not write {path}: {e.strerror or e} -- the previous file is kept") from e
        raise


def merge_results(path: str, rung: str, patch: dict, meta: dict | None = None) -> dict:
    """Merge measured numbers for one rung into a results.json (the file pollard-card --results reads:
    {rung: {ppl, kld, top1, toolcall, decision, retention, mt, ...}, _meta keys...}). Nested dicts are
    merged, not replaced, so taskeval / toolcall / verify can each add their part to the same rung."""
    try:
        res = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    except Exception:
        res = {}
    cur = res.setdefault(rung, {}) if rung else res
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(cur.get(k), dict):
            cur[k] = {**cur[k], **v}
        else:
            cur[k] = v
    for k, v in (meta or {}).items():
        res[k] = v
    write_json_atomic(path, res, indent=2)
    return res


def mark_verified(build_path: str, ok: bool, ppl=None):
    """Find the build with this path in any model's MANIFEST and stamp its verified flag (+ ppl).
    No-op if the build isn't in the workspace (e.g. user built to an explicit --out elsewhere)."""
    model, entry, slug = _find_build(build_path)
    if not slug:
        return False
    patch = {"verified": bool(ok)}
    if ppl is not None:
        patch["ppl"] = ppl
    return update_build(slug, build_path, patch)


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


def _model_dir_of(model_id):
    if os.path.isdir(model_id):
        return model_id
    try:
        from huggingface_hub import hf_hub_download
        return os.path.dirname(hf_hub_download(model_id, "config.json"))
    except Exception:                                                     # noqa: BLE001
        return None


def resolve_trust_remote_code(model_id, mode="auto") -> bool:
    """May this checkpoint's own Python code run? Custom architectures (Spark2_5, Nemotron-3.5-Bolt) ship their
    modelling files and name them in config.json's auto_map. That code NEVER runs silently (owner, 2026-10-16):
      'on'   consent given (--trust-remote-code on / release plan trust_remote_code: true) -- every file is named;
      'off'  never;
      'auto' no auto_map -> nothing to run (False); an auto_map -> ask y/N on a terminal, refuse anywhere else with
             the files and the flag that consents (POLLARD_TRUST_REMOTE_CODE=1 also consents).
    Decided from config.json read as plain JSON -- no code is executed to make the decision. Shared by every lane."""
    if mode in (True, "on"):
        d = _model_dir_of(model_id)
        if d:
            import pollard_stream as S
            if S.needs_remote_code(d):
                return S.consent(d, "on")
        return True
    if mode in (False, "off", None):
        return False
    d = _model_dir_of(model_id)
    if not d:
        return False
    import pollard_stream as S
    return S.consent(d, mode)


def config_dict(model_id) -> dict:
    """config.json as plain JSON (local dir, or the one file from the Hub) -- never runs checkpoint code. {} if
    it cannot be read."""
    d = _model_dir_of(model_id) if not os.path.isdir(str(model_id)) else model_id
    try:
        with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                                     # noqa: BLE001
        return {}


def text_config_dict(cfg: dict) -> dict:
    """The language model's config inside a wrapper (llm_config / text_config / thinker_config.text_config), merged
    over the top level so plain keys (num_hidden_layers, num_experts ...) read the same for every model kind."""
    out = dict(cfg or {})
    for k in ("llm_config", "language_config", "text_config", "thinker_config"):
        sub = (cfg or {}).get(k)
        if isinstance(sub, dict):
            if k == "thinker_config" and isinstance(sub.get("text_config"), dict):
                sub = sub["text_config"]
            out.update(sub)
            break
    return out


def parse_layers(spec) -> set:
    """Parse a layer spec like '3,4,8' or '3-8,16,20-22' into a set of ints. Empty -> empty set.
    Shared by the allocation lanes for --focus-layers (steer the budget to chosen layers)."""
    out = set()
    for part in str(spec or "").replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out
