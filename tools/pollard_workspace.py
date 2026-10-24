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
    cache/                                    _work dirs: probe GGUFs, base logits (safe to delete;
                                              `pollard-ls --clean-cache`)
    mirror/<model>/                           sparse / streamed source mirrors (remote sources, MLX --hf);
                                              `pollard-ls --clean-mirrors` once every build is on the Hub
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
    """Delete `path` (a folder or file strictly inside `root`) and then every folder between it and `root` left
    empty. Refuses anything else -- a path that resolves (through a symlink or ..) outside `root`, or `root` itself.
    A symlink is never followed: one inside `root` is unlinked (whatever it points at is left alone). True when
    removed. Self-contained -- shipped to the relay machine like scan_staging."""
    import shutil
    r = os.path.normcase(os.path.realpath(root))
    if os.path.islink(path):
        p = os.path.normcase(os.path.join(os.path.realpath(os.path.dirname(os.path.abspath(path))),
                                          os.path.basename(os.path.abspath(path))))
    else:
        p = os.path.normcase(os.path.realpath(path))
    if p == r or not p.startswith(r.rstrip(os.sep) + os.sep):
        raise ValueError(f"refusing to delete {path}: not inside {root}")
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
        path = p
    else:
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


# --------------------------------------------------------------------------------------------- reclaim space
# `pollard-ls --clean-cache / --clean-mirrors / --clean-all`. Everything removed here is strictly inside
# $POLLARD_HOME (remove_within), symlinks are never followed out, and models/ (the builds) is never touched.

#: cache/ subfolders that hold many independent cache files (each file is its own entry); every other
#: top-level item under cache/ (a probe / eval work dir, a verify reference, a KL base) is one entry
CACHE_BUCKETS = ("toolcall", "d3-ref")
#: an entry written to in the last this-many minutes may belong to a running job: kept
CLEAN_MIN_AGE_MIN = 30


def mirror_root(home: str | None = None) -> str:
    """$POLLARD_HOME/mirror: sparse / streamed source mirrors (remote sources, the MLX lane's --hf mirrors)."""
    return os.path.join(home or pollard_home(), "mirror")


def _disk_bytes(st) -> int:
    """What a file really holds on disk: a sparse mirror shard (a 5 GB header-only file with a few tensors fetched)
    counts its allocated blocks, not its apparent size; a small file counts its own size."""
    blocks = getattr(st, "st_blocks", None)
    return min(st.st_size, blocks * 512) if blocks is not None else st.st_size


def tree_bytes(path) -> int:
    """Bytes `path` holds on disk, counted with lstat: a symlink counts as itself, never as what it points at."""
    try:
        st = os.lstat(path)
    except OSError:
        return 0
    if not os.path.isdir(path) or os.path.islink(path):
        return _disk_bytes(st)
    total = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            try:
                total += _disk_bytes(os.lstat(os.path.join(r, f)))
            except OSError:
                pass
    return total


def newest_mtime(path) -> float:
    """The latest mtime of `path` or anything under it (lstat, symlinks not followed)."""
    try:
        newest = os.lstat(path).st_mtime
    except OSError:
        return 0.0
    if os.path.isdir(path) and not os.path.islink(path):
        for r, ds, fs in os.walk(path):
            for n in ds + fs:
                try:
                    newest = max(newest, os.lstat(os.path.join(r, n)).st_mtime)
                except OSError:
                    pass
    return newest


def held_open(root):
    """Paths under `root` some process has open right now, or None when this machine cannot tell (no /proc and no
    lsof, e.g. Windows -- where deleting an open file fails on its own anyway)."""
    root = os.path.realpath(root)
    pre = root.rstrip(os.sep) + os.sep
    if os.path.isdir("/proc/self/fd"):                                     # Linux: every fd we may read
        out = set()
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                fds = os.listdir(f"/proc/{pid}/fd")
            except OSError:
                continue
            for fd in fds:
                try:
                    t = os.readlink(f"/proc/{pid}/fd/{fd}")
                except OSError:
                    continue
                if t == root or t.startswith(pre):
                    out.add(t)
        return out
    import shutil as _sh
    import subprocess
    lsof = _sh.which("lsof")
    if not lsof:
        return None
    try:
        r = subprocess.run([lsof, "-n", "-P", "-F", "n", "+D", root], capture_output=True, text=True, timeout=120)
    except Exception:                                                     # noqa: BLE001
        return None
    if r.returncode not in (0, 1):                                        # 1 = nothing open under it
        return None
    return {ln[1:] for ln in r.stdout.splitlines() if ln.startswith("n") and ln[1:].startswith((pre, root))}


def running_job(home: str | None = None):
    """{pid, what} of a live process holding this workspace's machine lock (a Pollard build / measure), else None."""
    path = os.environ.get("POLLARD_MACHINE_LOCK") or os.path.join(home or pollard_home(), ".pollard.lock")
    try:
        with open(path, encoding="utf-8") as f:
            prev = json.loads(f.read() or "{}")
        pid = int(prev.get("pid", 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    if not pid or pid == os.getpid():
        return None
    try:
        if os.name == "nt":
            import subprocess
            alive = str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True,
                                               text=True, timeout=10).stdout
        else:
            os.kill(pid, 0)
            alive = True
    except PermissionError:
        alive = True
    except Exception:                                                     # noqa: BLE001
        alive = False
    return prev if alive else None


def _in_use(path, opened, cutoff):
    """Why `path` looks in use (a file under it is open, or it was written after `cutoff`), else ''."""
    real = os.path.realpath(path)
    pre = real.rstrip(os.sep) + os.sep
    if opened and any(p == real or p.startswith(pre) for p in opened):
        return "a running process has a file in it open"
    m = newest_mtime(path)
    if m > cutoff:
        return f"written {max(0, (time.time() - m)) / 60:.0f} min ago (may belong to a running job)"
    return ""


def _remove_logged(root, e, why, log, tag):
    try:
        remove_within(root, e["path"])
    except (OSError, ValueError) as err:
        log(f"   {tag}: could not remove {e['path']} ({err}) -- kept")
        return False
    log(f"   {tag}: removed {e['path']} ({human(e['bytes'])}) -- {why}")
    return True


def cache_entries(home: str | None = None) -> list:
    """Each reclaimable item under cache/: the files of the bucket folders (toolcall/, d3-ref/), every other
    top-level file or folder whole. [{path, bytes}]"""
    root = os.path.join(home or pollard_home(), "cache")
    if not os.path.isdir(root):
        return []
    out = []
    for n in sorted(os.listdir(root)):
        p = os.path.join(root, n)
        if n in CACHE_BUCKETS and os.path.isdir(p) and not os.path.islink(p):
            out += [{"path": os.path.join(p, k), "bytes": tree_bytes(os.path.join(p, k))} for k in sorted(os.listdir(p))]
        else:
            out.append({"path": p, "bytes": tree_bytes(p)})
    return out


def clean_cache(home: str | None = None, min_age_min: float = CLEAN_MIN_AGE_MIN, log=print, now=None):
    """Remove everything under cache/ (verify reference logits, KL bases, probe / eval work dirs, toolcall and d3
    reference caches -- all rebuilt on demand) EXCEPT what a running job may be using: an entry with a file open in
    some process, or written in the last `min_age_min` minutes. Returns (bytes freed, [(entry, action, why)])."""
    home = home or pollard_home()
    root = os.path.join(home, "cache")
    entries = cache_entries(home)
    if not entries:
        return 0, []
    cutoff = (time.time() if now is None else now) - min_age_min * 60
    job = running_job(home)
    if job:
        log(f"   cache: a Pollard job is running (pid {job.get('pid')}: {job.get('what', '?')}) -- anything it "
            "may be using is kept")
    opened = held_open(root)
    if opened is None:
        log(f"   cache: this machine cannot list open files -- anything written in the last {min_age_min:g} min "
            "is kept instead")
    freed, plan = 0, []
    for e in entries:
        why = _in_use(e["path"], opened, cutoff)
        if why:
            log(f"   cache: KEPT {e['path']} -- {why}")
            plan.append((e, "keep", why))
        elif _remove_logged(root, e, "cache entry (rebuilt on demand)", log, "cache"):
            freed += e["bytes"]
            plan.append((e, "remove", "cache entry"))
        else:
            plan.append((e, "keep", "could not remove"))
    _prune_empty(root, False, log)
    return freed, plan


# ---- sources (mirror/<model>/, downloads/<Org>__<Model>/): reclaimable once every build made from them is on the Hub

def source_manifests(src_dir, home: str | None = None) -> list:
    """[(slug, MANIFEST)] of every model whose builds were made from the source folder `src_dir`. The release
    records each lane under its own key -- the MLX lane under the source folder itself, GPTQ / MX / EXL3 under
    <source>-smoothed, GGUF under <source>-f16.gguf -- so all three are looked up, plus any MANIFEST whose
    `model` is the folder (or inside it)."""
    home = home or pollard_home()
    name = os.path.basename(str(src_dir).rstrip("/\\"))
    base = name[:-len("-smoothed")] if name.endswith("-smoothed") else name
    want = {base, base + "-smoothed", base + "-f16.gguf"}
    real = os.path.normcase(os.path.realpath(src_dir))
    roots = {real, os.path.normcase(os.path.realpath(os.path.join(os.path.dirname(src_dir), base)))}
    out = []
    mroot = os.path.join(home, "models")
    for slug in (sorted(os.listdir(mroot)) if os.path.isdir(mroot) else []):
        p = os.path.join(mroot, slug, "MANIFEST.json")
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                man = json.load(f)
        except (OSError, ValueError):
            continue
        m = str(man.get("model") or "")
        mp = os.path.normcase(os.path.realpath(m)) if m and (os.path.isabs(m) or m.startswith(".")) else ""
        if slug in want or (mp and any(mp == r or mp.startswith(r.rstrip(os.sep) + os.sep) or
                                       mp in (r + "-smoothed", r + "-f16.gguf") for r in roots)):
            out.append((slug, man))
    return out


def _lane_dest(b):
    """Where a build sits in its repo when the MANIFEST does not say (`uploaded_as`): the GGUF file at the root,
    a directory lane in <LANE>/<rung>/."""
    if b.get("uploaded_as"):
        return b["uploaded_as"]
    lane = str(b.get("lane") or "")
    name = b.get("name") or os.path.basename(str(b.get("path", "")).rstrip("/\\"))
    if lane == "gguf":
        return name
    folder = {"mlx": "MLX", "gptq": "GPTQ", "mx": "MX", "exl3": "EXL3", "exlr8": "EXLR8", "draft": "DRAFT"}.get(lane)
    if not folder:
        return None
    marker = f"-Pollard-{lane.upper()}-"
    sub = (name.split(marker, 1)[1] if marker in name else str(b.get("tag") or name)).lower()
    return folder if lane == "draft" else f"{folder}/{sub}"


def _build_pairs(b, dest):
    """[(path in repo, local file | recorded {size, sha256})] for one build: what its upload was confirmed with
    (`hub_files`), else its local files (hashed), plus the shards a streamed rung already uploaded + freed."""
    rec = b.get("hub_files")
    if isinstance(rec, dict) and rec:
        return [(rp, v) for rp, v in sorted(rec.items()) if isinstance(v, dict)]
    path = str(b.get("path") or "")
    pairs = []
    if os.path.isfile(path) and not os.path.islink(path):
        pairs.append((dest, path))
    elif os.path.isdir(path) and not os.path.islink(path):
        streamed = {}
        sp = os.path.join(path, "pollard-streamed.json")
        if os.path.isfile(sp):
            try:
                with open(sp, encoding="utf-8") as f:
                    streamed = json.load(f).get("files") or {}
            except (OSError, ValueError):
                streamed = {}
        for r, _, fs in os.walk(path):
            for f in fs:
                full = os.path.join(r, f)
                rel = os.path.relpath(full, path).replace(os.sep, "/")
                if rel == "pollard-streamed.json" or rel.endswith(".tmp") or os.path.islink(full):
                    continue
                pairs.append((f"{dest}/{rel}", full))
        pairs += [(e["path_in_repo"], {"size": e["size"], "sha256": e["sha256"], "git_sha1": e.get("git_sha1")})
                  for e in streamed.values() if isinstance(e, dict) and e.get("path_in_repo")]
    return sorted(pairs, key=lambda x: x[0])


def build_on_hub(api, b, repos=(), files_of=None):
    """(True | False | None, why) -- is this build on the Hub intact? True: every file matches by size + sha256 (or,
    for a rung freed after its upload was confirmed that way, the Hub still holds it). False: it is not (never
    uploaded, missing, differs). None: the Hub could not be asked (offline, no huggingface_hub) -- never a yes."""
    import pollard_hubcheck as H
    if api is None:
        return None, "the Hub cannot be asked (huggingface_hub not installed / offline)"
    targets = [b["uploaded"]] if b.get("uploaded") else list(repos)
    if not targets:
        return False, "never uploaded (no repo in its MANIFEST entry -- name one with --repo)"
    dest = _lane_dest(b)
    if not dest:
        return False, f"no repo path known for a {b.get('lane')!r} build"
    pairs = _build_pairs(b, dest)
    last = None
    for repo in targets:
        if pairs:
            bad = H.mismatches(api, repo, pairs)
            if not bad:
                return True, f"{repo}:{dest} (size + sha256 confirmed)"
            if bad[0].startswith("could not read"):
                return None, bad[0]
            last = f"{repo}: " + "; ".join(bad[:3])
        elif b.get("uploaded") == repo and b.get("freed"):
            # freed after its upload was confirmed by size + sha256 (publish_rung): the Hub must still hold it
            try:
                files = files_of(repo) if files_of else set(api.list_repo_files(repo))
            except Exception as e:                                        # noqa: BLE001
                return None, f"could not read the file list of {repo} ({type(e).__name__}: {e})"
            if files is None:
                return None, f"could not read the file list of {repo}"
            if dest in files or any(f.startswith(dest.rstrip("/") + "/") for f in files):
                return True, f"{repo}:{dest} (confirmed by size + sha256 when uploaded + freed; still on the Hub)"
            last = f"{repo}: {dest} is no longer on the Hub"
        else:
            last = "its local copy is gone and no upload of it is recorded"
    return False, last or "not confirmed on the Hub"


def source_status(src_dir, home=None):
    """Offline view for pollard-ls: (reclaimable?, why) from the MANIFESTs alone -- no Hub call, no hashing."""
    builds = [b for _, man in source_manifests(src_dir, home) for b in man.get("builds", [])]
    if not builds:
        return False, "no build recorded from it"
    up = [b for b in builds if b.get("uploaded")]
    if len(up) < len(builds):
        return False, f"{len(builds) - len(up)} of {len(builds)} build(s) not uploaded"
    repos = sorted({b["uploaded"] for b in up})
    return True, f"all {len(builds)} build(s) recorded as uploaded ({', '.join(repos)})"


def source_entries(root):
    """[{path, bytes}] for each folder under a sources root (mirror/ or downloads/)."""
    if not os.path.isdir(root):
        return []
    return [{"path": os.path.join(root, n), "bytes": tree_bytes(os.path.join(root, n))}
            for n in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, n)) or
            os.path.islink(os.path.join(root, n))]


def clean_sources(root, api=None, repos=(), force=False, home=None, min_age_min=CLEAN_MIN_AGE_MIN, log=print,
                  now=None, tag="mirror", hidden=False, only=None):
    """Reclaim the source folders under `root` (mirror/ or downloads/): one goes when EVERY build its MANIFESTs
    record is confirmed on the Hub (build_on_hub). One with an unpublished build, or no recorded build, or whose
    Hub check cannot be done, is kept -- unless `force`, which removes it anyway and says so. A folder a running job
    may be using (a file open, or written in the last `min_age_min` minutes) is kept even with `force`. `hidden`
    also takes dot-folders (downloads/.hf-cache) -- only ever with `force`. `only` limits it to these folders.
    Returns (bytes freed, [(entry, action, why)])."""
    home = home or pollard_home()
    entries = [e for e in source_entries(root)
               if (hidden or not os.path.basename(e["path"]).startswith("."))
               and (only is None or os.path.normcase(os.path.realpath(e["path"])) in
                    {os.path.normcase(os.path.realpath(o)) for o in only})]
    if not entries:
        return 0, []
    cutoff = (time.time() if now is None else now) - min_age_min * 60
    opened = held_open(root)
    if opened is None:
        log(f"   {tag}: this machine cannot list open files -- anything written in the last {min_age_min:g} min "
            "is kept instead")
    listings = {}

    def files_of(repo):
        if repo not in listings:
            try:
                listings[repo] = set(api.list_repo_files(repo))
            except Exception:                                             # noqa: BLE001
                listings[repo] = None
        return listings[repo]

    freed, plan = 0, []
    for e in entries:
        busy = _in_use(e["path"], opened, cutoff)
        if busy:
            log(f"   {tag}: KEPT {e['path']} -- {busy}")
            plan.append((e, "keep", busy))
            continue
        builds = [b for _, man in source_manifests(e["path"], home) for b in man.get("builds", [])]
        bad = None
        if not builds:
            bad = "no build recorded from it (nothing built yet, or a build is still going)"
        for b in builds:
            ok, why = build_on_hub(api, b, repos, files_of)
            if ok is not True:
                bad = (f"{b.get('name') or b.get('path')}: {why}" if ok is False else
                       f"the Hub check could not be done ({why})")
                break
        if bad and not force:
            log(f"   {tag}: KEPT {e['path']} ({human(e['bytes'])}) -- {bad}; `--force` removes it anyway")
            plan.append((e, "keep", bad))
            continue
        why = (f"every build ({len(builds)}) confirmed on the Hub" if not bad else
               f"FORCED (--force) -- NOT confirmed on the Hub: {bad}")
        if bad:
            log(f"   {tag}: !! --force: removing {e['path']} although it is NOT confirmed on the Hub ({bad})")
        if _remove_logged(root, e, why, log, tag):
            freed += e["bytes"]
            plan.append((e, "remove", why))
        else:
            plan.append((e, "keep", "could not remove"))
    return freed, plan


def clean_mirrors(api=None, repos=(), force=False, home=None, min_age_min=CLEAN_MIN_AGE_MIN, log=print, now=None,
                  only=None):
    """clean_sources over $POLLARD_HOME/mirror."""
    return clean_sources(mirror_root(home), api, repos, force, home, min_age_min, log, now, "mirror", only=only)


def clean_downloads_confirmed(api=None, repos=(), force=False, home=None, min_age_min=CLEAN_MIN_AGE_MIN, log=print,
                              now=None):
    """clean_sources over $POLLARD_HOME/downloads (with `force`, the hub cache folder .hf-cache too)."""
    return clean_sources(os.path.join(home or pollard_home(), "downloads"), api, repos, force, home, min_age_min,
                         log, now, "downloads", hidden=force)


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
