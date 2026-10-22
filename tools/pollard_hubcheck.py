"""Is this exact file on the Hub? One answer for every place Pollard deletes a local copy after an upload.

`pollard release --free-after-upload` (a whole rung) and pollard_stream's per-shard upload (a rung too big for the
disk, written + uploaded + freed one safetensors shard at a time) both free local bytes only after the Hub holds that
file byte for byte: the size and the hash the Hub reports for it -- the LFS sha256, or the git blob id of a small
file stored in git -- read back with get_paths_info, compared with the local file (or with the size + sha256 recorded
when a streamed shard was uploaded). Anything else -- a missing file, a short one, another hash, no hash at all, a Hub
that cannot be asked -- is a mismatch, and nothing is freed on a guess."""
from __future__ import annotations

import hashlib
import os


def sha256_file(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    """The id git (and the Hub) gives a small, non-LFS file: sha1 of "blob <size>\\0" + its bytes."""
    h = hashlib.sha1(f"blob {os.path.getsize(path)}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _lfs_sha(f):
    lfs = getattr(f, "lfs", None)
    if not lfs:
        return None
    return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)


def mismatches(api, repo, pairs, revision=None):
    """[why] for every (path in repo, local file | {"size", "sha256"}) the Hub does NOT hold intact; [] = all there.
    A recorded entry (a streamed shard whose local copy is already gone) is checked against its recorded size and
    sha256, which were computed from the bytes before they were uploaded. `revision`: a branch (a streamed rung's
    staging branch) instead of the default one."""
    if not pairs:
        return ["nothing to check"]
    try:
        paths = [p for p, _ in pairs]
        got = api.get_paths_info(repo, paths, revision=revision) if revision else api.get_paths_info(repo, paths)
        info = {f.path: f for f in got}
    except Exception as e:                                       # noqa: BLE001
        return [f"could not read the file list of {repo} back ({type(e).__name__}: {e})"]
    bad = []
    for rp, local in pairs:
        f = info.get(rp)
        rec = local if isinstance(local, dict) else None
        size = rec["size"] if rec else os.path.getsize(local)
        if f is None:
            bad.append(f"{rp}: not on the Hub")
            continue
        if getattr(f, "size", None) != size:
            bad.append(f"{rp}: {getattr(f, 'size', None)} bytes on the Hub, {size} expected")
            continue
        sha = _lfs_sha(f)
        if sha:
            want = rec["sha256"] if rec else sha256_file(local)
            if sha != want:
                bad.append(f"{rp}: sha256 differs")
        elif getattr(f, "blob_id", None) and not rec:
            if f.blob_id != git_blob_sha1(local):
                bad.append(f"{rp}: git blob id differs")
        elif getattr(f, "blob_id", None) and rec and rec.get("git_sha1"):
            if f.blob_id != rec["git_sha1"]:
                bad.append(f"{rp}: git blob id differs")
        else:
            bad.append(f"{rp}: the Hub gave no hash to check against")
    return bad
