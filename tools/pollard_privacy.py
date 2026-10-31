#!/usr/bin/env python3
"""pollard-privacy -- nothing Pollard publishes carries the publisher's local paths.

llama-quantize writes `quantize.imatrix.file` and `quantize.imatrix.dataset` into every GGUF it builds with an imatrix,
as whatever paths it was handed; llama-imatrix stores the calibration file's path inside the .imatrix; calibration and
sensitivity sidecars record the folders they read. Published as they are, every repo shows the builder's home folder and
user name (PollardWeights' did, 2026-10-31). This rewrites each local path to its file name before anything is uploaded,
and refuses an upload that still carries one.

    pollard-privacy check <file-or-folder>...     # list what would leak (exit 1 if anything does)
    pollard-privacy scrub <file-or-folder>...     # rewrite in place, then check

GGUF: only the header is rebuilt -- metadata strings shortened, tensor infos byte-for-byte, the tensor data streamed
across untouched -- so it works for every quant type (ik_llama's included) and for each shard of a split model.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys

#: an absolute local path: a home / scratch / mount root on macOS, Linux or Windows, and everything up to a delimiter
_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|(?<![\w.:/])/)(?:Users|home|root|private|var|tmp|Volumes|mnt|opt|srv|"
                      r"workspace|content|data|scratch|Documents and Settings)[\\/][^\s\"'<>|,;)\]}]*")
TEXT_EXT = (".json", ".jsonl", ".md", ".yaml", ".yml", ".txt", ".jinja", ".csv", ".tsv")
_BIG_TEXT = 64 << 20                 # bigger text files (a calibration corpus) are only searched for the home folder


def _home_strings():
    out = set()
    for h in (os.path.expanduser("~"), os.environ.get("USERPROFILE"), os.environ.get("HOME")):
        if h and len(h) > 3:
            out.add(h)
            out.add(h.replace("\\", "/"))
    return sorted(out, key=len, reverse=True)


def _base(p):
    parts = [x for x in re.split(r"[\\/]+", p) if x]
    return parts[-1] if parts else p


def scrub_text(s):
    """`s` with every local path cut to its file name (and any remaining home-folder prefix dropped)."""
    s = _PATH_RE.sub(lambda m: _base(m.group(0)), s)
    for h in _home_strings():
        s = s.replace(h + "/", "").replace(h + "\\", "").replace(h, "~")
    return s


def leaks_in_text(s):
    found = [m.group(0) for m in _PATH_RE.finditer(s)]
    found += [h for h in _home_strings() if h in s]
    return sorted(set(found))


# ------------------------------------------------------------------------------------------------ GGUF header

_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class _Reader:
    def __init__(self, f):
        self.f = f

    def take(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF header")
        return b

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64())


def _str(b):
    return struct.pack("<Q", len(b)) + b


def _read_value(r, vt):
    """(raw bytes of the value, [decoded strings inside it] , rebuild(fn) -> new raw bytes)."""
    if vt in _FIXED:
        raw = r.take(_FIXED[vt])
        return raw, [], None, raw
    if vt == 8:
        s = r.string()
        return _str(s), [s], "str", s
    if vt == 9:
        et, n = r.u32(), r.u64()
        head = struct.pack("<IQ", et, n)
        if et in _FIXED:
            return head + r.take(_FIXED[et] * n), [], None, None
        if et == 8:
            items = [r.string() for _ in range(n)]
            return head + b"".join(_str(x) for x in items), items, "arr", items
        parts = [_read_value(r, et)[0] for _ in range(n)]
        return head + b"".join(parts), [], None, None
    raise ValueError(f"unknown GGUF value type {vt}")


def _gguf_header(path):
    """(version, n_tensors, [(key, vtype, raw, strings, kind, payload)], tensor_info_bytes, alignment, data_start)."""
    with open(path, "rb") as f:
        r = _Reader(f)
        if r.take(4) != b"GGUF":
            return None
        ver, n_t, n_kv = r.u32(), r.u64(), r.u64()
        kvs, align = [], 32
        for _ in range(n_kv):
            key = r.string()
            vt = r.u32()
            raw, strs, kind, payload = _read_value(r, vt)
            if key == b"general.alignment" and vt == 4:
                align = struct.unpack("<I", raw)[0] or 32
            kvs.append((key, vt, raw, strs, kind, payload))
        ti_start = f.tell()
        for _ in range(n_t):
            r.string()
            nd = r.u32()
            r.take(8 * nd + 4 + 8)
        ti_end = f.tell()
        f.seek(ti_start)
        ti = f.read(ti_end - ti_start)
    data_start = -(-ti_end // align) * align
    return ver, n_t, kvs, ti, align, data_start


def _raw_leaks(path, n=4 << 20):
    """A file whose header cannot be parsed: search its first bytes for a local path instead."""
    with open(path, "rb") as f:
        blob = f.read(n).decode("latin-1")
    return [("raw", x) for x in leaks_in_text(blob)]


def gguf_leaks(path):
    try:
        h = _gguf_header(path)
    except (ValueError, struct.error, OverflowError, MemoryError):
        return _raw_leaks(path)
    if not h:
        return []
    out = []
    for key, vt, raw, strs, kind, payload in h[2]:
        for s in strs:
            t = s.decode("utf-8", "replace")
            if leaks_in_text(t):
                out.append((key.decode(), t[:200]))
    return out


def scrub_gguf(path):
    """Rewrite the header with local paths cut to file names; the tensor data is copied across untouched."""
    try:
        h = _gguf_header(path)
    except (ValueError, struct.error, OverflowError, MemoryError):
        return False                                    # not a header we can rebuild: check() reports any leak
    if not h:
        return False
    ver, n_t, kvs, ti, align, data_start = h
    changed = False
    body = [b"GGUF", struct.pack("<IQQ", ver, n_t, len(kvs))]
    for key, vt, raw, strs, kind, payload in kvs:
        if kind == "str" and leaks_in_text(payload.decode("utf-8", "replace")):
            raw = _str(scrub_text(payload.decode("utf-8", "replace")).encode())
            changed = True
        elif kind == "arr" and any(leaks_in_text(x.decode("utf-8", "replace")) for x in payload):
            items = [scrub_text(x.decode("utf-8", "replace")).encode() for x in payload]
            raw = struct.pack("<IQ", 8, len(items)) + b"".join(_str(x) for x in items)
            changed = True
        body.append(_str(key) + struct.pack("<I", vt) + raw)
    if not changed:
        return False
    head = b"".join(body) + ti
    head += b"\x00" * ((-len(head)) % align)
    tmp = path + ".pollard-privacy.tmp"
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        dst.write(head)
        src.seek(data_start)
        shutil.copyfileobj(src, dst, 64 << 20)
    shutil.copystat(path, tmp)
    os.replace(tmp, path)
    return True


# ------------------------------------------------------------------------------------------------ legacy .imatrix

def _imatrix_tail(path):
    """(offset of the trailing dataset record, dataset bytes) of a legacy llama-imatrix file, or None."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        r = _Reader(f)
        try:
            n = struct.unpack("<i", r.take(4))[0]
            if not 0 < n < 1_000_000:
                return None
            for _ in range(n):
                ln = struct.unpack("<i", r.take(4))[0]
                if not 0 < ln < 4096:
                    return None
                r.take(ln + 4)                                      # name + ncall
                nval = struct.unpack("<i", r.take(4))[0]
                f.seek(4 * nval, 1)
            if f.tell() + 4 > size:
                return None
            r.take(4)                                               # last_call
            off = f.tell()
            if off + 4 > size:
                return off, b""
            ln = struct.unpack("<i", r.take(4))[0]
            return off, (f.read(ln) if 0 <= ln <= size - off - 4 else b"")
        except (ValueError, struct.error):
            return None


def _is_gguf(path):
    with open(path, "rb") as f:
        return f.read(4) == b"GGUF"


def imatrix_leaks(path):
    if _is_gguf(path):
        return gguf_leaks(path)
    t = _imatrix_tail(path)
    return [("imatrix.dataset", t[1].decode("utf-8", "replace"))] if t and leaks_in_text(
        t[1].decode("utf-8", "replace")) else []


def scrub_imatrix(path):
    if _is_gguf(path):
        return scrub_gguf(path)
    t = _imatrix_tail(path)
    if not t or not leaks_in_text(t[1].decode("utf-8", "replace")):
        return False
    new = scrub_text(t[1].decode("utf-8", "replace")).encode()
    with open(path, "r+b") as f:
        f.seek(t[0])
        f.write(struct.pack("<i", len(new)) + new)
        f.truncate()
    return True


# ------------------------------------------------------------------------------------------------ files and trees

def file_leaks(path):
    low = path.lower()
    try:
        if low.endswith(".gguf"):
            return gguf_leaks(path)
        if low.endswith(".imatrix") or low.endswith(".dat") and _imatrix_tail(path):
            return imatrix_leaks(path)
        if low.endswith(TEXT_EXT):
            if os.path.getsize(path) > _BIG_TEXT:
                with open(path, "rb") as f:
                    blob = f.read().decode("utf-8", "replace")
                return [(path, h) for h in _home_strings() if h in blob]
            s = open(path, encoding="utf-8", errors="replace").read()
            return [(os.path.basename(path), x) for x in leaks_in_text(s)]
    except OSError:
        return []
    return []


def scrub_file(path):
    low = path.lower()
    if low.endswith(".gguf"):
        return scrub_gguf(path)
    if low.endswith(".imatrix"):
        return scrub_imatrix(path)
    if low.endswith(TEXT_EXT):
        s = open(path, encoding="utf-8", errors="replace").read()
        if os.path.getsize(path) > _BIG_TEXT:
            new = s
            for h in _home_strings():
                new = new.replace(h + "/", "").replace(h + "\\", "").replace(h, "~")
        else:
            new = scrub_text(s)
        if new != s:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
            return True
    return False


def _walk(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for n in files:
                    yield os.path.join(root, n)
        elif os.path.isfile(p):
            yield p


def scrub(paths, log=print):
    """Scrub every file under `paths`; returns the leaks still left afterwards ({file: [(where, what)]})."""
    for p in _walk(paths):
        try:
            if scrub_file(p):
                log(f"   privacy: local paths removed from {os.path.basename(p)}")
        except (OSError, ValueError) as e:
            log(f"   privacy: could not scrub {os.path.basename(p)} ({e})")
    return check(paths)


def check(paths):
    return {p: lk for p in _walk(paths) if (lk := file_leaks(p))}


class LocalPathLeak(RuntimeError):
    pass


def guard(paths, log=print):
    """Scrub, then refuse (LocalPathLeak) if anything still names a local path."""
    left = scrub(paths, log)
    if left:
        first = next(iter(left.items()))
        raise LocalPathLeak(f"refusing to publish: {os.path.basename(first[0])} still carries a local path "
                            f"({first[1][0][0]}: {first[1][0][1][:120]})")


class SafeApi:
    """huggingface_hub.HfApi (looked up when created, so a stub stands in for it the same way) whose uploads scrub local
    paths from what they send and refuse what still carries one; every other call passes straight through."""

    def __init__(self, *args, **kw):
        import huggingface_hub
        self._api = huggingface_hub.HfApi(*args, **kw)

    def __getattr__(self, name):
        return getattr(self._api, name)

    def upload_file(self, *args, path_or_fileobj=None, **kw):
        if isinstance(path_or_fileobj, (str, os.PathLike)) and os.path.isfile(path_or_fileobj):
            guard([str(path_or_fileobj)])
        return self._api.upload_file(*args, path_or_fileobj=path_or_fileobj, **kw)

    def upload_folder(self, *args, folder_path=None, **kw):
        if folder_path and os.path.isdir(folder_path):
            guard([str(folder_path)])
        return self._api.upload_folder(*args, folder_path=folder_path, **kw)

    def upload_large_folder(self, *args, folder_path=None, **kw):
        if folder_path and os.path.isdir(folder_path):
            guard([str(folder_path)])
        return self._api.upload_large_folder(*args, folder_path=folder_path, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pollard-privacy", description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("check", "scrub"))
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args(argv)
    left = scrub(a.paths) if a.cmd == "scrub" else check(a.paths)
    for p, lk in left.items():
        for where, what in lk:
            print(f"LEAK  {p}  {where}: {what[:160]}")
    print("clean" if not left else f"{len(left)} file(s) carry local paths")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
