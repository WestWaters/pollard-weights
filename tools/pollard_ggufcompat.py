#!/usr/bin/env python3
"""pollard-ggufcheck — which runtime can actually load a GGUF we built?

A file's name does not decide this; its tensor types do. Stock llama.cpp rejects any ggml type id
above 42 outright, so one protected tensor carrying an ik_llama-only atom makes the whole file
ik_llama-only -- even when every other tensor is a plain K-quant and the filename says `IQ4_XS`.

That is easy to ship by accident, because the measured allocation is *supposed* to reach for a
better atom on a sensitive tensor. It just has to be said on the card.

  pollard-ggufcheck model.gguf [more.gguf ...]      # local files
  pollard-ggufcheck --repo PollardWeights/<Model>   # a published repo, over range requests
  pollard-ggufcheck --json *.gguf                   # machine-readable

Exit code is 1 if any file needs ik_llama, so this works as a pre-publish gate."""
import argparse
import json
import os
import struct
import sys

# Highest ggml type id stock llama.cpp knows. Anything above is a fork-only atom.
STOCK_MAX = 42

STOCK_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
               10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
               16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
               22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
               29: "IQ1_M", 30: "BF16", 31: "TQ1_0", 32: "TQ2_0"}

# ik_llama.cpp's extra atoms (ggml/include/ggml.h). Named so a card can say which one is in play.
IK_NAMES = {133: "Q6_0", 134: "IQ1_BN", 135: "IQ2_BN", 136: "Q8_K64", 137: "IQ2_K", 138: "IQ3_K",
            139: "IQ4_K", 140: "IQ5_K", 141: "IQ6_K", 144: "IQ4_KS", 145: "IQ2_KS", 146: "IQ4_KSS",
            147: "Q8_K16", 148: "Q8_K32", 149: "Q8_KR8", 150: "Q8_K128", 151: "Q8_KV",
            152: "IQ5_KS", 153: "IQ2_KT", 154: "IQ3_KT", 155: "IQ4_KT", 156: "IQ3_KS",
            157: "IQ2_KL", 158: "IQ1_KT"}


def type_name(t):
    return STOCK_NAMES.get(t) or IK_NAMES.get(t) or f"type{t}"


class _Local:
    """Byte source for a file on disk."""

    def __init__(self, path):
        self.fh = open(path, "rb")

    def take(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError("file ended inside the header")
        return b

    def close(self):
        self.fh.close()


class _Remote:
    """Byte source for a URL, pulled in range-request windows.

    Only the header, metadata and tensor table are ever fetched, so checking a 20 GB file costs a
    few MB.
    """

    def __init__(self, url, window=1 << 22):
        import urllib.request
        self._req = urllib.request
        self.url, self.window = url, window
        self.buf, self.pos = b"", 0

    def _fill(self, upto):
        while len(self.buf) < upto:
            end = max(upto, len(self.buf) + self.window) - 1
            r = self._req.Request(self.url, headers={"Range": f"bytes={len(self.buf)}-{end}",
                                                    "User-Agent": "pollard-ggufcheck"})
            with self._req.urlopen(r, timeout=90) as fh:
                got = fh.read()
            if not got:
                raise EOFError("range request returned nothing")
            self.buf += got

    def take(self, n):
        self._fill(self.pos + n)
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def close(self):
        pass


class _Rdr:
    """The little slice of GGUF needed to walk to the tensor table."""

    def __init__(self, src):
        self.src = src

    def take(self, n): return self.src.take(n)
    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def s(self): return self.take(self.u64()).decode("utf-8", "replace")

    def skip_value(self, t):
        fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t == 8:
            self.s(); return
        if t == 9:
            et = self.u32(); n = self.u64()
            if et == 8:
                for _ in range(n):
                    self.s()
            elif et == 9:
                raise ValueError("nested arrays are not expected in a GGUF header")
            else:
                self.take(fixed[et] * n)
            return
        self.take(fixed[t])


def tensor_types(path_or_url):
    """{ggml type id: tensor count} for a GGUF, local path or http(s) URL.

    Raises ValueError when the magic is wrong -- which is itself worth catching, since a file whose
    header is missing will not load anywhere no matter what the card says.
    """
    remote = str(path_or_url).startswith(("http://", "https://"))
    src = _Remote(path_or_url) if remote else _Local(path_or_url)
    try:
        r = _Rdr(src)
        magic = r.take(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF: magic is {magic!r}, expected b'GGUF'")
        r.u32()                                     # version
        ntensor, nkv = r.u64(), r.u64()
        for _ in range(nkv):
            r.s(); r.skip_value(r.u32())
        counts = {}
        for _ in range(ntensor):
            r.s()
            for _ in range(r.u32()):
                r.u64()                             # dims
            t = r.u32(); r.u64()                     # type, offset
            counts[t] = counts.get(t, 0) + 1
        return counts
    finally:
        src.close()


def fork_only(counts):
    """The fork-only atoms in a type histogram, as {name: tensor count}."""
    return {type_name(t): n for t, n in sorted(counts.items()) if t > STOCK_MAX}


def runtime_of(path_or_url):
    """('stock'|'ik_llama', {atom: count}) -- which runtime this file needs, and why."""
    fo = fork_only(tensor_types(path_or_url))
    return ("ik_llama" if fo else "stock"), fo


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("files", nargs="*", help="GGUF paths (or URLs)")
    ap.add_argument("--repo", help="check every .gguf in a published HF repo instead")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()

    targets = list(a.files)
    if a.repo:
        import urllib.request
        url = f"https://huggingface.co/api/models/{a.repo}"
        with urllib.request.urlopen(urllib.request.Request(
                url, headers={"User-Agent": "pollard-ggufcheck"}), timeout=60) as fh:
            info = json.load(fh)
        targets += [f"https://huggingface.co/{a.repo}/resolve/main/{s['rfilename']}"
                    for s in info.get("siblings", []) if s["rfilename"].endswith(".gguf")]
    if not targets:
        ap.error("give some GGUF files or --repo")

    rows, rc = [], 0
    for t in targets:
        name = os.path.basename(str(t).split("?")[0])
        try:
            rt, fo = runtime_of(t)
        except Exception as e:                                            # noqa: BLE001
            rows.append({"file": name, "error": str(e)})
            rc = 1
            if not a.json:
                print(f"  UNREADABLE  {name}\n              {e}")
            continue
        rows.append({"file": name, "runtime": rt, "fork_only": fo})
        if rt == "ik_llama":
            rc = 1
        if not a.json:
            why = "  [" + ", ".join(f"{k} x{v}" for k, v in fo.items()) + "]" if fo else ""
            print(f"  {'needs ik_llama' if fo else 'stock llama.cpp':<16}  {name}{why}")

    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        need = [r for r in rows if r.get("runtime") == "ik_llama"]
        bad = [r for r in rows if r.get("error")]
        print(f"\n{len(rows)} file(s): {len(rows)-len(need)-len(bad)} stock, {len(need)} ik_llama"
              + (f", {len(bad)} unreadable" if bad else ""))
        if need:
            print("Say so on the card: these will not load in stock llama.cpp, Ollama or LM Studio.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
