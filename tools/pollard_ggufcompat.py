#!/usr/bin/env python3
"""pollard-ggufcheck — which runtime can actually load a GGUF we built?

A file's name does not decide this. Two things in its header do, and either one alone is enough to
make a file unloadable in stock llama.cpp:

  * **tensor types.** Stock rejects any ggml type id above 42 outright, so one protected tensor
    carrying an ik_llama-only atom makes the whole file ik_llama-only -- even when every other tensor
    is a plain K-quant and the filename says `IQ4_XS`. Easy to ship by accident, because the measured
    allocation is *supposed* to reach for a better atom on a sensitive tensor.
  * **the architecture string.** `general.architecture` must be one stock knows. A brand-new model
    whose support lives only in a vendor fork produces a file with perfectly ordinary K-quants that
    still cannot open anywhere else. Our K2-Horizon repos shipped exactly that: `k2-horizon` is not
    among upstream's 146 architectures, every tensor is stock, and all three cards said the K-quants
    run anywhere.

Either way the fix is the same -- say so on the card.

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

# Architectures upstream llama.cpp can load, snapshotted from LLM_ARCH_NAMES in
# runtime/llama.cpp/src/llama-arch.cpp (146 entries; `clip` dropped -- it is a quantize-only dummy).
# A vendored checkout is read in preference to this list when one is present, so a newer runtime is
# believed over the snapshot.
STOCK_ARCHS = {
    "afmoe", "apertus", "arcee", "arctic", "arwkv7", "baichuan", "bailingmoe", "bailingmoe2",
    "bailingmoe3", "bert", "bitnet", "bloom", "chameleon", "chatglm", "codeshell", "cogvlm",
    "cohere2", "cohere2moe", "command-r", "dbrx", "deci", "deepseek", "deepseek2",
    "deepseek2-ocr", "deepseek32", "deepseek4", "dflash", "dots1", "dots3note", "dream",
    "eagle3", "ernie4_5", "ernie4_5-moe", "eurobert", "exaone", "exaone-moe", "exaone4",
    "falcon", "falcon-h1", "gemma", "gemma-embedding", "gemma2", "gemma3", "gemma3n", "gemma4",
    "gemma4-assistant", "glm-dsa", "glm4", "glm4moe", "gpt-oss", "gpt2", "gptj", "gptneox",
    "granite", "granite_swa", "granitehybrid", "granitemoe", "graniteswitch", "grok", "grovemoe",
    "hunyuan-dense", "hunyuan-moe", "hunyuan_vl", "hy_v3", "internlm2", "jais", "jais2", "jamba",
    "jina-bert-v2", "jina-bert-v3", "kimi-k3", "kimi-linear", "laguna", "lfm2", "lfm2moe",
    "llada", "llada-moe", "llama", "llama-embed", "llama4", "maincoder", "mamba", "mamba2",
    "mellum", "mimo2", "minicpm", "minicpm3", "minimax-01", "minimax-m2", "minimax-m3",
    "mistral3", "mistral4", "modern-bert", "mpt", "muse-glimmer", "nanbeige", "nemotron",
    "nemotron_h", "nemotron_h_moe", "neo-bert", "nomic-bert", "nomic-bert-moe", "olmo", "olmo2",
    "olmoe", "openelm", "orion", "paddleocr", "pangu-embedded", "phi2", "phi3", "phimoe",
    "plamo", "plamo2", "plamo3", "plm", "pockettts", "qwen", "qwen2", "qwen2moe", "qwen2vl",
    "qwen3", "qwen35", "qwen35moe", "qwen3moe", "qwen3next", "qwen3tts", "qwen3vl", "qwen3vlmoe",
    "refact", "rnd1", "rwkv6", "rwkv6qwen2", "rwkv7", "seed_oss", "smallthinker", "smollm3",
    "stablelm", "starcoder", "starcoder2", "step35", "t5", "t5encoder", "talkie",
    "wavtokenizer-dec", "xverse"
}

# Architectures Pollard can build that stock cannot load, and where support actually lives.
# name, url -- kept apart so a card can render a link and a console line can render plain text.
FORK_ARCHS = {
    "k2-horizon": ("the MBZUAI-IFM llama.cpp fork", "https://github.com/MBZUAI-IFM/llama.cpp"),
}


def stock_archs(vendored="runtime/llama.cpp/src/llama-arch.cpp"):
    """The architecture names stock llama.cpp knows.

    Prefers a vendored checkout so a fresher runtime wins over the snapshot; falls back to the
    snapshot when the source is not there (an installed tool usually has no runtime beside it).
    """
    try:
        import re as _re
        src = open(vendored, encoding="utf-8").read()
        m = _re.search(r"LLM_ARCH_NAMES\s*=\s*\{(.*?)\n\};", src, _re.S)
        if m:
            got = set(_re.findall(r'"\s*([a-z0-9._\-]+)\s*"', m.group(1)))
            if len(got) > 50:                    # sanity: a real list, not a stray match
                return got - {"clip"}
    except OSError:
        pass
    return set(STOCK_ARCHS)


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

    def read_value(self, t):
        """Read a metadata value (strings only; everything else is skipped and returns None)."""
        if t == 8:
            return self.s()
        self.skip_value(t)
        return None

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


def read_header(path_or_url):
    """(architecture, {ggml type id: tensor count}) for a GGUF, local path or http(s) URL.

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
        arch = ""
        for _ in range(nkv):
            key = r.s()
            val = r.read_value(r.u32())
            if key == "general.architecture" and val:
                arch = val
        counts = {}
        for _ in range(ntensor):
            r.s()
            for _ in range(r.u32()):
                r.u64()                             # dims
            t = r.u32(); r.u64()                     # type, offset
            counts[t] = counts.get(t, 0) + 1
        return arch, counts
    finally:
        src.close()


def tensor_types(path_or_url):
    """{ggml type id: tensor count} -- kept for callers that only care about atoms."""
    return read_header(path_or_url)[1]


def fork_only(counts):
    """The fork-only atoms in a type histogram, as {name: tensor count}."""
    return {type_name(t): n for t, n in sorted(counts.items()) if t > STOCK_MAX}


def runtime_of(path_or_url, archs=None):
    """('stock'|'ik_llama'|'fork', reasons) -- which runtime this file needs, and why.

    `reasons` is a dict: fork-only atoms by name, plus an "architecture" key when
    `general.architecture` is one stock llama.cpp does not know. Either is disqualifying on its own,
    and the architecture is the one a tensor-type check alone cannot see.
    """
    arch, counts = read_header(path_or_url)
    reasons = fork_only(counts)
    known = archs if archs is not None else stock_archs()
    if arch and arch not in known:
        reasons = dict(reasons)
        reasons["architecture"] = arch
        where = FORK_ARCHS.get(arch)
        if where:
            reasons["supported_by"], reasons["supported_url"] = where
        return "fork", reasons
    return ("ik_llama" if reasons else "stock"), reasons


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
        rows.append({"file": name, "runtime": rt, "reasons": fo})
        if rt != "stock":
            rc = 1
        if not a.json:
            arch = fo.get("architecture")
            atoms = ", ".join(f"{k} x{v}" for k, v in fo.items()
                              if k not in ("architecture", "supported_by", "supported_url"))
            if arch:
                label = "needs a fork"
                bits = [f"architecture `{arch}` is not one stock llama.cpp knows"]
                if fo.get("supported_by"):
                    bits.append(f"supported by {fo['supported_by']} ({fo.get('supported_url','')})"
                                .replace(" ()", ""))
                if atoms:
                    bits.append(f"also fork-only atoms: {atoms}")
                why = "  [" + "; ".join(bits) + "]"
            elif atoms:
                label, why = "needs ik_llama", f"  [{atoms}]"
            else:
                label, why = "stock llama.cpp", ""
            print(f"  {label:<16}  {name}{why}")

    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        ik = [r for r in rows if r.get("runtime") == "ik_llama"]
        fk = [r for r in rows if r.get("runtime") == "fork"]
        bad = [r for r in rows if r.get("error")]
        print(f"\n{len(rows)} file(s): {len(rows)-len(ik)-len(fk)-len(bad)} stock, {len(ik)} ik_llama"
              + (f", {len(fk)} fork-only architecture" if fk else "")
              + (f", {len(bad)} unreadable" if bad else ""))
        if ik or fk:
            print("Say so on the card: these will not load in stock llama.cpp, Ollama or LM Studio.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
