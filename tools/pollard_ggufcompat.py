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
  pollard-ggufcheck --offline model.gguf            # never consult upstream (claims less)

Exit code is 1 if any file needs ik_llama, so this works as a pre-publish gate."""
import argparse
import json
import os
import struct
import sys

# Highest ggml type id stock llama.cpp knows. Anything above is a fork-only atom.
STOCK_MAX = 42

# Architectures upstream llama.cpp can load, snapshotted from LLM_ARCH_NAMES on ggml-org master,
# 2026-09-10 (149 entries; `clip` dropped -- it is a quantize-only dummy).
#
# A SNAPSHOT GOES STALE, and stale in the dangerous direction: a new architecture that upstream has
# merged looks fork-only, which is a claim about someone else's runtime. That already happened here --
# a list taken from the vendored runtime was three entries behind master and reported Spark-X2.5-4B as
# fork-only when upstream had merged `spark2_5` days earlier. So a miss against this list is not the
# answer; it is the trigger to go and ask, which `arch_support()` does.
STOCK_ARCHS = {
    "afmoe", "apertus", "arcee", "arctic", "arwkv7", "baichuan", "bailingmoe", "bailingmoe2",
    "bailingmoe3", "bert", "bitnet", "bloom", "chameleon", "chatglm", "codeshell", "cogvlm",
    "cohere2", "cohere2moe", "command-r", "dbrx", "deci", "deepseek", "deepseek2",
    "deepseek2-ocr", "deepseek32", "deepseek4", "dflash", "dots1", "dots3note", "dream",
    "eagle3", "ernie4_5", "ernie4_5-moe", "eurobert", "exaone", "exaone-moe", "exaone4",
    "falcon", "falcon-h1", "gemma", "gemma-embedding", "gemma2", "gemma3", "gemma3n", "gemma4",
    "gemma4-assistant", "glm-dsa", "glm4", "glm4moe", "gpt-oss", "gpt2", "gptj", "gptneox",
    "granite", "granite_swa", "granitehybrid", "granitemoe", "graniteswitch", "grok", "grovemoe",
    "hunyuan-dense", "hunyuan-moe", "hunyuan_vl", "hy_v3", "hy_v4", "internlm2", "jais", "jais2",
    "jamba", "jina-bert-v2", "jina-bert-v3", "kimi-k3", "kimi-linear", "laguna", "lfm2",
    "lfm2moe", "llada", "llada-moe", "llama", "llama-embed", "llama4", "maincoder", "mamba",
    "mamba2", "mellum", "mimo2", "minicpm", "minicpm3", "minimax-01", "minimax-m2", "minimax-m3",
    "mistral3", "mistral4", "modern-bert", "mpt", "muse-glimmer", "nanbeige", "nemotron",
    "nemotron_h", "nemotron_h_moe", "neo-bert", "nomic-bert", "nomic-bert-moe", "olmo", "olmo2",
    "olmoe", "openelm", "orion", "paddleocr", "pangu-embedded", "phi2", "phi3", "phimoe",
    "plamo", "plamo2", "plamo3", "plm", "pockettts", "qwen", "qwen2", "qwen2moe", "qwen2vl",
    "qwen3", "qwen35", "qwen35moe", "qwen3moe", "qwen3next", "qwen3tts", "qwen3vl", "qwen3vlmoe",
    "qwen4exp", "refact", "rnd1", "rwkv6", "rwkv6qwen2", "rwkv7", "seed_oss", "smallthinker",
    "smollm3", "spark2_5", "stablelm", "starcoder", "starcoder2", "step35", "t5", "t5encoder",
    "talkie", "wavtokenizer-dec", "xverse"
}

# Architectures Pollard can build that stock cannot load, and where support actually lives.
# name, url -- kept apart so a card can render a link and a console line can render plain text.
FORK_ARCHS = {
    "k2-horizon": ("the MBZUAI-IFM llama.cpp fork", "https://github.com/MBZUAI-IFM/llama.cpp"),
}


def stock_archs(vendored="runtime/llama.cpp/src/llama-arch.cpp"):
    """The architecture names the llama.cpp BESIDE US knows.

    Prefers a vendored checkout, because that is the runtime this machine would actually build and run
    with. Falls back to the snapshot when there is no source to read (an installed tool usually has no
    runtime next to it). This is a local fact, not a claim about upstream -- see arch_support().
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


_UPSTREAM_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp"
_upstream_cache = None


def upstream_archs(offline=False):
    """What ggml-org master implements right now, or None if it could not be read.

    None matters: it means "unknown", and an unknown must never be reported as a fork-only
    architecture. Cached per process -- one fetch however many files are checked.
    """
    global _upstream_cache
    if offline:
        return None
    if _upstream_cache is not None:
        return _upstream_cache or None
    try:
        import re as _re
        import urllib.request
        req = urllib.request.Request(_UPSTREAM_URL, headers={"User-Agent": "pollard-ggufcheck"})
        with urllib.request.urlopen(req, timeout=45) as fh:
            src = fh.read().decode("utf-8", "replace")
        m = _re.search(r"LLM_ARCH_NAMES\s*=\s*\{(.*?)\n\};", src, _re.S)
        got = set(_re.findall(r'"\s*([a-z0-9._\-]+)\s*"', m.group(1))) - {"clip"} if m else set()
        _upstream_cache = got if len(got) > 50 else set()
    except Exception:
        _upstream_cache = set()
    return _upstream_cache or None


def arch_support(arch, local=None, offline=False):
    """('stock'|'newer'|'fork'|'unknown', detail) for an architecture name.

    Three different answers that a single list cannot tell apart, and conflating them makes a false
    claim about someone else's runtime:

      stock   the llama.cpp here already knows it
      newer   upstream has merged it, the runtime here is behind -- the user needs a newer llama.cpp,
              not a fork
      fork    upstream does not implement it at all; support lives somewhere specific
      unknown the local list misses it and upstream could not be consulted -- say so, claim nothing
    """
    known = stock_archs() if local is None else local
    if not arch or arch in known:
        return "stock", ""
    up = upstream_archs(offline=offline)
    if up is None:
        return "unknown", ("not in the llama.cpp here, and upstream could not be checked "
                           "(offline?) -- no claim made")
    if arch in up:
        return "newer", "upstream llama.cpp implements it; the build here is older"
    where = FORK_ARCHS.get(arch)
    return "fork", (where[0] if where else "no upstream support and no fork on record")


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


# What a non-CUDA/Metal BACKEND will accept, and what it silently does to the rest.
#
# "Which runtime opens this file" is only half the question. A backend can open a file and then change
# it: OpenVINO's NPU path requantizes Q6_K to Q4_0_128, which discards a measured allocation while
# reporting success. That is the same failure shape as a fork-only atom -- the file works, and is not
# what the card promised.
#
# Source: llama.cpp docs/backend/OPENVINO.md. `accepts` is the documented scheme list; `rewrites` maps
# an accepted type to what the backend turns it into on that device.
BACKENDS = {
    "openvino-cpu": {
        "label": "OpenVINO (Intel CPU)",
        "accepts": {"F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q4_K", "Q5_K", "Q6_K"},
        "rewrites": {"Q5_K": "Q8_0_C", "Q6_K": "Q8_0_C"},
        "note": "BF16 is Xeon-only",
    },
    "openvino-gpu": {
        "label": "OpenVINO (Intel GPU)",
        "accepts": {"F16", "Q8_0", "Q4_0", "Q4_1", "Q4_K", "Q5_K", "Q6_K"},
        "rewrites": {"Q5_K": "Q8_0_C", "Q6_K": "Q8_0_C"},
        "note": "",
    },
    "openvino-npu": {
        "label": "OpenVINO (Intel NPU)",
        "accepts": {"F16", "Q8_0", "Q4_0", "Q4_1", "Q4_K", "Q5_K", "Q6_K"},
        "rewrites": {"Q6_K": "Q4_0_128", "Q5_K": "Q4_0_128"},
        "note": "Q4_0 is the primary scheme; embedding Q6_K goes to Q8_0_C and the token embedding "
                "is dequantized to fp16",
    },
}


def backend_report(counts):
    """{backend key: (verdict, detail)} for a tensor-type histogram.

    verdict is 'ok' (every type accepted and kept as-is), 'rewritten' (accepted but the backend
    changes some tensors) or 'unsupported' (a type the backend does not take at all).
    """
    out = {}
    present = {type_name(t) for t in counts}
    for key, spec in BACKENDS.items():
        # F32 is always fine: it is the accumulate/norm type, not a quantization scheme
        quant = {n for n in present if n != "F32"}
        missing = sorted(n for n in quant if n not in spec["accepts"])
        if missing:
            out[key] = ("unsupported", f"does not accept {', '.join(missing)}")
            continue
        changed = sorted(f"{n} -> {spec['rewrites'][n]}" for n in quant if n in spec["rewrites"])
        if changed:
            out[key] = ("rewritten", "; ".join(changed))
        else:
            out[key] = ("ok", "")
    return out


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


def runtime_of(path_or_url, archs=None, offline=False):
    """('stock'|'ik_llama'|'newer'|'fork'|'unknown', reasons) -- what this file needs, and why.

    Two independent disqualifiers live in a GGUF header and a tensor-type check sees only one:

      * a fork-only atom (any ggml type above 42)      -> 'ik_llama'
      * an architecture the local llama.cpp lacks       -> resolved by arch_support(), which
        separates "upstream merged it, you are behind" from "no upstream support at all"

    The architecture answer takes precedence when it is disqualifying, because no quant type can
    rescue a file whose architecture will not load.
    """
    arch, counts = read_header(path_or_url)
    reasons = fork_only(counts)
    verdict, detail = arch_support(arch, local=archs, offline=offline)
    if verdict != "stock":
        reasons = dict(reasons)
        reasons["architecture"] = arch
        reasons["arch_detail"] = detail
        if verdict == "fork":
            where = FORK_ARCHS.get(arch)
            if where:
                reasons["supported_by"], reasons["supported_url"] = where
        return verdict, reasons
    return ("ik_llama" if reasons else "stock"), reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("files", nargs="*", help="GGUF paths (or URLs)")
    ap.add_argument("--repo", help="check every .gguf in a published HF repo instead")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--backends", action="store_true",
                    help="also report what non-CUDA/Metal backends do with this file. A backend can "
                         "ACCEPT a file and still change it -- OpenVINO's NPU path requantizes Q6_K "
                         "to Q4_0_128, which discards a measured allocation while reporting success")
    ap.add_argument("--offline", action="store_true",
                    help="do not ask upstream what it implements. An architecture missing from the "
                         "local list then reports as unverified instead of being called fork-only -- "
                         "a stale list must not turn into a claim about someone else's runtime")
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
            rt, fo = runtime_of(t, offline=a.offline)
        except Exception as e:                                            # noqa: BLE001
            rows.append({"file": name, "error": str(e)})
            rc = 1
            if not a.json:
                print(f"  UNREADABLE  {name}\n              {e}")
            continue
        row = {"file": name, "runtime": rt, "reasons": fo}
        if a.backends:
            try:
                row["backends"] = backend_report(read_header(t)[1])
            except Exception:                                              # noqa: BLE001
                row["backends"] = {}
        rows.append(row)
        if rt != "stock":
            rc = 1                     # 'newer' counts: the runtime here still cannot open it
        if not a.json:
            arch = fo.get("architecture")
            atoms = ", ".join(f"{k} x{v}" for k, v in fo.items()
                              if k not in ("architecture", "supported_by", "supported_url",
                                           "arch_detail"))
            if arch:
                label = {"newer": "needs newer llama.cpp", "fork": "needs a fork",
                         "unknown": "cannot verify"}.get(rt, "needs a fork")
                bits = [f"architecture `{arch}`: {fo.get('arch_detail','')}".rstrip(": ")]
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
            for key, (verdict, detail) in (row.get("backends") or {}).items():
                mark = {"ok": "loads as built", "rewritten": "REWRITTEN",
                        "unsupported": "UNSUPPORTED"}[verdict]
                line = f"      {BACKENDS[key]['label']:<24} {mark}"
                print(line + (f"  ({detail})" if detail else ""))

    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        by = {}
        for r in rows:
            by.setdefault(r.get("runtime") or "error", []).append(r)
        parts = [f"{len(v)} {k}" for k, v in sorted(by.items())]
        print(f"\n{len(rows)} file(s): " + ", ".join(parts))
        if by.get("newer"):
            print("Say so on the card: these need a llama.cpp new enough to carry the architecture. "
                  "That is a version requirement, NOT a fork.")
        if by.get("ik_llama") or by.get("fork"):
            print("Say so on the card: these will not load in stock llama.cpp, Ollama or LM Studio.")
        if by.get("unknown"):
            print("Could not reach upstream to confirm an unrecognised architecture; nothing claimed.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
