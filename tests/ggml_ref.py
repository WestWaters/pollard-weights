"""The real ggml, through ctypes, as a reference for tests: quantize(W, imatrix) and dequant(bytes) from
the built libggml-base. Used by the passthrough requant check and the pollard_fq bit-exact gate.

Finds the library in $POLLARD_GGML_LIB (a file) / $POLLARD_LLAMA_BIN_DIR (a build/bin dir), then the
managed runtime (runtime/llama.cpp/build/bin). `load()` returns None when no library is present, so a
test can skip cleanly in CI."""
from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ggml_type enum values (ggml.h)
TYPES = {"f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q5_0": 6, "q5_1": 7, "q8_0": 8, "q2_k": 10, "q3_k": 11,
         "q4_k": 12, "q5_k": 13, "q6_k": 14, "iq2_xxs": 16, "iq2_xs": 17, "iq3_xxs": 18, "iq1_s": 19,
         "iq4_nl": 20, "iq3_s": 21, "iq2_s": 22, "iq4_xs": 23, "iq1_m": 29, "bf16": 30, "tq1_0": 34,
         "tq2_0": 35, "mxfp4": 39, "nvfp4": 40}


class _Traits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64),
                ("blck_size_interleave", ctypes.c_int64), ("type_size", ctypes.c_size_t),
                ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]


_TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)


def _lib_names(d):
    pats = ("libggml-base.dylib", "libggml-base.so", "ggml-base.dll", "libggml-base.so.*", "libggml-base.*.dylib")
    for p in pats:
        hits = sorted(glob.glob(os.path.join(d, p)))
        if hits:
            return hits[0]
    return None


def candidates():
    env = os.environ.get("POLLARD_GGML_LIB")
    if env:
        yield env
    for d in (os.environ.get("POLLARD_LLAMA_BIN_DIR"), str(ROOT / "runtime" / "llama.cpp" / "build" / "bin")):
        if d and os.path.isdir(d):
            p = _lib_names(d)
            if p:
                yield p


class GGML:
    def __init__(self, path):
        self.path = path
        lib = ctypes.CDLL(path)
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        lib.ggml_row_size.restype = ctypes.c_size_t
        lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        lib.ggml_get_type_traits.restype = ctypes.POINTER(_Traits)
        lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        self.lib = lib

    def type_id(self, t):
        return TYPES[str(t).lower()]

    def has(self, t):
        try:
            tr = self.lib.ggml_get_type_traits(self.type_id(t)).contents
            return bool(tr.type_name) and bool(tr.to_float)
        except Exception:
            return False

    def quantize(self, t, x, imatrix=None) -> bytes:
        """ggml_quantize_chunk over a [rows, n_per_row] float32 matrix (imatrix: [n_per_row] or None)."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        rows, n = x.shape
        tid = self.type_id(t)
        out = ctypes.create_string_buffer(self.lib.ggml_row_size(tid, n) * rows)
        im = None
        if imatrix is not None:
            imatrix = np.ascontiguousarray(imatrix, dtype=np.float32)
            im = imatrix.ctypes.data
        size = self.lib.ggml_quantize_chunk(tid, x.ctypes.data, out, 0, rows, n, im)
        return out.raw[:size]

    def dequant(self, t, data: bytes, rows, n) -> np.ndarray:
        tid = self.type_id(t)
        tr = self.lib.ggml_get_type_traits(tid).contents
        fn = _TO_FLOAT(tr.to_float)
        src = ctypes.create_string_buffer(bytes(data), len(data))
        y = np.empty((rows, n), dtype=np.float32)
        fn(ctypes.cast(src, ctypes.c_void_p), y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), rows * n)
        return y


def load():
    for p in candidates():
        try:
            return GGML(p)
        except OSError:
            continue
    return None


def llama_bin(name):
    """A llama.cpp tool next to the library (llama-quantize), or None."""
    dirs = [os.environ.get("POLLARD_LLAMA_BIN_DIR"), str(ROOT / "runtime" / "llama.cpp" / "build" / "bin")]
    for d in dirs:
        if not d:
            continue
        for n in (name, name + ".exe"):
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    return None


# ------------------------------------------------------------------------------------------------
# STQ1_0 lives only in Pollard's patched tree (runtime-patches/llama-stq-stq1_0-quants.patch), whose
# prebuilt libggml can lag its source. The reference is compiled from that tree's ggml-quants.c with the
# same -O3 the runtime uses, cached by source hash; skipped when no tree or no C compiler is present.
# ------------------------------------------------------------------------------------------------
STQ_TREES = ("~/pollard-stq/llama.cpp", "C:/pollard/llama-stq")
STQ_ROW = {"qs": 32, "sign": 8, "d": 2}


def _stq_tree():
    for c in [os.environ.get("POLLARD_STQ_SRC")] + list(STQ_TREES):
        if c and os.path.isfile(os.path.join(os.path.expanduser(c), "ggml", "src", "ggml-quants.c")):
            src = os.path.join(os.path.expanduser(c), "ggml", "src", "ggml-quants.c")
            if "quantize_row_stq1_0_imatrix" in open(src, encoding="utf-8", errors="replace").read():
                return os.path.expanduser(c)
    return None


class STQRef:
    """quantize_stq1_0 / dequantize_row_stq1_0 from a compiled copy of the patched ggml-quants.c."""
    row_bytes = 42                                     # qs[32] + sign[8] + fp16 d per 256 weights

    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        self.lib.quantize_stq1_0.restype = ctypes.c_size_t
        self.lib.quantize_stq1_0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                             ctypes.c_void_p]
        self.lib.dequantize_row_stq1_0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

    def quantize(self, x, imatrix=None):
        x = np.ascontiguousarray(x, dtype=np.float32)
        rows, n = x.shape
        out = ctypes.create_string_buffer(rows * (n // 256) * self.row_bytes)
        im = None if imatrix is None else np.ascontiguousarray(imatrix, dtype=np.float32)
        self.lib.quantize_stq1_0(x.ctypes.data, out, rows, n, None if im is None else im.ctypes.data)
        return out.raw

    def dequant(self, data, rows, n):
        y = np.empty((rows, n), dtype=np.float32)
        src = ctypes.create_string_buffer(bytes(data), len(data))
        self.lib.dequantize_row_stq1_0(ctypes.cast(src, ctypes.c_void_p), y.ctypes.data, rows * n)
        return y


def stq_reference():
    import hashlib
    import shutil
    import subprocess
    import tempfile
    env = os.environ.get("POLLARD_STQ_GGML_LIB")
    if env and os.path.isfile(env):
        return STQRef(env)
    tree, base = _stq_tree(), load()
    # the CommandLineTools clang first: /usr/bin/clang is an xcrun shim that refuses to run until the Xcode
    # licence is accepted, even when the CLT toolchain itself works
    cc = next((c for c in (os.environ.get("POLLARD_CC"), "/Library/Developer/CommandLineTools/usr/bin/clang",
                           shutil.which("clang"), shutil.which("cc")) if c and os.path.exists(c)), None)
    if tree is None or base is None or cc is None:
        return None
    ctypes.CDLL(base.path, mode=ctypes.RTLD_GLOBAL)        # ggml_abort & co. resolve against the runtime
    src = os.path.join(tree, "ggml", "src", "ggml-quants.c")
    tag = hashlib.sha256(open(src, "rb").read()).hexdigest()[:16]
    ext = ".dylib" if os.uname().sysname == "Darwin" else ".so"
    out = os.path.join(tempfile.gettempdir(), "pollard-fq-ref", f"stq-{tag}{ext}")
    if not os.path.isfile(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        # ggml_row_size would resolve against the runtime's libggml, which has no type 43 -- give this
        # translation unit its own (STQ1_0 rows only; nothing else in it is called)
        wrap = out + ".c"
        with open(wrap, "w") as f:
            f.write('#define ggml_row_size pollard_stq_row_size\n#include "ggml-quants.c"\n'
                    "size_t pollard_stq_row_size(enum ggml_type type, int64_t ne) { (void) type; "
                    "return (size_t) (ne / 256) * 42; }\n")
        cmd = [cc, "-O3", "-DNDEBUG", "-DGGML_BUILD", "-std=gnu11", "-fPIC", "-shared",
               "-I", os.path.join(tree, "ggml", "src"), "-I", os.path.join(tree, "ggml", "include"), wrap, "-o", out]
        if ext == ".dylib":
            cmd[1:1] = ["-arch", "arm64", "-undefined", "dynamic_lookup", "-D_DARWIN_C_SOURCE", "-D_XOPEN_SOURCE=600"]
            sdk = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
            if os.path.isdir(sdk):
                cmd[1:1] = ["-isysroot", sdk]
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            return None
    try:
        return STQRef(out)
    except OSError:
        return None
