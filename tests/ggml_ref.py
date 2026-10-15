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
