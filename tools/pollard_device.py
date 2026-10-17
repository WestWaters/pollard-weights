#!/usr/bin/env python3
"""pollard_device -- one place that decides where a tensor job runs: CUDA on NVIDIA (Linux and Windows),
MPS on Apple Silicon (MLX lanes use Metal through mlx itself), CPU otherwise.

Every tool used to carry its own two lines of `cuda if torch.cuda.is_available() else ...`. That answer is
wrong in three ways this helper does not repeat:

  * `is_available()` reports the DRIVER, not a usable device. With CUDA_VISIBLE_DEVICES=-1 (the shared build box
    hides its GPU while someone else is on it) or an empty mask on Windows it can answer True with zero devices,
    and the first device query dies.
  * A torch wheel built without code for this GPU imports fine, reports CUDA available, and then fails on the
    first kernel ("no kernel image is available for execution on the device") -- a cu124 wheel on a Blackwell
    card (sm_120) does exactly that. `cuda_ok()` checks the wheel's arch list against the card.
  * Asking for `mps` on a machine without it, or `cuda` on a Mac, should land on the best device that IS here,
    with a note -- not a traceback an hour into a build.

    import pollard_device as D
    dev = D.pick(a.device)            # "auto" | "cuda" | "cuda:1" | "mps" | "cpu" -> what this machine can run
    D.hide_cuda()                     # before `import torch`, for an explicit CPU run

torch is imported lazily: this module itself is stdlib-only, so the setup checker and the core install can
import it without the heavy extras.
"""
from __future__ import annotations

import os
import re
import sys


def hide_cuda():
    """Make an explicit CPU run MEAN cpu. Must run before torch is imported (torch caches availability).
    "-1", not "": on Windows an empty mask leaves is_available() True with zero devices."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _torch(torch=None):
    if torch is not None:
        return torch
    try:
        import torch as _t                                       # noqa: F401
        return _t
    except Exception:                                            # noqa: BLE001  (ImportError, a broken DLL)
        return None


def wheel_covers(cap, arch_list):
    """Does a torch wheel whose arch list is `arch_list` (torch.cuda.get_arch_list()) carry code for a GPU of
    compute capability `cap` ((12, 0))? Real code for the exact sm, or PTX (compute_XY) at or below it -- the
    driver JITs PTX forward. `sm_120a`-style entries count for their number."""
    if not arch_list:
        return True                                              # unknown: do not refuse on a guess
    want = cap[0] * 10 + cap[1]
    for a in arch_list:
        m = re.match(r"^(sm|compute)_(\d+)", str(a))
        if not m:
            continue
        n = int(m.group(2))
        if m.group(1) == "sm" and n == want:
            return True
        if m.group(1) == "compute" and n <= want:
            return True
    return False


def cuda_ok(torch=None):
    """(usable, reason). Usable = a visible device AND this wheel has code for it."""
    t = _torch(torch)
    if t is None:
        return False, "torch is not installed"
    try:
        if not t.cuda.is_available():
            return False, ("this torch has no CUDA build" if not getattr(t.version, "cuda", None)
                           else "no CUDA device visible")
        if t.cuda.device_count() < 1:
            return False, "no CUDA device visible (CUDA_VISIBLE_DEVICES hides them)"
        cap = tuple(t.cuda.get_device_capability(0))
        arches = list(t.cuda.get_arch_list() or [])
    except Exception as e:                                       # noqa: BLE001
        return False, f"CUDA query failed: {e}"
    if not wheel_covers(cap, arches):
        return False, (f"torch {t.__version__} (CUDA {t.version.cuda}) has no kernels for this GPU "
                       f"(sm_{cap[0]}{cap[1]}; wheel carries {', '.join(arches) or 'none'}) -- "
                       "reinstall torch from the matching CUDA index (`pollard setup --lanes torch`)")
    return True, f"sm_{cap[0]}{cap[1]}"


def mps_ok(torch=None):
    t = _torch(torch)
    if t is None:
        return False, "torch is not installed"
    try:
        mps = getattr(t.backends, "mps", None)
        if mps is not None and mps.is_available():
            return True, "Apple GPU (Metal)"
        if sys.platform == "darwin":
            return False, "MPS not available (Intel Mac, or a torch built without it)"
        return False, "not an Apple machine"
    except Exception as e:                                       # noqa: BLE001
        return False, f"MPS query failed: {e}"


def best(torch=None):
    """The best device this machine can actually run torch on: cuda > mps > cpu."""
    if cuda_ok(torch)[0]:
        return "cuda"
    if mps_ok(torch)[0]:
        return "mps"
    return "cpu"


def pick(want="auto", torch=None, note=None):
    """Resolve a requested device to one that works here. `auto` -> best(); an explicit device that is not usable
    falls back to best() with a note (printed, or passed to `note`) instead of failing later on the first kernel.
    `cpu` is always honoured. `cuda:N` keeps its index when CUDA is usable."""
    want = (want or "auto").strip().lower()
    if want == "auto":
        return best(torch)
    if want == "cpu":
        return "cpu"
    kind = want.split(":")[0]
    if kind == "cuda":
        ok, why = cuda_ok(torch)
    elif kind == "mps":
        ok, why = mps_ok(torch)
    else:
        return want                                              # xpu / meta / something we do not judge
    if ok:
        return want
    got = best(torch)
    msg = f"   [device] {want} is not usable here ({why}) -- using {got}"
    (note or (lambda m: print(m, file=sys.stderr, flush=True)))(msg)
    return got


def dtype_for(dev, torch=None):
    """A sensible compute dtype: bf16 on a CUDA card that supports it, fp16 on MPS / older CUDA, fp32 on CPU."""
    t = _torch(torch)
    if t is None:
        return None
    d = str(dev).split(":")[0]
    if d == "cuda":
        try:
            return t.bfloat16 if t.cuda.is_bf16_supported() else t.float16
        except Exception:                                        # noqa: BLE001
            return t.float16
    if d == "mps":
        return t.float16
    return t.float32


def accel_bytes(dev, torch=None):
    """Usable accelerator memory for `dev` (90% of it), 0 for the CPU or when unknown."""
    t = _torch(torch)
    d = str(dev).split(":")[0]
    try:
        if t is not None and d == "cuda" and cuda_ok(t)[0]:
            idx = int(str(dev).split(":")[1]) if ":" in str(dev) else 0
            return int(t.cuda.get_device_properties(idx).total_memory * 0.90)
        if t is not None and d == "mps":
            rec = getattr(getattr(t, "mps", None), "recommended_max_memory", None)
            return int(rec() * 0.90) if rec else 0
    except Exception:                                            # noqa: BLE001
        return 0
    return 0


def describe(torch=None):
    """What the setup report prints: torch version, its CUDA build, and which devices work."""
    t = _torch(torch)
    if t is None:
        return {"torch": None, "device": "cpu", "cuda": (False, "torch is not installed"),
                "mps": (False, "torch is not installed")}
    return {"torch": t.__version__, "torch_cuda": getattr(t.version, "cuda", None),
            "cuda": cuda_ok(t), "mps": mps_ok(t), "device": best(t)}
