"""Load a .pt checkpoint without running whatever a pickle tells it to.

An unrestricted torch.load (weights_only off) is plain pickle: a .pt from anywhere -- a checkpoint someone shared,
a resumed training state on a shared disk -- can run arbitrary code the moment it is opened. weights_only=True is
torch's restricted unpickler: tensors, dicts, lists, tuples, numbers, strings and nothing callable.

Some checkpoints also hold numpy arrays (a graph's edge lists, say), which that unpickler refuses by default.
`numpy=True` allows those by name and nothing else: numpy's array rebuild, ndarray, dtype and the dtype classes.
They rebuild data; none of them runs code. The trained .pt files on disk load bit-identically to the old
unrestricted load.
"""
from __future__ import annotations

__all__ = ["safe_torch_load", "numpy_safe_globals"]


def numpy_safe_globals():
    """The numpy names a saved ndarray needs, under both spellings its pickle may use (numpy 1 wrote
    numpy.core.multiarray, numpy 2 writes numpy._core.multiarray)."""
    import numpy as np
    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:                                   # numpy 1.x
        from numpy.core.multiarray import _reconstruct
    allowed = [np.ndarray, np.dtype,
               (_reconstruct, "numpy.core.multiarray._reconstruct"),
               (_reconstruct, "numpy._core.multiarray._reconstruct")]
    dts = getattr(np, "dtypes", None)
    if dts is not None:
        allowed += [getattr(dts, n) for n in dir(dts) if n.endswith("DType") and isinstance(getattr(dts, n), type)]
    return allowed


def safe_torch_load(f, map_location="cpu", numpy=False):
    """torch.load with weights_only=True. `numpy=True` additionally allows numpy arrays.
    A payload that needs anything else raises torch's UnpicklingError instead of running it."""
    import torch
    if not numpy:
        return torch.load(f, map_location=map_location, weights_only=True)
    ctx = getattr(torch.serialization, "safe_globals", None)
    if ctx is None:
        raise RuntimeError(f"torch {torch.__version__} cannot load numpy arrays safely (torch.serialization.safe_globals "
                           "needs torch >= 2.5); upgrade torch")
    with ctx(numpy_safe_globals()):
        return torch.load(f, map_location=map_location, weights_only=True)
