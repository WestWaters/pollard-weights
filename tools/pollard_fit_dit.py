#!/usr/bin/env python3
"""pollard-fit-dit — memory-fit builds for ANY GGUF architecture, pure Python.

llama-quantize only speaks LLM architectures; diffusion models (DiTs — video,
image, audio) live in GGUF too but get rejected. This builder sidesteps the
C++ entirely: it reads any GGUF with gguf-py, chooses a per-tensor type from a
protection policy and a byte budget, quantizes with gguf-py's own kernels
(Q4_0 / Q5_0 / Q8_0 / F16 ladder — the same mechanism community DiT quants
are made with), and writes a normal GGUF back out.

Usage:
  pollard-fit-dit --gguf model-bf16.gguf --budget-gb 11 --out model-pollard.gguf
  pollard-fit-dit --gguf model.gguf --budget-gb 11 --plan-only
  pollard-fit-dit --gguf model.gguf --budget-gb 11 --protect "adaln,norm,bias,emb"

Protection: matching tensors keep high precision (Q8_0, or F16 if tiny).
Everything else starts at Q5_0 and the largest tensors step down to Q4_0,
biggest-first, until the projection fits the budget. Best from a bf16/f16
source; requantizing an already-4-bit file cannot recover quality.
"""
import argparse
import sys

import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter
from gguf.quants import quantize
from gguf.constants import GGMLQuantizationType as T

LADDER = [T.Q8_0, T.Q5_0, T.Q4_0]
BPW = {T.F32: 32, T.F16: 16, T.BF16: 16, T.Q8_0: 8.5, T.Q5_0: 5.5, T.Q4_0: 4.6}
DEFAULT_PROTECT = ["adaln", "norm", "bias", "emb", "patch_proj", "head",
                   "modulation", "time_", "t_table", "final"]
TINY = 1 << 16          # tensors below this stay F16 — overhead beats savings


def plan(reader, budget_gb, protect_frags):
    """Return {name: target_type} and the projected size."""
    assign, sizes = {}, {}
    for t in reader.tensors:
        n_params = int(np.prod(t.shape))
        sizes[t.name] = n_params
        low = t.name.lower()
        if n_params < TINY or t.data.ndim == 1:
            # keep the SOURCE type: llama.cpp kernels expect norms/biases in
            # F32; "optimizing" them to F16 crashes compute
            assign[t.name] = T.F32 if t.tensor_type == T.F32 else T.F16
        elif any(f in low for f in protect_frags):
            assign[t.name] = T.Q8_0
        else:
            assign[t.name] = T.Q5_0

    def projected():
        return sum(sizes[n] * BPW[q] / 8 for n, q in assign.items()) / 1e9

    # step the largest unprotected tensors down first until we fit
    order = sorted((n for n, q in assign.items() if q == T.Q5_0),
                   key=lambda n: -sizes[n])
    i = 0
    while projected() > budget_gb and i < len(order):
        assign[order[i]] = T.Q4_0
        i += 1
    return assign, projected()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--budget-gb", type=float, required=True,
                    help="target on-disk size for the build")
    ap.add_argument("--out")
    ap.add_argument("--protect", default=",".join(DEFAULT_PROTECT),
                    help="comma-separated name fragments kept at high precision")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()

    reader = GGUFReader(a.gguf)
    arch = "unknown"
    for f in reader.fields.values():
        if f.name == "general.architecture":
            arch = str(bytes(f.parts[-1]), "utf-8")
    protect = [p.strip().lower() for p in a.protect.split(",") if p.strip()]
    assign, gb = plan(reader, a.budget_gb, protect)

    from collections import Counter
    c = Counter(q.name for q in assign.values())
    print(f"== pollard-fit-dit :: {a.gguf}  [{arch}]")
    print(f"tensors             : {len(assign)}")
    print(f"projected build     : {gb:.2f} GB  (budget {a.budget_gb:g} GB)")
    print(f"type mix            : " + ", ".join(f"{n}×{k}" for k, n in sorted(c.items())))
    if a.plan_only:
        return
    if gb > a.budget_gb * 1.02:
        sys.exit("ERROR: cannot reach the budget with the Q4_0 floor — "
                 "lower --budget-gb expectations or prune first.")

    out = a.out or a.gguf.rsplit(".gguf", 1)[0] + "-pollard.gguf"
    w = GGUFWriter(out, arch)
    # copy every metadata field except the ones GGUFWriter writes itself
    from gguf.constants import GGUFValueType as VT
    for f in reader.fields.values():
        if f.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                      "general.architecture"):
            continue
        vtype = f.types[0]
        sub = f.types[1] if vtype == VT.ARRAY and len(f.types) > 1 else None
        w.add_key_value(f.name, f.contents(), vtype, sub_type=sub)
    for t in reader.tensors:
        q = assign[t.name]
        data = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        shape = tuple(int(d) for d in t.shape)
        data = data.reshape(shape[::-1])          # reader shape is reversed
        if q not in (T.F16, T.F32) and data.shape[-1] % 32 != 0:
            q = T.F16                     # block quants need 32-wide rows
        if q == T.F32:
            w.add_tensor(t.name, data.astype(np.float32))
            continue
        if q == T.F16:
            w.add_tensor(t.name, data.astype(np.float16))
        else:
            qd = quantize(data, q)
            # raw_shape is the QUANTIZED byte-shape; gguf-py recovers the
            # logical shape from it (passing the logical shape here = crash)
            w.add_tensor(t.name, qd, raw_shape=qd.shape, raw_dtype=q)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"done: {out}")


if __name__ == "__main__":
    main()
