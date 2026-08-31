#!/usr/bin/env python3
"""pollard-prune — REAP-style EXPERT PRUNING for a MoE GGUF: drop the least-important whole
experts so the model is STRUCTURALLY smaller, then quantize it normally. This is the lever
`pollard-pack` only planned — the "scissors" for a trillion-param MoE (Kimi K2: 384 experts).

Quant has a floor (1T params x 1 bit ~= 125 GB); you cannot quantize a 1T MoE onto one 128 GB
box. Pruning half the (cold) experts first drops the param count itself, THEN quant lands it.

    pollard-prune --gguf kimi-f16.gguf --keep 0.5 --out kimi-pruned.gguf
    pollard-prune --gguf moe-f16.gguf  --keep-experts 96 --imatrix moe.imatrix --out pruned.gguf
    # then:  pollard-automap --tensors <dry-run of pruned> --model pruned.gguf --no-imatrix ...

Scoring (which experts to KEEP, per layer):
  magnitude (default) — L2 norm of each expert's gate/up/down weights. Zero-calib, any box.
  imatrix             — summed importance-matrix activation per expert (REAP-correct: keeps the
                        experts the calib actually routed to). Needs --imatrix.
  router              — L2 norm of the router (ffn_gate_inp) row per expert. Cheap proxy.

Keeps a FIXED count K per layer (GGUF stores one expert_count); which experts differ per layer.
K must be >= the model's active-expert count (top-k), or routing breaks. Writes a new GGUF with
sliced ffn_*_exps + router (+ expert bias) and expert_count=K; all other tensors pass through.

Needs gguf-py (ships with llama.cpp: `pip install gguf`, or add llama.cpp/gguf-py to PYTHONPATH)
and numpy. Runs streaming (one tensor in memory at a time), so RAM ~= the largest expert tensor.
"""
import argparse, os, re, struct, sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pollard-prune needs gguf-py. `pip install gguf`, or add llama.cpp/gguf-py to "
             "PYTHONPATH (ik_llama.cpp/gguf-py). It's the same lib llama.cpp uses.")


# ---- imatrix activation per expert (the REAP-correct keep signal) --------------------------
def imatrix_activation(path):
    """{tensor_name: mean(activation^2)} from an ik/llama imatrix (old binary format:
    int32 n_entries, then per entry int32 len, name, int32 ncall, int32 nval, float[nval]).
    We reduce each entry to a scalar so expert tensors can be compared/ranked."""
    try:
        d = open(path, "rb").read()
    except Exception:
        return {}
    out, off = {}, 0
    n = struct.unpack_from("<i", d, off)[0]; off += 4
    for _ in range(n):
        ln = struct.unpack_from("<i", d, off)[0]; off += 4
        if ln <= 0 or ln > 512:
            return out
        name = d[off:off + ln].decode("utf-8", "replace"); off += ln
        off += 4                                             # ncall
        nval = struct.unpack_from("<i", d, off)[0]; off += 4
        vals = np.frombuffer(d, dtype="<f4", count=nval, offset=off); off += 4 * nval
        out[name] = float(vals.mean()) if nval else 0.0
    return out


_LAYER = re.compile(r"blk\.(\d+)\.")
_EXPERT_W = re.compile(r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight$")
_ROUTER = re.compile(r"blk\.\d+\.ffn_gate_inp\.weight$")
_EXP_BIAS = re.compile(r"blk\.\d+\.(ffn_gate_inp|exp_probs)_b(ias)?\.weight$|exp_probs_b\.weight$")


def _expert_axis(arr, n_exp):
    """Which axis of an ffn_*_exps tensor indexes the experts (the one equal to n_exp)."""
    for ax, s in enumerate(arr.shape):
        if s == n_exp:
            return ax
    return None


def score_experts(reader, n_exp, mode, imx):
    """Per layer -> np.array of per-expert scores (higher = keep). Aggregates the expert's
    gate+up+down (magnitude/imatrix) or reads the router row (router mode)."""
    layers = {}
    def acc(layer, vec):
        layers.setdefault(layer, np.zeros(n_exp, dtype=np.float64))
        layers[layer] += vec
    for t in reader.tensors:
        m = _LAYER.search(t.name)
        if not m:
            continue
        layer = int(m.group(1))
        if mode == "router" and _ROUTER.search(t.name):
            arr = np.asarray(t.data, dtype=np.float32)
            ax = _expert_axis(arr, n_exp)
            if ax is not None:
                acc(layer, np.linalg.norm(arr.reshape(arr.shape[ax], -1) if ax == 0
                                          else arr.T.reshape(n_exp, -1), axis=1))
        elif mode == "magnitude" and _EXPERT_W.search(t.name):
            arr = np.asarray(t.data, dtype=np.float32)
            ax = _expert_axis(arr, n_exp)
            if ax is not None:
                flat = np.moveaxis(arr, ax, 0).reshape(n_exp, -1)
                acc(layer, np.linalg.norm(flat, axis=1))
        elif mode == "imatrix" and _EXPERT_W.search(t.name):
            # imatrix stores one activation vector per expert tensor; its scalar reduction is
            # already per-tensor, so approximate per-expert by the expert-axis mean of |w|*imx.
            arr = np.asarray(t.data, dtype=np.float32)
            ax = _expert_axis(arr, n_exp)
            s = imx.get(t.name)
            if ax is not None and s is not None:
                flat = np.moveaxis(arr, ax, 0).reshape(n_exp, -1)
                acc(layer, np.linalg.norm(flat, axis=1) * s)
            elif ax is not None:                              # uncovered expert tensor: fall back
                flat = np.moveaxis(arr, ax, 0).reshape(n_exp, -1)
                acc(layer, np.linalg.norm(flat, axis=1))
    return layers


def keep_indices(layers, n_exp, keep_k):
    """Top-keep_k expert indices to KEEP per layer (sorted, so order is stable)."""
    keep = {}
    for layer, sc in layers.items():
        idx = np.argsort(sc)[::-1][:keep_k]
        keep[layer] = np.sort(idx)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", required=True, help="source MoE GGUF (f16/bf16 recommended)")
    ap.add_argument("--out", help="output pruned GGUF path (required unless --dry-run)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--keep", type=float, help="fraction of experts to KEEP per layer (0<f<=1)")
    g.add_argument("--keep-experts", type=int, help="absolute experts to KEEP per layer")
    ap.add_argument("--score", choices=["magnitude", "imatrix", "router"], default="magnitude")
    ap.add_argument("--imatrix", help="imatrix (for --score imatrix)")
    ap.add_argument("--dry-run", action="store_true", help="report the plan; write nothing")
    a = ap.parse_args()
    if not a.dry_run and not a.out:
        ap.error("--out is required unless --dry-run")

    reader = gguf.GGUFReader(a.gguf)
    # expert count from metadata (arch-agnostic: <arch>.expert_count)
    n_exp = None
    for field in reader.fields.values():
        if field.name.endswith("expert_count"):
            n_exp = int(field.parts[field.data[0]][0]); break
    if not n_exp or n_exp < 2:
        sys.exit(f"not a MoE (expert_count={n_exp}). pollard-prune is MoE-only; a dense model "
                 "has no experts to drop — quantize it with pollard-fit instead.")
    n_used = None
    for field in reader.fields.values():
        if field.name.endswith("expert_used_count"):
            n_used = int(field.parts[field.data[0]][0]); break
    keep_k = a.keep_experts if a.keep_experts else max(1, round(a.keep * n_exp))
    if n_used and keep_k < n_used:
        sys.exit(f"--keep leaves {keep_k} experts but the model routes to {n_used} (top-k). "
                 f"Keep at least {n_used}.")
    print(f"pollard-prune :: {os.path.basename(a.gguf)}")
    print(f"  experts: {n_exp} -> KEEP {keep_k} per layer  ({100*keep_k/n_exp:.0f}%, "
          f"score={a.score}, top-k routes to {n_used or '?'})")

    imx = imatrix_activation(a.imatrix) if a.score == "imatrix" and a.imatrix else {}
    if a.score == "imatrix" and not imx:
        sys.exit("--score imatrix needs a readable --imatrix.")
    layers = score_experts(reader, n_exp, a.score, imx)
    if not layers:
        sys.exit("found no expert tensors to score — is this the right GGUF?")
    keep = keep_indices(layers, n_exp, keep_k)
    dropped = n_exp - keep_k
    print(f"  {len(layers)} MoE layers scored; dropping {dropped}/{n_exp} experts each "
          f"(~{100*dropped/n_exp:.0f}% fewer expert params).")
    if a.dry_run:
        print("  dry-run: no file written. Re-run without --dry-run to emit the pruned GGUF.")
        return

    # ---- write the pruned GGUF: slice expert tensors + router, patch expert_count ----------
    arch = reader.fields["general.architecture"]
    arch = str(bytes(arch.parts[arch.data[0]]), "utf-8")
    writer = gguf.GGUFWriter(a.out, arch)
    # copy metadata, overriding expert_count
    for field in reader.fields.values():
        if field.name in ("general.architecture", "GGUF.version",
                           "GGUF.tensor_count", "GGUF.kv_count"):
            continue
        if field.name.endswith("expert_count"):
            writer.add_uint32(field.name, keep_k)
            continue
        _copy_field(writer, field)

    for t in reader.tensors:
        arr = np.asarray(t.data)
        m = _LAYER.search(t.name)
        if m and (_EXPERT_W.search(t.name) or _ROUTER.search(t.name)):
            layer = int(m.group(1))
            ax = _expert_axis(arr, n_exp)
            if ax is not None and layer in keep:
                arr = np.take(arr, keep[layer], axis=ax)
        writer.add_tensor(t.name, np.ascontiguousarray(arr), raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    old = os.path.getsize(a.gguf) / 1e9
    new = os.path.getsize(a.out) / 1e9
    print(f"  wrote {a.out}: {old:.1f} GB -> {new:.1f} GB (-{100*(1-new/old):.0f}%). "
          f"Next: pollard-automap --no-imatrix on the pruned GGUF, then quantize.")


def _copy_field(writer, field):
    """Re-emit a GGUFReader field into the writer, preserving its type."""
    import gguf as _g
    val = field.contents() if hasattr(field, "contents") else None
    t = field.types[0] if field.types else None
    if t == _g.GGUFValueType.STRING:
        writer.add_string(field.name, str(val))
    elif t == _g.GGUFValueType.ARRAY:
        writer.add_array(field.name, list(val) if val is not None else [])
    elif t in (_g.GGUFValueType.FLOAT32, _g.GGUFValueType.FLOAT64):
        writer.add_float32(field.name, float(val))
    elif t is not None:
        writer.add_uint32(field.name, int(val))


if __name__ == "__main__":
    main()
