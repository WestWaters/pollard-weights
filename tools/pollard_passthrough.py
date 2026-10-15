#!/usr/bin/env python3
"""Quantized-GGUF passthrough -- build a smaller rung FROM an already-quantized GGUF without
touching the tensors that are already right.

The case it exists for is a vendor QAT release (NVIDIA Nemotron-3.5-Bolt-24GB-GGUF: experts trained
into IQ2_XXS, Mamba ssm_in/out NVFP4, attention / embeddings Q8_0). Requantizing that file the
ordinary way dequantizes every tensor and quantizes it again -- the QAT experts lose the exact codes
they were trained into, and the Q8_0 / NVFP4 tensors get a second rounding for nothing. Passthrough
is the other policy:

  * every tensor whose assigned type equals its current type is COPIED VERBATIM (no dequant /
    requant -- llama-quantize writes the source bytes when target == current type, and Pollard
    checks byte identity afterwards);
  * a tensor assigned a LOWER type is requantized (--allow-requantize) from its current data;
  * a tensor is never "upgraded": a plan or --tier that asks for more bits than the tensor already
    has is refused -- the information is not there to recover.

Which tensors change is the measured allocation's call: start from the source exactly, then take the
cheapest KL-per-byte downgrades (only inside the changeable classes, default `bulk` = routed experts
and dense FFN) until the budget fits. Everything else -- attention, shared experts, SSM, embeddings,
vendor-only formats -- stays byte-identical.

REQUANT SOURCE CAVEAT: a downgraded tensor is quantized from dequant(current), not from the original
bf16 weights. That compounds two roundings (the source format's and the new one's) and the imatrix
can only steer the second. For a QAT source this is still the right input -- the QAT codes ARE the
model -- but for an ordinary PTQ file, a rung requantized from bf16 will usually beat it at the same
bytes. The sensitivity probe and the imatrix both run on the quantized GGUF as host, so what they
measure is the cost of moving away from THIS file.

Used by `pollard-fit --passthrough` (auto-on when the source is already quantized) and
`pollard --gguf <quantized.gguf> --ladder`; `pollard-sensitivity --passthrough` measures the profile.
"""
import hashlib
import heapq
import math
import os
import re
from collections import Counter

# ---------------------------------------------------------------------------------------------
# type facts (from gguf-py's GGML_QUANT_SIZES when present, so a new ggml type is picked up)
# ---------------------------------------------------------------------------------------------
_FALLBACK_SIZES = {   # name -> (block elements, block bytes)
    "f32": (1, 4), "f16": (1, 2), "bf16": (1, 2), "q4_0": (32, 18), "q4_1": (32, 20), "q5_0": (32, 22),
    "q5_1": (32, 24), "q8_0": (32, 34), "q2_k": (256, 84), "q3_k": (256, 110), "q4_k": (256, 144),
    "q5_k": (256, 176), "q6_k": (256, 210), "iq2_xxs": (256, 66), "iq2_xs": (256, 74), "iq3_xxs": (256, 98),
    "iq1_s": (256, 50), "iq4_nl": (32, 18), "iq3_s": (256, 110), "iq2_s": (256, 82), "iq4_xs": (256, 136),
    "iq1_m": (256, 56), "tq1_0": (256, 54), "tq2_0": (256, 66), "mxfp4": (32, 17), "nvfp4": (64, 36),
}


def _sizes():
    try:
        from gguf import GGML_QUANT_SIZES
        return {t.name.lower(): v for t, v in GGML_QUANT_SIZES.items()}
    except Exception:
        return dict(_FALLBACK_SIZES)


SIZES = _sizes()
# llama-quantize's spelling of each type (ggml_type_name), which --tensor-type parses case-insensitively;
# kept in ggml's case so the file reads like the rest of Pollard's tensor-type files
_GGML_NAME = {"q2_k": "q2_K", "q3_k": "q3_K", "q4_k": "q4_K", "q5_k": "q5_K", "q6_k": "q6_K", "q8_k": "q8_K"}


def norm_type(t):
    return str(t).strip().lower()


def ggml_name(t):
    t = norm_type(t)
    return _GGML_NAME.get(t, t)


def bpw(t):
    """Exact bits per weight of a ggml type (block overhead included), None if unknown (ik-only types)."""
    s = SIZES.get(norm_type(t))
    return None if s is None else s[1] * 8.0 / s[0]


def block(t):
    s = SIZES.get(norm_type(t))
    return None if s is None else s[0]


def nbytes(t, n_elements):
    blk, sz = SIZES[norm_type(t)]
    return (int(n_elements) // blk) * sz


# the downgrade ladder a passthrough rung walks (high -> low). q8_0 heads it so a bf16/f16 tensor in a
# quantized file has a first step; the 1-bit floor is part of it because a ~2-bit QAT source has no other
# lever left -- they still need an imatrix (see candidates()).
PT_LADDER = ["q8_0", "q6_K", "q5_K", "iq4_xs", "iq3_s", "iq2_s", "iq2_xxs", "iq1_m", "iq1_s"]
IMATRIX_REQUIRED = {"iq2_xxs", "iq2_xs", "iq2_s", "iq1_s", "iq1_m"}
NOIMATRIX_SUB = {"iq2_xxs": "q2_K", "iq2_xs": "q2_K", "iq2_s": "q2_K"}
# KL per unit importance of a uniform build at each type -- pollard_fit.NOISE, plus q2_K (between iq2_s
# and iq3_s by the e11 granite measurement: iq2_s beat q2_K by ~23%)
DEFAULT_NOISE = {"q8_0": 0.00048, "q6_K": 0.00479, "q5_K": 0.01033, "iq4_xs": 0.04090, "iq3_s": 0.12282,
                 "iq2_s": 0.63394, "q2_K": 0.78, "iq2_xxs": 1.22235, "iq1_m": 2.17, "iq1_s": 2.96}
CLASSES = ("bulk", "attn", "shexp", "emb", "other")


class PassthroughError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------
# reading the source
# ---------------------------------------------------------------------------------------------
def _shards(path):
    try:
        from pollard_calc import _shard_paths
        return _shard_paths(path)
    except Exception:
        return [path]


def read_tensors(path):
    """[{name, type, ne, n_elements, n_bytes, ndim, shard, offset}] for every tensor in every shard.
    Only the header is parsed; the data stays memory-mapped and untouched."""
    from gguf import GGUFReader
    out = []
    for sp in _shards(path):
        r = GGUFReader(sp)
        for t in r.tensors:
            ne = [int(x) for x in t.shape]
            out.append({"name": t.name, "type": t.tensor_type.name.lower(), "ne": ne,
                        "n_elements": int(t.n_elements), "n_bytes": int(t.n_bytes), "ndim": len(ne),
                        "shard": sp, "offset": int(t.data_offset)})
    return out


def is_quantized_source(tensors, min_share=0.5):
    """Is this GGUF already quantized? True when most of the weight BYTES sit in block-quantized types
    (a bf16 file with a q8_0 token_embd is not; Bolt's IQ2_XXS / NVFP4 / Q8_0 file is)."""
    tot = q = 0
    for t in tensors:
        if t["ndim"] < 2:
            continue
        tot += t["n_bytes"]
        if (block(t["type"]) or 1) > 1:
            q += t["n_bytes"]
    return bool(tot) and q / tot >= min_share


# ---------------------------------------------------------------------------------------------
# classifying tensors
# ---------------------------------------------------------------------------------------------
_LAYER = re.compile(r"(?:^|\.)blk\.(\d+)\.")
_BULK = re.compile(r"(?:^|\.)blk\.\d+\.ffn_(up|gate|down)(_exps)?\.weight$")
_SHEXP = re.compile(r"(?:^|\.)blk\.\d+\.ffn_(up|gate|down)_(shexp|chexps?)\.weight$")
_ATTN = re.compile(r"(?:^|\.)blk\.\d+\.attn_[a-z0-9_]+\.weight$")
# what llama-quantize refuses to touch whatever it is told (tensor_allows_quantization)
_NEVER = ("_norm.weight", "ffn_gate_inp.weight", "ffn_gate_tid2eid", "altup", "laurel", "per_layer_model_proj",
          "ssm_conv1d", "shortconv.conv.weight", "indexer.k_proj.weight", "indexer.q_proj.weight", "time_mix_",
          "attn_rel_b.weight", ".position_embd", "pos_embd", "token_types", "rel_pos", "patch_embd",
          "patch_merger", "sam.", "rvq.codebook", "code_embd")


def layer_of(name):
    m = _LAYER.search(name)
    return int(m.group(1)) if m else None


def tensor_class(name):
    if name in ("token_embd.weight", "output.weight", "per_layer_token_embd.weight"):
        return "emb"
    if _BULK.search(name):
        return "bulk"
    if _SHEXP.search(name):
        return "shexp"
    if _ATTN.search(name):
        return "attn"
    return "other"


def ffn_part(name):
    """'down' / 'gateup' for a bulk tensor (the granular sensitivity profile splits them), else None."""
    m = _BULK.search(name)
    if not m:
        return None
    return "down" if m.group(1) == "down" else "gateup"


def quantizable(t):
    """Would llama-quantize consider this tensor at all? (Everything else it copies regardless.)"""
    n = t["name"]
    return t["ndim"] >= 2 and n.endswith("weight") and not any(s in n for s in _NEVER)


# ---------------------------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------------------------
def candidates(t, imatrix_ok=True, covered=None, ladder=PT_LADDER):
    """The types this tensor may be DOWNGRADED to, high -> low: strictly fewer bits than it has now,
    row length compatible, imatrix-only types only where the imatrix covers it (a 1-bit floor without
    one is not offered at all; IQ2 falls back to q2_K)."""
    cur = bpw(t["type"])
    if cur is None:
        return []                                         # an ik-only / unknown type: leave it alone
    has_im = imatrix_ok and (covered is None or t["name"] in covered)
    out = []
    for ty in ladder:
        if not has_im and norm_type(ty) in IMATRIX_REQUIRED:
            ty = NOIMATRIX_SUB.get(norm_type(ty))
            if ty is None:
                continue
        b, blk = bpw(ty), block(ty)
        if b is None or b >= cur - 1e-9 or (t["ne"] and t["ne"][0] % blk):
            continue
        if ty not in out:
            out.append(ty)
    return sorted(out, key=lambda x: -bpw(x))


def _interp_noise(b):
    """Default-curve noise at an arbitrary bpw (log-linear in bpw between the measured points)."""
    pts = sorted((bpw(t), math.log(v)) for t, v in DEFAULT_NOISE.items())
    if b >= 15.9:
        return 0.0
    if b <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
    elif b >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
    else:
        for i in range(len(pts) - 1):
            if pts[i][0] <= b <= pts[i + 1][0]:
                (x0, y0), (x1, y1) = pts[i], pts[i + 1]
                break
    return math.exp(y0 + (y1 - y0) * (b - x0) / max(1e-9, x1 - x0))


def noise_model(sensitivity=None):
    """(noise_of_type, noise_of_current_type).

    A passthrough profile (pollard-sensitivity --passthrough) measured each type's KL RELATIVE TO THE
    SOURCE, so the current type costs 0 and unmeasured types are the default curve scaled to the
    measured ones. Otherwise: the profile's (or the default) uniform-from-f16 curve, and the current
    type's own cost interpolated on it -- the downgrade pays only the difference."""
    prof = (sensitivity or {}).get("noise") or {}
    meas = {norm_type(k): float(v) for k, v in prof.items() if v}
    base = {norm_type(k): v for k, v in DEFAULT_NOISE.items()}
    ratios = [meas[k] / base[k] for k in meas if k in base and base[k] > 0]
    scale = sorted(ratios)[len(ratios) // 2] if ratios else 1.0
    pt = bool((sensitivity or {}).get("passthrough"))

    def of_type(ty):
        k = norm_type(ty)
        if k in meas:
            return meas[k]
        if k in base:
            return base[k] * scale
        b = bpw(k)
        return _interp_noise(b) * scale if b else 1.0

    def of_current(ty):
        if pt:
            return 0.0
        k = norm_type(ty)
        if k in meas:
            return meas[k]
        b = bpw(k)
        return (_interp_noise(b) * scale) if b else 0.0
    return of_type, of_current


def _importance(t, sensitivity):
    """Measured sensitivity of the group this tensor belongs to (layer x class), 1.0 when unmeasured."""
    if not sensitivity:
        return 1.0
    li, cls = layer_of(t["name"]), tensor_class(t["name"])
    if li is None:
        return 1.0
    key = str(li)
    if cls == "bulk":
        part = ffn_part(t["name"])
        if part == "down" and sensitivity.get("ffn_down"):
            v = sensitivity["ffn_down"].get(key)
        elif part == "gateup" and sensitivity.get("ffn_gateup"):
            v = sensitivity["ffn_gateup"].get(key)
        else:
            v = (sensitivity.get("ffn") or {}).get(key)
    elif cls == "attn":
        v = (sensitivity.get("attn") or {}).get(key)
    else:
        v = None
    return max(float(v), 1e-9) if v is not None else 1.0


def plan_passthrough(tensors, budget_bytes, sensitivity=None, classes=("bulk",), imatrix_ok=True,
                     covered=None, pins=None, ladder=PT_LADDER):
    """Assign every tensor a type: its CURRENT type, unless the budget forces a downgrade.

    Greedy on marginal KL-per-byte (pollard_fit's allocator, per tensor instead of per arch group):
    everything starts where the source has it, and the cheapest single step down is taken until the
    total fits. Only tensors in `classes` may move; `pins` {class: type} force a class to one type
    (refused if that would be an upgrade). Returns {"assign", "bytes", "source_bytes", "changed",
    "fits", "summary"}; `fits` False means even the floor is over budget (caller decides)."""
    pins = {k: norm_type(v) for k, v in (pins or {}).items()}
    classes = set(classes)
    of_type, of_cur = noise_model(sensitivity)
    assign = {t["name"]: t["type"] for t in tensors}
    size = {t["name"]: t["n_bytes"] for t in tensors}
    for t in tensors:
        cls = tensor_class(t["name"])
        if cls in pins and quantizable(t):
            ty = pins[cls]
            if ty == t["type"]:
                continue
            if bpw(ty) is None or bpw(t["type"]) is None or bpw(ty) > bpw(t["type"]) + 1e-9:
                raise PassthroughError(
                    f"--tier {cls}={ty} would UPGRADE {t['name']} ({t['type']} -> {ty}): the bits are not in "
                    f"this file to recover. Passthrough only copies or downgrades.")
            assign[t["name"]], size[t["name"]] = ty, nbytes(ty, t["n_elements"])
    items = []                                           # (tensor, [steps], importance)
    for t in tensors:
        if tensor_class(t["name"]) not in classes or tensor_class(t["name"]) in pins or not quantizable(t):
            continue
        steps = candidates(t, imatrix_ok, covered, ladder)
        if steps:
            items.append((t, steps, _importance(t, sensitivity) * t["n_elements"]))
    tot_imp = sum(i for _, _, i in items) or 1.0
    lvl = [-1] * len(items)                              # -1 = the source's own type

    def marginal(j):
        t, steps, imp = items[j]
        i = lvl[j]
        if i + 1 >= len(steps):
            return None
        a = t["type"] if i < 0 else steps[i]
        b = steps[i + 1]
        db = (t["n_bytes"] if i < 0 else nbytes(a, t["n_elements"])) - nbytes(b, t["n_elements"])
        na = of_cur(a) if i < 0 else of_type(a)
        dkl = (imp / tot_imp) * max(0.0, of_type(b) - na)
        return ((dkl / db) if db > 0 else float("inf"), j)

    total = sum(size.values())
    heap = [m for j in range(len(items)) if (m := marginal(j)) is not None]
    heapq.heapify(heap)
    while total > budget_bytes and heap:
        _, j = heapq.heappop(heap)
        t, steps, _ = items[j]
        old = size[t["name"]]
        lvl[j] += 1
        new_t = steps[lvl[j]]
        assign[t["name"]], size[t["name"]] = new_t, nbytes(new_t, t["n_elements"])
        total += size[t["name"]] - old
        nxt = marginal(j)
        if nxt is not None:
            heapq.heappush(heap, nxt)
    src = sum(t["n_bytes"] for t in tensors)
    changed = {t["name"]: (t["type"], assign[t["name"]]) for t in tensors if norm_type(assign[t["name"]]) != t["type"]}
    moves = Counter(f"{tensor_class(n)} {a}->{b}" for n, (a, b) in changed.items())
    summary = ", ".join(f"{c} x{k}" for k, c in sorted(moves.items(), key=lambda kv: (-kv[1], kv[0]))) or "nothing changes"
    return {"assign": assign, "bytes": total, "source_bytes": src, "changed": changed,
            "fits": total <= budget_bytes, "summary": summary}


def floor_bytes(tensors, classes=("bulk",), imatrix_ok=True, covered=None):
    """Smallest a passthrough rung can get: every changeable tensor at its lowest candidate."""
    tot = 0
    for t in tensors:
        steps = candidates(t, imatrix_ok, covered) if (tensor_class(t["name"]) in classes and quantizable(t)) else []
        tot += nbytes(steps[-1], t["n_elements"]) if steps else t["n_bytes"]
    return tot


def check_no_upgrade(tensors, assign):
    """Refuse any assignment with more bits than the tensor has (or an unknown type swap)."""
    bad = []
    for t in tensors:
        ty = norm_type(assign.get(t["name"], t["type"]))
        if ty == t["type"]:
            continue
        a, b = bpw(t["type"]), bpw(ty)
        if a is None or b is None or b > a + 1e-9:
            bad.append(f"{t['name']}: {t['type']} -> {ty}")
    if bad:
        raise PassthroughError(f"passthrough refuses {len(bad)} upgrade(s) -- an already-quantized tensor cannot "
                               f"gain back bits it does not have: " + "; ".join(bad[:6]) + (" ..." if len(bad) > 6 else ""))


# ---------------------------------------------------------------------------------------------
# emitting the llama-quantize call
# ---------------------------------------------------------------------------------------------
_RX_META = set(".^$|?*+()[]{}\\")


def anchored(name):
    """An exact-name pattern for --tensor-type: anchored, so `blk.1.ffn_up.weight` can never also hit
    `blk.11.` / `dec.blk.1.`; only regex metacharacters escaped (std::regex on MSVC rejects identity
    escapes like `\\-` that Python's re.escape emits)."""
    return "^" + "".join("\\" + c if c in _RX_META else c for c in name) + "$"


def quantize_plan(tensors, assign):
    """(tensor_type_lines, flags, stats). One anchored line PER quantizable tensor -- every one of them
    named explicitly, so llama-quantize's first-match-wins order, its base preset and its own mixture
    logic never get a say: a tensor at its current type is written byte-for-byte (cur == new -> copy),
    a lower one is requantized. Lines for the passthrough ones come first only for readability; with
    exact anchors no two lines can match the same tensor."""
    check_no_upgrade(tensors, assign)
    keep, down = [], []
    flags = []
    for t in tensors:
        ty = norm_type(assign.get(t["name"], t["type"]))
        if t["name"] == "token_embd.weight":
            flags += ["--token-embedding-type", ggml_name(ty)]
        elif t["name"] == "output.weight":
            flags += ["--output-tensor-type", ggml_name(ty)]
        if not quantizable(t):
            continue
        (keep if ty == t["type"] else down).append(f"{anchored(t['name'])}={ggml_name(ty)}")
    if down:
        flags = ["--allow-requantize"] + flags
    stats = {"copy": len(keep), "requant": len(down),
             "copy_bytes": sum(t["n_bytes"] for t in tensors if norm_type(assign.get(t["name"], t["type"])) == t["type"])}
    return keep + down, flags, stats


# the positional base preset llama-quantize insists on. Every quantizable tensor is named explicitly, so
# it decides nothing -- Q8_0 because it never requires an imatrix (an IQ preset makes llama-quantize
# refuse to start without one, even when no tensor would use it).
BASE_PRESET = "Q8_0"


def build_command(llama_quantize, src, out, tt_file, flags, imatrix=None, threads=None):
    cmd = [llama_quantize] + list(flags)
    if imatrix:
        cmd += ["--imatrix", imatrix]
    cmd += ["--tensor-type-file", tt_file, src, out, BASE_PRESET]
    if threads:
        cmd += [str(threads)]
    return cmd


# ---------------------------------------------------------------------------------------------
# verifying the build
# ---------------------------------------------------------------------------------------------
def _digest(t, chunk=64 << 20):
    h = hashlib.sha256()
    with open(t["shard"], "rb") as f:
        f.seek(t["offset"])
        left = t["n_bytes"]
        while left:
            b = f.read(min(chunk, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def verify_build(src_tensors, out_path, assign):
    """Byte-check the passthrough promise on the written file: every tensor assigned its current type
    must be byte-identical to the source; every downgraded one must carry the assigned type."""
    out = {t["name"]: t for t in read_tensors(out_path)}
    rep = {"identical": 0, "requantized": 0, "mismatch": [], "missing": []}
    for t in src_tensors:
        o = out.get(t["name"])
        if o is None:
            rep["missing"].append(t["name"])
            continue
        want = norm_type(assign.get(t["name"], t["type"]))
        if o["type"] != want:
            rep["mismatch"].append(f"{t['name']}: type {o['type']} != planned {want}")
        elif want == t["type"]:
            if o["n_bytes"] != t["n_bytes"] or _digest(o) != _digest(t):
                rep["mismatch"].append(f"{t['name']}: bytes differ from the source")
            else:
                rep["identical"] += 1
        else:
            rep["requantized"] += 1
    rep["ok"] = not rep["mismatch"] and not rep["missing"]
    return rep


# ---------------------------------------------------------------------------------------------
# ladder rungs for `pollard --gguf <quantized> --ladder`
# ---------------------------------------------------------------------------------------------
PT_RUNGS = (0.25, 0.5, 0.75, 1.0)    # how far each rung goes from the source size toward the floor


def ladder_targets(tensors, classes=("bulk",), imatrix_ok=True, covered=None, rungs=PT_RUNGS):
    """[(label, target_bytes)] -- rungs spaced between the source (nothing changes) and the floor
    (every changeable tensor at its lowest candidate). A vendor QAT source has no f16 to take fractions
    of, so the rungs are placed by how much of the reducible gap they spend."""
    src = sum(t["n_bytes"] for t in tensors)
    flo = floor_bytes(tensors, classes, imatrix_ok, covered)
    if flo >= src:
        return []
    return [(f"PT{int(round(f * 100))}", int(src - f * (src - flo))) for f in rungs]


def describe(plan):
    gb = 1e9
    return (f"{plan['source_bytes']/gb:.2f} GB -> {plan['bytes']/gb:.2f} GB  "
            f"({len(plan['changed'])} tensor(s) requantized: {plan['summary']})")


# ---------------------------------------------------------------------------------------------
# measuring on the quantized host (pollard-sensitivity --passthrough)
# ---------------------------------------------------------------------------------------------
GROUP_CLASS = {"ffn": "bulk", "attn": "attn", "shexp": "shexp"}


def _crush(t, probe, imatrix_ok, covered):
    """The probe type for this tensor if it is a legal downgrade, else its lowest legal one."""
    steps = candidates(t, imatrix_ok, covered)
    if not steps:
        return None
    return probe if probe in steps else steps[-1]


def default_probe(tensors, classes=("bulk",), imatrix_ok=True, covered=None):
    """The lowest type the changeable tensors can reach (iq1_s for a QAT IQ2 source with an imatrix)."""
    lows = [candidates(t, imatrix_ok, covered)[-1] for t in tensors
            if tensor_class(t["name"]) in classes and quantizable(t) and candidates(t, imatrix_ok, covered)]
    return min(lows, key=bpw) if lows else None


def probe_assign(tensors, group, layer, probe, imatrix_ok=True, covered=None):
    """Everything as the source has it, except group `group` (ffn/attn/shexp) of `layer` crushed to `probe`.
    Returns (assign, n_crushed)."""
    cls = GROUP_CLASS.get(group, group)
    assign = {t["name"]: t["type"] for t in tensors}
    n = 0
    for t in tensors:
        if tensor_class(t["name"]) == cls and layer_of(t["name"]) == layer and quantizable(t):
            ty = _crush(t, probe, imatrix_ok, covered)
            if ty:
                assign[t["name"]] = ty
                n += 1
    return assign, n


def noise_types(tensors, classes=("bulk",), imatrix_ok=True, covered=None):
    """The ladder types at least one changeable tensor can be moved to -- the passthrough noise curve's points."""
    seen = set()
    for t in tensors:
        if tensor_class(t["name"]) in classes and quantizable(t):
            seen.update(candidates(t, imatrix_ok, covered))
    return sorted(seen, key=lambda x: -bpw(x))


def noise_assign(tensors, ty, classes=("bulk",), imatrix_ok=True, covered=None):
    """Every changeable tensor that can take `ty` at `ty`, the rest as the source has it."""
    assign = {t["name"]: t["type"] for t in tensors}
    for t in tensors:
        if tensor_class(t["name"]) in classes and quantizable(t) and ty in candidates(t, imatrix_ok, covered):
            assign[t["name"]] = ty
    return assign


def sweep(tensors, layers, groups, probe, build, kl, imatrix_ok=True, covered=None, log=print):
    """The passthrough sensitivity sweep. `build(assign) -> path or None` writes a probe GGUF and
    `kl(path) -> float or None` scores it against the SOURCE's own logits (the quantized host is the
    reference: a passthrough build of the whole file IS the file, so its KL is 0 by construction).
    Returns the profile dict pollard-fit --passthrough reads."""
    classes = tuple(GROUP_CLASS.get(g, g) for g in groups)
    noise = {}
    for ty in noise_types(tensors, classes, imatrix_ok, covered):
        p = build(noise_assign(tensors, ty, classes, imatrix_ok, covered))
        noise[ty] = kl(p) if p else None
        log(f"  noise {ty:8} {noise[ty]}")
    profile = {g: {} for g in groups}
    failed = []
    for i in range(layers):
        for g in groups:
            assign, n = probe_assign(tensors, g, i, probe, imatrix_ok, covered)
            if not n:
                continue                                   # nothing changeable in this group/layer
            p = build(assign)
            k = kl(p) if p else None
            profile[g][str(i)] = max(0.0, k) if k is not None else None
            if k is None:
                failed.append(f"{g}.{i}")
        log(f"  layer {i:>3}/{layers}  " + "  ".join(f"{g}={profile[g].get(str(i))}" for g in groups))
    for g in groups:                                       # unmeasured -> PROTECTED, never "free"
        vals = [v for v in profile[g].values() if v is not None]
        hi = max(vals) if vals else 1.0
        for key, v in profile[g].items():
            if v is None:
                profile[g][key] = hi
    return {**profile, "noise": noise, "passthrough": True, "probe": probe, "layers": layers,
            "failed": failed, "ref": "source (passthrough)", "base": "the quantized source itself"}
