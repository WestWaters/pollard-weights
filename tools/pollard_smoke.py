#!/usr/bin/env python3
"""pollard-smoke -- will Pollard handle THIS model on THIS machine, before you spend the afternoon?

Pollard's tools are exercised by whatever models have been run through them. A model only breaks one
when it is the first to walk some combination nothing walked before -- a bigger-than-memory
checkpoint, an architecture whose layers differ from one another, a row length a K-quant cannot
hold, a text stack hidden under a vision wrapper. Every one of those is a property of the model's
SHAPE, so none of it needs the model's weights to test.

Two ways to use it:

    pollard-smoke                      # synthetic shapes -> exercise every structural path
    pollard-smoke --model <checkpoint>  # can this machine build THIS model? (exit 1 = no)

The second is the one worth running before a long build. It answers, in seconds, the questions that
otherwise surface hours in: is there a converter here that knows this architecture, where will the
weights actually fit, and can the tools reach this model's layers.

Nothing here is specific to any model or machine -- shapes are generated, paths come from your
environment.
"""
from __future__ import annotations

import argparse, json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


QK_K = 256                      # K-quant block: a row not divisible by this cannot hold one


# ---- synthetic shapes ---------------------------------------------------------------------------
# Module trees, not checkpoints: the walking/allocation bugs are structural, so a stand-in exercises
# the same code for a few bytes instead of a few gigabytes.
class _Node:
    """Stands in for an nn.Module for the tools that only walk attributes."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _mlp(gate=True, up=True, down=True):
    return _Node(gate_proj="W" if gate else None,
                 up_proj="W" if up else None,
                 down_proj="W" if down else None)


def _attn(full=True):
    return _Node(q_proj="W", k_proj="W", v_proj="W" if full else None,
                 o_proj="W" if full else None)


def shape_dense(n=4):
    """Every layer identical -- the case every tool was written against."""
    return _Node(model=_Node(layers=[_Node(mlp=_mlp(), self_attn=_attn()) for _ in range(n)]))


def shape_layer_varying(n=4):
    """Layers that differ: the unused submodules are declared and left None (Gemma4 does this).

    hasattr() is True for them, so a tool that trusts it puts a None where a module belongs."""
    layers = []
    for i in range(n):
        odd = i % 2
        layers.append(_Node(mlp=_mlp(gate=not odd, down=not odd), self_attn=_attn(full=not odd)))
    return _Node(model=_Node(layers=layers))


def shape_vision_language(n=4):
    """A text stack under a vision wrapper -- `model.language_model`, not `model.model`."""
    return _Node(model=_Node(language_model=_Node(
        layers=[_Node(mlp=_mlp(), self_attn=_attn()) for _ in range(n)])))


def shape_missing_group(n=3):
    """A layer with no MLP at all. Legitimate; must contribute nothing rather than raise."""
    return _Node(model=_Node(layers=[_Node(self_attn=_attn()) for _ in range(n)]))


SHAPES = {"dense": shape_dense,
          "layer-varying": shape_layer_varying,
          "vision-language": shape_vision_language,
          "missing-group": shape_missing_group}


# ---- checks -------------------------------------------------------------------------------------
def check_layer_access(shape_name, model):
    """text_layers() reaches the decoder layers, and _linears() never yields a non-module."""
    # Imported here, at the point of use, so this tool stays runnable without the torch extra --
    # the converter check is its most useful mode and needs none. Module-qualified on purpose, so
    # the shared-loader guard keeps its exact meaning and needs no exception for this file.
    from pollard_probe import _linears
    from pollard_load import text_layers as _text_layers
    layers = _text_layers(model)
    if not layers:
        return False, "text_layers() found no decoder layers"
    for i in range(len(layers)):
        for g in ("ffn", "attn"):
            got = _linears(model, i, g)
            if any(x is None for x in got):
                return False, f"layer {i} {g}: a None reached the caller as if it were a module"
    return True, f"{len(layers)} layers reachable, no None submodules"


def check_row_widths(_shape_name=None, _model=None):
    """A K-quant needs rows divisible by QK_K; forcing one elsewhere truncates the file."""
    from pollard_fit import block_safe_type
    bad = []
    for row in (QK_K, QK_K * 3, 896, 1536, 5120, 1):          # 896 and 1 are NOT divisible
        t = block_safe_type("q4_K", row)
        if row % QK_K and t == "q4_K":
            bad.append(f"row {row} kept a K-quant it cannot hold")
    return (not bad), ("; ".join(bad) if bad else "K-quants refused on rows that cannot hold them")


def check_placement(model_dir=None):
    """Where would the weights actually go on this machine?"""
    from pollard_probe import plan_placement, _host_bytes
    if _host_bytes() <= 0:
        return False, f"RAM is unreadable on {sys.platform} -- every budget built on it is wrong"
    if model_dir:
        dev, kw, note = plan_placement(model_dir, "cpu", os.path.join(tempfile.gettempdir(), "pollard-smoke"))
        return True, note or f"placed on {dev}"
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.truncate(400 * (1 << 30))                            # bigger than any machine
    _, kw, note = plan_placement(d, "cpu", os.path.join(d, "off"))
    if not kw and "paged" not in note:
        return False, f"a 400GB model was neither paged nor offloaded: {note}"
    return True, note


def check_converter(model_dir=None):
    """Is there a converter here that knows this architecture?"""
    from pollard_convert import find_converter, model_architectures
    conv, note = find_converter(model_dir)
    if conv is None:
        return False, note
    archs = model_architectures(model_dir) if model_dir else []
    return True, note if archs else f"{note} (no model given -- capability not verified)"


# Types llama-quantize REFUSES to produce without importance data for the tensor. Assigning one of
# these to an uncovered tensor is not a warning -- it is GGML_ASSERT(imatrix != NULL) and a dead run.
IMATRIX_REQUIRED = {"iq2_xxs", "iq2_xs", "iq2_s", "iq1_s", "iq1_m", "q2_k_s",
                    "iq1_kt", "iq2_kt", "iq3_kt", "iq4_kt"}


def check_imatrix_plan(gguf=None, imatrix=None, ftype=None, out_type=None, emb_type=None):
    """Would this quantize ABORT on a tensor the imatrix does not cover?

    llama-quantize hard-aborts with GGML_ASSERT(imatrix != NULL) when a tensor is assigned an
    imatrix-REQUIRED type and the imatrix holds no entry for it -- and it does so AFTER loading the
    model and printing the entire plan. On a 27B that is a long wait for a crash whose cause is one
    line of output scrolled far off the top.

    It is not an exotic case. llama-imatrix does not collect token_embd/output at all, so any build
    tight enough to push those down the ladder walks straight into it; and MTP/`nextn` heads look
    like ordinary attention (`blk.64.attn_k.weight`) while never being calibrated.

    Everything needed to answer this is metadata -- tensor names and imatrix keys -- so it costs
    seconds and no weights. Run it before the build, not after.
    """
    from pollard_calc import imatrix_covered_tensors, read_gguf_tensor_names
    if not gguf:
        return True, "no --gguf given -- not checked"
    names = read_gguf_tensor_names(gguf)
    if not names:
        return False, f"no tensors readable from {gguf}"
    if not imatrix:
        planned = {(ftype or "").lower(), (out_type or "").lower(), (emb_type or "").lower()}
        bad = sorted(t for t in planned if t in IMATRIX_REQUIRED)
        if bad:
            return False, (f"no --imatrix, but {', '.join(bad)} REQUIRES one -- this aborts. "
                           f"Build an imatrix, or pick a K-quant.")
        return True, "no imatrix needed for these types"

    covered = imatrix_covered_tensors(imatrix)
    if covered is None:
        return False, (f"could not read {imatrix} as either a GGUF or a legacy .dat imatrix -- "
                       f"coverage unknown, so an imatrix-required type cannot be cleared")

    # Only MATMULS are quantized -- norms and biases stay F32, so flagging `attn_norm.weight` or
    # `ssm_dt.bias` is noise that buries the real hits. The GGUF itself says which is which: a
    # 2-D tensor is a matmul. That is ground truth from the file, so this needs no rule table and
    # cannot drift from one.
    try:
        from gguf import GGUFReader
        names = [t.name for t in GGUFReader(gguf).tensors if len(t.shape) >= 2]
    except Exception:
        pass                                                 # names-only fallback: over-report, never miss
    base = (ftype or "").lower()
    overrides = {"output.weight": (out_type or "").lower(),
                 "token_embd.weight": (emb_type or "").lower()}
    risky = []
    for nm in names:
        ty = overrides.get(nm) or base
        if ty in IMATRIX_REQUIRED and nm not in covered:
            risky.append((nm, ty))
    if risky:
        shown = ", ".join(f"{n} -> {t}" for n, t in risky[:4])
        more = f" (+{len(risky) - 4} more)" if len(risky) > 4 else ""
        fix = ""
        if any(n in ("output.weight", "token_embd.weight") for n, _ in risky):
            fix = ("  FIX: --output-tensor-type / --token-embedding-type to a non-imatrix type "
                   "(Q6_K, Q5_K, IQ4_XS, IQ3_S, Q2_K).")
        elif risky:
            fix = "  FIX: pin these to a non-imatrix type, or extend the calibration to cover them."
        return False, (f"{len(risky)} tensor(s) would take an imatrix-required type with NO "
                       f"coverage -- llama-quantize ABORTS on these: {shown}{more}.{fix}")
    return True, f"{len(covered)} covered; every imatrix-required assignment is backed"


def check_arch_coverage(gguf=None, imatrix=None):
    """Does Pollard RECOGNISE every matmul this model's calibration covers?

    The imatrix is the ground truth for what actually gets quantized. A covered tensor that no
    group rule matches is a weight family the tooling has never seen -- and the silent result is
    whole layers scored at cost 0.0, which the allocator reads as 'free to crush'.

    Qwen3.8-27B is how this was found: 48 of its 65 blocks mix with a state-space operator rather
    than attention, 240 of 496 covered matmuls matched nothing, and the profile looked healthy."""
    if not (gguf and imatrix):
        return True, "needs --gguf and --imatrix -- not checked"
    import re as _re
    from pollard_calc import imatrix_covered_tensors
    from pollard_probe import _gguf_slot
    covered = imatrix_covered_tensors(imatrix)
    if covered is None:
        return False, "imatrix unreadable -- cannot verify architecture coverage"
    unknown = sorted({_re.sub(r"^blk\.\d+\.", "", n) for n in covered
                      if _re.match(r"^blk\.\d+\..+\.weight$", n)
                      and _gguf_slot(n, ["ffn", "attn"]) is None})
    if unknown:
        return False, (f"{len(unknown)} calibrated tensor kind(s) match no group rule, so their "
                       f"layers would score 0.0 (= free to crush): {', '.join(unknown)}. "
                       f"Add them to GROUP_GGUF in pollard_probe.py.")
    return True, "every calibrated matmul maps to a group"


# ---- run ----------------------------------------------------------------------------------------
def _line(ok, name, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name:22} {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", help="a real checkpoint to check (else synthetic shapes only)")
    ap.add_argument("--shapes", default="all", help="comma-separated subset of: " + ", ".join(SHAPES))
    # The BUILD preflight: the questions that otherwise abort a 27B quantize an hour in.
    ap.add_argument("--gguf", help="the source GGUF you are about to quantize")
    ap.add_argument("--imatrix", help="the imatrix that build will use (.dat or GGUF)")
    ap.add_argument("--ftype", help="the quant type you are about to build (e.g. IQ2_XXS)")
    ap.add_argument("--output-tensor-type", help="as passed to llama-quantize")
    ap.add_argument("--token-embedding-type", help="as passed to llama-quantize")
    a = ap.parse_args()
    ok = True

    if a.gguf or a.imatrix:
        print(f"== pollard-smoke :: build preflight")
        for name, fn in (("imatrix plan", lambda: check_imatrix_plan(
                              a.gguf, a.imatrix, a.ftype,
                              a.output_tensor_type, a.token_embedding_type)),
                         ("arch coverage", lambda: check_arch_coverage(a.gguf, a.imatrix))):
            try:
                good, detail = fn()
            except Exception as e:
                good, detail = False, f"{type(e).__name__}: {e}"
            ok &= _line(good, name, detail)
        print()

    if a.model:
        print(f"== pollard-smoke :: {a.model}")
        for name, fn in (("converter", check_converter), ("placement", check_placement)):
            try:
                good, detail = fn(a.model)
            except Exception as e:                              # a check must never mask a result
                good, detail = False, f"{type(e).__name__}: {e}"
            ok &= _line(good, name, detail)
        print("\n  (structural checks below use synthetic shapes -- no weights loaded)")
    else:
        print("== pollard-smoke :: synthetic shapes (no model given)")

    names = list(SHAPES) if a.shapes == "all" else [s.strip() for s in a.shapes.split(",") if s.strip()]
    for name in names:
        if name not in SHAPES:
            ok &= _line(False, name, "unknown shape")
            continue
        try:
            good, detail = check_layer_access(name, SHAPES[name]())
        except Exception as e:
            good, detail = False, f"{type(e).__name__}: {e}"
        ok &= _line(good, name, detail)

    for name, fn in (("row widths", check_row_widths), ("placement", check_placement)):
        if a.model and name == "placement":
            continue                                            # already reported against the model
        try:
            good, detail = fn()
        except Exception as e:
            good, detail = False, f"{type(e).__name__}: {e}"
        ok &= _line(good, name, detail)

    print(f"\n  {'all checks passed' if ok else 'FAILURES above -- fix before a long build'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
