#!/usr/bin/env python3
"""Move a build between lanes without requantizing it.

Requantizing lane-to-lane is a trap. llama.cpp's own --allow-requantize warns it "can severely
reduce quality compared to quantizing from 16bit or 32bit", and it is right: you would be
quantizing already-quantized weights and compounding the error. The lanes do not share atoms
either -- GGUF K-quants, EXL3 trellis and NVFP4 are different representations for different
hardware.

But Pollard does not need to requantize, because Pollard's OWN allocation is the portable thing.
sensitivity.json is measured once, per model, in bits-of-KL-cost -- not in any lane's atoms -- and
every emitter reads it:

    pollard-fit      --sensitivity   -> GGUF
    pollard-mlx      --sensitivity   -> MLX
    pollard-mx       --sensitivity   -> NVFP4 / MXFP4
    pollard-export   --sensitivity   -> vLLM / SGLang
    pollard-gptq     --recipe        -> GPTQ
    pollard-exl3     --recipe        -> EXL3

So converting GGUF -> MLX is not a conversion at all. It is the same measured allocation emitted
again from the source, which is why it costs nothing in quality.

Nothing here refuses. If a piece is missing, it says which tool produces it and puts that step
first, because "you cannot do that" is never as useful as "here is the order to do it in".

    python -m pollard_studio.convert <build path> MLX
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# how each lane takes Pollard's allocation, and what it emits
LANES = {
    "GGUF": {"tool": "pollard_fit", "takes": "--sensitivity", "source": "gguf",
             "note": "llama.cpp / ik_llama. Trellis atoms need the ik_llama fork."},
    "MLX": {"tool": "pollard_mlx", "takes": "--sensitivity", "source": "hf",
            "note": "Apple silicon."},
    "MX": {"tool": "pollard_mx", "takes": "--sensitivity", "source": "hf",
           "note": "NVFP4 (Blackwell) or MXFP4."},
    "GPTQ": {"tool": "pollard_gptq", "takes": "--recipe", "source": "hf",
             "note": "full-Hessian error feedback; takes a recipe, not a raw profile."},
    "EXL3": {"tool": "pollard_exl3", "takes": "--recipe", "source": "hf",
             "note": "exllamav3 trellis; takes a per-tensor YAML recipe."},
    "VLLM": {"tool": "pollard_export", "takes": "--sensitivity", "source": "hf",
             "note": "vLLM / SGLang-loadable checkpoint."},
}
# what produces the portable allocation, cheapest first
PROFILERS = [
    ("pollard_probe", "cheap, runs on any box -- same output shape as the measured one"),
    ("pollard_sensitivity", "measured KL cost per tensor group; slower, and the better input"),
    ("pollard_palette", "mixed-alphabet allocation for targets below 2 bits"),
]


def find_source(build: Path, workspace_models: list | None = None) -> dict:
    """The fp16/bf16 weights a build came from. Looked for, never assumed."""
    build = Path(build)
    root = build.parent if build.is_file() else build
    stem = build.stem.split("-Pollard")[0].split(".gguf")[0]

    cands: list[Path] = []
    home = root
    for _ in range(4):                               # walk up to $POLLARD_HOME
        dl = home / "downloads"
        if dl.is_dir():
            cands += [p for p in dl.iterdir()
                      if p.is_dir() and list(p.glob("*.safetensors"))]
            cands += [p for p in dl.glob("*.gguf")]
            break
        home = home.parent

    def score(p: Path) -> int:
        n = p.name.lower().replace("__", "/").replace("-", "").replace("_", "")
        s = stem.lower().replace("-", "").replace("_", "")
        return sum(1 for i in range(0, len(s), 4) if s[i:i + 4] and s[i:i + 4] in n)

    ranked = sorted(cands, key=score, reverse=True)
    best = ranked[0] if ranked and score(ranked[0]) > 1 else None
    return {"found": best is not None, "path": str(best) if best else None,
            "kind": ("hf" if best and best.is_dir() else "gguf" if best else None),
            "candidates": [str(p) for p in ranked[:5]]}


# How good a source is, by MEASURED bits per weight. A source does not have to be fp16 -- Q8_0 is
# near-lossless and a perfectly good place to emit from, and refusing it would send people off to
# re-download 300 GB for nothing. What matters is saying plainly what a given source costs, so the
# choice is informed rather than blocked.
SOURCE_TIERS = (
    (15.0, "ideal", "f16/bf16 source -- nothing has been discarded yet."),
    (7.5, "excellent", "Q8-class source. Near-lossless; emitting from this is fine and is what "
                       "most people should do rather than re-downloading the f16."),
    (5.0, "usable", "Q6-class source. Slightly lossy, so the new build inherits a little error. "
                    "Fine for a smaller target; prefer f16 or Q8 if you have one."),
    (3.5, "lossy", "Q4-class source. The new build inherits this one's error ON TOP of its own. "
                   "Worth it only if the original weights are genuinely gone."),
    (0.0, "poor", "Sub-4-bit source. Emitting from here compounds two lots of quantization error "
                  "and the result will not reflect what Pollard can do. Find the original if you "
                  "possibly can -- but it will still run if you choose to."),
)


def source_quality(path: str | Path) -> dict:
    """What emitting from this source costs, measured rather than assumed from the filename."""
    p = Path(path)
    bpw = None
    try:
        if p.is_file() and p.suffix == ".gguf":
            from . import ggufread
            bpw = ggufread.summarise(ggufread.read(p))["bpw"]
        elif p.is_dir():
            from . import saferead
            bpw = saferead.summarise(saferead.read(p))["bpw"]
    except Exception:
        bpw = None
    if bpw is None:
        return {"bpw": None, "tier": "unknown",
                "note": "could not read this source's precision; it will still be used."}
    for floor, tier, note in SOURCE_TIERS:
        if bpw >= floor:
            return {"bpw": round(bpw, 2), "tier": tier, "note": note}
    return {"bpw": round(bpw, 2), "tier": "poor", "note": SOURCE_TIERS[-1][2]}


def find_profile(build: Path) -> dict:
    """An existing allocation next to the build, in any of the forms Pollard writes."""
    build = Path(build)
    root = build.parent if build.is_file() else build
    for name in ("sensitivity.json", "profile.json", "pollard-sensitivity.json"):
        for where in (root, root.parent):
            p = where / name
            if p.exists():
                return {"found": True, "path": str(p), "kind": "sensitivity"}
    side = build.with_suffix(build.suffix + ".tensor-types.txt")
    if side.exists():
        # the per-tensor recipe the build was GIVEN. Lane-specific (ggml atom names), so it
        # records what happened rather than transferring to another lane's atom space.
        return {"found": True, "path": str(side), "kind": "tensor-types",
                "portable": False}
    return {"found": False, "path": None, "kind": None}


def plan(build: str | Path, target: str) -> dict:
    """Ordered steps to get this model onto `target`. Always a route, never a refusal."""
    target = target.upper()
    if target not in LANES:
        # not a refusal either: name what IS available
        return {"ok": False, "steps": [],
                "guidance": f"No emitter named {target!r}. Pollard emits: "
                            + ", ".join(LANES) + "."}
    spec = LANES[target]
    build = Path(build)
    src = find_source(build)
    prof = find_profile(build)
    steps, notes = [], []

    if not src["found"]:
        steps.append({
            "tool": "pollard_ls", "args": ["--paths"],
            "why": "locate the fp16/bf16 source this build came from. Every lane emits FROM the "
                   "source -- requantizing the quantized build would compound its error.",
            "blocking": True,
        })
        notes.append("source weights not found near the build; "
                     + (f"closest matches: {', '.join(Path(c).name for c in src['candidates'][:2])}"
                        if src["candidates"] else "nothing similar in downloads/"))
    elif target == "GGUF" and src["kind"] == "hf":
        steps.append({"tool": "pollard_convert", "args": ["--model", src["path"]],
                      "why": "this lane starts from a GGUF; convert the source once first.",
                      "blocking": False})
    elif target != "GGUF" and src["kind"] == "gguf":
        notes.append("the source found is a GGUF; this lane wants the original HF weights. "
                     "Point --model at the HF checkpoint if you still have it.")

    quality = source_quality(src["path"]) if src["found"] else {"tier": "unknown", "bpw": None}
    if src["found"]:
        notes.append(f"source is {quality['bpw']} bpw ({quality['tier']}). {quality['note']}")

    portable = prof["found"] and prof.get("portable", True)
    if not portable:
        steps.append({
            "tool": PROFILERS[0][0], "args": ["--model", src["path"] or "<source>"],
            "why": ("a .tensor-types.txt records what THIS build was given, in ggml atom names -- "
                    "it does not transfer to another lane's atoms. sensitivity.json is measured in "
                    "KL cost, which every emitter reads."
                    if prof["found"] else
                    "no portable allocation yet. This is the artifact that makes a lane change "
                    "free: measured once, read by every emitter."),
            "blocking": False,
        })

    emit = {"tool": spec["tool"], "args": [], "why": f"emit into {target}. {spec['note']}",
            "blocking": False}
    if src["found"]:
        emit["args"] += ["--model" if spec["source"] == "hf" else "--gguf", src["path"]]
    if portable and spec["takes"] == "--sensitivity":
        emit["args"] += ["--sensitivity", prof["path"]]
    elif spec["takes"] == "--recipe":
        emit["why"] += (" Takes a per-tensor recipe rather than the raw profile, so the "
                        "allocation is translated into this lane's atoms first.")
    steps.append(emit)

    return {
        "ok": True, "target": target, "steps": steps, "notes": notes,
        "source": src, "profile": prof, "source_quality": quality,
        "requantize": False,
        "guidance": (f"{len(steps)} step{'s' if len(steps) != 1 else ''} from here to {target}. "
                     "Nothing is requantized: the allocation is measured once and re-emitted."),
    }


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    p = plan(sys.argv[1], sys.argv[2])
    print(p["guidance"])
    for i, s in enumerate(p["steps"], 1):
        mark = "!" if s.get("blocking") else " "
        print(f"  {mark}{i}. {s['tool'].replace('_', '-')} {' '.join(map(str, s['args']))}")
        print(f"      {s['why']}")
    for n in p.get("notes", []):
        print(f"  note: {n}")


if __name__ == "__main__":
    main()
