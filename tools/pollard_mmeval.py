#!/usr/bin/env python3
"""pollard-mmeval -- did quantization break the model's EYES?

Every number on a Pollard card measures the text path: perplexity, KL divergence, top-1. None of
them touch the projector. So a multimodal build can read perfectly on the board and be blind in
practice, and nothing we ship would notice -- gemma-4-12B-it went out with a byte-exact 175MB
mmproj and a card whose every figure is text perplexity.

This is the same shape as pollard-bench, pointed at the other half of the model: run the SAME
inputs through a rung and (optionally) through the reference, and report what changed.

    pollard-mmeval --gguf model-IQ2_XXS.gguf --mmproj mmproj-BF16.gguf
    pollard-mmeval --gguf rung.gguf --mmproj mm.gguf --ref f16.gguf --ref-mmproj mm.gguf

The probes are GENERATED, not downloaded: flat-colour shapes, counts, positions and a rendered
word, each with an answer that is true by construction. That buys three things a dataset does not.
It is deterministic, so two runs on two machines compare. It needs no network and no licence. And
it cannot leak into calibration, because the images did not exist until this ran -- the trap that
put gemma-4-12B-it's perplexity on its own calibration corpus does not have a visual analogue here.

What this DOES measure: whether the vision pathway still resolves colour, count, shape, position
and rendered text after the weights were crushed. That is the gross-damage question, and it is the
one that decides whether a rung is publishable as multimodal.

What it does NOT measure: fine-grained visual reasoning, OCR at length, or anything about audio.
A model can pass every probe here and still be worse at a real VQA benchmark. Read a pass as "the
eyes still work", not as a quality score.

This tool imports nothing from the rest of Pollard and is imported by nothing. It owns its probe
generation, its subprocess handling and its scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys


# ---- probes ------------------------------------------------------------------------------------
# (name, draw(fn), prompt, accepted answers). Answers are lowercase substrings; a reply counts if
# ANY of them appears. Kept deliberately easy: this is a damage detector, not a benchmark. A model
# that cannot say a solid red square is red has lost the vision path, whatever its perplexity says.
_COLOURS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 60)}


def _probe_specs():
    specs = []
    for name, rgb in _COLOURS.items():
        specs.append((f"colour-{name}", ("solid", rgb),
                      "What color is the shape in this image? Answer with one word.",
                      (name,)))
    for n in (2, 3, 5):
        specs.append((f"count-{n}", ("count", n),
                      "How many squares are in this image? Answer with just the number.",
                      (str(n), _WORDS[n])))
    for shape in ("circle", "square", "triangle"):
        specs.append((f"shape-{shape}", ("shape", shape),
                      "What shape is in this image? Answer with one word: circle, square, or triangle.",
                      (shape,)))
    for side in ("left", "right"):
        specs.append((f"position-{side}", ("position", side),
                      "Is the circle on the left or the right side of this image? Answer left or right.",
                      (side,)))
    for word in ("CAT", "DOG"):
        specs.append((f"text-{word}", ("text", word),
                      "What word is written in this image? Answer with just the word.",
                      (word.lower(),)))
    return specs


_WORDS = {2: "two", 3: "three", 5: "five"}


def _draw(kind, arg, path, size=448):
    """Render one probe. Big, high-contrast, centred -- a failure here is the pathway, not acuity."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    m = size // 6
    if kind == "solid":
        d.rectangle([m, m, size - m, size - m], fill=arg)
    elif kind == "count":
        # a row of identical squares, generously spaced so counting is not a crowding test
        w = size // (arg * 2 + 1)
        for i in range(arg):
            x = w + i * 2 * w
            d.rectangle([x, size // 2 - w // 2, x + w, size // 2 + w // 2], fill=(30, 60, 220))
    elif kind == "shape":
        if arg == "circle":
            d.ellipse([m, m, size - m, size - m], fill=(30, 60, 220))
        elif arg == "square":
            d.rectangle([m, m, size - m, size - m], fill=(30, 60, 220))
        else:
            d.polygon([(size // 2, m), (m, size - m), (size - m, size - m)], fill=(30, 60, 220))
    elif kind == "position":
        cx = size // 4 if arg == "left" else 3 * size // 4
        d.ellipse([cx - m, size // 2 - m, cx + m, size // 2 + m], fill=(30, 60, 220))
    elif kind == "text":
        font = None
        for cand in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                     "C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            if os.path.exists(cand):
                try:
                    font = ImageFont.truetype(cand, size // 3)
                    break
                except Exception:
                    pass
        if font is None:                                   # bitmap fallback: small, but legible
            font = ImageFont.load_default()
        box = d.textbbox((0, 0), arg, font=font)
        d.text(((size - (box[2] - box[0])) // 2, (size - (box[3] - box[1])) // 2 - box[1]),
               arg, fill=(0, 0, 0), font=font)
    img.save(path)
    return path


def build_probes(outdir):
    os.makedirs(outdir, exist_ok=True)
    out = []
    for name, (kind, arg), prompt, answers in _probe_specs():
        p = os.path.join(outdir, f"{name}.png")
        _draw(kind, arg, p)
        out.append({"name": name, "image": p, "prompt": prompt, "answers": answers})
    return out


# ---- running -----------------------------------------------------------------------------------
def find_mtmd(explicit=None):
    """llama-mtmd-cli, wherever this machine keeps it. Named plainly so nothing else is picked up."""
    if explicit:
        return explicit if os.path.exists(explicit) or shutil.which(explicit) else None
    for cand in ("llama-mtmd-cli", "llama-mtmd-cli.exe"):
        hit = shutil.which(cand)
        if hit:
            return hit
    home = os.environ.get("POLLARD_HOME", os.path.expanduser("~/pollard"))
    for root in (os.path.join(home, "bin"),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                              "runtime", "llama.cpp", "build", "bin")):
        for cand in ("llama-mtmd-cli", "llama-mtmd-cli.exe"):
            p = os.path.join(root, cand)
            if os.path.exists(p):
                return p
    return None


def ask(cli, model, mmproj, image, prompt, ngl, timeout=300):
    """One image + one question -> the model's reply, or None if it produced nothing.

    --jinja is tried FIRST. Gemma4's chat template is one llama.cpp's built-in parser refuses --
    it aborts with `std::runtime_error: this custom template is not supported, try using --jinja`
    before generating a single token. Without this the tool reports <no output> on every probe,
    which reads exactly like a model that has gone blind. It is worth being loud about: a harness
    bug that mimics the failure it is looking for is the worst kind, and this one would have had us
    "discover" that a published rung could not see.
    """
    kw = {}
    if os.name == "nt":                                    # keep console events off the child
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    base = [cli, "-m", model, "--mmproj", mmproj, "--image", image,
            # 220, not 48: a reasoning-tuned model spends its first hundred-odd tokens inside a
            # thinking block, so a short budget cuts it off before the answer and scores a MISS on
            # a model that was about to be right. pollard-bench's gate learned this the same way.
            "-p", prompt, "-ngl", str(ngl), "-n", "220", "--temp", "0"]
    r = None
    for extra in (["--jinja"], []):                        # builds without --jinja fall through
        try:
            r = subprocess.run(base + extra, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=timeout, stdin=subprocess.DEVNULL, **kw)
        except subprocess.TimeoutExpired:
            return None
        blob = (r.stdout or "") + (r.stderr or "")
        if "unrecognized argument" not in blob and "invalid argument" not in blob:
            if "not supported, try using --jinja" in blob:
                continue                                   # this build needs it and did not get it
            break
    if r is None:
        return None
    text = (r.stdout or "")
    # mtmd-cli echoes the prompt and prints timings; keep what follows the prompt, drop the log tail
    if prompt in text:
        text = text.split(prompt, 1)[1]
    text = re.split(r"\n(?:llama_|main:|load_|clip_|encode_|decode_)", text)[0]
    return text.strip()


def scored(reply, answers):
    if reply is None:
        return False
    low = reply.lower()
    return any(a.lower() in low for a in answers)


# ---- report ------------------------------------------------------------------------------------
def run_model(cli, model, mmproj, probes, ngl, label):
    rows, hits = [], 0
    print(f"\n== {label} ==", flush=True)
    for p in probes:
        reply = ask(cli, model, mmproj, p["image"], p["prompt"], ngl)
        ok = scored(reply, p["answers"])
        hits += ok
        short = (reply or "<no output>").replace("\n", " ")[:56]
        print(f"  {'OK ' if ok else 'MISS'}  {p['name']:16} -> {short}", flush=True)
        rows.append({"probe": p["name"], "reply": reply, "pass": bool(ok)})
    pct = 100.0 * hits / max(len(probes), 1)
    print(f"  {hits}/{len(probes)} probes ({pct:.1f}%)", flush=True)
    return rows, pct


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True, help="the quantized model to check")
    ap.add_argument("--mmproj", required=True, help="its projector (mmproj-*.gguf)")
    ap.add_argument("--ref", help="reference model (f16) -- scores the SAME probes for comparison")
    ap.add_argument("--ref-mmproj", help="the reference's projector (default: --mmproj)")
    ap.add_argument("--ngl", type=int, default=0, help="layers on the GPU (default 0)")
    ap.add_argument("--probes-dir", help="where to write the generated images")
    ap.add_argument("--out", help="write a results.json")
    ap.add_argument("--llama-mtmd-cli", help="path to llama-mtmd-cli")
    a = ap.parse_args()

    cli = find_mtmd(a.llama_mtmd_cli)
    if not cli:
        sys.exit("llama-mtmd-cli not found -- build llama.cpp with multimodal support, or pass "
                 "--llama-mtmd-cli. Without it the projector cannot be exercised at all.")
    for f in (a.gguf, a.mmproj) + ((a.ref,) if a.ref else ()):
        if not os.path.exists(f):
            sys.exit(f"file not found: {f}")

    pdir = a.probes_dir or os.path.join(os.path.dirname(os.path.abspath(a.gguf)), "mmeval-probes")
    probes = build_probes(pdir)
    print(f"== pollard-mmeval :: {len(probes)} generated probes in {pdir}")
    print(f"   cli {cli}")

    rows, pct = run_model(cli, a.gguf, a.mmproj, probes, a.ngl, os.path.basename(a.gguf))
    payload = {"model": a.gguf, "mmproj": a.mmproj, "probes": len(probes),
               "pass_pct": pct, "rows": rows}

    if a.ref:
        ref_rows, ref_pct = run_model(cli, a.ref, a.ref_mmproj or a.mmproj, probes, a.ngl,
                                      os.path.basename(a.ref) + " (reference)")
        agree = sum(1 for x, y in zip(rows, ref_rows) if x["pass"] == y["pass"])
        payload.update({"ref": a.ref, "ref_pass_pct": ref_pct,
                        "agreement_pct": 100.0 * agree / max(len(probes), 1),
                        "ref_rows": ref_rows})
        print(f"\n  reference {ref_pct:.1f}%   this build {pct:.1f}%   "
              f"delta {pct - ref_pct:+.1f} points")
        # The reference is the ceiling. A rung that drops far below it lost the vision path in
        # quantization; a reference that itself scores low means these probes do not suit this
        # model, and the comparison -- not the absolute -- is what to read.
        if ref_pct < 60:
            print("  NOTE: the REFERENCE scores low, so these probes do not suit this model. "
                  "Compare the two numbers, not the absolute.")
        elif pct < ref_pct - 25:
            print("  WARNING: this build is far below its own f16. The projector survived the "
                  "conversion but the language side can no longer use it -- do not publish this "
                  "rung as multimodal without saying so.")

    if a.out:
        json.dump(payload, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
