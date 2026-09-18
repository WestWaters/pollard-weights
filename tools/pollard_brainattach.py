#!/usr/bin/env python3
"""pollard-brainattach -- ship a trained brain alongside a finished build.

A brain is an OPTIONAL thing a user may want to carry with a model. It is not part of quantizing
one, so it does not belong inside the build driver: `pollard` builds models and knows nothing about
brains, and this attaches a brain to a build that already exists.

    pollard-brainattach --brain FlyBrain-Pollard-CNSv1.pt --build ./my-build-dir --format gguf

The brain is a separate ~11 MB file, not weights to quantize, so it copies in untouched whatever
lane the build is. What differs per lane is whether it can be USED there yet: the GPTQ lane loads
under transformers, so a brain attaches directly; GGUF, MLX and EXL3 run under their own engines,
which have no hook to inject memory tokens into the residual stream. The file still travels with the
build so the pair never gets separated, and BRAIN.md says which case this is rather than leaving
someone to find out at run time.
"""
from __future__ import annotations

import argparse
import os

LANES = ("gguf", "gptq", "mlx", "exl3", "mx")
LIVE_LANES = ("gptq",)          # lanes whose runtime can actually host a brain today


def attach_brain(brain_path: str, out_dir: str, fmt: str) -> None:
    """Copy `brain_path` into the build at `out_dir` and write BRAIN.md next to it."""
    import shutil
    if not os.path.isfile(brain_path):
        raise SystemExit(f"--brain: no such file {brain_path}")
    dest_dir = out_dir if os.path.isdir(out_dir) else os.path.dirname(out_dir) or "."
    name = os.path.basename(brain_path)
    shutil.copy2(brain_path, os.path.join(dest_dir, name))

    meta = {}
    try:
        import torch
        meta = torch.load(brain_path, map_location="cpu", weights_only=False).get("meta", {})
    except Exception:
        pass
    live = fmt in LIVE_LANES
    note = [f"# Brain: {name}", ""]
    note.append(f"- slots x width : {meta.get('neurons','?')} x {meta.get('width','?')}"
                f"  ({int(meta.get('neurons',0)) * int(meta.get('width',0)) * 4 / 1e6:.1f} MB live state)"
                if meta.get("neurons") else "- (could not read brain metadata)")
    if meta.get("hidden"):
        note.append(f"- trained against a backbone with hidden size {meta['hidden']}")
    note.append("")
    if live:
        note += ["Attach it at run time:", "",
                 "```python", "from pollard_flybrain import FlyBrain, load_backbone",
                 "brain = FlyBrain.load(\"" + name + "\").bind(model, tok)",
                 "brain.feed(open(\"long_document.txt\").read())", "```", ""]
    else:
        note += [f"The {fmt.upper()} lane runs under its own engine, which has no hook to inject",
                 "memory tokens into the residual stream, so the brain cannot attach to THIS build",
                 "yet. It ships here so the pair stays together; use it with the transformers copy of",
                 "the same backbone, or carry it to another model with:", "",
                 "```", f"pollard-flybrain --train 900 --continue-from {name} --model <hf-id> ...", "```", ""]
    note += ["Verify any brain with:", "",
             "```", f"pollard-brainverify --brain {name} --model <hf-id> --filler corpus.txt", "```"]
    with open(os.path.join(dest_dir, "BRAIN.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(note) + "\n")
    print(f"   brain: {name} -> {dest_dir}"
          + ("  (attaches at run time)" if live else f"  (ships with the build; {fmt} cannot host it yet)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--brain", required=True, help="a trained .pt brain (pollard-flybrain / connectome)")
    ap.add_argument("--build", required=True, help="the finished build directory (or a file inside it)")
    ap.add_argument("--format", default="gguf", choices=LANES,
                    help="which lane the build is, so BRAIN.md says whether it can host a brain yet")
    a = ap.parse_args()
    attach_brain(a.brain, a.build, a.format)


if __name__ == "__main__":
    main()
