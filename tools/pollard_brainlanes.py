#!/usr/bin/env python3
"""pollard-brainlanes -- which runtimes can host a brain on THIS machine?

A brain needs four things from a backbone and nothing else: embeddings in, per-token hidden states
out, and the output embedding for its token codes. Whether a given runtime can do that is a property
of the machine -- the platform, the build, the GPU -- not of the brain, so it is measured here rather
than claimed in a document.

  pollard-brainlanes                       # what is importable, and why not
  pollard-brainlanes --model <path-or-id>  # actually run a forward pass in each lane

Without --model this reports what is INSTALLED. That is the cheap check and it is honest about being
cheap: an import proving nothing about whether a forward pass works. With --model it runs the real
thing -- six tokens through each available lane -- and reports the shapes, which is the only answer
that counts.

Platform notes, measured rather than assumed: MLX is Apple-only and is the natural lane on a Mac.
llama.cpp builds everywhere. exllamav3 and vLLM are CUDA-first and their story differs by platform,
which is exactly why this prints what it finds instead of what it expects.
"""
from __future__ import annotations

import argparse
import importlib.util
import platform
import sys

LANES = [
    ("transformers", "transformers", "the reference path -- every published brain number"),
    ("mlx",          "mlx_lm",       "Apple silicon; native lane on a Mac"),
    ("gguf",         "llama_cpp",    "llama.cpp; builds on every platform"),
    ("exl3",         "exllamav3",    "exllamav3; CUDA-first"),
    ("vllm",         "vllm",         "the serving lane; retrieval only, see the backend notes"),
]


def installed(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--model", help="run a real forward pass in every available lane")
    ap.add_argument("--lane", help="check just this one")
    a = ap.parse_args()

    print(f"\n  {platform.system()} {platform.machine()}, python {sys.version.split()[0]}")
    try:
        import torch
        dev = ("cuda" if torch.cuda.is_available() else
               "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
               else "cpu")
        print(f"  torch {torch.__version__}, device {dev}\n")
    except ImportError:
        raise SystemExit("torch is required: pip install 'pollard-weights[flybrain]'")

    lanes = [l for l in LANES if not a.lane or l[0] == a.lane]
    usable = []
    for name, mod, note in lanes:
        ok = installed(mod)
        print(f"  {name:14s} {'available' if ok else 'not installed':16s} {note}")
        if ok:
            usable.append(name)

    if not a.model:
        print(f"\n  {len(usable)}/{len(lanes)} importable. Pass --model to actually run each one --"
              f"\n  an import does not prove a forward pass works, and that is the claim that counts.")
        return

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from pollard_brain_backends import open_backend
    print()
    for name in usable:
        try:
            b = open_backend(name if name != "transformers" else "auto", a.model, device=dev)
            ids = torch.tensor([[785, 6234, 3409, 374, 4244, 13]])
            emb = b.embed(ids)
            try:
                logits, h = b.forward(emb)
                shapes = f"emb {tuple(emb.shape)}  logits {tuple(logits.shape)}  hidden {tuple(h.shape)}"
            except NotImplementedError:
                h = b.forward_ids(ids)
                shapes = f"emb {tuple(emb.shape)}  hidden {tuple(h.shape)}  (retrieval only)"
            print(f"  {name:14s} WORKS   {shapes}")
        except Exception as e:
            msg = str(e).split("\n")[0][:96]
            print(f"  {name:14s} no      {type(e).__name__}: {msg}")


if __name__ == "__main__":
    main()
