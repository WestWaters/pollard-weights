#!/usr/bin/env python3
"""pollard-convert -- HF weights -> f16/bf16 GGUF, on whatever machine you are sitting at.

Pollard does not reimplement the conversion: llama.cpp carries a model class per architecture
(`conversion/gemma.py` alone registers four Gemma4 variants), and racing upstream on every new model
release is a maintenance tax with no win at the end of it. What Pollard OWNS is the question
"can THIS machine convert THIS model, and if not, exactly what is missing" -- which is the part that
kept breaking.

The old behaviour was to look in two hard-coded spots and then return the bare string
"convert_hf_to_gguf.py" and hope the shell found it. On a box where it was absent or too old, the
one-shot died deep in a build with an error about the model rather than about the toolchain -- and
the fix looked like "move a 23.8GB GGUF across the network" instead of "copy a 3MB script". Both
machines had the weights the whole time.

So: find a converter that actually registers this model's architecture, say which one and why, and
refuse clearly when none does.

    pollard-convert --model <hf-dir> --outfile model-f16.gguf
    pollard-convert --model <hf-dir> --check          # can this machine do it? (exit 1 = no)
    pollard-convert --list                            # every converter found, and what it knows
"""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

# Where a converter plausibly lives, in priority order. Every entry is checked on every platform --
# a Mac-only or Linux-only search path is how a tool silently stops working on the build box.
def _search_paths():
    here = Path(__file__).resolve().parents[1]
    env = os.environ.get("POLLARD_CONVERTER")
    cands = []
    if env:
        cands.append(Path(env))
    home = Path(os.environ.get("POLLARD_HOME", Path.home() / "pollard"))
    for root in (here / "runtime" / "llama.cpp",      # the runtime beside this Pollard checkout
                 home / "convert",                    # a staged converter in the workspace
                 home / "llama.cpp",
                 Path("C:/pollard/convert"),          # the Windows build box
                 Path("C:/pollard/llama-current"),
                 Path.home() / "llama.cpp",
                 Path("/usr/local/share/llama.cpp"),
                 Path("/opt/llama.cpp")):
        cands.append(root / "convert_hf_to_gguf.py")
    found = shutil.which("convert_hf_to_gguf.py")
    if found:
        cands.append(Path(found))
    return cands


def model_architectures(model_dir) -> list[str]:
    """The `architectures` a checkpoint declares -- the exact class names a converter registers."""
    cfg = Path(model_dir) / "config.json"
    if not cfg.is_file():
        return []
    try:
        d = json.loads(cfg.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    archs = d.get("architectures") or []
    text = d.get("text_config") or {}
    archs += text.get("architectures") or []
    return [str(a) for a in archs]


def converter_registers(conv_py, arch: str) -> bool:
    """Does this converter carry a model class for `arch`?

    Upstream split the monolithic script into a package, so the entry point is ~16KB of CLI and the
    registrations live in `conversion/*.py`. Grepping only the script would call every modern
    converter incapable -- so search the script AND the package beside it."""
    conv_py = Path(conv_py)
    if not conv_py.is_file():
        return False
    targets = [conv_py]
    pkg = conv_py.parent / "conversion"
    if pkg.is_dir():
        targets += sorted(pkg.rglob("*.py"))
    for f in targets:
        try:
            if arch in f.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False


def find_converter(model_dir=None):
    """(path, note) for a converter that can handle `model_dir`, or (None, why-not).

    With no model, returns the first converter present -- capability is only meaningful against a
    specific architecture."""
    archs = model_architectures(model_dir) if model_dir else []
    present = [c for c in _search_paths() if Path(c).is_file()]
    if not present:
        return None, ("no convert_hf_to_gguf.py found. Point POLLARD_CONVERTER at one, or put a "
                      "llama.cpp checkout beside Pollard (runtime/llama.cpp).")
    if not archs:
        return present[0], f"{present[0]} (architecture unknown -- not verified)"
    for c in present:
        for a in archs:
            if converter_registers(c, a):
                return c, f"{c} registers {a}"
    names = ", ".join(archs)
    where = "\n  ".join(str(p) for p in present)
    where_arg = model_dir or "<repo-or-dir>"
    # A converter that does not know this architecture IS the new-architecture case, which Pollard
    # already has a path for: onboard it, and the contribution makes the next person's model of that
    # family one-shot. Dead-ending here throws that away.
    return None, (f"no converter on this machine registers {names}.\n  Checked:\n  {where}\n"
                  f"  -> onboard this architecture:  pollard-onboard --model {where_arg} --contribute\n"
                  "     (audits the real tensor names against Pollard's matchers and writes a\n"
                  "      PR-ready contribution -- read-only, no weights downloaded)\n"
                  "  If the converter is merely older than the model, copying a newer\n"
                  "  convert_hf_to_gguf.py + conversion/ + gguf-py/ beside it is ~3MB.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", help="HF checkout to convert")
    ap.add_argument("--outfile", help="output GGUF (default <model>-f16.gguf beside the model)")
    ap.add_argument("--outtype", default="f16", help="f16 (default), bf16, f32, q8_0")
    ap.add_argument("--check", action="store_true", help="report capability only; exit 1 if it cannot")
    ap.add_argument("--list", action="store_true", help="every converter found on this machine")
    ap.add_argument("--python", default=sys.executable, help="interpreter to run the converter with")
    a = ap.parse_args()

    if a.list:
        archs = model_architectures(a.model) if a.model else []
        for c in _search_paths():
            if not Path(c).is_file():
                continue
            known = [x for x in archs if converter_registers(c, x)]
            print(f"  {c}" + (f"   registers: {', '.join(known)}" if known
                              else ("   (does NOT register " + ", ".join(archs) + ")") if archs else ""))
        return

    if not a.model:
        ap.error("--model is required (or use --list)")
    conv, note = find_converter(a.model)
    archs = model_architectures(a.model)
    print(f"== pollard-convert :: {a.model}")
    print(f"   architecture : {', '.join(archs) or 'unknown'}")
    print(f"   converter    : {note}")
    if conv is None:
        raise SystemExit(1)
    if a.check:
        return

    out = a.outfile or str(Path(a.model).with_name(Path(a.model).name + f"-{a.outtype}.gguf"))
    cmd = [a.python, str(conv), str(a.model), "--outtype", a.outtype, "--outfile", out]
    print("   " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0 or not os.path.exists(out):
        raise SystemExit(f"\n   conversion failed (exit {r.returncode}) -- no GGUF at {out}")
    print(f"   wrote {out}  ({os.path.getsize(out)/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
