#!/usr/bin/env python3
"""pollard-diffusion -- generate with a diffusion LLM (Dream / LLaDA / RND1 / DiffusionGemma) the way it
actually decodes: unmasking a canvas over N steps, through llama-diffusion-cli.

llama-server cannot decode these architectures at all (upstream included), so every place Pollard makes
a model GENERATE -- the coherence gate, Studio chat, the post-build sanity -- routes here when the GGUF's
general.architecture says diffusion. Quantizing, imatrix and the memory-fit math are unchanged: a Dream
GGUF is a Qwen2 body with Qwen2 tensor names and quantizes like one. What changes is how you run it and
what a "score" means -- AR perplexity / next-token KL are not this model's objective, so the gate (does the
unmasked canvas read as an answer?) is the check that carries weight until a masked-prediction KL lands.

    pollard-diffusion --gguf model.gguf -p "Explain in two sentences why the sky is blue." -n 128
    pollard-diffusion --gguf model.gguf --is-diffusion            # exit 0 if the arch decodes by diffusion

Everything here is model-side and self-contained: stdlib + pollard_calc's GGUF header reader.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# Kept in ONE place; pollard_modelkind imports it. GGUF spells it "diffusion-gemma", HF "diffusion_gemma".
DIFFUSION_ARCHS = frozenset({"dream", "llada", "llada-moe", "rnd1", "diffusion-gemma"})

#: sensible defaults per architecture family -- Dream is timestep-scheduled, LLaDA block-scheduled
_DEFAULTS = {
    "dream":           {"steps": 128, "algorithm": 3, "eps": 0.001},          # 3 = entropy-based
    "rnd1":            {"steps": 128, "algorithm": 3, "eps": 0.001},
    "llada":           {"steps": 128, "algorithm": 0, "block_length": 32},
    "llada-moe":       {"steps": 128, "algorithm": 0, "block_length": 32},
    "diffusion-gemma": {"steps": 128, "algorithm": 0, "block_length": 32},
}


def norm_arch(arch) -> str:
    return str(arch or "").strip().lower().replace("_", "-")


def is_diffusion_arch(arch) -> bool:
    return norm_arch(arch) in DIFFUSION_ARCHS


def arch_of(gguf: str) -> str:
    """general.architecture of a GGUF, or '' if it cannot be read. Header only -- never loads weights."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pollard_calc import read_gguf_meta
        return norm_arch(read_gguf_meta(gguf).get("general.architecture", ""))
    except Exception:
        return ""


def is_diffusion(gguf: str) -> bool:
    return is_diffusion_arch(arch_of(gguf))


def cli_bin(explicit: str | None = None) -> str | None:
    """Where llama-diffusion-cli is: told, else the runtime build install.sh makes, else PATH."""
    if explicit and os.path.isfile(explicit):
        return explicit
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pollard_calc import find_llama_bin
        return find_llama_bin("llama-diffusion-cli")
    except Exception:
        return None


def generate(model: str, prompt: str, n_predict: int = 256, *, steps: int | None = None,
             algorithm: int | None = None, block_length: int | None = None, eps: float | None = None,
             ngl: int = 0, ctx: int = 4096, temperature: float = 0.0, top_k: int | None = None,
             top_p: float | None = None, seed: int | None = None, system: str | None = None,
             binary: str | None = None, timeout: float = 1800) -> dict:
    """One diffusion generation. Returns {"ok", "text", "tokens", "stop_reason", "seconds", "argv", "error"}.

    The CLI applies the model's own chat template (add_generation_prompt) and prints the detokenized
    canvas as its final log line; with --log-jsonl every line is a JSON object, so the result is the
    last info message -- no scraping of a mixed stream. stop_reason is always "length": a diffusion
    decoder fills the canvas it was given rather than emitting an end-of-turn token, so the gate's
    "did it stop on its own" signal does not exist here and the text is what gets judged.
    """
    exe = cli_bin(binary)
    if not exe:
        return {"ok": False, "error": "llama-diffusion-cli not found -- rebuild the runtime "
                                      "(./install.sh) or pass --diffusion-cli", "text": ""}
    if not os.path.isfile(model):
        return {"ok": False, "error": f"no such build: {model}", "text": ""}
    fam = arch_of(model)
    d = dict(_DEFAULTS.get(fam, _DEFAULTS["dream"]))
    if steps is not None:
        d["steps"] = int(steps)
    if algorithm is not None:
        d["algorithm"] = int(algorithm)
    if block_length is not None:
        d["block_length"] = int(block_length); d.pop("eps", None)
    if eps is not None:
        d["eps"] = float(eps); d.pop("block_length", None)
    argv = [exe, "-m", model, "-p", prompt, "-n", str(int(n_predict)), "-c", str(int(ctx)),
            "-ngl", str(int(ngl)), "--temp", str(float(temperature)),
            "--diffusion-steps", str(d["steps"]), "--diffusion-algorithm", str(d["algorithm"]),
            "--log-jsonl"]
    if "block_length" in d:
        argv += ["--diffusion-block-length", str(d["block_length"])]
    if "eps" in d:
        argv += ["--diffusion-eps", str(d["eps"])]
    if top_k is not None:
        argv += ["--top-k", str(int(top_k))]
    if top_p is not None:
        argv += ["--top-p", str(float(top_p))]
    if seed is not None:
        argv += ["--seed", str(int(seed))]
    if system:
        argv += ["-sys", system]
    t0 = time.time()
    try:
        # stdin=DEVNULL: a CLI that inherits a tty may wait on it; the gate has no one to answer it.
        r = subprocess.run(argv, capture_output=True, text=True, errors="replace",
                           timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"llama-diffusion-cli timed out after {timeout:.0f}s",
                "text": "", "argv": argv}
    except OSError as e:
        return {"ok": False, "error": str(e), "text": "", "argv": argv}
    text, errors = "", []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        lvl, msg = str(o.get("level", "")), str(o.get("msg", ""))
        if lvl == "error":
            errors.append(msg.strip())
        elif lvl == "info" and msg.startswith("\n"):        # the canvas is printed as "\n%s\n"
            text = msg.strip("\n")
    if r.returncode != 0 or (not text and errors):
        why = errors[-1] if errors else (r.stderr or "").strip().splitlines()[-1:] or ["exit %d" % r.returncode]
        return {"ok": False, "error": why if isinstance(why, str) else why[0], "text": text,
                "argv": argv, "seconds": round(time.time() - t0, 1)}
    if text == "Error: diffusion generation failed":
        return {"ok": False, "error": text, "text": "", "argv": argv}
    return {"ok": True, "text": text, "tokens": int(n_predict), "stop_reason": "length",
            "seconds": round(time.time() - t0, 1), "argv": argv, "arch": fam}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("-p", "--prompt", default="Explain in two sentences why the sky is blue.")
    ap.add_argument("-n", type=int, default=128, help="canvas length = tokens to generate")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--algorithm", type=int, help="0 origin, 1 confidence, 2 margin, 3 entropy, 4 random")
    ap.add_argument("--block-length", type=int)
    ap.add_argument("--eps", type=float)
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--diffusion-cli", help="explicit llama-diffusion-cli path")
    ap.add_argument("--is-diffusion", action="store_true", help="only report whether the arch is diffusion")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    arch = arch_of(a.gguf)
    if a.is_diffusion:
        print(f"{a.gguf}: architecture={arch or '?'} diffusion={is_diffusion_arch(arch)}")
        sys.exit(0 if is_diffusion_arch(arch) else 1)
    if not is_diffusion_arch(arch):
        print(f"note: {arch or 'unknown arch'} is not a diffusion architecture; the CLI may still refuse it",
              file=sys.stderr)
    r = generate(a.gguf, a.prompt, a.n, steps=a.steps, algorithm=a.algorithm, block_length=a.block_length,
                 eps=a.eps, ngl=a.ngl, ctx=a.ctx, temperature=a.temp, binary=a.diffusion_cli)
    if a.json:
        print(json.dumps(r, indent=2))
    elif r["ok"]:
        print(r["text"])
        print(f"\n[{arch} - {r['tokens']} tokens - {r['seconds']}s]", file=sys.stderr)   # ASCII: cp1252 consoles
    else:
        print(f"error: {r['error']}", file=sys.stderr)
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
