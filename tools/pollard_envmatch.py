#!/usr/bin/env python3
"""pollard_envmatch — build a custom-architecture model in a Python env matched to the transformers
version it was SAVED with, so its remote modeling code actually runs.

A model whose config.json says `transformers_version: 4.57.1` and ships its own modeling code (auto_map)
will crash under transformers 5.x — the internal APIs its code calls (e.g. `create_causal_mask`) drift
between majors. Spark-X2.5-4B is the canonical case: `create_causal_mask() got an unexpected keyword
argument 'input_embeds'` under transformers 5.15.

This reads that version and provisions a CACHED venv at `$POLLARD_HOME/envs/tv-<ver>/` with the matched
transformers + the lane's deps + Pollard (editable), so the export lanes build there instead of crashing.
Stock archs (no auto_map) are left alone — they're forward-compatible and use the current env.

Used by the `pollard` one-shot automatically (--match-transformers auto|on|off); also runnable:
  pollard-envmatch --model XHToken/Spark-X2.5-4B --lane gptq   # print/ensure the matched env
"""
import argparse
import json
import os
import subprocess
import sys

# per-lane pip extras the matched env needs (transformers is pinned separately)
LANE_DEPS = {
    "gptq": ["torch", "safetensors", "gptqmodel", "datasets"],
    "mx": ["torch", "safetensors", "llmcompressor", "datasets"],
    "mlx": ["mlx-lm"],
    "exl3": ["torch", "safetensors", "exllamav3", "datasets"],
    "probe": ["torch", "safetensors", "datasets"],
    "smooth": ["torch", "safetensors"],
}


def _read_config(model):
    try:
        if os.path.isdir(model):
            return json.load(open(os.path.join(model, "config.json")))
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download(model, "config.json")))
    except Exception:
        return {}


def model_transformers_version(model):
    return _read_config(model).get("transformers_version")


def has_custom_code(model):
    return bool(_read_config(model).get("auto_map"))


def installed_transformers_version():
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return None


def needs_matched_env(model):
    """Return the target transformers version to match, or None if the current env is fine.
    Only custom-code models (auto_map) can break on a version drift; stock archs stay on the
    current transformers. Trigger on a MAJOR mismatch (4.x vs 5.x) — the API-breaking kind."""
    if not has_custom_code(model):
        return None
    target = model_transformers_version(model)
    cur = installed_transformers_version()
    if not target or not cur:
        return None
    if target.split(".")[0] != cur.split(".")[0]:      # different major -> API drift risk
        return target
    return None


def _env_python(envdir):
    win = os.path.join(envdir, "Scripts", "python.exe")
    nix = os.path.join(envdir, "bin", "python")
    return win if os.path.exists(win) else nix


def ensure_env(version, lane, pollard_repo=None):
    """Create (or reuse) a cached venv pinned to transformers==version with the lane's deps + Pollard.
    Returns the path to that env's python. Cached at $POLLARD_HOME/envs/tv-<version>/ and reused."""
    home = os.path.abspath(os.environ.get("POLLARD_HOME", os.path.expanduser("~/pollard")))
    envdir = os.path.join(home, "envs", f"tv-{version}")
    py = _env_python(envdir)
    marker = os.path.join(envdir, f".ready-{lane}")
    if os.path.exists(py) and os.path.exists(marker):
        return py
    if not os.path.exists(py):
        print(f"   [envmatch] creating matched env (transformers=={version}) at {envdir}")
        # --system-site-packages so the matched env INHERITS the base env's heavy CUDA stack (the right
        # cu-tagged torch, gptqmodel, exllamav3) and we only overlay the pinned transformers. A plain venv
        # would pull CPU-only torch from PyPI and the GPU lanes would fail.
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", envdir], check=True)
        py = _env_python(envdir)
    repo = pollard_repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # transformers is pinned (shadows the inherited one); the rest are installed only if the base env
    # doesn't already satisfy them (pip skips inherited torch/gptqmodel/etc. via --system-site-packages).
    deps = ["transformers==" + version] + LANE_DEPS.get(lane, ["torch", "safetensors"])
    print(f"   [envmatch] pinning transformers=={version} (inheriting torch/CUDA from the base env); "
          "installing {lane} deps + pollard as needed (cached after first run)".replace("{lane}", lane))
    subprocess.run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=False)
    r = subprocess.run([py, "-m", "pip", "install", "-q", *deps, "-e", repo])
    if r.returncode != 0:
        print("   [envmatch] pip install failed — the lane will run in the current env (may crash on the arch)")
        return None
    open(marker, "w").write(version)
    return py


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF repo id or local dir")
    ap.add_argument("--lane", default="gptq", choices=sorted(LANE_DEPS), help="lane whose deps to install")
    ap.add_argument("--ensure", action="store_true", help="actually create/populate the env (else just report)")
    a = ap.parse_args()

    tv = model_transformers_version(a.model)
    cur = installed_transformers_version()
    custom = has_custom_code(a.model)
    target = needs_matched_env(a.model)
    print(f"== pollard-envmatch :: {a.model}")
    print(f"   model transformers_version: {tv or 'unspecified'} | installed: {cur or 'none'} | custom code: {custom}")
    if not target:
        print("   verdict: current env is fine (stock arch or matching major) — no matched env needed.")
        return
    print(f"   verdict: MATCHED ENV needed — build the {a.lane} lane under transformers=={target}")
    if a.ensure:
        py = ensure_env(target, a.lane)
        print(f"   env python: {py}" if py else "   env setup failed.")
    else:
        print(f"   (run with --ensure to provision it, or `pollard --hf {a.model} --format {a.lane} --run` "
              "does it automatically)")


if __name__ == "__main__":
    main()
