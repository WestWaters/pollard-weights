#!/usr/bin/env python3
"""pollard_envmatch — build a custom-architecture model in a Python env matched to the transformers
version it was SAVED with, so its remote modeling code actually runs.

A model whose config.json says `transformers_version: 4.57.1` and ships its own modeling code (auto_map)
will crash under transformers 5.x — the internal APIs its code calls (e.g. `create_causal_mask`) drift
between majors. Spark-X2.5-4B is the canonical case: `create_causal_mask() got an unexpected keyword
argument 'input_embeds'` under transformers 5.15.

This reads that version and provisions a CACHED PYTHONPATH overlay at `$POLLARD_HOME/tvover/tv-<ver>/`
holding just `transformers==<ver>` (+ its light deps). Prepended to PYTHONPATH, it shadows the base
env's transformers while torch/gptqmodel/exllamav3 keep coming from the base env — so a custom-arch
model builds under the transformers it expects, on the SAME (CUDA) interpreter. No separate venv (a
child venv can't inherit a parent venv's site-packages via --system-site-packages). Stock archs
(no auto_map) are left alone — they're forward-compatible and use the current env.

Used by the `pollard` one-shot automatically (--match-transformers auto|on|off); also runnable:
  pollard-envmatch --model XHToken/Spark-X2.5-4B --lane gptq --ensure   # provision the overlay
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


def ensure_overlay(version, lane=None):
    """Provision a PYTHONPATH OVERLAY that pins transformers to `version`, and return its directory.

    Why an overlay, not a venv: the base env is itself a venv (e.g. the CUDA venv with the right
    cu-tagged torch + gptqmodel). A child venv with --system-site-packages inherits the SYSTEM python's
    packages, NOT the parent venv's — so it comes up empty and has to reinstall torch (CPU-only). Instead
    we `pip install --target <dir> transformers==version` (transformers + its light deps only — NOT torch)
    and prepend <dir> to PYTHONPATH: transformers==version shadows the base's, while torch/gptqmodel/
    exllamav3 keep coming from the base env. Lighter (no venv, no 2.5GB torch) and correct.

    Cached at $POLLARD_HOME/tvover/tv-<version>/ and reused. Returns the dir, or None on failure."""
    home = os.path.abspath(os.environ.get("POLLARD_HOME", os.path.expanduser("~/pollard")))
    tdir = os.path.join(home, "tvover", f"tv-{version}")
    marker = os.path.join(tdir, ".ready")
    if os.path.exists(marker):
        return tdir
    os.makedirs(tdir, exist_ok=True)
    print(f"   [envmatch] provisioning transformers=={version} overlay at {tdir} "
          "(torch/gptqmodel stay inherited from the base env; cached after first run)")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", tdir,
                        "transformers==" + version])
    if r.returncode != 0:
        print("   [envmatch] overlay install failed — the lane will run in the current env (may crash on the arch)")
        return None
    open(marker, "w").write(version)
    return tdir


# backward-compatible alias (older callers used ensure_env)
def ensure_env(version, lane=None, pollard_repo=None):
    return ensure_overlay(version, lane)


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
    print(f"   verdict: MATCHED OVERLAY needed — build the {a.lane} lane under transformers=={target}")
    if a.ensure:
        tdir = ensure_overlay(target, a.lane)
        if tdir:
            print(f"   overlay dir: {tdir}")
            print(f"   use:  PYTHONPATH={tdir}{os.pathsep}$PYTHONPATH  <build command>")
        else:
            print("   overlay setup failed.")
    else:
        print(f"   (run with --ensure to provision it, or `pollard --hf {a.model} --format {a.lane} --run` "
              "does it automatically)")


if __name__ == "__main__":
    main()
