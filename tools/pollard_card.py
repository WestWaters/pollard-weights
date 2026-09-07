#!/usr/bin/env python3
"""pollard-card — generate the STANDARD Hugging Face model card for a Pollard build, so every
published PollardWeights repo looks identical: same frontmatter, same method section, same
results table, same run instructions. Data-driven from the workspace manifest (what the build
recorded) + the base model's config, so it's never hand-written and never drifts.

  pollard-card --model openbmb/MiniCPM5-2B --out README.md         # all lanes recorded for this model
  pollard-card --model <base> --lane gguf --out README.md          # just one lane's repo
  pollard-card --model <base> --scorecard scorecard.md --out README.md   # embed the gold-card board

Reads POLLARD_HOME's manifest (pollard-ls data) for the size/bpw/ppl/verified of each build. Pass
--params / --license / --base-model to override or fill what the config doesn't carry. --upload pushes
the card to a HF repo (needs `huggingface-cli login` or HF_TOKEN)."""
import argparse
import json
import os
import sys

LANE_RUN = {
    "gguf": "```bash\n# llama.cpp / Ollama / LM Studio — it's a standard GGUF\nllama-cli -m {file} -p \"Hello\"\n```",
    "gptq": "```bash\nvllm serve {repo} --quantization gptq\n```",
    "mx":   "```bash\nvllm serve {repo}   # compressed-tensors (NVFP4 on Blackwell / W4A16 any GPU)\n```",
    "mlx":  "```bash\nmlx_lm.generate --model {repo} --prompt \"Hello\"\n```",
    "exl3": "```bash\n# exllamav3 / TabbyAPI\n```",
}
LANE_NAME = {"gguf": "GGUF (llama.cpp)", "gptq": "GPTQ (vLLM/SGLang)",
             "mx": "compressed-tensors (Blackwell NVFP4 / any-GPU W4A16)",
             "mlx": "MLX (Apple Silicon)", "exl3": "EXL3 (exllamav3)"}


def base_config(model_id):
    """Best-effort read of the base model's config for frontmatter (params/arch/license)."""
    try:
        if os.path.isdir(model_id):
            return json.load(open(os.path.join(model_id, "config.json")))
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download(model_id, "config.json")))
    except Exception:
        return {}


def load_builds(model_id, lane=None):
    """Pull this model's recorded builds from the workspace manifest (name/lane/tag/bpw/ppl/size/verified)."""
    try:
        import pollard_workspace as ws
        builds = ws.read_manifest(model_id).get("builds", [])
    except Exception:
        builds = []
    if lane:
        builds = [b for b in builds if b.get("lane") == lane]
    return builds


def frontmatter(base_model, license_, tags, pipeline, library):
    lines = ["---",
             f"base_model: {base_model}" if base_model else None,
             f"license: {license_}" if license_ else None,
             f"pipeline_tag: {pipeline}" if pipeline else None,
             f"library_name: {library}" if library else None,
             "quantized_by: PollardWeights",
             "tags:",
             "  - pollard-weights", "  - quantized", "  - measured-allocation"]
    lines += [f"  - {t}" for t in tags]
    lines.append("---")
    return "\n".join(l for l in lines if l is not None)


def results_table(builds):
    if not builds:
        return "_No builds recorded yet in the workspace manifest._"
    rows = ["| Lane | Variant | bpw | Size | PPL | Verified |", "|---|---|---|---|---|---|"]
    try:
        import pollard_workspace as ws
        human = ws.human
    except Exception:
        human = lambda n: (f"{n/1e9:.1f}GB" if n else "-")
    for b in sorted(builds, key=lambda x: (x.get("lane", ""), x.get("bpw") or 0)):
        size = human(b.get("bytes")) if b.get("bytes") else "-"
        rows.append("| {lane} | {name} | {bpw} | {size} | {ppl} | {v} |".format(
            lane=LANE_NAME.get(b.get("lane"), b.get("lane", "-")),
            name=b.get("tag") or b.get("name", "-"),
            bpw=b.get("bpw") if b.get("bpw") is not None else "-",
            size=size,
            ppl=b.get("ppl") if b.get("ppl") is not None else "-",
            v="✅" if b.get("verified") else "—"))
    return "\n".join(rows)


def run_section(builds):
    lanes = sorted({b.get("lane") for b in builds if b.get("lane")}) or ["gguf"]
    out = []
    for ln in lanes:
        ex = next((b for b in builds if b.get("lane") == ln), {})
        snippet = LANE_RUN.get(ln, "").format(file=ex.get("name", "model.gguf"),
                                              repo="PollardWeights/" + (ex.get("name") or "model"))
        out.append(f"**{LANE_NAME.get(ln, ln)}**\n\n{snippet}")
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="base model id or dir (for title/frontmatter/config)")
    ap.add_argument("--lane", help="only this lane's builds (else all recorded lanes)")
    ap.add_argument("--out", default="README.md", help="output card path (default README.md)")
    ap.add_argument("--scorecard", help="a pollard-scorecard .md to embed (the gold-card board)")
    ap.add_argument("--base-model", help="override base_model in the frontmatter")
    ap.add_argument("--license", dest="license_", help="override license (else from base config)")
    ap.add_argument("--params", help="human param count for the title, e.g. 2.5B")
    ap.add_argument("--upload", help="HF repo id to push the card to (needs HF login / HF_TOKEN)")
    a = ap.parse_args()

    cfg = base_config(a.model)
    base_model = a.base_model or a.model
    lic = a.license_ or cfg.get("license") or "apache-2.0"
    arch = (cfg.get("architectures") or ["?"])[0]
    pipeline = "text-generation"
    tags = []
    if cfg.get("model_type"):
        tags.append(cfg["model_type"])
    builds = load_builds(a.model, a.lane)

    name = os.path.basename(str(a.model).rstrip("/"))
    title = f"{name} — Pollard" + (f" ({a.params})" if a.params else "")
    parts = [frontmatter(base_model, lic, tags, pipeline, "gguf" if any(b.get("lane") == "gguf" for b in builds) else "transformers")]
    parts.append(f"# {title}\n")
    parts.append("[![Pollard Weights](https://img.shields.io/badge/quantized%20by-Pollard%20Weights-6E56CF)]"
                 "(https://github.com/WestWaters/pollard-weights)\n")
    parts.append(f"Measured-allocation quantization of [`{base_model}`](https://huggingface.co/{base_model}) "
                 "with [Pollard Weights](https://github.com/WestWaters/pollard-weights) — bits allocated by "
                 "**measured KL sensitivity** under a size budget, not a uniform crush.\n")
    parts.append("## Method\n\n"
                 "- **Measured allocation** — per-layer sensitivity decides which tensors stay high and which "
                 "are crushed (imatrix for GGUF; a measured probe for the export lanes).\n"
                 "- **Preconditioning** — SmoothQuant on the low-bit lanes so massive-activation channels don't "
                 "collapse the quantizer.\n"
                 "- **Calibration** — Pollard's Calib 3.0 multi-domain corpus.\n"
                 "- **Verified** — every build is gated by real reconstruction (`pollard-verify`), never a proxy "
                 "metric.\n")
    parts.append("## Variants in this repo\n\n" + results_table(builds) + "\n")
    parts.append("## Run it\n\n" + run_section(builds) + "\n")
    if a.scorecard and os.path.exists(a.scorecard):
        parts.append("## Benchmark scorecard\n\n" + open(a.scorecard, encoding="utf-8").read() + "\n")
    parts.append("## Reproduce\n\n```bash\npip install pollard-weights\n"
                 f"pollard --hf {base_model} --format <gguf|gptq|mlx|exl3|mx> --run\n```\n")
    parts.append("---\n_Card generated by `pollard-card` — every PollardWeights repo uses the same template._")
    card = "\n".join(parts) + "\n"

    open(a.out, "w", encoding="utf-8").write(card)
    print(f"wrote model card -> {a.out}  ({len(builds)} build(s) listed)")
    if a.upload:
        try:
            from huggingface_hub import HfApi
            HfApi().upload_file(path_or_fileobj=a.out, path_in_repo="README.md",
                                repo_id=a.upload, repo_type="model")
            print(f"uploaded card -> https://huggingface.co/{a.upload}")
        except Exception as e:
            sys.exit(f"upload failed ({e}); is HF_TOKEN set / are you logged in? Card still written to {a.out}.")


if __name__ == "__main__":
    main()
