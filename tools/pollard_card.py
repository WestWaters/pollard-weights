#!/usr/bin/env python3
"""pollard-card — THE master template for every PollardWeights Hugging Face repo. One template, every
model, so the repos read as one shelf instead of fourteen one-offs.

Sections, in order. Ones that depend on the model only appear when they apply, so a card never
advertises something the repo does not ship:

  frontmatter · title + shrink hero + size table · what these are
  Model details            params / arch / input support / imatrix / measured
  Which file should I choose?   the rung guide, sized off the real bytes
  Available files          PPL / size / tok-s / Mean KLD / notes
  Prompt format            --prompt-format
  Multimodal               --mmproj
  Fill-in-the-middle       --fim (coder models)
  Download a specific file · How to run
  imatrix (calibration)    --imatrix-file / --calib-file / --calib-note
  ARM / AVX · Errata · Credits & license

Data-driven from the workspace manifest + the base model's config — never hand-written, never drifts.

  pollard-card --model openbmb/MiniCPM5-2B --params 2.5B --out README.md
  pollard-card --model <base> --builds-from <manifest-key> --results results.json --out README.md
  pollard-card ... --repo PollardWeights/<Model>-Pollard --upload PollardWeights/<Model>-Pollard

`--results` (optional) supplies per-file PPL / Mean-KLD / eval string so the files table carries real
numbers; without it those columns show "—". `--builds-from` reads builds recorded under a different
manifest key (e.g. the f16 GGUF path). Sizes/bpw come from the manifest."""
import argparse
import json
import os
import re
import sys

LANE_TAGS = {"gguf": ["gguf", "llama.cpp", "ik_llama.cpp", "trellis", "imatrix"],
             "gptq": ["gptq", "vllm", "compressed-tensors"],
             "mx": ["nvfp4", "compressed-tensors", "vllm", "blackwell"],
             "mlx": ["mlx", "apple-silicon"], "exl3": ["exl3", "exllamav3"]}
# format size multipliers vs f16 (bytes/param relative to 2.0) — for the shrink size table
FMT_BPP = {"Q8_0": 1.06, "Q6_K": 0.82, "Q5_K_M": 0.69, "Q4_K_M": 0.58, "NVFP4": 0.53, "IQ4_XS": 0.55}


def base_config(model_id):
    try:
        if os.path.isdir(model_id):
            return json.load(open(os.path.join(model_id, "config.json")))
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download(model_id, "config.json")))
    except Exception:
        return {}


def _frontmatter_license(path):
    """Read `license:` out of a model card's YAML frontmatter."""
    try:
        with open(path, encoding="utf-8") as fh:
            if fh.readline().strip() != "---":
                return None
            for line in fh:
                if line.strip() == "---":
                    return None
                m = re.match(r"license:\s*(\S+)", line)
                if m:
                    return m.group(1).strip("\"'")
    except OSError:
        pass
    return None


def _hub_license(model_id):
    """Ask the Hub for a model's declared license, over stdlib.

    Deliberately not via `huggingface_hub`: reading one public metadata field should not require the
    `[hf]` extra, and when that import is missing the alternative is guessing.
    """
    import urllib.request
    try:
        req = urllib.request.Request(f"https://huggingface.co/api/models/{model_id}",
                                     headers={"User-Agent": "pollard-card"})
        with urllib.request.urlopen(req, timeout=30) as fh:
            return ((json.load(fh).get("cardData") or {}).get("license"))
    except Exception:
        return None


def base_license(model_id, cfg):
    """Resolve the base model's license.

    `config.json` almost never carries a license -- the Hub keeps it in the card frontmatter -- so
    reading config alone falls through to whatever default sits behind it and labels every card the
    same regardless of the base. Ask the card first, and return None rather than guessing: the
    license line is a legal claim about someone else's weights.
    """
    if cfg.get("license"):
        return cfg["license"]
    if os.path.isdir(model_id):
        return _frontmatter_license(os.path.join(model_id, "README.md"))
    return _hub_license(model_id)


def build_runtimes(builds, repo=None):
    """{build path: 'stock'|'ik_llama'} for the GGUF builds we can actually open.

    A file's name does not decide which runtime loads it; its tensor types do. Stock llama.cpp
    rejects any ggml type above 42, so one protected tensor carrying an ik_llama-only atom makes the
    whole file ik_llama-only even when the filename says `IQ4_XS`. The measured allocation is meant
    to reach for a better atom on a sensitive tensor, so this is normal -- it just has to be said on
    the card instead of assumed from the name.

    With `repo` set, a build whose local file is gone is read from the published copy instead, over
    range requests -- a rung already on the Hub can be described without pulling it back down.

    Builds that can be read neither way are simply absent from the result; the caller states nothing
    about a rung it could not read rather than guessing.
    """
    out = {}
    try:
        import pollard_ggufcompat as gc
    except ImportError:
        return out
    for b in builds:
        path = b.get("path") or ""
        if b.get("lane", "gguf") != "gguf" or not path:
            continue
        src, where = path, "local file"
        if not os.path.isfile(path):
            if not repo:
                continue
            src = f"https://huggingface.co/{repo}/resolve/main/{b.get('name') or os.path.basename(path)}"
            where = "the published copy"
        try:
            verdict, reasons = gc.runtime_of(src)
            out[path] = verdict
            if isinstance(reasons, dict) and reasons.get("supported_by"):
                url = reasons.get("supported_url")
                out.setdefault("_supported_by",
                               f"[{reasons['supported_by']}]({url})" if url else reasons["supported_by"])
                out.setdefault("_supported_plain", reasons["supported_by"])
            arch = reasons.get("architecture") if isinstance(reasons, dict) else None
            if arch:
                out.setdefault("_fork_arch", arch)
        except Exception as e:                                            # noqa: BLE001
            print(f"WARNING: could not read the header of {os.path.basename(path)} via {where} "
                  f"({e}); the card will not state a runtime for it.", file=sys.stderr)
    return out


def load_builds(key, lane=None):
    try:
        import pollard_workspace as ws
        builds = ws.read_manifest(key).get("builds", [])
    except Exception:
        builds = []
    return [b for b in builds if (not lane or b.get("lane") == lane)]


def human_gb(nbytes):
    return f"{nbytes/1e9:.2f} GB" if nbytes else "—"


def parse_params_b(params, cfg):
    if params:
        s = str(params).upper().replace("B", "").strip()
        try:
            return float(s)
        except ValueError:
            pass
    # rough estimate from config if not given
    h = cfg.get("hidden_size", 0); L = cfg.get("num_hidden_layers", 0); v = cfg.get("vocab_size", 0)
    return round((12 * L * h * h + 2 * v * h) / 1e9, 2) if h and L else 0.0


def frontmatter(base_model, lic, lanes, model_type):
    tags = ["pollard-weights", "pollard"]
    for ln in lanes:
        tags += LANE_TAGS.get(ln, [ln])
    if model_type:
        tags.append(model_type)
    tags += ["quantized", "mixed-precision", "measured-allocation", "conversational"]
    seen, uniq = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t); uniq.append(t)
    lines = ["---", f"license: {lic}", f"base_model: {base_model}", "base_model_relation: quantized",
             "quantized_by: PollardWeights", "pipeline_tag: text-generation", "language:", "- en", "tags:"]
    lines += [f"- {t}" for t in uniq]
    lines.append("---")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="base model id or dir (title/frontmatter/config)")
    ap.add_argument("--builds-from", help="workspace manifest key for builds (default: --model)")
    ap.add_argument("--base-model", help="base_model for the frontmatter (default: --model)")
    ap.add_argument("--title", help="card title (default: basename of base model)")
    ap.add_argument("--params", help="param count, e.g. 2.5B (else estimated from config)")
    ap.add_argument("--license", dest="license_", help="license (else read off the base model's card)")
    ap.add_argument("--lane", help="only this lane's builds")
    ap.add_argument("--results", help="JSON: {file_or_tag: {ppl, kld, tps, note}} + optional "
                    "{_eval, _f16_ppl, _hw}. `tps` adds a tok/s column; `_hw` names the machine.")
    ap.add_argument("--repo", help="HF repo id (for ollama/usage lines; default from base name)")
    ap.add_argument("--out", default="README.md")
    ap.add_argument("--upload", help="HF repo id to push the card to (needs HF login / HF_TOKEN)")
    ap.add_argument("--runtime-from-repo", action="store_true",
                    help="for rungs whose local build file is gone, read which runtime they need "
                         "from the copy already published in --repo, over range requests (a few MB "
                         "per file, not the whole download)")
    ap.add_argument("--prompt-format", help="chat template name or a fenced example (Model details + "
                    "Prompt format). Omit and the section is skipped.")
    ap.add_argument("--arch", help="architecture line for Model details (else the base config's model_type)")
    ap.add_argument("--input-support", default="text",
                    help="Model details: what the model takes (text / text+image / text+image+video)")
    ap.add_argument("--imatrix-file", help="imatrix filename shipped in the repo (adds the calibration "
                    "section and sets Model details imatrix=yes)")
    ap.add_argument("--calib-file", help="calibration corpus filename shipped in the repo")
    ap.add_argument("--calib-note", help="one line describing the calibration corpus (domains, tokens)")
    ap.add_argument("--mmproj", help="mmproj filename shipped alongside (adds the Multimodal section)")
    ap.add_argument("--fim", action="store_true", help="coder model: add the fill-in-the-middle section")
    ap.add_argument("--credits", action="append", default=[],
                    help="extra credit bullet (repeatable); base model + llama.cpp + Pollard are automatic")
    ap.add_argument("--base-owner", help="who published the base model, for Credits")
    ap.add_argument("--extra-md", help="markdown file of EXTRA `## ` sections for this model -- the "
                    "per-model prose the template cannot know (a measured comparison, an arch note, a "
                    "quirk). Inserted after Available files, in file order, so regenerating a card "
                    "never silently drops hand-written analysis.")
    a = ap.parse_args()

    cfg = base_config(a.base_model or a.model)
    base_model = a.base_model or a.model
    lic = a.license_ or base_license(base_model, cfg)
    if not lic:
        lic = "other"
        print(f"WARNING: could not resolve the license of {base_model}. Wrote `other`; pass "
              f"--license to set it. A card must not guess -- it is a legal claim about "
              f"someone else's weights.", file=sys.stderr)
    mtype = cfg.get("model_type", "")
    builds = load_builds(a.builds_from or a.model, a.lane)
    lanes = sorted({b.get("lane") for b in builds if b.get("lane")}) or (["gguf"])
    name = a.title or os.path.basename(str(base_model).rstrip("/"))
    repo = a.repo or f"PollardWeights/{name}-Pollard"
    results = {}
    if a.results and os.path.exists(a.results):
        results = json.load(open(a.results))
    eval_str = results.get("_eval", "")
    f16_ppl = results.get("_f16_ppl")

    runtimes = build_runtimes(builds, repo if a.runtime_from_repo else None)
    # A rung can fail stock llama.cpp two ways -- a fork-only atom, or a fork-only architecture -- and
    # the card has to name which. `k2-horizon` is not among upstream's 146 architectures, so all three
    # K2 repos shipped ordinary K-quants that still open nowhere but the IFM fork.
    fork_arch = runtimes.get("_fork_arch")
    fork_where = runtimes.get("_supported_by")
    fork_plain = runtimes.get("_supported_plain") or "a vendor fork"
    runtimes = {k: v for k, v in runtimes.items() if not k.startswith("_")}
    ik_builds = [b for b in builds if runtimes.get(b.get("path")) in ("ik_llama", "fork")]

    pb = parse_params_b(a.params, cfg)
    f16_gb = pb * 2.0
    builds_sorted = sorted(builds, key=lambda b: -(b.get("bytes") or 0))
    smallest = builds_sorted[-1] if builds_sorted else {}
    small_gb = (smallest.get("bytes") or 0) / 1e9
    pct = (1 - small_gb / f16_gb) * 100 if f16_gb else 0
    x = f16_gb / small_gb if small_gb else 0

    primary = a.lane or (lanes[0] if lanes else "gguf")
    lane_word = {"gguf": "", "mlx": " for Apple Silicon", "gptq": " for vLLM/SGLang",
                 "mx": " for Blackwell/vLLM", "exl3": " for exllamav3"}.get(primary, "")
    # ---- frontmatter + hero
    out = [frontmatter(base_model, lic, lanes, mtype), "", f"# {name} — Pollard", ""]
    if f16_gb and small_gb:
        out += [f"> ### Pollard shrank this model{lane_word}: **{f16_gb:.2f} GB (f16) → {small_gb:.2f} GB** — "
                f"**{pct:.0f}% smaller, {x:.1f}× down**.",
                "> The smallest rung here; larger, higher-fidelity rungs are listed below."]
        if primary == "gguf":          # the format size table is GGUF-specific
            out += [">", "> | format | this model's size |", "> |---|---:|", f"> | f16 | {f16_gb:.2f} GB |"]
            for fmt, mult in FMT_BPP.items():
                if fmt in ("Q8_0", "Q6_K", "Q4_K_M"):
                    out.append(f"> | {fmt} | ~{pb*mult:.2f} GB |")
            out.append(f"> | **PollardMix (this repo's {smallest.get('tag','best')})** | **{small_gb:.2f} GB** |")
        out.append("")
    out += [f"Pollard builds of [{base_model}](https://huggingface.co/{base_model}) made with "
            "[Pollard Weights](https://github.com/WestWaters/pollard-weights) — a ladder of "
            "**measured-allocation** quants (bits placed by per-layer sensitivity, not a uniform crush).", ""]
    if "gguf" in lanes:
        # Stated from the files' own tensor types, not from their names: a rung can carry a
        # fork-only atom on one protected tensor and still be called IQ4_XS, and saying "the
        # K-quants run anywhere" would then be wrong for a file people are told to download.
        IK_URL = "https://github.com/ikawrakow/ik_llama.cpp"
        if not runtimes:
            out += ["**GGUF for llama.cpp / ik_llama.cpp, Ollama, LM Studio.** Rungs built on "
                    f"[ik_llama.cpp]({IK_URL})-only atoms (the trellis `IQ*_KT` family among them) "
                    "need that build; the rest run in any recent llama.cpp.", ""]
        elif not ik_builds:
            out += ["**Standard GGUF — every file here runs in stock llama.cpp / ik_llama.cpp, "
                    "Ollama, LM Studio.**", ""]
        elif len(ik_builds) == len(runtimes):
            if fork_arch:
                where = fork_where or "the vendor's llama.cpp fork"
                out += [f"**These files need {where}.** This model's architecture "
                        f"(`{fork_arch}`) is not one upstream llama.cpp knows, so stock llama.cpp -- "
                        "and therefore Ollama and LM Studio -- cannot load them whatever the quant "
                        "types are. The quants themselves are ordinary K-quants.", ""]
            else:
                out += [f"**These files need [ik_llama.cpp]({IK_URL}).** The measured allocation "
                        "places ik_llama-only atoms on this model's sensitive tensors, so stock "
                        "llama.cpp (and therefore Ollama and LM Studio) will not load them.", ""]
        else:
            need = ", ".join(f"`{b.get('tag') or b.get('name')}`" for b in
                             sorted(ik_builds, key=lambda x: -(x.get("bytes") or 0)))
            out += ["**Standard GGUF — runs in stock llama.cpp / ik_llama.cpp, Ollama, LM Studio, "
                    f"except where noted.** {need} need [ik_llama.cpp]({IK_URL}): their allocation "
                    "puts ik_llama-only atoms on the tensors it protects. The rest run anywhere.", ""]

    # ---- Model details: the at-a-glance table every good Pollard card opens with
    arch = a.arch or mtype or "—"
    out += ["## Model details", "", "| | |", "|---|---|"]
    out.append(f"| Parameter count | ~{pb:.1f}B |" if pb else "| Parameter count | — |")
    out.append(f"| Architecture | `{arch}` |")
    out.append(f"| Input support | {a.input_support} |")
    out.append(f"| imatrix | {'**yes** — see [calibration](#imatrix-calibration)' if a.imatrix_file else 'no'} |")
    out.append(f"| Perplexity measured | {'**yes** — table below' if f16_ppl or results else 'pending'} |")
    out.append("")

    # ---- Which file should I choose? -- the rung guide, sized off the real bytes
    if builds:
        out += ["## Which file should I choose?", "",
                "Every rung is the **same weights**, sized to a different RAM budget by the measured "
                "allocation. Pick the largest one that fits your machine with room for context:", ""]
        for b in sorted(builds, key=lambda x: (x.get("bytes") or 0), reverse=True):
            gb = (b.get("bytes") or 0) / 1e9
            r = results.get(b.get("name", ""), results.get(b.get("tag", ""), {}))
            note = str(r.get("note", "")).strip()
            is_rec = "recommended" in note.lower()
            rec = "" if (is_rec and note) else (" **Recommended.**" if is_rec else "")
            if is_rec and note:
                note = f"**{note}**"
            head = f"- **~{gb + 2:.0f} GB RAM / VRAM** → **`{b.get('tag','')}`** ({gb:.2f} GB)."
            # This list is where people actually pick a file, so a rung that stock llama.cpp cannot
            # open has to say so here too -- not only in the table further down.
            rtb = runtimes.get(b.get("path"))
            if rtb == "ik_llama":
                head += " *(ik_llama.cpp)*"
            elif rtb == "fork":
                head += f" *(needs {fork_plain})*"
            out.append(f"{head} {note[:110]}{rec}" if note else f"{head}{rec}")
        out.append("")

    # ---- available files
    out += [f"## Available files{(' (' + eval_str + ')') if eval_str else ''}", ""]
    if f16_ppl:
        out.append(f"_f16 reference PPL {f16_ppl}._\n")
    # tok/s is a column people actually shop on, and the table had no way to carry it -- so every
    # generated card was silently speed-less no matter what had been measured. Shown only when at
    # least one rung reports it, so cards without speed data do not grow an empty column.
    has_tps = any((results.get(b.get("name", ""), results.get(b.get("tag", ""), {})) or {}).get("tps")
                  for b in builds)
    tps_h = " tok/s |" if has_tps else ""
    tps_s = "---:|" if has_tps else ""
    # "runs in" appears only when the ladder is actually mixed -- a uniform ladder says it once
    # in the line above the table instead of repeating itself on every row.
    mixed = bool(ik_builds) and len(ik_builds) != len(runtimes)
    rt_h = " runs in |" if mixed else ""
    rt_s = "---|" if mixed else ""
    out += [f"| file | PPL | size |{tps_h} Mean KLD |{rt_h} notes |",
            f"|---|---:|---:|{tps_s}---:|{rt_s}---|"]
    for b in sorted(builds, key=lambda x: (x.get("bytes") or 0)):
        r = results.get(b.get("name", ""), results.get(b.get("tag", ""), {}))
        tps_c = f" {r.get('tps','—')} |" if has_tps else ""
        rt = runtimes.get(b.get("path"))
        rt_lbl = {"ik_llama": "ik_llama", "fork": fork_plain,
                  "stock": "any llama.cpp"}.get(rt, "—")
        rt_c = (" " + rt_lbl + " |") if mixed else ""
        out.append(f"| `{b.get('name','-')}` | {r.get('ppl','—')} | {human_gb(b.get('bytes'))} |{tps_c} "
                   f"{r.get('kld','—')} |{rt_c} {r.get('note', b.get('tag',''))} |")
    if has_tps:
        hw = results.get("_hw")
        out.append("")
        out.append(f"_tok/s measured on {hw}._" if hw else
                   "_tok/s is hardware-specific; the machine it was measured on is stated in the errata._")
    if not results:
        out.append("")
        out.append("_PPL / Mean-KLD benchmarking pending — sizes and allocation are final._")
    out.append("")

    # ---- usage
    def _recommended(b):
        r = results.get(b.get("name", ""), results.get(b.get("tag", ""), {}))
        return "recommended" in str(r.get("note", "")).lower()
    ex = next((b for b in builds if _recommended(b)),
              min(builds, key=lambda b: (b.get("bytes") or 1e18), default={}))
    exn = ex.get("name", "model.gguf")
    extag = ex.get("tag", "")

    # ---- per-model prose. The template covers what is true of EVERY Pollard repo; this carries what
    # is true of one -- a measured comparison, an architecture note. Kept in a file beside the repo so
    # regenerating a card never costs analysis that was written by hand.
    if a.extra_md:
        if not os.path.exists(a.extra_md):
            sys.exit(f"--extra-md not found: {a.extra_md}")
        extra = open(a.extra_md, encoding="utf-8").read().strip()
        if extra:
            if not extra.lstrip().startswith("#"):
                sys.exit("--extra-md must contain `## ` sections, so the card keeps one heading level")
            out += [extra, ""]

    # ---- Prompt format (only when we actually know it)
    if a.prompt_format:
        out += ["## Prompt format", ""]
        if "\n" in a.prompt_format or "<" in a.prompt_format:
            out += ["```", a.prompt_format.strip(), "```", ""]
        else:
            out += [f"{a.prompt_format}", ""]

    # ---- Multimodal: only for a repo that actually ships the projector
    if a.mmproj:
        out += ["## Multimodal", "",
                f"Vision needs the projector shipped alongside: **`{a.mmproj}`** — download it too and "
                "pass it with `--mmproj`. It is **not quantized**; it is small and the text ladder is "
                "where the size lives.", "",
                "```bash", f"llama-server -m {exn} --mmproj {a.mmproj} -ngl 99", "```", ""]

    # ---- Fill-in-the-middle: coder models only
    if a.fim:
        out += ["## Fill-in-the-middle (code completion)", "",
                "Use the FIM tokens the base model was trained with, not a chat turn:", "",
                "```", "<|fim_prefix|>{before}<|fim_suffix|>{after}<|fim_middle|>", "```", ""]

    # ---- Download a specific file
    out += ["## Download a specific file", "", "```bash",
            'pip install -U "huggingface_hub[cli]"',
            f"hf download {repo} \\", f'  --include "{exn}" --local-dir ./', "```", ""]

    # ---- How to run
    out += ["## How to run", ""]
    if "gguf" in lanes:
        # The featured command has to name a file that command can actually open. If the
        # recommended rung is ik_llama-only, showing it behind a stock `llama-server -hf` sends
        # people to a load error -- so the stock example moves to a rung that loads, and the
        # recommended one is shown with the build it needs.
        ex_ik = runtimes.get(ex.get("path")) in ("ik_llama", "fork")
        stock = [b for b in builds if runtimes.get(b.get("path")) == "stock"]
        stock_pick = max(stock, key=lambda b: (b.get("bytes") or 0), default=None)
        if not ex_ik:
            out += ["These are standard GGUF and run with **llama.cpp**:", "", "```bash",
                    f"llama-server -hf {repo}:{extag}" if extag else f"llama-server -hf {repo}",
                    "```", "", "or from a local file:", "", "```bash",
                    f'llama-cli    -m {exn} -ngl 99 -p "Explain why the sky is blue."',
                    f"llama-server -m {exn} -ngl 99      # OpenAI-compatible API + web UI at :8080",
                    "```", ""]
            out += ["They also work in anything built on llama.cpp — **LM Studio, koboldcpp, Jan, "
                    f"ramalama, Ollama** (`ollama run hf.co/{repo}`).", ""]
        else:
            lead = (f"This model's architecture (`{fork_arch}`) needs "
                    f"{fork_where or 'the vendor llama.cpp fork'}, so every file here runs there"
                    if fork_arch else
                    f"`{extag or exn}` is built on ik_llama-only atoms, so it runs with "
                    "**[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)**")
            out += [lead + ":", "", "```bash",
                    f'llama-cli    -m {exn} -ngl 99 -p "Explain why the sky is blue."',
                    f"llama-server -m {exn} -ngl 99", "```", ""]
            if stock_pick:
                sn, st = stock_pick.get("name", "model.gguf"), stock_pick.get("tag", "")
                out += [f"For stock llama.cpp, Ollama or LM Studio, use `{st or sn}` instead:", "",
                        "```bash", f"llama-server -hf {repo}:{st}" if st else f"llama-server -hf {repo}",
                        f'llama-cli    -m {sn} -ngl 99 -p "Explain why the sky is blue."', "```", ""]
            else:
                out += ["No rung in this repo loads in stock llama.cpp, so Ollama and LM Studio "
                        "cannot run these files.", ""]
    if "mlx" in lanes:
        out += ["```bash", f'mlx_lm.generate --model {repo} --prompt "Hello"', "```", ""]
    if "gptq" in lanes or "mx" in lanes:
        out += ["```bash", f"vllm serve {repo}", "```", ""]

    # ---- imatrix / calibration: what the allocation was measured on
    if a.imatrix_file:
        out += ["## imatrix (calibration)", "",
                f"The importance matrix (`{a.imatrix_file}`, included) was computed on "
                + (a.calib_note or "a mixed-domain corpus so the matrix sees every register the model serves")
                + ".", ""]
        if a.calib_file:
            out += [f"The exact corpus is included as `{a.calib_file}`, so the allocation can be "
                    "reproduced rather than taken on trust.", ""]

    # ---- ARM / AVX: same boilerplate on every GGUF card, so it stops drifting
    if "gguf" in lanes:
        out += ["## ARM / AVX", "",
                "llama.cpp repacks weights into an interleaved layout at load time for faster inference "
                "on ARM and AVX machines — no special file needed, online repacking covers these quants. "
                "The old `Q4_0_4_4/4_8/8_8` variants are not required.", ""]

    # ---- errata + footer
    out += ["## Errata", ""]
    if "gguf" in lanes:
        if ik_builds and fork_arch:
            out.append(f"- `general.architecture` is `{fork_arch}`, which upstream llama.cpp does not "
                       f"implement, so these files load only in "
                       f"{fork_where or 'the vendor fork that adds it'} — the quant types are "
                       "ordinary and irrelevant to that. Checked with `pollard-ggufcheck`, which "
                       "reads the architecture and the tensor types out of the header.")
        elif ik_builds:
            names = ", ".join(f"`{b.get('tag') or b.get('name')}`" for b in
                              sorted(ik_builds, key=lambda x: -(x.get("bytes") or 0)))
            out.append(f"- {names} carry ik_llama-only atoms and need ik_llama.cpp to run; "
                       "stock llama.cpp rejects any ggml type above 42 outright. Checked with "
                       "`pollard-ggufcheck`, from the files' tensor types rather than their names.")
        elif runtimes:
            out.append("- Every file here loads in stock llama.cpp — verified from the tensor types "
                       "with `pollard-ggufcheck`, not assumed from the filenames.")
        else:
            out.append("- Trellis (`IQ*_KT`) quants need ik_llama.cpp to build/run; K-quants run in "
                       "any recent llama.cpp.")
    out += ["- Measured allocation places bits by per-layer sensitivity under a size budget.",
            "- Single machine; replication invited."]
    # ---- Credits & license: every card names the base model, the tooling, and the method
    owner = f" ({a.base_owner})" if a.base_owner else ""
    out += ["", "## Credits & license", "",
            f"- Base model: [`{base_model}`](https://huggingface.co/{base_model}){owner}",
            "- Quantization tooling: [llama.cpp](https://github.com/ggml-org/llama.cpp) (ggml-org)",
            "- Method + tooling: [Pollard Weights](https://github.com/WestWaters/pollard-weights) — "
            "*measure first, no claim before a number.*"]
    out += [f"- License: `{lic}`, inherited from the base model."]
    for c in a.credits:
        out.append("- " + c.lstrip("- ").strip())

    out += ["",
            "*Built with [Pollard Weights](https://github.com/WestWaters/pollard-weights) — "
            "frontier models, small hardware, no compromise.*"]

    card = "\n".join(out) + "\n"
    open(a.out, "w", encoding="utf-8").write(card)
    print(f"wrote model card -> {a.out}  ({len(builds)} file(s), lanes: {','.join(lanes)})")
    if a.upload:
        try:
            from huggingface_hub import HfApi
            HfApi().upload_file(path_or_fileobj=a.out, path_in_repo="README.md",
                                repo_id=a.upload, repo_type="model")
            print(f"uploaded card -> https://huggingface.co/{a.upload}")
        except Exception as e:
            sys.exit(f"upload failed ({e}); card still written to {a.out}.")


if __name__ == "__main__":
    main()
