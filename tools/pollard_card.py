#!/usr/bin/env python3
"""pollard-card -- THE master template for every PollardWeights Hugging Face repo. One template, every
model, so the repos read as one shelf instead of fourteen one-offs.

Sections, in order. Ones that depend on the model only appear when they apply, so a card never
advertises something the repo does not ship:

  frontmatter  |  title + shrink hero + size table  |  what these are
  Model details            params / arch / input support / imatrix / measured
  Which file should I choose?   the rung guide, sized off the real bytes
  Available files          PPL / size / tok-s / Mean KLD / notes
  Prompt format            --prompt-format
  Multimodal               --mmproj
  Fill-in-the-middle       --fim (coder models)
  Download a specific file  |  How to run  |  Use it in your harness (pollard_harness.HARNESSES)
  imatrix (calibration)    --imatrix-file / --calib-file / --calib-note
  ARM / AVX  |  Errata  |  Credits & license

Data-driven from the workspace manifest + the base model's config -- never hand-written, never drifts.

  pollard-card --model openbmb/MiniCPM5-2B --params 2.5B --out README.md
  pollard-card --model <base> --builds-from <manifest-key> --results results.json --out README.md
  pollard-card ... --repo PollardWeights/<Model>-Pollard --upload PollardWeights/<Model>-Pollard

`--results` (optional) supplies per-file PPL / Mean-KLD / eval string so the files table carries real
numbers; without it those columns show "--". `--builds-from` reads builds recorded under a different
manifest key (e.g. the f16 GGUF path). Sizes/bpw come from the manifest."""
import argparse, glob
import json
import os
import re
import sys

LANE_TAGS = {"gguf": ["gguf", "llama.cpp", "ik_llama.cpp", "trellis", "imatrix"],
             "gptq": ["gptq", "vllm", "compressed-tensors"],
             # rubin / vera-rubin / fp8 belong to the mx lane only: Rubin runs the same NVFP4 / MXFP4 / FP8
             # formats as Blackwell through vLLM. llama.cpp has no FP4 tensor-core path on sm_100/107, so
             # a GGUF card must never carry a Rubin tag that reads as an optimized build for it.
             "mx": ["nvfp4", "fp8", "mxfp6", "compressed-tensors", "vllm", "blackwell", "rubin", "vera-rubin"],
             "mlx": ["mlx", "apple-silicon"], "exl3": ["exl3", "exllamav3"], "exlr8": ["exlr8", "trellis"]}
# lane -> the hero's "Pollard shrank this model<...>" phrase
LANE_WORD = {"gguf": "", "mlx": " for Apple Silicon", "gptq": " for vLLM/SGLang",
             "mx": " for Blackwell/Rubin (vLLM)", "exl3": " for exllamav3",
             "exlr8": " in EXLR8 (one trellis encode, every rung a manifest)"}
RUBIN_VLLM_IMAGE = "vllm/vllm-openai:cu134-nightly"


def checkpoint_is_mixed(path):
    """Does this GPTQ / MX rung folder hold a MIXED-precision checkpoint -- layers at different schemes? A GPTQ
    `dynamic` map that changes bits, a Quark per-module `layer_quant_config`, more than one compressed-tensors config
    group, or a pollard-mx MXFP6 / mix rung. Those serve on vLLM with `--linear-backend humming` (pollard_mxfp6)."""
    import pollard_mxfp6 as M6
    p = str(path or "")
    if not p or not os.path.isdir(p):
        return False
    if M6.meta_is_mixed(M6.read_meta(p)):
        return True
    if gptq_is_mixed(p):
        return True
    try:
        with open(os.path.join(p, "config.json"), encoding="utf-8") as f:
            q = json.load(f).get("quantization_config") or {}
    except (OSError, ValueError, AttributeError):
        return False
    if q.get("quant_method") == "quark" and q.get("layer_quant_config"):
        return True
    return q.get("quant_method") == "compressed-tensors" and (
        len(q.get("config_groups") or {}) > 1 or q.get("format") == "mixed-precision")


def vllm_line(path, rung_dir=None, flags=""):
    """`vllm serve <path>`, with the Humming flag when the rung at `rung_dir` is a mixed-precision checkpoint."""
    import pollard_mxfp6 as M6
    return M6.humming_serve(path, flags) if checkpoint_is_mixed(rung_dir) else f"vllm serve {path}{flags}"


def vllm_serve_block(repo, lanes, builds=None):
    """The vLLM run lines for the gptq / mx lanes. The mx lane adds Vera Rubin: the nightly CUDA 13.4
    image and the flags that select its paths. Commands only -- no throughput claim, because no number
    on a card may be one we did not measure, and Pollard has not run on Rubin. `builds`: the gptq / mx rungs; any
    mixed-precision one puts `--linear-backend humming` on the line."""
    if "gptq" not in lanes and "mx" not in lanes:
        return []
    import pollard_mxfp6 as M6
    mixed = any(checkpoint_is_mixed(b.get("path")) for b in builds or [] if isinstance(b, dict))
    out = ["```bash", M6.humming_serve(repo) if mixed else f"vllm serve {repo}", "```", ""]
    if mixed:
        out += [M6.HUMMING_NOTE, ""]
    if "mx" in lanes:
        out += ["On Vera Rubin, use vLLM's CUDA 13.4 nightly image (Rubin support is upstream-nightly):", "",
                "```bash",
                f"docker run --gpus all --ipc=host -p 8000:8000 {RUBIN_VLLM_IMAGE} {repo}",
                "# NVFP4 MoE: add --moe-backend flashinfer_cutedsl   |   FP8 KV cache: add --kv-cache-dtype fp8",
                "```", ""]
    return out
# format size multipliers vs f16 (bytes/param relative to 2.0) -- for the shrink size table
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


_HUB_META = {}


def _hub_meta(model_id):
    """Ask the Hub for a model's public metadata, over stdlib, once per model.

    Deliberately not via `huggingface_hub`: reading public metadata should not require the `[hf]`
    extra, and when that import is missing the alternative is guessing.
    """
    if model_id in _HUB_META:
        return _HUB_META[model_id]
    import urllib.request
    try:
        req = urllib.request.Request(f"https://huggingface.co/api/models/{model_id}",
                                     headers={"User-Agent": "pollard-card"})
        with urllib.request.urlopen(req, timeout=30) as fh:
            meta = json.load(fh) or {}
    except Exception:
        meta = {}
    _HUB_META[model_id] = meta
    return meta


def _hub_license(model_id):
    """The base model's declared license, from the Hub's card metadata."""
    return ((_hub_meta(model_id).get("cardData") or {}).get("license"))


# Signals in config.json that a model takes more than text. Keyed by the pipeline tag they imply.
_MODALITY_KEYS = [
    ("video-text-to-text", ("video_config", "video_tower")),
    ("image-text-to-text", ("vision_config", "vision_tower", "image_token_index", "mm_vision_tower")),
    ("audio-text-to-text", ("audio_config", "audio_tower")),
]


def detect_pipeline_tag(model_id, cfg, explicit=None, mmproj=None, input_support=None):
    """What kind of model this is, for the frontmatter `pipeline_tag`.

    This was hardcoded to `text-generation` on every card, which quietly mislabels every multimodal
    build we ship -- Carnice-V3 carries an mmproj and a video preprocessor and was still advertised as
    text-only, so it never surfaced in a Hub search for vision models and the tag contradicted the
    repo's own file list.

    Resolution order, first hit wins:
      1. an explicit --pipeline-tag
      2. what the BASE model declares on the Hub -- its authors' own answer, and authoritative
      3. modality signals in the base config.json (vision/video/audio towers)
      4. what this repo actually ships: an mmproj, or --input-support naming image/video
      5. text-generation
    """
    if explicit:
        return explicit

    # A caller stating the artifact is text-only outranks the base being multimodal: quantizing a VLM
    # without shipping its projector produces a repo that cannot see, and tagging it image-text-to-text
    # would promise vision the files cannot deliver.
    if (input_support or "").strip().lower() == "text" and not mmproj:
        return "text-generation"

    hub = _hub_meta(model_id).get("pipeline_tag") if not os.path.isdir(model_id) else None
    if hub:
        return hub

    for tag, keys in _MODALITY_KEYS:
        if any(k in cfg for k in keys):
            return tag
    archs = " ".join(cfg.get("architectures") or []).lower()
    if "vision" in archs or "vl" in archs.split():
        return "image-text-to-text"

    sup = (input_support or "").lower()
    if "video" in sup:
        return "video-text-to-text"
    if mmproj or "image" in sup:
        return "image-text-to-text"
    return "text-generation"


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
        if not os.path.isfile(path) and b.get("runtime"):
            out[path] = b["runtime"]                      # read before the rung was freed (pollard release)
            if b["runtime"] == "pollard" and b.get("runtime_arch"):
                out.setdefault("_fork_arch", b["runtime_arch"])
                out.setdefault("_arch_verdict", "pollard")
            continue
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
                out.setdefault("_arch_verdict", verdict)
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


PLACEHOLDER_OWNER = "YOUR_HF_ACCOUNT"


def publishing_account(explicit=None, repo=None, upload=None):
    """Whose name goes on this card.

    The template used to hardcode `quantized_by: PollardWeights` and default every download and usage
    line to a PollardWeights repo id. That is fine while we are the only ones running it and wrong the
    moment anyone else clones Pollard: their card credits us for their work and points readers at our
    repos instead of theirs. No credential leaks either way -- huggingface_hub resolves the token from
    the caller's own login -- but the attribution followed the tool instead of the publisher.

    Resolution order, first hit wins: an explicit --quantized-by, the owner segment of the repo id
    being written to or uploaded to, then whoever is logged in locally. None of those is a guess.
    """
    if explicit:
        return explicit
    for rid in (repo, upload):
        if rid and "/" in rid:
            owner = rid.split("/", 1)[0].strip()
            if owner:
                return owner
    try:
        from huggingface_hub import HfApi
        return (HfApi().whoami() or {}).get("name") or None
    except Exception:
        return None


def rung_agreement(n):
    """Verb and pronoun agreement for a list of rungs, so a single fork-only rung is not plural.

    With one rung needing ik_llama the card used to read "`IQ1_KT` need ik_llama.cpp: their
    allocation puts ...", which is the one sentence on a measured-results card that announces nobody
    read it. Both Qwen repos hit it the moment the stock rebuild left exactly one trellis rung.
    """
    return ("needs", "carries", "its") if n == 1 else ("need", "carry", "their")


def human_gb(nbytes):
    return f"{nbytes/1e9:.2f} GB" if nbytes else "--"





def _fmt_num(v, nd=4):
    """Numbers a reader can scan: 56.8803, not 56.880311."""
    return f"{float(v):.{nd}f}".rstrip("0").rstrip(".") if isinstance(v, (int, float)) else "--"


def auto_note(idx, n, ppl, f16_ppl, tag, rec_idx=None):
    """The one-line verdict a reader wants per rung: what it costs against f16, and which to take.

    Said in PERCENT, not raw PPL, and earned rather than assigned by position. "near-lossless"
    meant +0.08 on a 11.28 baseline for one model and would have meant +3.57 on 23.04 for the next
    -- 15% worse, printed as near-lossless, on a published card.

    `rec_idx` is the ONE rung that earned "recommended default" (see pick_recommended). Every
    middle rung used to claim it, so a six-rung ladder shipped four recommended defaults and the
    word stopped meaning anything.
    """
    if not (ppl and f16_ppl):
        base = {0: "smallest"}.get(idx, "largest" if idx == n - 1 else "")
        return ("recommended default" if idx == rec_idx else base)
    pct = (float(ppl) - float(f16_ppl)) / float(f16_ppl) * 100.0
    if pct <= 1.0:
        cost = "near-lossless"
    elif pct <= 5.0:
        cost = f"+{pct:.1f}% vs f16"
    else:
        cost = f"+{pct:.0f}% vs f16"
    if idx == rec_idx:
        return f"recommended default -- {cost}"
    if idx == 0:
        return f"smallest -- {cost}"
    if idx == n - 1:
        return f"highest fidelity here -- {cost}"
    return cost


def pick_recommended(rows):
    """Which single rung is THE recommendation, earned from the measurement.

    `rows` is [(idx, ppl_or_None)] smallest-first. The pick is the best measured PPL; a rung with
    no measurement cannot be recommended, because the recommendation would then be a guess about
    the one thing this card exists to state. If nothing is measured, nothing is recommended.
    """
    scored = [(i, float(p)) for i, p in rows if p]
    if not scored:
        return None
    return min(scored, key=lambda x: x[1])[0]


def detect_card_facts(model_id, builds_dir, cfg, builds=()):
    """Input support, imatrix and parameter count -- read off what was actually built.

    These were flags a person had to remember. Forgetting one puts a wrong fact on a published
    card: gemma-4-12B-it went out reading "Input support: text" while its 175MB projector sat in
    the same folder, and "imatrix: no" for a build an imatrix produced. What shipped is knowable
    from the model and the build directory, so read it."""
    facts = {"input": None, "imatrix": None, "params_b": None, "decision": False, "decision_format": None,
             "tools": None}
    # classify needs something on disk. A repo id reads nothing, so fall back to a built GGUF
    # (which carries the chat template) or the pulled source in the workspace.
    probe = model_id if os.path.isdir(str(model_id)) else None
    if not probe:
        probe = next((b["path"] for b in builds
                      if isinstance(b, dict) and str(b.get("path", "")).endswith(".gguf")
                      and os.path.isfile(b["path"])), None)
    if not probe:
        home = os.environ.get("POLLARD_HOME") or os.path.expanduser("~/pollard")
        cand = os.path.join(home, "downloads", str(model_id).replace("/", "__"))
        probe = cand if os.path.isdir(cand) else model_id
    try:
        from pollard_modelkind import classify
        k = classify(probe)
        facts["decision"] = bool(k.get("decision"))
        facts["decision_format"] = k.get("decision_format")
        facts["tools"] = k.get("tools")
        mods = [m for m in k.get("modalities", []) if m != "speech_out"]
        # A modality only ships if the projector that carries it ships too.
        # The text GGUF carries no modality signals by construction -- the projector does, and it
        # says so itself (clip.has_vision_encoder / clip.has_audio_encoder). Ask the file that
        # actually ships the capability rather than the one that cannot.
        mm = glob.glob(os.path.join(builds_dir, "*mmproj*.gguf")) if builds_dir else []
        shipped = []
        if mm:
            try:
                from pollard_calc import read_gguf_meta
                meta = read_gguf_meta(mm[0])
                if meta.get("clip.has_vision_encoder"):
                    shipped.append("image")
                if meta.get("clip.has_audio_encoder"):
                    shipped.append("audio")
            except Exception:
                shipped = mods
        facts["input"] = ", ".join(["text"] + shipped)
        if mods and not shipped:
            facts["input"] += ("   (the source also does " + "/".join(mods) +
                               " -- build the projector with --mmproj to ship it)")
    except Exception:
        pass
    # Parameter count: the config may be unreachable for a gated/renamed repo, but a built GGUF
    # records what it holds, so a card need never print "--" for a model we actually built.
    try:
        from pollard_calc import read_gguf_meta, gguf_to_config, analyse
        g = next((b["path"] for b in builds
                  if isinstance(b, dict) and str(b.get("path", "")).endswith(".gguf")
                  and os.path.isfile(b["path"])), None)
        if g:
            tot = read_gguf_meta(g).get("_tensor_param_sum")
            if tot:
                facts["params_b"] = float(tot) / 1e9
    except Exception:
        pass
    if builds_dir:
        home = os.environ.get("POLLARD_HOME") or os.path.expanduser("~/pollard")
        # the imatrix is written beside the SOURCE gguf, which is a sibling tree of the builds
        im = []
        for d in (builds_dir, os.path.dirname(builds_dir.rstrip("/\\")),
                  os.path.join(home, "downloads")):
            for pat in ("*.imatrix", "*.dat"):
                im += glob.glob(os.path.join(d, pat))
        facts["imatrix"] = im[0] if im else None
    return facts


def source_params(d):
    """Parameter count of an HF checkpoint folder from its safetensors HEADERS (every tensor's shape; no weights are
    read): the index's metadata.total_parameters when it records one, else the sum over the shards. 0 = unknown.
    The card's config estimate (12 L h^2 + 2 V h) is a dense formula -- it called the 24B LFM2.5 MoE ~2.5B."""
    import struct
    d = str(d or "")
    if not os.path.isdir(d):
        return 0
    idx = os.path.join(d, "model.safetensors.index.json")
    files = []
    if os.path.isfile(idx):
        try:
            with open(idx, encoding="utf-8") as f:
                ix = json.load(f)
            tp = (ix.get("metadata") or {}).get("total_parameters")
            if isinstance(tp, (int, float)) and tp > 0:
                return int(tp)
            files = sorted(set((ix.get("weight_map") or {}).values()))
        except (OSError, ValueError):
            files = []
    if not files:
        files = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    total = 0
    for fn in files:
        try:
            with open(os.path.join(d, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
        except (OSError, ValueError, struct.error):
            return 0
        for k, v in hdr.items():
            if k != "__metadata__":
                p = 1
                for x in v.get("shape") or []:
                    p *= int(x)
                total += p
    return total


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


TOOLCALL_MIN_RATE = 0.75      # pollard-toolcall's own --min-rate default


#: pollard-toolcall backend -> the server the card names for it
TOOL_SERVERS = {"gguf": "`llama-server --jinja`", "mlx": "`mlx_lm.server`",
                "vllm": "`vllm serve --enable-auto-tool-choice --tool-call-parser <family>`",
                "exl3": "exllamav3 (in-process, the model's chat template)",
                "hf": "transformers (in-process, the model's chat template)", "url": "an OpenAI-compatible server"}


def _ref_word(tc):
    """What the retention is a share of: the f16 GGUF for a GGUF rung (or a cross-backend fallback), the bf16 original
    for a directory lane served the same way as its reference."""
    rb = tc.get("ref_backend") or tc.get("backend") or "gguf"
    if tc.get("cross_backend"):
        return "f16 GGUF, cross-backend" if rb == "gguf" else f"bf16 via {rb}, cross-backend"
    return "f16" if rb == "gguf" else "bf16"


def tool_ready_cell(tc):
    """The Tool-ready cell from a pollard-toolcall result. The full gate (served by the lane's own server, 60 probes)
    reports retention vs the reference: 'yes (98% of f16; CI 90-104)', 'yes (97% of bf16)' for an MLX / vLLM rung
    against its bf16 original. Without a reference, or from the --quick check: 'yes (95%)' / 'no (60%)'. A rung this
    machine could not serve: 'not measured (reason)'; a failed measurement: 'error (reason)'.
    '--' when nothing was recorded."""
    if not isinstance(tc, dict):
        return "--"
    if tc.get("status") in ("not-measured", "error"):
        label = "error" if tc["status"] == "error" else "not measured"
        why = str(tc.get("reason") or "").strip().replace("|", "/")
        return f"{label} ({why})" if why else label
    rate = tc.get("rate")
    ok = tc.get("pass")
    if ok is None and rate is not None:
        ok = float(rate) >= TOOLCALL_MIN_RATE
    if ok is None:
        return "--"
    word = "yes" if ok else "no"
    ret, ci = tc.get("retention"), tc.get("retention_ci")
    if ret is not None:
        lo_hi = f"; CI {float(ci[0]):.0f}-{float(ci[1]):.0f}" if isinstance(ci, (list, tuple)) and len(ci) == 2 \
            and None not in ci else ""
        return f"{word} ({float(ret):.0f}% of {_ref_word(tc)}{lo_hi})"
    return word + (f" ({float(rate):.0%})" if rate is not None else "")


#: the name earlier cards and tests used
agent_ready_cell = tool_ready_cell


def tool_ready_note(builds, results):
    """The Tool-ready footnote, naming the server each lane in the table was actually measured through."""
    seen = []
    for b in builds:
        tc = (results.get(b.get("name", ""), results.get(b.get("tag", ""), {})) or {}).get("toolcall")
        if isinstance(tc, dict) and tc.get("status") not in ("not-measured", "error"):
            be = tc.get("backend") or "gguf"
            if be not in seen:
                seen.append(be)
    servers = ", ".join(TOOL_SERVERS.get(be, be) for be in seen)
    measurement = (f"served with {servers}, asked 60 tool probes" if seen else
                   "no completed measurement. The check asks 60 tool probes")
    return ("_Tool-ready = the rung still calls tools ([pollard-toolcall](https://github.com/WestWaters/"
            f"pollard-weights)): {measurement} (single call, choose among eight tools, "
            "no call needed, nested arguments) through OpenAI-style `tools`, and scored on the calls that server "
            "parses -- shown as success relative to the reference (the f16 for GGUF, the bf16 original for the other "
            "lanes) with a 95% paired-bootstrap CI, or as a bare success rate when no reference run was recorded. "
            "\"cross-backend\" = the reference ran on a different server than the rung; \"not measured\" = this "
            "lane's server could not run where the rung was built; \"error\" = request failures prevented a valid "
            "capability measurement._")


def tool_serve_lines(tools, lanes, repo, example_file, builds=None):
    """One serve line per lane, each with the flags that make THAT server parse this model's tool calls."""
    lc = tools.get("llama_cpp") or {}
    out = []
    if "gguf" in lanes:
        out.append(f"- **llama.cpp** (GGUF): `llama-server -m {example_file} --jinja` -- the model's own template "
                   f"renders the tools and llama.cpp parses the calls ({lc.get('format') or 'its template-driven parser'}"
                   "); pass `tools` in `/v1/chat/completions`.")
    if "mlx" in lanes:
        out.append(f"- **MLX**: `mlx_lm.server --model {repo}` -- the tokenizer's chat template renders `tools` and "
                   "mlx-lm's tool parser fills `tool_calls` in `/v1/chat/completions`.")
    vllm_lanes = [l for l in ("gptq", "mx") if l in lanes]
    if not (vllm_lanes or "gguf" in lanes):
        pass                                                     # an MLX / EXL3 repo: vLLM is not how it is served
    elif tools.get("vllm_parser"):
        note = "" if tools.get("vllm_parser_confirmed") else " (parser name not yet confirmed for this model)"
        target = repo if vllm_lanes else "<the original weights>"
        what = " (" + " / ".join({"gptq": "GPTQ", "mx": "NVFP4 / MXFP4 / MXFP6 / FP8"}[l] for l in vllm_lanes) + ")" \
            if vllm_lanes else ""
        mixed = vllm_lanes and any(checkpoint_is_mixed(b.get("path")) for b in builds or [] if isinstance(b, dict))
        hum = " --linear-backend humming" if mixed else ""
        out.append(f"- **vLLM**{what}: `vllm serve {target} {tools['vllm_flags']}{hum}`{note}"
                   + (" (mixed-precision rungs: vLLM >= 0.32.0)" if mixed else ""))
    else:
        out.append("- **vLLM**: no `--tool-call-parser` reads this syntax; parse the calls client-side.")
    if "exlr8" in lanes:
        parser = f" {tools['vllm_flags']}" if tools.get("vllm_parser") else ""
        out.append(f"- **EXLR8**: decode a rung (`pollard-exlr8 --decode <rung folder> --out ./bf16`), then "
                   f"`vllm serve ./bf16{parser}` -- the decode is bit-exact to the stored codes.")
    if "exl3" in lanes:
        out.append(f"- **ExLlamaV3** (EXL3): serve a rung folder with [TabbyAPI](https://github.com/theroyallab/tabbyAPI) "
                   "(`python main.py --model-dir . --model-name <rung folder>`) and pass `tools` in "
                   "`/v1/chat/completions`; TabbyAPI renders them with the model's chat template.")
    return out


def tool_serving_section(tools, lanes, repo, example_file, builds=None):
    """'Tool calling': the call syntax the chat template uses and, per lane, the server flags that parse it -- the
    setting users get wrong most often. From pollard-modelkind's toolcall_info; nothing when the template writes no
    tool calls."""
    if not isinstance(tools, dict) or tools.get("family", "none") == "none":
        return []
    fam = tools["family"]
    out = ["## Tool calling", "",
           f"The chat template writes tool calls in the **`{fam}`** syntax (detected by rendering it with "
           "[pollard-modelkind](https://github.com/WestWaters/pollard-weights)):", ""]
    if tools.get("example"):
        ex = tools["example"].encode("ascii", "backslashreplace").decode()
        out += ["```", ex, "```", ""]
    out.append("Serve with:")
    out.append("")
    out += tool_serve_lines(tools, lanes, repo, example_file, builds)
    if not tools.get("supports_tools"):
        out.append("- The template does not render a tool list, so describe the tools in the system prompt.")
    out.append("")
    return out


def _fmt_kl(v):
    """Option KL spans 1e-5 (Q6_K) to 1e-1 (2-bit): two significant figures keep every rung readable."""
    if v is None:
        return "--"
    v = float(v)
    return "0" if v == 0 else f"{v:.2g}" if v < 0.01 else f"{v:.3f}"


def rung_subfolder(build, lane):
    """Where a directory-lane rung sits in its repo: the part of its name after -Pollard-<LANE>- (lowercased),
    e.g. Qwen3-8B-Pollard-MLX-4bit -> 4bit. pollard release uploads to exactly this folder."""
    name = build.get("name") or os.path.basename(str(build.get("path", "")).rstrip("/\\"))
    marker = f"-Pollard-{lane.upper()}-"
    return (name.split(marker, 1)[1] if marker in name else str(build.get("tag") or name)).lower()


def rungs_section(builds, lane, repo):
    """For a directory lane (MLX / GPTQ / MX / EXL3): which subfolder holds which rung, and how to load one."""
    if lane == "gguf" or not builds:
        return []
    run = {"mlx": "mlx_lm.generate --model ./{d} --prompt \"Hello\"", "gptq": "vllm serve ./{d}",
           "mx": "vllm serve ./{d}", "exl3": "# load ./{d} in exllamav3 / TabbyAPI",
           "exlr8": "pollard-exlr8 --decode ./{d} --out ./{d}-bf16 && vllm serve ./{d}-bf16"}.get(lane, "")
    rows = [f"| `{rung_subfolder(b, lane)}/` | {human_gb(b.get('bytes'))} | {b.get('bpw') or '--'} |"
            for b in sorted(builds, key=lambda x: -(x.get("bytes") or 0))]
    pick_b = min(builds, key=lambda x: x.get("bytes") or 0)
    pick = rung_subfolder(pick_b, lane)
    if lane == "mx":
        return mx_rungs_section(builds, repo, pick)
    line = vllm_line(f"./{pick}", pick_b.get("path")) if lane == "gptq" else run.format(d=pick)
    return ["## Rungs in this repo", "",
            "Each rung is a complete checkpoint in its own folder; the repo root holds the default rung, so "
            "pointing a runtime at the repo id loads that one.", "",
            "| folder | size | avg bits |", "|---|---:|---:|", *rows, "",
            "```bash", f'hf download {repo} --include "{pick}/*" --local-dir .', line, "```", ""]


#: ONE repo per model (owner, 2026-10-16): GGUF rungs + mmproj at the root, every other lane in its own folder
LANE_FOLDER = {"mlx": "MLX", "gptq": "GPTQ", "mx": "MX", "exl3": "EXL3", "exlr8": "EXLR8", "draft": "DRAFT"}
LANE_TITLE = {"mlx": "MLX (Apple Silicon)", "gptq": "GPTQ (vLLM / SGLang)",
              "mx": "MX -- NVFP4 / MXFP4 / MXFP6 / FP8 (vLLM, Blackwell)", "exl3": "EXL3 (ExLlamaV3 / TabbyAPI)",
              "exlr8": "EXLR8 (one trellis encode, every rung a manifest)"}


def lane_dir(repo):
    """The local folder `hf download <repo> --local-dir` writes to in the commands on the card."""
    return "./" + str(repo).split("/")[-1]


def lane_serve(lane, path, tools=None, rung_dir=None):
    """The serve command for one directory-lane rung at a local `path`, with the tool-call flags that server needs.
    `rung_dir`: the rung's local build folder -- a mixed-precision GPTQ / MX checkpoint gets `--linear-backend humming`."""
    tools = tools if isinstance(tools, dict) else {}
    vflags = f" {tools['vllm_flags']}" if tools.get("vllm_parser") and tools.get("vllm_flags") else ""
    if lane == "mlx":
        return [f"mlx_lm.server --model {path}"]
    if lane in ("gptq", "mx"):
        return [vllm_line(path, rung_dir, vflags)]
    if lane == "exl3":
        return [f"# TabbyAPI: python main.py --model-dir {os.path.dirname(path) or '.'} "
                f"--model-name {os.path.basename(path)}"]
    if lane == "exlr8":
        return [f"pollard-exlr8 --decode {path} --out {path}-bf16", f"vllm serve {path}-bf16{vflags}"]
    return []


def provenance_lines(lane, builds):
    """What each rung was built FROM, read off its own sidecars: the streamed engine and text backbone, an NVFP4 source
    tensor kept byte for byte, tensors quantized from an already-quantized source, an EXL3 of the language model only.
    One line per fact, never a guess -- a rung without a sidecar says nothing here."""
    import json as _j
    out = []
    for b in builds:
        p = b.get("path", "")
        if not p or not os.path.isdir(p):
            continue
        name = rung_subfolder(b, lane)
        facts = []
        for side in ("pollard_mx.json", "pollard-gptq.json", "pollard-exl3.json"):
            f = os.path.join(p, side)
            if not os.path.isfile(f):
                continue
            try:
                m = _j.load(open(f, encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if m.get("passthrough"):
                facts.append(f"{len(m['passthrough'])} linears kept byte for byte from the source's own NVFP4 "
                             "(NVIDIA's QAT tensors)")
            deq = m.get("dequantized_source") or {}
            if deq:
                facts.append("quantized from the dequantized source for " + ", ".join(
                    f"{n} {a.replace('ggml:', '')} linears" for a, n in sorted(deq.items())))
            if m.get("engine") == "pollard-stream":
                how = (m.get("backbone") or {}).get("how") or ""
                facts.append("streamed block by block" + (f" ({how})" if how else ""))
            if m.get("input") == "text":
                facts.append(f"language model only: {m.get('why')}")
        if facts:
            out.append(f"- `{name}/`: " + "; ".join(facts))
    return (["How these rungs were made:", "", *out, ""] if out else [])


def gptq_is_mixed(path):
    """True when a GPTQ rung's quantization_config carries a `dynamic` rule that changes a module's bits."""
    try:
        with open(os.path.join(str(path), "config.json"), encoding="utf-8") as f:
            q = json.load(f).get("quantization_config") or {}
    except (OSError, ValueError, AttributeError):
        return False
    base = q.get("bits")
    return any(isinstance(v, dict) and v.get("bits") not in (None, base) for v in (q.get("dynamic") or {}).values())


def lane_folder_section(lane, builds, results, repo, tools=None):
    """One lane of the ONE-repo card: its rungs in <LANE>/<rung>/ with the measured numbers (PPL / KLD / top-1 /
    Tool-ready / gate, whichever were recorded), and the exact download + serve command."""
    if not builds or lane not in LANE_FOLDER or lane == "draft":
        return []
    F = LANE_FOLDER[lane]
    rows = []
    if d3_boards(results):                     # a d3 decision model: its readout board, not text PPL / KLD
        for b in sorted(builds, key=lambda x: -(x.get("bytes") or 0)):
            d = _rung_result(results, b).get("decision") or {}
            g = lambda k, fmt: fmt.format(float(d[k])) if d.get(k) is not None else "--"
            rows.append(f"| `{F}/{rung_subfolder(b, lane)}/` | {human_gb(b.get('bytes'))} | {b.get('bpw') or '--'} | "
                        f"{d.get('runtime', '--')} | {g('agreement', '{:.1%}')} | {_fmt_kl(d.get('option_kl'))} | "
                        f"{g('ece', '{:.4f}')} |{verified_cell(b, results)}")
        pick = rung_subfolder(min(builds, key=lambda x: x.get("bytes") or 0), lane)
        here = lane_dir(repo)
        return [f"## {LANE_TITLE.get(lane, lane.upper())} -- `{F}/`", "",
                f"Each rung is a complete d3 package in its own folder under `{F}/` (the quantized body, the source's "
                "`readout.safetensors` and `decision_config.json` byte-exact, its own `MODEL_MANIFEST.json`). Measured "
                "on its readout against bf16 (see the decision-fidelity table):", "",
                "| folder | size | avg bits | runtime | agrees with bf16 | option KL | ECE | verified |",
                "|---|---:|---:|---|---:|---:|---:|---|", *rows, "",
                "```bash", f'hf download {repo} --include "{F}/{pick}/*" --local-dir {here}', "```", ""] \
            + provenance_lines(lane, builds)
    for b in sorted(builds, key=lambda x: -(x.get("bytes") or 0)):
        r = _rung_result(results, b)
        top1 = r.get("top1")
        rows.append(f"| `{F}/{rung_subfolder(b, lane)}/` | {human_gb(b.get('bytes'))} | {b.get('bpw') or '--'} | "
                    f"{_fmt_num(r.get('ppl'))} | {_fmt_num(r.get('kld'))} | "
                    f"{(f'{float(top1):.2f}%' if top1 else '--')} | {tool_ready_cell(r.get('toolcall'))} |"
                    f"{verified_cell(b, results)}")
    pick_b = min(builds, key=lambda x: x.get("bytes") or 0)
    pick = rung_subfolder(pick_b, lane)
    here = lane_dir(repo)
    path = f"{here}/{F}/{pick}"
    out = [f"## {LANE_TITLE.get(lane, lane.upper())} -- `{F}/`", "",
           f"Each rung is a complete checkpoint in its own folder under `{F}/`. Download only the one you want:", "",
           "| folder | size | avg bits | PPL | Mean KLD | Top-1 agree | Tool-ready | verified |",
           "|---|---:|---:|---:|---:|---:|---|---|", *rows, "",
           "```bash", f'hf download {repo} --include "{F}/{pick}/*" --local-dir {here}',
           *lane_serve(lane, path, tools, pick_b.get("path")), "```", ""]
    if lane in ("gptq", "mx") and any(checkpoint_is_mixed(b.get("path")) for b in builds):
        import pollard_mxfp6 as M6
        out += [M6.HUMMING_NOTE, ""]
    out += provenance_lines(lane, builds)
    if lane == "gptq" and any(gptq_is_mixed(b.get("path", "")) for b in builds):
        out += ["Mixed-precision rungs store each layer at the bits its sensitivity earned (a per-layer `dynamic` map in "
                "`quantization_config`). vLLM and gptqmodel (`GPTQModel.load`) honour that map; transformers' "
                "`from_pretrained` (optimum's GPTQ loader) builds every layer at the base bits and does not load these "
                "correctly -- use vLLM or gptqmodel.", ""]
    if lane == "mx":
        import pollard_mxfp6 as M6
        extra = []
        for b in builds:
            meta = M6.read_meta(b.get("path", "")) or {}
            d = f"{here}/{F}/{rung_subfolder(b, 'mx')}"
            for line in M6.serve_lines(meta, d) if meta else []:
                if line not in extra:
                    extra.append(line)
        if extra:
            out += ["Weight-only MXFP6 / mix rungs load in these runtimes:", "", "```bash", *extra, "```", ""]
    return out


def mx_rungs_section(builds, repo, pick):
    """The MX lane's rung table with a format column and, per rung, the runtimes that load it (pollard_mx.json, written
    by pollard-mx --format / --mix; an llm-compressor rung has none and is a compressed-tensors checkpoint for vLLM)."""
    import pollard_mxfp6 as M6
    rows, serve = [], []
    for b in sorted(builds, key=lambda x: -(x.get("bytes") or 0)):
        d = rung_subfolder(b, "mx")
        meta = M6.read_meta(b.get("path", "")) or {}
        if meta:
            fmt = meta["format"].upper() if meta["format"] != "mix" else "mix " + " / ".join(
                f"{a.upper().replace('_', '-')} {100 * v:.0f}%" for a, v in meta.get("atoms", {}).items())
            lay = meta.get("layout", "")
            loads = ", ".join(r for r, st, _ in (tuple(x) for x in meta.get("runtimes", [])) if st == "yes") or "--"
            for line in M6.serve_lines(meta, f"./{d}"):
                serve.append(line)
        else:
            fmt, lay, loads = str(b.get("tag") or d).upper(), "compressed-tensors", "vLLM"
            serve.append(vllm_line(f"./{d}", b.get("path")))
        rows.append(f"| `{d}/` | {fmt} | {lay} | {human_gb(b.get('bytes'))} | {b.get('bpw') or '--'} | {loads} |")
    notes = []
    for b in builds:
        meta = M6.read_meta(b.get("path", "")) or {}
        for rt, st, how in (tuple(x) for x in meta.get("runtimes", [])):
            line = f"- `{rung_subfolder(b, 'mx')}/` **{rt}** ({st}): {how}"
            if line not in notes:
                notes.append(line)
    out = ["## Rungs in this repo", "",
           "Each rung is a complete checkpoint in its own folder; the repo root holds the default rung, so "
           "pointing a runtime at the repo id loads that one. MXFP6 and mix rungs are weight-only (activations stay "
           "16-bit) OCP MX: 32-weight blocks with an E8M0 scale; mixes put each fused layer at MXFP8 / MXFP6 / FP4 by "
           "its measured sensitivity, with routers, lm_head and embeddings at the checkpoint's precision.", "",
           "| folder | format | layout | size | avg bits | loads in |", "|---|---|---|---:|---:|---|", *rows, ""]
    if notes:
        out += ["Runtimes per rung:", "", *notes, ""]
    out += ["```bash", f'hf download {repo} --include "{pick}/*" --local-dir .']
    out += [s for s in dict.fromkeys(serve) if f"./{pick}" in s] or [f"vllm serve ./{pick}"]
    return out + ["```", ""]


#: the one architecture GR8 (kindlingai/gr8) serves today -- GLM-5.3 on 3/4/6 DGX Spark (GB10) boxes
GR8_ARCHS = ("GlmMoeDsaForCausalLM",)


def exlr8_section(builds, repo, cfg=None):
    """How an EXLR8 rung runs, said plainly. GR8 (the EXLR8 engine) serves GLM-5.3 on GB10 boxes, from its release
    layout: per-unit routed-expert files (experts/exlr8-layer-LLL.kK) next to the dense records. A rung gets the GR8
    command only when it IS that model AND ships those files; every other rung -- today all of pollard-exlr8's,
    which writes dense records -- runs through the reference decode, and the card says so."""
    if not builds:
        return []
    arch = ((cfg or {}).get("architectures") or [None])[0]
    pick = rung_subfolder(min(builds, key=lambda x: x.get("bytes") or 0), "exlr8")
    gr8 = arch in GR8_ARCHS and all(os.path.isdir(os.path.join(str(b.get("path") or ""), "experts")) for b in builds)
    out = ["## Running EXLR8 rungs", "",
           "Each rung folder is an EXLR8 v2 fileset (`dense/exlr8-dense-LLL.safetensors` trellis records, "
           "`dense/exlr8-vocab.safetensors`, `manifests/exlr8-manifest-<rung>.json`). Every rung comes from ONE "
           "encode of each tensor at K = 2-6 bits; the manifest records the width each tensor got -- an exact "
           "knapsack on Pollard's measured sensitivity under the rung's byte budget.", ""]
    if gr8:
        out += ["[GR8](https://github.com/kindlingai/gr8) serves this architecture on DGX Spark (GB10) boxes with an "
                "OpenAI-compatible API on the head node:", "",
                "```bash", f"./gr8 glm53 up --tp 4 --peers <ip1>,<ip2>,<ip3>,<ip4> --model https://huggingface.co/{repo}",
                "```", ""]
    else:
        why = f"not `{arch}`" if arch and arch not in GR8_ARCHS else "and loads routed experts from per-unit expert " \
            "files these rungs do not carry"
        out += [f"[GR8](https://github.com/kindlingai/gr8), the EXLR8 engine, serves one model today (GLM-5.3 on DGX "
                f"Spark boxes), {why}. Run these rungs through the reference decode, which is bit-exact to the "
                "stored codes and is what the numbers on this card measured:", ""]
    out += ["```bash", 'pip install "pollard-weights[exlr8]"',
            f'hf download {repo} --include "{pick}/*" --local-dir .',
            f"pollard-exlr8 --decode ./{pick} --out ./{pick}-bf16",
            f"vllm serve ./{pick}-bf16        # or transformers: AutoModelForCausalLM.from_pretrained('./{pick}-bf16')",
            "```", ""]
    return out


def gate_verdict(b, results):
    """True / False if a pollard-verify --gate ran on this rung (MANIFEST first, then results.json), else None."""
    g = b.get("gate") if isinstance(b, dict) else None
    if isinstance(g, dict) and "pass" in g:
        return bool(g["pass"])
    v = _rung_result(results, b).get("verified")
    if isinstance(v, dict) and "pass" in v:
        return bool(v["pass"])
    return None


def gate_below(b, results):
    """The report-only checks this rung did NOT meet (pollard-verify's default, non-strict gate: measured and labelled
    here, never blocking) -- from the MANIFEST gate record, else results.json. [] when none or no gate ran."""
    g = b.get("gate") if isinstance(b, dict) else None
    if isinstance(g, dict) and "pass" in g:
        if g.get("report_only_failed") is not None:
            return list(g["report_only_failed"])
        return [c.get("check") for c in g.get("checks") or [] if not c.get("ok") and c.get("blocking") is False]
    v = _rung_result(results, b).get("verified")
    return list(v.get("report_only_failed") or []) if isinstance(v, dict) else []


def gate_strict(builds, results):
    """True when any rung's gate ran strict (every threshold blocking) -- the card's note then says so."""
    for b in builds:
        g = b.get("gate") if isinstance(b, dict) else None
        v = _rung_result(results, b).get("verified")
        if (isinstance(g, dict) and g.get("strict")) or (isinstance(v, dict) and v.get("strict")):
            return True
    return False


def verified_cell(b, results):
    v = gate_verdict(b, results)
    if v is None:
        return " -- |"
    if not v:
        return " **no** |"
    below = gate_below(b, results)
    return f" yes (below band: {', '.join(below)}) |" if below else " yes |"


D3_FORMAT = "d3-code-readout-v1"


def d3_boards(results):
    """{rung: board} for the d3 readout boards (pollard-d3 gate) in a results file."""
    return {k: v["decision"] for k, v in (results or {}).items()
            if isinstance(v, dict) and isinstance(v.get("decision"), dict) and v["decision"].get("format") == D3_FORMAT}


def d3_decision_section(builds, results):
    """The d3 board: every rung through ITS runtime, read out with the source's readout, against bf16 through the
    package's own runtime. Columns only for what was measured; the Decision Index (apolinario/decision-index, the
    public part of the Jev Decision Index) only when the opt-in overnight run recorded it."""
    boards = [(b, _rung_result(results, b).get("decision")) for b in sorted(builds, key=lambda x: -(x.get("bytes") or 0))]
    boards = [(b, d) for b, d in boards if isinstance(d, dict) and d.get("format") == D3_FORMAT]
    if not boards:
        return []
    ref = (results.get("bf16") or results.get("_d3_ref") or {}).get("decision") or {}
    ref_idx = ref.get("jev_index") if isinstance(ref, dict) else None
    has_idx = any(d.get("jev_index") is not None for _, d in boards)
    n = boards[0][1].get("n")
    qset = str(boards[0][1].get("set") or "")
    f = lambda d, k, fmt: (fmt.format(float(d[k])) if d.get(k) is not None else "--")
    rows = []
    for b, d in boards:
        idx = "--"
        if d.get("jev_index") is not None:
            idx = f"{float(d['jev_index']):.1f}"
            if ref_idx is not None:
                idx += f" ({float(d['jev_index']) - float(ref_idx):+.1f})"
        bt = d.get("by_type") or {}
        types = " / ".join(f"{t} {v['agreement']:.0%}" for t, v in bt.items()) or "--"
        rows.append(f"| `{b.get('name', '-')}` | {human_gb(b.get('bytes'))} | {d.get('runtime', '--')} | "
                    f"{f(d, 'agreement', '{:.1%}')} | {_fmt_kl(d.get('option_kl'))} | {f(d, 'option_kl_p95', '{:.4f}')} | "
                    f"{f(d, 'ece', '{:.4f}')} | {f(d, 'delta_ece', '{:+.4f}')} | {types} |"
                    + (f" {idx} |" if has_idx else ""))
    asked = f"the same {n} typed decisions" if n else "the same typed decisions"
    if qset.startswith("LocalLLaMA/typed-decisions"):
        rev = qset.split("@", 1)[1].split(" ")[0] if "@" in qset else ""
        asked = (f"{n} decisions of [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) "
                 f"(test split{', revision ' + rev if rev else ''}; choice, yes/no and score questions)")
    out = ["## Decision fidelity (d3 readout)", "",
           f"A d3 model answers through its `{D3_FORMAT}` readout: the body reads the whole prompt with non-causal "
           "full attention, the last token's hidden state goes through the FP32 `readout.safetensors` (255 answer "
           "codes), and the answer is a probability per option. So each rung is measured where the model's product "
           f"is: on {asked}, run through the rung's own runtime and read out with the source's readout (shipped "
           "byte-exact in every rung), against the **bf16** checkpoint through the package's own runtime "
           "([pollard-d3](https://github.com/WestWaters/pollard-weights)).", "",
           "| file | size | runtime | agrees with bf16 | option KL | KL p95 | ECE | ECE vs bf16 | agreement by type |"
           + (" Decision Index (vs bf16) |" if has_idx else ""),
           "|---|---:|---|---:|---:|---:|---:|---:|---|" + ("---:|" if has_idx else ""), *rows, "",
           "_Agrees = the rung picks the same option as bf16. Option KL = KL(bf16 || rung) of the answer "
           "distribution, averaged over decisions (p95 = the 95th percentile). ECE = expected calibration error "
           "against the gold labels (10 bins, top label); `ECE vs bf16` = the change from bf16's own._"]
    if has_idx:
        out.append("_Decision Index = the public index of the Decision Index 0.3 kit "
                   "([apolinario/decision-index](https://github.com/apolinario/decision-index), MIT) run on the rung "
                   "through `/v1/systemone`" + (f"; bf16 scored {float(ref_idx):.1f} on the same run" if ref_idx is not None
                                                else "") + "._")
    return out + [""]


def decision_section(builds, results):
    """The board a decision model is judged on: each rung against the f16, same typed questions, same
    readout (pollard-decision). Agreement = same answer as the f16; option KL = how far the option
    probabilities moved; drift = mean absolute change per option."""
    if d3_boards(results):
        return d3_decision_section(builds, results)
    rows, n = [], None
    boards = [(b, (results.get(b.get("name", ""), results.get(b.get("tag", ""), {})) or {}).get("decision"))
              for b in sorted(builds, key=lambda x: -(x.get("bytes") or 0))]
    # calibration columns only when the boards carry them (pollard-decision since 2026-10-11): older
    # results files have none, and an all-"--" column would read as "measured, nothing there"
    has_ece = any(isinstance(d, dict) and d.get("ece") is not None for _, d in boards)
    has_flips = any(isinstance(d, dict) and d.get("flips") is not None for _, d in boards)
    for b, d in boards:
        if not isinstance(d, dict):
            continue
        n = n or d.get("n")
        f = lambda k, fmt: (fmt.format(float(d[k])) if d.get(k) is not None else "--")
        extra = (f" {f('ece', '{:.4f}')} |" if has_ece else "") + \
                (f" {d['flips']} |" if has_flips and d.get("flips") is not None else (" -- |" if has_flips else ""))
        rows.append(f"| `{b.get('name','-')}` | {human_gb(b.get('bytes'))} | {f('agreement', '{:.0%}')} | "
                    f"{_fmt_kl(d.get('option_kl'))} | {f('mean_abs_drift', '{:.4f}')} | {f('accuracy', '{:.0%}')} |{extra}")
    asked = f"the same {n} typed questions" if n else "the same typed questions"
    qset = next((str((r or {}).get("decision", {}).get("set") or "") for r in results.values()
                 if isinstance(r, dict) and isinstance(r.get("decision"), dict)), "")
    if qset.startswith("LocalLLaMA/typed-decisions"):
        rev = qset.split("@", 1)[1] if "@" in qset else ""
        asked = (f"the {n} decisions of the [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)"
                 f" benchmark (test split{', revision ' + rev if rev else ''}: 400 states x 5 typed questions)")
    ref = decision_ref(results)
    return ["## Decision fidelity", "",
            "This is a **decision model**: it answers typed questions (choice / score / yes-no) with a "
            f"probability per option, so it is measured on its decisions, not on text. Each rung and the {ref} "
            f"answered {asked} through llama-server's `/v1/systemone`, read the same way "
            "([pollard-decision](https://github.com/WestWaters/pollard-weights)).", "",
            f"| file | size | agrees with {ref} | option KL vs {ref} | mean prob. drift | accuracy |"
            + (" ECE |" if has_ece else "") + (f" flips vs {ref} |" if has_flips else ""),
            "|---|---:|---:|---:|---:|---:|" + ("---:|" if has_ece else "") + ("---:|" if has_flips else ""),
            *rows, "",
            f"_Agrees = the rung picks the same option as the {ref}. Option KL = how far its option "
            f"probabilities moved from the {ref}'s (0 = identical); it is the number that separates the rungs._"
            + (" _ECE = expected calibration error against the gold labels (10 bins, top label; lower = its "
               "confidence means what it says)._" if has_ece else "")
            + (f" _Flips = decisions whose answer differs from the {ref}'s._" if has_flips else ""),
            ""] + decision_only_lines(results, ref)


def decision_only_lines(results, ref="f16"):
    """A decision-ONLY model (no per-token logits: clef / clef-omni): the card says KLD / PPL are not applicable, as
    pollard-verify recorded them, and that the board above is what each rung was gated on."""
    why = next((str(r.get("kld_reason") or "") for r in (results or {}).values()
                if isinstance(r, dict) and r.get("kld_status") == "not-applicable"), None)
    if why is None:
        return []
    return [f"_Mean KLD / PPL: {why or 'not applicable: decision-only model'}. Each rung's publish gate is this "
            f"board: agreement and option KL vs the {ref}._", ""]


RETENTION_CATS = ["Knowledge & Reasoning", "Math", "Coding", "Instruction Following", "Vision", "Other"]


def _pct(v, nd=1):
    return f"{float(v):.{nd}f}%" if v is not None else "--"


def _rung_result(results, b):
    return (results.get(b.get("name", ""), results.get(b.get("tag", ""), {})) or {})


def moe_allocation_section(builds):
    """The measured MoE rung: how much was pruned and why -- every candidate pollard-moefit auto screened (prune
    fraction x bit mix at the rung's bytes) with what it predicted and, for the ones built, what they MEASURED."""
    out = []
    for b in builds:
        m = b.get("moefit")
        if not m:
            continue
        pct = int(round(100 * float(m.get("chosen_prune") or 0)))
        out += [f"## MoE allocation: `{b.get('name', '-')}`", "",
                (f"**{pct}% of the routed experts pruned** (keep {m.get('keep')}/{m.get('experts')} per layer, "
                 f"scored by {'REAP' if m.get('scores') == 'reap' else 'router traffic'})" if pct else
                 f"**No experts pruned** (all {m.get('experts')} kept)")
                + f", the remaining experts mixed per tensor at {m.get('budget_gb')} GB. Chosen by measurement, "
                "not a fixed rule: every prune fraction was screened with its best bit mix at the same bytes, and "
                "the best candidates were built and measured against the full model on held-out text (mean KLD, "
                "top-1, PPL) with the tool-call gate.", "", f"Why: {m.get('why')}.", "", m.get("table") or "", "",
                "_Screened KL = the pruned model's own measured KL + the allocator's predicted KL for its bit mix "
                "(1-bit experts priced by their measured cost). Measured mean KLD is vs the full model._", ""]
    return out


def retention_section(builds, results):
    """"Retention vs f16": what each rung keeps of the f16 on real work, from pollard-taskeval (per-category
    task retention), its mt suite (chrF++ per language), pollard-toolcall (tool-call validity vs the f16's)
    and pollard-decision (agreement with the f16's decisions). Only columns something measured appear,
    and a rung nothing measured is left out rather than shown as a row of dashes."""
    ordered = sorted(builds, key=lambda x: -(x.get("bytes") or 0))
    rows = [(b, _rung_result(results, b)) for b in ordered]
    rows = [(b, r) for b, r in rows if any(isinstance(r.get(k), dict) for k in ("retention", "mt", "toolcall", "decision"))]
    if not rows:
        return []
    cats = [c for c in RETENTION_CATS if any(c in ((r.get("retention") or {}).get("categories") or {}) for _, r in rows)]
    has_mt = any(isinstance(r.get("mt"), dict) and any("retention" in v for v in r["mt"].values() if isinstance(v, dict))
                 for _, r in rows)
    has_tc = any(isinstance(r.get("toolcall"), dict) for _, r in rows)
    has_dec = any(isinstance(r.get("decision"), dict) and r["decision"].get("agreement") is not None for _, r in rows)
    has_all = any((r.get("retention") or {}).get("overall") is not None for _, r in rows)
    head = ["file"] + cats + (["Translation (chrF++)"] if has_mt else []) + (["Tool calls"] if has_tc else []) \
        + (["Decisions (agree)"] if has_dec else []) + (["**Overall**"] if has_all else [])
    out = ["## Retention vs f16", "",
           "What each rung keeps of the f16 on real work -- quantized score / f16 score on the same tasks, the same "
           "settings, the same harness (100% = no loss). Measured against our own f16, so the rungs compare "
           "directly; it is not comparable to a retention figure published on a different suite.", "",
           "| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for b, r in rows:
        ret = r.get("retention") or {}
        cells = [f"`{b.get('name', '-')}`"] + [_pct((ret.get("categories") or {}).get(c)) for c in cats]
        if has_mt:
            vals = [v["retention"] for v in (r.get("mt") or {}).values() if isinstance(v, dict) and "retention" in v]
            cells.append(_pct(sum(vals) / len(vals)) if vals else "--")
        if has_tc:
            tc = r.get("toolcall")
            if isinstance(tc, dict) and tc.get("status") == "not-measured":
                cells.append("not measured")
            elif isinstance(tc, dict) and tc.get("status") == "error":
                cells.append("error")
            elif not isinstance(tc, dict) or tc.get("rate") is None:
                cells.append("--")
            elif tc.get("retention") is not None:
                cells.append(_pct(float(tc["retention"])))
            elif tc.get("ref_rate"):
                cells.append(_pct(100 * float(tc["rate"]) / float(tc["ref_rate"])))
            else:
                cells.append(f"{float(tc['rate']):.0%} valid")
        if has_dec:
            d = r.get("decision") or {}
            cells.append(_pct(100 * float(d["agreement"]), 0) if d.get("agreement") is not None else "--")
        if has_all:
            cells.append(f"**{_pct(ret.get('overall'))}**" if ret.get("overall") is not None else "--")
        out.append("| " + " | ".join(cells) + " |")
    notes = []
    suite = results.get("_retention_suite") or {}
    if cats and suite:
        tasks = ", ".join(f"`{t}`" for t in suite.get("tasks") or [])
        notes.append(f"_Tasks: {tasks or suite.get('suite', '?')} ({suite.get('harness', 'lm_eval')}, "
                     f"limit {suite.get('limit', 'full')}), scored by pollard-taskeval._")
    if has_tc:
        notes.append("_Tool calls = the rung's tool-call success as a share of the f16's "
                     "([pollard-toolcall](https://github.com/WestWaters/pollard-weights)); \"valid\" when no f16 run was "
                     "recorded. Decisions = share of typed decisions identical to the f16's._")
    for n in notes:
        out += ["", n]
    out.append("")
    if has_mt:
        out += mt_section(rows, results)
    return out


def mt_section(rows, results):
    """Per-language translation table: one row per pair, chrF++ and retention per rung."""
    pairs = []
    for _, r in rows:
        for p in (r.get("mt") or {}):
            if p not in pairs:
                pairs.append(p)
    mrows = [(b, r) for b, r in rows if isinstance(r.get("mt"), dict)]
    ref_of = {}
    for _, r in mrows:
        for p, v in r["mt"].items():
            if isinstance(v, dict) and v.get("ref_chrf") is not None:
                ref_of.setdefault(p, v["ref_chrf"])
    meta = results.get("_mt_suite") or {}
    head = ["pair", "f16 chrF++"] + [f"`{b.get('name', '-')}`" for b, _ in mrows]
    out = ["### Translation, per language", "",
           f"chrF++ on {('[' + meta['set'] + '](https://huggingface.co/datasets/' + meta['set'] + ')') if meta.get('set') else 'the mt suite'}"
           + (f" ({meta.get('split')} split, {meta.get('limit')} sentences per pair)" if meta.get("limit") else "")
           + "; each cell is the rung's chrF++ and, in brackets, its retention of the f16's.", "",
           "| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for p in pairs:
        cells = [p, f"{ref_of[p]:.1f}" if p in ref_of else "--"]
        for _, r in mrows:
            v = r["mt"].get(p)
            if not isinstance(v, dict):
                cells.append("--")
                continue
            c = f"{float(v['chrf']):.1f}"
            if v.get("retention") is not None:
                c += f" ({float(v['retention']):.0f}%)"
            if v.get("comet") is not None:
                c += f" / COMET {float(v['comet']):.3f}"
            cells.append(c)
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return out


def decision_ref(results):
    """What the decision board was measured against: the f16 unless the results say otherwise. A model too big
    to serve at f16 on the build machine (Clef 27B: 54 GB) is gated against Q8_0, and the card must say so."""
    return str((results or {}).get("_decision_ref") or "f16")


def pollard_runtime_lines(need, build):
    """'How to run' for a `pollard` architecture: the plain need line, then how to build and serve with Pollard's
    runtime (pollard-runtime --update, or the pinned upstream base + the patch)."""
    return [f"**{need}.** Build it, then serve any file here:", "", "```bash", *build, "```", ""]


def decision_usage(exn, newer_arch=None):
    lead = (f"This model's architecture (`{newer_arch}`) needs a llama.cpp new enough to carry it. "
            if newer_arch else "")
    return [lead + "A decision model is served on **`/v1/systemone`** (no chat or completions):", "",
            "```bash", f"llama-server -m {exn} -ngl 99 --port 8080",
            "curl http://127.0.0.1:8080/v1/systemone -H \"Content-Type: application/json\" -d '{",
            '  "state": "Customer message: I was charged twice for my order last week and nobody has replied.",',
            '  "questions": {',
            '    "route":   {"type": "choice", "instructions": "Which team should handle this?",',
            '                "criteria": {"billing": null, "shipping": null, "technical": null}},',
            '    "angry":   {"type": "noul",   "instructions": "Is the customer angry?"},',
            '    "urgency": {"type": "score",  "instructions": "How urgent is this?",',
            '                "criteria": ["can wait", "this week", "today", "right now"]}',
            "  }", "}'", "```", "",
            "Each answer carries the probability of every option. Some decision models evaluate the whole "
            "prompt in one batch, so a long `state` may need a larger `--ubatch-size`.", ""]


def frontmatter(base_model, lic, lanes, model_type, quantized_by=None,
                pipeline_tag="text-generation", decision=False, ik=True):
    tags = ["pollard-weights", "pollard"]
    for ln in lanes:
        # ik_llama / trellis tags only when a rung actually needs ik_llama: a stock-only ladder
        # (clef-flash) was tagged with a runtime and a quant family it does not use.
        tags += [t for t in LANE_TAGS.get(ln, [ln]) if ik or t not in ("ik_llama.cpp", "trellis")]
    if model_type:
        tags.append(model_type)
    tags += ["quantized", "mixed-precision", "measured-allocation"]
    # A decision model answers typed questions with probabilities; it does not chat.
    tags += ["decision-model", "systemone", "structured-output"] if decision else ["conversational"]
    seen, uniq = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t); uniq.append(t)
    lines = ["---", f"license: {lic}", f"base_model: {base_model}", "base_model_relation: quantized"]
    # Say which library the files are for. Left unset, the Hub guesses from the tags: our "trellis" tag (the
    # ik_llama quant family) made it file every ik repo under Microsoft's TRELLIS 3D library, and a repo
    # without that tag (clef-flash-Pollard) showed "Downloads are not tracked for this model".
    library = "gguf" if "gguf" in lanes else "mlx" if "mlx" in lanes else None
    if library:
        lines.append(f"library_name: {library}")
    # Better to say nothing than to credit the wrong account: an unattributed card is incomplete,
    # a misattributed one is false.
    if quantized_by:
        lines.append(f"quantized_by: {quantized_by}")
    lines += [f"pipeline_tag: {pipeline_tag}", "language:", "- en", "tags:"]
    lines += [f"- {t}" for t in uniq]
    lines.append("---")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="base model id or dir (title/frontmatter/config)")
    ap.add_argument("--builds-from", help="workspace manifest key for builds (default: --model)")
    ap.add_argument("--base-model", help="base_model for the frontmatter (default: --model)")
    ap.add_argument("--pipeline-tag", dest="pipeline_tag",
                    help="HF pipeline tag (default: detected from the base model's Hub metadata, "
                         "its config's modality signals, then what this repo ships)")
    ap.add_argument("--quantized-by", dest="quantized_by",
                    help="HF account to credit (default: the owner of --repo/--upload, else your "
                         "Hugging Face login)")
    ap.add_argument("--title", help="card title (default: basename of base model)")
    ap.add_argument("--params", help="param count, e.g. 2.5B (else estimated from config)")
    ap.add_argument("--license", dest="license_", help="license (else read off the base model's card)")
    ap.add_argument("--lane", help="only this lane's builds")
    ap.add_argument("--results", help="JSON: {file_or_tag: {ppl, kld, tps, note}} + optional "
                    "{_eval, _f16_ppl, _hw}. `tps` adds a tok/s column; `_hw` names the machine.")
    ap.add_argument("--thinklean", help="pollard-thinklean --json output; a MEASURED preset adds a 'Think-lean' section")
    ap.add_argument("--repo", help="HF repo id (for ollama/usage lines; default from base name)")
    ap.add_argument("--out", default="README.md")
    ap.add_argument("--upload", help="HF repo id to push the card to (needs HF login / HF_TOKEN)")
    ap.add_argument("--verified-only", action="store_true",
                    help="list only rungs that passed pollard-verify --gate (what pollard release publishes)")
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
    ap.add_argument("--draft-results", help="results-draft.json from pollard-draft measure: adds a Speculative "
                    "decoding section (matched drafter, measured tokens per target pass, this lane's serve line)")
    ap.add_argument("--draft-repo", help="the drafter repo (<M>-Pollard-DRAFT) the section points at")
    ap.add_argument("--also", nargs=3, action="append", default=[], metavar=("LANE", "BUILDS_FROM", "RESULTS"),
                    help="ONE repo per model: add this lane's rungs (in its <LANE>/ folder) to the card, read from "
                         "the MANIFEST key BUILDS_FROM and its results file (repeatable). The drafter (draft lane) "
                         "is then described in DRAFT/ of the same repo.")
    ap.add_argument("--extra-md", help="markdown file of EXTRA `## ` sections for this model -- the "
                    "per-model prose the template cannot know (a measured comparison, an arch note, a "
                    "quirk). Inserted after Available files, in file order, so regenerating a card "
                    "never silently drops hand-written analysis.")
    a = ap.parse_args()

    cfg = base_config(a.base_model or a.model)
    base_model = a.base_model or a.model
    if os.path.exists(str(base_model)):
        print(f"WARNING: base_model is a local path ({base_model}); the Hub wants the repo id (org/name). Pass "
              "--base-model <org/name> (pollard release: `base_model:` in the plan).", file=sys.stderr)
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
    account = publishing_account(a.quantized_by, a.repo, a.upload)
    # A placeholder reads as "fill this in"; a real-but-wrong repo id reads as an instruction, and
    # sends people to download someone else's weights.
    repo = a.repo or f"{account or PLACEHOLDER_OWNER}/{name}-Pollard"
    results = {}
    if a.results and os.path.exists(a.results):
        results = json.load(open(a.results))
    if a.verified_only:
        dropped = [b.get("name") for b in builds if gate_verdict(b, results) is not True]
        builds = [b for b in builds if gate_verdict(b, results) is True]
        if dropped:
            print(f"   --verified-only: left off {len(dropped)} rung(s) that did not pass the gate: "
                  + ", ".join(str(d) for d in dropped), file=sys.stderr)
        lanes = sorted({b.get("lane") for b in builds if b.get("lane")}) or lanes
    eval_str = results.get("_eval", "")
    f16_ppl = results.get("_f16_ppl")
    also = []                                        # [(lane, builds, results)] -- the ONE-repo card's other lanes
    for lane_, key_, res_ in a.also:
        if lane_ == "draft":
            continue
        rr = json.load(open(res_, encoding="utf-8")) if res_ and os.path.exists(res_) else {}
        bb = load_builds(key_, lane_)
        if a.verified_only:
            bb = [b for b in bb if gate_verdict(b, rr) is True]
        if bb:
            also.append((lane_, bb, rr))
    one_repo = bool(a.also)

    runtimes = build_runtimes(builds, repo if a.runtime_from_repo else None)
    # A rung can fail stock llama.cpp two ways -- a fork-only atom, or a fork-only architecture -- and
    # the card has to name which. `k2-horizon` is not among upstream's 146 architectures, so all three
    # K2 repos shipped ordinary K-quants that still open nowhere but the IFM fork.
    fork_arch = runtimes.get("_fork_arch")
    # 'newer' and 'fork' are different promises to a reader: one is "update llama.cpp", the other is
    # "you need someone else's build". Conflating them once told people Spark-X2.5 needed a fork when
    # upstream had merged the architecture days earlier.
    arch_verdict = runtimes.get("_arch_verdict")
    fork_where = runtimes.get("_supported_by")
    fork_plain = runtimes.get("_supported_plain") or "a vendor fork"
    if arch_verdict == "newer":
        fork_where = "a current llama.cpp"
        fork_plain = "a current llama.cpp"
    # 'pollard' = an architecture only a runtime patch POLLARD SHIPS implements (runtime-patches/, upstream PR
    # pending): never "stock", never someone else's fork -- the card names the patch and how to build with it.
    pol_need = pol_build = None
    if arch_verdict == "pollard" and fork_arch:
        import pollard_ggufcompat as _gc
        _pi = _gc.pollard_archs().get(fork_arch) or {}
        pol_need = _gc.pollard_need_line(fork_arch, _pi)
        pol_build = _gc.pollard_build_lines(fork_arch, _pi)
        fork_where = f"llama.cpp with Pollard's `{fork_arch}` patch"
        fork_plain = "llama.cpp + Pollard patch"
    runtimes = {k: v for k, v in runtimes.items() if not k.startswith("_")}
    # Anything that is not plainly 'stock' has to be said on the card. 'newer' and 'unknown' belong
    # here too: a reader whose llama.cpp cannot open the file does not care why it cannot.
    NONSTOCK = ("ik_llama", "fork", "newer", "pollard", "unknown", "prism")
    ik_builds = [b for b in builds if runtimes.get(b.get("path")) in NONSTOCK]
    # online-RHT rungs (prism.hadamard manifest): stock llama.cpp, but only from b11556 (#29600) --
    # a version requirement, not a fork -- and an older build loads them silently WRONG.
    prism_builds = [b for b in builds if runtimes.get(b.get("path")) == "prism"]
    try:
        from pollard_ggufcompat import PRISM_MIN_BUILD
    except Exception:                                                    # noqa: BLE001
        PRISM_MIN_BUILD = "b11556"
    PRISM_NEED = f"llama.cpp >= {PRISM_MIN_BUILD}"

    # Detect first: the parameter count drives f16_gb, which drives the shrink block. Detecting
    # afterwards left a card with no "Pollard shrank this model" hero at all whenever the config
    # was unreachable -- the single most useful line on the page, silently absent.
    bdir = next((os.path.dirname(b["path"]) for b in builds
                 if isinstance(b, dict) and b.get("path") and os.path.dirname(b["path"])), None)
    facts = detect_card_facts(a.model, bdir, cfg, builds)
    # A DECISION model (a decision head like clef, or a Jev / OpenJev-style model) is measured on its
    # decisions, not its text: perplexity does not apply, and a clef server cannot chat at all. Shown
    # only when detected (pollard-modelkind) or when the results carry a pollard-decision board.
    dec_rows = {k: v["decision"] for k, v in results.items()
                if isinstance(v, dict) and isinstance(v.get("decision"), dict)}
    decision = bool(facts.get("decision") or dec_rows)
    # what was MEASURED beats the formula: --params, the source's safetensors headers, a built GGUF's tensor sum, and
    # only then the dense config estimate (12 L h^2 + 2 V h), which reads a 24B MoE as ~2.5B
    pb = (parse_params_b(a.params, {}) or
          next((source_params(d) / 1e9 for d in (a.model, a.base_model) if source_params(d)), 0.0) or
          facts.get("params_b") or parse_params_b(None, cfg) or 0.0)
    f16_gb = pb * 2.0
    builds_sorted = sorted(builds, key=lambda b: -(b.get("bytes") or 0))
    smallest = builds_sorted[-1] if builds_sorted else {}
    small_gb = (smallest.get("bytes") or 0) / 1e9
    pct = (1 - small_gb / f16_gb) * 100 if f16_gb else 0
    x = f16_gb / small_gb if small_gb else 0

    primary = a.lane or (lanes[0] if lanes else "gguf")
    lane_word = LANE_WORD.get(primary, "")
    # ---- frontmatter + hero
    # Both of these name a file that sits at the ROOT of the published repo, so only the basename
    # can ever be correct on the card. Accepting a path and printing it verbatim put
    # "C:\pollard\phome\mmproj-....gguf" in front of every reader and told them to pass it to
    # --mmproj, which works on exactly one machine on earth.
    if a.mmproj:
        a.mmproj = os.path.basename(a.mmproj.rstrip("/\\"))
    if a.imatrix_file:
        a.imatrix_file = os.path.basename(a.imatrix_file.rstrip("/\\"))
    ptag = ("text-classification" if decision and not a.pipeline_tag else
            detect_pipeline_tag(base_model, cfg, a.pipeline_tag, a.mmproj, a.input_support))
    uses_ik = any(runtimes.get(b.get("path")) == "ik_llama" for b in builds)
    out = [frontmatter(base_model, lic, lanes + [l for l, _, _ in also if l not in lanes], mtype, account, ptag,
                       decision, uses_ik), "",
           f"# {name} -- Pollard", ""]
    if f16_gb and small_gb:
        out += [f"> ### Pollard shrank this model{lane_word}: **{f16_gb:.2f} GB (f16) -> {small_gb:.2f} GB** -- "
                f"**{pct:.0f}% smaller, {x:.1f}x down**.",
                "> The smallest rung here; larger, higher-fidelity rungs are listed below."]
        if primary == "gguf":          # the format size table is GGUF-specific
            out += [">", "> | format | this model's size |", "> |---|---:|", f"> | f16 | {f16_gb:.2f} GB |"]
            # If the repo SHIPS this format, quote the real file. An estimated "~22.40 GB" three
            # lines above the shipped 22.43 GB reads as two different files.
            shipped = {str(x.get("tag", "")).upper(): (x.get("bytes") or 0) / 1e9 for x in builds}
            for fmt, mult in FMT_BPP.items():
                if fmt in ("Q8_0", "Q6_K", "Q4_K_M"):
                    real = shipped.get(fmt.upper())
                    out.append(f"> | {fmt} | {real:.2f} GB |" if real
                               else f"> | {fmt} | ~{pb*mult:.2f} GB |")
            out.append(f"> | **PollardMix (this repo's {smallest.get('tag','best')})** | **{small_gb:.2f} GB** |")
        out.append("")
    out += [f"Pollard builds of [{base_model}](https://huggingface.co/{base_model}) made with "
            "[Pollard Weights](https://github.com/WestWaters/pollard-weights) -- a ladder of "
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
            out += ["**Standard GGUF -- every file here runs in stock llama.cpp / ik_llama.cpp, "
                    "Ollama, LM Studio.**", ""]
        elif prism_builds and len(prism_builds) == len(runtimes):
            out += [f"**These files need {PRISM_NEED}** ([#29600](https://github.com/ggml-org/llama.cpp/pull/29600)). "
                    "They are online-RHT rungs: the weights are stored Hadamard-folded (W.D.H) and the "
                    "runtime applies the matching transform to each matmul input. Any llama.cpp from "
                    f"{PRISM_MIN_BUILD} on runs them -- Ollama and LM Studio once they bundle one that new. "
                    "**An older llama.cpp loads them without an error and produces garbage**, and "
                    "ik_llama.cpp is not supported.", ""]
        elif len(ik_builds) == len(runtimes):
            if fork_arch and arch_verdict == "newer":
                out += [f"**Standard GGUF, but you need a recent llama.cpp.** This model's "
                        f"architecture (`{fork_arch}`) is implemented upstream, so any build new "
                        "enough to carry it runs these files -- llama.cpp itself, and Ollama or "
                        "LM Studio once they ship a runtime with it. An older build will refuse them "
                        "with `unknown model architecture`. The quants are ordinary K-quants.", ""]
            elif fork_arch and pol_need:
                out += [f"**{pol_need}.** This model's architecture (`{fork_arch}`) is implemented by a runtime "
                        "patch Pollard ships and is not in upstream llama.cpp yet, so stock llama.cpp -- and "
                        "therefore Ollama and LM Studio -- cannot load these files until it lands. Every rung here was "
                        "built and verified with that runtime (see How to run).", ""]
            elif fork_arch:
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
            ik_only = [b for b in ik_builds if runtimes.get(b.get("path")) != "prism"]
            line = "**Standard GGUF -- runs in stock llama.cpp / ik_llama.cpp, Ollama, LM Studio, except where noted.**"
            if ik_only:
                need = ", ".join(f"`{b.get('tag') or b.get('name')}`" for b in
                                 sorted(ik_only, key=lambda x: -(x.get("bytes") or 0)))
                v_need, _, pron = rung_agreement(len(ik_only))
                line += (f" {need} {v_need} [ik_llama.cpp]({IK_URL}): {pron} allocation puts "
                         "ik_llama-only atoms on the tensors it protects.")
            if prism_builds:
                need = ", ".join(f"`{b.get('tag') or b.get('name')}`" for b in
                                 sorted(prism_builds, key=lambda x: -(x.get("bytes") or 0)))
                v_need, _, _pron = rung_agreement(len(prism_builds))
                line += (f" {need} {v_need} {PRISM_NEED} "
                         "([#29600](https://github.com/ggml-org/llama.cpp/pull/29600)) -- online-RHT "
                         "weights; an older llama.cpp loads them silently wrong, and ik_llama.cpp is "
                         "not supported.")
            out += [line + " The rest run anywhere.", ""]

    # ---- Model details: the at-a-glance table every good Pollard card opens with
    arch = a.arch or mtype or "--"
    out += ["## Model details", "", "| | |", "|---|---|"]
    # Detected from the model and the build directory; an explicit flag still wins.
    # the directory the builds actually live in, from the manifest entries themselves
    bdir = next((os.path.dirname(b["path"]) for b in builds
                 if isinstance(b, dict) and b.get("path") and os.path.dirname(b["path"])), None)
    facts = detect_card_facts(a.model, bdir, cfg, builds)
    inp = a.input_support if a.input_support and a.input_support != "text" else (facts["input"] or a.input_support)
    imat = a.imatrix_file or facts["imatrix"]
    out.append(f"| Parameter count | ~{pb:.1f}B |" if pb else "| Parameter count | -- |")
    out.append(f"| Architecture | `{arch}` |")
    out.append(f"| Input support | {inp} |")
    out.append(f"| imatrix | {'**yes** -- see [calibration](#imatrix-calibration)' if imat else 'no'} |")
    if decision:
        out.append("| Measured | " + (f"**decision fidelity vs {decision_ref(results)}** "
                   + ("through each rung's own runtime, d3 readout -- " if d3_boards(results) else
                      "through `/v1/systemone` -- ")
                   + "[table below](#decision-fidelity) (perplexity does not apply: a decision model "
                   "answers with probabilities, not text)" if dec_rows else "decision fidelity -- pending") + " |")
    else:
        out.append(f"| Perplexity measured | {'**yes** -- table below' if f16_ppl or results else 'pending'} |")
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
            # One decimal, not :.0f. Rounding to whole GB printed 6.53, 7.25 and 7.36 GB as the
            # same "~9 GB" budget, and told a 9 GB machine it could hold a 7.36 GB build -- the
            # 2 GB of context headroom this line exists to add, rounded away.
            head = f"- **~{gb + 2:.1f} GB RAM / VRAM** -> **`{b.get('tag','')}`** ({gb:.2f} GB)."
            # This list is where people actually pick a file, so a rung that stock llama.cpp cannot
            # open has to say so here too -- not only in the table further down.
            rtb = runtimes.get(b.get("path"))
            if rtb == "ik_llama":
                head += " *(ik_llama.cpp)*"
            elif rtb == "prism":
                head += f" *(needs {PRISM_NEED})*"
            elif rtb in ("fork", "newer", "pollard", "unknown"):
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
    # Top-1 agreement: of the tokens the f16 would have picked, how many does this rung still pick.
    # llama-perplexity reports it as "Same top p" on every --kl-divergence run, so it is measured
    # anywhere a KLD is -- it was simply never carried onto the card. It is the figure a reader
    # understands without knowing what a nat of divergence is, which is why people keep asking for it.
    has_top1 = any((results.get(b.get("name", ""), results.get(b.get("tag", ""), {})) or {}).get("top1")
                   for b in builds)
    t1_h = " Top-1 agree |" if has_top1 else ""
    t1_s = "---:|" if has_top1 else ""
    # "runs in" appears only when the ladder is actually mixed -- a uniform ladder says it once
    # in the line above the table instead of repeating itself on every row.
    mixed = bool(ik_builds) and len(ik_builds) != len(runtimes)
    rt_h = " runs in |" if mixed else ""
    rt_s = "---|" if mixed else ""
    # Tool-ready: whether the rung still calls tools (pollard-toolcall), when the results carry
    # it. A quant can keep its perplexity and lose the narrow band of layers that formats tool calls, and
    # an agent harness is exactly where that shows -- so it is a column, not a footnote. Never measured here.
    has_tc = any(isinstance((results.get(b.get("name", ""), results.get(b.get("tag", ""), {})) or {}).get("toolcall"),
                            dict) for b in builds)
    tc_h = " Tool-ready |" if has_tc else ""
    tc_s = "---|" if has_tc else ""
    # verified: the pollard-verify --gate verdict, from the MANIFEST (record_gate) or results.json -- only
    # when a gate actually ran, so a card never implies a check nobody made
    has_v = any(gate_verdict(b, results) is not None for b in builds)
    v_h = " verified |" if has_v else ""
    v_s = "---|" if has_v else ""
    if decision:
        out += [f"| file | size | agrees with {decision_ref(results)} | option KL |{rt_h}{v_h} notes |",
                f"|---|---:|---:|---:|{rt_s}{v_s}---|"]
    else:
        out += [f"| file | PPL | size |{tps_h} Mean KLD |{t1_h}{rt_h}{tc_h}{v_h} notes |",
                f"|---|---:|---:|{tps_s}---:|{t1_s}{rt_s}{tc_s}{v_s}---|"]
    _ordered = sorted(builds, key=lambda x: (x.get("bytes") or 0))
    _rec = pick_recommended(
        [(i, results.get(b.get("name", ""), results.get(b.get("tag", ""), {})).get("ppl"))
         for i, b in enumerate(_ordered)])
    for _i, b in enumerate(_ordered):
        r = results.get(b.get("name", ""), results.get(b.get("tag", ""), {}))
        tps_c = f" {r.get('tps','--')} |" if has_tps else ""
        rt = runtimes.get(b.get("path"))
        rt_lbl = {"ik_llama": "ik_llama", "fork": fork_plain, "newer": fork_plain, "pollard": fork_plain,
                  "unknown": "unverified", "stock": "any llama.cpp",
                  "prism": PRISM_NEED}.get(rt, "--")
        rt_c = (" " + rt_lbl + " |") if mixed else ""
        _ppl, _kld = r.get("ppl"), r.get("kld")
        note = r.get("note") or auto_note(_i, len(builds), _ppl, results.get("_f16_ppl"),
                                          b.get("tag", ""), _rec)
        t1 = r.get("top1")
        t1_c = (f" {float(t1):.2f}% |" if t1 else " -- |") if has_top1 else ""
        if decision:
            d = r.get("decision") or {}
            ag = f"{float(d['agreement']):.0%}" if d.get("agreement") is not None else "--"
            note = r.get("note") or ""
            out.append(f"| `{b.get('name','-')}` | {human_gb(b.get('bytes'))} | {ag} | "
                       f"{_fmt_kl(d.get('option_kl'))} |{rt_c}{verified_cell(b, results) if has_v else ''} {note} |")
            continue
        tc_c = f" {tool_ready_cell(r.get('toolcall'))} |" if has_tc else ""
        v_c = verified_cell(b, results) if has_v else ""
        out.append(f"| `{b.get('name','-')}` | {_fmt_num(_ppl)} | {human_gb(b.get('bytes'))} |{tps_c} "
                   f"{_fmt_num(_kld)} |{t1_c}{rt_c}{tc_c}{v_c} {note} |")
    if has_top1:
        out += ["", "_Top-1 agree = share of tokens where the rung's most-likely token is the same "
                "one the f16 would have picked (llama-perplexity's `Same top p` for GGUF, the same logit "
                "comparison in pollard-verify for the other lanes). Higher is closer to the original model._"]
    if has_v and d3_boards(results):
        out += ["", "_Verified = the rung passed `pollard-d3 gate`: through its own runtime it picks the same option as "
                "bf16 on at least 90% of the typed decisions with a mean option KL of at most 0.05, and it ships the "
                "readout and decision config byte-exact (pollard-complete). A rung that failed is not published._"]
    elif has_v and gate_strict(builds, results):
        out += ["", "_Verified = the rung passed `pollard-verify --gate --strict`: its KLD / top-1 against the source "
                "and every other threshold in the gate file (retention, tool calls, runtime, coherence) were measured "
                "and met._"]
    elif has_v:
        out += ["", "_Verified = the rung passed `pollard-verify --gate`: it is the complete model (every tower, "
                "projector and processor file of the source), it loads, and it passes the coherence and runtime "
                "checks. Its KLD / top-1, tool-call and retention numbers are measured and shown here as they are; "
                "\"below band\" names a number outside the gate file's band for its bit width -- reported, not "
                "blocking._"]
    if has_tc and not decision:
        out += ["", tool_ready_note(builds, results)]
    if has_tps:
        hw = results.get("_hw")
        out.append("")
        out.append(f"_tok/s measured on {hw}._" if hw else
                   "_tok/s is hardware-specific; the machine it was measured on is stated in the errata._")
    if not results:
        out.append("")
        out.append("_Decision-fidelity measurement pending -- sizes and allocation are final._" if decision else
                   "_PPL / Mean-KLD benchmarking pending -- sizes and allocation are final._")
    out.append("")
    if decision and dec_rows:
        out += decision_section(builds, results)
    out += retention_section(builds, results)
    out += moe_allocation_section(builds)
    if one_repo and primary != "gguf":
        out += lane_folder_section(primary, builds, results, repo, facts.get("tools"))
    else:
        out += rungs_section(builds, primary, repo)
    if also:
        out += ["## Every format in this repo", "",
                "One repo holds every Pollard build of this model: the GGUF files above sit at the root, each other "
                "format in its own folder: " + ", ".join(f"`{LANE_FOLDER[l]}/`" for l, _, _ in also), ""]
    for lane_, bb, rr in also:
        out += lane_folder_section(lane_, bb, rr, repo, facts.get("tools"))

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
                f"Vision needs the projector shipped alongside: **`{a.mmproj}`** -- download it too and "
                "pass it with `--mmproj`. It is **not quantized**; it is small and the text ladder is "
                "where the size lives.", "",
                "```bash", f"llama-server -m {exn} --mmproj {a.mmproj} -ngl 99", "```", ""]

    # ---- Fill-in-the-middle: coder models only
    if a.fim:
        out += ["## Fill-in-the-middle (code completion)", "",
                "Use the FIM tokens the base model was trained with, not a chat turn:", "",
                "```", "<|fim_prefix|>{before}<|fim_suffix|>{after}<|fim_middle|>", "```", ""]

    # ---- Download a specific file (a directory lane's rungs are folders: its rungs section says how)
    if primary == "gguf" or not builds:
        out += ["## Download a specific file", "", "```bash",
                'pip install -U "huggingface_hub[cli]"',
                f"hf download {repo} \\", f'  --include "{exn}" --local-dir ./', "```", ""]

    # ---- How to run
    out += ["## How to run", ""]
    if decision and (facts.get("decision_format") == D3_FORMAT or d3_boards(results)):
        import pollard_d3 as D3
        out += D3.serve_lines(D3.info(a.model) if os.path.isdir(str(a.model)) else {"repo_id": base_model},
                              repo, lanes)
    elif "gguf" in lanes and decision:
        if pol_need:
            out += pollard_runtime_lines(pol_need, pol_build)
        out += decision_usage(exn, fork_arch if arch_verdict == "newer" else None)
    elif "gguf" in lanes:
        # The featured command has to name a file that command can actually open. If the
        # recommended rung is ik_llama-only, showing it behind a stock `llama-server -hf` sends
        # people to a load error -- so the stock example moves to a rung that loads, and the
        # recommended one is shown with the build it needs.
        ex_ik = runtimes.get(ex.get("path")) in NONSTOCK
        stock = [b for b in builds if runtimes.get(b.get("path")) == "stock"]
        stock_pick = max(stock, key=lambda b: (b.get("bytes") or 0), default=None)
        if ex_ik and pol_need:
            out += pollard_runtime_lines(pol_need, pol_build)
        elif not ex_ik:
            out += ["These are standard GGUF and run with **llama.cpp**:", "", "```bash",
                    f"llama-server -hf {repo}:{extag}" if extag else f"llama-server -hf {repo}",
                    "```", "", "or from a local file:", "", "```bash",
                    f'llama-cli    -m {exn} -ngl 99 -p "Explain why the sky is blue."',
                    f"llama-server -m {exn} -ngl 99      # OpenAI-compatible API + web UI at :8080",
                    "```", ""]
            out += ["They also work in anything built on llama.cpp -- **LM Studio, koboldcpp, Jan, "
                    f"ramalama, Ollama** (`ollama run hf.co/{repo}`).", ""]
        if ex_ik:
            lead = (f"This model's architecture (`{fork_arch}`) needs llama.cpp with Pollard's patch, so every "
                    "file here runs in the build above"
                    if pol_need else
                    f"This model's architecture (`{fork_arch}`) needs a llama.cpp new enough to "
                    "carry it, so build or update from upstream first"
                    if fork_arch and arch_verdict == "newer" else
                    f"This model's architecture (`{fork_arch}`) needs "
                    f"{fork_where or 'the vendor llama.cpp fork'}, so every file here runs there"
                    if fork_arch else
                    f"`{extag or exn}` is an online-RHT rung, so it needs **{PRISM_NEED}** "
                    "(update first -- an older build loads it silently wrong)"
                    if runtimes.get(ex.get("path")) == "prism" else
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
            elif arch_verdict == "pollard":
                out += ["Ollama and LM Studio will run these once upstream llama.cpp merges "
                        f"`{fork_arch}` and they bundle a build that carries it.", ""]
            elif arch_verdict == "newer":
                out += ["Ollama and LM Studio will run these once their bundled llama.cpp carries "
                        f"`{fork_arch}`.", ""]
            elif prism_builds and len(prism_builds) == len(ik_builds):
                out += [f"Ollama and LM Studio will run these once their bundled llama.cpp is "
                        f"{PRISM_MIN_BUILD} or newer.", ""]
            else:
                out += ["No rung in this repo loads in stock llama.cpp, so Ollama and LM Studio "
                        "cannot run these files.", ""]
    if "mlx" in lanes:
        out += ["```bash", f'mlx_lm.generate --model {repo} --prompt "Hello"', "```", ""]
    vllm_builds = (list(builds) if primary in ("gptq", "mx") else []) + [
        x for l_, bb_, _ in also if l_ in ("gptq", "mx") for x in bb_]
    out += vllm_serve_block(repo, lanes, vllm_builds)
    if "exlr8" in lanes:
        out += exlr8_section(builds, repo, cfg)
    if not decision:
        out += tool_serving_section(facts.get("tools"), lanes, repo, exn, vllm_builds)
    # ---- Speculative decoding: the matched drafter (pollard-draft), measured against these quantized rungs
    if a.draft_results and os.path.exists(a.draft_results) and a.draft_repo:
        import pollard_draft as PD
        org = repo.split("/")[0] if "/" in repo else (account or PLACEHOLDER_OWNER)
        trepos = {**PD.target_repos_for(org, base_model, lanes), primary: repo}
        if one_repo:                                 # the drafter lives in DRAFT/ of this repo
            here = lane_dir(repo)
            trepos = {"gguf": repo, **{l: f"{here}/{LANE_FOLDER[l]}/<rung>" for l, _, _ in also
                                       if l in ("mlx", "gptq", "mx")}}
            out += PD.card_section(json.load(open(a.draft_results, encoding="utf-8")), f"{here}/DRAFT", trepos,
                                   primary, folder_of=repo)
        else:
            out += PD.card_section(json.load(open(a.draft_results, encoding="utf-8")), a.draft_repo, trepos, primary)

    # ---- Use it in your harness: rendered from pollard_harness.HARNESSES, the same table `pollard serve`
    # prints from, so the card and the CLI cannot drift apart
    if "gguf" in lanes and builds and d3_boards(results):
        out += ["## Use it in your harness", "",
                "A d3 rung has no chat endpoint. `pollard-d3 serve --rung <rung>.gguf --source . --port 8080` puts "
                "`/v1/systemone` in front of llama-server (the readout applied to the last token's hidden state), so "
                "anything that calls `/v1/systemone` -- Oh My Pi's `api: typesafe` judge role, the decision-index "
                "`http` engine -- uses it unchanged.", ""]
    elif "gguf" in lanes and builds:
        import pollard_harness as PH
        rt_ex = runtimes.get(ex.get("path"))
        stock_rungs = [b for b in builds if runtimes.get(b.get("path")) == "stock"]
        stock_tag = extag if rt_ex == "stock" else \
            (max(stock_rungs, key=lambda b: (b.get("bytes") or 0)).get("tag") if stock_rungs else None)
        needs = {"ik_llama": "ik_llama.cpp", "fork": fork_plain, "newer": "a current llama.cpp", "pollard": fork_where,
                 "prism": PRISM_NEED}.get(rt_ex)
        out += PH.card_section(repo, exn, extag or None, decision=decision, stock_tag=stock_tag, needs=needs)

    # ---- imatrix / calibration: what the allocation was measured on
    if a.imatrix_file:
        # A note detailed enough to be worth reading ends in its own punctuation; appending a period
        # unconditionally gave the Qwen cards "...the allocation was tuned on.." on the one line whose
        # job is to make the measurement credible.
        note = (a.calib_note or
                "a mixed-domain corpus so the matrix sees every register the model serves").rstrip()
        if note[-1:] not in ".!?":
            note += "."
        out += ["## imatrix (calibration)", "",
                f"The importance matrix (`{a.imatrix_file}`, included) was computed on {note}", ""]
        if a.calib_file:
            out += [f"The exact corpus is included as `{a.calib_file}`, so the allocation can be "
                    "reproduced rather than taken on trust.", ""]

    # ---- ARM / AVX: same boilerplate on every GGUF card, so it stops drifting
    if "gguf" in lanes:
        out += ["## ARM / AVX", "",
                "llama.cpp repacks weights into an interleaved layout at load time for faster inference "
                "on ARM and AVX machines -- no special file needed, online repacking covers these quants. "
                "The old `Q4_0_4_4/4_8/8_8` variants are not required.", ""]

    # ---- errata + footer
    tl = json.load(open(a.thinklean)) if a.thinklean and os.path.exists(a.thinklean) else None
    if tl and tl.get("measured"):
        m = tl["measured"]
        out += ["## Think-lean preset (shorter reasoning)", "",
                "Quantized reasoning models over-use hedging words (\"wait\", \"but\", \"alternatively\"...) -- "
                "Meta FAIR measured it in [arXiv 2606.00206](https://arxiv.org/abs/2606.00206). A small negative logit "
                f"bias on those tokens ({len(tl['token_ids'])} ids for this tokenizer) trims the rumination.", "",
                f"Measured on `{m['gguf']}`, {m['bench']}:", "",
                "| | Accuracy | Mean tokens | Median tokens |", "|---|---|---|---|",
                f"| default | {m['default']['accuracy']:.1%} | {m['default']['mean_tokens']:.0f} | {m['default']['median_tokens']} |",
                f"| think-lean | {m['think-lean']['accuracy']:.1%} | {m['think-lean']['mean_tokens']:.0f} | {m['think-lean']['median_tokens']} |", "",
                f"Tokens {m['token_change_pct']:+.1f}%, accuracy {m['accuracy_change_pts']:+.1f} points.", "",
                "```bash", f"llama-server -m <file>.gguf {tl['llama_cpp_flags']}", "```", ""]
    out += ["## Errata", ""]
    if "gguf" in lanes:
        if ik_builds and fork_arch and arch_verdict == "newer":
            out.append(f"- `general.architecture` is `{fork_arch}`, which upstream llama.cpp added "
                       "recently. A build older than that support refuses these files with `unknown "
                       "model architecture` -- update llama.cpp rather than looking for a different "
                       "quant. Checked with `pollard-ggufcheck`, which reads the architecture and the "
                       "tensor types out of the header and asks upstream what it implements.")
        elif ik_builds and fork_arch and pol_need:
            out.append(f"- `general.architecture` is `{fork_arch}`, which upstream llama.cpp does not implement "
                       f"yet. {pol_need} -- the quant types are ordinary and irrelevant to that. Checked with "
                       "`pollard-ggufcheck`, which reads the architecture out of the header and the patches "
                       "Pollard ships out of `runtime-patches/`.")
        elif ik_builds and fork_arch:
            out.append(f"- `general.architecture` is `{fork_arch}`, which upstream llama.cpp does not "
                       f"implement, so these files load only in "
                       f"{fork_where or 'the vendor fork that adds it'} -- the quant types are "
                       "ordinary and irrelevant to that. Checked with `pollard-ggufcheck`, which "
                       "reads the architecture and the tensor types out of the header.")
        elif ik_builds:
            names = ", ".join(f"`{b.get('tag') or b.get('name')}`" for b in
                              sorted(ik_builds, key=lambda x: -(x.get("bytes") or 0)))
            v_need, v_carry, _ = rung_agreement(len(ik_builds))
            out.append(f"- {names} {v_carry} ik_llama-only atoms and {v_need} ik_llama.cpp to run; "
                       "stock llama.cpp rejects any ggml type above 42 outright. Checked with "
                       "`pollard-ggufcheck`, from the files' tensor types rather than their names.")
        elif runtimes:
            out.append("- Every file here loads in stock llama.cpp -- verified from the tensor types "
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
            "- Method + tooling: [Pollard Weights](https://github.com/WestWaters/pollard-weights) -- "
            "*measure first, no claim before a number.*"]
    out += [f"- License: `{lic}`, inherited from the base model."]
    for c in a.credits:
        out.append("- " + c.lstrip("- ").strip())

    out += ["",
            "*Built with [Pollard Weights](https://github.com/WestWaters/pollard-weights) -- "
            "frontier models, small hardware, no compromise.*"]

    card = "\n".join(out) + "\n"
    if os.path.dirname(os.path.abspath(a.out)):
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
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
