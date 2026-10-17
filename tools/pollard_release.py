#!/usr/bin/env python3
"""pollard release -- ONE run that builds EVERY lane of a model, each lane as a full LADDER of rungs, gates
every rung, cards each lane with its measured numbers, and (only when asked) publishes it.

    pollard release --hf Qwen/Qwen3-8B --plan release.yaml              # = --dry-run: print the full plan
    pollard release --hf Qwen/Qwen3-8B --plan release.yaml --run        # build + verify + card, serially
    pollard release --hf Qwen/Qwen3-8B --plan release.yaml --run --upload   # ...and publish passing rungs
    pollard release --init release.yaml --preset moe                     # write a plan to edit
    pollard release --hf <m> --preset vlm --lanes gguf,mlx --run         # just these lanes on this machine

The lanes and their ladders (what each lane's runtime supports):

  gguf  the publish ladder (pollard --hf <m> --ladder: imatrix + measured sensitivity, the way Pollard builds),
        Q6_K / IQ4_XS / IQ3_S / 2-bit, plus the IQ*_KT flagship where ik_llama builds it      -> <M>-Pollard
  mlx   2 / 3 / 4 / 5 / 6 / 8-bit, measured allocation (pollard-mlx --bits)                     -> <M>-Pollard-MLX
  gptq  int2 / int3 / int4 / int8, measured 4/8 dynamic mix (pollard-export --bits)             -> <M>-Pollard-GPTQ
  mx    NVFP4 / MXFP4 / pure FP8 W8A8 (routers + lm_head kept high, Cohere's recipe), MXFP6 E2M3 / E3M2, and
        measured MX mixes (mix<bpw>: MXFP8 / MXFP6 / FP4 per fused layer, pollard-mx --mix)          -> <M>-Pollard-MX
  exl3  several bpw on EXL3's own allocator, smoothed + Calib 3.0 (pollard-exl3 --bpw)           -> <M>-Pollard-EXL3
  exlr8 ONE trellis encode at K=2..6 (E4M3 codebook, EXLR8 v2 records), then every rung is a manifest: exact
        knapsack on the measured sensitivity, 2.5 / 3 / 3.25 / 4 / 5 / 6 bpw (pollard-exlr8)   -> <M>-Pollard-EXLR8
  draft a speculative drafter matched to the quantized target (pollard-draft): the best published DFlash / DSpark /
        EAGLE-3 head (or a same-family small model), GGUF ladder Q8_0 / Q4_K_M + its HF folder, MEASURED as tokens
        per target pass against every GGUF rung of this release, gated on that                -> <M>-Pollard-DRAFT

ONE repo per model, <org>/<M>-Pollard, holds every lane: the GGUF rungs (<M>-Pollard-<TAG>.gguf) and the mmproj
at the root -- so `llama-server -hf <repo>:<TAG>` works as it always has -- and each other lane in its own folder with
a subfolder per rung: MLX/<rung>/, GPTQ/<rung>/, MX/<rung>/, EXL3/<bpw>/, EXLR8/<bpw>/, and the drafter in DRAFT/ (its
GGUF rungs + its HF folder). ONE card covers every lane: the rung table per lane with the measured numbers and
Tool-ready, and per lane the exact `hf download --include "<LANE>/<rung>/*"` + serve command. Every lane's card step
regenerates it as that lane lands. The repo is created PRIVATE and the last step makes it public only once every
lane that ships this run is uploaded and the file list (README, every rung in its folder, the mmproj) checks out.
Repos published before this (<M>-Pollard-MLX and friends) are left as they are.

How it runs -- and why:
  * pollard-smoke preflight FIRST, before anything is spent.
  * SERIAL, one heavy job at a time, under the machine lock. Lanes this machine cannot build (MLX needs Apple
    Silicon; GPTQ / MX / EXL3 need CUDA) are skipped with the command to run them where they can be built.
  * RESUMABLE: every step is recorded in <workspace>/models/<M>/release-state.json, and a rung the MANIFEST
    already shows built AND verified is never rebuilt -- so the Mac can do MLX and the box the CUDA lanes,
    each re-running the same plan.
  * every rung goes through `pollard-verify --gate` (gates.yaml); a rung that fails is not carded and not
    uploaded. The cards are pollard-card per lane, --verified-only, from the numbers this run measured.
  * nothing is uploaded unless the plan says `upload: true` AND the command line says --upload.
  * --free-after-upload (with --upload): for a box whose free disk is smaller than a lane's whole ladder. Each rung
    is built -> measured -> gated -> uploaded on its own, and its LOCAL file is deleted only once the Hub holds that
    exact file (size + sha256 read back from the Hub). A rung that failed or skipped a measure / verify / gate, or
    whose upload did not land intact, is KEPT on disk unpublished and listed at the end. The GGUF ladder is built one rung at a time (pollard --rung).
    Every build step first checks free disk against the rung's estimated size and holds the rung (not the release)
    when it does not fit. Only files this release built are ever deleted -- never the source, f16, imatrix or calib.
  * existing inputs can be named in the plan instead of rebuilt: `f16:` (an f16/bf16 GGUF), `imatrix:`,
    `sensitivity:` (the raw-weights profile), `calib_file:` + `heldout_file:` (skip pollard-calib).
  * the raw-weights sensitivity profile comes from one of two probes (`probe:` in the plan, default auto):
    the transformers forward probe on the HF weights, or the imatrix probe (pollard-probe --from-imatrix) on the
    GGUF input, which needs no model load. auto takes the imatrix probe whenever a GGUF input exists and the HF
    weights cannot be used -- transformers has no class for the architecture, the checkpoint needs custom code,
    the folder holds no weights, or the weights are bigger than free RAM + accelerator -- and builds the imatrix
    first when the plan names none (on the GGUF itself, or on a Q8_0 host when an f16 is bigger than RAM, at a
    partial -ngl; `imatrix_host:`, `imatrix_ngl:`, `imatrix_chunks:` override).
  * the lanes that export from HF weights (gptq / mx / exl3 / exlr8) STREAM the model block by block from disk
    (pollard_stream): SmoothQuant, GPTQ, the MX calibration pass, the EXLR8 encode and the verify gate each hold
    one block + the activations, preflighted against free RAM (`max_ram_gb:` caps it). A VLM / omni checkpoint is
    built through its language model (towers, talkers and projectors copied through at source precision); a
    checkpoint with its own code runs it only with consent (`trust_remote_code: true` here, or y/N on a terminal),
    unless transformers' native classes cover it (Bolt: native NemotronH blocks, one config per layer);
    `hf_weights:` names the checkpoint the HF lanes read when the --hf folder holds no weights (an already-quantized
    one is dequantized on read -- and an NVFP4 source tensor passes through the MX NVFP4 rung byte for byte). A lane
    is held only for an exact reason: no consent for custom code, exllamav3 has no architecture for the model,
    no weights to read, the disk.
"""
from __future__ import annotations

import argparse, copy, json, os, platform, shutil, subprocess, sys, time

TEMPLATE = """\
# pollard release plan -- every lane, each a ladder of rungs. `pollard release --hf <model> --plan this.yaml`
# prints the plan; add --run to build, --upload to publish (and set upload: true below).
org: PollardWeights            # the HF account the lane repos go to
preset: text                   # text | vlm | decision | moe -- fills anything not set here
ngl: 0                         # GPU layers for llama.cpp steps. 0 = CPU (shared GPU boxes); auto = what fits
upload: false                  # publishing needs this AND --upload on the command line
gates: null                    # gates.yaml for pollard-verify --gate (null = the built-in defaults)
trust_remote_code: false       # true = consent to run the checkpoint's own .py files (only used where no native class fits)
hf_weights: null               # the checkpoint the HF lanes read (dir or repo id) when the --hf folder holds no weights
max_ram_gb: null               # cap the streamed HF lanes' RAM (null = 85% of what is free)
calib:
  profile: auto                # pollard-calib --profile (auto = from pollard-modelkind)
  per_domain: 300
  self_generate: null          # an OpenAI-compatible URL serving the f16, for target-generated calibration
  reasoning: auto              # auto | on | off | toggle -- thinking traces in the model's own delimiters
  reasoning_share: null        # null = 0.35 of the corpus when the model reasons
measure:                       # extra numbers for the card + gate (GGUF rungs; served by llama-server)
  taskeval: {suite: quick, limit: 200}   # null to skip; retention vs f16 per task
  mt: null                     # {limit: 100} -> chrF++ per language (translation models)
  toolcall: auto               # tool-call retention vs f16 (auto = on when pollard-modelkind finds a tool-call syntax)
  decision: false              # decision board vs f16 (decision models)
lanes:
  gguf: {enabled: true}        # the publish ladder + flagship, built the Pollard way (pollard --ladder)
  mlx:  {enabled: true, rungs: [2, 3, 4, 5, 6, 8], default: 4}      # default = the rung also at the repo root
  gptq: {enabled: true, rungs: [2, 3, 4, 8], default: 4}
  mx:   {enabled: true, rungs: [nvfp4, mxfp4, fp8, mxfp6_e2m3, mxfp6_e3m2, mix4.5, mix5, mix5.5], default: nvfp4}
  exl3: {enabled: true, rungs: [2.5, 3.0, 4.0, 5.0, 6.0], default: 4.0}
  # one encode at every width, each rung a manifest over it (GR8's format; dense decoder layers -- MoE experts not yet)
  exlr8: {enabled: true, rungs: [2.5, 3.0, 3.25, 4.0, 5.0, 6.0], default: 4.0, widths: [2, 3, 4, 5, 6]}
  draft: {enabled: false, drafter: auto, rungs: [Q8_0, Q4_K_M], targets: all, limit: 0, max_tokens: 128,
          prefer_own: false, train: {kind: eagle3, rung: Q4_K_M, taps: rung, samples: null, max_steps: null}}
                               # speculative drafter (pollard-draft): drafter = auto (search the Hub) or a repo id;
                               # measured against `targets` (all = every GGUF rung of this release, or a list of tags).
                               # Nothing redistributable published (or prefer_own) -> Pollard trains its own head on a
                               # CUDA box, quant-aware vs train.rung, measured vs the rungs AND f16
"""

PRESETS = {
    # a plain text model: every lane, every rung
    "text": {},
    # a vision-language / omni model: the GGUF lane ships the mmproj (pollard does that itself); MLX holds the tower
    # high; the HF lanes quantize the language model and copy the towers through; EXL3 runs when exllamav3 has the
    # architecture (or its language model as a plain causal LM -- text only) and is held with the reason otherwise
    "vlm": {"measure": {"taskeval": {"suite": "quick", "limit": 200}}},
    # a decision model answers on /v1/systemone from first-position logits: llama.cpp only, judged on decisions
    "decision": {"lanes": {"mlx": {"enabled": False}, "gptq": {"enabled": False}, "mx": {"enabled": False},
                           "exl3": {"enabled": False}, "exlr8": {"enabled": False}},
                 "measure": {"taskeval": None, "decision": True, "toolcall": False}},
    # MoE: vLLM's fused-MoE kernels want one scheme for all experts (--moe-protect-attn-only); 2/3-bit GPTQ
    # MoE kernels are poor, so the GPTQ ladder is 4 and 8. EXLR8 encodes every expert matrix as its own dense record
    # (checkpoint names), decoded by the reference loader -- not EXLR8's per-unit expert files
    "moe": {"lanes": {"gptq": {"rungs": [4, 8]}, "mlx": {"rungs": [3, 4, 6, 8]},
                      "mx": {"moe_protect_attn_only": True}, "exl3": {"rungs": [2.5, 3.0, 4.0]},
                      "exlr8": {"rungs": [2.5, 3.0, 4.0, 5.0]}}},
}
LANES = ["gguf", "mlx", "gptq", "mx", "exl3", "exlr8", "draft"]
MX_SCHEMES = {"nvfp4": "NVFP4", "mxfp4": "MXFP4", "fp8": "FP8", "fp8_dynamic": "FP8_DYNAMIC"}
#: model-free MX rungs (pollard_mxfp6): uniform MXFP6, and `mix<bpw>` measured mixes
MX6_FORMATS = {"mxfp6_e2m3": "mxfp6-e2m3", "mxfp6_e3m2": "mxfp6-e3m2"}


def mx_mix_bpw(rung):
    """`mix5.5` / `mix-5.5` -> 5.5; None for any other rung name."""
    r = str(rung).lower()
    if not r.startswith("mix"):
        return None
    return float(r[3:].lstrip("-_").removesuffix("bpw"))


# the machine each lane builds on -- one table, pollard_setup.LANES, which also lists what each lane needs installed
from pollard_setup import MACHINE as _MACHINE
REQUIRES = {lane: _MACHINE[lane] for lane in LANES}


# ------------------------------------------------------------------------------- plan file

def _yaml(txt):
    try:
        import yaml
        return yaml.safe_load(txt) or {}
    except ImportError:                                  # PyYAML ships with gguf; JSON parses either way
        return json.loads(txt)


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_plan(path=None, preset=None):
    """The template's defaults <- the preset <- the plan file. A preset named on the command line wins over
    the file's `preset:` line, so one plan can be reused across model kinds."""
    user = _yaml(open(path, encoding="utf-8").read()) if path else {}
    name = preset or user.get("preset") or "text"
    if name not in PRESETS:
        raise SystemExit(f"unknown preset '{name}' -- choose from {sorted(PRESETS)}")
    cfg = deep_merge(deep_merge(_yaml(TEMPLATE), PRESETS[name]), user)
    cfg["preset"] = name
    return cfg


def repo_name(org, model, lane=None):
    """ONE repo per model (owner, 2026-10-16): <org>/<M>-Pollard holds every lane -- the GGUF rungs + mmproj at the
    root (so `llama-server -hf <repo>:<TAG>` keeps working), each other lane in its own folder (lane_folder). The
    older per-lane repos (<M>-Pollard-MLX / -GPTQ / ...) stay where they are; nothing new goes to one."""
    import pollard_workspace as ws
    return f"{org}/{ws.model_basename(model)}-Pollard"


def lane_folder(lane):
    """Where a lane lives in the one repo: '' (the root) for GGUF, MLX/ GPTQ/ MX/ EXL3/ EXLR8/ DRAFT/ otherwise."""
    from pollard_card import LANE_FOLDER
    return LANE_FOLDER.get(lane, "")


def rung_tag(lane, rung):
    """The rung's name inside its lane -- the build dir suffix and the branch it is published on."""
    if lane == "mlx":
        return f"{int(rung)}bit"
    if lane == "gptq":
        return f"int{int(rung)}"
    if lane == "mx":
        if mx_mix_bpw(rung) is not None:
            return f"MIX-{mx_mix_bpw(rung):g}bpw"
        if str(rung).lower() in MX6_FORMATS:
            return MX6_FORMATS[str(rung).lower()].upper()
        return MX_SCHEMES[str(rung).lower()]
    if lane == "exl3":
        return f"{float(rung):.1f}bpw"
    if lane == "exlr8":
        from pollard_exlr8 import bpw_tag
        return bpw_tag(rung)                     # 3.25 -> 3.25bpw (EXL3's .1f would print 3.2)
    return str(rung)


# ------------------------------------------------------------------------------- machine

def machine():
    """What this box can build: Apple Silicon (MLX) and/or CUDA (GPTQ / MX / EXL3)."""
    apple = sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")
    cuda = bool(shutil.which("nvidia-smi"))
    return {"any": True, "apple": apple, "cuda": cuda}


# ------------------------------------------------------------------------------- steps

class Step(dict):
    """id, lane, rung, kind (prep/build/measure/verify/card/upload/expand/note), requires, cmd or fn, out."""
    def __getattr__(self, k):
        return self.get(k)


def _ctx(cfg, src):
    import pollard_workspace as ws
    hf_dir = src if os.path.isdir(src) else ws.source_dir(src)
    hfw = cfg.get("hf_weights")
    hf_repo = None
    if hfw:                                        # the HF lanes read another checkpoint (one that has weights)
        if os.path.isdir(str(hfw)):
            hf_dir = os.path.abspath(str(hfw))
        else:
            hf_repo, hf_dir = str(hfw), ws.source_dir(str(hfw))
    mdir = ws.model_dir(src)
    cal = os.path.join(mdir, "calibration")
    name = ws.model_basename(src)
    base = os.path.basename(hf_dir.rstrip("/\\"))
    smooth = hf_dir.rstrip("/\\") + "-smoothed"
    gdir = os.path.join(mdir, "gguf")
    ctx = {
        "src": src, "hf_dir": hf_dir, "mdir": mdir, "name": name, "smooth_dir": smooth,
        "calib": os.path.join(cal, "release-calib.txt"), "heldout": os.path.join(cal, "release-calib.heldout.txt"),
        "sens_raw": os.path.join(cal, f"{name}.sensitivity.json"),
        "sens_smooth": os.path.join(cal, f"{name}-smoothed.sensitivity.json"),
        "gguf_dir": gdir, "f16": os.path.join(gdir, f"{base}-f16.gguf"),
        "exlr8_store": os.path.join(mdir, "exlr8-store"),
        "state": os.path.join(mdir, "release-state.json"), "release_dir": os.path.join(mdir, "release"),
        "ngl": str(cfg.get("ngl", 0)), "gates": cfg.get("gates"),
        "imatrix": cfg.get("imatrix"), "free": bool(cfg.get("_free_after_upload")),
        "org": cfg.get("org", "PollardWeights"), "hf_repo": hf_repo,
        "trc": "on" if cfg.get("trust_remote_code") is True or str(cfg.get("trust_remote_code")).lower() in
               ("true", "on", "yes") else "auto",
        "max_ram_gb": cfg.get("max_ram_gb"),
    }
    # inputs that already exist (a big model's f16 / imatrix / profile / corpus): used as they are, never rebuilt
    for key, plan_key in (("f16", "f16"), ("sens_raw", "sensitivity"), ("calib", "calib_file"),
                          ("heldout", "heldout_file")):
        if cfg.get(plan_key):
            ctx[key] = os.path.abspath(str(cfg[plan_key]))
    ctx["given_f16"] = bool(cfg.get("f16"))
    if cfg.get("measure_ref"):                           # the served measurements' reference (see measure_ref)
        ctx["measure_ref"] = os.path.abspath(str(cfg["measure_ref"]))
    return ctx


def results_path(c, lane):
    return os.path.join(c["mdir"], "reports", f"results-{lane}.json")


def lane_key(c, lane):
    """The MANIFEST key each lane's tool records its builds under (what pollard-card --builds-from reads)."""
    return {"gguf": c["f16"], "mlx": c["hf_dir"], "draft": c["src"]}.get(lane, c["smooth_dir"])


def _verify_gate(c):
    return ["--gate"] + ([c["gates"]] if c["gates"] else [])


PROBE_ROUTES = ("auto", "hf", "imatrix")
#: imatrix chunks when the plan does not say: the full Calib 3.0 corpus is ~9k chunks of 512, hours on a big model
IMATRIX_CHUNKS = 1000


def transformers_loadable(hf_dir):
    """(True, "") when transformers can load this checkpoint as a causal LM or a vision-language model without
    running custom code, else (False, why); (None, why) when this machine cannot tell (no transformers).
    Decided from config.json, without importing the model."""
    cfgp = os.path.join(hf_dir, "config.json")
    if not os.path.isfile(cfgp):
        return False, "no config.json in the HF folder"
    try:
        conf = json.load(open(cfgp, encoding="utf-8"))
    except Exception as e:                                        # noqa: BLE001
        return False, f"config.json is unreadable ({e})"
    if conf.get("auto_map"):
        return False, "the checkpoint needs custom modelling code (config.json auto_map), which the probe never runs"
    mt = conf.get("model_type")
    try:
        from transformers.models.auto.modeling_auto import (MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as LM,
                                                            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES as VL)
    except Exception:                                             # noqa: BLE001
        return None, "transformers is not installed here"            # unknown: not a verdict on the model
    if mt not in LM and mt not in VL:
        return False, f"transformers has no causal-LM or image-text-to-text class for model_type {mt!r}"
    return True, ""


def probe_fits(hf_dir):
    """(True, "") when the HF weights fit free RAM + the accelerator, else (False, why). The forward probe can
    shard a bigger model to disk, but paging tens of GB through it is the wall the imatrix probe exists for."""
    from pollard_auto import _accel_gb, _weights_gb
    from pollard_calc import detect_available_ram_gb
    w = _weights_gb(hf_dir) * (1 << 30) / 1e9
    if not w:
        return False, "the HF folder holds no weights"
    ram, acc = detect_available_ram_gb() or 0.0, _accel_gb() or 0.0
    if w > ram + acc:
        return False, f"{w:.1f} GB of HF weights > {ram:.1f} GB free RAM + {acc:.1f} GB accelerator"
    return True, ""


def probe_route(cfg, c):
    """("hf" | "imatrix", why, transformers can load it). See `probe:` in the module doc."""
    want = str(cfg.get("probe") or "auto").lower()
    if want not in PROBE_ROUTES:
        raise SystemExit(f"probe: {want!r} -- choose from {PROBE_ROUTES}")
    gguf = c["f16"] if (c.get("given_f16") or os.path.isfile(c["f16"])) else None
    local = os.path.isdir(c["hf_dir"])
    if want == "hf":
        return "hf", "probe: hf in the plan", True
    if want == "imatrix":
        if not gguf:
            raise SystemExit("probe: imatrix needs a GGUF input (`f16:` in the plan)")
        ok, why = transformers_loadable(c["hf_dir"]) if local else (True, "")
        return "imatrix", "probe: imatrix in the plan" + (f" ({why})" if ok is False else ""), ok is not False
    if not gguf or not local:
        return "hf", "", True
    ok, why = transformers_loadable(c["hf_dir"])
    if ok is False:
        return "imatrix", why, False
    fits, why = probe_fits(c["hf_dir"])
    return ("hf", "", True) if fits else ("imatrix", why, True)


def prep_steps(cfg, c, lanes):
    cal = cfg.get("calib") or {}
    smoothed = any(l in lanes for l in ("gptq", "mx", "exl3", "exlr8"))
    raw_probe = any(l in lanes for l in ("gguf", "mlx"))
    steps = []
    if not os.path.isdir(c["src"]) or (c.get("hf_repo") and not os.path.isdir(c["hf_dir"])):
        what = c["src"] if not os.path.isdir(c["src"]) else c["hf_repo"]
        steps.append(Step(id="prep/fetch", lane="prep", kind="prep", requires="any", fn="fetch",
                          desc=f"pull {what} -> {c['hf_dir']}"))
    steps.append(Step(id="prep/smoke", lane="prep", kind="prep", requires="any",
                      cmd=["pollard-smoke", "--model", c["hf_dir"]]))
    calib = ["pollard-calib", "--profile", str(cal.get("profile", "auto")), "--model", c["hf_dir"],
             "--per-domain", str(cal.get("per_domain", 300)), "--out", c["calib"], "--held-out", c["heldout"],
             "--record-model", c["src"]]
    if cal.get("reasoning") not in (None, "auto"):              # auto = pollard-modelkind decides
        calib += ["--reasoning", str(cal["reasoning"])]
    if cal.get("reasoning_share") is not None:
        calib += ["--reasoning-share", str(cal["reasoning_share"])]
    if cal.get("self_generate"):
        calib += ["--self-generate", "--server", str(cal["self_generate"])]
    if cfg.get("calib_file"):
        if not cfg.get("heldout_file"):
            raise SystemExit("calib_file: needs heldout_file: too (the verify gate measures on held-out text)")
        steps.append(Step(id="prep/calib", lane="prep", kind="note", requires="any",
                          desc=f"calibration given in the plan: {c['calib']} (held-out {c['heldout']})"))
    else:
        steps.append(Step(id="prep/calib", lane="prep", kind="prep", requires="any", cmd=calib, out=c["calib"]))
    route, why, loadable = probe_route(cfg, c)
    c["probe_route"] = route
    if route == "imatrix" and not c.get("imatrix"):
        c["imatrix"] = os.path.join(c["mdir"], "calibration", f"{c['name']}.imatrix")
    if route == "imatrix" and c.get("imatrix") and not os.path.exists(str(c["imatrix"])):
        steps.append(Step(id="prep/imatrix", lane="prep", kind="prep", requires="any", fn="imatrix",
                          out=c["imatrix"], desc=f"llama-imatrix on the GGUF input (calibration: {c['calib']}) -> "
                                                 f"{c['imatrix']}"))
    if raw_probe and cfg.get("sensitivity"):
        steps.append(Step(id="prep/probe", lane="prep", kind="note", requires="any",
                          desc=f"sensitivity profile given in the plan: {c['sens_raw']}"))
    elif raw_probe and route == "imatrix":
        print(f"   probe: from the imatrix on {c['f16']} -- {why}")
        steps.append(Step(id="prep/probe", lane="prep", kind="prep", requires="any", out=c["sens_raw"],
                          cmd=["pollard-probe", "--from-imatrix", str(c["imatrix"]), "--gguf", c["f16"],
                               "--out", c["sens_raw"]]))
    elif raw_probe:
        probe = ["pollard-probe", "--model", c["hf_dir"], "--eval", c["heldout"], "--out", c["sens_raw"]]
        if c["ngl"] == "0" and machine()["cuda"]:
            probe += ["--device", "cpu"]                 # a shared GPU is not ours (see pollard --ngl 0)
        steps.append(Step(id="prep/probe", lane="prep", kind="prep", requires="any", cmd=probe, out=c["sens_raw"]))
    held = [l for l in lanes if l in ("gptq", "mx", "exl3", "exlr8")]
    access, why_not = hf_lanes_access(c) if smoothed else (True, "")
    if smoothed and not access:
        steps.append(Step(id="prep/smooth", lane="prep", kind="hold", requires="any", blocks=held,
                          desc=f"{', '.join(held)} held: {why_not}"))
    elif smoothed:
        # the LOCKED gold recipe for the low-bit trellis / error-feedback lanes: SmoothQuant first, STREAMED (one
        # block in memory). A small model gets a full smoothed copy (the transformers probe reads it); a big one an
        # overlay of scales when a full copy would not fit the disk (the streamed lanes read it as the smoothed model)
        full = route == "hf" and loadable
        smooth = ["pollard-hf-smooth", "--model", c["hf_dir"], "--out", c["smooth_dir"], "--calib", c["calib"],
                  "--overlay", "off" if full else "on"] + hf_flags(c)
        steps.append(Step(id="prep/smooth", lane="prep", kind="prep", requires="cuda", cmd=smooth,
                          out=c["smooth_dir"], disk_gb=_size_gb(c["hf_dir"]) * 1.02 if full else 0,
                          blocks=held))
        if full:
            steps.append(Step(id="prep/probe-smoothed", lane="prep", kind="prep", requires="cuda",
                              cmd=["pollard-probe", "--model", c["smooth_dir"], "--eval", c["heldout"],
                                   "--out", c["sens_smooth"]], out=c["sens_smooth"], blocks=held))
        else:
            c["sens_smooth"] = c["sens_raw"]
            steps.append(Step(id="prep/probe-smoothed", lane="prep", kind="note", requires="any",
                              desc="the smoothed lanes allocate from the raw-weights profile "
                                   f"({os.path.basename(c['sens_raw'])}): the transformers forward probe cannot run "
                                   f"on this model whole ({why or 'not a plain causal LM'}) -- smoothing is an identity "
                                   "fold, so the layer ranking carries over"))
    return steps


def hf_flags(c, device=True):
    """The flags every streamed HF-lane command shares: consent for custom code (the plan's trust_remote_code),
    the device (CPU when the release runs llama.cpp at -ngl 0 on a shared GPU), the RAM cap."""
    out = ["--trust-remote-code", c.get("trc", "auto")]
    if device and c.get("ngl") == "0" and machine()["cuda"]:
        out += ["--device", "cpu"]
    if c.get("max_ram_gb"):
        out += ["--max-ram-gb", str(c["max_ram_gb"])]
    return out


def hf_lanes_access(c):
    """(True, "") when the streamed HF lanes can build this checkpoint, else (False, the exact reason). Decided from
    config.json + the safetensors index (pollard_stream.static_backbone): nothing is built, no code runs."""
    d = c["hf_dir"]
    if not os.path.isdir(d):
        return True, ""                                # fetched first; the build steps decide
    try:
        import pollard_stream as S
    except Exception:                                  # noqa: BLE001
        return True, ""
    if not os.path.isfile(os.path.join(d, "config.json")):
        return False, f"no config.json in {d}"
    try:
        files = S.weight_files(d)
    except Exception:                                  # noqa: BLE001
        files = []
    missing = [f for f in files if not os.path.exists(f)]
    if missing:                                        # an index whose shards are not here (only the small files)
        return False, (f"{d} holds the index but not the weights ({len(missing)} of {len(files)} shards missing) -- "
                       "name a checkpoint that has them with `hf_weights:` in the plan (an already-quantized one is "
                       "dequantized on read)")
    st = S.static_backbone(d)
    if st["route"] in ("native", "puzzle", "unknown"):
        return True, ""
    if st["route"] == "remote":
        if c.get("trc") == "on":
            return True, ""
        return False, (f"the checkpoint needs its own code ({', '.join(st.get('files') or [])}) and the plan does not "
                       "consent: set `trust_remote_code: true` to run it")
    return False, f"no way to build the text backbone: {st['why']}"


def toolcall_wanted(cfg, c):
    """measure.toolcall: true / false as written; `auto` (the default) = on for an AGENTIC model, i.e. one whose chat
    template writes tool calls in a syntax pollard-modelkind recognises. Asked of the f16 GGUF (it carries the
    template) or the HF checkout, whichever exists by the time the rungs are expanded."""
    v = (cfg.get("measure") or {}).get("toolcall", "auto")
    if v is None or isinstance(v, bool):
        return bool(v)
    if str(v).lower() in ("true", "on", "yes"):
        return True
    if str(v).lower() != "auto":
        return False
    for src in (c.get("f16"), c.get("hf_dir")):
        if src and os.path.exists(src):
            try:
                from pollard_modelkind import classify
                return (classify(src).get("tools") or {}).get("family", "none") != "none"
            except (Exception, SystemExit):                      # noqa: BLE001 -- unreadable: no gate
                return False
    return False


#: lane -> pollard-toolcall's backend for its rungs (the server its users run it with)
TOOLCALL_BACKEND = {"gguf": "gguf", "mlx": "mlx", "gptq": "vllm", "mx": "vllm", "exl3": "exl3", "exlr8": "hf"}


def toolcall_cmd(c, lane, out, name, res):
    """pollard-toolcall for one directory-lane rung. The reference is the ORIGINAL bf16 checkout (not the smoothed
    copy), run the rung's own way where it can be: mlx_lm.server for MLX, vLLM for GPTQ / MX (both fall back to the
    in-process transformers backend where no usable vLLM is installed, e.g. Windows). An EXL3 rung runs in-process
    in exllamav3 and an EXLR8 rung in-process in transformers on its reference decode, so their reference runs
    in-process in transformers -- same template rendering, same call reader. When that cannot run here (no free GPU,
    too big) the f16 GGUF through llama-server stands in, and the result is labelled cross-backend. A backend this
    machine lacks records the rung as not measured and exits 0 -- never a fake number."""
    cmd = ["pollard-toolcall", "--model", out, "--backend", TOOLCALL_BACKEND[lane], "--ref", c["hf_dir"],
           "--ngl", c["ngl"], "--results", res, "--rung", name]
    if lane in ("exl3", "exlr8"):
        cmd += ["--ref-backend", "hf"]
    if lane != "mlx":
        cmd += ["--ref-fallback", c["f16"]]
    return cmd


def gguf_rung_steps(cfg, c, builds):
    """Measure + verify for every rung the GGUF ladder produced (read back from the MANIFEST)."""
    m, res, steps = dict(cfg.get("measure") or {}), results_path(c, "gguf"), []
    m["toolcall"] = toolcall_wanted(cfg, c)
    ref = c.get("measure_ref") or c["f16"]                        # served measurements; verify stays on the f16
    for b in builds:
        tag, path = b.get("tag") or b.get("name"), b.get("path")
        common = ["--results", res, "--rung", tag]
        rid = f"gguf/{tag}"
        if m.get("taskeval"):
            t = m["taskeval"]
            steps.append(Step(id=f"{rid}/taskeval", lane="gguf", rung=tag, kind="measure", requires="any",
                              cmd=["pollard-taskeval", "--model", path, "--ref", ref, "--suite",
                                   str(t.get("suite", "quick")), "--limit", str(t.get("limit", 200)),
                                   "--ngl", c["ngl"], "--out", os.path.join(c["mdir"], "reports", "taskeval", tag)]
                              + common))
        if m.get("mt"):
            steps.append(Step(id=f"{rid}/mt", lane="gguf", rung=tag, kind="measure", requires="any",
                              cmd=["pollard-taskeval", "--model", path, "--ref", ref, "--suite", "mt",
                                   "--limit", str((m["mt"] or {}).get("limit", 100)), "--ngl", c["ngl"]] + common))
        if m.get("toolcall"):
            steps.append(Step(id=f"{rid}/toolcall", lane="gguf", rung=tag, kind="measure", requires="any",
                              cmd=["pollard-toolcall", "--gguf", path, "--ref", ref, "--ngl", c["ngl"]] + common,
                              ok_codes=(0, 1)))                        # a FAIL is a number for the gate, not a crash
        if m.get("decision"):
            steps.append(Step(id=f"{rid}/decision", lane="gguf", rung=tag, kind="measure", requires="any",
                              cmd=["pollard-decision", "--gguf", path, "--ref", ref, "--ngl", c["ngl"]] + common,
                              ok_codes=(0, 1)))
        steps.append(Step(id=f"{rid}/verify", lane="gguf", rung=tag, kind="verify", requires="any", out=path,
                          cmd=["pollard-verify", "--lane", "gguf", "--model", path, "--ref", c["f16"],
                               "--eval", c["heldout"], "--ngl", c["ngl"], "--coherence"] + _verify_gate(c) + common))
        if c.get("free"):
            steps.append(_publish_rung_step(c, "gguf", tag, path))
    return steps


def _publish_rung_step(c, lane, tag, path):
    repo = repo_name(c["org"], c["src"])
    return Step(id=f"{lane}/{tag}/publish", lane=lane, rung=tag, kind="publish", requires="any", fn="publish_rung",
                out=path, repo=repo, desc=f"upload this rung -> https://huggingface.co/{repo}, then free its local copy")


def draft_steps(cfg, c, lc):
    """find -> convert (GGUF ladder + HF folder) -> expand: measure + gate each drafter rung against the target rungs."""
    found = os.path.join(c["mdir"], "draft", "drafter.json")
    find = ["pollard-draft", "find", "--target", c["src"], "--target-dir", c["hf_dir"], "--out", found]
    if lc.get("drafter") not in (None, "auto"):
        find += ["--drafter", str(lc["drafter"])]
    rungs = ",".join(str(r) for r in lc.get("rungs") or ["Q8_0", "Q4_K_M"])
    conv = ["pollard-draft", "convert", "--target", c["src"], "--target-dir", c["hf_dir"], "--from", found,
            "--key", c["src"], "--rungs", rungs, "--skip-if-trained"]
    tc = lc.get("train") or {}
    train = ["pollard-draft", "train", "--target", c["src"], "--target-dir", c["hf_dir"], "--key", c["src"],
             "--if-needed", found, "--kind", str(tc.get("kind") or "eagle3"), "--rung-tag", str(tc.get("rung") or
                                                                                              "Q4_K_M"),
             "--rung-key", c["f16"], "--taps", str(tc.get("taps") or "rung"), "--rungs", rungs, "--ngl", c["ngl"],
             "--measure-target", "f16=" + c["f16"], "--run"]
    for k in ("samples", "max_steps"):
        if tc.get(k):
            train += ["--" + k.replace("_", "-"), str(tc[k])]
    if lc.get("prefer_own"):
        train += ["--prefer-own"]
    return [Step(id="draft/find", lane="draft", kind="build", requires="any", cmd=find, out=found,
                 desc="the best published drafter for this model, checked against its config",
                 ok_codes=(0, 1)),                             # none found is an answer: the train step takes over
            Step(id="draft/train", lane="draft", kind="build", requires="cuda", cmd=train,
                 out=os.path.join(c["mdir"], "draft", "train-" + str(tc.get("kind") or "eagle3")),
                 desc="Pollard's own quant-aware head when nothing redistributable is published"),
            Step(id="draft/convert", lane="draft", kind="build", requires="any", cmd=conv,
                 out=os.path.join(c["mdir"], "draft")),
            Step(id="draft/rungs", lane="draft", kind="expand", requires="any")]


def draft_targets(cfg, c):
    """(tag, path) of the target rungs a drafter is measured against: this release's GGUF rungs (all, or the plan's
    list), else the f16 -- labelled as such on the card."""
    want = ((cfg.get("lanes") or {}).get("draft") or {}).get("targets", "all")
    rungs = [(b.get("tag") or b.get("name"), b.get("path")) for b in gguf_builds(c) if b.get("path")]
    if want not in (None, "all"):
        keep = {str(x).lower() for x in (want if isinstance(want, list) else [want])}
        rungs = [r for r in rungs if str(r[0]).lower() in keep]
    return rungs or [("f16", c["f16"])]


def draft_rung_steps(cfg, c, builds, targets=None):
    """Measure + verify every drafter rung the MANIFEST holds. The HF folder is measured through its BF16 GGUF (the
    same weights) so its gate has a number; the BF16 GGUF itself is an intermediate, not a rung."""
    lc = (cfg.get("lanes") or {}).get("draft") or {}
    res, steps = results_path(c, "draft"), []
    targets = list(targets or draft_targets(cfg, c))
    if any(b.get("source") == "pollard-trained" for b in builds) and not any(p == c["f16"] for _, p in targets):
        targets.append(("f16", c["f16"]))                     # a quant-aware head shows the bf16 number beside it
    for b in builds:
        if b.get("publish") is False or b.get("format") == "mlx":
            continue
        tag, path = b.get("tag") or b.get("name"), b.get("path")
        rid = f"draft/{tag}"
        m = ["pollard-draft", "measure", "--drafter", path, "--rung", tag, "--results", res, "--ngl", c["ngl"],
             "--max-tokens", str(lc.get("max_tokens", 128))]
        if lc.get("limit"):
            m += ["--limit", str(lc["limit"])]
        if lc.get("n_max") not in (None, "auto"):
            m += ["--n-max", str(lc["n_max"])]
        if b.get("format") == "hf" and b.get("via"):
            m += ["--via", b["via"]]
        for t, p in targets:
            m += ["--target", p, "--target-tag", t]
        steps.append(Step(id=f"{rid}/measure", lane="draft", rung=tag, kind="measure", requires="any", cmd=m))
        steps.append(Step(id=f"{rid}/verify", lane="draft", rung=tag, kind="verify", requires="any", out=path,
                          cmd=["pollard-verify", "--lane", "draft", "--model", path, "--rung", tag, "--results", res]
                          + _verify_gate(c)))
    return steps


def lane_steps(cfg, c, lane):
    lc = (cfg.get("lanes") or {}).get(lane) or {}
    import pollard_workspace as ws
    if lane == "draft":
        return draft_steps(cfg, c, lc)
    if lane == "gguf":
        src = ["--gguf", c["f16"]] if c.get("given_f16") else ["--hf", c["hf_dir"], "--output", c["gguf_dir"]]
        cmd = ["pollard"] + src + ["--calib", c["calib"],
               "--sensitivity", c["sens_raw"], "--ladder", "--run", "--no-card", "--force",
               "--no-toolcall",                          # release measures every rung itself (gguf/rungs)
               "--ngl", "auto" if c["ngl"] == "auto" else c["ngl"],
               "--calib-profile", str((cfg.get("calib") or {}).get("profile", "auto"))]
        if c.get("imatrix"):
            cmd += ["--imatrix", os.path.abspath(str(c["imatrix"]))]
        if not c.get("free"):
            return [Step(id="gguf/ladder", lane="gguf", kind="build", requires="any", cmd=cmd, out=c["f16"],
                         desc="convert -> imatrix -> measured publish ladder (+ IQ*_KT flagship where ik_llama builds)"),
                    Step(id="gguf/rungs", lane="gguf", kind="expand", requires="any")]
        # one rung at a time: build -> measure + gate -> upload -> free, so the peak disk is ONE rung, not the ladder
        steps = []
        for label, gb in gguf_ladder_labels(c):
            steps.append(Step(id=f"gguf/ladder-{label}", lane="gguf", rung=label, kind="build", requires="any",
                              cmd=cmd + ["--rung", label], out=None, disk_gb=gb,
                              desc=f"the {label} rung of the measured publish ladder"))
            steps.append(Step(id=f"gguf/rungs-{label}", lane="gguf", kind="expand", requires="any"))
        return steps
    steps, res = [], results_path(c, lane)
    tool = toolcall_wanted(cfg, c)
    cfg_dir = next((d for d in (c["hf_dir"], c["src"]) if os.path.isfile(os.path.join(str(d), "config.json"))), None)
    if lane == "exl3" and cfg_dir:
        try:
            import pollard_exl3 as E
            ep = E.exl3_input_plan(cfg_dir)
        except Exception as e:                                   # noqa: BLE001
            ep = {"mode": "direct", "why": f"(not checked here: {e})"}
        if ep["mode"] == "hold":
            return [Step(id="exl3/hold", lane="exl3", kind="hold", requires="any", blocks=["exl3"],
                         desc=f"exl3 held: {ep['why']}")]
        if ep["mode"] == "text":
            steps.append(Step(id="exl3/text-backbone", lane="exl3", kind="note", requires="any",
                              desc=f"EXL3 of the language model only: {ep['why']}"))
    if lane == "exlr8":
        # the ONE encode every rung of this lane is a manifest over (resumable per layer inside pollard-exlr8)
        widths = ",".join(str(int(k)) for k in (lc.get("widths") or [2, 3, 4, 5, 6]))
        steps.append(Step(id="exlr8/encode", lane="exlr8", kind="build", requires=REQUIRES["exlr8"],
                          cmd=["pollard-exlr8", "--model", c["smooth_dir"], "--store", c["exlr8_store"], "--calib",
                               c["calib"], "--widths", widths, "--encode-only"] + hf_flags(c), out=c["exlr8_store"],
                          desc="encode every matrix once at each width (LDLQ + tail-biting Viterbi, E4M3 codebook)"))
    for rung in lc.get("rungs") or []:
        tag = rung_tag(lane, rung)
        out = ws.resolve_out(c["src"], lane, tag, create_parent=False)
        name = os.path.basename(out)
        if lane == "mlx":
            cmd = ["pollard-mlx", "--model", c["hf_dir"], "--sensitivity", c["sens_raw"], "--bits", str(int(rung)),
                   "--out", out]
            src = c["hf_dir"]
        elif lane == "gptq":
            cmd = ["pollard-export", "--model", c["smooth_dir"], "--sensitivity", c["sens_smooth"],
                   "--calib", c["calib"], "--bits", str(int(rung)), "--out", out, "--engine", "auto"] + hf_flags(c)
            src = c["smooth_dir"]
        elif lane == "mx" and (mx_mix_bpw(rung) is not None or str(rung).lower() in MX6_FORMATS):
            # model-free, weight-only, CPU: no calibration set, no llm-compressor
            cmd = ["pollard-mx", "--model", c["smooth_dir"], "--sensitivity", c["sens_smooth"], "--out", out,
                   "--trust-remote-code", c.get("trc", "auto")]
            cmd += (["--mix", f"{mx_mix_bpw(rung):g}"] if mx_mix_bpw(rung) is not None
                    else ["--format", MX6_FORMATS[str(rung).lower()]])
            src = c["smooth_dir"]
        elif lane == "mx":
            scheme = MX_SCHEMES[str(rung).lower()]
            cmd = ["pollard-mx", "--model", c["smooth_dir"], "--calib", c["calib"], "--scheme", scheme, "--out", out,
                   "--engine", "auto"] + hf_flags(c)
            if scheme not in ("FP8", "FP8_DYNAMIC"):
                cmd += ["--sensitivity", c["sens_smooth"]]
                if lc.get("moe_protect_attn_only"):
                    cmd += ["--moe-protect-attn-only"]
            src = c["smooth_dir"]
        elif lane == "exlr8":                            # a manifest over the encode: seconds, no re-encode
            cmd = ["pollard-exlr8", "--model", c["smooth_dir"], "--store", c["exlr8_store"], "--calib", c["calib"],
                   "--sensitivity", c["sens_smooth"], "--widths", ",".join(str(int(k)) for k in (lc.get("widths") or [2, 3, 4, 5, 6])),
                   "--bpw", str(float(rung)), "--out", out] + hf_flags(c)
            src = c["smooth_dir"]
        else:                                            # exl3
            cmd = ["pollard-exl3", "--model", c["smooth_dir"], "--bpw", str(float(rung)), "--calib-text", c["calib"],
                   "--out", out]
            src = c["smooth_dir"]
        rid = f"{lane}/{tag}"
        st_build = Step(id=f"{rid}/build", lane=lane, rung=tag, kind="build", requires=REQUIRES[lane], cmd=cmd,
                        out=out, src=src, disk_gb=_rung_gb(lane, rung, c["hf_dir"]) if c.get("free") else 0,
                        held_codes=(EXIT_HELD,) if lane == "exl3" else ())
        if c.get("free") and lane in STREAMABLE:
            # bigger than the free disk: written + uploaded + confirmed + freed one shard at a time, not held
            from pollard_card import rung_subfolder
            st_build["stream_to"] = (repo_name(c["org"], c["src"]),
                                     f"{lane_folder(lane)}/{rung_subfolder({'name': name, 'tag': tag}, lane)}")
        steps.append(st_build)
        if tool:
            steps.append(Step(id=f"{rid}/toolcall", lane=lane, rung=tag, kind="measure", requires=REQUIRES[lane],
                              cmd=toolcall_cmd(c, lane, out, name, res), ok_codes=(0, 1)))
        # gated against the ORIGINAL weights, not the smoothed copy: fidelity to the model people asked for
        ver = ["pollard-verify", "--lane", lane, "--model", out, "--source", c["hf_dir"], "--rung", name,
               "--results", res] + _verify_gate(c)
        if lane in ("gptq", "mx", "exlr8"):
            ver += ["--trust-remote-code", c.get("trc", "auto")]
            if c.get("ngl") == "0" and machine()["cuda"]:
                ver += ["--device", "cpu"]
        ver += ["--end-to-end"] if lane == "exl3" else ["--eval", c["heldout"]]
        steps.append(Step(id=f"{rid}/verify", lane=lane, rung=tag, kind="verify", requires=REQUIRES[lane],
                          cmd=ver, out=out))
        if c.get("free"):
            st = _publish_rung_step(c, lane, tag, out)
            st["default"] = lc.get("default") is not None and rung_tag(lane, lc["default"]) == tag
            steps.append(st)
    return steps


#: directory lanes whose writers can stream a rung to the Hub shard by shard (pollard_stream.ShardWriter: GPTQ;
#: pollard_mxfp6.write_shards: every pollard-engine MX rung)
STREAMABLE = ("gptq", "mx")
#: free disk a streamed build still needs: the shard being written (a source shard can be ~5 GB) + the margin
STREAM_SHARD_GB = 6.0


def gguf_ladder_labels(c):
    """[(label, estimated GB)] of the GGUF ladder, rung by rung, for --free-after-upload. An f16/bf16 source gets
    pollard's publish ladder (Q6_K / IQ4_XS / IQ3_S / IQ2 + the automap flagship, sized as fractions of the f16); an
    already-quantized source (a vendor QAT GGUF) gets the passthrough ladder (PT25..PT100, sized from its tensors)."""
    f16_gb = _size_gb(c["f16"])
    try:
        import pollard_passthrough as pt
        ts = pt.read_tensors(c["f16"]) if os.path.isfile(c["f16"]) else None
        if ts is not None and pt.is_quantized_source(ts):
            return [(lb, b / 1e9 * 1.02) for lb, b in pt.ladder_targets(ts, imatrix_ok=bool(c.get("imatrix")))]
    except Exception:                                              # noqa: BLE001 -- unreadable: the f16 ladder
        pass
    from pollard_auto import LADDER
    return [(label, frac * f16_gb * 1.05) for label, _, frac in list(LADDER) + [("flagship", (), 0.20)]]


def _draft_on(cfg):
    return bool(((cfg.get("lanes") or {}).get("draft") or {}).get("enabled"))


def card_cmd(cfg, c, lanes):
    """The ONE card for the one repo: the primary lane (GGUF when it is in the run) in full, every other lane as its
    own folder section with its measured numbers and download + serve commands, the drafter in DRAFT/. Every lane's
    card step regenerates it, so the README always describes what has landed so far."""
    repo = repo_name(cfg.get("org", "PollardWeights"), c["src"])
    body = [l for l in lanes if l != "draft"]
    primary = "gguf" if "gguf" in body else (body[0] if body else "gguf")
    card = ["pollard-card", "--model", c["src"] if not os.path.isdir(c["src"]) else c["hf_dir"],
            "--base-model", c["src"], "--lane", primary, "--builds-from", lane_key(c, primary),
            "--results", results_path(c, primary), "--repo", repo, "--out", os.path.join(c["release_dir"], "README.md"),
            "--verified-only"]
    for other in body:
        if other != primary:
            card += ["--also", other, lane_key(c, other), results_path(c, other)]
    if "draft" in lanes or _draft_on(cfg):
        card += ["--also", "draft", c["src"], results_path(c, "draft"),
                 "--draft-results", results_path(c, "draft"), "--draft-repo", repo]
    if primary == "gguf":
        card += ["--calib-file", os.path.basename(c["calib"]),
                 "--calib-note", f"the pollard-calib '{(cfg.get('calib') or {}).get('profile', 'auto')}' profile "
                                 "(sources, revisions and licences in the shipped .sources.json)"]
    if c.get("free"):
        card += ["--runtime-from-repo"]                  # freed rungs are read from their published copy
    return card


def publish_steps(cfg, c, lane, lanes=None):
    repo = repo_name(cfg.get("org", "PollardWeights"), c["src"])
    outdir = c["release_dir"]
    steps = [Step(id=f"{lane}/card", lane=lane, kind="card", requires="any", cmd=card_cmd(cfg, c, lanes or [lane]),
                  out=outdir, repo=repo)]
    where = f"{lane_folder(lane)}/ in " if lane_folder(lane) else ""
    steps.append(Step(id=f"{lane}/upload", lane=lane, kind="upload", requires="any", fn="upload", repo=repo,
                      out=outdir, desc=f"publish the verified {lane} rungs + card -> {where}"
                                       f"https://huggingface.co/{repo}"))
    return steps


def public_step(cfg, c, lanes):
    repo = repo_name(cfg.get("org", "PollardWeights"), c["src"])
    return Step(id="release/public", lane="release", kind="finalize", requires="any", fn="go_public", repo=repo,
                lanes=list(lanes), desc=f"every lane that ships this run is on {repo} and its file list checks out "
                                        "-> make the repo public (it is created private)")


def plan_steps(cfg, src, only=None):
    c = _ctx(cfg, src)
    lanes = [l for l in LANES if ((cfg.get("lanes") or {}).get(l) or {}).get("enabled")
             and (not only or l in only)]
    steps = prep_steps(cfg, c, lanes)
    # the ONE card describes every lane the plan enables -- also the ones an earlier pass shipped (a pass-2 run with
    # --lanes gptq,mx keeps the GGUF section it did not rebuild); a lane with no verified rung adds nothing to it
    card_lanes = [l for l in LANES if ((cfg.get("lanes") or {}).get(l) or {}).get("enabled")]
    # the drafter is measured against the GGUF rungs and every lane's card quotes it: build + gate it right after the
    # GGUF ladder, before any card is written
    early_draft = "draft" in lanes and "gguf" in lanes
    for lane in lanes:
        if lane != "draft" or not early_draft:
            steps += lane_steps(cfg, c, lane)
        if lane == "gguf" and early_draft:
            steps += lane_steps(cfg, c, "draft")
        steps += publish_steps(cfg, c, lane, card_lanes)
    if lanes:
        steps.append(public_step(cfg, c, lanes))
    return c, lanes, steps


# ------------------------------------------------------------------------------- state / resume

def load_state(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}


def save_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)


def _manifest_entry(path):
    import pollard_workspace as ws
    return ws.find_build(path)[1] if path else None


def resume_reason(step, state, redo=()):
    """Why this step can be skipped on a re-run, or None to run it."""
    if any(step.id == r or step.id.startswith(r.rstrip("/") + "/") for r in redo):
        return None
    if (state.get(step.id) or {}).get("status") == "done" and step.kind != "upload":
        return "done"
    if step.kind in ("build", "verify"):
        b = _manifest_entry(step.out)
        if b and b.get("verified") is True and os.path.exists(step.out):
            return "built + verified (MANIFEST)"
    return None


def gguf_builds(c):
    import pollard_workspace as ws
    return [b for b in ws.read_manifest(c["f16"]).get("builds", []) if b.get("lane") == "gguf"]


# ------------------------------------------------------------------------------- execution

def _fmt_cmd(cmd):
    """A command for the plan printout, with the workspace root shortened to $POLLARD_HOME."""
    try:
        import pollard_workspace as ws
        home = ws.pollard_home()
    except Exception:
        home = None
    parts = []
    for x in cmd:
        x = str(x)
        if home and x.startswith(home):
            x = "$POLLARD_HOME" + x[len(home):]
        parts.append(f'"{x}"' if " " in x else x)
    return " ".join(parts)


def fetch(c):
    import pollard_workspace as ws
    if not os.path.isdir(c["src"]):
        ws.fetch_source(c["src"])
    if c.get("hf_repo") and not os.path.isdir(c["hf_dir"]):
        ws.fetch_source(c["hf_repo"])


def canonical_gguf(c):
    """Give every ladder rung its published name, <M>-Pollard-<TAG>.gguf, in place (a rename, no copy), and
    re-point its MANIFEST entry -- so the verify, the card and the repo all say the same file name. pollard-fit
    names a build after the f16 it came from, which is not a name to publish."""
    import pollard_workspace as ws
    key = c["f16"]
    man = ws.read_manifest(key)
    changed = False
    for b in man.get("builds", []):
        if b.get("lane") != "gguf" or not b.get("tag"):
            continue
        want = f"{c['name']}-Pollard-{b['tag']}.gguf"
        path = b.get("path", "")
        if os.path.basename(path) == want or not os.path.exists(path):
            continue
        target = os.path.join(os.path.dirname(path), want)
        os.replace(path, target)
        b["path"], b["name"] = os.path.abspath(target), want
        changed = True
    if changed:
        # a re-run's rung lands on the published name an earlier, freed / kept / replaced build of the same tag had:
        # the newest entry for a path is the build that is there now, so older entries for it go (publish_rung would
        # otherwise find the stale one first -- unverified -- and keep the fresh, verified file off the Hub)
        seen, keepl = set(), []
        for b in reversed(man.get("builds", [])):
            p = os.path.normcase(os.path.abspath(str(b.get("path", ""))))
            if b.get("lane") == "gguf" and p in seen:
                continue
            seen.add(p)
            keepl.append(b)
        man["builds"] = list(reversed(keepl))
        ws._write_manifest(key, man)


def default_build(builds, lane, want):
    """The rung that also goes at the repo root: the plan's default if it verified, else the largest."""
    from pollard_card import rung_subfolder
    if want is not None:
        tag = rung_tag(lane, want).lower()
        for b in builds:
            if rung_subfolder(b, lane) == tag:
                return b
    return max(builds, key=lambda b: b.get("bytes") or 0) if builds else None


def lane_expect(c, lane, builds, repo):
    """The repo paths a lane's verified rungs must have once published (the file-list check)."""
    from pollard_card import rung_subfolder
    F = lane_folder(lane)
    out = set()
    for b in builds:
        if lane == "gguf":
            out.add(b.get("uploaded_as") or b.get("name") or os.path.basename(b["path"]))
        elif lane == "draft":
            if b.get("publish") is False:
                continue
            if b.get("format") == "gguf":
                out.add(f"{F}/{b.get('name') or os.path.basename(b['path'])}")
            elif b.get("format") == "hf":
                out.add(f"{F}/config.json")
        else:
            out.add(f"{F}/{rung_subfolder(b, lane)}/config.json")
    if lane == "gguf" and os.path.isdir(c["gguf_dir"]):
        out |= {f for f in os.listdir(c["gguf_dir"]) if f.startswith("mmproj-")}
    return out


def _lane_builds(c, lane, repo):
    import pollard_workspace as ws
    return [b for b in ws.read_manifest(lane_key(c, lane)).get("builds", [])
            if b.get("lane") == lane and b.get("verified") is True
            and (os.path.exists(b.get("path", "")) or b.get("uploaded") == repo)]


def upload(step, c, cfg, run_upload):
    """Publish the verified rungs of one lane + the (combined) card into the ONE repo: GGUF files at the root, a
    directory lane as <LANE>/<rung>/, the drafter in DRAFT/. Refuses a lane with no verified rung, and checks the repo
    afterwards: a lane whose files are not all there -- or a repo without its README -- is not published."""
    from pollard_card import rung_subfolder
    lane, repo = step.lane, step.repo
    builds = _lane_builds(c, lane, repo)
    card = os.path.join(step.out, "README.md")
    if not builds:
        raise RuntimeError(f"no verified {lane} rung to publish")
    if not os.path.exists(card):
        raise RuntimeError(f"no card at {card} -- the card step must run first")
    if not run_upload:
        print(f"      (not uploading: {len(builds)} verified rung(s) ready -- needs `upload: true` in the plan "
              "AND --upload)")
        return "held"
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")          # Xet write tokens expire mid-transfer
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True, private=True)   # public only once complete
    F = lane_folder(lane)
    if lane == "draft":
        for b in builds:
            if b.get("publish") is False:
                continue
            if b.get("format") == "gguf":
                name = b.get("name") or os.path.basename(b["path"])
                api.upload_file(path_or_fileobj=b["path"], path_in_repo=f"{F}/{name}", repo_id=repo)
            elif b.get("format") == "hf":                      # the HF drafter: vLLM / SGLang / MLX load DRAFT/
                api.upload_folder(folder_path=b["path"], path_in_repo=F, repo_id=repo)
    elif lane == "gguf":
        for b in builds:
            name = b.get("uploaded_as") or b.get("name") or os.path.basename(b["path"])
            if b.get("uploaded") != repo:                      # --free-after-upload already put it there
                api.upload_file(path_or_fileobj=b["path"], path_in_repo=name, repo_id=repo)
        if c.get("imatrix") and os.path.exists(str(c["imatrix"])):
            api.upload_file(path_or_fileobj=str(c["imatrix"]), path_in_repo=f"{c['name']}.imatrix", repo_id=repo)
        for side in (c["calib"], c["calib"] + ".sources.json"):
            if os.path.exists(side):
                api.upload_file(path_or_fileobj=side, path_in_repo=os.path.basename(side), repo_id=repo)
        for f in sorted(os.listdir(c["gguf_dir"])) if os.path.isdir(c["gguf_dir"]) else []:
            if f.endswith(".imatrix") or f.startswith("mmproj-"):
                api.upload_file(path_or_fileobj=os.path.join(c["gguf_dir"], f), path_in_repo=f, repo_id=repo)
    else:
        for b in builds:
            if b.get("uploaded") != repo:
                api.upload_folder(folder_path=b["path"], path_in_repo=f"{F}/{rung_subfolder(b, lane)}", repo_id=repo)
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo)
    missing = (lane_expect(c, lane, builds, repo) | {"README.md"}) - set(api.list_repo_files(repo))
    if missing:
        raise RuntimeError(f"{repo} is missing {sorted(missing)} after upload -- not live")
    print(f"      landed: {len(builds)} {lane} rung(s) in https://huggingface.co/{repo}" + (f" ({F}/)" if F else ""))
    return "uploaded"


def go_public(step, c, cfg, run_upload, statuses, out=print):
    """The last step: the repo was created private and stays private until EVERY lane that ships this run is
    uploaded and the whole file list (README, each lane's rungs in its folder, the mmproj) is on the Hub. A lane
    whose upload failed keeps it private; a lane that built nothing (held / failed / not this machine) ships
    nothing and does not block."""
    if not run_upload:
        out("      (not publishing: uploads are off)")
        return "held"
    ups = {l: statuses.get(f"{l}/upload") for l in (step.lanes or [])}
    bad = sorted(l for l, v in ups.items() if v == "failed")
    if bad:
        raise RuntimeError(f"{step.repo} stays private: the {', '.join(bad)} upload failed")
    shipped = [l for l, v in ups.items() if v == "done"]
    if not shipped:
        out(f"      (nothing shipped this run -- {step.repo} stays as it is)")
        return "held"
    import pollard_workspace as ws
    stranded = [p for l in LANES for b in ws.read_manifest(lane_key(c, l)).get("builds", [])
                if b.get("lane") == l and b.get("streamed_unverified") and b.get("verified") is not True
                for p in b["streamed_unverified"][:1]]
    if stranded:
        raise RuntimeError(f"{step.repo} stays private: it holds streamed shards of a rung that did not pass its gate "
                           f"({', '.join(sorted({os.path.dirname(p) for p in stranded}))}) -- verify it or remove that "
                           "folder, then re-run")
    from huggingface_hub import HfApi
    api = HfApi()
    want = {"README.md"}
    for l in shipped:
        want |= lane_expect(c, l, _lane_builds(c, l, step.repo), step.repo)
    files = set(api.list_repo_files(step.repo))
    missing = want - files
    if missing:
        raise RuntimeError(f"{step.repo} stays private: missing {sorted(missing)}")
    try:
        api.update_repo_settings(repo_id=step.repo, private=False)
    except AttributeError:                                       # older huggingface_hub
        api.update_repo_visibility(step.repo, private=False)
    out(f"      PUBLIC: https://huggingface.co/{step.repo}  ({', '.join(shipped)}; {len(files)} files checked)")
    return "done"


def _size_gb(path):
    """Size of a file or a directory tree in GB (0 when it is not there yet)."""
    if not path or not os.path.exists(path):
        return 0.0
    if os.path.isfile(path):
        return os.path.getsize(path) / 1e9
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot / 1e9


#: average bits per weight of a directory-lane rung, for the disk estimate (not a claim about the build)
_MX_BITS = {"nvfp4": 4.5, "mxfp4": 4.25, "fp8": 8.0, "fp8_dynamic": 8.0, "mxfp6_e2m3": 6.25, "mxfp6_e3m2": 6.25}


def _rung_gb(lane, rung, hf_dir):
    """Estimated size of one directory-lane rung: the bf16 checkpoint scaled by the rung's bits, plus 10%."""
    r = str(rung).lower()
    if lane == "mx":
        bits = mx_mix_bpw(rung) if mx_mix_bpw(rung) is not None else _MX_BITS.get(r, 8.0)
    else:
        try:
            bits = float(rung)
        except ValueError:
            bits = 8.0
    return _size_gb(hf_dir) * bits / 16.0 * 1.1


def disk_free_gb(path):
    p = os.path.abspath(path or ".")
    while not os.path.exists(p) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    return shutil.disk_usage(p).free / 1e9


def _protected_paths(c):
    """Never deleted by --free-after-upload: the inputs, whatever a build entry says."""
    keep = {c.get("f16"), c.get("hf_dir"), c.get("smooth_dir"), c.get("calib"), c.get("heldout"), c.get("imatrix"),
            c.get("sens_raw"), c.get("sens_smooth"), c.get("src")}
    return {os.path.normcase(os.path.abspath(str(k).rstrip("/\\"))) for k in keep if k}


def _freeable(c, b):
    """A build file this release made: under the workspace model dirs, or the automap flagship mix the ladder
    wrote next to the f16. Anything else (an input, a file someone put there) is left alone."""
    import pollard_workspace as ws
    path = os.path.normcase(os.path.abspath(str(b.get("path", "")).rstrip("/\\")))
    if not path or path in _protected_paths(c):
        return False
    roots = [ws.model_dir(c["src"]), ws.model_dir(c["f16"]), ws.model_dir(c["hf_dir"]),
             ws.model_dir(c["smooth_dir"]), c["mdir"]]
    if any(path.startswith(os.path.normcase(os.path.abspath(r)) + os.sep) for r in roots):
        return True
    # the flagship mix lives next to the f16: as automap wrote it (<stem>-mix.gguf) or under its published name
    # once canonical_gguf renamed it (<M>-Pollard-<TAG>.gguf)
    name = os.path.basename(path)
    return bool(b.get("flagship")) and os.path.dirname(path) == os.path.normcase(
        os.path.dirname(os.path.abspath(c["f16"]))) and (
        path.endswith("-mix.gguf") or name.startswith(os.path.normcase(f"{c['name']}-Pollard-")))


def _free_local(c, b, out=print):
    path = b.get("path", "")
    if not _freeable(c, b):
        out(f"      (kept {path}: not a file this release built)")
        return False
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
    out(f"      freed {path} (disk free now {disk_free_gb(path):.1f} GB)")
    return True


def _sha256(path):
    import pollard_hubcheck as H
    return H.sha256_file(path)


def _git_blob_sha1(path):
    import pollard_hubcheck as H
    return H.git_blob_sha1(path)


def _local_pairs(path, path_in_repo):
    """[(path in the repo, local file)] for one file or every file under a folder."""
    if os.path.isfile(path):
        return [(path_in_repo, path)]
    pairs = []
    for root, _, files in os.walk(path):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            pairs.append((f"{path_in_repo}/{rel}" if path_in_repo else rel, full))
    return sorted(pairs)


def remote_mismatches(api, repo, pairs):
    """[why] for every (path in repo, local file | recorded {size, sha256}) the Hub does NOT hold intact (missing,
    another size, another hash, no hash); [] = every file is there. See pollard_hubcheck."""
    import pollard_hubcheck as H
    return H.mismatches(api, repo, pairs)


def _keep(c, lane, key, man, b, why, out):
    """A rung that is NOT freed: recorded in the MANIFEST (kept = why) and said out loud."""
    b["kept"] = why
    b.pop("freed", None)
    ws_write(key, man)
    out(f"      KEPT {b.get('path')} -- {why}")


def ws_write(key, man):
    import pollard_workspace as ws
    ws._write_manifest(key, man)


def publish_rung(step, c, cfg, run_upload, out=print):
    """--free-after-upload: upload ONE gated rung, confirm THAT file is on the Hub byte-for-byte (size + sha256
    against the Hub's own listing), mark it uploaded in the MANIFEST, and only then delete the local copy.

    Anything short of that keeps the file: a rung that failed or skipped its measure / verify / gate is NOT
    published and NOT deleted (the LFM2.5 release, 2026-10-16, deleted three unverified, never-uploaded rungs --
    five hours of builds -- because "did not pass the gate" was treated as "done with it"); an upload that raises,
    or a file the Hub does not hold intact, keeps the local copy and fails the step."""
    import pollard_workspace as ws
    from pollard_card import rung_subfolder
    lane, repo = step.lane, step.repo
    key = lane_key(c, lane)
    man = ws.read_manifest(key)
    want = os.path.normcase(os.path.abspath(str(step.out).rstrip("/\\")))
    b = next((x for x in man.get("builds", []) if x.get("lane") == lane and
              os.path.normcase(os.path.abspath(str(x.get("path", "")).rstrip("/\\"))) == want), None)
    if b is None:
        if not os.path.exists(str(step.out)):
            out("      (nothing to publish: the rung was not built)")
            return "skipped: not built"
        raise RuntimeError(f"{step.out} is not in the MANIFEST")
    if b.get("uploaded") and not os.path.exists(b.get("path", "")):
        return "done"
    import pollard_stream as PS
    streamed = PS.read_streamed(b["path"]) if lane != "gguf" and os.path.isdir(b.get("path", "")) else {}
    if b.get("verified") is not True:
        if streamed:
            b["streamed_unverified"] = sorted(e["path_in_repo"] for e in streamed.values())
        _keep(c, lane, key, man, b, "not verified (measure / verify / gate failed or did not run) -- not published"
              + (f"; its {len(streamed)} streamed shard(s) stay in the PRIVATE repo, which will not go public "
                 "until they are verified or removed" if streamed else ""), out)
        return "kept: not verified"
    if not run_upload:
        raise RuntimeError("--free-after-upload needs uploading on (`upload: true` + --upload)")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True, private=True)   # public only once complete
    if lane == "gguf":
        name = b.get("name") or os.path.basename(b["path"])
        try:                                     # pollard-ggufcheck before it leaves: the card states the runtime
            import pollard_ggufcompat as gc
            b["runtime"] = gc.runtime_of(b["path"])[0]
            out(f"      ggufcheck: {name} -> {b['runtime']}")
        except Exception as e:                                   # noqa: BLE001
            out(f"      ggufcheck: could not read {name} ({e}) -- the card will not state its runtime")
        dest = name
        if not b.get("uploaded"):
            api.upload_file(path_or_fileobj=b["path"], path_in_repo=dest, repo_id=repo)
    else:
        dest = f"{lane_folder(lane)}/{rung_subfolder(b, lane)}"
        if not b.get("uploaded") and streamed:
            # the shards are already there (streamed + confirmed one by one); the rest, then the index LAST -- a
            # loader sees the rung only once every shard it maps is in place
            api.upload_folder(folder_path=b["path"], path_in_repo=dest, repo_id=repo,
                              ignore_patterns=["model.safetensors.index.json", PS.STREAMED, "*.tmp"])
            if os.path.isfile(os.path.join(b["path"], "model.safetensors.index.json")):
                api.upload_file(path_or_fileobj=os.path.join(b["path"], "model.safetensors.index.json"),
                                path_in_repo=f"{dest}/model.safetensors.index.json", repo_id=repo)
        elif not b.get("uploaded"):
            api.upload_folder(folder_path=b["path"], path_in_repo=dest, repo_id=repo)
    b["uploaded_as"] = dest
    pairs = [(rp, lp) for rp, lp in _local_pairs(b["path"], dest) if os.path.basename(lp) != PS.STREAMED]
    pairs += [(e["path_in_repo"], {"size": e["size"], "sha256": e["sha256"], "git_sha1": e.get("git_sha1")})
              for e in streamed.values()]
    bad = remote_mismatches(api, repo, pairs)
    if bad:
        _keep(c, lane, key, man, b, "upload not confirmed: " + "; ".join(bad[:5]), out)
        raise RuntimeError(f"{dest} is not intact in {repo} after the upload ({bad[0]}) -- keeping the local copy")
    b["uploaded"] = repo
    b.pop("kept", None)
    if streamed:
        b["bytes"] = int(_size_gb(b["path"]) * 1e9) + sum(int(e["size"]) for e in streamed.values())
    ws._write_manifest(key, man)
    out(f"      uploaded {dest} -> https://huggingface.co/{repo} (size + hash confirmed on the Hub)")
    if _free_local(c, b, out):
        b["freed"] = "uploaded + confirmed on the Hub"
        ws._write_manifest(key, man)
    return "done"


def _free_vram_mib():
    """Free VRAM on the first NVIDIA GPU in MiB, 0 when there is none (pollard_llamaserve, shared with the tools)."""
    import pollard_llamaserve as LS
    return LS.free_vram_mib()


def partial_ngl(gguf, free_mib, headroom_mib=1536, overhead_mib=1536):
    """-ngl from FREE VRAM: the last N blocks that fit, 99 only when the whole file does (pollard_llamaserve)."""
    import pollard_llamaserve as LS
    return LS.partial_ngl(gguf, free_mib, headroom_mib, overhead_mib)


def _total_ram_gb():
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
            m = MS()
            m.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.total / 1e9
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:                                             # noqa: BLE001
        return 0.0


def measure_ref(cfg, c):
    """The GGUF the SERVED measurements (taskeval / toolcall / mt / decision) compare each rung against. The f16,
    unless the plan names `measure_ref:` -- or the f16 cannot be served here: bigger than RAM + VRAM it pages off
    disk at well under a token a second (the LFM2.5 f16, 48 GB on a 32 GB + 16 GB box: 0.75 tok/s, then the server
    died). Then the release's own Q8_0 imatrix host stands in when it exists, and says so. The KLD verify always
    runs against the f16 (prompt processing only, which paging survives)."""
    if cfg.get("measure_ref"):
        return os.path.abspath(str(cfg["measure_ref"]))
    f16 = c["f16"]
    room = _total_ram_gb() * 0.85 + _free_vram_mib() / 1024 * 1.073741824
    if room and _size_gb(f16) > room:
        for cand in (os.path.join(c["gguf_dir"], f"{c['name']}-Q8_0-host.gguf"),
                     os.path.join(os.path.dirname(f16), f"{c['name']}-Q8_0.gguf")):
            if os.path.isfile(cand):
                print(f"   measure reference: {cand} (the f16 is {_size_gb(f16):.0f} GB, more than RAM + VRAM "
                      f"~{room:.0f} GB, so it cannot be served at a usable speed; the KLD verify stays on the f16)")
                return cand
    return f16


def imatrix_host(cfg, c):
    """(GGUF the imatrix runs on, quant type to build it as or None). The GGUF input itself when it is already
    quantized (a vendor QAT file) or fits in RAM; else a Q8_0 host next to it -- an f16 bigger than RAM pages
    off disk for every chunk (the 30B lesson: compute on a smaller host at a partial -ngl)."""
    if cfg.get("imatrix_host"):
        return os.path.abspath(str(cfg["imatrix_host"])), None
    src = c["f16"]
    try:
        import pollard_passthrough as pt
        if pt.is_quantized_source(pt.read_tensors(src)):
            return src, None
    except Exception:                                             # noqa: BLE001
        pass
    from pollard_calc import detect_available_ram_gb
    if _size_gb(src) <= (detect_available_ram_gb() or 0.0) * 0.8:
        return src, None
    return os.path.join(c["gguf_dir"], f"{c['name']}-Q8_0-host.gguf"), "Q8_0"


def imatrix_ngl(cfg, c, host):
    if cfg.get("imatrix_ngl") is not None:
        return str(int(cfg["imatrix_ngl"]))
    if c["ngl"] != "auto":
        return c["ngl"]
    free = _free_vram_mib()
    if free:
        return str(partial_ngl(host, free))
    from pollard_auto import _fit_ngl                             # Apple / no nvidia-smi: the unified-memory fit
    return _fit_ngl(host, "auto")


def build_imatrix(c, cfg, out=print):
    """The prep/imatrix step: (Q8_0 host if needed ->) llama-imatrix over the release calibration -> c['imatrix']
    (legacy .dat, which ik_llama's trellis flagship also reads). A failure at a partial -ngl retries once on the
    CPU; nothing is renamed into place unless llama-imatrix exited 0."""
    from pollard_auto import _imatrix_shape
    from pollard_calc import find_llama_bin
    imx = str(c["imatrix"])
    host, qtype = imatrix_host(cfg, c)
    if qtype and not os.path.exists(host):
        need = _size_gb(c["f16"]) * 0.55
        free = disk_free_gb(host)
        if free < need + float(cfg.get("disk_margin_gb", 3)):
            raise RuntimeError(f"a {qtype} imatrix host needs ~{need:.1f} GB, {free:.1f} GB free")
        q = find_llama_bin("llama-quantize")
        if not q:
            raise RuntimeError("llama-quantize not found (pollard-runtime)")
        os.makedirs(os.path.dirname(host), exist_ok=True)
        out(f"      {qtype} imatrix host: {q} {c['f16']} {host} {qtype}")
        run_cmd([q, c["f16"], host + ".partial", qtype])
        os.replace(host + ".partial", host)
    binim = find_llama_bin("llama-imatrix")
    if not binim:
        raise RuntimeError("llama-imatrix not found (pollard-runtime)")
    os.makedirs(os.path.dirname(os.path.abspath(imx)), exist_ok=True)
    ngl = imatrix_ngl(cfg, c, host)
    base = [binim, "-m", host, "-f", c["calib"], "-o", imx + ".partial", "--output-format", "dat"]
    base += _imatrix_shape(host) or ["-c", "512", "-b", "512", "-ub", "512"]
    chunks = cfg.get("imatrix_chunks", IMATRIX_CHUNKS)
    if chunks:
        base += ["--chunks", str(int(chunks))]
    if os.environ.get("POLLARD_THREADS"):
        base += ["-t", os.environ["POLLARD_THREADS"], "-tb", os.environ["POLLARD_THREADS"]]
    try:                                                          # chat-template tokens in the corpus stay tokens
        if "parse-special" in subprocess.run([binim, "--help"], capture_output=True, text=True, timeout=60).stdout:
            base += ["--parse-special"]
    except Exception:                                             # noqa: BLE001
        pass
    for attempt in ([ngl, "0"] if ngl != "0" else ["0"]):
        if os.path.exists(imx + ".partial"):
            os.remove(imx + ".partial")
        out(f"      $ {_fmt_cmd(base + ['-ngl', attempt])}")
        try:
            run_cmd(base + ["-ngl", attempt])
        except RuntimeError as e:
            out(f"      llama-imatrix -ngl {attempt}: {e}" + (" -- retrying on the CPU" if attempt != "0" else ""))
            continue
        if os.path.exists(imx + ".partial"):
            os.replace(imx + ".partial", imx)
            return imx
    raise RuntimeError(f"llama-imatrix produced no imatrix at {imx}")


#: a lane tool that exits with this code HELD the step for an exact reason it printed (no exllamav3 architecture, no
#: disk for the files exllamav3 reads) -- recorded as held, not failed
EXIT_HELD = 3


class Held(Exception):
    pass


def _toolcall_not_measured(st, why):
    """Record a rung's tool calls as NOT MEASURED, with the reason, the way pollard-toolcall does (the gate's
    toolcall_not_measured policy and the card read it from there)."""
    cmd = list(st.cmd)
    res, rung = cmd[cmd.index("--results") + 1], cmd[cmd.index("--rung") + 1]
    from pollard_taskeval import save_results
    save_results(res, rung, {"toolcall": {"status": "not-measured", "reason": why, "lane": st.lane}})
    return "done"


def run_cmd(cmd, ok_codes=(0,), env=None):
    from pollard_auto import _resolve_tool
    r = subprocess.run(_resolve_tool(list(cmd)), env=env) if env else subprocess.run(_resolve_tool(list(cmd)))
    if r.returncode not in ok_codes:
        raise RuntimeError(f"exit {r.returncode}")


def execute(steps, c, cfg, run=False, run_upload=False, redo=(), stop_on_fail=False, have=None, out=print):
    """Print (dry-run) or run the plan, serially. Returns {step id: status}."""
    have = have or machine()
    state = load_state(c["state"])
    statuses, failed_rungs, failed_lanes, unbuildable = {}, set(), set(), set()
    kept = []                                        # (lane, rung, path) left on disk unpublished -- reported at the end
    streamed = set()                                 # (lane, rung) built in streaming mode (shards already on the Hub)
    queue, n = list(steps), 0
    while queue:
        st = queue.pop(0)
        n += 1
        head = f"{n:3d}. [{st.lane}{'/' + st.rung if st.rung else ''}] {st.kind}"
        if st.kind == "expand" and st.lane == "draft":
            import pollard_draft as D
            builds = D.draft_builds(c["src"])
            if builds:
                queue[0:0] = draft_rung_steps(cfg, c, builds)
                out(f"{head}: {len(builds)} drafter rung(s) in the MANIFEST -> measure vs the target rungs + gate each")
            elif run:
                out(f"{head}: no drafter rungs in the MANIFEST (find/convert failed or was skipped)")
            else:
                tg = None if gguf_builds(c) else [("<each GGUF rung>", "<rung .gguf>")]
                for s in draft_rung_steps(cfg, c, [{"tag": "<drafter rung>", "path": "<each drafter .gguf>"}], tg):
                    out(f"     (per rung) {s.kind}: " + (s.desc if s.fn else f"$ {_fmt_cmd(s.cmd)}"))
            n -= 1
            continue
        if st.kind == "expand":
            if run and have.get("any"):
                canonical_gguf(c)
                if "measure_ref" not in c:
                    c["measure_ref"] = measure_ref(cfg, c)
            builds = gguf_builds(c)
            if c.get("free"):                        # one rung at a time: only what is built and not yet published
                done = {s.get("id") for s in queue} | set(statuses)
                builds = [b for b in builds if not b.get("uploaded") and not b.get("freed")
                          and os.path.exists(b.get("path", ""))
                          and f"gguf/{b.get('tag') or b.get('name')}/publish" not in done]
            if builds:
                queue[0:0] = gguf_rung_steps(cfg, c, builds)
                out(f"{head}: {len(builds)} ladder rung(s) in the MANIFEST -> measure + verify each")
            elif run:
                out(f"{head}: no ladder rungs in the MANIFEST (the ladder step failed or was skipped)")
            else:
                for s in gguf_rung_steps(cfg, c, [{"tag": "<rung>", "path": "<each ladder rung .gguf>"}]):
                    out(f"     (per rung) {s.kind}: " + (s.desc if s.fn else f"$ {_fmt_cmd(s.cmd)}"))
            n -= 1
            continue
        if st.kind == "note":
            out(f"{head}: {st.desc}")
            statuses[st.id] = "note"
            continue
        if st.kind == "hold":                         # a lane this model cannot take: recorded, the rest carry on
            out(f"{head}: HELD -- {st.desc}")
            statuses[st.id] = f"held: {st.desc}"
            failed_lanes.update(st.blocks or [])
            if run:
                state[st.id] = {"status": statuses[st.id], "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                save_state(c["state"], state)
            continue
        what = st.desc if st.fn else f"$ {_fmt_cmd(st.cmd)}"
        if not have.get(st.requires, False):
            out(f"{head}: SKIP on this machine (needs {st.requires}) -- {what}")
            statuses[st.id] = f"skipped: needs {st.requires}"
            unbuildable.update(l for l, r in REQUIRES.items() if r == st.requires)
            continue
        if st.lane in unbuildable and st.kind in ("card", "upload"):
            out(f"{head}: SKIP (this lane builds on a {REQUIRES[st.lane]} machine -- run the same plan there "
                f"with --lanes {st.lane}) -- {what}")
            statuses[st.id] = f"skipped: needs {REQUIRES[st.lane]}"
            continue
        if st.kind == "publish" and (st.lane in failed_lanes or (st.lane, st.rung) in failed_rungs):
            # free mode: a rung whose build / measure / verify failed or was skipped is neither published NOR deleted
            out(f"{head}: KEPT locally, not published (an earlier step of this rung failed) -- {st.out}")
            statuses[st.id] = "kept: upstream failed"
            kept.append((st.lane, st.rung, st.out))
            continue
        if st.lane in failed_lanes or (st.lane, st.rung) in failed_rungs:
            out(f"{head}: SKIP (an earlier step of this {'rung' if st.rung else 'lane'} failed) -- {what}")
            statuses[st.id] = "skipped: upstream failed"
            continue
        need = st.disk_gb or 0
        stream_env = None
        if run and need and st.kind in ("build", "prep") and resume_reason(st, state, redo) is None:
            free = disk_free_gb(st.out or c["mdir"])
            margin = float(cfg.get("disk_margin_gb", 3))
            if free < need + margin and st.stream_to and run_upload and free >= STREAM_SHARD_GB + margin:
                import pollard_stream as PS
                repo, sub = st.stream_to
                stream_env = dict(os.environ, **{PS.SHARD_UPLOAD_ENV: json.dumps(
                    {"out": os.path.abspath(st.out), "repo": repo, "path_in_repo": sub})})
                streamed.add((st.lane, st.rung))
                out(f"{head}: ~{need:.1f} GB > {free:.1f} GB free -- STREAMING: each shard written -> uploaded to "
                    f"{repo}/{sub}/ -> confirmed -> freed before the next (the repo stays private)")
            elif free < need + margin:
                msg = f"held: disk (needs ~{need:.1f} GB + {margin:g} margin, {free:.1f} GB free)"
                out(f"{head}: {msg} -- {what}")
                statuses[st.id] = msg
                if st.rung:
                    failed_rungs.add((st.lane, st.rung))
                elif st.blocks:
                    failed_lanes.update(st.blocks)
                else:
                    failed_lanes.add(st.lane)
                continue
        why = resume_reason(st, state, redo)
        if why:
            out(f"{head}: [{why}] {what}")
            statuses[st.id] = f"skipped: {why}"
            continue
        out(f"{head}: {what}")
        if not run:
            statuses[st.id] = "planned"
            continue
        try:
            if st.fn == "fetch":
                fetch(c)
                res = "done"
            elif st.fn == "upload":
                res = "done" if upload(st, c, cfg, run_upload) == "uploaded" else "held"
            elif st.fn == "go_public":
                res = go_public(st, c, cfg, run_upload, statuses, out)
            elif st.fn == "imatrix":
                build_imatrix(c, cfg, out)
                res = "done"
            elif st.fn == "publish_rung":
                res = publish_rung(st, c, cfg, run_upload, out)
            elif st.kind == "measure" and (st.lane, st.rung) in streamed and "pollard-toolcall" in str(st.cmd[0]):
                res = _toolcall_not_measured(st, "the rung was streamed to the Hub shard by shard while it was built "
                                                 "(bigger than the free disk); the tool-call probe needs it on disk")
                out("      tool calls: not measured (streamed rung)")
            else:
                try:
                    run_cmd(st.cmd, st.ok_codes or (0,), **({"env": stream_env} if stream_env else {}))
                except RuntimeError as e:
                    if str(e) in {f"exit {h}" for h in (st.held_codes or ())}:
                        raise Held(f"{e} (the reason is the HELD: line above)") from None
                    raise
                res = "done"
        except Held as e:
            out(f"      HELD: {e}")
            statuses[st.id] = f"held: {e}"
            if st.rung:
                failed_rungs.add((st.lane, st.rung))
            else:
                failed_lanes.add(st.lane)
            continue
        except Exception as e:                                   # noqa: BLE001
            res = "failed"
            out(f"      FAILED: {e}")
            if st.blocks and not stop_on_fail:            # e.g. the smoothed copy: only its lanes depend on it
                failed_lanes.update(st.blocks)
                statuses[st.id] = res
                state[st.id] = {"status": res, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                save_state(c["state"], state)
                continue
            if st.lane == "prep" or stop_on_fail:
                statuses[st.id] = res
                state[st.id] = {"status": res, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                save_state(c["state"], state)
                raise SystemExit(f"   release stopped at {st.id}" + ("" if st.lane != "prep" else
                                 " -- a preparation step failed, nothing after it can be trusted"))
            if st.rung:
                failed_rungs.add((st.lane, st.rung))
            elif st.kind in ("build", "card"):
                failed_lanes.add(st.lane)                    # no card -> nothing to publish
        statuses[st.id] = res
        if str(res).startswith("kept"):
            kept.append((st.lane, st.rung, st.out))
        if res != "held":
            state[st.id] = {"status": res, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            save_state(c["state"], state)
    if run and kept:
        out("\n   KEPT on disk, NOT published (fix the measure / gate and re-run the plan to publish them):")
        for lane, rung, path in kept:
            out(f"     {lane}/{rung}: {path}")
    return statuses


# ------------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(prog="pollard release", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--hf", help="HF repo id or local HF dir of the f16/bf16 model (or `model:` in the plan)")
    ap.add_argument("--plan", help="release.yaml (default: the built-in template)")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="model-kind preset (overrides the plan's preset:)")
    ap.add_argument("--lanes", help="comma list: only these lanes this run (e.g. mlx on the Mac, gptq,mx,exl3 on the box)")
    ap.add_argument("--run", action="store_true", help="execute (default: print the plan)")
    ap.add_argument("--dry-run", action="store_true", help="print the full plan and exit (the default)")
    ap.add_argument("--upload", action="store_true", help="publish verified rungs (also needs `upload: true` in the plan)")
    ap.add_argument("--redo", action="append", default=[], metavar="STEP",
                    help="re-run this step id (or prefix, e.g. mlx/4bit) even if the state says done")
    ap.add_argument("--stop-on-fail", action="store_true", help="stop at the first failed rung (default: carry on)")
    ap.add_argument("--init", metavar="PATH", help="write the plan template (with --preset applied) and exit")
    ap.add_argument("--force", action="store_true", help="ignore the machine lock")
    ap.add_argument("--skip-setup-check", action="store_true",
                    help="run even when `pollard setup` finds a lane's dependencies missing on this machine")
    ap.add_argument("--free-after-upload", action="store_true",
                    help="build -> gate -> upload -> delete ONE rung at a time (a disk smaller than the ladder); "
                         "needs --upload and `upload: true`")
    a = ap.parse_args(argv)

    if a.init:
        if os.path.exists(a.init):
            raise SystemExit(f"{a.init} exists -- not overwriting")
        txt = TEMPLATE if not a.preset else TEMPLATE.replace("preset: text", f"preset: {a.preset}", 1)
        open(a.init, "w", encoding="utf-8").write(txt)
        print(f"wrote {a.init} (preset {a.preset or 'text'}) -- edit it, then: pollard release --hf <model> --plan {a.init}")
        return 0
    cfg = load_plan(a.plan, a.preset)
    if a.free_after_upload:
        if not (a.upload and cfg.get("upload")):
            ap.error("--free-after-upload deletes each rung once it is published: it needs --upload AND "
                     "`upload: true` in the plan")
        cfg["_free_after_upload"] = True
    src = a.hf or cfg.get("model")
    if not src:
        ap.error("pass --hf <repo-or-dir> (or set `model:` in the plan)")
    only = {x.strip() for x in a.lanes.split(",")} if a.lanes else None
    if only and only - set(LANES):
        ap.error(f"unknown lane(s) {sorted(only - set(LANES))}; choose from {LANES}")
    run = a.run and not a.dry_run
    c, lanes, steps = plan_steps(cfg, src, only)
    have = machine()
    print(f"== pollard release :: {src}  (preset {cfg['preset']}, {'RUN' if run else 'plan only'})")
    print(f"   lanes: {', '.join(lanes) or 'none'}   |   this machine: "
          + ", ".join(k for k in ("apple", "cuda") if have[k]) + (" (CPU only)" if not (have['apple'] or have['cuda']) else ""))
    for lane in lanes:
        lc = (cfg["lanes"].get(lane) or {})
        rungs = "publish ladder (Q6_K / IQ4_XS / IQ3_S / 2-bit + flagship)" if lane == "gguf" else \
            ("drafter " + str(lc.get("drafter", "auto")) + ": GGUF " + ", ".join(str(r) for r in lc.get("rungs") or [])
             + " + HF folder" if lane == "draft" else ", ".join(rung_tag(lane, r) for r in lc.get("rungs") or []))
        print(f"   {lane:5s} -> {repo_name(cfg.get('org', 'PollardWeights'), src, lane):48s} {rungs}")
    upload_ok = bool(cfg.get("upload")) and a.upload
    print(f"   upload: {'ON' if upload_ok else 'off'}"
          + ("" if upload_ok else " (needs `upload: true` in the plan AND --upload)"))
    print(f"   state: {c['state']}\n")
    # what each lane needs installed, checked BEFORE the first step: a lane missing gptqmodel stops here with the
    # fix command, not hours in. Lanes this machine cannot build at all are left to execute() (it skips them).
    import pollard_setup as S
    here = [lane for lane in lanes if have.get(REQUIRES.get(lane, "any"), False)]
    if run and not a.skip_setup_check:
        if S.require(here):
            return 2
    elif not run:
        S.hint(here, out=print)
    if run:
        from pollard_auto import MachineLock
        with MachineLock("pollard release " + src, force=a.force):
            st = execute(steps, c, cfg, True, upload_ok, a.redo, a.stop_on_fail, have)
    else:
        st = execute(steps, c, cfg, False, False, a.redo, a.stop_on_fail, have)
        print("\n   plan only -- add --run to execute (serially, resumable), --upload to publish.")
        return 0
    bad = [k for k, v in st.items() if v == "failed"]
    print(f"\n   {sum(v == 'done' for v in st.values())} step(s) done, {len(bad)} failed"
          + (": " + ", ".join(bad) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
