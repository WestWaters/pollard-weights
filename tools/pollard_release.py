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
regenerates it as that lane lands. The repo is created PUBLIC (`private: true` in the plan keeps it private). The
card goes up FIRST: as soon as a rung is measured and has passed, the card is rebuilt from the rungs built and passed
so far and uploaded, and the rung's files follow it. The last step checks the file list (README, every rung in its
folder, the mmproj) of every lane that shipped this run. Repos published before this (<M>-Pollard-MLX and friends)
are left as they are.

How it runs -- and why:
  * pollard-smoke preflight FIRST, before anything is spent; then the engine preflight (prep/engines,
    pollard-preflight): every llama.cpp / MLX step the plan runs is resolved the way that step resolves it and loads
    THIS model on the CPU in seconds. A lane that cannot run is held there, with the fix (`pollard-runtime --update`,
    "converter for <arch> only in <tree>"); `critical: true` on a lane stops the whole release instead.
  * SERIAL, one heavy job at a time: each COMPUTE step (build / measure / verify) holds the machine lock (and
    POLLARD_HEAVY_LOCK, a shared lock directory, when set) for its own length; uploads never hold it. Lanes this
    machine cannot build (MLX needs Apple Silicon; GPTQ / MX / EXL3 need CUDA) are skipped with the command to run
    them where they can be built.
  * RESUMABLE: every step is recorded in <workspace>/models/<M>/release-state.json, and a rung the MANIFEST
    already shows built AND verified is never rebuilt -- so the Mac can do MLX and the box the CUDA lanes,
    each re-running the same plan.
  * every rung goes through `pollard-verify --gate` (gates.yaml); a rung that fails is not carded and not
    uploaded. The quality numbers (KLD, top-1, tool calls, retention, EXL3 end-to-end) are REPORT-ONLY by default:
    measured and carded with their pass / fail label, never blocking. What blocks: the complete model, the file
    loading, coherence, runtime. `strict_gates: true` in the plan (pollard-verify --strict) or `strict: true` in the
    gates file makes the quality thresholds block too. The cards are pollard-card, --verified-only, from the numbers
    this run measured.
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

import argparse, contextlib, copy, json, os, platform, re, shutil, subprocess, sys, time

TEMPLATE = """\
# pollard release plan -- every lane, each a ladder of rungs. `pollard release --hf <model> --plan this.yaml`
# prints the plan; add --run to build, --upload to publish (and set upload: true below).
org: PollardWeights            # the HF account the lane repos go to
base_model: null               # the HF repo id the card names as base_model (null = the --hf id / its config)
preset: text                   # text | vlm | decision | d3 | moe -- fills anything not set here (a d3
                               # decision model -- decision_config.json d3-code-readout-v1 -- picks d3 itself)
ngl: 0                         # GPU layers for llama.cpp steps. 0 = CPU (shared GPU boxes); auto = what fits
upload: false                  # publishing needs this AND --upload on the command line
private: false                 # false = the repo is created PUBLIC and the card goes up with the first rung;
                               # true = create (and keep) it private -- flip it yourself
gates: null                    # gates.yaml for pollard-verify --gate (null = the built-in defaults)
strict_gates: false            # true = pollard-verify --strict: KLD / top-1 / tool calls / retention BLOCK a rung
                               # (default: measured + carded with their pass / fail label, never blocking)
trust_remote_code: false       # true = consent to run the checkpoint's own .py files (only used where no native class fits)
hf_weights: null               # the checkpoint the HF lanes read (dir or repo id) when the --hf folder holds no weights
max_ram_gb: null               # cap the streamed HF lanes' RAM (null = 85% of what is free)
remote_source: null            # ssh://user@host/<dir>?python=..&key=..: the MLX lane reads the source through a sparse
                               # local mirror at --hf (tensors fetched as read, cached within the free disk)
reclaim_mirror: true           # once the release is checked, remove its $POLLARD_HOME/mirror/<model> source mirror if
                               # every build from it is confirmed on the Hub (size + sha256); false = keep it
calib:
  profile: auto                # pollard-calib --profile (auto = from pollard-modelkind)
  per_domain: 300
  self_generate: null          # an OpenAI-compatible URL serving the f16, for target-generated calibration
  reasoning: auto              # auto | on | off | toggle -- thinking traces in the model's own delimiters
  reasoning_share: null        # null = 0.35 of the corpus when the model reasons
measure:                       # extra numbers for the card + gate (GGUF rungs; served by llama-server)
  taskeval: {suite: quick, limit: 200}   # OPT-IN (--taskeval, or later --taskeval-only): ~1 h per rung of lm-eval,
                                         # so the release path does not run it; retention vs f16 per task
  mt: null                     # {limit: 100} -> chrF++ per language (translation models)
  toolcall: auto               # tool-call retention vs f16 (auto = on when pollard-modelkind finds a tool-call syntax)
  decision: false              # decision board vs f16 (decision models)
card_args: []                  # extra pollard-card flags for this model (e.g. [--params, 7.996B, --mmproj, mmproj-X.gguf])
lanes:                         # any lane may add `critical: true`: if the engine preflight finds it cannot run here,
                               # the whole release stops before calibration (default: only that lane is held)
  gguf: {enabled: true}        # the publish ladder + flagship, built the Pollard way (pollard --ladder)
  #   quality: [Q90, Q85]  quality-targeted rungs (pollard_qtarget): the SMALLEST measured-allocation build whose
  #                        measured KLD / top-1 vs the f16 meet the class (Q90: KLD <= 0.08, top-1 >= 90; Q85: KLD <=
  #                        0.2, top-1 >= 85), named <M>-Pollard-<class>-<size>GB.gguf; needs free-after-upload
  #   ladder: [Q6_K, flagship]  only these size-targeted labels (drop the ones the quality rungs replace)
  mlx:  {enabled: true, rungs: [2, 3, 4, 5, 6, 8], default: 4}      # default = the rung also at the repo root
  gptq: {enabled: true, rungs: [2, 3, 4, 8], default: 4}
  mx:   {enabled: true, rungs: [mxfp4, nvfp4, mix4.5, mix5, mix5.5, mxfp6_e2m3, mxfp6_e3m2, fp8], default: nvfp4}
  exl3: {enabled: true, rungs: [2.5, 3.0, 4.0, 5.0, 6.0], default: 4.0}
  #   recipe_choice: auto (default) = the default rung is converted smoothed / raw x Calib 3.0 / exllamav3's own
  #   calibration, each measured by KLD through exllamav3 on the held-out text; the winner is that rung and every
  #   other rung builds the same way. pollard | raw-calib | smooth-default | plain pin one (no A/B).
  # one encode at every width, each rung a manifest over it (GR8's format; dense decoder layers -- MoE experts not yet)
  exlr8: {enabled: true, rungs: [2.5, 3.0, 3.25, 4.0, 5.0, 6.0], default: 4.0, widths: [2, 3, 4, 5, 6]}
  draft: {enabled: false, drafter: auto, rungs: [Q4_K_M, Q8_0], targets: all, limit: 0, max_tokens: 128,
          prefer_own: false, train: {kind: eagle3, rung: Q4_K_M, taps: rung, samples: null, max_steps: null}}
                               # speculative drafter (pollard-draft): drafter = auto (search the Hub) or a repo id;
                               # measured against `targets` (all = every GGUF rung of this release, or a list of tags).
                               # Nothing redistributable published (or prefer_own) -> Pollard trains its own head on a
                               # CUDA box, quant-aware vs train.rung, measured vs the rungs AND f16
"""

PRESETS = {
    # a plain text model: every lane, every rung
    "text": {},
    # a vision-language / omni model: every lane ships the whole model, gated by pollard-complete -- the GGUF lane the
    # mmproj (+ source-precision sidecars for a talker); MLX through mlx-vlm; the HF lanes quantize the language model
    # and copy the towers / talker through; EXL3 converts when exllamav3 has the architecture (or its language model as
    # a plain causal LM, recomposed into the complete checkpoint) and is held with the reason otherwise
    "vlm": {"measure": {"taskeval": {"suite": "quick", "limit": 200}}},
    # a decision model answers on /v1/systemone from first-position logits: llama.cpp only, judged on decisions
    "decision": {"lanes": {"mlx": {"enabled": False}, "gptq": {"enabled": False}, "mx": {"enabled": False},
                           "exl3": {"enabled": False}, "exlr8": {"enabled": False}},
                 "measure": {"taskeval": None, "decision": True, "toolcall": False}},
    # a d3 decision model (vllm-sr/d3: decision_config.json `d3-code-readout-v1`): every lane with a d3 runtime
    # quantizes the BODY only and ships readout + decision config byte-exact; each rung is gated by pollard-d3 (its
    # readout distribution vs bf16 through its own runtime) instead of text KLD / perplexity, which do not apply to a
    # non-causal body read at one position. EXL3 / EXLR8 plan a hold with the reason (lane_steps): no runtime serves a
    # d3 rung in those formats.
    "d3": {"measure": {"taskeval": None, "decision": True, "toolcall": False},
           "calib": {"reasoning": "off"},             # the base template's <think> is never used: d3 never generates
           "d3": {"set": "typed-decisions", "limit": 0}},
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
    # the lanes whose rung order the PLAN wrote (`rungs:` / `ladder:`): built in that order; every other lane's rungs
    # (the template's / the preset's) are built and uploaded smallest first (order_rungs)
    cfg["_explicit_order"] = sorted(l for l, lc in ((user.get("lanes") or {}).items())
                                    if isinstance(lc, dict) and ("rungs" in lc or "ladder" in lc))
    return cfg


def auto_preset(cfg, src, preset=None, plan=None):
    """A d3 decision model picks the d3 preset itself when neither the command line nor the plan names one, and a
    decision-HEAD model (clef / clef-omni: joint_head_config.json) the decision preset."""
    if preset or cfg.get("preset") != "text" or not src or not os.path.isdir(str(src)):
        return cfg
    if plan and _yaml(open(plan, encoding="utf-8").read()).get("preset"):
        return cfg
    import pollard_d3 as D3
    if D3.decision_config(src) is not None:
        print(f"   {src}: a d3 decision model ({D3.FORMAT_ID}) -> preset d3")
        return load_plan(plan, "d3")
    if os.path.isfile(os.path.join(str(src), "joint_head_config.json")):
        print(f"   {src}: a decision-head model (joint_head_config.json) -> preset decision")
        return load_plan(plan, "decision")
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
    """id, lane, rung, kind (prep/build/measure/verify/gate/card/upload/expand/note), requires, cmd or fn, out.
    A `gate` is re-run on every resume (never skipped by a MANIFEST "verified"): the complete-model check."""
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
        "ngl": str(cfg.get("ngl", 0)), "gates": cfg.get("gates"), "strict_gates": cfg.get("strict_gates") is True,
        "upload_plan": cfg.get("upload") is True and not cfg.get("_rungs_only"),
        "imatrix": cfg.get("imatrix"), "free": bool(cfg.get("_free_after_upload")),
        "org": cfg.get("org", "PollardWeights"), "hf_repo": hf_repo,
        "trc": "on" if cfg.get("trust_remote_code") is True or str(cfg.get("trust_remote_code")).lower() in
               ("true", "on", "yes") else "auto",
        "max_ram_gb": cfg.get("max_ram_gb"),
        "toolcall_ref_rows": os.path.abspath(str(cfg["toolcall_ref_rows"])) if cfg.get("toolcall_ref_rows") else None,
    }
    # inputs that already exist (a big model's f16 / imatrix / profile / corpus): used as they are, never rebuilt
    for key, plan_key in (("f16", "f16"), ("sens_raw", "sensitivity"), ("calib", "calib_file"),
                          ("heldout", "heldout_file")):
        if cfg.get(plan_key):
            ctx[key] = os.path.abspath(str(cfg[plan_key]))
    ctx["given_f16"] = bool(cfg.get("f16"))
    ctx["gguf_lane"] = dict((cfg.get("lanes") or {}).get("gguf") or {})     # ladder: / quality: (gguf_ladder_labels)
    try:                                                 # a d3 decision model: gated on its readout, not on text
        import pollard_d3 as D3
        ctx["d3"] = D3.decision_config(hf_dir) is not None
    except ImportError:
        ctx["d3"] = False
    if cfg.get("measure_ref"):                           # the served measurements' reference (see measure_ref)
        ctx["measure_ref"] = os.path.abspath(str(cfg["measure_ref"]))
    ctx["decision_only"] = decision_only_reason(ctx)
    return ctx


def decision_only_reason(c):
    """Why this model is DECISION-ONLY (its graph ends in a decision head: no per-token logits), or None. Read off the
    HF folder (clef's joint_head_config.json, a d3 package), else the f16 GGUF when it exists. Such a model's GGUF
    rungs are gated on the decision board (pollard-decision vs the reference) and pollard-verify records KLD / PPL as
    not applicable -- llama-perplexity has no logits to read."""
    import pollard_decision as PD
    for p in (c.get("hf_dir"), c.get("f16")):
        why = PD.decision_only(p) if p else None
        if why:
            return why
    return None


def engine_arch(c):
    """POLLARD_ARCH for this run, from the f16 GGUF as soon as it exists: every binary lookup in this process and in
    every step it starts (rung builds, imatrix, quantize, pollard-decision, pollard-verify) then resolves through
    pollard_calc.find_llama_bin to the engine tree that carries what the architecture needs (engines.json + the
    patches' `archs`), not the stock tree by default."""
    from pollard_calc import model_arch, set_model_arch
    f16 = c.get("f16")
    if f16 and os.path.isfile(str(f16)) and not c.get("_arch_set"):
        if set_model_arch(f16):
            c["_arch_set"] = True
    return model_arch()


def results_path(c, lane):
    return os.path.join(c["mdir"], "reports", f"results-{lane}.json")


def lane_key(c, lane):
    """The MANIFEST key each lane's tool records its builds under (what pollard-card --builds-from reads)."""
    return {"gguf": c["f16"], "mlx": c["hf_dir"], "draft": c["src"]}.get(lane, c["smooth_dir"])


def _verify_gate(c):
    return ["--gate"] + ([c["gates"]] if c["gates"] else []) + (["--strict"] if c.get("strict_gates") else [])


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
    if cfg.get("remote_source"):                    # the source stays on another machine: a sparse local mirror
        if not os.path.isfile(os.path.join(c["hf_dir"], ".pollard-remote")):
            steps.append(Step(id="prep/mirror", lane="prep", kind="prep", requires="any", out=c["hf_dir"],
                              cmd=["pollard-mlx", "--model", c["hf_dir"], "--remote-source",
                                   str(cfg["remote_source"]), "--mirror-only"],
                              desc=f"mirror {cfg['remote_source']} -> {c['hf_dir']} (headers + small files)"))
    elif not os.path.isdir(c["src"]) or (c.get("hf_repo") and not os.path.isdir(c["hf_dir"])):
        what = c["src"] if not os.path.isdir(c["src"]) else c["hf_repo"]
        steps.append(Step(id="prep/fetch", lane="prep", kind="prep", requires="any", fn="fetch",
                          desc=f"pull {what} -> {c['hf_dir']}"))
    steps.append(Step(id="prep/smoke", lane="prep", kind="prep", requires="any",
                      cmd=["pollard-smoke", "--model", c["hf_dir"]]))
    steps.append(Step(id="prep/engines", lane="prep", kind="prep", requires="any", fn="engine_preflight",
                      lanes=list(lanes),
                      desc="engine preflight: every llama.cpp / MLX step this plan runs, resolved as the step resolves "
                           "it and loaded on THIS model on the CPU (seconds) -- a lane that cannot run stops here"))
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
    elif raw_probe and cfg.get("remote_source") and not os.path.exists(c["sens_raw"]):
        c["no_profile"] = True                       # the forward probe would read the mirror's holes: never run it
        steps.append(Step(id="prep/probe", lane="prep", kind="note", requires="any",
                          desc="no sensitivity profile (none in the plan, and the HF weights are a remote mirror the "
                               "forward probe cannot read): the MLX rungs allocate without one -- router, embeddings, "
                               "towers and projector held high, the body at the rung's bits"))
    elif raw_probe and route == "imatrix":
        print(f"   probe: from the imatrix on {c['f16']} -- {why}")
        probe = ["pollard-probe", "--from-imatrix", str(c["imatrix"]), "--gguf", c["f16"], "--out", c["sens_raw"]]
        tok = next((d for d in (c.get("hf_dir"), c.get("src")) if d and os.path.isfile(os.path.join(str(d),
                                                                                                    "tokenizer.json"))),
                   None)
        if tok:                                      # a lookup table (Gemma 4's per-layer embedding) is measured too
            probe += ["--text", c["calib"], "--tokenizer", str(tok)]
        steps.append(Step(id="prep/probe", lane="prep", kind="prep", requires="any", out=c["sens_raw"], cmd=probe))
    elif raw_probe:
        probe = ["pollard-probe", "--model", c["hf_dir"], "--eval", c["heldout"], "--out", c["sens_raw"]]
        if c["ngl"] == "0" and machine()["cuda"]:
            probe += ["--device", "cpu"]                 # a shared GPU is not ours (see pollard --ngl 0)
        steps.append(Step(id="prep/probe", lane="prep", kind="prep", requires="any", cmd=probe, out=c["sens_raw"]))
    held = [l for l in lanes if l in ("gptq", "mx", "exl3", "exlr8")]
    if smoothed and c.get("d3"):
        # a d3 body is not smoothed: SmoothQuant would fold scales measured on CAUSAL next-token activations into a
        # body that decides non-causally, and the smoothed copy is written as transformers' causal LM (other tensor
        # names, no tower) -- not a d3 package any more. The HF lanes quantize the source body as it is.
        c["smooth_dir"], c["sens_smooth"] = c["hf_dir"], c["sens_raw"]
        steps.append(Step(id="prep/smooth", lane="prep", kind="note", requires="any",
                          desc="d3 decision model: no SmoothQuant -- the HF lanes quantize the source body "
                               "(its own names, towers and readout) with the raw-weights profile"))
        smoothed = False
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
    if c.get("toolcall_ref_rows"):                  # the reference measured on another machine (labelled cross-backend)
        cmd += ["--ref-rows", c["toolcall_ref_rows"]]
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
        if m.get("taskeval") and (cfg.get("_taskeval") or cfg.get("_taskeval_only")):
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
        if c.get("d3"):                                    # the readout board vs bf16 IS this rung's gate
            steps.append(d3_gate_step(cfg, c, "gguf", tag, path, res))
            steps += _after_gate(c, "gguf", tag, path)
            continue
        if m.get("decision") or c.get("decision_only"):     # decision-only: the board IS the fidelity number
            steps.append(Step(id=f"{rid}/decision", lane="gguf", rung=tag, kind="measure", requires="any",
                              cmd=["pollard-decision", "--gguf", path, "--ref", ref, "--ngl", c["ngl"]]
                              + (["--source", c["hf_dir"]] if c.get("hf_dir") and os.path.isdir(str(c["hf_dir"])) else [])
                              + common,
                              ok_codes=(0, 1)))
        steps.append(Step(id=f"{rid}/verify", lane="gguf", rung=tag, kind="verify", requires="any", out=path,
                          cmd=["pollard-verify", "--lane", "gguf", "--model", path, "--ref", c["f16"],
                               "--eval", c["heldout"], "--ngl", c["ngl"], "--coherence"] + _verify_gate(c) + common))
        steps += _after_gate(c, "gguf", tag, path)
    return steps


def taskeval_only_steps(cfg, c):
    """`pollard release --taskeval-only`: the task-retention numbers for rungs that are ALREADY published, after the
    fact (overnight), then the card again. Each verified GGUF rung is measured from its local file, or -- freed after
    its upload -- from a temporary download of its published copy, deleted again once measured. Not part of the
    publish path: Pollard publishes on KLD + PPL + the tool-call gate + the complete-model gate."""
    t = dict((cfg.get("measure") or {}).get("taskeval") or {"suite": "quick", "limit": 200})
    ref = c.get("measure_ref") or c["f16"]
    repo = repo_name(c["org"], c["src"])
    steps = []
    for b in gguf_builds(c):
        if b.get("verified") is not True:
            continue
        tag = b.get("tag") or b.get("name")
        cmd = ["pollard-taskeval", "--model", b["path"], "--ref", ref, "--suite", str(t.get("suite", "quick")),
               "--limit", str(t.get("limit", 200)), "--ngl", c["ngl"],
               "--out", os.path.join(c["mdir"], "reports", "taskeval", tag), "--results", results_path(c, "gguf"),
               "--rung", tag]
        steps.append(Step(id=f"gguf/{tag}/taskeval", lane="gguf", rung=tag, kind="measure", requires="any",
                          fn="taskeval_rung", cmd=cmd, out=b["path"], repo=repo,
                          remote=b.get("uploaded_as") if b.get("uploaded") == repo else None,
                          desc=f"task retention of {tag} ({t.get('suite', 'quick')}, limit {t.get('limit', 200)})"))
    steps.append(Step(id="gguf/card", lane="gguf", kind="card", requires="any", cmd=card_cmd(cfg, c, ["gguf"]),
                      out=c["release_dir"], repo=repo))
    steps.append(Step(id="gguf/card-upload", lane="gguf", kind="upload", requires="any", fn="card_upload", repo=repo,
                      out=c["release_dir"], desc=f"the card with the retention numbers -> https://huggingface.co/{repo}"))
    return steps


def jev_index_only_steps(cfg, c, kit_python, lanes=("gguf", "mlx", "gptq", "mx"), limit=0):
    """`pollard release --jev-index-only KIT_PYTHON` (a d3 decision model): the Decision Index (apolinario/decision-
    index, the public part of the Jev Decision Index) for the bf16 source and every VERIFIED rung, each served on
    /v1/systemone by pollard-d3 and run through the kit's http engine, then the card again with the index and its
    delta vs bf16. Hours per rung on the full suite: an overnight step, never part of the publish path (a d3 rung
    publishes on its pollard-d3 gate + the complete-model gate)."""
    if not c.get("d3"):
        raise SystemExit("--jev-index-only is for a d3 decision model (decision_config.json d3-code-readout-v1)")
    repo = repo_name(c["org"], c["src"])
    jdir = os.path.join(c["mdir"], "reports", "jev-index")
    base = ["--source", c["hf_dir"], "--kit-python", kit_python] + (["--limit", str(limit)] if limit else [])
    steps = [Step(id="d3/bf16/jev-index", lane="gguf", rung="bf16", kind="measure", requires="any",
                  cmd=["pollard-d3", "jev-index", "--rung", c["hf_dir"]] + base +
                      ["--out", os.path.join(jdir, "bf16"), "--results", results_path(c, "gguf"), "--rung-name", "bf16"],
                  desc="the Decision Index of the bf16 source (the delta every rung is shown against)")]
    for lane in lanes:
        builds = gguf_builds(c) if lane == "gguf" else _lane_builds(c, lane, repo)
        for b in builds:
            if b.get("verified") is not True or not os.path.exists(str(b.get("path"))):
                continue
            tag = b.get("tag") or b.get("name")
            steps.append(Step(id=f"{lane}/{tag}/jev-index", lane=lane, rung=tag, kind="measure",
                              requires=REQUIRES[lane] if lane == "mlx" else "any",
                              cmd=["pollard-d3", "jev-index", "--rung", b["path"]] + base +
                                  ["--out", os.path.join(jdir, f"{lane}-{tag}"), "--results", results_path(c, lane),
                                   "--rung-name", tag],
                              desc=f"Decision Index of {lane} {tag} through /v1/systemone"))
    steps.append(Step(id="gguf/card", lane="gguf", kind="card", requires="any", cmd=card_cmd(cfg, c, list(lanes)),
                      out=c["release_dir"], repo=repo))
    steps.append(Step(id="gguf/card-upload", lane="gguf", kind="upload", requires="any", fn="card_upload", repo=repo,
                      out=c["release_dir"], desc=f"the card with the Decision Index -> https://huggingface.co/{repo}"))
    return steps


def taskeval_rung(step, c, run_upload, out=print):
    """Measure one published rung: its local file, else a temporary download of its published copy (deleted after)."""
    cmd = list(step.cmd)
    if os.path.exists(str(step.out)):
        run_cmd(cmd)
        return "done"
    if not step.remote:
        raise RuntimeError(f"{step.out} is neither on disk nor recorded as published")
    from huggingface_hub import hf_hub_download
    tmp = os.path.join(c["mdir"], "taskeval-download")
    os.makedirs(tmp, exist_ok=True)
    out(f"      fetching the published {step.remote} for the measurement (deleted afterwards)")
    local = hf_hub_download(step.repo, step.remote, local_dir=tmp)
    try:
        cmd[cmd.index("--model") + 1] = local
        run_cmd(cmd)
    finally:
        if os.path.exists(local):
            os.remove(local)
    return "done"


def card_upload(step, c, run_upload, out=print):
    card = os.path.join(step.out, "README.md")
    if not run_upload:
        out("      (not uploading the card: uploads are off)")
        return "held"
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=step.repo)
    return "done"


def _rung_card_step(c, lane, tag, path):
    """Without --free-after-upload the rungs upload with their lane, but the card does not wait for that: right after
    this rung is measured and passed, the card is rebuilt from the rungs passed so far and uploaded."""
    repo = repo_name(c["org"], c["src"])
    return Step(id=f"{lane}/{tag}/card", lane=lane, rung=tag, kind="upload", requires="any", fn="rung_card",
                out=path, repo=repo, desc=f"card with this rung's numbers -> https://huggingface.co/{repo} (the rung "
                                          "follows with its lane's upload)")


def _after_gate(c, lane, tag, path):
    """What follows a rung's gate: --free-after-upload = publish it (card first, then the rung); else, when the plan
    uploads, the card refresh."""
    if c.get("free"):
        return [_publish_rung_step(c, lane, tag, path)]
    return [_rung_card_step(c, lane, tag, path)] if c.get("upload_plan") else []


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
    rungs = ",".join(str(r) for r in order_rungs(cfg, "draft", lc.get("rungs") or ["Q4_K_M", "Q8_0"]))
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


def _has_weights(d):
    """A local HF checkpoint folder (config + safetensors): the reference the complete-model gate reads."""
    try:
        return bool(d) and os.path.isfile(os.path.join(str(d), "config.json")) and any(
            f.endswith(".safetensors") for f in os.listdir(str(d)))
    except OSError:
        return False


def d3_gate_step(cfg, c, lane, tag, path, res):
    """A d3 rung's verify step: pollard-d3 gate -- the rung through its own runtime (llama-server embeddings / MLX /
    the d3 runtime on the rung's decoded weights), read out with the source's readout, against the bf16 reference:
    option KL, top-choice agreement, calibration. Text KLD / perplexity / coherence do not apply to a non-causal body
    read at its last token. Exit 0 = PASS; WEAK / FAIL block the rung like any failed gate."""
    d = cfg.get("d3") or {}
    cmd = ["pollard-d3", "gate", "--rung", path, "--source", c["hf_dir"], "--set", str(d.get("set", "typed-decisions")),
           "--limit", str(int(d.get("limit") or 0)), "--results", res, "--rung-name", tag]
    if lane == "gguf":
        cmd += ["--ngl", c["ngl"]]
    if d.get("device"):
        cmd += ["--device", str(d["device"])]
    elif c.get("ngl") == "0" and machine()["cuda"]:
        cmd += ["--device", "cpu"]                       # a shared GPU is not ours (see pollard --ngl 0)
    return Step(id=f"{lane}/{tag}/verify", lane=lane, rung=tag, kind="verify", out=path,
                requires=REQUIRES[lane] if lane == "mlx" else "any", cmd=cmd,
                mem="mlx-verify" if lane == "mlx" else None,
                desc="d3 readout board vs bf16 (agreement / option KL / ECE) through the rung's own runtime")


def lane_steps(cfg, c, lane):
    lc = (cfg.get("lanes") or {}).get(lane) or {}
    import pollard_workspace as ws
    if lane == "draft":
        return draft_steps(cfg, c, lc)
    if lane == "gguf":
        src = ["--gguf", c["f16"]] if c.get("given_f16") else ["--hf", c["hf_dir"], "--output", c["gguf_dir"]]
        cmd = ["pollard"] + src + ["--calib", c["calib"], "--heldout", c["heldout"],
               "--sensitivity", c["sens_raw"], "--ladder", "--run", "--no-card", "--force",
               "--no-toolcall",                          # release measures every rung itself (gguf/rungs)
               "--ngl", "auto" if c["ngl"] == "auto" else c["ngl"],
               "--calib-profile", str((cfg.get("calib") or {}).get("profile", "auto"))]
        if c.get("imatrix"):
            cmd += ["--imatrix", os.path.abspath(str(c["imatrix"]))]
        complete = []
        if _has_weights(c["hf_dir"]):                    # the GGUF repo carries the whole model: mmproj + sidecars
            complete = [Step(id="gguf/complete", lane="gguf", kind="gate", requires="any", out=c["gguf_dir"],
                             blocks=["gguf"],
                             cmd=["pollard-complete", "--source", c["hf_dir"], "--out", c["gguf_dir"], "--lane",
                                  "gguf", "--fill"],
                             desc="every vision / audio tower in an mmproj, speech output and side weights as "
                                  "source-precision sidecars -- never a text-only GGUF repo")]
        if not c.get("free"):
            return [Step(id="gguf/ladder", lane="gguf", kind="build", requires="any", cmd=cmd, out=c["f16"],
                         desc="convert -> imatrix -> measured publish ladder (+ IQ*_KT flagship where ik_llama builds)"),
                    Step(id="gguf/rungs", lane="gguf", kind="expand", requires="any")] + complete
        # one rung at a time: build -> measure + gate -> upload -> free, so the peak disk is ONE rung, not the ladder
        steps = []
        for label, gb in gguf_ladder_labels(c):
            steps.append(Step(id=f"gguf/ladder-{label}", lane="gguf", rung=label, kind="build", requires="any",
                              cmd=cmd + ["--rung", label], out=None, disk_gb=gb,
                              desc=f"the {label} rung of the measured publish ladder"))
            if complete and len(steps) == 1:             # the first build wrote the mmproj: check before any upload
                steps += complete
            steps.append(Step(id=f"gguf/rungs-{label}", lane="gguf", kind="expand", requires="any"))
        return steps
    steps, res = [], results_path(c, lane)
    if c.get("d3") and lane in ("exl3", "exlr8"):
        import pollard_d3 as D3
        return [Step(id=f"{lane}/hold", lane=lane, kind="hold", requires="any", blocks=[lane],
                     desc=f"{lane} held for this d3 decision model: {D3.HOLD_REASON[lane]}")]
    tool = toolcall_wanted(cfg, c) and not c.get("d3")
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
            steps.append(Step(id="exl3/recompose", lane="exl3", kind="note", requires="any",
                              desc=f"EXL3 shipped complete: {ep['why']}"))
    if lane == "exlr8":
        # the ONE encode every rung of this lane is a manifest over (resumable per layer inside pollard-exlr8)
        widths = ",".join(str(int(k)) for k in (lc.get("widths") or [2, 3, 4, 5, 6]))
        steps.append(Step(id="exlr8/encode", lane="exlr8", kind="build", requires=REQUIRES["exlr8"],
                          cmd=["pollard-exlr8", "--model", c["smooth_dir"], "--store", c["exlr8_store"], "--calib",
                               c["calib"], "--widths", widths, "--encode-only"] + hf_flags(c), out=c["exlr8_store"],
                          desc="encode every matrix once at each width (LDLQ + tail-biting Viterbi, E4M3 codebook)"))
    rungs = order_rungs(cfg, lane, lc.get("rungs") or [])
    ab_rung = None
    if lane == "exl3" and rungs and str(lc.get("recipe_choice", "auto")) == "auto":
        # the recipe A/B (pollard_exl3.run_ab) runs on ONE rung -- the default, else the first -- and its winner is
        # the recipe every later rung builds with, so that rung goes first
        ab_rung = lc.get("ab_bpw", lc.get("default"))
        ab_rung = next((r for r in rungs if float(r) == float(ab_rung)), rungs[0]) if ab_rung is not None else rungs[0]
        rungs = [ab_rung] + [r for r in rungs if r is not ab_rung]
        c.setdefault("exl3_ab", os.path.join(c["mdir"], "reports", "exl3-recipe-ab.json"))
    for rung in rungs:
        tag = rung_tag(lane, rung)
        out = ws.resolve_out(c["src"], lane, tag, create_parent=False)
        name = os.path.basename(out)
        if lane == "mlx":
            cmd = ["pollard-mlx", "--model", c["hf_dir"]] + ([] if c.get("no_profile") else
                                                              ["--sensitivity", c["sens_raw"]]) + [
                   "--bits", str(int(rung)), "--out", out]
            if c.get("imatrix"):                         # the range search weighted by activations, as GGUF's
                cmd += ["--imatrix", os.path.abspath(str(c["imatrix"]))]
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
            choice = str(lc.get("recipe_choice", "auto"))
            if choice != "pollard":                      # smoothing / Calib 3.0 only where they measure better
                cmd += ["--raw-model", c["hf_dir"], "--recipe-choice", choice]
                if choice == "auto":
                    cmd += ["--ab-record", c["exl3_ab"]] + (["--ab-eval", c["heldout"]] if rung is ab_rung else [])
            src = c["smooth_dir"]
        rid = f"{lane}/{tag}"
        st_build = Step(id=f"{rid}/build", lane=lane, rung=tag, kind="build", requires=REQUIRES[lane], cmd=cmd,
                        out=out, src=src, disk_gb=_rung_gb(lane, rung, c["hf_dir"]) if c.get("free") else 0,
                        held_codes=(EXIT_HELD,) if lane in ("exl3", "mlx") else (),
                        mem="mlx-build" if lane == "mlx" else None, bits=int(rung) if lane == "mlx" else None)
        if c.get("free") and lane in STREAMABLE:
            # bigger than the free disk: written + uploaded + confirmed + freed one shard at a time, not held
            from pollard_card import rung_subfolder
            st_build["stream_to"] = (repo_name(c["org"], c["src"]),
                                     f"{lane_folder(lane)}/{rung_subfolder({'name': name, 'tag': tag}, lane)}")
        steps.append(st_build)
        if _has_weights(c["hf_dir"]):                    # never a text-only rung: the gate lists what is missing
            # a d3 rung: --fill puts the decision files byte-exact + the rung's own MODEL_MANIFEST.json in first
            steps.append(Step(id=f"{rid}/complete", lane=lane, rung=tag, kind="gate", requires="any", out=out,
                              cmd=["pollard-complete", "--source", c["hf_dir"], "--out", out, "--lane", lane]
                              + (["--fill"] if c.get("d3") else []),
                              desc="the rung carries every tower / projector / talker / side weight and processor "
                                   "file of the source" + (" + the d3 readout and decision config, byte-exact"
                                                           if c.get("d3") else "")))
        if c.get("d3"):
            steps.append(d3_gate_step(cfg, c, lane, tag, out, res))
            if c.get("free"):
                st = _publish_rung_step(c, lane, tag, out)
                st["default"] = lc.get("default") is not None and rung_tag(lane, lc["default"]) == tag
                steps.append(st)
            else:
                steps += _after_gate(c, lane, tag, out)
            continue
        if tool:
            steps.append(Step(id=f"{rid}/toolcall", lane=lane, rung=tag, kind="measure", requires=REQUIRES[lane],
                              cmd=toolcall_cmd(c, lane, out, name, res), ok_codes=(0, 1), rung_dir=out, results=res,
                              rung_name=name, mem="mlx-serve" if lane == "mlx" else None))
        # gated against the ORIGINAL weights, not the smoothed copy: fidelity to the model people asked for
        ver = ["pollard-verify", "--lane", lane, "--model", out, "--source", c["hf_dir"], "--rung", name,
               "--results", res] + _verify_gate(c)
        if lane in ("gptq", "mx", "exlr8"):
            ver += ["--trust-remote-code", c.get("trc", "auto")]
            if c.get("ngl") == "0" and machine()["cuda"]:
                ver += ["--device", "cpu"]
            if c.get("max_ram_gb"):                      # the streamed verify plans both passes under the plan's cap
                ver += ["--max-ram-gb", str(c["max_ram_gb"])]
        ver += (["--end-to-end"] if lane == "exl3" else []) + ["--eval", c["heldout"]]
        steps.append(Step(id=f"{rid}/verify", lane=lane, rung=tag, kind="verify", requires=REQUIRES[lane],
                          cmd=ver, out=out, mem="mlx-verify" if lane == "mlx" else None,
                          # disk / RAM short (MLX; the HF lanes' reference logits): held, rung kept
                          held_codes=(EXIT_HELD,) if lane in ("mlx", "gptq", "mx", "exlr8") else ()))
        if c.get("free"):
            st = _publish_rung_step(c, lane, tag, out)
            st["default"] = lc.get("default") is not None and rung_tag(lane, lc["default"]) == tag
            steps.append(st)
        else:
            steps += _after_gate(c, lane, tag, out)
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
    from pollard_auto import LADDER, MOE_BUDGET_LABEL, MOE_TAG, QUALITY_RANGE
    labels = [(label, frac * f16_gb * 1.05) for label, _, frac in list(LADDER) + [("flagship", (), 0.20)]]
    if _is_moe_gguf(c["f16"]):                     # the measured prune x bit-mix rung, at the 2-bit rung's bytes
        labels.append((MOE_TAG, next(f for lb, _, f in LADDER if lb == MOE_BUDGET_LABEL) * f16_gb * 1.05))
    gl = c.get("gguf_lane") or {}
    norm = lambda x: str(x).upper() if str(x).lower() != "flagship" else "flagship"   # noqa: E731
    if gl.get("ladder") is not None:               # `lanes.gguf.ladder:` -- only these labels, in the plan's order
        keep = [norm(x) for x in gl["ladder"]]
        labels = sorted([(lb, gb) for lb, gb in labels if norm(lb) in keep], key=lambda x: keep.index(norm(x[0])))
    else:                                          # smallest first: flagship / 2-bit, then up to Q6_K
        labels.sort(key=lambda x: (x[1], x[0] != "flagship"))
    import pollard_qtarget as QT
    for q in gl.get("quality") or []:              # `lanes.gguf.quality:` -- quality-targeted rungs (pollard_qtarget)
        cls = QT.parse_class(q)
        if not cls:
            raise SystemExit(f"lanes.gguf.quality: {q!r} -- choose from {sorted(QT.QUALITY_CLASSES)}")
        labels.append((cls, QUALITY_RANGE[1] * f16_gb * 1.05))    # the candidate at the top of its search range
    return labels


def _is_moe_gguf(path):
    """Does this GGUF route experts (expert_count > 1)? False when unreadable."""
    try:
        from pollard_calc import read_gguf_meta
        meta = read_gguf_meta(path)
        return any(k.endswith(".expert_count") and int(v or 0) > 1 for k, v in meta.items())
    except Exception:                                              # noqa: BLE001
        return False


def _draft_on(cfg):
    return bool(((cfg.get("lanes") or {}).get("draft") or {}).get("enabled"))


_HF_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def _is_hf_id(x):
    """An HF repo id (org/name), not a local path that happens to contain one slash."""
    x = str(x or "")
    return bool(_HF_ID.match(x)) and not os.path.exists(x)


def base_model_id(cfg, c):
    """The HF repo id the card names as base_model (its frontmatter, links, licence lookup): the plan's `base_model:`,
    the --hf argument when it is a repo id, `hf_weights:` when that is one, the source config's `_name_or_path`, a
    workspace folder named <Org>__<Model>. None when nothing says -- a local path is never a base_model (the LFM2.5
    card went out with base_model: C:\\pollard\\src\\LFM2.5-24B-A2B, 2026-10-17)."""
    if _is_hf_id(cfg.get("base_model")):
        return str(cfg["base_model"])
    if _is_hf_id(c.get("src")):
        return str(c["src"])
    if _is_hf_id(c.get("hf_repo")):
        return str(c["hf_repo"])
    for d in (c.get("hf_dir"), c.get("src")):
        try:
            with open(os.path.join(str(d), "config.json"), encoding="utf-8") as f:
                nop = json.load(f).get("_name_or_path")
            if _is_hf_id(nop):
                return str(nop)
        except (OSError, ValueError, AttributeError):
            pass
    for d in (c.get("hf_dir"), c.get("src")):            # a d3 package names its own repo in MODEL_MANIFEST.json
        try:
            import pollard_d3 as D3
            rid = (D3.info(str(d)) or {}).get("repo_id")
            if _is_hf_id(rid):
                return str(rid)
        except Exception:                                    # noqa: BLE001
            pass
    for d in (c.get("hf_dir"), c.get("src")):
        base = os.path.basename(str(d or "").rstrip("/\\"))
        if base.count("__") == 1 and _is_hf_id(base.replace("__", "/")):
            return base.replace("__", "/")
    return None


def source_params(d):
    """pollard_card.source_params (parameters from the safetensors headers)."""
    from pollard_card import source_params as sp
    return sp(d)


def card_cmd(cfg, c, lanes):
    """The ONE card for the one repo: the primary lane (GGUF when it is in the run) in full, every other lane as its
    own folder section with its measured numbers and download + serve commands, the drafter in DRAFT/. Every lane's
    card step regenerates it, so the README always describes what has landed so far."""
    repo = repo_name(cfg.get("org", "PollardWeights"), c["src"])
    body = [l for l in lanes if l != "draft"]
    primary = "gguf" if "gguf" in body else (body[0] if body else "gguf")
    base = base_model_id(cfg, c)
    card = ["pollard-card", "--model", c["src"] if not os.path.isdir(c["src"]) else c["hf_dir"],
            "--base-model", base or c["src"], "--lane", primary, "--builds-from", lane_key(c, primary),
            "--results", results_path(c, primary), "--repo", repo, "--out", os.path.join(c["release_dir"], "README.md"),
            "--verified-only"]
    for other in body:
        if other != primary:
            card += ["--also", other, lane_key(c, other), results_path(c, other)]
    if "draft" in lanes or _draft_on(cfg):
        card += ["--also", "draft", c["src"], results_path(c, "draft"),
                 "--draft-results", results_path(c, "draft"), "--draft-repo", repo]
    params = source_params(c.get("hf_dir")) or source_params(c.get("src"))
    if params:
        card += ["--params", f"{params / 1e9:.3f}B"]
    if primary == "gguf" and c.get("imatrix"):
        # the release uploads it as <name>.imatrix (upload()); the card then says imatrix: yes and links it
        card += ["--imatrix-file", f"{c['name']}.imatrix"]
    if primary == "gguf":
        card += ["--calib-file", os.path.basename(c["calib"]),
                 "--calib-note", f"the pollard-calib '{(cfg.get('calib') or {}).get('profile', 'auto')}' profile "
                                 "(sources, revisions and licences in the shipped .sources.json)"]
    if c.get("free"):
        card += ["--runtime-from-repo"]                  # freed rungs are read from their published copy
    card += [str(x) for x in (cfg.get("card_args") or [])]   # e.g. --params / --mmproj a config-only mirror cannot give
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
                                        "(the repo is public from its first upload unless the plan says private)")


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
        if not cfg.get("_rungs_only"):
            steps += publish_steps(cfg, c, lane, card_lanes)
    if lanes and not cfg.get("_rungs_only"):
        steps.append(public_step(cfg, c, lanes))
    return c, lanes, steps


# ------------------------------------------------------------------------------- state / resume

def load_state(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}


import threading as _threading
_STATE_LOCK = _threading.RLock()


def save_entry(path, sid, entry):
    """Write ONE step's state under a lock, merged into what is on disk: several executes (the parallel scheduler's
    build slots + measure queue) share one release-state.json, and none may overwrite another's steps."""
    with _STATE_LOCK:
        disk = load_state(path)
        disk[sid] = entry
        try:
            save_state(path, disk)
        except OSError as e:                      # a full disk: the old state stays whole, this step re-runs next time
            print(f"      WARNING: {e} -- {sid} is not recorded and runs again on the next release run",
                  file=sys.stderr, flush=True)
            return False
    return True


def save_state(path, state):
    """tmp + fsync + rename (pollard_workspace.write_json_atomic): ENOSPC leaves the previous state intact."""
    import pollard_workspace as ws
    ws.write_json_atomic(path, state, indent=1)


def _manifest_entry(path):
    import pollard_workspace as ws
    return ws.find_build(path)[1] if path else None


def stale_imatrix_profile(path):
    """True when `path` is a `pollard-probe --from-imatrix` profile made before MoE experts were weighted by routing
    frequency (no `"experts"` key): on an MoE it costs every expert as if every token reached it, and attention lands
    far too low (clef-omni IQ4_XS, 2026-10-22). Re-making it takes minutes and no forward pass."""
    try:
        with open(path, encoding="utf-8") as f:
            prof = json.load(f)
    except (OSError, ValueError, TypeError):
        return False
    return isinstance(prof, dict) and prof.get("method") == "imatrix-hessian" and not prof.get("experts")


def resume_reason(step, state, redo=()):
    """Why this step can be skipped on a re-run, or None to run it."""
    if any(step.id == r or step.id.startswith(r.rstrip("/") + "/") for r in redo):
        return None
    if step.fn == "engine_preflight":
        return None                                  # the engines may have changed since: re-checked (cached passes)
    if step.id == "prep/probe" and "--from-imatrix" in (step.get("cmd") or []) and stale_imatrix_profile(step.out):
        return None                                  # a pre-fix imatrix profile: re-probe (cheap) instead of reusing it
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


class PreflightFailed(RuntimeError):
    """The engine preflight found a step the release cannot run -- raised before any long step."""


def engine_preflight(st, c, cfg, have=None, out=print):
    """prep/engines: prove every engine step this plan runs works on THIS model, before calibration, the imatrix or
    any build (pollard_preflight). Returns the lanes to hold: a lane whose critical step cannot run is held here and
    the other lanes carry on. Raises PreflightFailed -- the release stops, nothing long has started -- when a lane
    marked `critical: true` in the plan is among them, when every lane this machine runs is, or when the imatrix the
    sensitivity probe needs (probe route imatrix) cannot be computed."""
    import pollard_preflight as PF
    if PF.disabled():
        out("      engine preflight: off (POLLARD_ENGINE_PREFLIGHT=0)")
        return set()
    have = have or machine()
    lanes = [l for l in (st.lanes or []) if have.get(REQUIRES.get(l, "any"), False)]
    gguf = str(c["f16"]) if c.get("f16") and os.path.isfile(str(c["f16"])) else None
    hf = c["hf_dir"] if os.path.isfile(os.path.join(str(c.get("hf_dir") or ""), "config.json")) else None
    ik_dir = None
    if "gguf" in lanes:
        from pollard_auto import find_ik_build
        ik_dir = find_ik_build()
    have_imatrix = bool(c.get("imatrix")) and os.path.exists(str(c["imatrix"]))
    res = PF.run(lanes, gguf=gguf, hf_dir=hf, mmproj=PF.find_mmproj(gguf) if gguf else None,
                 toolcall=toolcall_wanted(cfg, c) and not c.get("d3"), flagship=bool(ik_dir),
                 imatrix=not have_imatrix, ik_dir=ik_dir, out=out,
                 convert=not c.get("d3"),                # a d3 body converts through pollard-d3's own view
                 mlx_shard=not cfg.get("remote_source"))  # a sparse mirror's shards are holes until read
    held = set(res["failed_lanes"])
    crit = {l for l in lanes if ((cfg.get("lanes") or {}).get(l) or {}).get("critical")}
    imx = [r for r in res["rows"] if r["step"] == "imatrix" and not r["ok"]]
    why = []
    if held & crit:
        why.append(f"critical lane(s) {', '.join(sorted(held & crit))} cannot run")
    if lanes and held >= set(lanes):
        why.append("no lane this machine runs can")
    if imx and c.get("probe_route") == "imatrix" and not have_imatrix:
        why.append("the sensitivity probe needs an imatrix and llama-imatrix cannot load this model")
    if why:
        raise PreflightFailed("engine preflight: " + "; ".join(why) + " -- fix shown above, nothing long has started")
    for lane in sorted(held):
        out(f"      HELD {lane}: an engine step it needs cannot run here (above); the other lanes carry on")
    return held


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
    with ws.manifest_transaction(key) as man:
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
        out |= {f for f in os.listdir(c["gguf_dir"]) if f.startswith(("mmproj-", "components-"))}
    return out


def _build_bytes(b):
    """On-disk bytes of a build (file or folder); a freed one (already on the Hub) sorts first -- nothing to send."""
    p = b.get("path") or ""
    if os.path.isfile(p):
        return os.path.getsize(p)
    if os.path.isdir(p):
        return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
    return 0


def _lane_builds(c, lane, repo):
    import pollard_workspace as ws
    return [b for b in ws.read_manifest(lane_key(c, lane)).get("builds", [])
            if b.get("lane") == lane and b.get("verified") is True
            and (os.path.exists(b.get("path", "")) or b.get("uploaded") == repo)]


def repo_private(cfg):
    """The plan's `private:` -- False (the default) = the repo is public from its first upload."""
    return cfg.get("private") is True


def open_repo(api, repo, cfg, out=print):
    """Create the ONE repo the way the plan says: PUBLIC by default, private only with `private: true`. A repo an
    older Pollard created private (it flipped repos public only at the very end) is made public here, at its first
    upload of this run, unless the plan asks for private."""
    private = repo_private(cfg)
    api.create_repo(repo, repo_type="model", exist_ok=True, private=private)
    if private:
        return True
    try:
        api.update_repo_settings(repo_id=repo, private=False)
    except AttributeError:                                       # older huggingface_hub
        try:
            api.update_repo_visibility(repo, private=False)
        except Exception as e:                                   # noqa: BLE001
            out(f"      ({repo}: could not confirm it is public: {e})")
    except Exception as e:                                       # noqa: BLE001
        out(f"      ({repo}: could not confirm it is public: {e})")
    return False


def release_card_cmd(cfg, c):
    """The ONE card over every lane the plan enables (what each lane's card step writes)."""
    return card_cmd(cfg, c, [l for l in LANES if ((cfg.get("lanes") or {}).get(l) or {}).get("enabled")])


#: where each machine's measured numbers meet in the ONE repo: pollard/results-<lane>.<host>.json =
#: {"lane", "host", "results": <reports/results-<lane>.json>, "builds": [the lane's MANIFEST builds]}
SIDECAR_DIR = "pollard"
SIDECAR_RE = re.compile(r"^pollard/results-([a-z0-9]+)\.([A-Za-z0-9_.-]+)\.json$")
#: the repo folder of each non-root lane (the GGUF lane lives at the root)
_LANE_DIR = {"mlx": "MLX", "gptq": "GPTQ", "mx": "MX", "exl3": "EXL3", "exlr8": "EXLR8", "draft": "DRAFT"}


def host_name(cfg=None):
    """This machine's name in its sidecars: the plan's `host:`, else the short hostname."""
    h = (cfg or {}).get("host") or platform.node().split(".")[0] or "host"
    return re.sub(r"[^A-Za-z0-9_-]", "-", str(h)).lower()


def lane_sidecar(c, lane, host):
    """This machine's sidecar document for one lane: its results and its MANIFEST builds of that lane."""
    import pollard_workspace as ws
    try:
        res = json.load(open(results_path(c, lane), encoding="utf-8"))
    except (OSError, ValueError):
        res = {}
    builds = [b for b in ws.read_manifest(lane_key(c, lane)).get("builds", []) if b.get("lane") == lane]
    return {"lane": lane, "host": host, "results": res, "builds": builds}


def push_sidecar(api, repo, c, cfg, lane, out=print):
    """Best-effort: this machine's numbers for `lane` as pollard/results-<lane>.<host>.json in the repo, so any machine
    that writes the card next keeps them. Never raises."""
    host = host_name(cfg)
    try:
        doc = lane_sidecar(c, lane, host)
        if not doc["results"] and not doc["builds"]:
            return False
        d = os.path.join(c["mdir"], "reports")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"sidecar-{lane}.{host}.json")
        json.dump(doc, open(p, "w", encoding="utf-8"), indent=1)
        api.upload_file(path_or_fileobj=p, path_in_repo=f"{SIDECAR_DIR}/results-{lane}.{host}.json", repo_id=repo)
        out(f"      sidecar: {SIDECAR_DIR}/results-{lane}.{host}.json ({len(doc['results'])} result(s), "
            f"{len(doc['builds'])} build(s))")
        return True
    except Exception as e:                                       # noqa: BLE001 -- never holds a rung back
        out(f"      sidecar for {lane} not uploaded ({type(e).__name__}: {e}) -- the next one carries it")
        return False


def _on_hub(b, lane, files):
    """Is this build's file / folder in the repo listing?"""
    from pollard_card import rung_subfolder
    n = b.get("uploaded_as") or b.get("name") or os.path.basename(str(b.get("path", "")))
    if lane == "gguf":
        return n in files
    if lane == "draft":
        return any(f.startswith("DRAFT/") and (n in f or (b.get("tag") or "~") in f) for f in files)
    sub = b.get("uploaded_as") or f"{_LANE_DIR.get(lane, lane.upper())}/{rung_subfolder(b, lane)}"
    return any(f.startswith(sub.rstrip("/") + "/") for f in files)


def hub_sidecars(api, repo, files, out=print):
    """{lane: {"results": {...}, "builds": [...]}} merged over every pollard/results-*.json in the repo. An
    unreadable sidecar is skipped with a warning (its rungs show as numbers pending)."""
    merged = {}
    for f in sorted(x for x in files if SIDECAR_RE.match(x)):
        try:
            p = api.hf_hub_download(repo_id=repo, filename=f) if hasattr(api, "hf_hub_download") else None
            if p is None:
                from huggingface_hub import hf_hub_download
                p = hf_hub_download(repo_id=repo, filename=f)
            d = json.load(open(p, encoding="utf-8"))
        except Exception as e:                                   # noqa: BLE001
            out(f"      (sidecar {f} unreadable: {type(e).__name__}: {e})")
            continue
        lane = d.get("lane") or SIDECAR_RE.match(f).group(1)
        m = merged.setdefault(lane, {"results": {}, "builds": []})
        m["results"].update(d.get("results") or {})
        m["builds"] += [b for b in d.get("builds") or [] if isinstance(b, dict)]
    return merged


def pending_rows(card_text, files):
    """Rungs in the repo that the card does not name (no numbers anywhere yet): the 'numbers pending' section."""
    rungs = sorted({"/".join(f.split("/")[:2]) + "/" for f in files
                    if f.split("/")[0] in set(_LANE_DIR.values()) - {"DRAFT"} and f.count("/") >= 2}
                   | {f for f in files if f.endswith(".gguf") and "/" not in f and not f.startswith("mmproj")})
    missing = [r for r in rungs if r not in card_text]
    if not missing:
        return card_text, []
    rows = ["", "## Also on the Hub (numbers pending)", "",
            "These rungs are published; their measured numbers land in this card once the machine that built them "
            "has put its results sidecar in the repo.", "", "| File / folder | Numbers |", "|---|---|"]
    rows += [f"| `{m}` | numbers pending |" for m in missing]
    cut = card_text.find("\n## Credits")
    text = (card_text[:cut] + "\n".join(rows) + "\n" + card_text[cut:]) if cut >= 0 else \
        card_text + "\n".join(rows) + "\n"
    return text, missing


@contextlib.contextmanager
def merged_lane_files(c, cfg, side, files):
    """For the length of one card build: this machine's results + MANIFESTs overlaid with every other machine's
    sidecar (results merged, builds added -- the other machines' only where the Hub holds them). Restored after."""
    import pollard_workspace as ws
    saved = {}

    def keep(p):
        if p not in saved:
            saved[p] = open(p, "rb").read() if os.path.exists(p) else None

    try:
        for lane, s_ in side.items():
            p = results_path(c, lane)
            keep(p)
            try:
                mine = json.load(open(p, encoding="utf-8"))
            except (OSError, ValueError):
                mine = {}
            os.makedirs(os.path.dirname(p), exist_ok=True)
            json.dump(dict(s_["results"], **mine), open(p, "w", encoding="utf-8"), indent=2)   # this machine wins
            key = lane_key(c, lane)
            mp = os.path.join(ws.model_dir(key, create=True), "MANIFEST.json")
            keep(mp)
            man = ws.read_manifest(key) or {}
            have = {b.get("name") for b in man.get("builds", [])}
            add = [b for b in s_["builds"] if b.get("lane") == lane and b.get("name") not in have
                   and _on_hub(b, lane, files)]
            if add:
                man = dict(man, model=man.get("model") or key, builds=list(man.get("builds", [])) + add)
                json.dump(man, open(mp, "w", encoding="utf-8"), indent=2)
        yield
    finally:
        for p, data in saved.items():
            if data is None:
                if os.path.exists(p):
                    os.remove(p)
            else:
                open(p, "wb").write(data)


def push_card(api, repo, c, cfg, out=print):
    """The card goes up FIRST, and it UPDATES what the repo shows rather than replacing it: rebuilt from the union of
    this machine's rungs built and passed so far, every other machine's numbers in the repo's pollard/results-*.json
    sidecars, and the rungs actually on the Hub (one with no numbers anywhere is listed 'numbers pending'). Every
    lane any machine has published keeps its rows."""
    try:
        files = set(api.list_repo_files(repo))
    except Exception:                                            # noqa: BLE001 -- a new repo: nothing to merge
        files = set()
    side = hub_sidecars(api, repo, files, out) if files else {}
    lanes = [l for l in LANES if ((cfg.get("lanes") or {}).get(l) or {}).get("enabled") or l in side]
    with merged_lane_files(c, cfg, side, files):
        run_cmd(card_cmd(cfg, c, lanes))
    card = os.path.join(c["release_dir"], "README.md")
    if not os.path.exists(card):
        raise RuntimeError(f"pollard-card wrote no card at {card}")
    text, missing = pending_rows(open(card, encoding="utf-8").read(), files)
    if missing:
        open(card, "w", encoding="utf-8").write(text)
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo)
    out(f"      card: README.md -> https://huggingface.co/{repo} (every lane's rungs so far"
        + (f"; {len(missing)} with numbers pending" if missing else "") + ")")


def rung_card(step, c, cfg, run_upload, out=print):
    """A rung just measured and passed: its card goes up now, ahead of the lane's upload of the rung itself."""
    if not run_upload:
        out("      (not uploading the card: uploads are off)")
        return "held"
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        api = hf_api(cfg)
        open_repo(api, step.repo, cfg, out)
        push_card(api, step.repo, c, cfg, out)
    except Exception as e:                                       # noqa: BLE001
        # never holds the rung back: the lane's own card + upload step puts the card up before the rungs anyway
        out(f"      card not uploaded yet ({type(e).__name__}: {e}) -- the lane's upload puts it up before the rungs")
        return "card not uploaded"
    return "done"


def upload(step, c, cfg, run_upload):
    """Publish the verified rungs of one lane + the (combined) card into the ONE repo: the card FIRST, then the GGUF
    files at the root, a directory lane as <LANE>/<rung>/, the drafter in DRAFT/. Refuses a lane with no verified rung,
    and checks the repo afterwards: a lane whose files are not all there -- or a repo without its README -- raises."""
    from pollard_card import rung_subfolder
    lane, repo = step.lane, step.repo
    builds = sorted(_lane_builds(c, lane, repo), key=_build_bytes)   # smallest first
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
    api = hf_api(cfg)
    open_repo(api, repo, cfg)
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo)   # the card first, then the rungs
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
            if f.endswith(".imatrix") or f.startswith(("mmproj-", "components-")):
                api.upload_file(path_or_fileobj=os.path.join(c["gguf_dir"], f), path_in_repo=f, repo_id=repo)
    else:
        for b in builds:
            if b.get("uploaded") != repo:
                api.upload_folder(folder_path=b["path"], path_in_repo=f"{F}/{rung_subfolder(b, lane)}", repo_id=repo)
    missing = (lane_expect(c, lane, builds, repo) | {"README.md"}) - set(api.list_repo_files(repo))
    if missing:
        raise RuntimeError(f"{repo} is missing {sorted(missing)} after upload -- not live")
    print(f"      landed: {len(builds)} {lane} rung(s) in https://huggingface.co/{repo}" + (f" ({F}/)" if F else ""))
    return "uploaded"


def go_public(step, c, cfg, run_upload, statuses, out=print):
    """The last step: the repo has been public since its first upload (open_repo; `private: true` keeps it private),
    so there is nothing to flip -- this checks that EVERY lane that shipped this run is on the Hub with its whole file
    list (README, each lane's rungs in its folder, the mmproj), then reclaims the source mirror. A lane whose upload
    failed fails it; a lane that built nothing (held / failed / not this machine) ships nothing and does not count."""
    if not run_upload:
        out("      (not publishing: uploads are off)")
        return "held"
    ups = {l: statuses.get(f"{l}/upload") for l in (step.lanes or [])}
    bad = sorted(l for l, v in ups.items() if v == "failed")
    if bad:
        raise RuntimeError(f"{step.repo}: the {', '.join(bad)} upload failed")
    shipped = [l for l, v in ups.items() if v == "done"]
    if not shipped:
        out(f"      (nothing shipped this run -- {step.repo} stays as it is)")
        return "held"
    import pollard_workspace as ws
    stranded = [p for l in LANES for b in ws.read_manifest(lane_key(c, l)).get("builds", [])
                if b.get("lane") == l and b.get("streamed_unverified") and b.get("verified") is not True
                for p in b["streamed_unverified"][:1]]
    if stranded:
        raise RuntimeError(f"{step.repo} holds streamed shards of a rung that did not pass its gate "
                           f"({', '.join(sorted({os.path.dirname(p) for p in stranded}))}) -- verify it or remove that "
                           "folder, then re-run")
    api = hf_api(cfg)
    want = {"README.md"}
    for l in shipped:
        want |= lane_expect(c, l, _lane_builds(c, l, step.repo), step.repo)
    files = set(api.list_repo_files(step.repo))
    missing = want - files
    if missing:
        raise RuntimeError(f"{step.repo} is missing {sorted(missing)}")
    vis = "PRIVATE (private: true in the plan)" if repo_private(cfg) else "PUBLIC"
    out(f"      {vis}: https://huggingface.co/{step.repo}  ({', '.join(shipped)}; {len(files)} files checked)")
    reclaim_after_public(c, cfg, step.repo, api, out)
    return "done"


def reclaim_after_public(c, cfg, repo, api, out=print):
    """Once the repo is public: the source mirror this release read ($POLLARD_HOME/mirror/<model>, and its
    -smoothed twin) goes when EVERY build its MANIFESTs record is confirmed on the Hub by size + sha256 -- the same
    check as `pollard-ls --clean-mirrors` (a mirror with an unpublished build, or a Hub that cannot be read, is
    kept). `reclaim_mirror: false` in the plan keeps it. Never fails the release. cache/ is not touched here (its
    entries are content-keyed, not per model): `pollard-ls --clean-cache` reclaims it."""
    import pollard_workspace as ws
    if cfg.get("reclaim_mirror") is False:
        return 0
    root = os.path.normcase(os.path.realpath(ws.mirror_root())) + os.sep
    dirs = [d for d in (c.get("hf_dir"), c.get("smooth_dir"))
            if d and os.path.isdir(d) and os.path.normcase(os.path.realpath(d)).startswith(root)]
    if not dirs:
        return 0
    try:
        # this release's own steps are finished, so no age rule: only a file some process holds open keeps it
        freed, _ = ws.clean_mirrors(api, [repo], only=dirs, min_age_min=0, log=out)
    except Exception as e:                                                # noqa: BLE001
        out(f"      mirror: reclaim skipped ({type(e).__name__}: {e})")
        return 0
    if freed:
        out(f"      mirror: freed {ws.human(freed)} (reclaim cache/ with `pollard-ls --clean-cache`)")
    return freed


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


#: drafter GGUF rungs (pollard-draft), bits per weight -- for the build / upload order only
_GGUF_BITS = {"Q2_K": 2.6, "Q3_K_M": 3.9, "Q4_0": 4.5, "Q4_K_S": 4.6, "Q4_K_M": 4.85, "Q5_K_M": 5.7, "Q6_K": 6.6,
              "Q8_0": 8.5, "F16": 16.0, "BF16": 16.0}


def rung_bits(lane, rung):
    """Expected average bits per weight of one rung (mlx / gptq bits, exl3 / exlr8 bpw, an MX format or mix, a drafter
    GGUF type). An estimate for ordering and disk planning, not a claim about the build."""
    r = str(rung).lower()
    if lane == "mx":
        return mx_mix_bpw(rung) if mx_mix_bpw(rung) is not None else _MX_BITS.get(r, 8.0)
    if lane == "draft":
        return _GGUF_BITS.get(str(rung).upper(), 8.0)
    try:
        return float(rung)
    except ValueError:
        return 8.0


def order_rungs(cfg, lane, rungs):
    """The order a lane's rungs are built and published in: the plan's own order when the plan lists them, else
    smallest first (owner, 2026-10-24: "we always do the smallest first") -- ties keep the listed order."""
    rungs = list(rungs)
    if lane in ((cfg or {}).get("_explicit_order") or ()):
        return rungs
    return sorted(rungs, key=lambda r: rung_bits(lane, r))


def _rung_gb(lane, rung, hf_dir):
    """Estimated size of one directory-lane rung: the bf16 checkpoint scaled by the rung's bits, plus 10%."""
    return _size_gb(hf_dir) * rung_bits(lane, rung) / 16.0 * 1.1


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


def _local_files(path):
    """{path relative to the build (its file name for a one-file build): full path} of every file a build has on disk
    now -- a streamed rung's shard registry (pollard_stream.STREAMED) is the build's own bookkeeping, not counted."""
    import pollard_stream as PS
    if os.path.isfile(path):
        return {os.path.basename(path): path}
    out = {}
    for root, _, files in os.walk(path):
        for f in files:
            full = os.path.join(root, f)
            if f != PS.STREAMED:
                out[os.path.relpath(full, path).replace(os.sep, "/")] = full
    return out


def _freeable(c, key, b):
    """(True, "") when --free-after-upload may delete this rung, else (False, why). Decided by what the release
    RECORDED, never by where the file sits (the Bolt passthrough ladder's rungs were written beside the vendor GGUF in
    downloads/ and were kept, 38 GB, as "not a file this release built", 2026-10-24): the path is a build output in
    this release's MANIFEST (`key`), its upload is recorded there as confirmed on the Hub by size + sha256
    (`confirmed_files`, hashed from these bytes), and every file at the path still has the recorded size. An input of
    the release (the vendor GGUF / f16, the HF folder, calibration, imatrix, profiles) -- or a folder holding one --
    is never freed, whatever a MANIFEST entry says."""
    import pollard_workspace as ws
    raw = str(b.get("path", "")).rstrip("/\\")
    if not raw:
        return False, "no path recorded"
    path = os.path.normcase(os.path.abspath(raw))
    prot = _protected_paths(c)
    if path in prot:
        return False, "an input of this release"
    if any(p.startswith(path + os.sep) for p in prot):
        return False, "holds an input of this release"
    for k in ("f16", "hf_dir", "src", "smooth_dir", "imatrix", "calib", "heldout"):
        try:
            if c.get(k) and os.path.exists(str(c[k])) and os.path.samefile(raw, str(c[k])):
                return False, "the same file as an input of this release"
        except OSError:
            pass
    entry = next((x for x in ws.read_manifest(key).get("builds", [])
                  if os.path.normcase(os.path.abspath(str(x.get("path", "")).rstrip("/\\"))) == path
                  and x.get("lane") == b.get("lane")), None)
    if entry is None:
        return False, "not a build output recorded in this release's MANIFEST"
    conf = entry.get("confirmed_files") or {}
    if not entry.get("uploaded") or not conf:
        return False, "no upload confirmed by size + sha256 recorded for it"
    now = _local_files(raw)
    if set(now) != set(conf):
        extra = sorted(set(now) - set(conf))
        return False, ("files the upload did not confirm: " + ", ".join(extra[:3])) if extra else \
            "files missing since the upload was confirmed"
    for rel, full in now.items():
        if os.path.getsize(full) != int(conf[rel]["size"]):
            return False, f"{rel} changed size since the upload was confirmed"
    return True, ""


def _free_local(c, key, b, out=print):
    path = b.get("path", "")
    ok, why = _freeable(c, key, b)
    if not ok:
        out(f"      (kept {path}: {why})")
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


def _keep(key, b, why, out):
    """A rung that is NOT freed: recorded in the MANIFEST (kept = why) and said out loud."""
    import pollard_workspace as ws
    b["kept"] = why
    b.pop("freed", None)
    fields = ("kept", "streamed_unverified", "runtime", "runtime_arch", "uploaded_as")
    ws.update_build(key, b["path"], {k: b[k] for k in fields if k in b}, remove=("freed",))
    out(f"      KEPT {b.get('path')} -- {why}")


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
        if streamed and b.get("verified") is False and PS.staged_revisions(streamed):
            # the gate measured it and it FAILED: its staging branch goes, main never held a byte of it
            discard_streamed(b["path"], out)
            _keep(key, b, "failed its gate -- not published; its streamed shards were on a staging branch, now "
                  "deleted (main never held them)", out)
            return "kept: not verified"
        if streamed:
            b["streamed_unverified"] = sorted(e["path_in_repo"] for e in streamed.values())
        staged = bool(PS.staged_revisions(streamed))
        _keep(key, b, "not verified (measure / verify / gate failed or did not run) -- not published"
              + (f"; its {len(streamed)} streamed shard(s) stay on "
                 + ("their staging branch (not main) until the rung is verified" if staged else
                    "the repo's MAIN branch (streamed there by an older Pollard) until they are verified or "
                    "removed -- the release's last step fails on them")
                 if streamed else ""), out)
        return "kept: not verified"
    if not run_upload:
        raise RuntimeError("--free-after-upload needs uploading on (`upload: true` + --upload)")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    api = hf_api(cfg)
    open_repo(api, repo, cfg, out)
    if not b.get("uploaded"):
        try:                                     # the card with this rung's numbers goes up first, the rung after it
            push_card(api, repo, c, cfg, out)
        except Exception as e:                   # noqa: BLE001 -- never holds the rung back
            out(f"      card not uploaded yet ({type(e).__name__}: {e}) -- the next rung's card carries it")
    if lane == "gguf":
        name = b.get("name") or os.path.basename(b["path"])
        try:                                     # pollard-ggufcheck before it leaves: the card states the runtime
            import pollard_ggufcompat as gc
            b["runtime"], _why = gc.runtime_of(b["path"])
            if b["runtime"] == "pollard":      # the card names the patch even after the rung is freed
                b["runtime_arch"] = _why.get("architecture")
            out(f"      ggufcheck: {name} -> {b['runtime']}")
        except Exception as e:                                   # noqa: BLE001
            out(f"      ggufcheck: could not read {name} ({e}) -- the card will not state its runtime")
        dest = name
        if not b.get("uploaded"):
            api.upload_file(path_or_fileobj=b["path"], path_in_repo=dest, repo_id=repo)
    else:
        dest = f"{lane_folder(lane)}/{rung_subfolder(b, lane)}"
        if not b.get("uploaded") and streamed:
            # the shards are already there (streamed + confirmed one by one) -- on the rung's staging branch: copied
            # onto main now that it is verified (server side, nothing re-sent), the branch deleted; then the rest,
            # then the index LAST -- a loader sees the rung only once every shard it maps is in place
            if PS.staged_revisions(streamed):
                PS.promote_staging(PS._hf_api(), streamed, log=out)
            api.upload_folder(folder_path=b["path"], path_in_repo=dest, repo_id=repo,
                              ignore_patterns=["model.safetensors.index.json", PS.STREAMED, "*.tmp"])
            if os.path.isfile(os.path.join(b["path"], "model.safetensors.index.json")):
                api.upload_file(path_or_fileobj=os.path.join(b["path"], "model.safetensors.index.json"),
                                path_in_repo=f"{dest}/model.safetensors.index.json", repo_id=repo)
        elif not b.get("uploaded"):
            api.upload_folder(folder_path=b["path"], path_in_repo=dest, repo_id=repo)
    b["uploaded_as"] = dest
    import pollard_hubcheck as H
    local = [(rp, lp) for rp, lp in _local_pairs(b["path"], dest) if os.path.basename(lp) != PS.STREAMED]
    # every local file hashed ONCE (size + sha256 + git blob id): the Hub is checked against these numbers, and they
    # are what the MANIFEST records as confirmed -- the free step deletes only the files that still match them
    recs = {lp: H.file_record(lp) for _, lp in local}
    pairs = [(rp, recs[lp]) for rp, lp in local]
    pairs += [(e["path_in_repo"], {"size": e["size"], "sha256": e["sha256"], "git_sha1": e.get("git_sha1")})
              for e in streamed.values()]
    bad = remote_mismatches(api, repo, pairs)
    if bad:
        _keep(key, b, "upload not confirmed: " + "; ".join(bad[:5]), out)
        raise RuntimeError(f"{dest} is not intact in {repo} after the upload ({bad[0]}) -- keeping the local copy")
    b["uploaded"] = repo
    b.pop("kept", None)
    base = b["path"] if os.path.isdir(b["path"]) else os.path.dirname(b["path"])
    b["confirmed_files"] = {os.path.relpath(lp, base).replace(os.sep, "/"): {
        "size": recs[lp]["size"], "sha256": recs[lp]["sha256"], "path_in_repo": rp} for rp, lp in local}
    if streamed:
        b["bytes"] = int(_size_gb(b["path"]) * 1e9) + sum(int(e["size"]) for e in streamed.values())
    # what the Hub holds for it (size + sha256 / git blob id): pollard-ls --clean-mirrors checks the Hub against
    # this once the local copy is freed
    rec = H.records(api, repo, [rp for rp, _ in pairs])
    if rec and len(rec) == len(pairs):
        b["hub_files"] = rec
    fields = ("uploaded", "uploaded_as", "runtime", "runtime_arch", "bytes", "hub_files", "confirmed_files")
    out(f"      uploaded {dest} -> https://huggingface.co/{repo} (size + hash confirmed on the Hub)")
    if not ws.update_build(key, b["path"], {k: b[k] for k in fields if k in b}, remove=("kept",)):
        # the upload is confirmed but the MANIFEST no longer has this build to say so: freeing the file now would
        # leave nothing on this machine recording where the rung went -- keep it
        out(f"      KEPT {b.get('path')} -- uploaded, but its MANIFEST entry is gone, so the upload is not recorded")
        return "kept: upload not recorded"
    if _free_local(c, key, b, out):
        b["freed"] = "uploaded + confirmed on the Hub"
        ws.update_build(key, b["path"], {"freed": b["freed"]})
    push_sidecar(api, repo, c, cfg, lane, out)   # this machine's numbers in the repo, best-effort
    return "done"


def discard_streamed(path, out=print):
    """A streamed rung that will never ship (its build or its gate failed): every staging branch its shards sit on is
    deleted and the registry with it -- main never held them. Shards streamed onto main by an older Pollard (no
    branch) are left alone: the release's last step (go_public) fails on them until someone removes them."""
    import pollard_stream as PS
    streamed = PS.read_streamed(path) if os.path.isdir(str(path)) else {}
    revs = PS.staged_revisions(streamed)
    if not revs:
        return False
    api = PS._hf_api()
    for repo, rev in revs:
        PS.discard_staging(api, repo, rev, log=lambda m: out("   " + m.strip()))
    if all(e.get("revision") for e in streamed.values()):
        try:
            os.remove(os.path.join(path, PS.STREAMED))
        except OSError:
            pass
    return True


def drop_failed_stream(st, statuses, out=print):
    """The publish step of a rung that will not ship: when its BUILD failed or its gate FAILED (not merely held or not
    run -- that rung is resumable), the staging branch its streamed shards sit on is deleted. Never raises."""
    if not st.out or st.lane == "gguf":
        return False
    import pollard_workspace as ws
    try:
        b = ws.find_build(st.out)[1] or {}
        if str(statuses.get(f"{st.lane}/{st.rung}/build", "")) == "failed" or b.get("verified") is False:
            return discard_streamed(st.out, out)
    except Exception as e:                                       # noqa: BLE001
        out(f"      could not delete the staging branch of {st.out}: {e} -- delete it on the Hub by hand")
    return False


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


# ------------------------------------------------------------------------------- memory preflight (MLX on a Mac)
#
# The MLX lane builds and verifies streamed (one tensor / one block at a time), but a 16 GB Mac has rebooted under
# memory pressure before: every MLX step is checked against the RAM that is free NOW, minus a floor
# (POLLARD_MIN_FREE_GB, 2 GB) -- and the plan's max_ram_gb -- before it starts. A build or verify that does not fit is
# HELD with the numbers (the rest of the ladder carries on); a tool-call gate whose rung cannot be served whole in
# mlx_lm.server is recorded as not measured, with the reason, and the card says so.

def mirror_cache_gb(c):
    """GB a remote-source mirror at the HF folder holds as cache -- disk it evicts when a rung needs it."""
    try:
        import pollard_mlx as M
        m = M.RemoteMirror.of(c["hf_dir"], log=lambda *a: None) if os.path.isdir(str(c.get("hf_dir"))) else None
        return m.resident_gb() if m is not None else 0.0
    except Exception:                                 # noqa: BLE001
        return 0.0


def memory_need(st, c):
    """GB of RAM an MLX step needs on this machine, or None when it cannot be told yet (source not there)."""
    try:
        import pollard_mlx as M
        if st.mem == "mlx-build":
            low, high = M.rung_bits(int(st.bits))
            return M.plan_build(c["hf_dir"], low, high)["peak_gb"]
        if st.mem == "mlx-verify":
            return M.plan_verify(c["hf_dir"])["peak_gb"]
        if st.mem == "mlx-serve":                     # mlx_lm.server holds the whole rung + its cache
            gb = _size_gb(st.rung_dir) if st.rung_dir and os.path.exists(st.rung_dir) else 0.0
            return gb * 1.15 + 1.0 if gb else None
    except (Exception, SystemExit):                   # noqa: BLE001 -- unreadable here: the step's own checks decide
        return None
    return None


def memory_short(st, c, cfg):
    """The reason this step does not fit the RAM free now, or "" when it does (or cannot be told)."""
    need = memory_need(st, c)
    if not need:
        return ""
    try:
        import pollard_stream as S
        free = S.free_ram_gb()
    except Exception:                                 # noqa: BLE001
        free = None
    if free is None:
        return ""
    floor = float(os.environ.get("POLLARD_MIN_FREE_GB", 2.0))
    cap = cfg.get("max_ram_gb")
    usable = min(free - floor, float(cap)) if cap else free - floor
    if need <= usable:
        return ""
    what = {"mlx-build": "the streamed MLX build", "mlx-verify": "the streamed MLX verify",
            "mlx-serve": "mlx_lm.server with the whole rung"}[st.mem]
    return (f"{what} needs ~{need:.1f} GB of RAM and {max(usable, 0):.1f} GB is usable here ({free:.1f} GB free, "
            f"{floor:g} GB floor" + (f", max_ram_gb {cap}" if cap else "") + ")")


def record_toolcall_not_measured(st, why):
    import pollard_workspace as ws
    if st.results:
        ws.merge_results(st.results, st.rung_name or st.rung, {"toolcall": {
            "status": "not-measured", "reason": why, "backend": "mlx", "mode": "full", "lane": st.lane}})


# ------------------------------------------------------------------------------- upload relay (--upload-via)

class RelayApi:
    """The huggingface_hub calls a release makes, run on ANOTHER machine over ssh -- for a builder with no HF login
    (the Mac builds MLX, the box holds the token). `ssh://user@host[:port]?python=<python>&key=<identity>
    &min_free_gb=40[&home=<that machine's POLLARD_HOME>]`. A folder or file is copied with scp into that machine's
    Pollard workspace -- <its POLLARD_HOME>/staging/<repo slug>/<path in repo>/, one fixed folder per rung, found by
    asking its python (or `home=`) -- after checking it keeps min_free_gb free for the bytes still to send; uploaded
    from there with that machine's own login; checked on the Hub file by file (size, and sha256 for LFS files)
    against the local copy; and only then is the staging copy deleted. A transfer that dies part way (the box
    rebooted mid-scp) leaves its folder for the next run, which sends only the files not already there at the
    right size + sha256. Nothing local is touched. A path in the URL (the old <staging dir>) is no longer used."""

    def __init__(self, url, runner=None, out=print):
        from urllib.parse import urlparse, parse_qs, unquote
        import pollard_workspace as ws
        u = urlparse(url)
        if u.scheme != "ssh" or not u.hostname:
            raise SystemExit(f"--upload-via {url}: expected ssh://user@host[:port]?python=<python>")
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        self.host = f"{u.username}@{u.hostname}" if u.username else u.hostname
        self.port = str(u.port or 22)
        path = unquote(u.path)
        self.url_path = (path[1:] if len(path) > 2 and path[2] == ":" else path).rstrip("/") or None  # /C:/x -> C:/x
        self.python = q.get("python", "python3")
        self.key = os.path.expanduser(q["key"]) if q.get("key") else None
        self.min_free = float(q.get("min_free_gb", 40))
        self.report_days = float(q.get("report_days", ws.STAGING_REPORT_DAYS))
        self._home = q.get("home", "").replace("\\", "/").rstrip("/") or None
        self.run_, self.out = runner or subprocess.run, out

    # -------------------------------------------------------------------------------------------- the remote side
    def _base(self, tool="ssh"):
        return [tool] + (["-i", self.key] if self.key else []) + ["-o", "BatchMode=yes",
                                                                   "-P" if tool == "scp" else "-p", self.port]

    def _py(self, code):
        """Run python on the relay (code on stdin); the last stdout line is JSON."""
        pre = "import json, os\nos.environ.setdefault('HF_HUB_DISABLE_XET', '1')\n"
        r = self.run_(self._base() + [self.host, f'"{self.python}" -'], input=pre + code, capture_output=True,
                      text=True)
        if r.returncode:
            raise RuntimeError(f"relay {self.host}: python exited {r.returncode}: {(r.stderr or '')[-800:]}")
        lines = [x for x in (r.stdout or "").splitlines() if x.strip()]
        return json.loads(lines[-1]) if lines else None

    @staticmethod
    def _ship(*fns):
        """The source of self-contained pollard_workspace helpers, to define them on the relay machine (which may run
        an older Pollard, or none)."""
        import inspect
        import textwrap
        return "".join(textwrap.dedent(inspect.getsource(f)) + "\n" for f in fns)

    @property
    def home(self):
        """The relay machine's Pollard workspace: `home=` in the URL, else what its pollard_workspace says (its
        $POLLARD_HOME, default ~/pollard), asked once."""
        if self._home is None:
            h = self._py("try:\n    import pollard_workspace as _ws\n    h = _ws.pollard_home()\nexcept Exception:\n"
                         "    h = os.path.abspath(os.environ.get('POLLARD_HOME', os.path.expanduser('~/pollard')))\n"
                         "print(json.dumps(h.replace(os.sep, '/')))\n")
            if not h:
                raise RuntimeError(f"relay {self.host}: could not read its POLLARD_HOME -- pass home= in the URL")
            self._home = str(h).replace("\\", "/").rstrip("/")
            if self.url_path and self.url_path.rstrip("/") != self.staging:
                self.out(f"      relay: the URL's staging path {self.url_path} is no longer used -- staging goes "
                         f"to {self.staging} (the workspace of {self.host})")
        return self._home

    @property
    def staging(self):
        return f"{self.home}/staging"

    def stage_dir(self, repo_id, path_in_repo):
        import pollard_workspace as ws
        return f"{self.staging}/{ws.model_slug(repo_id)}/{ws.staging_key(path_in_repo)}"

    def _api(self, call):
        return self._py(f"from huggingface_hub import HfApi\napi = HfApi()\nprint(json.dumps({call}))\n")

    def create_repo(self, repo_id, repo_type="model", exist_ok=True, private=True):
        self._api(f"str(api.create_repo({repo_id!r}, repo_type={repo_type!r}, exist_ok={exist_ok!r}, "
                  f"private={private!r}))")

    def list_repo_files(self, repo_id):
        return self._api(f"api.list_repo_files({repo_id!r})")

    def update_repo_settings(self, repo_id, private):
        self._api(f"str(api.update_repo_settings(repo_id={repo_id!r}, private={private!r}))")

    def get_paths_info(self, repo_id, paths):
        """HfApi.get_paths_info on the relay (what pollard_hubcheck reads back after an upload): path, size, LFS
        sha256 and git blob id per file."""
        from types import SimpleNamespace
        rows = self._py("from huggingface_hub import HfApi\napi = HfApi()\nd = []\n"
                        f"for f in api.get_paths_info({repo_id!r}, {list(paths)!r}):\n"
                        "    if hasattr(f, 'size') and not hasattr(f, 'tree_id'):\n"
                        "        lfs = getattr(f, 'lfs', None)\n"
                        "        sha = (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else None\n"
                        "        d.append([f.path, f.size, sha, getattr(f, 'blob_id', None)])\n"
                        "print(json.dumps(d))\n") or []
        return [SimpleNamespace(path=p, size=s, lfs=({"sha256": h} if h else None), blob_id=b) for p, s, h, b in rows]

    def listing(self, repo_id):
        """{path: [size, lfs sha256 or None]} of every file in the repo, read on the relay."""
        return self._py("from huggingface_hub import HfApi\napi = HfApi()\nd = {}\n"
                        f"for f in api.list_repo_tree({repo_id!r}, recursive=True):\n"
                        "    if hasattr(f, 'size') and not hasattr(f, 'tree_id'):\n"
                        "        d[f.path] = [f.size, (f.lfs.sha256 if getattr(f, 'lfs', None) else None)]\n"
                        "print(json.dumps(d))\n")

    def _drop(self, stage):
        """Delete one staging folder on the relay (only ever inside its workspace staging area)."""
        import pollard_workspace as ws
        self._py(self._ship(ws.remove_within) + f"remove_within({self.staging!r}, {stage!r})\nprint('null')\n")

    # ------------------------------------------------------------------------------------------- staging + resume
    @staticmethod
    def _local_plan(local, prefix, ignore=()):
        """[(path inside the staging folder, local file, path in repo)] as upload_file / upload_folder lay it out --
        a folder's files keep their relative paths, a file sits in the folder under its own name."""
        from fnmatch import fnmatch
        if os.path.isfile(local):
            return [(os.path.basename(local), local, prefix)]
        items = []
        for r, _, fs in os.walk(local):
            for f in fs:
                rel = os.path.relpath(os.path.join(r, f), local).replace(os.sep, "/")
                if any(fnmatch(rel, p) for p in ignore or ()):
                    continue
                items.append((rel, os.path.join(r, f), "/".join(x for x in (prefix.strip("/"), rel) if x)))
        return sorted(items)

    @staticmethod
    def local_files(local, prefix, ignore=()):
        """{path in repo: (size, sha256)} of a local file or folder as upload_file / upload_folder lay it out."""
        import pollard_hubcheck as H
        return {rp: (os.path.getsize(p), H.sha256_file(p)) for _, p, rp in RelayApi._local_plan(local, prefix, ignore)}

    def _stage(self, local, repo_id, path_in_repo, ignore=()):
        """Copy `local` into its fixed staging folder on the relay, resuming: files already there at the same size
        and sha256 are not sent again, a stray file (a different build of the rung) is removed from the folder.
        Returns (staging folder, {path in repo: (size, sha256)})."""
        import pollard_hubcheck as H
        import pollard_workspace as ws
        plan = self._local_plan(local, path_in_repo, ignore)
        want = {rel: (os.path.getsize(p), H.sha256_file(p)) for rel, p, _ in plan}
        stage = self.stage_dir(repo_id, path_in_repo)
        mark = {"repo_id": repo_id, "path_in_repo": path_in_repo, "source": os.path.abspath(local),
                "files": {rp: list(want[rel]) for rel, _, rp in plan},
                "staged": time.strftime("%Y-%m-%dT%H:%M:%S")}
        got = self._py(self._ship(ws.sha256_files) + "import shutil\n"
                       f"stage, want, mark = {stage!r}, {({k: v[0] for k, v in want.items()})!r}, {mark!r}\n"
                       "os.makedirs(stage, exist_ok=True)\n"
                       "have = {}\n"
                       "for r, _, fs in os.walk(stage):\n"
                       "    for f in fs:\n"
                       "        rel = os.path.relpath(os.path.join(r, f), stage).replace(os.sep, '/')\n"
                       f"        if rel != {ws.STAGING_MARK!r}:\n"
                       "            have[rel] = os.path.getsize(os.path.join(r, f))\n"
                       "for rel in [x for x in have if x not in want]:\n"
                       "    os.remove(os.path.join(stage, rel))\n"
                       "    del have[rel]\n"
                       "for rel in want:\n"
                       "    os.makedirs(os.path.dirname(os.path.join(stage, rel)), exist_ok=True)\n"
                       f"with open(os.path.join(stage, {ws.STAGING_MARK!r} + '.tmp'), 'w', encoding='utf-8') as f:\n"
                       "    json.dump(mark, f)\n"
                       f"os.replace(os.path.join(stage, {ws.STAGING_MARK!r} + '.tmp'), os.path.join(stage, {ws.STAGING_MARK!r}))\n"
                       "shas = sha256_files(stage, [r for r, s in want.items() if have.get(r) == s])\n"
                       "print(json.dumps({'have': have, 'shas': shas, 'free': shutil.disk_usage(stage).free / 1e9}))\n")
        have, shas, free = got["have"], got["shas"], float(got["free"])
        send = [(rel, p) for rel, p, _ in plan if not (have.get(rel) == want[rel][0] and shas.get(rel) == want[rel][1])]
        need = sum(max(0, want[rel][0] - int(have.get(rel, 0))) for rel, _ in send) / 1e9
        kept = len(plan) - len(send)
        if free - need < self.min_free:
            raise RuntimeError(f"relay {self.host}: {need:.1f} GB still to stage and {free:.1f} GB free there -- it "
                               f"must keep {self.min_free:g} GB free (the partial copy in {stage} is kept)")
        self.out(f"      relay: {local} -> {self.host}:{stage}/ (" + (
            f"resuming: {kept} of {len(plan)} file(s) already there, " if kept else "")
            + f"{need:.2f} GB to send; {free:.0f} GB free there)")
        groups = {}
        for rel, p in send:
            groups.setdefault(rel.rsplit("/", 1)[0] if "/" in rel else "", []).append(p)
        for sub, paths in sorted(groups.items()):
            dest = f"{stage}/{sub}/" if sub else f"{stage}/"
            r = self.run_(self._base("scp") + ["-q"] + paths + [f"{self.host}:{dest}"])
            if r.returncode:
                raise RuntimeError(f"relay {self.host}: scp exited {r.returncode} (the partial copy in {stage} is "
                                   "kept: the next run resumes it)")
        return stage, {rp: want[rel] for rel, _, rp in plan}

    def check(self, repo_id, want):
        got = self.listing(repo_id) or {}
        bad = sorted(rp for rp, (size, sha) in want.items()
                     if rp not in got or got[rp][0] != size or (got[rp][1] and got[rp][1] != sha))
        if bad:
            raise RuntimeError(f"{repo_id}: after the relay upload {len(bad)} file(s) differ from the local copy "
                               f"(size / sha256): {bad[:5]}")
        self.out(f"      relay: {len(want)} file(s) on the Hub match the local copy (size + sha256)")

    def _upload(self, local, path_in_repo, repo_id, call, ignore=()):
        stage, want = self._stage(local, repo_id, path_in_repo, ignore)
        try:
            self._api(call(stage))
            self.check(repo_id, want)
        except BaseException:
            self.out(f"      relay: upload not confirmed -- the staging copy {self.host}:{stage} is kept (the next "
                     "run resumes it)")
            raise
        self._drop(stage)                               # only after the Hub holds every file, size + sha256
        return stage

    def upload_folder(self, folder_path, path_in_repo, repo_id, ignore_patterns=None, **kw):
        import pollard_workspace as ws
        ignore = list(ignore_patterns or [])
        pats = [ws.STAGING_MARK] + ignore
        self._upload(folder_path, path_in_repo, repo_id, lambda stage: (
            f"str(api.upload_folder(folder_path={stage!r}, path_in_repo={path_in_repo!r}, repo_id={repo_id!r}, "
            f"ignore_patterns={pats!r}))"), ignore)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, **kw):
        name = os.path.basename(path_or_fileobj)
        self._upload(path_or_fileobj, path_in_repo, repo_id, lambda stage: (
            f"str(api.upload_file(path_or_fileobj={stage + '/' + name!r}, path_in_repo={path_in_repo!r}, "
            f"repo_id={repo_id!r}))"))

    # ------------------------------------------------------------------------------------------------- the sweep
    def sweep(self, repos=(), force=False, legacy_root=None):
        """At the start of a run: on the relay machine, remove empty staging folders and copies already confirmed on
        the Hub (size + sha256), report anything else older than report_days, keep recent partial copies for resume.
        The legacy folder (C:/pollard/mlx-staging, random-named copies the old relay never reclaimed) is swept the
        same way -- emptied of copies already on the Hub, removed once empty -- and nothing else outside the
        workspace is ever touched. Returns (bytes freed, [(path, action, why)])."""
        import pollard_workspace as ws
        legacy = ws.LEGACY_STAGING if legacy_root is None else legacy_root
        scans = self._py(self._ship(ws.scan_staging) +
                         f"out = [scan_staging({self.staging!r}, 2), []]\n"
                         f"if {legacy!r} and os.path.isdir({legacy!r}):\n"
                         f"    out[1] = scan_staging({legacy!r}, 1)\n"
                         f"print(json.dumps(out + [bool({legacy!r}) and os.path.isdir({legacy!r})]))\n")
        cache = {}

        def listing_of(repo):
            if repo not in cache:
                try:
                    cache[repo] = self.listing(repo)
                except Exception:                                         # noqa: BLE001
                    cache[repo] = None
            return cache[repo]

        def shas_of(e, rels):
            return self._py(self._ship(ws.sha256_files) + f"print(json.dumps(sha256_files({e['path']!r}, {rels!r})))\n")

        freed, report = 0, []
        for root, entries, is_legacy in ((self.staging, scans[0], False), (legacy, scans[1], True)):
            if is_legacy and not scans[2]:
                continue
            plan = ws.plan_sweep(entries, listing_of, shas_of, repos, None, self.report_days, is_legacy)
            gone = [e for e, act, _ in plan if act == "remove" or (force and not is_legacy)]
            if gone:
                self._py(self._ship(ws.remove_within) +
                         f"for p in {[e['path'] for e in gone]!r}:\n    remove_within({root!r}, p)\nprint('null')\n")
            for e, act, why in plan:
                if e in gone:
                    freed += e["bytes"]
                    why = why if act == "remove" else "forced"
                    act = "remove"
                    self.out(f"   relay staging: removed {self.host}:{e['path']} -- {why}")
                elif act == "report":
                    self.out(f"   relay staging: KEPT {self.host}:{e['path']} -- {why}")
                report.append((e["path"], act, why))
            left = self._py(self._ship(ws._prune_empty) + f"_prune_empty({root!r}, {is_legacy!r}, log=lambda m: None)\n"
                            f"print(json.dumps(os.path.isdir({root!r})))\n")
            if is_legacy and not left:
                self.out(f"   relay staging: removed the empty legacy folder {self.host}:{root}")
        return freed, report


def hf_api(cfg):
    """HfApi() here, or the --upload-via relay (another machine's login) when the plan names one."""
    if cfg.get("upload_via"):
        return RelayApi(str(cfg["upload_via"]))
    from huggingface_hub import HfApi
    return HfApi()


def sweep_staging(cfg, c, out=print, relay=None):
    """Start of an uploading run: reclaim staging copies left by earlier runs -- this machine's workspace staging area
    and, with --upload-via, the relay machine's (plus its legacy C:/pollard/mlx-staging). Empty folders and copies
    already confirmed on the Hub (size + sha256) go; recent partial copies stay for the upload to resume; anything
    else older than a week is reported, not deleted. Never fails the run."""
    import pollard_workspace as ws
    repos = [repo_name(c.get("org", cfg.get("org", "PollardWeights")), c["src"])]
    if os.path.isdir(ws.staging_root()) or os.path.isdir(ws.LEGACY_STAGING):
        try:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
            except ImportError:
                api = None
            freed, _ = ws.sweep_staging(api, repos, log=out)
            if freed:
                out(f"   staging: freed {ws.human(freed)} on this machine")
        except Exception as e:                                            # noqa: BLE001
            out(f"   staging: sweep of {ws.staging_root()} skipped ({type(e).__name__}: {e})")
    if cfg.get("upload_via"):
        try:
            r = relay or RelayApi(str(cfg["upload_via"]), out=out)
            freed, _ = r.sweep(repos)
            if freed:
                out(f"   relay staging: freed {ws.human(freed)} on {r.host}")
        except Exception as e:                                            # noqa: BLE001
            out(f"   relay staging: sweep skipped ({type(e).__name__}: {e})")


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


#: tools that take --device (the streamed HF lanes and their gate) and the ones with no CPU path at all
DEVICE_TOOLS = ("pollard-export", "pollard-mx", "pollard-exlr8", "pollard-hf-smooth", "pollard-verify", "pollard-probe")
GPU_ONLY_TOOLS = ("pollard-exl3",)


def gpu_returned():
    """True once POLLARD_GPU_UNTIL has passed (pollard_llamaserve.gpu_lent) on a CUDA machine."""
    if not os.environ.get("POLLARD_GPU_UNTIL"):
        return False
    import pollard_llamaserve as LS
    return not LS.gpu_lent()


#: the lanes whose steps load exllamav3 (its JIT-built CUDA extension): the build, the in-process tool gate, the verify
EXL3_ENV_LANES = ("exl3",)


def _is_system_account(env):
    """True when this process runs as Windows' SYSTEM account (a scheduled task with no user logged in)."""
    user = str(env.get("USERNAME") or "").strip().upper()
    prof = str(env.get("USERPROFILE") or env.get("LOCALAPPDATA") or "").lower()
    return user == "SYSTEM" or user.endswith("$") or "systemprofile" in prof


def _user_torch_extensions(env, exe, users_root=None, P=os.path):
    """The torch_extensions cache of the person who owns this install (where exllamav3's extension was built), for a
    SYSTEM run whose own profile has none: the user that POLLARD_HOME / the venv / this checkout lives under, else the
    one user profile that holds a cache. `P` = the path module of the target OS (ntpath on Windows)."""
    import glob
    sub = P.join("AppData", "Local", "torch_extensions", "torch_extensions", "Cache")
    root = users_root or P.join((env.get("SystemDrive") or "C:") + P.sep, "Users")
    rootn = P.normcase(P.normpath(root))
    owners = []
    for p in (env.get("POLLARD_HOME"), exe, os.path.abspath(__file__)):
        p = P.normcase(P.normpath(str(p or "")))
        if p.startswith(rootn + P.sep):
            name = p[len(rootn) + 1:].split(P.sep)[0]
            if name and name not in owners:
                owners.append(name)
    for name in owners:
        cand = P.join(root, name, sub)
        if os.path.isdir(cand):
            return cand
    found = sorted(d for d in glob.glob(P.join(root, "*", sub)) if os.path.isdir(d))
    return found[0] if len(found) == 1 else None


def exl3_env(base=None, exe=None, platform_name=None, users_root=None):
    """The environment an EXL3 step runs in. exllamav3 builds its CUDA extension on first use with torch's JIT, which
    needs `ninja` on PATH and finds the built extension in the account's torch_extensions cache. A Windows scheduled
    task as SYSTEM has neither -- the venv's Scripts is not on its PATH and SYSTEM's profile holds no cache -- so the
    import died with 'DLL load failed' on the box (2026-10-17, qualification) while the same step ran fine as the user.
    The venv's Scripts / bin dir goes first on PATH; as SYSTEM, TORCH_EXTENSIONS_DIR points at the owner's cache
    (an explicit TORCH_EXTENSIONS_DIR is never overridden)."""
    import ntpath, posixpath
    env = dict(os.environ if base is None else base)
    win = (platform_name or sys.platform).startswith("win")
    P, sep = (ntpath, ";") if win else (posixpath, ":")
    exe = exe or sys.executable
    scripts = P.dirname(exe)
    key = next((k for k in env if k.upper() == "PATH"), "PATH")       # Windows spells it Path
    parts = [p for p in str(env.pop(key, "") or "").split(sep) if p]
    norm = lambda p: P.normcase(P.normpath(p))                        # noqa: E731
    if scripts and norm(scripts) not in {norm(p) for p in parts}:
        parts.insert(0, scripts)
    env[key] = sep.join(parts)
    if win and not env.get("TORCH_EXTENSIONS_DIR") and _is_system_account(env):
        cache = _user_torch_extensions(env, exe, users_root, P)
        if cache:
            env["TORCH_EXTENSIONS_DIR"] = cache
    return env


def run_cmd(cmd, ok_codes=(0,), env=None):
    from pollard_auto import _resolve_tool
    r = subprocess.run(_resolve_tool(list(cmd)), env=env) if env else subprocess.run(_resolve_tool(list(cmd)))
    if r.returncode not in ok_codes:
        raise RuntimeError(f"exit {r.returncode}")


#: step kinds that compute (quantize, imatrix, perplexity / KLD, a model loaded to verify) -- the machine locks cover
#: these and nothing else
HEAVY_KINDS = ("build", "prep", "measure", "verify", "gate")
#: preparation steps that only move bytes (a download / a sparse mirror): network, not compute
LIGHT_PREP = ("prep/fetch", "prep/mirror")


def is_heavy(st):
    """True for a compute-heavy step (build / measure / verify / gate / compute prep). Publish, upload, relay,
    card and finalize steps are not: an upload-only release must not hold the machine lock while it uploads."""
    if st.kind not in HEAVY_KINDS:
        return False
    if st.id in LIGHT_PREP or st.fn == "fetch":
        return False
    return st.fn not in ("upload", "publish_rung", "card_upload", "rung_card", "go_public")


class _ExecutionState:
    def __init__(self):
        self.failed_rungs, self.failed_lanes, self.unbuildable = set(), set(), set()
        self.held_mem = set()
        self.streamed = set()


def execute(steps, c, cfg, run=False, run_upload=False, redo=(), stop_on_fail=False, have=None, out=print,
            execution=None):
    """Print (dry-run) or run the plan, serially. Returns {step id: status}."""
    have = have or machine()
    state = load_state(c["state"])
    execution = execution or _ExecutionState()
    statuses = {}
    failed_rungs, failed_lanes, unbuildable = execution.failed_rungs, execution.failed_lanes, execution.unbuildable
    held_mem = execution.held_mem                    # (lane, rung) held for RAM: kept on disk for a re-run
    kept = []                                        # (lane, rung, path) left on disk unpublished -- reported at the end
    streamed = execution.streamed                   # (lane, rung) built in streaming mode (shards already on the Hub)
    queue, n = list(steps), 0
    while queue:
        st = queue.pop(0)
        n += 1
        if run:
            engine_arch(c)
        head = f"{n:3d}. [{st.lane}{'/' + st.rung if st.rung else ''}] {st.kind}"
        if st.kind == "expand" and st.lane in failed_lanes:
            out(f"{head}: SKIP (an earlier step of this lane failed)")
            statuses[st.id] = "skipped: upstream failed"
            continue
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
                with _STATE_LOCK:                    # the parallel scheduler: each rung is measured by ONE chain
                    claimed = c.setdefault("_claimed", set())
                    builds = [b for b in builds if os.path.abspath(b["path"]) not in claimed]
                    claimed.update(os.path.abspath(b["path"]) for b in builds)
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
                save_entry(c["state"], st.id, state[st.id])
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
        if st.kind == "publish" and (st.lane, st.rung) in held_mem:
            out(f"{head}: KEPT -- a step of this rung was held for memory; re-run the plan to finish it -- {what}")
            statuses[st.id] = "held: memory (rung kept for a re-run)"
            continue
        if st.kind == "publish" and (st.lane in failed_lanes or (st.lane, st.rung) in failed_rungs):
            # free mode: a rung whose build / measure / verify failed or was skipped is neither published NOR deleted
            out(f"{head}: KEPT locally, not published (an earlier step of this rung failed) -- {st.out}")
            statuses[st.id] = "kept: upstream failed"
            kept.append((st.lane, st.rung, st.out))
            if run:
                drop_failed_stream(st, statuses, out)
            continue
        if st.lane in failed_lanes or (st.lane, st.rung) in failed_rungs:
            out(f"{head}: SKIP (an earlier step of this {'rung' if st.rung else 'lane'} failed) -- {what}")
            statuses[st.id] = "skipped: upstream failed"
            continue
        need = st.disk_gb or 0
        stream_env = None
        if run and need and st.kind in ("build", "prep") and resume_reason(st, state, redo) is None:
            free = disk_free_gb(st.out or c["mdir"]) + mirror_cache_gb(c)     # a mirror's cache gives room back
            margin = float(cfg.get("disk_margin_gb", 3))
            if free < need + margin and st.stream_to and run_upload and free >= STREAM_SHARD_GB + margin:
                import pollard_stream as PS
                repo, sub = st.stream_to
                # always onto the rung's own staging branch, never main: the repo may already be public (an earlier
                # lane went out), and an unverified shard must not be on the branch people download
                rev = PS.staging_branch(sub)
                stream_env = dict(os.environ, **{PS.SHARD_UPLOAD_ENV: json.dumps(
                    {"out": os.path.abspath(st.out), "repo": repo, "path_in_repo": sub, "revision": rev,
                     "private": repo_private(cfg)})})
                streamed.add((st.lane, st.rung))
                out(f"{head}: ~{need:.1f} GB > {free:.1f} GB free -- STREAMING: each shard written -> uploaded to "
                    f"{repo}@{rev}/{sub}/ -> confirmed -> freed before the next (main gets it only once verified)")
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
        if run and st.cmd and gpu_returned():
            # POLLARD_GPU_UNTIL passed: the GPU goes back to its owner. The streamed tools move to the CPU; work that
            # only runs on a GPU (EXL3, training a drafter) is HELD -- resumable on the next run -- never squeezed in
            if str(st.cmd[0]) in GPU_ONLY_TOOLS or st.id == "draft/train":
                msg = "held: GPU-only step and the GPU is not ours after POLLARD_GPU_UNTIL -- re-run the plan to resume"
                out(f"{head}: {msg} -- {what}")
                statuses[st.id] = msg
                if st.rung:
                    failed_rungs.add((st.lane, st.rung))
                else:
                    failed_lanes.add(st.lane)
                continue
            if str(st.cmd[0]) in DEVICE_TOOLS and "--device" not in st.cmd:
                st = Step(st, cmd=list(st.cmd) + ["--device", "cpu"])
        out(f"{head}: {what}")
        if not run:
            if st.mem:
                need = memory_need(st, c)
                if need:
                    out(f"      memory: needs ~{need:.1f} GB of RAM here (checked against free RAM before it runs)")
            statuses[st.id] = "planned"
            continue
        if st.mem:
            short = memory_short(st, c, cfg)
            if short and st.mem == "mlx-serve":           # a number nobody can measure here is said so, not faked
                record_toolcall_not_measured(st, short)
                out(f"      NOT MEASURED: {short}")
                statuses[st.id] = "done"
                state[st.id] = {"status": "done", "not_measured": short, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                save_state(c["state"], state)
                continue
            if short:
                out(f"      HELD: memory -- {short}")
                statuses[st.id] = f"held: memory ({short})"
                if st.rung:
                    failed_rungs.add((st.lane, st.rung))
                    held_mem.add((st.lane, st.rung))
                else:
                    failed_lanes.add(st.lane)
                continue
        # the machine locks cover the COMPUTE only: a build / measure / verify waits for (and holds) them; a publish,
        # upload, relay or card runs outside them -- an upload-only release used to hold the machine for hours
        from pollard_auto import heavy_step
        with (heavy_step(f"pollard release {c.get('src', '')}: {st.id}", force=bool(c.get("_force_lock")), log=out)
              if is_heavy(st) else contextlib.nullcontext()):
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
                elif st.fn == "engine_preflight":
                    held = engine_preflight(st, c, cfg, have, out)
                    failed_lanes.update(held)
                    res = "done"
                elif st.fn == "publish_rung":
                    res = publish_rung(st, c, cfg, run_upload, out)
                elif st.fn == "taskeval_rung":
                    res = taskeval_rung(st, c, run_upload, out)
                elif st.fn == "card_upload":
                    res = card_upload(st, c, run_upload, out)
                elif st.fn == "rung_card":
                    res = rung_card(st, c, cfg, run_upload, out)
                elif st.kind == "measure" and (st.lane, st.rung) in streamed and "pollard-toolcall" in str(st.cmd[0]):
                    res = _toolcall_not_measured(st, "the rung was streamed to the Hub shard by shard while it was built "
                                                     "(bigger than the free disk); the tool-call probe needs it on disk")
                    out("      tool calls: not measured (streamed rung)")
                else:
                    env = stream_env
                    if st.lane in EXL3_ENV_LANES:
                        env = exl3_env(stream_env)
                    if st.lane == "mlx" and cfg.get("disk_margin_gb") is not None:
                        # the streamed MLX tools size their disk reserve from the plan's margin (never under their floor)
                        env = dict(env or os.environ, POLLARD_DISK_MARGIN_GB=str(float(cfg["disk_margin_gb"])))
                    from pollard_auto import heavy_env
                    env = heavy_env(env)                     # a child never waits on the lock held for it
                    try:
                        run_cmd(st.cmd, st.ok_codes or (0,), **({"env": env} if env else {}))
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
                    save_entry(c["state"], st.id, state[st.id])
                    continue
                if st.lane == "prep" or stop_on_fail:
                    statuses[st.id] = res
                    state[st.id] = {"status": res, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    save_entry(c["state"], st.id, state[st.id])
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
            save_entry(c["state"], st.id, state[st.id])
    if run and kept:
        out("\n   KEPT on disk, NOT published (fix the measure / gate and re-run the plan to publish them):")
        for lane, rung, path in kept:
            out(f"     {lane}/{rung}: {path}")
    return statuses


# ------------------------------------------------------------------------------- parallel scheduler

#: lanes whose rungs build in the GPU slot (streamed HF lanes, EXL3, the drafter); everything else (the llama-quantize
#: GGUF rungs) builds in the CPU slot
GPU_SLOT_LANES = ("gptq", "mx", "exlr8", "exl3", "draft", "mlx")
#: RAM / VRAM a job may use at most, GB -- the box: 32 GB RAM, 16 GB VRAM (a little kept back for the OS / driver)
RAM_TOTAL_GB, VRAM_TOTAL_GB = 30.0, 15.0


def job_peak(st, c, cfg):
    """(ram GB, vram GB, disk GB) one step may take at its peak -- what admission control reserves for it.
    GGUF quantize: a small resident set (the tensors stream through) + its output on disk. A streamed HF-lane build:
    the plan's max_ram_gb (the streamed planner keeps its peak under it). EXL3: exllamav3 on the GPU. A GGUF measure:
    the rung's own size split over VRAM / RAM by the -ngl pollard_llamaserve picks (so its RAM is what does not fit)."""
    lane = st.lane or ""
    max_ram = float(cfg.get("max_ram_gb") or 24)
    on_gpu = not (c.get("ngl") == "0")
    if st.kind == "build" and lane == "gguf":
        return 4.0, 0.0, float(st.disk_gb or 0)
    if st.kind == "build" and lane == "exl3":
        return 16.0, VRAM_TOTAL_GB - 1, float(st.disk_gb or 0)
    if st.kind == "build":
        return max_ram, (12.0 if on_gpu else 0.0), float(st.disk_gb or 0)
    if lane == "gguf" and st.kind in ("measure", "verify", "gate"):
        size = _size_gb(st.out) if st.out else 0.0
        vram = min(size, VRAM_TOTAL_GB - 2) if on_gpu else 0.0
        return max(2.0, size - vram + 2.0), vram, 0.0
    if st.kind in ("measure", "verify", "gate"):
        return max_ram, (12.0 if on_gpu else 0.0), 0.0
    return 1.0, 0.0, 0.0


class Admission:
    """Admission control for the slots: a job starts only when its planned peak fits what is FREE now and what the
    jobs already running have reserved (RAM <= 30 GB, VRAM <= 15 GB, disk with the plan's margin). A job that does not
    fit WAITS -- never an OOM. Alone on the machine a job always starts (its own guards then decide)."""

    def __init__(self, cfg, c, free_ram=None, free_vram=None, free_disk=None, sleep=time.sleep, out=print):
        self.lock = _threading.Condition()
        self.running = {}                                # step id -> (ram, vram, disk)
        self.margin = float(cfg.get("disk_margin_gb", 3))
        self.c = c
        self._ram = free_ram or (lambda: __import__("pollard_stream").free_ram_gb())
        self._vram = free_vram or (lambda: _free_vram_mib() / 1024.0)
        self._disk = free_disk or (lambda: disk_free_gb(c["mdir"]))
        self.sleep, self.out = sleep, out

    def fits(self, peak):
        ram, vram, disk = peak
        if not self.running:
            return True
        used = [sum(x[i] for x in self.running.values()) for i in range(3)]
        if used[0] + ram > RAM_TOTAL_GB or used[1] + vram > VRAM_TOTAL_GB:
            return False
        if ram and self._ram() < ram + 1.0:
            return False
        if vram and self._vram() < vram:
            return False
        if disk and self._disk() - used[2] < disk + self.margin:
            return False
        return True

    def acquire(self, sid, peak, slot, poll_s=30):
        waited = False
        with self.lock:
            while not self.fits(peak):
                if not waited:
                    self.out(f"   [{slot}] WAIT {sid}: needs ~{peak[0]:.0f} GB RAM / {peak[1]:.0f} GB VRAM / "
                             f"{peak[2]:.0f} GB disk; running {sorted(self.running)}")
                    waited = True
                self.lock.wait(timeout=poll_s)
            self.running[sid] = peak

    def release(self, sid):
        with self.lock:
            self.running.pop(sid, None)
            self.lock.notify_all()


def rung_chains(steps):
    """[(lane, rung, [steps])] in plan order: a GGUF rung = its build + its expand; a directory-lane rung = build,
    complete gate, tool calls, verify, publish. Lane-level steps (card / upload / public) are not part of a chain."""
    chains, idx, last = [], {}, {}
    for st in steps:
        if st.kind in ("card", "upload", "finalize", "note", "hold") and not st.rung:
            continue
        # a lane-level build (the EXLR8 encode, the drafter's find / train / convert / expand) is its lane's own chain,
        # in plan order -- so it runs before that lane's rungs in the same slot
        key = (st.lane, st.rung or (st.id.split("rungs-", 1)[1] if "rungs-" in st.id else "_lane"))
        if st.kind == "gate" and not st.rung and st.lane in (st.blocks or []):
            # A whole-lane completion gate follows the build that produced its sidecars, before that rung expands.
            key = last.get(st.lane, key)
        if key not in idx:
            idx[key] = len(chains)
            chains.append((key[0], key[1], []))
        chains[idx[key]][2].append(st)
        last[st.lane] = key
    return chains


def execute_parallel(steps, c, cfg, run_upload=False, redo=(), have=None, out=print, timeline=None,
                     admission=None, runner=None):
    """`pollard release --parallel`: the shared prep ONCE (calib, imatrix, sensitivity, smoothing), then two build
    slots side by side -- CPU (GGUF quantize) and GPU (streamed GPTQ / MX / EXLR8, EXL3, the drafter), each running
    its lane's rungs back to back -- and ONE serialized measure queue (KLD / PPL, tool calls, the complete-model gate,
    ggufcheck, then upload + confirmed free) fed as each build lands. Admission control holds any job whose peak does
    not fit. The lane cards / uploads and the public flip run last, serially. Per-slot timeline -> timeline."""
    import queue
    have = have or machine()
    execution = _ExecutionState()
    runner = runner or (lambda ss: execute(ss, c, cfg, True, run_upload, redo, False, have, out, execution))
    admission = admission or Admission(cfg, c, out=out)
    tl_path = timeline or os.path.join(c["mdir"], "release-timeline.log")
    os.makedirs(os.path.dirname(tl_path), exist_ok=True)
    tl_lock = _threading.Lock()
    statuses = {}

    def log(slot, what, sid):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {slot:7s} {what:6s} {sid}"
        with tl_lock:
            with open(tl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            out(f"   [timeline] {line}")

    prep = [s for s in steps if s.lane == "prep"]
    final = [s for s in steps if s.kind in ("card", "upload", "finalize") and not s.rung]
    statuses.update(runner(prep))                     # shared prep, once; a failure stops the release (SystemExit)
    chains = rung_chains([s for s in steps if s not in prep and s not in final])
    mq = queue.Queue()
    slots = {"cpu": [], "gpu": []}
    for lane, rung, ss in chains:
        slots["gpu" if lane in GPU_SLOT_LANES else "cpu"].append((lane, rung, ss))

    def run_job(slot, sid, ss, peak):
        admission.acquire(sid, peak, slot)
        log(slot, "start", sid)
        try:
            got = runner(ss)
        except SystemExit as e:                        # one rung's hard stop is that rung's, not the release's
            got = {s.id: f"failed: {e}" for s in ss}
        finally:
            admission.release(sid)
        log(slot, "end", sid + "  " + ",".join(sorted({str(v).split(":")[0] for v in got.values()})))
        with tl_lock:
            statuses.update(got)
        return got

    pending = set()                                   # gate Events a slot is waiting on (under tl_lock)
    measure_down = _threading.Event()                 # set when the measure worker exits, for any reason
    measure_crash = []

    def skip_unmeasured(ss, why):
        with tl_lock:
            statuses.update({s.id: why for s in ss if s.id not in statuses})

    def slot_worker(slot):
        for i, (lane, rung, ss) in enumerate(slots[slot]):
            if measure_down.is_set():
                # nothing queued from here on would ever be measured: say so instead of building rungs nobody gates
                for _, _, left in slots[slot][i:]:
                    skip_unmeasured(left, "skipped: the measure worker stopped")
                break
            builds = [s for s in ss if s.kind == "build"]
            rest = [s for s in ss if s.kind != "build"]
            ok = True
            for b in builds:
                got = run_job(slot, b.id, [b], job_peak(b, c, cfg))
                ok = ok and str(got.get(b.id, "")).split(":")[0] in ("done", "skipped")
            if not ok:
                with tl_lock:
                    statuses.update({s.id: "skipped: upstream failed" for s in rest if s.kind != "publish"})
                    statuses.update({s.id: "kept: upstream failed" for s in rest if s.kind == "publish"})
                for s in rest:
                    if s.kind == "publish":
                        drop_failed_stream(s, statuses, out)
                continue
            if rest:
                checked = _threading.Event() if any(s.kind == "gate" and lane in (s.blocks or []) for s in rest) else None
                if checked:
                    with tl_lock:
                        pending.add(checked)
                mq.put((lane, rung, rest, checked))
                if measure_down.is_set():                  # put after the worker drained the queue: never measured
                    skip_unmeasured(rest, "skipped: the measure worker stopped")
                if checked:
                    # Later builds in this lane depend on its complete-model gate. Other slots can carry on. A measure
                    # worker that died never sets it: notice that rather than wait forever.
                    while not checked.wait(1.0) and not measure_down.is_set():
                        pass
                    with tl_lock:
                        pending.discard(checked)
        mq.put(None)

    def measure_worker(n_feeders):
        done_feeders = 0
        try:
            while done_feeders < n_feeders:
                item = mq.get()
                if item is None:
                    done_feeders += 1
                    continue
                lane, rung, rest, checked = item
                try:
                    peak = max((job_peak(s, c, cfg) for s in rest), key=lambda p: p[0] + p[1]) if rest else (1, 0, 0)
                    run_job("measure", f"{lane}/{rung}", rest, peak)
                except BaseException:
                    skip_unmeasured(rest, "failed: the measure worker stopped")
                    raise
                finally:
                    if checked:
                        checked.set()
        except BaseException as e:                     # noqa: BLE001 -- reported after the join, never swallowed
            measure_crash.append(e)
        finally:
            # every slot waiting on a gate wakes up; whatever is still queued will not be measured
            measure_down.set()
            with tl_lock:
                for ev in list(pending):
                    ev.set()
            while True:
                try:
                    item = mq.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    skip_unmeasured(item[2], "skipped: the measure worker stopped")
                    if item[3]:
                        item[3].set()

    feeders = [_threading.Thread(target=slot_worker, args=(k,), name=f"slot-{k}", daemon=True) for k in slots]
    meas = _threading.Thread(target=measure_worker, args=(len(feeders),), name="measure", daemon=True)
    for t in feeders + [meas]:
        t.start()
    for t in feeders + [meas]:
        t.join()
    if measure_crash:
        # no card, upload or public flip on top of rungs that were never gated
        skip_unmeasured(final, "skipped: the measure worker stopped")
        raise RuntimeError(f"the measure worker stopped: {measure_crash[0]!r}") from measure_crash[0]
    statuses.update(runner(final))
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
    ap.add_argument("--force", action="store_true", help="ignore the machine locks (the per-step compute hold)")
    ap.add_argument("--skip-setup-check", action="store_true",
                    help="run even when `pollard setup` finds a lane's dependencies missing on this machine")
    ap.add_argument("--free-after-upload", action="store_true",
                    help="build -> gate -> upload -> delete ONE rung at a time (a disk smaller than the ladder); "
                         "needs --upload and `upload: true`")
    ap.add_argument("--upload-via", metavar="SSH_URL",
                    help="upload through another machine's HF login: ssh://user@host[:port]?python="
                         "<python>&key=<identity>&min_free_gb=40[&home=<its POLLARD_HOME>] (scp into its workspace "
                         "staging/<repo>/<path>/, resumed on a rerun -> upload there -> size/sha256 checked on the "
                         "Hub -> staging deleted). Also `upload_via:` in the plan")
    ap.add_argument("--rungs-only", action="store_true",
                    help="with --free-after-upload: publish each verified rung into its lane folder and stop there -- "
                         "no card, no README upload, no visibility change (for a machine that builds one lane of a "
                         "release another machine cards)")
    ap.add_argument("--parallel", action="store_true",
                    help="two build slots (CPU: GGUF quantize; GPU: streamed HF lanes / EXL3 / drafter) + one serialized "
                         "measure queue, with admission control on RAM / VRAM / disk; shared prep once")
    ap.add_argument("--taskeval", action="store_true",
                    help="also run pollard-taskeval (lm-eval gsm8k/ifeval, ~1 h per rung) on every GGUF rung before "
                         "it is gated -- off by default: the release publishes on KLD + PPL + tool calls + coherence")
    ap.add_argument("--jev-index-only", metavar="KIT_PYTHON",
                    help="a d3 decision model, after publishing (overnight): the Decision Index of bf16 and every "
                         "verified rung via apolinario/decision-index installed in KIT_PYTHON, then the card again")
    ap.add_argument("--jev-index-limit", type=int, default=0, help="with --jev-index-only: first N requests (smoke)")
    ap.add_argument("--taskeval-only", action="store_true",
                    help="after publishing: task retention for every verified GGUF rung (its local file or a temporary "
                         "download of the published one), then regenerate + upload the card")
    a = ap.parse_args(argv)

    if a.init:
        if os.path.exists(a.init):
            raise SystemExit(f"{a.init} exists -- not overwriting")
        txt = TEMPLATE if not a.preset else TEMPLATE.replace("preset: text", f"preset: {a.preset}", 1)
        open(a.init, "w", encoding="utf-8").write(txt)
        print(f"wrote {a.init} (preset {a.preset or 'text'}) -- edit it, then: pollard release --hf <model> --plan {a.init}")
        return 0
    cfg = load_plan(a.plan, a.preset)
    cfg = auto_preset(cfg, a.hf or cfg.get("model"), a.preset, a.plan)
    if a.free_after_upload:
        if not (a.upload and cfg.get("upload")):
            ap.error("--free-after-upload deletes each rung once it is published: it needs --upload AND "
                     "`upload: true` in the plan")
        cfg["_free_after_upload"] = True
    if a.upload_via:
        cfg["upload_via"] = a.upload_via
    if a.rungs_only:
        if not a.free_after_upload:
            ap.error("--rungs-only publishes rung by rung: it needs --free-after-upload")
        cfg["_rungs_only"] = True
    src = a.hf or cfg.get("model")
    if not src:
        ap.error("pass --hf <repo-or-dir> (or set `model:` in the plan)")
    only = {x.strip() for x in a.lanes.split(",")} if a.lanes else None
    if only and only - set(LANES):
        ap.error(f"unknown lane(s) {sorted(only - set(LANES))}; choose from {LANES}")
    run = a.run and not a.dry_run
    cfg["_taskeval"] = bool(a.taskeval)
    cfg["_taskeval_only"] = bool(a.taskeval_only)
    c, lanes, steps = plan_steps(cfg, src, only)
    if a.taskeval_only:
        steps = taskeval_only_steps(cfg, c)
    if a.jev_index_only:
        steps = jev_index_only_steps(cfg, c, a.jev_index_only, [l for l in lanes if l in ("gguf", "mlx", "gptq", "mx")],
                                     a.jev_index_limit)
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
    if "mlx" in here and _has_weights(c["hf_dir"]):
        try:
            import pollard_mlx as M
            if M.needs_mlx_vlm(c["hf_dir"]):              # towers / projector / talker: mlx-vlm ships it whole
                here = ["mlx-vlm" if lane == "mlx" else lane for lane in here]
        except Exception:                                  # noqa: BLE001
            pass
    if run and not a.skip_setup_check:
        if S.require(here):
            return 2
    elif not run:
        S.hint(here, out=print)
    if run:
        # no whole-run lock: each compute step takes the machine lock (and POLLARD_HEAVY_LOCK) for its own length
        # (execute -> heavy_step), so the hours a release spends uploading never block another job
        c["_force_lock"] = bool(a.force)
        if upload_ok:
            sweep_staging(cfg, c)
        if a.parallel and not (a.taskeval_only or a.jev_index_only):
            st = execute_parallel(steps, c, cfg, upload_ok, a.redo, have)
        else:
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
