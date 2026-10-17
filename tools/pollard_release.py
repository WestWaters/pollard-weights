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

One repo per lane (the PollardWeights naming: <M>-Pollard for GGUF, <M>-Pollard-<LANE> otherwise). The drafter repo
holds the drafter's HF folder at its root (what vLLM / SGLang / dflash-mlx load by repo id) and its GGUF rungs beside
it (<M>-Pollard-DRAFT-<Kind>-<QUANT>.gguf, for llama.cpp -md); every target lane's card gets a Speculative decoding
section with the measured numbers and that lane's serve line. GGUF rungs
are files at the repo root, named <M>-Pollard-<TAG>.gguf like every published Pollard GGUF; a directory lane puts
each rung in its own subfolder (`4bit/`, `int4/`, `nvfp4/`, `4.0bpw/`) with the lane's default rung ALSO at the
root, so `mlx_lm.generate --model <repo>` / `vllm serve <repo>` work as-is -- the layout MiniCPM5-2B-Pollard-MLX
already ships.

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
    is built -> measured -> gated -> uploaded on its own, then its LOCAL file is deleted before the next rung (a
    rung that fails the gate is deleted unpublished). The GGUF ladder is built one rung at a time (pollard --rung).
    Every build step first checks free disk against the rung's estimated size and holds the rung (not the release)
    when it does not fit. Only files this release built are ever deleted -- never the source, f16, imatrix or calib.
  * existing inputs can be named in the plan instead of rebuilt: `f16:` (an f16/bf16 GGUF), `imatrix:`,
    `sensitivity:` (the raw-weights profile), `calib_file:` + `heldout_file:` (skip pollard-calib).
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
    # a vision-language model: the GGUF lane ships the mmproj (pollard does that itself); MLX holds the tower
    # high; EXL3 is off because exllamav3's vision support is per-architecture -- turn it on once checked
    "vlm": {"lanes": {"exl3": {"enabled": False}, "exlr8": {"enabled": False}},
            "measure": {"taskeval": {"suite": "quick", "limit": 200}}},
    # a decision model answers on /v1/systemone from first-position logits: llama.cpp only, judged on decisions
    "decision": {"lanes": {"mlx": {"enabled": False}, "gptq": {"enabled": False}, "mx": {"enabled": False},
                           "exl3": {"enabled": False}, "exlr8": {"enabled": False}},
                 "measure": {"taskeval": None, "decision": True, "toolcall": False}},
    # MoE: vLLM's fused-MoE kernels want one scheme for all experts (--moe-protect-attn-only); 2/3-bit GPTQ
    # MoE kernels are poor, so the GPTQ ladder is 4 and 8. EXLR8 is off: pollard-exlr8 writes dense records only,
    # not EXLR8's per-unit expert files yet
    "moe": {"lanes": {"gptq": {"rungs": [4, 8]}, "mlx": {"rungs": [3, 4, 6, 8]},
                      "mx": {"moe_protect_attn_only": True}, "exl3": {"rungs": [2.5, 3.0, 4.0]},
                      "exlr8": {"enabled": False}}},
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


def repo_name(org, model, lane):
    """PollardWeights naming, consistent with the repos already published: <M>-Pollard (GGUF),
    <M>-Pollard-MLX / -GPTQ / -MX / -EXL3 / -EXLR8 for the others."""
    import pollard_workspace as ws
    base = ws.model_basename(model)
    return f"{org}/{base}-Pollard" + ("" if lane == "gguf" else f"-{lane.upper()}")


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
        "org": cfg.get("org", "PollardWeights"),
    }
    # inputs that already exist (a big model's f16 / imatrix / profile / corpus): used as they are, never rebuilt
    for key, plan_key in (("f16", "f16"), ("sens_raw", "sensitivity"), ("calib", "calib_file"),
                          ("heldout", "heldout_file")):
        if cfg.get(plan_key):
            ctx[key] = os.path.abspath(str(cfg[plan_key]))
    ctx["given_f16"] = bool(cfg.get("f16"))
    return ctx


def results_path(c, lane):
    return os.path.join(c["mdir"], "reports", f"results-{lane}.json")


def lane_key(c, lane):
    """The MANIFEST key each lane's tool records its builds under (what pollard-card --builds-from reads)."""
    return {"gguf": c["f16"], "mlx": c["hf_dir"], "draft": c["src"]}.get(lane, c["smooth_dir"])


def _verify_gate(c):
    return ["--gate"] + ([c["gates"]] if c["gates"] else [])


def prep_steps(cfg, c, lanes):
    cal = cfg.get("calib") or {}
    smoothed = any(l in lanes for l in ("gptq", "mx", "exl3", "exlr8"))
    raw_probe = any(l in lanes for l in ("gguf", "mlx"))
    steps = []
    if not os.path.isdir(c["src"]):
        steps.append(Step(id="prep/fetch", lane="prep", kind="prep", requires="any", fn="fetch",
                          desc=f"pull {c['src']} -> {c['hf_dir']}"))
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
    if raw_probe and cfg.get("sensitivity"):
        steps.append(Step(id="prep/probe", lane="prep", kind="note", requires="any",
                          desc=f"sensitivity profile given in the plan: {c['sens_raw']}"))
    elif raw_probe:
        probe = ["pollard-probe", "--model", c["hf_dir"], "--eval", c["heldout"], "--out", c["sens_raw"]]
        if c["ngl"] == "0" and machine()["cuda"]:
            probe += ["--device", "cpu"]                 # a shared GPU is not ours (see pollard --ngl 0)
        steps.append(Step(id="prep/probe", lane="prep", kind="prep", requires="any", cmd=probe, out=c["sens_raw"]))
    if smoothed:
        # the LOCKED gold recipe for the low-bit trellis / error-feedback lanes: SmoothQuant first
        steps.append(Step(id="prep/smooth", lane="prep", kind="prep", requires="cuda",
                          cmd=["pollard-hf-smooth", "--model", c["hf_dir"], "--out", c["smooth_dir"],
                               "--calib", c["calib"]], out=c["smooth_dir"],
                          disk_gb=_size_gb(c["hf_dir"]) * 1.02,
                          blocks=[l for l in lanes if l in ("gptq", "mx", "exl3", "exlr8")]))
        steps.append(Step(id="prep/probe-smoothed", lane="prep", kind="prep", requires="cuda",
                          cmd=["pollard-probe", "--model", c["smooth_dir"], "--eval", c["heldout"],
                               "--out", c["sens_smooth"]], out=c["sens_smooth"],
                          blocks=[l for l in lanes if l in ("gptq", "mx", "exl3", "exlr8")]))
    return steps


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
    for b in builds:
        tag, path = b.get("tag") or b.get("name"), b.get("path")
        common = ["--results", res, "--rung", tag]
        rid = f"gguf/{tag}"
        if m.get("taskeval"):
            t = m["taskeval"]
            steps.append(Step(id=f"{rid}/taskeval", lane="gguf", rung=tag, kind="measure", requires="any",
                              cmd=["pollard-taskeval", "--model", path, "--ref", c["f16"], "--suite",
                                   str(t.get("suite", "quick")), "--limit", str(t.get("limit", 200)),
                                   "--ngl", c["ngl"], "--out", os.path.join(c["mdir"], "reports", "taskeval", tag)]
                              + common))
        if m.get("mt"):
            steps.append(Step(id=f"{rid}/mt", lane="gguf", rung=tag, kind="measure", requires="any",
                              cmd=["pollard-taskeval", "--model", path, "--ref", c["f16"], "--suite", "mt",
                                   "--limit", str((m["mt"] or {}).get("limit", 100)), "--ngl", c["ngl"]] + common))
        if m.get("toolcall"):
            steps.append(Step(id=f"{rid}/toolcall", lane="gguf", rung=tag, kind="measure", requires="any",
                              cmd=["pollard-toolcall", "--gguf", path, "--ref", c["f16"], "--ngl", c["ngl"]] + common,
                              ok_codes=(0, 1)))                        # a FAIL is a number for the gate, not a crash
        if m.get("decision"):
            steps.append(Step(id=f"{rid}/decision", lane="gguf", rung=tag, kind="measure", requires="any",
                              cmd=["pollard-decision", "--gguf", path, "--ref", c["f16"], "--ngl", c["ngl"]] + common,
                              ok_codes=(0, 1)))
        steps.append(Step(id=f"{rid}/verify", lane="gguf", rung=tag, kind="verify", requires="any", out=path,
                          cmd=["pollard-verify", "--lane", "gguf", "--model", path, "--ref", c["f16"],
                               "--eval", c["heldout"], "--ngl", c["ngl"], "--coherence"] + _verify_gate(c) + common))
        if c.get("free"):
            steps.append(_publish_rung_step(c, "gguf", tag, path))
    return steps


def _publish_rung_step(c, lane, tag, path):
    repo = repo_name(c["org"], c["src"], lane)
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
    if lane == "exlr8":
        # the ONE encode every rung of this lane is a manifest over (resumable per layer inside pollard-exlr8)
        widths = ",".join(str(int(k)) for k in (lc.get("widths") or [2, 3, 4, 5, 6]))
        steps.append(Step(id="exlr8/encode", lane="exlr8", kind="build", requires=REQUIRES["exlr8"],
                          cmd=["pollard-exlr8", "--model", c["smooth_dir"], "--store", c["exlr8_store"], "--calib",
                               c["calib"], "--widths", widths, "--encode-only"], out=c["exlr8_store"],
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
                   "--calib", c["calib"], "--bits", str(int(rung)), "--out", out]
            src = c["smooth_dir"]
        elif lane == "mx" and (mx_mix_bpw(rung) is not None or str(rung).lower() in MX6_FORMATS):
            # model-free, weight-only, CPU: no calibration set, no llm-compressor
            cmd = ["pollard-mx", "--model", c["smooth_dir"], "--sensitivity", c["sens_smooth"], "--out", out]
            cmd += (["--mix", f"{mx_mix_bpw(rung):g}"] if mx_mix_bpw(rung) is not None
                    else ["--format", MX6_FORMATS[str(rung).lower()]])
            src = c["smooth_dir"]
        elif lane == "mx":
            scheme = MX_SCHEMES[str(rung).lower()]
            cmd = ["pollard-mx", "--model", c["smooth_dir"], "--calib", c["calib"], "--scheme", scheme, "--out", out]
            if scheme not in ("FP8", "FP8_DYNAMIC"):
                cmd += ["--sensitivity", c["sens_smooth"]]
                if lc.get("moe_protect_attn_only"):
                    cmd += ["--moe-protect-attn-only"]
            src = c["smooth_dir"]
        elif lane == "exlr8":                            # a manifest over the encode: seconds, no re-encode
            cmd = ["pollard-exlr8", "--model", c["smooth_dir"], "--store", c["exlr8_store"], "--calib", c["calib"],
                   "--sensitivity", c["sens_smooth"], "--widths", ",".join(str(int(k)) for k in (lc.get("widths") or [2, 3, 4, 5, 6])),
                   "--bpw", str(float(rung)), "--out", out]
            src = c["smooth_dir"]
        else:                                            # exl3
            cmd = ["pollard-exl3", "--model", c["smooth_dir"], "--bpw", str(float(rung)), "--calib-text", c["calib"],
                   "--out", out]
            src = c["smooth_dir"]
        rid = f"{lane}/{tag}"
        steps.append(Step(id=f"{rid}/build", lane=lane, rung=tag, kind="build", requires=REQUIRES[lane], cmd=cmd,
                          out=out, src=src, disk_gb=_rung_gb(lane, rung, c["hf_dir"]) if c.get("free") else 0))
        if tool:
            steps.append(Step(id=f"{rid}/toolcall", lane=lane, rung=tag, kind="measure", requires=REQUIRES[lane],
                              cmd=toolcall_cmd(c, lane, out, name, res), ok_codes=(0, 1)))
        # gated against the ORIGINAL weights, not the smoothed copy: fidelity to the model people asked for
        ver = ["pollard-verify", "--lane", lane, "--model", out, "--source", c["hf_dir"], "--rung", name,
               "--results", res] + _verify_gate(c)
        ver += ["--end-to-end"] if lane == "exl3" else ["--eval", c["heldout"]]
        steps.append(Step(id=f"{rid}/verify", lane=lane, rung=tag, kind="verify", requires=REQUIRES[lane],
                          cmd=ver, out=out))
        if c.get("free"):
            st = _publish_rung_step(c, lane, tag, out)
            st["default"] = lc.get("default") is not None and rung_tag(lane, lc["default"]) == tag
            steps.append(st)
    return steps


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


def publish_steps(cfg, c, lane):
    import pollard_workspace as ws
    repo = repo_name(cfg.get("org", "PollardWeights"), c["src"], lane)
    outdir = os.path.join(c["release_dir"], lane)
    card = ["pollard-card", "--model", c["src"] if not os.path.isdir(c["src"]) else c["hf_dir"],
            "--base-model", c["src"], "--lane", lane, "--builds-from", lane_key(c, lane),
            "--results", results_path(c, lane), "--repo", repo, "--out", os.path.join(outdir, "README.md"),
            "--verified-only"]
    if lane == "draft":
        card = ["pollard-draft", "card", "--target", c["src"], "--key", c["src"], "--results", results_path(c, lane),
                "--repo", repo, "--org", cfg.get("org", "PollardWeights"), "--out", os.path.join(outdir, "README.md"),
                "--verified-only", "--lanes", ",".join(l for l in LANES if l not in ("draft", "exl3", "exlr8") and
                                                      ((cfg.get("lanes") or {}).get(l) or {}).get("enabled"))]
    elif _draft_on(cfg):
        card += ["--draft-results", results_path(c, "draft"),
                 "--draft-repo", repo_name(cfg.get("org", "PollardWeights"), c["src"], "draft")]
    if lane == "gguf":
        card += ["--calib-file", os.path.basename(c["calib"]),
                 "--calib-note", f"the pollard-calib '{(cfg.get('calib') or {}).get('profile', 'auto')}' profile "
                                 "(sources, revisions and licences in the shipped .sources.json)"]
    if c.get("free") and lane != "draft":
        card += ["--runtime-from-repo"]                  # freed rungs are read from their published copy
    steps = [Step(id=f"{lane}/card", lane=lane, kind="card", requires="any", cmd=card, out=outdir, repo=repo)]
    steps.append(Step(id=f"{lane}/upload", lane=lane, kind="upload", requires="any", fn="upload", repo=repo,
                      out=outdir, desc=f"publish the verified {lane} rungs + card -> https://huggingface.co/{repo}"))
    return steps


def plan_steps(cfg, src, only=None):
    c = _ctx(cfg, src)
    lanes = [l for l in LANES if ((cfg.get("lanes") or {}).get(l) or {}).get("enabled")
             and (not only or l in only)]
    steps = prep_steps(cfg, c, lanes)
    # the drafter is measured against the GGUF rungs and every lane's card quotes it: build + gate it right after the
    # GGUF ladder, before any card is written
    early_draft = "draft" in lanes and "gguf" in lanes
    for lane in lanes:
        if lane != "draft" or not early_draft:
            steps += lane_steps(cfg, c, lane)
        if lane == "gguf" and early_draft:
            steps += lane_steps(cfg, c, "draft")
        steps += publish_steps(cfg, c, lane)
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
    ws.fetch_source(c["src"])


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


def upload(step, c, cfg, run_upload):
    """Publish the verified rungs of one lane + its card. GGUF: files at the root. Directory lanes: one subfolder
    per rung plus the default rung at the root. Refuses a lane with no verified rung, and checks the repo
    afterwards: a repo without its README is not live."""
    import pollard_workspace as ws
    from pollard_card import rung_subfolder
    lane, repo = step.lane, step.repo
    builds = [b for b in ws.read_manifest(lane_key(c, lane)).get("builds", [])
              if b.get("lane") == lane and b.get("verified") is True
              and (os.path.exists(b.get("path", "")) or b.get("uploaded") == repo)]
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
    api.create_repo(repo, repo_type="model", exist_ok=True)
    expect = {"README.md"}
    if lane == "draft":
        for b in builds:
            if b.get("publish") is False:
                continue
            if b.get("format") == "gguf":
                name = b.get("name") or os.path.basename(b["path"])
                api.upload_file(path_or_fileobj=b["path"], path_in_repo=name, repo_id=repo)
                expect.add(name)
            elif b.get("format") == "hf":                      # the HF drafter at the root: vLLM / SGLang / MLX
                api.upload_folder(folder_path=b["path"], repo_id=repo)
                expect.add("config.json")
    elif lane == "gguf":
        for b in builds:
            name = b.get("uploaded_as") or b.get("name") or os.path.basename(b["path"])
            if b.get("uploaded") != repo:                      # --free-after-upload already put it there
                api.upload_file(path_or_fileobj=b["path"], path_in_repo=name, repo_id=repo)
            expect.add(name)
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
            sub = rung_subfolder(b, lane)
            if b.get("uploaded") != repo:
                api.upload_folder(folder_path=b["path"], path_in_repo=sub, repo_id=repo)
            expect.add(f"{sub}/config.json")
        if not any(b.get("root") for b in builds):
            local = [b for b in builds if os.path.exists(b.get("path", ""))]
            root = default_build(local, lane, ((cfg.get("lanes") or {}).get(lane) or {}).get("default"))
            if root:
                api.upload_folder(folder_path=root["path"], repo_id=repo)
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo)
    missing = expect - set(api.list_repo_files(repo))
    if missing:
        raise RuntimeError(f"{repo} is missing {sorted(missing)} after upload -- not live")
    print(f"      live: https://huggingface.co/{repo}  ({len(builds)} rung(s))")
    return "uploaded"


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
    return bool(b.get("flagship")) and os.path.dirname(path) == os.path.normcase(
        os.path.dirname(os.path.abspath(c["f16"]))) and path.endswith("-mix.gguf")


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


def publish_rung(step, c, cfg, run_upload, out=print):
    """--free-after-upload: upload ONE gated rung to its lane repo, check it landed, mark it uploaded in the MANIFEST,
    then delete the local copy. A rung that did not pass the gate is not published; its file is freed too (its
    numbers stay in the results file)."""
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
    if b.get("uploaded"):
        return "done"
    if b.get("verified") is not True:
        _free_local(c, b, out)
        b["freed"] = "did not pass the gate -- not published"
        ws._write_manifest(key, man)
        return "done"
    if not run_upload:
        raise RuntimeError("--free-after-upload needs uploading on (`upload: true` + --upload)")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    if lane == "gguf":
        name = b.get("name") or os.path.basename(b["path"])
        api.upload_file(path_or_fileobj=b["path"], path_in_repo=name, repo_id=repo)
        landed = name in set(api.list_repo_files(repo))
        b["uploaded_as"] = name
    else:
        sub = rung_subfolder(b, lane)
        api.upload_folder(folder_path=b["path"], path_in_repo=sub, repo_id=repo)
        if step.default:
            api.upload_folder(folder_path=b["path"], repo_id=repo)
            b["root"] = True
        landed = f"{sub}/config.json" in set(api.list_repo_files(repo))
        b["uploaded_as"] = sub
    if not landed:
        raise RuntimeError(f"{b['uploaded_as']} is not in {repo} after the upload -- keeping the local copy")
    b["uploaded"] = repo
    ws._write_manifest(key, man)
    out(f"      uploaded {b['uploaded_as']} -> https://huggingface.co/{repo}")
    if _free_local(c, b, out):
        b["freed"] = "uploaded"
        ws._write_manifest(key, man)
    return "done"


def run_cmd(cmd, ok_codes=(0,)):
    from pollard_auto import _resolve_tool
    r = subprocess.run(_resolve_tool(list(cmd)))
    if r.returncode not in ok_codes:
        raise RuntimeError(f"exit {r.returncode}")


def execute(steps, c, cfg, run=False, run_upload=False, redo=(), stop_on_fail=False, have=None, out=print):
    """Print (dry-run) or run the plan, serially. Returns {step id: status}."""
    have = have or machine()
    state = load_state(c["state"])
    statuses, failed_rungs, failed_lanes, unbuildable = {}, set(), set(), set()
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
        if st.kind == "publish" and (st.lane, st.rung) in failed_rungs and st.lane not in failed_lanes:
            pass                                     # free mode: a rung that failed is still freed (not published)
        elif st.lane in failed_lanes or (st.lane, st.rung) in failed_rungs:
            out(f"{head}: SKIP (an earlier step of this {'rung' if st.rung else 'lane'} failed) -- {what}")
            statuses[st.id] = "skipped: upstream failed"
            continue
        need = st.disk_gb or 0
        if run and need and st.kind in ("build", "prep") and resume_reason(st, state, redo) is None:
            free = disk_free_gb(st.out or c["mdir"])
            margin = float(cfg.get("disk_margin_gb", 3))
            if free < need + margin:
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
            elif st.fn == "publish_rung":
                res = publish_rung(st, c, cfg, run_upload, out)
            else:
                run_cmd(st.cmd, st.ok_codes or (0,))
                res = "done"
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
        if res != "held":
            state[st.id] = {"status": res, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            save_state(c["state"], state)
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
