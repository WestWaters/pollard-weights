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
  mx    NVFP4 / MXFP4 / pure FP8 W8A8 (routers + lm_head kept high, Cohere's recipe)            -> <M>-Pollard-MX
  exl3  several bpw on EXL3's own allocator, smoothed + Calib 3.0 (pollard-exl3 --bpw)           -> <M>-Pollard-EXL3
  draft EAGLE3 draft head -- declared, NOT wired yet (no pollard-draft); the plan says so.

One repo per lane (the PollardWeights naming: <M>-Pollard for GGUF, <M>-Pollard-<LANE> otherwise). GGUF rungs
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
  mx:   {enabled: true, rungs: [nvfp4, mxfp4, fp8], default: nvfp4}
  exl3: {enabled: true, rungs: [2.5, 3.0, 4.0, 5.0, 6.0], default: 4.0}
  draft: {enabled: false}      # EAGLE3 draft head per rung -- not wired yet
"""

PRESETS = {
    # a plain text model: every lane, every rung
    "text": {},
    # a vision-language model: the GGUF lane ships the mmproj (pollard does that itself); MLX holds the tower
    # high; EXL3 is off because exllamav3's vision support is per-architecture -- turn it on once checked
    "vlm": {"lanes": {"exl3": {"enabled": False}},
            "measure": {"taskeval": {"suite": "quick", "limit": 200}}},
    # a decision model answers on /v1/systemone from first-position logits: llama.cpp only, judged on decisions
    "decision": {"lanes": {"mlx": {"enabled": False}, "gptq": {"enabled": False}, "mx": {"enabled": False},
                           "exl3": {"enabled": False}},
                 "measure": {"taskeval": None, "decision": True, "toolcall": False}},
    # MoE: vLLM's fused-MoE kernels want one scheme for all experts (--moe-protect-attn-only); 2/3-bit GPTQ
    # MoE kernels are poor, so the GPTQ ladder is 4 and 8
    "moe": {"lanes": {"gptq": {"rungs": [4, 8]}, "mlx": {"rungs": [3, 4, 6, 8]},
                      "mx": {"moe_protect_attn_only": True}, "exl3": {"rungs": [2.5, 3.0, 4.0]}}},
}
LANES = ["gguf", "mlx", "gptq", "mx", "exl3", "draft"]
MX_SCHEMES = {"nvfp4": "NVFP4", "mxfp4": "MXFP4", "fp8": "FP8", "fp8_dynamic": "FP8_DYNAMIC"}
REQUIRES = {"gguf": "any", "mlx": "apple", "gptq": "cuda", "mx": "cuda", "exl3": "cuda", "draft": "cuda"}


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
    <M>-Pollard-MLX / -GPTQ / -MX / -EXL3 for the others."""
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
        return MX_SCHEMES[str(rung).lower()]
    if lane == "exl3":
        return f"{float(rung):.1f}bpw"
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
    return {
        "src": src, "hf_dir": hf_dir, "mdir": mdir, "name": name, "smooth_dir": smooth,
        "calib": os.path.join(cal, "release-calib.txt"), "heldout": os.path.join(cal, "release-calib.heldout.txt"),
        "sens_raw": os.path.join(cal, f"{name}.sensitivity.json"),
        "sens_smooth": os.path.join(cal, f"{name}-smoothed.sensitivity.json"),
        "gguf_dir": gdir, "f16": os.path.join(gdir, f"{base}-f16.gguf"),
        "state": os.path.join(mdir, "release-state.json"), "release_dir": os.path.join(mdir, "release"),
        "ngl": str(cfg.get("ngl", 0)), "gates": cfg.get("gates"),
    }


def results_path(c, lane):
    return os.path.join(c["mdir"], "reports", f"results-{lane}.json")


def lane_key(c, lane):
    """The MANIFEST key each lane's tool records its builds under (what pollard-card --builds-from reads)."""
    return {"gguf": c["f16"], "mlx": c["hf_dir"]}.get(lane, c["smooth_dir"])


def _verify_gate(c):
    return ["--gate"] + ([c["gates"]] if c["gates"] else [])


def prep_steps(cfg, c, lanes):
    cal = cfg.get("calib") or {}
    smoothed = any(l in lanes for l in ("gptq", "mx", "exl3"))
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
    steps.append(Step(id="prep/calib", lane="prep", kind="prep", requires="any", cmd=calib, out=c["calib"]))
    if raw_probe:
        probe = ["pollard-probe", "--model", c["hf_dir"], "--eval", c["heldout"], "--out", c["sens_raw"]]
        if c["ngl"] == "0" and machine()["cuda"]:
            probe += ["--device", "cpu"]                 # a shared GPU is not ours (see pollard --ngl 0)
        steps.append(Step(id="prep/probe", lane="prep", kind="prep", requires="any", cmd=probe, out=c["sens_raw"]))
    if smoothed:
        # the LOCKED gold recipe for the low-bit trellis / error-feedback lanes: SmoothQuant first
        steps.append(Step(id="prep/smooth", lane="prep", kind="prep", requires="cuda",
                          cmd=["pollard-hf-smooth", "--model", c["hf_dir"], "--out", c["smooth_dir"],
                               "--calib", c["calib"]], out=c["smooth_dir"]))
        steps.append(Step(id="prep/probe-smoothed", lane="prep", kind="prep", requires="cuda",
                          cmd=["pollard-probe", "--model", c["smooth_dir"], "--eval", c["heldout"],
                               "--out", c["sens_smooth"]], out=c["sens_smooth"]))
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
TOOLCALL_BACKEND = {"gguf": "gguf", "mlx": "mlx", "gptq": "vllm", "mx": "vllm", "exl3": "exl3"}


def toolcall_cmd(c, lane, out, name, res):
    """pollard-toolcall for one directory-lane rung. The reference is the ORIGINAL bf16 checkout (not the smoothed
    copy), run the rung's own way where it can be: mlx_lm.server for MLX, vLLM for GPTQ / MX (both fall back to the
    in-process transformers backend where no usable vLLM is installed, e.g. Windows). An EXL3 rung runs in-process
    in exllamav3, so its reference runs in-process in transformers -- same template rendering, same call reader.
    When that cannot run here (no free GPU, too big) the f16 GGUF through llama-server stands in, and the result is
    labelled cross-backend. A backend this machine lacks records the rung as not measured and exits 0 -- never a
    fake number."""
    cmd = ["pollard-toolcall", "--model", out, "--backend", TOOLCALL_BACKEND[lane], "--ref", c["hf_dir"],
           "--ngl", c["ngl"], "--results", res, "--rung", name]
    if lane == "exl3":
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
    return steps


def lane_steps(cfg, c, lane):
    lc = (cfg.get("lanes") or {}).get(lane) or {}
    import pollard_workspace as ws
    if lane == "draft":
        return [Step(id="draft/note", lane="draft", kind="note", requires="any",
                     desc="EAGLE3 draft heads are declared but NOT wired: there is no pollard-draft tool yet "
                          "(llama.cpp b9606+ serves EAGLE3; train/measure one per rung when it lands).")]
    if lane == "gguf":
        cmd = ["pollard", "--hf", c["hf_dir"], "--output", c["gguf_dir"], "--calib", c["calib"],
               "--sensitivity", c["sens_raw"], "--ladder", "--run", "--no-card", "--force",
               "--no-toolcall",                          # release measures every rung itself (gguf/rungs)
               "--ngl", "auto" if c["ngl"] == "auto" else c["ngl"],
               "--calib-profile", str((cfg.get("calib") or {}).get("profile", "auto"))]
        return [Step(id="gguf/ladder", lane="gguf", kind="build", requires="any", cmd=cmd, out=c["f16"],
                     desc="convert -> imatrix -> measured publish ladder (+ IQ*_KT flagship where ik_llama builds)"),
                Step(id="gguf/rungs", lane="gguf", kind="expand", requires="any")]
    steps, res = [], results_path(c, lane)
    tool = toolcall_wanted(cfg, c)
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
        elif lane == "mx":
            scheme = MX_SCHEMES[str(rung).lower()]
            cmd = ["pollard-mx", "--model", c["smooth_dir"], "--calib", c["calib"], "--scheme", scheme, "--out", out]
            if scheme not in ("FP8", "FP8_DYNAMIC"):
                cmd += ["--sensitivity", c["sens_smooth"]]
                if lc.get("moe_protect_attn_only"):
                    cmd += ["--moe-protect-attn-only"]
            src = c["smooth_dir"]
        else:                                            # exl3
            cmd = ["pollard-exl3", "--model", c["smooth_dir"], "--bpw", str(float(rung)), "--calib-text", c["calib"],
                   "--out", out]
            src = c["smooth_dir"]
        rid = f"{lane}/{tag}"
        steps.append(Step(id=f"{rid}/build", lane=lane, rung=tag, kind="build", requires=REQUIRES[lane], cmd=cmd,
                          out=out, src=src))
        if tool:
            steps.append(Step(id=f"{rid}/toolcall", lane=lane, rung=tag, kind="measure", requires=REQUIRES[lane],
                              cmd=toolcall_cmd(c, lane, out, name, res), ok_codes=(0, 1)))
        # gated against the ORIGINAL weights, not the smoothed copy: fidelity to the model people asked for
        ver = ["pollard-verify", "--lane", lane, "--model", out, "--source", c["hf_dir"], "--rung", name,
               "--results", res] + _verify_gate(c)
        ver += ["--end-to-end"] if lane == "exl3" else ["--eval", c["heldout"]]
        steps.append(Step(id=f"{rid}/verify", lane=lane, rung=tag, kind="verify", requires=REQUIRES[lane],
                          cmd=ver, out=out))
    return steps


def publish_steps(cfg, c, lane):
    import pollard_workspace as ws
    repo = repo_name(cfg.get("org", "PollardWeights"), c["src"], lane)
    outdir = os.path.join(c["release_dir"], lane)
    card = ["pollard-card", "--model", c["src"] if not os.path.isdir(c["src"]) else c["hf_dir"],
            "--base-model", c["src"], "--lane", lane, "--builds-from", lane_key(c, lane),
            "--results", results_path(c, lane), "--repo", repo, "--out", os.path.join(outdir, "README.md"),
            "--verified-only"]
    if lane == "gguf":
        card += ["--calib-file", os.path.basename(c["calib"]),
                 "--calib-note", f"the pollard-calib '{(cfg.get('calib') or {}).get('profile', 'auto')}' profile "
                                 "(sources, revisions and licences in the shipped .sources.json)"]
    steps = [Step(id=f"{lane}/card", lane=lane, kind="card", requires="any", cmd=card, out=outdir, repo=repo)]
    steps.append(Step(id=f"{lane}/upload", lane=lane, kind="upload", requires="any", fn="upload", repo=repo,
                      out=outdir, desc=f"publish the verified {lane} rungs + card -> https://huggingface.co/{repo}"))
    return steps


def plan_steps(cfg, src, only=None):
    c = _ctx(cfg, src)
    lanes = [l for l in LANES if ((cfg.get("lanes") or {}).get(l) or {}).get("enabled")
             and (not only or l in only)]
    steps = prep_steps(cfg, c, lanes)
    for lane in lanes:
        steps += lane_steps(cfg, c, lane)
        if lane != "draft":
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
              if b.get("lane") == lane and b.get("verified") is True and os.path.exists(b.get("path", ""))]
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
    if lane == "gguf":
        for b in builds:
            name = b.get("name") or os.path.basename(b["path"])
            api.upload_file(path_or_fileobj=b["path"], path_in_repo=name, repo_id=repo)
            expect.add(name)
        for side in (c["calib"], c["calib"] + ".sources.json"):
            if os.path.exists(side):
                api.upload_file(path_or_fileobj=side, path_in_repo=os.path.basename(side), repo_id=repo)
        for f in sorted(os.listdir(c["gguf_dir"])) if os.path.isdir(c["gguf_dir"]) else []:
            if f.endswith(".imatrix") or f.startswith("mmproj-"):
                api.upload_file(path_or_fileobj=os.path.join(c["gguf_dir"], f), path_in_repo=f, repo_id=repo)
    else:
        for b in builds:
            sub = rung_subfolder(b, lane)
            api.upload_folder(folder_path=b["path"], path_in_repo=sub, repo_id=repo)
            expect.add(f"{sub}/config.json")
        root = default_build(builds, lane, ((cfg.get("lanes") or {}).get(lane) or {}).get("default"))
        api.upload_folder(folder_path=root["path"], repo_id=repo)
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo)
    missing = expect - set(api.list_repo_files(repo))
    if missing:
        raise RuntimeError(f"{repo} is missing {sorted(missing)} after upload -- not live")
    print(f"      live: https://huggingface.co/{repo}  ({len(builds)} rung(s))")
    return "uploaded"


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
        if st.kind == "expand":
            if run and have.get("any"):
                canonical_gguf(c)
            builds = gguf_builds(c)
            if builds:
                queue[0:0] = gguf_rung_steps(cfg, c, builds)
                out(f"{head}: {len(builds)} ladder rung(s) in the MANIFEST -> measure + verify each")
            elif run:
                out(f"{head}: no ladder rungs in the MANIFEST (the ladder step failed or was skipped)")
            else:
                for s in gguf_rung_steps(cfg, c, [{"tag": "<rung>", "path": "<each ladder rung .gguf>"}]):
                    out(f"     (per rung) {s.kind}: $ {_fmt_cmd(s.cmd)}")
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
        if st.lane in failed_lanes or (st.lane, st.rung) in failed_rungs:
            out(f"{head}: SKIP (an earlier step of this {'rung' if st.rung else 'lane'} failed) -- {what}")
            statuses[st.id] = "skipped: upstream failed"
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
            else:
                run_cmd(st.cmd, st.ok_codes or (0,))
                res = "done"
        except Exception as e:                                   # noqa: BLE001
            res = "failed"
            out(f"      FAILED: {e}")
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
    a = ap.parse_args(argv)

    if a.init:
        if os.path.exists(a.init):
            raise SystemExit(f"{a.init} exists -- not overwriting")
        txt = TEMPLATE if not a.preset else TEMPLATE.replace("preset: text", f"preset: {a.preset}", 1)
        open(a.init, "w", encoding="utf-8").write(txt)
        print(f"wrote {a.init} (preset {a.preset or 'text'}) -- edit it, then: pollard release --hf <model> --plan {a.init}")
        return 0
    cfg = load_plan(a.plan, a.preset)
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
        if lane == "draft":
            continue
        lc = (cfg["lanes"].get(lane) or {})
        rungs = "publish ladder (Q6_K / IQ4_XS / IQ3_S / 2-bit + flagship)" if lane == "gguf" else \
            ", ".join(rung_tag(lane, r) for r in lc.get("rungs") or [])
        print(f"   {lane:5s} -> {repo_name(cfg.get('org', 'PollardWeights'), src, lane):48s} {rungs}")
    upload_ok = bool(cfg.get("upload")) and a.upload
    print(f"   upload: {'ON' if upload_ok else 'off'}"
          + ("" if upload_ok else " (needs `upload: true` in the plan AND --upload)"))
    print(f"   state: {c['state']}\n")
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
