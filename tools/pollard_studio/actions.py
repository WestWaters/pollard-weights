#!/usr/bin/env python3
"""What each button in Studio actually runs.

One source of truth, in Python, next to the thing that executes it -- so the command the UI prints
and the command that runs cannot drift apart.

EVERY FLAG HERE EXISTS. They were read out of the tools' own argparse definitions (see
manifest.py), not remembered. A control in the UI that has no flag behind it is a control that
lies about what the build will do, which is worse than not offering it.

    python actions.py build '{"ram": 16}'      # print the argv, run nothing
"""
from __future__ import annotations

import json
import sys

# lane -> the tool that emits it
LANE_TOOL = {"GGUF": "pollard_fit", "GPTQ": "pollard_gptq", "MLX": "pollard_mlx",
             "EXL3": "pollard_exl3", "MX": "pollard_mx"}


def _opt(args: list, flag: str, value) -> None:
    """Append `--flag value`, but only when there is a value to append."""
    if value not in (None, "", False):
        args += [flag, str(value)]


def _switch(args: list, flag: str, on) -> None:
    if on:
        args.append(flag)


def _frac(v):
    """The UI holds hot-frac as a percentage; the tools take 0..1."""
    if v in (None, "", False):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f / 100, 4) if f > 1 else round(f, 4)


def build(r: dict) -> tuple[str, list]:
    """pollard-fit: solve an allocation against a memory budget."""
    a: list[str] = []
    _opt(a, "--gguf", r.get("source"))
    _opt(a, "--ram", r.get("ram"))
    _opt(a, "--reserve", r.get("reserve"))
    _opt(a, "--imatrix", r.get("imatrix"))
    _opt(a, "--sensitivity", r.get("sensitivity"))
    _opt(a, "--tier", r.get("tier"))
    _opt(a, "--out", r.get("out"))
    _switch(a, "--plan-only", r.get("planOnly", True))
    _switch(a, "--allow-1bit", r.get("allow1bit"))
    _switch(a, "--allow-grow", r.get("allowGrow"))
    _opt(a, "--threads", r.get("threads"))
    return "pollard_fit", a


def automap(r: dict) -> tuple[str, list]:
    """pollard-automap: generate the per-tensor recipe."""
    a: list[str] = []
    _opt(a, "--model", r.get("source"))
    _opt(a, "--body", r.get("body"))
    _opt(a, "--protect", r.get("protect"))
    _opt(a, "--imatrix", r.get("imatrix"))
    _opt(a, "--fragile", r.get("fragileScan"))
    _opt(a, "--eval", r.get("evalDir"))
    _opt(a, "--ngl", r.get("ngl", 0))
    _opt(a, "--rival", r.get("rival"))
    _opt(a, "--out", r.get("out"))
    _switch(a, "--mix-only", r.get("mixOnly"))
    _switch(a, "--no-imatrix", r.get("noImatrix"))
    _switch(a, "--no-gate", r.get("noGate"))
    _switch(a, "--allow-dense", r.get("allowDense"))
    return "pollard_automap", a


def lane(r: dict) -> tuple[str, list]:
    """Emit into whichever lane the LANE knob is on, routed by what the source actually is.

    pollard-fit takes --gguf and only --gguf. Pointing it at a safetensors directory builds a
    command that looks right and cannot run, so a non-GGUF source goes to convert (for the GGUF
    lane) or straight to the lane's own tool, which takes --model.
    """
    want = r.get("lane", "GGUF")
    src = str(r.get("source") or "")
    is_gguf = src.endswith(".gguf")

    # An unrecognised lane used to fall back to pollard_fit while still passing --model, which
    # fit does not take -- a command that looks right and argparse rejects. GGUF is the honest
    # default, and it routes on what the source actually is.
    if want not in LANE_TOOL:
        want = "GGUF"

    if want == "GGUF":
        if is_gguf:
            return build(r)
        a: list[str] = []                      # safetensors -> f16 GGUF first
        _opt(a, "--model", src)
        _opt(a, "--outfile", r.get("out"))     # convert says --outfile, not --out
        _opt(a, "--outtype", r.get("outtype"))
        return "pollard_convert", a

    tool = LANE_TOOL[want]
    a = []
    _opt(a, "--model", src)
    _opt(a, "--out", r.get("out"))

    # THE POINT OF THE WHOLE THING: one measured allocation, emitted into any lane. Without this
    # every lane but GGUF was built on its tool's defaults, which throws away the measurement.
    # Each lane takes the profile in its own form -- MLX/MX read sensitivity.json directly, EXL3
    # wants a per-tensor YAML. GPTQ has no profile-file input at all; its --recipe is a named mix.
    if tool in ("pollard_mlx", "pollard_mx"):
        _opt(a, "--sensitivity", r.get("sensitivity"))
    elif tool == "pollard_exl3":
        _opt(a, "--recipe", r.get("recipeFile"))

    # and the lane's own knobs, each declared by that lane's tool
    if tool == "pollard_mlx":
        _opt(a, "--group-size", r.get("groupSize"))
        _opt(a, "--hot-frac", _frac(r.get("hotFrac")))
        _opt(a, "--focus-layers", r.get("focusLayers"))
    elif tool == "pollard_mx":
        _opt(a, "--scheme", r.get("scheme"))
        _opt(a, "--protect-scheme", r.get("protectScheme"))
        _opt(a, "--hot-frac", _frac(r.get("hotFrac")))
        _opt(a, "--focus-layers", r.get("focusLayers"))
    elif tool == "pollard_exl3":
        _opt(a, "--bpw", r.get("bpw"))
        _opt(a, "--head-bits", r.get("headBits"))
        _opt(a, "--devices", r.get("devices"))
    elif tool == "pollard_gptq":
        _opt(a, "--bits", r.get("bits"))
        _opt(a, "--groupsize", r.get("groupsize"))
        if (r.get("qmode") or "int") != "int":
            _opt(a, "--qmode", r.get("qmode"))
    return tool, a


def bench(r: dict) -> tuple[str, list]:
    a: list[str] = []
    _opt(a, "--gguf", r.get("gguf"))
    _opt(a, "--vs", r.get("vs"))
    _opt(a, "--ref", r.get("ref"))
    _opt(a, "--eval", r.get("evalFile"))
    _opt(a, "--chunks", r.get("chunks"))
    _opt(a, "--ngl", r.get("ngl", 0))          # 0 unless the box is genuinely free
    _switch(a, "--speed", r.get("speed", True))
    _switch(a, "--coherence", r.get("coherence"))
    _switch(a, "--quick", r.get("quick"))
    _opt(a, "--threads", r.get("threads"))
    return "pollard_bench", a


def evaluate(r: dict) -> tuple[str, list]:
    """pollard-eval: trajectory divergence against a reference build."""
    a: list[str] = []
    _opt(a, "--ref", r.get("ref"))
    _opt(a, "--quants", r.get("gguf"))
    _opt(a, "--eval", r.get("evalFile"))
    _opt(a, "--prompts", r.get("promptsFile"))     # your own prompts for trajectory mode
    _opt(a, "--rpc", r.get("rpc"))                 # pool the scoring across linked boxes
    _opt(a, "--gen-tokens", r.get("genTokens"))
    _opt(a, "--out", r.get("out"))
    _switch(a, "--trajectory", r.get("trajectory", True))
    _switch(a, "--chart", r.get("chart"))
    return "pollard_eval", a


def doctor(r: dict) -> tuple[str, list]:
    a: list[str] = []
    _opt(a, "--model", r.get("gguf"))
    _opt(a, "--source", r.get("source"))
    _opt(a, "--lane", (r.get("lane") or "gguf").lower())
    _switch(a, "--predict", r.get("predict", True))
    return "pollard_doctor", a


def smoke(r: dict) -> tuple[str, list]:
    a: list[str] = []
    _opt(a, "--model", r.get("source"))
    _opt(a, "--gguf", r.get("gguf"))
    _opt(a, "--imatrix", r.get("imatrix"))
    _opt(a, "--ftype", r.get("ftype"))
    _opt(a, "--token-embedding-type", r.get("emb"))
    return "pollard_smoke", a


def ggufcheck(r: dict) -> tuple[str, list]:
    return "pollard_ggufcompat", ([r["gguf"]] if r.get("gguf") else [])


def verify(r: dict) -> tuple[str, list]:
    a: list[str] = []
    _opt(a, "--model", r.get("gguf"))
    _opt(a, "--source", r.get("source"))
    _opt(a, "--lane", (r.get("lane") or "gguf").lower())
    return "pollard_verify", a


def card(r: dict) -> tuple[str, list]:
    a: list[str] = []
    _opt(a, "--model", r.get("modelName"))
    _opt(a, "--builds-from", r.get("dir"))
    _opt(a, "--out", r.get("out"))
    return "pollard_card", a


def fragile(r: dict) -> tuple[str, list]:
    a: list[str] = []
    _opt(a, "--gguf", r.get("gguf") or r.get("source"))
    _opt(a, "--top", r.get("top"))
    _opt(a, "--protect", r.get("protect"))
    _opt(a, "--out", r.get("out"))
    return "pollard_fragile", a


def train(r: dict) -> tuple[str, list]:
    """pollard-gptq: the full-Hessian error-feedback solver. Pollard trains as well as quantizes."""
    a: list[str] = []
    _opt(a, "--model", r.get("source"))
    _opt(a, "--method", r.get("method"))
    _opt(a, "--bits", r.get("bits"))
    _opt(a, "--groupsize", r.get("groupsize"))
    _opt(a, "--nsamples", r.get("nsamples"))
    _opt(a, "--seqlen", r.get("seqlen"))
    _opt(a, "--head-bits", r.get("headBits"))
    _opt(a, "--embed-bits", r.get("embedBits"))
    _opt(a, "--calib-file", r.get("calibFile"))      # train on YOUR corpus, not the HF datasets
    _opt(a, "--eval-file", r.get("trainEvalFile"))
    _opt(a, "--eval-chunks", r.get("evalChunks"))
    _opt(a, "--device", r.get("device"))
    _opt(a, "--threads", r.get("threads"))
    _opt(a, "--work-dir", r.get("workDir"))
    # these three are choice flags whose default IS "none"/"int" -- passing the default is noise,
    # and --recipe/--ablate only mean anything on the gptq-seq methods
    if (r.get("qmode") or "int") != "int":
        _opt(a, "--qmode", r.get("qmode"))
    if (r.get("recipe") or "none") != "none":
        _opt(a, "--recipe", r.get("recipe"))
    if (r.get("ablate") or "none") != "none":
        _opt(a, "--ablate", r.get("ablate"))
    _switch(a, "--resume", r.get("resume"))
    _switch(a, "--offload", r.get("offload"))
    return "pollard_gptq", a


def mmeval(r: dict) -> tuple[str, list]:
    """pollard-mmeval: did quantization break the model's eyes (or ears)?"""
    a: list[str] = []
    _opt(a, "--gguf", r.get("gguf"))
    _opt(a, "--mmproj", r.get("mmproj"))
    _opt(a, "--ref", r.get("ref"))
    _opt(a, "--out", r.get("out"))
    return "pollard_mmeval", a


def taskeval(r: dict) -> tuple[str, list]:
    """pollard-taskeval: what the quantization costs on real tasks, not just on the logits."""
    a: list[str] = []
    _opt(a, "--model", r.get("gguf"))          # taskeval says --model, not --gguf
    _opt(a, "--ref", r.get("ref"))
    _opt(a, "--suite", r.get("suite"))
    _opt(a, "--tasks", r.get("tasks"))         # explicit lm-eval task names, overrides --suite
    _opt(a, "--limit", r.get("limit"))
    _opt(a, "--ngl", r.get("ngl", 0))
    _opt(a, "--out", r.get("out"))
    return "pollard_taskeval", a


def probes(r: dict) -> tuple[str, list]:
    """pollard-probes: task accuracy on YOUR benchmark file.

    The three datafile flags are mutually exclusive in the tool -- whichever one is set picks
    the format -- so only the one the user chose is passed.
    """
    a: list[str] = []
    _opt(a, "--gguf", r.get("gguf"))
    _opt(a, "--label", r.get("label"))
    _opt(a, "--tasks", r.get("probeTasks"))
    fmt = (r.get("probeFormat") or "hellaswag").lower()
    data = r.get("probeData")
    if data:
        _opt(a, {"hellaswag": "--hellaswag-data",
                 "winogrande": "--winogrande",
                 "multiple-choice": "--multiple-choice"}.get(fmt, "--hellaswag-data"), data)
    _opt(a, "--ngl", r.get("ngl", 0))
    _opt(a, "--ctx", r.get("ctx"))
    _opt(a, "--out-dir", r.get("out"))
    return "pollard_probes", a


def placement(r: dict) -> tuple[str, list]:
    """pollard-run: measured expert placement across whatever hardware you actually have.

    --rpc is how a model bigger than any one box gets built and served: ggml-rpc-server on each
    node, pooled here. Pollard is not capped on model size; the cluster is the cap.
    """
    a: list[str] = []
    _opt(a, "--gguf", r.get("gguf"))
    _opt(a, "--profile", r.get("heatProfile"))
    _opt(a, "--vram", r.get("vram"))
    _opt(a, "--rpc", r.get("rpc"))
    # -ngl and the per-device split ride in --extra, which pollard-run appends verbatim after its
    # own "-ngl 999" -- llama.cpp takes the last one, so these win.
    extra = " ".join(str(x) for x in (r.get("placement") or []))
    both = " ".join(x for x in (extra, r.get("extraArgs") or "") if x).strip()
    _opt(a, "--extra", both)
    return "pollard_run", a


def vllmcheck(r: dict) -> tuple[str, list]:
    """pollard-vllm: will this build serve under vLLM, and at which tensor-parallel degree?"""
    a: list[str] = []
    _opt(a, "--model", r.get("gguf") or r.get("source"))
    _opt(a, "--tp", r.get("tp"))
    _opt(a, "--max-len", r.get("maxLen"))
    _switch(a, "--serve", r.get("serveCmd"))
    return "pollard_vllm", a


def failmode(r: dict) -> tuple[str, list]:
    """pollard-failmode: is this build's damage repairable, or did a component fail outright?"""
    a: list[str] = []
    _opt(a, "--ref", r.get("ref") or r.get("source"))
    _opt(a, "--model", r.get("gguf"))
    _opt(a, "--calib", r.get("calibFile"))
    _opt(a, "--device", r.get("device"))
    _opt(a, "--out", r.get("out"))
    return "pollard_failmode", a


def toolcall(r: dict) -> tuple[str, list]:
    """pollard-toolcall: can the build still emit a valid tool call?"""
    a: list[str] = []
    _opt(a, "--gguf", r.get("gguf"))
    _opt(a, "--ref", r.get("ref"))
    _opt(a, "--ngl", r.get("ngl", 0))
    _opt(a, "--threads", r.get("threads"))
    _opt(a, "--min-rate", r.get("minRate"))
    _opt(a, "--out", r.get("out"))
    return "pollard_toolcall", a


def kvsweep(r: dict) -> tuple[str, list]:
    """pollard-bench --kv-sweep: what each KV cache precision costs on this model."""
    a: list[str] = []
    _opt(a, "--gguf", r.get("gguf"))
    # the sweep has its own corpus field; fall back to the one the eval screen set
    _opt(a, "--eval", r.get("kvEvalFile") or r.get("evalFile"))
    _opt(a, "--kv-ctx", r.get("kvCtx"))
    _opt(a, "--chunks", r.get("chunks"))
    _opt(a, "--ngl", r.get("ngl", 0))
    _opt(a, "--threads", r.get("threads"))
    _opt(a, "--out", r.get("out"))
    a.append("--kv-sweep")
    return "pollard_bench", a


def routecheck(r: dict) -> tuple[str, list]:
    """pollard-routecheck: did quantization change which experts fire?

    MoE only. Perplexity cannot see a routing change, because the model still produces plausible
    text -- with a different set of experts than the one that was measured.
    """
    a: list[str] = []
    _opt(a, "--ref", r.get("ref") or r.get("source"))
    _opt(a, "--model", r.get("gguf"))
    _opt(a, "--calib", r.get("calibFile"))
    _opt(a, "--nsamples", r.get("nsamples"))
    _opt(a, "--top-k", r.get("topK"))
    _opt(a, "--swap-budget", r.get("swapBudget"))
    _opt(a, "--out", r.get("out"))
    return "pollard_routecheck", a


# --- the levers, the diagnostics and the emitters -----------------------------------------------
# These were reachable only through the generic Tools screen. Everything Pollard does should have
# a place where it belongs, not just a place where it can be found.

# Each of these takes the flags ITS OWN argparse declares. A generic "--model/--out" helper was
# tried and test_every_flag_exists rejected nineteen of them at once -- pollard-rotate wants
# --gguf, pollard-scorecard wants --results, pollard-brainattach wants --brain and --build.
# Guessing a common shape across 64 tools is how a UI ends up printing commands that cannot run.

def probe(r):
    a = []
    _opt(a, "--model", r.get("source") or r.get("gguf"))
    _opt(a, "--eval", r.get("evalFile"))
    _opt(a, "--out", r.get("out"))
    return "pollard_probe", a


def sensitivity(r):
    a = []
    _opt(a, "--gguf", r.get("gguf") or r.get("source"))
    _opt(a, "--imatrix", r.get("imatrix"))
    _opt(a, "--eval", r.get("evalFile"))
    _opt(a, "--rpc", r.get("rpc"))          # pool the forward passes across linked boxes
    _opt(a, "--ngl", r.get("ngl"))
    _opt(a, "--out", r.get("out"))
    return "pollard_sensitivity", a


def precondition(r):
    a = []
    _opt(a, "--gguf", r.get("gguf") or r.get("source"))
    _opt(a, "--imatrix", r.get("imatrix"))
    _opt(a, "--calib", r.get("calibFile"))
    _opt(a, "--eval", r.get("evalFile"))
    _opt(a, "--target", r.get("target"))
    return "pollard_precondition", a


def rotate(r):
    a = []
    _opt(a, "--gguf", r.get("gguf") or r.get("source"))
    _opt(a, "--out", r.get("out"))
    _opt(a, "--kind", r.get("rotKind"))
    return "pollard_rotate", a


def smooth(r):
    a = []
    _opt(a, "--gguf", r.get("gguf") or r.get("source"))
    _opt(a, "--imatrix", r.get("imatrix"))
    _opt(a, "--out", r.get("out"))
    return "pollard_smooth", a


def hf_smooth(r):
    a = []
    _opt(a, "--model", r.get("source"))
    _opt(a, "--calib", r.get("calibFile"))
    _opt(a, "--out", r.get("out"))
    return "pollard_hf_smooth", a


def palette(r):
    a = []
    _opt(a, "--model", r.get("source"))
    _opt(a, "--calib-file", r.get("calibFile"))
    _opt(a, "--eval-file", r.get("evalFile"))
    return "pollard_palette", a


def lowbit(r):
    a = []
    _opt(a, "--model", r.get("source"))
    _opt(a, "--calib-file", r.get("calibFile"))
    _opt(a, "--eval-file", r.get("evalFile"))
    _opt(a, "--bits", r.get("bits"))
    return "pollard_lowbit", a


def experts(r):
    a = []
    _opt(a, "--jsonl", r.get("routeCapture"))
    _opt(a, "--out", r.get("out"))
    return "pollard_experts", a


def route(r):
    a = []
    _opt(a, "--model", r.get("gguf") or r.get("source"))
    _opt(a, "--prompts", r.get("promptsFile"))
    _opt(a, "--out", r.get("out"))
    return "pollard_route", a


def prune(r):
    a = []
    _opt(a, "--gguf", r.get("gguf"))
    _opt(a, "--imatrix", r.get("imatrix"))
    _opt(a, "--out", r.get("out"))
    return "pollard_prune", a


def modelkind(r):
    a = []
    _opt(a, "--model", r.get("gguf") or r.get("source"))
    return "pollard_modelkind", a


def archfp(r):
    a = []
    _opt(a, "--gguf", r.get("gguf"))
    return "pollard_archfp", a


def refcheck(r):
    a = []
    _opt(a, "--model", r.get("source") or r.get("gguf"))
    _opt(a, "--calib", r.get("calibFile"))
    return "pollard_refcheck", a


def scorecard(r):
    a = []
    _opt(a, "--results", r.get("results"))
    _opt(a, "--out", r.get("out"))
    return "pollard_scorecard", a


def recard(r):
    a = []
    _opt(a, "--repo", r.get("repo"))
    _opt(a, "--out", r.get("out"))
    return "pollard_recard", a


def reclaim(r):
    a = []
    _opt(a, "--home", r.get("home"))
    _switch(a, "--scan", True)               # never --delete from a button
    return "pollard_reclaim", a


def brainattach(r):
    a = []
    _opt(a, "--brain", r.get("brain"))
    _opt(a, "--build", r.get("gguf"))
    return "pollard_brainattach", a


def brainverify(r):
    a = []
    _opt(a, "--brain", r.get("brain"))
    _opt(a, "--model", r.get("gguf") or r.get("source"))
    _opt(a, "--filler", r.get("filler"))
    return "pollard_brainverify", a


def brainlanes(r):
    a = []
    _opt(a, "--model", r.get("gguf"))
    return "pollard_brainlanes", a


def connectome(r):
    a = []
    _opt(a, "--out", r.get("out"))
    return "pollard_connectome", a


def flybrain(r):
    a = []
    _opt(a, "--brain", r.get("brain"))
    _opt(a, "--model", r.get("source") or r.get("gguf"))
    _opt(a, "--probes", r.get("probesFile"))
    return "pollard_flybrain", a


def envmatch(r):
    a = []
    _opt(a, "--model", r.get("source"))
    _opt(a, "--lane", (r.get("lane") or "gguf").lower())
    return "pollard_envmatch", a


def health(_r):
    return "pollard_health", []


def runtime(_r):
    return "pollard_runtime", []


def stop(_r):
    return "pollard_stop", ["--watch"]       # report, do not kill, from a button


def calc(r):
    a = []
    _opt(a, "--gguf", r.get("gguf"))
    _opt(a, "--ram", r.get("ram"))
    return "pollard_calc", a


def emit(r):
    """Emit into whichever lane the Convert screen is pointed at."""
    return lane(r)


def calib(r):
    """pollard-calib: a multi-domain calibration corpus with an honest held-out split."""
    a = []
    _opt(a, "--out", r.get("out"))
    _opt(a, "--domains", r.get("domains"))
    _opt(a, "--per-domain", r.get("perDomain"))
    _opt(a, "--held-out", r.get("heldOut"))
    _opt(a, "--held-frac", (r.get("heldFrac") / 100) if r.get("heldFrac") else None)
    return "pollard_calib", a


def ls(_r: dict) -> tuple[str, list]:
    return "pollard_ls", ["--paths"]


def card_local(r: dict) -> tuple[str, list]:
    """Write the card to disk. Always safe -- nothing leaves the machine."""
    return card(r)


def publish(r: dict) -> tuple[str, list]:
    """pollard-card --upload: this one actually pushes to Hugging Face."""
    a: list[str] = []
    _opt(a, "--model", r.get("modelName"))
    _opt(a, "--builds-from", r.get("dir"))
    _opt(a, "--results", r.get("results"))
    _opt(a, "--repo", r.get("repo"))
    _opt(a, "--upload", r.get("uploadRepo") or r.get("repo"))
    return "pollard_card", a


ACTIONS = {
    "build": lane, "fit": build, "automap": automap, "bench": bench, "bench-all": bench,
    "eval": evaluate, "doctor": doctor, "smoke": smoke, "ggufcheck": ggufcheck,
    "verify": verify, "gate": verify, "card": card_local, "fragile": fragile, "train": train,
    "ls": ls, "publish": publish, "routecheck": routecheck,
    "mmeval": mmeval, "taskeval": taskeval, "failmode": failmode,
    "toolcall": toolcall, "kvsweep": kvsweep,
    # levers
    "probe": probe, "sensitivity": sensitivity, "precondition": precondition,
    "rotate": rotate, "smooth": smooth, "hf-smooth": hf_smooth,
    "palette": palette, "lowbit": lowbit,
    # MoE
    "experts": experts, "route": route, "prune": prune,
    # diagnostics
    "calc": calc, "health": health, "runtime": runtime, "refcheck": refcheck,
    "modelkind": modelkind, "archfp": archfp, "envmatch": envmatch, "stop": stop,
    # publish
    "scorecard": scorecard, "recard": recard, "reclaim": reclaim,
    # brains
    "brainattach": brainattach, "brainverify": brainverify, "brainlanes": brainlanes,
    "connectome": connectome, "flybrain": flybrain,
    "emit": emit, "calib": calib, "probes": probes,
    # distributed / multi-box
    "placement": placement, "vllmcheck": vllmcheck,
}

# Actions that leave the machine. These are held to a stricter confirmation than a long build:
# a bad build wastes an afternoon, a bad publish is public.
OUTBOUND = {"publish"}

# Anything that writes weights, uploads, or occupies the box for a long time. The UI asks before
# these; the read-only ones just go.
CONFIRM = {"build", "fit", "automap", "train", "bench", "bench-all", "eval", "card", "publish",
           "routecheck", "mmeval", "taskeval", "failmode", "toolcall", "kvsweep",
           "probe", "sensitivity", "precondition", "rotate", "smooth", "hf-smooth",
           "palette", "lowbit", "prune", "route", "emit", "recard", "reclaim",
           "brainattach", "flybrain", "connectome", "calib", "probes"}


def resolve(action: str, recipe: dict) -> tuple[str, list] | None:
    """(tool, args) for an action, or None when the action is UI-only."""
    key = action.split(":", 1)[0]
    if key == "run" and ":" in action:                     # 'run:pollard_x' from the Tools screen
        return action.split(":", 1)[1], []
    if key == "help" and ":" in action:
        return action.split(":", 1)[1], ["--help"]
    fn = ACTIONS.get(key)
    return fn(recipe) if fn else None


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    recipe = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    got = resolve(sys.argv[1], recipe)
    print("UI-only action, nothing to run" if got is None
          else f"{got[0].replace('_', '-')} " + " ".join(got[1]))


if __name__ == "__main__":
    main()
