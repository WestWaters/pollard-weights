#!/usr/bin/env python3
"""pollard -- the autoaware entry point. Point it at ANY model; it detects dense vs MoE
and drives the WINNING method, so a user never runs a losing path or wastes hours.

WINNING PATH -- the SAME for dense AND MoE (no losing fallback):
  (1) the imatrix K-quant ladder (pollard-fit) -- the honest fit-your-RAM baseline, plus
  (2) the mixed-precision FLAGSHIP mix (automap trellis) -- the hand-coded winner (crush body,
      protect attn/down/first-last; MoE = expert-allocation) -- WHEN an imatrix is present.
  No --imatrix supplied -> AUTO-BUILD one (Calib 3.0 corpus -> llama-imatrix) so the DEFAULT is the
  flagship mix with ZERO manual steps. --no-auto-imatrix falls back to the stock K-quant ladder.
  (There is NO "imatrix-free mix": it loses to stock Q2_K, so it's deprecated, not a path here.)

STOP LOSING / TRAP paths -- never the default, opt-in only:
  - sensitivity SWEEP on a BIG model = ~2*layers full-model quantizes = many HOURS; opt-in R&D only
    (dense is NOT refused -- it is a cost guard, not an arch guard; bench the two arms).
  - the 3-bar comparison + KL/PPL = the BENCHMARK, opt-in via --benchmark (see benchmarks/).

Plans by default (prints the exact commands for THIS model); `--run` executes them.

    pollard --gguf model-f16.gguf --run             # ONE-SHOT GGUF: auto-calib -> imatrix -> flagship mix
    pollard --hf Qwen/Qwen3-8B --run                # ONE-SHOT from a HF repo: download -> convert -> build
    pollard --hf ./my-local-model --run             # ...or a model already on disk (any arch)
    pollard --hf Qwen/Qwen3-8B --format gptq --run  # GPTQ for vLLM/SGLang
    pollard --hf Qwen/Qwen3-8B --format mlx --run   # MLX for Apple Silicon
    pollard --hf Qwen/Qwen3-8B --format exl3 --run  # EXL3 (exllamav3 -- heavy trellis)
    pollard --hf Qwen/Qwen3-8B --format mx --run    # MX: Blackwell NVFP4 / any-GPU W4A16 (compressed-tensors)
    pollard --hf Qwen/Qwen3-8B --format exlr8 --run # EXLR8: one trellis encode, the whole bpw ladder as manifests
    pollard --gguf model-f16.gguf --imatrix m.imatrix --run    # bring your own imatrix (skips auto-calib)
    pollard --gguf model-f16.gguf --benchmark --run            # + the gold-card board (slow)
    pollard tune --base Qwen/Qwen3.5-9B --data cases.csv --out runs/x --ship   # fine-tune + eval + ship (pollard-tune)
    pollard serve Org/Model:Q4_K_M --harness claude-code       # run a rung for an agent harness (pollard-serve)
    pollard release --hf Qwen/Qwen3-8B --plan release.yaml     # EVERY lane as a ladder, gated + carded (pollard-release)
    pollard setup                                              # what each lane needs here; install what is missing

EVERY lane runs the GOLD Pollard method one-shot: GGUF = auto Calib-3.0 imatrix -> measured automap mix
-> coherence gate; GPTQ/MLX/MX = smoothing (default, low-bit lanes) + auto-measured allocation (a cheap
pollard-probe, --no-measure to skip); EXL3 = smoothing + Calib 3.0 packed to -cd + EXL3's native allocator.
"""
import argparse, contextlib, json, os, re, subprocess, sys, time

from pollard_calc import (read_gguf_meta, gguf_to_config, analyse, find_llama_bin,
                          detect_gpu_gb, detect_available_ram_gb)


def _accel_gb():
    """GB the accelerator can actually hold: VRAM on NVIDIA, the shared pool on Apple Silicon
    (where the GPU spends host RAM, so nameplate VRAM is the wrong question). 0 = unknown."""
    vram = detect_gpu_gb()
    if vram:
        return vram * 0.90
    if sys.platform == "darwin":
        return detect_available_ram_gb() * 0.80
    return 0.0


def _weights_gb(hf_dir):
    """On-disk weight GB for an HF checkout. 0 = unknown."""
    import glob
    tot = 0
    for pat in ("*.safetensors", "*.bin"):
        tot += sum(os.path.getsize(f) for f in glob.glob(os.path.join(hf_dir, pat)))
    return tot / (1 << 30)


def _use_stream_probe(hf_dir, forced, need_gb=None):
    """Which sensitivity estimator can actually finish on this box.

    The default perturb+KL probe crushes one group and re-runs the eval, so it costs layers*groups
    full forward passes -- ~100 for a 48-layer model. That is fine for a model that sits in memory
    and hopeless for one streaming off disk, which is precisely when a measured allocation matters
    most. The gold path would be 'available' and simply never finish. The one-pass Hessian-diagonal
    estimator ranks the same groups for a single forward, so use it when the weights do not fit."""
    if forced in ("stream", "kl"):
        return forced == "stream"
    # `need_gb` states the weight size rather than requiring a checkpoint on disk -- exercising the
    # bigger-than-RAM path should not mean writing a 400GB file, which is free only on a filesystem
    # with sparse files and writes real bytes on one without.
    need = need_gb if need_gb is not None else _weights_gb(hf_dir)
    ram = detect_available_ram_gb() or 0.0
    return bool(need and ram and need * 1.15 > ram)


def _fit_ngl(gguf, ngl):
    """How many layers actually fit on the accelerator.

    `-ngl 99` asks llama.cpp to offload EVERY layer. That is right for a model that fits and fatal
    for one that doesn't -- and because a failed imatrix only surfaces later, as llama-quantize
    being unable to open the file, the whole build lands on a uniform allocation with nobody the
    wiser. Offload the share that fits and stream the rest."""
    if str(ngl).lower() != "auto":
        return str(ngl)
    try:
        size_gb = os.path.getsize(gguf) / (1 << 30)
        layers = int(gguf_to_config(read_gguf_meta(gguf), gguf).get("num_hidden_layers") or 0)
    except Exception:
        return "99"
    budget = _accel_gb()
    if not budget or not layers:
        return "99"
    if size_gb * 1.10 <= budget:
        return "99"
    n = max(int(layers * budget / (size_gb * 1.10)), 0)
    print(f"      (-ngl auto: {size_gb:.1f}GB model vs {budget:.1f}GB usable -> {n}/{layers} layers "
          f"offloaded, rest streamed)")
    return str(n)

# reference bits-per-weight for the common formats, so a user can see where a Pollard
# build lands vs f16 / NVFP4 / the usual GGUF tiers ("half the size of NVFP4" etc.).
_REF_BPW = [("f16", 16.0), ("Q8_0", 8.5), ("Q6_K", 6.6), ("Q5_K_M", 5.5),
            ("NVFP4", 4.25), ("Q4_K_M", 4.85), ("IQ2 (~2-bit)", 2.1)]


def size_ladder(params_b, pollard_gb=None, pollard_label="Pollard"):
    """Print the shrink story: f16 size, where Pollard's build lands, and the same model
    at each reference format -- so people SEE what Pollard did and can compare to NVFP4."""
    if not params_b:
        return
    gb = lambda bpw: params_b * 1e9 * bpw / 8 / 1e9
    f16 = gb(16.0)
    print("   --- what Pollard did (size) ---")
    if pollard_gb:
        pct = 100 * (1 - pollard_gb / f16)
        ratio = f16 / max(pollard_gb, 1e-9)
        bpw = pollard_gb * 8 * 1e9 / (params_b * 1e9)
        print(f"   {pollard_label}: {f16:.1f} GB (f16) -> {pollard_gb:.2f} GB  "
              f"(-{pct:.0f}%, {ratio:.1f}x smaller, {bpw:.2f} bpw)")
    print("   same model at each format:  " +
          "  ".join(f"{n} {gb(b):.1f}GB" for n, b in _REF_BPW) +
          (f"  ->  {pollard_label} {pollard_gb:.2f}GB" if pollard_gb else ""))


def _resolve_tool(cmd):
    """A `pollard-*` step runs even when its console launcher was never installed.

    An editable checkout keeps the code current but only writes launchers when pip runs, and a
    scheduled task on Windows does not have the venv's Scripts on PATH either -- FrogNano died after a
    clean convert because `pollard-calib` was "not found" on a box that had the module right here. If
    the launcher is missing, run the module behind it with this interpreter instead."""
    import shutil
    name = str(cmd[0]) if cmd else ""
    if not cmd or not (name == "pollard" or name.startswith("pollard-")) or shutil.which(name):
        return cmd
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        proj = open(os.path.join(os.path.dirname(here), "pyproject.toml"), encoding="utf-8").read()
        m = re.search(rf'^{re.escape(str(cmd[0]))}\s*=\s*"([\w_]+):', proj, re.M)
    except OSError:
        m = None
    mod = m.group(1) if m else str(cmd[0]).replace("-", "_")
    path = os.path.join(here, mod + ".py")
    return [sys.executable, path, *cmd[1:]] if os.path.isfile(path) else cmd


def _run(cmd, do_run, cwd=None):
    cmd = _resolve_tool(list(cmd))
    print("   $ " + " ".join(str(c) for c in cmd))
    if do_run:
        r = subprocess.run(cmd, cwd=cwd)
        if r.returncode != 0:
            sys.exit(f"   step failed (exit {r.returncode}).")
    return not do_run


def _arch_name(gguf_path):
    """The GGUF's `general.architecture` string, or None."""
    try:
        meta = read_gguf_meta(gguf_path)
        for k in ("general.architecture", "architecture", "arch"):
            v = meta.get(k) if isinstance(meta, dict) else None
            if v:
                return str(v)
    except Exception:
        pass
    return None


def _scan_for(path, needle):
    """Substring search over a file, chunked, without loading it whole."""
    try:
        overlap = len(needle) - 1
        prev = b""
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    return False
                if needle in prev + chunk:          # `prev` carries the chunk boundary
                    return True
                prev = chunk[-overlap:] if overlap else b""
    except Exception:
        return False


def _build_knows_arch(bin_dir, arch):
    """Does this llama.cpp build know `arch`? Arch names are string literals in the arch table, which
    lives in the SHARED LIBRARY (llama.dll / libllama.so), not the CLI executable -- scanning only the
    exe reports false negatives for architectures the build genuinely supports. So scan the whole bin
    directory. Returns True/False, or None when the directory is missing/unreadable."""
    if not arch or not bin_dir or not os.path.isdir(bin_dir):
        return None
    needle = arch.encode()
    try:
        files = [os.path.join(bin_dir, n) for n in os.listdir(bin_dir)]
    except OSError:
        return None
    files = [p for p in files if os.path.isfile(p)]
    if not files:
        return None
    return any(_scan_for(p, needle) for p in files)


def _is_ik_build(bin_dir):
    """Can this bin dir build the trellis atoms? Only ik_llama.cpp has them: its ggml type table
    names "iq1_kt", which no stock llama.cpp carries. A llama-quantize alone is not enough -- the
    stock one on PATH rejects every ik type, and that is how a run that had just said "ik_llama
    build not found" went on to plan the IQ1_KT mix and die in llama-quantize. The name lives in
    libggml (or in llama-quantize itself in a static build), so only those files are scanned."""
    if not bin_dir or not os.path.isdir(bin_dir) or not os.path.isfile(_bin_in(bin_dir, "llama-quantize")):
        return False
    try:
        libs = sorted(n for n in os.listdir(bin_dir) if "ggml" in n.lower())
    except OSError:
        return False
    files = [_bin_in(bin_dir, "llama-quantize")] + [os.path.join(bin_dir, n) for n in libs]
    return any(os.path.isfile(f) and _scan_for(f, b"iq1_kt") for f in files)


#: the ik_llama.cpp build pollard-runtime manages, next to this checkout
_MANAGED_IK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "runtime", "ik_llama.cpp", "build", "bin")


def find_ik_build(ik_bin_dir=None):
    """The ik_llama.cpp bin dir the trellis flagship will be built with, or None. --bin, then
    $IK_LLAMA_BIN, then the managed runtime, then wherever pollard-automap would look on its own --
    each one checked to really be ik_llama, so the plan and the build agree on what is installed."""
    cands = [ik_bin_dir, os.environ.get("IK_LLAMA_BIN"), _MANAGED_IK]
    try:
        from pollard_automap import find_ik_bin
        cands.append(find_ik_bin(None))
    except Exception:
        pass
    for cand in cands:
        if _is_ik_build(cand):
            return cand
    return None


def lane_report(gguf_path, ik_dir=None):
    """Autoaware lane availability: the trellis FLAGSHIP needs ik_llama.cpp, and ik_llama only
    builds architectures it knows. Say so BEFORE planning a flagship that cannot build, and say
    exactly what is missing so it can be ported instead of silently downgraded. `ik_dir` is the
    build find_ik_build() found (None = no ik_llama on this machine)."""
    arch = _arch_name(gguf_path)
    if not arch:
        return None
    known = _build_knows_arch(ik_dir, arch) if ik_dir else None
    print(f"   arch: {arch}")
    if known is None:
        print("   trellis flagship: ik_llama.cpp build not found -- set IK_LLAMA_BIN (or --bin) to an "
              "ik_llama.cpp build/bin to enable the IQ*_KT flagship (the K-quant ladder still runs).")
    elif known:
        print("   trellis flagship: available (ik_llama.cpp knows this arch)")
    else:
        print(f"   trellis flagship: UNAVAILABLE - ik_llama.cpp does not know '{arch}'.")
        print( "     The K-quant ladder still runs and is honest; the flagship needs the arch ported.")
        # Don't leave the user guessing how hard that port is -- fingerprint the layout and say so.
        try:
            from pollard_archfp import fingerprint, twin, report as fp_report
            meta = read_gguf_meta(gguf_path)
            names = meta.get("_tensor_names") or []
            if names:
                hp = {k: v for k, v in meta.items() if not k.startswith("_")}
                res = twin(fingerprint(names), hp)
                for line in fp_report(res).splitlines():
                    print("     " + line)
                if res["exact"]:
                    print(f"     -> port it by giving ik_llama an arch enum + the name '{arch}' + the "
                          f"{res['twin']} tensor map, and reusing its existing builder.")
                else:
                    print("     -> NOT a drop-in: the deltas above are real work, not a rename.")
        except Exception as e:
            print(f"     (layout fingerprint unavailable: {e})")
        print( "     See notes/custom-arch-onboarding.md.")
    return known


#: The publish ladder: (label, presets that count as it, target size as a fraction of the f16 file).
#: The fractions are where the published Pollard ladders landed (FrogNano / FrogMini / clef-flash):
#: within each label's band of budgets, the rung nearest its fraction is the one that ships.
LADDER = [("Q6_K", ("Q6_K",), 0.42),
          ("IQ4_XS", ("IQ4_XS",), 0.31),
          ("IQ3_S", ("IQ3_S",), 0.26),
          ("IQ2", ("IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS"), 0.20)]


def _bin_in(bin_dir, name):
    """`name` inside a --bin directory, with .exe on Windows. A bare join handed pollard-fit
    "...\\build\\bin\\llama-quantize", which does not exist on Windows: the Clef 27B ladder (2026-10-10) died
    "llama-quantize not found" after a 2-hour imatrix, while the same build without --bin worked."""
    p = os.path.join(bin_dir, name)
    if os.name == "nt" and not p.lower().endswith(".exe") and os.path.isfile(p + ".exe"):
        return p + ".exe"
    return p


def ladder_points(gguf, imatrix=None, sensitivity=None, step=0.1):
    """[(ram, preset, projected_gb)] over a budget grid, from pollard-fit's own planner (reserve 0,
    so --ram is the weights budget it plans against). Nothing is quantized; it is the --plan-only
    search the hand-written ladder scripts did by trial and error, done in-process."""
    from pollard_fit import plan_allocation
    from pollard_calc import imatrix_covered_tensors
    from pollard_ngram import disk_plan
    meta = read_gguf_meta(gguf)
    arch = analyse(gguf_to_config(meta, gguf))
    covered = imatrix_covered_tensors(imatrix) if imatrix else None
    emb_ok = bool(covered) and {"token_embd.weight", "output.weight"} <= covered
    disk = disk_plan(meta, "auto", "q8_0")
    sens = json.load(open(sensitivity)) if sensitivity else None
    top = os.path.getsize(gguf) / 1e9 * 1.25 if os.path.exists(gguf) else 64.0
    pts, r = [], 1.0
    while r <= top:
        try:
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                _, _, gb, preset, _ = plan_allocation(arch, r, 0.0, sens, False, None, emb_ok,
                                                      disk if disk and disk["params"] else None)
            pts.append((round(r, 2), str(preset), float(gb)))
        except (SystemExit, Exception):
            pass
        r += step
    return pts


def pick_ladder(points, f16_gb, ik_flagship=False):
    """{label: (ram, preset, gb)} -- per LADDER label, the budget whose preset counts as it and whose
    projected size is nearest the label's fraction of the f16. The 2-bit rung is left to the IQ*_KT
    flagship when ik_llama can build it (that is the rung Pollard publishes there); otherwise a stock
    IQ2 type, which mainline llama.cpp runs. A label no budget reaches is simply absent."""
    out = {}
    for label, presets, frac in LADDER:
        if label == "IQ2" and ik_flagship:
            continue
        band = [p for p in points if p[1] in presets]
        if band:
            out[label] = min(band, key=lambda p: (abs(p[2] - frac * f16_gb), p[0]))
    return out


def _wanted_rungs(a):
    """--rung LABEL (repeatable / comma list): build only these ladder labels (`flagship` = the automap mix).
    None = the whole ladder + flagship. `pollard release --free-after-upload` builds one rung at a time with it, so
    a box with less free disk than the whole ladder can still publish every rung."""
    want = [x.strip() for r in (getattr(a, "rung", None) or []) for x in str(r).split(",") if x.strip()]
    return {w.upper() if w.lower() != "flagship" else "flagship" for w in want} or None


def _build_ladder(a, ik_flagship):
    pts = ladder_points(a.gguf, a.imatrix, a.sensitivity)
    f16_gb = os.path.getsize(a.gguf) / 1e9
    picks = pick_ladder(pts, f16_gb, ik_flagship)
    print(f"   1) the publish ladder ({', '.join(picks) or 'nothing reachable'}):")
    for label, (ram, preset, gb) in picks.items():
        print(f"      {label:<7} --ram {ram:<5} -> {preset} ~{gb:.1f} GB")
    want = _wanted_rungs(a)
    if want is not None and "IQ2" in want and "IQ2" not in picks and ik_flagship:
        # asked for BY NAME: the stock 2-bit rung (mainline llama.cpp runs it) beside the ik-only flagship. The
        # ladder leaves IQ2 to the flagship by default, so `--rung IQ2` used to build nothing at all.
        extra = pick_ladder(pts, f16_gb, ik_flagship=False).get("IQ2")
        if extra:
            picks["IQ2"] = extra
            print(f"      IQ2     --ram {extra[0]:<5} -> {extra[1]} ~{extra[2]:.1f} GB  (asked for by --rung: the "
                  "stock 2-bit rung beside the trellis flagship)")
    if want is not None:
        picks = {k: v for k, v in picks.items() if k.upper() in want}
        print(f"      --rung: building {', '.join(picks) or 'no ladder rung'} this run")
    binq = _bin_in(a.bin, "llama-quantize") if a.bin else None
    for label, (ram, preset, gb) in picks.items():
        cmd = ["pollard-fit", "--gguf", a.gguf, "--ram", str(ram), "--reserve", "0"]
        if a.imatrix: cmd += ["--imatrix", a.imatrix]
        if a.sensitivity: cmd += ["--sensitivity", a.sensitivity]
        if binq: cmd += ["--llama-quantize", binq]
        if not a.run: cmd += ["--plan-only"]
        _run(cmd, a.run)


def _passthrough_source(gguf):
    """The source's tensor table when it is ALREADY quantized (a vendor QAT GGUF), else None."""
    try:
        import pollard_passthrough as pt
        ts = pt.read_tensors(gguf)
        return ts if pt.is_quantized_source(ts) else None
    except Exception:
        return None


def _rung_out(gguf, label):
    stem = os.path.splitext(os.path.basename(gguf))[0]
    return os.path.join(os.path.dirname(os.path.abspath(gguf)) or ".", f"{stem}-{label}.gguf")


def _build_ladder_passthrough(a, tensors):
    """The publish ladder for an already-quantized source: rungs between the file as it is and its
    floor (every changeable tensor at its lowest legal type), each built by pollard-fit's passthrough
    policy -- untouched tensors copied byte-for-byte, the measured allocation choosing what drops."""
    import pollard_passthrough as pt
    from pollard_calc import imatrix_covered_tensors
    covered = imatrix_covered_tensors(a.imatrix) if a.imatrix and os.path.exists(a.imatrix) else None
    rungs = pt.ladder_targets(tensors, imatrix_ok=bool(a.imatrix), covered=covered)
    src = sum(t["n_bytes"] for t in tensors)
    print(f"   1) the passthrough ladder ({src/1e9:.2f} GB source -> "
          f"{', '.join(f'{lb} ~{b/1e9:.2f} GB' for lb, b in rungs) or 'nothing reachable'}):")
    if not rungs:
        print("      nothing in the changeable classes can go lower" +
              ("" if a.imatrix else " without an imatrix (the 1-bit / IQ2 rungs need one)") + ".")
        return
    want = _wanted_rungs(a)
    if want is not None:
        rungs = [(lb, t) for lb, t in rungs if lb.upper() in want]
        print(f"      --rung: building {', '.join(lb for lb, _ in rungs) or 'no passthrough rung'} this run")
    binq = _bin_in(a.bin, "llama-quantize") if a.bin else None
    for label, target in rungs:
        cmd = ["pollard-fit", "--gguf", a.gguf, "--ram", f"{(target + 1) / 0.85e9:.9f}", "--reserve", "0",
               "--passthrough", "on", "--out", _rung_out(a.gguf, label)]
        if a.imatrix: cmd += ["--imatrix", a.imatrix]
        if a.sensitivity: cmd += ["--sensitivity", a.sensitivity]
        if binq: cmd += ["--llama-quantize", binq]
        if not a.run: cmd += ["--plan-only"]
        _run(cmd, a.run)


def _ensure_passthrough_sensitivity(a):
    """Measured allocation for a passthrough ladder: pollard-sensitivity --passthrough, on the quantized
    file as its own host (imatrix from the same host, held-out slice of Calib 3.0 as the eval)."""
    if a.sensitivity or not a.measure or not a.imatrix:
        if not a.sensitivity:
            print("   (passthrough: no measured profile -- uniform importance across the changeable layers)")
        return a.sensitivity
    here = os.path.dirname(os.path.abspath(a.gguf)) or "."
    calib = a.calib or os.path.join(here, "pollard_calib.txt")
    evalf = calib + ".heldout" if os.path.exists(calib + ".heldout") or not a.run else calib
    prof = os.path.join(here, os.path.splitext(os.path.basename(a.gguf))[0] + ".passthrough.sensitivity.json")
    cmd = ["pollard-sensitivity", "--gguf", a.gguf, "--imatrix", a.imatrix, "--eval", evalf,
           "--passthrough", "--out", prof]
    if str(a.ngl) != "auto":
        cmd += ["--ngl", str(a.ngl)]
    if getattr(a, "allow_slow", False):
        cmd += ["--allow-slow"]
    print("   measured allocation on the quantized host (pollard-sensitivity --passthrough):")
    _run(cmd, a.run)
    return prof


def _automap_mix(a, is_moe, ik_dir=None):
    """Emit + (with --run) build the automap mix -- the hand-coded mixed-precision flagship.
    MoE: expert-allocation (crush cold experts, protect router/down/shared/attn). DENSE: the
    IQ1_KT crush-body / protect-attn+down+edges flagship (automap-on-dense, --allow-dense).
    Fast by default (--mix-only --no-eval); --benchmark emits the 3-bar + PPL board instead.
    This is what keeps the winning hand-coded mix a first-class BUILD, not benchmark-only."""
    binq = find_llama_bin("llama-quantize") if not a.bin else _bin_in(a.bin, "llama-quantize")
    here = os.path.dirname(os.path.abspath(a.gguf)) or "."
    tensors = os.path.join(here, "pollard_auto_tensors.txt")
    print(f"   tensor list (Q6_K dry-run): {binq} --dry-run {os.path.basename(a.gguf)} x.gguf Q6_K > tensors")
    if a.run:
        with open(tensors, "w") as f:
            subprocess.run([binq, "--dry-run", a.gguf, "x.gguf", "Q6_K"],
                           stdout=f, stderr=subprocess.STDOUT)
    out = os.path.join(here, "pollard_auto_build.bat" if is_moe else "pollard_auto_flagship.bat")
    # the flagship is the TRELLIS mix (winner) -- always imatrix-guided. No K-quant fallback.
    am = ["pollard-automap", "--tensors", tensors, "--model", a.gguf, "--out", out,
          "--imatrix", a.imatrix]
    if not is_moe:
        am += ["--allow-dense"]                             # dense flagship = the hand-coded mix
    if not a.benchmark:
        am += ["--mix-only", "--no-eval"]                   # plain build = ONE model, no benchmark
    if not getattr(a, "gate", True):
        am += ["--no-gate"]                                 # user opted out of the auto coherence gate
    if ik_dir or a.bin:
        am += ["--bin", ik_dir or a.bin]                    # the build find_ik_build() checked
    _run(am, a.run, cwd=here)
    if a.run and not a.benchmark:
        _record_flagship(a, here)
    mode = "3-bar BENCHMARK + PPL" if a.benchmark else "ONE mix model, no eval (fast)"
    print(f"   -> run {os.path.basename(out)} on the box -> {mode}"
          + ("" if a.benchmark else "  (add --benchmark for the gold-card numbers)"))
    return out


def flagship_mix_path(gguf, here=None):
    """Where pollard-automap writes the mix: <stem>-mix.gguf next to the f16, where <stem> is the file name with
    `-f16.gguf` / `.f16.gguf` / `.gguf` cut off (pollard_automap's own rule). The recorder used the plain stem, so
    LFM2.5-24B-A2B-f16.gguf looked for LFM2.5-24B-A2B-f16-mix.gguf while automap wrote LFM2.5-24B-A2B-mix.gguf --
    the flagship built and was never recorded, measured or published."""
    import re as _re
    base = os.path.basename(gguf)
    stem = _re.sub(r"[-.]f16\.gguf$|\.gguf$", "", base, flags=_re.I)
    return os.path.join(here or os.path.dirname(os.path.abspath(gguf)) or ".", f"{stem}-mix.gguf")


def _record_flagship(a, here, body=None):
    """Put the automap mix in the MANIFEST beside the ladder rungs, so pollard-release measures, gates, cards and
    publishes it like any other rung. automap writes `<stem>-mix.gguf` next to the f16; the tag is its body atom
    (iq1_kt by default with an imatrix -> IQ1_KT, the name the published flagships use)."""
    mix = flagship_mix_path(a.gguf, here)
    if not os.path.isfile(mix):
        print(f"   (warning: the flagship mix is not at {mix} -- not recorded, so it will not be measured or published)")
        return None
    try:
        import pollard_workspace as ws
        arch = analyse(gguf_to_config(read_gguf_meta(a.gguf), a.gguf))
        bpw = round(os.path.getsize(mix) * 8.0 / arch["total"], 2) if arch.get("total") else None
        return ws.record_build(a.gguf, "gguf", mix, tag=str(body or "iq1_kt").upper(), bpw=bpw,
                               extra={"flagship": True})
    except Exception as e:                                               # noqa: BLE001
        print(f"   (warning: could not record the flagship in the workspace manifest: {e})")
        return None


def _find_convert(hf_dir=None):
    """Locate a converter that can handle THIS model, not just any file with the right name.

    This used to check two paths and then hand back the bare string "convert_hf_to_gguf.py",
    trusting the shell. When the converter was absent or too old for the architecture, the failure
    surfaced deep in the build as something about the model -- and the apparent fix was to move a
    23.8GB GGUF across the network rather than copy a 3MB script."""
    from pollard_convert import find_converter
    conv, note = find_converter(hf_dir)
    if conv is None and "registers" in note:
        # The converter exists but predates this architecture: update the managed engine and look again.
        from pollard_runtime_update import auto_update_for
        from pollard_convert import model_architectures
        if auto_update_for(",".join(model_architectures(hf_dir)) or "this model"):
            conv, note = find_converter(hf_dir)
    if conv is None:
        raise SystemExit(f"\n   cannot convert on this machine: {note}")
    if hf_dir:
        print(f"   converter: {note}")
    return str(conv)


def _precondition_hf(a, hf_dir):
    """OPT-IN FP16-level transforms applied BEFORE any lane, so they compose across ALL lanes:
      --abliterate : uncensor (orthogonalize residual writers vs the refusal direction). Behaviour-
                     changing, the user's own model; measure the quality delta (pollard-kl), don't assume.
      --smooth     : SmoothQuant preconditioning (reliable low-bit quality lever; GGUF has its own).
    Returns the (possibly transformed) HF dir."""
    here = os.path.dirname(os.path.abspath(__file__))
    cur = hf_dir
    trc = ["--trust-remote-code", getattr(a, "trust_remote_code", "auto")]   # custom-arch passthrough
    if getattr(a, "abliterate", False):
        out = cur.rstrip("/\\") + "-abliterated"
        cmd = [sys.executable, os.path.join(here, "pollard_abliterate.py"), "--model", cur, "--out", out] + trc
        if a.harmful: cmd += ["--harmful", a.harmful]
        if a.harmless: cmd += ["--harmless", a.harmless]
        print(f"   [precondition] abliterate (uncensor, FP16, opt-in): {' '.join(cmd)}")
        if a.run and subprocess.run(cmd).returncode != 0:
            sys.exit("   abliterate failed.")
        cur = out
    if getattr(a, "smooth", False):
        out = cur.rstrip("/\\") + "-smoothed"
        cmd = [sys.executable, os.path.join(here, "pollard_hf_smooth.py"), "--model", cur, "--out", out] + trc
        calib = a.calib or _smooth_calib(a, cur)
        if calib: cmd += ["--calib", calib]
        print(f"   [precondition] smooth (SmoothQuant, FP16): {' '.join(cmd)}")
        if a.run and subprocess.run(cmd).returncode != 0:
            sys.exit("   smooth failed.")
        cur = out
    return cur


def _smooth_calib(a, hf_dir):
    """The corpus SmoothQuant measures its activation scales on: the same Calib 3.0 corpus the export lane then
    calibrates on (built here, first, when the lane is an export lane) -- so an agentic model's smoothing sees its
    tool-call traces rendered through its own template, not a bland built-in sentence."""
    if getattr(a, "format", "gguf") not in ("gptq", "mx", "exl3", "exlr8", "mlx"):
        return None
    here = os.path.dirname(os.path.abspath(a.output)) if a.output else os.getcwd()
    prev = getattr(a, "_hf_dir", None)
    a._hf_dir = hf_dir                                  # the template the corpus renders through
    try:
        return _ensure_calib_text(a, here)
    finally:
        a._hf_dir = prev


def _preflight(a, hf_dir):
    """Can this machine finish the build, asked BEFORE it starts one.

    Every capability this needs is knowable in seconds from the model's shape -- is there a
    converter here that knows the architecture, will the weights fit anywhere sensible. Asking
    afterwards is how a missing 3MB converter surfaces hours in, looking like a problem with the
    model. A build that cannot finish should say so before it spends anything."""
    if not getattr(a, "preflight", True):
        return
    try:
        from pollard_convert import find_converter, model_architectures
        from pollard_probe import plan_placement
    except ImportError:
        return                                          # no extra installed: nothing to check with
    archs = model_architectures(hf_dir)
    conv, note = find_converter(hf_dir)
    print(f"   preflight: {', '.join(archs) or 'architecture unknown'}")
    if conv is None:
        raise SystemExit(f"   STOPPING before this build spends anything -- {note}\n"
                         "   (--no-preflight to try anyway)")
    print(f"      converter  {note}")
    try:
        _, _, where = plan_placement(hf_dir, "cpu", os.path.join(hf_dir, ".pollard-offload"))
        if where:
            print(f"      placement  {where}")
    except Exception:
        pass                                            # placement is advisory, never a blocker


def _resolve_hf(a):
    """A local HF dir is used as-is; a repo id is downloaded (snapshot) so users can point at either.
    Opt-in --abliterate/--smooth transforms are applied here so every lane inherits them."""
    if os.path.isdir(a.hf):
        _preflight(a, a.hf)
        a._hf_src = a.hf                                # the original weights: the tool-call reference
        a._hf_dir = _precondition_hf(a, a.hf)
        return a._hf_dir
    import pollard_workspace as ws
    local = ws.source_dir(a.hf)                         # $POLLARD_HOME/downloads/<slug> -- one findable place
    print(f"   fetch HF repo -> {local}  (workspace downloads/)")
    if a.run:
        local = ws.fetch_source(a.hf)                   # single copy, no ~/.cache dup
    if os.path.isdir(local):
        _preflight(a, local)
    a._hf_src = local
    a._hf_dir = _precondition_hf(a, local)
    return a._hf_dir


def _hf_to_gguf(a):
    """HF weights -> f16 GGUF so the GGUF flagship pipeline can run one-shot from a repo/dir."""
    hf_dir = _resolve_hf(a)
    conv = _find_convert(hf_dir)
    here = os.path.abspath(a.output) if a.output else os.path.dirname(os.path.abspath(hf_dir)) or "."
    out = os.path.join(here, os.path.basename(hf_dir.rstrip("/\\")) + "-f16.gguf")
    print(f"   convert HF -> f16 GGUF: python {os.path.basename(conv)} {hf_dir} --outtype f16 --outfile {out}")
    _emit_mmproj(a, conv, hf_dir, out)
    if a.run:
        subprocess.run([sys.executable, conv, hf_dir, "--outtype", "f16", "--outfile", out])
    return out


def _ensure_calib_text(a, here):
    """Return a Calib 3.0 text corpus path, auto-building it if the user gave none (one-shot = no
    manual calibration step). Shared by the gptq / mx / exl3 lanes."""
    calib = a.calib or os.path.join(here, "pollard_calib.txt")
    if not a.calib:
        cmd = ["pollard-calib", "--out", calib, "--held-out", calib + ".heldout"] + _calib_profile_args(a)
        print(f"   auto-calib (Calib 3.0): {' '.join(os.path.basename(c) if c in (calib, calib + '.heldout') else c for c in cmd)}")
        if a.run and not os.path.exists(calib):
            _run(cmd, True, cwd=here)
    return calib


def _calib_profile_args(a):
    """--profile for pollard-calib: what the model is FOR decides the calibration mix (a translator
    calibrated on English prose protects the wrong tensors). 'auto' asks pollard-modelkind; the chat
    samples are rendered through the model's own template when the HF checkout is at hand."""
    prof = getattr(a, "calib_profile", "auto") or "auto"
    src = getattr(a, "_hf_dir", None) or (a.hf if a.hf and os.path.isdir(str(a.hf)) else None) or a.gguf
    args = ["--profile", prof]
    if src:
        args += ["--model", src]
    key = a.hf or a.gguf
    if key:
        args += ["--record-model", key]
    return args


def _ensure_sensitivity(a, hf_dir, calib, here):
    """The GOLD lever for the export lanes: a measured sensitivity profile so allocation is Pollard's,
    not uniform. Uses --sensitivity if given; else (unless --no-measure) auto-runs the cheap any-box
    probe (pollard-probe) against a held-out slice of Calib 3.0. Returns a path or None (uniform)."""
    if a.sensitivity:
        return a.sensitivity
    if not a.measure:
        print("   (--no-measure: uniform allocation for this lane)")
        return None
    prof = os.path.join(here, os.path.basename(hf_dir.rstrip("/\\")) + ".sensitivity.json")
    heldout = (calib + ".heldout") if calib else None
    evalf = heldout if (heldout and (not a.run or os.path.exists(heldout))) else calib
    print(f"   auto-measure allocation (gold): pollard-probe --model {hf_dir} --eval {os.path.basename(evalf or 'calib')} --out {os.path.basename(prof)}")
    if a.run:
        cmd = ["pollard-probe", "--model", hf_dir, "--eval", evalf, "--out", prof]
        if str(getattr(a, "ngl", "auto")) == "0":
            # --ngl 0 means "this GPU is not ours" (a shared box). It used to reach only llama.cpp, so the
            # torch probe still picked cuda on the son's GPU and crashed (0xC0000005) mid-session.
            cmd += ["--device", "cpu"]
        if _use_stream_probe(hf_dir, getattr(a, "probe_method", "auto")):
            cmd.append("--stream")
            print(f"      ({_weights_gb(hf_dir):.1f}GB of weights vs {detect_available_ram_gb() or 0:.1f}GB "
                  "free -- one-pass estimator; perturb+KL would need ~layers*groups full passes)")
        r = subprocess.run(_resolve_tool(cmd), cwd=here)
        # This used to fall back to uniform "rather than killing the whole build". But a uniform
        # allocation is not a lesser Pollard build -- it is the thing pollard-fit itself warns has
        # no quality win over a stock K-quant. Degrading to it silently spends hours producing a
        # model whose entire selling point is missing. Uniform is a CHOICE (--no-measure), not a
        # fallback.
        if r.returncode != 0 or not os.path.exists(prof):
            raise SystemExit(
                f"\n   pollard-probe failed (exit {r.returncode}) -- no sensitivity profile.\n"
                "   REFUSING to fall back to a uniform allocation: the measured per-layer profile\n"
                "   IS the Pollard method, and without it this build is a stock K-quant wearing\n"
                "   our name. See the probe's error above (it places a too-big model automatically,\n"
                "   so an OOM here usually means something else).\n"
                "   To accept a uniform build on purpose, pass --no-measure.")
    return prof


def _emit_nongguf(a):
    """GPTQ (vLLM/SGLang), MLX (Apple), EXL3 (exllamav3), and MX (Blackwell/any-GPU compressed-tensors)
    emit straight from HF weights -- same Pollard method (smoothing default + measured allocation), a
    different emitter. Calib 3.0 and the sensitivity profile are auto-built so it's a true one-shot."""
    if not a.hf:
        sys.exit(f"--format {a.format} exports from HF weights -- pass --hf <repo-or-dir> "
                 f"(a GGUF can't be re-exported to {a.format}; use --format gguf for a GGUF input).")
    hf_dir = _resolve_hf(a)
    out = a.output or (os.path.basename(hf_dir.rstrip("/\\")) + f"-Pollard-{a.format.upper()}")
    here = os.path.dirname(os.path.abspath(out)) or "."

    trc = ["--trust-remote-code", a.trust_remote_code]     # custom-arch passthrough (Spark2_5 etc.)
    if a.focus_layers:                                     # user-steered budget passthrough
        trc += ["--focus-layers", a.focus_layers]
    if a.format == "gptq":
        calib = _ensure_calib_text(a, here)
        sens = _ensure_sensitivity(a, hf_dir, calib, here)
        cmd = ["pollard-export", "--model", hf_dir, "--calib", calib, "--out", out] + trc
        if sens:
            cmd += ["--sensitivity", sens]
    elif a.format == "mx":                                  # Blackwell NVFP4 / any-GPU W4A16 (compressed-tensors)
        calib = _ensure_calib_text(a, here)                # NVFP4 activation scales need calibration
        sens = _ensure_sensitivity(a, hf_dir, calib, here)
        cmd = ["pollard-mx", "--model", hf_dir, "--calib", calib, "--out", out] + trc
        if sens:
            cmd += ["--sensitivity", sens]
    elif a.format == "exlr8":
        # one encode (Hessians from the same Calib 3.0 corpus) -> every rung of the ladder as a manifest chosen by
        # the measured sensitivity; --output is the parent of the rung folders
        calib = _ensure_calib_text(a, here)
        sens = _ensure_sensitivity(a, hf_dir, calib, here)
        from pollard_exlr8 import DEFAULT_LADDER
        cmd = ["pollard-exlr8", "--model", hf_dir, "--calib", calib, "--out", out,
               "--ladder", ",".join(str(b) for b in DEFAULT_LADDER)]
        if sens:
            cmd += ["--sensitivity", sens]
    elif a.format == "exl3":
        # GOLD EXL3 = smoothing (applied in _resolve_hf) + Calib 3.0 (-cd) + EXL3's native allocator.
        calib = _ensure_calib_text(a, here)
        cmd = ["pollard-exl3", "--model", hf_dir, "--out", out, "--calib-text", calib]
    else:                                                   # mlx (Apple) -- measured 4/8 mix; smoothing N/A
        calib = _ensure_calib_text(a, here)                # only for the probe's held-out eval corpus
        sens = _ensure_sensitivity(a, hf_dir, calib, here)
        cmd = ["pollard-mlx", "--model", hf_dir, "--out", out] + trc
        if sens:
            cmd += ["--sensitivity", sens]
        if getattr(a, "imatrix", None) and os.path.exists(a.imatrix):
            cmd += ["--imatrix", a.imatrix]                # activation-weighted range search (--range search)
    print(f"   {a.format.upper()} export (Pollard method -- smoothing default + measured allocation):")
    _run(cmd, a.run)
    _rt = {"gptq": "vllm serve / sglang", "mlx": "mlx_lm.generate",
           "exl3": "exllamav3 / TabbyAPI", "mx": "vllm serve (compressed-tensors)",
           "exlr8": "pollard-exlr8 --decode <rung> -> vllm serve / transformers"}
    print(f"   -> {out}  ({_rt.get(a.format, '')})")
    if a.format == "exlr8":                                 # the ladder's parent: the tool gate takes the 4.0 rung
        from pollard_exlr8 import bpw_tag
        return os.path.join(out, f"{os.path.basename(hf_dir.rstrip('/'))}-Pollard-EXLR8-{bpw_tag(4.0)}")
    return out


def _imatrix_shape(gguf):
    """Extra llama-imatrix flags a model's runtime shape demands. A decision HEAD (clef: a
    "<arch>.decision.*" block) runs in embedding mode, and llama.cpp then forces n_batch = n_ubatch
    = 512 after imatrix has sized its sequences from the defaults -- it dies on
    GGML_ASSERT(params.n_ctx == n_seq * n_ctx) (exit 0xC0000409 on Windows) before the first chunk.
    Asking for that shape up front is what it ends up with anyway. The perplexity it prints is then
    meaningless (no LM head) -- the activations it collects are not."""
    try:
        meta = read_gguf_meta(gguf)
    except Exception:
        return []
    if any(".decision." in str(k) for k in meta):
        return ["-c", "512", "-b", "512", "-ub", "512"]
    return []


def _ensure_imatrix(a):
    """TRUE one-shot: if the user gave no --imatrix, auto-build one (Calib 3.0 multi-domain
    corpus -> llama-imatrix) so they never run a manual calibration step. Returns the imatrix
    path (and, with --run, actually builds it). Plan mode just prints the exact commands."""
    here = os.path.dirname(os.path.abspath(a.gguf)) or "."
    calib = a.calib or os.path.join(here, "pollard_calib.txt")
    imat = os.path.join(here, os.path.splitext(os.path.basename(a.gguf))[0] + ".imatrix")
    binim = (_bin_in(a.bin, "llama-imatrix") if a.bin
             else find_llama_bin("llama-imatrix")) or "llama-imatrix"
    print("   0) auto-imatrix (Calib 3.0 -> llama-imatrix) -- no manual calibration step:")
    calib_cmd = ["pollard-calib", "--out", calib] + _calib_profile_args(a)
    if getattr(a, "_passthrough", False):                # the passthrough sweep needs a held-out eval slice
        calib_cmd[3:3] = ["--held-out", calib + ".heldout"]
    if not a.calib:
        print(f"      pollard-calib --out {os.path.basename(calib)} {' '.join(calib_cmd[3:])}")
    ngl = _fit_ngl(a.gguf, a.ngl)
    print(f"      {binim} -m {os.path.basename(a.gguf)} -f {os.path.basename(calib)} "
          f"-o {os.path.basename(imat)} -ngl {ngl}")
    if a.run:
        if not a.calib and not os.path.exists(calib):
            _run(calib_cmd, True, cwd=here)
        # --output-format dat, NOT the gguf default. Mainline llama.cpp reads both, but ik_llama --
        # which builds the trellis flagship, the whole reason an imatrix is computed here -- reads
        # only the legacy format and fails with "load_imatrix: failed reading number of values".
        # The ladder still builds, so the loss is silent: the flagship is simply skipped.
        r = subprocess.run([binim, "-m", a.gguf, "-f", calib, "-o", imat, "-ngl", ngl,
                            "--output-format", "dat"] + _imatrix_shape(a.gguf), cwd=here)
        # An unchecked imatrix is how a build gets all the way to llama-quantize before anyone finds
        # out there is no imatrix -- at which point it reports "failed to open" and quietly ships a
        # stock K-quant. Fail here, where the cause is still on screen.
        if r.returncode != 0 or not os.path.exists(imat):
            raise SystemExit(
                f"\n   llama-imatrix failed (exit {r.returncode}) -- no imatrix at {imat}.\n"
                "   Pollard will NOT continue without it: the imatrix is what makes this a Pollard\n"
                "   mix rather than a stock K-quant ladder. If the model is too big for this box,\n"
                "   lower --ngl (or leave it 'auto') or build the imatrix on a Q8_0 host.\n"
                "   To deliberately accept the stock ladder instead, pass --no-auto-imatrix.")
    print("      (big MoE won't fit f16 for the forward pass -> compute on a Q6_K host at a "
          "partial --ngl; see SKILL.md. Undercovered experts hard-fail low-bit -- Calib 3.0 covers them.)")
    return imat


def _pid_alive(pid):
    try:
        if sys.platform == "win32":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True,
                                 text=True, timeout=10).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except Exception:
        return False


class MachineLock:
    """One heavy Pollard job on a machine at a time.

    Quantizing, computing an imatrix and running perplexity all want the same GPU, the same disk and
    most of the cores. Started together they do not go faster -- they thrash, and on a shared machine
    the other person notices first. Nothing enforced this, so the only thing standing between a user
    and an unusable desktop was remembering not to.

    Advisory and self-healing: a lock whose process is gone is taken over, never a reason to be
    stuck. --force ignores it outright."""

    def __init__(self, what, force=False):
        home = os.environ.get("POLLARD_HOME") or os.path.expanduser("~/pollard")
        os.makedirs(home, exist_ok=True)
        self.path, self.what, self.force, self.held = os.path.join(home, ".pollard.lock"), what, force, False

    def __enter__(self):
        if self.force:
            return self
        try:
            if os.path.exists(self.path):
                prev = json.loads(open(self.path, encoding="utf-8").read() or "{}")
                pid = int(prev.get("pid", 0))
                if pid and pid != os.getpid() and _pid_alive(pid):
                    raise SystemExit(
                        f"\n   another Pollard job is running on this machine (pid {pid}): "
                        f"{prev.get('what', '?')}\n"
                        "   Running two at once does not finish sooner -- they contend for the GPU,\n"
                        "   the disk and the cores, and on a shared machine someone else feels it.\n"
                        "   Wait for it, or pass --force if you know it is finished.")
        except SystemExit:
            raise
        except Exception:
            pass                                            # an unreadable lock never blocks a build
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "what": self.what}, f)
            self.held = True
        except OSError:
            pass
        return self

    def __exit__(self, *exc):
        if self.held:
            try:
                os.remove(self.path)
            except OSError:
                pass
        return False




def _emit_mmproj(a, conv, hf_dir, gguf_out):
    """Export the multimodal projector when the model has one, so the build can still see and hear.

    A text GGUF is only the language half of a multimodal model. Ship it alone and the model loses
    image and audio with nothing saying so -- the weights for them are right there in the source.

    NOT --outtype f16. Gemma 4 is encoder-free: the 550M vision encoder is replaced by one large
    matmul, so `v.patch_embd.weight` IS the vision pathway, and the reference build keeps it at F32
    while the projections are BF16. Passing f16 downcast it and produced a 122MB projector where the
    reference is 175MB. bf16 reproduces the reference byte for byte."""
    try:
        from pollard_modelkind import classify
        k = classify(hf_dir)
    except Exception:
        return
    mods = [m for m in k.get("modalities", []) if m != "speech_out"]
    if not mods:
        return
    out = os.path.join(os.path.dirname(os.path.abspath(gguf_out)) or ".",
                       "mmproj-" + os.path.basename(gguf_out).replace("-f16.gguf", "-BF16.gguf"))
    print(f"   + multimodal projector ({'/'.join(mods)}): --mmproj --outtype bf16 -> "
          f"{os.path.basename(out)}")
    if not a.run:
        return
    r = subprocess.run([sys.executable, str(conv), str(hf_dir), "--mmproj",
                        "--outtype", "bf16", "--outfile", out])
    if r.returncode != 0 or not os.path.exists(out):
        print("   (mmproj export failed -- the text build is unaffected, but this model's image/"
              "audio will not work until one is produced)")
        return
    print(f"   wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


def _agentic_family(src):
    """The tool-call syntax pollard-modelkind finds in this model's chat template, or None (not agentic / unreadable)."""
    try:
        from pollard_modelkind import classify
        fam = (classify(src).get("tools") or {}).get("family", "none")
    except (Exception, SystemExit):                              # noqa: BLE001
        return None
    return None if fam in (None, "none") else fam


def _f16_gguf_for(src):
    """An f16 GGUF of this model already in the workspace (a GGUF run made one): the cross-backend reference a
    directory lane falls back to when its own backend cannot serve the bf16 here."""
    if not src:
        return None
    try:
        import glob
        import pollard_workspace as ws
        hits = sorted(glob.glob(os.path.join(ws.model_dir(src), "**", "*-f16.gguf"), recursive=True))
        return hits[0] if hits else None
    except Exception:                                            # noqa: BLE001
        return None


def _toolcall_cmd(path, ref, ngl, results, rung, backend="auto", ref_fallback=None):
    cmd = ["pollard-toolcall", "--model", path, "--backend", backend, "--ngl", ngl, "--results", results,
           "--rung", rung]
    if ref:
        cmd += ["--ref", ref]
    if ref_fallback:
        cmd += ["--ref-fallback", ref_fallback]
    return _resolve_tool(cmd)


def _toolcall_gate(a, since, lane_out=None):
    """An AGENTIC model (pollard-modelkind: its chat template writes tool calls in a known syntax) gets the tool-call
    gate on every rung this run built, in whichever lane it built: pollard-toolcall serves each rung with that
    lane's own server (llama-server --jinja for GGUF, mlx_lm.server for MLX, vllm serve --tool-call-parser for
    GPTQ / MX, exllamav3 in-process for EXL3), asks 60 tool probes through OpenAI-style `tools`, and records retention vs the
    reference in results.json -- which the card turns into its Tool-ready column. A quant can keep its perplexity and
    lose the narrow band of layers that formats tool calls, and an agent harness is exactly where that shows.
    On by default; --no-toolcall skips it. A lane this machine cannot serve is recorded as not measured."""
    if not a.run or not getattr(a, "toolcall", True):
        return
    # a shared GPU is not ours unless told: 'auto' never becomes a GPU claim for llama-server here
    ngl = str(a.ngl) if str(a.ngl).isdigit() else "0"
    if lane_out:                                                 # a directory lane: the one rung this run emitted
        ref = getattr(a, "_hf_src", None) or getattr(a, "_hf_dir", None)
        fam = _agentic_family(lane_out) or _agentic_family(ref)
        if not fam or not os.path.exists(lane_out):
            return
        results = getattr(a, "results", None) or os.path.join(os.path.dirname(os.path.abspath(lane_out)) or ".",
                                                             "results.json")
        print(f"\n   tool-call gate: agentic model ({fam} call syntax) -- the {a.format.upper()} rung vs the "
              "original weights (--no-toolcall to skip)")
        cmd = _toolcall_cmd(lane_out, ref, ngl, results, os.path.basename(lane_out.rstrip("/\\")),
                            ref_fallback=_f16_gguf_for(a.hf))
        print("   $ " + " ".join(str(c) for c in cmd))
        r = subprocess.run(cmd)
        if r.returncode not in (0, 1):                           # 1 = FAIL, a number for the card; else a crash
            print(f"   (tool-call gate on {os.path.basename(lane_out)} did not finish: exit {r.returncode})")
        a.results = results
        return
    if not a.gguf:
        return
    fam = _agentic_family(a.gguf)
    if not fam:
        return
    try:
        import pollard_workspace as ws
        builds = [b for b in ws.read_manifest(a.gguf).get("builds", [])
                  if b.get("lane") == "gguf" and os.path.isfile(b.get("path", ""))
                  and os.path.getmtime(b["path"]) >= since - 5]
    except Exception:                                            # noqa: BLE001
        builds = []
    if not builds:
        return
    results = getattr(a, "results", None) or os.path.join(os.path.dirname(os.path.abspath(a.gguf)) or ".",
                                                         "results.json")
    print(f"\n   tool-call gate: agentic model ({fam} call syntax) -- {len(builds)} rung(s) vs the f16 "
          "(--no-toolcall to skip)")
    for b in builds:
        cmd = _toolcall_cmd(b["path"], a.gguf, ngl, results, b.get("name") or os.path.basename(b["path"]),
                            backend="gguf")
        print("   $ " + " ".join(str(c) for c in cmd))
        r = subprocess.run(cmd)
        if r.returncode not in (0, 1):                           # 1 = FAIL, a number for the card; else a crash
            print(f"   (tool-call gate on {os.path.basename(b['path'])} did not finish: exit {r.returncode})")
    a.results = results


def _emit_card(a):
    """Write the model card for what was just built.

    A build is not finished when the files exist: without a card nobody knows what the rungs are,
    which runtime each needs, or what was measured. That step lived in whatever script happened to
    be driving the build, which means it existed for us and not for anyone else."""
    src = a.gguf or getattr(a, "_hf_dir", None) or a.hf
    if not src:
        return
    cmd = ["pollard-card", "--model", src]
    results = getattr(a, "results", None)
    if results and os.path.exists(results):
        cmd += ["--results", results]
    out = os.path.join(os.path.dirname(os.path.abspath(a.out or src)) or ".", "README.md")
    cmd += ["--out", out]
    print("\n   3) the card (what was built, which runtime each rung needs, what was measured):")
    r = subprocess.run(_resolve_tool(cmd))
    if r.returncode != 0:
        print("   (card step failed -- the builds are fine; run pollard-card yourself to write it)")
        return
    print(f"   wrote {out}")
    if not results:
        print("   NOTE: no --results, so the card carries sizes but no measured numbers. Produce "
              "them with\n         pollard-bench --gguf <rung> --ref <f16> --out results.json, "
              "then re-run with --results.")


def main():
    # `pollard tune ...` -- fine-tune on your data, eval base vs tuned, then hand the merged checkpoint back to
    # this very entry point (--ship). Dispatched before argparse, which would reject the positional.
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        from pollard_tune import main as tune_main
        sys.exit(tune_main(sys.argv[2:]))
    # `pollard setup ...` -- is everything each lane needs here and working; offer to install what is not
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        from pollard_setup import main as setup_main
        sys.exit(setup_main(sys.argv[2:]))
    # `pollard release ...` -- every lane, each a ladder of rungs, gated, carded, (optionally) published
    if len(sys.argv) > 1 and sys.argv[1] == "release":
        from pollard_release import main as release_main
        sys.exit(release_main(sys.argv[2:]))
    # `pollard serve ...` -- the build's other half: run a rung for an agent harness (pollard_harness).
    # Dispatched before argparse, which would otherwise reject the positional as an unknown argument.
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        from pollard_harness import main as serve_main
        sys.exit(serve_main(sys.argv[2:]))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", help="f16/bf16 source GGUF (or use --hf to point at HF weights)")
    ap.add_argument("--hf", help="HuggingFace repo id OR local HF model dir -- Pollard downloads/converts/"
                    "routes it (so a user can one-shot straight from a repo or a model already on disk)")
    ap.add_argument("--format", default="gguf", choices=["gguf", "gptq", "mlx", "exl3", "mx", "exlr8"],
                    help="output lane: gguf (llama.cpp/Ollama, default)  |  gptq (vLLM/SGLang)  |  mlx (Apple) "
                    " |  exl3 (exllamav3 -- the heavy trellis lane)  |  mx (Blackwell NVFP4 / any-GPU W4A16, "
                    "compressed-tensors)  |  exlr8 (one trellis encode, the whole bpw ladder as manifests)")
    ap.add_argument("--output", help="output dir/file for the gptq/mlx/mx/exl3 export (else auto-named)")
    ap.add_argument("--results", help="measured numbers (pollard-bench --out) to put on the card")
    ap.add_argument("--no-card", dest="card", action="store_false",
                    help="skip writing the model card after a build")
    ap.set_defaults(card=True)
    ap.add_argument("--force", action="store_true",
                    help="ignore the machine lock and start even if another Pollard job is "
                         "running here. They will contend for the GPU, disk and cores.")
    ap.add_argument("--no-preflight", dest="preflight", action="store_false",
                    help="skip the up-front capability check (converter knows this architecture, "
                         "weights fit somewhere) and start the build regardless")
    ap.set_defaults(preflight=True)
    ap.add_argument("--probe-method", choices=("auto", "stream", "kl"), default="auto",
                    help="sensitivity estimator: 'kl' = perturb+KL (layers*groups forward passes), "
                         "'stream' = one-pass Hessian diagonal, 'auto' (default) picks stream when "
                         "the weights do not fit memory")
    ap.add_argument("--sensitivity", help="Pollard sensitivity.json (gptq/mlx/mx allocation; else auto-measured)")
    ap.add_argument("--no-measure", dest="measure", action="store_false",
                    help="skip the auto sensitivity probe on the gptq/mlx/mx lanes (falls back to uniform "
                         "allocation). By default the one-shot measures allocation (pollard-probe) -- the gold path.")
    ap.set_defaults(measure=True)
    ap.add_argument("--trust-remote-code", default="auto", choices=["auto", "on", "off"],
                    help="run a model's own modeling code for custom archs (Spark2_5 etc.); 'auto' enables "
                         "it only when config.json declares an auto_map. Passed through to the export lanes.")
    ap.add_argument("--focus-layers", help="steer the budget: force these layer indices to HIGH bits on "
                    "the export lanes (gptq/mlx/mx), e.g. '3,4,8' or '3-8,16'. Unions into the measured hot set.")
    ap.add_argument("--match-transformers", default="auto", choices=["auto", "on", "off"],
                    help="build a custom-arch model in a cached env pinned to the transformers version it was "
                         "SAVED with (config.json), so its remote code doesn't crash on a newer transformers. "
                         "'auto' = only when the majors differ; 'off' = always use the current env.")
    ap.add_argument("--imatrix", help="importance matrix (auto-generated from Calib 3.0 if omitted)")
    ap.add_argument("--calib", help="calibration corpus for auto-imatrix (else Calib 3.0 auto-built)")
    ap.add_argument("--calib-profile", default="auto", choices=["auto", "general", "translation", "agent", "code"],
                    help="pollard-calib domain mix for the auto-built corpus. 'auto' (default) picks it from "
                         "pollard-modelkind: agentic -> agent, coder -> code, translator -> translation, else general")
    ap.add_argument("--ngl", default="auto",
                    help="GPU layers for auto-imatrix. 'auto' (default) offloads the share that "
                         "fits this box and streams the rest; a number forces it.")
    ap.add_argument("--no-auto-imatrix", dest="auto_imatrix", action="store_false",
                    help="do NOT auto-generate an imatrix when --imatrix is omitted (K-quant ladder only)")
    ap.set_defaults(auto_imatrix=True)
    ap.add_argument("--ram", default="16", help="RAM budget in GB for the dense memory-fit build")
    ap.add_argument("--ladder", action="store_true",
                    help="build the whole publish ladder (Q6_K / IQ4_XS / IQ3_S / 2-bit) instead of one "
                         "--ram build: budgets are found by pollard-fit's own planner, the 2-bit rung is the "
                         "IQ*_KT flagship where ik_llama can build it and a stock IQ2 type where it cannot")
    ap.add_argument("--rung", action="append", metavar="LABEL",
                    help="with --ladder: build only this rung (Q6_K / IQ4_XS / IQ3_S / IQ2 / flagship); repeatable "
                         "or a comma list. Default: every rung + the flagship")
    ap.add_argument("--allow-slow", action="store_true",
                    help="let the passthrough sensitivity sweep run on a big model (it is one quantize + KL pass "
                         "per layer; pollard-sensitivity refuses multi-hour sweeps without this)")
    ap.add_argument("--out", help="output path (dense build)")
    ap.add_argument("--eval", default="wikitext2_test.txt")
    ap.add_argument("--bin", help="llama.cpp bin dir (for the MoE dry-run/build)")
    ap.add_argument("--smooth", dest="smooth", action="store_true", default=None,
                    help="SmoothQuant preconditioning (pollard-hf-smooth) on the FP16 model before the lane. "
                         "DEFAULT-ON for the low-bit trellis/error-feedback lanes (EXL3/GPTQ/MX) -- it's the "
                         "locked gold recipe there (prevents the massive-activation broken build). Use --no-smooth to skip.")
    ap.add_argument("--no-smooth", dest="smooth", action="store_false",
                    help="skip the default preconditioning on the EXL3/GPTQ/MX lanes")
    ap.add_argument("--abliterate", action="store_true",
                    help="OPT-IN: uncensor the FP16 model (pollard-abliterate) before the lane -- composes "
                         "across ALL lanes. Behaviour-changing, your own model; measure quality with pollard-kl")
    ap.add_argument("--harmful", help="abliterate: prompts to stop refusing (one/line)")
    ap.add_argument("--harmless", help="abliterate: matched benign prompts (one/line)")
    ap.add_argument("--run", action="store_true", help="execute the path (default: plan/print it)")
    ap.add_argument("--benchmark", "--reproduce", dest="benchmark", action="store_true",
                    help="ALSO run the gold-card benchmark (3-bar comparison + PPL) -- the "
                         "hour-long validation. OFF by default: a normal build makes ONE model "
                         "fast and skips the eval. Use this only to reproduce our published numbers.")
    ap.add_argument("--force-dense", action="store_true", help="override detection -> dense path")
    ap.add_argument("--force-moe", action="store_true", help="override detection -> MoE path")
    ap.add_argument("--no-toolcall", dest="toolcall", action="store_false",
                    help="skip the tool-call gate. By default an AGENTIC model (its chat template writes tool calls "
                         "in a syntax pollard-modelkind knows) gets pollard-toolcall on every rung this run built, in "
                         "every lane: 60 tool probes through the lane's own server (llama-server --jinja, "
                         "mlx_lm.server, vllm serve --tool-call-parser, exllamav3 in-process), retention vs the reference, into "
                         "results.json and the card's Tool-ready column. A lane this machine cannot serve is recorded "
                         "as not measured, never faked.")
    ap.add_argument("--toolcall", dest="toolcall", action="store_true",
                    help="(the default; kept so older command lines still parse)")
    ap.set_defaults(toolcall=True)
    ap.add_argument("--no-gate", dest="gate", action="store_false",
                    help="skip the auto coherence gate (loop check + sampling sweep) the one-shot "
                         "MoE build appends after the mix -- by default it runs so you learn if the "
                         "model is usable and which sampling to ship, without a manual step.")
    ap.set_defaults(gate=True)
    ap.add_argument("--skip-setup-check", action="store_true",
                    help="start even when `pollard setup` says this lane is missing a piece here")
    a = ap.parse_args()

    # First-run hook: a fast, cached (per environment) check that the requested lane can run here -- a one-line
    # hint when it cannot. With --run, a lane missing a piece stops HERE with the fix, not an hour into the build.
    try:
        import pollard_setup as S
        if a.run and not a.skip_setup_check:
            if S.require([a.format]):
                sys.exit(2)
        else:
            S.hint([a.format])
    except SystemExit:
        raise
    except Exception:                                            # noqa: BLE001  (a check never blocks a build)
        pass

    # One heavy job per machine. Everything below contends for the same GPU, disk and cores.
    with MachineLock(" ".join(sys.argv[1:])[:120], force=getattr(a, "force", False)) if a.run \
            else contextlib.nullcontext():
        _dispatch(a)


def _dispatch(a):
    a._t0 = time.time()
    if not a.gguf and not a.hf:
        # argparse-style usage error; `ap` lives in main(), so ap.error here was a NameError (Joey's review).
        sys.exit("pollard: error: pass --gguf <file> or --hf <repo-or-dir>")

    # LOCKED gold default: precondition (smooth) the low-bit trellis/error-feedback lanes unless opted out.
    # These lanes silently break on massive-activation outliers without it (EXL3 3090->8.699); smoothing +
    # our Calib 3.0 is the measured EXL3 win (8.670 < exl3-default 8.699). GGUF has its own smoothing; MLX not needed.
    if a.smooth is None:
        a.smooth = a.format in ("exl3", "exlr8", "gptq", "mx")
        if a.smooth:
            print(f"   [{a.format}] LOCKED default: preconditioning ON (--no-smooth to skip)")

    # NON-GGUF lanes (GPTQ/MLX) emit straight from HF weights -- route and done.
    if a.format in ("gptq", "mlx", "exl3", "mx", "exlr8"):
        print(f"pollard :: {a.hf or a.gguf}  -> {a.format.upper()} lane")
        # Auto-version onboarding: a custom-arch model saved with an older transformers major will crash
        # its remote code under the current one. Build the whole lane in a cached env pinned to the model's
        # transformers_version instead (Spark2_5 needs 4.57.1 vs a 5.x box). Re-invoke this same one-shot
        # under that env (with --match-transformers off to avoid recursion) so smooth/probe/export all match.
        if a.hf and a.match_transformers != "off":
            try:
                import pollard_envmatch as em
                target = (em.needs_matched_env(a.hf) if a.match_transformers == "auto"
                          else em.model_transformers_version(a.hf))
                if target:
                    print(f"   [match-transformers] {a.hf} was saved with transformers {target}; "
                          f"building in a matched env (its remote code needs it)")
                    if not a.run:
                        print("   (plan) --run would provision the matched env and build the lane there.")
                        return
                    tdir = em.ensure_overlay(target, a.format)
                    if tdir:
                        # Re-run the SAME python with the transformers-overlay dir prepended to PYTHONPATH,
                        # so this build (and its calib/probe/smooth subprocesses, which inherit the env)
                        # imports the pinned transformers while torch/gptqmodel stay from the base env.
                        argv = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + \
                               ["--match-transformers", "off"]
                        env = dict(os.environ)
                        env["POLLARD_AUTO"] = "1"   # silence pollard-fit's step-1 notice: we run the rest
                        env["PYTHONPATH"] = tdir + os.pathsep + env.get("PYTHONPATH", "")
                        sys.exit(subprocess.run(argv, env=env).returncode)
                    print("   [match-transformers] env setup failed -- falling back to the current env")
            except Exception as e:
                print(f"   [match-transformers] skipped ({e}); using the current env")
        out = _emit_nongguf(a)
        if a.run and out:
            _toolcall_gate(a, since=a._t0, lane_out=out)
        if not a.run:
            print("\n   plan only -- re-run with --run to execute.")
        return

    # GGUF lane: get an f16 GGUF (convert from HF if the user pointed at a repo/dir).
    if not a.gguf:
        a.gguf = _hf_to_gguf(a)
        if not a.run:                                      # plan mode: the GGUF doesn't exist yet
            print("\n   plan only -- re-run with --run to execute (convert + build).")
            return

    arch = analyse(gguf_to_config(read_gguf_meta(a.gguf), a.gguf))
    is_moe = (arch.get("n_experts") or 0) > 0 or "moe" in str(arch.get("kind", "")).lower()
    if a.force_dense: is_moe = False
    if a.force_moe: is_moe = True
    tag = "MoE" if is_moe else "DENSE"
    print(f"pollard :: {os.path.basename(a.gguf)}")
    print(f"   detected {tag}  ({(arch.get('total') or 0)/1e9:.1f}B total, "
          f"{(arch.get('active') or arch.get('total') or 0)/1e9:.1f}B active, {arch.get('layers')}L, "
          f"{arch.get('n_experts') or 0} experts)")
    # show the size ladder so the user sees the shrink + can compare to NVFP4/Q4/etc.
    # (actual built size is reported by the build step; this is where Pollard will land)
    size_ladder((arch.get("total") or 0) / 1e9)

    # ALREADY-QUANTIZED source (a vendor QAT GGUF): the passthrough policy. Its QAT tensors are copied
    # byte-for-byte; the imatrix and the sensitivity sweep run on this file as host; the measured
    # allocation only decides which tensors drop. No f16 ladder, no trellis flagship (both would
    # requantize everything).
    pt_tensors = _passthrough_source(a.gguf)
    if pt_tensors is not None:
        return _dispatch_passthrough(a, pt_tensors)

    # Autoaware: which lanes are actually open for THIS architecture (never plan an impossible build)
    ik_dir = find_ik_build(getattr(a, "bin", None))
    ik_known = lane_report(a.gguf, ik_dir=ik_dir)

    # WINNING PATH -- SAME shape for dense AND MoE (no losing fallback):
    #   (1) the imatrix K-quant ladder (pollard-fit) -- the honest, fit-your-RAM baseline, and
    #   (2) the mixed-precision FLAGSHIP mix (automap trellis) -- the hand-coded winner --
    #       WHEN an imatrix is given. No imatrix -> just the ladder (stock K-quants), which
    #       is the correct imatrix-free build; there is NO "imatrix-free mix" (it loses to Q2_K).
    flagship = "PollardMix expert-allocation" if is_moe else "IQ1_KT"
    print(f"   path: {tag} -> K-quant ladder (pollard-fit) + the {flagship} mixed-precision "
          f"flagship (the hand-coded winner) when an imatrix is present")
    # TRUE one-shot: auto-build the imatrix (Calib 3.0) if none supplied, so the DEFAULT output
    # is the flagship mix -- no manual calib/fit/calc step. --no-auto-imatrix opts back to ladder-only.
    if not a.imatrix and a.auto_imatrix:
        a.imatrix = _ensure_imatrix(a)
    # MEASURED allocation for the ladder too (not just uniform K-quants): if we have the HF weights
    # (came from --hf) and no profile was given, auto-probe one -- same gold lever as the export lanes.
    hf_dir = getattr(a, "_hf_dir", None)
    if hf_dir and not a.sensitivity and a.measure:
        here = os.path.dirname(os.path.abspath(a.gguf)) or "."
        calib = a.calib or os.path.join(here, "pollard_calib.txt")
        a.sensitivity = _ensure_sensitivity(a, hf_dir, calib, here)
    want = _wanted_rungs(a)
    if a.ladder and (want is None or want - {"flagship"}):
        _build_ladder(a, ik_flagship=bool(a.imatrix) and ik_known is True)
    elif a.ladder and want == {"flagship"}:
        # --rung flagship: ONLY the automap mix below. This used to fall through to the bare fit-your-RAM build
        # (pollard-fit --ram 16): the LFM2.5 release got an unasked 10.7 GB IQ3_S under the IQ3_S rung's name.
        print("   1) (the publish ladder is not in --rung this run -- flagship only)")
    else:
        cmd = ["pollard-fit", "--gguf", a.gguf, "--ram", str(a.ram)]
        if a.imatrix: cmd += ["--imatrix", a.imatrix]
        if a.sensitivity: cmd += ["--sensitivity", a.sensitivity]   # measured per-layer allocation
        if a.out: cmd += ["--out", a.out]
        if not a.run: cmd += ["--plan-only"]
        print("   1) the K-quant ladder (measured allocation, fits your RAM budget):")
        _run(cmd, a.run)
    if a.imatrix and ik_known is False:
        print(f"   2) (no {flagship} flagship: ik_llama.cpp does not know this architecture -- see above. "
              "The ladder is what ships until it is ported.)")
    elif a.imatrix and not ik_dir:
        # Every automap atom (IQ*_KT, iq1_bn, ...) is ik-only: stock llama-quantize rejects them, so
        # without an ik_llama build the flagship is not planned at all. With --ladder the 2-bit rung
        # was already given to a stock IQ2 type above (ik_flagship False).
        print(f"   2) (skipped the {flagship} flagship: no ik_llama.cpp build found, and its trellis atoms are "
              "ik-only" + (" -- the 2-bit rung is the stock IQ2 above" if a.ladder else "")
              + ". Build ik_llama.cpp and set IK_LLAMA_BIN (or --bin) to its build/bin to add it.)")
    elif a.imatrix and want is not None and "flagship" not in want:
        print(f"   2) (the {flagship} flagship is not in --rung this run)")
    elif a.imatrix:
        print(f"   2) the {flagship} mixed-precision flagship (automap trellis mix):")
        _automap_mix(a, is_moe=is_moe, ik_dir=ik_dir)
    else:
        print(f"   2) (--no-auto-imatrix set and no --imatrix: stock K-quant ladder only. Drop the "
              f"flag for the {flagship} flagship -- the winning build, auto-calibrated.)")
    if a.run:
        _toolcall_gate(a, since=getattr(a, "_t0", 0.0))
    if a.run and getattr(a, "card", True):
        _emit_card(a)
    if not a.run:
        print("\n   plan only -- re-run with --run to execute.")


def _dispatch_passthrough(a, tensors):
    from collections import Counter
    types = Counter(t["type"] for t in tensors if t["ndim"] >= 2)
    print("   source is ALREADY QUANTIZED (" + ", ".join(f"{n}x {t}" for t, n in types.most_common(5)) + ")")
    print("   path: PASSTHROUGH -- untouched tensors copied byte-for-byte, measured downgrades only, no upgrades;"
          " imatrix + sensitivity measured on this file as host")
    if not a.imatrix and a.auto_imatrix:
        a._passthrough = True
        a.imatrix = _ensure_imatrix(a)
    a.sensitivity = _ensure_passthrough_sensitivity(a)
    if a.ladder:
        _build_ladder_passthrough(a, tensors)
    else:
        cmd = ["pollard-fit", "--gguf", a.gguf, "--ram", str(a.ram), "--passthrough", "on"]
        if a.imatrix: cmd += ["--imatrix", a.imatrix]
        if a.sensitivity: cmd += ["--sensitivity", a.sensitivity]
        if a.out: cmd += ["--out", a.out]
        if a.bin: cmd += ["--llama-quantize", _bin_in(a.bin, "llama-quantize")]
        if not a.run: cmd += ["--plan-only"]
        print("   1) one passthrough build that fits --ram:")
        _run(cmd, a.run)
    print("   2) (no trellis flagship for a quantized source: automap would requantize every tensor, QAT ones included)")
    if a.run:
        _toolcall_gate(a, since=getattr(a, "_t0", 0.0))
    if a.run and getattr(a, "card", True):
        _emit_card(a)
    if not a.run:
        print("\n   plan only -- re-run with --run to execute.")


if __name__ == "__main__":
    main()
