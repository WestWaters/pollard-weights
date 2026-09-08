#!/usr/bin/env python3
"""pollard — the autoaware entry point. Point it at ANY model; it detects dense vs MoE
and drives the WINNING method, so a user never runs a losing path or wastes hours.

WINNING PATH — the SAME for dense AND MoE (no losing fallback):
  (1) the imatrix K-quant ladder (pollard-fit) — the honest fit-your-RAM baseline, plus
  (2) the mixed-precision FLAGSHIP mix (automap trellis) — the hand-coded winner (crush body,
      protect attn/down/first-last; MoE = expert-allocation) — WHEN an imatrix is present.
  No --imatrix supplied -> AUTO-BUILD one (Calib 3.0 corpus -> llama-imatrix) so the DEFAULT is the
  flagship mix with ZERO manual steps. --no-auto-imatrix falls back to the stock K-quant ladder.
  (There is NO "imatrix-free mix": it loses to stock Q2_K, so it's deprecated, not a path here.)

⛔ LOSING / TRAP paths — never the default, opt-in only:
  - sensitivity SWEEP on DENSE = loses (no expert redundancy) -> pollard-sensitivity refuses dense.
  - sensitivity SWEEP on a BIG MoE = ~2*layers full-model quantizes = many HOURS; opt-in R&D only.
  - the 3-bar comparison + KL/PPL = the BENCHMARK, opt-in via --benchmark (see benchmarks/).

Plans by default (prints the exact commands for THIS model); `--run` executes them.

    pollard --gguf model-f16.gguf --run             # ONE-SHOT GGUF: auto-calib -> imatrix -> flagship mix
    pollard --hf Qwen/Qwen3-8B --run                # ONE-SHOT from a HF repo: download -> convert -> build
    pollard --hf ./my-local-model --run             # ...or a model already on disk (any arch)
    pollard --hf Qwen/Qwen3-8B --format gptq --run  # GPTQ for vLLM/SGLang
    pollard --hf Qwen/Qwen3-8B --format mlx --run   # MLX for Apple Silicon
    pollard --hf Qwen/Qwen3-8B --format exl3 --run  # EXL3 (exllamav3 — heavy trellis)
    pollard --hf Qwen/Qwen3-8B --format mx --run    # MX: Blackwell NVFP4 / any-GPU W4A16 (compressed-tensors)
    pollard --gguf model-f16.gguf --imatrix m.imatrix --run    # bring your own imatrix (skips auto-calib)
    pollard --gguf model-f16.gguf --benchmark --run            # + the gold-card board (slow)

EVERY lane runs the GOLD Pollard method one-shot: GGUF = auto Calib-3.0 imatrix -> measured automap mix
-> coherence gate; GPTQ/MLX/MX = smoothing (default, low-bit lanes) + auto-measured allocation (a cheap
pollard-probe, --no-measure to skip); EXL3 = smoothing + Calib 3.0 packed to -cd + EXL3's native allocator.
"""
import argparse, os, subprocess, sys

from pollard_calc import read_gguf_meta, gguf_to_config, analyse, find_llama_bin

# reference bits-per-weight for the common formats, so a user can see where a Pollard
# build lands vs f16 / NVFP4 / the usual GGUF tiers ("half the size of NVFP4" etc.).
_REF_BPW = [("f16", 16.0), ("Q8_0", 8.5), ("Q6_K", 6.6), ("Q5_K_M", 5.5),
            ("NVFP4", 4.25), ("Q4_K_M", 4.85), ("IQ2 (~2-bit)", 2.1)]


def size_ladder(params_b, pollard_gb=None, pollard_label="Pollard"):
    """Print the shrink story: f16 size, where Pollard's build lands, and the same model
    at each reference format — so people SEE what Pollard did and can compare to NVFP4."""
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


def _run(cmd, do_run, cwd=None):
    print("   $ " + " ".join(str(c) for c in cmd))
    if do_run:
        r = subprocess.run(cmd, cwd=cwd)
        if r.returncode != 0:
            sys.exit(f"   step failed (exit {r.returncode}).")
    return not do_run


def _automap_mix(a, is_moe):
    """Emit + (with --run) build the automap mix — the hand-coded mixed-precision flagship.
    MoE: expert-allocation (crush cold experts, protect router/down/shared/attn). DENSE: the
    IQ1_KT crush-body / protect-attn+down+edges flagship (automap-on-dense, --allow-dense).
    Fast by default (--mix-only --no-eval); --benchmark emits the 3-bar + PPL board instead.
    This is what keeps the winning hand-coded mix a first-class BUILD, not benchmark-only."""
    binq = find_llama_bin("llama-quantize") if not a.bin else os.path.join(a.bin, "llama-quantize")
    here = os.path.dirname(os.path.abspath(a.gguf)) or "."
    tensors = os.path.join(here, "pollard_auto_tensors.txt")
    print(f"   tensor list (Q6_K dry-run): {binq} --dry-run {os.path.basename(a.gguf)} x.gguf Q6_K > tensors")
    if a.run:
        with open(tensors, "w") as f:
            subprocess.run([binq, "--dry-run", a.gguf, "x.gguf", "Q6_K"],
                           stdout=f, stderr=subprocess.STDOUT)
    out = os.path.join(here, "pollard_auto_build.bat" if is_moe else "pollard_auto_flagship.bat")
    # the flagship is the TRELLIS mix (winner) — always imatrix-guided. No K-quant fallback.
    am = ["pollard-automap", "--tensors", tensors, "--model", a.gguf, "--out", out,
          "--imatrix", a.imatrix]
    if not is_moe:
        am += ["--allow-dense"]                             # dense flagship = the hand-coded mix
    if not a.benchmark:
        am += ["--mix-only", "--no-eval"]                   # plain build = ONE model, no benchmark
    if not getattr(a, "gate", True):
        am += ["--no-gate"]                                 # user opted out of the auto coherence gate
    if a.bin:
        am += ["--bin", a.bin]
    _run(am, a.run, cwd=here)
    mode = "3-bar BENCHMARK + PPL" if a.benchmark else "ONE mix model, no eval (fast)"
    print(f"   -> run {os.path.basename(out)} on the box -> {mode}"
          + ("" if a.benchmark else "  (add --benchmark for the gold-card numbers)"))
    return out


def _find_convert():
    """Locate convert_hf_to_gguf.py (our runtime llama.cpp first, then PATH/common spots)."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for c in (os.path.join(repo, "runtime", "llama.cpp", "convert_hf_to_gguf.py"),
              os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py")):
        if os.path.exists(c):
            return c
    return "convert_hf_to_gguf.py"                          # assume on PATH / same dir


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
        if a.calib: cmd += ["--calib", a.calib]
        print(f"   [precondition] smooth (SmoothQuant, FP16): {' '.join(cmd)}")
        if a.run and subprocess.run(cmd).returncode != 0:
            sys.exit("   smooth failed.")
        cur = out
    return cur


def _resolve_hf(a):
    """A local HF dir is used as-is; a repo id is downloaded (snapshot) so users can point at either.
    Opt-in --abliterate/--smooth transforms are applied here so every lane inherits them."""
    if os.path.isdir(a.hf):
        a._hf_dir = _precondition_hf(a, a.hf)
        return a._hf_dir
    local = os.path.join(os.path.abspath(a.output or "."), a.hf.split("/")[-1])
    print(f"   fetch HF repo: huggingface-cli download {a.hf} --local-dir {local}")
    if a.run:
        from huggingface_hub import snapshot_download
        local = snapshot_download(a.hf, local_dir=local)
    a._hf_dir = _precondition_hf(a, local)
    return a._hf_dir


def _hf_to_gguf(a):
    """HF weights -> f16 GGUF so the GGUF flagship pipeline can run one-shot from a repo/dir."""
    hf_dir = _resolve_hf(a)
    conv = _find_convert()
    here = os.path.abspath(a.output) if a.output else os.path.dirname(os.path.abspath(hf_dir)) or "."
    out = os.path.join(here, os.path.basename(hf_dir.rstrip("/\\")) + "-f16.gguf")
    print(f"   convert HF -> f16 GGUF: python {os.path.basename(conv)} {hf_dir} --outtype f16 --outfile {out}")
    if a.run:
        subprocess.run([sys.executable, conv, hf_dir, "--outtype", "f16", "--outfile", out])
    return out


def _ensure_calib_text(a, here):
    """Return a Calib 3.0 text corpus path, auto-building it if the user gave none (one-shot = no
    manual calibration step). Shared by the gptq / mx / exl3 lanes."""
    calib = a.calib or os.path.join(here, "pollard_calib.txt")
    if not a.calib:
        print(f"   auto-calib (Calib 3.0): pollard-calib --out {os.path.basename(calib)} --held-out {os.path.basename(calib)}.heldout")
        if a.run and not os.path.exists(calib):
            _run(["pollard-calib", "--out", calib, "--held-out", calib + ".heldout"], True, cwd=here)
    return calib


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
        # tolerate a probe failure — fall back to uniform rather than killing the whole build
        r = subprocess.run(["pollard-probe", "--model", hf_dir, "--eval", evalf, "--out", prof], cwd=here)
        if r.returncode != 0 or not os.path.exists(prof):
            print("   (probe unavailable/failed — falling back to uniform allocation for this lane)")
            return None
    return prof


def _emit_nongguf(a):
    """GPTQ (vLLM/SGLang), MLX (Apple), EXL3 (exllamav3), and MX (Blackwell/any-GPU compressed-tensors)
    emit straight from HF weights — same Pollard method (smoothing default + measured allocation), a
    different emitter. Calib 3.0 and the sensitivity profile are auto-built so it's a true one-shot."""
    if not a.hf:
        sys.exit(f"--format {a.format} exports from HF weights — pass --hf <repo-or-dir> "
                 f"(a GGUF can't be re-exported to {a.format}; use --format gguf for a GGUF input).")
    hf_dir = _resolve_hf(a)
    out = a.output or (os.path.basename(hf_dir.rstrip("/\\")) + f"-Pollard-{a.format.upper()}")
    here = os.path.dirname(os.path.abspath(out)) or "."

    trc = ["--trust-remote-code", a.trust_remote_code]     # custom-arch passthrough (Spark2_5 etc.)
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
    elif a.format == "exl3":
        # GOLD EXL3 = smoothing (applied in _resolve_hf) + Calib 3.0 (-cd) + EXL3's native allocator.
        calib = _ensure_calib_text(a, here)
        cmd = ["pollard-exl3", "--model", hf_dir, "--out", out, "--calib-text", calib]
    else:                                                   # mlx (Apple) — measured 4/8 mix; smoothing N/A
        calib = _ensure_calib_text(a, here)                # only for the probe's held-out eval corpus
        sens = _ensure_sensitivity(a, hf_dir, calib, here)
        cmd = ["pollard-mlx", "--model", hf_dir, "--out", out] + trc
        if sens:
            cmd += ["--sensitivity", sens]
    print(f"   {a.format.upper()} export (Pollard method — smoothing default + measured allocation):")
    _run(cmd, a.run)
    _rt = {"gptq": "vllm serve / sglang", "mlx": "mlx_lm.generate",
           "exl3": "exllamav3 / TabbyAPI", "mx": "vllm serve (compressed-tensors)"}
    print(f"   -> {out}  ({_rt.get(a.format, '')})")


def _ensure_imatrix(a):
    """TRUE one-shot: if the user gave no --imatrix, auto-build one (Calib 3.0 multi-domain
    corpus -> llama-imatrix) so they never run a manual calibration step. Returns the imatrix
    path (and, with --run, actually builds it). Plan mode just prints the exact commands."""
    here = os.path.dirname(os.path.abspath(a.gguf)) or "."
    calib = a.calib or os.path.join(here, "pollard_calib.txt")
    imat = os.path.join(here, os.path.splitext(os.path.basename(a.gguf))[0] + ".imatrix")
    binim = (os.path.join(a.bin, "llama-imatrix") if a.bin
             else find_llama_bin("llama-imatrix")) or "llama-imatrix"
    print("   0) auto-imatrix (Calib 3.0 -> llama-imatrix) — no manual calibration step:")
    if not a.calib:
        print(f"      pollard-calib --out {os.path.basename(calib)}")
    print(f"      {binim} -m {os.path.basename(a.gguf)} -f {os.path.basename(calib)} "
          f"-o {os.path.basename(imat)} -ngl {a.ngl}")
    if a.run:
        if not a.calib and not os.path.exists(calib):
            _run(["pollard-calib", "--out", calib], True, cwd=here)
        subprocess.run([binim, "-m", a.gguf, "-f", calib, "-o", imat, "-ngl", str(a.ngl)], cwd=here)
    print("      (big MoE won't fit f16 for the forward pass -> compute on a Q6_K host at a "
          "partial --ngl; see SKILL.md. Undercovered experts hard-fail low-bit — Calib 3.0 covers them.)")
    return imat


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", help="f16/bf16 source GGUF (or use --hf to point at HF weights)")
    ap.add_argument("--hf", help="HuggingFace repo id OR local HF model dir — Pollard downloads/converts/"
                    "routes it (so a user can one-shot straight from a repo or a model already on disk)")
    ap.add_argument("--format", default="gguf", choices=["gguf", "gptq", "mlx", "exl3", "mx"],
                    help="output lane: gguf (llama.cpp/Ollama, default) · gptq (vLLM/SGLang) · mlx (Apple) "
                    "· exl3 (exllamav3 — the heavy trellis lane) · mx (Blackwell NVFP4 / any-GPU W4A16, "
                    "compressed-tensors)")
    ap.add_argument("--output", help="output dir/file for the gptq/mlx/mx/exl3 export (else auto-named)")
    ap.add_argument("--sensitivity", help="Pollard sensitivity.json (gptq/mlx/mx allocation; else auto-measured)")
    ap.add_argument("--no-measure", dest="measure", action="store_false",
                    help="skip the auto sensitivity probe on the gptq/mlx/mx lanes (falls back to uniform "
                         "allocation). By default the one-shot measures allocation (pollard-probe) — the gold path.")
    ap.set_defaults(measure=True)
    ap.add_argument("--trust-remote-code", default="auto", choices=["auto", "on", "off"],
                    help="run a model's own modeling code for custom archs (Spark2_5 etc.); 'auto' enables "
                         "it only when config.json declares an auto_map. Passed through to the export lanes.")
    ap.add_argument("--match-transformers", default="auto", choices=["auto", "on", "off"],
                    help="build a custom-arch model in a cached env pinned to the transformers version it was "
                         "SAVED with (config.json), so its remote code doesn't crash on a newer transformers. "
                         "'auto' = only when the majors differ; 'off' = always use the current env.")
    ap.add_argument("--imatrix", help="importance matrix (auto-generated from Calib 3.0 if omitted)")
    ap.add_argument("--calib", help="calibration corpus for auto-imatrix (else Calib 3.0 auto-built)")
    ap.add_argument("--ngl", default="99", help="GPU layers for auto-imatrix (lower for a big model)")
    ap.add_argument("--no-auto-imatrix", dest="auto_imatrix", action="store_false",
                    help="do NOT auto-generate an imatrix when --imatrix is omitted (K-quant ladder only)")
    ap.set_defaults(auto_imatrix=True)
    ap.add_argument("--ram", default="16", help="RAM budget in GB for the dense memory-fit build")
    ap.add_argument("--out", help="output path (dense build)")
    ap.add_argument("--eval", default="wikitext2_test.txt")
    ap.add_argument("--bin", help="llama.cpp bin dir (for the MoE dry-run/build)")
    ap.add_argument("--smooth", dest="smooth", action="store_true", default=None,
                    help="SmoothQuant preconditioning (pollard-hf-smooth) on the FP16 model before the lane. "
                         "DEFAULT-ON for the low-bit trellis/error-feedback lanes (EXL3/GPTQ/MX) — it's the "
                         "locked gold recipe there (prevents the massive-activation broken build). Use --no-smooth to skip.")
    ap.add_argument("--no-smooth", dest="smooth", action="store_false",
                    help="skip the default preconditioning on the EXL3/GPTQ/MX lanes")
    ap.add_argument("--abliterate", action="store_true",
                    help="OPT-IN: uncensor the FP16 model (pollard-abliterate) before the lane — composes "
                         "across ALL lanes. Behaviour-changing, your own model; measure quality with pollard-kl")
    ap.add_argument("--harmful", help="abliterate: prompts to stop refusing (one/line)")
    ap.add_argument("--harmless", help="abliterate: matched benign prompts (one/line)")
    ap.add_argument("--run", action="store_true", help="execute the path (default: plan/print it)")
    ap.add_argument("--benchmark", "--reproduce", dest="benchmark", action="store_true",
                    help="ALSO run the gold-card benchmark (3-bar comparison + PPL) — the "
                         "hour-long validation. OFF by default: a normal build makes ONE model "
                         "fast and skips the eval. Use this only to reproduce our published numbers.")
    ap.add_argument("--force-dense", action="store_true", help="override detection -> dense path")
    ap.add_argument("--force-moe", action="store_true", help="override detection -> MoE path")
    ap.add_argument("--no-gate", dest="gate", action="store_false",
                    help="skip the auto coherence gate (loop check + sampling sweep) the one-shot "
                         "MoE build appends after the mix — by default it runs so you learn if the "
                         "model is usable and which sampling to ship, without a manual step.")
    ap.set_defaults(gate=True)
    a = ap.parse_args()
    if not a.gguf and not a.hf:
        ap.error("pass --gguf <file> or --hf <repo-or-dir>")

    # LOCKED gold default: precondition (smooth) the low-bit trellis/error-feedback lanes unless opted out.
    # These lanes silently break on massive-activation outliers without it (EXL3 3090→8.699); smoothing +
    # our Calib 3.0 is the measured EXL3 win (8.670 < exl3-default 8.699). GGUF has its own smoothing; MLX not needed.
    if a.smooth is None:
        a.smooth = a.format in ("exl3", "gptq", "mx")
        if a.smooth:
            print(f"   [{a.format}] LOCKED default: preconditioning ON (--no-smooth to skip)")

    # NON-GGUF lanes (GPTQ/MLX) emit straight from HF weights — route and done.
    if a.format in ("gptq", "mlx", "exl3", "mx"):
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
                        env["PYTHONPATH"] = tdir + os.pathsep + env.get("PYTHONPATH", "")
                        sys.exit(subprocess.run(argv, env=env).returncode)
                    print("   [match-transformers] env setup failed — falling back to the current env")
            except Exception as e:
                print(f"   [match-transformers] skipped ({e}); using the current env")
        _emit_nongguf(a)
        if not a.run:
            print("\n   plan only — re-run with --run to execute.")
        return

    # GGUF lane: get an f16 GGUF (convert from HF if the user pointed at a repo/dir).
    if not a.gguf:
        a.gguf = _hf_to_gguf(a)
        if not a.run:                                      # plan mode: the GGUF doesn't exist yet
            print("\n   plan only — re-run with --run to execute (convert + build).")
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

    # WINNING PATH — SAME shape for dense AND MoE (no losing fallback):
    #   (1) the imatrix K-quant ladder (pollard-fit) — the honest, fit-your-RAM baseline, and
    #   (2) the mixed-precision FLAGSHIP mix (automap trellis) — the hand-coded winner —
    #       WHEN an imatrix is given. No imatrix -> just the ladder (stock K-quants), which
    #       is the correct imatrix-free build; there is NO "imatrix-free mix" (it loses to Q2_K).
    flagship = "PollardMix expert-allocation" if is_moe else "IQ1_KT"
    print(f"   path: {tag} -> K-quant ladder (pollard-fit) + the {flagship} mixed-precision "
          f"flagship (the hand-coded winner) when an imatrix is present")
    # TRUE one-shot: auto-build the imatrix (Calib 3.0) if none supplied, so the DEFAULT output
    # is the flagship mix — no manual calib/fit/calc step. --no-auto-imatrix opts back to ladder-only.
    if not a.imatrix and a.auto_imatrix:
        a.imatrix = _ensure_imatrix(a)
    # MEASURED allocation for the ladder too (not just uniform K-quants): if we have the HF weights
    # (came from --hf) and no profile was given, auto-probe one — same gold lever as the export lanes.
    hf_dir = getattr(a, "_hf_dir", None)
    if hf_dir and not a.sensitivity and a.measure:
        here = os.path.dirname(os.path.abspath(a.gguf)) or "."
        calib = a.calib or os.path.join(here, "pollard_calib.txt")
        a.sensitivity = _ensure_sensitivity(a, hf_dir, calib, here)
    cmd = ["pollard-fit", "--gguf", a.gguf, "--ram", str(a.ram)]
    if a.imatrix: cmd += ["--imatrix", a.imatrix]
    if a.sensitivity: cmd += ["--sensitivity", a.sensitivity]   # measured per-layer allocation
    if a.out: cmd += ["--out", a.out]
    if not a.run: cmd += ["--plan-only"]
    print("   1) the K-quant ladder (measured allocation, fits your RAM budget):")
    _run(cmd, a.run)
    if a.imatrix:
        print(f"   2) the {flagship} mixed-precision flagship (automap trellis mix):")
        _automap_mix(a, is_moe=is_moe)
    else:
        print(f"   2) (--no-auto-imatrix set and no --imatrix: stock K-quant ladder only. Drop the "
              f"flag for the {flagship} flagship — the winning build, auto-calibrated.)")
    if not a.run:
        print("\n   plan only — re-run with --run to execute.")


if __name__ == "__main__":
    main()
