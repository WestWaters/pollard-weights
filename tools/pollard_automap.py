#!/usr/bin/env python3
"""pollard-automap — generate the memory-fit mix recipe for ANY model, automatically.

The Pollard POLICY (measured on 7B, Grok-blessed) as a function of the model's tensor
list, so we never hand-roll a map again:
  - crush the fat body               -> IQ1_KT   (dense: ffn_gate/up ; MoE: the *_exps experts)
  - keep ffn_down LOW                -> IQ2_KT   (raising it to IQ3 leaves the 1-bit size class -> IQ2 wins)
  - fat attention (q, output)        -> IQ2_KT   ; k,v reclaimed -> IQ1_KT
  - protect first-2 / last-2 blocks  -> IQ2_KT
  - MoE router (ffn_gate_inp) + shared experts -> keep high (Q6_K / never IQ1)
  - fat head/embeddings              -> output Q6_K , token_embd Q4_K  (never < 4-bit)
  - norms                            -> F32

Reads a `llama-quantize --dry-run` tensor list (the ground truth of what's actually in
the GGUF), detects layer count + dense/MoE, and emits the three build commands
(uniform IQ1_KT baseline / PollardMix / uniform IQ2_KT ceiling) as a ready .bat.

Usage:
  # on the box:  llama-quantize --dry-run ... model-f16.gguf x.gguf IQ1_KT > tensors.txt
  pollard-automap --tensors tensors.txt --model model-f16.gguf --imatrix ik.imatrix \
      --out build_mix.bat --bin C:\\pollard\\ik_llama.cpp\\build\\bin
"""
import argparse, re, sys


def imatrix_covered(path):
    """The set of tensor names an ik_llama imatrix ACTUALLY covers. ik writes the OLD
    binary format (not GGUF): int32 n_entries, then per entry int32 name_len, name,
    int32 ncall, int32 nval, float[nval]. A MoE routes to only some experts over a short
    calib, so many `*_exps` tensors get NO entry — and a very-low-bit build hard-fails on
    an uncovered tensor. Read the real coverage so we can pin the uncovered ones. Returns
    the set, or None if it can't be parsed (caller then skips pinning)."""
    import struct
    try:
        d = open(path, "rb").read()
        n = struct.unpack_from("<i", d, 0)[0]; off = 4
        cov = set()
        for _ in range(n):
            ln = struct.unpack_from("<i", d, off)[0]; off += 4
            if ln <= 0 or ln > 512:
                return None
            cov.add(d[off:off + ln].decode("utf-8", "replace")); off += ln
            nval = struct.unpack_from("<i", d, off + 4)[0]; off += 8 + 4 * nval
        return cov
    except Exception:
        return None


# matmul/expert tensors that a very-low-bit build needs an imatrix for; anything here
# NOT covered by the imatrix must be pinned to a non-imatrix type or the build hard-fails.
_NEEDS_IMATRIX = re.compile(
    r"blk\.\d+\.(ffn_(up|down|gate)(_exps|_shexp)?|attn_(q|k|v|qkv|output))\.weight$")


def uncovered_pins(all_names, imatrix, fallback="q6_K"):
    """--custom-q rules pinning every imatrix-REQUIRED tensor the imatrix doesn't cover
    to `fallback`, so an aggressive MoE build can't crash on a rarely-routed expert. THIS
    is what makes automap robust on MoE. Empty if the imatrix can't be read (build may
    still fail on uncovered experts — rerun the imatrix with more/diverse chunks)."""
    cov = imatrix_covered(imatrix)
    if cov is None:
        return [], None
    pins = [rf"{re.escape(nm)}={fallback}" for nm in all_names
            if _NEEDS_IMATRIX.search(nm) and nm not in cov]
    return pins, len(cov)


def parse_tensors(path):
    """Return (names, n_layers, is_moe, is_hyv4). Accepts dry-run lines or bare tensor names."""
    names = []
    for ln in open(path, encoding="utf-8", errors="ignore"):
        m = re.search(r"(blk\.\d+\.[\w.]+|token_embd\.weight|output[\w.]*\.weight|output_norm\.weight)", ln)
        if m:
            names.append(m.group(1))
    names = sorted(set(names))
    layers = [int(m.group(1)) for n in names for m in [re.match(r"blk\.(\d+)\.", n)] if m]
    n_layers = (max(layers) + 1) if layers else 0
    is_moe = any("exps" in n or "ffn_gate_inp" in n for n in names)
    # HYV4 (Tencent Hy4-preview / hy_v4): a Deepseek-derived MLA + DSA-indexer + hyper-connection
    # MoE. Its tensors (hc_*, indexer.*, attn_k_b/v_b, exp_probs_b) are named nothing like Qwen3-
    # MoE, so it needs its OWN recipe — the THIRD canonical path.
    is_hyv4 = any(".hc_" in n or ".indexer." in n or "attn_k_b" in n or "exp_probs_b" in n
                  for n in names)
    return names, n_layers, is_moe, is_hyv4


# ---- the real ik_llama.cpp extreme-low ladder (from `llama-quantize --help`) ----
# bpw + family, so a chosen atom is picked from what the runtime actually ships and
# users can dial the crush all the way to the floor instead of stopping at 1.75.
QUANT_BPW = {
    "iq1_s_r4": 1.50, "iq1_s": 1.56, "iq1_bn": 1.62,   # sub-1.75 floor (iq1_bn = Bitnet ternary)
    "iq1_kt": 1.75, "iq1_m": 1.75,                      # trellis / i-quant 1.75
    "iq2_bn": 2.00, "iq2_xxs": 2.06, "iq2_kt": 2.125,   # ~2-bit band (iq2_bn = Bitnet ternary)
    "iq2_ks": 2.19, "iq2_xs": 2.31, "iq2_k": 2.375,
    "iq3_kt": 3.125,
    # imatrix-FREE K-quants: these build with NO importance matrix (the MoE decoupled path),
    # so a MoE never has to go through the 6-hour, coverage-hungry, kill-prone imatrix step.
    # These are the SINGLE ggml tensor types (what --custom-q takes); the uniform-bar
    # positional arg uses the matching preset via _bar_type().
    "q2_k": 2.63, "q3_k": 3.91, "q4_k": 4.85, "q5_k": 5.50, "q6_k": 6.56,
}

# uniform-bar positional preset for a single K-quant type (custom-q takes q4_k; the whole-
# model positional arg wants the preset Q4_K_M). Trellis/I-quants pass through as-is (upper).
_BAR_PRESET = {"q2_k": "Q2_K", "q3_k": "Q3_K_M", "q4_k": "Q4_K_M", "q5_k": "Q5_K_M", "q6_k": "Q6_K"}


def is_kquant(atom):
    """A K-quant (QN_K*) builds WITHOUT an imatrix; an I-/trellis quant (IQ*, *_KT) requires
    one. This is what lets the MoE path skip the imatrix entirely."""
    return bool(re.match(r"q\d_k", (atom or "").lower()))


def _bar_type(atom):
    """The positional quantize preset for a uniform bar of `atom` (K-quant single type ->
    its preset; trellis/I-quant -> the uppercase type name)."""
    a = _atom(atom)
    return _BAR_PRESET.get(a, a.upper())


def _cq(atom):
    """The EXACT ggml type name --custom-q expects. K-quants are `qN_K` (capital K, e.g.
    q3_K); i-/trellis quants are all-lowercase (iq1_kt). llama-quantize rejects `q3_k`
    ('Invalid quantization type') — the tensor-type table is case-sensitive on the K."""
    a = _atom(atom)
    m = re.match(r"(q\d)_k$", a)
    return f"{m.group(1)}_K" if m else a
# Frontier mixed-ultra-low formats (e.g. Hy4 MIX-STQ1_0's sparse-ternary STQ1_0) map onto
# the nearest thing this runtime can actually emit: Bitnet ternary.
ALIASES = {"stq1_0": "iq1_bn", "stq2_0": "iq2_bn"}
# atoms known to route reliably through --custom-q (lowercase); the trellis role-flags
# silently fell back for some types (Mix-v3 lesson), so the Mix goes 100% through custom-q.
BODY_CHOICES = ["iq1_kt", "iq1_bn", "iq1_s", "iq1_s_r4", "iq2_xxs", "iq2_kt"]
PROTECT_CHOICES = ["iq2_kt", "iq2_xxs", "iq2_k", "iq3_kt"]


def _atom(name):
    return ALIASES.get(name.lower(), name.lower())


def recipe_flags(n_layers, is_moe, body="iq1_kt", protect="iq2_kt"):
    """Emit the Mix as (base_type, custom-q rules). base_type = the crush atom (fills
    everything not matched); every protected role is named explicitly via custom-q
    (lowercase = the reliable path — role-flags silently fell back for some atoms).
    Rules are ordered general->specific; edge-block rules go LAST so they win.
    Verified against a dry-run (which prints the actual per-tensor type chosen).

    If body+protect are BOTH K-quants, the recipe is imatrix-free (the decoupled MoE
    path): the one hard-coded trellis atom (the shared-expert writer) drops to the
    protect K-quant so no tensor needs an importance matrix."""
    kfree = is_kquant(body) and is_kquant(protect)      # fully imatrix-free build?
    shexp_down = protect if kfree else "iq3_kt"         # trellis atom only when imatrix is present
    # custom-q needs the exact ggml type name (K-quants capital-K: q3_K, not q3_k).
    body, protect, shexp_down = _cq(body), _cq(protect), _cq(shexp_down)
    flags = ["--output-tensor-type Q6_K", "--token-embedding-type Q4_K"]  # head/embed: never < 4-bit
    cq = []
    # --custom-q is FIRST-MATCH-WINS (verified via dry-run): list overrides first.
    # (1) edge whole-block protection FIRST so it beats the general role rules below,
    #     fully protecting the first-2 / last-2 blocks (attn AND ffn), as the flag-based
    #     original did (custom-q used to override the role flags).
    edge = [0, 1, n_layers - 2, n_layers - 1]
    for i in sorted(set(x for x in edge if 0 <= x < n_layers)):
        cq.append(rf"blk\.{i}\.={protect}")
    if is_moe:
        # (2) crush the cold bulk experts to the body atom; keep the residual writer
        # (ffn_down_exps) a tier up, the router high, and the shared experts protected.
        cq += [f"ffn_gate_exps={body}", f"ffn_up_exps={body}", f"ffn_down_exps={protect}",
               "ffn_gate_inp=q6_K",                       # router: never crushed
               f"ffn_gate_shexp={protect}", f"ffn_up_shexp={protect}", f"ffn_down_shexp={shexp_down}"]
    # (3) general roles. Grok's policy: PROTECT attn v/o (they carry the distribution); q/k
    # are less critical. On DENSE the shipped 7B/14B recipe crushed k,v and still won, so keep
    # it. On MoE, crushing attn_v was measured to LOSE KLD vs uniform IQ1 (30B: mix 0.371 >
    # uniform 0.360) — protect attn_v/k there (attention is a small fraction of a MoE anyway).
    attn_kv = protect if (kfree or is_moe) else body
    cq += [f"attn_k={attn_kv}", f"attn_v={attn_kv}",
           f"attn_q={protect}", f"attn_output={protect}", f"ffn_down={protect}"]
    return flags, cq


def recipe_flags_hyv4(n_layers, body="iq1_kt", protect="iq2_kt"):
    """THE HYV4 canonical recipe (Tencent Hy4-preview / hy_v4) — the THIRD path. A Deepseek-
    derived MLA + DSA-indexer + hyper-connection MoE, tensors named nothing like Qwen3-MoE.
    Policy: crush the fat expert body (ffn_gate/up_exps); protect the residual writer
    (ffn_down_exps), shared experts, dense block-0 FFN, router, ALL MLA attention, the
    hyper-connections (hc_*_fn), the DSA indexer, and the output hc. Norms / base / scale /
    exp_probs_b / attn_sinks stay F32 (llama-quantize keeps those F32 — no rule needed).
    attn_k_b kept a tier HIGHER: Frank measured IQ4_XS regressed it (MLA absorption weight).
    Rules use `\\.weight` so `attn_q_a` doesn't also grab `attn_q_a_norm`.  v1 — MEASURE on the
    real model (Frank's rig) and tighten like any card; do NOT assume it's final."""
    body, protect = _cq(body), _cq(protect)
    kb = "q8_0" if is_kquant(protect) else _cq("iq3_kt")   # attn_k_b one tier up
    flags = ["--output-tensor-type Q6_K", "--token-embedding-type Q4_K"]
    cq = []
    edge = [0, 1, n_layers - 2, n_layers - 1]
    for i in sorted(set(x for x in edge if 0 <= x < n_layers)):
        cq.append(rf"blk\.{i}\.={protect}")                # edge whole-block protect (incl. dense blk.0)
    cq += [
        # crush the fat expert body
        f"ffn_gate_exps\\.weight={body}", f"ffn_up_exps\\.weight={body}",
        # residual writer + shared experts + dense block-0 FFN + router
        f"ffn_down_exps\\.weight={protect}",
        f"ffn_gate_shexp\\.weight={protect}", f"ffn_up_shexp\\.weight={protect}",
        f"ffn_down_shexp\\.weight={protect}",
        f"ffn_gate\\.weight={protect}", f"ffn_up\\.weight={protect}", f"ffn_down\\.weight={protect}",
        "ffn_gate_inp\\.weight=q6_K",
        # MLA attention — all protected; k_b a tier higher
        f"attn_q_a\\.weight={protect}", f"attn_q_b\\.weight={protect}", f"attn_k_b\\.weight={kb}",
        f"attn_v_b\\.weight={protect}", f"attn_kv_a_mqa\\.weight={protect}",
        f"attn_gate\\.weight={protect}", f"attn_output\\.weight={protect}",
        # hyper-connections + DSA indexer + output hc
        f"hc_attn_fn\\.weight={protect}", f"hc_ffn_fn\\.weight={protect}",
        f"indexer\\.attn_k\\.weight={protect}", f"indexer\\.attn_q_b\\.weight={protect}",
        f"indexer\\.proj\\.weight={protect}", f"output_hc_fn\\.weight={protect}",
    ]
    return flags, cq


def emit_bat(a, n_layers, is_moe, names, is_hyv4=False):
    body, protect = _atom(a.body), _atom(a.protect)
    kfree = is_kquant(body) and is_kquant(protect)     # imatrix-free (decoupled MoE) build?
    if is_hyv4:
        flags, cq = recipe_flags_hyv4(n_layers, body, protect)   # THE third canonical recipe
    else:
        flags, cq = recipe_flags(n_layers, is_moe, body, protect)
    base = a.model
    stem = re.sub(r"[-.]f16\.gguf$|\.gguf$", "", base.split("\\")[-1].split("/")[-1], flags=re.I)
    # robustness: pin imatrix-uncovered matmul/expert tensors to q6_K (first => wins), so a
    # MoE build can't hard-fail on a rarely-routed expert the imatrix never saw. NOT needed
    # in the imatrix-free path (K-quants don't consult an imatrix -> nothing to be uncovered).
    pins, ncov = ([], None) if kfree else uncovered_pins(names, a.imatrix)
    pin_cq = (",".join(pins) + ",") if pins else ""
    cqs = pin_cq + ",".join(cq)                     # pins FIRST (custom-q is first-match-wins)
    im_flag = "" if kfree else "--imatrix %IM% "    # the whole point: no imatrix on the K-quant path
    L = ["@echo off", f"set BIN={a.bin}", f"set IM={a.imatrix}", f"set SRC={base}",
         f"set EV={a.eval}", f"set LOG={a.log}",
         f"echo ===AUTOMAP {stem}  layers={n_layers}  moe={is_moe}  imatrix={'no (K-quant)' if kfree else 'yes'}  "
         f"body={body}({QUANT_BPW.get(body,'?')}) protect={protect}({QUANT_BPW.get(protect,'?')})"
         f"=== 1> %LOG% 2>&1", ""]
    def build(name, typ, extra=""):
        return (f"%BIN%\\llama-quantize.exe {im_flag}{extra} %SRC% "
                f"{stem}-{name}.gguf {typ} 1>> %LOG% 2>&1")
    def ppl(name):
        # PPL is offload-invariant (ngl changes speed, not the number) — so a partial
        # offload keeps every bar comparable AND stops a big bar OOMing the card.
        return (f"%BIN%\\llama-perplexity.exe -m {stem}-{name}.gguf -f %EV% -c 2048 "
                f"-ngl {a.ngl} 1>> %LOG% 2>&1")
    pin_extra = f' --custom-q "{pin_cq[:-1]}"' if pins else ""   # uniform bars need the pins too
    if pins:
        L += [f"echo == pinned {len(pins)} imatrix-uncovered tensor(s) to q6_K "
              f"(covered={ncov}) == 1>> %LOG% 2>&1"]
    # --no-eval skips PPL (a plain user BUILD doesn't need the benchmark); the 3-bar
    # comparison (uniform baseline + ceiling) is the BENCHMARK, emitted only when NOT --mix-only.
    def maybe_ppl(name):
        return [] if a.no_eval else [ppl(name)]
    if not a.mix_only:
        # baseline (uniform at the crush tier) — the bar the Mix must beat at ~same size
        L += [f"echo ==uniform {body}== 1>> %LOG% 2>&1",
              build(f"u-{body}", _bar_type(body), pin_extra), *maybe_ppl(f"u-{body}"), ""]
        # ceiling (uniform at the protect tier)
        L += [f"echo ==uniform {protect}== 1>> %LOG% 2>&1",
              build(f"u-{protect}", _bar_type(protect), pin_extra), *maybe_ppl(f"u-{protect}"), ""]
        # optional rival: a strong mixed/uniform low-bit to beat head-to-head (Grok's 4th bar)
        if a.rival:
            rv = _atom(a.rival)
            L += [f"echo ==rival uniform {rv}== 1>> %LOG% 2>&1",
                  build(f"rival-{rv}", _bar_type(rv)), *maybe_ppl(f"rival-{rv}"), ""]
    # PollardMix: base fills with the body atom, custom-q protects the sensitive roles. This is
    # the deliverable — always emitted; with --mix-only it's the ONLY thing built (the fast path).
    mix_extra = " ".join(flags) + f' --custom-q "{cqs}"'
    L += ["echo ==PollardMix (automap)== 1>> %LOG% 2>&1",
          build("mix", _bar_type(body), mix_extra), *maybe_ppl("mix"), ""]
    L += [f"echo AUTOMAP_DONE_EXIT_%ERRORLEVEL% 1>> %LOG% 2>&1"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tensors", required=True, help="llama-quantize --dry-run tensor list")
    ap.add_argument("--model", required=True, help="source F16 gguf path (as seen on the box)")
    ap.add_argument("--imatrix", default="ik.imatrix")
    ap.add_argument("--eval", default="wikitext2_test.txt")
    ap.add_argument("--ngl", type=int, default=99,
                    help="GPU layers for the PPL eval. Lower it for a big build that would OOM "
                         "the card (PPL is offload-invariant, so bars stay comparable).")
    ap.add_argument("--bin", default=r"C:\pollard\ik_llama.cpp\build\bin")
    ap.add_argument("--log", default=r"C:\pollard\bench\automap.log")
    ap.add_argument("--out", default="build_automap.bat")
    ap.add_argument("--body", default=None, help=f"crush atom for the fat body/cold experts {BODY_CHOICES} (stq1_0->iq1_bn)")
    ap.add_argument("--protect", default=None, help=f"protect atom for attn-q/output/ffn_down/edge {PROTECT_CHOICES}")
    ap.add_argument("--no-imatrix", "--kquant", dest="no_imatrix", action="store_true",
                    help="FALLBACK (not a win): imatrix-FREE K-quant MoE mix that builds off the "
                         "F16 with no imatrix. Use ONLY when a covered imatrix is impractical — it "
                         "does NOT beat stock Q2_K (measured). The winning path is the trellis mix "
                         "WITH an imatrix.")
    ap.add_argument("--mix-only", dest="mix_only", action="store_true",
                    help="emit ONLY the PollardMix build — the deliverable model. Skips the "
                         "uniform baseline/ceiling bars (those are the BENCHMARK). This is the "
                         "fast user-build path; without it you get the full 3-bar comparison.")
    ap.add_argument("--no-eval", dest="no_eval", action="store_true",
                    help="skip the PPL eval lines — a plain build doesn't need the benchmark. "
                         "(Reproduce the gold-card numbers with the benchmark path instead.)")
    ap.add_argument("--rival", default="", help="optional 4th bar: a uniform tier to beat head-to-head, e.g. iq2_xxs")
    ap.add_argument("--allow-dense", action="store_true",
                    help="permit a DENSE model (automap is the MoE path; dense uses imatrix "
                         "K-quants). Only for the research 1-bit-mix case (the gold-card).")
    a = ap.parse_args()
    # atom defaults: trellis (imatrix) by default; K-quant (imatrix-free) when --no-imatrix.
    if a.no_imatrix:
        print("(!) --no-imatrix is DEPRECATED: the K-quant mix does NOT beat stock Q2_K "
              "(measured). For a real MoE build make an imatrix and use the trellis mix; for an "
              "imatrix-free build just use a stock K-quant preset (pollard-fit). Continuing anyway.",
              file=sys.stderr)
        a.body = a.body or "q2_k"          # cold experts / attn k,v
        a.protect = a.protect or "q3_k"    # residual writer + attn q/out + edge: one tier up,
        #                                    NOT q4_k (that bloats the mix past the accept gate)
    else:
        a.body = a.body or "iq1_kt"
        a.protect = a.protect or "iq2_kt"
    names, n_layers, is_moe, is_hyv4 = parse_tensors(a.tensors)
    if not n_layers:
        sys.exit("no blk.N tensors found — is this a dry-run tensor list?")
    # GUARDRAIL: automap is the MoE path. A dense model has no experts to allocate — its
    # win is imatrix-guided K-quants, not this. Refuse dense (saves everyone the wrong-tool
    # run) unless --allow-dense (the research 1-bit-mix / gold-card case). HYV4 IS a MoE.
    if not is_moe and not a.allow_dense:
        sys.exit("REFUSED: this is a DENSE model, and automap is the MoE path.\n"
                 "  Dense models -> imatrix-guided K-quants (IQ3_S/IQ4_XS/Q6_K); the measured\n"
                 "  expert-allocation here doesn't apply (no expert redundancy to reallocate).\n"
                 "  Rule: imatrix = dense, automap = MoE. Pass --allow-dense only for the\n"
                 "  research 1-bit-mix case (the gold-card).")
    body, protect = _atom(a.body), _atom(a.protect)
    kfree = is_kquant(body) and is_kquant(protect)
    kind = "HYV4 (MLA+DSA+hyper-conn MoE)" if is_hyv4 else ("MoE" if is_moe else "dense")
    print(f"parsed: {len(names)} tensors, {n_layers} layers, {kind}")
    print(f"atoms: body={body} ({QUANT_BPW.get(body,'?')} bpw)  protect={protect} ({QUANT_BPW.get(protect,'?')} bpw)"
          f"  ->  {'imatrix-FREE (K-quant) build' if kfree else 'imatrix-guided (trellis) build'}")
    flags, cq = recipe_flags_hyv4(n_layers, body, protect) if is_hyv4 else recipe_flags(n_layers, is_moe, body, protect)
    print("Mix policy:" + ("  [HYV4 third canonical recipe — v1, measure on the real model]" if is_hyv4 else ""))
    print("  flags   :", " ".join(flags))
    print("  custom-q:", ",".join(cq))
    if kfree:
        print("  imatrix : NONE — every atom is a K-quant, so the build reads no importance "
              "matrix (no coverage problem, no 6-hour imatrix step, kill-proof).")
    else:
        pins, ncov = uncovered_pins(names, a.imatrix)
        if ncov is None:
            print("  imatrix : could not read coverage (skipping pins — build may fail on "
                  "uncovered experts; rerun the imatrix with more/diverse chunks, or use "
                  "--no-imatrix for a K-quant MoE build that needs no imatrix at all).")
        else:
            print(f"  imatrix : {ncov} tensors covered; PINNING {len(pins)} uncovered "
                  f"matmul/expert tensor(s) to q6_K so the build can't hard-fail."
                  + ("  (!) many uncovered - use --no-imatrix (K-quant) or a fuller/diverse imatrix."
                     if len(pins) > n_layers else ""))
        if not is_moe and pins:
            print("  (dense model with uncovered tensors — unusual; check the imatrix.)")
    open(a.out, "w").write(emit_bat(a, n_layers, is_moe, names, is_hyv4=is_hyv4))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
