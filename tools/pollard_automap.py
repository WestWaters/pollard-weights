#!/usr/bin/env python3
"""pollard-automap -- generate the memory-fit mix recipe for ANY model, automatically.

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
  # first:  llama-quantize --dry-run ... model-f16.gguf x.gguf IQ1_KT > tensors.txt
  # point --bin at YOUR ik_llama.cpp build (or set $POLLARD_IK_BIN):
  pollard-automap --tensors tensors.txt --model model-f16.gguf --imatrix ik.imatrix \
      --out build_mix.bat --bin path/to/ik_llama.cpp/build/bin
"""
import argparse, os, re, subprocess, sys, tempfile


def imatrix_covered(path):
    """The set of tensor names an ik_llama imatrix ACTUALLY covers. ik writes the OLD
    binary format (not GGUF): int32 n_entries, then per entry int32 name_len, name,
    int32 ncall, int32 nval, float[nval]. A MoE routes to only some experts over a short
    calib, so many `*_exps` tensors get NO entry -- and a very-low-bit build hard-fails on
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
# Every matmul that a trellis (iq*_kt) build consults an imatrix for. Includes the MLA
# up-/down-projections (attn_q_a/q_b, attn_k_b, attn_v_b, attn_kv_a_mqa, attn_kv_b) -- ik_llama's
# imatrix STRUCTURALLY skips attn_k_b/v_b/kv_b (its MLA forward path uses a different layout),
# and they can't be copy-covered (their input is the compressed KV latent, shared with nothing),
# so when uncovered they MUST be pinned to a K-quant or the low-bit build hard-fails ("Missing
# importance matrix ... bailing out"). Norms (attn_*_norm) are excluded -- they stay F32.
# A HYBRID (Mamba/SSM) block mixes sequence information with a state-space operator instead of
# attention, and its projections (ssm_in/out, the alpha/beta/x/dt projections) are ordinary
# matmuls the imatrix covers exactly like q/k/v -- verified on Qwen3.8-27B, where all 240 of them
# carry entries. Leaving them out of this pattern means a low-bit build never checks their
# coverage, which is how "Missing importance matrix ... bailing out" arrives at build time.
_NEEDS_IMATRIX = re.compile(
    r"blk\.\d+\.(ffn_(up|down|gate)(_exps|_shexp)?"
    r"|ssm_(in|out|alpha|beta|x|dt)"
    r"|attn_(q|k|v|qkv|gate|output|q_a|q_b|k_b|v_b|kv_b|kv_a_mqa))\.weight$")


def uncovered_pins(all_names, imatrix, fallback="q6_K"):
    """--custom-q rules pinning every imatrix-REQUIRED tensor the imatrix doesn't cover
    to `fallback`, so an aggressive MoE build can't crash on a rarely-routed expert. THIS
    is what makes automap robust on MoE. Empty if the imatrix can't be read (build may
    still fail on uncovered experts -- rerun the imatrix with more/diverse chunks)."""
    cov = imatrix_covered(imatrix)
    if cov is None:
        return [], None
    pins = [rf"{re.escape(nm)}={fallback}" for nm in all_names
            if _NEEDS_IMATRIX.search(nm) and nm not in cov]
    return pins, len(cov)


def ensure_gate_coverage(imatrix_path):
    """Auto-cover the SwiGLU gate side of a fused MoE ffn. ik_llama's imatrix routinely SKIPS
    ffn_gate_exps / ffn_gate_shexp / the dense ffn_gate (observed on Qwen3-30B-A3B, DeepSeek-V2,
    Hy4) -- gate and up share the SAME input, so their importance is identical. Left uncovered,
    the biggest param group (gate) either bloats to a q6 pin or hard-fails the low-bit build.
    Copy every covered `ffn_up*` entry to its `ffn_gate*` name, write a sibling `*.gatefix.imatrix`,
    and return its path so the whole flow (pins + build) uses the covered imatrix with NO manual
    step. Returns (path_to_use, n_copied); the original path + 0 when nothing needed copying or the
    file can't be parsed (caller then proceeds unchanged)."""
    import struct
    try:
        d = open(imatrix_path, "rb").read()
        n = struct.unpack_from("<i", d, 0)[0]; pos = 4
        entries = []
        for _ in range(n):
            ln = struct.unpack_from("<i", d, pos)[0]; pos += 4
            if ln <= 0 or ln > 512:
                return imatrix_path, 0
            name = d[pos:pos + ln]; pos += ln
            ncall = struct.unpack_from("<i", d, pos)[0]; pos += 4
            nval = struct.unpack_from("<i", d, pos)[0]; pos += 4
            floats = d[pos:pos + 4 * nval]; pos += 4 * nval
            entries.append((name, ncall, nval, floats))
    except Exception:
        return imatrix_path, 0
    have = {e[0] for e in entries}
    added = 0
    for name, ncall, nval, floats in list(entries):
        if b"ffn_up" in name:
            g = name.replace(b"ffn_up", b"ffn_gate")
            if g not in have:
                entries.append((g, ncall, nval, floats)); have.add(g); added += 1
    if not added:
        return imatrix_path, 0
    out = os.path.splitext(imatrix_path)[0] + ".gatefix.imatrix"
    try:
        buf = struct.pack("<i", len(entries))
        for name, ncall, nval, floats in entries:
            buf += struct.pack("<i", len(name)) + name + struct.pack("<i", ncall) + struct.pack("<i", nval) + floats
        open(out, "wb").write(buf)
    except Exception:
        return imatrix_path, 0          # couldn't write -> proceed with the original (pins will cover)
    return out, added


def tensor_list(model, bin_dir=None, imatrix=None, out=None):
    """Produce the tensor listing ourselves instead of demanding one.

    --tensors was a required argument, which meant every caller had to know to run
    `llama-quantize --dry-run` first and where to put the output. That is a step Pollard can do,
    so it does it.

    Choosing the listing type is the other half. A dry-run at IQ1_S is REFUSED outright when no
    importance matrix is present -- llama-quantize prints the refusal instead of the tensor names,
    and the caller sees an empty file rather than a reason. We only want names, so: use the
    imatrix when one was given, and otherwise ask for a type that never needs one. Detect and
    choose, rather than stop.
    """
    exe = os.path.join(bin_dir, "llama-quantize.exe" if os.name == "nt" else "llama-quantize") \
        if bin_dir else ("llama-quantize.exe" if os.name == "nt" else "llama-quantize")
    out = out or os.path.join(tempfile.gettempdir(), "pollard_tensors.txt")
    sink = os.path.join(tempfile.gettempdir(), "pollard_dryrun.gguf")

    # imatrix-free first: Q8_0 lists the same tensors and is never refused for want of one.
    attempts = [["--dry-run", model, sink, "Q8_0"]]
    if imatrix:
        attempts.insert(0, ["--dry-run", "--imatrix", imatrix, model, sink, "IQ1_S"])

    last = ""
    for args in attempts:
        try:
            r = subprocess.run([exe] + args, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=900)
        except (OSError, subprocess.TimeoutExpired) as e:
            last = str(e)
            continue
        blob = (r.stdout or "") + (r.stderr or "")
        if re.search(r"blk\.\d+\.", blob):
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(blob)
            return out
        last = blob.strip().splitlines()[-1] if blob.strip() else "no output"
    raise SystemExit(
        f"ERROR: could not list the tensors of {model}.\n"
        f"  Tried a dry-run with and without an importance matrix; the last thing it said was:\n"
        f"    {last[:300]}\n"
        f"  Pass --tensors with a `llama-quantize --dry-run` listing, or --bin with the directory "
        f"holding llama-quantize.")


def parse_tensors(path):
    """Return (names, n_layers, is_moe, arch). Detect the architecture CLASS generically and
    route by it -- dense -> dense recipe; ANY MoE -> THE MoE recipe (which covers both the
    Qwen-style tensor names AND the MLA / hyper-connection / DSA-indexer variants that Deepseek,
    Hy4, etc. use). `arch` is a human descriptor for the label from the features present -- never
    a per-model special case. Accepts dry-run lines or bare tensor names."""
    names = []
    for ln in open(path, encoding="utf-8", errors="ignore"):
        m = re.search(r"(blk\.\d+\.[\w.]+|token_embd\.weight|output[\w.]*\.weight|output_norm\.weight)", ln)
        if m:
            names.append(m.group(1))
    names = sorted(set(names))
    layers = [int(m.group(1)) for n in names for m in [re.match(r"blk\.(\d+)\.", n)] if m]
    n_layers = (max(layers) + 1) if layers else 0
    is_moe = any("exps" in n or "ffn_gate_inp" in n for n in names)
    feats = []                                             # generic architecture features
    if any("attn_k_b" in n or "kv_a_mqa" in n or "attn_q_b" in n for n in names): feats.append("MLA")
    if any(".hc_" in n for n in names): feats.append("hyper-conn")
    if any(".indexer." in n for n in names): feats.append("DSA-indexer")
    # HYBRID: most blocks mix with a state-space operator, only a minority carry real attention.
    # Worth naming in the arch line -- on Qwen3.8-27B it is 48 SSM blocks to 17 attention ones,
    # and a recipe written for "dense" silently crushes the mixing path of the other 48.
    n_ssm = len({m.group(1) for n in names for m in [re.match(r"blk\.(\d+)\.ssm_", n)] if m})
    if n_ssm:
        n_attn = len({m.group(1) for n in names
                      for m in [re.match(r"blk\.(\d+)\.attn_(q|qkv)\.weight", n)] if m})
        feats.append(f"hybrid-SSM {n_ssm}ssm/{max(n_attn - n_ssm, 0)}attn")
    arch = ("MoE" if is_moe else "dense") + (f" +{'+'.join(feats)}" if feats else "")
    return names, n_layers, is_moe, arch


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
    ('Invalid quantization type') -- the tensor-type table is case-sensitive on the K."""
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



def fragile_rules(path, protect, floor=3.0, cap=4):
    """Turn a pollard-fragile scan into protect rules, so the fragile tensors are HANDLED.

    Reporting fragility and leaving the build to crush it anyway is the wrong half of the job. A
    heavy-tailed tensor is one an absmax scale cannot represent, and that is knowable from the
    weights before anything is built -- so the allocator should act on it rather than print a
    warning nobody reads.

    This is why it matters in practice: on gemma-4-12B-it the scan puts token_embd at kurtosis 17.9
    with a crest factor of 378, five times worse than anything else in the model. That is the tensor
    whose lost resolution made the Gemma4 flagship loop on <|channel>thought and fail the coherence
    gate TWICE before anyone found it by building. Consuming the scan turns two dead builds into a
    rule emitted before the first one.

    Reads a JSON file, the same way --imatrix reads a file: data between tools, not tools threaded
    through each other. `floor` keeps ordinary tensors out (a mildly heavy tail is normal) and `cap`
    stops a model whose every kind is peaky from protecting itself into no compression at all.
    """
    import json
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print(f"  (fragility scan unreadable: {e}) -- continuing without it")
        return [], [], {}
    kinds = [k for k in (d.get("kinds") or []) if k.get("kurtosis", 0) >= floor]
    kinds.sort(key=lambda k: -k["kurtosis"])
    rules, notes, lift = [], [], {}
    for k in kinds[:cap]:
        base = re.sub(r"\.weight$", "", k["kind"])
        tag = f"{base} (kurtosis {k['kurtosis']:.1f}, crest {k['crest']:.0f})"
        # token_embd and output are NOT custom-q territory: they have dedicated flags, and the
        # protect atom is a LOW-bit trellis type -- emitting `token_embd=iq2_kt` would push the
        # most fragile tensor in the model DOWN to 2.125 bpw, the exact opposite of protecting it.
        # Raise their own flag instead, which is precisely the fix that rescued Gemma4's flagship.
        if base in ("token_embd", "output"):
            lift[base] = "Q8_0" if k["kurtosis"] >= 10 else "Q6_K"
            notes.append(f"{tag} -> {lift[base]} via its own flag")
        else:
            rules.append(f"{re.escape(base)}={protect}")
            notes.append(tag)
    return rules, notes, lift


def recipe_flags(n_layers, is_moe, body="iq1_kt", protect="iq2_kt"):
    """Emit the Mix as (base_type, custom-q rules). base_type = the crush atom (fills
    everything not matched); every protected role is named explicitly via custom-q
    (lowercase = the reliable path -- role-flags silently fell back for some atoms).
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
        # (2b) HYV4 (MLA + hyper-connection + DSA-indexer MoE) runs through THIS recipe, not a
        # separate one. Its MLA attn (attn_k_b/v_b/q_a/q_b/kv_a_mqa) is already caught by the
        # attn_q/k/v rules below (substring); these add the tensors those rules MISS --
        # attn_gate, the hyper-connections, indexer.proj, output hc, and the dense block-0 FFN --
        # protected at the same tier. All no-ops on Qwen3-MoE (those tensors don't exist there).
        cq += [f"attn_gate={protect}", f"hc_attn_fn={protect}", f"hc_ffn_fn={protect}",
               f"indexer\\.proj={protect}", f"output_hc_fn={protect}",
               f"ffn_gate\\.weight={protect}", f"ffn_up\\.weight={protect}"]
    # (3) general roles. Grok's policy: PROTECT attn v/o (they carry the distribution); q/k
    # are less critical. On DENSE the shipped 7B/14B recipe crushed k,v and still won, so keep
    # it. On MoE, crushing attn_v was measured to LOSE KLD vs uniform IQ1 (30B: mix 0.371 >
    # uniform 0.360) -- protect attn_v/k there (attention is a small fraction of a MoE anyway).
    attn_kv = protect if (kfree or is_moe) else body
    cq += [f"attn_k={attn_kv}", f"attn_v={attn_kv}",
           f"attn_q={protect}", f"attn_output={protect}", f"ffn_down={protect}"]
    # (4) HYBRID (Mamba/SSM) blocks mix the sequence with a state-space operator instead of
    # attention, so the rules above -- which name attention tensors -- reach none of them and the
    # whole mixing path falls through to the body crush atom. On Qwen3.8-27B that is 48 of 65
    # blocks: ssm_out alone is ~1.5B parameters, the mixer's OUTPUT projection, crushed to ~1 bit.
    # Apply the same policy attention gets: protect the writer (ssm_out, the analogue of
    # attn_output) and the gate, and the alpha/beta projections, which are tiny (48 x n_embd) and
    # cost nothing to keep. The fused attn_qkv is already caught by the attn_q rule above.
    # No-ops on a model with no SSM blocks.
    cq += [f"ssm_out={protect}", f"attn_gate={protect}",
           f"ssm_alpha={protect}", f"ssm_beta={protect}", f"ssm_in={protect}"]
    return flags, cq



def emit_bat(a, n_layers, is_moe, names):
    body, protect = _atom(a.body), _atom(a.protect)
    kfree = is_kquant(body) and is_kquant(protect)     # imatrix-free (decoupled MoE) build?
    flags, cq = recipe_flags(n_layers, is_moe, body, protect)   # dense/MoE (MoE covers MLA/hc/DSA)
    base = a.model
    stem = re.sub(r"[-.]f16\.gguf$|\.gguf$", "", base.split("\\")[-1].split("/")[-1], flags=re.I)
    # robustness: pin imatrix-uncovered matmul/expert tensors to q6_K (first => wins), so a
    # MoE build can't hard-fail on a rarely-routed expert the imatrix never saw. NOT needed
    # in the imatrix-free path (K-quants don't consult an imatrix -> nothing to be uncovered).
    pins, ncov = ([], None) if kfree else uncovered_pins(names, a.imatrix)
    pin_cq = (",".join(pins) + ",") if pins else ""
    # Fragile kinds are protected ahead of the general role rules: custom-q is FIRST-MATCH-WINS, so
    # a rule placed later would lose to the recipe's own entry for the same tensor.
    frag_rules, frag_notes, frag_lift = (fragile_rules(a.fragile, protect)
                                         if getattr(a, "fragile", None) else ([], [], {}))
    # a fragile embedding/output raises ITS OWN flag rather than taking a custom-q rule
    for _t, _ty in frag_lift.items():
        _flag = "--token-embedding-type" if _t == "token_embd" else "--output-tensor-type"
        flags = [f for f in flags if not f.startswith(_flag)] + [f"{_flag} {_ty}"]
    frag_cq = (",".join(frag_rules) + ",") if frag_rules else ""
    cqs = pin_cq + frag_cq + ",".join(cq)           # pins FIRST (custom-q is first-match-wins)
    if frag_notes:
        print("  fragile     : auto-protected -> " + ", ".join(frag_notes))
    im_flag = "" if kfree else "--imatrix %IM% "    # the whole point: no imatrix on the K-quant path
    # The eval corpus has to suit the MODEL, or the PPL lines describe the mismatch rather than the
    # build. Ask what this model is instead of defaulting everyone to raw Wikipedia.
    ev = a.eval
    if not ev:
        ev, why = "wikitext2_test.txt", ""
        try:
            from pollard_modelkind import classify, describe
            k = classify(base)
            if k["eval"] != "raw-text":
                ev = "pollard_eval_heldout.txt"
                why = (f"   # {describe(k)}: raw text would score the mismatch. Build this with\n"
                       f"   #   pollard-calib --out train.txt --held-out {ev}")
        except Exception:
            pass
    # ---- the plan, as ARGV rather than shell text ----------------------------------------------
    # It used to be assembled as Windows batch -- "@echo off", %BIN%\\llama-quantize.exe,
    # backslashes -- which meant automap on a Mac or a Linux box wrote a file nothing could run.
    # Building argv instead makes the plan portable, and makes it RUNNABLE: the same list renders
    # to a .bat or a .sh for anyone who wants the script, and executes directly for everyone who
    # wanted the model.
    exe = ".exe" if sys.platform == "win32" else ""
    quantize = os.path.join(a.bin, "llama-quantize" + exe)
    perplexity = os.path.join(a.bin, "llama-perplexity" + exe)
    steps: list = []          # (label, argv)

    header = (f"===AUTOMAP {stem}  layers={n_layers}  moe={is_moe}  "
              f"imatrix={'no (K-quant)' if kfree else 'yes'}  "
              f"body={body}({QUANT_BPW.get(body,'?')}) protect={protect}({QUANT_BPW.get(protect,'?')})===")

    def build(name, typ, extra=()):
        out = f"{stem}-{name}.gguf"
        argv = [quantize]
        if not kfree:
            argv += ["--imatrix", a.imatrix]
        argv += list(extra) + [base, out, typ]
        return out, argv

    def ppl(out):
        # PPL is offload-invariant (ngl changes speed, not the number) -- so a partial offload
        # keeps every bar comparable AND stops a big bar OOMing the card.
        return [perplexity, "-m", out, "-f", ev, "-c", "2048", "-ngl", str(a.ngl)]

    pin_extra = ["--custom-q", pin_cq[:-1]] if pins else []
    if pins:
        print(f"  pinned      : {len(pins)} imatrix-uncovered tensor(s) -> q6_K (covered={ncov})")

    def maybe_ppl(out):
        return [] if a.no_eval else [(f"ppl {os.path.basename(out)}", ppl(out))]

    if not a.mix_only:
        out, argv = build(f"u-{body}", _bar_type(body), pin_extra)
        steps += [(f"uniform {body}", argv), *maybe_ppl(out)]
        out, argv = build(f"u-{protect}", _bar_type(protect), pin_extra)
        steps += [(f"uniform {protect}", argv), *maybe_ppl(out)]
        if a.rival:
            rv = _atom(a.rival)
            out, argv = build(f"rival-{rv}", _bar_type(rv))
            steps += [(f"rival uniform {rv}", argv), *maybe_ppl(out)]

    # PollardMix: base fills with the body atom, custom-q protects the sensitive roles. This is
    # the deliverable -- always emitted; with --mix-only it's the ONLY thing built.
    mix_extra = []
    for f in flags:
        mix_extra += f.split(" ", 1)
    mix_extra += ["--custom-q", cqs]
    mix_out, mix_argv = build("mix", _bar_type(body), mix_extra)
    steps += [("PollardMix (automap)", mix_argv), *maybe_ppl(mix_out)]

    # Auto coherence gate on the finished mix -- so a build also tells you whether it is USABLE.
    if getattr(a, "gate", True):
        gate_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pollard_bench.py")
        steps.append(("coherence gate", [sys.executable, gate_py, "--gguf", mix_out,
                                         "--coherence", "--ngl", str(a.ngl),
                                         "--llama-cli", os.path.join(a.bin, "llama-cli" + exe)]))
    return {"header": header, "steps": steps, "mix": mix_out, "log": a.log}


def render_script(plan, for_windows=None):
    """The plan as a script, for anyone who wants to read or edit it before running."""
    win = sys.platform == "win32" if for_windows is None else for_windows
    q = (lambda x: f'"{x}"' if " " in str(x) else str(x))
    log = plan["log"]
    if win:
        out = ["@echo off", f'echo {plan["header"]} 1> {q(log)} 2>&1']
        for label, argv in plan["steps"]:
            out += [f'echo == {label} == 1>> {q(log)} 2>&1',
                    " ".join(q(x) for x in argv) + f" 1>> {q(log)} 2>&1"]
        out.append(f"echo AUTOMAP_DONE_EXIT_%ERRORLEVEL% 1>> {q(log)} 2>&1")
    else:
        out = ["#!/bin/sh", "set -e", f'echo {q(plan["header"])} > {q(log)} 2>&1']
        for label, argv in plan["steps"]:
            out += [f'echo "== {label} ==" >> {q(log)} 2>&1',
                    " ".join(q(x) for x in argv) + f" >> {q(log)} 2>&1"]
        out.append(f'echo "AUTOMAP_DONE_EXIT_$?" >> {q(log)} 2>&1')
    return "\n".join(out)


def run_plan(plan):
    """Actually build it. Returns the exit code of the first step that failed, or 0.

    This is what automap is for. Emitting a script and stopping left the user to re-run the
    thing they had already asked for -- and on any machine that is not Windows, to re-run it by
    hand because the script was batch.
    """
    print(f"  {plan['header']}")
    with open(plan["log"], "w", encoding="utf-8", errors="replace") as log:
        log.write(plan["header"] + "\n")
        for label, argv in plan["steps"]:
            print(f"  -> {label}", flush=True)
            log.write(f"\n== {label} ==\n")
            log.flush()
            r = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                msg = f"  FAILED at '{label}' (exit {r.returncode}) -- see {plan['log']}"
                print(msg)
                log.write(msg + "\n")
                return r.returncode
    print(f"  built {plan['mix']}")
    return 0


def find_ik_bin(explicit=None):
    """Where ik_llama.cpp's binaries are, without making the user say so.

    The default used to be the RELATIVE path ik_llama.cpp\\build\\bin, which resolves against
    whatever directory the run happens to start in -- so the same install worked from one folder
    and failed from another, and the backslash made it Windows-only. A tool that cannot find a
    binary sitting in an obvious place should look, not stop.
    """
    exe = ".exe" if sys.platform == "win32" else ""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    home = os.environ.get("POLLARD_HOME") or os.path.expanduser("~/pollard")
    roots = [explicit, os.environ.get("POLLARD_IK_BIN")]
    for base in (os.getcwd(), home, here, os.path.dirname(home), os.path.expanduser("~")):
        for sub in ("ik_llama.cpp/build/bin", "ik_llama/build/bin", "bin",
                    "runtime/ik_llama.cpp/build/bin"):
            roots.append(os.path.join(base, *sub.split("/")))
    for r in roots:
        if r and os.path.isfile(os.path.join(r, "llama-quantize" + exe)):
            return r
    # PATH, last: a distro build may be older than the trellis atoms this needs
    from shutil import which
    got = which("llama-quantize")
    return os.path.dirname(got) if got else (explicit or "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tensors", help="llama-quantize --dry-run tensor list. Optional: without "
                                      "it Pollard runs the dry-run itself, picking a listing "
                                      "type that does not need an importance matrix if none "
                                      "was supplied.")
    ap.add_argument("--model", required=True, help="source F16 gguf path (as seen on the box)")
    ap.add_argument("--imatrix", default="ik.imatrix")
    ap.add_argument("--eval", default="",
                    help="held-out eval corpus for the PPL lines. Left empty, Pollard picks one "
                         "that suits the model: raw text for a base model, in-domain (Calib 3.0 "
                         "held-out) for an instruct/reasoning model, because a model tuned away "
                         "from raw-text modelling scores its own mismatch on WikiText -- "
                         "gemma-4-12B-it reads ~664 there where a plain 7B reads 5.4.")
    ap.add_argument("--ngl", type=int, default=99,
                    help="GPU layers for the PPL eval. Lower it for a big build that would OOM "
                         "the card (PPL is offload-invariant, so bars stay comparable).")
    ap.add_argument("--bin", default=None,
                    help="dir holding ik_llama.cpp binaries (llama-quantize/-perplexity/-cli). "
                         "Found automatically when it is anywhere obvious; override with "
                         "$POLLARD_IK_BIN or point it at YOUR build")
    ap.add_argument("--log", default=os.environ.get("POLLARD_AUTOMAP_LOG", "automap.log"),
                    help="build/eval log path (default: ./automap.log; or $POLLARD_AUTOMAP_LOG)")
    ap.add_argument("--out", default="",
                    help="also write the plan as a script here (.bat on Windows, .sh elsewhere). "
                         "Optional: automap BUILDS by default, so a script is only needed when "
                         "you want to read or edit the plan first.")
    ap.add_argument("--plan-only", action="store_true",
                    help="work out the mix and stop without building it. Pair with --out to get "
                         "a script you can inspect.")
    ap.add_argument("--fragile", help="a pollard-fragile --out scan.json. The heaviest-tailed\n                        tensor kinds are PROTECTED automatically instead of merely reported.")
    ap.add_argument("--body", default=None, help=f"crush atom for the fat body/cold experts {BODY_CHOICES} (stq1_0->iq1_bn)")
    ap.add_argument("--protect", default=None, help=f"protect atom for attn-q/output/ffn_down/edge {PROTECT_CHOICES}")
    ap.add_argument("--no-imatrix", "--kquant", dest="no_imatrix", action="store_true",
                    help="FALLBACK (not a win): imatrix-FREE K-quant MoE mix that builds off the "
                         "F16 with no imatrix. Use ONLY when a covered imatrix is impractical -- it "
                         "does NOT beat stock Q2_K (measured). The winning path is the trellis mix "
                         "WITH an imatrix.")
    ap.add_argument("--mix-only", dest="mix_only", action="store_true",
                    help="emit ONLY the PollardMix build -- the deliverable model. Skips the "
                         "uniform baseline/ceiling bars (those are the BENCHMARK). This is the "
                         "fast user-build path; without it you get the full 3-bar comparison.")
    ap.add_argument("--no-eval", dest="no_eval", action="store_true",
                    help="skip the PPL eval lines -- a plain build doesn't need the benchmark. "
                         "(Reproduce the gold-card numbers with the benchmark path instead.)")
    ap.add_argument("--rival", default="", help="optional 4th bar: a uniform tier to beat head-to-head, e.g. iq2_xxs")
    ap.add_argument("--allow-dense", action="store_true",
                    help="accepted and ignored -- dense is no longer refused. (was: MoE path; dense uses imatrix "
                         "K-quants). Only for the research 1-bit-mix case (the gold-card).")
    ap.add_argument("--no-gate", dest="gate", action="store_false",
                    help="skip the auto coherence gate appended after the mix build. By default the "
                         "emitted build runs a quick loop-check + sampling sweep on the finished mix "
                         "(PASS+recommended sampling, or BELOW-FLOOR+bump-a-tier) so a one-shot build "
                         "tells you if it's usable -- no manual pollard-bench --coherence needed.")
    ap.set_defaults(gate=True)
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
    # --tensors is optional now: if it was not given, make the listing rather than refusing.
    a.bin = find_ik_bin(a.bin)
    tensors = a.tensors or tensor_list(a.model, bin_dir=a.bin, imatrix=a.imatrix)
    names, n_layers, is_moe, arch = parse_tensors(tensors)
    if not n_layers:
        sys.exit("no blk.N tensors found -- is this a dry-run tensor list?")
    # Dense runs. automap has carried a real dense recipe all along (crush ffn_gate/up, protect
    # attn+down+edges) -- refusing dense and then applying that recipe the moment someone passed
    # --allow-dense was the tool arguing with itself. The shipped dense flagships all came out of
    # the forced path, so the forced path IS the path.
    # Transparency: an aliased atom (e.g. Hy4's STQ1_0) is an APPROXIMATION, not the real format --
    # say so, so nobody thinks they built a true 1.31-bit STQ1_0 when they built 1.62-bit iq1_bn.
    for label, raw in [("--body", a.body), ("--protect", a.protect), ("--rival", a.rival)]:
        if raw and raw.lower() in ALIASES:
            real = ALIASES[raw.lower()]
            print(f"(!) {label} {raw}: NOT emittable by this llama.cpp -> building {real} "
                  f"({QUANT_BPW.get(real, '?')} bpw), the nearest ternary. True {raw} (Hy4 MIX-STQ1_0's "
                  f"~1.31-bit forced-zero 3:4 pack) needs AngelSlim/upstream STQ kernels; this is an "
                  f"approximation at a higher bpw.", file=sys.stderr)
    body, protect = _atom(a.body), _atom(a.protect)
    kfree = is_kquant(body) and is_kquant(protect)
    kind = arch
    print(f"parsed: {len(names)} tensors, {n_layers} layers, {kind}")
    print(f"atoms: body={body} ({QUANT_BPW.get(body,'?')} bpw)  protect={protect} ({QUANT_BPW.get(protect,'?')} bpw)"
          f"  ->  {'imatrix-FREE (K-quant) build' if kfree else 'imatrix-guided (trellis) build'}")
    flags, cq = recipe_flags(n_layers, is_moe, body, protect)   # HY4 runs through THE MoE recipe
    print(f"Mix policy: [{arch} -> {'MoE' if is_moe else 'dense'} recipe]")
    print("  flags   :", " ".join(flags))
    print("  custom-q:", ",".join(cq))
    if kfree:
        print("  imatrix : NONE -- every atom is a K-quant, so the build reads no importance "
              "matrix (no coverage problem, no 6-hour imatrix step, kill-proof).")
    else:
        # AUTO gate-copy: cover the SwiGLU gate side (ik's imatrix skips it) BEFORE pinning, so the
        # experts crush cleanly instead of bloating to q6 -- no manual imatrix_fix_gate step. Uses
        # the fixed imatrix for both the pins below AND the emitted build (a.imatrix is what set IM=).
        if is_moe:
            fixed, ncopied = ensure_gate_coverage(a.imatrix)
            if ncopied:
                print(f"  gate-copy: covered {ncopied} SwiGLU gate tensor(s) from up (wrote "
                      f"{fixed.split(chr(92))[-1].split('/')[-1]}); build uses the covered imatrix.")
                a.imatrix = fixed
        pins, ncov = uncovered_pins(names, a.imatrix)
        if ncov is None:
            print("  imatrix : could not read coverage (skipping pins -- build may fail on "
                  "uncovered experts; rerun the imatrix with more/diverse chunks, or use "
                  "--no-imatrix for a K-quant MoE build that needs no imatrix at all).")
        else:
            print(f"  imatrix : {ncov} tensors covered; PINNING {len(pins)} uncovered "
                  f"matmul/expert tensor(s) to q6_K so the build can't hard-fail."
                  + ("  (!) many uncovered - use --no-imatrix (K-quant) or a fuller/diverse imatrix."
                     if len(pins) > n_layers else ""))
        if not is_moe and pins:
            print("  (dense model with uncovered tensors -- unusual; check the imatrix.)")
    plan = emit_bat(a, n_layers, is_moe, names)

    # Write the script whenever one was asked for, so the plan stays readable and editable.
    if a.out:
        open(a.out, "w", encoding="utf-8").write(render_script(plan))
        print(f"wrote {a.out}")

    if a.plan_only:
        print("  --plan-only: nothing was built. Run the script above, or drop --plan-only.")
        return
    raise SystemExit(run_plan(plan))


if __name__ == "__main__":
    main()
