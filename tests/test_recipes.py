#!/usr/bin/env python3
"""Pollard regression suite — assert every canonical recipe, guard, and encoder rule, so a
change to one path can't silently break another (the exact class of bug that cost a weekend:
the MoE attn_v crush, the q3_k casing, the dense guard). Runnable two ways:

    python tests/test_recipes.py      # plain asserts, prints PASS/FAIL, exits non-zero on fail
    pytest tests/test_recipes.py      # same functions as test_*

Add a case whenever a recipe/guard changes — never fewer rows than the tools have behaviors.
"""
import os, re, subprocess, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_automap as A


# ---- helpers ---------------------------------------------------------------------------------
def _tensorfile(names):
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.write("\n".join(f"[{i}] {n} - type f16" for i, n in enumerate(names)))
    f.close()
    return f.name


def _dense(nl=4):
    L = ["token_embd.weight", "output.weight", "output_norm.weight"]
    for i in range(nl):
        for t in ["ffn_gate", "ffn_up", "ffn_down", "attn_q", "attn_k", "attn_v",
                  "attn_output", "attn_norm", "ffn_norm"]:
            L.append(f"blk.{i}.{t}.weight")
    return L


def _moe(nl=4):
    L = ["token_embd.weight", "output.weight", "output_norm.weight"]
    for i in range(nl):
        for t in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_inp",
                  "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
                  "attn_q", "attn_k", "attn_v", "attn_output", "attn_norm"]:
            L.append(f"blk.{i}.{t}.weight")
    return L


def _hyv4(nl=4):
    L = ["token_embd.weight", "output.weight", "output_hc_fn.weight",
         "output_hc_base.weight", "output_hc_scale.weight", "output_norm.weight"]
    kinds = ["attn_gate", "attn_k_b", "attn_kv_a_mqa", "attn_kv_a_norm", "attn_norm",
             "attn_output", "attn_q_a", "attn_q_a_norm", "attn_q_b", "attn_sinks", "attn_v_b",
             "exp_probs_b", "ffn_down", "ffn_down_exps", "ffn_down_shexp", "ffn_gate",
             "ffn_gate_exps", "ffn_gate_inp", "ffn_gate_shexp", "ffn_norm", "ffn_up",
             "ffn_up_exps", "ffn_up_shexp", "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
             "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale", "indexer.attn_k", "indexer.attn_q_b",
             "indexer.k_norm", "indexer.proj"]
    for i in range(nl):
        for k in kinds:
            L.append(f"blk.{i}.{k}.weight")
    return L


def _cq_map(cq):
    """{tensor_rule: type} from a recipe_flags custom-q list."""
    return dict(r.rsplit("=", 1) for r in cq)


def _apply(cq, base, name):
    """Simulate --custom-q (first-match-wins, ik uses re.search) on one tensor name."""
    for rule in cq:
        pat, typ = rule.rsplit("=", 1)
        if re.search(pat, name):
            return typ
    return base


# ---- detection -------------------------------------------------------------------------------
def test_detection():
    # generic architecture-class detection (not per-model): dense / MoE / MoE +features
    for names, exp_moe, feats in [(_dense(), False, []), (_moe(), True, []),
                                  (_hyv4(), True, ["MLA", "hyper-conn", "DSA-indexer"])]:
        _, nl, is_moe, arch = A.parse_tensors(_tensorfile(names))
        assert nl == 4, f"layers {nl}"
        assert is_moe == exp_moe, f"is_moe {is_moe} != {exp_moe}"
        assert arch.startswith("MoE" if exp_moe else "dense"), f"arch '{arch}'"
        for f in feats:
            assert f in arch, f"feature {f} not detected in arch '{arch}'"


# ---- encoder rules (the casing bug) ----------------------------------------------------------
def test_stq_alias():
    # STQ1_0 (Hy4 MIX-STQ1_0's ~1.31-bit format) is NOT in ik_llama -> aliased to the nearest
    # emittable ternary (iq1_bn, 1.62). Documents the approximation (main() warns at runtime).
    assert A._atom("stq1_0") == "iq1_bn" and A._atom("stq2_0") == "iq2_bn"
    assert A._atom("iq1_kt") == "iq1_kt"                              # a real atom passes through


def test_cq_casing():
    assert A._cq("q3_k") == "q3_K" and A._cq("q2_k") == "q2_K"        # K-quants: capital K
    assert A._cq("iq1_kt") == "iq1_kt" and A._cq("iq2_kt") == "iq2_kt"  # trellis: lowercase
    assert A._bar_type("q2_k") == "Q2_K" and A._bar_type("q4_k") == "Q4_K_M"  # bar preset
    assert A._bar_type("iq1_kt") == "IQ1_KT"


# ---- MLA up-projections need imatrix (regression: build hard-fails if unpinned) --------------
def test_mla_tensors_need_imatrix():
    # ik_llama's imatrix structurally skips attn_k_b/v_b/kv_b; when uncovered they MUST be
    # flagged (so uncovered_pins pins them to q6_K) or the trellis build bails out. This is the
    # exact bug that killed the DeepSeek-V2-Lite board (blk.0.attn_v_b, "Missing importance
    # matrix in a very low-bit quantization").
    for nm in ["blk.0.attn_k_b.weight", "blk.5.attn_v_b.weight", "blk.9.attn_kv_b.weight",
               "blk.3.attn_kv_a_mqa.weight", "blk.7.attn_q_a.weight", "blk.7.attn_q_b.weight",
               "blk.1.ffn_gate_exps.weight", "blk.1.attn_output.weight"]:
        assert A._NEEDS_IMATRIX.search(nm), f"{nm} must be flagged as imatrix-required"
    # norms and 1D tensors stay F32 -> must NOT be flagged (would pin harmlessly but noisily)
    for nm in ["blk.0.attn_norm.weight", "blk.0.attn_kv_a_norm.weight", "blk.0.attn_q_a_norm.weight",
               "blk.0.ffn_norm.weight"]:
        assert not A._NEEDS_IMATRIX.search(nm), f"{nm} is a norm, must NOT be flagged"


# ---- coherence-gate loop detector (pure heuristic; the real mix failures vs coherent Q8) ------
def test_loop_detector():
    import pollard_bench as B
    loops = [
        "as big as the " * 15,                                              # phrase loop (the 1-bit mix)
        "Sg" * 60,                                                          # char loop ('SgSgSg...')
        "The largest planet in our solar system is the largest planet in our solar system. " * 6,
        "Planet of France and Planet of France " * 8,
    ]
    for t in loops:
        is_loop, m, why = B.detect_loop(t)
        assert is_loop, f"should flag loop: {t[:40]!r} -> {why}"
    coherent = [
        "The largest planet in our solar system is Jupiter. There are eight planets: Mercury, "
        "Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune, each orbiting the sun.",
        "def fib(n):\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n"
        "        return fib(n-1) + fib(n-2)\n\nfor i in range(10):\n    print(fib(i))",
        "Photosynthesis is how green plants convert sunlight, water, and carbon dioxide into "
        "glucose and oxygen, using chlorophyll in their leaves to capture the light energy.",
    ]
    for t in coherent:
        is_loop, m, why = B.detect_loop(t)
        assert not is_loop, f"should NOT flag coherent: {t[:40]!r} -> {why} (metric {m})"
    # too-short output is undecided, not a loop
    assert not B.detect_loop("Jupiter.")[0]


# ---- auto gate-copy wired into automap (no manual imatrix_fix_gate step) ----------------------
def test_auto_gate_copy():
    import struct, tempfile
    def entry(nm, vals):
        return (struct.pack("<i", len(nm)) + nm + struct.pack("<i", 7)
                + struct.pack("<i", len(vals)) + b"".join(struct.pack("<f", v) for v in vals))
    ents = [entry(b"blk.0.ffn_up_exps.weight", [1.0, 2.0, 3.0]),      # up covered, gate MISSING
            entry(b"blk.0.ffn_down_exps.weight", [4.0, 5.0])]
    f = tempfile.NamedTemporaryFile(suffix=".imatrix", delete=False)
    f.write(struct.pack("<i", len(ents)) + b"".join(ents)); f.close()
    fixed, n = A.ensure_gate_coverage(f.name)
    assert n == 1 and fixed.endswith(".gatefix.imatrix"), f"expected 1 gate copied, got {n} -> {fixed}"
    cov = A.imatrix_covered(fixed)
    assert "blk.0.ffn_gate_exps.weight" in cov, "gate must now be covered (copied from up)"
    # idempotent: re-running on the fixed file copies nothing more
    _, n2 = A.ensure_gate_coverage(fixed)
    assert n2 == 0, f"gate already covered -> should copy 0, got {n2}"


# ---- dense recipe (shipped, MUST NOT change: crush attn k/v, protect q/out/down) -------------
def test_dense_recipe():
    _, cq = A.recipe_flags(4, is_moe=False, body="iq1_kt", protect="iq2_kt")
    m = _cq_map(cq)
    assert m["attn_k"] == "iq1_kt" and m["attn_v"] == "iq1_kt"     # dense crushes k,v (won 7B/14B)
    assert m["attn_q"] == "iq2_kt" and m["attn_output"] == "iq2_kt" and m["ffn_down"] == "iq2_kt"


# ---- MoE recipe (the attn_v fix — REGRESSION GUARD for the KLD-losing bug) -------------------
def test_moe_recipe_protects_attn_v():
    _, cq = A.recipe_flags(4, is_moe=True, body="iq1_kt", protect="iq2_kt")
    m = _cq_map(cq)
    assert m["attn_v"] == "iq2_kt", "MoE MUST protect attn_v (crushing it lost KLD on the 30B)"
    assert m["attn_k"] == "iq2_kt"
    assert m["ffn_gate_exps"] == "iq1_kt" and m["ffn_up_exps"] == "iq1_kt"   # crush cold experts
    assert m["ffn_down_exps"] == "iq2_kt"                                    # protect residual writer
    assert m["ffn_gate_inp"] == "q6_K"                                       # router high


# ---- HYV4 runs through THE MoE recipe (no separate recipe; no Frank-buggy values) ------------
def test_hyv4_via_moe_recipe():
    # HY4 IS a MoE — the SAME recipe, applied to HY4's tensor names (8 layers so blk.4 is middle).
    _, cq = A.recipe_flags(8, is_moe=True, body="iq1_kt", protect="iq2_kt")
    ap = lambda n: _apply(cq, "iq1_kt", n)
    # experts crush; residual writer + router protect (MoE policy, unchanged)
    assert ap("blk.4.ffn_gate_exps.weight") == "iq1_kt" and ap("blk.4.ffn_up_exps.weight") == "iq1_kt"
    assert ap("blk.4.ffn_down_exps.weight") == "iq2_kt" and ap("blk.4.ffn_gate_inp.weight") == "q6_K"
    # HY4 MLA attention caught by the attn_q/k/v substring rules -> protected, same tier as attn
    assert ap("blk.4.attn_k_b.weight") == "iq2_kt"       # attn_k substring (NOT a Frank iq3_kt special-case)
    assert ap("blk.4.attn_v_b.weight") == "iq2_kt"       # attn_v
    assert ap("blk.4.attn_q_a.weight") == "iq2_kt"       # attn_q
    assert ap("blk.4.attn_kv_a_mqa.weight") == "iq2_kt"  # attn_k
    # HY4-only tensors the substring rules miss -> explicit MoE-recipe protect rules
    assert ap("blk.4.attn_gate.weight") == "iq2_kt"
    assert ap("blk.4.hc_attn_fn.weight") == "iq2_kt" and ap("blk.4.hc_ffn_fn.weight") == "iq2_kt"
    assert ap("blk.4.indexer.proj.weight") == "iq2_kt"
    assert ap("blk.4.output_hc_fn.weight") == "iq2_kt"
    # Non-norm special tensors that must NOT get a named rule (they fall through to base; llama-
    # quantize then keeps them F32). Norms ARE substring-matched by attn_q/k rules but that's
    # harmless — llama-quantize keeps norms/1D F32 regardless (proven on the 30B), so we don't
    # over-assert on them here.
    for f32 in ["exp_probs_b", "attn_sinks", "hc_attn_base", "hc_ffn_scale", "hc_attn_scale"]:
        assert ap(f"blk.4.{f32}.weight") == "iq1_kt", f"{f32} must fall through to base, not a named rule"
    # the removed separate recipe must be gone
    assert not hasattr(A, "recipe_flags_hyv4"), "recipe_flags_hyv4 should be deleted — HY4 uses the MoE recipe"


# ---- build vs benchmark (the split): --mix-only=1 build, --no-eval=0 PPL ----------------------
class _Args:
    def __init__(self, **k):
        d = dict(model="M-f16.gguf", imatrix="x.imatrix", eval="e.txt", ngl=99, bin="",
                 log="l.log", body="iq1_kt", protect="iq2_kt", no_imatrix=False, mix_only=False,
                 no_eval=False, rival="", allow_dense=False)
        d.update(k); self.__dict__.update(d)


def test_build_vs_benchmark_split():
    names = _moe()
    bat_full = A.emit_bat(_Args(), 4, True, names)
    bat_fast = A.emit_bat(_Args(mix_only=True, no_eval=True), 4, True, names)
    assert bat_full.count("llama-quantize") == 3 and bat_full.count("llama-perplexity") == 3  # 3-bar benchmark
    assert bat_fast.count("llama-quantize") == 1 and bat_fast.count("llama-perplexity") == 0   # ONE model, no eval


def test_gate_appended_to_oneshot_build():
    # the one-shot (mix-only) build auto-appends the coherence gate on the finished mix
    names = _moe()
    bat = A.emit_bat(_Args(mix_only=True, no_eval=True), 4, True, names)
    assert "--coherence" in bat and "pollard_bench.py" in bat, "one-shot build must append the gate"
    assert "deepseek" not in bat  # sanity: uses the emitted stem, not a stray path
    bat_off = A.emit_bat(_Args(mix_only=True, no_eval=True, gate=False), 4, True, names)
    assert "--coherence" not in bat_off, "--no-gate must omit the gate"


# ---- guards (dense refused; deprecation warns) -----------------------------------------------
def test_dense_guard():
    tf = _tensorfile(_dense())
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "tools",
                        "pollard_automap.py"), "--tensors", tf, "--model", "d.gguf",
                        "--out", os.path.join(tempfile.gettempdir(), "g.bat")],
                       capture_output=True, text=True)
    assert "REFUSED" in (r.stdout + r.stderr), "automap must REFUSE a dense model without --allow-dense"


def _glm(nl=8):
    """GLM4-MoE (GLM-4.5/4.6/5.x class) GGUF tensor names, from gguf-py's GLM4_MOE arch:
    standard MoE + shared experts + q/k norms + the GLM-specific NEXTN (MTP) tail on the last layer."""
    L = ["token_embd.weight", "output.weight", "output_norm.weight"]
    for i in range(nl):
        for t in ["attn_q", "attn_k", "attn_v", "attn_output", "attn_norm",
                  "attn_q_norm", "attn_k_norm", "attn_post_norm"]:
            L.append(f"blk.{i}.{t}.weight")
        if i == 0:                                          # GLM MoE has dense first layer(s)
            for t in ["ffn_gate", "ffn_up", "ffn_down", "ffn_norm"]:
                L.append(f"blk.{i}.{t}.weight")
        else:
            for t in ["ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
                      "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "exp_probs_b", "ffn_norm"]:
                L.append(f"blk.{i}.{t}.weight")
    for t in ["nextn.eh_proj", "nextn.embed_tokens", "nextn.enorm", "nextn.hnorm",
              "nextn.shared_head.head", "nextn.shared_head.norm"]:
        L.append(f"blk.{nl-1}.{t}.weight")                 # MTP head on the last layer
    return L


def test_glm_moe_routing():
    # A non-Qwen family (GLM) one-shots through THE MoE recipe by detected features, no hand edits.
    _, nl, is_moe, arch = A.parse_tensors(_tensorfile(_glm(8)))
    assert is_moe and arch.startswith("MoE"), f"GLM must detect MoE, got '{arch}'"
    _, cq = A.recipe_flags(8, is_moe=True, body="iq1_kt", protect="iq2_kt")
    ap = lambda n: _apply(cq, "iq1_kt", n)
    assert ap("blk.4.ffn_gate_exps.weight") == "iq1_kt" and ap("blk.4.ffn_up_exps.weight") == "iq1_kt"
    assert ap("blk.4.ffn_down_exps.weight") == "iq2_kt"    # residual writer protected
    assert ap("blk.4.ffn_gate_inp.weight") == "q6_K"       # router protected
    assert ap("blk.4.ffn_gate_shexp.weight") != "iq1_kt"   # shared experts NOT crushed to body
    # GLM-specific NEXTN/MTP tail is on the LAST layer -> edge-protect tier, never the 1-bit body
    assert ap("blk.7.nextn.eh_proj.weight") == "iq2_kt", "GLM MTP tail must be protected, not crushed"
    assert ap("blk.7.nextn.shared_head.head.weight") == "iq2_kt"
# ---- alloc-v2: finer granular FFN allocation (gate+up vs down) -------------------------------
def test_alloc_granular_protects_down():
    import pollard_fit as F
    h = 2048; ffn = 3 * h * 5632; attn = 4 * h * h; L = 12
    arch = dict(kind="dense", layers=L, hidden=h, total=L * (ffn + attn) + 2 * h * 32000,
                expert_params=0, n_experts=0, dense_ffn_params=ffn, attn_params=attn)
    def down_type(sens):
        ov, *_ = F.plan_allocation(arch, 0.55, 0.05, sensitivity=sens)
        m = {p.replace("\\", ""): t for p, t in ov}
        return m.get("blk.0.ffn_gate.weight"), m.get("blk.0.ffn_down.weight")
    # lumped profile: gate and down share a rung
    g0, d0 = down_type(dict(layers=L, ffn={str(i): 0.5 for i in range(L)},
                            attn={str(i): 0.2 for i in range(L)}))
    assert g0 == d0, f"non-granular should share a rung, got gate={g0} down={d0}"
    # granular profile with down MORE sensitive: down must land on a HIGHER rung than gate/up
    gg, dg = down_type(dict(layers=L, ffn={str(i): 0.5 for i in range(L)},
                            attn={str(i): 0.2 for i in range(L)},
                            ffn_gateup={str(i): 0.2 for i in range(L)},
                            ffn_down={str(i): 0.95 for i in range(L)}))
    assert F.BPW[dg] > F.BPW[gg], f"granular must protect down above gate/up, got gate={gg} down={dg}"


def test_card_license_is_never_invented():
    """A card's license line is a legal claim about someone else's weights, so the template must
    read it, not default it. The old code asked `config.json` -- which almost never carries a
    license -- and fell through to a hardcoded "apache-2.0", which published Ling-3.0-tiny (MIT)
    under the wrong license. An unresolvable base must come back None so the tool warns.
    """
    import pollard_card as C

    # config wins when it actually carries one
    assert C.base_license("whoever/whatever", {"license": "mit"}) == "mit"

    # a local checkout resolves from the card frontmatter, not from a default
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("---\nlicense: mit\npipeline_tag: text-generation\n---\n# hi\n")
    assert C.base_license(d, {}) == "mit", "must read the frontmatter of a local base model"

    # frontmatter with no license must not invent one
    d2 = tempfile.mkdtemp()
    with open(os.path.join(d2, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("---\npipeline_tag: text-generation\n---\n# hi\n")
    assert C.base_license(d2, {}) is None, "no license present must resolve to None, not a guess"

    # and the module must not carry a fallback license string anywhere
    src = open(os.path.join(os.path.dirname(__file__), "..", "tools", "pollard_card.py"),
               encoding="utf-8").read()
    assert "apache-2.0" not in src, "pollard_card must not hardcode any license"


def _mini_gguf(types, path=None, arch=b"llama"):
    """Write a minimal GGUF with one tensor per entry in `types` (ggml type ids).

    The architecture is a real upstream one by default: pollard-ggufcheck rejects an unknown
    `general.architecture` on its own, so a placeholder here would make every atom assertion fail for
    the wrong reason.
    """
    import struct
    path = path or tempfile.NamedTemporaryFile(suffix=".gguf", delete=False).name
    with open(path, "wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<I", 3))                 # version
        fh.write(struct.pack("<Q", len(types)))        # tensor count
        fh.write(struct.pack("<Q", 1))                 # one kv pair
        # kv: general.architecture -> a REAL upstream architecture, so this fixture exercises the
        # atom check without tripping the separate architecture check
        k = b"general.architecture"
        fh.write(struct.pack("<Q", len(k))); fh.write(k)
        fh.write(struct.pack("<I", 8))
        fh.write(struct.pack("<Q", len(arch))); fh.write(arch)
        for i, t in enumerate(types):
            n = f"blk.{i}.weight".encode()
            fh.write(struct.pack("<Q", len(n))); fh.write(n)
            fh.write(struct.pack("<I", 1))             # 1 dimension
            fh.write(struct.pack("<Q", 32))            # dim 0
            fh.write(struct.pack("<I", t))             # ggml type
            fh.write(struct.pack("<Q", 0))             # offset
    return path


def test_ggufcheck_reads_runtime_from_types_not_names():
    """Which runtime loads a GGUF is decided by its tensor types, never by its filename.

    Stock llama.cpp rejects any ggml type above 42 outright, so a single protected tensor carrying an
    ik_llama-only atom makes the whole file ik_llama-only -- which is exactly how a rung named
    `IQ4_XS` shipped with IQ5_K on every attn_v and a card saying it ran anywhere. The allocator is
    supposed to reach for that atom; the card just has to say so.
    """
    import pollard_ggufcompat as gc

    # stock-only ladder: F32 + IQ4_XS(23) + Q6_K(14)
    p = _mini_gguf([0, 23, 23, 14])
    assert gc.runtime_of(p) == ("stock", {}), "an all-stock file must not be called ik_llama"

    # the real shape of the bug: IQ4_XS body, IQ5_K (140) on the protected tensors
    p2 = _mini_gguf([0, 23, 23, 140, 140])
    rt, fo = gc.runtime_of(p2)
    assert rt == "ik_llama", "a fork-only atom anywhere makes the file ik_llama-only"
    assert fo == {"IQ5_K": 2}, f"must name the atom and count it, got {fo}"

    # the trellis family must be named too, not reported as a bare number
    rt3, fo3 = gc.runtime_of(_mini_gguf([0, 158, 153]))
    assert rt3 == "ik_llama" and set(fo3) == {"IQ1_KT", "IQ2_KT"}, fo3

    # 42 is the last stock id; 43 is not
    assert gc.runtime_of(_mini_gguf([0, 42]))[0] == "stock"
    assert gc.runtime_of(_mini_gguf([0, 43]))[0] == "ik_llama"

    # a file with no GGUF magic must raise, not quietly read as stock -- a zeroed header means the
    # file will not load in ANY runtime, which is worth surfacing rather than defaulting.
    blank = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
    blank.write(b"\0" * 4096); blank.close()
    try:
        gc.runtime_of(blank.name)
        raise AssertionError("a header-less file must raise, not report a runtime")
    except ValueError:
        pass


def test_recard_harvest_is_idempotent():
    """Recarding a card that has already been recarded must not double its sections.

    Two of the harvest passes can see the same prose: one keeps every non-template section verbatim,
    the other gathers loose paragraphs carrying measured numbers into a "Measured notes" section.
    A card that already has that section feeds it to both, so the section grew a fresh copy on every
    run -- caught with two identical "## Measured notes" blocks on Qwen2.5-7B and -14B before they
    were uploaded. Same card in, same card out.
    """
    import re as _re

    import pollard_recard as R

    card = (
        "---\nlicense: apache-2.0\nbase_model: Qwen/Qwen2.5-7B-Instruct\n---\n\n"
        "# M\n\n"
        "## Available files\n\n"
        "| file | PPL | size |\n|---|---:|---:|\n| `m-Q6_K.gguf` | 6.55 | 6.25 GB |\n\n"
        "## Measured notes\n\n"
        "The flagship reaches PPL 10.23 at 0.537 Mean KLD, a real step over the baseline.\n\n"
        "## Errata\n\n- something\n"
    )
    files = ["m-Q6_K.gguf"]

    def heads(t):
        return _re.findall(r"^##\s+(.+)$", t or "", _re.M)

    _, e1 = R.harvest(card, files)
    h1 = heads(e1)
    assert h1.count("Measured notes") == 1, f"harvested it {h1.count('Measured notes')} times: {h1}"

    # feed the harvest back in, the way recarding an already-recarded repo does
    _, e2 = R.harvest(card + "\n\n" + (e1 or ""), files)
    h2 = heads(e2)
    assert h2 == h1, f"not a fixed point: {h1} then {h2}"
    assert len(h2) == len(set(h2)), f"duplicate sections after a second pass: {h2}"

    # a section appearing twice in the source card is also collapsed to one
    _, e3 = R.harvest(card + "\n## Measured notes\n\nThe flagship reaches PPL 10.23 at 0.537 Mean "
                              "KLD, a real step over the baseline.\n", files)
    assert heads(e3).count("Measured notes") == 1, heads(e3)


def test_reclaim_never_mistakes_a_build_for_a_source():
    """The one mistake pollard-reclaim must not make: offering a published ladder as a "source".

    The workspace stores a ladder INSIDE a directory named after the f16 body it was built from
    (`models/FrogMini-14B-f16.gguf/FrogMini-14B-f16.gguf-Pollard-GGUF-Q6_K.gguf`), so any test that
    looks at the path -- or even at the basename alone -- sees "-f16" and calls the rung a source.
    Under --delete --sources that deletes the whole published ladder. Caught on the real box before
    anything was removed.
    """
    import pollard_reclaim as R

    ladder = os.path.join("models", "FrogMini-14B-f16.gguf",
                          "FrogMini-14B-f16.gguf-Pollard-GGUF-Q6_K.gguf")
    assert not R.is_source_name(ladder), "a build carrying -Pollard is never a source"
    assert not R.is_source_name("Qwen2.5-7B-Instruct-Pollard-IQ4_XS.gguf")

    # a real unquantized body still is one
    assert R.is_source_name("Qwen2.5-7B-Instruct-f16.gguf")
    assert R.is_source_name(os.path.join("downloads", "FrogMini-14B-bf16.gguf"))
    # and an ordinary quant with no marker is neither
    assert not R.is_source_name("some-model-Q6_K.gguf")


def test_reclaim_repo_guesses_strip_source_decoration():
    """A manifest key is often the source GGUF's path, not an HF id.

    `FrogMini-14B-f16.gguf` was published as `PollardWeights/FrogMini-14B-Pollard`, so the extension
    and the f16/bf16 marker have to come off before the repo is guessed, or nothing matches and
    nothing is ever reclaimable.
    """
    import pollard_reclaim as R

    g = R.repo_guesses("C:/pollard/phome/models/FrogMini-14B-f16.gguf", "PollardWeights")
    assert "PollardWeights/FrogMini-14B-Pollard" in g, g
    g2 = R.repo_guesses("Qwen/Qwen2.5-7B-Instruct", "PollardWeights")
    assert "PollardWeights/Qwen2.5-7B-Instruct-Pollard" in g2, g2
def test_rope_conventions_are_distinct_and_correct():
    """The one-line difference that invalidates every activation-derived measurement.

    Rotate-half (NeoX) pairs element i with i+d/2; interleaved (Megatron/PTM) pairs 2i with 2i+1.
    Applying the wrong one to a checkpoint scrambles relative positions in every layer, and nothing
    about the shapes or the key names looks wrong -- the model loads and generates, it is just
    measuring the wrong thing. It cost Hy4-preview a bf16 NLL of 5.02 instead of 1.855 with routing
    mass off by 24 points (issue #65), and K2-Horizon nearly shipped with NORM where the reference
    fork says NEOX.

    Both conventions are checked against an explicit pairwise rotation, so the test states which is
    which rather than just pinning whatever the code happens to do.
    """
    import numpy as np

    import pollard_refcheck as RC

    T, D = 6, 8
    pos = np.arange(T)[:, None]
    inv = 1.0 / (10000 ** (np.arange(0, D, 2) / D))          # d/2 distinct angles
    ang = pos * inv[None, :]
    # transformers builds cos/sin in the half-width layout: the d/2 angles, concatenated
    cos = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1)
    sin = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1)
    x = np.random.default_rng(0).standard_normal((T, D))

    half = RC.rope_rotate_half(x, cos, sin, np)
    inter = RC.rope_interleaved(x, cos, sin, np)
    assert not np.allclose(half, inter), "the two conventions must not be the same function"

    # interleaved: rotate each adjacent (2i, 2i+1) pair by its own angle
    gt = np.empty_like(x)
    for t in range(T):
        for i in range(D // 2):
            c, s2 = np.cos(ang[t, i]), np.sin(ang[t, i])
            a0, a1 = x[t, 2 * i], x[t, 2 * i + 1]
            gt[t, 2 * i] = a0 * c - a1 * s2
            gt[t, 2 * i + 1] = a1 * c + a0 * s2
    assert np.allclose(inter, gt, atol=1e-12), "interleaved must rotate ADJACENT pairs"

    # rotate-half: pair i with i + d/2
    d = D // 2
    gt2 = np.empty_like(x)
    for t in range(T):
        for i in range(d):
            c, s2 = np.cos(ang[t, i]), np.sin(ang[t, i])
            a0, a1 = x[t, i], x[t, i + d]
            gt2[t, i] = a0 * c - a1 * s2
            gt2[t, i + d] = a1 * c + a0 * s2
    assert np.allclose(half, gt2, atol=1e-12), "rotate-half must pair i with i+d/2"

    # a rotation changes direction, never length
    for got in (half, inter):
        assert np.allclose(np.linalg.norm(got, axis=-1), np.linalg.norm(x, axis=-1)), \
            "rotary must preserve the norm"


def test_refcheck_flags_a_broken_forward():
    """The gate has to fire on the measured signature, not just on a threshold.

    Hy4's broken forward scored 5.02 nats AND got worse along the sequence (5.1 at positions 0-64 ->
    5.8 at 1024-2047). A single mean can be argued away as hard rows; loss that RISES with distance is
    positional damage, so both signals are checked.
    """
    import pollard_refcheck as RC

    broken = {"nll": 5.02, "ppl": 151.0, "head_nll": 5.1, "tail_nll": 5.8,
              "tail_ratio": 5.8 / 5.1, "tokens": 16384}
    ok, why = RC.verdict(broken, expect=1.86)
    assert not ok and len(why) >= 2, why
    assert any("RISES" in w for w in why), "the rising-loss signature must be named"

    fixed = {"nll": 1.855, "ppl": 6.39, "head_nll": 1.9, "tail_nll": 1.82,
             "tail_ratio": 1.82 / 1.9, "tokens": 16384}
    ok2, why2 = RC.verdict(fixed, expect=1.86)
    assert ok2, why2

    # hy_v4 is on record with the expected value the gate compares against
    assert "hy_v4" in RC.KNOWN_DEFECTS
    assert RC.KNOWN_DEFECTS["hy_v4"]["expect_nll"] < RC.NLL_SUSPECT


def test_ggufcheck_catches_a_fork_only_architecture():
    """Stock-only atoms are not enough: the architecture string has to be one stock llama.cpp knows.

    `general.architecture` is checked by name against upstream's LLM_ARCH_NAMES. A brand-new model
    whose support lives in a vendor fork produces a file of perfectly ordinary K-quants that opens
    nowhere else -- which is what all three K2-Horizon repos shipped. Every tensor is stock, the
    tensor-type check passed them clean, and all three cards said the K-quants run anywhere.
    """
    import pollard_ggufcompat as gc

    known = gc.stock_archs()
    assert len(known) > 100, f"the architecture list looks wrong ({len(known)} entries)"
    for ordinary in ("llama", "qwen2", "qwen3", "qwen3moe", "bailingmoe3", "deepseek2"):
        assert ordinary in known, f"{ordinary} must be recognised as stock-loadable"

    # the ones we actually ship that upstream does not implement
    assert "k2-horizon" not in known, "k2-horizon is not an upstream architecture"
    assert "k2-horizon" in gc.FORK_ARCHS, "a fork-only architecture must record where it IS supported"
    name, url = gc.FORK_ARCHS["k2-horizon"]
    assert "MBZUAI" in name and url.startswith("https://"), (name, url)

    # `clip` is a quantize-only dummy and must not be offered as a loadable architecture
    assert "clip" not in known


def test_arch_support_never_calls_a_new_architecture_a_fork():
    """A stale architecture list must not become a claim about someone else's runtime.

    The first version of this check compared against one snapshot and reported anything missing as
    fork-only. The snapshot came from the vendored runtime, which was three entries behind master, so
    Spark-X2.5-4B was published as needing a fork when upstream had merged `spark2_5` four days
    earlier (llama.cpp #27868, 2026-09-06). "Update llama.cpp" and "go get someone else's build" are
    different instructions and only one of them was true.

    So a miss against the local list is the trigger to ask upstream, and there are four answers, not
    two. Unreachable upstream must produce "unknown" -- never "fork".
    """
    import pollard_ggufcompat as gc

    local = {"llama", "qwen2", "k2-horizon-not-this"}

    # in the local list -> stock, and upstream is never consulted
    assert gc.arch_support("llama", local=local, offline=True)[0] == "stock"

    # missing locally and upstream cannot be reached -> claim nothing
    v, why = gc.arch_support("spark2_5", local=local, offline=True)
    assert v == "unknown", (v, why)
    assert "no claim" in why.lower(), why

    # missing locally but upstream has it -> a version requirement, not a fork
    saved = gc._upstream_cache
    try:
        gc._upstream_cache = {"llama", "qwen2", "spark2_5"}
        v2, why2 = gc.arch_support("spark2_5", local=local)
        assert v2 == "newer", (v2, why2)
        # and one upstream does NOT have is a fork, named
        v3, why3 = gc.arch_support("k2-horizon", local=local)
        assert v3 == "fork", (v3, why3)
        assert "MBZUAI" in why3, why3
    finally:
        gc._upstream_cache = saved

    # the shipped snapshot must be at least as new as the architectures we publish against
    assert "spark2_5" in gc.STOCK_ARCHS, "the snapshot is stale again"
    assert "k2-horizon" not in gc.STOCK_ARCHS, "k2-horizon is not upstream"

    # the atom tables must survive any future edit to the architecture block -- they were once
    # deleted by a rewrite of it, which silently made every atom verdict raise
    assert gc.type_name(140) == "IQ5_K" and gc.type_name(23) == "IQ4_XS"
    assert gc.fork_only({0: 1, 23: 10, 140: 28}) == {"IQ5_K": 28}


def test_runtime_reads_archs_from_source_and_binaries():
    """A runtime's architecture list has to be read from the binary, not only the source tree.

    A tree can be reset, re-pointed or rebuilt after a model was made, and then the source says one
    thing while the binary that actually runs says another. Spark-X2.5-4B was quantized on 2026-09-08
    against a llama.cpp carrying `spark2_5` (upstream #27868, merged 2026-09-06); the tree was later
    rebuilt as the IFM fork, whose base predates that merge, and the published model stopped loading
    anywhere on either machine. Source-only inspection cannot see that happen.
    """
    import pollard_runtime as R

    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "src"), exist_ok=True)
    with open(os.path.join(d, "src", "llama-arch.cpp"), "w", encoding="utf-8") as fh:
        fh.write("static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {\n")
        fh.write('    { LLM_ARCH_CLIP, "clip" },\n')
        for i in range(30):
            fh.write(f'    {{ LLM_ARCH_M{i}, "made-up-{i}" }},\n')
        fh.write('    { LLM_ARCH_K2, "k2-horizon" },\n')
        fh.write("};\n")
    got = R._arch_list_from_source(d)
    assert got and "k2-horizon" in got, got
    assert "clip" not in got, "clip is a quantize-only dummy, not a loadable architecture"

    # a binary is searched for the literal strings, independently of the source
    os.makedirs(os.path.join(d, "build", "bin"), exist_ok=True)
    binpath = os.path.join(d, "build", "bin", "llama-cli")
    with open(binpath, "wb") as fh:
        # literals in a real binary are NUL-delimited; `notspark2_5x` is deliberately adjacent text
        # that must NOT count as a hit
        fh.write(b"\x00notspark2_5x\x00" + b"spark2_5\x00" + b"\x00" * 32 + b"qwen3moe\x00")
    found, nfiles = R.archs_in_binaries(d, ["spark2_5", "qwen3moe", "k2-horizon"])
    assert nfiles >= 1, "the binary should have been scanned"
    assert found == {"spark2_5", "qwen3moe"}, found
    assert "k2-horizon" not in found, "must not report an architecture the binary does not carry"

    # a name embedded in a longer identifier is not a hit on its own
    only_embedded = os.path.join(d, "build", "bin", "llama-perplexity")
    with open(only_embedded, "wb") as fh:
        fh.write(b"\x00xxk2-horizonyy\x00")
    got2, _ = R.archs_in_binaries(os.path.dirname(only_embedded), ["k2-horizon"])
    assert got2 == set(), got2

    # and the source list did NOT contain spark2_5 -- the two views genuinely differ, which is the point
    assert "spark2_5" not in got


def test_runtime_patch_verification_survives_line_ending_drift():
    """A captured runtime patch must verify by CONTENT, not by whether git can reverse-apply it.

    Our runtime trees are CRLF on the Windows box and the patches get read on a Mac. One captured patch
    would not reverse-apply even with --ignore-whitespace while every added line was still in the file,
    so a reverse-apply-only check reported LOST on three patches that were all applied. That is worse
    than not checking, because the whole point is to notice when support really has gone -- which is
    how spark2_5 was lost.
    """
    import pollard_runtime as R

    patch = (
        "diff --git a/src/unicode.cpp b/src/unicode.cpp\n"
        "--- a/src/unicode.cpp\n"
        "+++ b/src/unicode.cpp\n"
        "@@ -1,3 +1,5 @@\n"
        " context line\n"
        "+static void k2_horizon_split() {\n"
        "+    return;\n"
        " more context\n"
    )
    want = R._added_lines(patch)
    assert want == {"static void k2_horizon_split() {", "return;"}, want
    assert not any(l.startswith("+++") for l in want), "the +++ header is not an added line"

    # the same additions, arriving with CRLF endings and different indentation, must still match
    crlf = patch.replace("+static void k2_horizon_split() {",
                         "+  static void k2_horizon_split() {\r").replace("+    return;", "+\treturn;\r")
    assert R._added_lines(crlf) == want, R._added_lines(crlf)

    # a blank added line carries no content and must not count toward the total
    assert R._added_lines("+++ b/x\n+\n+real\n") == {"real"}


def test_install_state_catches_a_declared_but_uninstalled_command():
    """A synced repo can still be missing the command for a tool it declares.

    An editable install keeps module code current -- `import pollard_card` resolves straight into the
    checkout -- but pip only writes command launchers when it runs. Add a tool, sync, and the code is
    there while the command is not, which nobody notices until they type the name. Every tool added in
    one session sat in exactly that state on both machines.
    """
    import pollard_runtime as R

    d = tempfile.mkdtemp()
    with open(os.path.join(d, "pyproject.toml"), "w", encoding="utf-8") as fh:
        fh.write('[project]\nname = "x"\n\n[project.scripts]\n'
                 'pollard-alpha = "a:main"\npollard-beta = "b:main"\n')
    got = R.declared_scripts(d)
    assert got == {"pollard-alpha": "a:main", "pollard-beta": "b:main"}, got

    st = R.install_state(d)
    assert st is not None
    assert set(st["declared"]) == {"pollard-alpha", "pollard-beta"}
    # this interpreter has neither, so both must be reported missing rather than assumed fine
    assert set(st["missing"]) == {"pollard-alpha", "pollard-beta"}, st["missing"]

    # no pyproject at all is "cannot tell", not "all good"
    assert R.install_state(tempfile.mkdtemp()) is None


def test_capture_refuses_encoding_damage():
    """A captured runtime patch must not smuggle encoding damage in as if it were work.

    An editor that reads a UTF-8 source as Latin-1 and writes it back replaces lines with broken
    copies of themselves. Captured, that becomes the thing you re-apply after a clone, so the damage
    is permanent. Real case: 27 of 39 hunks in a k2-horizon capture were a byte-order mark or
    mojibake, and the mojibake sat inside a DeepSeek pre-tokenizer regex -- any build made from that
    patch would mis-tokenize DeepSeek models.
    """
    import pollard_runtime as R

    good = ("diff --git a/x.c b/x.c\n--- a/x.c\n+++ b/x.c\n@@ -1,1 +1,2 @@\n"
            " int main(void) {\n+    return 0;\n")
    assert R.encoding_noise(good) == "", R.encoding_noise(good)

    bom = "+\ufeff#include <stdio.h>\n-#include <stdio.h>\n"
    assert "byte-order mark" in R.encoding_noise(bom)

    # 'A-tilde circumflex' is how an em dash looks after a UTF-8 -> Latin-1 -> UTF-8 round trip
    moji = "+// clear the graph \u00c3\u00a2\u00c2\u0080\u00c2\u0094 before reuse\n-// clear the graph before reuse\n"
    assert "Latin-1" in R.encoding_noise(moji), R.encoding_noise(moji)

    # a legitimate non-ASCII addition is NOT damage: real unicode in a tokenizer regex must pass
    legit = '+        "\\s?[!-/:-~\uff01-\uff0f\u2018-\u201f\u3000-\u3002]+",\n'
    assert R.encoding_noise(legit) == "", R.encoding_noise(legit)


def test_bench_coherence_does_not_swallow_speed():
    """Asking pollard-bench for two measurements must not silently return one.

    The coherence gate exited as soon as it had a verdict, which fired BEFORE the --speed branch. So
    `--coherence --speed` printed a gate, exited 0, and produced no tok/s at all, with nothing saying
    a requested measurement had been dropped. That is very likely why published cards have been
    missing tok/s: the obvious command looks like it did everything.

    Checked against the source rather than by running llama-cli, since the bug is purely one of
    control flow -- the early exit must be conditional on nothing else having been asked for.
    """
    import re as _re

    src = open(os.path.join(os.path.dirname(__file__), "..", "tools", "pollard_bench.py"),
               encoding="utf-8").read()

    # the gate's early exit must not fire when --speed was also requested
    m = _re.search(r"gate_passed = print_gate\(res\)\s*\n\s*if ([^\n:]+):", src)
    assert m, "the gate no longer stores its verdict before deciding to exit"
    cond = m.group(1)
    assert "a.speed" in cond, f"the gate still exits without considering --speed: {cond!r}"

    # and the speed branch must come AFTER the gate, so both can run in one invocation
    i_gate = src.index("gate_passed = print_gate(res)")
    i_speed = src.index("=== decode speed ===")
    assert i_gate < i_speed, "speed must run after the gate for both to be reachable"

    # the speed-only exit must still report a gate verdict when a gate ran
    tail = src[i_speed:i_speed + 2000]
    assert "gate_passed is None" in tail, \
        "the speed exit must distinguish 'no gate ran' from 'gate failed'"


def test_backend_report_flags_rewrites_not_just_rejections():
    """A backend that ACCEPTS a file can still change it, and that must be reported too.

    OpenVINO's NPU path requantizes Q6_K to Q4_0_128. The load succeeds, the model runs, and the
    measured allocation is gone -- the same failure shape as a fork-only atom or an unknown
    architecture, which is the third time this pattern has cost us. And no IQ type is on OpenVINO's
    accepted list at all, so most of what Pollard publishes cannot load there in any form.
    """
    import pollard_ggufcompat as gc

    # an IQ ladder rung: unsupported everywhere on OpenVINO
    iq = {0: 100, 23: 200, 21: 50}            # F32, IQ4_XS, IQ3_S
    rep = gc.backend_report(iq)
    for key in ("openvino-cpu", "openvino-gpu", "openvino-npu"):
        verdict, detail = rep[key]
        assert verdict == "unsupported", (key, verdict, detail)
        assert "IQ4_XS" in detail, detail

    # Q6_K: accepted everywhere, but rewritten -- and to something WORSE on NPU
    q6 = {0: 100, 14: 200}                    # F32, Q6_K
    rep2 = gc.backend_report(q6)
    assert rep2["openvino-cpu"][0] == "rewritten", rep2["openvino-cpu"]
    assert "Q8_0_C" in rep2["openvino-cpu"][1]
    assert rep2["openvino-npu"][0] == "rewritten", rep2["openvino-npu"]
    assert "Q4_0_128" in rep2["openvino-npu"][1], rep2["openvino-npu"][1]

    # Q4_0 is the one scheme that survives untouched on every OpenVINO device
    q4 = {0: 100, 2: 200}                     # F32, Q4_0
    rep3 = gc.backend_report(q4)
    assert all(v == "ok" for v, _ in rep3.values()), rep3

    # F32 alone must never be called a quantization problem
    assert all(v == "ok" for v, _ in gc.backend_report({0: 10}).values())


def test_kv_counts_only_the_layers_that_produce_it():
    """KV must be counted on the layers that PRODUCE it, and the indexer cache on every arch.

    DeepSeek-V4.1-Flash runs CSA2: most layers read a previous layer's KV and keep none of their own.
    Its config says which produce it -- kv_source_layer_ids = [2, 8, 14, 20], four of forty layers.
    Counting all forty overstates the cache by 10x, and at a 1M context that is the number deciding
    whether the model fits a box at all.

    The second bug was worse because it was silent: the indexer cache was added only on the MLA
    branch. This model has no kv_lora_rank, so its 8 indexer layers vanished and the estimate halved.

    Validated against a served measurement: 12 GiB pinned holding 2.56 M tokens with fp4 KV
    = 4.92 KB/token. The model lands at 4.00, 19 % under -- the right order, where it had been 2.5x
    out before.
    """
    import pollard_calc as pc

    cfg = {
        "num_hidden_layers": 40,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "index_head_dim": 128,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "hidden_size": 5120,
        "num_attention_heads": 64,
    }
    a = pc.analyse(cfg)
    assert a["layers"] == 40
    assert a["n_kv_layers"] == 4, a.get("n_kv_layers")
    assert a["n_indexer"] == 8, a.get("n_indexer")

    kb = pc.kv_cache_bytes(a, 1_000_000, 0.5) / 1_000_000 / 1024
    assert 3.5 < kb < 6.0, f"{kb} KB/token is not near the measured 4.92"

    # counting all 40 layers -- the old behaviour -- must be far away from measured
    naive = dict(a); naive["n_kv_layers"] = 0; naive["n_full"] = 0
    kb_naive = pc.kv_cache_bytes(naive, 1_000_000, 0.5) / 1_000_000 / 1024
    assert kb_naive > 2 * kb, f"all-layer count {kb_naive} should dwarf {kb}"

    # and the indexer must be counted even with no MLA latent present
    no_idx = dict(a); no_idx["index_head_dim"] = 0
    assert pc.kv_cache_bytes(no_idx, 1_000_000, 0.5) < pc.kv_cache_bytes(a, 1_000_000, 0.5), \
        "the indexer cache must add bytes on a non-MLA architecture"


def test_archfp_surfaces_per_layer_structure_a_twin_would_miss():
    """A layout can score as a known twin while the config says it behaves nothing like one.

    DeepSeek-V4.1-Flash's tensor names fingerprint as an ordinary MLA MoE. What makes it different
    lives in LIST-valued hparams the numeric gap check could not see: kv_source_layer_ids says only
    4 of its 40 layers produce KV at all. A twin that ignores that overstates the cache tenfold, on a
    model whose 1M context makes KV the number that decides whether it fits anything.
    """
    import pollard_archfp as A

    names = ["token_embd.weight", "output.weight", "output_norm.weight"]
    for i in range(4):
        for t in ("attn_q_a", "attn_q_b", "attn_kv_a", "attn_kv_b", "attn_out", "attn_norm",
                  "ffn_norm", "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            names.append(f"blk.{i}.{t}.weight")
    fp = A.fingerprint(names)

    hp = {
        "num_hidden_layers": 40,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "engram_layer_ids": [1, 14],
        "compress_ratios": [0, 0] + [2] * 38,
        "hc_mult": 4,
    }
    res = A.twin(fp, hp)
    what = " | ".join(g["what"] for g in res["gaps"])

    assert not res["exact"], "a model with per-layer KV sourcing must never read as an exact twin"
    assert "kv_source_layer_ids" in what, what
    assert "4 of 40 layers" in what, what          # the count, not just the key name
    assert "index_source_layer_ids" in what, what
    assert "engram_layer_ids" in what, what
    assert "hc_mult" in what, what

    # every gap must say what supporting it would take, not merely refuse
    for g in res["gaps"]:
        assert g.get("to_support"), g

    # an ordinary model must not collect these gaps
    plain = A.twin(fp, {"num_hidden_layers": 40})
    assert "kv_source_layer_ids" not in " ".join(g["what"] for g in plain["gaps"])


def test_speed_parser_reads_the_classic_timing_block():
    """Decode speed must parse on ik_llama too, not only on builds that print "Generation: t/s".

    pollard-bench matched one format. ik_llama and older llama.cpp print the classic timing block
    instead, so every trellis build came back "no speed line" -- which is why no IQ*_KT rung has ever
    carried a tok/s figure on a card. Measuring the wrong thing is bad; silently measuring nothing and
    reporting success is worse.
    """
    import pollard_bench as pb

    classic = (
        "main: prompt eval time =     217.23 ms /     1 tokens (  217.23 ms per token,"
        "     4.60 tokens per second)\n"
        "main:        eval time =     156.00 ms /    32 tokens (    4.88 ms per token,"
        "   205.12 tokens per second)\n"
        "main:       total time =     400.00 ms\n"
    )
    gen, pro = pb._classic_speeds(classic)
    assert gen == 205.12, gen          # generation is the plain "eval time" line
    assert pro == 4.60, pro            # prompt is the "prompt eval time" line

    # the two lines must not be confused: prompt is far slower here, so a mix-up is obvious
    assert gen > pro

    # a build that prints neither format yields nothing rather than a wrong number
    assert pb._classic_speeds("main: total time = 400.00 ms\n") == (None, None)

    # lines mentioning tokens per second that are NOT timing lines are ignored
    assert pb._classic_speeds("note: we measured 999.0 tokens per second once\n") == (None, None)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}")
        except Exception as e:
            fails += 1; print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests)-fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
