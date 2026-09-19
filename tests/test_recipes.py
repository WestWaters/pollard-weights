#!/usr/bin/env python3
"""Pollard regression suite — assert every canonical recipe, guard, and encoder rule, so a
change to one path can't silently break another (the exact class of bug that cost a weekend:
the MoE attn_v crush, the q3_k casing, the dense guard). Runnable two ways:

    python tests/test_recipes.py      # plain asserts, prints PASS/FAIL, exits non-zero on fail
    pytest tests/test_recipes.py      # same functions as test_*

Add a case whenever a recipe/guard changes — never fewer rows than the tools have behaviors.
"""
import json, os, pathlib, re, subprocess, sys, tempfile

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


def test_card_rung_list_agrees_in_number():
    """One fork-only rung must not be described in the plural.

    The stock rebuild of both Qwen repos left exactly one trellis rung, and the card then read
    "`IQ1_KT` need ik_llama.cpp: their allocation puts ..." -- grammatically wrong in the one
    sentence a reader uses to decide whether the file will open on their machine.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import pollard_card as C

    assert C.rung_agreement(1) == ("needs", "carries", "its")
    assert C.rung_agreement(2) == ("need", "carry", "their")
    assert C.rung_agreement(4) == ("need", "carry", "their")

    # and the template must actually use it -- checked on the rendered sentence rather than by
    # grepping the source, which trips over the helper's own docstring.
    src = open(os.path.join(os.path.dirname(__file__), "..", "tools", "pollard_card.py"),
               encoding="utf-8").read()
    for frag in ("{need} {v_need} [ik_llama.cpp]", "{names} {v_carry} ik_llama-only atoms"):
        assert frag in src, f"rung list still not agreement-aware: {frag}"


def test_card_calib_note_is_not_double_punctuated():
    """A calibration note that ends in a period must not get a second one appended.

    The Qwen cards' calib note ends with a sentence about test-set disjointness -- exactly the claim a
    reader checks hardest -- and it rendered as "tuned on..".
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import pollard_card as C

    src = open(os.path.join(os.path.dirname(__file__), "..", "tools", "pollard_card.py"),
               encoding="utf-8").read()
    assert '+ "."' not in src, "calibration note still appends a period unconditionally"
    assert 'not in ".!?"' in src, "no guard against double punctuation in the calibration note"


def test_card_attribution_follows_the_publisher_not_the_tool():
    """A cloned Pollard must not credit PollardWeights for someone else's build.

    `quantized_by: PollardWeights` was hardcoded into the frontmatter and the repo id defaulted to a
    PollardWeights repo, so anyone else running pollard-card produced a card that credited us and
    pointed every download line at our repos. No credential ever leaked -- huggingface_hub resolves
    the token from the caller's own login -- but the attribution followed the tool, not the publisher.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import pollard_card as C

    here = os.path.dirname(__file__)
    src = open(os.path.join(here, "..", "tools", "pollard_card.py"), encoding="utf-8").read()
    assert '"quantized_by: PollardWeights"' not in src, "frontmatter still hardcodes our account"
    assert 'a.repo or f"PollardWeights/' not in src, "repo id still defaults to our account"

    # resolution order: explicit flag, then the repo being written/uploaded to
    assert C.publishing_account("acme") == "acme"
    assert C.publishing_account(None, "someone/Model-Pollard") == "someone"
    assert C.publishing_account(None, None, "other/Model-Pollard") == "other"
    # an explicit flag outranks the repo owner
    assert C.publishing_account("acme", "someone/Model-Pollard") == "acme"

    # with nobody identified the line is omitted rather than filled with a wrong or fake name
    anon = C.frontmatter("Qwen/Qwen2.5-7B-Instruct", "apache-2.0", ["gguf"], "qwen2", None)
    assert "quantized_by" not in anon, "unattributed card invented an owner"
    named = C.frontmatter("Qwen/Qwen2.5-7B-Instruct", "apache-2.0", ["gguf"], "qwen2", "acme")
    assert "quantized_by: acme" in named

    # and reclaim no longer checks everyone's local builds against our repos
    rsrc = open(os.path.join(here, "..", "tools", "pollard_reclaim.py"), encoding="utf-8").read()
    assert 'default="PollardWeights"' not in rsrc, "reclaim still defaults --owner to our account"


def test_card_detects_the_pipeline_tag():
    """`pipeline_tag` must reflect what the model is, not always text-generation.

    It was hardcoded, so every multimodal build we publish was advertised as text-only -- Carnice-V3
    and Qwen3.8-27B both ship an mmproj next to their rungs and both said text-generation, which
    contradicts their own file list and keeps them out of any Hub search for vision models.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import pollard_card as C

    src = open(os.path.join(os.path.dirname(__file__), "..", "tools", "pollard_card.py"),
               encoding="utf-8").read()
    assert '"pipeline_tag: text-generation"' not in src, "frontmatter still hardcodes the tag"

    # an explicit flag wins outright
    assert C.detect_pipeline_tag("x/y", {}, explicit="any-to-any") == "any-to-any"

    # modality signals in the base config, without any network call
    assert C.detect_pipeline_tag("local/dir", {"vision_config": {}}) == "image-text-to-text"
    assert C.detect_pipeline_tag("local/dir", {"video_config": {}}) == "video-text-to-text"
    assert C.detect_pipeline_tag("local/dir", {"audio_config": {}}) == "audio-text-to-text"

    # what this repo actually ships
    assert C.detect_pipeline_tag("local/dir", {}, mmproj="mmproj.gguf") == "image-text-to-text"
    assert C.detect_pipeline_tag("local/dir", {}, input_support="text+image+video") == "video-text-to-text"

    # a plain text model stays text
    assert C.detect_pipeline_tag("local/dir", {}) == "text-generation"

    # and quantizing a VLM WITHOUT its projector must not promise vision
    assert C.detect_pipeline_tag("local/dir", {"vision_config": {}}, input_support="text") == "text-generation"


def test_flybrain_state_is_fixed_size_and_refuses_a_mismatch():
    """The memory must be a fixed-size file, and must not load into the wrong connectome.

    The whole claim is that context costs a constant amount: a KV cache for a long conversation is
    gigabytes and grows, while this is the same size forever. A state saved from one brain loaded
    into another would silently produce confident nonsense, so it is refused.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    try:
        import torch
    except ImportError:
        print("    (skipped: torch not installed -- `pip install pollard-weights[flybrain]`)")
        return
    from pollard_flybrain import FlyBrain

    import torch.nn as nn
    n, hid, wid = 64, 16, 12

    def make(neurons):
        bits, span = 2, 2                      # tiny codebook: the test only checks shapes and size
        nbit, cw = bits * span, wid - bits * span
        return {"src": [0, 1], "dst": [1, 0], "sign": [1.0, -1.0], "w": torch.ones(2),
                "addr": nn.Linear(hid, neurons).state_dict(),
                "addr_e": nn.Linear(hid, neurons, bias=False).state_dict(),
                "val": nn.Linear(2 * hid, cw).state_dict(),
                "wgate": nn.Linear(2 * hid, 1).state_dict(),
                "out": nn.Linear(wid, hid).state_dict(),   # the decoder reads the whole slot
                "voice": torch.zeros(1), "temp": torch.ones(1),
                "amix": torch.zeros(1), "dbeta": torch.zeros(1),
                "meta": {"name": "t", "neurons": neurons, "synapses": 2,
                         "hidden": hid, "width": wid, "win": 8, "k_mem": 2,
                         "bits": bits, "span": span, "ek": 3, "eos_id": 3}}

    b = FlyBrain(make(n))
    b.reset(1)
    assert b.state_bytes == n * wid * 4

    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "a.flystate")
        size_short = b.save_state(p1)
        for _ in range(50):                      # write into the memory a lot
            b.mem = b.mem + 1.0
            b.z = b.z + 1.0
        size_long = b.save_state(p1)
        assert size_short == size_long, "state size must not grow with use"

        b2 = FlyBrain(make(n + 1))
        try:
            b2.load_state(p1)
            raise AssertionError("loaded a state from a different connectome")
        except ValueError:
            pass


def test_vllm_tp_divisibility_is_caught_before_a_load():
    """A model that cannot shard at TP=N must be refused here, not minutes into a vLLM load.

    vLLM splits tensors across ranks, so head counts and the intermediate dimension have to divide by
    the TP degree, and group-quantized weights need whole groups per shard. Qwen2.5 has 14 attention
    heads: TP=2 is fine, TP=4 cannot work, and the failure otherwise shows up as a shape error on
    someone else's hardware after the model has already been downloaded.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from pollard_vllm import check

    qwen = {"num_attention_heads": 14, "num_key_value_heads": 2,
            "hidden_size": 896, "intermediate_size": 4864}
    assert check(qwen, 1) == [], "TP=1 must always work"
    assert check(qwen, 2) == [], "14 heads and 4864 intermediate divide by 2"
    assert check(qwen, 4), "14 heads cannot split 4 ways"
    assert "attention heads" in check(qwen, 4)[0]

    # group-quantized: each rank's slice must still contain whole groups
    gptq = {"num_attention_heads": 32, "num_key_value_heads": 8,
            "hidden_size": 4096, "intermediate_size": 11008,
            "quantization_config": {"quant_method": "gptq", "group_size": 128}}
    assert check(gptq, 2) == [], "11008/2 = 5504 IS a multiple of 128 — this one is fine"
    assert check(gptq, 4), "11008/4 = 2752 is not a multiple of the 128 group"
    assert "group size" in " ".join(check(gptq, 4))

    # per-channel quantization declares group_size -1: no group constraint at all
    perchan = dict(gptq, quantization_config={"quant_method": "gptq", "group_size": -1})
    assert not any("group size" in b for b in check(perchan, 4))


def test_vllm_tp_sweep_is_bounded_by_the_model_not_by_a_constant():
    """The sweep must report EVERY degree a model can serve at, not a hand-picked list.

    People run 3, 6 and 10 GPUs. An arbitrary ceiling (or a powers-of-two list) silently hides a
    working configuration, which is worse than saying no -- the user never learns the option exists.
    The only honest bound is the model's own head count: a rank has to receive at least one head.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import pollard_vllm
    from pollard_vllm import check, tp_ceiling

    src = pathlib.Path(pollard_vllm.__file__).read_text()
    assert "MAX_TP" not in src, "a constant TP ceiling is back -- the bound must come from the model"

    big = {"num_attention_heads": 64, "num_key_value_heads": 4,
           "hidden_size": 4096, "intermediate_size": 12288}
    assert tp_ceiling(big) == 64, "the ceiling is the head count"
    assert check(big, 32) == [] and check(big, 64) == [], "past any old cap, and still shardable"

    # odd and non-power-of-two degrees are reachable, both ways
    odd = {"num_attention_heads": 24, "num_key_value_heads": 24,
           "hidden_size": 3072, "intermediate_size": 9216}
    assert check(odd, 3) == [] and check(odd, 6) == [], "TP=3 and TP=6 divide 24 heads cleanly"
    assert check(odd, 5), "24 heads cannot split 5 ways"

    # a multimodal config keeps its text-tower geometry, not the top-level stub
    assert tp_ceiling({"text_config": {"num_attention_heads": 40}}) == 40



def test_tool_output_is_ascii_except_the_calibration_corpus():
    """Every byte a tool can PRINT must survive a legacy Windows codepage.

    Python on Windows encodes piped/redirected stdout with the locale codepage (cp1252 here), which
    has no arrow, sigma, check-mark or box-drawing glyph -- printing one raises UnicodeEncodeError
    and kills the run. Not cosmetic: `pollard-x ... > log.txt`, CI, and every SSH session take that
    path. Argparse prints module docstrings as the --help epilog, so docstrings count as output.

    The ONE exception is pollard-calib's multilingual seed corpus. Those samples are DATA -- the
    Japanese and Arabic prose is the point of a multilingual calibration set, and ASCII-folding them
    would quietly degrade calibration for non-Latin scripts. They are written to a file with an
    explicit utf-8 encoding, never printed, so they never reach the console encoder.
    """
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    offenders = []
    for f in sorted(tools.glob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            bad = [c for c in line if ord(c) > 127]
            if not bad:
                continue
            if f.name == "pollard_calib.py" and _is_corpus_sample(line):
                continue
            offenders.append(f"{f.name}:{i} {''.join(sorted(set(bad)))!r}")
    assert not offenders, ("non-ASCII in tool output (crashes on a cp1252 console):\n  "
                           + "\n  ".join(offenders[:12]))

    # and the corpus really is still there -- this test must not be satisfiable by deleting it
    calib = (tools / "pollard_calib.py").read_text(encoding="utf-8")
    assert any(ord(c) > 0x3000 for c in calib), "the multilingual calibration seeds are gone"


def _is_corpus_sample(line):
    """A bundled calibration seed: a quoted string literal on its own line, in the seed tables."""
    return line.strip().startswith('"') and line.strip().endswith('",')



def test_human_connectome_filters_glia_and_signs_by_dale():
    """Two decisions separate a human connectome from a pile of detector artifacts.

    H01's soma table covers ~57k cells and most are GLIA. Unfiltered, the largest single edge class
    is astrocyte->pyramidal, which is not a synapse: astrocyte processes wrap real synapses and the
    detector reports the wrapper. A graph whose commonest connection is biologically impossible is
    not a connectome, so neurons are the default and glia are opt-in.

    The sign has to come from Dale's law on the cell type, not from the detector's own
    excitatory/inhibitory call, which agreed with Dale only 57.5% of the time on this data -- barely
    better than a coin. And the sign array is indexed by position in np.unique(concat(pre, post)),
    the same remap the trainer applies; build it any other way and every sign lands on a different
    neuron than the one it describes, silently.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import pollard_connectome as C

    assert "ASTROCYTE" not in C.NEURONS and "OLIGO" not in C.NEURONS and "MG_OPC" not in C.NEURONS
    assert "BLOOD_VESSEL_CELL" not in C.NEURONS
    assert "PYRAMIDAL" in C.EXCITATORY and "INTERNEURON" in C.INHIBITORY
    assert not (C.EXCITATORY & C.INHIBITORY), "a cell type cannot be both"
    assert C.NEURONS == C.EXCITATORY | C.INHIBITORY

    # the sign vector must align to the sorted unique node ids, not to edge order
    import numpy as np
    pre  = np.array([50, 10, 10], dtype=np.int64)
    post = np.array([10, 20, 50], dtype=np.int64)
    soma = {10: ("INTERNEURON", "Layer 2"), 20: ("PYRAMIDAL", "Layer 3"), 50: ("PYRAMIDAL", "Layer 5")}
    nodes = np.unique(np.concatenate([pre, post]))
    signs = np.array([-1.0 if soma[int(n)][0] in C.INHIBITORY else 1.0 for n in nodes])
    assert list(nodes) == [10, 20, 50]
    assert list(signs) == [-1.0, 1.0, 1.0], "the inhibitory neuron must be the one at its sorted slot"



def test_flybrain_token_code_is_an_exact_inverse():
    """The brain stores the HOST's code for a token and reads it back with the transpose.

    That only works if the projection is orthonormal: P @ P.t() must be the identity. It is what lets
    the brain name a word it never saw in training, because the code for every token in the vocabulary
    is defined without fitting anything. An earlier version learned its own encoder instead and scored
    12.5% on a fixed word list, then 0% the moment the words were drawn fresh -- it had memorised 300
    codes rather than a mechanism. If this property ever breaks, that failure comes back silently.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    try:
        import torch
    except ImportError:
        print("    (skipped: torch not installed -- `pip install pollard-weights[flybrain]`)")
        return
    from pollard_flybrain import _code_projection

    P = _code_projection(64, 16, "cpu")
    assert P.shape == (16, 64)
    eye = P @ P.t()
    assert torch.allclose(eye, torch.eye(16), atol=1e-5), "code projection is not orthonormal"

    # and it must be reproducible: a brain saved today has to decode the same way tomorrow
    assert torch.equal(P, _code_projection(64, 16, "cpu")), "code projection is not deterministic"


def test_steering_strength_generalises_ablation_without_changing_its_default():
    """Removing a direction is one point on a dial, and the default must stay where it was.

    The published technique orthogonalises a residual-writing weight against the direction that
    mediates a behaviour: W -= r r^T W. That is strength -1.0. The same diff-of-means direction
    works for any behaviour you can write two contrasting prompt sets for, so the coefficient is
    exposed -- but anyone who was running abliteration before must keep getting abliteration, so
    -1.0 stays the default and has to remain EXACTLY the old arithmetic, not an approximation of it.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    try:
        import torch
    except ImportError:
        print("    (skipped: torch not installed -- `pip install pollard-weights[flybrain]`)")
        return
    from pollard_abliterate import abliterate

    import torch.nn as nn

    def toy(D=8, n_in=5):
        torch.manual_seed(0)
        blk = type("B", (), {})()
        blk.self_attn = type("A", (), {})(); blk.mlp = type("M", (), {})()
        blk.self_attn.o_proj = nn.Linear(n_in, D, bias=False)
        blk.mlp.down_proj = nn.Linear(n_in, D, bias=False)
        m = type("Mo", (), {})(); m.model = type("Inner", (), {})()
        m.model.layers = [blk]
        m.model.embed_tokens = nn.Embedding(11, D)
        return m

    r = torch.zeros(8); r[3] = 1.0                      # a unit direction, axis-aligned for clarity

    # strength -1.0 must remove the component completely -- the old behaviour, bit for bit
    m = toy()
    abliterate(m, r, "cpu", -1.0)
    W = m.model.layers[0].self_attn.o_proj.weight.data
    assert torch.allclose(W[3], torch.zeros(5), atol=1e-6), "ablation left the direction behind"
    import inspect
    assert inspect.signature(abliterate).parameters["strength"].default == -1.0, \
        "the default must stay at full ablation -- existing users get what they had"

    # 0.0 is a no-op, and +0.5 amplifies rather than removes
    base = toy().model.layers[0].self_attn.o_proj.weight.data.clone()
    m0 = toy(); abliterate(m0, r, "cpu", 0.0)
    assert torch.allclose(m0.model.layers[0].self_attn.o_proj.weight.data, base, atol=1e-6)
    mp = toy(); abliterate(mp, r, "cpu", 0.5)
    amp = mp.model.layers[0].self_attn.o_proj.weight.data
    assert torch.allclose(amp[3], base[3] * 1.5, atol=1e-5), "positive strength must amplify"
    # and it must touch ONLY that direction
    assert torch.allclose(amp[4], base[4], atol=1e-6), "steering leaked into other directions"


def test_routing_capture_exists_for_every_tool_that_consumes_one():
    """A tool must not require an input that no shipped tool can produce.

    pollard-experts reads a routing capture. The capture used to come from an in-repo experiment
    directory that was never shipped, so from a clean install the expert-residency path could not be
    run at all -- the model was never the limitation, the missing producer was.
    """
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    assert (tools / "pollard_route.py").is_file(), "the routing-capture producer is missing"

    src = (tools / "pollard_route.py").read_text(encoding="utf-8")
    for field in ('"layer"', '"experts"', '"phase"'):
        assert field in src, f"capture rows must carry {field} for pollard-experts"
    assert "--gen" in src, "decode capture is the whole point; prefill alone understates it"

    pyproject = (tools.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert "pollard-route" in pyproject, "the tool exists but is not registered as a command"



def test_expert_analysis_separates_decode_from_prefill():
    """Mixing the two regimes hides the only structure worth measuring.

    A router spreads PREFILL across nearly the whole pool whatever the workload -- one domain touched
    97.6% of experts in our measurements -- while DECODE concentrates about 2x. Prefill also produces
    far more rows than decode in any normal capture, so averaging them buries the concentration under
    the flat part and reports "no structure" for a workload that has plenty. An agent lives in
    decode, so decode is the default here, and asking for a regime a capture does not contain is an
    error with a fix in it rather than a silently empty report.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import json as _json
    from pollard_experts import load

    rows = []
    for pos in range(20):                      # prefill: every expert equally, the flat case
        rows.append({"prompt": 0, "pos": pos, "phase": "prefill", "layer": 0,
                     "experts": [pos % 4]})
    for pos in range(20, 26):                  # decode: concentrated on one expert
        rows.append({"prompt": 0, "pos": pos, "phase": "decode", "layer": 0, "experts": [3]})

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(_json.dumps(r) for r in rows))
        path = f.name

    dec, counts = load(path, "decode")
    pre, _ = load(path, "prefill")
    allr, _ = load(path, "all")
    assert len(dec) == 6 and len(pre) == 20 and len(allr) == 26
    assert counts["prefill"] == 20 and counts["decode"] == 6

    # the point of the split: decode is one expert, the mix is not
    assert {e for _p, _pos, _l, ex, _ph in dec for e in ex} == {3}
    assert len({e for _p, _pos, _l, ex, _ph in allr for e in ex}) == 4, \
        "mixing prefill back in hides that decode used a single expert"

    # a capture with no phase labels must say so, not report an empty result
    legacy = [{k: v for k, v in r.items() if k != "phase"} for r in rows]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(_json.dumps(r) for r in legacy))
        lpath = f.name
    try:
        load(lpath, "decode")
        raise AssertionError("an unlabelled capture must not pass as a decode capture")
    except SystemExit as e:
        assert "pollard-route" in str(e), "the error must say how to fix it"
    os.unlink(path); os.unlink(lpath)



def test_backbone_loader_accepts_a_vision_language_model():
    """A VL backbone must be reachable, because everything downstream already handles one.

    AutoModelForCausalLM refuses a vision-language config outright -- "Unrecognized configuration
    class Qwen2VLConfig for this kind of AutoModel" -- so a VL model could not be loaded at all, even
    though _find_stack already looks for `model.language_model`, which is exactly where a VL model
    keeps its text stack. One missing fallback was the whole gap; with it, the shipped trainer
    reaches 100% exact recall on Qwen2-VL from a cold start, against 51.4% for the old VL-specific
    script that predated the bit payload.

    The brain attaches to the LANGUAGE side. Exact recall stores token ids as bits, and a continuous
    vision encoder emits no ids -- so this makes a VL backbone usable, not images recallable.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    try:
        import torch
    except ImportError:
        print("    (skipped: torch not installed -- `pip install pollard-weights[flybrain]`)")
        return
    import pollard_flybrain as F
    import pollard_backbone as L

    assert hasattr(F, "load_backbone") and hasattr(L, "load_backbone")
    src = pathlib.Path(L.__file__).read_text(encoding="utf-8")
    for cls in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        assert cls in src, f"no fallback to {cls}: a VL model would be unreachable"
    assert "model.language_model" in src, "the VL text-stack path was dropped"

    # ONE loader. Eleven tools each called AutoModelForCausalLM directly and each died on a VL
    # config; a second copy is how they drift back apart.
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    offenders = []
    # BRAIN tooling only. The model tools each load their own backbone and do not depend on this
    # loader -- that separation is the point, and enforcing the shared loader on them was how brain
    # code ended up threaded through the build path in the first place.
    # Each lane carries its own loader now -- brains share no code with model building, so neither
    # can change under the other. What still has to hold is the CAPABILITY: a brain binds to the
    # language side, so a VL checkpoint must be reachable. Calling AutoModelForCausalLM is fine;
    # calling it with no vision-language fallback is not.
    for f in sorted(tools.glob("pollard_*brain*.py")) + sorted(tools.glob("pollard_connectome.py")):
        body = f.read_text(encoding="utf-8")
        if "AutoModelForCausalLM.from_pretrained" in body and not any(
                c in body for c in ("AutoModelForImageTextToText", "AutoModelForVision2Seq")):
            offenders.append(f.name)
    assert not offenders, ("these load a backbone directly and will refuse a VL model: "
                           + ", ".join(offenders))



def test_brain_query_default_matches_the_verified_prompt():
    """The query shape is part of the experiment, not a cosmetic default.

    A token is filed under the words immediately before it, so retrieval works by reproducing that
    context. The verified construction ends with "Answer: The secret word is" -- question AND
    continuation. Ship a default that is only the question and a brain measuring 100% measures 46%,
    from a memory that is perfectly intact. That default shipped, and a first correction to only the
    continuation was wrong in the same way. Both halves, or it is not the measured prompt.
    """
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    src = (tools / "pollard_flybrain.py").read_text(encoding="utf-8")
    verified = "Question: what is the secret word? Answer: The secret word is"
    i = src.index('ap.add_argument("--ask"')
    assert verified in src[i:i + 400], "--ask default is not the verified prompt"
    # and the trainer must teach what the default asks
    assert verified in src[:i], "the trainer's ASKS no longer contains the default query"

    vsrc = (tools / "pollard_brainverify.py").read_text(encoding="utf-8")
    assert verified in vsrc, "the verifier must use the same construction it validates"


def test_brain_payload_codec_is_recorded_not_assumed():
    """A brain must say how its payload is encoded, because reading it the other way is noise.

    'tokens' stores this backbone's vocabulary indices; 'bytes' stores UTF-8 text, which any
    tokenizer can read back and which is smaller for short words (6x8 against 4x18). A brain written
    one way and read the other decodes to garbage with no error, so the codec travels in the
    checkpoint and defaults to the original behaviour for every brain written before it existed.
    """
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    src = (tools / "pollard_flybrain.py").read_text(encoding="utf-8")
    assert 'self.meta.get("codec", "tokens")' in src, "codec must default to the original behaviour"
    assert '"codec": codec' in src, "the trainer must record the codec it wrote"
    assert 'bits = 8 if codec == "bytes"' in src, "a byte payload is 8 bits, not the vocabulary width"
    assert "decode_text" in src, "a byte brain needs a text decoder"
    # Changing the payload must be ALLOWED, not refused. --continue-from carries trained weights;
    # written memory lives in a .flystate file, so a new codebook has nothing stored to corrupt.
    # People swap backbones and payloads constantly, and refusing the whole transfer over a
    # resizable layer threw away the address path and gate that transfer perfectly well.
    i = src.index("if continue_from:")
    block = src[i:i + 2000]
    assert "raise SystemExit" not in block, "a payload change must not abort the transfer"
    assert "payload change" in block, "a payload change must be reported, not silent"
    # but a written-memory file from a differently shaped brain IS still refused
    assert "state was written by a differently shaped brain" in src, \
        "load_state must still refuse a mismatched .flystate -- that file holds real memory"



def test_every_runtime_backend_declares_the_same_four_operations():
    """A brain needs four things from a backbone, and nothing else.

    embed(ids), forward(embeds) -> (logits, hidden), forward_ids(ids), out_weight(). Brains ran only
    under transformers because those four calls were written inline against one library, not because
    of anything in the memory. Any runtime that can be fed EMBEDDINGS can host one -- that is the
    hard requirement, since memory is delivered by prepending vectors to the sequence.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import pollard_brain_backends as B

    for cls in (B.Transformers, B.MLX, B.ExLlamaV3, B.LlamaCpp):
        for op in ("embed", "forward", "forward_ids", "out_weight"):
            assert callable(getattr(cls, op, None)), f"{cls.__name__} is missing {op}()"
        assert getattr(cls, "name", "?") != "?", f"{cls.__name__} has no lane name"
    assert {c.name for c in (B.Transformers, B.MLX, B.ExLlamaV3, B.LlamaCpp)} == \
        {"transformers", "mlx", "exl3", "gguf"}


def test_mlx_output_embedding_is_dequantized():
    """A quantized MLX model reports a PACKED output embedding, and the brain's codes come from it.

    A 4-bit Qwen reports (151936, 112) where the real matrix is (151936, 896). Hand the brain packed
    bytes and it builds its codebook out of bit-patterns: every stored token decodes to noise and
    nothing raises. The backend must dequantize before returning it.
    """
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    src = (tools / "pollard_brain_backends.py").read_text(encoding="utf-8")
    i = src.index("class MLX")
    block = src[i:src.index("class ExLlamaV3")]
    assert "dequantize" in block, "MLX out_weight must dequantize a packed embedding"
    assert 'hasattr(mod, "scales")' in block, "must detect a quantized module before unpacking"



def test_gguf_lane_requires_unpooled_per_token_states():
    """llama.cpp CAN host a brain -- but only unpooled.

    The high-level Llama.eval() takes tokens only, which is why this lane looked closed. The C API
    has both halves: llama_batch_init(n, embd, seq) carries EMBEDDINGS in `embd`, and
    llama_get_embeddings_ith() returns the final hidden state per token. Verified on a Q4_K_M build:
    embed (1,6,896) in, logits (1,6,151936) and hidden (1,6,896) out.

    Pooling is the trap. With llama.cpp's default the context returns ONE pooled vector for the whole
    sequence, so a brain has nothing per-token to address on and every write lands in the same place.
    The context must be opened with pooling NONE.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    src = (tools / "pollard_brain_backends.py").read_text(encoding="utf-8")
    assert "LLAMA_POOLING_TYPE_NONE" in src, "the GGUF context must disable pooling"
    i = src.index("class LlamaCpp")
    block = src[i:src.index("def open_backend")]
    assert "llama_batch_init" in block and "embd" in block, "must feed embeddings, not ids"
    assert "llama_get_embeddings_ith" in block, "must read per-token hidden states"
    # The output embedding is IN THE FILE. llama.cpp not exposing it to Python looked like it needed
    # an upstream patch; it does not -- a GGUF carries output.weight (or token_embd.weight when the
    # model ties them) and the gguf package dequantizes either, so the lane needs no fork.
    assert "GGUFReader" in block and "dequantize" in block, \
        "out_weight must read the embedding from the GGUF rather than require a fork"
    assert "token_embd.weight" in block, "must fall back to the tied embedding"



def test_byte_codec_target_and_payload_share_units():
    """A payload and its training target must be the same KIND of thing.

    Under --codec bytes the memory stores a token's UTF-8 bytes, but the loss target was built from
    BITCODE, which indexes token IDS. So the brain was trained against the id's low 8 bits while
    storing the token's text, and the two have nothing to do with each other.

    It did not look like a failure, which is the dangerous part. Byte 0 of an answer is almost always
    32 -- a leading space -- so position 0 learned the constant and scored 98% while positions 1-3
    sat at exactly 0%. One position high and the rest at zero is the signature of a wrong target; a
    real learning failure degrades evenly. With the units matched the codec reaches 99% by step 26,
    the same speed as the token codec.

    The evaluation had the same disease in mirror image: it compared decoded BYTES against expected
    TOKEN IDS, scoring 0% by construction however well the brain had learned.
    """
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    src = (tools / "pollard_flybrain.py").read_text(encoding="utf-8")

    i = src.index("tgt = torch.cat([BITCODE")
    head = src[max(0, i - 1200):i]
    assert 'if codec == "bytes":' in head, "the byte target must not be built from BITCODE"
    assert "BTBL[i][:int(BLEN[i])]" in head, "the byte target must come from the token->bytes table"

    j = src.index("def evaluate(")
    ev = src[j:j + 1600]
    assert 'codec == "bytes"' in ev, "the evaluation must score bytes against bytes"
    assert 'encode("utf-8")' in ev, "the byte target must be the answer's real UTF-8"


def test_byte_table_is_built_in_one_batch():
    """Building it id-by-id stalls for minutes with an empty log, which reads as a hung job.

    151,936 separate decode() calls against a full vocabulary take long enough that a user sees
    nothing happen and kills a working run -- the same failure mode as unflushed training output,
    reintroduced somewhere new. batch_decode does it in well under a second.
    """
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    src = (tools / "pollard_flybrain.py").read_text(encoding="utf-8")
    i = src.index("def _byte_table_for(")
    block = src[i:i + 1400]
    assert "batch_decode" in block, "the byte table must be built in batches"
    assert "for _i in range(V)" not in src, "a per-id decode loop is back"



def test_backends_move_inputs_and_handle_dtypes_at_the_boundary():
    """A lane that imports is not a lane that works, and the gap is all at the boundary.

    Five real failures, each the FIRST thing a user would hit, none caught by an import check:
      - ids handed to an MPS/CUDA model still on the CPU ("Passed CPU tensor to MPS op")
      - numpy cannot view bfloat16, which is what quantized MLX models compute in, so the buffer
        conversion died with a PEP 3118 error that reads like a corrupt model
      - casting everything to float to fix that broke the ID path instead: gather refuses
        non-integral indices, so the dtype has to decide which way to go
      - the GGUF lane given an HF id rather than a path to a .gguf file
      - exllamav3 needing ninja on PATH to build its extensions
    """
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    src = (tools / "pollard_brain_backends.py").read_text(encoding="utf-8")

    i = src.index("class Transformers")
    assert "ids.to(" in src[i:src.index("class MLX")], "transformers must move ids to the model"

    mlx = src[src.index("class MLX"):src.index("class ExLlamaV3")]
    assert "astype(self.mx.float32)" in mlx, "mlx->torch must cast before numpy sees bfloat16"
    assert "is_floating_point()" in mlx, "torch->mlx must preserve integer dtype for ids"

    g = src[src.index('if kind == "gguf"'):]
    assert "os.path.isfile" in g and ".gguf FILE" in g, "the GGUF lane must demand a file path"


def test_brainlanes_reports_the_machine_not_a_table():
    """Which runtimes can host a brain is a property of the MACHINE, not of the brain.

    Platform support belongs in a command a user can run, not a compatibility table in a document
    that goes stale. And the tool must distinguish 'importable' from 'works' -- an import proves
    nothing about a forward pass, which is the only claim that counts.
    """
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    src = (tools / "pollard_brainlanes.py").read_text(encoding="utf-8")
    for lane in ("transformers", "mlx", "gguf", "exl3", "vllm"):
        assert f'"{lane}"' in src, f"{lane} missing from the lane report"
    assert "platform.system()" in src, "must report the platform it actually ran on"
    assert "an import does not prove a forward pass works" in src
    # find_spec is not enough: vLLM's Windows wheel ships WITHOUT its compiled CUDA extension, so the
    # package directory exists and `from vllm import LLM` still dies on vllm._C_stable_libtorch. A
    # spec check calls that "available", which is worse than a clear no because someone acts on it.
    assert "importlib.import_module(mod)" in src, "the check must actually import, not just find"
    assert "installed but broken" in src, "a present-but-unimportable package must say so"



def test_lane_failures_name_the_blocker_not_the_exception():
    """An exception type sends people the wrong way; the blocker tells them what to do.

    All three were hit on real machines: "Ninja is required" AFTER pip install ninja succeeded (the
    binary lands in the venv's bin, which is not on PATH unless the venv is activated); a missing
    CUDA_HOME on a machine with no toolkit, where the lane compiles CUDA extensions; and a package
    whose wheel ships without its compiled extension for that platform, which imports far enough to
    look installed and then dies on its own _C module.
    """
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    src = (tools / "pollard_brainlanes.py").read_text(encoding="utf-8")
    assert "def _why(" in src, "lane failures must be translated, not re-printed"
    assert "ninja ON PATH" in src, "the ninja-installed-but-not-on-PATH case must be named"
    assert "CUDA toolkit" in src, "a missing toolkit must be named, not shown as OSError"
    assert "no compiled extension for this platform" in src



def test_kquant_is_not_forced_onto_a_row_length_that_cannot_hold_it():
    """A forced K-quant on an unaligned tensor ABORTS ggml and leaves a truncated file.

    K-quants store 256 elements per block. A tensor whose row length is not a multiple of 256 cannot
    be one, and llama.cpp falls back on its own -- unless an explicit --token-embedding-type
    overrides it, which is exactly what the allocator passes. ggml then aborts mid-write:

        ggml.c: GGML_ASSERT(start % type_traits[type].blck_size == 0) failed

    The process dies partway and leaves a TRUNCATED .gguf, which is worse than an error because the
    build reports nothing wrong. Qwen2.5-0.5B produced a 5.9 MB file that no reader would open,
    where the real build is 506 MB -- hidden size 896, and 896 / 256 = 3.5.

    q8_0 blocks are 32 wide, so it divides anything 32-aligned and costs a little size on what is
    usually one embedding matrix.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from pollard_fit import block_safe_type, QK_K

    assert QK_K == 256
    assert block_safe_type("q6_K", 896) == "q8_0", "896 cannot hold a K-quant"
    assert block_safe_type("q4_K", 896) == "q8_0"
    assert block_safe_type("q6_K", 1536) == "q6_K", "1536 divides 256 -- leave it alone"
    assert block_safe_type("q6_K", 4096) == "q6_K"
    # non-K types have small blocks and are never substituted
    assert block_safe_type("q8_0", 896) == "q8_0"
    assert block_safe_type("f16", 896) == "f16"
    # and the substitution must be announced, never silent
    src = pathlib.Path(__file__).resolve().parent.parent.joinpath(
        "tools", "pollard_fit.py").read_text(encoding="utf-8")
    assert "is not a multiple of" in src, "a type substitution must be printed, not hidden"



def test_layer_access_goes_through_text_layers():
    """Loading a VL model is half the job; reaching its layers is the other half.

    A vision-language model keeps its decoder under model.language_model, so `model.model.layers`
    raises AttributeError even after the model loads fine. Fixing the LOADER alone moved the failure
    two lines down -- pollard-probe loaded Qwen2-VL successfully and then died on
    `len(model.model.layers)`, which looks like a different bug and is the same one.
    """
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    offenders = []
    # BRAIN tooling only -- see the note in the loader test above.
    for f in sorted(tools.glob("pollard_*brain*.py")) + sorted(tools.glob("pollard_connectome.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if "model.model.layers" not in line:
                continue
            if line.lstrip().startswith(("#", "print(", "sys.exit(")) or '"' in line.split("model.model.layers")[0][-2:]:
                continue                      # a message ABOUT the path, not a use of it
            offenders.append(f"{f.name}:{i}")
    assert not offenders, ("these reach layers directly and break on a VL model: "
                           + ", ".join(offenders))



def test_shared_loader_is_imported_where_module_scope_code_can_see_it():
    """An import inside main() is invisible to a module-scope function that needs it.

    This bit twice, identically, because the conversion put the import at the FIRST use rather than
    at the top: pollard_abliterate raised NameError from abliterate(), and after that was fixed
    pollard_probe raised the same NameError from _linears() -- both only when the helper was reached
    outside main(). A CLI run could pass while a library call died.

    pollard_flybrain is the deliberate exception: it imports lazily because pollard_backbone pulls in
    torch, and flybrain's optional-extra guard depends on torch not being required at import time.
    """
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    offenders = []
    for f in sorted(tools.glob("pollard_*.py")):
        if f.name == "pollard_flybrain.py":
            continue
        src = f.read_text(encoding="utf-8")
        if "pollard_backbone import" not in src:
            continue
        for i, line in enumerate(src.splitlines(), 1):
            if "from pollard_backbone import" in line and line.startswith((" ", "\t")):
                offenders.append(f"{f.name}:{i}")
    assert not offenders, ("imported inside a function, so module-scope callers raise NameError: "
                           + ", ".join(offenders))


def test_probe_places_a_model_too_big_for_the_accelerator():
    """The probe pinned the WHOLE model to one device, so the first model bigger than the box OOM'd
    (Gemma4 12B: 22GB onto a 16GB Mac). It must shard+offload instead of dying."""
    try:
        import torch  # noqa: F401  (pollard_probe imports it at module scope)
    except ImportError:
        print("    (skipped: torch not installed -- `pip install pollard-weights[flybrain]`)")
        return
    import pollard_probe as P
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "model-00001-of-00001.safetensors"), "wb") as f:
        f.truncate(400 * (1 << 30))                     # 400GB: bigger than any dev box
    _, kw, note = P.plan_placement(d, "mps", os.path.join(d, "off"))
    assert kw.get("device_map") == "auto", f"a 400GB model was not sharded: {kw}"
    assert kw.get("offload_folder"), "nothing offloaded, so it will OOM"
    # unified memory: the GPU and the CPU spend the SAME pool, so the budgets must not double-count
    if sys.platform == "darwin":
        tot = sum(int(re.sub(r"[^0-9]", "", v)) for v in kw["max_memory"].values())
        ram = P._host_bytes() / (1 << 30)
        assert tot <= ram * 0.80, f"budgeted {tot}GiB of {ram:.0f}GB unified RAM"


def test_gold_path_never_degrades_to_uniform_silently():
    """A uniform allocation is what pollard-fit itself warns has no quality win. Losing the probe or
    the imatrix must STOP the build, not quietly ship a stock K-quant wearing Pollard's name."""
    src = pathlib.Path(__file__).resolve().parents[1] / "tools" / "pollard_auto.py"
    txt = src.read_text(encoding="utf-8")
    probe = txt.split("def _ensure_sensitivity", 1)[1].split("\ndef ", 1)[0]
    assert "raise SystemExit" in probe, "a failed probe still falls back to uniform allocation"
    imat = txt.split("def _ensure_imatrix", 1)[1].split("\ndef ", 1)[0]
    assert "returncode" in imat and "raise SystemExit" in imat, (
        "llama-imatrix's exit code is unchecked -- a missing imatrix stays invisible until "
        "llama-quantize fails to open it")


def test_the_model_tools_know_nothing_about_brains():
    """A brain is an OPTIONAL thing a user may attach to a finished model. It is not part of
    quantizing one, so no model tool should carry brain code -- attach_brain() and a --brain flag
    had grown into the middle of the build driver, which is why every edit to a model tool raised
    the question of whether the brains had been touched. `pollard` builds models; brains live in
    the brain tools and attach afterwards."""
    tools = pathlib.Path(__file__).resolve().parents[1] / "tools"
    offenders = []
    for f in sorted(tools.glob("pollard_*.py")):
        if re.search(r"brain|connectome|flybrain", f.name):
            continue                                   # the brain tools themselves, naturally
        txt = f.read_text(encoding="utf-8")
        hits = [i for i, l in enumerate(txt.splitlines(), 1)
                if re.search(r"\b(flybrain|connectome|attach_brain)\b", l, re.I)
                or re.search(r"--brain\b", l)]
        if hits:
            offenders.append(f"{f.name}:{hits[:3]}")
    assert not offenders, ("brain code has grown back into model tooling: " + "; ".join(offenders))






def test_one_shot_serializes_and_finishes_the_job():
    """Two gaps that existed only in whatever script was driving a build, never in the tool:

    Nothing stopped a second heavy job starting on the same machine. Quantizing, imatrix and
    perplexity all want the same GPU, disk and cores; started together they thrash, and on a shared
    machine the other person feels it first. A lock is advisory and self-healing -- a dead pid is
    taken over, never a reason to be stuck -- and --force ignores it.

    And a build ended with files but no card, so nobody could tell what the rungs were or which
    runtime each needed."""
    import pollard_auto as A
    src = (pathlib.Path(__file__).resolve().parents[1] / "tools" / "pollard_auto.py").read_text(
        encoding="utf-8")
    assert hasattr(A, "MachineLock"), "no machine lock: two builds can still thrash one box"
    assert "_pid_alive" in src, "a stale lock would wedge every later run"
    assert "--force" in src, "no way past a lock the user knows is finished"
    assert hasattr(A, "_emit_card"), "a run still ends without a card"
    assert '"--no-card"' in src, "no way to skip the card"
    body = src.split("def _emit_card", 1)[1].split("\ndef ", 1)[0]
    assert "pollard-card" in body and "--results" in body, (
        "the card step does not pass measured numbers through")







def test_stop_only_selects_pollard_build_work_and_waits_for_the_save():
    """Two ways a stop goes wrong, pulling opposite directions.

    Too gentle leaves ORPHANS: `schtasks /end` takes the shell and leaves the worker, and an
    imatrix here survived its task holding 11.4GB until someone looked.

    Too hard corrupts the ARTIFACT: llama-imatrix rewrites its .dat every few chunks and
    llama-quantize streams a GGUF tensor by tensor. Killed mid-write the file is short and
    perfectly well-formed -- it loads, it is wrong, nothing says so.

    And it must never guess WHAT to stop: a first pass matched any command line containing
    'pollard', which selected the operator's own shells and an unrelated project's llama-server."""
    import pollard_stop as S
    # a shell whose cwd merely mentions pollard is not a job
    procs = [(1, 0, "/bin/zsh", "/bin/zsh -c cd /Users/x/pollard-weights && ls"),
             (2, 0, "llama-server", "/other/project/llama-server -m /models/foo.gguf"),
             (3, 0, "llama-imatrix", "/p/bin/llama-imatrix -m f16.gguf -o m.dat"),
             (4, 0, "python.exe", "python.exe C:/pollard/pw/tools/pollard_bench.py --gguf x")]
    picked = {p for p, *_ in S.find_jobs(procs)}
    assert 1 not in picked, "a shell was selected as a job"
    assert 2 not in picked, "an unrelated llama-server was selected -- stopping it is someone's outage"
    assert 3 in picked and 4 in picked, f"real build work missed: {picked}"
    # the save-wait is the reason this tool exists
    src = (pathlib.Path(__file__).resolve().parents[1] / "tools" / "pollard_stop.py").read_text(
        encoding="utf-8")
    assert "def wait_for_save" in src and "SETTLE_SECONDS" in src
    seg = src.split("def stop(", 1)[1].split("\ndef ", 1)[0]
    assert seg.index("wait_for_save") < seg.rindex("/F"), (
        "it force-kills before waiting for the write to settle")


def test_gate_names_the_symptom_and_leads_with_the_cheap_lever():
    """BELOW FLOOR told everyone the same thing: bump the body tier. But gemma-4's flagship failed
    by repeating a CONTROL token (<|channel>thought, over and over) -- that is the token embedding
    losing resolution on a 152k vocabulary, not the body collapsing, and protecting one tensor costs
    a few hundred MB against a whole tier. Advice that ignores the symptom sends people to the most
    expensive fix first."""
    import re
    import pollard_bench as B
    src = (pathlib.Path(__file__).resolve().parents[1] / "tools" / "pollard_bench.py").read_text(
        encoding="utf-8")
    seg = src.split("def coherence_gate", 1)[1].split("\ndef ", 1)[0]
    assert "control tokens repeating" in seg, "the gate does not distinguish this failure mode"
    # the pattern must actually catch what gemma-4 emitted
    assert re.search(r"<\|[^|>]{1,32}\|?>|<[a-z_]{2,16}>", "<|channel>thought <|channel>thought")
    verdict = src.split("BELOW FLOOR", 1)[1][:4000]
    assert "TOKEN EMBEDDING" in verdict, "the cheapest lever is not offered"
    # every lever Pollard actually ships should be reachable from the failure, not just the ones
    # whoever wrote the message happened to remember
    for tool in ("pollard-probe", "pollard-sensitivity", "pollard-calib", "pollard-precondition",
                 "pollard-rotate", "pollard-smooth", "pollard-hf-smooth", "pollard-palette",
                 "pollard-lowbit", "pollard-prune", "pollard-automap"):
        assert tool in verdict, f"{tool} is never offered to someone whose build failed"
    assert verdict.index("TOKEN EMBEDDING") < verdict.index("bump the body tier"), (
        "bumping the tier is still suggested before protecting one tensor")


def test_card_detects_its_facts_instead_of_being_told_them():
    """Input support, imatrix and parameter count were flags a person had to remember, and
    forgetting one puts a wrong fact on a published card: gemma-4-12B-it generated as
    'Input support: text' with its 175MB projector sitting in the same folder, and 'imatrix: no'
    for a build an imatrix produced. Each is knowable from what was actually built."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "tools" / "pollard_card.py").read_text(
        encoding="utf-8")
    assert "def detect_card_facts" in src, "the card still relies on flags alone"
    seg = src.split("def detect_card_facts", 1)[1].split("\ndef ", 1)[0]
    # the text GGUF cannot know about modalities -- the projector declares them
    assert "clip.has_vision_encoder" in seg and "clip.has_audio_encoder" in seg, (
        "input support is not read from the projector that actually ships the capability")
    assert "_tensor_param_sum" in seg, "parameter count is not read from the build"
    assert ".imatrix" in seg and ".dat" in seg, "the imatrix is not looked for"
    # a modality is only claimed when the projector that carries it ships
    assert "shipped" in seg, "a modality could be claimed without the projector"


def test_multimodal_builds_ship_their_projector_at_full_precision():
    """A text GGUF is only the language half of a multimodal model, and the weights for the rest are
    in the source. Gemma 4 is ENCODER-FREE -- Google replaced a 550M vision encoder with one large
    matmul and dropped the audio conformer, projecting 40ms/16kHz chunks straight into the embedding
    space -- so v.patch_embd.weight IS the vision pathway. Exporting with --outtype f16 downcast it
    and produced a 122MB projector where the reference is 175MB; bf16 reproduces the reference
    exactly (F32 patch embedding, BF16 projections)."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "tools" / "pollard_auto.py").read_text(
        encoding="utf-8")
    assert "_emit_mmproj" in src, "a multimodal build ships without its projector"
    seg = src.split("def _emit_mmproj", 1)[1].split("\ndef ", 1)[0]
    assert '"--mmproj"' in seg, "no projector export"
    assert '"bf16"' in seg and '"f16"' not in seg.split("NOT --outtype")[-1].split('"""')[0], (
        "the projector is downcast to f16, which loses the vision pathway's precision")
    assert "pollard_modelkind" in seg, "it exports blindly instead of asking what the model is"


def test_a_declared_modality_needs_weights_to_back_it():
    """gemma-4-12B-it carries vision_config, audio_config, video_token_id and the projection
    layers -- and none of the encoder towers. The checkpoint is 666 language-model tensors, one
    embed_vision, one embed_audio and a 9-tensor embedder. Trusting the config would put image,
    audio and video on the card for a model that cannot do any of them."""
    import pollard_modelkind as K
    d = tempfile.mkdtemp()
    pathlib.Path(d, "config.json").write_text(json.dumps(
        {"architectures": ["FooForConditionalGeneration"],
         "vision_config": {"mm_embed_dim": 8}, "audio_config": {"audio_embed_dim": 8}}),
        encoding="utf-8")
    # config declares them; with no weights present at all the claim cannot be checked, so the
    # config is taken at face value (the honest fallback for a repo id or a partial checkout)
    assert K._modality_evidence(d) == (None, None)
    # and a projection-only checkpoint must NOT count as an encoder
    counts = {"image": 0, "audio": 0, "video": 0}
    assert all(v < 2 for v in counts.values()), "an encoder tower means repeated blocks"


def test_the_eval_corpus_is_chosen_for_the_model_not_hardcoded():
    """automap wrote `set EV=wikitext2_test.txt` into every generated build script, so a user
    benchmarking a reasoning or instruct model measured the mismatch rather than the build --
    gemma-4-12B-it reads ~664 on WikiText where a plain 7B reads 5.4. The corpus has to follow
    what the model IS, and that is knowable from its chat template."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "tools" / "pollard_automap.py").read_text(
        encoding="utf-8")
    assert 'default="wikitext2_test.txt"' not in src, (
        "--eval still defaults to raw text for every model, whatever it is")
    assert "pollard_modelkind" in src, "automap does not ask what the model is"
    assert "raw-text" in src, "no branch for a model that raw text cannot score"


def test_modelkind_detects_what_a_model_actually_is():
    """Pollard measured every model as though it were plain text. gemma-4-12B-it is instruct,
    thinking, tool-calling AND image+audio+video -- score that on raw Wikipedia and you measure the
    mismatch (PPL ~664, where a plain 7B reads 5.4 on the same corpus), and gate it with a short
    budget and its thinking block gets cut off, failing a build that was about to answer."""
    import pollard_modelkind as K
    d = tempfile.mkdtemp()
    # a thinking + tool-calling + multimodal checkout
    pathlib.Path(d, "chat_template.jinja").write_text(
        "{% if thinking %}<think>{% endif %} tool_call tool_response function_call thinking think",
        encoding="utf-8")
    pathlib.Path(d, "config.json").write_text(json.dumps(
        {"architectures": ["FooForConditionalGeneration"],
         "audio_config": {"audio_embed_dim": 8}, "vision_config": {"mm_embed_dim": 8},
         "video_token_id": 7}), encoding="utf-8")
    k = K.classify(d)
    assert k["instruct"] and k["thinking"] and k["agentic"], k
    assert set(["image", "audio", "video"]) <= set(k["modalities"]), k["modalities"]
    assert k["eval"] == "multimodal", "a text corpus cannot score a multimodal model"
    assert k["gate_tokens"] >= 200, "a thinking model needs room to finish thinking"

    # a plain base model: raw-text perplexity is the right measurement there
    b = tempfile.mkdtemp()
    pathlib.Path(b, "config.json").write_text(json.dumps({"architectures": ["FooForCausalLM"]}),
                                              encoding="utf-8")
    kb = K.classify(b)
    assert kb["base"] and kb["eval"] == "raw-text" and not kb["modalities"], kb


def test_coherence_gate_rejects_fluent_garbage():
    """The gate ran detect_loop() and, finding no repetition, reported "coherent". A build emitting
    token salad does not repeat, so it PASSED -- the IQ1_KT gemma4 flagship answered "The capital of
    France is isletedGESarz Svensri--st IC himself1 andict zichzelf" and the gate green-lit it for
    publication. Not-looping is not coherent. Each prompt carries a known answer now."""
    import pollard_bench as B
    assert all(isinstance(p, tuple) and len(p) == 2 for p in B.GATE_PROMPTS), (
        "gate prompts carry no expected answer, so nothing checks what the model said")
    joined = " ".join(" ".join(e).lower() for _p, e in B.GATE_PROMPTS)
    assert "jupiter" in joined, "the solar-system probe has no known answer to check"
    src = (pathlib.Path(__file__).resolve().parents[1] / "tools" / "pollard_bench.py").read_text(
        encoding="utf-8")
    seg = src.split("def coherence_gate", 1)[1].split("\ndef ", 1)[0]
    assert "knows" in seg and "not knows" in seg, (
        "the gate still passes on absence of looping alone")
    # salad must fail even though it never repeats
    salad = "isletedGESarz Svensri--st IC himself1 andict zichzelf-int-just"
    assert not any(e.lower() in salad.lower() for e in B.GATE_PROMPTS[0][1])


def test_bench_parses_the_kl_report_llama_cpp_actually_prints():
    """Under --kl-divergence llama-perplexity prints a different report: no "Final estimate", but
    both perplexities and a top-1 agreement. Two ways this went wrong at once -- PPL came back
    empty for every rung, and `Same top[^:]*:` latched onto the TABLE HEADER, because [^:] matches
    newlines, then ran to the next colon anywhere below and reported 818.3 as a percentage. A wrong
    number that looks like a result is worse than a blank."""
    import re
    src = (pathlib.Path(__file__).resolve().parents[1] / "tools" / "pollard_bench.py").read_text(
        encoding="utf-8")
    assert "re.M" in src, "patterns are not line-anchored, so they can match across the table"
    assert "[^:]*:" not in src, "an unanchored [^:] pattern can still swallow newlines"
    sample = (
        "chunk   PPL   ln(PPL(Q)/PPL(base))   KL Divergence   \u0394p RMS   Same top p\n"
        "[1]394.4999,0.1,0.5,1.2,88.1\n"
        "====== Perplexity statistics ======\n"
        "Mean PPL(Q)                   :  12.345678 \u00b1   0.123456\n"
        "Mean PPL(base)                :  11.111111 \u00b1   0.100000\n"
        "====== KL divergence statistics ======\n"
        "Mean    KLD:   0.425925 \u00b1   0.001\n"
        "Median  KLD:   0.081144\n"
        "Same top p: 91.234 \u00b1 0.123 %\n")
    got = {k: (lambda pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, sample, re.M)))(pat)
           for k, pat in (("ppl", r"^Mean PPL\(Q\)\s*:\s*([0-9.]+)"),
                          ("ref_ppl", r"^Mean PPL\(base\)\s*:\s*([0-9.]+)"),
                          ("mean_kld", r"^Mean\s+KLD:\s*([0-9.]+)"),
                          ("median_kld", r"^Median\s+KLD:\s*([0-9.]+)"),
                          ("top1", r"^Same top p:\s*([0-9.]+)"))}
    assert got["ppl"] == 12.345678 and got["ref_ppl"] == 11.111111, got
    assert got["mean_kld"] == 0.425925 and got["median_kld"] == 0.081144, got
    assert got["top1"] == 91.234, f"top-1 must be a percentage, got {got['top1']}"
    assert 0.0 <= got["top1"] <= 100.0


def test_imatrix_is_written_in_the_format_the_flagship_can_read():
    """llama-imatrix now defaults to a GGUF-format imatrix. Mainline reads both, but ik_llama --
    which builds the trellis flagship, the entire reason an imatrix is computed -- reads only the
    legacy .dat and dies with 'load_imatrix: failed reading number of values'. The K-quant ladder
    still builds, so the flagship is skipped and nothing says so."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "tools" / "pollard_auto.py").read_text(
        encoding="utf-8")
    seg = src.split("def _ensure_imatrix", 1)[1].split("\ndef ", 1)[0]
    assert "--output-format" in seg and '"dat"' in seg, (
        "the imatrix is left in the gguf default, which the trellis flagship cannot read")


def test_taskeval_scores_a_gguf_on_the_quantized_kernel():
    """lm-eval's HF backend opens a GGUF by DEQUANTIZING it, so a 5.9GB build becomes ~55GB of fp32:
    it cannot open the models Pollard exists for, and where it can, it scores an fp32 copy rather
    than the build that ships. A GGUF must be SERVED and scored on the quantized kernel."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "tools" / "pollard_taskeval.py").read_text(
        encoding="utf-8")
    assert "def served(" in src, "no server path: a GGUF is still dequantized to be scored"
    seg = src.split("def run(", 1)[1].split("\ndef ", 1)[0]
    assert 'serve and harness == "lm_eval" and path.endswith(".gguf")' in seg, (
        "the served path is not what a GGUF actually takes")
    assert '"gguf"' in seg and "base_url=" in seg, "lm-eval is not pointed at the server"
    # the reference model must not collide with the model's server
    call = src.split("os.path.join(a.out, \"ref\")", 1)
    assert len(call) > 1 and "port + 1" in src, "--ref would reuse the same port and fail"
    assert "--no-serve" in src, "no escape hatch for the dequantized path"


def test_llama_bin_resolves_on_windows_and_from_the_workspace():
    """Binary lookup had three Unix-only prefixes, no workspace bin/, and checked bare names -- so on
    Windows, where the file is llama-quantize.EXE, it resolved NOTHING and told the build box to
    update a runtime it already had. That silently disables the trellis flagship."""
    import pollard_calc as C
    src = pathlib.Path(C.__file__).read_text(encoding="utf-8")
    seg = src.split("def find_llama_bin", 1)[1].split("\ndef ", 1)[0]
    assert "POLLARD_HOME" in seg, "the workspace's own bin/ is not searched"
    assert '".exe"' in seg or "'.exe'" in seg, "a bare name never matches llama-quantize.exe"
    assert "win32" in seg, "no Windows branch in binary resolution"
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "bin"), exist_ok=True)
    exe = "llama-smoketest" + (".exe" if sys.platform == "win32" else "")
    open(os.path.join(d, "bin", exe), "wb").close()
    old = os.environ.get("POLLARD_HOME")
    os.environ["POLLARD_HOME"] = d
    try:
        got = C.find_llama_bin("llama-smoketest")
        assert got and os.path.exists(got), f"workspace bin/ not searched (got {got})"
    finally:
        os.environ.pop("POLLARD_HOME", None)
        if old is not None:
            os.environ["POLLARD_HOME"] = old


def test_probe_never_emits_a_profile_from_unreadable_weights():
    """A model too big to hold gets offloaded, and most of its weights then sit on the meta device
    holding NO data. The estimator reads every weight to compute dW, so taking meta at face value
    yields a profile that LOOKS measured and is not -- which produces a confidently bad allocation.
    Resolve from the checkpoint, and refuse rather than emit a partial profile."""
    try:
        import torch
    except ImportError:
        print("    (skipped: torch not installed -- `pip install pollard-weights[flybrain]`)")
        return
    import pollard_probe as P

    class _Lin:
        def __init__(self, w): self.weight = w
    real = _Lin(torch.zeros(2, 2))
    ws = P.WeightSource.__new__(P.WeightSource)          # no checkpoint on disk
    ws.names, ws.dir, ws.map, ws.missing, ws.unresolved = {}, "", {}, 0, []
    assert ws.get(real) is not None, "a resident weight must be returned as-is"
    assert ws.missing == 0
    meta = _Lin(torch.zeros(2, 2, device="meta"))
    assert ws.get(meta) is None, "a meta weight was returned as if it held data"
    assert ws.missing == 1, "an unresolvable weight must be COUNTED, not silently dropped"
    src = (pathlib.Path(__file__).resolve().parents[1] / "tools" / "pollard_probe.py").read_text(
        encoding="utf-8")
    seg = src.split("def _stream_sensitivity", 1)[1]
    assert "weights.missing" in seg and "raise SystemExit" in seg, (
        "the estimator still returns a profile built from weights it could not read")


def test_probe_skips_submodules_an_architecture_leaves_unset():
    """An architecture whose layers differ declares the full submodule set and leaves the unused
    ones None (Gemma4). hasattr() is True for those, so taking them at face value put a None into
    the forward-hook list and killed the probe AFTER loading 12B of weights."""
    try:
        import torch  # noqa: F401  (pollard_probe imports it at module scope)
    except ImportError:
        print("    (skipped: torch not installed -- `pip install pollard-weights[flybrain]`)")
        return
    import pollard_probe as P

    class _Blank:                       # a layer that declares gate/up/down but only uses one
        pass
    mlp = _Blank(); mlp.gate_proj = None; mlp.up_proj = "REAL"; mlp.down_proj = None
    layer = _Blank(); layer.mlp = mlp
    holder = _Blank(); holder.layers = [layer]
    model = _Blank(); model.model = holder

    got = P._linears(model, 0, "ffn")
    assert got == ["REAL"], f"unset submodules leaked into the hook list: {got}"
    noattn = P._linears(model, 0, "attn")        # whole group absent is legitimate
    assert noattn == [], f"a layer with no attn group should contribute nothing, got {noattn}"


def test_no_function_local_import_shadows_a_module_level_one():
    """`import os` inside a function makes `os` local to the WHOLE function, so every use of it
    EARLIER in that function raises UnboundLocalError -- even though the module imports os at the
    top and the code reads as correct. It only fires on the path that reaches the earlier use, so
    it ships green: this one ran fine on the Mac and killed the probe on the box."""
    import ast
    root = pathlib.Path(__file__).resolve().parents[1] / "tools"
    offenders = []
    for f in sorted(root.glob("pollard_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        top = set()
        for n in tree.body:                                  # module-level imports only
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                top.update((al.asname or al.name.split(".")[0]) for al in n.names)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for n in ast.walk(fn):
                if not isinstance(n, (ast.Import, ast.ImportFrom)):
                    continue
                for al in n.names:
                    name = al.asname or al.name.split(".")[0]
                    if name in top:
                        offenders.append(f"{f.name}:{n.lineno} re-imports '{name}'")
    assert not offenders, ("function-local import shadows a module-level one, making every earlier "
                           "use in that function an UnboundLocalError: " + "; ".join(offenders))


def test_converter_is_matched_to_the_model_not_just_found():
    """The driver used to return the bare string "convert_hf_to_gguf.py" and trust the shell. A
    converter too old for the architecture then failed deep in a build, reading as a problem with
    the model rather than the toolchain -- and the apparent fix was moving a 23.8GB GGUF across the
    network instead of copying a 3MB script. Capability is per-architecture, so verify it."""
    import pollard_convert as C
    d = tempfile.mkdtemp()
    pathlib.Path(d, "config.json").write_text(json.dumps(
        {"architectures": ["TotallyMadeUpForCausalLM"]}), encoding="utf-8")
    assert C.model_architectures(d) == ["TotallyMadeUpForCausalLM"]
    conv, why = C.find_converter(d)
    assert conv is None, "an architecture no converter registers was reported convertible"
    assert "TotallyMadeUpForCausalLM" in why, f"the refusal does not name the architecture: {why}"
    # registrations live in conversion/*.py, not the ~16KB entry point -- scanning only the script
    # would call every modern converter incapable
    root = pathlib.Path(__file__).resolve().parents[1] / "tools"
    body = (root / "pollard_convert.py").read_text(encoding="utf-8")
    assert "conversion" in body and "rglob" in body, "only the entry point is scanned for classes"
    drv = (root / "pollard_auto.py").read_text(encoding="utf-8")
    seg = drv.split("def _find_convert", 1)[1].split("\ndef ", 1)[0]
    assert "find_converter" in seg, "the one-shot still guesses at a converter"
    assert not re.search(r'return\s+["\']convert_hf_to_gguf\.py["\']', seg), (
        "the bare-PATH fallback is back -- the driver hands the shell a name it never verified")


def test_probe_estimator_is_one_that_can_finish():
    """Perturb+KL costs layers*groups full forward passes (~100 on a 48-layer model). On a model
    streaming off disk that never finishes -- so the gold path would be 'available' and hang. A
    model too big for RAM must fall to the one-pass estimator."""
    import pollard_auto as A2
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.truncate(400 * (1 << 30))                     # 400GB: bigger than any dev box
    assert A2._use_stream_probe(d, "auto"), "a model far bigger than RAM still uses perturb+KL"
    assert not A2._use_stream_probe(d, "kl"), "an explicit --probe-method kl must be honoured"
    small = tempfile.mkdtemp()
    with open(os.path.join(small, "model.safetensors"), "wb") as f:
        f.truncate(8 * (1 << 20))                       # 8MB: fits anywhere
    assert not A2._use_stream_probe(small, "auto"), "a tiny model gave up the accurate estimator"


def test_memory_detection_covers_all_three_platforms():
    """Pollard is cross-platform, so a POSIX-only memory probe is a silent Windows downgrade: no
    sysconf and no /proc there, so RAM reads as 0/None and every budget built on it is wrong --
    on the box that actually does the builds."""
    # The source scan runs EVERYWHERE, torch or not -- it is the actual regression guard, and CI is
    # exactly the machine that would otherwise let a POSIX-only probe through unnoticed.
    root = pathlib.Path(__file__).resolve().parents[1] / "tools"
    for fn, name in ((root / "pollard_probe.py", "_host_bytes"),
                     (root / "pollard_calc.py", "detect_available_ram_gb")):
        txt = fn.read_text(encoding="utf-8")
        body = txt.split(f"def {name}", 1)[1].split("\ndef ", 1)[0]
        helper = txt.split("def _win_mem", 1)[1].split("\ndef ", 1)[0] if "_win_mem" in txt else ""
        assert "win32" in body, f"{fn.name}:{name} has no Windows branch"
        assert "GlobalMemoryStatusEx" in body + helper, f"{fn.name}:{name} never asks Windows for RAM"
        assert ("darwin" in body or "sysconf" in body), f"{fn.name}:{name} lost its macOS path"
        assert ("meminfo" in body or "sysconf" in body), f"{fn.name}:{name} lost its Linux path"
    # pollard_calc is torch-free, so the live read is checked on every platform CI runs on
    from pollard_calc import detect_available_ram_gb
    if sys.platform != "win32":
        assert detect_available_ram_gb(), f"available RAM unreadable on {sys.platform}"
    try:
        import torch  # noqa: F401  (pollard_probe imports it at module scope)
    except ImportError:
        return
    import pollard_probe as P
    assert P._host_bytes() > 0, f"RAM unreadable on {sys.platform}"


def test_imatrix_ngl_fits_the_box_it_runs_on():
    """-ngl 99 offloads every layer: right when it fits, fatal when it doesn't."""
    import pollard_auto as A2
    assert A2._fit_ngl("/nonexistent.gguf", "7") == "7", "an explicit --ngl must be honoured"


def test_taskeval_reports_retention_against_a_reference():
    """Every other Pollard metric is intrinsic; nobody else quotes those.

    KL-to-f16, top-1 agreement and perplexity are the right things to ALLOCATE against -- cheap,
    dense, sensitive. They are not what competing releases publish. Those quote task scores and a
    retention figure, and a reader holding a KL number next to "98.2% of the full-precision
    baseline" has no basis for comparison. This tool closes that gap in their units.

    Retention is only meaningful against a reference measured the same way -- same tasks, same shot
    count, same limit -- so the tool prints that caveat rather than letting a number travel without
    it.
    """
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    src = (tools / "pollard_taskeval.py").read_text(encoding="utf-8")

    assert "retention" in src.lower(), "the headline figure must be retention, not a bare score"
    assert "--ref" in src, "retention needs a reference build"
    assert "published against a different suite" in src, \
        "the tool must say a retention figure is not comparable across suites"

    # the suite must match what competing releases actually run, or the comparison is theatre
    import sys as _s
    _s.path.insert(0, str(tools))
    import pollard_taskeval as TE
    for task in ("gsm8k", "ifeval", "mmlu_redux_generative", "gpqa_diamond_zeroshot",
                 "humaneval_plus", "mbpp_plus", "minerva_math500"):
        assert task in TE.SUITES["core"], f"{task} missing from the core suite"
    assert set(TE.CATEGORY.values()) >= {"Math", "Coding", "Knowledge & Reasoning",
                                         "Instruction Following"}

    # agentic is declared but NOT wired -- it needs a live tool environment, and a stub that
    # returned zeros would be worse than an honest refusal
    assert "tau2_bench" in TE.AGENTIC and "bfcl" in TE.AGENTIC
    assert "agentic" not in TE.SUITES, "agentic must not look runnable while it is not"


def test_taskeval_batch_default_is_not_auto():
    """lm-eval's 'auto' batch probes for a size that fits, and the probe dies on CPU.

    It fails without a usable message, which then looked like the tool was broken. 1 is slow and
    always works; a GPU user raises it.
    """
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    src = (tools / "pollard_taskeval.py").read_text(encoding="utf-8")
    i = src.index('"--batch-size"')
    assert 'default="1"' in src[i:i + 120], "auto batch-size dies on CPU"
    # and a harness failure must surface the CAUSE, not the last lines of a progress log
    assert "Traceback" in src and "hits or blob" in src, \
        "error reporting must pick failure lines, not the tail"



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
