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
    for names, exp_moe, exp_hyv4 in [(_dense(), False, False), (_moe(), True, False),
                                     (_hyv4(), True, True)]:
        _, nl, is_moe, is_hyv4 = A.parse_tensors(_tensorfile(names))
        assert nl == 4, f"layers {nl}"
        assert is_moe == exp_moe, f"is_moe {is_moe} != {exp_moe}"
        assert is_hyv4 == exp_hyv4, f"is_hyv4 {is_hyv4} != {exp_hyv4}"


# ---- encoder rules (the casing bug) ----------------------------------------------------------
def test_cq_casing():
    assert A._cq("q3_k") == "q3_K" and A._cq("q2_k") == "q2_K"        # K-quants: capital K
    assert A._cq("iq1_kt") == "iq1_kt" and A._cq("iq2_kt") == "iq2_kt"  # trellis: lowercase
    assert A._bar_type("q2_k") == "Q2_K" and A._bar_type("q4_k") == "Q4_K_M"  # bar preset
    assert A._bar_type("iq1_kt") == "IQ1_KT"


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


# ---- guards (dense refused; deprecation warns) -----------------------------------------------
def test_dense_guard():
    tf = _tensorfile(_dense())
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "tools",
                        "pollard_automap.py"), "--tensors", tf, "--model", "d.gguf",
                        "--out", os.path.join(tempfile.gettempdir(), "g.bat")],
                       capture_output=True, text=True)
    assert "REFUSED" in (r.stdout + r.stderr), "automap must REFUSE a dense model without --allow-dense"


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
