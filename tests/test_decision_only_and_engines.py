"""What the clef-omni night run (box, 2026-10-21) needed box-local wrappers for, now in Pollard itself:

1. A DECISION-ONLY model (its graph ends in a decision head; no per-token logits: clef / clef-omni, a d3 readout
   GGUF) has no llama-perplexity KLD / PPL -- the KL base killed every verify. pollard-verify records them as
   "not applicable: decision-only model" and gates the rung on the decision board (agreement + option KL vs the
   reference) with per-bit-band defaults; the release plans the decision step for such a model on its own.
2. pollard-decision --ngl auto crashed (argparse type=int, exit 2): it now fits each served file to free VRAM.
3. Every llama.cpp step resolves its binary through ONE resolver (pollard_calc.find_llama_bin) that picks the engine
   carrying what the architecture needs -- clef-omni -> the tree with the dec.blk quantize patch (a staged
   runtime/llama.cpp.bolt), not the live runtime/llama.cpp.

Stdlib + gguf (a core dependency) only: runs in every CI shape."""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_calc as PC  # noqa: E402
import pollard_decision as PD  # noqa: E402
import pollard_verify as V  # noqa: E402

EXE = ".exe" if sys.platform == "win32" else ""


def _gguf(path, arch, tensors=("token_embd.weight",), kv=None):
    gguf = pytest.importorskip("gguf")
    import numpy as np
    w = gguf.GGUFWriter(str(path), arch)
    for k, v in (kv or {}).items():
        w.add_uint32(k, v)
    for name in tensors:
        w.add_tensor(name, np.zeros((4, 4), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


# ------------------------------------------------------------------------------------------- 1. decision-only

def test_decision_only_is_read_off_the_model():
    assert PD.decision_only_meta({"general.architecture": "clef-omni"})
    assert PD.decision_only_meta({"general.architecture": "clef"})
    assert PD.decision_only_meta({"general.architecture": "qwen35", "qwen35.decision.temperature.choice": 1.2})
    assert PD.decision_only_meta({"general.architecture": "x", "_tensor_params": {"dec.blk.0.ffn_down.weight": 1}})
    # a letter-readout decision model (OpenJev: a stock body) HAS logits -- KLD still applies to it
    assert PD.decision_only_meta({"general.architecture": "qwen35", "general.name": "OpenJev-27B"}) is None
    assert PD.decision_only_meta({"general.architecture": "qwen3"}) is None


def test_decision_only_from_files(tmp_path):
    hf = tmp_path / "clef-omni"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    assert PD.decision_only(str(hf)) is None
    (hf / "joint_head_config.json").write_text("{}")
    assert "joint_head_config" in PD.decision_only(str(hf))
    assert PD.decision_only(_gguf(tmp_path / "c.gguf", "clef-omni"))
    assert PD.decision_only(_gguf(tmp_path / "q.gguf", "qwen3")) is None
    assert PD.decision_only(str(tmp_path / "missing.gguf")) is None


def test_built_in_gates_judge_a_decision_only_rung_on_its_board():
    g = V.load_gates(None)
    q4 = V.thresholds_for(g, "Q4_K_M", "gguf", ["decision_only"])
    assert "kld_max" not in q4 and "top1_min" not in q4
    assert q4["decision_agreement_min"] == 92 and q4["decision_option_kl_max"] == 0.05
    q8 = V.thresholds_for(g, "Q8_0", "gguf", ["decision_only"])
    q2 = V.thresholds_for(g, "IQ2_XXS", "gguf", ["decision_only"])
    assert q8["decision_agreement_min"] > q4["decision_agreement_min"] > q2["decision_agreement_min"]
    assert q8["decision_option_kl_max"] < q4["decision_option_kl_max"] < q2["decision_option_kl_max"]
    # a plain rung is untouched
    assert V.thresholds_for(g, "Q4_K_M", "gguf", [])["kld_max"] == 0.08


def test_a_custom_gates_file_cannot_hold_a_decision_rung_to_a_kld_that_cannot_exist():
    custom = {"missing": "fail", "default": {"kld_max": 0.1, "top1_min": 90}}
    th = V.decision_only_thresholds(V.thresholds_for(custom, "IQ3_S", "gguf", ["decision_only"]), custom, "IQ3_S")
    assert "kld_max" not in th and "top1_min" not in th
    assert th["decision_agreement_min"] == 88 and th["decision_option_kl_max"] == 0.10
    # the file's own decision thresholds win when it has them
    mine = {"default": {"kld_max": 0.1, "decision_agreement_min": 99}}
    assert V.decision_only_thresholds(V.thresholds_for(mine, "IQ3_S"), mine, "IQ3_S") == {"decision_agreement_min": 99}


def test_a_decision_only_rung_with_a_tool_template_is_not_held_to_tool_calls():
    # clef-omni's chat template carries tools, so pollard-modelkind also reports `agentic`; its decision-only graph
    # writes no text, so the release (box, 2026-10-22) failed Q6_K on "tool-call retention: None (want >= 90)"
    # while its decision board was 100% / option KL 0.0001
    g = V.load_gates(None)
    kinds = ["instruct", "thinking", "agentic", "decision", "decision_only"]
    for rung in ("Q6_K", "IQ4_XS", "IQ3_S", "IQ2_XXS"):
        th = V.decision_only_thresholds(V.thresholds_for(g, rung, "gguf", kinds), g, rung)
        assert "toolcall_retention_min" not in th and "toolcall_pass" not in th, rung
        assert "decision_agreement_min" in th and "decision_option_kl_max" in th
        ok = V.evaluate_gate({"decision_agreement": 100, "decision_option_kl": 0.0001},
                             {k: v for k, v in th.items() if k.startswith(("decision_", "toolcall_"))}, "fail")
        assert ok["pass"], (rung, ok)
    # a custom gates file whose agentic section has no decision_only skip is covered too
    custom = {"missing": "fail", "kinds": {"agentic": {"toolcall_retention_min": 90}}}
    th = V.decision_only_thresholds(V.thresholds_for(custom, "Q6_K", "gguf", ["agentic", "decision_only"]), custom, "Q6_K")
    assert "toolcall_retention_min" not in th
    # an agentic model that is NOT decision-only keeps its tool-call gate
    assert V.thresholds_for(g, "Q6_K", "gguf", ["agentic"])["toolcall_retention_min"] == 90


def test_option_kl_is_a_gate_metric():
    m = V.metrics_from_results({"Q4_K_M": {"decision": {"agreement": 0.95, "option_kl": 0.02}}}, "Q4_K_M")
    assert m == {"decision_agreement": 95.0, "decision_option_kl": 0.02}
    th = {"decision_agreement_min": 92, "decision_option_kl_max": 0.05}
    assert V.evaluate_gate(m, th)["pass"]
    assert not V.evaluate_gate({**m, "decision_option_kl": 0.2}, th)["pass"]
    assert not V.evaluate_gate({"decision_option_kl": 0.01}, th)["pass"], "no board = not verified"


def _verify(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["pollard-verify", *argv])
    # which llama.cpp loads the file depends on the trees on this machine (clef-omni: our patch) -- not this test's
    monkeypatch.setattr(V, "gguf_runtime", lambda path: "stock")
    with pytest.raises(SystemExit) as e:
        V.main()
        raise SystemExit(0)
    return e.value.code or 0


def test_verify_skips_llama_perplexity_and_gates_on_the_decision_board(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    rung = _gguf(tmp_path / "clef-omni-Q4_K_M.gguf", "clef-omni")
    f16 = _gguf(tmp_path / "clef-omni-f16.gguf", "clef-omni")
    res = tmp_path / "results.json"
    res.write_text(json.dumps({"Q4_K_M": {"decision": {"agreement": 0.97, "option_kl": 0.01}, "coherence": "PASS"}}))
    monkeypatch.setattr(V, "measure_gguf", lambda *a, **k: pytest.fail("llama-perplexity on a decision-only model"))
    code = _verify(monkeypatch, ["--lane", "gguf", "--model", rung, "--ref", f16, "--eval", str(tmp_path / "e.txt"),
                                 "--gate", "--rung", "Q4_K_M", "--results", str(res)])
    out = json.loads(res.read_text())["Q4_K_M"]
    assert code == 0, out
    assert out["kld_status"] == "not-applicable" and out["kld_reason"].startswith(PD.NOT_APPLICABLE)
    assert out["verified"]["pass"] is True
    assert os.environ.get("POLLARD_ARCH") == "clef-omni", "the verify's binaries come from clef-omni's engine"


def test_verify_fails_a_decision_rung_that_changed_its_decisions(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    rung = _gguf(tmp_path / "clef-omni-IQ4_XS.gguf", "clef-omni")
    res = tmp_path / "results.json"
    res.write_text(json.dumps({"IQ4_XS": {"decision": {"agreement": 0.70, "option_kl": 0.3}, "coherence": "PASS"}}))
    monkeypatch.setattr(V, "measure_gguf", lambda *a, **k: pytest.fail("no KLD for a decision-only model"))
    code = _verify(monkeypatch, ["--lane", "gguf", "--model", rung, "--ref", rung, "--eval", "x", "--gate",
                                 "--rung", "IQ4_XS", "--results", str(res)])
    out = json.loads(res.read_text())["IQ4_XS"]
    assert code == 1
    assert set(out["verified"]["failed"]) == {"decision agreement %", "decision option KL"}


def test_a_plain_gguf_still_measures_kld(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    rung = _gguf(tmp_path / "q-Q8_0.gguf", "qwen3")
    called = []
    monkeypatch.setattr(V, "measure_gguf", lambda *a, **k: called.append(a) or {"kld": 0.001, "top1": 99.0, "ppl": 9.0})
    _verify(monkeypatch, ["--lane", "gguf", "--model", rung, "--ref", rung, "--eval", "x"])
    assert called


def test_the_card_says_kld_ppl_do_not_apply():
    import pollard_card as C
    results = {"Q4_K_M": {"decision": {"agreement": 0.97, "option_kl": 0.01, "n": 2000},
                          "kld_status": "not-applicable", "kld_reason": PD.NOT_APPLICABLE + " (clef-omni)"}}
    text = "\n".join(C.decision_section([{"name": "m-Q4_K_M.gguf", "tag": "Q4_K_M", "bytes": 1}], results))
    assert "## Decision fidelity" in text and "Mean KLD / PPL: not applicable: decision-only model" in text
    assert "not applicable" not in "\n".join(C.decision_section([{"name": "x", "bytes": 1}],
                                                                {"x": {"decision": {"agreement": 1.0}}}))


# ------------------------------------------------------------------------------------- the release plans it

@pytest.fixture
def clef(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    d = tmp_path / "clef-omni"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3_omni_moe", "architectures": ["ClefOmniModel"]}))
    (d / "joint_head_config.json").write_text("{}")
    return str(d)


def test_release_plans_the_decision_board_for_a_decision_only_model(clef):
    import pollard_release as R
    cfg = R.auto_preset(R.load_plan(None), clef)
    assert cfg["measure"]["decision"] is True and cfg["lanes"]["mlx"]["enabled"] is False, "preset decision"
    cfg = R.load_plan(None)
    cfg["measure"]["decision"] = False                       # even a plan that forgot it
    cfg["ngl"] = "auto"
    c = R._ctx(cfg, clef)
    assert c["decision_only"]
    steps = R.gguf_rung_steps(cfg, c, [{"tag": "Q4_K_M", "path": "/x/clef-omni-Q4_K_M.gguf"}])
    ids = [s.id for s in steps]
    assert ids.index("gguf/Q4_K_M/decision") < ids.index("gguf/Q4_K_M/verify")
    dec = next(s for s in steps if s.id == "gguf/Q4_K_M/decision").cmd
    assert dec[dec.index("--ngl") + 1] == "auto"


def test_release_points_every_step_at_the_models_engine(clef, tmp_path):
    import pollard_release as R
    c = R._ctx(R.load_plan(None), clef)
    assert R.engine_arch(c) is None                          # no f16 yet: nothing to read
    os.makedirs(os.path.dirname(c["f16"]), exist_ok=True)
    _gguf(c["f16"], "clef-omni")
    assert R.engine_arch(c) == "clef-omni" and os.environ["POLLARD_ARCH"] == "clef-omni"


# --------------------------------------------------------------------------------- 2. pollard-decision --ngl auto

def test_decision_ngl_takes_auto():
    assert PD._ngl_arg("auto") == "auto" and PD._ngl_arg("12") == "12" and PD._ngl_arg("AUTO") == "auto"
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        PD._ngl_arg("lots")


def test_decision_fits_each_served_file_on_its_own(monkeypatch):
    import pollard_bench as B
    import pollard_llamaserve as LS
    served = []

    @contextlib.contextmanager
    def fake_served(model, ngl, ctx=4096, port=0, extra=()):
        served.append((os.path.basename(model), ngl))
        yield "http://127.0.0.1:1"

    monkeypatch.setattr(B, "_served", fake_served)
    monkeypatch.setattr(LS, "resolve_ngl", lambda g, n: ("40" if "rung" in g else "12") if n == "auto" else str(n))
    monkeypatch.setattr(PD, "run", lambda base, post, items=None: [
        {"question": q, "options": o, "answer": a, "first": "", "probs": [1.0] + [0.0] * (len(o) - 1), "mass": 1.0}
        for _, q, o, a in PD.DECISION_SET])
    res = PD._gate("/m/rung.gguf", "auto", ref="/m/ref-f16.gguf")
    assert served == [("rung.gguf", "40"), ("ref-f16.gguf", "12")]
    assert res["board"]["agreement"] == 1.0


def test_decision_cli_accepts_ngl_auto(monkeypatch, tmp_path):
    got = {}
    monkeypatch.setattr(PD, "gate", lambda m, ngl, *a, **k: got.update(ngl=ngl) or {"verdict": "PASS", "rows": []})
    monkeypatch.setattr(sys, "argv", ["pollard-decision", "--gguf", str(tmp_path / "r.gguf"), "--ngl", "auto"])
    with pytest.raises(SystemExit) as e:
        PD.main()
    assert e.value.code == 0 and got["ngl"] == "auto"


# ------------------------------------------------------------------------------------------ 3. engine resolver

PATCH = """diff --git a/src/llama-quant.cpp b/src/llama-quant.cpp
--- a/src/llama-quant.cpp
+++ b/src/llama-quant.cpp
@@ -1,2 +1,4 @@
 int a = 0;
+    int n_dec_blk     = 0;
+            if (sscanf(name, "dec.blk.%d.", &i_layer) == 1 && n_dec_blk > 0) {
 int b = 0;
"""


def _bin(tree, name="llama-quantize", archs=(b"clef-omni",)):
    d = tree / "build" / "bin"
    d.mkdir(parents=True, exist_ok=True)
    p = d / (name + EXE)
    p.write_bytes(b"\x00".join(archs) + b"\x00")
    p.chmod(0o755)
    return str(p)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A Pollard checkout with two engines: the live runtime/llama.cpp (loads clef-omni, no dec.blk patch) and a
    staged runtime/llama.cpp.bolt (pollard-runtime lock file) whose source carries it -- the box on 2026-10-21."""
    r = tmp_path / "pw"
    (r / "runtime-patches").mkdir(parents=True)
    (r / "runtime-patches" / "engines.json").write_text(json.dumps({"_doc": "x", "llama.cpp": {"url": "u"}}))
    (r / "runtime-patches" / "llama.cpp-quantize-dec-blk.patch").write_text(PATCH)
    (r / "runtime-patches" / "llama.cpp-quantize-dec-blk.json").write_text(json.dumps(
        {"name": "quantize-dec-blk", "files": ["src/llama-quant.cpp"], "archs": ["clef-omni"]}))
    live, bolt = r / "runtime" / "llama.cpp", r / "runtime" / "llama.cpp.bolt"
    for t, patched in ((live, False), (bolt, True)):
        (t / "src").mkdir(parents=True)
        body = ["int a = 0;", "int b = 0;"]
        if patched:   # CRLF, as on the Windows box: compared by content, never by git
            body[1:1] = ['    int n_dec_blk     = 0;', '            if (sscanf(name, "dec.blk.%d.", &i_layer) == 1 '
                         '&& n_dec_blk > 0) {']
        (t / "src" / "llama-quant.cpp").write_bytes(("\r\n".join(body) + "\r\n").encode())
    (r / "runtime" / "llama.cpp.bolt.lock.json").write_text("{}")
    (r / "runtime" / "llama.cpp.prev.lock.json").write_text("{}")
    monkeypatch.setattr(PC, "_repo_root", lambda: str(r))
    monkeypatch.setattr(PC, "_PATCH_IN_TREE", {})
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    return r


def test_engine_trees_are_engines_json_plus_lock_files(repo):
    names = [n for n, _ in PC.engine_trees()]
    assert names == ["llama.cpp", "llama.cpp.bolt"], "an update's .prev / .next is never an engine"


def test_clef_omni_resolves_to_the_engine_with_its_quantize_patch(repo):
    live = _bin(repo / "runtime" / "llama.cpp")
    bolt = _bin(repo / "runtime" / "llama.cpp.bolt")
    assert PC.find_llama_bin("llama-quantize", "clef-omni") == bolt
    assert PC.find_llama_bin("llama-quantize") == live, "no architecture: the live tree, as always"
    PC.set_model_arch("clef-omni")                           # what the release / tools do from the GGUF
    assert PC.find_llama_bin("llama-quantize") == bolt
    # the same engine for every step of that model
    _bin(repo / "runtime" / "llama.cpp", "llama-imatrix")
    imx = _bin(repo / "runtime" / "llama.cpp.bolt", "llama-imatrix")
    assert PC.find_llama_bin("llama-imatrix") == imx


def test_an_engine_without_the_binary_or_the_arch_is_passed_over(repo):
    live = _bin(repo / "runtime" / "llama.cpp")
    assert PC.find_llama_bin("llama-quantize", "clef-omni") == live, "bolt has no llama-quantize built"
    _bin(repo / "runtime" / "llama.cpp.bolt", archs=(b"qwen3",))
    assert PC.find_llama_bin("llama-quantize", "clef-omni") == live, "bolt cannot load clef-omni"


def test_architectures_without_declared_patches_keep_the_old_order(repo):
    live = _bin(repo / "runtime" / "llama.cpp", archs=(b"qwen3", b"clef-omni"))
    _bin(repo / "runtime" / "llama.cpp.bolt", archs=(b"qwen3", b"clef-omni"))
    assert PC.find_llama_bin("llama-quantize", "qwen3") == live
    assert PC.find_llama_bin("llama-quantize", "nonesuch") is None, "explicit arch nobody loads -> None (caller explains)"
    PC.set_model_arch("nonesuch")
    assert PC.find_llama_bin("llama-quantize") == live, "an implicit arch never makes a binary disappear"


def test_patch_presence_is_by_content(repo):
    m = PC.required_patches("clef-omni")
    assert [x["name"] for x in m] == ["quantize-dec-blk"] and PC.required_patches("qwen3") == []
    assert PC.tree_has_patch(str(repo / "runtime" / "llama.cpp.bolt"), m[0])
    assert not PC.tree_has_patch(str(repo / "runtime" / "llama.cpp"), m[0])


def test_set_model_arch_reads_the_gguf(tmp_path):
    assert PC.set_model_arch(_gguf(tmp_path / "m.gguf", "clef-omni")) == "clef-omni"
    assert os.environ["POLLARD_ARCH"] == "clef-omni"
    assert PC.set_model_arch(str(tmp_path / "missing.gguf")) is None and os.environ["POLLARD_ARCH"] == "clef-omni"


def test_the_shipped_dec_blk_patch_declares_clef_omni():
    """The repo's own manifest is what sends clef-omni to the patched engine (skipped only where runtime-patches/ is not
    found at all -- neither in the checkout nor as an installed package's data)."""
    p = Path(PC.patches_dir()) / "llama.cpp-quantize-dec-blk.json"
    if not p.is_file():
        pytest.skip("no runtime-patches/ in this checkout or install")
    assert "clef-omni" in json.loads(p.read_text())["archs"]
