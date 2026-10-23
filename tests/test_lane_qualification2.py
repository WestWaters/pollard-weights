r"""Regression tests for lane qualification round 2 (box, Qwen2.5-0.5B-Instruct on main 0773add, 2026-10-18,
C:\pollard\release\qualification.json "bugs" + "quality_findings"):

1. pollard-verify --lane exl3 --end-to-end built Cache(max_num_tokens=length + 64) = 576; exllamav3 asserts a multiple
   of 256 -> every EXL3 rung failed its verify.
2. pollard_stream's decoded loader could not read llm-compressor's MXFP4 (mxfp4-pack-quantized): "no tensor for
   model.layers.0.self_attn.q_proj.weight" -> no CPU tool gate for that format.
3. pollard-toolcall counted always-on Windows shell / overlay apps (TabTip, XboxPcApp, XboxPcTray) as GPU users.
4. clef-omni: llama-quantize could not read a dec.blk.N tensor's layer on an MoE body; llama-imatrix's perplexity
   pass read logits the decision head does not produce; automap read `dec.blk.0.x` as `blk.0.x`.
5. The HF lanes were scored on 256-token BOS-less chunks from position 0, the GGUF lane by llama-perplexity at
   -c 2048 (second half): numbers on one card did not compare.
6. Pollard EXL3 lost to exllamav3's defaults by KLD on the 0.5B: the smoothed bf16 checkpoint alone was KLD 0.0011
   from its source (rounded fold), and the recipe is now chosen by a measured A/B.

numpy-only checks run in every CI shape; the tiny-model ones need torch + transformers (+ tokenizers); the patch
checks need git."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

import pollard_automap as A  # noqa: E402
import pollard_exl3 as E3  # noqa: E402
import pollard_mxfp6 as M6  # noqa: E402
import pollard_release as R  # noqa: E402
import pollard_stream as S  # noqa: E402
import pollard_toolcall as TC  # noqa: E402
import pollard_verify as V  # noqa: E402


def _need_torch():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")


# ------------------------------------------------------------------------- 1. EXL3 cache: a multiple of 256

def test_exl3_cache_is_a_whole_number_of_pages():
    assert V.exl3_cache_tokens(512) == 768                        # 512 + 64 = 576 was the crash
    for n in (1, 63, 192, 255, 256, 511, 1984, 2048, 4000):
        t = V.exl3_cache_tokens(n)
        assert t % 256 == 0 and t >= n + 64 and t - (n + 64) < 256


def _fake_exllamav3(monkeypatch, logits_fn, seen):
    """An exllamav3 module that enforces what the real one asserts (max_num_tokens % 256 == 0)."""
    mod = types.ModuleType("exllamav3")

    class Config:
        @staticmethod
        def from_directory(d):
            return Config()

    class Model:
        @staticmethod
        def from_config(cfg):
            return Model()

        def load(self):
            seen.append("load")

        def unload(self):
            pass

        def forward(self, ids, *a, **k):
            return logits_fn(ids)

    class Cache:
        def __init__(self, model, max_num_tokens):
            assert max_num_tokens % 256 == 0, "max_num_tokens must be a multiple of 256"
            seen.append(max_num_tokens)

    class Tokenizer:
        @staticmethod
        def from_config(cfg):
            import torch

            class T:
                def encode(self, text):
                    return torch.tensor([ord(c) % 250 for c in text])
            return T()

    mod.Config, mod.Model, mod.Cache, mod.Tokenizer = Config, Model, Cache, Tokenizer
    monkeypatch.setitem(sys.modules, "exllamav3", mod)
    return mod


def test_exl3_end_to_end_builds_a_cache_exllamav3_accepts(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = []
    _fake_exllamav3(monkeypatch, lambda ids: torch.randn(1, ids.shape[-1], 250), seen)
    acc, ppl = V.end_to_end("m", "s", "cpu", 1, 512)                 # crashed: Cache(..., 576)
    assert seen[0] == 768 and 0.0 <= acc <= 1.0 and ppl > 0


# ------------------------------------------------------------------------- 2. compressed-tensors MXFP4 / MXFP8

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def _spec_mxfp4(packed, scales):
    """The OCP MX v1.0 decode written out independently: low nibble first, E2M1 codes, w = v * 2^(e - 127)."""
    p = np.asarray(packed, np.uint8)
    codes = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), np.int64)
    codes[..., 0::2], codes[..., 1::2] = p & 15, p >> 4
    v = FP4[codes]
    N, K = v.shape
    return (v.reshape(N, K // 32, 32) * np.ldexp(1.0, np.asarray(scales, np.int64) - 127)[..., None]).reshape(N, K)


def _llmc_mxfp4_checkpoint(d, weights, extra=None, fmt="mxfp4-pack-quantized"):
    """Write {module: [out, in] float} the way llm-compressor saves MXFP4: weight_packed U8 [out, in/2] +
    weight_scale U8 [out, in/32] (E8M0), no global scale, config format mxfp4-pack-quantized."""
    os.makedirs(d, exist_ok=True)
    t = dict(extra or {})
    for mod, w in weights.items():
        codes, sc = M6.mx_encode(np.asarray(w, np.float32), "e2m1", "ocp")
        t[f"{mod}.weight_packed"] = ("U8", M6.pack4(codes))
        t[f"{mod}.weight_scale"] = ("U8", sc)
    S._st_write(os.path.join(d, "model.safetensors"), t, {"format": "pt"})
    qc = {"quant_method": "compressed-tensors", "format": fmt, "ignore": ["lm_head"],
          "config_groups": {"group_0": {"format": fmt, "targets": ["Linear"],
                                        "weights": {"num_bits": 4, "type": "float", "group_size": 32,
                                                    "strategy": "group", "scale_dtype": "torch.uint8"}}}}
    cfgp = os.path.join(d, "config.json")
    cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
    cfg["quantization_config"] = qc
    json.dump(cfg, open(cfgp, "w"))
    return t


def test_source_decodes_llmcompressor_mxfp4_by_the_spec(tmp_path):
    rng = np.random.default_rng(0)
    w = {"model.layers.0.self_attn.q_proj": rng.normal(size=(8, 64)).astype(np.float32) * 0.05,
         "model.layers.0.mlp.down_proj": rng.normal(size=(4, 96)).astype(np.float32)}
    t = _llmc_mxfp4_checkpoint(str(tmp_path / "rung"), w,
                               {"model.layers.0.self_attn.q_proj.bias": ("F32", np.zeros(8, np.float32))})
    src = S.Source(str(tmp_path / "rung"))
    for mod, ref in w.items():
        key = mod + ".weight"
        assert key in src.logical, "the logical .weight the decoded loader asks for"
        assert src.encoding(key)["algo"] == "mxfp4" and src.shape(key) == ref.shape
        got = src.read(key)
        want = _spec_mxfp4(t[mod + ".weight_packed"][1], t[mod + ".weight_scale"][1])
        assert np.array_equal(got, want.astype(np.float32))
        assert np.abs(got - ref).max() <= np.abs(ref).max() * 0.5    # it IS the weight, at 4 bits
    assert "model.layers.0.self_attn.q_proj.bias" in src.logical      # plain tensors read as stored
    rd = S.RungReader(str(tmp_path / "rung"))
    assert rd.read("model.layers.0.mlp.down_proj.weight").shape == (4, 96)
    assert S.decoded_bytes(str(tmp_path / "rung")) == (8 * 64 + 4 * 96 + 8) * 2


def test_nvfp4_is_still_nvfp4_and_mxfp8_decodes_its_e8m0_groups(tmp_path):
    rng = np.random.default_rng(1)
    w = rng.normal(size=(4, 64)).astype(np.float32)
    parts, dec = M6.encode_atom(w, "nvfp4")
    t = {f"m.nv.{k}": v for k, v in parts.items()}
    codes, sc = M6.mx_encode(w, "e4m3", "ocp")
    t["m.mx8.weight"], t["m.mx8.weight_scale"] = ("F8_E4M3", codes.astype(np.uint8)), ("U8", sc)
    os.makedirs(tmp_path / "r")
    S._st_write(str(tmp_path / "r" / "model.safetensors"), t)
    src = S.Source(str(tmp_path / "r"))
    assert src.encoding("m.nv.weight")["algo"] == "nvfp4" and np.allclose(src.read("m.nv.weight"), dec)
    assert src.encoding("m.mx8.weight")["algo"] == "mxfp8"
    assert np.allclose(src.read("m.mx8.weight"), M6.mx_decode(codes, sc, "e4m3"))


def test_decoded_loader_runs_an_llmcompressor_mxfp4_rung(tmp_path):
    """The CPU tool gate's loader: a plain transformers model of the MXFP4 rung, weights from its own bytes."""
    _need_torch()
    import torch
    import tiny_models as TM
    from transformers import AutoModelForCausalLM
    src = TM.llama(str(tmp_path / "src"), layers=2)
    ref = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
    sd = {k: v.numpy() for k, v in ref.state_dict().items()}
    rung = str(tmp_path / "rung")
    shutil.copytree(src, rung)
    os.remove(os.path.join(rung, "model.safetensors"))
    lin = {k[:-len(".weight")]: v for k, v in sd.items()
           if k.startswith("model.layers.") and k.endswith("_proj.weight")}
    plain = {k: ("F32", v) for k, v in sd.items() if k[:-len(".weight")] not in lin}     # lm_head: ignored
    t = _llmc_mxfp4_checkpoint(rung, lin, plain)
    m = S.decoded_hf_model(rung, "cpu", False, dtype=torch.float32)       # "no tensor for ...q_proj.weight" before
    got = m.state_dict()
    for mod in lin:
        want = _spec_mxfp4(t[mod + ".weight_packed"][1], t[mod + ".weight_scale"][1])
        assert np.allclose(got[mod + ".weight"].numpy(), want), mod
    ids = torch.tensor([[5, 17, 99, 3, 42, 7, 250, 11]])
    with torch.no_grad():
        kl = V.kld_top1(ref(ids).logits[0].numpy(), m(ids).logits[0].numpy())
    assert kl["kld"] < 0.05


# ------------------------------------------------------------------------- 3. the shared GPU's always-on apps

SMI = """\
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|=========================================================================================|
|    0   N/A  N/A            1201    C+G   C:\\Windows\\System32\\TabTip.exe            N/A      |
|    0   N/A  N/A            1202    C+G   ...\\Microsoft.GamingApp\\XboxPcApp.exe       N/A      |
|    0   N/A  N/A            1203    C+G   ...\\Microsoft.GamingApp\\XboxPcTray.exe      N/A      |
|    0   N/A  N/A            1204    C+G   ...\\Microsoft.XboxGamingOverlay\\GameBar.exe N/A      |
|    0   N/A  N/A            1205    C+G   ...\\WidgetBoard.exe                          N/A      |
|    0   N/A  N/A            1206    C+G   ...\\PhoneExperienceHost.exe                  N/A      |
|    0   N/A  N/A            1207    C+G   ...\\SnippingTool\\SnippingTool.exe            N/A      |
+-----------------------------------------------------------------------------------------+
"""


def test_windows_shell_and_overlay_apps_do_not_hold_the_gpu(monkeypatch):
    monkeypatch.delenv("POLLARD_GPU_IGNORE", raising=False)
    assert TC.gpu_users(SMI, "") == [], "an idle Windows 11 desktop: the GPU is free"
    game = SMI.replace("...\\WidgetBoard.exe     ", "...\\FortniteClient.exe")
    assert TC.gpu_users(game, "") == ["FortniteClient.exe"], "a game still counts"
    assert TC.gpu_users("", "4242, python.exe, 9000\n") == ["python.exe"], "a compute job still counts"


def test_kvbench_shares_the_same_shell_list():
    import pollard_kvbench as K
    ok = {x.lower() for x in K.SHARED_OK}
    for name in ("tabtip", "xboxpcapp", "xboxpctray", "gamebar"):
        assert name in ok


# ------------------------------------------------------------------------- 4. clef-omni: quantize, imatrix, automap

PATCH_NAME = "llama.cpp-quantize-dec-blk"
#: the runtime-patches/ the tools themselves read: the checkout's, or the installed package's data copy
_PATCHES = Path(__import__("pollard_calc").patches_dir())
_REAL_PATCH = _PATCHES / f"{PATCH_NAME}.patch"
#: the patch itself (the nix package ships no runtime-patches/): kept identical to the file, checked below
DEC_BLK_PATCH = """\
diff --git a/src/llama-quant.cpp b/src/llama-quant.cpp
index 9c6066796..a72518973 100644
--- a/src/llama-quant.cpp
+++ b/src/llama-quant.cpp
@@ -179,6 +179,9 @@ struct quantize_state_impl {
 
     int n_fallback    = 0;
 
+    // blocks of a second stack numbered apart from the body: clef-omni's decision head (dec.blk.N) on an MoE body
+    int n_dec_blk     = 0;
+
     bool has_imatrix = false;
 
     // used to figure out if a model has tied embeddings (tok_embd shares weights with output)
@@ -458,13 +461,17 @@ static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type
         return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
     };
     const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
-    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
+    const int n_dec_blk = qs.n_dec_blk;
+    auto layer_info = [n_expert, n_dec_blk] (int i_layer, int n_layer, const char * name) {
         if (n_expert > 1) {
             // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but occasionally randomly
             // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
             // for getting the current layer as I initially thought, and we need to resort to parsing the
             // tensor name.
-            if (sscanf(name, "blk.%d.", &i_layer) != 1) {
+            // A decoder stack next to the body (dec.blk.N, e.g. a decision head) is its own stack of n_dec_blk blocks.
+            if (sscanf(name, "dec.blk.%d.", &i_layer) == 1 && n_dec_blk > 0) {
+                n_layer = n_dec_blk;
+            } else if (sscanf(name, "blk.%d.", &i_layer) != 1) {
                 throw std::runtime_error(format("Failed to determine layer for tensor %s", name));
             }
             if (i_layer < 0 || i_layer >= n_layer) {
@@ -941,6 +948,11 @@ static void init_quantize_state_counters(quantize_state_impl & qs, std::vector<t
         if (cat == tensor_category::OUTPUT) {
             qs.has_tied_embeddings = false;
         }
+
+        int i_dec = -1;
+        if (sscanf(tm.name.c_str(), "dec.blk.%d.", &i_dec) == 1 && i_dec >= qs.n_dec_blk) {
+            qs.n_dec_blk = i_dec + 1;
+        }
     }
     qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)qs.model.hparams.n_layer_all;
 }
@@ -1500,6 +1512,7 @@ void llama_quant_compute_types(
     qs->i_ffn_gate          = 0;
     qs->i_ffn_up            = 0;
     qs->n_fallback          = 0;
+    qs->n_dec_blk           = 0;
     qs->has_imatrix         = false;
     qs->has_tied_embeddings = true;
 
"""


def test_the_embedded_patch_is_the_shipped_one():
    if not _REAL_PATCH.exists():
        pytest.skip("no runtime-patches/ in this checkout or install")
    assert _REAL_PATCH.read_text(encoding="utf-8") == DEC_BLK_PATCH


def test_the_dec_blk_patch_is_declared_after_clef_omni_and_its_manifest_matches():
    if not _REAL_PATCH.exists():
        pytest.skip("no runtime-patches/ in this checkout or install")
    eng = json.load(open(_PATCHES / "engines.json"))["llama.cpp"]["patches"]
    assert PATCH_NAME in eng and eng.index(PATCH_NAME) > eng.index("llama.cpp-clef-omni")
    man = json.load(open(_PATCHES / f"{PATCH_NAME}.json"))
    import pollard_runtime_update as U
    assert set(man["files"]) == U._touched_files(DEC_BLK_PATCH) - {"/dev/null"} == {"src/llama-quant.cpp"}
    assert man["tree_slug"] == "llama.cpp" and man["remote"].endswith("ggml-org/llama.cpp")


def _preimage(diff):
    """The pre-patch text of every hunk, in order: a file the patch applies to (git tolerates the offsets)."""
    out = []
    for ln in diff.splitlines():
        if ln.startswith("@@"):
            out.append("// ...")
        elif ln[:1] in (" ", "-") and not ln.startswith("---"):
            out.append(ln[1:])
    return "\n".join(out) + "\n"


def _git(*a, cwd):
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="t",
               GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, env=env)


def test_the_patch_applies_reads_dec_blk_layers_and_the_patch_checks_see_it(tmp_path):
    """git apply + the content checks pollard-runtime uses: applied, the patch is live (reverse-applies) and
    absorbed (its added lines are in the file); on the untouched tree it is neither."""
    if not shutil.which("git"):
        pytest.skip("no git")
    import pollard_runtime_update as U
    tree = tmp_path / "llama.cpp"
    (tree / "src").mkdir(parents=True)
    (tree / "src" / "llama-quant.cpp").write_text(_preimage(DEC_BLK_PATCH))
    pp = tmp_path / "p.patch"
    pp.write_text(DEC_BLK_PATCH)
    assert _git("init", "-q", cwd=tree).returncode == 0
    _git("add", "-A", cwd=tree), _git("commit", "-qm", "base", cwd=tree)
    m = {"_patch": str(pp), "files": ["src/llama-quant.cpp"], "adds_strings": []}
    assert not U.patch_is_live(str(tree), m) and not U._absorbed(str(tree), m)
    r = _git("apply", "--check", str(pp), cwd=tree)
    assert r.returncode == 0, r.stderr
    _git("apply", str(pp), cwd=tree)
    assert U.patch_is_live(str(tree), m) and U._absorbed(str(tree), m)
    src = (tree / "src" / "llama-quant.cpp").read_text()
    # the layer of a dec.blk tensor is parsed from its own prefix, on its own stack's count, before blk.N's rule
    assert src.index('sscanf(name, "dec.blk.%d.", &i_layer)') < src.index('sscanf(name, "blk.%d.", &i_layer)')
    assert "n_layer = n_dec_blk;" in src
    assert 'sscanf(tm.name.c_str(), "dec.blk.%d.", &i_dec)' in src and "qs.n_dec_blk = i_dec + 1;" in src
    assert "qs->n_dec_blk           = 0;" in src, "llama_quant_compute_types resets it with the other counters"


CLEF_OMNI_NAMES = (
    ["token_embd.weight", "output_norm.weight"]
    + [f"blk.{i}.{k}.weight" for i in range(4) for k in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_inp",
                                                          "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "attn_norm")]
    + [f"dec.blk.0.{k}.weight" for k in ("cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_o",
                                         "cross_attn_norm", "ffn_up", "ffn_down")]
    + [f"dec.blk.1.{k}.weight" for k in ("attn_q", "attn_k", "attn_v", "attn_o", "cross_attn_q", "cross_attn_o",
                                         "ffn_up", "ffn_down")]
    + ["decision.proj_memory.weight", "decision.proj_question.weight", "decision.scorer.weight",
       "decision.scorer_out.weight", "decision.hidden_norm.weight", "decision.scales"])


def _apply(cq, base, name):
    for rule in cq:                                       # custom-q / --tensor-type: first match wins, re.search
        pat, typ = rule.rsplit("=", 1)
        if re.search(pat, name):
            return typ
    return base


def test_automap_reads_the_decision_head_as_its_own_stack(tmp_path):
    p = tmp_path / "names.txt"
    p.write_text("\n".join(f"llama_tensor_get_type: {n}  - type q8_0" for n in CLEF_OMNI_NAMES))
    names, nl, is_moe, arch = A.parse_tensors(str(p))
    assert "dec.blk.0.ffn_down.weight" in names, "kept whole, not read as blk.0.ffn_down.weight"
    assert names.count("blk.0.ffn_down.weight") == 0 and "decision.scorer.weight" in names
    assert nl == 4 and is_moe and A.dec_layers(names) == 2 and "decision-head 2dec" in arch


def test_automap_rules_give_dec_blk_the_body_rules_on_its_own_numbering():
    names = list(CLEF_OMNI_NAMES)
    _, cq = A.recipe_flags(8, True, "iq1_kt", "iq2_kt", n_dec=A.dec_layers(names))
    ap = lambda n: _apply(cq, "iq1_kt", n)                                     # noqa: E731
    for n in [x for x in names if x.startswith(("dec.", "decision.")) and "norm" not in x]:
        assert ap(n) == "iq2_kt", f"{n}: the decision head is protected, never the crush atom"
    assert ap("blk.4.ffn_up_exps.weight") == "iq1_kt"                           # the body recipe is unchanged
    assert ap("blk.0.ffn_up_exps.weight") == "iq2_kt"                           # body edge block
    # the body's edge rules no longer reach the head by accident (and the head's never reach the body)
    _, cq0 = A.recipe_flags(8, True, "iq1_kt", "iq2_kt")
    assert not any(re.search(r.rsplit("=", 1)[0], "dec.blk.0.ffn_up.weight") for r in cq0 if "blk" in r)


def test_uncovered_pins_name_exactly_one_tensor(monkeypatch):
    monkeypatch.setattr(A, "imatrix_covered", lambda p: {"dec.blk.0.ffn_down.weight"})
    pins, _ = A.uncovered_pins(["blk.0.ffn_down.weight", "dec.blk.0.ffn_down.weight", "dec.blk.1.attn_o.weight",
                                "decision.proj_memory.weight", "decision.hidden_norm.weight"], "x.imatrix")
    pats = [p.rsplit("=", 1)[0] for p in pins]
    hit = lambda n: [p for p in pats if re.search(p, n)]                       # noqa: E731
    assert len(hit("blk.0.ffn_down.weight")) == 1
    assert hit("dec.blk.0.ffn_down.weight") == [], "covered: the body's pin must not catch it"
    assert len(hit("dec.blk.1.attn_o.weight")) == 1 and len(hit("decision.proj_memory.weight")) == 1
    assert hit("decision.hidden_norm.weight") == []


def test_release_imatrix_skips_the_perplexity_pass_on_a_decision_head(tmp_path, monkeypatch):
    import pollard_auto as PA
    import pollard_calc
    host = tmp_path / "clef-omni-Q8_0-host.gguf"
    host.write_bytes(b"x")
    monkeypatch.setattr(PA, "read_gguf_meta", lambda p: {"general.architecture": "clef-omni",
                                                        "clef-omni.decision.block_count": 1})
    monkeypatch.setattr(R, "imatrix_host", lambda cfg, c: (str(host), None))
    monkeypatch.setattr(R, "imatrix_ngl", lambda cfg, c, h: "0")
    monkeypatch.setattr(pollard_calc, "find_llama_bin", lambda n, arch=None: str(tmp_path / n))
    ran = []

    def run(cmd, *a, **k):
        ran.append(list(cmd))
        open(cmd[cmd.index("-o") + 1], "wb").write(b"imatrix")
    monkeypatch.setattr(R, "run_cmd", run)
    c = {"imatrix": str(tmp_path / "cal" / "m.imatrix"), "f16": str(host), "calib": str(tmp_path / "c.txt"),
         "ngl": "0", "gguf_dir": str(tmp_path), "name": "m"}
    assert R.build_imatrix(c, {}, out=lambda *_: None) == c["imatrix"]
    assert "--no-ppl" in ran[0], "clef-omni's head has no LM logits: the perplexity pass crashed (0xC0000005)"


# ------------------------------------------------------------------------- 5. one protocol for every lane

class _Tok:
    def __init__(self, bos=None, adds=False, template=""):
        self.bos_token_id, self.bos_token, self.chat_template, self._adds = bos, "<s>" if bos is not None else None, \
            template, adds

    def encode(self, text):
        ids = [ord(c) % 200 + 2 for c in text]
        return ([self.bos_token_id] if self._adds and self.bos_token_id is not None else []) + ids

    def __call__(self, text):
        return {"input_ids": self.encode(text)}


def test_hf_lanes_cut_bos_led_chunks_and_score_the_second_half(tmp_path):
    ev = tmp_path / "ev.txt"
    ev.write_text("held-out text " * 400)
    parts, starts = V._hf_tokens(_Tok(bos=1, adds=True), str(ev), 2048, 2)
    assert len(parts) == 2 and all(len(p) == 2048 and p[0] == 1 for p in parts)
    assert parts[0].count(1) == 1, "a tokenizer that adds BOS: not doubled"
    assert starts == [1024, 1024]
    parts, starts = V._hf_tokens(_Tok(bos=None), str(ev), 512, 3)               # Qwen: no BOS
    assert [len(p) for p in parts] == [512] * 3 and starts == [256] * 3
    assert V.PROTOCOL == V.MLX_PROTOCOL and V.PROTOCOL_LENGTH == 2048


def test_reference_cache_is_keyed_on_the_protocol():
    old = V._ref_cache("s", "e", 2048, 16, "hf")
    assert old != V._ref_cache("s", "e", 2048, 16, f"hf-{V.PROTOCOL}"), "an old-protocol cache must never be reused"


def test_the_release_measures_every_hf_lane_at_the_protocol_length():
    import inspect
    for fn in (V.measure_hf, V.measure_stream, V.measure_exlr8, V.measure_exl3):
        assert inspect.signature(fn).parameters["length"].default is None, fn.__name__   # -> PROTOCOL_LENGTH


@pytest.fixture(scope="module")
def tiny_mx(tmp_path_factory):
    """A tiny llama and an MXFP4 rung of it (llm-compressor layout)."""
    _need_torch()
    import torch
    import tiny_models as TM
    from transformers import AutoModelForCausalLM
    root = tmp_path_factory.mktemp("proto")
    src = TM.llama(str(root / "src"), layers=2)
    sd = {k: v.numpy() for k, v in AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).state_dict().items()}
    rung = str(root / "rung")
    shutil.copytree(src, rung)
    os.remove(os.path.join(rung, "model.safetensors"))
    lin = {k[:-len(".weight")]: v for k, v in sd.items() if k.startswith("model.layers.") and k.endswith("_proj.weight")}
    _llmc_mxfp4_checkpoint(rung, lin, {k: ("F32", v) for k, v in sd.items() if k[:-len(".weight")] not in lin})
    ev = root / "ev.txt"
    ev.write_text("".join(chr(32 + (i * 7) % 90) for i in range(4000)))
    return {"src": src, "rung": rung, "eval": str(ev)}


def test_streamed_and_whole_hf_paths_agree_under_the_protocol(tiny_mx, monkeypatch, tmp_path):
    import torch
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    whole = V.measure_hf(tiny_mx["rung"], tiny_mx["src"], tiny_mx["eval"], device="cpu", length=64, chunks=3,
                         loader=lambda p: S.decoded_hf_model(p, "cpu", False, dtype=torch.float32))
    streamed = V.measure_stream(tiny_mx["rung"], tiny_mx["src"], tiny_mx["eval"], device="cpu", length=64, chunks=3,
                                trust_remote_code="off")
    assert whole["n"] == streamed["n"] == 3 * 32, "the second half of each chunk, nothing else"
    assert "bos-second-half" in whole["protocol"] and "bos-second-half" in streamed["protocol"]
    assert whole["kld"] == pytest.approx(streamed["kld"], rel=0.05, abs=1e-4)
    assert whole["ppl"] == pytest.approx(streamed["ppl"], rel=0.02)
    assert whole["ref_ppl"] == pytest.approx(streamed["ref_ppl"], rel=0.02)
    same = V.measure_hf(tiny_mx["src"], tiny_mx["src"], tiny_mx["eval"], device="cpu", length=64, chunks=3)
    assert same["kld"] < 1e-6 and same["top1"] == 100.0


def test_exl3_kld_uses_the_same_reference_as_the_hf_lanes(tiny_mx, monkeypatch, tmp_path):
    """measure_exl3 reads exllamav3's logits for the protocol's chunks against the SAME cached reference; a rung
    whose logits are the source's own reads KLD 0."""
    import torch
    from transformers import AutoModelForCausalLM
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    srcm = AutoModelForCausalLM.from_pretrained(tiny_mx["src"], dtype=torch.float32).eval()
    seen = []
    _fake_exllamav3(monkeypatch, lambda ids: srcm(ids.cpu()).logits.detach(), seen)
    r = V.measure_exl3("exl3-dir", tiny_mx["src"], tiny_mx["eval"], device="cpu", length=64, chunks=3)
    assert seen[0] % 256 == 0 and r["n"] == 96 and r["kld"] < 1e-5 and r["decoder"] == "exllamav3"


# ------------------------------------------------------------------------- 6. EXL3: an exact fold, a measured recipe

def test_pow2_scales_make_the_fold_exact_in_bf16():
    sc = np.array([0.013, 0.7, 1.0, 1.41, 1.42, 37.0, 99.0, 250.0], np.float32)
    p = __import__("pollard_hf_smooth").pow2_scales(sc, (1e-2, 1e2))
    assert np.all(np.log2(p) == np.round(np.log2(p))) and p.min() >= 1e-2 and p.max() <= 1e2
    inside = (sc >= 2.0 ** -6) & (sc <= 2.0 ** 6)                   # the powers of two inside the clamp
    assert np.all(np.abs(np.log2(p[inside]) - np.log2(sc[inside])) <= 0.5 + 1e-6)
    assert p[sc > 64].max() == 64.0 and p[sc < 2.0 ** -6].min() == 2.0 ** -6
    rng = np.random.default_rng(0)
    w = rng.normal(size=(16, 8)).astype(np.float32)
    bf = S.to_storage(w, "BF16")                                     # a bf16 checkpoint's weights
    w16 = (bf.astype(np.uint32) << 16).view(np.float32)
    for s in (p, sc):
        back = (S.to_storage(w16 * s[None, :], "BF16").astype(np.uint32) << 16).view(np.float32) / s[None, :]
        exact = np.array_equal(back, w16)
        assert exact == (s is p), "pow2: bit-exact; the raw ratio: rounded again"


def test_smoothing_is_an_exact_identity_on_a_tiny_model(tmp_path):
    _need_torch()
    import torch
    import tiny_models as TM
    import pollard_hf_smooth as H
    from transformers import AutoModelForCausalLM
    src = TM.llama(str(tmp_path / "src"), layers=2)
    m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    m.save_pretrained(src)                                             # a bf16 checkpoint, like the real ones
    cal = tmp_path / "cal.txt"
    cal.write_text("".join(chr(32 + (i * 11) % 90) for i in range(3000)))
    H.smooth_stream(src, str(tmp_path / "sm"), str(cal), rows=4, cols=48, log=lambda *_: None)
    rep = json.load(open(tmp_path / "sm" / "pollard-smooth-report.json"))
    assert rep["scales"] == "pow2" and rep["seams"] > 0
    a = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
    b = AutoModelForCausalLM.from_pretrained(str(tmp_path / "sm"), dtype=torch.float32).eval()
    ids = torch.tensor([[5, 17, 99, 3, 42, 7, 250, 11, 9, 80]])
    with torch.no_grad():
        la, lb = a(ids).logits, b(ids).logits
    assert torch.allclose(la, lb, atol=1e-4, rtol=1e-4), "the smoothed checkpoint computes what the source does"
    changed = [k for k, v in a.state_dict().items() if not torch.equal(v, b.state_dict()[k])]
    assert changed, "and it did fold something"


class _A:
    def __init__(self, tmp, **kw):
        self.out = str(tmp / "M-Pollard-EXL3-4.0bpw")
        self.bpw, self.head_bits, self.mtp_bits, self.recipe, self.vision_bits, self.devices = 4.0, 6, 4, None, None, "0"
        self.ab_record, self.ab_eval, self.ab_chunks, self.ab_variants = str(tmp / "ab.json"), "heldout.txt", 8, None
        self.raw_source = str(tmp / "raw")
        self.__dict__.update(kw)


def _fake_convert(made):
    def run(cmd):
        out = cmd[cmd.index("-o") + 1]
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, "config.json"), "w").write("{}")
        open(os.path.join(out, "model.safetensors"), "wb").write(b"x")
        made.append((cmd[cmd.index("-i") + 1], "-cd" in cmd, out))
        return types.SimpleNamespace(returncode=0)
    return run


def test_recipe_ab_keeps_the_measured_winner_as_the_rung(tmp_path):
    made = []
    kld = {"pollard": 0.0421, "raw-calib": 0.0390, "smooth-default": 0.0330, "plain": 0.0313}   # the 0.5B, 4.0 bpw
    a = _A(tmp_path)
    measure = lambda d: {"kld": kld[d.rsplit(".ab-", 1)[1]], "top1": 90.0}                      # noqa: E731
    win = E3.run_ab(a, "SMOOTHED", "RAW", "cal.safetensors", False, measure=measure, run=_fake_convert(made),
                    log=lambda *_: None)
    assert win == "plain"
    assert {(m, cd) for m, cd, _ in made} == {("SMOOTHED", True), ("RAW", True), ("SMOOTHED", False), ("RAW", False)}
    assert os.path.exists(os.path.join(a.out, "config.json")), "the winner IS this rung"
    assert not [p for p in os.listdir(tmp_path) if ".ab-" in p], "the losers (and work dirs) are gone"
    rec = E3.read_ab(a.ab_record)
    assert rec["winner"] == "plain" and rec["variants"]["pollard"]["kld"] == 0.0421
    # every later rung builds the winning way
    assert E3.variant_inputs(rec["winner"], "SMOOTHED", "RAW", "cal") == ("RAW", None)
    assert E3.variant_inputs("pollard", "SMOOTHED", "RAW", "cal") == ("SMOOTHED", "cal")


def test_recipe_ab_keeps_smoothing_where_it_measures_better(tmp_path):
    a = _A(tmp_path, ab_variants=["pollard", "plain"])
    kld = {"pollard": 0.020, "plain": 0.031}
    win = E3.run_ab(a, "S", "R", "c", False, measure=lambda d: {"kld": kld[d.rsplit(".ab-", 1)[1]]},
                    run=_fake_convert([]), log=lambda *_: None)
    assert win == "pollard" and set(E3.read_ab(a.ab_record)["variants"]) == {"pollard", "plain"}
    assert E3.ab_winner({"pollard": {"kld": 0.03}, "plain": {"kld": 0.03}}) == "pollard"      # a tie keeps the recipe
    assert E3.ab_winner({"pollard": {"kld": None}}) is None


def test_release_runs_the_ab_on_the_default_rung_first_and_the_rest_follow(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    src = tmp_path / "Tiny-1B"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama",
                                                 "num_hidden_layers": 2}))
    cfg = R.load_plan(None)
    cfg["lanes"] = {"exl3": {"enabled": True, "rungs": [3.0, 4.0, 5.0], "default": 4.0}}
    cfg["measure"]["toolcall"] = False
    c, lanes, steps = R.plan_steps(cfg, str(src), {"exl3"})
    builds = [s for s in steps if s.id.startswith("exl3/") and s.id.endswith("/build")]
    assert [s.rung for s in builds] == ["4.0bpw", "3.0bpw", "5.0bpw"]
    first, rest = builds[0].cmd, builds[1:]
    assert first[first.index("--recipe-choice") + 1] == "auto" and first[first.index("--raw-model") + 1] == c["hf_dir"]
    assert first[first.index("--ab-eval") + 1] == c["heldout"]
    for s in rest:
        assert "--ab-eval" not in s.cmd and s.cmd[s.cmd.index("--ab-record") + 1] == first[first.index("--ab-record") + 1]
    cfg["lanes"]["exl3"]["recipe_choice"] = "pollard"
    _, _, steps = R.plan_steps(cfg, str(src), {"exl3"})
    assert all("--recipe-choice" not in s.cmd for s in steps if s.id.endswith("/build"))


def test_recipe_ab_that_cannot_measure_keeps_the_gold_recipe_and_says_so(tmp_path):
    a = _A(tmp_path, ab_variants=["pollard", "plain"])

    def measure(d):
        raise RuntimeError("no exllamav3 forward for this architecture")
    win = E3.run_ab(a, "S", "R", "c", False, measure=measure, run=_fake_convert([]), log=lambda *_: None)
    rec = E3.read_ab(a.ab_record)
    assert win == "pollard" and rec["winner"] == "pollard" and rec["measured"] is False
    assert "no exllamav3 forward" in rec["variants"]["plain"]["error"]
