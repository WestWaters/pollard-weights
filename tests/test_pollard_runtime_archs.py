"""`pollard` runtime class: an architecture only a runtime patch POLLARD SHIPS implements.

What it protects: clef-omni loads in llama.cpp only through runtime-patches/llama.cpp-clef-omni.patch (upstream PR
pending). Reading the patched runtime/llama.cpp beside the checkout called it "stock" (the tree knows it); reading
from any other directory fell back to the snapshot and called it "no upstream support and no fork on record". Both
are false. The truthful answer is its own class -- `pollard` -- that the gate accepts only when the release
built/verified the rung with Pollard's runtime, and that the card states plainly: "Needs llama.cpp with Pollard's
`clef-omni` patch (runtime-patches/...; upstream PR pending)", with how to build it. Third-party forks (k2-horizon)
stay `fork`; stock stays stock.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_ggufcompat as gc  # noqa: E402
import pollard_verify as V  # noqa: E402


def _gguf(path, arch, types=(12,)):
    a = arch.encode()
    with open(path, "wb") as fh:
        fh.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(types)) + struct.pack("<Q", 1))
        k = b"general.architecture"
        fh.write(struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(a)) + a)
        for i, t in enumerate(types):
            n = f"blk.{i}.weight".encode()
            fh.write(struct.pack("<Q", len(n)) + n + struct.pack("<I", 1) + struct.pack("<Q", 32)
                     + struct.pack("<I", t) + struct.pack("<Q", 0))
    return str(path)


#: the checks that read the SHIPPED clef-omni patch run wherever the tool finds runtime-patches/ -- the checkout, or
#: the copy every install (pip, nix) ships as data; the synthetic-patch-dir test covers the logic everywhere.
shipped_patches = pytest.mark.skipif(not (Path(gc.PATCH_DIR) / "llama.cpp-clef-omni.patch").exists(),
                                     reason="no runtime-patches/ in this checkout or install")


@pytest.fixture
def no_upstream(monkeypatch):
    """Upstream unreachable (an empty cache reads as 'could not check'), and no network call made."""
    monkeypatch.setattr(gc, "_upstream_cache", set())
    gc._pollard_cache.clear()
    yield
    gc._pollard_cache.clear()


def _fake_tree(root, extra=()):
    """A repo checkout with a vendored runtime/llama.cpp whose arch list carries our patched arch too."""
    src = root / "runtime" / "llama.cpp" / "src"
    src.mkdir(parents=True)
    names = sorted(gc.STOCK_ARCHS) + list(extra)
    body = "\n".join(f'    {{ LLM_ARCH_X{i}, "{n}" }},' for i, n in enumerate(names))
    (src / "llama-arch.cpp").write_text("static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {\n"
                                        + body + "\n};\n")


@shipped_patches
def test_shipped_clef_omni_patch_is_a_pollard_arch_from_any_cwd(tmp_path, monkeypatch, no_upstream):
    monkeypatch.chdir(tmp_path)                       # nowhere near the checkout
    pol = gc.pollard_archs()
    assert "clef-omni" in pol, pol
    info = pol["clef-omni"]
    assert info["patch"] == "llama.cpp-clef-omni" and info["engine"] == "llama.cpp"
    assert info["url"].endswith("runtime-patches/llama.cpp-clef-omni.patch")
    assert "k2-horizon" not in pol, "a port of a third-party fork's arch is not Pollard support"
    v, why = gc.arch_support("clef-omni")
    assert (v, why) == ("pollard", "llama.cpp-clef-omni; upstream PR pending")
    rt, fo = gc.runtime_of(_gguf(tmp_path / "c.gguf", "clef-omni"))
    assert rt == "pollard" and fo["pollard_patch"] == "llama.cpp-clef-omni"
    assert fo["pollard_needs"] == ("Needs llama.cpp with Pollard's `clef-omni` patch "
                                   "(runtime-patches/llama.cpp-clef-omni.patch; upstream PR pending)")


@shipped_patches
def test_stock_archs_reads_the_repo_tree_not_the_cwd_and_drops_our_patched_archs(tmp_path, monkeypatch,
                                                                                  no_upstream):
    repo = tmp_path / "repo"
    _fake_tree(repo, extra=("clef-omni", "brand-new-local"))
    monkeypatch.setattr(gc, "REPO_ROOT", str(repo))
    elsewhere = tmp_path / "models" / "Org__M"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    known = gc.stock_archs()
    assert "brand-new-local" in known, "the vendored tree beside the checkout was not read"
    assert "clef-omni" not in known, "an arch only our patch adds is not stock"
    assert "clef-omni" in gc.engine_archs()
    assert gc.arch_support("clef-omni")[0] == "pollard"
    assert gc.arch_support("brand-new-local")[0] == "stock"


def test_stock_and_fork_are_unchanged(tmp_path, monkeypatch, no_upstream):
    monkeypatch.chdir(tmp_path)
    assert gc.arch_support("llama")[0] == "stock"
    assert gc.runtime_of(_gguf(tmp_path / "l.gguf", "llama"))[0] == "stock"
    monkeypatch.setattr(gc, "_upstream_cache", {"llama", "qwen3"})
    v, why = gc.arch_support("k2-horizon", local={"llama"})
    assert v == "fork" and "MBZUAI" in why
    rt, fo = gc.runtime_of(_gguf(tmp_path / "k.gguf", "k2-horizon"), archs={"llama"})
    assert rt == "fork" and "MBZUAI" in fo["supported_by"]


@shipped_patches
def test_upstream_merging_the_arch_wins_over_pollard(no_upstream, monkeypatch):
    monkeypatch.setattr(gc, "_upstream_cache", set(gc.STOCK_ARCHS) | {"clef-omni"})
    assert gc.arch_support("clef-omni", local=set(gc.STOCK_ARCHS))[0] == "newer"
    assert gc.arch_support("clef-omni", local=set(gc.STOCK_ARCHS) | {"clef-omni"})[0] == "stock"


def test_patch_rules_engine_listed_and_upstream_remote_only(tmp_path):
    d = tmp_path / "rp"
    d.mkdir()
    arch_hunk = ('diff --git a/src/llama-arch.cpp b/src/llama-arch.cpp\n--- a/src/llama-arch.cpp\n'
                 '+++ b/src/llama-arch.cpp\n@@ -1,2 +1,3 @@\n     {{ LLM_ARCH_QWEN3, "qwen3" }},\n'
                 '+    {{ LLM_ARCH_{u}, "{a}" }},\n')
    for name, remote, arch in (("llama.cpp-ours", "https://github.com/ggml-org/llama.cpp", "ours-arch"),
                               ("llama.cpp-unlisted", "https://github.com/ggml-org/llama.cpp", "unlisted-arch"),
                               ("ik_llama.cpp-port", "https://github.com/ikawrakow/ik_llama.cpp", "port-arch")):
        (d / f"{name}.patch").write_text(arch_hunk.format(u=arch.upper().replace("-", "_"), a=arch))
        (d / f"{name}.json").write_text(json.dumps({"name": name, "remote": remote, "base_commit": "abc1234",
                                                    "upstream": {"pr": "https://github.com/ggml-org/llama.cpp/pull/1"}
                                                    if name == "llama.cpp-ours" else {}}))
    (d / "engines.json").write_text(json.dumps({
        "llama.cpp": {"url": "https://github.com/ggml-org/llama.cpp", "patches": ["llama.cpp-ours"]},
        "ik_llama.cpp": {"url": "https://github.com/ikawrakow/ik_llama.cpp", "patches": ["ik_llama.cpp-port"]}}))
    pol = gc.pollard_archs(str(d))
    assert set(pol) == {"ours-arch"}, pol
    assert pol["ours-arch"]["upstream"] == "upstream PR https://github.com/ggml-org/llama.cpp/pull/1"
    lines = gc.pollard_build_lines("ours-arch", pol["ours-arch"])
    assert "pollard-runtime --update" in lines and "git checkout abc1234" in lines
    assert "git apply ../pollard-weights/runtime-patches/llama.cpp-ours.patch" in lines


@pytest.mark.parametrize("want,runtime,engine,ok", [
    ("stock", "stock", None, True),
    ("stock", "pollard", "pollard", True),       # built/verified with Pollard's runtime: passes, card says so
    ("stock", "pollard", None, False),           # no evidence: the card and the runtime could disagree
    ("stock", "fork", None, False),
    ("stock", "ik_llama", None, False),
    ("stock", "newer", None, False),
    ("ik_llama", "pollard", "pollard", True),
    ("ik_llama", "pollard", None, False),
    ("ik_llama", "ik_llama", None, True),
])
def test_runtime_gate_matrix(want, runtime, engine, ok):
    m = {"runtime": runtime}
    if engine:
        m["runtime_engine"] = engine
    res = V.evaluate_gate(m, {"runtime": want})
    assert res["pass"] is ok, res
    if runtime == "pollard":
        assert "Pollard's runtime" in res["checks"][0]["why"]


def test_runtime_engine_evidence(tmp_path, monkeypatch, no_upstream):
    p = _gguf(tmp_path / "c.gguf", "clef-omni")
    assert V.pollard_runtime_engine(p, {"kld": 0.01}) == "pollard"            # the measurement loaded it here
    assert V.pollard_runtime_engine(p, {"decision_agreement": 99.0}) == "pollard"
    monkeypatch.setattr(gc, "engine_archs", lambda *a, **k: None)
    assert V.pollard_runtime_engine(p, {}) is None
    monkeypatch.setattr(gc, "engine_archs", lambda *a, **k: {"clef-omni", "llama"})
    assert V.pollard_runtime_engine(p, {}) == "pollard"
    assert V.metrics_from_results({"Q4_K_M": {"runtime_engine": "pollard"}}, "Q4_K_M")["runtime_engine"] == "pollard"


def _card(tmp_path, monkeypatch, arch, results=None):
    import pollard_card as C
    home = tmp_path / "home"
    src = tmp_path / "Org__M"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 1024,
                                                 "num_hidden_layers": 4, "vocab_size": 1000}))
    (src / "README.md").write_text("---\nlicense: mit\n---\n")
    d = home / "models" / "Org__M"
    d.mkdir(parents=True)
    builds = []
    for t, n in (("Q6_K", 900_000_000), ("Q4_K_M", 600_000_000)):
        p = _gguf(d / f"M-Pollard-{t}.gguf", arch)
        builds.append({"name": f"M-Pollard-{t}.gguf", "lane": "gguf", "tag": t, "path": p, "bytes": n})
    (d / "MANIFEST.json").write_text(json.dumps({"model": "Org/M", "builds": builds}))
    res = tmp_path / "results.json"
    res.write_text(json.dumps(results or {f"M-Pollard-{t}.gguf": {"ppl": 10.1, "kld": 0.01} for t in ("Q6_K", "Q4_K_M")}))
    out = tmp_path / "README.md"
    monkeypatch.setenv("POLLARD_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["pollard-card", "--model", str(src), "--builds-from", "Org/M",
                                      "--repo", "PollardWeights/M-Pollard", "--results", str(res), "--out", str(out)])
    C.main()
    return out.read_text()


CLEF_LINE = ("Needs llama.cpp with Pollard's `clef-omni` patch ([runtime-patches/llama.cpp-clef-omni.patch]"
             "(https://github.com/WestWaters/pollard-weights/blob/main/runtime-patches/llama.cpp-clef-omni.patch); "
             "upstream PR pending)")


@shipped_patches
def test_card_names_the_pollard_patch_and_never_says_stock(tmp_path, monkeypatch, no_upstream):
    md = _card(tmp_path, monkeypatch, "clef-omni")
    assert f"**{CLEF_LINE}.**" in md
    assert "pollard-runtime --update" in md and "git checkout b76b133" in md
    assert "git apply ../pollard-weights/runtime-patches/llama.cpp-clef-omni.patch" in md
    assert "Standard GGUF" not in md and "run in stock llama.cpp" not in md
    assert "any llama.cpp" not in md and "ollama run hf.co/" not in md
    assert "no upstream support and no fork on record" not in md
    assert "*(needs llama.cpp + Pollard patch)*" in md


def test_card_for_a_stock_arch_is_unchanged(tmp_path, monkeypatch, no_upstream):
    md = _card(tmp_path, monkeypatch, "qwen3")
    assert "runtime-patches/" not in md and "Pollard patch" not in md
    assert "Standard GGUF -- every file here runs in stock llama.cpp" in md
