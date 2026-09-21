"""The manual knobs: does every control reach the command, and is every option real?

Two bugs motivated this file, and both were silent:

  * the Train screen bound its sliders with `bindSlider(id)` -- no recipe key -- so bits,
    samples, seqlen, head and embed bits showed values that never reached the solver;
  * its Method dropdown offered `gptq-ef`, which is not one of --method's choices, so choosing
    it built a command argparse rejects.

A control that lies about what the run will do is worse than no control.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

# the package lives at <repo>/tools/pollard_studio, so ROOT is the tools dir:
# every path below stays written as "pollard_studio/..." and the import works too
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "tools"
sys.path.insert(0, str(ROOT))

from pollard_studio import actions, manifest  # noqa: E402

JS = (ROOT / "pollard_studio/ui/app.js").read_text()
REPO = Path(os.environ.get("POLLARD_REPO", REPO_ROOT))
pytestmark = pytest.mark.skipif(not (REPO / "tools").is_dir(), reason="Pollard repo not present")


def _tools():
    return {t["module"]: t for t in manifest.build(REPO / "tools")}


def _flag(mod, name):
    for f in _tools()[mod]["flags"]:
        if name in f["names"]:
            return f
    raise AssertionError(f"{mod} has no {name}")


def _screen(name, nxt):
    return JS[JS.index(f"SCREENS.{name}"):JS.index(f"SCREENS.{nxt}")]


# ── every option offered is an option the tool accepts ──────────────────────────────────────────

@pytest.mark.parametrize("sel_id,mod,flag", [
    ("tr-method", "pollard_gptq", "--method"),
    ("tr-qmode", "pollard_gptq", "--qmode"),
    ("tr-recipe", "pollard_gptq", "--recipe"),
    ("tr-ablate", "pollard_gptq", "--ablate"),
])
def test_dropdown_options_are_real_choices(sel_id, mod, flag):
    mo = re.search(r'sel\("%s",\s*(\[.*?\]),' % re.escape(sel_id), JS, re.S)
    assert mo, f"{sel_id} not found"
    offered = {v for v in re.findall(r'"([a-z0-9_.-]*)"', mo.group(1))}
    offered.discard("")                      # "" means "leave it at the default"
    real = set(_flag(mod, flag)["choices"] or [])
    assert offered <= real, f"{sel_id} offers {sorted(offered - real)}; {flag} accepts {sorted(real)}"


def test_the_train_screen_no_longer_offers_a_method_that_does_not_exist():
    """gptq-ef was the default in the dropdown and is not a --method choice at all."""
    assert "gptq-ef" not in _screen("train", "chat")


# ── every control reaches the command ───────────────────────────────────────────────────────────

def test_no_slider_or_select_is_bound_to_nothing():
    """bindSlider("x") / bindSel("x") with no key silently discards the value."""
    keyless = re.findall(r'bind(?:Slider|Sel)\("([a-z0-9-]+)"\)', JS)
    # these three are read straight off the DOM when the request is made, by design
    allowed = {"c-max", "c-temp", "art-vol"}
    assert set(keyless) <= allowed, f"bound to nothing: {sorted(set(keyless) - allowed)}"


@pytest.mark.parametrize("key,flag", [
    ("bits", "--bits"), ("nsamples", "--nsamples"), ("seqlen", "--seqlen"),
    ("headBits", "--head-bits"), ("embedBits", "--embed-bits"),
    ("method", "--method"), ("groupsize", "--groupsize"), ("device", "--device"),
    ("calibFile", "--calib-file"), ("trainEvalFile", "--eval-file"),
    ("evalChunks", "--eval-chunks"), ("workDir", "--work-dir"),
])
def test_train_controls_reach_the_solver(key, flag):
    _, args = actions.resolve("train", {"source": "/hf", key: 7 if "its" in key or key in
                                        ("nsamples", "seqlen", "evalChunks") else "x"})
    assert flag in args, f"R.{key} never becomes {flag}"


@pytest.mark.parametrize("key,flag", [("resume", "--resume"), ("offload", "--offload")])
def test_train_switches_reach_the_solver(key, flag):
    _, args = actions.resolve("train", {"source": "/hf", key: True})
    assert flag in args


def test_a_default_choice_is_not_passed_as_noise():
    """--recipe none / --ablate none / --qmode int ARE the defaults; sending them explicitly just
    makes the command harder to read."""
    _, args = actions.resolve("train", {"source": "/hf", "qmode": "int",
                                        "recipe": "none", "ablate": "none"})
    assert args == ["--model", "/hf"]


# ── across machines ─────────────────────────────────────────────────────────────────────────────

def test_the_cluster_panel_is_on_the_build_screen():
    build = _screen("build", "monitor")
    assert "ACROSS MACHINES" in build
    # cl-tp is a sel(), which builds its own id attribute, so match the identifier itself
    for ident in ("cl-rpc", "cl-tp", "cl-vram", "cl-dev"):
        assert f'"{ident}"' in build, f"{ident} missing from the build screen"


def test_nothing_is_pinned_to_one_persons_network():
    """Placeholders must be generic. A real address baked into the UI is somebody's home lab."""
    ip = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    for f in (ROOT / "pollard_studio").rglob("*"):
        if f.suffix not in {".js", ".css", ".html", ".py", ".json"} or not f.is_file():
            continue
        for n, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            for hit in ip.findall(line):
                # 192.0.2.x is RFC 5737 TEST-NET-1: reserved for documentation and guaranteed
                # unroutable, which is why it is used to find the local interface without
                # sending anything. It is nobody's real address.
                ok = {"0.0.0.0", "127.0.0.1", "192.0.2.1"}
                assert hit in ok, f"{f.name}:{n} hardcodes {hit}"
            assert "/Users/" not in line, f"{f.name}:{n} hardcodes a home directory"


def test_rpc_pool_reaches_the_tools_that_accept_it():
    """--rpc is how a model bigger than one box gets measured and placed."""
    for act, tool in (("placement", "pollard_run"), ("sensitivity", "pollard_sensitivity"),
                      ("eval", "pollard_eval")):
        got_tool, args = actions.resolve(act, {"gguf": "/m.gguf", "rpc": "a:1,b:2"})
        assert got_tool == tool
        assert args[args.index("--rpc") + 1] == "a:1,b:2", f"{act} drops --rpc"


def test_every_tool_given_rpc_actually_declares_it():
    """Passing --rpc to a tool that does not take it builds a command argparse rejects."""
    tools = _tools()
    for act in ("placement", "sensitivity", "eval"):
        mod, args = actions.resolve(act, {"gguf": "/m.gguf", "rpc": "a:1"})
        names = {n for f in tools[mod]["flags"] for n in f["names"]}
        assert "--rpc" in names, f"{mod} does not declare --rpc"


def test_tensor_parallel_degree_reaches_vllm():
    tool, args = actions.resolve("vllmcheck", {"gguf": "/m", "tp": 16})
    assert tool == "pollard_vllm" and args[args.index("--tp") + 1] == "16"


def test_cuda_devices_only_reach_the_lane_that_takes_them():
    """--devices is an EXL3 flag. pollard-fit would reject it."""
    tool, args = actions.resolve("build", {"lane": "EXL3", "source": "/hf", "devices": "0,1"})
    assert tool == "pollard_exl3" and args[args.index("--devices") + 1] == "0,1"
    tool, args = actions.resolve("build", {"lane": "GGUF", "source": "/m.gguf", "devices": "0,1"})
    assert tool == "pollard_fit" and "--devices" not in args


# ── the flag contract, everywhere ───────────────────────────────────────────────────────────────

def test_every_flag_any_action_emits_exists_on_its_tool():
    """The whole-surface version of the check that has caught this four times now."""
    tools = _tools()
    probe = {k: "1" for k in (
        "gguf", "source", "ref", "imatrix", "out", "evalFile", "promptsFile", "calibFile",
        "trainEvalFile", "rpc", "vram", "devices", "tp", "workDir", "probeData", "tasks",
        "heatProfile", "extraArgs", "maxLen", "vs", "modelName", "dir", "body", "protect",
        "label", "sensitivity", "tier", "bpw", "mmproj", "routeCapture", "suite", "limit",
        "topK", "swapBudget", "nsamples", "method", "bits", "groupsize", "seqlen", "headBits",
        "embedBits", "device", "threads", "evalChunks", "ngl", "ctx", "chunks", "minRate",
        "kvCtx", "kvEvalFile", "top", "emb", "ftype", "probeTasks", "genTokens", "lane")}
    probe.update(qmode="ternary", recipe="handmix", ablate="attn", probeFormat="winogrande")
    bad = []
    for name in sorted(actions.ACTIONS):
        got = actions.resolve(name, dict(probe))
        if not got:
            continue
        mod, args = got
        if mod not in tools:
            continue
        declared = {n for f in tools[mod]["flags"] for n in f["names"]}
        for a in args:
            if a.startswith("--") and a not in declared:
                bad.append(f"{name} -> {mod} {a}")
    assert bad == [], "flags that do not exist on the tool: " + "; ".join(bad)


# ── all lanes, not just GGUF ────────────────────────────────────────────────────────────────────

from pollard_studio import validate as _v  # noqa: E402


def test_lane_is_read_from_the_files_not_the_name(tmp_path):
    """A directory called "whatever-GGUF" holding safetensors is not a GGUF."""
    g = tmp_path / "m.gguf"; g.write_bytes(b"x")
    assert _v.lane_of(str(g)) == "GGUF"
    d = tmp_path / "whatever-GGUF"; d.mkdir()
    (d / "model.safetensors").write_text("")
    assert _v.lane_of(str(d)) == "HF"
    q = tmp_path / "q"; q.mkdir(); (q / "quantize_config.json").write_text("{}")
    assert _v.lane_of(str(q)) == "GPTQ"
    assert _v.lane_of(str(tmp_path / "does-not-exist")) == "?"


def test_a_gguf_only_tool_given_another_lane_names_the_route(tmp_path):
    """Pollard's answer to a mismatch is a route, not a refusal -- the message has to say what
    to do, not just that it is wrong."""
    d = tmp_path / "hf"; d.mkdir(); (d / "model.safetensors").write_text("")
    msgs = _v.lane_fit("pollard_bench", {"--gguf": str(d)})
    assert len(msgs) == 1
    assert "Convert" in msgs[0] and "GGUF" in msgs[0]


def test_a_lane_agnostic_tool_is_never_blocked(tmp_path):
    """Anything taking --model reads whatever lane it is handed."""
    d = tmp_path / "hf"; d.mkdir(); (d / "model.safetensors").write_text("")
    for tool in ("pollard_gptq", "pollard_mlx", "pollard_exl3", "pollard_taskeval"):
        assert _v.lane_fit(tool, {"--model": str(d)}) == []


def test_a_matching_lane_passes_quietly(tmp_path):
    g = tmp_path / "m.gguf"; g.write_bytes(b"x")
    assert _v.lane_fit("pollard_bench", {"--gguf": str(g)}) == []


def test_an_unknown_path_does_not_raise_a_false_alarm():
    """The file may simply not exist yet -- that is --gguf's own job to report."""
    assert _v.lane_fit("pollard_bench", {"--gguf": "/not/here/yet.gguf"}) == []


# ── the allocation card is per-lane ─────────────────────────────────────────────────────────────

LANE_MOD = {"GGUF": "pollard_fit", "GPTQ": "pollard_gptq", "MLX": "pollard_mlx",
            "EXL3": "pollard_exl3", "MX": "pollard_mx"}


def _lane_alloc():
    """Parse LANE_ALLOC out of the UI: {lane: [(flag, kind, options, key), ...]}."""
    blk = JS[JS.index("const LANE_ALLOC"):JS.index("/* Render the allocation controls")]
    # split on the lane headers, so multi-line entries and formatting changes do not matter
    marks = [(lane, blk.index(f"{lane}: {{ tool:")) for lane in LANE_MOD if f"{lane}: {{ tool:" in blk]
    assert len(marks) == len(LANE_MOD), f"missing lanes: {set(LANE_MOD) - {m[0] for m in marks}}"
    marks.sort(key=lambda m: m[1])
    out = {}
    for i, (lane, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(blk)
        body = blk[start:end]
        out[lane] = re.findall(
            r'\["[^"]+",\s*"(--[a-z-]+)",\s*"(\w+)",\s*(\[[^\]]*\]|"[^"]*"|\w+),\s*"\w+",\s*"(\w+)"\]', body)
        assert out[lane], f"{lane} has no parsable fields"
    return out


def test_every_lane_offers_only_flags_its_own_tool_declares():
    """iq1_s and q6_K are llama.cpp container types; offering them on the MLX lane is a control
    the run cannot honour."""
    tools = _tools()
    for lane, rows in _lane_alloc().items():
        for flag, _kind, _opts, owner in rows:
            declared = {n for f in tools[owner]["flags"] for n in f["names"]}
            assert flag in declared, f"{lane} offers {flag}, which {owner} does not take"


@pytest.mark.parametrize("lane,flag", [("MX", "--scheme"), ("MX", "--protect-scheme"),
                                       ("GPTQ", "--qmode")])
def test_lane_dropdowns_offer_real_choices(lane, flag):
    rows = {f: (k, o, w) for f, k, o, w in _lane_alloc()[lane]}
    kind, opts, owner = rows[flag]
    assert kind == "sel"
    offered = set(re.findall(r'"([A-Za-z0-9_]+)"', opts))
    real = set(_flag(owner, flag)["choices"] or [])
    assert offered <= real, f"{lane} {flag} offers {sorted(offered - real)}, accepts {sorted(real)}"


def test_the_gguf_atoms_are_not_shown_on_other_lanes():
    alloc = _lane_alloc()
    assert any(f == "--body" for f, _, _, _ in alloc["GGUF"])
    for lane in ("MLX", "MX", "EXL3", "GPTQ"):
        flags = {f for f, _, _, _ in alloc[lane]}
        assert "--body" not in flags and "--protect" not in flags, \
            f"{lane} still offers GGUF container atoms"


def test_the_size_projection_uses_each_lanes_own_units():
    """It fell back to 32 bpw on every non-GGUF lane, which makes the projected size nonsense."""
    fn = JS[JS.index("function laneBpw"):JS.index("function project()")]
    for lane in ("GPTQ", "EXL3", "MLX", "MX"):
        assert f'case "{lane}"' in fn, f"laneBpw has no case for {lane}"
    assert "NVFP4: 4" in fn, "MX widths not derived from the scheme"


# ── one allocation, every lane ──────────────────────────────────────────────────────────────────

def test_the_measured_profile_reaches_every_lane_that_can_read_one():
    """This is the whole method: one measurement, emitted into any lane. It reached NONE of the
    non-GGUF lanes, so every one of them was built on its tool's defaults."""
    base = {"source": "/hf/model", "sensitivity": "/w/sensitivity.json",
            "recipeFile": "/w/recipe.yaml"}
    for lane, flag, val in (("MLX", "--sensitivity", "/w/sensitivity.json"),
                            ("MX", "--sensitivity", "/w/sensitivity.json"),
                            ("EXL3", "--recipe", "/w/recipe.yaml")):
        _, args = actions.resolve("build", {**base, "lane": lane})
        assert flag in args, f"{lane} never receives the allocation"
        assert args[args.index(flag) + 1] == val


def test_gptq_is_not_handed_a_profile_it_cannot_read():
    """pollard-gptq's --recipe is a NAMED MIX (none/handmix/aggr), not a path. Passing a file
    there builds a command argparse rejects."""
    _, args = actions.resolve("build", {"source": "/hf", "lane": "GPTQ",
                                        "sensitivity": "/w/s.json", "recipeFile": "/w/r.yaml"})
    assert "--sensitivity" not in args
    assert "/w/r.yaml" not in args


def test_hot_frac_is_converted_to_the_units_the_tools_take():
    """The fader is a percentage because 0..1 is unusable on a slider; the tools want 0..1."""
    _, args = actions.resolve("build", {"source": "/hf", "lane": "MLX", "hotFrac": 35})
    assert args[args.index("--hot-frac") + 1] == "0.35"
    _, args = actions.resolve("build", {"source": "/hf", "lane": "MX", "hotFrac": 0.25})
    assert args[args.index("--hot-frac") + 1] == "0.25"


def test_each_lanes_own_knobs_reach_its_own_tool():
    tools = _tools()
    cases = {
        "MLX": ({"groupSize": 64, "focusLayers": "0,1"}, ["--group-size", "--focus-layers"]),
        "MX": ({"scheme": "MXFP4", "protectScheme": "FP8"}, ["--scheme", "--protect-scheme"]),
        "EXL3": ({"bpw": 4, "headBits": 6, "devices": "0,1"},
                 ["--bpw", "--head-bits", "--devices"]),
        "GPTQ": ({"bits": 3, "groupsize": 128}, ["--bits", "--groupsize"]),
    }
    for lane, (recipe, want) in cases.items():
        mod, args = actions.resolve("build", {"source": "/hf", "lane": lane, **recipe})
        declared = {n for f in tools[mod]["flags"] for n in f["names"]}
        for flag in want:
            assert flag in args, f"{lane}: {flag} never reaches {mod}"
            assert flag in declared, f"{mod} does not declare {flag}"


# ── the repo is the source of truth, at render time ─────────────────────────────────────────────

def test_rescan_rereads_the_repos_tools_not_just_the_workspace():
    """Every form in Studio is generated from Pollard's own argparse declarations. A fix that
    lands in the repo -- a new flag, a corrected choice list -- has to reach the UI without
    restarting Studio, or the panel keeps offering yesterday's contract."""
    src = (ROOT / "pollard_studio/app.py").read_text()
    fn = src[src.index("    def rescan(self)"):src.index("    # -- execution")]
    assert "self._tools = None" in fn, "rescan leaves the tool manifest cached"
    assert "self._flagspec = None" in fn, "rescan leaves the validation spec cached"


def test_the_manifest_is_generated_from_the_repo_never_hand_written():
    """A hand-written flag list is a UI that lies the moment a tool changes."""
    src = (ROOT / "pollard_studio/app.py").read_text()
    fn = src[src.index("def load_manifest("):src.index("class Api")]
    assert "runner.REPO" in fn and "manifest.py" in fn
    assert "subprocess" in fn, "the manifest should be parsed from the repo's own tools"


def test_the_live_repo_beats_a_stale_snapshot(tmp_path, monkeypatch):
    """A manifest.json on disk was preferred over parsing the repo, so a flag added to a tool was
    rejected by the validator as unknown -- which reads to a user as Pollard refusing a build it
    should accept."""
    from pollard_studio import validate as v
    src = (ROOT / "pollard_studio/validate.py").read_text()
    fn = src[src.index("def load_manifest("):src.index("def _coerce(")]
    assert "tools_dir.is_dir()" in fn, "the repo is not consulted first"
    i_repo, i_snap = fn.index("tools_dir"), fn.index("snapshot")
    assert i_repo < i_snap, "the snapshot is still read before the repo"
    # and the live parse really does reach the repo
    live = v.load_manifest()
    if True or live:
        assert live, "nothing came back from the live repo"


def test_a_snapshot_still_works_when_there_is_no_repo(tmp_path, monkeypatch):
    """Studio installed on its own must still validate, just against what it shipped with."""
    from pollard_studio import runner, validate as v
    monkeypatch.setattr(runner, "REPO", tmp_path / "nowhere")
    snap = tmp_path / "manifest.json"
    snap.write_text(json.dumps({"tools": [
        {"module": "pollard_x", "flags": [{"names": ["--y"], "is_flag": True}]}]}))
    got = v.load_manifest(snap)
    assert got["pollard_x"]["--y"]["is_flag"] is True


# ── the app is the instrument, including in the dock ────────────────────────────────────────────

def test_the_icon_is_drawn_from_the_same_geometry_as_the_mark():
    """The dock showed a generic Python rocket. The mark in the UI is pure ellipses and circles,
    so it can be redrawn with PIL alone -- and this keeps the two in step."""
    from pollard_studio import icon
    html = (ROOT / "pollard_studio/ui/index.html").read_text()
    sym = html[html.index('id="i-bear"'):html.index("</symbol>", html.index('id="i-bear"'))]
    import re as _re
    svg_shapes = len(_re.findall(r"<(?:ellipse|circle)\b", sym))
    assert svg_shapes == len(icon._FUR) + len(icon._DARK), \
        "the icon geometry has drifted from the symbol in the UI"


def test_the_icon_renders_and_is_cached():
    from pollard_studio import icon
    p = icon.png_path(128)
    assert p is not None and p.exists(), "no icon was produced"
    assert p.stat().st_size > 2000, "the icon looks empty"
    before = p.stat().st_mtime
    assert icon.png_path(128) == p and p.stat().st_mtime == before, "regenerated needlessly"


def test_setting_the_icon_never_takes_the_app_down():
    """An icon is chrome. If the platform will not take it, the app still has to start."""
    from pollard_studio import icon
    got = icon.apply(None)
    assert isinstance(got, str) and got, "apply() should report what it did"


def test_startup_sets_the_icon():
    src = (ROOT / "pollard_studio/app.py").read_text()
    assert "_icon.apply(" in src
    assert src.index("_icon.apply(") < src.index("webview.start("), \
        "the icon must be set before the event loop starts"
