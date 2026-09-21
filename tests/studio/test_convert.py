"""Lane conversion: a route, never a refusal, and never a requantize.

Two invariants. Pollard's allocation (sensitivity.json) is measured in KL cost, not in any lane's
atoms, so moving between lanes is re-emission from source -- not translating one quantized
representation into another, which compounds error. And when something is missing the planner says
which tool produces it and puts that step first, because "you cannot do that" is never as useful
as "here is the order to do it in".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# the package lives at <repo>/tools/pollard_studio, so ROOT is the tools dir:
# every path below stays written as "pollard_studio/..." and the import works too
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "tools"
sys.path.insert(0, str(ROOT))

from pollard_studio import convert, workspace   # noqa: E402


@pytest.mark.parametrize("target", sorted(convert.LANES))
def test_every_lane_produces_a_route(target):
    p = convert.plan("/nowhere/some-model-Pollard-IQ3_S.gguf", target)
    assert p["ok"] and p["steps"], f"{target} produced no route"


@pytest.mark.parametrize("target", sorted(convert.LANES))
def test_no_route_ever_requantizes(target):
    p = convert.plan("/nowhere/some-model-Pollard-IQ3_S.gguf", target)
    assert p["requantize"] is False
    blob = json.dumps(p).lower()
    assert "allow-requantize" not in blob and "allow_requantize" not in blob


def test_a_missing_source_is_a_first_step_not_a_failure():
    """The whole point: no dead ends."""
    p = convert.plan("/nowhere/thing.gguf", "MLX")
    assert p["ok"], "a missing source must not fail the plan"
    assert p["steps"][0]["blocking"] is True
    assert "locate" in p["steps"][0]["why"].lower()


def test_an_unknown_lane_still_says_what_is_available():
    p = convert.plan("/nowhere/thing.gguf", "TPU")
    assert "GGUF" in p["guidance"] and "MLX" in p["guidance"]


def test_the_last_step_is_always_the_emitter():
    for target, spec in convert.LANES.items():
        p = convert.plan("/nowhere/thing.gguf", target)
        assert p["steps"][-1]["tool"] == spec["tool"], target


def test_gguf_target_from_hf_source_converts_first(tmp_path):
    """pollard-fit starts from a GGUF, so an HF source needs one conversion first."""
    home = tmp_path / "pollard"
    (home / "downloads/Org__My-Model").mkdir(parents=True)
    (home / "downloads/Org__My-Model/model.safetensors").write_bytes(b"x")
    (home / "rungs").mkdir()
    b = home / "rungs/My-Model-Pollard-IQ3_S.gguf"
    b.write_bytes(b"x")
    p = convert.plan(b, "GGUF")
    tools = [s["tool"] for s in p["steps"]]
    assert "pollard_convert" in tools and tools[-1] == "pollard_fit"


def test_tensor_types_sidecar_is_not_treated_as_portable(tmp_path):
    """It records ggml atom names -- what THIS build was given. It does not carry to another lane."""
    home = tmp_path / "pollard"
    (home / "downloads/Org__My-Model").mkdir(parents=True)
    (home / "downloads/Org__My-Model/model.safetensors").write_bytes(b"x")
    (home / "rungs").mkdir()
    b = home / "rungs/My-Model-Pollard-IQ3_S.gguf"
    b.write_bytes(b"x")
    b.with_suffix(b.suffix + ".tensor-types.txt").write_text("ffn_down=iq4_xs\n")
    prof = convert.find_profile(b)
    assert prof["found"] and prof["kind"] == "tensor-types" and prof["portable"] is False
    p = convert.plan(b, "MLX")
    assert any(s["tool"] in {t for t, _ in convert.PROFILERS} for s in p["steps"]), \
        "must offer to measure a portable profile first"


def test_a_portable_profile_is_passed_to_the_emitter(tmp_path):
    home = tmp_path / "pollard"
    (home / "downloads/Org__My-Model").mkdir(parents=True)
    (home / "downloads/Org__My-Model/model.safetensors").write_bytes(b"x")
    (home / "rungs").mkdir()
    b = home / "rungs/My-Model-Pollard-IQ3_S.gguf"
    b.write_bytes(b"x")
    (home / "rungs/sensitivity.json").write_text("{}")
    p = convert.plan(b, "MLX")
    emit = p["steps"][-1]
    assert "--sensitivity" in emit["args"], "the measured allocation should be carried forward"


def test_recipe_lanes_explain_the_translation():
    """GPTQ and EXL3 take a recipe, not the raw profile -- the user should know why."""
    for target in ("GPTQ", "EXL3"):
        p = convert.plan("/nowhere/thing.gguf", target)
        assert "recipe" in p["steps"][-1]["why"].lower(), target


def test_source_lookup_finds_a_real_one():
    root = workspace.home()
    b = next((root / "rungs").glob("Qwen2.5-0.5B-Instruct-Pollard-*.gguf"), None)
    if b is None:
        pytest.skip("no reference build in the workspace")
    got = convert.find_source(b)
    assert got["found"] and "0.5B" in got["path"]


def test_lane_table_matches_the_tools_that_exist():
    mf = ROOT / "manifest.json"
    if not mf.exists():
        pytest.skip("no manifest")
    known = {t["module"] for t in json.loads(mf.read_text())["tools"]}
    for target, spec in convert.LANES.items():
        assert spec["tool"] in known, f"{target} points at a tool that does not exist"
    for tool, _ in convert.PROFILERS:
        assert tool in known, tool


# ── a source does not have to be f16 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bpw,expect", [
    (16.0, "ideal"), (8.5, "excellent"), (8.0, "excellent"),
    (6.6, "usable"), (4.25, "lossy"), (2.1, "poor"),
])
def test_source_tiers_land_where_they_should(bpw, expect):
    """Q8-class is excellent and should be used rather than re-downloading the f16. Q6_K is 6.6
    bpw, which is Q6-class -- honest, not flattering."""
    tier = next(t for floor, t, _ in convert.SOURCE_TIERS if bpw >= floor)
    assert tier == expect, f"{bpw} bpw -> {tier}"


def test_a_high_precision_source_is_never_refused():
    """Refusing a Q8 source would send someone off to re-download 300 GB for nothing."""
    root = workspace.home()
    b = next((root / "rungs").glob("*Q6_K.gguf"), None)
    if b is None:
        pytest.skip("no high-precision build in the workspace")
    q = convert.source_quality(b)
    assert q["tier"] not in ("poor", "unknown"), q
    assert convert.plan(b, "MLX")["ok"] is True


def test_a_low_bit_source_warns_but_still_routes():
    root = workspace.home()
    low = None
    for p in root.rglob("*.gguf"):
        bpw = convert.source_quality(p).get("bpw")
        if bpw is not None and bpw < 4:
            low = p
            break
    if low is None:
        pytest.skip("no sub-4-bit build in the workspace")
    q = convert.source_quality(low)
    assert q["tier"] == "poor"
    assert "compounds" in q["note"]
    assert "will still run" in q["note"], "a warning, not a refusal"
    assert convert.plan(low, "MLX")["ok"] is True


def test_source_quality_is_measured_not_guessed_from_the_name(tmp_path):
    """A file called 'f16' that is really 4-bit must report 4-bit."""
    d = tmp_path / "definitely-f16-honest"
    d.mkdir()
    import struct as st
    hdr = json.dumps({"w": {"dtype": "I32", "shape": [64, 64],
                            "data_offsets": [0, 64 * 64 * 4]}}).encode()
    (d / "model.safetensors").write_bytes(st.pack("<Q", len(hdr)) + hdr + b"\0" * (64 * 64 * 4))
    q = convert.source_quality(d)
    assert q["bpw"] is not None and q["bpw"] > 15, "should report what the bytes say"


def test_unreadable_source_still_routes():
    q = convert.source_quality("/nowhere/at/all")
    assert q["tier"] == "unknown" and "still be used" in q["note"]


@pytest.mark.parametrize("tier_floor,tier,_note", convert.SOURCE_TIERS)
def test_every_tier_explains_itself(tier_floor, tier, _note):
    assert _note and len(_note) > 30, f"{tier} has no useful explanation"


# ── what the UI is allowed to show ──────────────────────────────────────────────────────────────
def test_no_screen_prints_a_home_directory():
    """A path with someone's username in it ends up in every screenshot they paste anywhere."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    assert "function short(" in js, "there should be one place that shortens paths"
    # anywhere a raw workspace/repo path is INTERPOLATED it must go through short().
    # short()'s own body compares against S.home, which is not output.
    import re
    body_start = js.index("function short(")
    body_end = js.index("const shortArgs")
    for var in ("S.home", "S.repo"):
        for m in re.finditer(re.escape("${" + var), js):
            if body_start <= m.start() < body_end:
                continue
            ctx = js[max(0, m.start() - 40):m.start() + 40]
            assert "short(" in ctx, f"{var} interpolated raw: ...{ctx}..."


def test_the_shortener_handles_both_platforms():
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    body = js[js.index("function short("):js.index("const shortArgs")]
    assert "POLLARD_HOME" in body, "a workspace path should read as $POLLARD_HOME"
    assert "Users|home" in body, "posix homes"
    assert "A-Z" in body and "Users" in body, "windows homes"


def test_hero_banners_are_gone():
    """Screens should match each other; a page-title banner on six of eleven does not."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    assert 'class="page"' not in js


def test_slider_fill_accounts_for_the_thumb():
    """The thumb centre travels width-minus-thumb, so a raw percentage puts the fill ahead of it."""
    css = (ROOT / "pollard_studio/ui/app.css").read_text()
    assert "--thumb" in css
    assert "100% - var(--thumb)" in css, "fill should be computed against the real travel"
    assert "var(--pct" not in css, "the old uncorrected variable should be gone"


def test_slider_variable_is_unitless_so_calc_works():
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    assert '"--p", ((v - r.min)' in js, "must set a unitless --p, not a percentage string"
    assert '--p:${((val - min)' in js


# ── layout invariants ───────────────────────────────────────────────────────────────────────────
def test_the_deck_row_fits_its_own_contents():
    """The deck held 80px of LCD + 6px gap + 26px spectrum. A 116px row clipped the spectrum
    against the BUILD/GATE buttons."""
    css = (ROOT / "pollard_studio/ui/app.css").read_text()
    import re
    m = re.search(r"#app \{[^}]*grid-template-rows: (\d+)px 1fr (\d+)px", css)
    assert m, "could not read the app grid rows"
    deck_h = int(m.group(2))
    lcd = 80
    spec = int(re.search(r"\.spec \{[^}]*height: (\d+)px", css).group(1))
    gap = int(re.search(r"\.spec \{[^}]*margin-top: (\d+)px", css).group(1))
    pad = 2 * int(re.search(r"#deck \{[^}]*padding: (\d+)px", css).group(1))
    assert deck_h >= lcd + spec + gap + pad, \
        f"deck row {deck_h}px cannot hold {lcd + spec + gap + pad}px of content"


def test_an_idle_output_panel_still_reads_out():
    """A blank instrument panel looks broken. At rest it should say what is loaded."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    assert "function idleReadout()" in js
    assert "nothing has been run yet" not in js, "the bare one-liner should be gone"
    body = js[js.index("function idleReadout()"):]
    body = body[:body.index("\n}")]
    for field in ("model", "build", "lane", "runtime", "state"):
        assert f'"{field}"' in body, f"idle readout should report {field}"


def test_the_recipe_panel_reads_the_file_not_a_sidecar():
    """A .tensor-types.txt is often absent. The types are in the build itself, so 'not recorded'
    was never the right answer."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    assert "function renderRecipe()" in js, "the function must exist, not just be called"
    assert "renderRecipe();" in js, "and it must be called"
    assert "no .tensor-types.txt sidecar" not in js.lower()
    body = js[js.index("function renderRecipe()"):]
    body = body[:body.index("\nfunction ")]
    assert "b.types" in body, "it should read the counted types from the header"


def test_chart_panels_get_the_same_scanline_as_the_other_lcds():
    """Eval, Bench, Train, Ladder and Monitor render charts in .chartbox. Without the scanline
    they looked like a different instrument to Build."""
    css = (ROOT / "pollard_studio/ui/app.css").read_text()
    assert ".chartbox::before" in css
    block = css[css.index(".chartbox::before"):]
    block = block[:block.index("}")]
    assert "repeating-linear-gradient" in block


def test_the_rotated_fader_cannot_drift_off_its_rail():
    """The input is rotated -90deg, so its vertical axis becomes the horizontal one on screen.
    Unequal heights put the cap 1px off, and an asymmetric shadow pulled it further."""
    css = (ROOT / "pollard_studio/ui/app.css").read_text()
    seg = css[css.index(".fader input[type=range] {"):]
    seg = seg[:seg.index(".fader .fl")]
    import re
    heights = re.findall(r"height: (\d+)px", seg)
    assert len(set(heights)) == 1, f"input, track and thumb must share a height, got {heights}"
    assert "margin-top: 0" in seg, "no residual centring for the UA to resolve"
    shadow = re.search(r"box-shadow: (0|\d+px) (0|\d+px)", seg)
    assert shadow, f"no box-shadow found on the thumb: {seg[-300:]}"
    x, y = shadow.group(1), shadow.group(2)
    assert x in ("0", "0px") and y in ("0", "0px"), \
        f"an offset shadow ({x} {y}) becomes a SIDEWAYS shadow once rotated"


def test_the_layout_collapses_instead_of_clipping():
    css = (ROOT / "pollard_studio/ui/app.css").read_text()
    assert "@media (max-width: 1400px)" in css
    assert "@media (max-width: 1100px)" in css
    assert "@media (max-width: 820px)" in css
    assert "min-width: 0" in css, "grid children need this or they refuse to shrink"


# ── defaults come from the machine, not from a literal ──────────────────────────────────────────
def test_the_bridge_reads_the_real_machine():
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from pollard_studio import app
    hw = app.Api().hardware()
    assert hw["cpus"] and hw["cpus"] > 0
    assert hw["ram_gb"] is None or hw["ram_gb"] > 0
    if hw["ram_gb"]:
        assert 0 < hw["suggest_target_gb"] <= hw["ram_gb"], "a target above total RAM is useless"
        assert hw["suggest_reserve_gb"] >= 1


def test_no_screen_hardcodes_a_memory_target():
    """16 GB is meaningful on a 16 GB laptop and meaningless everywhere else."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    assert "ramMax()" in js, "the RAM slider should size itself to the machine"
    import re
    for m in re.finditer(r'id="(?:cv-ram|f-ram)"[^>]*max="([^"]+)"', js):
        assert "ramMax" in m.group(1), f"RAM slider still has a literal max: {m.group(1)}"


def test_the_hardware_suggestion_is_applied_at_boot():
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    assert "await bridge().hardware()" in js
    assert "R.ram = S.hw.suggest_target_gb" in js
    assert "R.threads = Math.max(1, Math.round(S.hw.cpus * 0.6))" in js, \
        "thread default should leave the machine usable"


def test_the_panels_read_the_selected_build_not_a_constant():
    """Switching model must change what the readouts say."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    body = js[js.index("function refreshBuild("):]
    body = body[:body.index("\nasync function updateCmd")]
    for src in ("p.ref.tag", "p.params", "p.ref.blocks", "p.gb", "p.bpw"):
        assert src in body, f"the allocation readout does not use {src}"


# ── version is Pollard's, not a second number ───────────────────────────────────────────────────
def test_studio_version_tracks_pollard_weights():
    """A separate Studio version is only a thing to forget to bump."""
    import sys as _s
    _s.path.insert(0, str(ROOT))
    import pollard_studio
    repo = Path(convert.__file__).resolve().parents[2]
    import os
    repo = Path(os.environ.get("POLLARD_REPO", REPO_ROOT))
    if not (repo / "pyproject.toml").exists():
        pytest.skip("no Pollard repo to compare against")
    import re
    want = None
    for line in (repo / "pyproject.toml").read_text().splitlines():
        m = re.match(r'\s*version\s*=\s*["\']([^"\']+)["\']', line)
        if m:
            want = m.group(1)
            break
    assert pollard_studio.__version__ == want, \
        f"studio says {pollard_studio.__version__}, Pollard says {want}"


def test_the_version_is_shown_in_the_panel():
    html = (ROOT / "pollard_studio/ui/index.html").read_text()
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    assert 'id="ver"' in html
    assert '$("#ver")' in js and "S.version" in js


def test_a_missing_repo_does_not_crash_the_import(monkeypatch, tmp_path):
    """Studio ships WITH Pollard now, so its version is the installed distribution's -- the same
    number pip knows, not one derived from a path that may not exist. Pointing POLLARD_REPO at
    nothing must still import cleanly and still report a real version."""
    import importlib

    import pollard_studio
    monkeypatch.setenv("POLLARD_REPO", str(tmp_path / "nowhere"))
    importlib.reload(pollard_studio)
    assert pollard_studio.__version__
    assert pollard_studio.__version__ != pollard_studio.FALLBACK, \
        "the installed distribution should answer even with no checkout"



def test_advanced_is_generated_from_the_manifest():
    """A hand-written list of 562 flags is wrong the day someone adds one. An audit found 122
    knobs across 21 tools missing from the hand-written version."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    assert "function flagForm(" in js
    assert "ADV_SECTIONS.map(" in js
    assert "toolBlock(m," in js


def test_every_advanced_section_names_tools_that_exist():
    import json as _j
    import re
    mf = ROOT / "manifest.json"
    if not mf.exists():
        pytest.skip("no manifest")
    known = {t["module"] for t in _j.loads(mf.read_text())["tools"]}
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    block = js[js.index("const ADV_SECTIONS = ["):js.index("SCREENS.advanced")]
    named = set(re.findall(r'"(pollard_\w+)"', block))
    assert named, "no tools listed"
    assert not (named - known), f"Advanced names tools that do not exist: {sorted(named - known)}"


def test_advanced_covers_every_knob_of_the_tools_it_lists():
    """The point of generating it: a section cannot be missing a flag."""
    import json as _j
    import re
    mf = ROOT / "manifest.json"
    if not mf.exists():
        pytest.skip("no manifest")
    tools = {t["module"]: t for t in _j.loads(mf.read_text())["tools"]}
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    block = js[js.index("const ADV_SECTIONS = ["):js.index("SCREENS.advanced")]
    named = re.findall(r'"(pollard_\w+)"', block)
    total = sum(len(tools[m]["flags"]) for m in named if m in tools)
    assert total > 150, f"only {total} flags reachable from Advanced; generation is not working"


def test_every_section_explains_why_it_exists():
    import re
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    block = js[js.index("const ADV_SECTIONS = ["):js.index("SCREENS.advanced")]
    whys = re.findall(r'why:\s*"', block)
    titles = re.findall(r'title:\s*"', block)
    assert len(whys) == len(titles), "a section without a reason is a section nobody will open"


def test_diagnostics_live_on_doctor_not_advanced():
    """They answer questions about the ENVIRONMENT, not the build, so they belong with the other
    environment checks."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    doc = js[js.index("SCREENS.doctor"):js.index("SCREENS.brains")]
    adv = js[js.index("const ADV_SECTIONS"):js.index("SCREENS.tools")]
    for a in ("health", "runtime", "refcheck", "modelkind", "archfp", "envmatch", "calc", "stop"):
        # the card generates its buttons from an array, so look for the action NAME
        assert f'["{a}",' in doc, f"{a} should be on Doctor"
        assert f"act('{a}')" not in adv, f"{a} is still on Advanced"


def test_the_diagnostics_card_is_not_duplicated():
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    assert js.count('DIAGNOSTICS<span class="rt">is the machine') == 1


def test_the_version_badge_is_actually_visible():
    """It rendered at 9px gold-on-cream, which is not the same as being visible."""
    css = (ROOT / "pollard_studio/ui/app.css").read_text()
    rule = css[css.index(".brand .ver {"):]
    rule = rule[:rule.index("}")]
    assert "background" in rule, "a badge, not faint text"
    assert "color: #3a2c08" in rule or "color:#3a2c08" in rule


def test_every_advanced_tool_exists_in_the_manifest():
    """A typo'd module name renders an empty section rather than erroring, so it has to be
    caught here."""
    import re
    from pollard_studio import manifest
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    adv = js[js.index("const ADV_SECTIONS"):js.index("SCREENS.advanced")]
    named = set(re.findall(r'"(pollard_[a-z0-9_]+)"', adv))
    import os
    repo = Path(os.environ.get("POLLARD_REPO", REPO_ROOT))
    if not (repo / "tools").is_dir():
        pytest.skip("Pollard repo not present")
    have = {t["module"] for t in manifest.build(repo / "tools")}
    assert named, "ADV_SECTIONS names no tools"
    assert named <= have, f"Advanced names tools that do not exist: {sorted(named - have)}"


def test_advanced_covers_the_manual_allocation_knobs():
    """These are the tools a user reaches for to hand-tune a build; they were reachable only from
    the generic Tools list before."""
    import re
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    adv = js[js.index("const ADV_SECTIONS"):js.index("SCREENS.advanced")]
    for mod in ("pollard_fragile", "pollard_errsrc", "pollard_errtype", "pollard_fit_dit",
                "pollard_run", "pollard_exl3_band", "pollard_onboard", "pollard_probes",
                "pollard_serve_eval"):
        assert f'"{mod}"' in adv, f"{mod} is not curated onto Advanced"


def test_brain_tools_can_actually_run_not_just_help():
    """The brains screen used to offer only a --help button for three tools, and did not mention
    connectome or flybrain at all."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    steps = js[js.index("const BRAIN_STEPS"):js.index("SCREENS.brains")]
    for mod in ("pollard_connectome", "pollard_flybrain", "pollard_brainattach",
                "pollard_brainverify", "pollard_brainlanes"):
        assert f'"{mod}"' in steps, f"{mod} missing from the brains screen"
    screen = js[js.index("SCREENS.brains"):js.index("SCREENS.publish")]
    assert "toolBlock(" in screen, "brains screen has no runnable form"


def _curated_modules():
    """Modules a user can reach with a purpose-built UI -- named directly by a generated form, or
    run by an action key the UI invokes. The Tools screen can run all 64 generically; this is
    about whether the tool has a HOME."""
    import os, re
    from pollard_studio import actions, manifest
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    strings = set(re.findall(r"['\"]([a-z][a-z0-9_:-]*)['\"]", js))
    named = set(re.findall(r'"(pollard_[a-z0-9_]+)"', js))
    # ask the REAL registry what each action key runs, rather than inferring it
    by_key = set()
    for key in actions.ACTIONS:
        if key not in strings:
            continue
        try:
            got = actions.resolve(key, {})
        except Exception:
            continue
        if got:
            by_key.add(got[0])
    repo = Path(os.environ.get("POLLARD_REPO", REPO_ROOT))
    have = {t["module"] for t in manifest.build(repo / "tools")}
    return have, (named | by_key) & have


def test_every_tool_has_a_home_screen():
    """Mario: 'i dont think everything that is manually advanced in pollard is in the advanced
    screen'. Reaching a tool only through the generic Tools list is not the same as it having a
    place, so this asserts each one is curated somewhere."""
    import os
    repo = Path(os.environ.get("POLLARD_REPO", REPO_ROOT))
    if not (repo / "tools").is_dir():
        pytest.skip("Pollard repo not present")
    have, curated = _curated_modules()
    assert have - curated == set(), f"no home screen: {sorted(have - curated)}"
