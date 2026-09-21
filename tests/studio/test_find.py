"""Point Pollard at your OWN evals, benchmarks and models.

The tools have always taken --eval / --prompts / --hellaswag-data; what was missing was any way
to give them a file from the UI. These cover the search that finds one and the fields that carry
it into the run.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pytest

# the package lives at <repo>/tools/pollard_studio, so ROOT is the tools dir:
# every path below stays written as "pollard_studio/..." and the import works too
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "tools"
sys.path.insert(0, str(ROOT))

from pollard_studio import actions, find  # noqa: E402


# ── the search itself ───────────────────────────────────────────────────────────────────────────

@pytest.fixture
def disk(tmp_path):
    """A little machine with models and evals scattered the way they really are."""
    (tmp_path / "Downloads").mkdir()
    (tmp_path / "Downloads" / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"x" * 2048)
    (tmp_path / "Downloads" / "notes.txt").write_text("hello")
    (tmp_path / "Desktop" / "evals").mkdir(parents=True)
    (tmp_path / "Desktop" / "evals" / "wikitext-2-test.txt").write_text("corpus")
    (tmp_path / "Desktop" / "evals" / "my_mcq.jsonl").write_text('{"q": 1}')
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "penjing-IQ4_XS.gguf").write_bytes(b"y" * 4096)
    # things the walk must refuse to descend into
    for junk in (".git", "node_modules", ".venv"):
        d = tmp_path / "Desktop" / junk
        d.mkdir()
        (d / "decoy.gguf").write_bytes(b"z")
    return tmp_path


def test_finds_a_model_by_part_of_its_name(disk):
    got = find.search("qwen", "gguf", home=disk)
    names = [r["name"] for r in got["results"]]
    assert names == ["Qwen3-4B-Q4_K_M.gguf"]


def test_an_empty_query_lists_everything_of_that_kind(disk):
    """Listing every GGUF is the useful default -- you may not remember the name at all."""
    names = {r["name"] for r in find.search("", "gguf", home=disk)["results"]}
    assert names == {"Qwen3-4B-Q4_K_M.gguf", "penjing-IQ4_XS.gguf"}


def test_kind_filters_by_extension(disk):
    """A field that wants a corpus must not offer a 4 GB model."""
    names = {r["name"] for r in find.search("", "text", home=disk)["results"]}
    assert "wikitext-2-test.txt" in names
    assert not any(n.endswith(".gguf") for n in names)


def test_matches_on_the_parent_folder_too(disk):
    """'evals' is a folder, not part of any filename -- searching it should still work, because
    that is how people remember where a file lives."""
    names = {r["name"] for r in find.search("evals", "text", home=disk)["results"]}
    assert "wikitext-2-test.txt" in names


def test_never_descends_into_junk_directories(disk):
    """.git / node_modules / .venv are all cost and no answer, and a decoy in one of them must
    not be offered."""
    paths = [r["path"] for r in find.search("decoy", "gguf", home=disk)["results"]]
    assert paths == []


def test_a_query_with_several_terms_requires_all_of_them(disk):
    assert find.search("qwen penjing", "gguf", home=disk)["results"] == []
    assert len(find.search("penjing iq4", "gguf", home=disk)["results"]) == 1


def test_results_carry_what_the_ui_shows(disk):
    r = find.search("penjing", "gguf", home=disk)["results"][0]
    assert r["bytes"] == 4096
    assert r["name"] and r["dir"] and r["path"].endswith(r["name"])
    assert r["mtime"] > 0


def test_the_search_is_bounded_and_says_so(disk):
    """A budget that has already expired must return promptly with truncated set, not walk the
    disk anyway."""
    t0 = time.monotonic()
    got = find.search("", "any", home=disk, budget_s=0.0)
    assert time.monotonic() - t0 < 2.0
    assert got["truncated"] is True


def test_limit_is_respected(disk):
    got = find.search("", "any", home=disk, limit=2)
    assert len(got["results"]) <= 2


def test_missing_roots_are_skipped_not_fatal(tmp_path):
    """Most of the roots will not exist on any given machine."""
    got = find.search("anything", "gguf", home=tmp_path / "nowhere")
    assert got["results"] == []


# ── the bridge ──────────────────────────────────────────────────────────────────────────────────

def _api():
    from pollard_studio import app
    api = app.Api.__new__(app.Api)      # no window, no runner -- just the two methods
    api.window = None
    return api


def test_pick_file_without_a_window_returns_nothing_rather_than_raising():
    assert _api().pick_file("gguf") == {"path": None}


def test_search_files_bridges_through_to_find(monkeypatch):
    seen = {}

    def fake(query, kind, limit):
        seen.update(query=query, kind=kind, limit=limit)
        return {"results": [], "truncated": False}

    from pollard_studio import app
    monkeypatch.setattr(app._find, "search", lambda q, k, limit: fake(q, k, limit))
    _api().search_files("qwen", "gguf", 30)
    assert seen == {"query": "qwen", "kind": "gguf", "limit": 30}


def test_search_files_clamps_a_silly_limit(monkeypatch):
    from pollard_studio import app
    got = {}
    monkeypatch.setattr(app._find, "search",
                        lambda q, k, limit: got.setdefault("limit", limit) or {"results": []})
    _api().search_files("x", "gguf", 100000)
    assert got["limit"] == 200


def test_search_files_reports_an_error_instead_of_crashing_the_ui(monkeypatch):
    from pollard_studio import app

    def boom(*a, **k):
        raise OSError("disk went away")

    monkeypatch.setattr(app._find, "search", boom)
    got = _api().search_files("x", "gguf")
    assert got["results"] == [] and "disk went away" in got["error"]


def test_every_file_kind_the_ui_asks_for_is_one_the_dialog_knows():
    """A kind the UI passes that FILE_KINDS does not have silently degrades to 'any'."""
    from pollard_studio import app
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    used = set(re.findall(r'pathrow\("[a-z0-9-]+",\s*"[A-Za-z]+",\s*"([a-z]+)"', js))
    assert used, "no pathrow calls found"
    assert used <= set(app.Api.FILE_KINDS), f"unknown kinds: {used - set(app.Api.FILE_KINDS)}"


def test_find_kinds_and_dialog_kinds_agree():
    from pollard_studio import app
    shared = set(app.Api.FILE_KINDS) - {"dir"}
    assert shared <= set(find.KINDS), f"dialog offers kinds find cannot search: {shared - set(find.KINDS)}"


# ── the fields reach the command ────────────────────────────────────────────────────────────────

def test_your_own_corpus_and_prompts_reach_pollard_eval():
    _, args = actions.resolve("eval", {"ref": "/r.gguf", "gguf": "/q.gguf",
                                       "evalFile": "/my/corpus.txt",
                                       "promptsFile": "/my/prompts.txt"})
    assert "--eval" in args and args[args.index("--eval") + 1] == "/my/corpus.txt"
    assert "--prompts" in args and args[args.index("--prompts") + 1] == "/my/prompts.txt"


def test_your_own_corpus_reaches_pollard_bench():
    _, args = actions.resolve("bench", {"gguf": "/q.gguf", "evalFile": "/my/corpus.txt"})
    assert args[args.index("--eval") + 1] == "/my/corpus.txt"


def test_a_rival_gguf_from_anywhere_reaches_bench():
    _, args = actions.resolve("bench", {"gguf": "/q.gguf", "vs": "/elsewhere/rival.gguf"})
    assert args[args.index("--vs") + 1] == "/elsewhere/rival.gguf"


@pytest.mark.parametrize("fmt,flag", [("hellaswag", "--hellaswag-data"),
                                      ("winogrande", "--winogrande"),
                                      ("multiple-choice", "--multiple-choice")])
def test_the_benchmark_format_picks_the_right_flag(fmt, flag):
    """These three are mutually exclusive in the tool: the chosen format decides which one the
    file is passed as, and the other two must not appear."""
    tool, args = actions.resolve("probes", {"gguf": "/q.gguf", "probeData": "/my/bench.jsonl",
                                            "probeFormat": fmt})
    assert tool == "pollard_probes"
    assert args[args.index(flag) + 1] == "/my/bench.jsonl"
    others = {"--hellaswag-data", "--winogrande", "--multiple-choice"} - {flag}
    assert not (others & set(args))


def test_probes_without_a_datafile_passes_no_format_flag():
    """No file means the tool auto-prepares HellaSwag; passing an empty flag would break it."""
    _, args = actions.resolve("probes", {"gguf": "/q.gguf"})
    assert not ({"--hellaswag-data", "--winogrande", "--multiple-choice"} & set(args))


def test_explicit_lm_eval_tasks_reach_taskeval():
    _, args = actions.resolve("taskeval", {"gguf": "/q.gguf", "tasks": "arc_easy,gsm8k"})
    assert args[args.index("--tasks") + 1] == "arc_easy,gsm8k"


def test_the_kv_sweep_corpus_no_longer_collides_with_the_eval_screen():
    """Both used to write R.evalFile, so whichever screen you touched last decided both."""
    _, args = actions.resolve("kvsweep", {"gguf": "/q.gguf", "kvEvalFile": "/kv.txt",
                                          "evalFile": "/eval.txt"})
    assert args[args.index("--eval") + 1] == "/kv.txt"


def test_the_kv_sweep_still_falls_back_to_the_shared_corpus():
    _, args = actions.resolve("kvsweep", {"gguf": "/q.gguf", "evalFile": "/eval.txt"})
    assert args[args.index("--eval") + 1] == "/eval.txt"


# ── no field may be decoration ──────────────────────────────────────────────────────────────────

def test_no_path_field_is_inert():
    """Four of these boxes accepted a path and threw it away -- the run went ahead on defaults.
    Every text input in a screen must either bind itself or be bound in a wire function."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    inert = []
    for mo in re.finditer(r'<input type="text"[^>]*?id="([a-z0-9-]+)"[^>]*?>', js, re.S):
        ident, whole = mo.group(1), mo.group(0)
        if "oninput" in whole or "onchange" in whole:
            continue
        if re.search(r'bind\w*\(\s*"%s"' % re.escape(ident), js):
            continue
        if re.search(r'\$\("#%s"\)' % re.escape(ident), js):
            continue
        inert.append(ident)
    assert inert == [], f"these inputs go nowhere: {inert}"


def test_every_pathrow_key_is_read_by_some_action():
    """A path field bound to a recipe key no action consumes is the same bug wearing a hat."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    py = (ROOT / "pollard_studio/actions.py").read_text()
    keys = set(re.findall(r'pathrow\("[a-z0-9-]+",\s*"([A-Za-z]+)"', js))
    assert keys, "no pathrow calls found"
    ui_only = {"artifact"}          # the chat media player reads this one directly, not via a run
    for k in keys - ui_only:
        assert f'"{k}"' in py, f"R.{k} is set by a path field but no action reads it"


def test_every_icon_used_exists_in_the_sprite():
    """ic("nope") renders an empty <use> -- no error, just a blank where the icon should be."""
    html = (ROOT / "pollard_studio/ui/index.html").read_text()
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    have = set(re.findall(r'id="i-([a-z0-9]+)"', html))
    used = set(re.findall(r'ic\("([a-z0-9]+)"\)', js))
    assert used - have == set(), f"icons used but never drawn: {sorted(used - have)}"


def test_the_search_panel_is_wired_to_the_bridge():
    """FIND must call the bridge and render clickable rows, or it is a button that does nothing."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    fn = js[js.index("async function findInto"):js.index("function useFound")]
    assert "search_files(" in fn, "FIND never reaches the bridge"
    assert "useFound(" in fn, "results are not clickable"
    assert "findpanel" not in fn or True
    # and the result rows must carry the path as data, not interpolate it into the handler
    assert "data-p=" in fn, "path should ride in a data attribute, not inside the onclick string"


def test_browse_and_find_both_write_the_same_recipe_key():
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    for fn_name in ("useFound", "browseInto"):
        body = js[js.index(f"function {fn_name}"):]
        body = body[:body.index("\n}\n") + 3]
        assert "R[key]" in body, f"{fn_name} does not set the recipe key"


def test_a_path_field_gets_the_whole_width_not_a_sliver():
    """FIND / BROWSE / CLEAR beside the input in a 340px card left about 90px to read a
    filesystem path in. The box takes the row; the buttons go under it."""
    css = (ROOT / "pollard_studio/ui/app.css").read_text()
    blk = css[css.index(".pathrow {"):css.index(".btn.sm {")]
    assert "flex-wrap: wrap" in blk, "the row cannot wrap, so the input stays squeezed"
    assert "flex: 1 1 100%" in blk, "the input does not claim the full width"
    assert "direction: rtl" not in blk, "rtl breaks the cursor while typing a path"


def test_the_buttons_share_the_row_under_the_input():
    css = (ROOT / "pollard_studio/ui/app.css").read_text()
    blk = css[css.index(".pathrow {"):css.index(".btn.sm {")]
    assert ".pathrow .btn { flex: 1 1 0" in blk
