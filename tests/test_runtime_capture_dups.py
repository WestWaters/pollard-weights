"""The nightly runtime update must not re-capture a declared patch as a new 'pre-update' patch every night."""
import os, subprocess, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_runtime_update as U


def _git(d, *a):
    subprocess.run(["git", "-C", d, *a], check=True, capture_output=True)


def _engine(tmp):
    """A tiny engine tree with one committed source file, plus a declared patch that edits it."""
    tree = os.path.join(tmp, "engine")
    os.makedirs(os.path.join(tree, "src"))
    _git(tmp, "init", "-q", tree)
    _git(tree, "config", "user.email", "t@t"); _git(tree, "config", "user.name", "t")
    src = os.path.join(tree, "src", "quants.c")
    open(src, "w").write("int a;\n#if defined(__SSE2__)\nint helper;\n#endif\nint b;\n")
    _git(tree, "add", "."); _git(tree, "commit", "-q", "-m", "base")
    open(src, "w").write("int a;\n#if defined(__SSE2__) || defined(__SSSE3__)\nint helper;\n#endif\nint b;\n")
    patch = os.path.join(tmp, "engine-msvc-sse2.patch")
    open(patch, "w").write(subprocess.run(["git", "-C", tree, "diff", "HEAD"], capture_output=True, text=True).stdout)
    return tree, src, patch


def _declare(monkeypatch, patch, live=True):
    monkeypatch.setattr(U, "_patches_for", lambda tree: [{"_patch": patch}])
    monkeypatch.setattr(U, "patch_is_live", lambda tree, m: live)


def test_exactly_the_declared_patch_is_not_captured_again(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tree, src, patch = _engine(tmp)
        _declare(monkeypatch, patch)
        assert U.dirty_is_declared(tree)


def test_declared_patch_saved_with_crlf_still_matches(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tree, src, patch = _engine(tmp)
        text = open(patch).read().replace("\n", "\r\n")
        open(patch, "w", newline="").write(text)
        _declare(monkeypatch, patch)
        assert U.dirty_is_declared(tree)


def test_an_extra_change_is_still_captured(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tree, src, patch = _engine(tmp)
        open(src, "a").write("int new_work;\n")
        _declare(monkeypatch, patch)
        assert not U.dirty_is_declared(tree), "work beyond the declared patch must be captured"


def test_an_untracked_source_file_is_still_captured(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tree, src, patch = _engine(tmp)
        open(os.path.join(tree, "src", "new-sampler.cpp"), "w").write("int x;\n")
        _declare(monkeypatch, patch)
        assert not U.dirty_is_declared(tree), "a new source file is not part of any declared patch"


def test_a_patch_that_is_not_live_does_not_explain_the_changes(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tree, src, patch = _engine(tmp)
        _declare(monkeypatch, patch, live=False)
        assert not U.dirty_is_declared(tree)


def test_changed_lines_ignores_headers_inside_hunks():
    diff = ("diff --git a/x.c b/x.c\n--- a/x.c\n+++ b/x.c\n@@ -1,2 +1,2 @@\n"
            "--- a removed line that looks like a header\n+++ an added one too\n")
    assert U._changed_lines(diff) == {"x.c": sorted(["--- a removed line that looks like a header", "+++ an added one too"])}


def test_a_mode_only_change_beyond_the_patch_is_still_captured(monkeypatch):
    """A chmod carries no +/- lines, so content-only dedup missed it (Joey's review)."""
    with tempfile.TemporaryDirectory() as tmp:
        tree, src, patch = _engine(tmp)
        other = os.path.join(tree, "src", "run.sh")
        open(other, "w").write("echo hi\n")
        _git(tree, "add", "src/run.sh"); _git(tree, "commit", "-q", "-m", "script")
        os.chmod(other, 0o755)
        _declare(monkeypatch, patch)
        assert not U.dirty_is_declared(tree), "a mode change beyond the declared patch must be captured"


def _engine_with_new_file(tmp):
    """A declared patch that edits a tracked file AND adds a new one (like composed-sampler.cpp)."""
    tree, src, _ = _engine(tmp)
    new = os.path.join(tree, "src", "sampler-new.cpp")
    open(new, "w").write("int composed;\n")
    _git(tree, "add", "-N", "src/sampler-new.cpp")
    patch = os.path.join(tmp, "engine-new-file.patch")
    open(patch, "w").write(subprocess.run(["git", "-C", tree, "diff", "HEAD"], capture_output=True, text=True).stdout)
    _git(tree, "reset", "-q", "--", "src/sampler-new.cpp")          # applied directly: the new file is untracked
    return tree, new, patch


def test_a_new_file_the_declared_patch_creates_is_declared(monkeypatch):
    """The box re-captured all four llama.cpp patches (73 KB) because composed-sampler.cpp was untracked."""
    with tempfile.TemporaryDirectory() as tmp:
        tree, new, patch = _engine_with_new_file(tmp)
        _declare(monkeypatch, patch)
        assert U.dirty_is_declared(tree)


def test_an_untracked_file_no_patch_creates_is_still_captured(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tree, new, patch = _engine_with_new_file(tmp)
        open(os.path.join(tree, "src", "my-work.cpp"), "w").write("int mine;\n")
        _declare(monkeypatch, patch)
        assert not U.dirty_is_declared(tree)


def test_edits_inside_a_patch_created_file_are_still_captured(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tree, new, patch = _engine_with_new_file(tmp)
        open(new, "a").write("int extra_work;\n")
        _declare(monkeypatch, patch)
        assert not U.dirty_is_declared(tree)
