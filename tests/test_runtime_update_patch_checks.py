"""pollard-runtime --update: which patches count as superseded, and what happens when git refuses a tree."""
import os, stat, subprocess, sys, tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_runtime_update as U
import pollard_runtime as R

REPO = os.path.join(os.path.dirname(__file__), "..")
_REAL_CLEF_OMNI = os.path.join(REPO, "runtime-patches", "llama.cpp-clef-omni.patch")
# The registration hunks of runtime-patches/llama.cpp-clef-omni.patch. The nix package ships only the .py files
# of tests/ and no runtime-patches/, so the tests fall back to this excerpt there; it carries the same arch
# registrations (and the neighbouring upstream `clef` lines that tripped the old any()-over-strings check).
_CLEF_OMNI_EXCERPT = """\
diff --git a/gguf-py/gguf/constants.py b/gguf-py/gguf/constants.py
--- a/gguf-py/gguf/constants.py
+++ b/gguf-py/gguf/constants.py
@@ -573,6 +573,7 @@ class MODEL_ARCH(IntEnum):
     QWEN35MOE        = auto()
     CLEF             = auto()
     VLLM_DECISION2   = auto()
+    CLEF_OMNI        = auto()
     QWEN4EXP         = auto()
@@ -1377,6 +1378,7 @@ MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
     MODEL_ARCH.QWEN35MOE:        "qwen35moe",
     MODEL_ARCH.CLEF:             "clef",
     MODEL_ARCH.VLLM_DECISION2:   "vllm-decision2",
+    MODEL_ARCH.CLEF_OMNI:        "clef-omni",
     MODEL_ARCH.QWEN4EXP:         "qwen4exp",
diff --git a/src/llama-arch.cpp b/src/llama-arch.cpp
--- a/src/llama-arch.cpp
+++ b/src/llama-arch.cpp
@@ -42,6 +42,7 @@ static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
     { LLM_ARCH_QWEN35MOE,        "qwen35moe"        },
     { LLM_ARCH_CLEF,             "clef"             },
     { LLM_ARCH_VLLM_DECISION2,   "vllm-decision2"   },
+    { LLM_ARCH_CLEF_OMNI,        "clef-omni"        },
     { LLM_ARCH_QWEN4EXP,         "qwen4exp"         },
"""


def _clef_omni_patch():
    if os.path.exists(_REAL_CLEF_OMNI):
        return _REAL_CLEF_OMNI
    p = os.path.join(tempfile.gettempdir(), "pollard-test-clef-omni-excerpt.patch")
    with open(p, "w", encoding="utf-8") as f:
        f.write(_CLEF_OMNI_EXCERPT)
    return p


CLEF_OMNI = _clef_omni_patch()
# upstream at 620d222: has clef (and vllm-decision2), not clef-omni
UPSTREAM_620D222 = {"llama", "qwen35", "qwen35moe", "clef", "vllm-decision2", "qwen3vlmoe", "nemotron_h_moe"}


def _patch(tmp, body):
    p = os.path.join(tmp, "x.patch")
    open(p, "w").write(body)
    return {"_patch": p, "adds_strings": []}


# ---- (b) _is_arch_patch: the exact architectures, all of them ----------------------------------------------

def test_clef_omni_is_not_superseded_by_upstream_clef():
    m = {"_patch": CLEF_OMNI, "adds_strings": ["clef", "clef-omni", "attn_norm", "models.h"]}
    # the old any()-over-adds_strings test said True here because "clef" is upstream
    assert not U._is_arch_patch(m, UPSTREAM_620D222)


def test_clef_omni_is_superseded_once_upstream_has_clef_omni():
    m = {"_patch": CLEF_OMNI}
    assert U._is_arch_patch(m, UPSTREAM_620D222 | {"clef-omni"})


def test_patch_arch_names_reads_registrations_only():
    text = open(CLEF_OMNI, encoding="utf-8").read()
    assert U.patch_arch_names(text) == {"clef-omni"}


def test_every_registered_arch_must_be_upstream():
    with tempfile.TemporaryDirectory() as tmp:
        m = _patch(tmp, '+++ b/src/llama-arch.cpp\n'
                        '+    { LLM_ARCH_FOO,     "foo"     },\n'
                        '+    { LLM_ARCH_FOO_BAR, "foo-bar" },\n'
                        '+++ b/gguf-py/gguf/constants.py\n'
                        '+    MODEL_ARCH.FOO_BAR:        "foo-bar",\n')
        assert not U._is_arch_patch(m, {"foo"})
        assert U._is_arch_patch(m, {"foo", "foo-bar"})


def test_patch_that_registers_no_arch_is_never_an_arch_patch():
    with tempfile.TemporaryDirectory() as tmp:
        m = _patch(tmp, '+++ b/src/x.cpp\n+    const char * s = "clef";\n')
        m["adds_strings"] = ["clef"]
        assert not U._is_arch_patch(m, {"clef"})


def test_unreadable_patch_is_not_an_arch_patch():
    assert not U._is_arch_patch({"_patch": "/nonexistent/x.patch", "adds_strings": ["clef"]}, {"clef"})


def test_reapply_blocks_instead_of_calling_clef_omni_not_needed(monkeypatch):
    """The 620d222 staging run: clef-omni did not apply, upstream had `clef`, the patch was reported
    'not needed'. It must block (rebase it) instead."""
    with tempfile.TemporaryDirectory() as tmp:
        staging = os.path.join(tmp, "staging")
        os.makedirs(os.path.join(staging, "src"))
        names = sorted(UPSTREAM_620D222) + [f"arch{i}" for i in range(25)]
        open(os.path.join(staging, "src", "llama-arch.cpp"), "w").write(
            "static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {\n"
            + "".join(f'    {{ LLM_ARCH_X{i}, "{n}" }},\n' for i, n in enumerate(names)) + "};\n")
        subprocess.run(["git", "init", "-q", staging], check=True)
        subprocess.run(["git", "-C", staging, "add", "."], check=True)
        subprocess.run(["git", "-C", staging, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "b"], check=True)
        m = {"_patch": CLEF_OMNI, "files": ["src/llama-arch.cpp"], "adds_strings": ["clef", "clef-omni"]}
        monkeypatch.setattr(U, "_patches_for", lambda tree: [m])
        monkeypatch.setattr(U, "patch_is_live", lambda tree, mm: True)
        monkeypatch.setattr(U, "_absorbed", lambda staging, mm: False)
        monkeypatch.setattr(U, "REPO", tmp)                      # no runtime-patches/scripts here
        block, notes = U.reapply(staging, os.path.join(tmp, "live"))
        assert block and "rebase" in block[0], (block, notes)
        assert not any("not needed" in n for n in notes)


# ---- (c) git refusing a tree (Windows SYSTEM task): loud, never "patch absent" -------------------------------

DUBIOUS = ("fatal: detected dubious ownership in repository at '{t}'\n"
           "To add an exception for this directory, call:\n\n\tgit config --global --add safe.directory {t}\n")


@pytest.fixture
def refusing_git(monkeypatch):
    """A `git` first on PATH that answers every command the way git does for a tree owned by someone else."""
    if sys.platform == "win32":
        pytest.skip("fake git is a shell script")
    with tempfile.TemporaryDirectory() as tmp:
        bindir = os.path.join(tmp, "bin")
        os.makedirs(bindir)
        g = os.path.join(bindir, "git")
        open(g, "w").write("#!/bin/sh\nprintf \"%s\" \"" + DUBIOUS.format(t="/tree").replace('"', '\\"') + "\" >&2\nexit 128\n")
        os.chmod(g, os.stat(g).st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", bindir + os.pathsep + os.environ.get("PATH", ""))
        tree = os.path.join(tmp, "tree")
        os.makedirs(os.path.join(tree, ".git"))
        yield tree


def test_patch_is_live_raises_instead_of_reporting_absent(refusing_git):
    with pytest.raises(U.GitOwnershipError, match="dubious ownership"):
        U.patch_is_live(refusing_git, {"_patch": CLEF_OMNI})


def test_git_helper_raises_instead_of_returning_empty(refusing_git):
    with pytest.raises(U.GitOwnershipError):
        U._git(refusing_git, "rev-parse", "HEAD")


def test_update_engine_stops_loudly(refusing_git, monkeypatch, capsys):
    monkeypatch.setitem(U.ENGINES, "fake", {"dir": refusing_git, "url": "https://github.com/x/y"})
    monkeypatch.setattr(U, "upstream_head", lambda url, ref=None: ("abc1234", None))
    monkeypatch.setattr(U.os, "environ", dict(os.environ))       # trust_trees writes here; keep it local
    assert U.update_engine("fake") is False
    out = capsys.readouterr().out
    assert "NOT updating" in out and "dubious ownership" in out and "safe.directory" in out


def test_capture_does_not_report_no_changes(refusing_git):
    meta, msg = R.capture(refusing_git, "pre-update-x", tempfile.gettempdir())
    assert meta is None and "dubious ownership" in msg and "no uncommitted" not in msg


def test_other_git_failures_are_not_ownership_errors():
    r = U._run(["git", "-C", tempfile.gettempdir(), "apply", "--check", "/nonexistent.patch"])
    assert r.returncode != 0                      # an ordinary failure still returns, as before


# ---- trust_trees: safe.directory reaches git, once ------------------------------------------------------------

def test_trust_trees_sets_command_scope_safe_directory(monkeypatch):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_CONFIG_")}
    monkeypatch.setattr(U.os, "environ", env)
    U.trust_trees("/some/tree", "/some/tree.next")
    U.trust_trees("/some/tree")                                    # idempotent
    vals = [env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env["GIT_CONFIG_COUNT"]))]
    assert all(env[f"GIT_CONFIG_KEY_{i}"] == "safe.directory" for i in range(len(vals)))
    assert vals.count(os.path.abspath("/some/tree").replace("\\", "/")) == 1
    r = subprocess.run(["git", "config", "--get-all", "safe.directory"], capture_output=True, text=True, env=env)
    assert os.path.abspath("/some/tree").replace("\\", "/") in r.stdout.splitlines()


def test_trust_trees_keeps_existing_git_config_env(monkeypatch):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_CONFIG_")}
    env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="core.autocrlf", GIT_CONFIG_VALUE_0="false")
    monkeypatch.setattr(U.os, "environ", env)
    U.trust_trees("/t")
    assert env["GIT_CONFIG_KEY_0"] == "core.autocrlf" and env["GIT_CONFIG_COUNT"] in ("2", "3")
