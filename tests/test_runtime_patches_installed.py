"""runtime-patches/ ships with the installed package, and the tools find it there.

A nix install (and any non-editable pip install) carried tools/ only: pollard_ggufcompat looked for
<site-packages>/../runtime-patches, found nothing, and reported clef-omni as `unknown` instead of `pollard`. The
patches now install as data at <prefix>/share/pollard-weights/runtime-patches, and pollard_calc.patches_dir() finds
them when there is no checkout.
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_calc as PC  # noqa: E402

SRC_PATCHES = Path(PC.patches_dir())


def _need_patches():
    if not (SRC_PATCHES / "engines.json").is_file():
        pytest.skip("no runtime-patches/ in this checkout or install")


def test_pyproject_installs_every_runtime_patch_file():
    """the data-files globs cover every file runtime-patches/ holds -- a new kind of file (or a subfolder) would
    otherwise silently stay behind in the checkout"""
    src = ROOT / "runtime-patches"
    if not src.is_dir() or not (ROOT / "pyproject.toml").is_file():
        pytest.skip("not a source checkout")
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    df = cfg["tool"]["setuptools"]["data-files"]
    globs = df[PC.INSTALLED_PATCHES.replace(os.sep, "/")]
    have = sorted(p.relative_to(ROOT).as_posix() for p in src.rglob("*") if p.is_file())
    missed = [f for f in have if not any(fnmatch.fnmatchcase(f, g) and "/" not in f[len("runtime-patches/"):]
                                         for g in globs)]
    assert not missed, f"runtime-patches files the install would leave behind: {missed}"


def _installed_layout(tmp_path):
    """<prefix>/lib/python3.X/site-packages/{pollard_calc,pollard_ggufcompat}.py + <prefix>/share/.../runtime-patches,
    no checkout anywhere above it -- the shape pip / nix install."""
    prefix = tmp_path / "prefix"
    sp = prefix / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "site-packages"
    sp.mkdir(parents=True)
    for m in ("pollard_calc.py", "pollard_ggufcompat.py"):
        shutil.copy2(ROOT / "tools" / m, sp / m)
    shutil.copytree(SRC_PATCHES, prefix / PC.INSTALLED_PATCHES)
    return prefix, sp


def _probe(sp, cwd):
    code = ("import json, pollard_ggufcompat as gc, pollard_calc as pc\n"
            "print(json.dumps({'dir': gc.PATCH_DIR, 'calc': pc.patches_dir(),"
            " 'arch': gc.arch_support('clef-omni', offline=True),"
            " 'needs': len(pc.required_patches('clef-omni'))}))")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "POLLARD_ARCH")}
    env["PYTHONPATH"] = str(sp)
    env["HOME"] = str(cwd)
    r = subprocess.run([sys.executable, "-c", code], cwd=str(cwd), env=env, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_an_installed_package_finds_its_runtime_patches(tmp_path):
    _need_patches()
    prefix, sp = _installed_layout(tmp_path)
    elsewhere = tmp_path / "models"
    elsewhere.mkdir()
    got = _probe(sp, elsewhere)
    want = os.path.realpath(prefix / PC.INSTALLED_PATCHES)
    assert os.path.realpath(got["dir"]) == want and os.path.realpath(got["calc"]) == want
    assert got["arch"] == ["pollard", "llama.cpp-clef-omni; upstream PR pending"], got
    assert got["needs"], "find_llama_bin would not know clef-omni needs the dec.blk patch in an install"


def test_without_the_data_an_install_says_unknown_never_pollard(tmp_path):
    """the regression this fixes, kept honest: no patches anywhere -> `unknown`, no claim made"""
    _need_patches()
    prefix, sp = _installed_layout(tmp_path)
    shutil.rmtree(prefix / PC.INSTALLED_PATCHES)
    got = _probe(sp, tmp_path)
    assert got["arch"][0] == "unknown"


def test_a_checkout_still_wins_over_an_installed_copy():
    """an editable install / a checkout reads its own runtime-patches/ (the one being edited), not a stale copy"""
    if not (ROOT / "runtime-patches").is_dir():
        pytest.skip("not a source checkout")
    assert Path(PC.patches_dir()) == ROOT / "runtime-patches"
    assert Path(PC.patches_dir(str(ROOT / "nowhere"))) == ROOT / "nowhere" / "runtime-patches"
