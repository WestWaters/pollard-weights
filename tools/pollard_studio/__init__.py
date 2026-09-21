"""Pollard Studio -- a desktop window over the Pollard CLI.

The version is POLLARD'S. Studio is a front end for a specific toolchain, so a second version
number would only ever be a thing to forget to bump: if the tools move, this moved with them.
"""
from __future__ import annotations

import os
import pathlib
import re

FALLBACK = "0.0.0"


def _version() -> str:
    """Studio's version IS Pollard's version -- they ship together.

    Read from the installed distribution first, so it is the same number pip knows rather than
    one derived from a path that may not exist. In a checkout there is no distribution, so fall
    back to the pyproject beside the package.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("pollard-weights")
        except PackageNotFoundError:
            pass
    except ImportError:
        pass
    from .runner import repo_root
    for base in (repo_root(), pathlib.Path(os.environ.get("POLLARD_REPO", "")).expanduser()):
        try:
            for line in (base / "pyproject.toml").read_text().splitlines():
                m = re.match(r'\s*version\s*=\s*["\']([^"\']+)["\']', line)
                if m:
                    return m.group(1)
        except OSError:
            continue
    return FALLBACK


__version__ = _version()
