"""find -- search the machine for the files a run needs: models, eval corpora, benchmarks.

A file dialog only helps someone who already knows where the file is. Most of the time the
honest answer is "it's a GGUF somewhere, I downloaded it last month", and the thing that
actually unblocks the run is a search.

Three rules keep this from being a `find /` that hangs the window:

  * it walks a LIST OF ROOTS, not the whole disk -- the places models and datasets actually
    land (the workspace, Downloads, the HF cache, LM Studio, Ollama);
  * it is bounded twice, by depth and by a wall-clock budget, and returns what it found so far
    when either runs out, with `truncated` set so the UI can say so;
  * it skips the directories that are all cost and no answer -- version control, virtualenvs,
    node_modules, app bundles.

Matching is on the file NAME, never on contents, so it stays fast on a large disk.
"""

from __future__ import annotations

import os
import pathlib
import time

#: extensions worth offering for each kind of field, most-likely first
KINDS: dict[str, tuple[str, ...]] = {
    "gguf":    (".gguf",),
    "text":    (".txt", ".md", ".jsonl", ".json"),
    "data":    (".jsonl", ".json", ".csv", ".parquet", ".tsv", ".bin", ".txt"),
    "imatrix": (".imatrix", ".dat"),
    "safetensors": (".safetensors",),
    "any":     (),
}

#: never worth descending into -- big, and never holds a user's eval or model
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages",
    ".Trash", "Library", "Applications", ".cargo", ".rustup", ".gradle", ".npm",
    "build", "dist", ".next", ".terraform", "Photos Library.photoslibrary",
})

#: the caches that DO hold models, reached explicitly since Library/ is skipped above
EXTRA_ROOTS = (
    "~/.cache/huggingface/hub",
    "~/.cache/lm-eval",
    "~/.ollama/models",
    "~/.lmstudio/models",
    "~/Library/Application Support/nomic.ai/GPT4All",
)


def roots(home: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Where to look, in the order results should be ranked."""
    h = home or pathlib.Path.home()
    out: list[pathlib.Path] = []
    # An explicit `home` means "search this tree" -- mixing in an unrelated workspace from the
    # environment would quietly widen the search past what the caller asked for.
    env = os.environ.get("POLLARD_HOME") if home is None else None
    if env:
        out.append(pathlib.Path(env).expanduser())
    out += [h / "pollard", h / "Desktop", h / "Downloads", h / "Documents",
            h / "models", h / "data", h / "evals"]
    # resolved against `h`, not against ~, so an explicit home really is the whole search
    out += [h / p[2:] for p in EXTRA_ROOTS]
    seen, uniq = set(), []
    for p in out:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp not in seen and p.is_dir():
            seen.add(rp)
            uniq.append(p)
    return uniq


def _score(name: str, terms: list[str], depth: int) -> int:
    """Rank: all terms as a phrase beats scattered terms, shallow beats deep."""
    low = name.lower()
    s = 0
    if terms:
        joined = " ".join(terms)
        if joined in low:
            s += 60
        s += 20 * sum(1 for t in terms if t in low)
        if low.startswith(terms[0]):
            s += 15
    return s - depth


def search(query: str = "", kind: str = "any", *, limit: int = 80,
           budget_s: float = 2.5, max_depth: int = 6,
           home: pathlib.Path | None = None,
           extra: list[pathlib.Path] | None = None) -> dict:
    """Find files matching `query` whose extension suits `kind`.

    Every term in `query` must appear in the file name (or its parent folder, so that
    "wikitext" finds `wikitext-2/test.txt`). An empty query lists everything of that kind,
    which is the useful default for `gguf`.
    """
    exts = KINDS.get(kind, ())
    terms = [t for t in query.lower().split() if t]
    deadline = time.monotonic() + budget_s
    hits: list[tuple[int, dict]] = []
    truncated = False
    seen: set[str] = set()

    for root in (extra or []) + roots(home):
        if time.monotonic() > deadline:
            truncated = True
            break
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            if time.monotonic() > deadline:
                truncated = True
                break
            depth = len(pathlib.Path(dirpath).parts) - base_depth
            if depth >= max_depth:
                dirnames[:] = []
            # prune in place: os.walk honours this and never descends
            dirnames[:] = [d for d in dirnames
                           if d not in SKIP_DIRS and not d.startswith(".")
                           and not d.endswith((".app", ".framework"))]
            parent = os.path.basename(dirpath).lower()
            for fn in filenames:
                if exts and not fn.lower().endswith(exts):
                    continue
                if fn.startswith("."):
                    continue
                hay = fn.lower() + " " + parent
                if terms and not all(t in hay for t in terms):
                    continue
                full = os.path.join(dirpath, fn)
                if full in seen:
                    continue
                seen.add(full)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                hits.append((_score(fn + " " + parent, terms, depth), {
                    "path": full,
                    "name": fn,
                    "dir": dirpath,
                    "bytes": st.st_size,
                    "mtime": st.st_mtime,
                }))
                if len(hits) >= limit * 4:           # plenty to rank from; stop hunting
                    truncated = True
                    break
            if truncated:
                break
        if truncated and len(hits) >= limit:
            break

    # best score first, then newest -- a file you touched recently is usually the one you meant
    hits.sort(key=lambda h: (-h[0], -h[1]["mtime"]))
    return {"results": [h[1] for h in hits[:limit]],
            "truncated": truncated or len(hits) > limit,
            "searched": [str(r) for r in roots(home)]}
