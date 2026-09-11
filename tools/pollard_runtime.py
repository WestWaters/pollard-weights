#!/usr/bin/env python3
"""pollard-runtime - are the llama.cpp builds on this machine current, and do they still load what we shipped?

Pollard builds against whatever llama.cpp happens to be beside it, and that is a moving part nobody was
watching. Two ways it bites, both of which already happened here:

  * **Building something upstream already has.** We only find out by accident, after writing it.
  * **Going backward.** A working build gets replaced by a fork pinned to an older upstream base, and a
    model that built and measured fine last week stops loading. Spark-X2.5-4B was quantized and
    published on 2026-09-08 with a llama.cpp that carried `spark2_5` (upstream #27868, merged
    2026-09-06). The tree was then rebuilt as the IFM fork for K2, whose base predates that merge, so
    today no build on either machine opens the model we published. Nothing was lost from the model --
    the runtime moved out from under it.

**And the real one: our own runtime patches were never tracked.** Spark-X2.5-4B was built with
`ik_llama.cpp` -- its converter wrote the `spark2_5` header, its `llama-imatrix` computed the matrix, its
`llama-quantize` cut the ladder -- which means we had added `spark2_5` to that tree ourselves, because
ik_llama does not carry it. That work lived only as uncommitted edits in the working tree. A day later
the same tree was edited for the K2 trellis port and the spark support was overwritten. Nothing was
committed, nothing was exported, and a published model lost its runtime. The K2 support in there right
now is one `git checkout` away from going the same way.

So this tool also CAPTURES a runtime's local modifications into the repo as a named patch, and verifies
that a captured patch is still applied. A runtime change that a model depends on is an artifact, not a
working-tree edit.

  pollard-runtime                          # every build it can find, vs upstream master
  pollard-runtime --scan DIR               # look somewhere else too (repeatable)
  pollard-runtime --arch k2-horizon        # who has this architecture? does upstream?
  pollard-runtime --capture NAME           # export each tree's uncommitted changes to runtime-patches/
  pollard-runtime --verify                 # is every captured patch still applied? (non-zero if not)
  pollard-runtime --apply NAME             # re-apply a captured patch after a clone or reset

Exit code is 1 when a build is behind upstream on an architecture we have published against, or when a
captured patch has gone missing -- both states where a shipped repo has no runtime on this machine."""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

# Bytes that can appear INSIDE an architecture name. Used to require that a candidate sits alone in
# the binary rather than as part of a longer run: extracting maximal ASCII runs and intersecting them
# misses a name that happens to abut other text, which is how an early version of this failed to see
# `spark2_5` at all.
IDENT = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
BIN_NAMES = ("llama-cli", "llama-cli.exe", "llama-perplexity", "llama-perplexity.exe")
LIB_GLOBS = ("libllama*.dylib", "libllama*.so", "llama.dll", "*.dll")
UPSTREAM = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp"

# Only runtimes Pollard itself builds against. Nothing here reaches into unrelated projects that
# happen to vendor a llama.cpp -- use --scan for anything outside this list.
DEFAULT_SCAN = [
    "runtime/llama.cpp", "~/llama.cpp", "~/llama-current", "~/pollard-builds/ifm-llama",
    "~/pollard-stq/llama.cpp", "C:/pollard/llama-current", "C:/pollard/ifm-llama",
    "C:/pollard/ik_llama.cpp", "C:/pollard/llama-stq", "C:/pollard/pw/runtime/llama.cpp",
]


def _arch_list_from_source(tree):
    """Architecture names from a tree's llama-arch.cpp, or None."""
    for rel in ("src/llama-arch.cpp", "llama-arch.cpp"):
        p = os.path.join(tree, rel)
        if os.path.isfile(p):
            try:
                s = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            m = re.search(r"LLM_ARCH_NAMES\s*=\s*\{(.*?)\n\};", s, re.S)
            if m:
                got = set(re.findall(r'"\s*([a-z0-9._\-]+)\s*"', m.group(1))) - {"clip"}
                if len(got) > 20:
                    return got
    return None


def _delimited_hits(path, candidates, limit=256 << 20):
    """Which of `candidates` occur in this file as a standalone string.

    "Standalone" means not part of a longer identifier run -- a literal in a binary is delimited by a
    NUL or other non-identifier byte on each side. Checking that is what separates `spark2_5` from an
    accidental match inside some longer symbol.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read(limit)
    except OSError:
        return set()
    found = set()
    for cand in candidates:
        needle = cand.encode()
        start = 0
        while True:
            i = data.find(needle, start)
            if i < 0:
                break
            before = data[i - 1] if i > 0 else 0
            after = data[i + len(needle)] if i + len(needle) < len(data) else 0
            if before not in IDENT and after not in IDENT:
                found.add(cand)
                break
            start = i + 1
    return found


def archs_in_binaries(tree, candidates):
    """Which of `candidates` appear as literal strings in this tree's built binaries.

    Deliberately binary-first: a tree can be reset or re-pointed after a build, and the binary is what
    actually runs. That is how a build was found to have lost `spark2_5` while its source looked fine.
    """
    found, files = set(), []
    for root in (tree, os.path.join(tree, "build")):
        for pat in BIN_NAMES:
            files += glob.glob(os.path.join(root, "**", pat), recursive=True)
        for pat in LIB_GLOBS:
            files += glob.glob(os.path.join(root, "**", pat), recursive=True)
    todo = list(candidates)
    for f in files[:40]:
        found |= _delimited_hits(f, todo)
        todo = [c for c in todo if c not in found]
        if not todo:
            break
    return found, len(files)


def git_info(tree):
    def run(*a):
        try:
            return subprocess.run(["git", "-C", tree, *a], capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:                                                  # noqa: BLE001
            return ""
    return {"commit": run("rev-parse", "--short", "HEAD"),
            "date": run("log", "-1", "--format=%cs"),
            "describe": run("describe", "--tags", "--always"),
            "remote": run("config", "--get", "remote.origin.url")}


def upstream_archs():
    """ggml-org master's architecture list, or None if unreachable."""
    import urllib.request
    try:
        req = urllib.request.Request(UPSTREAM, headers={"User-Agent": "pollard-runtime"})
        with urllib.request.urlopen(req, timeout=45) as fh:
            s = fh.read().decode("utf-8", "replace")
        m = re.search(r"LLM_ARCH_NAMES\s*=\s*\{(.*?)\n\};", s, re.S)
        got = set(re.findall(r'"\s*([a-z0-9._\-]+)\s*"', m.group(1))) - {"clip"} if m else set()
        return got if len(got) > 50 else None
    except Exception:                                                      # noqa: BLE001
        return None


def find_trees(explicit):
    """Trees to act on.

    `--scan` NARROWS rather than adds: capturing a patch names one tree, and if the defaults came along
    too the same diff got written under every tree's name. Given no --scan, the defaults are used.
    """
    paths = list(explicit) if explicit else DEFAULT_SCAN
    out = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p) and p not in out:
            out.append(p)
    return out


def published_archs(owner="PollardWeights"):
    """Architectures our own published repos declare -- what a build here has to be able to load."""
    import urllib.request
    try:
        req = urllib.request.Request(f"https://huggingface.co/api/models?author={owner}&limit=100",
                                     headers={"User-Agent": "pollard-runtime"})
        with urllib.request.urlopen(req, timeout=60) as fh:
            repos = json.load(fh)
    except Exception:                                                      # noqa: BLE001
        return {}
    out = {}
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pollard_ggufcompat import read_header
    except Exception:                                                      # noqa: BLE001
        return {}
    for m in repos:
        rid = m["modelId"]
        try:
            req = urllib.request.Request(f"https://huggingface.co/api/models/{rid}",
                                         headers={"User-Agent": "pollard-runtime"})
            with urllib.request.urlopen(req, timeout=60) as fh:
                sib = [s["rfilename"] for s in json.load(fh).get("siblings", [])
                       if s["rfilename"].endswith(".gguf") and "mmproj" not in s["rfilename"]]
            if not sib:
                continue
            arch, _ = read_header(f"https://huggingface.co/{rid}/resolve/main/{sorted(sib)[0]}")
            out.setdefault(arch, []).append(rid.split("/")[1])
        except Exception:                                                  # noqa: BLE001
            continue
    return out


def declared_scripts(repo_root="."):
    """Console scripts pyproject declares, as {command: module}."""
    try:
        import tomllib
    except ImportError:                                                    # py<3.11
        return {}
    try:
        with open(os.path.join(repo_root, "pyproject.toml"), "rb") as fh:
            d = tomllib.load(fh)
        return dict(d.get("project", {}).get("scripts", {}))
    except Exception:                                                      # noqa: BLE001
        return {}


def installed_scripts():
    """Command launchers actually present next to the running interpreter."""
    bindir = os.path.dirname(sys.executable)
    out = set()
    for f in os.listdir(bindir) if os.path.isdir(bindir) else []:
        name = f[:-4] if f.lower().endswith(".exe") else f
        if name.startswith("pollard"):
            out.add(name)
    return out


def install_state(repo_root="."):
    """Is the install able to RUN what the repo declares?

    An editable install keeps module code current automatically -- `import pollard_card` resolves
    straight into the checkout -- but it only writes command launchers when pip runs. So adding a tool
    and syncing leaves the code present and the command missing, which is invisible until someone types
    the name. Every tool added in one session was in exactly that state on both machines.
    """
    declared = declared_scripts(repo_root)
    if not declared:
        return None
    present = installed_scripts()
    missing = sorted(c for c in declared if c not in present)
    return {"declared": sorted(declared), "present": sorted(present), "missing": missing,
            "bindir": os.path.dirname(sys.executable)}


PATCH_DIR = "runtime-patches"


def _slug(tree):
    return os.path.basename(os.path.abspath(tree).rstrip("/\\")) or "runtime"


def capture(tree, name, out_dir=PATCH_DIR):
    """Export a runtime tree's uncommitted changes as a tracked patch plus a manifest.

    The manifest records the remote, the base commit, and the architecture names the patch adds. That
    last field is what makes verification possible later: "is this patch still applied" becomes a
    question you can answer by looking at the tree, not by remembering.
    """
    # errors="replace" and a generous timeout on purpose: a real runtime patch can be large and can
    # carry bytes that strict UTF-8 rejects, and the whole point of this is not to lose the work.
    try:
        r = subprocess.run(["git", "-C", tree, "diff", "HEAD"], capture_output=True,
                           encoding="utf-8", errors="replace", timeout=600)
        diff = r.stdout or ""
    except Exception as e:                                                 # noqa: BLE001
        return None, f"git diff failed: {e}"
    if not diff.strip():
        return None, "no uncommitted changes to capture"

    added_archs = sorted({m for m in re.findall(r'^\+.*?"\s*([a-z][a-z0-9._\-]{2,31})\s*"',
                                                diff, re.M)})
    gi = git_info(tree)
    os.makedirs(out_dir, exist_ok=True)
    slug = _slug(tree)
    pp = os.path.join(out_dir, f"{slug}-{name}.patch")
    mp = os.path.join(out_dir, f"{slug}-{name}.json")
    with open(pp, "w", encoding="utf-8") as fh:
        fh.write(diff)
    files = sorted(re.findall(r"^\+\+\+ b/(.+)$", diff, re.M))
    meta = {"name": name, "tree_slug": slug, "remote": gi.get("remote"),
            "base_commit": gi.get("commit"), "base_describe": gi.get("describe"),
            "base_date": gi.get("date"), "files": files, "adds_strings": added_archs,
            "diff_bytes": len(diff)}
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return meta, f"wrote {pp} ({len(diff)} bytes, {len(files)} files)"


def load_captured(out_dir=PATCH_DIR):
    out = []
    for mp in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        try:
            m = json.load(open(mp, encoding="utf-8"))
            m["_manifest"] = mp
            m["_patch"] = mp[:-5] + ".patch"
            out.append(m)
        except Exception:                                                  # noqa: BLE001
            continue
    return out


def _added_lines(diff_text):
    """The content of a diff's added lines, stripped, as a set.

    Comparing these is line-ending and context agnostic, which `git apply --reverse --check` is not.
    These trees are CRLF on the Windows box and the patches are read back on a Mac, and even
    --ignore-whitespace could not reverse-apply one of them -- while the additions were plainly still
    in the file.
    """
    out = set()
    for ln in diff_text.splitlines():
        if ln.startswith("+") and not ln.startswith("+++"):
            body = ln[1:].strip().rstrip("\r")
            if body:
                out.add(body)
    return out


def _current_diff(tree, files):
    try:
        r = subprocess.run(["git", "-C", tree, "diff", "HEAD", "--"] + list(files),
                           capture_output=True, encoding="utf-8", errors="replace", timeout=600)
        return r.stdout or ""
    except Exception:                                                      # noqa: BLE001
        return ""


def verify_captured(trees, out_dir=PATCH_DIR):
    """For each captured patch, is the support it adds present in its tree right now?

    Checked against the tree's SOURCE and its BUILT BINARIES, because the two can disagree and the
    binary is what runs. A patch whose strings are in neither has been lost -- which is precisely what
    happened to `spark2_5` and was invisible until a published model would not load.
    """
    rows = []
    by_slug = {_slug(t): t for t in trees}
    for m in load_captured(out_dir):
        tree = by_slug.get(m.get("tree_slug"))
        want = [a for a in m.get("adds_strings") or [] if a]
        row = {"name": m.get("name"), "slug": m.get("tree_slug"), "tree": tree, "wants": want}
        if not tree:
            row["state"] = "tree missing"
            rows.append(row)
            continue
        # The general test, and the only one that works for a patch that adds no distinctive strings
        # (the MSVC regex fix and the STQ quant kernels both do not): if the patch REVERSE-applies
        # cleanly, its changes are present in the tree right now.
        # --ignore-whitespace is required, not cosmetic: these patches are captured on the Windows box
        # where the checkouts are CRLF, and read back on a Mac. Without it every verification reported
        # LOST on a patch that was applied, which is worse than not checking at all.
        rev = subprocess.run(["git", "-C", tree, "apply", "--reverse", "--check",
                              "--ignore-whitespace", os.path.abspath(m["_patch"])],
                             capture_output=True, text=True, timeout=300)
        row["reverse_applies"] = rev.returncode == 0
        # Content check: are the patch's added lines still in the tree's own diff? This is what
        # actually decides the verdict, because reverse-apply is defeated by line-ending drift.
        try:
            captured = open(m["_patch"], encoding="utf-8", errors="replace").read()
        except OSError:
            captured = ""
        want_lines = _added_lines(captured)
        have_lines = _added_lines(_current_diff(tree, m.get("files") or []))
        absent = want_lines - have_lines
        row["added_lines"] = len(want_lines)
        row["absent_lines"] = len(absent)
        row["present_frac"] = (1 - len(absent) / len(want_lines)) if want_lines else 1.0
        if want:
            src = _arch_list_from_source(tree) or set()
            in_src = [a for a in want if a in src]
            in_bin, _ = archs_in_binaries(tree, want)
            row["in_source"], row["in_binaries"] = in_src, sorted(in_bin)
            row["missing"] = [a for a in want if a not in src and a not in in_bin]
        else:
            row["missing"] = []
        if row["reverse_applies"] or row["present_frac"] >= 0.999:
            row["state"] = "APPLIED"
        elif row["present_frac"] >= 0.5:
            row["state"] = "DRIFTED"          # most of it is there; the tree moved under the patch
        else:
            row["state"] = "LOST"
        rows.append(row)
    return rows


def apply_patch(tree, name, out_dir=PATCH_DIR):
    pp = os.path.join(out_dir, f"{_slug(tree)}-{name}.patch")
    if not os.path.isfile(pp):
        return False, f"no captured patch at {pp}"
    r = subprocess.run(["git", "-C", tree, "apply", "--3way", "--ignore-whitespace",
                        os.path.abspath(pp)], capture_output=True, text=True, timeout=300)
    return r.returncode == 0, (r.stderr or r.stdout or "applied").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--scan", action="append", default=[],
                    help="llama.cpp tree to act on (repeatable). Given at all, ONLY these are used -- "
                         "so --capture writes one tree's diff under one tree's name")
    ap.add_argument("--arch", help="report who supports this architecture, upstream included")
    ap.add_argument("--no-published", action="store_true",
                    help="skip reading our published repos' architectures (saves network)")
    ap.add_argument("--capture", metavar="NAME",
                    help="export each tree's uncommitted changes to runtime-patches/ as a tracked "
                         "patch plus a manifest naming the base commit and the strings it adds")
    ap.add_argument("--verify", action="store_true",
                    help="check every captured patch is still applied in its tree (source AND "
                         "binaries); non-zero if any has been lost")
    ap.add_argument("--apply", metavar="NAME", help="re-apply a captured patch after a clone or reset")
    ap.add_argument("--patch-dir", default=PATCH_DIR, help=f"where patches live (default {PATCH_DIR})")
    ap.add_argument("--install", action="store_true",
                    help="check the install can RUN what the repo declares -- an editable install "
                         "keeps module code current but only writes command launchers when pip runs, "
                         "so a synced repo can still have missing commands. Non-zero if any is")
    ap.add_argument("--repo-root", default=".", help="repo to read pyproject.toml from")
    ap.add_argument("--dirty", action="store_true",
                    help="just list trees with uncommitted changes -- runtime work that is not yet "
                         "an artifact and would be lost by a checkout")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    trees = find_trees(a.scan)
    if not trees:
        print("no llama.cpp trees found. Pass --scan <dir>.")
        return 0

    if a.install:
        st = install_state(a.repo_root)
        if st is None:
            print(f"could not read {os.path.join(a.repo_root, 'pyproject.toml')} — pass --repo-root")
            return 0
        print(f"interpreter : {sys.executable}")
        print(f"launchers in: {st['bindir']}")
        print(f"declared    : {len(st['declared'])}")
        print(f"present     : {len(st['present'])}")
        if not st["missing"]:
            print("\nevery declared command is installed.")
            return 0
        print(f"\nMISSING {len(st['missing'])} command(s) — the code is there, the launcher is not:")
        for c in st["missing"]:
            print(f"   {c}")
        print(f"\nfix: {sys.executable} -m pip install -e {os.path.abspath(a.repo_root)} --no-deps")
        print("Run that wherever Pollard is installed after a sync that adds a tool.")
        return 1

    if a.dirty:
        rc = 0
        print("runtime trees with uncommitted changes (not yet an artifact):\n")
        for t in trees:
            try:
                st = subprocess.run(["git", "-C", t, "status", "--porcelain"], capture_output=True,
                                    text=True, timeout=60).stdout.strip().splitlines()
            except Exception:                                              # noqa: BLE001
                continue
            mod = [l for l in st if l[:2].strip() and not l.startswith("??")]
            if not mod:
                print(f"  clean   {t}")
                continue
            rc = 1
            gi = git_info(t)
            print(f"  DIRTY   {t}")
            print(f"          base {gi.get('describe') or '-'} {gi.get('commit') or '-'} "
                  f"({gi.get('date') or '-'})  remote {gi.get('remote') or '-'}")
            for l in mod[:14]:
                print(f"            {l}")
            if len(mod) > 14:
                print(f"            ... and {len(mod)-14} more")
            print(f"          capture it: pollard-runtime --scan {t} --capture <name>")
        if rc:
            print("\nUncommitted runtime changes are how spark2_5 was lost: a later edit to the same "
                  "tree overwrote it and a published model stopped loading. Capture them.")
        return rc

    if a.capture:
        rc = 0
        for t in trees:
            meta, msg = capture(t, a.capture, a.patch_dir)
            tag = "captured" if meta else "skipped "
            print(f"  {tag}  {t}: {msg}")
            if meta and meta.get("adds_strings"):
                print(f"            adds: {', '.join(meta['adds_strings'][:10])}")
        return rc

    if a.apply:
        rc = 0
        for t in trees:
            ok, msg = apply_patch(t, a.apply, a.patch_dir)
            print(f"  {'applied ' if ok else 'FAILED  '} {t}: {msg[:160]}")
            rc |= 0 if ok else 1
        return rc

    if a.verify:
        rows = verify_captured(trees, a.patch_dir)
        if not rows:
            print(f"no captured patches in {a.patch_dir}/ — nothing to verify.")
            return 0
        rc = 0
        for r in rows:
            print(f"  {r['state']:10s} {r['slug']}-{r['name']}")
            if r.get("wants"):
                print(f"             adds: {', '.join(r['wants'][:8])}")
            if r.get("added_lines"):
                print(f"             {r['added_lines'] - r.get('absent_lines', 0)}/"
                      f"{r['added_lines']} added lines present in the tree")
            if r["state"] == "DRIFTED":
                print("             most of it is there but the diff no longer matches -- the tree "
                      "moved under the patch; re-capture it")
            if r["state"] == "LOST":
                rc = 1
                if r.get("missing"):
                    print(f"             MISSING from source and binaries: {', '.join(r['missing'])}")
                print(f"             re-apply: pollard-runtime --scan {r['tree']} "
                      f"--apply {r['name']}")
            if r["state"] == "tree missing":
                rc = 1
                print(f"             no tree named `{r['slug']}` was scanned")
        return rc

    up = upstream_archs()

    print(f"upstream ggml-org master: {len(up)} architectures" if up
          else "upstream ggml-org master: UNREACHABLE (comparisons skipped)")

    rows = []
    for t in trees:
        src = _arch_list_from_source(t)
        gi = git_info(t)
        row = {"tree": t, "git": gi, "source_archs": len(src) if src else None}
        missing = sorted(up - src) if (up and src) else []
        row["behind"] = missing
        rows.append(row)
        print(f"\n{t}")
        print(f"   {gi.get('describe') or '-'}  {gi.get('commit') or '-'}  {gi.get('date') or '-'}")
        if gi.get("remote"):
            print(f"   remote {gi['remote']}")
        if src is None:
            print("   no llama-arch.cpp found (not a llama.cpp tree, or a different layout)")
            continue
        print(f"   knows {len(src)} architectures", end="")
        if up:
            print(f" -- {len(missing)} behind upstream" if missing else " -- current with upstream")
            if missing:
                print(f"   missing: {', '.join(missing[:12])}" + (" ..." if len(missing) > 12 else ""))
        else:
            print()

    if a.arch:
        print(f"\n=== who supports `{a.arch}` ===")
        print(f"   upstream master: {'YES' if up and a.arch in up else 'no' if up else 'unknown'}")
        for r in rows:
            src = _arch_list_from_source(r["tree"])
            in_src = bool(src and a.arch in src)
            in_bin, nfiles = archs_in_binaries(r["tree"], [a.arch])
            print(f"   {r['tree']}: source={'YES' if in_src else 'no'}  "
                  f"binaries={'YES' if a.arch in in_bin else 'no'} ({nfiles} scanned)")

    rc = 0
    if not a.no_published and up:
        pub = published_archs()
        if pub:
            print("\n=== architectures our published repos need ===")
            for arch, repos in sorted(pub.items()):
                holders = []
                for r in rows:
                    src = _arch_list_from_source(r["tree"])
                    if src and arch in src:
                        holders.append(os.path.basename(r["tree"].rstrip("/\\")))
                up_has = arch in up
                if holders:
                    state = f"loadable here ({', '.join(holders)})"
                elif up_has:
                    state = "UPSTREAM HAS IT, no build here does -- update a runtime"
                    rc = 1
                else:
                    state = "not upstream and not here -- needs the vendor fork"
                    rc = 1
                print(f"   {arch:16s} {state}")
                print(f"                    repos: {', '.join(sorted(repos))}")
    if a.json:
        print(json.dumps(rows, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
