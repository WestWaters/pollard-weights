#!/usr/bin/env python3
"""pollard-reclaim - free the disk a finished model is still holding, once it is safely published.

A ladder is built, measured, uploaded, and then sits on the build box forever. Ours reached 796 GB of
GGUFs and safetensors, which is the thing that stops the next, bigger run: a 552B body needs more free
space than any one model's leftovers.

The rule this encodes: **a local build is only disposable once the published copy is proven identical.**
So nothing is judged by name or by a manifest's word. Every candidate is matched against the file the
Hub actually serves, by exact byte length, and anything that does not match is kept. Sources (the f16
body a ladder was quantized from) are never published, so they are reported separately -- they are
re-derivable from the base model, which is a slow download, not a lost artifact.

Deleting is never the default. Run it to see the bill, then opt in:

  pollard-reclaim                       # report only -- what is published, what is not, what it frees
  pollard-reclaim --delete              # remove only the rungs verified published
  pollard-reclaim --delete --sources    # also remove f16/bf16 sources (re-downloadable, not published)
  pollard-reclaim --scan DIR            # look outside $POLLARD_HOME too (build dirs, bench/, downloads/)
  pollard-reclaim --owner someone-else  # match against a different HF account

Users keep whatever they want -- their machine, their call. This only ever removes what it has proved
is still downloadable."""
import argparse
import json
import os
import sys
import urllib.request

SOURCE_HINTS = ("-f16", "-bf16", "_f16", "_bf16")
BUILD_MARK = "-Pollard"            # workspace build names carry this; a source never does
MODEL_EXT = (".gguf", ".safetensors")
PROBE = 65536                      # bytes sampled at head/middle/tail to confirm a size match


def _api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pollard-reclaim"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.load(fh)


def published_sizes(repo):
    """{filename: exact byte size} for a published repo, or None if it could not be read.

    None and {} differ: None is "I could not check", {} is "the repo is empty". Either way nothing is
    reclaimed, but the reason is worth printing.
    """
    try:
        info = _api(f"https://huggingface.co/api/models/{repo}?blobs=true")
    except Exception:
        return None
    return {s["rfilename"]: s.get("size") for s in info.get("siblings", [])}


def human(n):
    return f"{n/1e9:.2f} GB"


def _range(url, off, n):
    req = urllib.request.Request(url, headers={"Range": f"bytes={off}-{off+n-1}",
                                              "User-Agent": "pollard-reclaim"})
    with urllib.request.urlopen(req, timeout=90) as fh:
        return fh.read()


def same_content(path, repo, remote_name, nbytes):
    """Confirm a local file really is the published one, by sampling both.

    Equal byte length across multi-GB files is already a strong signal, but this tool deletes things,
    so it is not the whole test. Head, middle and tail are compared against the bytes the Hub serves.
    A build renamed on upload still matches -- the name is not what is being trusted.
    """
    url = f"https://huggingface.co/{repo}/resolve/main/{remote_name}"
    spots = [0, max(0, nbytes // 2 - PROBE // 2), max(0, nbytes - PROBE)]
    try:
        with open(path, "rb") as fh:
            for off in spots:
                want = min(PROBE, nbytes - off)
                if want <= 0:
                    continue
                fh.seek(off)
                if fh.read(want) != _range(url, off, want):
                    return False
        return True
    except Exception:
        return False


def is_source_name(path):
    """True for an unquantized body (the thing a ladder was built FROM).

    Deliberately filename-only AND excluding build names: the workspace puts a ladder inside a
    directory named after its f16 source, so a path test would flag every published rung as a source
    and offer to delete it under --sources.
    """
    base = os.path.basename(path)
    if BUILD_MARK in base:
        return False
    return any(h in base.lower() for h in SOURCE_HINTS)


def iter_manifests(home):
    root = os.path.join(home, "models")
    if not os.path.isdir(root):
        return
    for d in sorted(os.listdir(root)):
        mp = os.path.join(root, d, "MANIFEST.json")
        if os.path.isfile(mp):
            try:
                yield mp, json.load(open(mp))
            except Exception as e:                                         # noqa: BLE001
                print(f"WARNING: unreadable manifest {mp} ({e})", file=sys.stderr)


def scan_loose(dirs):
    out = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(MODEL_EXT):
                    p = os.path.join(root, f)
                    try:
                        out.append((p, os.path.getsize(p)))
                    except OSError:
                        pass
    return out


def repo_guesses(model, owner):
    """Repo ids a manifest key might have been published as.

    The manifest key is often the source GGUF's path, not an HF id -- e.g.
    `FrogMini-14B-f16.gguf` published as `PollardWeights/FrogMini-14B-Pollard` -- so the f16/bf16 and
    extension decoration is stripped before guessing.
    """
    base = os.path.basename(str(model).rstrip("/"))
    for ext in (".gguf", ".safetensors"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
    out = []
    for cand in (base,) + tuple(base[: -len(h)] for h in SOURCE_HINTS if base.lower().endswith(h)):
        cand = cand.rstrip("-_.")
        if cand and cand not in out:
            out.append(cand)
    return [f"{owner}/{c}-Pollard" for c in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--home", default=os.environ.get("POLLARD_HOME", os.path.expanduser("~/pollard")),
                    help="workspace to read manifests from (default $POLLARD_HOME)")
    ap.add_argument("--scan", action="append", default=[],
                    help="extra directory to scan for model files no manifest claims (repeatable)")
    ap.add_argument("--owner", default=None,
                    help="HF account the builds were published under (default: your Hugging Face "
                         "login)")
    ap.add_argument("--repo", action="append", default=[],
                    help="also check against these repo ids (repeatable). Needed when a build was "
                         "published under a name the manifest key does not imply.")
    ap.add_argument("--delete", action="store_true",
                    help="remove the files verified as published (default: report only)")
    ap.add_argument("--sources", action="store_true",
                    help="with --delete, also remove unquantized f16/bf16 bodies -- re-derivable by "
                         "downloading the base model, but published nowhere")
    ap.add_argument("--min-gb", type=float, default=0.0, help="ignore files smaller than this")
    ap.add_argument("--exclude", action="append", default=[],
                    help="skip paths containing this substring (repeatable) -- use it to protect a "
                         "file a running build or imatrix pass is reading")
    ap.add_argument("--no-verify", action="store_true",
                    help="trust an exact byte-length match without sampling the published file. "
                         "Faster, and strictly weaker evidence -- not recommended before --delete.")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()

    # Defaulting this to PollardWeights meant anyone else running reclaim checked their local builds
    # against OUR repos, so nothing ever matched and nothing was ever reclaimed. Their own account is
    # the only sensible default; deletion is gated on a positive match, so an unknown owner simply
    # finds nothing rather than removing anything.
    from pollard_card import publishing_account
    owner = a.owner or publishing_account()
    if not owner:
        sys.exit("no HF account known: pass --owner, or log in with `hf auth login`.")

    cands, seen, repos = [], set(), set(a.repo)
    for _, man in iter_manifests(a.home):
        guesses = repo_guesses(man.get("model") or "", owner)
        repos.update(guesses)
        for b in man.get("builds", []):
            p = b.get("path")
            if not p or not os.path.isfile(p):
                continue
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            if p not in seen:
                seen.add(p)
                # from a manifest, so it is a BUILD -- never treated as a source
                cands.append({"path": p, "bytes": sz, "repos": guesses, "build": True})
    for p, sz in scan_loose(a.scan):
        if p not in seen:
            seen.add(p)
            cands.append({"path": p, "bytes": sz, "repos": None, "build": False})

    if a.exclude:
        before = len(cands)
        cands = [c for c in cands if not any(x.lower() in c["path"].lower() for x in a.exclude)]
        print(f"excluded {before - len(cands)} file(s) by --exclude")
    cands = [c for c in cands if c["bytes"] >= a.min_gb * 1e9]
    if not cands:
        print("nothing to consider - no build files found.")
        return 0

    remote = {}
    for r in sorted(repos):
        remote[r] = published_sizes(r)
    live = {r: m for r, m in remote.items() if m}

    groups = {"published": [], "source": [], "keep": []}
    for c in sorted(cands, key=lambda x: -x["bytes"]):
        path, nbytes = c["path"], c["bytes"]
        # A local build is often renamed on upload, so the published copy is found by exact byte
        # length within the candidate repos and then CONFIRMED by sampling its bytes. The filename is
        # never the evidence.
        search = {r: live[r] for r in (c["repos"] or live.keys()) if r in live}
        hit = None
        for r, m in search.items():
            for rname, rsize in m.items():
                if rsize == nbytes:
                    hit = (r, rname)
                    break
            if hit:
                break
        if hit:
            r, rname = hit
            if a.no_verify or same_content(path, r, rname, nbytes):
                groups["published"].append({
                    "path": path, "bytes": nbytes, "repo": r, "remote": rname,
                    "reason": f"matches {r}/{rname}"
                    + (" by length only (--no-verify)" if a.no_verify else " (head/middle/tail verified)")})
                continue
            groups["keep"].append({"path": path, "bytes": nbytes, "repo": r, "remote": rname,
                                   "reason": f"same length as {rname} but the bytes differ"})
            continue
        if not c["build"] and is_source_name(path):
            groups["source"].append({"path": path, "bytes": nbytes, "repo": None,
                                     "reason": "unquantized body, published nowhere"})
            continue
        why = (f"no published file of this length in {len(search)} checked repo(s)") if search \
              else "no readable published repo to compare against"
        groups["keep"].append({"path": path, "bytes": nbytes, "repo": None, "reason": why})

    if a.json:
        print(json.dumps(groups, indent=1))
    else:
        for key, title in (
                ("published", "SAFE TO RECLAIM - the published copy is verified identical"),
                ("source", "SOURCES - published nowhere; re-derivable by downloading the base model"),
                ("keep", "KEEPING")):
            g = groups[key]
            print(f"\n{title}  ({len(g)} file(s), {human(sum(x['bytes'] for x in g))})")
            show = 40 if key != "keep" else 8
            for x in g[:show]:
                print(f"   {human(x['bytes']):>10}  {x['path']}")
                print(f"               {x['reason']}")
            if len(g) > show:
                print(f"   ... and {len(g)-show} more")

    freeable = sum(x["bytes"] for x in groups["published"])
    src = sum(x["bytes"] for x in groups["source"])
    if not a.json and groups["keep"]:
        # Everything held back, totalled by directory. Whether an experiment's scratch is still
        # wanted is a judgement this tool cannot make, so it reports the bill and stops.
        per = {}
        for x in groups["keep"]:
            per[os.path.dirname(x["path"])] = per.get(os.path.dirname(x["path"]), 0) + x["bytes"]
        print(f"\nHELD BACK, by directory  (nothing here is provably published)")
        for d, n in sorted(per.items(), key=lambda kv: -kv[1])[:20]:
            print(f"   {human(n):>10}  {d}")
        if len(per) > 20:
            print(f"   ... and {len(per)-20} more directories")
    if not a.json:
        print(f"\nreclaimable now: {human(freeable)}"
              + (f"; plus {human(src)} of sources with --sources" if src else ""))
    if not a.delete:
        if not a.json:
            print("Nothing was deleted. Re-run with --delete to remove the verified-published files"
                  + (", or --delete --sources to include the sources." if src else "."))
        return 0

    doomed = list(groups["published"]) + (list(groups["source"]) if a.sources else [])
    removed, failed = 0, 0
    for x in doomed:
        try:
            os.remove(x["path"])
            removed += x["bytes"]
            print(f"removed {human(x['bytes']):>10}  {x['path']}")
        except OSError as e:
            failed += 1
            print(f"FAILED  {x['path']}: {e}", file=sys.stderr)
    print(f"\nfreed {human(removed)}" + (f"; {failed} could not be removed" if failed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
