#!/usr/bin/env python3
"""pollard-ls -- list what's in the Pollard workspace so you never hunt for a build. Reads each model's
MANIFEST.json and prints every build: lane, quant, size, PPL, and whether it passed pollard-verify.

  pollard-ls                 # everything in $POLLARD_HOME (default ~/pollard): builds + pulled sources
  pollard-ls Qwen2.5-3B      # just entries whose model matches this substring
  pollard-ls --paths         # print full paths (for scripting)
  pollard-ls --clean-downloads   # delete downloads/ (source models are re-fetchable) to reclaim disk
  pollard-ls --clean-staging     # reclaim staging/ (relay upload copies): empty folders + copies already on the Hub
  pollard-ls --clean-staging --force   # ...and every other staging copy too (each is only a copy of a rung)
  pollard-ls --clean-cache       # cache/ (verify refs, KL bases, probe work dirs, toolcall caches) -- not what a
                                 # running job may be using (a file open, or written in the last --min-age minutes)
  pollard-ls --clean-mirrors     # mirror/<model>/ once every build its MANIFEST records is confirmed on the Hub
                                 # (size + sha256); --force removes unconfirmed ones too, --repo names the Hub repo
  pollard-ls --clean-all         # all of it: downloads/ (confirmed on the Hub, or all with --force), staging/,
                                 # cache/ and mirror/, then a table of what each freed

Only paths strictly inside $POLLARD_HOME are ever removed, symlinks are never followed out, and models/ (the
builds) is never touched.
"""
import argparse, os
import pollard_workspace as ws


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("filter", nargs="?", default="", help="only models matching this substring")
    ap.add_argument("--paths", action="store_true", help="print full build paths")
    ap.add_argument("--clean-downloads", action="store_true",
                    help="delete the downloads/ staging tree (pulled source models are re-fetchable)")
    ap.add_argument("--clean-staging", action="store_true",
                    help="reclaim staging/ (copies waiting to be uploaded from here): removes empty folders and copies "
                         "already confirmed on the Hub (size + sha256); reports the rest")
    ap.add_argument("--clean-cache", action="store_true",
                    help="reclaim cache/ (verify reference logits, KL bases, probe / eval work dirs, toolcall caches -- "
                         "all rebuilt on demand); keeps anything a running job may be using")
    ap.add_argument("--clean-mirrors", action="store_true",
                    help="reclaim mirror/<model>/ (sparse / streamed source mirrors) once every build its MANIFEST "
                         "records is confirmed on the Hub (size + sha256); keeps the rest, and everything when the "
                         "Hub cannot be read")
    ap.add_argument("--clean-all", action="store_true",
                    help="downloads/ (only sources whose builds are all confirmed on the Hub; all with --force), "
                         "staging/, cache/ and mirror/ -- then a table of the space each freed")
    ap.add_argument("--force", action="store_true",
                    help="with --clean-staging / --clean-mirrors / --clean-all: also remove copies and mirrors NOT "
                         "confirmed on the Hub (said out loud for each). Anything a running job may be using is "
                         "kept even then")
    ap.add_argument("--repo", action="append", default=[],
                    help="the HF repo to check against when a MANIFEST does not record one (with --clean-staging: "
                         "match unmarked legacy copies against it)")
    ap.add_argument("--min-age", type=float, default=ws.CLEAN_MIN_AGE_MIN, metavar="MIN",
                    help=f"keep cache entries / mirrors written in the last MIN minutes (default "
                         f"{ws.CLEAN_MIN_AGE_MIN}) -- they may belong to a running job")
    a = ap.parse_args()
    _API.clear()

    if a.clean_all:
        return clean_all(a)
    if a.clean_staging:
        freed, plan = ws.sweep_staging(_hub_api(), a.repo, force=a.force)
        kept = [e for e, act, _ in plan if act != "remove"]
        print(f"cleaned staging/ -- freed {ws.human(freed)}"
              + (f"; kept {len(kept)} copy(ies), {ws.human(sum(e['bytes'] for e in kept))}" if kept else ""))
        if not (a.clean_cache or a.clean_mirrors):
            return
    if a.clean_cache:
        freed, plan = ws.clean_cache(min_age_min=a.min_age)
        print(f"cleaned cache/ -- freed {ws.human(freed)}" + _kept(plan))
    if a.clean_mirrors:
        freed, plan = ws.clean_mirrors(_hub_api(), a.repo, force=a.force, min_age_min=a.min_age)
        print(f"cleaned mirror/ -- freed {ws.human(freed)}" + _kept(plan))
    if a.clean_staging or a.clean_cache or a.clean_mirrors:
        return
    if a.force:
        ap.error("--force goes with --clean-staging, --clean-mirrors or --clean-all")

    if a.clean_downloads:
        freed = ws.clean_downloads()
        print(f"cleared downloads/ -- freed {ws.human(freed)}")
        return

    home = ws.pollard_home()
    print(f"Pollard workspace: {home}")
    models = [m for m in ws.list_models() if a.filter.lower() in m.lower()]
    if not models:
        print("  (no builds yet -- convert a model and it lands here automatically, or set $POLLARD_HOME)")
        _scratch(a)
        return

    total = 0
    for slug in models:
        man = ws.read_manifest(os.path.join(home, "models", slug))
        builds = man.get("builds", [])
        print(f"\n{slug}  ({len(builds)} build{'s' if len(builds) != 1 else ''})")
        if not builds:
            print("   (folder present, no recorded builds)")
        for b in sorted(builds, key=lambda x: (x.get("lane", ""), x.get("tag", ""))):
            total += b.get("bytes") or 0
            v = b.get("verified")
            vflag = "okverified" if v is True else ("xFAILED" if v is False else " |  unverified")
            ppl = f"ppl {b['ppl']:.2f}" if b.get("ppl") is not None else ""
            bpw = f"{b['bpw']:.2f}bpw" if b.get("bpw") is not None else ""
            meta = "  ".join(x for x in (b.get("lane", "").upper(), bpw, ws.human(b.get("bytes")), ppl, vflag) if x)
            print(f"   {b.get('name','?'):<44} {meta}")
            if a.paths:
                print(f"       {b.get('path','')}")
    print(f"\ntotal: {ws.human(total)} across {len(models)} model(s)")

    # pulled SOURCE models (the input side) -- show them so they're findable and easy to reclaim
    dl = ws.downloads_dir()
    if os.path.isdir(dl):
        srcs = sorted(d for d in os.listdir(dl) if os.path.isdir(os.path.join(dl, d)) and not d.startswith("."))
        if srcs:
            dtotal = 0
            print(f"\ndownloads/ (pulled sources):")
            for s in srcs:
                if a.filter.lower() not in s.lower():
                    continue
                sz = ws._dir_size(os.path.join(dl, s)); dtotal += sz
                print(f"   {s:<44} {ws.human(sz)}")
                if a.paths:
                    print(f"       {os.path.join(dl, s)}")
            if dtotal:
                print(f"   (source total: {ws.human(dtotal)} -- reclaim with `pollard-ls --clean-downloads`, or "
                      "`pollard-ls --clean-all` for only the sources whose builds are on the Hub, plus staging/ "
                      "cache/ mirror/)")
    _scratch(a)


_API = []


def _hub_api():
    """One HfApi for the run, or None (huggingface_hub not installed: nothing can be confirmed on the Hub)."""
    if not _API:
        try:
            from huggingface_hub import HfApi
            _API.append(HfApi())
        except ImportError:
            _API.append(None)
            print("   (huggingface_hub not installed: nothing can be confirmed on the Hub -- only what needs no Hub "
                  "check goes)")
    return _API[0]


def _kept(plan):
    kept = [e for e, act, _ in plan if act != "remove"]
    return f"; kept {len(kept)}, {ws.human(sum(e['bytes'] for e in kept))}" if kept else ""


def clean_all(a):
    """downloads/ + staging/ + cache/ + mirror/, then a table of what each freed and kept."""
    rows = []
    api = _hub_api()
    f, plan = ws.clean_downloads_confirmed(api, a.repo, force=a.force, min_age_min=a.min_age)
    rows.append(("downloads/", f) + _counts(plan))
    f, plan = ws.sweep_staging(api, a.repo, force=a.force)
    rows.append(("staging/", f) + _counts(plan))
    f, plan = ws.clean_cache(min_age_min=a.min_age)
    rows.append(("cache/", f) + _counts(plan))
    f, plan = ws.clean_mirrors(api, a.repo, force=a.force, min_age_min=a.min_age)
    rows.append(("mirror/", f) + _counts(plan))
    print(f"\nreclaimed in {ws.pollard_home()}:")
    print(f"   {'':<12} {'freed':>10}  {'kept':>10}")
    for name, freed, nkept, kbytes in rows:
        print(f"   {name:<12} {ws.human(freed):>10}  {(ws.human(kbytes) + f' ({nkept})') if nkept else '-':>10}")
    print(f"   {'total':<12} {ws.human(sum(r[1] for r in rows)):>10}")
    if any(r[2] for r in rows) and not a.force:
        print("   (kept = not confirmed on the Hub, or possibly in use -- see the lines above; --force removes what "
              "is only unconfirmed)")


def _counts(plan):
    kept = [e for e, act, _ in plan if act != "remove"]
    return len(kept), sum(e["bytes"] for e in kept)


def _scratch(a):
    """staging/ (relay upload copies) and cache/ (work dirs): what they hold, so nothing fills the disk unseen."""
    st = ws.staging_usage()
    if st:
        print(f"\nstaging/ (copies waiting to be uploaded from here):")
        for e in st:
            rel = os.path.relpath(e["path"], ws.staging_root())
            mark = e.get("mark") or {}
            what = f"{mark.get('repo_id')}:{mark.get('path_in_repo') or '/'}" if mark.get("repo_id") else "no marker"
            print(f"   {rel:<44} {ws.human(e['bytes']):>8}  {what}")
            if a.paths:
                print(f"       {e['path']}")
        print(f"   (staging total: {ws.human(sum(e['bytes'] for e in st))} -- reclaim with `pollard-ls --clean-staging`)")
    cd = ws.cache_dir()
    if os.path.isdir(cd):
        n = ws._dir_size(cd)
        if n:
            print(f"\ncache/ (work dirs, safe to delete): {ws.human(n)} -- reclaim with `pollard-ls --clean-cache`"
                  + (f"\n       {cd}" if a.paths else ""))
    mirrors = ws.source_entries(ws.mirror_root())
    if mirrors:
        print("\nmirror/ (sparse / streamed source mirrors):")
        free = 0
        for e in mirrors:
            ok, why = ws.source_status(e["path"])
            free += e["bytes"] if ok else 0
            print(f"   {os.path.basename(e['path']):<44} {ws.human(e['bytes']):>8}  "
                  f"{'reclaimable' if ok else 'kept'}: {why}")
            if a.paths:
                print(f"       {e['path']}")
        total = sum(e["bytes"] for e in mirrors)
        print(f"   (mirror total: {ws.human(total)}"
              + (f"; reclaimable {ws.human(free)} -- `pollard-ls --clean-mirrors` (confirms each on the Hub by "
                 f"size + sha256 first)" if free else "; none confirmed published -- `pollard-ls --clean-mirrors "
                 "--force` removes them anyway") + ")")


if __name__ == "__main__":
    main()
