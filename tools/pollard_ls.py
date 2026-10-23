#!/usr/bin/env python3
"""pollard-ls -- list what's in the Pollard workspace so you never hunt for a build. Reads each model's
MANIFEST.json and prints every build: lane, quant, size, PPL, and whether it passed pollard-verify.

  pollard-ls                 # everything in $POLLARD_HOME (default ~/pollard): builds + pulled sources
  pollard-ls Qwen2.5-3B      # just entries whose model matches this substring
  pollard-ls --paths         # print full paths (for scripting)
  pollard-ls --clean-downloads   # delete downloads/ (source models are re-fetchable) to reclaim disk
  pollard-ls --clean-staging     # reclaim staging/ (relay upload copies): empty folders + copies already on the Hub
  pollard-ls --clean-staging --force   # ...and every other staging copy too (each is only a copy of a rung)
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
    ap.add_argument("--force", action="store_true",
                    help="with --clean-staging: also remove staging copies NOT confirmed on the Hub (a staging copy "
                         "is only a copy -- the rung stays on the machine that built it until its upload is confirmed)")
    ap.add_argument("--repo", action="append", default=[],
                    help="with --clean-staging: also match unmarked legacy copies against this HF repo")
    a = ap.parse_args()

    if a.clean_staging:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
        except ImportError:
            api = None
            print("   (huggingface_hub not installed: nothing can be confirmed on the Hub -- only empty folders go)")
        freed, plan = ws.sweep_staging(api, a.repo, force=a.force)
        kept = [e for e, act, _ in plan if act != "remove"]
        print(f"cleaned staging/ -- freed {ws.human(freed)}"
              + (f"; kept {len(kept)} copy(ies), {ws.human(sum(e['bytes'] for e in kept))}" if kept else ""))
        return
    if a.force:
        ap.error("--force goes with --clean-staging")

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
                print(f"   (source total: {ws.human(dtotal)} -- reclaim with `pollard-ls --clean-downloads`)")
    _scratch(a)


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
            print(f"\ncache/ (work dirs, safe to delete): {ws.human(n)}" + (f"\n       {cd}" if a.paths else ""))


if __name__ == "__main__":
    main()
