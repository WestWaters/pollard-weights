#!/usr/bin/env python3
"""pollard-ls — list what's in the Pollard workspace so you never hunt for a build. Reads each model's
MANIFEST.json and prints every build: lane, quant, size, PPL, and whether it passed pollard-verify.

  pollard-ls                 # everything in $POLLARD_HOME (default ~/pollard)
  pollard-ls Qwen2.5-3B      # just builds whose model matches this substring
  pollard-ls --paths         # print full paths (for scripting)
"""
import argparse, os
import pollard_workspace as ws


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("filter", nargs="?", default="", help="only models matching this substring")
    ap.add_argument("--paths", action="store_true", help="print full build paths")
    a = ap.parse_args()

    home = ws.pollard_home()
    print(f"Pollard workspace: {home}")
    models = [m for m in ws.list_models() if a.filter.lower() in m.lower()]
    if not models:
        print("  (no builds yet — convert a model and it lands here automatically, or set $POLLARD_HOME)")
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
            vflag = "✓verified" if v is True else ("✗FAILED" if v is False else "· unverified")
            ppl = f"ppl {b['ppl']:.2f}" if b.get("ppl") is not None else ""
            bpw = f"{b['bpw']:.2f}bpw" if b.get("bpw") is not None else ""
            meta = "  ".join(x for x in (b.get("lane", "").upper(), bpw, ws.human(b.get("bytes")), ppl, vflag) if x)
            print(f"   {b.get('name','?'):<44} {meta}")
            if a.paths:
                print(f"       {b.get('path','')}")
    print(f"\ntotal: {ws.human(total)} across {len(models)} model(s)")


if __name__ == "__main__":
    main()
