#!/usr/bin/env python3
"""pollard-recard — bring already-published model repos onto the Pollard master card template,
without losing what their current cards already say.

Cards drift. Ours did: across 14 published repos there were 32 different section names and not one
section appeared in all of them, because early cards were hand-written and later ones hand-edited.
Regenerating them is easy; regenerating them WITHOUT dropping measured numbers is the hard part, and
that is what this does.

Three passes over each existing card, then a rebuild:

  1. numbers      the files table, in whatever shape it was written -- plain `file.gguf`, a markdown
                  link, or a headerless multi-column layout. PPL / tok-s / KLD / notes.
  2. sections     any `## ` section the master template does not generate, kept verbatim -- the
                  model-specific analysis a template cannot know.
  3. prose        sentences carrying measured numbers that live OUTSIDE any table, which pass 2
                  misses because they sit under a heading the template owns.

Sizes are then read from the REPO, not from the old card's prose -- which is how this found four
rungs on one of our repos whose stated sizes were GiB mislabelled as GB, understating every file.

  pollard-recard --repo PollardWeights/Ling-3.0-tiny-Pollard        # write locally, review
  pollard-recard --author PollardWeights --out regen                # the whole shelf
  pollard-recard --author PollardWeights --upload                   # ... and publish

Writes locally by default. Nothing is uploaded unless you pass --upload.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The published cards use at least three table shapes. Match the FILENAME wherever it sits in the row:
# a parser that only knew one shape silently dropped measured numbers from the others.
ROW = re.compile(r"^\|(.*?\.(?:gguf|safetensors).*)$", re.M)
FNAME = re.compile(r"\[?([A-Za-z0-9._\-]+\.(?:gguf|safetensors))\]?")
NUMERIC = re.compile(r"^\d+\.\d+$")
MEASURED = re.compile(r"\b(PPL|KLD|top-1|tok/s|bpw)\b", re.I)

# sections the master template generates itself; anything else is model-specific and must survive
TEMPLATE_SECTIONS = {
    "model details", "which file should i choose?", "available files", "prompt format", "multimodal",
    "fill-in-the-middle (code completion)", "download a specific file", "download", "download & run",
    "how to run", "how to run (text)", "imatrix (calibration)", "imatrix", "arm / avx", "errata",
    "credits", "credits & license", "usage", "notes", "verified",
}


def harvest(card, files):
    """(results, extra_markdown) from an existing card."""
    results, keep = {}, []

    b = re.search(r"^base_model:\s*(\S+)", card, re.M)
    if b:
        results["_base"] = b.group(1)
    for pat in (r"## Available files \(([^)]+)\)", r"## The numbers \(([^)]+)\)"):
        mt = re.search(pat, card)
        if mt:
            results["_eval"] = mt.group(1)
            break
    mt = re.search(r"f16 (?:reference )?(?:PPL|ppl)[^\d]*([\d.]+)", card)
    if mt:
        results["_f16_ppl"] = mt.group(1)

    hdr = None
    for line in card.splitlines():
        if line.startswith("|") and re.search(r"filename|file\b", line, re.I):
            hdr = [c.strip().lower() for c in line.strip("|").split("|")]
            break

    for mt in ROW.finditer(card):
        row = mt.group(1)
        fm = FNAME.search(row)
        if not fm:
            continue
        fn = fm.group(1)
        cells = [c.strip() for c in row.split("|")]
        rec = {}
        if hdr:
            for i, h in enumerate(hdr):
                if i >= len(cells):
                    break
                v = cells[i].strip()
                if not v or v == "—":
                    continue
                if "ppl" in h or "perplex" in h:
                    rec["ppl"] = v
                elif "tok" in h:
                    rec["tps"] = v
                elif "kld" in h:
                    rec["kld"] = v
                elif "description" in h or "note" in h:
                    rec["note"] = v
        if "ppl" not in rec:                      # headerless layout: first bare decimal is the PPL
            nums = [c for c in cells if NUMERIC.fullmatch(c)]
            if nums:
                rec["ppl"] = nums[0]
        if not rec.get("note"):
            tail = cells[-1] if cells else ""
            if tail and tail not in ("", "—") and ".gguf" not in tail:
                rec["note"] = tail
        if rec:
            # Cards abbreviate filenames ("`…-Q6_K.gguf`"), so a raw filename key matches nothing
            # downstream and the numbers vanish. Resolve to the real file, else key by quant tag.
            real = next((f for f in files if f.endswith(fn) or fn.endswith(f)), None)
            results[real or (fn[:-5].rstrip("-").split("-")[-1] or fn)] = rec

    for mt in re.finditer(r"^(## .+?)$(.*?)(?=^## |\Z)", card, re.S | re.M):
        title = mt.group(1)[3:].strip().lower()
        if title in TEMPLATE_SECTIONS or title.split(" (")[0] in TEMPLATE_SECTIONS:
            continue
        keep.append(mt.group(0).rstrip())

    rescued = []
    for para in re.split(r"\n\s*\n", card):
        p = para.strip()
        if not p or p.startswith(("|", "#", "---", "```", "- ", "> ")):
            continue
        if MEASURED.search(p) and re.search(r"\d+\.\d", p):
            rescued.append(p)
    if rescued:
        keep.append("## Measured notes\n\n" + "\n\n".join(rescued))

    return results, ("\n\n".join(keep) + "\n" if keep else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--repo", help="one repo id, e.g. you/Model-GGUF")
    g.add_argument("--author", help="every model repo under this HF account")
    ap.add_argument("--out", default="recard", help="directory for the regenerated cards")
    ap.add_argument("--upload", action="store_true", help="publish them (default: write locally only)")
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    sys.path.insert(0, HERE)
    import pollard_workspace as ws

    repos = [a.repo] if a.repo else [m.id for m in api.list_models(author=a.author)]
    os.makedirs(a.out, exist_ok=True)
    rows = []

    for rid in repos:
        name = rid.split("/")[-1]
        try:
            card = open(hf_hub_download(rid, "README.md")).read()
        except Exception:
            card = ""
        info = api.repo_info(rid, files_metadata=True)
        siblings = [(s.rfilename, s.size or 0) for s in info.siblings]
        weights = [(f, sz) for f, sz in siblings
                   if f.endswith((".gguf", ".safetensors")) and "mmproj" not in f]
        if not weights:
            rows.append((name, "skip", "no weight files")); continue

        results, extra = harvest(card, [f for f, _ in siblings])
        rp = os.path.join(a.out, f"{name}.results.json")
        json.dump(results, open(rp, "w"), indent=2)
        ep = None
        if extra:
            ep = os.path.join(a.out, f"{name}.extra.md")
            open(ep, "w").write(extra)

        # sizes come from the REPO, never from the old card's prose
        key = f"{rid}#recard"
        d = ws.model_dir(key, create=True)
        man = {"model": key, "builds": []}
        for fn, sz in sorted(weights, key=lambda x: -x[1]):
            tag = fn.replace(f"{name}-", "").rsplit(".", 1)[0]
            man["builds"].append({"name": fn, "lane": "gguf" if fn.endswith(".gguf") else "mlx",
                                  "tag": tag, "path": os.path.join(d, fn), "bpw": None, "ppl": None,
                                  "verified": True, "bytes": sz, "created": ""})
        json.dump(man, open(os.path.join(d, "MANIFEST.json"), "w"), indent=2)

        params = None
        src = results.get("_base") or rid
        try:
            st = api.model_info(src).safetensors
            if st and st.total:
                params = f"{st.total/1e9:.2f}B"
        except Exception:
            pass

        out_md = os.path.join(a.out, f"{name}.md")
        cmd = [sys.executable, os.path.join(HERE, "pollard_card.py"), "--model", src,
               "--title", name.replace("-Pollard", ""), "--builds-from", key,
               "--repo", rid, "--out", out_md, "--results", rp]
        if params:
            cmd += ["--params", params]
        if ep:
            cmd += ["--extra-md", ep]
        mm = next((f for f, _ in siblings if "mmproj" in f), None)
        im = next((f for f, _ in siblings if f.endswith(".imatrix")), None)
        if mm:
            cmd += ["--mmproj", mm]
        if im:
            cmd += ["--imatrix-file", im]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            rows.append((name, "FAIL", (r.stderr or r.stdout).strip().splitlines()[-1][:70])); continue

        # did any measured number in the old card fail to survive?
        new = open(out_md).read()
        onum = set(re.findall(r"\b\d+\.\d{2,}\b", card))
        lost = sorted(onum - set(re.findall(r"\b\d+\.\d{2,}\b", new)))
        note = f"{len(man['builds'])} files"
        if lost:
            note += f"; check {lost[:4]}"
        rows.append((name, "ok", note))

        if a.upload:
            api.upload_file(path_or_fileobj=out_md, path_in_repo="README.md", repo_id=rid,
                            repo_type="model", commit_message="Regenerate on the Pollard master card template")

    print(f"{'repo':44s} {'status':7s} detail")
    for n, st, d in sorted(rows):
        print(f"{n:44s} {st:7s} {d}")
    print(f"\ncards written to {a.out}/" + ("  (uploaded)" if a.upload else "  — review, then re-run with --upload"))
    print("A 'check' note lists numbers in the old card that are not in the new one. Some are "
          "corrections (real file sizes replacing stated ones); read them before publishing.")


if __name__ == "__main__":
    main()
