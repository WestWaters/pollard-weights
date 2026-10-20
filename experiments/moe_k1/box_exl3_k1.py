#!/usr/bin/env python3
"""box_exl3_k1.py -- the EXL3 K1 lane on a small real MoE (box: RTX 5070 Ti 16 GB; later the Spark), Pollard's
per-expert allocation vs exllamav3's own allocator at EQUAL bpw. NOT for a shared GPU: it refuses when another
process is resident on the card (the box's GPU is shared -- run it when the GPU is ours), and it never assumes a
model fits: the HF bf16 reference must fit the free VRAM (pollard-verify's measure_exl3 loads it).

What it runs, per target bpw (default 1.6, 2.0, 2.5):
  0. routing: `pollard-moefit routing --imatrix <GGUF imatrix of the SAME checkpoint>` (llama.cpp keeps HF's expert
     order), or --routing JSON. Sensitivity: --sensitivity (pollard-sensitivity / pollard-probe JSON) -- optional.
  1. pollard-k1-layer : `pollard-moefit exl3-recipe --granularity layer` (one K per layer projection: what STOCK
                        exllamav3 loads -- its MoE fast path asserts one K across a projection's experts,
                        modules/multilinear.py) -> convert.py -rcp
  2. pollard-k1-expert: `--granularity expert` (one K per expert matrix, the SAGE-EXL3 shape). Built, then a LOAD
                        CHECK: stock exllamav3 is expected to refuse mixed K inside one layer (MultiLinear assert);
                        the row records exactly what happened rather than assuming.
  3. exl3-own         : exllamav3's own budgeted allocator, convert.py -b <bpw> (the baseline at equal bpw)
  4. pollard-verify --lane exl3 --eval <held-out> (KLD / top-1 vs the bf16 source through exllamav3's kernels) and
     pollard-toolcall --model <rung> --ref <source> (tool retention).

    python box_exl3_k1.py --model C:\\pollard\\m\\granite-3.1-1b-a400m-instruct --imatrix full.imatrix.gguf \\
        --eval kld_eval.txt --exllamav3 C:\\pollard\\exllamav3 --work C:\\pollard\\k1 [--sensitivity s.json]
    python box_exl3_k1.py ... --dry-run          # print every command, run nothing (no GPU needed)

Results land in <work>/exl3_k1_results.json; the table prints at the end."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)


def gpu_check(min_free_gb, allow_shared):
    """(ok, why): nvidia-smi says the card is free enough and nobody else is on it."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False, "no nvidia-smi (no NVIDIA GPU here)"
    q = subprocess.run([smi, "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    p = subprocess.run([smi, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                       capture_output=True, text=True)
    try:
        free, total = (float(x) / 1024 for x in q.stdout.strip().splitlines()[0].split(","))
    except Exception:
        return False, f"could not read nvidia-smi: {q.stdout or q.stderr}"
    others = [l for l in p.stdout.strip().splitlines() if l.strip()]
    if others and not allow_shared:
        return False, f"another process is resident on the GPU ({'; '.join(others)}) -- not ours to take"
    if free < min_free_gb:
        return False, f"{free:.1f} GB free of {total:.1f} GB < {min_free_gb} GB needed"
    return True, f"{free:.1f} GB free of {total:.1f} GB"


def run(cmd, dry, log_path=None):
    print("$ " + " ".join(map(str, cmd)), flush=True)
    if dry:
        return 0, ""
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(r.stdout + r.stderr)
    return r.returncode, r.stdout + r.stderr


def load_check(rung, dry):
    """does stock exllamav3 load this rung? (the mixed-K-within-a-layer question)"""
    code = ("from exllamav3 import Config, Model\n"
            f"c = Config.from_directory({rung!r})\nm = Model.from_config(c)\nm.load()\nprint('LOADED')\n")
    rc, out = run([sys.executable, "-c", code], dry)
    if dry:
        return {"load": "dry-run"}
    if rc == 0 and "LOADED" in out:
        return {"load": "ok"}
    tail = [l for l in out.strip().splitlines() if l.strip()][-3:]
    return {"load": "refused", "load_error": " | ".join(tail)[:400]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF bf16 checkpoint of the MoE")
    ap.add_argument("--imatrix", help="GGUF-format imatrix of the same checkpoint (per-expert routing counts)")
    ap.add_argument("--routing", help="routing JSON (pollard-moefit routing --out) instead of --imatrix")
    ap.add_argument("--sensitivity")
    ap.add_argument("--eval", required=True, help="held-out text for the KLD measurement")
    ap.add_argument("--exllamav3", required=True, help="exllamav3 checkout (convert.py)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--bpw", default="1.6,2.0,2.5")
    ap.add_argument("--head-bits", default="8")
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--allow-shared-gpu", action="store_true", help="run even with another process on the GPU")
    ap.add_argument("--no-tool", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if not a.dry_run:
        ok, why = gpu_check(a.min_free_gb, a.allow_shared_gpu)
        print(f"GPU: {why}")
        if not ok:
            sys.exit("REFUSED: " + why)
    py = sys.executable
    moefit = [py, os.path.join(TOOLS, "pollard_moefit.py")]
    routing = a.routing or os.path.join(a.work, "routing.json")
    if not a.routing:
        if not a.imatrix:
            sys.exit("give --imatrix (GGUF imatrix of this checkpoint) or --routing")
        run(moefit + ["routing", "--imatrix", a.imatrix, "--out", routing], a.dry_run)
    convert = os.path.join(a.exllamav3, "convert.py")
    res_path = os.path.join(a.work, "exl3_k1_results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for bpw in [float(x) for x in a.bpw.split(",") if x.strip()]:
        arms = {
            f"pollard-k1-layer-{bpw}": ["layer"],
            f"pollard-k1-expert-{bpw}": ["expert"],
            f"exl3-own-{bpw}": None,
        }
        for tag, gran in arms.items():
            if tag in results and results[tag].get("mean_kld") is not None:
                continue
            out = os.path.join(a.work, tag)
            wdir = os.path.join(a.work, "w-" + tag)
            row = {"bpw_target": bpw}
            if gran:
                rec = os.path.join(a.work, tag + ".recipe.yaml")
                cmd = moefit + ["exl3-recipe", "--model", a.model, "--routing", routing, "--bpw", str(bpw),
                                "--granularity", gran[0], "--head-bits", a.head_bits, "--out", rec]
                if a.sensitivity:
                    cmd += ["--sensitivity", a.sensitivity]
                run(cmd, a.dry_run)
                conv = [py, convert, "-i", a.model, "-o", out, "-w", wdir, "-rcp", rec, "-hb", a.head_bits]
            else:
                conv = [py, convert, "-i", a.model, "-o", out, "-w", wdir, "-b", str(bpw), "-hb", a.head_bits]
            rc, _ = run(conv, a.dry_run, os.path.join(a.work, tag + ".convert.log"))
            if rc:
                row["status"] = f"convert failed ({rc}) -- see {tag}.convert.log"
                results[tag] = row
                continue
            if gran and gran[0] == "expert":
                row.update(load_check(out, a.dry_run))
                if row.get("load") == "refused":
                    results[tag] = row
                    json.dump(results, open(res_path, "w"), indent=1)
                    continue
            vj = os.path.join(a.work, tag + ".verify.json")
            run([py, os.path.join(TOOLS, "pollard_verify.py"), "--model", out, "--source", a.model, "--lane", "exl3",
                 "--eval", a.eval, "--results", vj, "--rung", tag], a.dry_run, os.path.join(a.work, tag + ".verify.log"))
            if not a.no_tool:
                run([py, os.path.join(TOOLS, "pollard_toolcall.py"), "--model", out, "--ref", a.model,
                     "--out", os.path.join(a.work, tag + ".toolcall.json")], a.dry_run,
                    os.path.join(a.work, tag + ".toolcall.log"))
            if not a.dry_run and os.path.exists(vj):
                v = json.load(open(vj)).get(tag, {})
                row.update({"mean_kld": v.get("kld"), "top1": v.get("top1"), "ppl": v.get("ppl")})
            tj = os.path.join(a.work, tag + ".toolcall.json")
            if not a.dry_run and os.path.exists(tj):
                t = json.load(open(tj))
                row["tool_retention"] = None if t.get("retention") is None else round(100 * t["retention"], 1)
            results[tag] = row
            json.dump(results, open(res_path, "w"), indent=1)
    if a.dry_run:
        return 0
    print("\n| rung | target bpw | mean KLD | top-1 | tool retention | load |")
    print("|---|---|---|---|---|---|")
    for k, v in sorted(results.items()):
        print(f"| {k} | {v.get('bpw_target')} | {v.get('mean_kld')} | {v.get('top1')} | {v.get('tool_retention')} | "
              f"{v.get('load', 'ok')} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
