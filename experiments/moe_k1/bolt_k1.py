#!/usr/bin/env python3
"""bolt_k1.py -- Nemotron-3.5-Bolt-74B-A9B at ~1.6-2.5 bpw with Pollard's per-expert-tensor allocation, judged at
EQUAL BYTES against NVIDIA's own QAT pack (Nemotron-3.5-Bolt-24GB-GGUF, 22.02 GB: experts QAT IQ2_XXS, Mamba
ssm_in/out NVFP4, attention / latent / embeddings / head Q8_0, avg 2.378 bpw).

Two hosts, two jobs:

  --host box    (RTX 5070 Ti 16 GB + 32 GB RAM; the GPU is shared -> CPU only, -ngl 0, --threads 19)
                LANE A, from NVIDIA's QAT GGUF: keep every QAT tensor byte-for-byte unless the measured knapsack moves
                it DOWN (cold expert tensors IQ2_XXS -> IQ1_M / IQ1_S); nothing is ever upgraded. The 22 GB QAT file
                is its own host for the imatrix and the KLD reference (it is what is being preserved).
  --host spark  (DGX Spark, GB10, 128 GB unified; Linux, CUDA 13; ours -> -ngl 99)
                LANE B, from the pre-QAT BF16 checkpoint: pollard-moefit errmap on the bf16 GGUF (one llama-quantize
                per type, iq1_s ... q8_0), REAP saliency streamed one block at a time (pollard-reap --stream: 150 GB
                bf16 > 128 GB), Mario's arm (REAP-prune 50% / 25% of routed experts -> ~4-bit at equal bytes, then
                ~2-bit at ~half), all judged against a Q8_0 host's logits (bf16 does not fit a forward pass) AND the
                official QAT pack measured the same way.

Every rung: llama-perplexity --kl-divergence (mean / p99 KLD, top-1, PPL) on a held-out eval that includes
thinking traces (pollard-calib --held-out, enable_thinking), and pollard-toolcall retention vs the reference.

Needs llama.cpp >= cd0a5b6 (nemotron_h_moe, #30520) -- checked. The Mac's Metal mul_mat_id segfaults on this arch
(#30610): do not run it there.

    python bolt_k1.py --host box   --work C:\\pollard\\bolt --qat-gguf C:\\pollard\\m\\Bolt-24GB\\model.gguf \\
        --llama-bin C:\\pollard\\llama.cpp\\build\\bin\\Release --calib calib.txt --eval held.txt
    python bolt_k1.py --host spark --work /data/bolt --bf16 /data/hf/Nemotron-3.5-Bolt-74B-A9B-BF16 \\
        --qat-gguf /data/hf/Bolt-24GB-GGUF/model.gguf --llama-src ~/llama.cpp --calib calib.txt --eval held.txt
    add --dry-run to print the whole plan without running anything.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)

NEED_COMMIT = "cd0a5b6"                      # llama.cpp #30520 (nemotron_h_moe)
QAT_GB = 22.02                                # NVIDIA's pack
HOSTS = {"box": {"ngl": "0", "threads": "19", "disk_floor_gb": 60, "lanes": ("A",)},
         "spark": {"ngl": "99", "threads": None, "disk_floor_gb": 450, "lanes": ("A", "B")}}


def exe(bin_dir, name):
    for cand in (name, name + ".exe"):
        p = os.path.join(bin_dir, cand)
        if os.path.exists(p):
            return p
    return os.path.join(bin_dir, name)


class Run:
    def __init__(self, a):
        self.a, self.h = a, HOSTS[a.host]
        self.w = os.path.abspath(a.work)
        os.makedirs(self.w, exist_ok=True)
        b = a.llama_bin or os.path.join(a.llama_src or "", "build", "bin")
        self.q, self.imat, self.ppl = (exe(b, n) for n in ("llama-quantize", "llama-imatrix", "llama-perplexity"))
        self.res_path = self.p("bolt_results.json")

    def p(self, *x):
        return os.path.join(self.w, *x)

    def sh(self, cmd, log=None, need_gb=0):
        print("$ " + " ".join(map(str, cmd)), flush=True)
        if self.a.dry_run:
            return ""
        free = shutil.disk_usage(self.w).free / 1e9
        if need_gb and free - need_gb < self.h["disk_floor_gb"]:
            sys.exit(f"STOP: {free:.0f} GB free, this step needs ~{need_gb} GB over the {self.h['disk_floor_gb']} GB floor")
        r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
        if log:
            open(self.p(log), "a", encoding="utf-8").write(r.stdout + r.stderr)
        if r.returncode:
            sys.exit(f"FAILED: {' '.join(map(str, cmd))}\n" + "\n".join((r.stdout + r.stderr).splitlines()[-12:]))
        return r.stdout + r.stderr

    def ngl(self):
        return ["-ngl", self.h["ngl"]] + (["-t", self.h["threads"]] if self.h["threads"] else [])

    def tool(self, name, *args):
        return [sys.executable, os.path.join(TOOLS, name)] + list(args)

    def save(self, tag, row):
        r = json.load(open(self.res_path)) if os.path.exists(self.res_path) else {}
        r[tag] = {**r.get(tag, {}), **row}
        json.dump(r, open(self.res_path, "w"), indent=1)

    def done(self, tag):
        r = json.load(open(self.res_path)) if os.path.exists(self.res_path) else {}
        return r.get(tag, {}).get("mean_kld") is not None


def check_llama(r):
    """the runtime must contain #30520; with a source checkout the commit is checked, else --version is shown."""
    if r.a.dry_run:
        return
    src = r.a.llama_src
    if src and os.path.isdir(os.path.join(src, ".git")):
        ok = subprocess.run(["git", "-C", src, "merge-base", "--is-ancestor", NEED_COMMIT, "HEAD"]).returncode == 0
        if not ok:
            sys.exit(f"llama.cpp at {src} predates {NEED_COMMIT} (nemotron_h_moe): update it and rebuild ALL targets")
    out = subprocess.run([r.q, "--version"], capture_output=True, text=True)
    print("llama-quantize:", (out.stdout + out.stderr).strip().splitlines()[-1:] or "?")


def kld(r, model, base, tag):
    out = r.sh([r.ppl, "-m", model, "-f", r.a.eval, "--kl-divergence", "--kl-divergence-base", base, "-c", "512",
                "--chunks", str(r.a.kld_chunks)] + r.ngl(), f"{tag}.kld.log")

    def g(pat):
        m = re.search(pat, out, re.M)
        return float(m.group(1)) if m else None
    return {"ppl": g(r"^Mean PPL\(Q\)\s*:\s*([0-9.]+)"), "mean_kld": g(r"^Mean\s+KLD:\s*([0-9.]+)"),
            "p99_kld": g(r"^99\.0%\s+KLD:\s*([0-9.]+)"), "top1": g(r"^Same top p:\s*([0-9.]+)")}


def toolgate(r, model, ref, tag):
    out = r.p(f"{tag}.toolcall.json")
    r.sh(r.tool("pollard_toolcall.py", "--gguf", model, "--ref", ref, "--ngl", r.h["ngl"], "--out", out),
         f"{tag}.toolcall.log")
    if r.a.dry_run or not os.path.exists(out):
        return {}
    d = json.load(open(out))
    return {"tool_retention": None if d.get("retention") is None else round(100 * d["retention"], 1)}


def rung(r, tag, src, imatrix, em, budget, base, ref, extra=(), requant=False):
    if r.done(tag):
        return
    plan = r.p(f"{tag}.plan.json")
    r.sh(r.tool("pollard_moefit.py", "plan", "--errmap", em, "--bytes", str(int(budget)), "--embd", "q8_0",
                "--head", "q8_0", "--out", plan, *extra))
    out = r.p(f"{tag}.gguf")
    r.sh(r.tool("pollard_moefit.py", "build", "--gguf", src, "--imatrix", imatrix, "--plan", plan, "--out", out,
                "--llama-quantize", r.q, *(["--allow-requantize"] if requant else [])), f"{tag}.build.log",
         need_gb=budget / 1e9 + 1)
    row = {"bytes": None if r.a.dry_run else os.path.getsize(out), **({} if r.a.dry_run else kld(r, out, base, tag))}
    if not r.a.no_tool:
        row.update(toolgate(r, out, ref, tag))
    r.save(tag, row)


def lane_a(r):
    """from NVIDIA's QAT GGUF: per-expert-tensor downgrades only, judged against the QAT file itself."""
    qat = r.a.qat_gguf
    imx = r.p("qat.imatrix.gguf")
    if not os.path.exists(imx):
        r.sh([r.imat, "-m", qat, "-f", r.a.calib, "-o", imx, "-c", "512", "--chunks", str(r.a.imatrix_chunks)]
             + r.ngl(), "qat.imatrix.log")
    base = r.p("qat.base.kld")
    if not os.path.exists(base):
        r.sh([r.ppl, "-m", qat, "-f", r.a.eval, "--kl-divergence-base", base, "-c", "512", "--chunks",
              str(r.a.kld_chunks)] + r.ngl(), "qat.base.log", need_gb=5)
    em = r.p("qat.errmap.json")
    if not os.path.exists(em):
        r.sh(r.tool("pollard_moefit.py", "errmap", "--gguf", qat, "--imatrix", imx, "--out", em, "--allow-requantize",
                    "--types", "iq1_s,iq1_m,iq2_xxs,iq4_xs,q6_K,q8_0", "--llama-quantize", r.q,
                    "--workdir", r.p("probes-qat")), "qat.errmap.log", need_gb=30)
    r.sh(r.tool("pollard_moefit.py", "routing", "--imatrix", imx, "--out", r.p("qat.routing.json")), "routing.log")
    qat_bytes = None if r.a.dry_run else os.path.getsize(qat)
    if not r.a.dry_run:
        r.save("nv-qat-24gb", {"bytes": qat_bytes, "mean_kld": 0.0, "note": "the reference itself (lane A)"})
    for gb in r.a.lane_a_gb:
        rung(r, f"A-{gb}gb", qat, imx, em, gb * 1e9, base, qat, requant=True)


def lane_b(r):
    """from the pre-QAT BF16: the full PTQ allocation and Mario's prune-then-quantize arm, equal bytes vs the QAT."""
    bf16_gguf = r.p("bolt-bf16.gguf")
    if not os.path.exists(bf16_gguf):
        r.sh([sys.executable, os.path.join(r.a.llama_src, "convert_hf_to_gguf.py"), r.a.bf16, "--outtype", "bf16",
              "--outfile", bf16_gguf], "convert.log", need_gb=160)
    host = r.p("bolt-q8_0.gguf")                       # forward-pass host: bf16 (150 GB) does not fit 128 GB
    if not os.path.exists(host):
        r.sh([r.q, bf16_gguf, host, "Q8_0"], "q8.log", need_gb=85)
    imx = r.p("bf16.imatrix.gguf")
    if not os.path.exists(imx):
        r.sh([r.imat, "-m", host, "-f", r.a.calib, "-o", imx, "-c", "512", "--chunks", str(r.a.imatrix_chunks)]
             + r.ngl(), "bf16.imatrix.log")
    base = r.p("q8.base.kld")
    if not os.path.exists(base):
        r.sh([r.ppl, "-m", host, "-f", r.a.eval, "--kl-divergence-base", base, "-c", "512", "--chunks",
              str(r.a.kld_chunks)] + r.ngl(), "q8.base.log", need_gb=5)
    # the official pack, judged by the SAME reference
    if not r.done("nv-qat-24gb@q8ref"):
        row = {"bytes": None if r.a.dry_run else os.path.getsize(r.a.qat_gguf)}
        if not r.a.dry_run:
            row.update(kld(r, r.a.qat_gguf, base, "nv-qat"))
        if not r.a.no_tool:
            row.update(toolgate(r, r.a.qat_gguf, host, "nv-qat"))
        r.save("nv-qat-24gb@q8ref", row)
    em = r.p("bf16.errmap.json")
    if not os.path.exists(em):
        r.sh(r.tool("pollard_moefit.py", "errmap", "--gguf", bf16_gguf, "--imatrix", imx, "--out", em,
                    "--llama-quantize", r.q, "--workdir", r.p("probes-bf16")), "bf16.errmap.log", need_gb=90)
    qat_b = QAT_GB * 1e9 if r.a.dry_run else os.path.getsize(r.a.qat_gguf)
    for gb in [qat_b / 1e9] + list(r.a.lane_b_gb):
        t = "eqQAT" if gb == qat_b / 1e9 else f"{gb}gb"
        rung(r, f"B-full-no1bit-{t}", bf16_gguf, imx, em, gb * 1e9, base, host, ("--no-1bit",))
        rung(r, f"B-full-1bit-{t}", bf16_gguf, imx, em, gb * 1e9, base, host)
    # Mario's arm: REAP (streamed) -> prune -> ~4-bit at equal bytes -> ~2-bit at about half
    reap = r.p("reap.json")
    if not os.path.exists(reap):
        r.sh(r.tool("pollard_reap.py", "--model", r.a.bf16, "--calib", r.a.calib, "--out", reap, "--stream",
                    "--device", "cuda", "--rows", "64", "--seqlen", "1024"), "reap.log")
    for drop in r.a.prunes:
        tag = f"p{int(round(100 * drop))}"
        pg = r.p(f"bolt-{tag}-bf16.gguf")
        if not os.path.exists(pg):
            r.sh(r.tool("pollard_prune.py", "--gguf", bf16_gguf, "--keep", str(round(1 - drop, 4)), "--score", "reap",
                        "--scores", reap, "--out", pg, "--keep-list", r.p(f"{tag}.keep.json")), f"{tag}.prune.log",
                 need_gb=160 * (1 - drop) + 5)
        pimx = r.p(f"{tag}.imatrix.gguf")
        phost = r.p(f"bolt-{tag}-q8_0.gguf")
        if not os.path.exists(phost):
            r.sh([r.q, pg, phost, "Q8_0"], f"{tag}.q8.log", need_gb=85 * (1 - drop) + 5)
        if not os.path.exists(pimx):
            r.sh([r.imat, "-m", phost, "-f", r.a.calib, "-o", pimx, "-c", "512", "--chunks", str(r.a.imatrix_chunks)]
                 + r.ngl(), f"{tag}.imatrix.log")
        pem = r.p(f"{tag}.errmap.json")
        if not os.path.exists(pem):
            r.sh(r.tool("pollard_moefit.py", "errmap", "--gguf", pg, "--imatrix", pimx, "--out", pem,
                        "--llama-quantize", r.q, "--workdir", r.p(f"probes-{tag}")), f"{tag}.errmap.log", need_gb=60)
        for gb in [qat_b / 1e9, qat_b / 2e9] + list(r.a.lane_b_gb):
            t = "eqQAT" if gb == qat_b / 1e9 else ("halfQAT" if gb == qat_b / 2e9 else f"{gb}gb")
            rung(r, f"B-{tag}-no1bit-{t}", pg, pimx, pem, gb * 1e9, base, host, ("--no-1bit",))
            rung(r, f"B-{tag}-1bit-{t}", pg, pimx, pem, gb * 1e9, base, host)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--host", required=True, choices=sorted(HOSTS))
    ap.add_argument("--work", required=True)
    ap.add_argument("--qat-gguf", required=True, help="NVIDIA's Nemotron-3.5-Bolt-24GB-GGUF model file")
    ap.add_argument("--bf16", help="the BF16 (pre-QAT) HF checkpoint (lane B)")
    ap.add_argument("--llama-src", help="llama.cpp checkout (>= cd0a5b6); its build/bin is used")
    ap.add_argument("--llama-bin", help="llama.cpp binaries dir (overrides --llama-src/build/bin)")
    ap.add_argument("--calib", required=True, help="pollard-calib corpus (enable_thinking traces for Bolt)")
    ap.add_argument("--eval", required=True, help="held-out eval (pollard-calib --held-out)")
    ap.add_argument("--imatrix-chunks", type=int, default=256)
    ap.add_argument("--kld-chunks", type=int, default=32)
    ap.add_argument("--lane-a-gb", type=lambda s: [float(x) for x in s.split(",")], default=[21.0, 20.0, 19.0, 18.0])
    ap.add_argument("--lane-b-gb", type=lambda s: [float(x) for x in s.split(",")], default=[18.0, 15.0])
    ap.add_argument("--prunes", type=lambda s: [float(x) for x in s.split(",")], default=[0.5, 0.25])
    ap.add_argument("--lanes", help="override the host's lanes (A,B)")
    ap.add_argument("--no-tool", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    r = Run(a)
    lanes = tuple(a.lanes.split(",")) if a.lanes else r.h["lanes"]
    if "B" in lanes and (a.host != "spark" or not a.bf16 or not a.llama_src):
        sys.exit("lane B runs on the Spark with --bf16 and --llama-src (convert_hf_to_gguf.py)")
    check_llama(r)
    if "A" in lanes:
        lane_a(r)
    if "B" in lanes:
        lane_b(r)
    if os.path.exists(r.res_path):
        res = json.load(open(r.res_path))
        print("\n| rung | GB | mean KLD | p99 KLD | top-1 % | PPL | tool retention |\n|---|---|---|---|---|---|---|")
        for k, v in sorted(res.items(), key=lambda kv: kv[1].get("bytes") or 0):
            gb = (v.get("bytes") or 0) / 1e9
            print(f"| {k} | {gb:.2f} | {v.get('mean_kld')} | {v.get('p99_kld')} | {v.get('top1')} | {v.get('ppl')} | "
                  f"{v.get('tool_retention')} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
