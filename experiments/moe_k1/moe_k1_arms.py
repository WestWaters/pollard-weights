#!/usr/bin/env python3
"""moe_k1_arms.py -- the small-MoE proof of the per-expert / 1-bit allocation and of the prune-then-quantize arm,
GGUF lane, at EQUAL BYTES. Run on one machine, one heavy job at a time (run_mac.sh takes the shared lock).

Arms (every rung judged the same way: llama-perplexity --kl-divergence vs the reference of the FULL model -- f16, or
Q8_0 with --src-type q8_0 when the disk cannot hold an f16 -- on a held-out eval disjoint from the calibration and
from the sensitivity eval, plus pollard-toolcall retention vs that reference):

  full-2bit    all experts, pollard-moefit knapsack, experts floored at iq2_xxs     (a) "measured 2-bit"
  full-1bit    all experts, knapsack with iq1_s / iq1_m on the table              (b) "1-bit-heavy mix"
  p50-4bit     REAP-prune 50% of routed experts, then the knapsack at the SAME bytes (Mario's arm, stage 1: the
               remaining experts land ~4-bit)
  p50-2bit     the same pruned model at ~2-bit (Mario's arm, stage 2: about half the bytes)
  p25-1bit     REAP-prune 25% + 1-bit-heavy knapsack (the combined arm); p50-1bit likewise
  uniform-*    stock references at nearby sizes: experts uniform at one type, the rest at the floors

Budgets: B_hi = bytes of a full-expert ~2.25 bpw-experts plan; B_mid = full-expert ~1.75 bpw-experts (the lowest a
full-expert mix reaches above iq1_s with room to choose); B_lo = the p50 ~2.2-bpw-experts plan's bytes (below what any
full-expert build can reach: iq1_s everywhere is the floor).

    python3 moe_k1_arms.py --work ~/pollard_k1 --hf ~/pollard_k1/hf --step all
    python3 moe_k1_arms.py --work ~/pollard_k1 --step table          # the results table from results.json

Steps are idempotent (an output that exists is reused) and each checks the disk floor before writing."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

DISK_FLOOR_GB = 10.0
RAM_FLOOR_GB = 2.0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def free_disk_gb(path):
    return shutil.disk_usage(path).free / 1e9


def free_ram_gb():
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = int(re.search(r"page size of (\d+)", out).group(1))
        pages = sum(int(re.search(rf"{k}:\s+(\d+)", out).group(1)) for k in ("Pages free", "Pages inactive",
                                                                             "Pages speculative"))
        return pages * page / 1e9
    except Exception:
        try:
            import pollard_calc
            return pollard_calc.detect_available_ram_gb() or 99.0
        except Exception:
            return 99.0


def floors(work, need_gb=0.0):
    d, r = free_disk_gb(work), free_ram_gb()
    if d - need_gb < DISK_FLOOR_GB:
        raise SystemExit(f"STOP: {d:.1f} GB free on disk, this step needs ~{need_gb:.1f} GB, floor {DISK_FLOOR_GB} GB")
    if r < RAM_FLOOR_GB:
        raise SystemExit(f"STOP: {r:.1f} GB RAM available, floor {RAM_FLOOR_GB} GB")


def run(cmd, logf=None, env=None):
    log("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env)
    if logf:
        with open(logf, "a") as f:
            f.write(r.stdout + r.stderr)
    if r.returncode:
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(map(str, cmd))}\n{tail}")
    return r


class Ctx:
    def __init__(self, a):
        self.a = a
        self.w = os.path.abspath(os.path.expanduser(a.work))
        self.bin = os.path.expanduser(a.bin)
        self.q = os.path.join(self.bin, "llama-quantize")
        self.imat = os.path.join(self.bin, "llama-imatrix")
        self.ppl = os.path.join(self.bin, "llama-perplexity")
        self.py = a.python
        self.torch_py = a.torch_python
        self.res_path = os.path.join(self.w, "results.json")

    def p(self, *x):
        return os.path.join(self.w, *x)

    def src(self):
        return self.p(f"full-{self.a.src_type}.gguf")

    def quantized(self):
        return self.a.src_type not in ("f16", "bf16", "f32")

    def requant(self):
        return ["--allow-requantize"] if self.quantized() else []

    def results(self):
        return json.load(open(self.res_path)) if os.path.exists(self.res_path) else {}

    def save(self, key, row):
        r = self.results()
        r[key] = {**r.get(key, {}), **row}
        json.dump(r, open(self.res_path, "w"), indent=1)

    def tool(self, name, *args):
        return [self.py, os.path.join(ROOT, "tools", name)] + list(args)


# ------------------------------------------------------------------------------------------------ corpora

def step_corpora(c):
    """sens_eval (allocation's held-out) and kld_eval (the judge), disjoint slices of pollard-calib's held-out file."""
    if os.path.exists(c.p("kld_eval.txt")) and os.path.exists(c.p("sens_eval.txt")):
        return
    held = open(c.p("held.txt"), encoding="utf-8").read().split("\n")
    half = len(held) // 2
    open(c.p("sens_eval.txt"), "w").write("\n".join(held[:half])[:16000])
    open(c.p("kld_eval.txt"), "w").write("\n".join(held[half:])[:60000])
    log("corpora: sens_eval.txt (first half of held-out), kld_eval.txt (second half)")


# ------------------------------------------------------------------------------------------------ model prep

def step_convert(c):
    out = c.src()
    if os.path.exists(out):
        return out
    floors(c.w, 1.5 if c.quantized() else 3.0)
    env = {**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(c.bin), "..", "gguf-py")}
    conv = os.path.join(os.path.dirname(os.path.dirname(c.bin)), "convert_hf_to_gguf.py")
    run([c.torch_py, conv, c.a.hf, "--outtype", c.a.src_type, "--outfile", out], c.p("convert.log"), env)
    return out


def step_reap(c):
    out = c.p("reap.json")
    if not os.path.exists(out):
        if not c.a.hf or not os.path.exists(os.path.join(c.a.hf, "config.json")):
            raise SystemExit("REAP needs the HF checkpoint (--hf)")
        floors(c.w)
        run([c.torch_py, os.path.join(ROOT, "tools", "pollard_reap.py"), "--model", c.a.hf, "--calib", c.p("calib.txt"),
             "--rows", str(c.a.reap_rows), "--seqlen", "512", "--out", out], c.p("reap.log"))
    return out


def step_imatrix(c, model, tag):
    out = c.p(f"{tag}.imatrix.gguf")
    if not os.path.exists(out):
        floors(c.w)
        run([c.imat, "-m", model, "-f", c.p("calib.txt"), "-o", out, "-ngl", str(c.a.ngl), "--chunks",
             str(c.a.imatrix_chunks), "-c", "512"], c.p(f"{tag}.imatrix.log"))
    return out


def step_prune(c, keep, reap):
    tag = f"p{int(round(100 * (1 - keep)))}"
    out = c.p(f"{tag}-{c.a.src_type}.gguf")
    if not os.path.exists(out):
        floors(c.w, 1.5)
        run(c.tool("pollard_prune.py", "--gguf", c.src(), "--keep", str(keep), "--score", "reap",
                   "--scores", reap, "--out", out, "--keep-list", c.p(f"{tag}.keep.json")), c.p(f"{tag}.prune.log"))
    return tag, out


def step_sensitivity(c, model, imatrix, tag):
    out = c.p(f"{tag}.sens.json")
    if not os.path.exists(out):
        floors(c.w, 2.0)
        pt = ["--passthrough", "--groups", "ffn,attn", "--probe", "iq2_xxs"] if c.quantized() else []
        run(c.tool("pollard_sensitivity.py", "--gguf", model, "--imatrix", imatrix, "--eval", c.p("sens_eval.txt"),
                   "--out", out, "--llama-quantize", c.q, "--llama-perplexity", c.ppl, "--ngl", str(c.a.ngl), *pt),
            c.p(f"{tag}.sens.log"))
    return out


def step_errmap(c, model, imatrix, tag):
    out = c.p(f"{tag}.errmap.json")
    if not os.path.exists(out):
        floors(c.w, 1.5)                       # one probe on disk at a time (<= the q8_0 probe)
        run(c.tool("pollard_moefit.py", "errmap", "--gguf", model, "--imatrix", imatrix, "--out", out,
                   "--llama-quantize", c.q, "--workdir", c.p(f"probes-{tag}"), *c.requant()), c.p(f"{tag}.errmap.log"))
        shutil.rmtree(c.p(f"probes-{tag}"), ignore_errors=True)
    return out


def step_base(c):
    out = c.p("base.kld")
    if not os.path.exists(out):
        floors(c.w, 2.0)
        run([c.ppl, "-m", c.src(), "-f", c.p("kld_eval.txt"), "--kl-divergence-base", out,
             "-ngl", str(c.a.ngl), "-c", "512", "--chunks", str(c.a.kld_chunks)], c.p("base.log"))
    return out


# ------------------------------------------------------------------------------------------------ rungs

def kld(c, model):
    r = subprocess.run([c.ppl, "-m", model, "-f", c.p("kld_eval.txt"), "--kl-divergence", "--kl-divergence-base",
                        c.p("base.kld"), "-ngl", str(c.a.ngl), "-c", "512", "--chunks", str(c.a.kld_chunks)],
                       capture_output=True, text=True)
    out = r.stdout + r.stderr

    def g(pat):
        m = re.search(pat, out, re.M)
        return float(m.group(1)) if m else None
    return {"ppl": g(r"^Mean PPL\(Q\)\s*:\s*([0-9.]+)"), "ref_ppl": g(r"^Mean PPL\(base\)\s*:\s*([0-9.]+)"),
            "mean_kld": g(r"^Mean\s+KLD:\s*([0-9.]+)"), "median_kld": g(r"^Median\s+KLD:\s*([0-9.]+)"),
            "p99_kld": g(r"^99\.0%\s+KLD:\s*([0-9.]+)"), "top1": g(r"^Same top p:\s*([0-9.]+)")}


def toolgate(c, model, tag):
    out = c.p(f"{tag}.toolcall.json")
    if not os.path.exists(out):
        r = subprocess.run(c.tool("pollard_toolcall.py", "--gguf", model, "--ref", c.src(), "--ngl",
                                  str(c.a.ngl), "--out", out), capture_output=True, text=True)
        with open(c.p(f"{tag}.toolcall.log"), "w") as f:
            f.write(r.stdout + r.stderr)
    try:
        d = json.load(open(out))
    except Exception:
        return {"tool_status": "not-measured"}
    sc, rs = d.get("score") or {}, d.get("ref_score") or {}
    ret = d.get("retention")
    return {"tool_status": "measured", "tool_rate": sc.get("rate"), "tool_ref_rate": rs.get("rate"),
            "tool_retention": round(100 * ret, 1) if ret is not None else None,
            "tool_n": sc.get("n")}


def rung(c, tag, src_model, imatrix, plan_args, label, arm, prune=None):
    res = c.results()
    if tag in res and res[tag].get("mean_kld") is not None and (c.a.no_tool or res[tag].get("tool_status")):
        return res[tag]
    plan_path = c.p(f"{tag}.plan.json")
    run(c.tool("pollard_moefit.py", "plan", *plan_args, "--embd", "q8_0", "--out", plan_path))
    p = json.load(open(plan_path))
    out = c.p(f"{tag}.gguf")
    floors(c.w, p["bytes"] / 1e9 + 0.2)
    run(c.tool("pollard_moefit.py", "build", "--gguf", src_model, "--imatrix", imatrix, "--plan", plan_path,
               "--out", out, "--llama-quantize", c.q, *c.requant()), c.p(f"{tag}.build.log"))
    size = os.path.getsize(out)
    row = {"label": label, "arm": arm, "prune": prune, "bytes": size, "gb": round(size / 1e9, 4),
           "planned_bytes": p["bytes"], "bpw_role": p["bpw_role"], "hist": p["hist"],
           "predicted": p["predicted_kld"], **kld(c, out)}
    c.save(tag, row)
    log(f"  {tag}: {row['gb']} GB  KLD {row['mean_kld']}  top-1 {row['top1']}  PPL {row['ppl']}")
    if not c.a.no_tool:
        c.save(tag, toolgate(c, out, tag))
    if not c.a.keep_rungs:
        os.remove(out)
    return c.results()[tag]


def plan_bytes(c, errmap, *args):
    tmp = c.p("_probe_plan.json")
    run(c.tool("pollard_moefit.py", "plan", "--errmap", errmap, *args, "--embd", "q8_0", "--out", tmp))
    b = json.load(open(tmp))["bytes"]
    os.remove(tmp)
    return b


def expert_bytes_budget(em_path, expert_bpw, other_type="iq4_xs", embd="q8_0"):
    """the file bytes of a plan whose experts average `expert_bpw` and everything else sits at its floor."""
    import pollard_moefit as M
    em = json.load(open(em_path))
    b = M.fixed_bytes(em, embd=embd)
    for t in em["tensors"].values():
        b += t["numel"] * expert_bpw / 8 if t["role"] == M.ROLE_EXPERT else M.type_bytes(other_type, t["numel"])
    return int(b)


def step_arms(c):
    f16 = c.src()
    reap = step_reap(c)
    imx = step_imatrix(c, f16, "full")
    sens = step_sensitivity(c, f16, imx, "full") if not c.a.no_sens else None
    em = step_errmap(c, f16, imx, "full")
    step_base(c)
    pr = {}
    for drop in [float(x) for x in c.a.prunes.split(",") if x.strip()]:
        tag, model = step_prune(c, round(1.0 - drop, 4), reap)
        pimx = step_imatrix(c, model, tag)
        psens = step_sensitivity(c, model, pimx, tag) if not c.a.no_sens else None
        pr[tag] = (model, pimx, psens, step_errmap(c, model, pimx, tag))
    S = (lambda s: ["--sensitivity", s] if s else [])
    B_hi = expert_bytes_budget(em, 2.25)
    B_mid = expert_bytes_budget(em, 1.75)
    B_lo = expert_bytes_budget(pr["p50"][3], 2.25) if "p50" in pr else expert_bytes_budget(em, 1.5625)
    c.save("_budgets", {"B_hi": B_hi, "B_mid": B_mid, "B_lo": B_lo})
    log(f"budgets: B_hi {B_hi / 1e9:.4f} GB  B_mid {B_mid / 1e9:.4f} GB  B_lo {B_lo / 1e9:.4f} GB")
    # references: stock-style uniform experts
    for q in ("iq2_xxs", "iq2_s", "iq1_m", "iq3_xxs"):
        rung(c, f"uniform-{q}", f16, imx, ["--errmap", em, "--uniform", q], f"full, experts uniform {q}", "ref")
    for B, bt in ((B_hi, "hi"), (B_mid, "mid"), (B_lo, "lo")):
        arms = [("full", f16, imx, sens, em, None)] + [(t, m, pi, ps, pe, t) for t, (m, pi, ps, pe) in pr.items()]
        for name, model, im, sn, e, prune in arms:
            if B >= expert_bytes_budget(e, 2.0625):          # every expert can sit at iq2_xxs or above
                rung(c, f"{name}-no1bit-{bt}", model, im, ["--errmap", e, "--bytes", str(B), "--no-1bit", *S(sn)],
                     f"{name}, measured knapsack, experts >= iq2_xxs @{bt}", "a" if prune is None else "c", prune)
            if B >= expert_bytes_budget(e, 1.5625):          # iq1_s is the floor
                rung(c, f"{name}-1bit-{bt}", model, im, ["--errmap", e, "--bytes", str(B), *S(sn)],
                     f"{name}, measured knapsack, 1-bit experts allowed @{bt}", "b" if prune is None else "d", prune)


# ------------------------------------------------------------------------------------------------ table

_CALL = re.compile(r'^\s*(?:<\|?tool_call\|?>)?\s*\[?\s*\{\s*"arguments"\s*:\s*(\{.*?\})\s*,\s*"name"\s*:\s*"([^"]+)"'
                   r'|^\s*(?:<\|?tool_call\|?>)?\s*\[?\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})')


def _call(sample):
    """(name, args text) of a call written the way the reference writes it, else None. A reply that echoes the tool
    SCHEMA ({"name": ..., "description": ...}) is not a call."""
    m = _CALL.search(sample or "")
    if not m:
        return None
    return (m.group(2), m.group(1)) if m.group(2) else (m.group(3), m.group(4))


def tool_agreement(c, tag):
    """(%, n): of the probes where the REFERENCE writes a call, the share where the rung writes a call to the same
    function with the same arguments. pollard-toolcall's served path can be blind here (llama-server's parser does
    not read granite-3.1-1b's `<tool_call>[{...}]`, so parse rate is 0% for reference and rung alike and retention
    only sees abstention); this reads the per-probe samples (first 160 chars) directly."""
    try:
        d = json.load(open(c.p(f"{tag}.toolcall.json")))
    except Exception:
        return None
    ref = {r["id"]: _call(r.get("sample")) for r in (d.get("ref_rows") or [])}
    pairs = [(ref[r["id"]], _call(r.get("sample"))) for r in d.get("rows") or [] if ref.get(r["id"])]
    if not pairs:
        return None
    same = sum(1 for a, b in pairs if b is not None and a[0] == b[0] and a[1] == b[1])
    return f"{100.0 * same / len(pairs):.0f}% of {len(pairs)}"


def step_table(c):
    res = c.results()
    rows = [(k, v) for k, v in res.items() if not k.startswith("_") and v.get("mean_kld") is not None]
    rows.sort(key=lambda kv: (kv[1]["bytes"], kv[1]["mean_kld"]))
    base_ppl = next((v.get("ref_ppl") for _, v in rows if v.get("ref_ppl")), None)
    lines = [f"{c.a.src_type} reference PPL {base_ppl}", "",
             "| rung | arm | GB | expert bpw | mean KLD | p99 KLD | top-1 % | PPL | tool retention | same call as ref "
             "| expert types |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in rows:
        eb = (v.get("bpw_role") or {}).get("expert")
        ht = v.get("hist", {}).get("expert", {})
        hs = " ".join(f"{q}:{n}" for q, n in sorted(ht.items()))
        tr = v.get("tool_retention")
        lines.append(f"| {k} | {v.get('arm')} | {v['gb']:.3f} | {eb if eb is not None else '-'} | "
                     f"{v['mean_kld']:.4f} | {v.get('p99_kld') or '-'} | {v.get('top1')} | {v.get('ppl')} | "
                     f"{tr if tr is not None else v.get('tool_status', '-')} | {tool_agreement(c, k)} | {hs} |")
    txt = "\n".join(lines)
    print(txt)
    open(c.p("results.md"), "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--hf", help="HF checkpoint dir (REAP + convert)")
    ap.add_argument("--bin", default="~/Desktop/Pollard-Weights/runtime/llama.cpp/build/bin")
    ap.add_argument("--python", default=sys.executable, help="python for the Pollard tools (numpy + gguf)")
    ap.add_argument("--torch-python", default=sys.executable, help="python with torch + transformers (REAP, convert)")
    ap.add_argument("--ngl", default="99")
    ap.add_argument("--src-type", default="f16", choices=("f16", "bf16", "q8_0"),
                    help="the reference / source GGUF: f16, or q8_0 when the disk cannot hold the f16 (the rungs are "
                         "then requantized from it and every KLD is vs the Q8_0)")
    ap.add_argument("--imatrix-chunks", type=int, default=128)
    ap.add_argument("--kld-chunks", type=int, default=16)
    ap.add_argument("--prunes", default="0.5,0.25", help="fractions of routed experts to REMOVE (REAP)")
    ap.add_argument("--free-hf", action="store_true", help="delete the HF safetensors after REAP + convert (disk)")
    ap.add_argument("--reap-rows", type=int, default=64)
    ap.add_argument("--no-sens", action="store_true", help="skip pollard-sensitivity (relative-error allocation)")
    ap.add_argument("--no-tool", action="store_true")
    ap.add_argument("--keep-rungs", action="store_true")
    ap.add_argument("--step", default="all", choices=("all", "corpora", "convert", "arms", "table"))
    a = ap.parse_args()
    c = Ctx(a)
    if a.step in ("all", "corpora"):
        step_corpora(c)
    if a.step in ("all", "convert"):
        if a.hf:
            step_reap(c)
        step_convert(c)
        if a.free_hf and a.hf:
            for f in os.listdir(a.hf):
                if f.endswith(".safetensors"):
                    os.remove(os.path.join(a.hf, f))
                    log(f"freed {f} (re-download to rebuild)")
    if a.step in ("all", "arms"):
        step_arms(c)
    if a.step in ("all", "table", "arms"):
        step_table(c)


if __name__ == "__main__":
    main()
