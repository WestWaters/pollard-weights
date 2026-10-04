#!/usr/bin/env python3
"""pollard-kvbench: what does compressing the KV cache cost, at real context lengths?

llama.cpp can store keys and values in different types (`-ctk` / `-ctv`), and keeping K in f16 while V
goes to q8_0 is the idea ANEMLL's "V8" cache ships on the Neural Engine (keys are the sensitive half).
Nobody has published the curve that decides it: KLD against an f16 cache AND speed, per format, from
8K to 64K. This measures it, the Pollard way -- KLD is the product metric.

  pollard-kvbench --model Qwen3.8-27B-Pollard-IQ2_XXS.gguf --text wiki.test.raw
  pollard-kvbench --model X.gguf --ctx 8192,32768 --configs f16/f16,f16/q8_0,q8_0/q8_0
  pollard-kvbench --model X.gguf --dry-run          # print the plan and the commands only

Per (context, format):
  * quality -- llama-perplexity at that context with an f16/f16 cache saves base logits; each format is
    then scored with --kl-divergence: mean / median KLD and top-1 agreement vs the f16 cache.
  * speed   -- llama-bench at that depth: prompt (prefill) and generation tok/s.
  * memory  -- peak VRAM sampled from nvidia-smi while it runs.

Guards, because the GPU on the build box is shared:
  * refuses to start while another program holds the GPU (games, browsers with GPU tabs...) unless
    --force -- run it when the machine is yours;
  * refuses a model + cache that would not fit VRAM (an -ngl 99 overrun has crashed this box);
  * warns when the build lacks GGML_CUDA_FA_ALL_QUANTS: mixed K/V types then fall back to CPU
    attention SILENTLY, and every mixed-format speed number would be measuring the wrong thing.
"""
import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pollard_calc import find_llama_bin  # noqa: E402

DEFAULT_CTX = [8192, 16384, 32768, 65536]
DEFAULT_CONFIGS = ["f16/f16", "f16/q8_0", "q8_0/q8_0", "f16/q4_0", "q8_0/q4_0", "q4_0/q4_0"]
BYTES = {"f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}
SHARED_OK = ("dwm.exe", "explorer.exe", "ShellHost", "SearchHost", "StartMenuExperienceHost", "TextInputHost",
             "ShellExperienceHost", "CrossDeviceResume", "SystemSettings", "ApplicationFrameHost", "ctfmon",
             "LockApp", "Widgets", "WindowsTerminal", "NVIDIA Overlay", "nvcontainer", "msedgewebview2", "DCv2")


def nvsmi(query):
    try:
        r = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return [l.strip() for l in r.stdout.splitlines() if l.strip()]
    except Exception:                                                      # noqa: BLE001
        return []


def gpu_busy():
    """Programs that look like someone using the GPU (not the desktop shell)."""
    apps = nvsmi("compute-apps=process_name,used_memory") + nvsmi("gpu=utilization.gpu")
    procs = [a.split(",")[0].strip() for a in apps[:-1]] if apps else []
    users = [p for p in procs if p and not any(ok.lower() in p.lower() for ok in SHARED_OK)]
    util = int(apps[-1]) if apps and apps[-1].isdigit() else 0
    return users, util


def vram_total_gb():
    v = nvsmi("gpu=memory.total")
    return int(v[0]) / 1024 if v and v[0].isdigit() else None


class VramPeak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, False

    def run(self):
        while not self.stop:
            v = nvsmi("gpu=memory.used")
            if v and v[0].isdigit():
                self.peak = max(self.peak, int(v[0]))
            time.sleep(0.5)


def gguf_meta(path):
    """Layer count, KV heads and head dims from the GGUF header (for the fit check)."""
    from pollard_calc import _read_one_gguf
    kv = _read_one_gguf(path)[0]
    arch = kv.get("general.architecture", "")
    g = lambda k, d=None: kv.get(f"{arch}.{k}", d)                         # noqa: E731
    heads_kv = g("attention.head_count_kv", g("attention.head_count", 8))
    if isinstance(heads_kv, list):                                        # per-layer arrays on hybrids
        attn_layers = sum(1 for h in heads_kv if h)
        heads_kv = max(heads_kv) or 8
    else:
        attn_layers = None
    n_layer = g("block_count", 32)
    full_every = g("full_attention_interval")                             # qwen3.5/3.8 hybrids
    if attn_layers is None:
        attn_layers = n_layer // full_every if full_every else n_layer
    kd = g("attention.key_length", 128)
    vd = g("attention.value_length", kd)
    return {"arch": arch, "attn_layers": attn_layers, "heads_kv": heads_kv, "k_dim": kd, "v_dim": vd}


def kv_gb(meta, ctx, kt, vt):
    per_tok = meta["attn_layers"] * meta["heads_kv"] * (meta["k_dim"] * BYTES.get(kt, 2) + meta["v_dim"] * BYTES.get(vt, 2))
    return per_tok * ctx / 1024 ** 3


def cmake_flag(bin_path, flag):
    d = os.path.dirname(os.path.abspath(bin_path))
    for _ in range(4):
        c = os.path.join(d, "CMakeCache.txt")
        if os.path.isfile(c):
            m = re.search(rf"^{flag}:\w+=(.*)$", open(c, encoding="utf-8", errors="replace").read(), re.M)
            return m.group(1) if m else None
        d = os.path.dirname(d)
    return None


def ppl_kld(ppl, model, text, ctx, kt, vt, ngl, base, save):
    cmd = [ppl, "-m", model, "-f", text, "-c", str(ctx), "--chunks", "1", "-ngl", str(ngl), "-fa", "on",
           "-ctk", kt, "-ctv", vt, "-b", "2048", "-ub", "512"]
    if save:
        cmd += ["--kl-divergence-base", base]
    else:
        cmd += ["--kl-divergence", "--kl-divergence-base", base]
    th = os.environ.get("POLLARD_THREADS")
    if th:
        cmd += ["-t", th]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr

    def g(pat):
        m = re.search(pat, out, re.M)
        return float(m.group(1)) if m else None
    return {"ok": r.returncode == 0, "mean_kld": g(r"^Mean\s+KLD:\s*([0-9.]+)"), "median_kld": g(r"^Median\s+KLD:\s*([0-9.]+)"),
            "top1": g(r"^Same top p:\s*([0-9.]+)"), "ppl": g(r"^Mean PPL\(Q\)\s*:\s*([0-9.]+)") or g(r"Final estimate:\s*PPL[^=\n]*=\s*([0-9.]+)"),
            "cpu_attn": bool(re.search(r"(fall(ing)? back|not supported).{0,40}(flash|FA|attention)", out, re.I)),
            "tail": "" if r.returncode == 0 else out[-600:]}


def bench_speed(bench, model, ctx, kt, vt, ngl):
    cmd = [bench, "-m", model, "-fa", "1", "-ctk", kt, "-ctv", vt, "-d", str(ctx), "-p", "512", "-n", "64",
           "-ngl", str(ngl), "-r", "2", "-o", "json"]
    th = os.environ.get("POLLARD_THREADS")
    if th:
        cmd += ["-t", th]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        rows = json.loads(r.stdout[r.stdout.index("["):])
    except (ValueError, json.JSONDecodeError):
        return {"ok": False, "tail": (r.stdout + r.stderr)[-600:]}
    pp = next((x["avg_ts"] for x in rows if x.get("n_prompt") and not x.get("n_gen")), None)
    tg = next((x["avg_ts"] for x in rows if x.get("n_gen") and not x.get("n_prompt")), None)
    return {"ok": True, "prefill_tps": pp, "decode_tps": tg}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="GGUF to test (its weights stay fixed; only the KV cache changes)")
    ap.add_argument("--text", default="wikitext2_test.txt", help="long plain text; must be longer than the largest context")
    ap.add_argument("--ctx", default=",".join(map(str, DEFAULT_CTX)))
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS), help="comma list of K/V type pairs")
    ap.add_argument("--ngl", default="99", help="GPU layers (99 = all; the fit check guards it)")
    ap.add_argument("--no-speed", action="store_true")
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--out", help="results JSON (default kvbench-<model>-<date>.json)")
    ap.add_argument("--workdir", default=tempfile.gettempdir(), help="where base logits go (several GB at 32K)")
    ap.add_argument("--force", action="store_true", help="run even though another program holds the GPU")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    ctxs = [int(x) for x in a.ctx.split(",") if x]
    cfgs = [tuple(c.split("/")) for c in a.configs.split(",") if c]
    meta = gguf_meta(a.model)
    model_gb = os.path.getsize(a.model) / 1024 ** 3
    ppl, bench = find_llama_bin("llama-perplexity", arch=meta["arch"]), find_llama_bin("llama-bench", arch=meta["arch"])
    print(f"== pollard-kvbench :: {os.path.basename(a.model)}  ({meta['arch']}, {model_gb:.1f} GB, "
          f"{meta['attn_layers']} attention layers x {meta['heads_kv']} KV heads)")
    if not ppl or not bench:
        sys.exit("no llama-perplexity / llama-bench that loads this architecture -- pollard-runtime --update")

    fa_all = cmake_flag(ppl, "GGML_CUDA_FA_ALL_QUANTS")
    mixed = any(k != v for k, v in cfgs)
    if mixed and fa_all not in ("ON", "1", "TRUE"):
        print("  WARNING  this build lacks GGML_CUDA_FA_ALL_QUANTS: mixed K/V formats fall back to CPU attention\n"
              "           silently. Quality numbers stay valid; mixed-format SPEED would be wrong. Rebuild with:\n"
              "           pollard-runtime --update --with GGML_CUDA_FA_ALL_QUANTS=ON")

    total = vram_total_gb()
    plan = []
    for c in ctxs:
        for kt, vt in cfgs:
            need = model_gb + kv_gb(meta, c, kt, vt) + 1.2                 # + compute buffers
            plan.append((c, kt, vt, need))
    print(f"  plan: {len(ctxs)} contexts x {len(cfgs)} formats; VRAM {total or '?'} GB")
    for c, kt, vt, need in plan:
        flag = "" if not total or a.ngl != "99" or need < total * 0.94 else "  <-- would not fit, skipped"
        print(f"    {c:>6} ctx  K={kt:<5} V={vt:<5}  ~{need:5.1f} GB{flag}")
    if a.dry_run:
        print(f"\n  quality: {ppl} -m <model> -f <text> -c <ctx> --chunks 1 -fa on -ctk K -ctv V --kl-divergence ...")
        print(f"  speed  : {bench} -m <model> -fa 1 -ctk K -ctv V -d <ctx> -p 512 -n 64 -o json")
        return 0

    users, util = gpu_busy()
    if users and not a.force:
        sys.exit(f"GPU in use by {users} (util {util}%) -- not starting. Run when the machine is free, or --force.")

    res = {"model": os.path.basename(a.model), "arch": meta["arch"], "model_gb": round(model_gb, 2), "fa_all_quants": fa_all,
           "when": _dt.datetime.now().isoformat(timespec="seconds"), "rows": []}
    out = a.out or f"kvbench-{os.path.splitext(os.path.basename(a.model))[0]}-{_dt.date.today():%Y%m%d}.json"
    for c in ctxs:
        base = os.path.join(a.workdir, f"kvbase-{c}.kld")
        have_base = False
        for kt, vt, need in [(k, v, n) for cc, k, v, n in plan if cc == c]:
            if total and a.ngl == "99" and need >= total * 0.94:
                res["rows"].append({"ctx": c, "k": kt, "v": vt, "skipped": "would not fit VRAM"})
                continue
            row = {"ctx": c, "k": kt, "v": vt, "kv_gb": round(kv_gb(meta, c, kt, vt), 2)}
            mon = VramPeak(); mon.start()
            if not a.no_quality:
                if not have_base:
                    print(f"  [{c}] base logits with an f16/f16 cache ...", flush=True)
                    b = ppl_kld(ppl, a.model, a.text, c, "f16", "f16", a.ngl, base, save=True)
                    if not b["ok"]:
                        print(f"  [{c}] base failed: {b['tail']}")
                        break
                    have_base = True
                if (kt, vt) != ("f16", "f16"):
                    q = ppl_kld(ppl, a.model, a.text, c, kt, vt, a.ngl, base, save=False)
                    row.update({k: q[k] for k in ("mean_kld", "median_kld", "top1", "ppl", "cpu_attn")})
                else:
                    row.update({"mean_kld": 0.0, "top1": 100.0})
            if not a.no_speed:
                row.update(bench_speed(bench, a.model, c, kt, vt, a.ngl))
            mon.stop = True; mon.join(1)
            row["vram_peak_gb"] = round(mon.peak / 1024, 2) if mon.peak else None
            res["rows"].append(row)
            print(f"  [{c}] K={kt:<5} V={vt:<5} KLD={row.get('mean_kld')}  top1={row.get('top1')}%  "
                  f"prefill={row.get('prefill_tps')}  decode={row.get('decode_tps')}  peak={row.get('vram_peak_gb')} GB", flush=True)
            json.dump(res, open(out, "w"), indent=1)
        try:
            os.remove(base)
        except OSError:
            pass
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
