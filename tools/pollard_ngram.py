#!/usr/bin/env python3
"""pollard-ngram -- keep a model's n-gram / per-layer embedding TABLE on the SSD, at full quality.

Some models carry a huge LOOKUP table next to the transformer: Qwen3.8-Flash-Next's 51B n-gram PLE table,
Gemma 4's per-layer embeddings, Engram / LongCat n-gram embeddings. The table does no arithmetic: each token
hashes its last few tokens and reads a handful of rows (a few KB). llama.cpp can read those rows straight from
the file on demand (`--lazy-mode`, needs mmap), so the table never has to sit in RAM or VRAM.

That changes how the model should be BUILT. A RAM-fit quant that counts the table as RAM spends the budget on
it and crushes everything else to make room. With the table on disk:
  * the table stays high precision (q8_0 by default -- disk is cheap), and
  * the transformer gets the whole RAM budget.
`pollard-fit` does this automatically (`--disk auto`); this tool is the rest of the workflow:

    pollard-ngram inspect model.gguf                 # which tensors are tables, how big, can this runtime lazy-read them
    pollard-ngram run model.gguf --ram 64 --vram 12  # the llama-server command that keeps the table on the SSD
    pollard-ngram watch [--pid N]                    # live disk writes / swap while it runs: are the writes swap?

About SSD wear. Reading the table never writes to the SSD. If the drive is being WRITTEN while the model runs,
that is the OS swapping (or compressing) memory because something else did not fit: the table was loaded with
--no-mmap / --load-mode none|mlock (so it was copied into RAM), or the rest of the model plus the KV cache is
bigger than RAM. `watch` shows which. Reads well under the drive's rated speed are normal: lookups are small
random reads, so the SSD's latency matters, not its sequential MB/s.
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, time

# Tensor names that are lookup tables (rows fetched by index, never multiplied). First match wins; the
# per_layer_token_embd name is the one llama.cpp uses for Gemma 4 and Qwen3.8-Flash-Next (qwen4exp).
TABLE_PATTERNS = [r"^per_layer_token_embd\.weight$", r"(^|[._])n_?gram([._]|$)", r"(^|[._])engram([._]|$)"]
# Architectures whose llama.cpp loader marks the table TENSOR_READ_LAZY (row reads from disk). Read live from
# the local runtime source when it is there; this list is the fallback (llama.cpp @ df03399, 2026-09-10).
LAZY_ARCHS_FALLBACK = {"gemma4", "qwen4exp"}
DISK_TYPES = {"f16": 16.0, "bf16": 16.0, "q8_0": 8.5, "q6_K": 6.6, "q5_K": 5.5}
# flags that copy the whole model into RAM and so defeat lazy reads (and are how a table ends up swapped)
BAD_FLAGS = ["--no-mmap", "--load-mode none", "--load-mode mlock", "-lm none", "-lm mlock", "--mlock", "--lazy-mode off"]


def find_tables(tensor_params, patterns=None):
    """{name: params} for every tensor matching a table pattern."""
    pats = [re.compile(p) for p in (patterns or TABLE_PATTERNS)]
    return {n: p for n, p in (tensor_params or {}).items() if any(r.search(n) for r in pats)}


def _runtime_root():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.environ.get("POLLARD_LLAMA_SRC"), os.path.join(here, "..", "runtime", "llama.cpp")):
        if c and os.path.isdir(os.path.join(c, "src")):
            return os.path.abspath(c)
    return None


def lazy_archs():
    """(arch set, source) -- architectures whose loader reads the table lazily, from the runtime's own source."""
    root = _runtime_root()
    if not root:
        return set(LAZY_ARCHS_FALLBACK), "built-in list (no runtime source found)"
    found = set()
    for f in glob.glob(os.path.join(root, "src", "models", "*.cpp")):
        try:
            if "TENSOR_READ_LAZY" in open(f, encoding="utf-8", errors="replace").read():
                found.add(os.path.splitext(os.path.basename(f))[0])
        except OSError:
            pass
    return (found or set(LAZY_ARCHS_FALLBACK)), (f"runtime source {root}" if found else "built-in list")


def disk_plan(meta, spec="auto", disk_type="q8_0"):
    """Decide which tensors live on the disk tier for a build.
    spec: 'auto' (detected tables, only if this runtime lazy-reads the arch), 'off', or comma-separated regexes.
    -> dict(names, params, type, gb, patterns, note) or None."""
    if spec == "off":
        return None
    tp = meta.get("_tensor_params") or {}
    arch = meta.get("general.architecture") or "unknown"
    if spec == "auto":
        tables = find_tables(tp)
        if not tables:
            return None
        archs, src = lazy_archs()
        if arch not in archs:
            gb = sum(tables.values()) * DISK_TYPES.get(disk_type, 8.5) / 8 / 1e9
            return {"names": [], "params": 0, "type": disk_type, "gb": 0.0, "patterns": [],
                    "note": (f"found {len(tables)} lookup table(s) (~{gb:.1f} GB at {disk_type}) but this runtime does "
                             f"not lazy-read '{arch}' ({src}), so they stay in the RAM budget. Force with --disk PATTERN.")}
        pats = [re.escape(n) for n in tables]
    else:
        pats = [p.strip() for p in spec.split(",") if p.strip()]
        tables = find_tables(tp, pats)
        if not tables:
            sys.exit(f"ERROR: --disk {spec!r} matched no tensor in this model.")
    params = sum(tables.values())
    return {"names": sorted(tables), "params": params, "type": disk_type, "patterns": pats,
            "gb": params * DISK_TYPES.get(disk_type, 8.5) / 8 / 1e9, "note": None}


def run_command(model, tables, ram=None, vram=None, ctx=8192, moe=False, server="llama-server"):
    """The launch line that keeps the tables on disk, plus what NOT to pass."""
    cmd = [server, "-m", model, "-c", str(ctx), "--load-mode", "mmap", "--lazy-mode", "on"]
    if tables:
        cmd += ["-ot", ",".join(f"{re.escape(n)}=CPU" for n in tables)]     # rows are read by the CPU, never uploaded
    if vram:
        cmd += ["-ngl", "999"]
        if moe:
            cmd += ["--n-cpu-moe", "<N>"]      # raise N until it fits VRAM; experts left on CPU stream from RAM
    return cmd


def _q(a):
    return a if re.match(r"^[\w@%+=:,./<>-]+$", a) else "'" + a.replace("'", "'\\''") + "'"


def cmd_inspect(a):
    from pollard_calc import read_gguf_meta
    meta = read_gguf_meta(a.model)
    tp = meta.get("_tensor_params") or {}
    total = sum(tp.values()) or 1
    tables = find_tables(tp, a.pattern.split(",") if a.pattern else None)
    arch = meta.get("general.architecture") or "unknown"
    archs, src = lazy_archs()
    print(f"== pollard-ngram inspect :: {a.model}")
    print(f"architecture        : {arch}")
    if not tables:
        print("lookup tables       : none found (nothing to offload; patterns: " + ", ".join(TABLE_PATTERNS) + ")")
        return
    tparams = sum(tables.values())
    print(f"lookup tables       : {len(tables)} tensor(s), {tparams/1e9:.2f}B params = {100*tparams/total:.0f}% of the model")
    for n, p in sorted(tables.items(), key=lambda x: -x[1]):
        print(f"   {n:<44} {p/1e9:7.2f}B   " + "  ".join(f"{t} {p*b/8/1e9:6.1f} GB" for t, b in DISK_TYPES.items()))
    rest = total - tparams
    print(f"rest of the model   : {rest/1e9:.2f}B params -- this is all the RAM/VRAM budget has to hold")
    ok = arch in archs
    print(f"lazy row reads      : {'YES' if ok else 'NO'} for '{arch}' ({src})")
    if not ok:
        print("   This runtime loads the whole table into RAM for this architecture. Update llama.cpp, or keep the\n"
              "   table in the RAM budget (pollard-fit --disk off).")
    print("\nbuild:  pollard-fit --gguf <f16 source> --ram <GB> --disk auto --disk-type q8_0")
    print(f"run:    pollard-ngram run {a.model} --ram <GB> --vram <GB>")


def cmd_run(a):
    from pollard_calc import read_gguf_meta
    meta = read_gguf_meta(a.model)
    tp = meta.get("_tensor_params") or {}
    tables = find_tables(tp)
    moe = any(".ffn_gate_exps." in n or ".ffn_up_exps." in n for n in tp)
    cmd = run_command(a.model, sorted(tables), a.ram, a.vram, a.ctx, moe, a.server)
    print("== pollard-ngram run :: keep the lookup table on the SSD")
    print("   " + " ".join(_q(c) for c in cmd))
    print("\nwhy each flag:")
    print("   --load-mode mmap      the file is mapped, not copied: untouched table rows never use RAM")
    print("   --lazy-mode on        table rows are read from disk on demand (default 'auto' only does it above 4 GiB)")
    if tables:
        print("   -ot <table>=CPU       the CPU does the row lookups; the table is never uploaded to the GPU")
    if moe and a.vram:
        print("   --n-cpu-moe <N>       start high, lower N until VRAM is full; the rest of the experts run from RAM")
    print("\nnever add: " + ", ".join(BAD_FLAGS) + "\n   each one copies the whole model (table included) into memory, which is what fills swap")
    if a.ram:
        total_bytes = meta.get("_total_file_bytes") or 0
        table_bytes = total_bytes * (sum(tables.values()) / max(1, sum(tp.values())))
        rest_gb = (total_bytes - table_bytes) / 1e9
        print(f"\nmemory check: the rest of the model is ~{rest_gb:.1f} GB (+ KV cache for -c {a.ctx}); RAM {a.ram:.0f} GB"
              + (f" + VRAM {a.vram:.0f} GB" if a.vram else ""))
        if rest_gb > a.ram * 0.85 + (a.vram or 0) * 0.9:
            print("   WARNING: that does not fit -- the OS will swap, and swap is what writes to the SSD.\n"
                  "   Rebuild smaller for this machine:  pollard-fit --gguf <f16> --ram " + f"{a.ram:.0f} --disk auto")


def _disk_counters():
    import psutil
    d = psutil.disk_io_counters()
    s = psutil.swap_memory()
    return d.read_bytes, d.write_bytes, getattr(s, "sin", 0) or 0, getattr(s, "sout", 0) or 0, s.used


def cmd_watch(a):
    try:
        import psutil
    except ImportError:
        sys.exit("pollard-ngram watch needs psutil:  pip install psutil")
    proc = psutil.Process(a.pid) if a.pid else None
    print("== pollard-ngram watch  (Ctrl-C to stop)  -- whole-machine disk I/O and swap, every "
          f"{a.interval:.0f}s" + (f"; process {a.pid}" if proc else ""))
    print(f"{'time':>8} {'read MB/s':>10} {'write MB/s':>11} {'swap-out MB/s':>14} {'swap used GB':>13}" + (f" {'RSS GB':>8}" if proc else ""))
    prev, t0 = _disk_counters(), time.time()
    tot_w = tot_so = 0.0
    try:
        for _ in range(a.samples or 10**9):
            time.sleep(a.interval)
            cur = _disk_counters()
            dt = a.interval
            r, w, so = (cur[0] - prev[0]) / dt / 1e6, (cur[1] - prev[1]) / dt / 1e6, (cur[3] - prev[3]) / dt / 1e6
            tot_w += (cur[1] - prev[1]); tot_so += (cur[3] - prev[3])
            line = f"{time.strftime('%H:%M:%S'):>8} {r:10.1f} {w:11.1f} {so:14.1f} {cur[4]/1e9:13.1f}"
            if proc:
                try:
                    line += f" {proc.memory_info().rss/1e9:8.1f}"
                except psutil.Error:
                    line += "   (gone)"
            print(line, flush=True)
            prev = cur
    except KeyboardInterrupt:
        pass
    hrs = max(1e-9, (time.time() - t0) / 3600)
    print(f"\nwritten: {tot_w/1e9:.2f} GB in {hrs*60:.0f} min" + (f" (~{tot_w/1e9/hrs*24:.0f} GB/day at this rate)" if hrs >= 1 / 6 else "")
          + f"; swapped out: {tot_so/1e9:.2f} GB")
    if hrs < 1 / 6:
        print("verdict: too short to judge (watch at least 10 minutes of real generation; 30+ is better).")
    elif tot_w > 0 and tot_so >= 0.5 * tot_w:
        print("verdict: the writes are SWAP. The model plus its KV cache does not fit in RAM. Check the launch flags\n"
              "         (pollard-ngram run), lower -c, or rebuild smaller with pollard-fit --disk auto.")
    elif tot_w / 1e9 / hrs > 2:
        print("verdict: heavy writes that are not swap (swap counters unavailable on this OS, or another program).\n"
              "         Reading a lazy table never writes; check what else is writing (macOS: Activity Monitor > Disk).")
    else:
        print("verdict: no meaningful writes -- the table is being read in place, which is what you want.")


def main():
    ap = argparse.ArgumentParser(prog="pollard-ngram", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect", help="find lookup tables, their size, and whether this runtime lazy-reads them")
    i.add_argument("model"); i.add_argument("--pattern", help="comma-separated regexes instead of the built-in table names")
    i.set_defaults(fn=cmd_inspect)
    r = sub.add_parser("run", help="print the launch command that keeps the tables on disk")
    r.add_argument("model"); r.add_argument("--ram", type=float); r.add_argument("--vram", type=float)
    r.add_argument("--ctx", type=int, default=8192); r.add_argument("--server", default="llama-server")
    r.set_defaults(fn=cmd_run)
    w = sub.add_parser("watch", help="live disk writes and swap while a model runs")
    w.add_argument("--pid", type=int); w.add_argument("--interval", type=float, default=30.0)
    w.add_argument("--samples", type=int, help="stop after N samples (default: until Ctrl-C)")
    w.set_defaults(fn=cmd_watch)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
