#!/usr/bin/env python3
"""dsv41_engram_extract.py — run ON the store host: extract exactly the Engram rows the capture kit needs from the fp8 tables in
shards 47/48 with ONE sequential pass per table (the store is spinning ZFS: random 264-byte reads would take hours; a chunked scan is
disk-bandwidth bound, ~10-15 min per 94 GB table). Input: one or more kit hash caches engram_hashes_r{start}_{R}[...].safetensors
({"hashes": int64 [R, S, n_engram_layers, 24]}). Output per (layer, hash file): the kit's cache format
engram_rows_L{LL}_r{start}_{R}[...].safetensors = {"uniq": int64 [U] sorted, "rows": uint8 [U, 256], "scales": uint8 [U, 8]}.
Pure numpy + safetensors. usage: dsv41_engram_extract.py --ckpt /hf/DeepSeek-V4.1-Flash --out DIR --hashes A.safetensors [B.safetensors ...]"""
import argparse, json, os, struct, time, numpy as np
from safetensors.numpy import load_file, save_file
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True); ap.add_argument("--hashes", nargs="+", required=True)
ap.add_argument("--layer-ids", default="1,14"); ap.add_argument("--chunk-mb", type=int, default=512); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
def log(m): print(time.strftime("[%H:%M:%S]"), m, flush=True)
wm = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]
def header(path):
    with open(path, "rb") as f: n = struct.unpack("<Q", f.read(8))[0]; return json.loads(f.read(n)), 8 + n
jobs = []  # (tag, hashes array)
for hp in a.hashes:
    base = os.path.basename(hp); tag = base[len("engram_hashes_"):-len(".safetensors")]; H = load_file(hp)["hashes"]; jobs.append((tag, H)); log(f"{base}: hashes {H.shape}")
for li, L in enumerate(int(x) for x in a.layer_ids.split(",")):
    uniqs = {tag: np.unique(H[:, :, li, :].reshape(-1)).astype(np.int64) for tag, H in jobs}
    outs = {tag: {"rows": np.empty((u.shape[0], 256), np.uint8), "scales": np.empty((u.shape[0], 8), np.uint8)} for tag, u in uniqs.items()}
    for kind, width, okey in (("weight", 256, "rows"), ("scale", 8, "scales")):
        key = f"layers.{L}.engram.embed.{kind}"; shard = wm[key]; path = os.path.join(a.ckpt, shard); hdr, hb = header(path); info = hdr[key]
        rows_n, rb = info["shape"]; s0, e0 = info["data_offsets"]; assert rb == width and e0 - s0 == rows_n * rb, (key, info)
        for tag, u in uniqs.items(): assert u.min() >= 0 and u.max() < rows_n, (tag, u.min(), u.max(), rows_n)
        chunk_rows = max(1, (a.chunk_mb << 20) // rb); t0 = time.time(); done = 0
        with open(path, "rb") as f:
            for r0 in range(0, rows_n, chunk_rows):
                r1 = min(rows_n, r0 + chunk_rows); need = False
                for tag, u in uniqs.items():
                    lo, hi = np.searchsorted(u, r0), np.searchsorted(u, r1)
                    if hi > lo: need = True
                if not need: continue
                f.seek(hb + s0 + r0 * rb); buf = np.frombuffer(f.read((r1 - r0) * rb), np.uint8).reshape(r1 - r0, rb)
                for tag, u in uniqs.items():
                    lo, hi = np.searchsorted(u, r0), np.searchsorted(u, r1)
                    if hi > lo: outs[tag][okey][lo:hi] = buf[u[lo:hi] - r0]
                done += r1 - r0
                if (r0 // chunk_rows) % 20 == 0: log(f"  L{L} {kind}: {done / rows_n * 100:5.1f} % ({(done * rb) / 1e9:.1f} GB) {time.time() - t0:.0f}s")
        log(f"L{L} {kind}: scanned {rows_n} rows in {time.time() - t0:.0f}s")
    for tag, u in uniqs.items():
        p = os.path.join(a.out, f"engram_rows_L{L:02d}_{tag}.safetensors")
        save_file({"uniq": u, "rows": outs[tag]["rows"], "scales": outs[tag]["scales"]}, p); log(f"L{L} {tag}: {u.shape[0]} rows → {p} ({os.path.getsize(p) / 1e6:.0f} MB)")
log("DSV41-ENGRAM-EXTRACT-DONE")
