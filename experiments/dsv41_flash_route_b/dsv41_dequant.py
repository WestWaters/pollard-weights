#!/usr/bin/env python3
"""dsv41_dequant.py — EXACT upscale of deepseek-ai/DeepSeek-V4.1-Flash to bf16 ("the bf16 source that DeepSeek never shipped").

Layouts (from the official inference/convert.py + kernel.py and the shard headers, 2026-09-10):
  * dense linears / shared experts / engram.wkv / mtp main_proj:  weight F8_E4M3 [N, K], scale F8_E8M0 [ceil(N/32), ceil(K/32)]
        W[n, k] = e4m3(weight[n, k]) * 2 ** (scale[n // 32, k // 32] - 127)
  * routed experts (w1, w3: I8 [I, H/2]; w2: I8 [H, I/2]) packed MXFP4, 2 values per byte along K, LOW nibble = even k:
        nibble = sign(bit 3) | e2m1 index;  LUT = [0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6]
        scale F8_E8M0 [N, K/32];   W[n, k] = LUT[nibble] * 2 ** (scale[n, k // 32] - 127)
  * Engram tables (layers.{1,14}.engram.embed.{weight,scale}, ~94 GiB fp8 each) are lookup tables: NOT dequantized, their shards are
    copied/hardlinked verbatim (a bf16 table would be 183 GiB per layer and buys nothing).
  * everything else (bf16 norms/embed/head/gates, fp32 hc_*/attn_sink/gate.bias) is copied as-is.
Every value is exactly representable in bf16 (3-bit mantissas × powers of two), so with --check each dequantized tensor is requantized
with the SAME scales and compared byte-for-byte to the source (proof of exactness; any mismatch aborts).

Reads shard headers itself (no safetensors dtype support needed for F8_E8M0); writes standard safetensors bf16 shards with the same
names + a rebuilt model.safetensors.index.json + config.json without quantization_config + all small files.

usage: dsv41_dequant.py --src SRC_DIR --dst DST_DIR [--shards 1-46] [--check] [--workers 4] [--link-engram]
"""
import argparse, json, os, struct, sys, time, shutil, concurrent.futures as cf
import numpy as np, torch
from safetensors.torch import save_file

FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
DT = {"BF16": (torch.bfloat16, 2), "F16": (torch.float16, 2), "F32": (torch.float32, 4), "F8_E4M3": (torch.float8_e4m3fn, 1),
      "F8_E8M0": (torch.uint8, 1), "I8": (torch.uint8, 1), "U8": (torch.uint8, 1), "I32": (torch.int32, 4), "I64": (torch.int64, 8)}

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    meta = h.pop("__metadata__", None); return h, 8 + n, meta

def read_raw(path, base, info):
    s, e = info["data_offsets"]
    with open(path, "rb") as f: f.seek(base + s); buf = f.read(e - s)
    dt, _ = DT[info["dtype"]]; t = torch.frombuffer(bytearray(buf), dtype=dt if dt != torch.float8_e4m3fn else torch.uint8)
    if info["dtype"] == "F8_E4M3": t = t.view(torch.float8_e4m3fn)
    return t.reshape(info["shape"])

def e8m0_to_f32(s_u8): return torch.ldexp(torch.ones(s_u8.shape, dtype=torch.float32), s_u8.to(torch.int32) - 127)

def dequant_fp8_block(w, s, block=32):
    N, K = w.shape; sf = e8m0_to_f32(s)  # [ceil(N/32), ceil(K/32)]
    wf = w.float(); pad_n = sf.shape[0] * block - N; pad_k = sf.shape[1] * block - K
    if pad_n or pad_k: wf = torch.nn.functional.pad(wf, (0, pad_k, 0, pad_n))
    out = (wf.unflatten(0, (-1, block)).unflatten(-1, (-1, block)) * sf[:, None, :, None]).flatten(2, 3).flatten(0, 1)
    return out[:N, :K].to(torch.bfloat16)

def requant_fp8_block(wb, s, block=32):
    N, K = wb.shape; sf = e8m0_to_f32(s); wf = wb.float(); pad_n = sf.shape[0] * block - N; pad_k = sf.shape[1] * block - K
    if pad_n or pad_k: wf = torch.nn.functional.pad(wf, (0, pad_k, 0, pad_n))
    q = (wf.unflatten(0, (-1, block)).unflatten(-1, (-1, block)) / sf[:, None, :, None]).flatten(2, 3).flatten(0, 1)[:N, :K]
    return q.to(torch.float8_e4m3fn).view(torch.uint8)

def dequant_fp4(packed_u8, s):
    N, Kh = packed_u8.shape; low = packed_u8 & 0x0F; high = (packed_u8 >> 4) & 0x0F
    vals = torch.stack([FP4_LUT[low.long()], FP4_LUT[high.long()]], dim=-1).flatten(1)  # [N, K], even k = low nibble
    sf = e8m0_to_f32(s)  # [N, K/32]
    out = vals.unflatten(-1, (-1, 32)) * sf[:, :, None]
    return out.flatten(1).to(torch.bfloat16)

def requant_fp4(wb, s):
    sf = e8m0_to_f32(s); v = (wb.float().unflatten(-1, (-1, 32)) / sf[:, :, None]).flatten(1)
    # exact match to a LUT entry (values are exact by construction); -0.0 vs 0.0: source may use either code for zero → compare via LUT values
    mag = (v.abs().unsqueeze(-1) == FP4_LUT[:8]).float().argmax(-1)          # e2m1 magnitude index 0..7
    ok = bool((FP4_LUT[:8][mag] == v.abs()).all())                             # every value is on the grid
    idx = mag + 8 * torch.signbit(v).long()                                    # sign bit incl. -0.0 (bf16 keeps signed zero)
    even, odd = idx[:, 0::2], idx[:, 1::2]
    return (even | (odd << 4)).to(torch.uint8), ok

def is_engram_table(name): return ".engram.embed." in name

def process_shard(args, shard, wm_all, dst_index_lock=None):
    src = os.path.join(args.src, shard); dst = os.path.join(args.dst, shard)
    h, base, meta = header(src); names = list(h.keys())
    if any(is_engram_table(n) for n in names):   # shards 47/48 hold the 94 GiB tables PLUS engram.wkv/q_weight/k_weight: never read into RAM; copy verbatim (wkv stays fp8+scale, the kit dequantizes it on the fly)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src): return shard, {n: shard for n in names}, "engram shard already present"
        if args.link_engram:
            try: os.link(src, dst); return shard, {n: shard for n in names}, "engram shard hardlinked"
            except OSError: pass
        shutil.copyfile(src, dst); return shard, {n: shard for n in names}, "engram shard copied"
    if os.path.exists(dst) and not os.path.exists(dst + ".tmp"):
        try:  # resume: a complete shard has a valid header and all its names → skip (a store-host job was killed mid-run on 09-10)
            h2, _, _ = header(dst)
            if set(h2.keys()) >= {n for n in names if not (n.endswith(".scale") and n[: -len(".scale")] + ".weight" in wm_all)}:
                return shard, {n: shard for n in h2}, "already present (resume)"
        except Exception: pass
    out = {}; stats = {"fp8": 0, "fp4": 0, "copy": 0, "bytes_in": 0, "bytes_out": 0}; t0 = time.time()
    scale_of = {n: n[: -len(".weight")] + ".scale" for n in names if n.endswith(".weight")}
    for n in names:
        info = h[n]; stats["bytes_in"] += info["data_offsets"][1] - info["data_offsets"][0]
        if n.endswith(".scale") and n[: -len(".scale")] + ".weight" in wm_all: continue  # consumed with its weight
        sname = scale_of.get(n)
        if sname and sname in wm_all and info["dtype"] in ("F8_E4M3", "I8") and not is_engram_table(n):
            sshard = wm_all[sname]; sp = os.path.join(args.src, sshard); sh, sbase, _ = header(sp) if sshard != shard else (h, base, None)
            s = read_raw(sp, sbase, sh[sname]); w = read_raw(src, base, info)
            if info["dtype"] == "F8_E4M3":
                wb = dequant_fp8_block(w, s)
                if args.check:
                    back = requant_fp8_block(wb, s); assert torch.equal(back, w.view(torch.uint8)), f"fp8 round-trip mismatch {n}"
                stats["fp8"] += 1
            else:
                wb = dequant_fp4(w, s)
                if args.check:
                    back, ok = requant_fp4(wb, s)
                    if not (ok and torch.equal(back, w)):
                        diff = (back != w); raise AssertionError(f"fp4 round-trip mismatch {n}: on-grid={ok} bytes differ {int(diff.sum())}/{diff.numel()} e.g. src {w[diff][:4].tolist()} back {back[diff][:4].tolist()}")
                stats["fp4"] += 1
            out[n] = wb.contiguous()
        else:
            out[n] = read_raw(src, base, info).clone(); stats["copy"] += 1
    stats["bytes_out"] = sum(t.numel() * t.element_size() for t in out.values())
    save_file(out, dst + ".tmp", metadata=meta or {"format": "pt"}); os.replace(dst + ".tmp", dst)   # atomic: a killed run never leaves a truncated shard
    return shard, {n: shard for n in out}, f"fp8 {stats['fp8']} fp4 {stats['fp4']} copy {stats['copy']} {stats['bytes_in'] / 1e9:.1f}→{stats['bytes_out'] / 1e9:.1f} GB {time.time() - t0:.0f}s"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--shards", default=None, help="e.g. 1-46 (1-based shard numbers); default all"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--link-engram", action="store_true"); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); torch.set_num_threads(a.threads); os.makedirs(a.dst, exist_ok=True)
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json"))); wm_all = idx["weight_map"]
    shards = sorted(set(wm_all.values()))
    if a.shards:
        lo, hi = [int(x) for x in a.shards.split("-")]; shards = [s for s in shards if lo <= int(s.split("-")[1]) <= hi]
    print(f"[{time.strftime('%H:%M:%S')}] {len(shards)} shards, {len(wm_all)} tensors, check={a.check}", flush=True)
    new_map = {}
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_shard, a, s, wm_all): s for s in shards}
        for f in cf.as_completed(futs):
            shard, m, msg = f.result(); new_map.update(m); print(f"[{time.strftime('%H:%M:%S')}] {shard}: {msg}", flush=True)
    # index (merge with any previously written shards in dst so partial runs compose)
    ip = os.path.join(a.dst, "model.safetensors.index.json")
    if os.path.exists(ip): old = json.load(open(ip))["weight_map"]; old.update(new_map); new_map = old
    total = sum(os.path.getsize(os.path.join(a.dst, s)) for s in set(new_map.values()) if os.path.exists(os.path.join(a.dst, s)))
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_map.items()))}, open(ip, "w"), indent=1)
    for f in os.listdir(a.src):
        p = os.path.join(a.src, f)
        if os.path.isfile(p) and not f.endswith(".safetensors") and f != "model.safetensors.index.json":
            shutil.copyfile(p, os.path.join(a.dst, f))
    for sub in ("inference", "assets", "encoding"):
        if os.path.isdir(os.path.join(a.src, sub)) and not os.path.exists(os.path.join(a.dst, sub)): shutil.copytree(os.path.join(a.src, sub), os.path.join(a.dst, sub))
    cfg = json.load(open(os.path.join(a.src, "config.json"))); qc = cfg.pop("quantization_config", None)
    cfg["_bot_lab_21_upscale"] = {"from": "deepseek-ai/DeepSeek-V4.1-Flash", "method": "exact dequant (fp8 e4m3 x 2^(ue8m0-127) 32x32 blocks; MXFP4 e2m1 x 2^(ue8m0-127) per-row-per-32) → bf16; Engram tables left fp8",
                                  "original_quantization_config": qc, "round_trip_verified": a.check}
    json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] DSV41-DEQUANT-DONE {len(new_map)} tensors, {total / 1e9:.0f} GB in {a.dst}", flush=True)

if __name__ == "__main__": main()
