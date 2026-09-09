#!/usr/bin/env python3
"""Make an exllamav3 GLM-5.3 / DeepSeek-style artifact loadable by vLLM's DeepSeekMTP drafter (cuda-exl3 backend).

vLLM builds `model.layers.<mtp>.eh_proj` as a plain nn.Linear (never quantized) but exllamav3 quantizes it with the MTP side
model (`mtp_bits`), so the loader dies with `KeyError: 'model.layers.78.eh_proj.mul1'`. Fix the artifact: drop the EXL3 tensors
of eh_proj (trellis/suh/svh/mul1) from EVERY shard that physically holds them and add the source bf16 `eh_proj.weight` in a new
small shard; rewrite the index.

Design note: walk the shard HEADERS, not the index. exllamav3's index for the MTP side model can point at the wrong shard
(it said shard 45; the tensors sat in shard 44, and 3 small norm tensors are written twice across shard boundaries). An index-driven version trusted
the index, rewrote the wrong shard and left the EXL3 tensors in place -> two failed boots.

usage: exl3_fix_mtp_ehproj.py <artifact_dir> <bf16_source_file_holding_eh_proj.weight> [--layer 78]
"""
import glob, json, os, struct, sys
from safetensors import safe_open
from safetensors.torch import save_file

art, src = sys.argv[1], sys.argv[2]
layer = sys.argv[sys.argv.index("--layer") + 1] if "--layer" in sys.argv else "78"
K = f"model.layers.{layer}.eh_proj."

def header(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))

holders = {}
for p in sorted(glob.glob(os.path.join(art, "*.safetensors"))):
    ks = [k for k in header(p) if k.startswith(K) and k != K + "weight"]
    if ks: holders[p] = ks
print("EXL3 eh_proj tensors by FILE:", {os.path.basename(p): v for p, v in holders.items()})
for p, ks in holders.items():
    keep = {}
    with safe_open(p, "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            if not (k.startswith(K) and k != K + "weight"): keep[k] = f.get_tensor(k)
    tmp = p + ".tmp"; save_file(keep, tmp, metadata=meta or {"format": "pt"}); os.replace(tmp, p)
    left = [k for k in header(p) if k.startswith(K)]
    print(f"rewrote {os.path.basename(p)}: kept {len(keep)} tensors, {os.path.getsize(p)/1e9:.2f} GB, eh_proj left: {left}")

idx_p = os.path.join(art, "model.safetensors.index.json"); idx = json.load(open(idx_p)); wm = idx["weight_map"]
for k in [k for k in wm if k.startswith(K) and k != K + "weight"]: del wm[k]
new = "model-mtp-eh_proj.safetensors"
if not os.path.exists(os.path.join(art, new)):
    with safe_open(src, "pt") as f: w = f.get_tensor(K + "weight")
    assert w.shape[1] == 2 * w.shape[0], f"eh_proj should be (hidden, 2*hidden); got {tuple(w.shape)}"
    save_file({K + "weight": w.contiguous()}, os.path.join(art, new), metadata={"format": "pt"})
    print(f"added {new} ({w.dtype}, {tuple(w.shape)})")
wm[K + "weight"] = new
idx["metadata"]["total_size"] = sum(os.path.getsize(os.path.join(art, f)) for f in set(wm.values()))
json.dump(idx, open(idx_p, "w"), indent=1)
# final audit: no EXL3 eh_proj tensor anywhere on disk, and every file key is indexed
stray = {os.path.basename(p): [k for k in header(p) if k.startswith(K) and k != K + "weight"] for p in glob.glob(os.path.join(art, "*.safetensors"))}
stray = {p: v for p, v in stray.items() if v}
print("EHPROJ-FIX-OK" if not stray else f"EHPROJ-FIX-FAILED stray={stray}", f"index now {len(wm)} tensors")
sys.exit(0 if not stray else 1)
