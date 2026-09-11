#!/usr/bin/env python3
"""pollard-exl3-band — band-parallel EXL3 conversion for models that do not fit one node.

The companion to `pollard-exl3`: same lane, same gold recipe, but for a body too large to cook
sequentially on one machine.

exllamav3's converter is sequential: module i+1 is calibrated on the quantized output of module i, and a 744B model would
take ~3 days on one 121 GB node. Its work dir, however, is plain: `args.json`, `ckpt/job.json` (`next_module_idx`,
`bad_rows`), `ckpt/state.safetensors` (one F32 `[1, cols, hidden]` tensor per calibration row: the residual stream entering
the next module), `ckpt/original_input_ids.safetensors` (I64 `[1, cols]` per row) and `qtensors/<module>.safetensors` per
finished module — and the bit strategy is a pure function of config + flags, so every node computes the identical strategy.
This tool exploits that:

  1. one streaming bf16 pass over the calibration rows produces the residual stream at each band start (your own harness,
     or `pollard-export --shard-plan`'s boundary-handoff contract: the last hidden state of band k is the input of band k+1);
  2. every node runs `band`: fresh convert with `--max_module 0` (quantizes only the embedding, writes the canonical work
     dir) → `inject` the band-start state and set `next_module_idx = first_layer + 1` → `--resume --max_module last_layer + 1`;
  3. one node runs `merge`: gather all `qtensors`, inject the post-last-layer state with `next_module_idx = num_layers + 1`,
     resume uncapped → final norm, `lm_head`, MTP (uncalibrated), compile.

Trade: at each band boundary the calibration inputs are the exact model's rather than the quantized prefix's (non-sequential
GPTQ makes this trade at every layer with little measured loss; here it is n_bands−1 of num_layers boundaries). Measured on
GLM-5.3 (78 layers, 10 bands, GB10): ~7 h wall + ~1 h merge instead of ~3 days.

Layer i is module i+1 (module 0 = embeddings, num_layers+1 = final norm, +2 = head). Boundary file = safetensors with one
tensor `[rows, cols, hidden]` (any float dtype; key --state-key, default "h").

    # on node k (band first..last), CONVERT = path to exllamav3/convert.py, extra flags after "--" go to the fresh run:
    pollard-exl3-band band --convert $CONVERT --in-dir $HF --work $W --first 32 --last 39 \
        --state boundary_32.safetensors --calib calib.safetensors --rows 384 --cols 2048 -- -b 3.2 -hb 6 -mb 8 -hq
    # merge node:
    pollard-exl3-band merge --convert $CONVERT --work $W --out $OUT --num-layers 78 --state boundary_78.safetensors \
        --calib calib.safetensors --rows 384 --qtensors /gather/band*/qtensors
    # lower level: fabricate/replace a checkpoint by hand
    pollard-exl3-band inject --work $W --state boundary_32.safetensors --calib calib.safetensors --next 33

Second pass (re-cook only some bands with a corrected --recipe, see experiments/exl3_depth_recipe.py): run `band` again for those bands
with `-- --recipe recipe.yaml ...` into a new work dir, then `merge` with `--qtensors` listing the new bands' dirs FIRST and
`--overwrite` (later dirs never overwrite earlier ones).
"""
import argparse, glob, json, os, shutil, struct, subprocess, sys


def _safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); h.pop("__metadata__", None)
    return h


def inject(work, state_path, calib_path, next_idx, rows=None, state_key="h"):
    """Write ckpt/{state,original_input_ids}.safetensors + job.json so a resume starts at module `next_idx`
    from the given residual-stream tensor. Streaming writer: never holds two copies of the (large) state."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    ck = os.path.join(work, "ckpt"); os.makedirs(ck, exist_ok=True)
    with safe_open(state_path, "pt") as f:
        key = state_key if state_key in f.keys() else list(f.keys())[0]
        sl = f.get_slice(key); R, S, H = sl.get_shape(); rows = rows or R
        assert R >= rows, f"boundary has {R} rows, need {rows}"
        nb = S * H * 4; hdr = {}; off = 0
        for i in range(rows):
            hdr[f"tensor.{i}"] = {"dtype": "F32", "shape": [1, S, H], "data_offsets": [off, off + nb]}; off += nb
        hb = json.dumps(hdr, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
        tmp = os.path.join(ck, "state.safetensors.tmp")
        with open(tmp, "wb") as o:
            o.write(struct.pack("<Q", len(hb))); o.write(hb)
            for i in range(rows):
                t = sl[i:i + 1].to(torch.float32).contiguous()
                if not torch.isfinite(t).all(): sys.exit(f"non-finite boundary row {i}")
                o.write(t.numpy().tobytes())
        os.replace(tmp, os.path.join(ck, "state.safetensors"))
    ids = load_file(calib_path)["input_ids"]
    assert ids.shape[0] >= rows and ids.shape[1] == S, f"calib {tuple(ids.shape)} vs state cols {S}"
    save_file({f"tensor.{i}": ids[i:i + 1].to(torch.int64).contiguous() for i in range(rows)}, os.path.join(ck, "original_input_ids.safetensors"))
    json.dump({"next_module_idx": int(next_idx), "q_strategy": None, "bad_rows": []}, open(os.path.join(ck, "job.json"), "w"), indent=4)
    print(f"inject: next_module_idx={next_idx} rows={rows} cols={S} hidden={H} state_bytes={off}")


def _run(cmd):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd); return r.returncode


def cmd_inject(a):
    inject(a.work, a.state, a.calib, a.next, a.rows, a.state_key)


def cmd_band(a):
    extra = a.extra[1:] if a.extra and a.extra[0] == "--" else a.extra
    py = sys.executable
    if not a.resume_only:
        for d in ("ckpt", "ckpt_old", "ckpt_new", "qtensors", "images"):
            shutil.rmtree(os.path.join(a.work, d), ignore_errors=True)
        for f in ("args.json",):
            try: os.remove(os.path.join(a.work, f))
            except FileNotFoundError: pass
        os.makedirs(os.path.join(a.work, "out"), exist_ok=True)
        rc = _run([py, a.convert, "-i", a.in_dir, "-w", a.work, "-o", os.path.join(a.work, "out"), "-cd", a.calib, "-cr", str(a.rows), "-cc", str(a.cols),
                   "-cpi", "0", "--max_module", "0", "-d", a.devices] + extra)
        if rc: sys.exit(f"template run failed rc={rc}")
        if not os.path.exists(os.path.join(a.work, "ckpt", "state.safetensors")): sys.exit("template run left no checkpoint")
        if a.first > 0:
            if not a.state: sys.exit("--state (band-start residual stream) is required when --first > 0")
            inject(a.work, a.state, a.calib, a.first + 1, a.rows, a.state_key)
    rc = _run([py, a.convert, "-w", a.work, "-r", "--max_module", str(a.last + 1), "-d", a.devices, "-cpi", str(a.checkpoint_interval)])
    done = sorted(glob.glob(os.path.join(a.work, "qtensors", "model.layers.*.safetensors")))
    print(f"band {a.first}-{a.last}: rc={rc}, layer qtensors present: {len(done)}")
    sys.exit(0 if rc == 0 and len(done) >= a.last - a.first + 1 else 1)


def cmd_merge(a):
    qt = os.path.join(a.work, "qtensors"); os.makedirs(qt, exist_ok=True); n = 0
    for pattern in a.qtensors:
        for d in sorted(glob.glob(pattern)):
            for f in sorted(glob.glob(os.path.join(d, "model.layers.*.safetensors"))):
                dst = os.path.join(qt, os.path.basename(f))
                if os.path.exists(dst) and not a.overwrite: continue
                if os.path.exists(dst): os.remove(dst)      # never write through a hardlink
                (os.link if a.hardlink else shutil.copy2)(f, dst); n += 1
    print(f"merge: {n} layer qtensors placed")
    missing = [i for i in range(a.num_layers) if not os.path.exists(os.path.join(qt, f"model.layers.{i}.safetensors"))]
    if missing: sys.exit(f"missing layer qtensors: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if not os.path.exists(os.path.join(qt, "model.embed_tokens.safetensors")): sys.exit("missing model.embed_tokens qtensor (run `band` for the first band in this work dir, or copy it)")
    inject(a.work, a.state, a.calib, a.num_layers + 1, a.rows, a.state_key)
    rc = _run([sys.executable, a.convert, "-w", a.work, "-r", "-o", a.out, "-d", a.devices, "-cpi", str(a.checkpoint_interval)])
    ok = rc == 0 and os.path.exists(os.path.join(a.out, "model.safetensors.index.json"))
    print("merge:", "DONE" if ok else f"FAILED rc={rc}", a.out); sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--work", required=True); common.add_argument("--calib", required=True, help="safetensors with input_ids [rows, cols] (the converter's -cd file)")
    common.add_argument("--rows", type=int, default=None); common.add_argument("--state-key", default="h")
    p = sp.add_parser("inject", parents=[common]); p.add_argument("--state", required=True); p.add_argument("--next", type=int, required=True)
    p = sp.add_parser("band", parents=[common]); p.add_argument("--convert", required=True, help="path to exllamav3 convert.py"); p.add_argument("--in-dir", required=True)
    p.add_argument("--first", type=int, required=True); p.add_argument("--last", type=int, required=True); p.add_argument("--state", help="residual stream entering --first (not needed for the first band)")
    p.add_argument("--cols", type=int, default=2048); p.add_argument("--devices", default="0"); p.add_argument("--checkpoint-interval", type=int, default=900)
    p.add_argument("--resume-only", action="store_true", help="skip template+inject, just resume (after a crash)"); p.add_argument("extra", nargs=argparse.REMAINDER, help="-- then converter flags (-b/-hb/-mb/-hq/--recipe ...)")
    p = sp.add_parser("merge", parents=[common]); p.add_argument("--convert", required=True); p.add_argument("--out", required=True); p.add_argument("--num-layers", type=int, required=True)
    p.add_argument("--state", required=True, help="residual stream after the last layer"); p.add_argument("--qtensors", nargs="+", required=True, help="dirs (globs ok) holding model.layers.*.safetensors; earlier wins unless --overwrite")
    p.add_argument("--overwrite", action="store_true"); p.add_argument("--hardlink", action="store_true"); p.add_argument("--devices", default="0"); p.add_argument("--checkpoint-interval", type=int, default=900)
    a = ap.parse_args()
    if a.cmd == "band" and a.rows is None:
        sys.exit("--rows is required for band (calibration rows)")
    {"inject": cmd_inject, "band": cmd_band, "merge": cmd_merge}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
