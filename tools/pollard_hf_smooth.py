#!/usr/bin/env python3
"""pollard-hf-smooth -- activation-aware SmoothQuant preconditioning for an HF model, IN PLACE of the
weights, folded into the RMSNorms as a mathematical identity. Run it BEFORE a low-bit convert (EXL3 /
GPTQ / MX lanes) so massive-activation input channels can't wreck the quantization.

Why it exists: low-bit trellis/GPTQ quant has NO input-outlier protection. A single massive-activation
input channel (e.g. the residual-stream outlier that emerges in the first couple of layers) collapses the
quantizer's global scale, so every NORMAL channel gets under-quantized and the layer forwards to garbage --
while per-tensor and Hessian-proxy metrics still look perfect (see legacy/PROXY_ERR_BANNED.md). Measured
case: Qwen2.5-3B layer 1 MLP-input activation max 453 -> 4bpw layer sqnr -14 (PPL 3090); after this
preconditioning -> sqnr +18, sane PPL, no bits spent.

The fix (SmoothQuant, per input channel j, exact/identity):
    s_j = max|X_j|^alpha / max|W_j|^(1-alpha)          (clamped)
    norm.weight[j] /= s_j ;  W[:, j] *= s_j   for every consumer linear W on that seam
X activations come from a calibration forward; the fold into RMSNorm.weight is exact (RMS is over the
pre-weight input), so the model computes the same thing -- the quantizer just sees a flatter input.

Seams (norm -> linears), auto-detected per architecture (SEAMS below; a norm that also feeds a router or experts
is left alone -- scaling only some of its readers would change the model):
    A: input_layernorm / operator_norm (LFM2) / norm (Nemotron-H attention) -> q_proj, k_proj, v_proj
    B: post_attention_layernorm -> gate_proj, up_proj ; ffn_norm (LFM2 dense) -> w1, w3

STREAMED (pollard_stream): the model is never loaded whole. The text backbone is resolved from config.json (native
transformers class, a VLM / omni's language model, a *_puzzle per-layer config, or -- with consent -- the
checkpoint's own code), the calibration rows run through one block at a time read from disk (dequantized on read
when the source is already quantized), and each block's seams are folded as it passes. Peak RAM = one block + the
activations, preflighted against free RAM (--max-ram-gb caps it). Towers, talkers and projectors are copied through
byte for byte. --overlay writes only the scales (megabytes) over the source for the streamed lanes; --materialize
turns an overlay into a full checkpoint for consumers that read files themselves (exllamav3, GGUF convert).

  pollard-hf-smooth --model ./Qwen2.5-3B --out ./Qwen2.5-3B-sm --calib cal.txt
  pollard-exl3 --model ./Qwen2.5-3B-sm --out ./M-exl3 --bpw 4.0        # then convert the smoothed model

Verify every build with pollard-verify (never trust a proxy metric).
"""
import argparse, json, os, sys, time


def load_backbone(model_id, dtype=None, device="cpu", eval_mode=True, **kw):
    """This tool's OWN loader -- model tooling does not depend on a shared/brain-side one.

    AutoModelForCausalLM refuses a vision-language config outright ("Unrecognized configuration
    class Qwen2VLConfig for this kind of AutoModel"), so fall back to the vision-language auto
    classes: a VL model's text stack quantizes like any other. Both failures are reported if
    neither works, not just the last one.

    Pass `device_map=` to shard across GPU+CPU+disk; accelerate owns placement after that, so
    `device` is ignored in that case.
    """
    import torch as _t, transformers as _tf
    from transformers import AutoModelForCausalLM
    if dtype is None:
        dtype = _t.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kw)
    except ValueError as text_only_err:
        model, errs = None, [f"AutoModelForCausalLM: {text_only_err}"]
        for _n in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            _c = getattr(_tf, _n, None)
            if _c is None:
                continue
            try:
                model = _c.from_pretrained(model_id, dtype=dtype, **kw)
                break
            except Exception as e:
                errs.append(f"{_n}: {e}")
        if model is None:
            raise SystemExit(f"could not load {model_id!r} as a causal LM or a vision-language "
                             "model:\n  " + "\n  ".join(str(e)[:160] for e in errs)) from None
    if kw.get("device_map") is None:            # accelerate already placed a dispatched model
        model = model.to(device)
    return model.eval() if eval_mode else model


def text_layers(model):
    """This tool's OWN layer walk. A VL model keeps its text stack under `model.language_model`."""
    for path in ("model.language_model", "language_model.model", "model"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        layers = getattr(node, "layers", None) if node is not None else None
        if layers is not None:
            return layers
    layers = getattr(model, "layers", None)
    if layers is not None:
        return layers
    raise SystemExit(f"could not find the decoder layers on {type(model).__name__}; "
                     "this tool's text_layers() needs a path for this architecture")


#: (norm, its consumers) per decoder block, by checkpoint name under the block. A seam is folded only when EVERY
#: consumer is present -- a norm that also feeds something else (a MoE router + experts) is left alone, because
#: scaling only some of its readers would change the model.
SEAMS = [
    ("A", "input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
    ("B", "post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
    ("A", "operator_norm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),      # LFM2
    ("B", "ffn_norm", ("feed_forward.w1", "feed_forward.w3")),                                  # LFM2 dense
    ("A", "norm", ("mixer.q_proj", "mixer.k_proj", "mixer.v_proj")),                            # Nemotron-H attn
]


def block_seams(keys, prefix):
    """[(tag, norm key, [consumer keys])] of one block, from its checkpoint keys (prefix = '<...>.layers.<i>.')."""
    have = set(keys)
    out = []
    for tag, norm, cons in SEAMS:
        nk = f"{prefix}{norm}.weight"
        ck = [f"{prefix}{c}.weight" for c in cons]
        if nk in have and all(c in have for c in ck):
            readers = [k for k in keys if k.startswith(prefix) and k.endswith(".weight") and k not in ck
                       and k != nk and _reads_norm(k[len(prefix):], norm)]
            if readers:
                continue
            out.append((tag, nk, ck))
    return out


def _reads_norm(rel, norm):
    """Another linear in the block that would read this norm's output (guards a fold that would miss a reader)."""
    if norm in ("input_layernorm", "operator_norm"):
        return rel.startswith(("self_attn.qkv_proj", "self_attn.q_a_proj", "self_attn.kv_a_proj"))
    if norm in ("post_attention_layernorm", "ffn_norm"):
        return rel.startswith(("mlp.gate.", "mlp.router", "mlp.experts.", "feed_forward.gate.", "feed_forward.experts."))
    return False


def _norm_offset(bb, norm_key):
    """True when the block's norm computes x * (1 + w) (Gemma-style): the fold is then (1 + w) / s - 1."""
    import inspect
    t = bb.kmap.get(norm_key)
    if not t:
        return False
    mod = bb.root.get_submodule(t[0].rsplit(".", 1)[0])
    try:
        src = inspect.getsource(type(mod).forward)
    except (OSError, TypeError):
        return False
    return "1.0 + self.weight" in src or "1 + self.weight" in src or "(1.0 + self.weight" in src


def pow2_scales(sc, clamp=(1e-2, 1e2)):
    """Each SmoothQuant scale rounded to the nearest power of two (in log2), kept inside `clamp`.

    Why: the fold is an identity in exact arithmetic only. Stored back in bf16, norm.weight / s and W[:, j] * s are
    each rounded again -- on Qwen2.5-0.5B-Instruct the smoothed bf16 checkpoint ALONE (no quantization) read KLD
    0.0011 / top-1 98.2% against its source (BOS-led 2048-token chunks, second half; wikitext-2 and the release
    held-out alike, 2026-10-18), about what an int8 rung costs, and every lane built from it started there. A power
    of two only moves the exponent, so the folded bf16 weights are exactly the source's (barring under/overflow,
    which the clamp keeps far away): the smoothed model IS the model."""
    import numpy as np
    sc = np.asarray(sc, np.float64)
    lo, hi = np.ceil(np.log2(clamp[0])), np.floor(np.log2(clamp[1]))
    e = np.clip(np.round(np.log2(np.clip(sc, 1e-30, None))), lo, hi)
    return np.exp2(e).astype(np.float32)


def smooth_stream(model, out, calib=None, alpha=0.5, clamp=(1e-2, 1e2), rows=12, cols=512, device="cpu",
                  trust_remote_code="auto", overlay=False, max_ram_gb=None, log=print, act_dtype=None,
                  scales="pow2"):
    """SmoothQuant, streamed one block at a time from disk. Writes either a full smoothed checkpoint (`out`) or,
    with overlay=True, a folder holding only the scales over the source (every streamed lane reads it as the
    smoothed model; it costs megabytes instead of a second copy of the weights). Returns a summary dict.

    scales="pow2" (default): every scale a power of two, so the fold is exact in the checkpoint's own bf16
    (pow2_scales); "exact": the raw SmoothQuant ratio (the fold then rounds every smoothed weight once more)."""
    import numpy as np, torch
    import pollard_stream as S
    t0 = time.time()
    bb = S.Backbone(model, trust_remote_code, log=log)
    log(f"   backbone: {bb.how}  ({bb.n_layers} blocks, {len(bb.other_keys)} tensors copied through)")
    plan = S.fit_memory(bb, rows, cols, S.ram_budget(max_ram_gb), log=log, hessian=False, act_bytes=4)
    rows = plan["nsamples"]
    data = S.calib_rows(bb.src.base_dir if bb.src.overlay else model, calib, rows, cols,
                        trust=bb.remote)
    bb.dtype = S.plan_dtype(plan)
    eng = S.Engine(bb, device=device, act_dtype=act_dtype or torch.float32, log=log)
    eng.record(data)
    ops, folded, worst = {}, 0, (0, 0.0)
    writer = None if overlay else S.ShardWriter(out)
    for i in range(bb.n_layers):
        eng.load_layer(i)
        pfx = bb.layer_ckpt_prefix + f"{i}."
        seams = block_seams(bb.layer_keys[i], pfx)
        with S.InputStats(eng, i, hessian=False, amax=True) as st:
            outs = eng.run_layer(i)
        new = {}
        for tag, nk, ck in seams:
            a_max = st.amax_for(ck[0])
            if a_max is None:
                continue
            a_max = a_max.float().cpu().numpy().clip(min=1e-8)
            Ws = {c: bb.src.read(c) for c in ck}
            w_max = np.max(np.stack([np.abs(w).max(0) for w in Ws.values()]), 0).clip(min=1e-8)
            if a_max.shape != w_max.shape:
                continue
            sc = np.clip(a_max ** alpha / w_max ** (1 - alpha), clamp[0], clamp[1]).astype(np.float32)
            if scales == "pow2":
                sc = pow2_scales(sc, clamp)
            off = _norm_offset(bb, nk)
            ops[nk] = ("div_offset" if off else "div", sc)
            for c in ck:
                ops[c] = ("mul_in", sc)
            if not overlay:
                nw = bb.src.read(nk)
                new[nk] = ((1.0 + nw) / sc - 1.0) if off else nw / sc
                for c in ck:
                    new[c] = Ws[c] * sc[None, :]
            folded += 1
            if float(a_max.max()) > worst[1]:
                worst = (i, float(a_max.max()))
        if writer is not None:
            for k in bb.layer_keys[i]:
                if k in new:
                    S.write_float(bb.src, writer, k, arr=new[k])
                else:
                    S.copy_raw(bb.src, writer, k)
        bb.free_layer(i)
        eng.hidden = outs                     # the fold is an identity: the block's outputs are unchanged
        log(f"   block {i + 1}/{bb.n_layers}: {len(seams)} seam(s)")
    summary = {"model": os.path.abspath(model), "alpha": alpha, "scales": scales, "seams": folded,
               "worst_outlier": {"layer": worst[0], "max_abs_x": round(worst[1], 2)}, "rows": len(data),
               "tokens": sum(len(r) for r in data), "backbone": bb.describe(), "memory_plan": plan,
               "peak_rss_gb": round(eng.peak_rss / 1e9, 2), "minutes": round((time.time() - t0) / 60, 2),
               "layout": "overlay" if overlay else "checkpoint"}
    os.makedirs(out, exist_ok=True)
    if overlay:
        S._st_write(os.path.join(out, S.OVERLAY_ST), {k: ("F32", v[1]) for k, v in sorted(ops.items())})
        base = os.path.abspath(bb.src.base_dir)
        prev = bb.src.overlay
        if prev:                            # an overlay over an overlay: compose into one over the base
            raise SystemExit(f"{model} is already a smoothing overlay; smooth its base ({base}) instead")
        json.dump({"format": "pollard-smooth-overlay/1", "base": base, "ops": {k: v[0] for k, v in ops.items()},
                   **{k: v for k, v in summary.items() if k != "backbone"}},
                  open(os.path.join(out, S.OVERLAY_JSON), "w"), indent=1)
        S.copy_side_files(base, out)
    else:
        for k in bb.outer_keys + bb.other_keys:
            S.copy_raw(bb.src, writer, k)
        writer.close()
        S.copy_side_files_of(bb.src, out)
        _drop_quantized_entries(out, [k[:-len(".weight")] for k in ops if bb.src.encoding(k)])
    json.dump(summary, open(os.path.join(out, "pollard-smooth-report.json"), "w"), indent=1)
    return summary


def _drop_quantized_entries(out, modules):
    """A smoothed tensor that was encoded in the source is written as bf16: its quantization_config entry goes."""
    if not modules:
        return
    p = os.path.join(out, "config.json")
    cfg = json.load(open(p, encoding="utf-8"))
    q = cfg.get("quantization_config") or {}
    ql = q.get("quantized_layers") or {}
    for m in modules:
        ql.pop(m, None)
    json.dump(cfg, open(p, "w", encoding="utf-8"), indent=2)


def materialize(overlay_dir, out, log=print):
    """A full smoothed checkpoint from an overlay (for a consumer that reads files itself: exllamav3, GGUF convert)."""
    import pollard_stream as S
    src = S.Source(overlay_dir)
    if not src.overlay:
        raise SystemExit(f"{overlay_dir} is not a smoothing overlay")
    w = S.ShardWriter(out)
    for k in src.logical:
        S.copy_raw(src, w, k)
    w.close()
    S.copy_side_files_of(src, out)
    _drop_quantized_entries(out, [k[:-len(".weight")] for k in src.overlay if src.encoding(k)])
    open(os.path.join(out, ".pollard-materialized"), "w").write(os.path.abspath(overlay_dir))
    log(f"   materialized {overlay_dir} -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="source HF model dir/id (fp16/bf16, or an already-quantized "
                    "checkpoint the stream reader decodes: modelopt NVFP4 / FP8, ggml-packed, compressed-tensors NVFP4)")
    ap.add_argument("--out", help="output dir for the smoothed HF model (default: workspace)")
    ap.add_argument("--calib", help="calibration text file (else a bland built-in string; give real text)")
    ap.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant migration strength [0,1]")
    ap.add_argument("--clamp", type=float, nargs=2, default=(1e-2, 1e2), help="min/max per-channel scale")
    ap.add_argument("--scales", default="pow2", choices=["pow2", "exact"],
                    help="pow2 (default): power-of-two scales, so the fold is exact in bf16 -- the smoothed checkpoint "
                         "computes what the source does; exact: the raw SmoothQuant ratio (rounds every folded weight)")
    ap.add_argument("--rows", type=int, default=12, help="calibration chunks")
    ap.add_argument("--cols", type=int, default=512, help="calibration chunk length (tokens)")
    ap.add_argument("--device", default="auto", help="auto (CUDA when present) | cpu | cuda[:N]")
    ap.add_argument("--trust-remote-code", default="auto", choices=["auto", "on", "off"],
                    help="run the checkpoint's own modelling code when no native transformers class covers it. "
                         "Never silent: 'auto' asks on a terminal and refuses elsewhere (naming the files)")
    ap.add_argument("--overlay", default="off", choices=["off", "on", "auto"],
                    help="on: write only the scales over the source (megabytes; the streamed lanes read it as the "
                         "smoothed model). auto: an overlay when a full copy would not fit the free disk")
    ap.add_argument("--max-ram-gb", type=float, help="cap the run's RAM (default: 85%% of what is free)")
    ap.add_argument("--materialize", metavar="OVERLAY", help="write a full checkpoint (--out) from an overlay")
    a = ap.parse_args()

    if a.materialize:
        materialize(a.materialize, a.out or (a.materialize.rstrip("/\\") + "-full"))
        return
    import pollard_stream as S
    dev = S.stream_device(a.device)                     # before torch: an explicit cpu hides the shared GPU
    import pollard_workspace as ws
    if not a.out:
        a.out = os.path.join(ws.model_dir(a.model, create=True), ws.model_basename(a.model) + "-smoothed")
        print(f"   (no --out) -> workspace: {a.out}")
    overlay = a.overlay == "on"
    if a.overlay == "auto":
        import shutil
        need = sum(os.path.getsize(p) for p in __import__("pollard_stream").weight_files(a.model)) * 1.02
        parent = os.path.dirname(os.path.abspath(a.out)) or "."
        while not os.path.exists(parent):
            parent = os.path.dirname(parent)
        free = shutil.disk_usage(parent).free
        margin = float(os.environ.get("POLLARD_DISK_MARGIN_GB", 5)) * 1e9
        overlay = need + margin > free
        print(f"   overlay auto: a full copy needs {need / 1e9:.1f} GB, {free / 1e9:.1f} GB free -> "
              + ("overlay (scales only)" if overlay else "full smoothed checkpoint"))
    print(f"== pollard-hf-smooth :: {a.model}  alpha={a.alpha}  device={dev}")
    sm = smooth_stream(a.model, a.out, a.calib, a.alpha, tuple(a.clamp), a.rows, a.cols, dev, a.trust_remote_code,
                       overlay, a.max_ram_gb, scales=a.scales)
    print(f"   folded {sm['seams']} seams; worst input-activation outlier: layer {sm['worst_outlier']['layer']}  "
          f"max|X|={sm['worst_outlier']['max_abs_x']:.1f}  (smoothed)")
    print(f"   peak RSS {sm['peak_rss_gb']} GB (plan {sm['memory_plan']['peak_gb']} GB), {sm['minutes']} min")
    print(f"wrote smoothed HF model ({sm['layout']}) -> {a.out}\n  then:  pollard-exl3 --model {a.out} --out <exl3> --bpw 4.0"
          f"\n  VERIFY: pollard-verify --model <exl3> --source {a.model} --end-to-end")


if __name__ == "__main__":
    main()
