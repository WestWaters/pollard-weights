#!/usr/bin/env python3
"""pollard-hf-smooth — activation-aware SmoothQuant preconditioning for an HF model, IN PLACE of the
weights, folded into the RMSNorms as a mathematical identity. Run it BEFORE a low-bit convert (EXL3 /
GPTQ / MX lanes) so massive-activation input channels can't wreck the quantization.

Why it exists: low-bit trellis/GPTQ quant has NO input-outlier protection. A single massive-activation
input channel (e.g. the residual-stream outlier that emerges in the first couple of layers) collapses the
quantizer's global scale, so every NORMAL channel gets under-quantized and the layer forwards to garbage —
while per-tensor and Hessian-proxy metrics still look perfect (see legacy/PROXY_ERR_BANNED.md). Measured
case: Qwen2.5-3B layer 1 MLP-input activation max 453 → 4bpw layer sqnr -14 (PPL 3090); after this
preconditioning → sqnr +18, sane PPL, no bits spent.

The fix (SmoothQuant, per input channel j, exact/identity):
    s_j = max|X_j|^alpha / max|W_j|^(1-alpha)          (clamped)
    norm.weight[j] /= s_j ;  W[:, j] *= s_j   for every consumer linear W on that seam
X activations come from a calibration forward; the fold into RMSNorm.weight is exact (RMS is over the
pre-weight input), so the model computes the same thing — the quantizer just sees a flatter input.

Seams (norm -> linears), auto-detected per architecture:
    A: input_layernorm         -> q_proj, k_proj, v_proj
    B: post_attention_layernorm -> gate_proj, up_proj

  pollard-hf-smooth --model ./Qwen2.5-3B --out ./Qwen2.5-3B-sm --calib cal.txt
  pollard-exl3 --model ./Qwen2.5-3B-sm --out ./M-exl3 --bpw 4.0        # then convert the smoothed model

Verify every build with pollard-verify (never trust a proxy metric).
"""
import argparse, os, sys


def _load_calib(path, tok, device, cols, rows):
    import torch
    text = open(path, encoding="utf-8", errors="ignore").read() if path else (
        "The quick brown fox jumps over the lazy dog. " * 4000)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    chunks = [ids[:, a:a + cols] for a in range(0, ids.shape[1] - cols, cols)][:rows]
    return chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="source HF model dir/id (fp16/bf16)")
    ap.add_argument("--out", help="output dir for the smoothed HF model (default: workspace)")
    ap.add_argument("--calib", help="calibration text file (else a bland built-in string; give real text)")
    ap.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant migration strength [0,1]")
    ap.add_argument("--clamp", type=float, nargs=2, default=(1e-2, 1e2), help="min/max per-channel scale")
    ap.add_argument("--rows", type=int, default=12, help="calibration chunks")
    ap.add_argument("--cols", type=int, default=512, help="calibration chunk length (tokens)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map=a.device)
    model.eval()
    try:
        layers = model.model.layers
    except AttributeError:
        sys.exit("ERROR: expected a decoder with model.model.layers (Llama/Qwen-style). "
                 "Add the seam map for this architecture.")

    # seam consumers, per layer, guarded (fused-QKV / MLA / missing modules simply skip)
    def seams(L):
        s = []
        an = getattr(L, "input_layernorm", None)
        attn = getattr(L, "self_attn", None)
        if an is not None and attn is not None:
            cons = [getattr(attn, n, None) for n in ("q_proj", "k_proj", "v_proj")]
            cons = [c for c in cons if c is not None]
            if cons: s.append(("A", an, attn.q_proj if attn.q_proj is not None else cons[0], cons))
        pn = getattr(L, "post_attention_layernorm", None)
        mlp = getattr(L, "mlp", None)
        if pn is not None and mlp is not None:
            cons = [getattr(mlp, n, None) for n in ("gate_proj", "up_proj")]
            cons = [c for c in cons if c is not None]
            if cons: s.append(("B", pn, cons[-1], cons))
        return s

    # capture per-input-channel abs-max at each seam's representative consumer input
    amax = {}
    def mk_hook(key):
        def hook(mod, inp, out):
            x = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).amax(dim=0).float()
            amax[key] = torch.maximum(amax[key], x) if key in amax else x
        return hook
    hooks = []
    for i, L in enumerate(layers):
        for tag, norm, probe, cons in seams(L):
            hooks.append(probe.register_forward_hook(mk_hook((i, tag))))

    with torch.no_grad():
        for ch in _load_calib(a.calib, tok, a.device, a.cols, a.rows):
            model(ch)
    for h in hooks: h.remove()

    folded = 0
    worst = (0, 0.0)
    for i, L in enumerate(layers):
        for tag, norm, probe, cons in seams(L):
            key = (i, tag)
            if key not in amax:
                continue
            w_max = torch.stack([c.weight.data.abs().amax(dim=0) for c in cons]).amax(dim=0).clamp_min(1e-8)
            a_max = amax[key].to(w_max.device).clamp_min(1e-8)
            if a_max.shape != w_max.shape:              # dimension mismatch -> skip, never corrupt
                continue
            s = (a_max.pow(a.alpha) / w_max.pow(1 - a.alpha)).clamp(a.clamp[0], a.clamp[1])
            norm.weight.data /= s.to(norm.weight.dtype)
            for c in cons:
                c.weight.data *= s.to(c.weight.dtype).unsqueeze(0)
            folded += 1
            if a_max.max().item() > worst[1]:
                worst = (i, a_max.max().item())

    print(f"== pollard-hf-smooth :: {a.model}  alpha={a.alpha}  folded {folded} seams")
    print(f"   worst input-activation outlier: layer {worst[0]}  max|X|={worst[1]:.1f}  (smoothed)")
    if not a.out:
        import pollard_workspace as ws
        a.out = os.path.join(ws.model_dir(a.model, create=True), ws.model_basename(a.model) + "-smoothed")
        print(f"   (no --out) -> workspace: {a.out}")
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    print(f"wrote smoothed HF model -> {a.out}\n  then:  pollard-exl3 --model {a.out} --out <exl3> --bpw 4.0"
          f"\n  VERIFY: pollard-verify --model <exl3> --source {a.model} --end-to-end")


if __name__ == "__main__":
    main()
