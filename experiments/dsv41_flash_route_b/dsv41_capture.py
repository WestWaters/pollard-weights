#!/usr/bin/env python3
"""dsv41_capture.py — streaming bf16 forward of DeepSeek-V4.1-Flash one layer at a time on ONE DGX Spark GB10 (128 GB unified),
with the Pollard-method captures our GLM-5.3 / Hy4 kits wrote: (1) SmoothQuant seam statistics — per-channel |x| max at the output
of attn_norm (feeds wq_a, wkv, the compressor, the indexer's weights_proj) and of ffn_norm (feeds the router, the routed experts, the
shared expert) — (2) routing concentration — per-expert pick count and routing-weight mass — and (3) extra seams / hyper-connection /
attention-sink / CSA2-indexer data, plus optional per-matrix input Hessians (X^T X, fp32) for GPTQ. Output format is the one our
GLM-5.3 / Hy4 captures wrote (smooth_L##.json / route_L##.json / extra_L##.json), so the Pollard-side analysis and fold_smooth reuse it.

Model code = DeepSeek's official reference `inference/model.py` (+ engram.py), imported by path. transformers has no deepseek_v41, and
the reference is the only complete description of: interleaved-complex RoPE on the trailing 64 dims with the conjugate rotation on the
attention output, the sqrt(softplus) top-6 router with selection bias and route_scale 1.5, SwiGLU clamped at 10, hyper-connections
(hc_mult 4) with Sinkhorn, attention sinks in the softmax denominator, CSA2 (sliding window 128 + compressed KV; Full/Reindex/Reuse
layers via one shared runtime object), the hierarchical candidate pool from layer 20, and Engram n-gram tables at layers 1 and 14.
The reference's TileLang kernels (kernel.py) are replaced by pure-torch equivalents in `_make_kernel_shim()` below — every shim cites
the kernel.py lines it stands in for; see CAPTURE_PORT_NOTES.md.

Weights: the bf16 checkpoint produced by dsv41_dequant.py (same tensor names as the reference: layers.N.attn.wq_a.weight,
layers.N.ffn.experts.E.w1.weight, embed.weight, head.weight; no `.scale` tensors except the Engram tables, which stay fp8 e4m3
[rows, 256] + ue8m0 [rows, 8] in shards 47/48). Shards are pulled per layer over HTTP (byte-range server), kept in a small on-disk
stage and released once no upcoming layer needs them. The two Engram tables (~94 GiB each) are NEVER loaded whole: the hash ids of the
calibration tokens are computed once (reference engram.py hashing, needs the tokenizer) and only the rows they address are gathered
by HTTP byte-range reads (row r of layers.L.engram.embed.weight = header_base + data_offset + r*256; scale row at +r*8; value =
e4m3 * 2**(ue8m0 - 127) per 32-block). Run inside the vLLM image (torch 2.9+, safetensors, numpy, tokenizers); no tilelang needed.

usage:
  dsv41_capture.py --http http://STORE_HOST:8767/bf16 --tok-http http://STORE_HOST:8767/fp4 --stage /dsv41/stage \
      --tokens /dsv41/calib_384x2048.safetensors --rows 384 --batch 4 --layers 0-39 --out /dsv41/stats \
      [--hessians /dsv41/hess --hessian-expert-mode hessian|dump|skip --hessian-expert-chunks 4] [--smoke] [--selftest]
  split across two nodes: --row-start 0 --rows 192 (sp1) / --row-start 192 --rows 192 (sp2); stats and Hessians are additive.
"""
import argparse, dataclasses, gc, glob, importlib, json, math, os, re, struct, sys, threading, time, types, urllib.request
import concurrent.futures as cf
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--http", default="http://STORE_HOST:8767/bf16", help="base URL of the read-only bf16 shard server (byte ranges)")
ap.add_argument("--tok-http", default="http://STORE_HOST:8767/fp4", help="base URL holding tokenizer.json + tokenizer_config.json")
ap.add_argument("--ref-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference"),
                help="DeepSeek reference inference/ dir (model.py, engram.py, config.json)")
ap.add_argument("--meta", default="", help="local dir with the bf16 config.json + model.safetensors.index.json (default: fetch into stage/meta)")
ap.add_argument("--stage", required=True); ap.add_argument("--tokens", default="")
ap.add_argument("--rows", type=int, default=384); ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--row-start", type=int, default=0, help="first calibration row (split rows across nodes; stats/Hessians are additive)")
ap.add_argument("--layers", default="0-39"); ap.add_argument("--out", required=True)
ap.add_argument("--stage-budget-gb", type=float, default=100.0, help="warn threshold on staged shard bytes (bf16 layer shard ~28 GB)")
ap.add_argument("--smoke", action="store_true", help="random tokens, 2 rows x 256, tiny run for plumbing checks")
ap.add_argument("--selftest", action="store_true", help="CPU-only checks of the kernel shims against naive formulas, then exit")
ap.add_argument("--state-out", default="", help="write streams + pre_mix + CSA2 runtime after the last layer (resume point), safetensors")
ap.add_argument("--state-in", default="", help="resume: state produced by an earlier run ending at --layers start-1")
ap.add_argument("--hessians", default="", help="DIR: accumulate per-matrix input Hessians (X^T X fp32) per layer, safetensors")
ap.add_argument("--hessian-expert-mode", default="hessian", choices=["hessian", "dump", "skip"],
                help="hessian: per-expert X^T X (48 GB/layer on disk, forward repeated --hessian-expert-chunks times); "
                     "dump: write the ffn input + routing per layer (8 GB/layer) for on-demand expert Hessians; skip: dense only")
ap.add_argument("--hessian-expert-chunks", type=int, default=4, help="expert Hessians held in memory per pass (384/chunks experts)")
ap.add_argument("--attn-chunk", type=int, default=128, help="query positions per sparse-attention chunk (memory knob)")
ap.add_argument("--engram-fetch", default="auto", choices=["auto", "stream", "ranges"],
                help="how to gather Engram rows: stream the whole fp8 table once and pick rows, or issue coalesced byte-range reads")
ap.add_argument("--engram-http", default="", help="base URL for the Engram table shards (47/48); default = --http. The tables are copied verbatim by the upscale, so the fp4 dir is byte-identical and usable while the bf16 dir is still being written")
ap.add_argument("--engram-hashes-only", action="store_true", help="compute + cache the Engram hash ids for this row range, then exit (rows are then extracted on the store host by dsv41_engram_extract.py into the same cache-file format)")
ap.add_argument("--engram-threads", type=int, default=8); ap.add_argument("--engram-chunk-mb", type=int, default=64)
ap.add_argument("--engram-fq", default="none", choices=["none", "mxfp4", "nvfp4"],
                help="fake-quantize the dequantized Engram rows to fp4 (mxfp4: e2m1 + per-32 ue8m0 power-of-two scale; nvfp4: e2m1 + per-16 e4m3 scale) to measure the NLL cost of an fp4-resident Engram table")
ap.add_argument("--exl3-recipe", default="", help="recipe json (layers.L.gu/down = K): replace the routed experts of every layer by the EXL3 reconstruction from --exl3-k3/--exl3-k4gu/--exl3-k4w2 (pre-splice NLL gate of a recipe)")
ap.add_argument("--exl3-k3", default=""); ap.add_argument("--exl3-k4gu", default=""); ap.add_argument("--exl3-k4w2", default="")
ap.add_argument("--tokenizer-dir", default="", help="local dir with tokenizer.json (default: fetch from --tok-http into stage/meta/tok)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); os.makedirs(a.stage, exist_ok=True)
if a.hessians: os.makedirs(a.hessians, exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# =====================================================================================================================
# kernel shims — pure torch stand-ins for inference/kernel.py (TileLang). Registered as the `kernel` module BEFORE model.py is
# imported, so `from kernel import act_quant, fp4_act_quant, fp4_gemm, fp8_gemm, hc_split_sinkhorn, sparse_attn` binds to these.
# =====================================================================================================================
SHIM = {"attn_chunk": a.attn_chunk, "last_o": None}
FP8_MAX, FP4_MAX = 448.0, 6.0

def _pow2_ceil(v):
    """2**ceil(log2(v)) for v > 0 (fp32), = kernel.py fast_round_scale (l.22-37: exponent bits + (mantissa != 0))."""
    mant, exp = torch.frexp(v)                                   # v = mant * 2**exp, mant in [0.5, 1)
    e = exp - (mant == 0.5).to(exp.dtype)                        # exact power of two -> its own exponent
    return torch.ldexp(torch.ones_like(v), e)

def _e2m1_round(v):
    """Round |x|/s in [0, 6] to the nearest e2m1 magnitude {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to even mantissa, = the hardware
    fp32->fp4 cast in kernel.py l.171 (T.Cast(FP4, ...) lowers to cvt.rn). Step 0.5 below 2, 1 in [2, 4), 2 in [4, 6]."""
    r1 = torch.round(v * 2) / 2; r2 = torch.round(v); r3 = torch.round(v / 2) * 2
    return torch.where(v < 2, r1, torch.where(v < 4, r2, r3))

def _make_kernel_shim():
    m = types.ModuleType("kernel")

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        """kernel.py act_quant (l.98-124) + act_quant_kernel (l.41-95): per-(row, block_size) fp8 e4m3 quantization. With scale_fmt
        set the scale is rounded up to a power of two (ue8m0); inplace=True writes the dequantized values back (window K path)."""
        N = x.size(-1); assert N % block_size == 0
        xf = x.float().unflatten(-1, (-1, block_size))
        amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)                                 # l.76
        inv = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32, device=x.device)                # fp8_max_inv as an fp32 constant
        s = _pow2_ceil(amax * inv) if scale_fmt is not None else amax * inv                    # l.77-80
        q = (xf / s).clamp(-FP8_MAX, FP8_MAX)                                                  # l.85 / l.89
        if inplace:
            x.copy_((q.to(torch.float8_e4m3fn).float() * s).flatten(-2).to(x.dtype)); return x  # l.85, l.121-123
        try: s_out = s.squeeze(-1).to(scale_dtype)
        except Exception: s_out = s.squeeze(-1)
        return q.to(torch.float8_e4m3fn).flatten(-2), s_out

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        """kernel.py fp4_act_quant (l.184-204) + fp4_quant_kernel (l.128-181): e2m1 per block with ue8m0 (power-of-two) scales for the
        indexer q/k, or e4m3 scales (block 16) for the compressed KV. inplace=True writes the dequantized values back."""
        assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
        N = x.size(-1); assert N % block_size == 0
        xf = x.float().unflatten(-1, (-1, block_size))
        amax = xf.abs().amax(-1, keepdim=True)
        if scale_dtype == torch.float8_e4m3fn:                                                 # l.160-163 (compressed KV)
            amax = amax.clamp_min(6 * 2.0 ** -9)
            s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
        else:                                                                                  # l.164-166 (indexer)
            amax = amax.clamp_min(6 * 2.0 ** -126)
            s = _pow2_ceil(amax * torch.tensor(1.0 / FP4_MAX, dtype=torch.float32, device=x.device))
        q = (xf / s).clamp(-FP4_MAX, FP4_MAX)                                                  # l.171 / l.175
        deq = torch.copysign(_e2m1_round(q.abs()), q) * s
        if inplace:
            x.copy_(deq.flatten(-2).to(x.dtype)); return x                                     # l.201-203
        return deq.flatten(-2).to(x.dtype), s.squeeze(-1)   # packed fp4 storage is not needed on the bf16 path

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        """kernel.py sparse_attn (l.392-403) + sparse_attn_kernel (l.311-389): per query, gather its top-k KV rows (-1 = empty slot ->
        zero row, -inf score), fp32 scores from bf16 q/kv, running max floored at -1e30, P cast to bf16 for the PV product, and the
        learnable sink joins the softmax denominator only (exp(sink - max)). Chunked over query positions to bound memory."""
        b, s, h, d = q.shape
        o = torch.empty_like(q); SHIM["last_o"] = o
        sink = attn_sink.float().view(1, 1, h)
        step = SHIM["attn_chunk"]
        for c0 in range(0, s, step):
            idx = topk_idxs[:, c0:c0 + step].long(); mc, t = idx.shape[1], idx.shape[2]
            valid = idx >= 0                                                                   # l.360-364
            g = kv.gather(1, idx.clamp_min(0).reshape(b, -1, 1).expand(-1, -1, d)).view(b, mc, t, d).float()
            sc = torch.einsum("bmhd,bmtd->bmht", q[:, c0:c0 + step].float(), g) * softmax_scale   # l.365-367 (fp32 accumulate)
            sc = sc.masked_fill(~valid.unsqueeze(2), float("-inf"))
            mx = sc.amax(-1).clamp_min(-1e30)                                                  # l.355, l.369
            p = torch.exp(sc - mx.unsqueeze(-1))                                               # l.373
            denom = p.sum(-1) + torch.exp(sink - mx)                                           # l.374-376, l.382-383
            pv = torch.einsum("bmht,bmtd->bmhd", p.to(torch.bfloat16).float(), g)              # l.377-380 (P -> bf16 before PV)
            o[:, c0:c0 + step] = (pv / denom.unsqueeze(-1)).to(o.dtype)                        # l.384-387
        return o

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        """kernel.py hc_split_sinkhorn (l.465-474) + hc_split_sinkhorn_kernel (l.407-462): split the hc projection into
        pre (sigmoid + eps), post (2*sigmoid) and comb (row softmax + eps, then alternating column/row normalisation)."""
        hc = hc_mult
        pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps                          # l.427
        post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])               # l.429
        comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))            # l.431 (row-major j*hc+k)
        comb = comb.softmax(-1) + eps                                                                    # l.437-443
        comb = comb / (comb.sum(-2, keepdim=True) + eps)                                                 # l.446-448
        for _ in range(sinkhorn_iters - 1):                                                             # l.450-458
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    def _dequant_fp8_blocks(w, ws, block):
        N, K = w.shape; sf = ws.float(); wf = w.float()
        wf = F.pad(wf, (0, sf.shape[1] * block - K, 0, sf.shape[0] * block - N))
        wf = (wf.unflatten(0, (-1, block)).unflatten(-1, (-1, block)) * sf[:, None, :, None]).flatten(2, 3).flatten(0, 1)
        return wf[:N, :K]

    def fp8_gemm(a_, a_s, b_, b_s, scale_dtype=torch.float32, block_size=128):
        """kernel.py fp8_gemm (l.277-307): C = (A * sa) @ (B * sb)^T. Dequantize both operands and use an fp32 matmul. NOT reached on
        the bf16 checkpoint (model.linear only calls it for fp8 weights); kept so the module is complete."""
        K = a_.size(-1)
        af = (a_.float().unflatten(-1, (-1, block_size)) * a_s.float().unsqueeze(-1)).flatten(-2)
        return (af @ _dequant_fp8_blocks(b_, b_s, block_size).T).to(torch.get_default_dtype())

    FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

    def fp4_gemm(a_, a_s, b_, b_s, scale_dtype=torch.float32, act_block_size=128):
        """kernel.py fp4_gemm (l.562-591): C = (A * sa) @ (B_fp4 * sb)^T; B packed two e2m1 per byte along K (low nibble = even k),
        scales per 32. Dequantize and matmul. NOT reached on the bf16 checkpoint."""
        K = a_.size(-1)
        af = (a_.float().unflatten(-1, (-1, act_block_size)) * a_s.float().unsqueeze(-1)).flatten(-2)
        packed = b_.view(torch.uint8); lut = FP4_LUT.to(b_.device)
        vals = torch.stack([lut[(packed & 0x0F).long()], lut[(packed >> 4).long()]], dim=-1).flatten(1)
        bf = (vals.unflatten(-1, (-1, 32)) * b_s.float()[:, :, None]).flatten(1)
        return (af @ bf.T).to(torch.get_default_dtype())

    m.act_quant, m.fp4_act_quant, m.sparse_attn, m.hc_split_sinkhorn, m.fp8_gemm, m.fp4_gemm = (
        act_quant, fp4_act_quant, sparse_attn, hc_split_sinkhorn, fp8_gemm, fp4_gemm)
    return m

def _sympy_stub():
    """engram.py imports sympy.isprime (torch depends on sympy, so this is normally present); deterministic Miller-Rabin fallback."""
    m = types.ModuleType("sympy")
    def isprime(n):
        if n < 2: return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if n % p == 0: return n == p
        d, r = n - 1, 0
        while d % 2 == 0: d //= 2; r += 1
        for a_ in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            x = pow(a_, d, n)
            if x in (1, n - 1): continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1: break
            else: return False
        return True
    m.isprime = isprime; return m

def _image_processor_stub():
    m = types.ModuleType("image_processor"); m.IMAGE_START, m.IMAGE, m.IMAGE_NEW_LINE, m.IMAGE_END = range(4); return m   # image_processor.py l.23

def _vision_stub():
    m = types.ModuleType("vision")
    class _NoVision:
        def __init__(self, *args, **kw): raise RuntimeError("vision tower is not used by the capture")
    m.ViT = m.Aligner = _NoVision; return m

def import_reference(ref_dir):
    sys.path.insert(0, ref_dir)
    sys.modules["kernel"] = _make_kernel_shim()                      # must precede `import model`
    try: importlib.import_module("sympy")
    except ImportError: sys.modules["sympy"] = _sympy_stub()
    for name, stub in (("image_processor", _image_processor_stub), ("vision", _vision_stub)):
        try: importlib.import_module(name)                           # real files if present (image_processor needs PIL)
        except Exception: sys.modules[name] = stub()
    if not hasattr(torch, "float4_e2m1fn_x2"):
        torch.float4_e2m1fn_x2 = None   # model.py l.186/219 compare weight dtypes against it; None never equals a real dtype
    dsm = importlib.import_module("model"); dse = importlib.import_module("engram")
    dsm.default_dtype = torch.bfloat16                               # Transformer.__init__ (l.1191) would set this from args.dtype
    _pfc = dsm.precompute_freqs_cis.__wrapped__                      # l.368-389, lru_cache(2): under the meta build it would cache META tensors
    def _pfc_cpu(*args_):
        with torch.device("cpu"): return _pfc(*args_)
    dsm.precompute_freqs_cis = _pfc_cpu                              # Attention.__init__ (l.688) resolves the module global at call time
    return dsm, dse

dsm, dse = import_reference(a.ref_dir)
kern = sys.modules["kernel"]

# =====================================================================================================================
# instrumented Indexer.forward: verbatim model.py l.527-580 plus the lines marked `## capture` (top-k / candidate-pool statistics).
# =====================================================================================================================
IDX = {"on": False, "topk_valid": 0.0, "topk_slots": 0, "free_in_pool": 0.0, "free_valid": 0.0, "pool_reach": 0.0, "reach": 0.0}
def _indexer_forward_instrumented(self, x, qr, latent, start_pos, offset):
    assert self.freqs_cis is not None
    bsz, seqlen, _ = x.size()
    ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen
    if self.owns_k and latent is not None:                                                                  # l.537-548
        freqs = (self.freqs_cis[: seqlen - seqlen % ratio : ratio] if start_pos == 0 else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0))
        k = self.k_norm(self.wk(latent))
        dsm.apply_rotary_emb(k[..., -rd:], freqs)
        dsm.fp4_act_quant(k, dsm.fp4_block_size, True)
        self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
        dsm.shared_attn.index_k = self.k_cache
    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))                              # l.550-552
    dsm.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
    dsm.fp4_act_quant(q, dsm.fp4_block_size, True)
    index_k = dsm.shared_attn.index_k[:bsz, : end_pos // ratio]                                             # l.554-557
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
    index_score = torch.einsum("bshd,btd->bsht", q, index_k)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    if start_pos == 0:                                                                                      # l.563-567
        compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
        index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
    else:
        compress_lens = end_pos // ratio
    topk = min(self.index_topk, end_pos // ratio)                                                           # l.578
    if self.is_candidate_source:                                                                            # l.569-572
        dsm.shared_attn.candidates = dsm.select_candidate_blocks(index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size)
    elif self.uses_candidates:                                                                              # l.573-575
        if IDX["on"]:  ## capture: what this layer would pick WITHOUT the pool, and how much of it the pool keeps
            free = index_score.topk(topk, dim=-1, sorted=False).indices
            fvalid = free < compress_lens
            IDX["free_in_pool"] += (dsm.shared_attn.candidates.gather(-1, free) & fvalid).sum().item(); IDX["free_valid"] += fvalid.sum().item()
        index_score = index_score.masked_fill(~dsm.shared_attn.candidates, -torch.inf)
    if IDX["on"] and (self.is_candidate_source or self.uses_candidates):  ## capture: pool coverage of the reachable positions
        reach = torch.arange(index_score.size(-1), device=x.device) < compress_lens
        IDX["pool_reach"] += (dsm.shared_attn.candidates & reach).sum().item(); IDX["reach"] += reach.sum().item()
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values                         # l.579-580
    out = torch.where(idxs < compress_lens, idxs + offset, -1).int()
    if IDX["on"]:  ## capture: fraction of top-k slots that hold a reachable position
        IDX["topk_valid"] += (out >= 0).sum().item(); IDX["topk_slots"] += out.numel()
    return out
dsm.Indexer.forward = _indexer_forward_instrumented

# =====================================================================================================================
# self-test of the shims (CPU, random data) — run before the first GPU launch
# =====================================================================================================================
if a.selftest:
    torch.manual_seed(0)
    # sinkhorn: comb doubly stochastic, pre in (eps, 1+eps), post in (0, 2)
    mixes = torch.randn(3, 5, 24); pre, post, comb = kern.hc_split_sinkhorn(mixes, torch.tensor([1.0, 0.5, 2.0]), torch.randn(24), 4, 20, 1e-6)
    # reference ends on a COLUMN normalisation (kernel.py l.456-458): columns sum to 1 exactly, rows only approximately after 20 iters
    assert torch.allclose(comb.sum(-2), torch.ones(3, 5, 4), atol=1e-4), "sinkhorn: columns must sum to 1"
    assert (comb.sum(-1) - 1).abs().max() < 0.2 and (comb > 0).all(), "sinkhorn: rows far from 1 / non-positive"
    ref = torch.softmax((mixes[..., 8:] * 2.0 + torch.zeros(16)).unflatten(-1, (4, 4)), -1) + 1e-6  # spot-check first iterate shape/order
    assert ref.shape == comb.shape
    assert (pre > 1e-6).all() and (pre < 1 + 1e-6).all() and (post > 0).all() and (post < 2).all()
    # sparse attention vs dense causal softmax with a sink column
    b, s, h, d = 2, 16, 4, 32; q = torch.randn(b, s, h, d).bfloat16(); kv = torch.randn(b, s, d).bfloat16(); sink = torch.randn(h)
    idx = torch.arange(s).view(1, 1, s).expand(b, s, s); idx = torch.where(idx <= torch.arange(s).view(1, s, 1), idx, -1).int()
    o = kern.sparse_attn(q, kv, sink, idx, d ** -0.5).float()
    sc = torch.einsum("bshd,btd->bsht", q.float(), kv.float()) * d ** -0.5
    sc = sc.masked_fill(torch.arange(s).view(1, 1, 1, s) > torch.arange(s).view(1, s, 1, 1), float("-inf"))
    sc = torch.cat([sc, sink.view(1, 1, h, 1).expand(b, s, h, 1)], -1); p = sc.softmax(-1)[..., :s]
    o_ref = torch.einsum("bsht,btd->bshd", p, kv.float()); assert (o - o_ref).abs().max() < 5e-2, (o - o_ref).abs().max()
    # act_quant / fp4_act_quant: outputs on the grid, bounded error
    x = torch.randn(4, 512).bfloat16() * 3; y = kern.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
    assert ((y.float() - x.float()).abs() <= x.float().abs() / 8 + 1e-3).all()
    z = kern.fp4_act_quant(x.clone(), 32, True); amax = x.float().abs().unflatten(-1, (-1, 32)).amax(-1, keepdim=True)
    assert ((z.float() - x.float()).abs().unflatten(-1, (-1, 32)) <= amax / 3 + 2e-2).all()   # pow2 scale <= 2*amax/6, widest e2m1 step 2
    z2 = kern.fp4_act_quant(x.clone(), 16, True, scale_dtype=torch.float8_e4m3fn); assert torch.isfinite(z2.float()).all()
    v = torch.rand(1000) * 100 + 1e-3; want = torch.tensor([2.0 ** math.ceil(math.log2(t)) for t in v.tolist()])
    got = _pow2_ceil(v); assert torch.equal(got, want) or ((got - want).abs() / want).max() < 1e-6
    assert torch.equal(_e2m1_round(torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.9])), torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]))
    log("SELFTEST-OK"); sys.exit(0)

# =====================================================================================================================
# meta: config + index (+ tokenizer) — fetched from the shard server when no local --meta is given
# =====================================================================================================================
def http_get(url, dst, retries=6):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dst + ".part", "wb") as o:
                while True:
                    buf = r.read(16 << 20)
                    if not buf: break
                    o.write(buf)
            os.replace(dst + ".part", dst); return dst
        except Exception as e:
            log(f"  GET {url}: {type(e).__name__} {e} (attempt {attempt})"); time.sleep(5)
    raise RuntimeError(f"could not fetch {url}")

meta_dir = a.meta or os.path.join(a.stage, "meta"); os.makedirs(meta_dir, exist_ok=True)
for fn in ("config.json", "model.safetensors.index.json"):
    if not os.path.exists(os.path.join(meta_dir, fn)): http_get(f"{a.http}/{fn}", os.path.join(meta_dir, fn))
cfg = json.load(open(os.path.join(meta_dir, "config.json")))
if "dim" not in cfg:  # HF-style config (nested text_config) → use the flat reference config shipped with inference/ (same values)
    ref_cfg = os.path.join(a.ref_dir, "config.json"); log(f"config.json is HF-style; using the reference {ref_cfg}"); cfg = json.load(open(ref_cfg))
_fields = {f.name for f in dataclasses.fields(dsm.ModelArgs)}
margs = dsm.ModelArgs(**{k: v for k, v in cfg.items() if k in _fields})
margs.dtype = "bf16"; margs.expert_dtype = None   # bf16 body: Linear/Expert build bf16 weights, model.linear -> F.linear (l.207)
L, D, HC, E, V = margs.n_layers, margs.dim, margs.hc_mult, margs.n_routed_experts, margs.vocab_size
dev = torch.device("cuda"); torch.cuda.set_device(0)
layout = dse.EngramLayout.from_args(margs)
ENGRAM_LAYERS = tuple(layout.layer_ids) if layout is not None else ()

# =====================================================================================================================
# shard staging (hy4_capture.py structure) + HTTP byte-range helpers for the huge Engram shards
# =====================================================================================================================
def canon(k):
    """convert.py l.105-115 name rules, idempotent on already-canonical (reference-style) names."""
    if k.startswith("model."): k = k[len("model."):]
    k = k.replace("self_attn", "attn")
    if not k.startswith("vision."): k = k.replace("mlp", "ffn")
    return k.replace("weight_scale_inv", "scale").replace("e_score_correction_bias", "bias")
_wm_raw = json.load(open(os.path.join(meta_dir, "model.safetensors.index.json")))["weight_map"]
wm = {canon(k): (k, s) for k, s in _wm_raw.items()}                         # canonical name -> (checkpoint name, shard)
HUGE = {s for k, (_, s) in wm.items() if ".engram.embed." in k}              # shards holding a 94 GiB table: never staged
def tensors_for(prefix): return sorted(k for k in wm if k.startswith(prefix) and ".engram.embed." not in k)
def shards_for(prefix): return sorted({wm[k][1] for k in tensors_for(prefix)} - HUGE)
SIZES, HEADERS = {}, {}
def remote_size(shard):
    if shard not in SIZES:
        url = f"{a.http}/{shard}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r: SIZES[shard] = int(r.headers["Content-Length"])
        except Exception:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60) as r:
                SIZES[shard] = int(r.headers["Content-Range"].split("/")[-1])
    return SIZES[shard]
def http_range(url, start, end, retries=8):
    """bytes [start, end) of url."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"}), timeout=120) as r:
                buf = r.read()
            if len(buf) == end - start: return buf
            log(f"  range {url} {start}-{end}: short {len(buf)} (attempt {attempt})")
        except Exception as e:
            log(f"  range {url} {start}-{end}: {type(e).__name__} {e} (attempt {attempt})"); time.sleep(3 + 3 * attempt)
    raise RuntimeError(f"range read failed {url} {start}-{end}")
def remote_header(shard, base_url=None):
    if shard not in HEADERS:
        url = f"{base_url or a.http}/{shard}"; n = struct.unpack("<Q", http_range(url, 0, 8))[0]
        HEADERS[shard] = (json.loads(http_range(url, 8, 8 + n)), 8 + n)
    return HEADERS[shard]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_dequant import dequant_fp8_block as _deq_fp8, dequant_fp4 as _deq_fp4   # exact upscale, same math as the store-host bf16 artifact
ST_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.uint8, "F8_E8M0": torch.uint8, "U8": torch.uint8,
         "I8": torch.int8, "I32": torch.int32, "I64": torch.int64, "BOOL": torch.bool}
def remote_tensor(shard, ckpt_name):
    hdr, base = remote_header(shard); info = hdr[ckpt_name]; s0, e0 = info["data_offsets"]
    t = torch.frombuffer(bytearray(http_range(f"{a.http}/{shard}", base + s0, base + e0)), dtype=ST_DT[info["dtype"]])
    if info["dtype"] == "F8_E4M3": t = t.view(torch.float8_e4m3fn)
    return t.reshape(info["shape"])
def staged_bytes(): return sum(os.path.getsize(p) for p in glob.glob(os.path.join(a.stage, "*.safetensors")))
def fetch(shard):
    dst = os.path.join(a.stage, shard); want = remote_size(shard)
    if os.path.exists(dst) and os.path.getsize(dst) == want: return dst
    part = dst + ".part"; have = os.path.getsize(part) if os.path.exists(part) else 0
    for attempt in range(6):
        try:
            req = urllib.request.Request(f"{a.http}/{shard}", headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as o:
                while True:
                    buf = r.read(16 << 20)
                    if not buf: break
                    o.write(buf); have += len(buf)
            if have == want: os.replace(part, dst); return dst
            log(f"  fetch {shard}: short ({have}/{want}), retrying")
        except Exception as e:
            log(f"  fetch {shard}: {type(e).__name__} {e} (attempt {attempt}) — retrying"); time.sleep(10)
    raise RuntimeError(f"could not fetch {shard}")
def prefetch(prefix):
    def run():
        try:
            for s in shards_for(prefix): fetch(s)
        except Exception as e: log(f"  prefetch {prefix}: {e}")
    t = threading.Thread(target=run, daemon=True); t.start(); return t
def release(keep_prefixes):
    keep = set(); [keep.update(shards_for(p)) for p in keep_prefixes]
    for p in glob.glob(os.path.join(a.stage, "*.safetensors")):
        if os.path.basename(p) not in keep: os.remove(p)
def load_into(module, prefix, ignore_missing=()):
    """Copy every checkpoint tensor under `prefix` straight into the module's parameters (dtype cast on copy, no GPU double buffer).
    Tensors in the huge Engram shards are range-read; everything else comes from the staged shard."""
    params = dict(module.named_parameters()); got, unexpected, bad = set(), [], []
    by_shard = {}
    for k in tensors_for(prefix): by_shard.setdefault(wm[k][1], []).append(k)
    def assign(k, t):
        local = k[len(prefix):]; p = params.get(local)
        if p is None: unexpected.append(local); return
        if tuple(p.shape) != tuple(t.shape): bad.append((local, tuple(p.shape), tuple(t.shape))); return
        p.data.copy_(t); got.add(local)
    def maybe_dequant(k, t, getter):
        """fp8 (e4m3 x 2^(ue8m0-127), 32x32 blocks) and packed MXFP4 (I8 [N, K/2] + ue8m0 [N, K/32]) → bf16, exactly as dsv41_dequant.py."""
        if t.dtype not in (torch.float8_e4m3fn, torch.int8, torch.uint8): return t   # bf16/fp32 source: nothing to do (and no scale to read)
        sk = k[: -len(".weight")] + ".scale" if k.endswith(".weight") else None
        if sk is None or sk not in wm: return t
        sc = getter(sk).view(torch.uint8).cpu(); t = t.cpu()
        with torch.device("cpu"):   # never let a CUDA default-device context pull the dequant temporaries onto the GPU
            if t.dtype == torch.float8_e4m3fn: return _deq_fp8(t, sc)
            if t.dtype in (torch.int8, torch.uint8): return _deq_fp4(t.view(torch.uint8), sc)
        return t
    for shard, keys in by_shard.items():
        if shard in HUGE:
            for k in keys:
                if k.endswith(".scale"): got.add(k[len(prefix):]); continue
                assign(k, maybe_dequant(k, remote_tensor(shard, wm[k][0]), lambda kk: remote_tensor(wm[kk][1], wm[kk][0])))
        else:
            with safe_open(fetch(shard), "pt", device="cpu") as f:
                def _get(kk):
                    ssh_, sck = wm[kk][1], wm[kk][0]
                    if ssh_ == shard: return f.get_tensor(sck)
                    with safe_open(fetch(ssh_), "pt", device="cpu") as f2: return f2.get_tensor(sck)
                for k in keys:
                    if k.endswith(".scale"): got.add(k[len(prefix):]); continue
                    assign(k, maybe_dequant(k, f.get_tensor(wm[k][0]), _get))
    torch.cuda.synchronize()
    missing = [m for m in params if m not in got and not any(m.startswith(i) for i in ignore_missing)]
    if bad: raise SystemExit(f"{prefix} shape mismatch: {bad[:6]}")
    if missing: raise SystemExit(f"{prefix} missing tensors: {missing[:8]}")
    unexpected = [u for u in unexpected if not u.endswith(".scale")]  # scales are consumed by the on-the-fly dequant
    return unexpected

# =====================================================================================================================
# inputs, streams, CSA2 runtime store
# =====================================================================================================================
lo, hi = [int(x) for x in a.layers.split("-")]; hi = min(hi, L - 1)
if a.smoke:
    R, S = 2, 256; ids = torch.randint(1000, 100000, (R, S), dtype=torch.int64)
else:
    if not a.tokens: raise SystemExit("--tokens is required (or --smoke)")
    ids = load_file(a.tokens)["input_ids"][a.row_start: a.row_start + a.rows].to(torch.int64); R, S = ids.shape
    log(f"calibration rows {a.row_start}..{a.row_start + R - 1}")
B = min(a.batch, R); margs.max_batch_size = B; margs.max_seq_len = S    # sizes the reference's per-layer caches to exactly our batch
log(f"rows {R} × {S} tokens, batch {B}, layers {lo}-{hi}, hc_mult {HC}, experts {E}, engram layers {ENGRAM_LAYERS}")
est = R * S * HC * D * 2 / 1e9
log(f"memory plan: streams {est:.1f} GB bf16 + CSA2 runtime {R * S * (margs.head_dim * 2 + margs.index_head_dim * 2 + min(margs.index_topk, S) * 4 + S) / 1e9:.1f} GB "
    f"+ one bf16 layer ~27.5 GB + activations (batch {B}) ~{B * 2.5:.0f} GB"
    + (f" + expert Hessians {E // a.hessian_expert_chunks * (D * D + margs.moe_inter_dim ** 2) * 4 / 1e9:.0f} GB per pass" if a.hessians and a.hessian_expert_mode == "hessian" else ""))

class RuntimeStore:
    """Per-row copies of what model.SharedAttentionRuntime carries between layers (l.1166-1180). The reference runs all layers for one
    batch and keeps one slot each; we run one layer for all rows, so the slots are kept per row and swapped in per batch."""
    def __init__(self, R, S):
        cand_w = S // max(margs.compress_ratios[margs.candidate_source_layer], 1) if margs.candidate_source_layer >= 0 else 1
        self.ckv = torch.zeros(R, S, margs.head_dim, dtype=torch.bfloat16, device=dev)        # compress_kv (fp4-dequantized latents)
        self.ik = torch.zeros(R, S, margs.index_head_dim, dtype=torch.bfloat16, device=dev)   # index_k
        self.topk = torch.full((R, S, min(margs.index_topk, S)), -1, dtype=torch.int32, device=dev)
        self.cand = torch.zeros(R, S, cand_w, dtype=torch.bool, device=dev)
    def tensors(self): return {"rt_compress_kv": self.ckv, "rt_index_k": self.ik, "rt_topk": self.topk, "rt_candidates": self.cand.to(torch.uint8)}
store = RuntimeStore(R, S)

if a.state_in:
    st = load_file(a.state_in); streams = st["streams"].to(dev); pre_mix = st["pre_mix"].to(dev)
    store.ckv.copy_(st["rt_compress_kv"]); store.ik.copy_(st["rt_index_k"]); store.topk.copy_(st["rt_topk"]); store.cand.copy_(st["rt_candidates"].bool())
    del st; log(f"resumed streams {tuple(streams.shape)} + runtime from {a.state_in}")
else:
    if lo != 0 and not a.smoke: raise SystemExit("--layers must start at 0 unless --state-in is given")
    emb = torch.empty(V, D, dtype=torch.bfloat16, device=dev)
    class _Emb(torch.nn.Module):
        def __init__(s): super().__init__(); s.weight = torch.nn.Parameter(emb, requires_grad=False)
    load_into(_Emb(), "embed.")
    streams = torch.empty(R, S, HC, D, device=dev, dtype=torch.bfloat16)
    for r in range(R): streams[r] = F.embedding(ids[r].to(dev), emb).unsqueeze(1).expand(-1, HC, -1)   # Transformer.forward l.1253, l.1258
    pre_mix = dsm.make_identity_pre_mix(streams, HC)                                                    # l.1260: one-hot on stream 0
    del emb; torch.cuda.empty_cache()
    log(f"streams {tuple(streams.shape)} = {streams.numel() * 2 / 1e9:.1f} GB on GPU")
release([f"layers.{lo}.", f"layers.{lo + 1}."])

# =====================================================================================================================
# Engram: hash ids for the calibration tokens (reference hashing) + byte-range gather of only the addressed table rows
# =====================================================================================================================
def load_tokenizer():
    tok_dir = a.tokenizer_dir or os.path.join(meta_dir, "tok"); os.makedirs(tok_dir, exist_ok=True)
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        if not os.path.exists(os.path.join(tok_dir, fn)): http_get(f"{a.tok_http}/{fn}", os.path.join(tok_dir, fn))
    try:
        from tokenizers import Tokenizer
        class _Tok:  # the two things engram.build_compressed_token_map uses (engram.py l.43-47): backend_tokenizer and len()
            def __init__(s, p): s.backend_tokenizer = Tokenizer.from_file(p)
            def __len__(s): return s.backend_tokenizer.get_vocab_size(with_added_tokens=True)
        return _Tok(os.path.join(tok_dir, "tokenizer.json")), tok_dir
    except ImportError:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tok_dir), tok_dir

def compute_engram_hashes():
    """[R, S, n_engram_layers, n_hash_cols] int64 via engram.NgramHashState (l.129-184), cached per row range."""
    p = os.path.join(a.out, f"engram_hashes_r{a.row_start}_{R}{'_smoke' if a.smoke else ''}.safetensors")
    if os.path.exists(p): return load_file(p)["hashes"]
    tok, tok_dir = load_tokenizer()
    hargs = dataclasses.replace(margs, max_batch_size=R)
    try: hs = dse.NgramHashState(hargs, layout, tok)
    except AssertionError as e:
        log(f"  compressed-vocab mismatch with the tokenizers wrapper ({e}); retrying with transformers AutoTokenizer")
        from transformers import AutoTokenizer; hs = dse.NgramHashState(hargs, layout, AutoTokenizer.from_pretrained(tok_dir))
    hashes = hs(ids, 0, None).contiguous(); save_file({"hashes": hashes}, p); return hashes

def gather_engram_rows(layer_id, ids_layer):
    """Unique row ids addressed by this layer's hashes -> (uniq int64, rows uint8 [U, 256], scales uint8 [U, 8]) from the fp8 table.
    Row r of the weight tensor is at header_base + data_offset + r*256, scale row at +r*8 (Tony DiskEngramTable / safetensors layout)."""
    uniq = torch.unique(ids_layer.reshape(-1)).contiguous()
    p = os.path.join(a.out, f"engram_rows_L{layer_id:02d}_r{a.row_start}_{R}{'_smoke' if a.smoke else ''}.safetensors")
    if os.path.exists(p):
        c = load_file(p)
        if torch.equal(c["uniq"], uniq): log(f"  engram L{layer_id}: {uniq.numel()} rows from cache {p}"); return uniq, c["rows"], c["scales"]
    wname, sname = f"layers.{layer_id}.engram.embed.weight", f"layers.{layer_id}.engram.embed.scale"
    (wck, wsh), (sck, ssh) = wm[wname], wm[sname]
    plans = []
    for shard, ck in ((wsh, wck), (ssh, sck)):
        hdr, base = remote_header(shard, base_url=(a.engram_http or a.http)); info = hdr[ck]; rows_n, rb = info["shape"]; s0, e0 = info["data_offsets"]
        assert (e0 - s0) == rows_n * rb and info["dtype"] in ("F8_E4M3", "F8_E8M0", "U8"), (ck, info)
        plans.append((f"{a.engram_http or a.http}/{shard}", base + s0, rows_n, rb))
    U = uniq.numel(); un = uniq.numpy(); out_w = np.empty((U, plans[0][3]), np.uint8); out_s = np.empty((U, plans[1][3]), np.uint8)
    # cluster the sorted ids for range reads: gap <= 256 rows (64 KB of dead bytes) joins, span <= 4096 rows per request
    clusters, c0 = [], 0
    for j in range(1, U + 1):
        if j == U or un[j] - un[j - 1] > 256 or un[j] - un[c0] > 4096: clusters.append((c0, j)); c0 = j
    range_bytes = sum((un[j - 1] - un[i] + 1) for i, j in clusters) * (plans[0][3] + plans[1][3])
    table_bytes = sum(rn * rb for _, _, rn, rb in plans)
    mode = a.engram_fetch if a.engram_fetch != "auto" else ("ranges" if (len(clusters) <= 150000 and range_bytes < 0.3 * table_bytes) else "stream")
    log(f"  engram L{layer_id}: {U} unique rows of {plans[0][2]} | ranges: {len(clusters)} requests / {range_bytes / 1e9:.1f} GB vs stream {table_bytes / 1e9:.1f} GB -> {mode}")
    t0 = time.time()
    for (url, start, rows_n, rb), out in ((plans[0], out_w), (plans[1], out_s)):
        if mode == "ranges":
            def one(cl):
                i, j = cl; r0, r1 = int(un[i]), int(un[j - 1]) + 1
                buf = np.frombuffer(http_range(url, start + r0 * rb, start + r1 * rb), np.uint8).reshape(-1, rb)
                out[i:j] = buf[un[i:j] - r0]
            with cf.ThreadPoolExecutor(max_workers=a.engram_threads) as ex: list(ex.map(one, clusters))
        else:
            chunk_rows = max(1, (a.engram_chunk_mb << 20) // rb); pos = 0; nxt_log = 0
            while pos < rows_n:
                try:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={start + pos * rb}-{start + rows_n * rb - 1}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        pend = b""
                        while pos < rows_n:
                            need = min(chunk_rows, rows_n - pos) * rb
                            while len(pend) < need:
                                buf = r.read(min(16 << 20, need - len(pend)))
                                if not buf: raise IOError("stream ended early")
                                pend += buf
                            blk = np.frombuffer(pend[:need], np.uint8).reshape(-1, rb); pend = pend[need:]
                            i, j = np.searchsorted(un, pos), np.searchsorted(un, pos + blk.shape[0])
                            if j > i: out[i:j] = blk[un[i:j] - pos]
                            pos += blk.shape[0]
                            if pos * rb >= nxt_log: log(f"    stream {url.rsplit('/', 1)[-1]} {pos * rb / 1e9:.0f}/{rows_n * rb / 1e9:.0f} GB, {j}/{U} rows"); nxt_log += 8 << 30
                except Exception as e:
                    log(f"    stream interrupted at row {pos}: {type(e).__name__} {e}; resuming"); time.sleep(5)
    rows_t, scales_t = torch.from_numpy(out_w), torch.from_numpy(out_s)
    save_file({"uniq": uniq, "rows": rows_t, "scales": scales_t}, p)
    log(f"  engram L{layer_id}: gathered {U} rows ({(out_w.nbytes + out_s.nbytes) / 1e9:.2f} GB) in {time.time() - t0:.0f}s -> {p}")
    return uniq, rows_t, scales_t

class GatheredEngramTable(torch.nn.Module):
    """Stand-in for model.ParallelEngramEmbedding (l.296-325, world_size 1) that holds only the gathered rows; same dequant math:
    values.float().unflatten(-1, (-1, 32)) * 2**(ue8m0 - 127), flattened, to bf16."""
    def __init__(self, uniq, rows, scales, block_size=32):
        super().__init__(); self.block_size = block_size
        self.register_buffer("uniq", uniq.to(dev)); self.register_buffer("rows", rows.to(dev)); self.register_buffer("scales", scales.to(dev))
    def forward(self, indices):
        local = torch.searchsorted(self.uniq, indices)
        vals = self.rows[local].view(torch.float8_e4m3fn).float().unflatten(-1, (-1, self.block_size))
        sc = torch.ldexp(torch.ones(self.scales.shape[-1], device=indices.device), (self.scales[local].int() - 127))
        out = (vals * sc.unsqueeze(-1)).flatten(-2)
        if a.engram_fq != "none": out = fq_fp4(out, a.engram_fq)
        return out.to(torch.bfloat16)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
def _to_e2m1(u):
    """round-to-nearest onto the e2m1 magnitude grid (ties to even as in the OCP MX / NVFP4 conversions); u = |v|/scale, expected <= 6."""
    g = E2M1.to(u.device); idx = torch.bucketize(u, g)                         # g[idx-1] <= u < g[idx]
    lo = g[(idx - 1).clamp(0, 7)]; hi = g[idx.clamp(0, 7)]
    mid = (lo + hi) / 2; pick_hi = (u > mid) | ((u == mid) & ((idx % 2) == 0))   # ties → even index
    return torch.where(pick_hi, hi, lo)
def fq_fp4(x, mode):
    """fake fp4 on the last dim of a float tensor: blocks of 32 (mxfp4, scale = 2**(floor(log2 amax) - 2), ue8m0) or 16 (nvfp4, scale = amax/6
    rounded to e4m3 under a per-table fp32 scale). Element = sign * e2m1(|v|/scale) * scale. Values > 6*scale saturate to 6 (as the hardware does)."""
    bs = 32 if mode == "mxfp4" else 16
    xb = x.float().unflatten(-1, (-1, bs)); amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    if mode == "mxfp4":
        sc = torch.exp2(torch.floor(torch.log2(amax)) - 2)                     # OCP MX shared exponent: floor(log2 amax) - emax(e2m1)=2
    else:
        gs = (x.abs().amax() / (6.0 * 448.0)).clamp_min(1e-30)                  # per-tensor fp32 scale so block scales fit e4m3 (<= 448)
        sc = (amax / 6.0 / gs).to(torch.float8_e4m3fn).float() * gs
    q = _to_e2m1((xb.abs() / sc).clamp(max=6.0)) * sc * torch.sign(xb)
    return q.flatten(-2)

engram_needed = [l for l in ENGRAM_LAYERS if lo <= l <= hi]
hashes, engram_tables = None, {}
if engram_needed:
    hashes = compute_engram_hashes().to(dev); log(f"engram hashes {tuple(hashes.shape)} int64")
    if a.engram_hashes_only:
        for l in ENGRAM_LAYERS:
            u = torch.unique(hashes[:, :, layout.layer_ids.index(l), :].reshape(-1)); log(f"  L{l}: {u.numel()} unique rows")
        log("DSV41-ENGRAM-HASHES-DONE"); raise SystemExit(0)
    for l in engram_needed:
        uniq, rows, scales = gather_engram_rows(l, hashes[:, :, layout.layer_ids.index(l), :].cpu())
        engram_tables[l] = GatheredEngramTable(uniq, rows, scales, dsm.fp8_block_size)

# =====================================================================================================================
# per-layer helpers
# =====================================================================================================================
def csa2_mode(i):
    r = margs.compress_ratios[i]
    if r == 0: return "SWA-only"
    if i in margs.kv_source_layers: return "Full"
    if i in margs.index_source_layers: return "Reindex"
    return "Reuse"
def fix_freqs(attn):
    """Attention.__init__ l.680-698 recomputed on the real device: to_empty() left the freqs_cis buffer uninitialised."""
    if attn.compress_ratio: osl, theta = margs.original_seq_len, margs.compress_rope_theta
    else: osl, theta = 0, margs.rope_theta
    fc = dsm.precompute_freqs_cis(attn.rope_head_dim, margs.max_seq_len, osl, theta, margs.rope_factor, margs.beta_fast, margs.beta_slow)
    attn.freqs_cis = fc.to(dev)
    if attn.indexer is not None: attn.indexer.freqs_cis = attn.freqs_cis
def build_block(i):
    with dsm.set_dtype(torch.bfloat16), torch.device("meta"): blk = dsm.Block(i, margs, layout)
    if blk.engram is not None: blk.engram.embed = torch.nn.Module()   # the 384M-row table is replaced by the gathered rows after to_empty
    blk.to_empty(device=dev); fix_freqs(blk.attn)
    if blk.engram is not None: blk.engram.embed = engram_tables[i]
    return blk
def category(n):
    if n.startswith("ffn.experts."): return "experts"
    if n.startswith("ffn.shared_experts."): return "shared_experts"
    if n.startswith("ffn.gate."): return "router"
    if n.startswith("hc_"): return "hc"
    if n.startswith("attn.indexer."): return "indexer"
    if n.startswith("attn.compressor."): return "compressor"
    if n.startswith("attn."): return "attention"
    if n.startswith("engram."): return "engram"
    return "norm"
@torch.no_grad()
def weight_stats(blk):
    """Per-tensor rms / absmax / outlier fraction (allocator inputs) and a byte inventory; the 384x3 expert matrices are summarised per
    matrix type with per-expert rms (chunked: one expert at a time). no_grad is essential: Parameters carry requires_grad, and without it
    the fp32 copies of 1,152 expert matrices stay alive in the autograd graph (~110 GB) — this OOM-reset sp8 on 09-10."""
    wst, inv, groups = {}, {}, {}
    for n, p in blk.named_parameters():
        if n.startswith("engram.embed."): continue
        inv[category(n)] = inv.get(category(n), 0) + p.numel() * p.element_size()
        m = re.match(r"ffn\.experts\.(\d+)\.(w[123])\.weight$", n)
        if m: groups.setdefault(m.group(2), []).append(p); continue
        if p.dim() >= 2 and p.numel() >= 4096:
            f = p.float(); rms = f.pow(2).mean().sqrt().item(); mx = f.abs().max().item()
            wst[n] = {"shape": list(p.shape), "dtype": str(p.dtype).replace("torch.", ""), "rms": rms, "absmax": mx,
                      "absmax_over_rms": mx / (rms + 1e-12), "frac_gt_6rms": (f.abs() > 6 * rms).float().mean().item()}
            del f
    for w, lst in groups.items():
        sq = torch.stack([t.float().pow(2).sum() for t in lst]); mx = torch.stack([t.float().abs().max() for t in lst])
        cnt = sum(t.numel() for t in lst); rms = (sq.sum() / cnt).sqrt().item(); per = (sq / lst[0].numel()).sqrt()
        n_out = sum((t.float().abs() > 6 * rms).sum() for t in lst).item()
        wst[f"ffn.experts.*.{w}.weight"] = {"shape": [len(lst), *lst[0].shape], "dtype": str(lst[0].dtype).replace("torch.", ""), "rms": rms,
                                            "absmax": mx.max().item(), "absmax_over_rms": mx.max().item() / (rms + 1e-12), "frac_gt_6rms": n_out / cnt,
                                            "expert_rms_min_med_max": [per.min().item(), per.median().item(), per.max().item()],
                                            "per_expert_rms": per.tolist(), "per_expert_absmax": mx.tolist()}
    return wst, inv

# =====================================================================================================================
# layer loop
# =====================================================================================================================
torch.set_grad_enabled(False)   # the kit never trains; keeps every stats pass and shim from building autograd graphs
HESS = bool(a.hessians)
DUMP_X = None
if HESS and a.hessian_expert_mode == "dump":
    DUMP_X = torch.empty(R * S, D, dtype=torch.bfloat16); DUMP_I = torch.empty(R * S, margs.n_activated_experts, dtype=torch.int16)
    DUMP_W = torch.empty(R * S, margs.n_activated_experts, dtype=torch.bfloat16)
def exl3_reconstruct(out_tensors):
    """(in, out) fp32 from trellis/suh/svh — same recipe as dsv41_exl3_experts.reconstruct / exl3_kernel_truth.py (needs the exllamav3 ext)."""
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant.exl3_lib.quantize import preapply_had_l, preapply_had_r, had_k, had_n
    tr = out_tensors["trellis"]; k, n, K = tr.shape[0] * 16, tr.shape[1] * 16, tr.shape[2] // 16
    w = torch.empty((k, n), dtype=torch.half, device=tr.device); ext.reconstruct(w, tr.contiguous(), K, False, True)
    w = preapply_had_l(w, had_k); w *= out_tensors["suh"].to(w.device).unsqueeze(1); w = preapply_had_r(w, had_n); w *= out_tensors["svh"].to(w.device).unsqueeze(0)
    return w.float()

def apply_exl3_experts(blk, layer):
    """Overwrite blk.ffn.experts[e].{w1,w3,w2}.weight with the EXL3 reconstruction chosen by the recipe (gu → K from --exl3-k3/--exl3-k4gu,
    down → --exl3-k3/--exl3-k4w2). Logs the weight-space relative error per matrix type."""
    from safetensors import safe_open
    rec = json.load(open(a.exl3_recipe)).get("layers", {}).get(str(layer), {"gu": 3, "down": 3})
    src = {"gu": a.exl3_k3 if int(rec["gu"]) == 3 else a.exl3_k4gu, "down": a.exl3_k3 if int(rec["down"]) == 3 else a.exl3_k4w2}
    err = {"w1": [], "w3": [], "w2": []}; t0 = time.time(); n_done = 0
    for kind, mats in (("gu", ("w1", "w3")), ("down", ("w2",))):
        path = os.path.join(src[kind], f"exl3_L{layer:02d}.safetensors")
        if not os.path.exists(path): raise SystemExit(f"L{layer}: exl3 source missing for {kind} K={rec[kind]}: {path}")
        with safe_open(path, "pt", device=str(dev)) as f:
            keys = set(f.keys())
            for e in range(E):
                for m in mats:
                    base = f"layers.{layer}.ffn.experts.{e}.{m}"
                    if f"{base}.trellis" not in keys: raise SystemExit(f"{path}: missing {base}.trellis")
                    W = exl3_reconstruct({t: f.get_tensor(f"{base}.{t}") for t in ("trellis", "suh", "svh")})     # (in, out)
                    p = getattr(blk.ffn.experts[e], m).weight                                                     # [out, in] bf16
                    ref = p.data.float(); q = W.T
                    err[m].append(((q - ref).norm() / (ref.norm() + 1e-9)).item()); p.data.copy_(q.to(p.dtype)); n_done += 1
                    del W, ref, q
    log(f"  L{layer:02d} EXL3 experts applied (gu K={rec['gu']} down K={rec['down']}): {n_done} tensors in {time.time() - t0:.0f}s; weight rel err "
        + " ".join(f"{m} {sum(v) / max(len(v), 1):.4f}" for m, v in err.items()))

for i in range(lo, hi + 1):
    t0 = time.time(); pre = f"layers.{i}."
    blk = build_block(i)
    unexpected = load_into(blk, pre, ignore_missing=("engram.embed.",))
    if a.exl3_recipe: apply_exl3_experts(blk, i)
    if unexpected: log(f"  L{i} unexpected checkpoint tensors ignored: {unexpected[:6]}")
    pf = prefetch(f"layers.{i + 1}.") if i + 1 <= hi else None
    wst, inv_bytes = weight_stats(blk); fetch_s = time.time() - t0
    blk.eval(); attn = blk.attn; ratio = attn.compress_ratio; mode = csa2_mode(i)
    tk = min(margs.index_topk, S // ratio) if ratio else 0
    has_cand = margs.candidate_source_layer >= 0 and i > margs.candidate_source_layer and ratio > 0
    ST = {"on": True}
    # ---- statistics accumulators
    sm = {"attn_in": torch.zeros(D, device=dev), "mlp_in": torch.zeros(D, device=dev), "tokens": 0}
    ex = {"q_norm": torch.zeros(margs.q_lora_rank, device=dev), "kv_norm": torch.zeros(margs.head_dim, device=dev),
          "ckv_norm": torch.zeros(margs.head_dim, device=dev), "idx_k_norm": torch.zeros(margs.index_head_dim, device=dev),
          "hc_in_pre": torch.zeros(HC, device=dev, dtype=torch.float64), "hc_attn_post": torch.zeros(HC, device=dev, dtype=torch.float64),
          "hc_ffn_pre": torch.zeros(HC, device=dev, dtype=torch.float64), "hc_ffn_post": torch.zeros(HC, device=dev, dtype=torch.float64),
          "hc_next_pre": torch.zeros(HC, device=dev, dtype=torch.float64), "hc_attn_comb_diag": torch.zeros(HC, device=dev, dtype=torch.float64),
          "hc_ffn_comb_diag": torch.zeros(HC, device=dev, dtype=torch.float64), "hc_n": 0,
          "eng_delta_sq": torch.zeros(HC, device=dev, dtype=torch.float64), "eng_value_sq": 0.0, "eng_n": 0}
    sal = {"count": torch.zeros(E, device=dev, dtype=torch.float64), "wmass": torch.zeros(E, device=dev, dtype=torch.float64), "tokens": 0}
    for k in IDX: IDX[k] = 0.0 if k != "on" else False
    def _amax_into(dst, t): dst.copy_(torch.maximum(dst, t.detach().abs().amax(dim=tuple(range(t.dim() - 1))).float()))
    # ---- Hessians (dense + shared; experts per chunk below)
    HS, HN = {}, {}
    def hess_alloc(name, K, lead=None):
        HS[name] = torch.zeros((lead, K, K) if lead else (K, K), dtype=torch.float32, device=dev); HN[name] = 0
    def hess_add(name, x):
        xf = x.detach().reshape(-1, x.shape[-1]).float(); HS[name].addmm_(xf.T, xf); HN[name] += xf.shape[0]
    if HESS:
        G = margs.o_groups
        hess_alloc("attn_in", D); hess_alloc("q_norm_out", margs.q_lora_rank); hess_alloc("wo_a_in", margs.n_heads * margs.head_dim // G, lead=G)
        hess_alloc("wo_b_in", G * margs.o_lora_rank); hess_alloc("ffn_in", D); hess_alloc("shared_w2_in", margs.moe_inter_dim)
        if attn.indexer is not None and attn.indexer.owns_k: hess_alloc("indexer_wk_in", margs.head_dim)
        if blk.engram is not None: hess_alloc("engram_wkv_in", blk.engram.wkv.in_features)
    # ---- hooks (module seams)
    def _h_attn_norm(mod, inp, out):
        if not ST["on"]: return
        _amax_into(sm["attn_in"], out); sm["tokens"] += out.shape[0] * out.shape[1]
        if HESS: hess_add("attn_in", out)
    def _h_ffn_norm(mod, inp, out):
        if not ST["on"]: return
        _amax_into(sm["mlp_in"], out)
        if HESS: hess_add("ffn_in", out)
        if DUMP_X is not None: DUMP_X[ST["tok0"]:ST["tok0"] + out.shape[0] * out.shape[1]].copy_(out.reshape(-1, D))
    def _h_q_norm(mod, inp, out):
        if not ST["on"]: return
        _amax_into(ex["q_norm"], out)
        if HESS: hess_add("q_norm_out", out)
    def _h_kv_norm(mod, inp, out):
        if ST["on"]: _amax_into(ex["kv_norm"], out)
    def _h_ckv_norm(mod, inp, out):
        if ST["on"]: _amax_into(ex["ckv_norm"], out)
    def _h_idxk_norm(mod, inp, out):
        if ST["on"]: _amax_into(ex["idx_k_norm"], out)
    def _h_wk_pre(mod, inp):
        if ST["on"] and HESS: hess_add("indexer_wk_in", inp[0])
    def _h_wo_b_pre(mod, inp):
        if not (ST["on"] and HESS): return
        hess_add("wo_b_in", inp[0])
        o = SHIM["last_o"]                                          # sparse_attn output after the in-place conjugate RoPE (l.781), = wo_a input (l.785-787)
        og = o.reshape(-1, margs.o_groups, o.shape[-1] * o.shape[-2] // margs.o_groups).float()
        HS["wo_a_in"].baddbmm_(og.transpose(0, 1).transpose(1, 2), og.transpose(0, 1)); HN["wo_a_in"] += og.shape[0]
    def _h_gate(mod, inp, out):
        if not ST["on"]: return
        w, ix = out; ixf = ix.reshape(-1); wf = w.reshape(-1).double()
        sal["count"].index_add_(0, ixf, torch.ones_like(wf)); sal["wmass"].index_add_(0, ixf, wf); sal["tokens"] += ix.shape[0]
        if DUMP_X is not None:
            DUMP_I[ST["tok0"]:ST["tok0"] + ix.shape[0]].copy_(ix.to(torch.int16)); DUMP_W[ST["tok0"]:ST["tok0"] + ix.shape[0]].copy_(w.to(torch.bfloat16))
    def _h_shared_w2_pre(mod, inp):
        if ST["on"] and HESS: hess_add("shared_w2_in", inp[0])
    def _h_engram_wkv_pre(mod, inp):
        if ST["on"] and HESS: hess_add("engram_wkv_in", inp[0])
    def _h_engram_wkv(mod, inp, out):
        if ST["on"]: ex["eng_value_sq"] += out[..., -D:].detach().float().pow(2).sum().item(); ex["eng_n"] += out.shape[0] * out.shape[1]
    def _h_engram(mod, inp, out):
        if ST["on"]: ex["eng_delta_sq"] += (out.detach().float() - inp[0].float()).pow(2).sum(dim=(0, 1, 3)).double()
    hooks = [blk.attn_norm.register_forward_hook(_h_attn_norm), blk.ffn_norm.register_forward_hook(_h_ffn_norm),
             attn.q_norm.register_forward_hook(_h_q_norm), attn.kv_norm.register_forward_hook(_h_kv_norm),
             attn.wo_b.register_forward_pre_hook(_h_wo_b_pre), blk.ffn.gate.register_forward_hook(_h_gate),
             blk.ffn.shared_experts.w2.register_forward_pre_hook(_h_shared_w2_pre)]
    if attn.compressor is not None: hooks.append(attn.compressor.norm.register_forward_hook(_h_ckv_norm))
    if attn.indexer is not None and attn.indexer.owns_k:
        hooks += [attn.indexer.k_norm.register_forward_hook(_h_idxk_norm), attn.indexer.wk.register_forward_pre_hook(_h_wk_pre)]
    if blk.engram is not None:
        hooks += [blk.engram.wkv.register_forward_pre_hook(_h_engram_wkv_pre), blk.engram.wkv.register_forward_hook(_h_engram_wkv),
                  blk.engram.register_forward_hook(_h_engram)]
    # hyper-connection coefficients: Block.hc_mixes (l.948-955) is a method, wrap it. The attn call yields (pre for THIS block's ffn,
    # post for the attention output, comb); the ffn call yields (pre for the NEXT layer's attention, post for the ffn output, comb).
    _orig_mixes = blk.hc_mixes
    def _hc_mixes(x, hc_fn, hc_scale, hc_base):
        pre_, post_, comb_ = _orig_mixes(x, hc_fn, hc_scale, hc_base)
        if ST["on"]:
            if hc_fn is blk.hc_attn_fn:
                ex["hc_ffn_pre"] += pre_.double().sum((0, 1)); ex["hc_attn_post"] += post_.double().sum((0, 1))
                ex["hc_attn_comb_diag"] += comb_.diagonal(dim1=-2, dim2=-1).double().sum((0, 1)); ex["hc_n"] += pre_.shape[0] * pre_.shape[1]
            else:
                ex["hc_next_pre"] += pre_.double().sum((0, 1)); ex["hc_ffn_post"] += post_.double().sum((0, 1))
                ex["hc_ffn_comb_diag"] += comb_.diagonal(dim1=-2, dim2=-1).double().sum((0, 1))
        return pre_, post_, comb_
    blk.hc_mixes = _hc_mixes
    # ---- passes: 1 normally; --hessian-expert-mode hessian repeats the forward per expert chunk (memory bound: 12 GB per 96 experts)
    n_pass = a.hessian_expert_chunks if (HESS and a.hessian_expert_mode == "hessian") else 1
    chunks = list(np.array_split(np.arange(E), n_pass)) if (HESS and a.hessian_expert_mode == "hessian") else []
    sa = dsm.shared_attn
    for p_ in range(n_pass):
        final = p_ == n_pass - 1; ST["on"] = final; IDX["on"] = final
        ehooks = []
        if chunks:
            ch = chunks[p_]; I_ = margs.moe_inter_dim
            EH13 = torch.zeros(len(ch), D, D, dtype=torch.float32, device=dev); EH2 = torch.zeros(len(ch), I_, I_, dtype=torch.float32, device=dev)
            EN = torch.zeros(len(ch), dtype=torch.int64)
            def _mk(j, Hbuf, count):
                def h(mod, inp):
                    xf = inp[0].detach().reshape(-1, inp[0].shape[-1]).float(); Hbuf[j].addmm_(xf.T, xf)
                    if count: EN[j] += xf.shape[0]
                return h
            for j, e in enumerate(ch.tolist()):
                ehooks.append(blk.ffn.experts[e].register_forward_pre_hook(_mk(j, EH13, True)))     # Expert.forward input x[idx] (l.841, l.900)
                ehooks.append(blk.ffn.experts[e].w2.register_forward_pre_hook(_mk(j, EH2, False)))  # w2 input = weights * silu(gate) * up (l.848-851)
        with torch.inference_mode(), dsm.set_dtype(torch.bfloat16), torch.device(dev):
            for b in range(0, R, B):
                nb = min(B, R - b); ST["tok0"] = b * S
                hsb = streams[b:b + nb]; pm = pre_mix[b:b + nb]
                if ratio:   # swap this batch's CSA2 runtime slots in (SharedAttentionRuntime l.1166-1180)
                    sa.compress_kv = store.ckv[b:b + nb]; sa.index_k = store.ik[b:b + nb]
                    sa.topk_idxs = store.topk[b:b + nb, :, :tk]; sa.candidates = store.cand[b:b + nb, :, :S // ratio] if has_cand else None
                if blk.engram is not None:   # Transformer.forward l.1262-1263: engram before the block
                    hsb = blk.engram(hsb, hashes[b:b + nb, :, blk.engram.layer_hash_index, :], None)
                out, nxt = blk(hsb, 0, pm, None)                                                   # l.1267
                if final:
                    streams[b:b + nb] = out; pre_mix[b:b + nb] = nxt; ex["hc_in_pre"] += pm.double().sum((0, 1))
                    if attn.is_kv_source:
                        store.ckv[b:b + nb, :S // ratio] = sa.compress_kv[:nb, :S // ratio]; store.ik[b:b + nb, :S // ratio] = sa.index_k[:nb, :S // ratio]
                    if attn.is_index_source: store.topk[b:b + nb, :, :tk] = sa.topk_idxs
                    if i == margs.candidate_source_layer: store.cand[b:b + nb, :, :S // ratio] = sa.candidates
        for h in ehooks: h.remove()
        if chunks:
            save_file({"experts_w13_in": EH13.cpu(), "experts_w2_in": EH2.cpu(), "expert_ids": torch.tensor(ch), "counts": EN},
                      os.path.join(a.hessians, f"hessian_L{i:02d}_experts_c{p_}of{n_pass}.safetensors"))
            del EH13, EH2; torch.cuda.empty_cache()
            log(f"  L{i:02d} expert Hessian chunk {p_ + 1}/{n_pass} saved ({len(ch)} experts)")
    for h in hooks: h.remove()
    if HESS:
        save_file({**{k: v.cpu() for k, v in HS.items()}, "counts": torch.tensor([HN[k] for k in HS]), },
                  os.path.join(a.hessians, f"hessian_L{i:02d}.safetensors"), metadata={"names": json.dumps(list(HS))})
        if DUMP_X is not None:
            save_file({"x": DUMP_X, "route_idx": DUMP_I, "route_w": DUMP_W}, os.path.join(a.hessians, f"ffn_in_L{i:02d}.safetensors"),
                      metadata={"layer": str(i), "rows": str(R), "seq": str(S), "row_start": str(a.row_start)})
        del HS; torch.cuda.empty_cache()
    # ---- outputs
    json.dump({"layer": i, "tokens": sm["tokens"], "attn_in_absmax": sm["attn_in"].tolist(), "mlp_in_absmax": sm["mlp_in"].tolist()},
              open(os.path.join(a.out, f"smooth_L{i:02d}.json"), "w"))
    json.dump({"layer": i, "tokens": sal["tokens"], "count": sal["count"].tolist(), "wmass": sal["wmass"].tolist(), "route_scale": margs.route_scale,
               "n_activated": margs.n_activated_experts}, open(os.path.join(a.out, f"route_L{i:02d}.json"), "w"))
    sinks = attn.attn_sink.detach().float().cpu()
    fl = {"kv_src": bool(attn.is_kv_source), "idx_src": bool(attn.is_index_source), "has_comp": attn.compressor is not None,
          "owns_k": attn.indexer is not None and attn.indexer.owns_k, "uses_cand": attn.indexer is not None and attn.indexer.uses_candidates,
          "cand_src": attn.indexer is not None and attn.indexer.is_candidate_source, "rows_gathered": int(engram_tables[i].uniq.numel()) if blk.engram is not None else None}
    # free the layer now: the hc_mixes wrapper is a closure over the block (a bound-method cycle), so break it explicitly, or the 27.5 GB
    # of expert weights would survive into the next layer's build (55 GB peak)
    del blk.hc_mixes; del blk, attn, _orig_mixes, _hc_mixes; hooks.clear(); gc.collect(); torch.cuda.empty_cache()
    n_tok = max(sm["tokens"], 1)
    acc = torch.zeros(HC, device=dev, dtype=torch.float64)
    for r in range(0, R, 8): acc += streams[r:r + 8].float().pow(2).sum(dim=(0, 1, 3)).double()
    stream_rms = (acc / (R * S * D)).sqrt().tolist()
    idx_stats = None
    if fl["idx_src"]:
        idx_stats = {"topk": tk, "topk_valid_frac": IDX["topk_valid"] / max(IDX["topk_slots"], 1)}
        if fl["uses_cand"]: idx_stats["unpooled_topk_in_pool_frac"] = IDX["free_in_pool"] / max(IDX["free_valid"], 1)
        if fl["uses_cand"] or fl["cand_src"]:
            idx_stats["pool_frac_of_reachable"] = IDX["pool_reach"] / max(IDX["reach"], 1)
            if margs.candidate_topk_blocks * margs.candidate_block_size >= S // ratio:
                idx_stats["note"] = f"candidate pool = {margs.candidate_topk_blocks} blocks x {margs.candidate_block_size} >= {S // ratio} positions at S={S}: the pool cannot restrict here"
    eng = None
    if i in ENGRAM_LAYERS:
        eng = {"delta_rms_per_stream": (ex["eng_delta_sq"] / max(n_tok, 1) / D).sqrt().tolist(), "value_rms": (ex["eng_value_sq"] / max(ex["eng_n"], 1) / D) ** 0.5,
               "rows_gathered": fl["rows_gathered"], "table_rows": layout.num_embeddings[layout.layer_ids.index(i)]}
    if pf is not None: pf.join()
    release([f"layers.{i + 1}.", f"layers.{i + 2}."])
    json.dump({"layer": i, "moe": True, "csa2_mode": mode, "compress_ratio": ratio, "is_kv_source": fl["kv_src"], "is_index_source": fl["idx_src"],
               "uses_candidates": bool(has_cand), "tokens": sm["tokens"],
               "q_norm_absmax": ex["q_norm"].tolist(), "kv_norm_absmax": ex["kv_norm"].tolist(),
               "compress_kv_norm_absmax": ex["ckv_norm"].tolist() if fl["has_comp"] else None,
               "indexer_k_norm_absmax": ex["idx_k_norm"].tolist() if fl["owns_k"] else None,
               "hc_attn_pre_mean": (ex["hc_in_pre"] / n_tok).tolist(), "hc_attn_post_mean": (ex["hc_attn_post"] / n_tok).tolist(),
               "hc_ffn_pre_mean": (ex["hc_ffn_pre"] / n_tok).tolist(), "hc_ffn_post_mean": (ex["hc_ffn_post"] / n_tok).tolist(),
               "hc_next_pre_mean": (ex["hc_next_pre"] / n_tok).tolist(),
               "hc_attn_comb_diag_mean": (ex["hc_attn_comb_diag"] / n_tok).tolist(), "hc_ffn_comb_diag_mean": (ex["hc_ffn_comb_diag"] / n_tok).tolist(),
               "sinks": {"mean": sinks.mean().item(), "min": sinks.min().item(), "max": sinks.max().item(), "values": sinks.tolist()},
               "indexer": idx_stats, "engram": eng, "stream_rms_after": stream_rms, "weights": wst, "bytes_by_category": inv_bytes,
               "hessians": ({"file": f"hessian_L{i:02d}.safetensors", "expert_mode": a.hessian_expert_mode, "expert_chunks": n_pass} if HESS else None),
               "seconds": {"fetch_and_load": fetch_s, "total": time.time() - t0}},
              open(os.path.join(a.out, f"extra_L{i:02d}.json"), "w"))
    log(f"L{i:02d} done {time.time() - t0:6.1f}s (load {fetch_s:.0f}s) {mode:8s} r={ratio} | attn_in max {sm['attn_in'].max():.1f} med {sm['attn_in'].median():.3f} "
        f"| mlp_in max {sm['mlp_in'].max():.1f} | route top/med count {sal['count'].max():.0f}/{sal['count'].median():.0f} "
        f"| stream rms {['%.3f' % r for r in stream_rms]} | sinks mean {sinks.mean():.2f} | staged {staged_bytes() / 1e9:.0f} GB")
    if staged_bytes() > a.stage_budget_gb * 1e9: log(f"  WARNING staged bytes over budget ({staged_bytes() / 1e9:.0f} GB)")

if a.state_out:
    d = {"streams": streams.cpu(), "pre_mix": pre_mix.cpu(), **{k: v.cpu() for k, v in store.tensors().items()}}
    save_file(d, a.state_out, metadata={"layer_done": str(hi), "row_start": str(a.row_start), "rows": str(R), "seq": str(S)})
    log(f"state written {a.state_out}")

# =====================================================================================================================
# bf16 reference NLL on the calibration rows (only when the pass ended at the last layer) — Transformer.forward l.1268-1269 with
# full_logits=True (the reference default returns the last position only, for generation)
# =====================================================================================================================
if hi == L - 1 and not a.smoke:
    with dsm.set_dtype(torch.bfloat16), torch.device("meta"):
        norm = dsm.RMSNorm(D, margs.norm_eps); head = dsm.ParallelHead(V, D, margs.norm_eps, margs.hc_eps)
    norm.to_empty(device=dev); head.to_empty(device=dev)
    load_into(norm, "norm."); load_into(head, "head.")   # head.weight: bf16 in the checkpoint, fp32 parameter (l.1005-1006)
    tot_nll, tot_tok, per_row = 0.0, 0, []
    with torch.inference_mode(), torch.device(dev):
        for r in range(R):
            h = dsm.Block.hc_pre(None, streams[r:r + 1], pre_mix[r:r + 1])        # l.1268: collapse with the last layer's ffn_pre
            logits = head(norm(h), full_logits=True)[0]                          # [S, V] fp32
            tgt = ids[r, 1:].to(dev); nll = F.cross_entropy(logits[:-1], tgt, reduction="sum").item()
            per_row.append(nll / (S - 1)); tot_nll += nll; tot_tok += S - 1
    json.dump({"rows": R, "row_start": a.row_start, "tokens": tot_tok, "mean_nll": tot_nll / tot_tok, "ppl": math.exp(tot_nll / tot_tok), "per_row_nll": per_row,
               "note": "bf16 reference (exact dequant body, fp8/fp4 KV-path quantizers kept) on the in-domain calibration rows, not held-out"},
              open(os.path.join(a.out, (f"ppl_exl3_{os.path.splitext(os.path.basename(a.exl3_recipe))[0]}_calib.json" if a.exl3_recipe else ("ppl_bf16_calib.json" if a.engram_fq == "none" else f"ppl_engram_{a.engram_fq}_calib.json"))), "w"), indent=1)
    log(f"bf16 reference NLL on calibration rows{'' if a.engram_fq == 'none' else ' with Engram ' + a.engram_fq}: {tot_nll / tot_tok:.4f} (ppl {math.exp(tot_nll / tot_tok):.2f})")
    release([])
log("DSV41-CAPTURE-DONE")
