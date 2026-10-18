#!/usr/bin/env python3
"""pollard-export -- take a Pollard sensitivity profile and emit a vLLM/SGLang-loadable
GPTQ checkpoint with Pollard's measured allocation carried over as a gptqmodel `dynamic`
4/8-bit mix. This is the GPU-runtime lane: llama.cpp/ik_llama.cpp gets the 1-bit trellis
flagship; vLLM/SGLang get a sensitivity-allocated 4/8 GPTQ that runs Marlin-accelerated.

Why 4/8 and not 1-2: vLLM/SGLang's fast Marlin kernel supports ONLY 4-bit and 8-bit, so
the measured mix lives there -- sensitive modules kept at 8-bit, the tolerant body crushed
to 4-bit, allocated by Pollard's profile. gptqmodel's per-module `dynamic` is "fully
integrated into vLLM"; SGLang loads GPTQ but its mixed-bit is fragile (layer fusion), so
`--uniform` emits a plain W4 for SGLang.

Hard runtime constraints baked in (verified Aug 2026):
  * NEVER split bits inside a fused group: q/k/v_proj share one bit-width; gate/up_proj
    share one. (o_proj, down_proj are free.) Marlin fuses these.
  * group_size 128, desc_act False, sym True  -> the well-trodden Marlin fast path.
  * bits in {4, 8} only.

    pollard-export --model Qwen/Qwen2.5-7B-Instruct --sensitivity qwen7b.sensitivity.json \\
        --calib calib.txt --out ./Qwen2.5-7B-Instruct-Pollard-GPTQ
    # then: vllm serve ./...-Pollard-GPTQ --quantization gptq

LADDER rungs: --bits {2,3,4,8} sets the body (hot layers at --high-bits, default 8; --bits 8 = uniform
int8). 4 and 8 run on Marlin; 2- and 3-bit GPTQ load in vLLM through its plain GPTQ (exllama) kernels --
slower, and SGLang does not take them -- so they are the small-VRAM rungs, not the fast ones.

    pollard-export --model <hf> --sensitivity s.json --calib calib.txt --bits 3 --out ./M-Pollard-GPTQ-int3

ENGINES (--engine): `gptqmodel` loads the model through transformers and runs gptqmodel's GPTQ (disk offload on,
one layer at a time, where the gptqmodel version has it). `pollard` is Pollard's own sequential GPTQ streamed
block by block from disk (pollard_stream + pollard_gptqpack): any size the machine can stream (peak = one block +
its Hessians + the activations, preflighted against free RAM / --max-ram-gb), a VLM / omni checkpoint through its
language model (towers, talkers and projectors copied through at source precision), the checkpoint's own code only
with consent (--trust-remote-code on), and an already-quantized source (modelopt NVFP4 / ggml blocks) dequantized
on read. It writes the same GPTQ v1 layout (qweight / qzeros / scales / g_idx, sym, group 128, desc_act False) vLLM
loads, with Pollard's allocation in `dynamic`. `auto` keeps gptqmodel for a plain causal LM that fits in RAM.
"""
import argparse, json, os, re, sys
import pollard_workspace as ws

HIGH, LOW = 8, 4                       # Marlin supports only these
GPTQ_BITS = (2, 3, 4, 8)               # what gptqmodel packs and vLLM's GPTQ path loads
# module groups within a decoder layer. q/k/v are ONE fused unit; gate/up are ONE fused unit.
FUSED = {"attn_qkv": ["q_proj", "k_proj", "v_proj"], "gate_up": ["gate_proj", "up_proj"]}
FREE = {"attn_o": ["o_proj"], "ffn_down": ["down_proj"]}


def detect_moe(model_id, layers_hint=0):
    """Return (is_moe, n_layers). MoE if the config advertises experts (Qwen-MoE
    num_experts / Mixtral num_local_experts / DeepSeek n_routed_experts)."""
    cfg = ws.text_config_dict(ws.config_dict(model_id))      # plain JSON: no checkpoint code runs to read it
    if not cfg:
        return False, layers_hint
    n = cfg.get("num_hidden_layers", 0) or layers_hint
    moe = any(isinstance(cfg.get(k), int) and cfg.get(k) > 1
              for k in ("num_experts", "num_local_experts", "n_routed_experts"))
    moe = moe or any(b.get("block_type") == "moe" for b in (cfg.get("block_configs") or []) if isinstance(b, dict))
    return bool(moe), int(n)


resolve_trust_remote_code = ws.resolve_trust_remote_code   # shared across every export lane


def detect_mla(model_id):
    """True if the model uses Multi-head Latent Attention (DeepSeek-V2/V3, GLM-4.5/5.3) -- its
    attention is q_a/q_b/kv_a/kv_b_proj, not q/k/v_proj. ATTN_PROJ matches both; this is only
    for the user-facing note. Detected from the low-rank KV config field."""
    return ws.text_config_dict(ws.config_dict(model_id)).get("kv_lora_rank") is not None


parse_layers = ws.parse_layers   # shared layer-spec parser


def allocate(sens, n_layers, hot_frac, focus=None, low=None, high=None):
    """Pollard profile -> {layer: {group: bits}}. The most-sensitive `hot_frac` of layers
    keep their group at 8-bit; the rest go 4-bit. Fused groups get ONE bit-width (constraint).
    'attn' sensitivity drives q/k/v+o; 'ffn' sensitivity drives gate/up+down."""
    LOW, HIGH = low or globals()["LOW"], high or globals()["HIGH"]
    focus = set(focus or ())                       # user-chosen layers to force HIGH (steer the budget)
    ffn = {int(k): float(v) for k, v in (sens.get("ffn") or {}).items()}
    attn = {int(k): float(v) for k, v in (sens.get("attn") or {}).items()}
    if not ffn and not attn:                       # no profile -> uniform W4, but honor --focus-layers
        return {i: {"attn": HIGH if i in focus else LOW,
                    "ffn": HIGH if i in focus else LOW} for i in range(n_layers)}

    def hot_layers(d):
        k = max(1, int(round(hot_frac * len(d))))
        return set(sorted(d, key=lambda i: d[i], reverse=True)[:k])
    hot_attn, hot_ffn = hot_layers(attn or ffn) | focus, hot_layers(ffn or attn) | focus
    return {i: {"attn": HIGH if i in hot_attn else LOW,
                "ffn": HIGH if i in hot_ffn else LOW} for i in range(n_layers)}


# Arch-agnostic projection matchers. Standard attention is q/k/v/o_proj; MLA (DeepSeek-V2/V3,
# GLM-4.5/5.3) replaces them with q_a_proj/q_b_proj/kv_a_proj_with_mqa/kv_b_proj -- all still end in
# "proj", none of the norms do -- so "any *proj* under self_attn" catches both without a per-arch table.
ATTN_PROJ = r"self_attn\.[a-z_]*proj[a-z0-9_]*"
# Dense FFN is gate/up/down_proj; MoE also carries the same names under experts.<i>. -- this catches both. vLLM
# matches the rules against its FUSED module names (`mlp.gate_up_proj`, `self_attn.qkv_proj`, one `mlp.experts` per
# layer), not the checkpoint's: `(?:gate|up|down)_proj` never matched `gate_up_proj`, so a hot layer's gate/up was
# built at the base bits while the checkpoint stored them at 8 -- `gate_up` is spelled out for it.
FFN_PROJ = r"(?:mlp|block_sparse_moe)(?:\.experts\.\d+)?\.(?:gate_up|gate|up|down)_proj"
#: every rule is a `+:` override (gptqmodel's and vLLM's own spelling; `-:` would exclude)
POS = "+:"


def dynamic_config(alloc, high=None):
    """gptqmodel `dynamic` regex map. Base is LOW (4-bit); we add 8-bit overrides for the
    hot groups. q/k/v+o (or MLA's a/b projections) share the attn bit; gate/up+down share the
    ffn bit (fused-safe). Projection names are matched arch-agnostically (see ATTN_PROJ/FFN_PROJ)."""
    HIGH = high or globals()["HIGH"]
    dyn = {}
    # Head-wise attention output gate (Spark2_5 `self_attn.g_proj`): selection-critical, always high.
    # No-op for arches without it. Placed first so it holds regardless of per-layer attn allocation.
    dyn[POS + r".*\.self_attn\.[a-z_]*g_proj"] = {"bits": HIGH}
    for i, g in alloc.items():
        if g["attn"] == HIGH:
            dyn[POS + rf".*\.layers\.{i}\.{ATTN_PROJ}"] = {"bits": HIGH}
        if g["ffn"] == HIGH:
            dyn[POS + rf".*\.layers\.{i}\.{FFN_PROJ}"] = {"bits": HIGH}
    return dyn


def moe_dynamic_config(alloc, high=None):
    """MoE `dynamic` map (Qwen-MoE / Mixtral / DeepSeek naming). The Pollard MoE policy carried
    into the GPTQ lane: base LOW (4-bit) crushes the cold experts; the ROUTER and SHARED experts
    are ALWAYS protected at HIGH (8-bit) -- crushing the router scrambles expert selection -- and the
    experts of the most-sensitive (`ffn`-hot) layers go HIGH. Attention follows the `attn` profile.
    Router = `mlp.gate` (Qwen) / `block_sparse_moe.gate` (Mixtral) -- matched with `\\.gate$` so it
    never catches an expert's `gate_proj`."""
    HIGH = high or globals()["HIGH"]
    dyn = {}
    # ALWAYS protect the router (selection integrity) and the shared expert (every-token path).
    # `shared_experts?` covers Qwen-MoE's singular `shared_expert` and DeepSeek/GLM's plural `shared_experts`.
    dyn[POS + r".*\.(mlp|block_sparse_moe)\.gate$"] = {"bits": HIGH}
    dyn[POS + r".*\.mlp\.shared_experts?(_gate|\.(gate_up_proj|gate_proj|up_proj|down_proj))"] = {"bits": HIGH}
    for i, g in alloc.items():
        if g["attn"] == HIGH:                      # standard OR MLA attention (see ATTN_PROJ)
            dyn[POS + rf".*\.layers\.{i}\.{ATTN_PROJ}"] = {"bits": HIGH}
        if g["ffn"] == HIGH:                       # hot-layer experts kept at 8-bit (Mixtral w1/w2/w3 too)
            # one rule for the layer's experts: gptqmodel matches `...experts.<e>.gate_proj` (any projection, w1/w2/w3
            # too), vLLM the fused `...mlp.experts` module it builds per layer -- both must land on the same bits
            dyn[POS + rf".*\.layers\.{i}\.(mlp|block_sparse_moe)\.experts(\.|$)"] = {"bits": HIGH}
    return dyn


def avg_bits(alloc):
    b = [g["attn"] for g in alloc.values()] + [g["ffn"] for g in alloc.values()]
    return sum(b) / len(b) if b else LOW


def shard_ranges(n_layers, n_nodes):
    """Split n_layers into n_nodes contiguous ranges, remainder spread over the first nodes.
    Contiguous (not round-robin) so each node holds a band and only ONE boundary hidden state
    crosses the wire between neighbours. Returns [(start, end_exclusive), ...]."""
    n_nodes = max(1, min(n_nodes, n_layers))
    base, rem = divmod(n_layers, n_nodes)
    ranges, s = [], 0
    for k in range(n_nodes):
        span = base + (1 if k < rem else 0)
        ranges.append((s, s + span)); s += span
    return ranges


def print_shard_plan(model_id, n_layers, n_nodes, bf16_gb):
    """Band-parallel layer-streaming plan: which contiguous layer band each node owns, its byte
    budget, and the boundary-handoff contract. The actual per-node quantize is the existing offload
    path restricted to that band; only the boundary hidden state crosses between neighbours."""
    ranges = shard_ranges(n_layers, n_nodes)
    per_layer_gb = (bf16_gb / n_layers) if (bf16_gb and n_layers) else 0.0
    print(f"== pollard-export band-parallel plan :: {model_id}")
    print(f"   {n_layers} decoder layers over {len(ranges)} node(s) -- contiguous bands, "
          "one boundary hidden state handed to the next node:")
    for k, (s, e) in enumerate(ranges):
        budget = f"  |  ~{per_layer_gb*(e-s):.1f} GB bf16 resident" if per_layer_gb else ""
        print(f"   node {k}: layers [{s}..{e-1}]  ({e-s} layers){budget}")
    print("   handoff contract: node k quantizes its band with the offload path, re-runs the quantized"
          "\n     band to produce the hidden state at its last layer, and passes THAT to node k+1 as"
          "\n     its input (node 0 starts from the embedded calibration tokens).")
    print("   NOTE: verify end-to-end on your own cluster -- storage egress (~100 MB/s/box in the field)"
          " is usually the wall-clock wall, not compute.")


def pick_engine(model, want="auto"):
    """('pollard' | 'gptqmodel', why). auto keeps gptqmodel for what it always handled -- a plain causal LM, no
    custom code, unquantized, that fits in RAM -- and streams everything else through Pollard's engine."""
    if want != "auto":
        return want, "asked for"
    try:
        import gptqmodel  # noqa: F401
    except Exception:                                                     # noqa: BLE001
        return "pollard", "gptqmodel is not installed here"
    if not os.path.isdir(model):
        return "gptqmodel", ""
    import pollard_stream as S
    if S.is_overlay(model):
        return "pollard", "a smoothing overlay (scales over the source) -- only the stream reader applies it"
    if S.needs_remote_code(model):
        return "pollard", "custom code: the stream engine runs a native mapping where one exists"
    cfg = ws.config_dict(model)
    if cfg.get("quantization_config"):
        return "pollard", "the source is already quantized: the stream reader dequantizes it"
    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as LM
        if cfg.get("model_type") not in LM:
            return "pollard", f"transformers has no causal-LM class for {cfg.get('model_type')!r} (VLM / omni)"
    except Exception:                                                     # noqa: BLE001
        return "pollard", "transformers is not importable"
    w = sum(os.path.getsize(p) for p in S.weight_files(model)) / 1e9
    free = S.free_ram_gb() or 0
    if w * 1.2 > free:
        return "pollard", f"{w:.1f} GB of weights vs {free:.1f} GB free RAM: streamed"
    return "gptqmodel", ""


def kv_note(model_id):
    """Victor's point: vLLM pre-allocates KV hard (gpu_memory_utilization ~0.9, paged
    attention) -- far more headroom than llama.cpp. Flag it so the target VRAM is realistic."""
    return ("vLLM reserves KV cache up front (gpu_memory_utilization ~0.9). Budget "
            "weights + a large KV pool + activations, not just the weight bytes -- a model "
            "that fits in GGUF on a card can OOM in vLLM. Lower --gpu-memory-utilization or "
            "--max-model-len if it won't fit; run one load test before publishing the fit.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF model dir or id (FP16/BF16)")
    ap.add_argument("--sensitivity", help="Pollard sensitivity.json (pollard-probe/-sensitivity)")
    ap.add_argument("--calib", help="calibration text (one sample per line or a corpus); "
                    "required to build, not for --plan-only / --shard-plan")
    ap.add_argument("--out", help="output dir for the GPTQ checkpoint (default: workspace)")
    ap.add_argument("--layers", type=int, default=0, help="n decoder layers (else read from config)")
    ap.add_argument("--hot-frac", type=float, default=0.35, help="fraction of layers kept at 8-bit")
    ap.add_argument("--focus-layers", help="force these layer indices to HIGH bits regardless of the "
                    "sensitivity profile -- steer the budget to layers you care about, e.g. '3,4,8' or "
                    "'3-8,16'. Unions into the measured hot set (works even with no profile).")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--bits", type=int, choices=GPTQ_BITS, default=None,
                    help="ladder rung: body bits (default 4). 8 = uniform int8; 2/3 load on vLLM's slower GPTQ kernel")
    ap.add_argument("--high-bits", type=int, choices=(4, 8), default=None,
                    help="bits for the measured-hot groups / router / shared experts (default 8)")
    ap.add_argument("--uniform", action="store_true", help="plain uniform W4 (for SGLang mixed-bit fragility)")
    ap.add_argument("--trust-remote-code", default="auto", choices=["auto", "on", "off"],
                    help="run a model's own modeling code (custom archs like Spark2_5). 'auto' enables it "
                         "only when config.json has an auto_map (default)")
    ap.add_argument("--engine", default="auto", choices=["auto", "pollard", "gptqmodel"],
                    help="who runs GPTQ: 'pollard' = Pollard's own sequential GPTQ streamed block by block from disk "
                         "(any size the machine can stream, VLM / omni text backbones, custom code with consent, "
                         "already-quantized sources); 'gptqmodel' = the gptqmodel library (loads through "
                         "transformers); 'auto' = gptqmodel for a plain causal LM that fits in RAM when it is "
                         "installed, Pollard's engine for everything else")
    ap.add_argument("--device", default="auto", help="pollard engine: auto (CUDA when present) | cpu | cuda[:N]")
    ap.add_argument("--nsamples", type=int, default=128, help="pollard engine: calibration rows (the memory "
                    "preflight lowers it when the activations would not fit)")
    ap.add_argument("--seqlen", type=int, default=2048, help="pollard engine: tokens per calibration row")
    ap.add_argument("--max-ram-gb", type=float, help="pollard engine: cap the run's RAM (default 85%% of free)")
    ap.add_argument("--moe", dest="moe", action="store_true", default=None,
                    help="force the MoE dynamic map (default: auto-detect from config)")
    ap.add_argument("--dense", dest="moe", action="store_false",
                    help="force the dense dynamic map")
    ap.add_argument("--plan-only", action="store_true", help="print the allocation + dynamic map, build nothing")
    ap.add_argument("--shard-plan", type=int, default=0, metavar="N",
                    help="band-parallel: print the contiguous layer range + byte budget each of N nodes owns "
                         "(for models too big for one box), then exit. Add --bf16-gb for the byte estimate.")
    ap.add_argument("--bf16-gb", type=float, default=0.0,
                    help="total BF16 size (GB) of the source model, for the --shard-plan byte budget")
    a = ap.parse_args()

    sens = json.load(open(a.sensitivity)) if a.sensitivity else {}
    n_layers = a.layers or int(sens.get("layers") or 0)
    auto_moe, det_layers = detect_moe(a.model, n_layers)
    n_layers = n_layers or det_layers
    if not n_layers:
        sys.exit("ERROR: could not read layer count -- pass --layers.")
    is_moe = auto_moe if a.moe is None else a.moe

    if a.shard_plan:
        print_shard_plan(a.model, n_layers, a.shard_plan, a.bf16_gb)
        return

    focus = parse_layers(a.focus_layers)
    if focus:
        print(f"   focus-layers: forcing layers {sorted(focus)} to HIGH (steered budget)")
    low = a.bits or LOW
    high = max(a.high_bits or HIGH, low)
    uniform = a.uniform or low == high
    alloc = allocate(sens, n_layers, a.hot_frac, focus=focus, low=low, high=high)
    dyn = {} if uniform else (moe_dynamic_config(alloc, high) if is_moe else dynamic_config(alloc, high))
    ab = low if uniform else avg_bits(alloc)
    tag = f"int{low}" if uniform else f"int{low}" + (f"-{high}" if high != HIGH else "")
    if low in (2, 3):
        print(f"   NOTE: {low}-bit GPTQ is outside Marlin (4/8): vLLM loads it on its plain GPTQ kernel "
              "(slower); SGLang will not. This is the small-VRAM rung.")
    kind = "MoE" if is_moe else "dense"
    print(f"== pollard-export :: {a.model}  [{kind}{' auto' if a.moe is None else ''}]")
    print(f"   {n_layers} layers  |  {f'UNIFORM W{low}' + (' (SGLang-safe)' if low == 4 else '') if uniform else f'{low}/{high} dynamic mix, avg {ab:.2f} bits'}"
          f"  |  group_size {a.group_size}  |  desc_act False  |  sym True")
    if is_moe and not uniform:
        print(f"   MoE: router + shared experts pinned {high}-bit (selection integrity); cold experts {low}-bit")
    if detect_mla(a.model) and not uniform:
        print("   MLA attention detected (q_a/q_b/kv_a/kv_b_proj) -- allocation matches it arch-agnostically")
    print(f"   {high}-bit modules: {len(dyn)} groups (sensitivity-ranked hot set)")
    print(f"   NOTE (KV/memory): {kv_note(a.model)}")
    if a.plan_only:
        print("   dynamic map:")
        for k, v in list(dyn.items())[:8]:
            print(f"     {k}  -> {v}")
        if len(dyn) > 8:
            print(f"     ... +{len(dyn)-8} more")
        return

    engine, why = pick_engine(a.model, a.engine)
    print(f"   engine: {engine}" + (f"  ({why})" if why else ""))
    if engine == "pollard":
        if not a.calib:
            sys.exit("ERROR: --calib is required to build (use --plan-only / --shard-plan for planning only).")
        if not a.out:
            a.out = ws.resolve_out(a.model, "gptq", tag=tag)
            print(f"   (no --out) -> workspace: {a.out}")
        import pollard_gptqpack as GP
        import pollard_stream as S
        dev = S.stream_device(a.device)
        side = GP.build_stream(a.model, a.out, a.calib, alloc=None if uniform else alloc, low=low,
                               group=a.group_size, nsamples=a.nsamples, seqlen=a.seqlen, device=dev,
                               trust_remote_code=a.trust_remote_code, max_ram_gb=a.max_ram_gb)
        ab = side["avg_bits"] - 16.0 / a.group_size
        json.dump({"model": a.model, "kind": kind, "layers": n_layers, "group_size": a.group_size,
                   "low_bits": low, "high_bits": high, "uniform": bool(uniform), "avg_bits": round(ab, 3),
                   "dynamic": dyn, "engine": "pollard-stream"},
                  open(os.path.join(a.out, "pollard-allocation.json"), "w"), indent=2)
        ws.record_build(a.model, "gptq", a.out, tag=f"GPTQ-{tag}", bpw=round(ab, 2))
        print(f"   peak RSS {side['peak_rss_gb']} GB (plan {side['memory_plan']['peak_gb']} GB), {side['minutes']} min")
        print(f"wrote GPTQ checkpoint -> {a.out}\n"
              f"  vLLM:   vllm serve {a.out} --quantization gptq")
        return
    # ---- build with gptqmodel (runs on a CUDA box; produces a vLLM/SGLang GPTQ checkpoint)
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except Exception:
        sys.exit("ERROR: gptqmodel not installed here. On the CUDA box: "
                 "pip install 'pollard-weights[gptq]' (or pip install gptqmodel). "
                 "The checkpoint it writes loads in vLLM/SGLang.")
    if not a.calib:
        sys.exit("ERROR: --calib is required to build (use --plan-only / --shard-plan for planning only).")
    calib = ws.calib_samples(a.calib)            # whole samples: an agent trace keeps its tool call in one piece
    qkw = dict(bits=low, group_size=a.group_size, desc_act=False, sym=True, dynamic=(dyn or None))
    try:                                         # gptqmodel >= 4: onload one layer at a time from disk
        import inspect
        if "offload_to_disk" in inspect.signature(QuantizeConfig).parameters or \
                "offload_to_disk" in getattr(QuantizeConfig, "__dataclass_fields__", {}):
            qkw["offload_to_disk"] = True
    except (TypeError, ValueError):
        pass
    qcfg = QuantizeConfig(**qkw)
    trc = resolve_trust_remote_code(a.model, a.trust_remote_code)
    model = GPTQModel.load(a.model, qcfg, trust_remote_code=trc)
    model.quantize(calib)
    if not a.out:
        a.out = ws.resolve_out(a.model, "gptq", tag=tag)
        print(f"   (no --out) -> workspace: {a.out}")
    model.save(a.out)
    # allocation-as-config: drop the exact bit plan beside the checkpoint. A/B'ing a different
    # allocation is then a re-pack against this record, not a full re-cook.
    try:
        pass  # os is imported at module scope
        cfg_path = os.path.join(a.out, "pollard-allocation.json")
        json.dump({"model": a.model, "kind": kind, "layers": n_layers, "group_size": a.group_size,
                   "low_bits": low, "high_bits": high, "uniform": bool(uniform),
                   "avg_bits": round(ab, 3), "dynamic": dyn}, open(cfg_path, "w"), indent=2)
        print(f"   allocation recorded -> {cfg_path}")
    except Exception as e:
        print(f"   (could not write allocation sidecar: {e})")
    ws.record_build(a.model, "gptq", a.out, tag=f"GPTQ-{tag}", bpw=round(ab, 2))
    print(f"wrote GPTQ checkpoint -> {a.out}\n"
          f"  vLLM:   vllm serve {a.out} --quantization gptq\n"
          f"  SGLang: python -m sglang.launch_server --model-path {a.out} --quantization gptq\n"
          f"  (mixed 4/8 loads dynamic-accelerated on vLLM; use --uniform for SGLang.)")


if __name__ == "__main__":
    main()
