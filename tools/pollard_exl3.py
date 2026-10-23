#!/usr/bin/env python3
"""pollard-exl3 -- emit an EXL3 (exllamav3) model with Pollard's allocation, same as the other lanes.

[locked] LOCKED GOLD RECIPE (EXL3) -- THE WIN: **the Pollard method (its smoothing preconditioner + Calib 3.0)
beats EXL3 on EXL3's OWN allocator, trellis atoms, and custom kernel.** MEASURED (Qwen2.5-3B @4bpw):
broken 3090 -> smoothed 8.699 -> smoothed + Calib 3.0 **8.670**, vs EXL3 out-of-box **8.699** -- a win on
their own turf, untuned and with no extra calibration. And smoothing alone takes unusable low-bit (PPL
3090) to 4bpw ~= 8bpw quality at HALF the size. First and only tool that does this.
`pollard --format exl3` runs the gold recipe by DEFAULT (--no-smooth to skip). proxy_err is banned (legacy/).

LOW-BIT mechanics: precondition first with `pollard-hf-smooth` (SmoothQuant folded into the RMSNorms).
Massive-activation input channels otherwise collapse the trellis global scale and silently wreck a layer
(measured: layer sqnr -14 -> +18 after smoothing). One-shot: `pollard-doctor --source <fp16> --repair
--lane exl3` (smooth -> reconvert -> verify), or `pollard --format exl3 --run`.

The GPU desktop lane: GGUF (llama.cpp), GPTQ (vLLM/SGLang), MLX (Apple), and THIS -- EXL3 for the
exllamav3 runtime. It's the compatibility lane: same one command into a 4th runtime.

ALLOCATION (gold path): keep **EXL3's native allocator** and win through the Pollard method's
preconditioning + calibration -- that's where the win comes from, and it lands on EXL3's own atoms. The
allocator is already strong on its trellis atoms, so don't spend your bits reallocating it: the GGUF
K-quant role-map is a DEAD LEVER on EXL3 (K-quant priors don't map to trellis atoms -- measured worse, not
worth the ~50 min). The gold recipe already beats EXL3 out-of-box without touching allocation. A finer
per-tensor-KL recipe measured directly in EXL3's atom space is an open R&D lever -- until one is shown to
beat the gold recipe, the gold path is smoothing + Calib 3.0 + native allocator. Judge by pollard-verify.

Two allocation modes:
  * budgeted (DEFAULT, the gold path): EXL3's allocator via --bits + --head_bits/--mtp_bits + --hq.
  * recipe (--recipe pollard|exl3|FILE): per-tensor bitrate YAML for EXL3's -rcp -- R&D only. `pollard` emits one
    from Pollard's measured sensitivity at --bpw (EXL3's tensor list, fused groups tied); `exl3` uses EXL3's own
    sc_optimize recipe. Don't port a GGUF role map here; the head-to-head decides (notes/exl3-recipe-bridge.md).

  pollard-exl3 --model Qwen/Qwen3-8B --out ./Qwen3-8B-Pollard-EXL3 --bpw 3.0   # budgeted (recommended)
  pollard-exl3 --model <hf> --out <dir> --plan-only          # print the exllamav3 command, build nothing
  pollard-exl3 --model <hf> --out <dir> --recipe plan.yaml   # EXPERIMENTAL explicit per-tensor plan
  pollard-exl3 --model <dir> --out <dir> --bpw 3.0 --recipe pollard --sensitivity s.json   # Pollard -> EXL3 recipe
  pollard-exl3 --model <dir> --out <dir> --bpw 3.0 --recipe exl3 --exl3-recipe recipe_3.00bpw.yaml

Needs exllamav3 installed AND its CUDA ext loadable (a bleeding-edge GPU like Blackwell may need a
source build -- the pip wheel's prebuilt ext can fail with 'DLL load failed importing exllamav3_ext')."""
import argparse, os, subprocess, sys


def detect_vl(model_id):
    """Does this checkpoint carry a vision or audio tower? Read the config, never the name."""
    import json as _json
    for fn in ("config.json",):
        p = os.path.join(model_id, fn)
        if not os.path.exists(p):
            continue
        try:
            cfg = _json.load(open(p, encoding="utf-8"))
        except Exception:
            return False
        blob = _json.dumps(cfg).lower()
        return any(k in blob for k in ("vision_config", "vision_tower", "audio_config",
                                       "image_token", "visual"))
    return False


def detect_moe(model_id, layers_hint=0):
    import pollard_workspace as ws
    cfg = ws.text_config_dict(ws.config_dict(model_id))      # plain JSON: no checkpoint code runs to read it
    if not cfg:
        return False, layers_hint
    n = cfg.get("num_hidden_layers", 0) or layers_hint
    moe = any(isinstance(cfg.get(k), int) and cfg.get(k) > 1
              for k in ("num_experts", "num_local_experts", "n_routed_experts"))
    return bool(moe), int(n)


#: exllamav3's architectures (arch_string of every exllamav3/architecture/*.py), read off the box's exllamav3 on
#: 2026-10-16 -- used only where exllamav3 itself is not importable (a dry run on another machine)
EXL3_ARCHS_STATIC = {
    "AfmoeForCausalLM", "ApertusForCausalLM", "ArceeForCausalLM", "CohereForCausalLM", "Cohere2ForCausalLM",
    "DeciLMForCausalLM", "DeepseekV3ForCausalLM", "DeepseekV4ForCausalLM", "DFlashDraftModel",
    "DFlashLagunaForCausalLM", "Dots1ForCausalLM", "Ernie4_5_ForCausalLM", "Ernie4_5_MoeForCausalLM",
    "Exaone4ForCausalLM", "Gemma2ForCausalLM", "Gemma3ForConditionalGeneration", "Gemma3ForCausalLM",
    "Gemma4ForConditionalGeneration", "Gemma4UnifiedForConditionalGeneration", "Glm4ForCausalLM",
    "Glm4vForConditionalGeneration", "Glm4vMoeForConditionalGeneration", "Glm4MoeForCausalLM",
    "Glm5NextForConditionalGeneration", "GlmMoeDsaForCausalLM", "GptOssForCausalLM", "HCXVisionV2ForCausalLM",
    "HyperCLOVAXForCausalLM", "HYV3ForCausalLM", "IQuestCoderForCausalLM", "LagunaForCausalLM", "Lfm2MoeForCausalLM",
    "LlamaForCausalLM", "MiMoForCausalLM", "MiniMaxM2ForCausalLM", "Ministral3ForCausalLM", "MistralForCausalLM",
    "Mistral3ForConditionalGeneration", "MixtralForCausalLM", "MuseGlimmerForConditionalGeneration",
    "NemotronHForCausalLM", "Olmo3ForCausalLM", "OlmoHybridForCausalLM", "Phi3ForCausalLM", "Qwen2ForCausalLM",
    "Qwen2_5_VLForConditionalGeneration", "Qwen3ForCausalLM", "Qwen3_5ForConditionalGeneration",
    "Qwen3MoeForCausalLM", "Qwen3NextForCausalLM", "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration", "SeedOssForCausalLM", "SmolLM3ForCausalLM", "SolarOpenMoeForCausalLM",
    "Step3p5ForCausalLM", "Step3p7ForCausalLM",
}


def exl3_archs():
    """(set of architecture strings exllamav3 converts, where the list came from)."""
    try:
        from exllamav3.architecture.architectures import ARCHITECTURES
        return set(ARCHITECTURES), "exllamav3"
    except Exception:                                               # noqa: BLE001
        return set(EXL3_ARCHS_STATIC), "the list recorded from exllamav3 on 2026-10-16"


def exl3_input_plan(model_dir):
    """What exllamav3 converts for this checkpoint, decided from config.json alone:
    {"mode": "direct" | "text" | "hold", "arch": ..., "why": ...}. `text` = the wrapper's language model lifted out
    as the plain causal LM exllamav3 knows (pollard_stream.TEXT_EQUIV, verified token for token) for the convert, then
    recomposed into the COMPLETE checkpoint (recompose_complete: the quantized language model back under the source's
    names, the source's config, every tower / talker / processor file at source precision) -- never shipped alone.
    `hold` carries the exact reason no EXL3 of this model would load in exllamav3."""
    import json as _j
    try:
        cfg = _j.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
    except Exception as e:                                          # noqa: BLE001
        return {"mode": "hold", "arch": None, "why": f"no readable config.json in {model_dir} ({e})"}
    archs, where = exl3_archs()
    arch = (cfg.get("architectures") or [None])[0]
    if arch is None:
        return {"mode": "direct", "arch": None, "why": "config.json names no architecture: exllamav3 decides"}
    if arch in archs:
        return {"mode": "direct", "arch": arch, "why": ""}
    import pollard_stream as S
    eq = S.TEXT_EQUIV.get(cfg.get("model_type"))
    if eq and eq["arch"] in archs:
        return {"mode": "text", "arch": eq["arch"], "why": f"exllamav3 here has no {arch}; its language model is "
                                                           f"{eq['arch']} ({eq['why']}): converted as that, then "
                                                           "recomposed into the complete checkpoint (towers, talker "
                                                           "and processor files at source precision)"}
    inner = None
    for k in ("llm_config", "text_config", "language_config"):
        v = cfg.get(k)
        if isinstance(v, dict) and v.get("architectures"):
            inner = v
            break
    detail = ""
    if inner:
        ia = (inner.get("architectures") or [None])[0]
        detail = f"; its language model is {ia}"
        if ia in archs:
            detail += " -- exllamav3 has it, but no verified text-only mapping from this wrapper is recorded"
        elif inner.get("model_type", "").endswith("_puzzle") or inner.get("moe_latent_size"):
            base = ia.replace("Puzzle", "") if ia else "?"
            detail += (f", which exllamav3's {base if base in archs else 'architectures'} cannot hold: "
                       + ", ".join(x for x in (
                           "per-layer block configs (expert count / width differ by layer)"
                           if inner.get("block_configs") else "",
                           "latent MoE (fc1/fc2_latent_proj)" if inner.get("moe_latent_size") else "") if x))
    return {"mode": "hold", "arch": arch,
            "why": f"exllamav3 has no architecture for {arch}{detail} ({where}) -- an EXL3 of it would not load in "
                   "any runtime"}


def exl3_input(model_dir, log=print):
    """The folder exllamav3 converts: the model itself, a materialized copy of a smoothing overlay, or the text
    backbone lifted out (TEXT_EQUIV). Cached next to the source (reused by every bpw rung)."""
    import pollard_stream as S
    plan = exl3_input_plan(S.read_overlay(model_dir)["base"] if S.is_overlay(model_dir) else model_dir)
    if plan["mode"] == "hold":
        return None, plan
    if plan["mode"] == "text":                 # the convert's input only: lifted out of the (overlay) source, one copy
        txt = model_dir.rstrip("/\\") + ".text"
        if not os.path.exists(os.path.join(txt, ".pollard-text-backbone.json")):
            _need_disk(model_dir, txt, text_only=True)
            S.extract_text(model_dir, txt, log=log)
        return txt, plan
    if S.is_overlay(model_dir):
        mat = model_dir.rstrip("/\\") + ".materialized"
        if not os.path.exists(os.path.join(mat, ".pollard-materialized")):
            _need_disk(model_dir, mat)
            from pollard_hf_smooth import materialize
            materialize(model_dir, mat, log=log)
        return mat, plan
    return model_dir, plan


def _need_disk(src, out, text_only=False):
    import shutil
    import pollard_stream as S
    s = S.Source(src)
    if text_only:                              # only the language model's tensors are copied
        spec = S.text_equivalent(src)
        pre = tuple(a for a, _ in spec["renames"]) if spec else ()
        need = sum(s.nbytes(k) for k in s.logical if k.startswith(pre)) * 1.02
    else:
        need = sum(os.path.getsize(p) for p in s.files) * 1.02
    parent = os.path.dirname(os.path.abspath(out)) or "."
    free = shutil.disk_usage(parent).free
    margin = float(os.environ.get("POLLARD_DISK_MARGIN_GB", 5)) * 1e9
    if need + margin > free:
        print(f"HELD: exllamav3 reads files, so {out} needs ~{need / 1e9:.1f} GB (+{margin / 1e9:g} margin); "
              f"{free / 1e9:.1f} GB free")
        sys.exit(EXIT_HELD)


EXIT_HELD = 3


def _rename_safetensors(path, rename):
    """Rewrite one safetensors file's header with renamed keys (the data bytes are copied unchanged)."""
    import json as _j, shutil, struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = _j.loads(f.read(n))
        new = {(k if k == "__metadata__" else rename(k)): v for k, v in hdr.items()}
        if list(new) == list(hdr):
            return False
        hb = _j.dumps(new, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        tmp = path + ".tmp"
        with open(tmp, "wb") as g:
            g.write(struct.pack("<Q", len(hb)))
            g.write(hb)
            f.seek(8 + n)
            shutil.copyfileobj(f, g, 64 << 20)
    os.replace(tmp, path)
    return True


def _rename_json(obj, rename):
    """Rename every dict key that is a tensor / module name (tensor_storage maps, weight_map). Values are left alone:
    weight_map's values are file names ("model.safetensors" is not a module)."""
    if isinstance(obj, dict):
        return {(rename(k) if isinstance(k, str) else k): _rename_json(v, rename) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rename_json(v, rename) for v in obj]
    return obj


def recompose_complete(source_dir, out, log=print):
    """Turn the EXL3 of a wrapper's lifted-out language model (exl3_input mode `text`) into the COMPLETE checkpoint,
    in place: the quantized tensors renamed back to the source's namespace (thinker.model.* / thinker.lm_head),
    the source's own config.json (+ the EXL3 quantization_config, its tensor names renamed the same way), every
    non-text component (vision / audio tower, projector, talker, code2wav, side weights) at source precision, and
    every processor / preprocessor / template file. Exactly what exllamav3 writes when it converts the source
    directly with the architecture present (Qwen3OmniMoeForConditionalGeneration: get_additional_compiled_tensors)."""
    import glob, json as _j
    import pollard_stream as S
    import pollard_complete as PC
    base = S.read_overlay(source_dir)["base"] if S.is_overlay(source_dir) else source_dir
    cfg = S.config_json(base)
    spec = S.TEXT_EQUIV.get(cfg.get("model_type"))
    if not spec:
        raise SystemExit(f"{base}: no text equivalent recorded, nothing to recompose")
    back = [(b, a) for a, b in spec["renames"]]                 # model. -> thinker.model., lm_head. -> thinker.lm_head.

    def rename(k):
        for b, a in back:
            if k.startswith(b) and not k.startswith(a):
                return a + k[len(b):]
            if k == b.rstrip(".") and b.startswith("lm_head"):   # the head's module key has no tensor suffix
                return a.rstrip(".")
        return k
    n = sum(_rename_safetensors(p, rename) for p in glob.glob(os.path.join(out, "*.safetensors")))
    exl_cfg = _j.load(open(os.path.join(out, "config.json"), encoding="utf-8"))
    qcfg = _rename_json(exl_cfg.get("quantization_config") or {}, rename)
    for f in ("quantization_config.json", "model.safetensors.index.json"):
        p = os.path.join(out, f)
        if os.path.isfile(p):
            d = _j.load(open(p, encoding="utf-8"))
            _j.dump(_rename_json(d, rename), open(p, "w", encoding="utf-8"), indent=4)
    # the source's side files (tokenizer, templates, processor configs, custom code) -- then its own config
    S.copy_side_files(base, out)
    full = dict(cfg)
    if qcfg:
        if full.get("quantization_config"):
            qcfg = dict(qcfg, original_quantization_config=full["quantization_config"])
        full["quantization_config"] = qcfg
    _j.dump(full, open(os.path.join(out, "config.json"), "w", encoding="utf-8"), indent=2)
    wrote = PC.fill(source_dir, out, "exl3")
    idx = os.path.join(out, "model.safetensors.index.json")
    if os.path.isfile(idx):                                     # keep the index whole: the sidecars are shards too
        d = _j.load(open(idx, encoding="utf-8"))
        for f in wrote:
            if f.endswith(".safetensors"):
                for k in PC._st_keys(os.path.join(out, f)):
                    d.setdefault("weight_map", {})[k] = f
        _j.dump(d, open(idx, "w", encoding="utf-8"), indent=4)
    res = PC.check(source_dir, out, "exl3")
    log(f"   [complete] {n} shard(s) renamed back to {', '.join(a for _, a in back)}; config = the source's "
        f"({cfg.get('architectures', ['?'])[0]}); added {', '.join(wrote) or 'nothing'}")
    if not res["ok"]:
        raise SystemExit("   [complete] still missing: " + "; ".join(res["missing"]))
    return res


def pack_calib(model, text_path, out_path, rows=256, cols=2048):
    """Tokenize a TEXT corpus with the model's own tokenizer and pack it into the -cd safetensors
    EXL3 wants: {input_ids: int64[rows, cols]}. Real tokens only -- NEVER tile; if the corpus is short,
    emit fewer rows. 256 rows is the measured sweet spot (more shifts the Hessian -> worse allocation)."""
    from transformers import AutoTokenizer
    import torch
    from safetensors.torch import save_file
    tok = AutoTokenizer.from_pretrained(model)
    ids = tok(open(text_path, encoding="utf-8", errors="ignore").read(), return_tensors="pt").input_ids[0]
    if ids.numel() < rows * cols:                              # not enough real tokens -> fewer rows, no tiling
        rows = max(1, ids.numel() // cols)
    ids = ids[:rows * cols].reshape(rows, cols).contiguous().to(torch.long)
    save_file({"input_ids": ids}, out_path)
    return out_path, rows


# ----------------------------------------------------------------- EXL3 recipe bridge (pollard -> -rcp YAML)
#
# ExLlamaV3 (MIT) >= 1.6 converts from a per-tensor bitrate recipe: `convert.py -rcp recipe.yaml`, a YAML with
# `tensors: {<module key>: <int bits 1-8 | 16 | 1.5/2.5/3.5>}` covering EXACTLY the budgeted ("bits") Linears,
# plus target_bpw/achieved_bpw and head_bits (doc/optimize.md; parser in conversion/convert_model.py +
# allocation.create_q_strategy_from_recipe). EXL3's own sc_optimize fills it from noise-injected KLD. This bridge
# fills it from POLLARD's measured allocation, so the two allocators can be compared on EXL3's own atoms at the
# same bpw (notes/exl3-recipe-bridge.md).
#
# Allocation model (the one sc_optimize uses): kld(t, K) = S_t * 4^(p - K) -- error amplitude halves per bit, KL
# is quadratic in it. S_t is the measured group sensitivity (pollard-probe/-sensitivity: attn / ffn per layer, at
# p = probe bits) split across the group's tensors by size. Solved exactly at the bit budget by greedy marginal
# KL per stored bit (convex curves), with EXL3's fused groups tied to one bitrate.

#: EXL3 sc_optimize's default ties: fused GEMMs in its inference paths must share a bitrate
EXL3_TIE = "q_proj+k_proj+v_proj,gate_proj+up_proj,in_proj_qkv+in_proj_z"
_ROUTER_RE = r"(\.(mlp|block_sparse_moe|feed_forward|moe)\.(gate|router)|shared_expert_gate|\.e_score_correction)$"
_NOT_BUDGETED = r"(norm|embed|lm_head|rotary|visual|vision|audio|mm_proj|projector|merger|(^|\.)mtp\.|conv1d)"


def _safetensors_headers(model_dir):
    """{tensor name: shape} from the safetensors headers alone -- no weights read, no deps."""
    import glob, json as _j, struct
    out = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = _j.loads(fh.read(n))
        out.update({k: v["shape"] for k, v in hdr.items() if k != "__metadata__"})
    return out


def _numel(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def exl3_budgeted_keys(model_dir, prefer_exl3=True):
    """[(module key, numel)] of the tensors EXL3 budgets with --bits (qbits_key "bits", qmap set).

    Exact when exllamav3 imports (it builds the model graph from config.json and walks it the way
    create_q_strategy_from_recipe does); otherwise read off the safetensors headers: every 2-D Linear weight in
    a decoder layer, routers / norms / embeddings / towers / MTP excluded, fused expert tensors
    (experts.gate_up_proj / experts.down_proj) expanded per expert -- the names EXL3 gives them."""
    if prefer_exl3:
        try:
            from exllamav3 import Config, Model
            from exllamav3.modules.linear import Linear
            model = Model.from_config(Config.from_directory(model_dir))
            out = []

            def walk(m):
                if isinstance(m, Linear) and m.qmap is not None and m.qbits_key == "bits":
                    out.append((m.key, int(m.weights_numel())))
                for sm in m.modules:
                    walk(sm)
            for m in model.modules:
                walk(m)
            if out:
                return out, "exllamav3"
        except Exception:                                   # noqa: BLE001 -- no CUDA ext here: headers
            pass
    import re as _re
    out = []
    for name, shape in _safetensors_headers(model_dir).items():
        if not name.endswith(".weight") or not _re.search(r"\.layers\.\d+\.", name):
            continue
        key = name[:-len(".weight")]
        if _re.search(_ROUTER_RE, key) or _re.search(_NOT_BUDGETED, key, _re.I):
            continue
        if len(shape) == 2:
            out.append((key, _numel(shape)))
        elif len(shape) == 3 and _re.search(r"experts\.(gate_up_proj|down_proj|gate_proj|up_proj)$", key):
            E, a, b = (int(x) for x in shape)
            base, kind = key.rsplit(".", 1)
            if kind == "gate_up_proj":
                for i in range(E):
                    out += [(f"{base}.{i}.gate_proj", a * b // 2), (f"{base}.{i}.up_proj", a * b // 2)]
            else:
                out += [(f"{base}.{i}.{kind}", a * b) for i in range(E)]
    for name, shape in _safetensors_headers(model_dir).items():          # gate_up_proj stored with .weight-less names
        if _re.search(r"experts\.(gate_up_proj|down_proj)$", name) and len(shape) == 3:
            E, a, b = (int(x) for x in shape)
            base, kind = name.rsplit(".", 1)
            if kind == "gate_up_proj":
                for i in range(E):
                    out += [(f"{base}.{i}.gate_proj", a * b // 2), (f"{base}.{i}.up_proj", a * b // 2)]
            else:
                out += [(f"{base}.{i}.down_proj", a * b) for i in range(E)]
    return out, "safetensors headers"


def _group_of(key):
    """Pollard's sensitivity group for a tensor: sequence mixing ("attn") or channel mixing ("ffn")."""
    import re as _re
    if _re.search(r"(self_attn|\.attn\.|attention|linear_attn|mixer|\.conv\.|token_mix|time_mix)", key):
        return "attn"
    return "ffn"


def _layer_of(key):
    import re as _re
    m = _re.search(r"\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def _ties(keys, tie=EXL3_TIE):
    """Tie groups: keys whose last part is in one tie set AND share a parent module."""
    sets = [set(p.split("+")) for p in tie.split(",") if p.strip()] if tie else []
    groups, index = [], {}
    for k in keys:
        parent, leaf = k.rsplit(".", 1) if "." in k else ("", k)
        tid = next((i for i, st in enumerate(sets) if leaf in st), None)
        if tid is None:
            groups.append([k])
            continue
        gk = (parent, tid)
        if gk not in index:
            index[gk] = len(groups)
            groups.append([])
        groups[index[gk]].append(k)
    return groups


#: below K2 the K2->K1 octave of EXL3's trellis runs steeper than the in-range law (sc_optimize --alpha_low note:
#: effective exponent ~2.35 vs ~2.0 measured on qwen3.8-27b). exllamav3 encodes and decodes K = 1..8
#: (quant/bits_k.cuh, exl3_kernel_map.cu), so K1 is a real width -- this keeps the estimate honest down there.
EXL3_ALPHA_LOW = 2.35


def kld_law(S, K, probe_bits=4, alpha=2.0, alpha_low=EXL3_ALPHA_LOW):
    """sc_optimize's error law kld(t, K) = S_t * 4^(p - K), with the steeper exponent below K2."""
    if K >= 2 or alpha_low is None:
        return S * 2.0 ** (alpha * (probe_bits - K))
    return S * 2.0 ** (alpha * (probe_bits - 2)) * 2.0 ** (alpha_low * (2 - K))


def pollard_recipe(budgeted, sens, target_bpw, head_bits=6, min_bits=2, max_bits=8, tie=EXL3_TIE,
                   probe_bits=None, alpha=2.0, alpha_low=EXL3_ALPHA_LOW):
    """EXL3 recipe dict from Pollard's sensitivity profile at `target_bpw` (same accounting as convert --bits:
    mean bits over the budgeted tensors). min_bits=1 puts the K1 trellis on the table (a MoE's cold experts --
    per-expert traffic is pollard-moefit exl3-recipe's job)."""
    import heapq, math, re as _re
    if not budgeted:
        raise ValueError("no budgeted tensors")
    if not (1 <= min_bits <= max_bits <= 8):
        raise ValueError(f"bit range [{min_bits}, {max_bits}] outside EXL3's K = 1..8")
    if not (min_bits <= target_bpw <= max_bits):
        raise ValueError(f"target {target_bpw} bpw outside [{min_bits}, {max_bits}]")
    numel = dict(budgeted)
    keys = [k for k, _ in budgeted]
    if probe_bits is None:
        m = _re.search(r"(\d+)", str((sens or {}).get("probe", "rtn4")))
        probe_bits = int(m.group(1)) if m else 4
    grp = {g: {int(k): float(v) for k, v in ((sens or {}).get(g) or {}).items()} for g in ("attn", "ffn")}
    allv = sorted(v for d in grp.values() for v in d.values())
    fallback = allv[len(allv) // 2] if allv else 1.0
    gsize = {}
    for k in keys:
        gsize[(_group_of(k), _layer_of(k))] = gsize.get((_group_of(k), _layer_of(k)), 0) + numel[k]
    S = {}
    for k in keys:
        g, li = _group_of(k), _layer_of(k)
        sg = grp[g].get(li, grp["ffn" if g == "attn" else "attn"].get(li, fallback)) if li is not None else fallback
        S[k] = max(sg, 0.0) * numel[k] / max(gsize[(g, li)], 1)
    groups = _ties(keys, tie)
    total = sum(numel.values())
    budget = target_bpw * total
    bits = {i: min_bits for i in range(len(groups))}
    used = min_bits * total
    kl = lambda gi, K: sum(kld_law(S[k], K, probe_bits, alpha, alpha_low) for k in groups[gi])
    heap = []
    for gi in range(len(groups)):
        cost = sum(numel[k] for k in groups[gi])
        if min_bits < max_bits:
            heapq.heappush(heap, (-(kl(gi, min_bits) - kl(gi, min_bits + 1)) / cost, gi))
    while heap:
        _, gi = heapq.heappop(heap)
        cost = sum(numel[k] for k in groups[gi])
        if used + cost > budget + 1e-6:
            continue                                          # this group no longer fits; smaller ones may
        bits[gi] += 1
        used += cost
        if bits[gi] < max_bits:
            K = bits[gi]
            heapq.heappush(heap, (-(kl(gi, K) - kl(gi, K + 1)) / cost, gi))
    tensors = {k: int(bits[gi]) for gi, g in enumerate(groups) for k in g}
    pred = sum(kl(gi, bits[gi]) for gi in range(len(groups)))
    return {"model": None, "source": "pollard", "target_bpw": float(target_bpw),
            "achieved_bpw": round(used / total, 4), "predicted_kld": round(pred, 6), "head_bits": head_bits,
            "tie": tie, "tensors": {k: tensors[k] for k in keys}}


def write_recipe(recipe, path):
    """Plain YAML in sc_optimize's layout; strings JSON-quoted (valid YAML), tensor keys bare (dots only)."""
    import json as _j
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Quantization recipe generated by pollard-exl3 (Pollard measured allocation)\n")
        for k, v in recipe.items():
            if k == "tensors":
                f.write("tensors:\n")
                for key, b in v.items():
                    f.write(f"  {_j.dumps(key) if any(c in key for c in '*?[:# ') else key}: {b}\n")
            elif v is None:
                f.write(f"{k}: null\n")
            elif isinstance(v, str):
                f.write(f"{k}: {_j.dumps(v)}\n")
            else:
                f.write(f"{k}: {v}\n")
    return path


def check_recipe(recipe, budgeted_keys):
    """The checks EXL3's convert applies to -rcp (v1.6.2), as a list of problems ([] = it will load)."""
    import fnmatch, re as _re
    errs = []
    t = recipe.get("tensors") if isinstance(recipe, dict) else None
    if not isinstance(t, dict) or not t:
        return ["Recipe must contain a non-empty 'tensors' mapping"]
    ok = lambda v: (not isinstance(v, bool) and isinstance(v, (int, float))
                    and ((float(v).is_integer() and (1 <= v <= 8 or v == 16)) or float(v) in (1.5, 2.5, 3.5)))
    bad = [k for k, v in t.items() if not ok(v)]
    if bad:
        errs.append(f"bitrates must be integers 1-8, 16 or 1.5/2.5/3.5: {bad[:5]}")
    if not (recipe.get("achieved_bpw") or recipe.get("target_bpw")):
        errs.append("no target_bpw/achieved_bpw")
    budgeted = set(budgeted_keys)
    pats = [(k, _re.compile(fnmatch.translate(k))) for k in t if any(c in k for c in "*?[")]
    exact = {k for k in t if not any(c in k for c in "*?[")}
    missing = [k for k in budgeted if k not in exact and not any(rx.match(k) for _, rx in pats)]
    if missing:
        errs.append(f"missing {len(missing)} budgeted tensors, e.g. {sorted(missing)[:5]}")
    unknown = [k for k in exact if k not in budgeted] + [k for k, rx in pats if not any(rx.match(b) for b in budgeted)]
    if unknown:
        errs.append(f"{len(unknown)} keys match no budgeted tensor, e.g. {unknown[:5]}")
    return errs


def exl3_sc_commands(model, out_dir, bpw, head_bits=6, devices="0"):
    """EXL3's own measured-recipe pipeline (doc/optimize.md, run from an exllamav3 checkout)."""
    w = out_dir.rstrip("/\\") + "_sc"
    return [
        f"python sc_trace.py -m <a 6.0bpw EXL3 of {model}> -o {w}/cal_trace.json -co {w}/cal_trace.safetensors -cs 262144 -ambs 32",
        f"python sc_rfn_probe.py -mq <a 2.0bpw EXL3 of {model}> -mr {model} -d {devices} -o {w}/rfn_2.00bpw.json",
        f"python sc_measure.py -m {model} -d {devices} --shaped -tr {w}/cal_trace.safetensors -rr {w}/rfn_2.00bpw.json -rs 1.0,0.5 -o {w}/noise_attrib.json",
        f"python sc_optimize.py -m {w}/noise_attrib.json -rr {w}/rfn_2.00bpw.json -b {bpw} -hb {head_bits} -al 2.0 -o {w}/recipe_{bpw}bpw.yaml",
    ]


# ----------------------------------------------------------------- the recipe A/B (smoothing x calibration, measured)
#
# The gold recipe (smoothing + Calib 3.0) was established on Qwen2.5-3B at 4bpw by PPL (8.670 vs EXL3 out-of-box 8.699).
# The 2026-10-18 lane qualification measured it by KLD on Qwen2.5-0.5B-Instruct and it LOST to exllamav3's own
# defaults at the same bytes: 0.1610 / 0.0421 vs 0.1205 / 0.0313 at 3.0 / 4.0 bpw (an independent exllamav3 measure,
# wikitext-2). EXL3 already applies its own Hadamard incoherence processing, which is what SmoothQuant's flattening
# is for on the other lanes, and Calib 3.0's profile follows the model's use (agent / chat for an instruct model),
# not the text a KLD is read on. So neither half of the recipe is assumed any more: `--recipe-choice auto` converts
# ONE rung each way, measures each through exllamav3 on the held-out text (pollard_verify.measure_exl3, the shared
# protocol), keeps the winner as that rung, and every later rung of the model builds the winning way.

#: variant -> (convert the SMOOTHED model, calibrate on the --calib-text corpus); False/False = exllamav3 out of box
RECIPE_VARIANTS = {"pollard": (True, True), "raw-calib": (False, True), "smooth-default": (True, False),
                   "plain": (False, False)}
#: the order the A/B runs (and breaks an exact tie in)
AB_ORDER = ("pollard", "raw-calib", "smooth-default", "plain")


def read_ab(path):
    """The recipe A/B record ({winner, variants: {name: {kld, top1, ppl}}, ...}) or None when there is none yet."""
    import json as _j
    try:
        with open(path, encoding="utf-8") as f:
            rec = _j.load(f)
    except (OSError, ValueError, TypeError):
        return None
    return rec if isinstance(rec, dict) and rec.get("winner") in RECIPE_VARIANTS else None


def ab_winner(rows):
    """The variant with the lowest measured KLD (AB_ORDER breaks an exact tie); None when nothing was measured."""
    ok = [(rows[n]["kld"], AB_ORDER.index(n), n) for n in AB_ORDER if n in rows and rows[n].get("kld") is not None]
    return min(ok)[2] if ok else None


def variant_inputs(name, smoothed, raw, cal_data):
    """(the folder to convert, the -cd calibration or None) for a recipe variant."""
    sm, cal = RECIPE_VARIANTS[name]
    return (smoothed if sm else raw), (cal_data if cal else None)


def convert_cmd(a, model, out, work, cal_data, hq):
    """exllamav3's convert_model command for one build."""
    cmd = [sys.executable, "-m", "exllamav3.conversion.convert_model", "-i", model]
    if out:
        cmd += ["-o", out]
    if work:
        cmd += ["-w", work]                                 # exllamav3 REQUIRES a work dir (checkpoints/resume)
    if a.recipe:                                            # explicit Pollard per-tensor plan
        cmd += ["-rcp", a.recipe]
    else:                                                   # budgeted: Pollard intent -> EXL3 knobs
        cmd += ["-b", str(a.bpw), "-hb", str(a.head_bits), "-mb", str(a.mtp_bits)]
        if hq:
            cmd += ["-hq"]                                  # protect select (MoE) layers
    if cal_data:
        cmd += ["-cd", cal_data]
    # Vision/audio tower. exllamav3 exposes -vb and defaults to 6 bpw on towers it has validated
    # (16, i.e. unquantized, on anything else). Pollard was not passing it at all, so the lane was
    # the only one where the user had no say. Default here is "leave exllamav3's architecture
    # default alone"; --vision-bits 16 matches what Pollard does on GGUF, where the projector is
    # never quantized at all.
    if a.vision_bits is not None:
        cmd += ["-vb", str(a.vision_bits)]
    cmd += ["-d", a.devices]
    return cmd


def run_convert(cmd, out, run=None):
    """Run one convert; trust the files, not the return code (convert_model can print an arg error yet exit 0)."""
    r = (run or subprocess.run)(cmd)
    made = out and os.path.isdir(out) and any(f.endswith(".safetensors") for f in os.listdir(out)) and \
        os.path.exists(os.path.join(out, "config.json"))
    if r.returncode != 0 or not made:
        sys.exit(f"exllamav3 convert failed (exit {r.returncode}; output {'present' if made else 'MISSING'}) "
                 f"-- see console above.")


def run_ab(a, smoothed, raw, cal_data, hq, measure=None, run=None, log=print):
    """Convert this rung every RECIPE_VARIANTS way, measure each through exllamav3 against the RAW source on
    --ab-eval, keep the winner AS this rung (moved to --out), drop the rest, write --ab-record. Returns the winner."""
    import json as _j
    import shutil
    if measure is None:
        import pollard_verify as V
        measure = lambda d: V.measure_exl3(d, a.raw_source, a.ab_eval, "cuda:0", None, a.ab_chunks)  # noqa: E731
    rows, dirs = {}, {}
    names = [n for n in AB_ORDER if n in (a.ab_variants or AB_ORDER)]
    for n in names:
        model, cd = variant_inputs(n, smoothed, raw, cal_data)
        d = a.out.rstrip("/\\") + f".ab-{n}"
        dirs[n] = d
        log(f"   [A/B] {n}: {'smoothed' if RECIPE_VARIANTS[n][0] else 'raw'} weights, "
            f"{'Calib 3.0' if cd else 'exllamav3 default'} calibration -> {d}")
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "config.json")):
            run_convert(convert_cmd(a, model, d, d + "_work", cd, hq), d, run)
        try:
            r = measure(d)
        except (Exception, SystemExit) as e:                # noqa: BLE001 -- a variant we cannot measure cannot win
            rows[n] = {"kld": None, "error": f"{type(e).__name__}: {e}"[:300]}
            log(f"   [A/B] {n}: not measured ({rows[n]['error']})")
            continue
        rows[n] = {k: r.get(k) for k in ("kld", "top1", "kld_p99", "ppl", "ref_ppl", "n", "protocol")}
        log(f"   [A/B] {n}: KLD {r['kld']:.5f}  top-1 {r.get('top1') or 0:.2f}%")
    win = ab_winner(rows)
    if win is None:                                         # nothing measured: the gold recipe, and the record says so
        win = "pollard" if "pollard" in dirs else names[0]
        log(f"   [A/B] no variant could be measured -> {win} (unmeasured)")
    rec = {"winner": win, "bpw": a.bpw, "eval": os.path.abspath(a.ab_eval), "source": os.path.abspath(a.raw_source),
           "variants": rows, "variant_inputs": {n: {"smoothed": RECIPE_VARIANTS[n][0],
                                                    "calib_text": RECIPE_VARIANTS[n][1]} for n in rows},
           "metric": "mean KLD vs the raw source through exllamav3 (pollard_verify.measure_exl3), lowest wins",
           "measured": ab_winner(rows) is not None}
    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.replace(dirs[win], a.out)
    for n, d in dirs.items():
        for x in ((d,) if n != win else ()) + (d + "_work",):
            shutil.rmtree(x, ignore_errors=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.ab_record)) or ".", exist_ok=True)
    with open(a.ab_record, "w", encoding="utf-8") as f:
        _j.dump(rec, f, indent=1)
    log(f"   [A/B] winner: {win} -> {a.out}  (record {a.ab_record})")
    return win


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF model dir or id (FP16/BF16 source)")
    ap.add_argument("--out", help="output EXL3 dir (required unless --plan-only)")
    ap.add_argument("--work-dir", help="conversion work dir (checkpoints/resume; default: <out>_work)")
    ap.add_argument("--bpw", type=float, default=3.0, help="target average bits/weight (EXL3 budgeted)")
    ap.add_argument("--head-bits", type=int, default=6, help="output/head layer bits (protected)")
    ap.add_argument("--mtp-bits", type=int, default=4, help="MTP layer bits (GLM/DeepSeek MTP head)")
    ap.add_argument("--recipe", default="none",
                    help="per-tensor bitrate recipe for EXL3's -rcp: 'none' (default, the gold path: EXL3's budgeted "
                         "allocator), 'pollard' (emit one from --sensitivity at --bpw), 'exl3' (EXL3's own sc_optimize "
                         "recipe: --exl3-recipe FILE, or print its pipeline), or a path to a recipe YAML")
    ap.add_argument("--sensitivity", help="Pollard sensitivity.json (pollard-probe/-sensitivity) for --recipe pollard")
    ap.add_argument("--exl3-recipe", help="an sc_optimize recipe.yaml for --recipe exl3")
    ap.add_argument("--recipe-out", help="where --recipe pollard writes its YAML (default <out>.pollard-recipe.yaml)")
    ap.add_argument("--min-bits", type=int, default=2,
                    help="--recipe pollard: lowest per-tensor bitrate (1 = the K1 trellis; exllamav3 encodes and "
                         "decodes K = 1..8). For a MoE use pollard-moefit exl3-recipe: per-expert traffic decides "
                         "which experts take K1")
    ap.add_argument("--max-bits", type=int, default=8, help="--recipe pollard: highest per-tensor bitrate")
    ap.add_argument("--tie", default=EXL3_TIE, help="--recipe pollard: fused groups sharing one bitrate "
                    "(EXL3's default; '' = independent)")
    ap.add_argument("--hq", dest="hq", action="store_true", default=None,
                    help="bump bitrate of select layers (MoE) -- default: auto-on for MoE")
    ap.add_argument("--no-hq", dest="hq", action="store_false")
    ap.add_argument("--cal-data", help="calibration data (safetensors token rows, -cd); else EXL3's bundled mix")
    ap.add_argument("--calib-text", help="a TEXT corpus (e.g. Calib 3.0) -- tokenized+packed to the -cd "
                    "safetensors here, so the locked gold recipe (smoothing + Calib 3.0) runs one-shot")
    ap.add_argument("--cal-rows", type=int, default=256, help="rows to pack from --calib-text (256 = the "
                    "measured EXL3 sweet spot; more is NON-monotonic and can hurt)")
    ap.add_argument("--cal-cols", type=int, default=2048, help="tokens per row for --calib-text packing")
    ap.add_argument("--devices", default="0", help="CUDA device list for the convert, e.g. 0,1")
    ap.add_argument("--vision-bits", type=int, default=None, metavar="N",
                    help="bits for a vision/audio tower (1-8, or 16 to leave it unquantized). "
                         "Default: exllamav3's architecture default -- 6 on towers it has "
                         "validated, 16 otherwise. Measure any change with pollard-mmeval.")
    ap.add_argument("--recipe-choice", default="pollard", choices=["auto"] + list(RECIPE_VARIANTS),
                    help="which inputs this rung converts: pollard (--model as given + --calib-text: the gold recipe), "
                         "raw-calib / smooth-default / plain (exllamav3 out of box) -- or auto: the winner recorded "
                         "in --ab-record, measured by an A/B (--ab-eval) when there is no record yet")
    ap.add_argument("--raw-model", help="the UNSMOOTHED source (when --model is the smoothed copy): the raw variants "
                                        "convert it and the A/B measures against it")
    ap.add_argument("--ab-record", help="the recipe A/B record (JSON) this model's rungs share")
    ap.add_argument("--ab-eval", help="held-out text: run the A/B on THIS rung when --ab-record does not exist yet")
    ap.add_argument("--ab-chunks", type=int, default=8, help="A/B: eval chunks of 2048 tokens")
    ap.add_argument("--ab-variants", type=lambda x: [v.strip() for v in x.split(",") if v.strip()], default=None,
                    help="A/B: comma list of variants to try (default: all four)")
    ap.add_argument("--plan-only", action="store_true", help="print the exllamav3 command, build nothing")
    a = ap.parse_args()
    if a.ab_variants and set(a.ab_variants) - set(RECIPE_VARIANTS):
        ap.error(f"--ab-variants: unknown {sorted(set(a.ab_variants) - set(RECIPE_VARIANTS))}")
    if a.recipe_choice != "pollard" and not a.raw_model:
        if a.recipe_choice == "auto" or not RECIPE_VARIANTS[a.recipe_choice][0]:
            ap.error(f"--recipe-choice {a.recipe_choice} needs --raw-model (the unsmoothed source)")
    a.raw_source = a.raw_model

    # --model is documented as "dir OR id", but exllamav3's converter takes a DIRECTORY: an id went
    # through untouched and died as FileNotFoundError on 'Qwen/Qwen2.5-0.5B-Instruct\\config.json',
    # which reads as a broken model rather than an unresolved name. Every other lane resolves ids, so
    # this one does too.
    if not a.plan_only and not os.path.isdir(a.model) and "/" in a.model:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit(f"--model {a.model!r} is not a directory and huggingface_hub is not "
                             "installed to fetch it; pass a local path") from None
        print(f"   resolving {a.model} from the Hub ...", flush=True)
        import pollard_workspace as ws        # one findable copy in the workspace downloads/, not ~/.cache
        a.model = ws.fetch_source(a.model, allow_patterns=[
            "*.safetensors", "*.json", "*.model", "*.txt"])
        print(f"   -> {a.model}", flush=True)

    a.recorded_as = a.model                                 # the MANIFEST key: the folder the release asked for
    if os.path.isdir(a.model):
        src, plan = exl3_input(a.model) if not a.plan_only else (a.model, exl3_input_plan(a.model))
        if src is None:
            print(f"HELD: {plan['why']}")
            sys.exit(EXIT_HELD)
        if plan["mode"] == "text":
            print(f"   {plan['why']}")
        a.exl3_plan = plan
        if src != a.model and not a.plan_only:
            print(f"   exllamav3 reads {src}")
            a.model = src
    a.raw_input = None
    if a.raw_model and os.path.isdir(a.raw_model):
        a.raw_input = exl3_input(a.raw_model)[0] if not a.plan_only else a.raw_model
    if not a.out and not a.plan_only:                       # no --out -> organized workspace path
        import pollard_workspace as ws
        a.out = ws.resolve_out(a.model, "exl3", tag=f"{a.bpw}bpw")
        print(f"   (no --out) -> workspace: {a.out}")
    # gold recipe: pack a TEXT calib (Calib 3.0) into the -cd safetensors, unless one was given directly
    if a.calib_text and not a.cal_data:
        packed = (a.out.rstrip("/\\") + ".cal.safetensors") if a.out else "pollard_exl3_cal.safetensors"
        if a.plan_only:
            print(f"   [cal] would pack {a.calib_text} -> {packed} ({a.cal_rows}x{a.cal_cols} tokens) for -cd")
            a.cal_data = packed
        else:
            try:
                a.cal_data, rows = pack_calib(a.model, a.calib_text, packed, a.cal_rows, a.cal_cols)
                print(f"   [cal] packed Calib 3.0 -> {a.cal_data} ({rows}x{a.cal_cols} real tokens, -cd)")
            except Exception as e:
                print(f"   [cal] could not pack --calib-text ({repr(e)[:70]}); using EXL3's bundled cal")
    a.recipe_mode = a.recipe if a.recipe in ("none", "pollard", "exl3") else "file"
    if a.recipe_mode == "none":
        a.recipe = None
    elif a.recipe_mode == "pollard":
        if not a.sensitivity or not os.path.exists(a.sensitivity):
            sys.exit("--recipe pollard needs --sensitivity <pollard sensitivity.json> (pollard-probe / -sensitivity)")
        if not os.path.isdir(a.model):
            sys.exit("--recipe pollard reads the checkpoint's tensor list: --model must be a local directory")
        import json as _j
        keys, how = exl3_budgeted_keys(a.model)
        rec = pollard_recipe(keys, _j.load(open(a.sensitivity)), a.bpw, a.head_bits, a.min_bits, a.max_bits,
                             a.tie)
        rec["model"] = a.model
        a.recipe = a.recipe_out or ((a.out.rstrip("/\\") + ".pollard-recipe.yaml") if a.out
                                    else "pollard-exl3-recipe.yaml")
        write_recipe(rec, a.recipe)
        problems = check_recipe(rec, [k for k, _ in keys])
        hist = {}
        for b in rec["tensors"].values():
            hist[b] = hist.get(b, 0) + 1
        print(f"   [recipe] Pollard allocation -> {a.recipe}: {len(keys)} budgeted tensors ({how}), target "
              f"{a.bpw} -> achieved {rec['achieved_bpw']} bpw, bits {dict(sorted(hist.items()))}")
        if problems:
            sys.exit("   [recipe] would not load in EXL3: " + "; ".join(problems))
        if how != "exllamav3":
            print("   [recipe] tensor list read from safetensors headers (exllamav3 not importable here); "
                  "EXL3 aborts on a missing/unknown key, so a mismatch fails fast at convert")
    elif a.recipe_mode == "exl3":
        if not a.exl3_recipe:
            print("   --recipe exl3: EXL3's measured recipe comes from its sc_* pipeline (exllamav3 checkout, GPU):")
            for c in exl3_sc_commands(a.model, a.out or "out", a.bpw, a.head_bits, a.devices):
                print("     $ " + c)
            sys.exit("   then rerun with --recipe exl3 --exl3-recipe <recipe.yaml>")
        a.recipe = a.exl3_recipe
    is_moe, _ = detect_moe(a.model)
    hq = is_moe if a.hq is None else a.hq
    kind = "MoE" if is_moe else "dense"
    # which inputs this rung converts (the recipe A/B): the winner on record, or the A/B itself on this rung
    choice, ab_run = a.recipe_choice, False
    if choice == "auto":
        rec = read_ab(a.ab_record) if a.ab_record else None
        if rec:
            choice = rec["winner"]
            print(f"   recipe: {choice} (the A/B on record, {a.ab_record}: "
                  + ", ".join(f"{n} {v['kld']:.4f}" for n, v in rec["variants"].items() if v.get("kld") is not None)
                  + ")")
        elif a.ab_eval and a.ab_record:
            ab_run = True
        else:
            choice = "pollard"
            print("   recipe: no A/B on record and no --ab-eval to run one -> the gold recipe (pollard)")
    model_in, cal_in = (a.model, a.cal_data) if ab_run else variant_inputs(choice, a.model, a.raw_input, a.cal_data)
    if model_in is None:
        sys.exit(f"--recipe-choice {choice} converts the unsmoothed source: --raw-model {a.raw_model!r} is not a folder")
    work = a.work_dir or ((a.out.rstrip("/\\") + "_work") if a.out else None)
    cmd = convert_cmd(a, model_in, a.out, work, cal_in, hq)

    if a.recipe and a.recipe_mode == "file":
        print(" !! EXPERIMENTAL --recipe FILE: a GGUF/K-quant role map LOSES to EXL3-budgeted (measured: "
              "17.15 vs 14.30 PPL at more bits). Only use a recipe measured in EXL3's atom space.")
    elif a.recipe:
        print(f"   recipe: {a.recipe_mode} (per-tensor -rcp; head {a.head_bits}) -- R&D until the head-to-head "
              "beats the gold path (notes/exl3-recipe-bridge.md)")
    print(f"== pollard-exl3 :: {a.model}  [{kind}]  "
          + (f"recipe {os.path.basename(a.recipe)}" if a.recipe
             else f"{a.bpw} bpw  |  head {a.head_bits}  |  mtp {a.mtp_bits}"
                  + ("  |  hq(MoE)" if hq else "")))
    is_vl = detect_vl(a.model)
    print("   Pollard intent -> EXL3: crush the body to target bpw, protect head/MTP high"
          + (", bump select MoE layers (--hq)" if hq else "") + ".")
    print("   NOTE: EXL3 trellis is the HEAVY lane (hours for a low-bit output) -- that's the format, "
          "not Pollard. Prefer GGUF/GPTQ/MLX unless you need the exllama runtime.")
    if is_vl:
        vb = a.vision_bits
        print("   VISION: vision-language checkpoint. Tower at "
              + (f"{vb} bpw (--vision-bits)" if vb is not None else
                 "exllamav3's architecture default (6 bpw on validated towers, else 16)")
              + ". Use --vision-bits 16 to leave it unquantized, as the GGUF lane does. "
                "Confirm with pollard-mmeval either way.")
    print("   $ " + " ".join(cmd))
    if a.plan_only:
        return
    if not a.out:
        sys.exit("ERROR: --out required unless --plan-only.")
    try:
        import exllamav3  # noqa: F401
    except Exception as e:
        sys.exit(f"ERROR: exllamav3 not usable here ({repr(e)[:80]}). Install it AND make sure its CUDA "
                 f"ext loads (a bleeding-edge GPU may need a source build). Then rerun.")
    if ab_run:
        print(f"   recipe: A/B on this rung ({a.bpw} bpw) -- {', '.join(a.ab_variants or AB_ORDER)}, measured on "
              f"{a.ab_eval}")
        choice = run_ab(a, a.model, a.raw_input, a.cal_data, hq)
        if not RECIPE_VARIANTS[choice][0]:
            model_in = a.raw_input
    else:
        if work:
            os.makedirs(work, exist_ok=True)
        run_convert(cmd, a.out)
    plan = getattr(a, "exl3_plan", None)
    if plan and plan["mode"] == "text":                     # never ship the lifted-out text model alone
        recompose_complete(a.recorded_as, a.out)
    elif plan and os.path.isdir(a.recorded_as):             # a direct convert: anything exllamav3 left out goes beside
        import pollard_complete as PC
        wrote = PC.fill(a.recorded_as, a.out, "exl3")
        if wrote:
            print(f"   [complete] exllamav3 left out {', '.join(wrote)}: added at source precision")
        res = PC.check(a.recorded_as, a.out, "exl3")
        if not res["ok"]:
            sys.exit("   [complete] the EXL3 is missing: " + "; ".join(res["missing"]))
    if plan:
        import json as _j
        with open(os.path.join(a.out, "pollard-exl3.json"), "w", encoding="utf-8") as f:
            _j.dump({"input": plan["mode"], "arch": plan.get("arch"), "why": plan.get("why"),
                     "converted_from": model_in, "recipe": choice,
                     "recipe_ab": os.path.abspath(a.ab_record) if a.ab_record and os.path.exists(a.ab_record) else None,
                     "shipped": "complete",
                     "runtime": "exllamav3 with this architecture" if plan["mode"] == "text" else "exllamav3"},
                    f, indent=1)
    try:
        import pollard_workspace as ws
        ws.record_build(getattr(a, "recorded_as", a.model), "exl3", a.out, tag=f"{a.bpw}bpw", bpw=a.bpw)
    except Exception as e:
        print(f"   (warning: could not record build to the workspace manifest: {e})")
    print(f"wrote EXL3 model -> {a.out}\n  run:  exllamav3 / TabbyAPI loads {a.out}"
          f"\n  VERIFY:  pollard-verify --model {a.out} --source {a.model} --end-to-end")


if __name__ == "__main__":
    main()
