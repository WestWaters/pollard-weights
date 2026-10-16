#!/usr/bin/env python3
"""pollard-draft train -- Pollard's own speculative drafter heads, trained against the QUANTIZED rung people run.

A drafter is only as good as its agreement with the model that verifies it. Published heads are trained against the
bf16 original; the rung people actually serve (Q4_K_M, IQ4_XS...) disagrees with bf16 a little on every token, and
every disagreement is a rejected draft. So Pollard trains its own:

  1. data     prompts from pollard-calib's corpus (chat + tool-call + code + reasoning), rendered through the model's
              own chat template, ANSWERED BY THE QUANTIZED RUNG (llama-server on the GGUF, or any OpenAI-compatible
              server on another lane's rung, or in-process transformers) -- the drafter learns what the rung says
  2. capture  hidden states of the target at the layers the drafter reads -- from bf16 (what the trainers expect)
              or, with --taps rung, from the rung's own weights (GGUF dequantized into the HF model), so the taps
              AND the verifier distribution the head is fit to are the rung's
  3. train    vLLM speculators (Apache-2.0): eagle3 / dflash / dspark, its `file` hidden-state backend, a recipe
              scaled to the model and the card it runs on
  4. convert  the existing `pollard-draft convert` (GGUF ladder + the HF folder vLLM / SGLang load), and
  5. measure  acceptance against the quantized rung AND against bf16, so the card shows both

    pollard-draft train --target Qwen/Qwen3-4B --rung Qwen3-4B-Pollard-Q4_K_M.gguf --rung-tag Q4_K_M \\
                        --kind eagle3 --taps rung --run
    pollard-draft train --target Qwen/Qwen3-0.6B --rung q4.gguf --smoke --run       # Mac: a tiny end-to-end proof

The head is Pollard's (Apache-2.0, as the repo); the card credits the trainer. Safety: a CUDA stage refuses while any
non-ignored process holds the GPU (POLLARD_GPU_IGNORE, shared with the tool-call gate) and a watchdog stops training
if one appears (the run resumes from its checkpoint); memory and disk are checked before anything starts; off CUDA
only a --smoke run (<= 1B target, a few dozen steps) is allowed. POLLARD_HEAVY_LOCK=<dir> makes every heavy stage
wait for that lock directory (mkdir) and release it after.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time

TRAINER = {"name": "speculators", "licence": "Apache-2.0", "url": "https://github.com/vllm-project/speculators",
           "pip": "speculators>=0.8.1"}
#: the licence of a head Pollard trains: the repo's own
HEAD_LICENSE = "apache-2.0"
SOURCE = "pollard-trained"
KINDS = ("eagle3", "dflash", "dspark")
#: pollard-calib domains a drafter learns from, and their weight. Agent turns are as heavy as chat: tool calls are
#: where people run drafters (short, highly predictable JSON) and where a generic chat head is weakest.
DEFAULT_MIX = {"chat": 3, "agent": 3, "codechat": 2, "reasoning": 1}
SMOKE_MIX = {"chat": 1, "agent": 1, "codechat": 1}           # no thinking traces: they blow the smoke budget
SMOKE_MAX_PARAMS_B = 1.0
SMOKE_MAX_STEPS = 200


class Refused(SystemExit):
    pass


# ----------------------------------------------------------------------------- when to train at all

def needs_training(found, prefer_own=False):
    """(train?, why) from `pollard-draft find --out` (drafter.json, or its dict)."""
    if isinstance(found, str):
        try:
            found = json.load(open(found, encoding="utf-8"))
        except (OSError, ValueError):
            found = {}
    pick = (found or {}).get("pick")
    if prefer_own:
        return True, "prefer_own: Pollard trains its own quant-aware head"
    if not pick:
        return True, "no compatible drafter is published"
    if not pick.get("redistributable"):
        return True, f"{pick.get('id')} ({pick.get('license') or 'no licence'}) may not be redistributed"
    return False, f"{pick.get('id')} ({pick.get('license')}) is published and redistributable"


# ----------------------------------------------------------------------------- the recipe

def params_b(cfg):
    """Rough parameter count (billions) from a config.json: the trainer's memory scales with it."""
    t = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) and not cfg.get("hidden_size") else cfg
    h, L = int(t.get("hidden_size") or 0), int(t.get("num_hidden_layers") or 0)
    inter = int(t.get("intermediate_size") or 4 * h)
    v = int(t.get("vocab_size") or 0)
    tied = bool(t.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))
    return round((L * (4 * h * h + 3 * h * inter) + v * h * (1 if tied else 2)) / 1e9, 3)


def default_layer_ids(n_layers):
    """speculators' own default taps ([2, n/2, n-3]); the final layer (n) is appended for the verifier."""
    n = int(n_layers)
    return sorted({min(2, n - 1), n // 2, max(0, n - 3)})


def train_vram_gb(kind, hidden, vocab, seq, draft_layers=1, draft_vocab=32000, inter=None):
    """What one training step holds on the GPU, conservatively: fp32 trainable weights + grads + two optimizer
    moments, the frozen verifier embedding / lm_head (fp32), and per packed token the verifier's full-vocab logits
    (bf16 + an fp32 transient), the draft's logits and layer activations -- x3 for EAGLE-3's test-time-training
    unroll. Measured runs beat this; it exists so a 16GB card gets a shorter sequence instead of an OOM."""
    inter = inter or 4 * hidden
    layer = 4 * hidden * hidden + 2 * hidden * hidden + 3 * hidden * inter       # 2x-wide qkv input (eagle3)
    out_vocab = draft_vocab if kind == "eagle3" else vocab
    trainable = draft_layers * layer + 3 * hidden * hidden + out_vocab * hidden
    frozen = 2 * vocab * hidden
    weights = trainable * 4 * 4 + frozen * 4
    ttt = 3 if kind == "eagle3" else 1
    acts = seq * (vocab * 6 + ttt * (out_vocab * 4 * 3 + hidden * 4 * 12 * draft_layers))
    return round((weights + acts) / 1e9 + 1.5, 1)


def recipe(kind, cfg, smoke=False, device="cuda", vram_gb=None, samples=None, max_steps=None, epochs=None):
    """Hyper-parameters for this model on this machine. Small targets get more epochs; a 16GB card gets a shorter
    packed sequence (and, for the 5-layer DFlash family, fewer draft layers) instead of an OOM."""
    pb = params_b(cfg)
    t = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) and not cfg.get("hidden_size") else cfg
    hidden, vocab = int(t.get("hidden_size") or 0), int(t.get("vocab_size") or 0)
    r = {"kind": kind, "params_b": pb, "lr": 1e-4, "optimizer": "muon", "draft_vocab_size": 32000 if kind == "eagle3"
         else None, "num_layers": None, "attn_impl": None, "loss_impl": None, "num_workers": 2, "log_freq": 10,
         "checkpoint_freq": 1.0, "noise_std": 0.05}
    if smoke:
        r.update(samples=samples or 24, max_new_tokens=96, total_seq_len=1024, epochs=epochs or 4,
                 max_steps=max_steps or 40, mix=dict(SMOKE_MIX), log_freq=1, num_workers=0)
        if kind != "eagle3":
            r["num_layers"] = 1
    else:
        r.update(samples=samples or (8000 if pb <= 4.5 else 12000), max_new_tokens=768, mix=dict(DEFAULT_MIX),
                 epochs=epochs or (10 if pb <= 4.5 else 6 if pb <= 15 else 4), max_steps=max_steps,
                 total_seq_len=8192)
        if vram_gb:
            r["fits"] = False
            for seq in (8192, 4096, 2048, 1536, 1024):
                r["total_seq_len"] = seq
                nl = 1 if kind == "eagle3" else (5 if vram_gb >= 40 else 3 if vram_gb >= 20 else 2)
                if kind != "eagle3":
                    r["num_layers"] = nl
                if train_vram_gb(kind, hidden, vocab, seq, nl, inter=t.get("intermediate_size")) <= vram_gb * 0.9:
                    r["fits"] = True
                    break
    if device != "cuda" or os.name == "nt":
        # flex attention and the fused loss are Triton kernels: no Triton on MPS / CPU, and none in a stock Windows
        # torch -- sdpa + the eager loss run everywhere (slower, same maths)
        r.update(attn_impl="sdpa", loss_impl="eager")
    if device != "cuda":
        r["num_workers"] = 0
    r["vram_est_gb"] = train_vram_gb(kind, hidden, vocab, r["total_seq_len"], r.get("num_layers") or
                                     (1 if kind == "eagle3" else 5), inter=t.get("intermediate_size"))
    return r


def train_argv(python, kind, verifier, data_dir, save_path, layer_ids, rec):
    """`python -m speculators train ...` for the cached-hidden-state (file backend) run. --on-missing raise: every
    sample's taps were captured by Pollard, so a missing one is a bug, not a cue to call a vLLM server."""
    argv = [python, "-m", "speculators", "train", "--speculator-type", kind,
            "--verifier-name-or-path", str(verifier), "--data-path", str(data_dir), "--save-path", str(save_path),
            "--hidden-states-backend", "file", "--on-missing", "raise",
            "--target-layer-ids", *[str(i) for i in layer_ids],
            "--total-seq-len", str(rec["total_seq_len"]), "--epochs", str(rec["epochs"]), "--lr", str(rec["lr"]),
            "--optimizer", rec["optimizer"], "--num-workers", str(rec["num_workers"]),
            "--log-freq", str(rec["log_freq"]), "--checkpoint-freq", str(rec["checkpoint_freq"]),
            "--noise-std", str(rec["noise_std"]), "--save-best"]
    if rec.get("num_workers") == 0:
        argv += ["--prefetch-factor", "2"]
    if rec.get("max_steps"):
        argv += ["--max-steps", str(rec["max_steps"])]
    if rec.get("draft_vocab_size"):
        argv += ["--draft-vocab-size", str(rec["draft_vocab_size"])]
    if rec.get("num_layers"):
        argv += ["--num-layers", str(rec["num_layers"])]
    if rec.get("attn_impl"):
        argv += ["--draft-attn-impl", rec["attn_impl"]]
    if rec.get("loss_impl"):
        argv += ["--loss-implementation", rec["loss_impl"]]
    return argv


# ----------------------------------------------------------------------------- 1. data

def prompt_records(n, mix=None, seed=0, offline=False, rng=None):
    """`n` chat records (messages [+ tools], ending in an assistant turn to regenerate) over pollard-calib's
    domains, by weight. offline: the bundled seeds only (no dataset download)."""
    import pollard_calib as PC
    mix = mix or DEFAULT_MIX
    rng = rng or random.Random(seed)
    total = sum(mix.values()) or 1
    out = []
    for d, w in mix.items():
        need = max(1, round(n * w / total))
        if offline:
            recs = PC._seed_records(d)
            got = []
            while recs and len(got) < need:
                b = recs[:]
                rng.shuffle(b)
                got += b
            recs = got[:need]
        else:
            recs, _ = PC.build_domain(d, need, 40, rng)
        for r in recs:
            if isinstance(r, dict) and any(m.get("role") == "assistant" for m in r.get("messages") or []):
                out.append({**r, "domain": d})
    rng.shuffle(out)
    return out[:n]


def regenerate(records, complete, reasoning_info=None, log=None):
    """Every record's last assistant turn re-answered by the rung. Records the rung could not answer (server error,
    thinking cut off by the budget, empty answer) are DROPPED, not kept with the dataset's reference answer: the
    head must learn what the rung says, nothing else. Returns (records, answered, dropped)."""
    import pollard_calib as PC
    out, _, _ = PC.self_generate(records, complete, log=log, info=reasoning_info)
    kept = [o for o, r in zip(out, records) if o is not r]       # self_generate hands back the SAME dict on failure
    return kept, len(kept), len(records) - len(kept)


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def render_rows(records, renderer, tokenizer, max_len):
    """speculators' training rows: input_ids of the whole conversation in the model's own template, loss_mask = 1 on
    the final assistant turn (where the full render departs from the generation prompt)."""
    rows = []
    for r in records:
        msgs, tools, thinking = r["messages"], r.get("tools"), r.get("thinking")
        full = renderer(msgs, tools, thinking)
        prompt = renderer._try(msgs[:-1], add_generation_prompt=True, tools=tools, **renderer.kwargs(thinking))
        if not full or prompt is None:
            continue
        ids = list(tokenizer(full, add_special_tokens=False)["input_ids"])
        pids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        cut = _common_prefix(ids, pids)
        if len(ids) > max_len:
            drop = len(ids) - max_len
            if drop <= cut - 16:                                  # a long history: drop its oldest tokens, keep
                ids, cut = ids[drop:], cut - drop                 # the answer whole (that is what the head learns)
            else:
                ids = ids[:max_len]
        mask = [0] * min(cut, len(ids)) + [1] * max(0, len(ids) - cut)
        if sum(mask) < 4:
            continue
        rows.append({"input_ids": ids, "loss_mask": mask, "domain": r.get("domain")})
    return rows


def write_dataset(rows, data_dir):
    """The prepared dataset speculators' trainer loads (`datasets` save_to_disk: input_ids, loss_mask, seq_len) and
    token_freq.pt (assistant-token counts, what --draft-vocab-size picks the draft vocabulary from)."""
    import torch
    from datasets import Dataset
    os.makedirs(data_dir, exist_ok=True)
    ds = Dataset.from_dict({"input_ids": [r["input_ids"] for r in rows], "loss_mask": [r["loss_mask"] for r in rows],
                            "seq_len": [len(r["input_ids"]) for r in rows]})
    ds.set_format(type="torch")
    ds.save_to_disk(data_dir)
    freq = {}
    for r in rows:
        for t, m in zip(r["input_ids"], r["loss_mask"]):
            if m:
                freq[t] = freq.get(t, 0) + 1
    torch.save(freq, os.path.join(data_dir, "token_freq.pt"))
    return data_dir


def hs_bytes(rows, n_taps, hidden, dtype_bytes=2):
    """Disk the cached hidden states take: every token x (taps + the final layer) x hidden."""
    return sum(len(r["input_ids"]) for r in rows) * (n_taps + 1) * hidden * dtype_bytes


# ----------------------------------------------------------------------------- 2. capture

#: archs whose GGUF converter permutes q/k rows for llama.cpp's rope layout (convert_hf_to_gguf LlamaModel.permute)
PERMUTED_ARCHS = {"llama"}


def _unpermute(w, n_head):
    """Inverse of convert_hf_to_gguf's LlamaModel.permute."""
    shape = w.shape
    return w.reshape(n_head, shape[0] // n_head // 2, 2, *shape[1:]).swapaxes(1, 2).reshape(shape)


def gguf_state(path):
    """{gguf tensor name: float32 numpy array in HF orientation} and the arch, from a GGUF rung."""
    import numpy as np
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(path)
    f = r.fields.get("general.architecture")
    arch = bytes(f.parts[f.data[0]]).decode() if f else ""
    out = {}
    for t in r.tensors:
        if t.tensor_type.name in ("F32", "F16", "BF16"):
            a = np.array(t.data, dtype=np.float32) if t.tensor_type.name != "BF16" else \
                dequantize(t.data, t.tensor_type).astype(np.float32)
        else:
            a = dequantize(t.data, t.tensor_type).astype(np.float32)
        out[t.name] = a.reshape([int(x) for x in reversed(t.shape)])
    return arch, out


def load_rung_weights(model, gguf_path, state=None):
    """Overwrite the HF model's weights with the GGUF rung's, dequantized: the forward pass then runs the rung's
    weights (activations stay bf16; llama.cpp's own matmul rounding is the only difference left). Refuses when the
    names do not map (an arch gguf-py's tensor map does not know, or an ik-only atom gguf-py cannot decode)."""
    import torch
    from gguf import MODEL_ARCH_NAMES, get_tensor_name_map
    arch, tensors = state or gguf_state(gguf_path)
    inv = {v: k for k, v in MODEL_ARCH_NAMES.items()}
    if arch not in inv:
        raise Refused(f"{gguf_path}: architecture {arch!r} unknown to gguf-py -- use --taps bf16")
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    tmap = get_tensor_name_map(inv[arch], int(cfg.num_hidden_layers))
    sd = model.state_dict()
    done, missing = 0, []
    with torch.no_grad():
        for name, p in sd.items():
            g = tmap.get_name(name, try_suffixes=(".weight", ".bias"))
            if name.endswith("lm_head.weight") and g not in tensors:
                g = "token_embd.weight"                           # tied: the GGUF stores the embedding once
            if not g or g not in tensors:
                missing.append(name)
                continue
            w = tensors[g]
            if arch in PERMUTED_ARCHS and name.endswith(("q_proj.weight", "k_proj.weight")):
                heads = cfg.num_attention_heads if "q_proj" in name else \
                    (getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads)
                w = _unpermute(w, int(heads))
            if tuple(w.shape) != tuple(p.shape):
                if w.size != p.numel():
                    missing.append(name)
                    continue
                w = w.reshape(tuple(p.shape))
            p.copy_(torch.from_numpy(w).to(p.dtype))
            done += 1
    if not done or len(missing) > 0.02 * len(sd):
        raise Refused(f"{gguf_path}: only {done}/{len(sd)} tensors map onto the HF model (e.g. {missing[:3]}) -- "
                      "use --taps bf16")
    return done, missing


def decoder_layers(model):
    """The decoder's layer list (model.model.layers, or a VLM's language model's)."""
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n = int(cfg.num_hidden_layers)
    best = None
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "ModuleList" and len(mod) == n and name.endswith("layers"):
            if best is None or len(name) < len(best[0]):
                best = (name, mod)
    if best is None:
        raise Refused("could not find the decoder layers to tap")
    return best[1]


def capture(rows, model, layer_ids, hs_dir, device, progress=print):
    """hs_<i>.safetensors per row, the file-backend layout speculators reads: token_ids [seq] and hidden_states
    [seq, len(layer_ids) + 1, hidden] -- layer id k = the residual stream entering layer k (vLLM's convention:
    0 = embeddings), and the last slot the final layer's output before the norm (the verifier's own state).
    Resumable: an existing file is kept."""
    import torch
    from safetensors.torch import save_file
    layers = decoder_layers(model)
    n = len(layers)
    order = list(layer_ids) + [n]
    store = {}

    def keep(k):
        def hook(_mod, _inp, out):
            store[k] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        return hook
    hooks = []
    if 0 in order:
        hooks.append(model.get_input_embeddings().register_forward_hook(keep(0)))
    for i, layer in enumerate(layers):
        if i + 1 in order:
            hooks.append(layer.register_forward_hook(keep(i + 1)))
    os.makedirs(hs_dir, exist_ok=True)
    body = model.get_decoder() if hasattr(model, "get_decoder") else model
    t0, wrote = time.time(), 0
    try:
        for idx, row in enumerate(rows):
            path = os.path.join(hs_dir, f"hs_{idx}.safetensors")
            if os.path.exists(path):
                continue
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            store.clear()
            with torch.no_grad():
                body(input_ids=ids, use_cache=False)
            hs = torch.stack([store[k][0] for k in order], dim=1).to(torch.bfloat16).cpu().contiguous()
            save_file({"token_ids": ids[0].cpu().contiguous(), "hidden_states": hs}, path)
            wrote += 1
            if progress and (wrote % 25 == 0 or idx == len(rows) - 1):
                progress(f"     captured {idx + 1}/{len(rows)} ({time.time() - t0:.0f}s)")
    finally:
        for h in hooks:
            h.remove()
    return wrote


def write_verifier_head(model, src_dir, out_dir):
    """The verifier speculators reads embed_tokens / lm_head / model.norm from -- here the RUNG's (dequantized), so
    the distribution the head is fit to is the one it will be verified by. config + tokenizer from the target."""
    import torch
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        if f.endswith((".json", ".txt", ".model", ".jinja", ".tiktoken")) and not f.endswith(".index.json"):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(out_dir, f))
    sd = model.state_dict()
    keep = {}
    for k, v in sd.items():
        for want in ("embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
            if k.endswith(want):
                keep.setdefault(want, v.detach().to(torch.bfloat16).cpu().contiguous())
    if "lm_head.weight" in keep and "embed_tokens.weight" in keep and \
            keep["lm_head.weight"].data_ptr() == keep["embed_tokens.weight"].data_ptr():
        keep["lm_head.weight"] = keep["lm_head.weight"].clone()
    save_file(keep, os.path.join(out_dir, "model.safetensors"))
    return out_dir


# ----------------------------------------------------------------------------- safety

@contextlib.contextmanager
def heavy_lock(path=None, poll=20, log=print):
    """Hold the shared machine lock (a directory; mkdir is atomic) for a heavy stage. POLLARD_HEAVY_LOCK names it."""
    path = path or os.environ.get("POLLARD_HEAVY_LOCK")
    if not path:
        yield
        return
    said = False
    while True:
        try:
            os.mkdir(path)
            break
        except FileExistsError:
            if not said and log:
                log(f"   waiting for the heavy-job lock {path} ...")
                said = True
            time.sleep(poll)
    try:
        yield
    finally:
        shutil.rmtree(path, ignore_errors=True)


def pick_device(want="auto"):
    if want != "auto":
        return want
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def gpu_guard(need_gb=0.0, users=None, free=None):
    """A CUDA stage runs only on a GPU nobody else is using (a game, a desktop session's compute) with room for it."""
    import pollard_toolcall as T
    users = T.gpu_users() if users is None else users
    free = T.gpu_free_gb() if free is None else free
    if free is None:
        raise Refused("no NVIDIA GPU here (nvidia-smi) -- train on the CUDA box, or --smoke on this machine")
    if users:
        raise Refused("the GPU has another process resident (" + ", ".join(users[:3]) + ") -- not ours to take. "
                      "Wait until it is free (POLLARD_GPU_IGNORE names always-on desktop apps to ignore).")
    if need_gb and need_gb > free:
        raise Refused(f"needs ~{need_gb:.1f}GB of VRAM, {free:.1f}GB free")


def preflight(device, smoke, pb, max_steps, samples, need_ram_gb=0.0, need_disk=0, disk_path=".", need_vram_gb=0.0,
              avail_ram=None, free_disk=None, users=None, free_vram=None):
    """Every reason not to start, before anything starts."""
    if device != "cuda":
        if not smoke:
            raise Refused(f"training a drafter on {device} is a smoke test only (--smoke: <= {SMOKE_MAX_PARAMS_B:g}B "
                          f"target, <= {SMOKE_MAX_STEPS} steps) -- the real run belongs on a CUDA GPU")
        if pb > SMOKE_MAX_PARAMS_B:
            raise Refused(f"--smoke off CUDA takes a <= {SMOKE_MAX_PARAMS_B:g}B target; this one is ~{pb:g}B")
        if (max_steps or 0) > SMOKE_MAX_STEPS or samples > 256:
            raise Refused(f"--smoke is capped at {SMOKE_MAX_STEPS} steps / 256 samples off CUDA")
    else:
        gpu_guard(need_vram_gb, users, free_vram)
    if need_ram_gb:
        import pollard_toolcall as T
        try:
            T.check_ram(need_ram_gb, avail_ram)
        except T.NotMeasured as e:
            raise Refused(str(e)) from None
    if need_disk:
        free = free_disk if free_disk is not None else shutil.disk_usage(_existing(disk_path)).free
        want = need_disk * 1.15 + 2e9
        if want > free:
            raise Refused(f"the hidden states need ~{want / 1e9:.0f}GB on {disk_path}, {free / 1e9:.0f}GB free -- "
                          "fewer --samples, or a bigger disk")


def _existing(p):
    p = os.path.abspath(p)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return p


class GpuWatch:
    """Stops a CUDA training process the moment someone else's process appears on the GPU (a game starting). The
    run resumes from its last checkpoint (speculators resumes by default) when started again."""

    def __init__(self, proc, every=30, users=None):
        self.proc, self.every, self.reason = proc, every, None
        self._users = users
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import pollard_toolcall as T
        mine = {str(self.proc.pid)}
        while not self._stop.wait(self.every):
            if self.proc.poll() is not None:
                return
            us = (self._users or T.gpu_users)()
            others = [u for u in us if u not in mine and not u.lower().startswith(("python", "torchrun"))]
            if others:
                self.reason = "GPU claimed by " + ", ".join(others[:3]) + " -- training stopped (resumes from checkpoint)"
                self.proc.terminate()
                return

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


# ----------------------------------------------------------------------------- 3-5. train, convert, measure

def latest_checkpoint(save_path):
    """checkpoint_best when the trainer kept one, else the highest epoch directory with a config.json."""
    best = os.path.join(save_path, "checkpoint_best")
    if os.path.exists(os.path.join(best, "config.json")):
        return os.path.realpath(best)
    eps = [d for d in os.listdir(save_path) if d.isdigit() and os.path.exists(os.path.join(save_path, d, "config.json"))] \
        if os.path.isdir(save_path) else []
    return os.path.join(save_path, max(eps, key=int)) if eps else None


def finalize(ckpt, out_dir, target, provenance):
    """The trained head as Pollard ships it: weights + config, the verifier recorded as the TARGET's id (not the
    local verifier-head dir it trained against), and pollard_train.json saying how it was made."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    for f in os.listdir(ckpt):
        src = os.path.join(ckpt, f)
        if os.path.isfile(src) and (f.endswith((".safetensors", ".json", ".npy")) or f.startswith("LICENSE")):
            shutil.copy2(src, os.path.join(out_dir, f))
    p = os.path.join(out_dir, "config.json")
    cfg = json.load(open(p, encoding="utf-8"))
    ver = (cfg.get("speculators_config") or {}).get("verifier")
    if isinstance(ver, dict):
        ver["name_or_path"] = str(target)
    json.dump(cfg, open(p, "w", encoding="utf-8"), indent=2)
    json.dump(provenance, open(os.path.join(out_dir, "pollard_train.json"), "w", encoding="utf-8"), indent=1)
    return out_dir


def build_meta(prov):
    """What the MANIFEST (and so the card) records about a Pollard-trained head."""
    return {"source": SOURCE, "license": HEAD_LICENSE, "redistributable": True,
            "trained_by": f"{TRAINER['name']} ({TRAINER['licence']})", "trainer_url": TRAINER["url"],
            "quant_aware_vs": prov.get("rung_tag"), "taps": prov.get("taps"), "train_samples": prov.get("samples")}


def _python_for_trainer(explicit=None):
    return explicit or os.environ.get("POLLARD_SPECULATORS_PYTHON") or sys.executable


def has_speculators(python):
    try:
        r = subprocess.run([python, "-c", "import speculators"], capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception:                                            # noqa: BLE001
        return False


def _complete_for(a, renderer, rec):
    """(context manager yielding a completion fn, label): who answers the training prompts."""
    import pollard_calib as PC
    budget = rec["max_new_tokens"]
    kw = dict(max_tokens=budget, template_kwargs=renderer.kwargs, reasoning_max_tokens=budget * 4, timeout=1800)
    if a.regen_url:
        return contextlib.nullcontext(PC.openai_complete(a.regen_url, **kw)), a.regen_url

    if a.regen == "hf":
        return contextlib.nullcontext(None), "in-process"

    @contextlib.contextmanager
    def served():
        import pollard_toolcall as T
        with T.served(a.rung, a.ngl, ctx=max(4096, rec["total_seq_len"]), threads=a.threads,
                      server=a.llama_server, log_path=os.path.join(a.work, "regen-server.log")) as base:
            yield PC.openai_complete(base, **kw)
    return served(), f"llama-server on {os.path.basename(a.rung)}"


def hf_complete(model, tokenizer, renderer, device, max_new_tokens):
    """In-process greedy answers (the transformers path): the record's prompt through the model's own template."""
    import torch

    def complete(messages, tools=None, thinking=None):
        prompt = renderer._try(messages, add_generation_prompt=True, tools=tools, **renderer.kwargs(thinking))
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return {"role": "assistant", "content": text, "_finish": "stop"}
    return complete


def cmd_train(a):
    """The whole pipeline, each stage resumable from what is on disk."""
    import pollard_draft as D
    import pollard_workspace as ws
    say = print
    if a.kind not in KINDS:
        raise Refused(f"--kind {a.kind}: Pollard trains {', '.join(KINDS)}")
    if a.if_needed:
        go, why = needs_training(a.if_needed, a.prefer_own)
        say(f"== pollard-draft train :: {why}")
        if not go:
            say("   not training -- `pollard-draft convert` ships the published head")
            return 0
    key = a.key or a.target
    tdir = a.target_dir or (a.target if os.path.isdir(a.target) else ws.fetch_source(a.target))
    tcfg = json.load(open(os.path.join(tdir, "config.json"), encoding="utf-8"))
    tcfg_t = tcfg.get("text_config") if isinstance(tcfg.get("text_config"), dict) and not tcfg.get("hidden_size") \
        else tcfg
    if a.rung_tag and not a.rung:
        a.rung = rung_from_manifest(a.rung_key or key, a.rung_tag)
    device = pick_device(a.device)
    vram = None
    if device == "cuda":
        import pollard_toolcall as T
        vram = T.gpu_free_gb()
    rec = recipe(a.kind, tcfg, smoke=a.smoke, device=device, vram_gb=vram, samples=a.samples, max_steps=a.max_steps,
                 epochs=a.epochs)
    if a.mix:
        rec["mix"] = {k: float(v) for k, v in (x.split("=") for x in a.mix.split(","))}
    n_layers = int(tcfg_t["num_hidden_layers"])
    layer_ids = [int(x) for x in a.layer_ids.split(",")] if a.layer_ids else default_layer_ids(n_layers)
    a.work = a.work or os.path.join(ws.model_dir(key, create=True), "draft", f"train-{a.kind}")
    os.makedirs(a.work, exist_ok=True)
    data_dir, hs_dir = os.path.join(a.work, "data"), os.path.join(a.work, "data", "hidden_states")
    regen_path = os.path.join(a.work, "regen.jsonl")
    rung_tag = a.rung_tag or (os.path.basename(a.rung) if a.rung else None)
    taps = a.taps if a.rung else "bf16"
    quant_aware = bool(a.rung or a.regen_url)
    say(f"== pollard-draft train :: {a.target} -> {a.kind} head, {rec['samples']} samples, "
        f"{'max ' + str(rec['max_steps']) + ' steps' if rec.get('max_steps') else str(rec['epochs']) + ' epochs'}, "
        f"device {device}, taps {layer_ids}+[{n_layers}] from {'the ' + rung_tag + ' rung' if taps == 'rung' else 'bf16'}")
    say(f"   answers by: {('the quantized rung ' + rung_tag) if a.rung else (a.regen_url or 'the bf16 target')}"
        + ("" if quant_aware else "  -- NOT quant-aware (pass --rung <quantized .gguf>)"))
    say(f"   trainer: {TRAINER['name']} ({TRAINER['licence']}, {TRAINER['url']}); head licence {HEAD_LICENSE}")
    if not a.run:
        say("   recipe: " + json.dumps({k: v for k, v in rec.items() if v is not None}))
        say("   plan only -- add --run (a CUDA box for the real run; --smoke for a tiny local proof)")
        return 0
    pb = rec["params_b"]
    need_disk = int(rec["samples"] * min(rec["total_seq_len"], 2 * rec["max_new_tokens"] + 400) * (len(layer_ids) + 1)
                    * int(tcfg_t["hidden_size"]) * 2)
    preflight(device, a.smoke, pb, rec.get("max_steps"), rec["samples"], need_ram_gb=pb * 2.2 + 2,
              need_disk=need_disk, disk_path=a.work,
              need_vram_gb=max(pb * 2.2 + 1.5, rec["vram_est_gb"]) if device == "cuda" else 0.0)
    py = _python_for_trainer(a.trainer_python)
    if not has_speculators(py):
        raise Refused(f"`{TRAINER['pip']}` is not importable by {py} -- pip install it there (Apache-2.0), or point "
                      "POLLARD_SPECULATORS_PYTHON / --trainer-python at an environment that has it")
    from pollard_calib import Renderer
    renderer = Renderer(tdir, a.trust_remote_code, "auto")
    tok = renderer.tok
    if tok is None:
        raise Refused(f"{tdir}: no tokenizer with a chat template -- cannot render training rows")

    # 1. data: the rung answers pollard-calib's prompts
    if os.path.exists(regen_path) and not a.force:
        records = [json.loads(x) for x in open(regen_path, encoding="utf-8") if x.strip()]
        say(f"   [exists] {len(records)} regenerated answers ({regen_path})")
    else:
        prompts = prompt_records(rec["samples"], rec["mix"], a.seed, a.offline)
        if renderer.rinfo.get("mode") == "toggle":
            prompts = [p if p.get("thinking") else {**p, "thinking": False} for p in prompts]
        ctx, who = _complete_for(a, renderer, rec)
        say(f"   1/5 data: {len(prompts)} prompts ({', '.join(sorted(rec['mix']))}) answered by {who}")
        with heavy_lock(), ctx as complete:
            hf_model = None
            if complete is None:
                hf_model = _load_target(tdir, device, a.rung if taps == "rung" else None, a.trust_remote_code)
                complete = hf_complete(hf_model, tok, renderer, device, rec["max_new_tokens"])
            records, ok, dropped = regenerate(prompts, complete, renderer.rinfo,
                                              log=lambda m: print(m, file=sys.stderr))
            del hf_model
        with open(regen_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({**r, "answered_by": rung_tag or who}, ensure_ascii=False) + "\n")
        say(f"      {ok} answered, {dropped} dropped (failed / cut off -- never kept with a reference answer)")
    rows = render_rows(records, renderer, tok, rec["total_seq_len"])
    if not rows:
        raise Refused("no training rows survived rendering")
    if not os.path.exists(os.path.join(data_dir, "dataset_info.json")) or a.force:
        if a.force and os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
        write_dataset(rows, data_dir)
    say(f"      {len(rows)} rows, {sum(sum(r['loss_mask']) for r in rows)} supervised tokens -> {data_dir}")

    # 2. capture
    verifier = tdir
    with heavy_lock():
        if device == "cuda":
            gpu_guard()
        model = _load_target(tdir, device, a.rung if taps == "rung" else None, a.trust_remote_code)
        say(f"   2/5 capture: {len(rows)} sequences, taps {layer_ids} + final, from "
            f"{'the ' + rung_tag + ' rung (GGUF weights dequantized)' if taps == 'rung' else 'bf16'}")
        capture(rows, model, layer_ids, hs_dir, device)
        if taps == "rung":
            verifier = write_verifier_head(model, tdir, os.path.join(a.work, "verifier-" + _slug(rung_tag)))
        del model
        _free(device)

    # 3. train
    save_path = os.path.join(a.work, "ckpt")
    argv = train_argv(py, a.kind, verifier, data_dir, save_path, layer_ids, rec)
    say("   3/5 train: $ " + " ".join(argv))
    env = dict(os.environ)
    if device == "mps":
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with heavy_lock():
        if device == "cuda":
            gpu_guard(rec["vram_est_gb"])
        log = open(os.path.join(a.work, "train.log"), "a", encoding="utf-8")
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=a.work)
        watch = GpuWatch(proc) if device == "cuda" else contextlib.nullcontext()
        with watch:
            rc = proc.wait()
        log.close()
    if getattr(watch, "reason", None):
        raise Refused(watch.reason)
    if rc != 0:
        raise SystemExit(f"speculators train exited {rc}; see {os.path.join(a.work, 'train.log')}")
    ckpt = latest_checkpoint(save_path)
    if not ckpt:
        raise SystemExit(f"no checkpoint under {save_path}")
    prov = {"tool": "pollard-draft train", "kind": a.kind, "target": a.target, "rung": a.rung and os.path.basename(a.rung),
            "rung_tag": rung_tag if quant_aware else None, "answers_by": rung_tag if a.rung else (a.regen_url or "bf16"),
            "taps": taps, "target_layer_ids": layer_ids, "samples": len(rows), "mix": rec["mix"],
            "recipe": {k: v for k, v in rec.items() if k not in ("mix",) and v is not None},
            "trainer": TRAINER, "license": HEAD_LICENSE, "device": device, "smoke": bool(a.smoke),
            "checkpoint": os.path.relpath(ckpt, a.work), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    final = finalize(ckpt, os.path.join(a.work, "head"), a.target, prov)
    say(f"      trained head -> {final}")

    # 4. convert (GGUF ladder + HF folder) with the existing lane step
    import argparse
    gg = D.gguf_convertible(D.repo_config(final), a.kind)
    ca = argparse.Namespace(target=a.target, target_dir=tdir, from_=None, drafter=final, kind=a.kind, rungs=a.rungs,
                            out_dir=a.out_dir, key=key, python=a.python, no_gguf=(gg is False) or a.no_gguf,
                            no_hf=False, mlx=False, mlx_bits=8, force=True, meta=build_meta(prov))
    say("   4/5 convert" + ("" if not ca.no_gguf else " -- HF folder only: llama.cpp's converter does not read a "
                             f"speculators-format {a.kind} head (eagle3 / dspark heads convert)"))
    D.cmd_convert(ca)

    # 5. measure vs the quantized rung AND vs bf16
    targets = [(t.split("=", 1)[0], t.split("=", 1)[1]) for t in (a.measure_target or [])]
    if a.rung and not any(p == a.rung for _, p in targets):
        targets.insert(0, (rung_tag, a.rung))
    if not targets:
        say("   5/5 measure: no target GGUF given (--measure-target TAG=path) -- the release's draft/rungs steps do it")
        return 0
    res = a.results or os.path.join(a.work, "results-draft.json")
    for b in D.draft_builds(key):
        if b.get("source") != SOURCE or b.get("publish") is False or b.get("format") not in ("gguf", "hf"):
            continue
        if b.get("format") == "hf" and not b.get("via"):
            continue
        ma = argparse.Namespace(target=[p for _, p in targets], target_tag=[t for t, _ in targets],
                                drafter=b["path"], via=b.get("via") if b.get("format") == "hf" else None,
                                kind=a.kind, rung=b.get("tag"), results=res, backend="gguf", limit=a.limit,
                                max_tokens=a.max_tokens, n_max=None, ngl=a.ngl, ngld=None, ctx=4096,
                                threads=a.threads, llama_server=a.llama_server, vllm=None, think=False,
                                baseline=False, keep_rows=False, hw=None, redistributable=True)
        say(f"   5/5 measure {b.get('tag')} vs {', '.join(t for t, _ in targets)}")
        with heavy_lock():
            D.cmd_measure(ma)
    say(f"   results -> {res}")
    return 0


def _slug(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s or "rung"))


def _free(device):
    import gc
    gc.collect()
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:                                            # noqa: BLE001
        pass


def _load_target(tdir, device, rung_gguf=None, trust_remote_code=False):
    import torch
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(tdir, dtype=dtype, trust_remote_code=trust_remote_code)
    if rung_gguf:
        n, miss = load_rung_weights(model, rung_gguf)
        print(f"     {n} tensors from {os.path.basename(rung_gguf)} (dequantized)"
              + (f"; kept bf16: {', '.join(miss[:4])}" if miss else ""))
    return model.to(device).eval()


def rung_from_manifest(key, tag):
    """The GGUF rung `tag` of this model's release (pollard-workspace MANIFEST)."""
    import pollard_workspace as ws
    keys = [key]
    try:
        keys.append(ws.model_slug(key))
    except Exception:                                            # noqa: BLE001
        pass
    for k in keys:
        try:
            builds = ws.read_manifest(k).get("builds", [])
        except Exception:                                        # noqa: BLE001
            continue
        for b in builds:
            if b.get("lane") == "gguf" and str(b.get("tag", "")).lower() == str(tag).lower() and b.get("path"):
                return b["path"]
    raise Refused(f"no GGUF rung tagged {tag} in the MANIFEST of {key} -- pass --rung <path>")


def add_args(t):
    """`pollard-draft train` flags (shared with the recipe printer in pollard_draft)."""
    t.add_argument("--target-dir", help="local checkout of the target (default: fetched)")
    t.add_argument("--key", help="MANIFEST key (default --target)")
    t.add_argument("--rung", help="the QUANTIZED target rung (.gguf): it answers the prompts (and gives the taps)")
    t.add_argument("--rung-tag", help="its tag (Q4_K_M ...); alone, the rung is looked up in the MANIFEST")
    t.add_argument("--rung-key", help="MANIFEST key holding the GGUF rungs (default --key)")
    t.add_argument("--regen-url", help="an OpenAI-compatible server on another lane's rung answers instead")
    t.add_argument("--regen", choices=["gguf", "hf"], default="gguf",
                   help="who answers without --rung/--regen-url: hf = in-process transformers")
    t.add_argument("--taps", choices=["bf16", "rung"], default="rung",
                   help="hidden states + verifier distribution from the rung's weights (default) or bf16")
    t.add_argument("--layer-ids", help="target layers to tap, comma list (default speculators': 2, n/2, n-3)")
    t.add_argument("--samples", type=int)
    t.add_argument("--max-steps", type=int)
    t.add_argument("--mix", help="domain weights, e.g. chat=3,agent=3,codechat=2,reasoning=1")
    t.add_argument("--offline", action="store_true", help="pollard-calib's bundled seed prompts only (no download)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    t.add_argument("--smoke", action="store_true", help="a tiny end-to-end proof (the only run allowed off CUDA)")
    t.add_argument("--work", help="working dir (default <workspace>/models/<target>/draft/train-<kind>)")
    t.add_argument("--rungs", default="Q8_0,Q4_K_M", help="GGUF quants of the trained head")
    t.add_argument("--out-dir", help="where the head's GGUF ladder goes (default <workspace>/models/<target>/draft)")
    t.add_argument("--no-gguf", action="store_true")
    t.add_argument("--python", help="interpreter for convert_hf_to_gguf.py")
    t.add_argument("--trainer-python", help="interpreter that has speculators (default POLLARD_SPECULATORS_PYTHON)")
    t.add_argument("--measure-target", action="append", help="TAG=path.gguf to measure against (repeatable); the "
                                                             "rung is always measured -- add the BF16/F16 too")
    t.add_argument("--results", help="results-draft.json (default <work>/results-draft.json)")
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--max-tokens", type=int, default=128)
    t.add_argument("--ngl", default="0", help="llama-server GPU layers for answers + measuring (default 0)")
    t.add_argument("--threads", type=int)
    t.add_argument("--llama-server", help="llama-server binary")
    t.add_argument("--if-needed", help="drafter.json from `find`: train only if nothing redistributable was found")
    t.add_argument("--prefer-own", action="store_true", help="with --if-needed: train even when one was found")
    t.add_argument("--trust-remote-code", action="store_true")
    t.add_argument("--force", action="store_true", help="redo every stage")


def main(argv=None):
    """`python pollard_draft_train.py ...` == `pollard-draft train ...` (the Studio runs tools by module)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pollard_draft as D
    return D.main(["train"] + list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
