#!/usr/bin/env python3
"""pollard-forge -- make your OWN foundation model, then ship it through the rest of Pollard.

The same recipe the big labs use for their small open models (NVIDIA's Minitron/Nemotron line, the distilled
"mini" models): start from scratch, or carve a smaller model out of a bigger one, train it, distill the bigger one
back into it, and publish. Every step writes a normal Hugging Face checkpoint, so the model then runs through
`pollard --hf <dir> --run` (GGUF ladder, card, upload) like any other.

    pollard-forge new   --family qwen3 --size 350m --tokenizer Qwen/Qwen3-0.6B --out my-350m
    pollard-forge prune --teacher Qwen/Qwen3-4B --calib calib.txt --keep-layers 0.75 --ffn 0.6 --out my-2b
    pollard-forge train --model my-2b --teacher Qwen/Qwen3-4B --kd 0.7 \\
                        --data hf:HuggingFaceFW/fineweb-edu@0.6 --data chat.jsonl@0.4 --tokens 2e9 --out my-2b
    pollard-forge card  --model my-2b --name "My-2B" --license apache-2.0
    pollard --hf my-2b --run                      # quantize the ladder + card, like any model

new    -- a fresh architecture (qwen3 / llama / qwen2 family) at a size preset or exact dims; the tokenizer is
          copied from any model, or trained on your corpus (--train-tokenizer).
prune  -- Minitron-style structured pruning from a teacher: drop the layers that change the hidden state least
          (block influence, measured on --calib) and the FFN channels that fire least. The student keeps the
          teacher's tokenizer, so it can be distilled from it.
train  -- pretraining, continued pretraining, or distillation (--teacher: CE + temperature-scaled KL on the
          teacher's logits). --data mixes sources by weight; keep chat data in the mix for an instruct model
          (QAT/distill on raw web text alone erases chat behaviour -- measured on STQ1_0). Checkpoints every
          --save-every steps and resumes with --resume. Multi-GPU: torchrun --nproc_per_node N pollard_forge.py train ...
card   -- README.md with the lineage (every forge step, its data and its measured losses) and the license.

Every step appends to <dir>/forge.json, so a published model carries exactly how it was made.
"""
import argparse, json, math, os, random, shutil, sys, time

SIZES = {   # (layers, hidden, heads, kv_heads, ffn) -- standard shapes at roughly these non-embedding sizes
    "60m": (8, 512, 8, 4, 1536), "125m": (12, 768, 12, 4, 2048), "350m": (24, 1024, 16, 4, 2816),
    "1b": (24, 2048, 16, 8, 5632), "3b": (36, 2560, 32, 8, 9728), "7b": (32, 4096, 32, 8, 14336),
}
FAMILIES = ("qwen3", "llama", "qwen2")


def _torch():
    try:
        import torch, transformers  # noqa: F401
        return torch
    except ImportError:
        sys.exit("pollard-forge needs torch + transformers (+ datasets for hf: data):  pip install 'pollard-weights[forge]'")


def _device(want="auto"):
    torch = _torch()
    if want != "auto":
        return want
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _lineage(d, entry):
    p = os.path.join(d, "forge.json")
    log = json.load(open(p)) if os.path.exists(p) else {"tool": "pollard-forge", "steps": []}
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
    log["steps"].append(entry)
    json.dump(log, open(p, "w"), indent=1)
    return log


def _carry_lineage(src, dst):
    s = os.path.join(src, "forge.json") if os.path.isdir(src) else None
    if s and os.path.exists(s) and os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(s, os.path.join(dst, "forge.json"))


def _params(model):
    return sum(p.numel() for p in model.parameters())


def _save(model, tok, out, dtype="bf16"):
    """Write the checkpoint in `dtype` (bf16 halves the files) while the live model keeps its fp32 master weights."""
    torch = _torch()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    sd = {k: (v.detach().to(dt) if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    old = getattr(model.config, "dtype", None)
    model.config.dtype = dt
    model.save_pretrained(out, state_dict=sd)
    model.config.dtype = old
    if tok is not None:
        tok.save_pretrained(out)


# ----------------------------- new: a fresh architecture -----------------------------
def build_config(family, layers, hidden, heads, kv, ffn, vocab, ctx=4096, tie=None, rope_theta=1e6):
    from transformers import AutoConfig
    if family not in FAMILIES:
        sys.exit(f"--family must be one of {', '.join(FAMILIES)}")
    if hidden % heads or heads % kv:
        sys.exit(f"hidden ({hidden}) must divide by heads ({heads}), and heads by kv-heads ({kv})")
    tie = (hidden <= 1024) if tie is None else tie        # small models tie embeddings (the vocab dominates them)
    kw = dict(num_hidden_layers=layers, hidden_size=hidden, num_attention_heads=heads, num_key_value_heads=kv,
              intermediate_size=ffn, vocab_size=vocab, max_position_embeddings=ctx, tie_word_embeddings=tie,
              rope_theta=rope_theta, rms_norm_eps=1e-6)
    if family == "qwen3":
        kw["head_dim"] = hidden // heads
    return AutoConfig.for_model(family, **kw)


def train_tokenizer(corpus, vocab, out):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    files = [corpus] if os.path.isfile(corpus) else [os.path.join(corpus, f) for f in sorted(os.listdir(corpus))]
    tok.train(files, tr)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|endoftext|>",
                                   additional_special_tokens=special[1:])
    fast.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    fast.save_pretrained(out)
    return fast


def cmd_new(a):
    torch = _torch()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    os.makedirs(a.out, exist_ok=True)
    if a.train_tokenizer:
        tok = train_tokenizer(a.train_tokenizer, a.vocab, a.out)
        tok_src = f"trained BPE on {a.train_tokenizer} (vocab {a.vocab})"
    elif a.tokenizer:
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        tok.save_pretrained(a.out)
        tok_src = a.tokenizer
    else:
        sys.exit("give --tokenizer <model id or dir> to reuse one, or --train-tokenizer <corpus> to train one")
    L, H, nh, kv, ffn = SIZES[a.size] if a.size else (None,) * 5
    L, H, nh, kv, ffn = a.layers or L, a.hidden or H, a.heads or nh, a.kv_heads or kv, a.ffn or ffn
    if None in (L, H, nh, kv, ffn):
        sys.exit("give --size (" + ", ".join(SIZES) + ") or all of --layers --hidden --heads --kv-heads --ffn")
    vocab = int(math.ceil(len(tok) / 64) * 64)               # pad to 64: faster matmuls, room for added tokens
    cfg = build_config(a.family, L, H, nh, kv, ffn, vocab, a.ctx, a.tie, a.rope_theta)
    cfg.bos_token_id = tok.bos_token_id
    cfg.eos_token_id = tok.eos_token_id
    torch.manual_seed(a.seed)
    model = AutoModelForCausalLM.from_config(cfg)
    _save(model, None, a.out, a.save_dtype)
    n = _params(model)
    _lineage(a.out, {"op": "new", "family": a.family, "layers": L, "hidden": H, "heads": nh, "kv_heads": kv, "ffn": ffn,
                     "vocab": vocab, "ctx": a.ctx, "params": n, "tokenizer": tok_src, "seed": a.seed})
    print(f"== pollard-forge new :: {a.out}\n   {a.family}, {L} layers x {H} hidden, {nh}/{kv} heads, ffn {ffn}, vocab {vocab}"
          f"\n   {n/1e6:.1f}M parameters, randomly initialised -- next: pollard-forge train --model {a.out} --data ...")


# ----------------------------- prune: carve a smaller model out of a teacher -----------------------------
def _decoder_layers(model):
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj, path
        except AttributeError:
            continue
    sys.exit("prune supports decoder-only models with model.layers (llama / qwen2 / qwen3 / mistral style)")


def measure_importance(model, tok, texts, device, seq=512):
    """Per layer: block influence = 1 - mean cos(h_in, h_out) (ShortGPT): a layer that barely turns the hidden
    state is the cheapest to remove. Per layer and FFN channel: mean |activation| entering down_proj (Minitron)."""
    torch = _torch()
    layers, _ = _decoder_layers(model)
    bi = [0.0] * len(layers); n_tok = 0
    act = [None] * len(layers)
    hooks = []
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "down_proj"):
            def h(mod, inp, i=i):
                x = inp[0].detach().float().abs().sum(dim=(0, 1))
                act[i] = x if act[i] is None else act[i] + x
            hooks.append(mlp.down_proj.register_forward_pre_hook(h))
    model.eval()
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=seq).input_ids.to(device)
            if ids.shape[1] < 8:
                continue
            hs = model(ids, output_hidden_states=True).hidden_states
            for i in range(len(layers)):
                a_, b_ = hs[i][0].float(), hs[i + 1][0].float()
                bi[i] += float((1 - torch.nn.functional.cosine_similarity(a_, b_, dim=-1)).sum())
            n_tok += ids.shape[1]
    for h in hooks:
        h.remove()
    if not n_tok:
        sys.exit("--calib produced no usable text (need lines/paragraphs of at least a few words)")
    return [b / n_tok for b in bi], [(x / n_tok).cpu() if x is not None else None for x in act]


def prune_model(model, bi, act, keep_layers, ffn_frac, protect_ends=True):
    """Drop the lowest-influence layers and keep the top FFN channels by activation. Edits the model in place."""
    torch = _torch()
    layers, path = _decoder_layers(model)
    L = len(layers)
    n_keep = max(1, round(L * keep_layers)) if keep_layers <= 1 else int(keep_layers)
    order = sorted(range(L), key=lambda i: bi[i])                         # least influential first
    fixed = {0, L - 1} if protect_ends and L > 2 else set()
    drop = [i for i in order if i not in fixed][:L - n_keep]
    keep = [i for i in range(L) if i not in drop]
    cfg = model.config
    if ffn_frac < 1:
        new_ffn = max(64, int(round(cfg.intermediate_size * ffn_frac / 64)) * 64)
        for i in keep:
            mlp = layers[i].mlp
            idx = torch.topk(act[i], new_ffn).indices.sort().values if act[i] is not None else torch.arange(new_ffn)
            for name in ("gate_proj", "up_proj"):
                old = getattr(mlp, name)
                lin = torch.nn.Linear(old.in_features, new_ffn, bias=old.bias is not None, dtype=old.weight.dtype, device=old.weight.device)
                lin.weight.data = old.weight.data[idx].clone()
                if old.bias is not None:
                    lin.bias.data = old.bias.data[idx].clone()
                setattr(mlp, name, lin)
            old = mlp.down_proj
            lin = torch.nn.Linear(new_ffn, old.out_features, bias=old.bias is not None, dtype=old.weight.dtype, device=old.weight.device)
            lin.weight.data = old.weight.data[:, idx].clone()
            if old.bias is not None:
                lin.bias.data = old.bias.data.clone()
            mlp.down_proj = lin
        cfg.intermediate_size = new_ffn
    new = torch.nn.ModuleList([layers[i] for i in keep])
    for j, layer in enumerate(new):                                       # KV-cache slots follow the new depth
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "layer_idx"):
            attn.layer_idx = j
    parent = model
    for part in path.split(".")[:-1]:
        parent = getattr(parent, part)
    setattr(parent, path.split(".")[-1], new)
    cfg.num_hidden_layers = len(keep)
    for key in ("layer_types",):                                          # per-layer lists (sliding window, etc.)
        v = getattr(cfg, key, None)
        if isinstance(v, list) and len(v) == L:
            setattr(cfg, key, [v[i] for i in keep])
    if getattr(cfg, "max_window_layers", None) is not None:
        cfg.max_window_layers = min(cfg.max_window_layers, len(keep))
    return keep, drop


def cmd_prune(a):
    torch = _torch()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = _device(a.device)
    tok = AutoTokenizer.from_pretrained(a.teacher)
    model = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.float32 if dev != "cuda" else torch.bfloat16).to(dev)
    before = _params(model)
    texts = [t for t in open(a.calib, encoding="utf-8", errors="replace").read().split("\n\n") if t.strip()][:a.calib_samples]
    print(f"== pollard-forge prune :: {a.teacher} -> {a.out}  (measuring on {len(texts)} calib samples, {dev})", flush=True)
    bi, act = measure_importance(model, tok, texts, dev, a.seq)
    keep, drop = prune_model(model, bi, act, a.keep_layers, a.ffn, not a.allow_ends)
    os.makedirs(a.out, exist_ok=True)
    _save(model, tok, a.out, a.save_dtype)
    _carry_lineage(a.teacher, a.out)
    after = _params(model)
    _lineage(a.out, {"op": "prune", "teacher": a.teacher, "params_before": before, "params_after": after,
                     "dropped_layers": sorted(drop), "kept_layers": keep, "ffn": model.config.intermediate_size,
                     "block_influence": [round(x, 5) for x in bi], "calib": os.path.basename(a.calib), "calib_samples": len(texts)})
    print(f"   layers {len(bi)} -> {len(keep)} (dropped {sorted(drop)}), ffn -> {model.config.intermediate_size}")
    print(f"   {before/1e6:.0f}M -> {after/1e6:.0f}M parameters ({100*after/before:.0f}%)")
    print(f"   a pruned model needs healing: pollard-forge train --model {a.out} --teacher {a.teacher} --kd 0.7 --data ...")


# ----------------------------- train: pretrain / continue / distill -----------------------------
def parse_source(spec):
    """'path.txt' | 'path.jsonl' | 'hf:org/name[:config][:split]' with an optional '@weight'."""
    w = 1.0
    if "@" in spec.rsplit("/", 1)[-1]:
        spec, ws = spec.rsplit("@", 1)
        w = float(ws)
    if spec.startswith("hf:"):
        parts = spec[3:].split(":")
        return {"kind": "hf", "name": parts[0], "config": parts[1] if len(parts) > 1 and parts[1] else None,
                "split": parts[2] if len(parts) > 2 else "train", "weight": w, "spec": spec}
    if not os.path.exists(spec):
        sys.exit(f"--data {spec}: no such file (use hf:org/name for a Hub dataset)")
    return {"kind": "jsonl" if spec.endswith((".jsonl", ".json")) else "text", "path": spec, "weight": w, "spec": spec}


def _texts(src, tok, rank=0, world=1):
    """Endless stream of training strings from one source (chat rows rendered with the chat template)."""
    def render(row):
        if isinstance(row, str):
            return row
        if row.get("messages") and getattr(tok, "chat_template", None):
            return tok.apply_chat_template(row["messages"], tokenize=False)
        for k in ("text", "content", "document"):
            if isinstance(row.get(k), str):
                return row[k]
        return None
    epoch = 0
    while True:
        if src["kind"] == "hf":
            from datasets import load_dataset
            ds = load_dataset(src["name"], src["config"], split=src["split"], streaming=True)
            ds = ds.shuffle(seed=epoch, buffer_size=10_000)
            it = (r for i, r in enumerate(ds) if i % world == rank)
        elif src["kind"] == "jsonl":
            it = (json.loads(l) for i, l in enumerate(open(src["path"], encoding="utf-8")) if l.strip() and i % world == rank)
        else:
            it = (p for i, p in enumerate(open(src["path"], encoding="utf-8", errors="replace").read().split("\n\n")) if p.strip() and i % world == rank)
        n = 0
        for row in it:
            t = render(row)
            if t:
                n += 1
                yield t
        if not n:
            sys.exit(f"--data {src['spec']}: no usable text (expects text / content / messages fields)")
        epoch += 1


def batches(sources, tok, seq, batch, seed, rank=0, world=1):
    """Mix sources by weight, tokenize, pack into (batch, seq+1) blocks."""
    torch = _torch()
    rng = random.Random(seed + rank)
    streams = [_texts(s, tok, rank, world) for s in sources]
    weights = [s["weight"] for s in sources]
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    buf = []
    while True:
        rows = []
        while len(rows) < batch:
            while len(buf) < seq + 1:
                buf.extend(tok(next(rng.choices(streams, weights)[0]), add_special_tokens=False).input_ids + [eos])
            rows.append(buf[:seq + 1]); buf = buf[seq + 1:]
        yield torch.tensor(rows, dtype=torch.long)


def kd_loss(student_logits, teacher_logits, T):
    torch = _torch()
    s = torch.nn.functional.log_softmax(student_logits.float() / T, dim=-1)
    t = torch.nn.functional.log_softmax(teacher_logits.float() / T, dim=-1)
    return torch.nn.functional.kl_div(s, t, log_target=True, reduction="batchmean") * (T * T) / student_logits.shape[1]


def cmd_train(a):
    torch = _torch()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0"))
    ddp = world > 1
    if ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        local = int(os.environ.get("LOCAL_RANK", "0"))
        dev = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
    else:
        dev = _device(a.device)
    main_proc = rank == 0
    out = a.out or a.model
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    bf16 = dev.startswith("cuda") and torch.cuda.is_bf16_supported()
    state_p = os.path.join(out, "forge-train-state.pt")
    load_from = out if (a.resume and os.path.exists(state_p)) else a.model
    model = AutoModelForCausalLM.from_pretrained(load_from, dtype=torch.float32).to(dev)
    if a.grad_checkpointing:
        model.gradient_checkpointing_enable()
    teacher = None
    if a.teacher:
        teacher = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16 if bf16 else torch.float32).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        tv, sv = teacher.config.vocab_size, model.config.vocab_size
        ttok = AutoTokenizer.from_pretrained(a.teacher)
        if ttok.get_vocab() != tok.get_vocab():
            sys.exit("--teacher must share the student's tokenizer (prune from it, or `new --tokenizer <teacher>`)")
    sources = [parse_source(s) for s in a.data]
    tok_per_step = a.seq * a.batch * a.accum * world
    steps = a.steps or max(1, int(float(a.tokens) // tok_per_step))
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    warm = max(1, int(steps * a.warmup)) if a.warmup < 1 else int(a.warmup)
    lr_at = lambda s: a.lr * (s + 1) / warm if s < warm else \
        a.lr * (a.min_lr + (1 - a.min_lr) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    start = 0
    if a.resume and os.path.exists(state_p):
        st = torch.load(state_p, map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"]); start = st["step"]
        if main_proc:
            print(f"   resumed at step {start}", flush=True)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev] if dev.startswith("cuda") else None)
    core = model.module if ddp else model
    eval_src = [parse_source(a.eval_data)] if a.eval_data else sources
    eval_batches = list(next(batches(eval_src, tok, a.seq, a.batch, 12345)) for _ in range(a.eval_batches))
    data = batches(sources, tok, a.seq, a.batch, a.seed + start, rank, world)
    log_p = os.path.join(out, "forge-train.jsonl")
    if main_proc:
        print(f"== pollard-forge train :: {a.model} -> {out}  ({dev}{', bf16' if bf16 else ''}{f', {world} ranks' if ddp else ''})")
        print(f"   {_params(core)/1e6:.1f}M params, {steps} steps x {tok_per_step:,} tokens = {steps*tok_per_step/1e6:.1f}M tokens"
              + (f", distilling from {a.teacher} (kd {a.kd}, T {a.temperature})" if teacher else ""), flush=True)

    def evaluate():
        core.eval(); tot = 0.0
        with torch.no_grad():
            for b in eval_batches:
                b = b.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    logits = core(b[:, :-1], use_cache=False).logits
                tot += float(torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), b[:, 1:].flatten()))
        core.train()
        return tot / max(1, len(eval_batches))

    first_eval = evaluate() if main_proc and start == 0 else None
    t0, seen, last = time.time(), 0, {}
    model.train()
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ce_acc = kd_acc = 0.0
        for micro in range(a.accum):
            b = next(data).to(dev)
            x, y = b[:, :-1], b[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                logits = model(x, use_cache=False).logits
                ce = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), y.flatten())
                loss = ce
                if teacher is not None:
                    with torch.no_grad():
                        tl = teacher(x, use_cache=False).logits
                    kd = kd_loss(logits, tl, a.temperature)
                    loss = (1 - a.kd) * ce + a.kd * kd
                    kd_acc += float(kd.detach()) / a.accum
            (loss / a.accum).backward()
            ce_acc += float(ce.detach()) / a.accum
            seen += x.numel() * world
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip))
        opt.step(); opt.zero_grad(set_to_none=True)
        if not math.isfinite(ce_acc):
            sys.exit(f"loss went non-finite at step {step} -- lower --lr or raise --warmup")
        if main_proc and (step % a.log_every == 0 or step == steps - 1):
            last = {"step": step + 1, "ce": round(ce_acc, 4), "lr": lr_at(step), "grad_norm": round(gn, 3),
                    "tokens": (step + 1) * tok_per_step, "tok_s": round(seen / max(1e-9, time.time() - t0))}
            if teacher is not None:
                last["kd"] = round(kd_acc, 4)
            if a.eval_every and (step + 1) % a.eval_every == 0:
                last["eval"] = round(evaluate(), 4)
            open(log_p, "a").write(json.dumps(last) + "\n")
            print("   " + "  ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in last.items()), flush=True)
        if main_proc and a.save_every and (step + 1) % a.save_every == 0 and step + 1 < steps:
            _save(core, tok, out, "fp32")              # mid-run checkpoints stay fp32 so --resume is exact
            torch.save({"opt": opt.state_dict(), "step": step + 1}, state_p)
    if main_proc:
        final_eval = evaluate()
        _save(core, tok, out, a.save_dtype)
        if os.path.exists(state_p):
            os.remove(state_p)
        if os.path.abspath(out) != os.path.abspath(a.model):
            _carry_lineage(a.model, out)
        _lineage(out, {"op": "distill" if teacher else "train", "from": a.model, "teacher": a.teacher, "kd": a.kd if teacher else None,
                       "data": [{"source": s["spec"], "weight": s["weight"]} for s in sources], "steps": steps,
                       "tokens": steps * tok_per_step, "seq": a.seq, "lr": a.lr, "eval_loss_start": first_eval,
                       "eval_loss_end": round(final_eval, 4), "eval_ppl_end": round(math.exp(min(final_eval, 50)), 2),
                       "device": dev, "ranks": world})
        print(f"   eval loss {first_eval if first_eval is None else round(first_eval, 4)} -> {final_eval:.4f}"
              f"  (ppl {math.exp(min(final_eval, 50)):.2f})  saved {out}")
    if ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


# ----------------------------- card -----------------------------
def _n(x):
    return next((f"{x/d:,.1f}{u}" for d, u in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")) if x >= d), f"{x:,.0f}")


def _src(spec):
    """A data source as a public card should show it: Hub ids as-is, local files by name only (no home paths)."""
    return spec if spec.startswith("hf:") else os.path.basename(spec)


def cmd_card(a):
    p = os.path.join(a.model, "forge.json")
    if not os.path.exists(p):
        sys.exit(f"{a.model} has no forge.json -- card documents a model made with pollard-forge")
    log = json.load(open(p))
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    steps = log["steps"]
    bases = sorted({s.get("teacher") for s in steps if s.get("teacher")})
    data = []
    for s in steps:
        for d in s.get("data") or []:
            if d["source"] not in data:
                data.append(d["source"])
    if not a.license:
        sys.exit("--license is required: the card states it, and Pollard never invents one. A pruned or distilled model "
                 "must respect its teacher's license" + (f" ({', '.join(bases)})" if bases else "") + ".")
    tokens = sum(s.get("tokens") or 0 for s in steps)
    n_params = next((s.get("params_after") or s.get("params") for s in reversed(steps) if s.get("params_after") or s.get("params")), None)
    last_eval = next((s for s in reversed(steps) if s.get("eval_loss_end") is not None), None)
    fm = ["---", f"license: {a.license}", "library_name: transformers", "pipeline_tag: text-generation",
          "tags:", "- pollard", "- pollard-forge"]
    if bases:
        fm += ["base_model:"] + [f"- {b}" for b in bases] + ["base_model_relation: finetune"]
    hf_data = [d[3:].split(":")[0] for d in data if d.startswith("hf:")]
    if hf_data:
        fm += ["datasets:"] + [f"- {d}" for d in hf_data]
    fm.append("---")
    lines = fm + ["", f"# {a.name}", "", a.description or f"A {cfg.get('model_type')} model built with pollard-forge.", "",
                  "## Model", "", "| | |", "|---|---|", f"| architecture | {cfg.get('model_type')} |",
                  f"| layers | {cfg.get('num_hidden_layers')} |", f"| hidden | {cfg.get('hidden_size')} |",
                  f"| attention heads (kv) | {cfg.get('num_attention_heads')} ({cfg.get('num_key_value_heads')}) |",
                  f"| FFN | {cfg.get('intermediate_size')} |", f"| vocabulary | {cfg.get('vocab_size')} |",
                  f"| context | {cfg.get('max_position_embeddings')} |"]
    if n_params:
        lines.append(f"| parameters | {_n(n_params)} |")
    if tokens:
        lines.append(f"| training tokens | {_n(tokens)} |")
    if last_eval:
        lines.append(f"| held-out loss / perplexity | {last_eval['eval_loss_end']} / {last_eval['eval_ppl_end']} |")
    lines += ["", "## How it was made", "", "| step | what | details |", "|---|---|---|"]
    for i, s in enumerate(steps, 1):
        if s["op"] == "new":
            d = f"{s['layers']}L x {s['hidden']}, {_n(s['params'])} params, tokenizer {s['tokenizer']}"
        elif s["op"] == "prune":
            d = (f"from {s['teacher']}: {_n(s['params_before'])} -> {_n(s['params_after'])}, "
                 f"dropped layers {s['dropped_layers']}, FFN {s['ffn']}")
        else:
            d = (f"{_n(s['tokens'])} tokens from " + ", ".join(f"{_src(x['source'])} ({x['weight']:g})" for x in s["data"])
                 + (f"; distilled from {s['teacher']} (kd {s['kd']})" if s.get("teacher") else "")
                 + f"; held-out loss {s.get('eval_loss_start') if s.get('eval_loss_start') is None else round(s['eval_loss_start'], 3)} -> {s['eval_loss_end']}")
        lines.append(f"| {i} | {s['op']} | {d} |")
    lines += ["", "## Use", "", "```python", "from transformers import AutoTokenizer, AutoModelForCausalLM",
              f'tok = AutoTokenizer.from_pretrained("{a.repo or a.name}")',
              f'model = AutoModelForCausalLM.from_pretrained("{a.repo or a.name}")', "```", "",
              "GGUF builds for llama.cpp, Ollama and LM Studio: `pollard --hf <this repo> --run`.", "",
              f"License: {a.license}." + (f" Derived from {', '.join(bases)}; their terms apply too." if bases else ""), ""]
    open(os.path.join(a.model, "README.md"), "w").write("\n".join(lines))
    print(f"== pollard-forge card :: {a.model}/README.md  ({len(steps)} forge step(s), license {a.license})")
    print(f"   next: pollard --hf {a.model} --run     then upload the model and its GGUF ladder")


def main():
    ap = argparse.ArgumentParser(prog="pollard-forge", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="a fresh architecture, randomly initialised")
    n.add_argument("--family", default="qwen3", choices=FAMILIES)
    n.add_argument("--size", choices=list(SIZES)); n.add_argument("--layers", type=int); n.add_argument("--hidden", type=int)
    n.add_argument("--heads", type=int); n.add_argument("--kv-heads", type=int); n.add_argument("--ffn", type=int)
    n.add_argument("--ctx", type=int, default=4096); n.add_argument("--rope-theta", type=float, default=1e6)
    n.add_argument("--tie", action=argparse.BooleanOptionalAction, default=None, help="tie input/output embeddings (default: yes up to 1024 hidden)")
    n.add_argument("--tokenizer", help="copy the tokenizer of this model id or dir")
    n.add_argument("--train-tokenizer", metavar="CORPUS", help="train a byte-level BPE on this file or directory instead")
    n.add_argument("--vocab", type=int, default=32000); n.add_argument("--seed", type=int, default=0)
    n.add_argument("--save-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    n.add_argument("--out", required=True); n.set_defaults(fn=cmd_new)
    p = sub.add_parser("prune", help="Minitron-style: a smaller model carved out of a teacher")
    p.add_argument("--teacher", required=True); p.add_argument("--calib", required=True, help="text file; blank-line separated samples")
    p.add_argument("--keep-layers", type=float, default=0.75, help="fraction (<=1) or count (>1) of layers to keep")
    p.add_argument("--ffn", type=float, default=1.0, help="fraction of FFN channels to keep (default 1 = depth only)")
    p.add_argument("--allow-ends", action="store_true", help="let the first and last layer be dropped too")
    p.add_argument("--calib-samples", type=int, default=128); p.add_argument("--seq", type=int, default=512)
    p.add_argument("--save-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="auto"); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_prune)
    t = sub.add_parser("train", help="pretrain / continue / distill")
    t.add_argument("--model", required=True); t.add_argument("--out", help="default: train in place")
    t.add_argument("--data", action="append", required=True, metavar="SRC[@W]",
                   help="file.txt | file.jsonl (text or messages) | hf:org/name[:config][:split]; repeat to mix, @weight to weight")
    t.add_argument("--eval-data", metavar="SRC", help="held-out source (default: a fixed sample of --data)")
    t.add_argument("--teacher", help="distill from this model (same tokenizer)"); t.add_argument("--kd", type=float, default=0.5)
    t.add_argument("--temperature", type=float, default=2.0)
    t.add_argument("--tokens", default="1e8", help="training budget in tokens (default 1e8); or --steps")
    t.add_argument("--steps", type=int); t.add_argument("--seq", type=int, default=2048); t.add_argument("--batch", type=int, default=8)
    t.add_argument("--accum", type=int, default=1); t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--min-lr", type=float, default=0.1, help="cosine floor as a fraction of --lr")
    t.add_argument("--warmup", type=float, default=0.02, help="fraction (<1) or steps"); t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--clip", type=float, default=1.0); t.add_argument("--grad-checkpointing", action="store_true")
    t.add_argument("--log-every", type=int, default=10); t.add_argument("--eval-every", type=int, default=0)
    t.add_argument("--eval-batches", type=int, default=4); t.add_argument("--save-every", type=int, default=0)
    t.add_argument("--resume", action="store_true"); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--save-dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="final checkpoint dtype")
    t.add_argument("--device", default="auto"); t.set_defaults(fn=cmd_train)
    c = sub.add_parser("card", help="README.md with the full lineage")
    c.add_argument("--model", required=True); c.add_argument("--name", required=True); c.add_argument("--license")
    c.add_argument("--repo", help="Hub repo id for the usage snippet"); c.add_argument("--description")
    c.set_defaults(fn=cmd_card)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
