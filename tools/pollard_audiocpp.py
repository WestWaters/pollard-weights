#!/usr/bin/env python3
"""pollard-audiocpp -- the Pollard lane for audio models that run on audio.cpp (github.com/0xShug0/audio.cpp).

audio.cpp runs 130+ audio families (TTS, ASR, voice conversion, separation, music) from its own GGUF packages
(`general.architecture = audiocpp`, the family's spec embedded). Its converter re-quantizes any of its GGUFs with
exact per-tensor rules (`audiocpp_gguf --input <gguf> --type <base> --keep-type <name>=<type>`, first match wins),
and each family's spec already keeps the tensors its maintainers found must stay at source precision (speaker
encoders, codecs, vocoders, convs, norms). What nobody publishes is a MEASURED mix. This lane makes one:

  eligible  the tensors the family itself quantizes: convert the source once at q8_0 and diff the types
  probe     per class (a layer / block of eligible tensors), push only that class low and score the model with
            audio.cpp itself -- ASR: transcript character error vs the source model's own transcripts; TTS:
            synthesize fixed sentences, transcribe them with a fixed ASR model, character error vs the text plus
            the |log| duration ratio vs the source audio (catches silence and runaway generation)
  plan      Pollard's exact budgeted allocation (pollard_exlr8.knapsack) over the measured costs; refuses to plan
            a class it has no measurement for -- never a guessed bit width
  build     audiocpp_gguf with the plan as exact-name rules (no wildcards)

    pollard-audiocpp eligible --source orig.gguf --family kokoro_tts --out eligible.json
    pollard-audiocpp probe --source orig.gguf --family kokoro_tts --eligible eligible.json --task tts \\
        --texts sentences.txt --asr-family citrinet_asr --asr-model citrinet.gguf --out sensitivity.json
    pollard-audiocpp plan --eligible eligible.json --sensitivity sensitivity.json --bpw 4.5 --out rules.json
    pollard-audiocpp build --source orig.gguf --family kokoro_tts --rules rules.json --out rung.gguf
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

#: the types audiocpp_gguf writes, highest first, with their bits per weight (ggml block layouts)
LADDER = ("q8_0", "q6_k", "q5_k", "q4_k", "q3_k", "q2_k")
BPW = {"orig": 16.0, "bf16": 16.0, "f16": 16.0, "f32": 32.0, "q8_0": 8.5, "q6_k": 6.5625, "q5_k": 5.5,
       "q4_k": 4.5, "q3_k": 3.4375, "q2_k": 2.625}
#: the type a probe pushes one class to (the bottom of the ladder: the class's worst case)
PROBE_TYPE = "q2_k"


# ------------------------------------------------------------------------------------------------ tools

def find_bin(name):
    """audio.cpp's binaries: $POLLARD_AUDIOCPP_BIN (dir), then runtime/audio.cpp/build/bin, then PATH."""
    for d in (os.environ.get("POLLARD_AUDIOCPP_BIN"),
              os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime", "audio.cpp",
                           "build", "bin")):
        if d and os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    p = shutil.which(name)
    if not p:
        raise SystemExit(f"{name} not found: set POLLARD_AUDIOCPP_BIN to audio.cpp's binary folder")
    return p


#: ggml type ids -> names (the ones audio.cpp writes plus the float sources)
GGML_TYPES = {0: "f32", 1: "f16", 8: "q8_0", 10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6_k",
              30: "bf16", 35: "tq2_0", 36: "tq2_1"}
_KV_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def tensor_types(path):
    """{name: (type name, n_elements, row length)} from a GGUF's header. Metadata values are skipped, never decoded: audio.cpp
    packages embed their spec, tokenizer and voice sidecars there, and gguf-py's reader takes minutes and gigabytes
    on them. Names are the logical ones `--keep-type` matches (`audiocpp.tensor_names`), not the physical aliases."""
    import struct
    with open(path, "rb") as f:
        rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF")
        rd("I")
        n_t, n_kv = rd("Q"), rd("Q")

        def skip(vt):
            if vt in _KV_FIXED:
                f.seek(_KV_FIXED[vt], 1)
            elif vt == 8:
                f.seek(rd("Q"), 1)
            elif vt == 9:
                et, n = rd("I"), rd("Q")
                if et in _KV_FIXED:
                    f.seek(_KV_FIXED[et] * n, 1)
                else:
                    for _ in range(n):
                        skip(et)
            else:
                raise SystemExit(f"{path}: unknown GGUF value type {vt}")
        logical = None
        for _ in range(n_kv):
            key = f.read(rd("Q")).decode()
            vt = rd("I")
            if key == "audiocpp.tensor_names" and vt == 9:
                rd("I")
                logical = [f.read(rd("Q")).decode() for _ in range(rd("Q"))]
            else:
                skip(vt)
        out = {}
        for i in range(n_t):
            name = f.read(rd("Q")).decode()
            if logical and i < len(logical):
                name = logical[i]             # long or duplicate names are stored as `_audiocpp.<i>` aliases
            dims = [rd("Q") for _ in range(rd("I"))]
            t = rd("I")
            rd("Q")
            out[name] = (GGML_TYPES.get(t, f"type{t}"), math.prod(dims), dims[0] if dims else 1)
    return out


def convert(source, family, out, base, rules=(), extra=()):
    """audiocpp_gguf: `source` (a GGUF or the family's weights) -> `out` at `base`, `rules` = [(name, type)] exact."""
    cmd = [find_bin("audiocpp_gguf"), "--input", source, "--output", out, "--type", base, "--overwrite"]
    if family:
        cmd += ["--family", family]
    for name, typ in rules:
        cmd += ["--keep-type", f"{name}={typ}"]
    r = subprocess.run(cmd + list(extra), capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError(f"audiocpp_gguf failed ({r.returncode}): {(r.stderr or r.stdout)[-600:]}")
    return out


# ------------------------------------------------------------------------------------------------ eligible + classes

def eligible_from(src_types, q8_types, ref_types=None):
    """Tensors the family quantizes: the ones a q8_0 conversion wrote as q8_0 (norms and biases that only change
    float width stay out), plus every tensor the family's own published quantized package (`ref_types`, e.g. the
    maintainers' q8_0 GGUF, which their prep scripts build with explicit overrides) carries quantized."""
    quantized = lambda t: t not in ("f32", "f16", "bf16")
    keep = {n for n, v in q8_types.items() if v[0] == "q8_0"}
    keep |= {n for n, v in (ref_types or {}).items() if quantized(v[0])}
    return {n: {"numel": numel, "row": row, "source": t} for n, (t, numel, row) in src_types.items()
            if n in keep and not quantized(t)}


def fit_type(t, row):
    """The type a tensor actually takes for a requested `t`: k-quant blocks are 256 wide, q8_0 blocks 32, so a
    tensor whose rows don't divide stays at q8_0 (audio.cpp refuses the override otherwise)."""
    return t if t == "q8_0" or row % 256 == 0 else "q8_0"


def class_of(name):
    """A tensor's class: its name up to and including the first numeric component (a layer / block), else the name.
    `thinker.layers.12.mlp.experts.gate_up` -> `thinker.layers.12`; `kokoro/decoder.decode.0.conv1.weight_v` ->
    `kokoro/decoder.decode.0`; `kokoro/predictor.lstm.weight_hh_l0` -> `kokoro/predictor.lstm`."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p.isdigit():
            return ".".join(parts[:i + 1])
    return ".".join(parts[:-1]) or name         # no layer index: the parent module (an LSTM's four matrices)


def classes(eligible):
    out = {}
    for n, v in eligible.items():
        c = out.setdefault(class_of(n), {"tensors": [], "numel": 0, "rows": {}})
        c["tensors"].append(n)
        c["numel"] += v["numel"]
        c["rows"][n] = [v["numel"], v["row"]]
    for c in out.values():
        c["tensors"].sort()
    return out


# ------------------------------------------------------------------------------------------------ scoring

def cer(ref, hyp):
    """Character error rate (edit distance / len(ref)) on lower-cased, whitespace-normalised text."""
    a, b = " ".join(ref.lower().split()), " ".join(hyp.lower().split())
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


def parse_cli(stdout):
    """audiocpp_cli's per-request output: {request_id: {"text": ..., "audio_duration_ms": ...}}."""
    out, cur = {}, None
    for line in stdout.splitlines():
        m = re.match(r"^request_id=(.+)$", line.strip())
        if m:
            cur = out.setdefault(m.group(1), {})
            continue
        m = re.match(r"^text_output=(.*)$", line.strip())
        if m and cur is not None:
            cur["text"] = m.group(1)
            continue
        m = re.match(r"^metrics\[(.+?)\]\.audio_duration_ms=([0-9.]+)", line.strip())
        if m:
            out.setdefault(m.group(1), {})["audio_duration_ms"] = float(m.group(2))
    return out


def default_backend():
    """audio.cpp has no auto: Metal on macOS, else CUDA when an NVIDIA driver answers, else the CPU."""
    if sys.platform == "darwin":
        return "metal"
    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def run_cli(task, family, model, requests, out_dir, backend=None, extra=()):
    os.makedirs(out_dir, exist_ok=True)
    req = os.path.join(out_dir, "requests.json")
    json.dump({"requests": requests}, open(req, "w"))
    cmd = [find_bin("audiocpp_cli"), "--task", task, "--family", family, "--model", model, "--backend", backend or default_backend(),
           "--request-sequence", req, "--out-dir", out_dir, "--metrics", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"audiocpp_cli {task} failed ({r.returncode}): {(r.stderr or r.stdout)[-600:]}")
    return parse_cli(r.stdout)


def score_asr(model, family, clips, ref_texts, work, backend):
    """Mean CER of `model`'s transcripts of `clips` against `ref_texts` (the source model's own transcripts)."""
    reqs = [{"id": f"c{i}", "audio": os.path.abspath(c)} for i, c in enumerate(clips)]
    got = run_cli("asr", family, model, reqs, work, backend)
    texts = [got.get(f"c{i}", {}).get("text", "") for i in range(len(clips))]
    return sum(cer(r, h) for r, h in zip(ref_texts, texts)) / max(len(clips), 1), texts


def score_tts(model, family, sentences, asr_family, asr_model, work, backend, ref_durations=None, voice=None,
              language=None):
    """TTS round trip: synthesize, transcribe with a fixed ASR model, mean CER vs the text; plus the mean |log|
    duration ratio against the source model's audio (silence and runaway generation are what quantization breaks)."""
    reqs = [{"id": f"s{i}", "text": s, **({"voice": voice} if voice else {}), **({"language": language} if language else {})}
            for i, s in enumerate(sentences)]
    got = run_cli("tts", family, model, reqs, work, backend)
    wavs = [os.path.join(work, f"s{i}.wav") for i in range(len(sentences))]
    durs = [got.get(f"s{i}", {}).get("audio_duration_ms") for i in range(len(sentences))]
    err, _ = score_asr(asr_model, asr_family, wavs, sentences, os.path.join(work, "asr"), backend)
    drift = 0.0
    if ref_durations:
        ds = [abs(math.log(d / r)) for d, r in zip(durs, ref_durations) if d and r]
        drift = sum(ds) / len(ds) if ds else 0.0
    return err + drift, durs


# ------------------------------------------------------------------------------------------------ plan

def plan(cls, sens, bpw, types=LADDER):
    """Exact budgeted allocation over measured classes. `sens` = {class: {type: cost}} (cost of that class at that
    type, everything else at source). A type with no measurement is interpolated between measured ones by bits;
    a class with no measurement at all is refused. Returns {class: type}."""
    import pollard_exlr8 as X
    missing = [c for c in cls if not sens.get(c)]
    if missing:
        raise SystemExit(f"{len(missing)} class(es) have no measurement ({missing[:3]}...): probe them first -- "
                         "Pollard never assigns a guessed width")
    items, keys = [], sorted(cls)
    for c in keys:
        meas = {t: float(v) for t, v in sens[c].items() if t in BPW}
        meas.setdefault(types[0], 0.0)                          # the top rung costs nothing relative to the source
        pts = sorted((BPW[t], v) for t, v in meas.items())
        opts = []
        for t in types:
            b = BPW[t]
            if t in meas:
                cost = meas[t]
            else:                                               # linear in bits between the nearest measurements
                lo = max((p for p in pts if p[0] <= b), default=pts[0])
                hi = min((p for p in pts if p[0] >= b), default=pts[-1])
                cost = lo[1] if hi[0] == lo[0] else lo[1] + (hi[1] - lo[1]) * (b - lo[0]) / (hi[0] - lo[0])
            size = class_bytes(cls[c], t)
            opts.append((size, max(0.0, cost) + size * 1e-12, t))   # equal cost -> the smaller type
        items.append(opts)
    numel = sum(c["numel"] for c in cls.values())
    budget = int(numel * bpw / 8)              # bpw over the eligible tensors (the protected rest is fixed)
    floor = sum(min(o[0] for o in opts) for opts in items)
    if budget < floor:                         # rows that only divide by 32 can't go below q8_0
        raise SystemExit(f"--bpw {bpw} is below this model's floor of {floor * 8 / numel:.3f} bpw over its eligible "
                         "tensors")
    res = X.knapsack(items, budget)
    return dict(zip(keys, res["choice"])), res


def class_bytes(c, t):
    return sum(int(math.ceil(numel * BPW[fit_type(t, row)] / 8)) for numel, row in c["rows"].values())


def class_rules(c, t):
    return [(n, fit_type(t, c["rows"][n][1])) for n in c["tensors"]]


def rules_for(cls, choice):
    """Exact-name rules, one per eligible tensor (first match wins in audiocpp_gguf, so no prefix can shadow)."""
    return [r for c in sorted(cls) for r in class_rules(cls[c], choice[c])]


# ------------------------------------------------------------------------------------------------ card + publish

def card_md(repo, family, base_model, rungs, stock=None, task="tts", texts=None, judge=None):
    """The repo card: what runs it, then every rung with its MEASURED numbers. `rungs` / `stock` are `score` results
    (bytes, delta, source_roundtrip) plus the rung's file name and label."""
    name = repo.split("/")[-1]
    metric = ("TTS round trip: the rung speaks held-out sentences, a fixed ASR model transcribes them -- character "
              "error vs the text plus |log| duration drift vs the source model's audio"
              if task == "tts" else "character error of the rung's transcripts vs the source model's")
    lines = [f"---\nbase_model: {base_model}\nlibrary_name: audio.cpp\ntags:\n- audio.cpp\n- gguf\n- pollard\n---", "",
             f"# {name}", "",
             f"Measured-allocation GGUF builds of [{base_model}](https://huggingface.co/{base_model}) for "
             f"[audio.cpp](https://github.com/0xShug0/audio.cpp), made with "
             f"[Pollard Weights](https://github.com/WestWaters/pollard-weights): every class of tensors was pushed "
             f"low on its own and scored by the model's own output, and the bits went where the damage was.", "",
             f"Run with `audiocpp_cli --family {family} --model <file>.gguf ...` (audio.cpp v0.9.3 or newer).", "",
             "| file | size | vs audio.cpp q8_0 | error vs source | notes |", "|---|---:|---:|---:|---|"]
    ordered = sorted(rungs, key=lambda r: r["bytes"])
    for i, r in enumerate(ordered):
        rel = f"{100 * (r['bytes'] / stock['bytes'] - 1):+.0f}%" if stock else "--"
        note = "smallest" if i == 0 else ("largest" if i == len(ordered) - 1 else "")
        lines.append(f"| `{r['file']}` | {r['bytes'] / 1e6:.0f} MB | {rel} | {r['delta']:+.4f} | {note} |")
    if stock:
        lines.append(f"| audio.cpp q8_0 (reference, not in this repo) | {stock['bytes'] / 1e6:.0f} MB | -- | "
                     f"{stock['delta']:+.4f} | |")
    lines += ["", f"_Error vs source: {metric}"
              + (f", on {texts} sentences the allocation never saw" if texts else "")
              + (f"; judge: {judge}" if judge else "") + ". Lower is better; 0 = the source model's own output._",
              "", "Every number above is measured, report-only, on the rung that is in this repo."]
    return "\n".join(lines) + "\n"


def publish(repo, files, readme, private=False, log=print):
    """Create the repo (public unless asked) and upload the card first, then the rungs -- through SafeApi, so nothing
    carries the publisher's local paths."""
    from pollard_privacy import SafeApi
    api = SafeApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=readme, path_in_repo="README.md", repo_id=repo)
    for f in files:
        api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=repo)
        log(f"   uploaded {os.path.basename(f)}")
    return f"https://huggingface.co/{repo}"


# ------------------------------------------------------------------------------------------------ cli

def main(argv=None):
    ap = argparse.ArgumentParser(prog="pollard-audiocpp", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eligible")
    e.add_argument("--source", required=True)
    e.add_argument("--family", required=True)
    e.add_argument("--reference", help="the family's published quantized GGUF: its quantized tensors are eligible too")
    e.add_argument("--out", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--source", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--eligible", required=True)
    p.add_argument("--task", choices=("asr", "tts"), required=True)
    p.add_argument("--clips", nargs="*", default=[], help="asr: audio clips")
    p.add_argument("--texts", help="tts: one sentence per line")
    p.add_argument("--voice")
    p.add_argument("--language", help="tts: the request's language (Ming: en / zh / ...)")
    p.add_argument("--asr-family")
    p.add_argument("--asr-model")
    p.add_argument("--types", default=PROBE_TYPE, help="comma list of types to probe each class at")
    p.add_argument("--backend", help="metal|cuda|vulkan|cpu (default: by hardware)")
    p.add_argument("--out", required=True)
    q = sub.add_parser("plan")
    q.add_argument("--eligible", required=True)
    q.add_argument("--sensitivity", required=True)
    q.add_argument("--bpw", type=float, required=True, help="bits per weight over the eligible tensors")
    q.add_argument("--out", required=True)
    m = sub.add_parser("score", help="measure one built GGUF against the source (the card's number)")
    m.add_argument("--source", required=True)
    m.add_argument("--model", required=True)
    m.add_argument("--family", required=True)
    m.add_argument("--task", choices=("asr", "tts"), required=True)
    m.add_argument("--clips", nargs="*", default=[])
    m.add_argument("--texts")
    m.add_argument("--voice")
    m.add_argument("--asr-family")
    m.add_argument("--asr-model")
    m.add_argument("--backend")
    u = sub.add_parser("publish", help="write the card from measured scores and upload the ladder")
    u.add_argument("--repo", required=True, help="org/name, e.g. PollardWeights/Ming-3.0-TTS-0.5B-Pollard")
    u.add_argument("--family", required=True)
    u.add_argument("--base-model", required=True)
    u.add_argument("--rung", action="append", required=True, metavar="FILE=SCORE_JSON",
                   help="a built rung and the `score` result for it (repeat)")
    u.add_argument("--stock", metavar="SCORE_JSON", help="the `score` result of audio.cpp's own q8_0 (the reference)")
    u.add_argument("--task", choices=("asr", "tts"), default="tts")
    u.add_argument("--texts", type=int, help="how many held-out sentences the scores used")
    u.add_argument("--judge", help="the ASR judge's name, for the card")
    u.add_argument("--private", action="store_true")
    u.add_argument("--dry-run", action="store_true", help="write README.md next to the first rung, upload nothing")
    b = sub.add_parser("build")
    b.add_argument("--source", required=True)
    b.add_argument("--family", required=True)
    b.add_argument("--rules", required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "eligible":
        with tempfile.TemporaryDirectory() as td:
            q8 = convert(a.source, a.family, os.path.join(td, "q8.gguf"), "q8_0")
            el = eligible_from(tensor_types(a.source), tensor_types(q8),
                               tensor_types(a.reference) if a.reference else None)
        json.dump({"source": a.source, "family": a.family, "eligible": el, "classes": classes(el)},
                  open(a.out, "w"), indent=1)
        print(f"{len(el)} eligible tensors in {len(classes(el))} classes -> {a.out}")
        return 0

    if a.cmd == "probe":
        el = json.load(open(a.eligible))
        cls = el["classes"]
        types = [t.strip() for t in a.types.split(",") if t.strip()]
        sens = json.load(open(a.out)) if os.path.exists(a.out) else {}
        with tempfile.TemporaryDirectory() as td:
            if a.task == "asr":
                _, ref_texts = score_asr(a.source, a.family, a.clips, [""] * len(a.clips), os.path.join(td, "ref"),
                                           a.backend)
                score = lambda m, w: score_asr(m, a.family, a.clips, ref_texts, w, a.backend)[0]
            else:
                sentences = [s.strip() for s in open(a.texts) if s.strip()]
                base_cost, ref_durs = score_tts(a.source, a.family, sentences, a.asr_family, a.asr_model,
                                                os.path.join(td, "ref"), a.backend, voice=a.voice,
                                                language=a.language)
                score = lambda m, w: score_tts(m, a.family, sentences, a.asr_family, a.asr_model, w, a.backend,
                                               ref_durs, a.voice, a.language)[0] - base_cost
            for i, c in enumerate(sorted(cls)):
                for t in types:
                    if t in sens.get(c, {}):
                        continue                                   # resumable: a measured cell is never redone
                    rules = class_rules(cls[c], t)
                    if all(ty == "q8_0" for _, ty in rules):   # rows that only take q8_0: nothing to measure below it
                        sens.setdefault(c, {})[t] = sens.get(c, {}).get("q8_0", 0.0)
                        json.dump(sens, open(a.out, "w"), indent=1)
                        continue
                    m = convert(a.source, a.family, os.path.join(td, "probe.gguf"), "orig", rules)
                    got = tensor_types(m)
                    if any(got.get(n, ("",))[0] != ty for n, ty in rules):
                        raise SystemExit(f"{c} @ {t}: the override did not land in the probe GGUF")  # never score a no-op
                    sens.setdefault(c, {})[t] = round(max(0.0, score(m, os.path.join(td, f"p{i}"))), 6)
                    json.dump(sens, open(a.out, "w"), indent=1)
                    print(f"  [{i + 1}/{len(cls)}] {c} @ {t}: {sens[c][t]}", flush=True)
        return 0

    if a.cmd == "plan":
        el = json.load(open(a.eligible))
        choice, res = plan(el["classes"], json.load(open(a.sensitivity)), a.bpw)
        rules = rules_for(el["classes"], choice)
        json.dump({"bpw": a.bpw, "choice": choice, "rules": rules, "predicted_cost": res["cost"],
                   "bytes": res["size"]}, open(a.out, "w"), indent=1)
        print(f"{len(rules)} rules over {len(choice)} classes, predicted cost {res['cost']:.4f} -> {a.out}")
        return 0

    if a.cmd == "score":
        with tempfile.TemporaryDirectory() as td:
            if a.task == "asr":
                _, ref = score_asr(a.source, a.family, a.clips, [""] * len(a.clips), os.path.join(td, "r"), a.backend)
                err, _ = score_asr(a.model, a.family, a.clips, ref, os.path.join(td, "m"), a.backend)
                out = {"cer_vs_source": round(err, 6)}
            else:
                sentences = [s.strip() for s in open(a.texts) if s.strip()]
                base, durs = score_tts(a.source, a.family, sentences, a.asr_family, a.asr_model,
                                       os.path.join(td, "r"), a.backend, voice=a.voice)
                err, _ = score_tts(a.model, a.family, sentences, a.asr_family, a.asr_model, os.path.join(td, "m"),
                                   a.backend, durs, a.voice)
                out = {"source_roundtrip": round(base, 6), "model_roundtrip": round(err, 6),
                       "delta": round(err - base, 6)}
        out.update(model=a.model, bytes=os.path.getsize(a.model), backend=a.backend or default_backend())
        print(json.dumps(out))
        return 0

    if a.cmd == "publish":
        rungs, files = [], []
        for spec in a.rung:
            f, _, js = spec.partition("=")
            d = json.load(open(js))
            rungs.append(dict(d, file=os.path.basename(f), bytes=os.path.getsize(f)))
            files.append(f)
        stock = json.load(open(a.stock)) if a.stock else None
        md = card_md(a.repo, a.family, a.base_model, rungs, stock, a.task, a.texts, a.judge)
        readme = os.path.join(os.path.dirname(os.path.abspath(files[0])), "README.md")
        open(readme, "w", encoding="utf-8").write(md)
        print(f"card -> {readme}")
        if not a.dry_run:
            print(publish(a.repo, files, readme, a.private))
        return 0

    if a.cmd == "build":
        r = json.load(open(a.rules))
        # base orig: protected tensors stay exactly as the source has them (a q8_0 base narrows f32 norms to f16,
        # which audio.cpp's Metal ops assert against); only the plan's exact rules change anything
        convert(a.source, a.family, a.out, "orig", [tuple(x) for x in r["rules"]])
        print(f"built {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
