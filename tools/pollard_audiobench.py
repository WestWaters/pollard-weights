#!/usr/bin/env python3
"""pollard-audiobench -- the benchmarks the model makers publish, measured on every rung.

Qwen, IBM and Liquid report their speech models on real recordings scored as word error rate: LibriSpeech test-clean /
test-other, FLEURS per language, the Open ASR Leaderboard's ESB sets. A rung's card should carry those same numbers.
pollard-audioeval asks the gross-damage question per language (does it still speak and listen); this answers the
makers' question, on their data and with their scoring.

    pollard-audiobench fetch --set librispeech-clean --n 500 --out sets/ls-clean
    pollard-audiobench fetch --set fleurs:ja_jp --n 100 --out sets/fleurs-ja
    pollard-audiobench fetch --set esb:earnings22 --n 200 --out sets/earnings22
    pollard-audiobench asr --gguf rung.gguf --mmproj mmproj.gguf --set sets/ls-clean --out r.json
    pollard-audiobench asr --audiocpp-family citrinet_asr --model rung.gguf --set sets/fleurs-ja --lang ja

Scoring follows the Open ASR Leaderboard: Whisper's EnglishTextNormalizer for English, its BasicTextNormalizer for
other languages, character error rate for Chinese / Japanese / Korean / Thai, corpus-level WER (total edits / total
reference words), and a 95% bootstrap interval resampled by SPEAKER (utterances from one speaker are not independent).

Data: LibriSpeech (CC-BY-4.0) and FLEURS (CC-BY-4.0) and the open ESB sets (AMI, Earnings-22, VoxPopuli). The fetch is
deterministic -- the same --n gives the same clips on every machine -- and spreads the clips across speakers.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unicodedata

#: no word boundaries in the script: score characters
CER_LANGS = {"zh", "ja", "ko", "th", "yue", "cmn"}
SEED = 1234

SETS = {
    "librispeech-clean": ("openslr/librispeech_asr", "clean/test/0000.parquet", "en"),
    "librispeech-other": ("openslr/librispeech_asr", "other/test/0000.parquet", "en"),
}
ESB_OPEN = {"ami", "earnings22", "voxpopuli"}          # gated / non-commercial ESB sets are left out on purpose


# ------------------------------------------------------------------------------------------------ text + scoring

def _normalizers():
    try:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
        return EnglishTextNormalizer({}), BasicTextNormalizer()
    except Exception:                                            # noqa: BLE001 -- the fallback below scores the same way
        return None, None


_EN, _BASIC = _normalizers()


def normalize(text, lang):
    """The leaderboard's normalization: Whisper's English normalizer for English, its basic one otherwise; a plain
    lower-case / punctuation-free fallback when transformers is not installed."""
    lang = (lang or "en").split("_")[0].lower()
    if lang == "en" and _EN is not None:
        t = _EN(text)
    elif _BASIC is not None:
        t = _BASIC(text)
    else:
        t = unicodedata.normalize("NFKC", text).lower()
        t = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in t)
    return " ".join(t.split())


def units(text, lang):
    """Words, or characters (spaces dropped) for scripts without word boundaries."""
    lang = (lang or "en").split("_")[0].lower()
    t = normalize(text, lang)
    return list(t.replace(" ", "")) if lang in CER_LANGS else t.split()


def edits(ref, hyp):
    """Levenshtein distance between two token lists."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def corpus_error(rows):
    """Corpus-level error rate: total edits / total reference units (the leaderboard's WER, not a mean of per-clip)."""
    e = sum(r["edits"] for r in rows)
    n = sum(r["ref_units"] for r in rows)
    return e / n if n else 0.0


def bootstrap_ci(rows, n_boot=1000, seed=SEED):
    """95% interval of the corpus error, resampling whole speakers with replacement."""
    by = {}
    for r in rows:
        by.setdefault(r.get("speaker") or r["id"], []).append(r)
    groups = list(by.values())
    if len(groups) < 2:
        v = corpus_error(rows)
        return v, v
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        pick = [x for _ in groups for x in rng.choice(groups)]
        vals.append(corpus_error(pick))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def clean_transcript(out, prompt=None):
    """The transcript in a CLI's stdout: after an `<asr_text>` tag (Qwen3-ASR), after `text_output=` (audio.cpp), else
    the last non-empty line; a prompt echoed back is removed, markup tags dropped."""
    m = re.findall(r"^text_output=(.*)$", out, re.M)
    if m:
        return m[-1].strip()
    text = out.strip()
    if "<asr_text>" in text:
        text = text.rsplit("<asr_text>", 1)[1]
    else:
        lines = [l for l in text.splitlines() if l.strip()]
        text = lines[-1] if lines else ""
    if prompt and text.endswith(prompt):
        text = text[: -len(prompt)]
    return re.sub(r"<[^>]{1,40}>", "", text).strip()


# ------------------------------------------------------------------------------------------------ fetch

def _to_wav(src_bytes, out, ext):
    """Any audio the sets ship (FLAC, WAV) -> 16 kHz mono 16-bit WAV, by ffmpeg or macOS afconvert."""
    with tempfile.NamedTemporaryFile(suffix="." + ext, delete=False) as f:
        f.write(src_bytes)
        tmp = f.name
    try:
        ff = shutil.which("ffmpeg")
        if ff:
            cmd = [ff, "-nostdin", "-loglevel", "error", "-y", "-i", tmp, "-ac", "1", "-ar", "16000", "-sample_fmt",
                   "s16", out]
        elif shutil.which("afconvert"):
            cmd = ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", tmp, out]
        else:
            raise SystemExit("ffmpeg (or macOS afconvert) is needed to decode the set's audio")
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(tmp)


def spread(items, n, key, seed=SEED):
    """`n` items, deterministic, round-robin over speakers so a big speaker cannot fill the sample."""
    rng = random.Random(seed)
    by = {}
    for it in sorted(items, key=lambda x: x["id"]):
        by.setdefault(key(it), []).append(it)
    for v in by.values():
        rng.shuffle(v)
    order = sorted(by)
    rng.shuffle(order)
    out = []
    while len(out) < n and any(by.values()):
        for k in order:
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return out


def _parquet_rows(repo, path_in_repo):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, path_in_repo, repo_type="dataset")
    return pq.read_table(p).to_pylist()


def fetch(set_name, n, out, log=print):
    os.makedirs(out, exist_ok=True)
    rows = []
    if set_name in SETS:
        repo, path, lang = SETS[set_name]
        for r in _parquet_rows(repo, path):
            rows.append({"id": r["id"], "text": r["text"], "speaker": str(r["speaker_id"]),
                         "bytes": r["audio"]["bytes"], "ext": "flac", "lang": lang})
    elif set_name.startswith("fleurs:"):
        code = set_name.split(":", 1)[1]
        from huggingface_hub import hf_hub_download
        tsv = hf_hub_download("google/fleurs", f"data/{code}/test.tsv", repo_type="dataset")
        meta = {}
        for line in open(tsv, encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 4:
                # FLEURS names no speakers; its sentence id groups the readings of one sentence, so the bootstrap
                # resamples by sentence there
                meta[f[1]] = {"id": f"{f[0]}-{f[1][:-4]}", "text": f[2], "speaker": f[0]}
        tgz = hf_hub_download("google/fleurs", f"data/{code}/audio/test.tar.gz", repo_type="dataset")
        chosen = {m["id"] for m in spread(list(meta.values()), n, lambda x: x["speaker"])}
        with tarfile.open(tgz, "r:gz") as tf:
            for mem in tf:
                name = os.path.basename(mem.name)
                if name in meta and meta[name]["id"] in chosen:
                    rows.append(dict(meta[name], bytes=tf.extractfile(mem).read(), ext="wav", lang=code))
    elif set_name.startswith("esb:"):
        sub = set_name.split(":", 1)[1]
        if sub not in ESB_OPEN:
            raise SystemExit(f"esb:{sub} -- only the open sets ({', '.join(sorted(ESB_OPEN))}); the others are gated or "
                             "non-commercial")
        from huggingface_hub import HfApi
        files = sorted(f for f in HfApi().list_repo_files("hf-audio/esb-datasets-test-only-sorted", repo_type="dataset")
                       if f.startswith(sub + "/") and f.endswith(".parquet"))
        for f in files[:2]:
            for i, r in enumerate(_parquet_rows("hf-audio/esb-datasets-test-only-sorted", f)):
                rows.append({"id": str(r.get("id") or f"{f}-{i}"), "text": r.get("text") or "",
                             "speaker": str(r.get("speaker_id") or r.get("id") or i),
                             "bytes": (r.get("audio") or {}).get("bytes"), "ext": "flac", "lang": "en"})
    else:
        raise SystemExit(f"unknown set {set_name!r}: librispeech-clean, librispeech-other, fleurs:<code>, esb:<open set>")
    rows = [r for r in rows if r.get("bytes") and r["text"].strip()]
    pick = spread(rows, n, lambda r: r["speaker"]) if not set_name.startswith("fleurs:") else rows
    manifest = []
    for r in pick:
        wav = os.path.join(out, re.sub(r"[^A-Za-z0-9_.-]", "_", r["id"]) + ".wav")
        if not os.path.exists(wav):
            _to_wav(r["bytes"], wav, r["ext"])
        manifest.append({"id": r["id"], "audio": os.path.basename(wav), "text": r["text"], "speaker": r["speaker"],
                         "lang": r["lang"]})
    with open(os.path.join(out, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    log(f"{set_name}: {len(manifest)} clips from {len({m['speaker'] for m in manifest})} speakers -> {out}")
    return manifest


# ------------------------------------------------------------------------------------------------ transcribe + score

def _llama_bin(name, hint=None):
    for d in (hint, os.environ.get("POLLARD_LLAMA_BIN"),
              os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime", "llama.cpp",
                           "build", "bin")):
        if d and os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    return shutil.which(name) or name


def transcribe_llama(gguf, mmproj, audio, system=None, prompt="Transcribe the audio.", ngl="99", bin_dir=None):
    cmd = [_llama_bin("llama-mtmd-cli", bin_dir), "-m", gguf, "--mmproj", mmproj, "--audio", audio, "-p", prompt,
           "-n", "256", "--temp", "0", "-ngl", str(ngl)]
    if system:
        cmd += ["-sys", system]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    return clean_transcript(r.stdout, prompt) if r.returncode == 0 else ""


def transcribe_audiocpp(family, model, audio, backend=None):
    exe = shutil.which("audiocpp_cli") or os.path.join(os.environ.get("POLLARD_AUDIOCPP_BIN", ""), "audiocpp_cli")
    with tempfile.TemporaryDirectory() as td:
        req = os.path.join(td, "r.json")
        json.dump({"requests": [{"id": "a", "audio": os.path.abspath(audio)}]}, open(req, "w"))
        r = subprocess.run([exe, "--task", "asr", "--family", family, "--model", model, "--backend",
                            backend or ("metal" if sys.platform == "darwin" else "cpu"), "--request-sequence", req,
                            "--out-dir", td], capture_output=True, text=True, timeout=900)
    return clean_transcript(r.stdout) if r.returncode == 0 else ""


def score_rows(manifest, hyps, lang=None):
    rows = []
    for m, h in zip(manifest, hyps):
        lg = lang or m.get("lang") or "en"
        ref, hyp = units(m["text"], lg), units(h, lg)
        rows.append({"id": m["id"], "speaker": m.get("speaker"), "edits": edits(ref, hyp), "ref_units": len(ref),
                     "hyp": h})
    return rows


def report(rows, lang):
    lo, hi = bootstrap_ci(rows)
    metric = "cer" if (lang or "en").split("_")[0].lower() in CER_LANGS else "wer"
    return {"metric": metric, "value": round(100 * corpus_error(rows), 2), "ci95": [round(100 * lo, 2),
                                                                                       round(100 * hi, 2)],
            "clips": len(rows), "speakers": len({r["speaker"] for r in rows})}


def run_asr(a):
    manifest = [json.loads(l) for l in open(os.path.join(a.set, "manifest.jsonl"), encoding="utf-8") if l.strip()]
    if a.limit:
        manifest = manifest[: a.limit]
    lang = a.lang or (manifest[0].get("lang") if manifest else "en")
    hyps = []
    for i, m in enumerate(manifest):
        wav = os.path.join(a.set, m["audio"])
        if a.audiocpp_family:
            hyps.append(transcribe_audiocpp(a.audiocpp_family, a.model, wav, a.backend))
        else:
            hyps.append(transcribe_llama(a.gguf, a.mmproj, wav, a.system, ngl=a.ngl, bin_dir=a.bin))
        if (i + 1) % 50 == 0:
            print(f"   {i + 1}/{len(manifest)}", flush=True)
    rows = score_rows(manifest, hyps, lang)
    res = dict(report(rows, lang), set=os.path.basename(os.path.normpath(a.set)), lang=lang,
               model=os.path.basename(a.gguf or a.model or ""))
    print(f"{res['set']}: {res['metric'].upper()} {res['value']}%  (95% CI {res['ci95'][0]}-{res['ci95'][1]}, "
          f"{res['clips']} clips, {res['speakers']} speakers)")
    if a.out:
        json.dump(dict(res, rows=rows), open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pollard-audiobench", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download a deterministic, speaker-spread subset of a benchmark set")
    f.add_argument("--set", required=True, help="librispeech-clean | librispeech-other | fleurs:<code> | esb:<set>")
    f.add_argument("--n", type=int, default=500)
    f.add_argument("--out", required=True)
    s = sub.add_parser("asr", help="transcribe a fetched set with a rung and score it the leaderboard's way")
    s.add_argument("--set", required=True)
    s.add_argument("--gguf", help="llama.cpp rung (with --mmproj)")
    s.add_argument("--mmproj")
    s.add_argument("--system", help="system prompt the model's card prescribes for ASR (e.g. 'Perform ASR in English.')")
    s.add_argument("--audiocpp-family", help="an audio.cpp ASR family instead (with --model)")
    s.add_argument("--model")
    s.add_argument("--backend")
    s.add_argument("--lang", help="override the set's language (scoring)")
    s.add_argument("--limit", type=int)
    s.add_argument("--ngl", default="99")
    s.add_argument("--bin")
    s.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch(a.set, a.n, a.out)
    else:
        if not (a.gguf and a.mmproj) and not (a.audiocpp_family and a.model):
            raise SystemExit("asr: --gguf + --mmproj, or --audiocpp-family + --model")
        run_asr(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
