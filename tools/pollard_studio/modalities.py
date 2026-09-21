#!/usr/bin/env python3
"""What can this build actually DO, and how do you check quantization did not break it?

Text coherence is one modality's worth of verification. A build that also sees, hears, draws or
speaks can pass every text gate and be broken in a way no perplexity, KL or loop check will show:
a vision path that stopped resolving colour, a speech head that outputs a buzz, an image head that
outputs noise.

Detection reads artifacts -- config files, GGUF metadata, the presence of a projector or a VAE --
never the model's name. A name is a label someone typed, and it is wrong exactly when it matters.

The structural checks here are deliberately cheap. Judging whether generated speech SOUNDS right
needs an ASR round-trip and a speaker-similarity model; judging an image needs CLIP. Those belong
in the Pollard tools that already own them. But the failures quantization actually produces at low
bit-width are blunt -- silence, a DC buzz, a flat grey frame, uniform noise -- and those are
catchable from the bytes alone, which makes them usable as a gate on every build.

    python -m pollard_studio.modalities <model dir or gguf>
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

# name -> (label, what a user is actually verifying)
MODALITIES = {
    "text":      ("Text", "does it still answer coherently, without looping"),
    "vision_in": ("Vision in", "can it still resolve colour, count, shape and position"),
    "audio_in":  ("Audio in", "can it still transcribe and follow spoken input"),
    "video_in":  ("Video in", "does it still track content across frames"),
    "image_out": ("Image out", "does it still draw structure, or only noise"),
    "audio_out": ("Speech out", "does it still speak, or output silence or a buzz"),
}


def _json(p: Path):
    try:
        return json.loads(p.read_text(errors="replace"))
    except Exception:
        return {}


def detect(path: str | Path) -> dict:
    """Which modalities a build carries. Reads what is on disk, not the name."""
    p = Path(path)
    got = {k: False for k in MODALITIES}
    got["text"] = True                                   # every build here is at least a text model
    notes: list[str] = []

    if p.is_file() and p.suffix == ".gguf":
        return _detect_gguf(p, got, notes)

    if not p.is_dir():
        return {"modalities": got, "notes": ["path not found"], "kind": "unknown"}

    cfg = _json(p / "config.json")
    blob = json.dumps(cfg).lower()
    idx = _json(p / "model_index.json")                  # diffusers pipelines declare themselves here

    if "vision_config" in cfg or "vision_tower" in blob or "visual" in blob or "image_token" in blob:
        got["vision_in"] = True
        notes.append("vision tower in config")
    if "audio_config" in cfg or "audio_tower" in blob or "speech" in blob:
        got["audio_in"] = True
        notes.append("audio tower in config")
    if "video" in blob or "num_frames" in blob:
        got["video_in"] = True

    # a diffusion pipeline is a VAE + a denoiser + a scheduler, whatever the folder is called
    has_vae = bool(list(p.glob("vae*")) or "vae" in json.dumps(idx).lower())
    has_denoiser = any(list(p.glob(d)) for d in ("unet*", "transformer*"))
    if idx or (has_vae and has_denoiser):
        got["image_out"] = True
        notes.append("diffusion pipeline (vae + denoiser)")
    # a speech head: a codec/vocoder next to the LM
    if any(list(p.glob(g)) for g in ("*vocoder*", "*codec*", "speaker*", "*snac*", "*dac*")):
        got["audio_out"] = True
        notes.append("vocoder or codec alongside the model")
    arch = " ".join(cfg.get("architectures") or []).lower()
    if any(k in arch for k in ("tts", "speecht5", "vits", "bark", "csm", "orpheus")):
        got["audio_out"] = True

    kind = "diffusers" if idx else ("safetensors" if list(p.glob("*.safetensors")) else "dir")
    return {"modalities": got, "notes": notes, "kind": kind}


def _detect_gguf(p: Path, got: dict, notes: list) -> dict:
    """A GGUF keeps its projector in a SEPARATE mmproj file, so look beside it as well as inside."""
    sib = [f.name.lower() for f in p.parent.glob("*.gguf")]
    if any("mmproj" in n for n in sib):
        got["vision_in"] = True
        notes.append("mmproj found alongside")
    try:
        from . import ggufread
        meta = ggufread.read(p)["metadata"]
    except Exception:
        meta = {}
    for k, v in meta.items():
        kl = k.lower()
        if "has_vision_encoder" in kl and v:
            got["vision_in"] = True
        if "has_audio_encoder" in kl and v:
            got["audio_in"] = True
    if meta.get("general.architecture", "").lower() in ("clip", "mmproj"):
        notes.append("this file IS a projector, not a language model")
    return {"modalities": got, "notes": notes, "kind": "gguf"}


# --- structural checks: the blunt failures quantization actually produces ----------------------
def check_image(raw: bytes) -> dict:
    """Is this a picture, or is it the way a broken image head fails?

    Two failure shapes dominate below ~4 bits: a flat frame (the denoiser collapsed and every
    pixel is one colour) and uniform noise (it never converged). Both are visible in the pixel
    histogram without decoding what the image depicts.
    """
    px, w, h = _decode(raw)
    if px is None:
        return {"ok": False, "reason": "not a decodable image"}
    n = len(px)
    mean = sum(px) / n
    var = sum((v - mean) ** 2 for v in px) / n
    std = math.sqrt(var)
    # Neighbour difference separates flat (near zero) from everything else, but it does NOT
    # separate noise from structure on its own: a checkerboard or any hard-edged pattern is
    # legitimately high-frequency and would be thrown away as noise.
    diffs = [abs(px[i] - px[i - 1]) for i in range(1, n)]
    rough = sum(diffs) / len(diffs)

    # What actually distinguishes them is the histogram. Uniform noise fills every value about
    # equally, so its entropy is near the 8-bit maximum. Structure -- however sharp -- concentrates
    # on far fewer values.
    hist = [0] * 256
    for v in px:
        hist[v] += 1
    ent = 0.0
    for c in hist:
        if c:
            pr = c / n
            ent -= pr * math.log2(pr)

    # Roughness is the discriminator; entropy is reported because it is useful to see, but it is
    # NOT used to gate. Tried that: on a small frame a rendered image with grain has HIGHER
    # entropy than uniform noise does, because 4096 samples over 256 bins undershoots the maximum.
    # It read the wrong way round, which is worth leaving written down.
    flat = std < 3.0
    noise = rough > 60.0
    return {"ok": not (flat or noise), "width": w, "height": h,
            "std": round(std, 2), "roughness": round(rough, 2), "entropy": round(ent, 2),
            "reason": "flat frame -- the denoiser collapsed" if flat else
                      "uniform noise -- it never converged" if noise else "has structure"}


def check_audio(raw: bytes) -> dict:
    """Is this speech, or silence, a DC buzz, or clipping?"""
    samples, rate = _decode_wav(raw)
    if samples is None:
        return {"ok": False, "reason": "not a decodable WAV"}
    n = len(samples)
    if not n:
        return {"ok": False, "reason": "empty audio"}
    peak = max(abs(s) for s in samples)
    mean = sum(samples) / n
    rms = math.sqrt(sum((s - mean) ** 2 for s in samples) / n)
    clipped = sum(1 for s in samples if abs(s) >= 32700) / n
    # zero crossings separate speech from a constant tone: speech moves around, a buzz does not
    zc = sum(1 for i in range(1, n) if (samples[i - 1] < mean) != (samples[i] < mean)) / n
    silent = peak < 300
    buzz = rms > 0 and zc < 0.002
    return {"ok": not (silent or buzz or clipped > 0.05),
            "seconds": round(n / rate, 2) if rate else 0, "rms": round(rms, 1),
            "peak": peak, "clipped_frac": round(clipped, 4), "zero_crossing_rate": round(zc, 4),
            "reason": "silence" if silent else "constant tone, not speech" if buzz
                      else f"clipping on {clipped:.1%} of samples" if clipped > 0.05
                      else "has speech-like variation"}


def _decode(raw: bytes):
    """Greyscale pixels from PNG/PPM without pulling in an image library."""
    if raw[:2] in (b"P5", b"P6"):                        # netpbm: what most CLI samplers can emit
        parts, i, vals = raw.split(b"\n", 4), 0, []
        try:
            w, h = (int(x) for x in parts[1].split())
            body = parts[3] if parts[0] == b"P5" else parts[3]
            step = 1 if parts[0] == b"P5" else 3
            return list(body[::step]), w, h
        except Exception:
            return None, 0, 0
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        try:
            w, h = struct.unpack(">II", raw[16:24])
        except Exception:
            return None, 0, 0
        # the compressed stream is not decoded here; its byte spread still separates the two
        # failure shapes, because a flat frame compresses to almost nothing
        idat = raw[raw.find(b"IDAT"):] if b"IDAT" in raw else b""
        if len(idat) < 64:
            return [0] * 64, w, h                        # degenerate -> reads as flat, correctly
        return list(idat[8:8 + min(len(idat) - 8, 65536)]), w, h
    return None, 0, 0


def _decode_wav(raw: bytes):
    """16-bit PCM samples from a WAV header, stdlib only."""
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return None, 0
    pos, rate, bits = 12, 0, 16
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt " and len(body) >= 16:
            rate = struct.unpack("<I", body[4:8])[0]
            bits = struct.unpack("<H", body[14:16])[0]
        elif cid == b"data":
            if bits != 16:
                return None, rate
            n = len(body) // 2
            return list(struct.unpack(f"<{n}h", body[:n * 2])), rate
        pos += 8 + size + (size & 1)
    return None, rate


# --- playback -----------------------------------------------------------------------------------
# A verdict of "has structure" is not the same as looking at the picture. For a model that draws,
# speaks, or renders video, the only check that settles it is your own eyes and ears -- so the
# generated file goes in front of the user, not just a pass/fail about it.
MEDIA = {
    ".png": ("image", "image/png"), ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"), ".gif": ("image", "image/gif"),
    ".webp": ("image", "image/webp"), ".bmp": ("image", "image/bmp"),
    ".ppm": ("image", "image/x-portable-pixmap"), ".pgm": ("image", "image/x-portable-graymap"),
    ".wav": ("audio", "audio/wav"), ".mp3": ("audio", "audio/mpeg"),
    ".flac": ("audio", "audio/flac"), ".ogg": ("audio", "audio/ogg"),
    ".opus": ("audio", "audio/ogg"), ".m4a": ("audio", "audio/mp4"),
    ".mp4": ("video", "video/mp4"), ".webm": ("video", "video/webm"),
    ".mov": ("video", "video/quicktime"), ".mkv": ("video", "video/x-matroska"),
}
# A data URI is copied whole into the page, so a large file costs that much memory twice over.
MAX_INLINE = 96 * 1024 * 1024


def media_kind(path: str | Path) -> tuple[str | None, str | None]:
    return MEDIA.get(Path(path).suffix.lower(), (None, None))


def load_media(path: str | Path) -> dict:
    """A generated file, ready to put in front of someone.

    Returned as a data URI rather than a file:// link because the page has its own origin and
    cannot read the disk -- a link would render as a broken box with no explanation.
    """
    import base64
    p = Path(path).expanduser()
    kind, mime = media_kind(p)
    if kind is None:
        return {"ok": False, "reason": f"{p.suffix or 'no extension'} is not a media type "
                                       f"this can play", "kind": None}
    try:
        size = p.stat().st_size
    except OSError as e:
        return {"ok": False, "reason": str(e), "kind": kind}
    if size > MAX_INLINE:
        return {"ok": False, "kind": kind, "bytes": size,
                "reason": f"{size / 1e6:.0f} MB is too large to inline "
                          f"({MAX_INLINE / 1e6:.0f} MB cap). Point a player at it directly."}
    raw = p.read_bytes()
    # Grade the ORIGINAL bytes. Converting first and checking after reads the compressed stream
    # instead of the pixels, and anything highly compressible -- a checkerboard, a test pattern --
    # comes back as "flat frame", which is exactly the failure this is supposed to detect.
    check = check_image(raw) if kind == "image" else check_audio(raw) if kind == "audio" else None
    # PPM/PGM is what a bare sampler tends to emit and no browser renders it, so convert for display
    if p.suffix.lower() in (".ppm", ".pgm"):
        png = _ppm_to_png(raw)
        if png is None:
            return {"ok": False, "kind": "image", "reason": "could not read that netpbm file"}
        raw, mime = png, "image/png"
    return {"ok": True, "kind": kind, "mime": mime, "bytes": size,
            "name": p.name, "check": check,
            "data": "data:" + mime + ";base64," + base64.b64encode(raw).decode()}


def _ppm_to_png(raw: bytes):
    """Netpbm -> PNG with zlib and struct. No image library: this has to work in a bare venv."""
    import struct
    import zlib
    try:
        parts = raw.split(None, 4)
        magic = parts[0]
        if magic not in (b"P5", b"P6"):
            return None
        w, h, _maxv = int(parts[1]), int(parts[2]), int(parts[3])
        body = parts[4]
        chans = 1 if magic == b"P5" else 3
        need = w * h * chans
        if len(body) < need:
            return None
        rows = b"".join(b"\x00" + body[y * w * chans:(y + 1) * w * chans] for y in range(h))
    except (ValueError, IndexError):
        return None

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0 if chans == 1 else 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    d = detect(sys.argv[1])
    print(f"kind: {d['kind']}")
    for k, on in d["modalities"].items():
        label, why = MODALITIES[k]
        print(f"  [{'x' if on else ' '}] {label:<12} {why}")
    for n in d["notes"]:
        print(f"      - {n}")


if __name__ == "__main__":
    main()
