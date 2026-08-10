# Recipe: MiniMax-H3 (33B video+audio) on a 16 GB Mac

The memory-fit method applied to a video diffusion model: a ~66 GB-native
audio+video DiT generating 1080p-class clips with synchronized sound on a
16 GB Apple Silicon machine. Every number below was measured on an M4 Mac
Mini (16 GB); the campaign chart in the README is this recipe's receipts.

This file is written to be handed to a coding agent ("follow
recipes/minimax-h3-16gb.md") or followed by hand.

## 1. The fit (why this works at all)

H3 is dense (no experts — the routing tools in this repo don't apply; the
*fitting* method does). The working stack totals ~31 GB of files but only
~17 GB of peak residency, staged so the 16 GB machine never holds the whole
pipeline at once:

| piece | build | size |
|---|---|---|
| DiT (FL2VA) | AdaLN-pruned (33B→20.1B) + Q4_0 | 11.4 GB |
| text encoder | Qwen3-VL-32B @ Q4_K_M | 14.6 GB |
| video VAE | fp16 | 5.2 GB |
| audio VAE | fp32 | 0.6 GB |
| upscaler | RealESRGAN x2plus | 67 MB |

Sources: `molbal/MiniMax-H3-GGUF` (pruned DiT builds), `realrebelai/MiniMax-H3_GGUFs`
(encoder), `Comfy-Org/MiniMax-H3` (VAEs). Verify with the calculator first:
`pollard-calc --gguf <dit.gguf> --ram auto` — the pruned Q4_0 reads ~20.1B
params / ~4.5 bpw and FITS RESIDENT with room for activations.

Two fit rules that matter more than they look:
- **Never use a 2-bit encoder.** Q2 encoders destroy on-screen text and fine
  structure. Q4_K_M is the floor for conditioning quality.
- **The pruned Q4_0 DiT beats the unpruned Q3_K_M at everything** — smaller,
  ~25% faster per step, higher fidelity. Prefer pruned bases.

## 2. Runtime: ComfyUI, and the launch ritual

ComfyUI (current master) + the ComfyUI-GGUF loader. Each line below was paid
for with a dead run:

```bash
# kill by PORT — pkill by script name misses "python main.py" argv
lsof -ti :8188 | xargs kill -9
python main.py --listen 127.0.0.1 --port 8188 \
    --novram --cache-none --use-pytorch-cross-attention \
    > /tmp/comfyui.log 2>&1 &     # nohup/redirect — piping through `head` kills the server (SIGPIPE)
```

- `--use-pytorch-cross-attention` is **mandatory**: the default sub-quadratic
  path is ~30% slower on Apple Silicon (fused SDPA is numerically verified).
- `--novram --cache-none`: the memory estimator otherwise full-loads the DiT
  and OOMs a 16 GB machine.
- New model files need a **symlink into `ComfyUI/models/<class>/`** before the
  server lists them.
- **Tear the server down after each batch.** A long-lived server accumulates
  memory pressure across model cycles; treat it like a per-batch worker.
- **Swap needs disk headroom.** Keep ≥20 GB free; renders cycle ~26 GB of
  models and macOS swap grows on disk. A full disk under memory pressure can
  panic the machine (measured the hard way, twice).

## 3. The speed levers, in measured order

Baseline → final: 61:31 → 49:46 wall for ~7× the pixels (README chart).

1. **Schedule**: 20 steps, `res_multistep` + `simple`, cfg 1.0 (CFG-free).
2. **Resolution strategy**: generate 832×480 or 960×544 native, finish with
   RealESRGAN ×2 → 1664×960 / 1920×1088. Native high-res is slower AND softer.
3. **Conditioning cache**: encode once per prompt, reuse the conditioning —
   the 14.6 GB encoder then never loads during iteration.
4. **torch.compile (inductor)**: ~1.4× on steps, lossless, ~4 min warmup.
   Skip it for multishot batches (memory-heavier).
5. **FirstBlockCache** (NVIDIA Sol-Engine recipe; community single-GPU port):
   ~35% at 20 steps, near-lossless. Turn OFF when motion smoothness is
   critiqued. Caution: H3 blocks mutate their input in place — any cache
   comparing input/output must `.clone()` first or it silently no-ops.
6. **Step floor for audio**: H3 runs video and audio on different flow shifts
   (12 / 3); flat samplers approximate both. The approximation holds at ~20
   steps and audibly degrades below ~16 (blown-out "video-game" sound). Use
   the SigmaShift node (video 12 / audio 3) and keep audio-critical renders
   at 20 steps.

## 3b. ref2va (reference-conditioned) — measured on the same 16 GB M4

The reference-to-video-audio variant conditions on identity images/video/audio
and is where a memory-fit build of the DiT itself pays off. All measured with a
pollard-fit-dit build of the pruned ref2va DiT (11.6 GB, quantized from the
community Q8_0 with AdaLN/norm/audio-projection tensors protected):

- **It renders end-to-end.** 56 frames @ 832x480, 20 steps: ~96 min. The
  measured-mix build won a same-seed A/B against the flat Q4_0 on background
  detail and glow effects; audio levels identical (-14.0 vs -14.1 dB mean).
- **Turbo LoRA works on ref2va** (undocumented by its creator — verified here):
  8 steps, euler + simple, strength 1.0, video shift 12, **audio shift 5**,
  same seed → clean output, ~1.8x wall-clock. Only ~3.5 min/step is sampling;
  ~26 min/run is fixed (encoder conditioning + load + decode), so the speedup
  approaches the full 2.5x in multishot where conditioning amortizes. Keep
  20 steps for final masters; 4-step needs a dual-clock sampler for audio.
- **API graph gotcha**: the native `MiniMaxH3ReferenceToVideo` node's autogrow
  reference inputs are dotted paths in API JSON — `ref_images.ref_image_0`
  (0-based) — while prompt tags stay 1-based (`<Picture 1>`).
- **Reference text bleeds.** Reference-image tokens ride through every step;
  a text-heavy reference sheet makes the model hallucinate garbled text into
  scenes. Use a text-free character crop for shots that don't need on-screen
  text, and quote exact strings for shots that do.

## 3c. GPU lane (community-reported, not measured here)

The same fit logic applies off-Mac; these are the levers CUDA users report,
unverified on this recipe's hardware: SageAttention (CUDA-only) on top of the
turbo LoRA; SeedVR2 / FlashVSR temporal upscalers instead of per-frame ESRGAN
(per-frame upscalers flicker on faces; temporal models don't — SeedVR2 also
runs on Apple MPS with fp32 forced, but 16 GB is tight); int8_convrot packs;
seed-scouting (preview several seeds at low steps, finish only the winner).
Measured numbers from GPU hardware are welcome via the profile zoo.

## 4. Quality rules (prompt-side)

- **On-screen text**: H3 spells *quoted strings* perfectly. Put exact text in
  double quotes + typography + "do not misspell it, do not add any other
  text." Everywhere else: "no visible text or lettering anywhere" — and for
  surfaces that inherently carry printing (table layouts, keyboards), either
  quote the text or say "plain unmarked X".
- **Identity across shots** = description density: re-describe the same 6–8
  concrete attributes verbatim in every shot. "The same man" drifts.
- **Speech budget** ≈ 2.5 words/second of shot; under-filled shots generate
  gibberish speech.
- Describe the audio in prose in every shot (ambience, SFX, quoted dialogue).

## 5. Long-form

Per-shot frame counts snap to H3's 17k+5 grid (39 = 1.6 s is the sweet spot
on 16 GB; 73 frames OOMs at 832×480). Long content = multishot chaining
(community node packs provide script-driven multi-shot samplers with memory
frames), not longer shots. 10 shots × 8 steps ≈ 5 h; reserve 20 steps for
final masters.

## 6. Credits

MiniMax (open weights) · molbal (pruned GGUF builds) · realrebelai (encoder
GGUFs) · Comfy-Org (ComfyUI + VAEs) · city96 (ComfyUI-GGUF) · NVIDIA
Sol-Engine + its community ports (FirstBlockCache lineage) · the MiniMax
prompt-writing guide (text/audio prompting rules).
