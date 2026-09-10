# The EXL3 low-bit break: one outlier channel, and the fix

This is the root-cause write-up that used to sit inside the 1.3.0 release entry in `VERSIONING.md`.
It is a finding, not a changelog line, so it belongs here with the rest of the experiment log.

## The symptom

The EXL3 lane produced a broken build at low bit-width, deterministically. Not noisy, not
occasionally — the same collapse every run, which is what made it findable.

## The cause

A **single outlier input channel** collapsing the trellis global scale.

A trellis quantizer fits one scale across a tensor. When one input channel carries activations orders
of magnitude larger than the rest, that channel alone sets the scale, and every other channel is left
quantizing inside a fraction of the available range. The result is not graceful degradation; the
tensor's useful precision is spent representing one column.

This is the "massive activation" phenomenon, and it is why the break was deterministic: the outlier is
a property of the trained weights, not of the run.

## The fix

`pollard-hf-smooth` — SmoothQuant preconditioning **folded into the RMSNorms** as an exact identity.
The per-channel scale is moved out of the activations and into the preceding norm, so the tensor the
quantizer sees no longer has the outlier, and the model computes exactly the same function. Nothing is
approximated: the fold is algebraically exact, which is what makes it safe to apply before measuring.

## Measured

| build | PPL |
|---|---:|
| smoothed, 4 bpw | **8.70** |
| 8 bpw reference | 8.28 |
| unsmoothed, 4 bpw | 3090 (broken) |

Smoothed 4 bpw lands near the 8 bpw reference at half the bits. The unsmoothed number is not a
degradation, it is a collapse — which is the point: without the fix this lane did not have a low-bit
build at all.

## Why it generalises

The same shape shows up wherever one scale spans a tensor with a fat channel, so the lever is not
EXL3-specific. `pollard-doctor` scans an fp16 source for massive-activation input channels for exactly
this reason, and `pollard-precondition` measures whether smoothing, rotation or nothing wins for a
given model and bit-width rather than assuming.

Related: `legacy/PROXY_ERR_BANNED.md` (why a quantizer's own error estimate is not evidence) ·
`notes/e13-measured-sensitivity-beats-uniform.md`
