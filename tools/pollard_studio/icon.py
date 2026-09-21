#!/usr/bin/env python3
"""The Pollard bear as the application icon, so the dock is not a generic Python rocket.

The mark in the UI is an SVG built entirely from ellipses and circles, which means it can be
redrawn with PIL alone -- no SVG renderer, no extra dependency, and the two stay in step because
this file carries the same geometry the symbol does.

Drawn at 4x and downsampled: PIL has no antialiasing on ellipse(), so the edges are jagged at
icon size unless you supersample.
"""

from __future__ import annotations

import pathlib
import sys

#: the i-bear symbol, in its own 64x64 viewBox. (cx, cy, rx, ry, rotation)
_FUR = [
    (26.5, 52.5, 5.4, 7.4, 0), (37.5, 52.5, 5.4, 7.4, 0),      # legs
    (24.8, 57.8, 6.2, 2.9, 0), (39.2, 57.8, 6.2, 2.9, 0),      # feet
    (32.0, 35.5, 11.6, 17.0, 0),                                # body
    (20.8, 32.0, 3.7, 8.6, -13), (43.4, 29.5, 3.9, 8.8, 31),   # arms
    (29.6, 7.6, 3.3, 3.3, 0), (41.2, 7.8, 3.0, 3.0, 0),        # ears
    (36.0, 14.2, 8.1, 7.1, 0),                                  # head
    (43.6, 16.8, 4.6, 3.3, 0),                                  # muzzle
]
_DARK = [(34.4, 12.4, 1.25, 1.25), (46.6, 16.2, 1.7, 1.3)]      # eye, nose

FUR_LIGHT, FUR_DARK = (255, 255, 255), (205, 198, 184)
OUTLINE, FEATURE = (141, 133, 119), (93, 86, 75)


#: the panel the app is, so the icon reads as the same object
CREAM_HI, CREAM_LO, TRIM = (253, 252, 250), (222, 217, 208), (157, 193, 218)


def render(size: int = 512) -> "object":
    """The bear on the instrument's own cream tile, as a PIL image.

    On a transparent ground a white bear disappears into a light dock, so it gets the panel it
    belongs to -- cream body, the thin blue piping, and the mark inset the way it sits in the
    header.
    """
    from PIL import Image, ImageDraw
    ss = 4                                    # supersample: ellipse() does not antialias
    n = size * ss
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # the tile: a rounded square in the panel gradient, with the light-blue piping inset
    pad, rad = int(n * 0.035), int(n * 0.22)
    for i in range(n):                        # vertical gradient, cheap and good enough
        t = i / max(n - 1, 1)
        d.line([(0, i), (n, i)], fill=tuple(
            int(CREAM_HI[c] + (CREAM_LO[c] - CREAM_HI[c]) * t) for c in range(3)) + (255,))
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([pad, pad, n - pad, n - pad], rad, fill=255)
    img.putalpha(mask)
    d = ImageDraw.Draw(img)
    ins = int(n * 0.075)
    d.rounded_rectangle([ins, ins, n - ins, n - ins], int(rad * 0.82),
                        outline=TRIM + (150,), width=max(2, int(n * 0.006)))

    # the mark sits inside the piping, not edge to edge
    k = (n * 0.62) / 64.0
    ox, oy = (n - 64 * k) / 2, (n - 64 * k) / 2

    def blob(cx, cy, rx, ry, rot, fill, outline):
        box = [ox + (cx - rx) * k, oy + (cy - ry) * k,
               ox + (cx + rx) * k, oy + (cy + ry) * k]
        if not rot:
            d.ellipse(box, fill=fill, outline=outline, width=max(1, int(1.15 * k)))
            return
        # rotate on its own layer, so the shape turns about its own centre like the SVG does
        w, h = int(2 * rx * k) + 8, int(2 * ry * k) + 8
        lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(lay).ellipse([4, 4, w - 4, h - 4], fill=fill, outline=outline,
                                    width=max(1, int(1.15 * k)))
        lay = lay.rotate(-rot, resample=Image.BICUBIC, expand=True)
        img.alpha_composite(lay, (int(ox + cx * k - lay.width / 2),
                                  int(oy + cy * k - lay.height / 2)))

    # a flat mid-fur reads better than a gradient once it is 32px in a dock
    for cx, cy, rx, ry, rot in _FUR:
        blob(cx, cy, rx, ry, rot, FUR_LIGHT, OUTLINE)
    for cx, cy, rx, ry in _DARK:
        d.ellipse([ox + (cx - rx) * k, oy + (cy - ry) * k,
                   ox + (cx + rx) * k, oy + (cy + ry) * k], fill=FEATURE)

    return img.resize((size, size), Image.LANCZOS)


def png_path(size: int = 512) -> pathlib.Path | None:
    """Write the icon next to the UI and return it, or None if PIL is not installed.

    Cached: the geometry only changes when this file does, so it is regenerated when the PNG is
    older than the source.
    """
    out = pathlib.Path(__file__).resolve().parent / "ui" / f"bear-{size}.png"
    try:
        if out.exists() and out.stat().st_mtime >= pathlib.Path(__file__).stat().st_mtime:
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        render(size).save(out)
        return out
    except Exception:
        return out if out.exists() else None


def apply(window=None) -> str:
    """Set the app icon for this platform. Returns what it did, for the log."""
    p = png_path()
    if p is None:
        return "no icon (Pillow not installed)"
    if sys.platform == "darwin":
        try:
            # pywebview already pulls in pyobjc on macOS, so this adds no dependency
            from AppKit import NSApplication, NSImage
            img = NSImage.alloc().initWithContentsOfFile_(str(p))
            if img is None:
                return "icon file would not load"
            NSApplication.sharedApplication().setApplicationIconImage_(img)
            return f"dock icon set from {p.name}"
        except Exception as e:
            return f"dock icon unavailable ({e})"
    if sys.platform.startswith("win"):
        try:
            import ctypes
            # give the app its own taskbar identity, or Windows groups it under python.exe and
            # shows python's icon no matter what the window carries
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("WestWaters.Pollard.Studio")
            return "taskbar identity set"
        except Exception as e:
            return f"taskbar identity unavailable ({e})"
    return "icon left to the window manager"


if __name__ == "__main__":
    print(png_path(int(sys.argv[1]) if len(sys.argv) > 1 else 512))
