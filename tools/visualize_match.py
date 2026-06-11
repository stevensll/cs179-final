#!/usr/bin/env python3
"""Visual aid: render log-magnitude spectrograms of segments from two songs
side by side (candidate window vs query window) so a claimed match can be
eyeballed. Uses ffmpeg for decoding and PIL for output (no matplotlib on box).

Usage:
  visualize_match.py A.wav A_offset B.wav B_offset duration out.png [max_hz]
"""

import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

SR = 22050
FFT = 2048
HOP = 256


def load(path, offset, duration):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(offset), "-t", str(duration),
           "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def spectrogram(x, max_hz):
    nfr = max(1, (len(x) - FFT) // HOP)
    win = np.hanning(FFT)
    frames = np.stack([x[i * HOP:i * HOP + FFT] * win for i in range(nfr)])
    S = np.abs(np.fft.rfft(frames, axis=1)).T  # bins x frames
    nbin = int(max_hz / (SR / 2) * S.shape[0])
    S = S[:nbin]
    S = 20 * np.log10(S + 1e-6)
    S -= S.max()
    return np.clip(S, -80, 0)


def to_img(S, h=400):
    # log-magnitude [-80,0] dB -> inferno-ish grayscale->color, low freq at bottom
    g = ((S + 80) / 80 * 255).astype(np.uint8)[::-1]
    img = Image.fromarray(g, "L").resize((max(g.shape[1], 256), h))
    return img.convert("RGB")


def main():
    if len(sys.argv) < 7:
        print(__doc__)
        sys.exit(1)
    a, aoff, b, boff, dur, out = sys.argv[1:7]
    max_hz = float(sys.argv[7]) if len(sys.argv) > 7 else 4000.0
    Sa = spectrogram(load(a, float(aoff), float(dur)), max_hz)
    Sb = spectrogram(load(b, float(boff), float(dur)), max_hz)
    ia, ib = to_img(Sa), to_img(Sb)
    w = max(ia.width, ib.width)
    pad, label_h = 4, 18
    canvas = Image.new("RGB", (w, ia.height + ib.height + 3 * label_h + pad), "black")
    d = ImageDraw.Draw(canvas)
    d.text((4, 2), f"{a} @{aoff}s (+{dur}s, 0-{int(max_hz)} Hz)", fill="yellow")
    canvas.paste(ia, (0, label_h))
    d.text((4, label_h + ia.height + 2), f"{b} @{boff}s", fill="yellow")
    canvas.paste(ib, (0, 2 * label_h + ia.height + pad))
    canvas.save(out)
    print(f"wrote {out} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
