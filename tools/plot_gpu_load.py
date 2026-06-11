#!/usr/bin/env python3
"""Sample nvidia-smi utilization and render a small PNG line chart (no
matplotlib on the box; PIL only).

Usage: plot_gpu_load.py [seconds] [out.png]    (defaults: 60 build/gpu_load.png)
Solid line = SM utilization, dim line = memory bandwidth utilization; one color
per GPU. Samples at ~2 Hz.
"""

import subprocess
import sys
import time

from PIL import Image, ImageDraw

W, H, PAD = 860, 280, 36
COLORS = {0: ((90, 200, 255), (40, 90, 115)), 1: ((120, 255, 120), (50, 110, 50))}


def sample():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    vals = {}
    for line in out.strip().splitlines():
        idx, sm, mem = [int(x) for x in line.split(",")]
        vals[idx] = (sm, mem)
    return vals


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    out_png = sys.argv[2] if len(sys.argv) > 2 else "build/gpu_load.png"
    interval = 0.5
    n = max(2, int(seconds / interval))
    series = {}  # gpu -> ([sm...], [mem...])
    for i in range(n):
        for gpu, (sm, mem) in sample().items():
            series.setdefault(gpu, ([], []))
            series[gpu][0].append(sm)
            series[gpu][1].append(mem)
        if i < n - 1:
            time.sleep(interval)

    img = Image.new("RGB", (W, H), (16, 16, 20))
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = PAD, PAD // 2, W - 8, H - PAD
    for pct in (0, 25, 50, 75, 100):
        y = y1 - (y1 - y0) * pct / 100
        d.line([(x0, y), (x1, y)], fill=(45, 45, 55))
        d.text((4, y - 6), f"{pct:3d}%", fill=(140, 140, 150))

    def xpix(i, ns):
        return x0 + (x1 - x0) * i / max(1, ns - 1)

    for gpu, (sms, mems) in sorted(series.items()):
        strong, dim = COLORS.get(gpu, ((255, 255, 255), (110, 110, 110)))
        for vals, color in ((mems, dim), (sms, strong)):
            pts = [(xpix(i, len(vals)), y1 - (y1 - y0) * v / 100)
                   for i, v in enumerate(vals)]
            d.line(pts, fill=color, width=2)
        avg_sm = sum(sms) / len(sms)
        d.text((x0 + 6 + gpu * 330, 4),
               f"GPU{gpu}: SM avg {avg_sm:.0f}% (solid), mem b/w (dim)",
               fill=strong)
    d.text((x0, H - PAD + 8),
           f"{seconds:.0f}s window, {interval}s samples - {time.strftime('%H:%M:%S')}",
           fill=(140, 140, 150))
    img.save(out_png)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
