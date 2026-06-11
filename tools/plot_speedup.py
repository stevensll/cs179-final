#!/usr/bin/env python3
"""Optimization-progression plot: per-iteration speedup on the 5-candidate
scan (the one workload measured at every step; numbers from the CLAUDE.md
build log) with accuracy overlaid. Reverted experiments excluded.

Output: plots/speedup_progression.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = 42.0  # project-baseline 5-candidate scan, seconds

# (concise label, scan seconds, gate-set MRR after the change)
# gate = the 5-query Sample100 subset used to gate every optimization
STEPS = [
    ("baseline",          42.0, 0.215),
    ("TF32 GEMMs",        37.0, 0.215),
    ("DTW band kernel",   21.0, 0.215),
    ("2× GPU",       12.8, 0.215),
    ("simult. NMF",        9.3, 0.354),
    ("fused W·H",     8.8, 0.354),
    ("init cache",         8.8, 0.354),
    ("tcache+prefetch",    7.5, 0.354),
    ("batch+fastmath",     6.9, 0.354),
    ("log-freq frontend",  3.2, 0.363),
    ("K=32 default",       3.0, None),   # gated on the full benchmark instead
]
# full-70-query benchmark MRR (different, harder metric than the gate set)
FULL_BENCH = [(10, 0.214, "heuristic"), (11, 0.557, "+ RF rerank")]

x = list(range(len(STEPS)))
speedup = [BASE / s for _, s, _ in STEPS]

fig, ax = plt.subplots(figsize=(9, 5))
ax2 = ax.twinx()

ax.plot(x, speedup, "o-", color="#1f6fb4", lw=2, ms=5, zorder=3,
        label="5-cand scan speedup")
for xi, sp in zip(x, speedup):
    if xi < 2:
        continue  # 1.0x/1.1x sit in the tick-label band; the 1x gridline says it
    off, ha = ((10, -2), "left") if xi == 2 else ((0, 7), "center")
    ax.annotate(f"{sp:.1f}×", (xi, sp), textcoords="offset points",
                xytext=off, ha=ha, fontsize=7, color="#1f6fb4")

gx = [i for i, (_, _, m) in enumerate(STEPS) if m is not None]
gy = [m for _, _, m in STEPS if m is not None]
ax2.plot(gx, gy, "s--", color="#d95f02", lw=1.4, ms=4, alpha=0.9,
         label="MRR (5-query gate set)")
fx = [i for i, _, _ in FULL_BENCH]
fy = [m for _, m, _ in FULL_BENCH]
ax2.plot(fx, fy, "D-", color="#7b3294", lw=1.6, ms=6,
         label="MRR (full 70-query benchmark)")
ax2.annotate(f"{FULL_BENCH[0][1]:.3f} heuristic", (FULL_BENCH[0][0], FULL_BENCH[0][1]),
             textcoords="offset points", xytext=(-6, -14), ha="right",
             fontsize=6.5, color="#7b3294")
ax2.annotate(f"{FULL_BENCH[1][1]:.3f} + RF", (FULL_BENCH[1][0], FULL_BENCH[1][1]),
             textcoords="offset points", xytext=(-4, -14), ha="right", va="top",
             fontsize=6.5, color="#7b3294")

# concise tick labels INSIDE the chart (vertical, small font)
labels = [lbl for lbl, _, _ in STEPS] + ["RF rerank"]
ax.set_xticks(list(range(len(labels))))
ax.set_xticklabels([])
ax.tick_params(axis="x", length=0)
for xi, lbl in enumerate(labels):
    ax.text(xi, 1.07, lbl, rotation=90, ha="center", va="bottom",
            fontsize=6.8, color="0.25", zorder=2, clip_on=True)

ax.set_yscale("log")
ax.set_yticks([1, 2, 4, 8, 16])
ax.set_yticklabels(["1×", "2×", "4×", "8×", "16×"])
ax.set_ylim(0.93, 22)
ax.set_xlim(-0.5, len(labels) - 0.5)
ax.set_xlabel("optimization iteration", fontsize=9)
ax.set_ylabel("speedup vs baseline (5-candidate scan, warm caches)",
              fontsize=9, color="#1f6fb4")
ax2.set_ylabel("accuracy (MRR, Sample100)", fontsize=9, color="#d95f02")
ax2.set_ylim(0, 0.62)
ax.grid(True, axis="y", alpha=0.25, which="both")

ax.text(0.02, 0.97,
        "full 70-query benchmark: ~8.5 h → ~13 min (~40×)\n"
        "caching + batching compound at library scale",
        transform=ax.transAxes, fontsize=7.5, va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="#eef4fb", ec="#1f6fb4", lw=0.8))
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=7.5, framealpha=0.9)

ax.set_title("GPU sample detection — optimization progression "
             "(reverted experiments excluded)", fontsize=10)
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.text(0.01, 0.008,
         "accuracy note: the 5-query gate set reads high vs the full 70-query "
         "benchmark (small subset); the purple series is the full-benchmark metric",
         fontsize=6.5, color="0.4")
fig.savefig("plots/speedup_progression.png", dpi=160)
print("wrote plots/speedup_progression.png")
