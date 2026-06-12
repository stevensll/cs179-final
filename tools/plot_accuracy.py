#!/usr/bin/env python3
"""Accuracy comparison figure for the README: heuristic vs +RF on Sample100
(64-candidate pools), with the paper's published numbers (10-candidate pools)
and our simulated run under the paper's protocol. Numbers from the full
benchmark + tools/pool_analysis.py (see CLAUDE.md build log).

Output: plots/accuracy_comparison.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), width_ratios=[3, 2])

# ---- left: ranking metrics on the 64-candidate benchmark ----
metrics = ["hit@1", "hit@3", "MRR"]
heur = [11.4, 21.4, 21.4]   # MRR plotted as % for shared axis
rf = [50.0, 57.1, 55.7]
x = np.arange(len(metrics))
w = 0.36
b1 = ax1.bar(x - w / 2, heur, w, label="heuristic ranking", color="#9ecae1")
b2 = ax1.bar(x + w / 2, rf, w, label="+ random forest (out-of-fold)", color="#1f6fb4")
for bars in (b1, b2):
    ax1.bar_label(bars, fmt="%.1f", fontsize=8)
ax1.set_xticks(x)
ax1.set_xticklabels(["hit@1 (%)", "hit@3 (%)", "MRR (×100)"], fontsize=9)
ax1.set_ylim(0, 70)
ax1.set_ylabel("Sample100, 70 queries × 64 candidates", fontsize=9)
ax1.axhline(1.6, color="0.6", lw=1, ls=":")
ax1.text(2.45, 2.6, "random hit@1", fontsize=7, color="0.4", ha="right")
ax1.legend(fontsize=8, loc="upper left")
ax1.set_title("Classifier stage lifts ranking", fontsize=10)

# ---- right: macro F under the paper's own protocol ----
labels = ["paper\n(10-cand pools)", "ours, 64-cand\npools", "ours @ paper's\nprotocol (10-cand)"]
vals = [62.5, 66.7, 75.0]
colors = ["0.65", "#74a9cf", "#1f6fb4"]
bars = ax2.bar(np.arange(3), vals, 0.6, color=colors)
ax2.bar_label(bars, fmt="%.1f", fontsize=9)
ax2.set_xticks(np.arange(3))
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylim(0, 85)
ax2.set_ylabel("macro F-measure (%)", fontsize=9)
ax2.set_title("Song-level F vs the paper", fontsize=10)

fig.tight_layout()
fig.savefig("plots/accuracy_comparison.png", dpi=160)
print("wrote plots/accuracy_comparison.png")
