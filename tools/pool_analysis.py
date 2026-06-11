#!/usr/bin/env python3
"""Pool-size and threshold analyses over the trainer's persisted out-of-fold
rerank scores (plots/rf/rerank_scores.csv) — no retraining needed.

1. 10-candidate-pool simulation (the paper's protocol: 1 true + 9 random):
   exact win probability per query via hypergeometrics, macro P/R/F at the
   train-chosen threshold via Monte Carlo.
2. Full-pool (64-candidate) precision/recall curve over the detection
   threshold, including precision at the paper's 50% recall operating point.

Per-query truth = the best-ranked true original (matches the hit@k
convention; a few queries sample multiple originals).
"""

import argparse
import csv
from collections import defaultdict
from math import comb

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="plots/rf/rerank_scores.csv")
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()

    by_q = defaultdict(lambda: {"true": [], "neg": []})
    thr = 0.5
    for r in csv.DictReader(open(args.scores)):
        by_q[r["query"]]["true" if r["is_true"] == "1" else "neg"].append(
            float(r["rf_prob"]))
        thr = float(r["thr"])
    queries = sorted(by_q)
    print(f"{len(queries)} queries, train-chosen threshold {thr:.3f}")

    # ---- 64-pool baseline (sanity: must reproduce the trainer's numbers) ----
    h1 = sum(max(d["true"]) > max(d["neg"], default=-1) for d in by_q.values())
    print(f"64-pool hit@1 (sanity): {h1}/{len(queries)} = {100*h1/len(queries):.1f}%")

    # ---- exact 10-pool hit@1: P(best true beats 9 uniform-random negatives) ----
    # b = negatives scoring above the best true; win iff all 9 draws come from
    # the N-b below: C(N-b,9)/C(N,9).
    probs = []
    for q in queries:
        s = max(by_q[q]["true"])
        neg = by_q[q]["neg"]
        N, b = len(neg), sum(n > s for n in neg)
        probs.append(comb(N - b, 9) / comb(N, 9) if N - b >= 9 else 0.0)
    print(f"\n10-pool top-1 (exact expectation): {100*np.mean(probs):.1f}%  "
          f"(paper protocol: 1 true + 9 random)")

    # ---- 10-pool macro P/R/F at the train threshold (Monte Carlo) ----
    rng = np.random.default_rng(42)
    tp = fp = fn = 0.0
    for q in queries:
        s = max(by_q[q]["true"])
        neg = np.asarray(by_q[q]["neg"])
        for _ in range(args.draws):
            pool_neg = rng.choice(neg, 9, replace=False)
            top = max(s, pool_neg.max())
            if top < thr: fn += 1
            elif s > pool_neg.max(): tp += 1
            else: fp += 1
    n = len(queries) * args.draws
    P, R = tp / max(1, tp + fp), tp / max(1, tp + fn)
    F = 2 * P * R / max(1e-9, P + R)
    print(f"10-pool macro @thr={thr:.3f}: P {100*P:.1f}%  R {100*R:.1f}%  "
          f"F {100*F:.1f}%   [paper: P 83.3  R 50.0  F 62.5]")

    # ---- full-pool P/R curve over the threshold ----
    tops = []
    for q in queries:
        s = max(by_q[q]["true"])
        m = max(by_q[q]["neg"], default=-1)
        tops.append((max(s, m), s > m))
    print(f"\n64-pool P/R over detection threshold "
          f"({len(queries)} queries, every query has a sample):")
    print(f"  {'thr':>6} {'P':>6} {'R':>6} {'F':>6}")
    best_at_r50 = None
    for t in sorted({t for t, _ in tops}, reverse=True):
        tp = sum(1 for top, ok in tops if top >= t and ok)
        fp = sum(1 for top, ok in tops if top >= t and not ok)
        fn = sum(1 for top, _ in tops if top < t)
        P = tp / max(1, tp + fp)
        R = tp / len(queries)
        F = 2 * P * R / max(1e-9, P + R)
        if R >= 0.5 and best_at_r50 is None:
            best_at_r50 = (t, P, R, F)
        print(f"  {t:6.3f} {100*P:6.1f} {100*R:6.1f} {100*F:6.1f}")
    if best_at_r50:
        t, P, R, F = best_at_r50
        print(f"\nat the paper's operating point (R >= 50%): thr {t:.3f} -> "
              f"P {100*P:.1f}%  R {100*R:.1f}%  F {100*F:.1f}%   "
              f"[paper: P 83.3 @ R 50.0]")


if __name__ == "__main__":
    main()
