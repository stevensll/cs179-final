#!/usr/bin/env python3
"""Train the paper's random-forest stage (3.4) on the feature corpus and
report leakage-safe metrics: out-of-fold AUC, reranked MRR/hit@k, and
paper-style song-level (macro) precision/recall/F at a train-chosen threshold.

Leakage protections:
- folds are grouped by CONNECTED COMPONENTS of the query<->original graph
  (queries sharing any sampled original always co-travel), strictly stronger
  than per-query grouping;
- all reported predictions are out-of-fold;
- the decision threshold for macro P/R/F is chosen on each fold's TRAIN side;
- RF hyperparameters are the paper's fixed choice (200 trees, sqrt features)
  — no tuning loop touches test data.

Labels: a row is positive iff its (query, candidate) is an annotated pair AND
band_start_s lies within TOL of any annotated t_original for that pair (the
sampled material itself, wherever it recurs in the query).

Usage: train_rf.py [--features-dir datasets/sample100/features]
                   [--tol 4] [--folds 5] [--out plots/rf]
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

FEATS = ["n_endpoints", "min_cost", "avg_cost", "std_cost", "best_len",
         "best_slope", "best_dev", "avg_slope", "std_slope", "avg_len",
         "std_len", "avg_dev", "std_dev"]


def load_pairs(data="datasets/sample100"):
    pairs = defaultdict(list)  # (query, candidate) -> [t_original_s, ...]
    for r in csv.DictReader(open(f"{data}/eval_pairs.csv")):
        pairs[(r["query_file"], r["expected_file"])].append(float(r["t_original_s"]))
    return pairs


def connected_components(queries, pairs):
    """Union queries that share any sampled original."""
    parent = {q: q for q in queries}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_orig = defaultdict(list)
    for (q, o) in pairs:
        if q in parent:
            by_orig[o].append(q)
    for qs in by_orig.values():
        for q in qs[1:]:
            ra, rb = find(qs[0]), find(q)
            if ra != rb:
                parent[rb] = ra
    return {q: find(q) for q in queries}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="datasets/sample100/features")
    ap.add_argument("--tol", type=float, default=4.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--neg-per-pos", type=float, default=20.0,
                    help="negative subsampling ratio (all positives kept; "
                         "candidates' BEST rows always kept so reranking has a "
                         "score for every (query,candidate))")
    ap.add_argument("--out", default="plots/rf")
    args = ap.parse_args()

    pairs = load_pairs()
    files = sorted(glob.glob(f"{args.features_dir}/*.csv"))
    X, y, qids, cands, heur = [], [], [], [], []
    for fp in files:
        for r in csv.DictReader(open(fp)):
            q, c = r["query"], r["candidate"]
            key = (q, c)
            pos = key in pairs and any(
                abs(float(r["band_start_s"]) - t) <= args.tol for t in pairs[key])
            X.append([float(r[k]) for k in FEATS])
            y.append(1 if pos else 0)
            qids.append(q)
            cands.append(c)
            heur.append(float(r["heur_sel"]))
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.int8)
    qids = np.asarray(qids)
    cands = np.asarray(cands)
    print(f"corpus: {len(y):,} rows from {len(files)} queries; "
          f"positives {int(y.sum()):,} ({100*y.mean():.2f}%)")
    if y.sum() == 0:
        sys.exit("no positive labels — check tol / annotations join")

    queries = sorted(set(qids))
    comp = connected_components(queries, pairs)
    comps = sorted(set(comp.values()))
    print(f"{len(queries)} queries in {len(comps)} leakage components")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    groups = np.asarray([comps.index(comp[q]) for q in qids])
    rng = np.random.default_rng(42)

    def train_subsample(tr):
        """All positives + negatives at --neg-per-pos; TRAINING only — test
        folds are always predicted in full, so reranking is exact."""
        pos = tr[y[tr] == 1]
        neg = tr[y[tr] == 0]
        n_neg = min(len(neg), int(len(pos) * args.neg_per_pos))
        return np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])

    oof = np.zeros(len(y), np.float32)
    gkf = GroupKFold(n_splits=args.folds)
    thresholds = []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        sub = train_subsample(tr)
        rf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                    class_weight="balanced", n_jobs=-1,
                                    random_state=42)
        rf.fit(X[sub], y[sub])
        oof[te] = rf.predict_proba(X[te])[:, 1]
        # train-side threshold: best F1 over (subsampled) train predictions
        tr = sub
        ptr = rf.predict_proba(X[tr])[:, 1]
        best_t, best_f = 0.5, -1
        for t in np.quantile(ptr[y[tr] == 1], np.linspace(0.05, 0.95, 19)):
            tp = ((ptr >= t) & (y[tr] == 1)).sum()
            fp = ((ptr >= t) & (y[tr] == 0)).sum()
            fn = ((ptr < t) & (y[tr] == 1)).sum()
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            if f1 > best_f:
                best_f, best_t = f1, t
        thresholds.append(best_t)
        print(f"  fold {k}: test rows {len(te):,}, "
              f"AUC {roc_auc_score(y[te], oof[te]):.3f}, train-F1 thr {best_t:.3f}")
    print(f"out-of-fold row AUC: {roc_auc_score(y, oof):.3f}")

    # ---- rerank candidates per query: score = max out-of-fold prob ----
    agg = defaultdict(lambda: (-1.0, 1e30))  # (q,c) -> (max prob, min heur)
    for i in range(len(y)):
        p, h = agg[(qids[i], cands[i])]
        agg[(qids[i], cands[i])] = (max(p, float(oof[i])), min(h, heur[i]))
    truth = defaultdict(set)
    for (q, c) in pairs:
        truth[q].add(c)

    def rank_metrics(score_of, reverse):
        mrr, h1, h3, n = 0.0, 0, 0, 0
        for q in queries:
            cs = sorted({c for (qq, c) in agg if qq == q},
                        key=lambda c: score_of(q, c), reverse=reverse)
            best = min((cs.index(c) + 1 for c in truth[q] if c in cs), default=10**9)
            if best < 10**9:
                mrr += 1.0 / best
                h1 += best == 1
                h3 += best <= 3
                n += 1
        return mrr / n, h1 / n, h3 / n, n

    m_rf = rank_metrics(lambda q, c: agg[(q, c)][0], True)
    m_he = rank_metrics(lambda q, c: -agg[(q, c)][1], True)
    print(f"\nheuristic rerank (from corpus): MRR {m_he[0]:.3f}  "
          f"hit@1 {100*m_he[1]:.1f}%  hit@3 {100*m_he[2]:.1f}%  (n={m_he[3]})")
    print(f"RF rerank (out-of-fold):        MRR {m_rf[0]:.3f}  "
          f"hit@1 {100*m_rf[1]:.1f}%  hit@3 {100*m_rf[2]:.1f}%")

    # ---- paper-style macro (song-level) P/R/F with train-chosen threshold ----
    thr = float(np.median(thresholds))
    tp = fp = fn = 0
    for q in queries:
        cs = {c for (qq, c) in agg if qq == q}
        top = max(cs, key=lambda c: agg[(q, c)][0])
        detect = agg[(q, top)][0] >= thr
        if detect and top in truth[q]:
            tp += 1
        elif detect:
            fp += 1
        else:
            fn += 1  # every query HAS a sample in this benchmark
    P = tp / max(1, tp + fp)
    R = tp / max(1, tp + fn)
    F = 2 * P * R / max(1e-9, P + R)
    print(f"macro (song-level, thr={thr:.3f}): P {100*P:.1f}%  R {100*R:.1f}%  "
          f"F {100*F:.1f}%   [paper, 10-cand pools: P 83.3 R 50.0 F 62.5]")

    os.makedirs(args.out, exist_ok=True)
    sub_all = train_subsample(np.arange(len(y)))
    rf_full = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                     class_weight="balanced", n_jobs=-1,
                                     random_state=42).fit(X[sub_all], y[sub_all])
    import joblib
    joblib.dump({"model": rf_full, "features": FEATS, "threshold": thr},
                f"{args.out}/rf_model.joblib")
    order = np.argsort(rf_full.feature_importances_)[::-1]
    print("\nfeature importances (full-data model):")
    for i in order:
        print(f"  {FEATS[i]:12s} {rf_full.feature_importances_[i]:.3f}")
    print(f"\nmodel saved to {args.out}/rf_model.joblib")


if __name__ == "__main__":
    main()
