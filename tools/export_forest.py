#!/usr/bin/env python3
"""Export the trained sklearn random forest to a flat binary for the CUDA
forest-inference kernel, plus a verification set (feature rows + sklearn's
probabilities) so GPU output can be checked for numerical parity.

Binary layout (little-endian):
  char[4]  magic "SDRF"
  int32    version (1), n_trees, n_features, n_nodes
  int32[n_trees]  root node index of each tree
  int32[n_nodes]  feature index per node (-1 = leaf)
  float[n_nodes]  threshold per node (for leaves: class-1 probability)
  int32[n_nodes]  left child   (x[feature] <= threshold -> left; sklearn rule)
  int32[n_nodes]  right child

Usage: export_forest.py [--model plots/rf/rf_model.joblib] [--out plots/rf]
                        [--verify-rows 200000]
"""

import argparse
import csv
import glob
import struct

import joblib
import numpy as np


def floor32(t):
    """Largest float32 <= float64 t. sklearn compares float32-cast features
    against float64 thresholds; no float32 lies in (floor32(t), t], so the
    float32 comparison x <= floor32(t) is decision-exact."""
    t32 = np.float32(t)
    return np.nextafter(t32, np.float32(-np.inf)) if t32 > t else t32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="plots/rf/rf_model.joblib")
    ap.add_argument("--features-dir", default="datasets/sample100/features")
    ap.add_argument("--out", default="plots/rf")
    ap.add_argument("--verify-rows", type=int, default=200000)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    rf, feats = bundle["model"], bundle["features"]

    roots, feat, thr, left, right = [], [], [], [], []
    for est in rf.estimators_:
        t = est.tree_
        base = len(feat)
        roots.append(base)
        value = t.value[:, 0, :]                       # (n_nodes, 2) class weights
        prob1 = value[:, 1] / np.maximum(1e-12, value.sum(axis=1))
        for n in range(t.node_count):
            is_leaf = t.children_left[n] == -1
            feat.append(-1 if is_leaf else int(t.feature[n]))
            thr.append(float(prob1[n]) if is_leaf else float(floor32(t.threshold[n])))
            left.append(-1 if is_leaf else base + int(t.children_left[n]))
            right.append(-1 if is_leaf else base + int(t.children_right[n]))

    out_bin = f"{args.out}/forest.bin"
    with open(out_bin, "wb") as f:
        f.write(b"SDRF")
        f.write(struct.pack("<iiii", 1, len(roots), len(feats), len(feat)))
        f.write(np.asarray(roots, "<i4").tobytes())
        f.write(np.asarray(feat, "<i4").tobytes())
        f.write(np.asarray(thr, "<f4").tobytes())
        f.write(np.asarray(left, "<i4").tobytes())
        f.write(np.asarray(right, "<i4").tobytes())
    print(f"wrote {out_bin}: {len(roots)} trees, {len(feat):,} nodes, "
          f"{len(feats)} features")

    # verification set: real corpus rows + float64 sklearn probabilities
    rows = []
    for fp in sorted(glob.glob(f"{args.features_dir}/*.csv")):
        for r in csv.DictReader(open(fp)):
            rows.append([float(r[k]) for k in feats])
            if len(rows) >= args.verify_rows:
                break
        if len(rows) >= args.verify_rows:
            break
    X = np.asarray(rows, np.float32)
    probs = rf.predict_proba(X)[:, 1].astype(np.float32)
    X.tofile(f"{args.out}/verify_rows.f32")
    probs.tofile(f"{args.out}/verify_probs.f32")
    print(f"wrote verification set: {len(X):,} rows "
          f"({args.out}/verify_rows.f32, verify_probs.f32)")


if __name__ == "__main__":
    main()
