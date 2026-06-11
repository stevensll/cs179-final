#!/usr/bin/env python3
"""Config-matrix sweep: build gpu_detect variants via -DSD_DEFS, measure each
variant's speed (warm 5-candidate scan) and accuracy (Sample100 eval subset),
write a CSV and a Pareto scatter (cost vs MRR).

Usage: sweep_configs.py [--queries T001,...] [--iters 60] [--configs name1,name2]
Outputs: plots/sweep/sweep_results.csv, plots/sweep/pareto.png
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = [
    # name        SD_DEFS (semicolon-separated; "" = production defaults)
    ("mel4",      ""),
    ("linear",    "SD_MEL_BINS=0"),
    ("mel2",      "SD_MEL_BINS=184;SD_BINS_PER_ST=2"),
    ("mel4_k24",  "SD_RANK_K=24"),
    ("mel4_k32",  "SD_RANK_K=32"),
    ("mel2_k24",  "SD_MEL_BINS=184;SD_BINS_PER_ST=2;SD_RANK_K=24"),
    ("hop2048",   "SD_HOP=2048"),
    ("shift05",   "SD_SHIFT_STEP4=2"),
    ("bandhop2",  "SD_WINDOW_HOP_SECONDS=2"),
    ("hop_shift", "SD_HOP=2048;SD_SHIFT_STEP4=2"),
]

DEFAULT_QUERIES = "T001,T004,T006,T008,T009,T012,T013,T016,T017,T020"


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def build_variant(name, defs):
    bdir = ROOT / "build" / "sweep" / name
    r = run(["cmake", "-S", ".", "-B", str(bdir), "-DCMAKE_BUILD_TYPE=Release",
             f"-DSD_DEFS={defs}"])
    if r.returncode != 0:
        sys.exit(f"cmake failed for {name}:\n{r.stderr[-2000:]}")
    r = run(["cmake", "--build", str(bdir), "-j", "--target", "gpu_detect"])
    if r.returncode != 0:
        sys.exit(f"build failed for {name}:\n{r.stderr[-2000:]}")
    return bdir / "gpu_detect"


def scan_seconds(binary, iters):
    """Warm 5-candidate scan time: first run warms this variant's tcache,
    second run is the measurement."""
    cmd = [str(binary), "--iters", str(iters), str(ROOT / "music/Hung Up.wav"),
           str(ROOT / "music")]
    run(cmd, timeout=1800)
    out = run(cmd, timeout=1800).stdout
    m = re.search(r"total ([0-9.]+) s", out)
    return float(m.group(1)) if m else float("nan")


def eval_metrics(binary, queries, iters):
    t0 = time.time()
    r = run(["python3", "tools/eval_sample100.py", "--binary", str(binary),
             "--queries", queries, "--gpus", "2", "--iters", str(iters)],
            timeout=7200)
    wall = time.time() - t0
    mrr = hit1 = float("nan")
    nq = 0
    m = re.search(r"MRR:\s+([0-9.]+)", r.stdout)
    if m:
        mrr = float(m.group(1))
    m = re.search(r"hit@1: (\d+)/(\d+)", r.stdout)
    if m:
        hit1, nq = int(m.group(1)), int(m.group(2))
    return mrr, hit1, nq, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=DEFAULT_QUERIES)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--configs", default="", help="subset of config names")
    args = ap.parse_args()

    chosen = [c for c in CONFIGS if not args.configs or c[0] in args.configs.split(",")]
    outdir = ROOT / "plots" / "sweep"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, defs in chosen:
        print(f"=== {name} ({defs or 'defaults'}) ===", flush=True)
        binary = build_variant(name, defs)
        scan_s = scan_seconds(binary, args.iters)
        mrr, hit1, nq, wall = eval_metrics(binary, args.queries, args.iters)
        per_query = wall / max(1, nq)
        rows.append({"config": name, "defs": defs, "scan_s": scan_s,
                     "eval_s_per_query": round(per_query, 1), "mrr": mrr,
                     "hit1": hit1, "n_queries": nq})
        print(f"  scan {scan_s:.1f} s | eval {per_query:.1f} s/query | "
              f"MRR {mrr:.3f} | hit@1 {hit1}/{nq}", flush=True)

    # merge with prior runs so incremental sweeps share one frontier
    csv_path = outdir / "sweep_results.csv"
    if csv_path.exists():
        fresh = {r["config"] for r in rows}
        for old in csv.DictReader(open(csv_path)):
            if old["config"] not in fresh:
                old["scan_s"] = float(old["scan_s"])
                old["mrr"] = float(old["mrr"])
                old["hit1"] = int(old["hit1"])
                old["n_queries"] = int(old["n_queries"])
                rows.append(old)
    rows.sort(key=lambda r: r["scan_s"])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [r["scan_s"] for r in rows]
    ys = [r["mrr"] for r in rows]
    # Pareto-efficient: no other config is both faster and more accurate
    pareto = [r for r in rows
              if not any(o["scan_s"] < r["scan_s"] and o["mrr"] >= r["mrr"] or
                         o["scan_s"] <= r["scan_s"] and o["mrr"] > r["mrr"]
                         for o in rows)]
    ax.scatter(xs, ys, c="#7fbfff", s=60, zorder=3)
    pf = sorted(pareto, key=lambda r: r["scan_s"])
    ax.plot([r["scan_s"] for r in pf], [r["mrr"] for r in pf],
            c="#7fff7f", lw=1.5, ls="--", zorder=2, label="Pareto frontier")
    for r in rows:
        mark = " *" if r in pareto else ""
        ax.annotate(r["config"] + mark, (r["scan_s"], r["mrr"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("5-candidate scan, warm cache (s)  [lower = faster]")
    ax.set_ylabel(f"MRR on {rows[0]['n_queries']}-query Sample100 subset")
    ax.set_title("Approximation vs accuracy — config sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "pareto.png", dpi=120)
    print(f"wrote {outdir}/sweep_results.csv and pareto.png")


if __name__ == "__main__":
    main()
