#!/usr/bin/env python3
"""
analyze_profile.py -- reusable nsys profile analyzer for the gpu_detect campaign.

Usage:
    python3 tools/analyze_profile.py <report.nsys-rep> <iteration_label> [--candidates N]

Example:
    python3 tools/analyze_profile.py /tmp/baseline.nsys-rep 00_baseline

For each iteration it writes plots/<iteration_label>/:
    kernels.png       top-15 kernels by total GPU time
    categories.png    GPU time by category (GEMM / NMF elementwise / DTW / ...)
    api_overhead.png  top CUDA API calls by total time
    summary.csv       machine-readable numbers behind the plots
    DIAGNOSTICS.md    written diagnosis + per-candidate arithmetic

Designed to be idempotent: re-running overwrites the iteration directory contents.
Never launches GPU work; only reads the .nsys-rep via `nsys stats` / `nsys export`.
"""

import argparse
import csv
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NSYS_CANDIDATES = [
    os.environ.get("NSYS", ""),
    "/usr/local/cuda/bin/nsys",
    shutil.which("nsys") or "",
]
NSYS = next((p for p in NSYS_CANDIDATES if p and os.path.exists(p)), "nsys")

# nsys stats report names (verified against Nsight Systems 2024.2; fallbacks for
# older/newer versions that renamed reports).
REPORTS = {
    "kern": ["cuda_gpu_kern_sum", "gpukernsum", "cudaapisum_gpu"],
    "mem": ["cuda_gpu_mem_time_sum", "gpumemtimesum"],
    "api": ["cuda_api_sum", "cudaapisum"],
    "gpu": ["cuda_gpu_sum", "gpusum"],  # combined kernels+memops coverage
}

PLOT_STYLE = "dark_background"
BAR_COLOR = "#4fc3f7"
ACCENT = "#ffb74d"

CATEGORY_ORDER = [
    "GEMM", "NMF elementwise", "DTW", "distance/znorm", "STFT",
    "other kernels", "memcpy/memset", "GPU idle (est.)",
]
CATEGORY_COLORS = {
    "GEMM": "#4fc3f7",
    "NMF elementwise": "#81c784",
    "DTW": "#e57373",
    "distance/znorm": "#ba68c8",
    "STFT": "#ffd54f",
    "other kernels": "#90a4ae",
    "memcpy/memset": "#ff8a65",
    "GPU idle (est.)": "#546e7a",
}


# --------------------------------------------------------------------------- util

def run_nsys_stats(report_path, report_names, outdir, tag):
    """Try report names in order; return path to the generated CSV or None."""
    for name in report_names:
        prefix = os.path.join(outdir, tag)
        cmd = [NSYS, "stats", "--report", name, "--format", "csv",
               "--force-export", "false", "--output", prefix, report_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"[warn] nsys stats {name} failed: {e}", file=sys.stderr)
            continue
        # nsys writes <prefix>_<reportname>.csv
        expected = f"{prefix}_{name}.csv"
        if proc.returncode == 0 and os.path.exists(expected) and os.path.getsize(expected) > 0:
            return expected
        print(f"[warn] report '{name}' unavailable on this nsys "
              f"(rc={proc.returncode}): {proc.stderr.strip()[:200]}", file=sys.stderr)
    return None


def read_csv_rows(path):
    """Read an nsys CSV, skipping any preamble lines before the header row."""
    if path is None or not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        lines = f.read().splitlines()
    # find the header line (first line containing 'Time' and a comma)
    start = 0
    for i, ln in enumerate(lines):
        if "," in ln and ("Time" in ln or "Name" in ln or "Operation" in ln):
            start = i
            break
    rows = list(csv.DictReader(lines[start:]))
    return rows


def col(row, *names):
    """Fetch the first matching column (exact, then substring match)."""
    for n in names:
        if n in row:
            return row[n]
    for key in row:
        for n in names:
            if n.lower() in key.lower():
                return row[key]
    return None


def to_float(v, default=0.0):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def ns_to_ms(ns):
    return ns / 1e6


def fmt_ms(ns):
    return f"{ns_to_ms(ns):,.2f} ms"


# ----------------------------------------------------------------- name handling

def shorten_kernel_name(name, maxlen=46):
    """Demangle-ish shortening of kernel names for plot labels."""
    n = name.strip().strip('"')
    low = n.lower()
    # Library kernels first
    if "sgemm" in low or ("gemm" in low and ("cutlass" in low or "ampere" in low
                                             or "cublas" in low)):
        m = re.search(r"(ampere_sgemm_\w+|sgemm_\w+|\w*gemm\w*)", low)
        detail = m.group(1) if m else "gemm"
        return f"cuBLAS GEMM [{detail[:28]}]"
    if "gemv" in low and ("cublas" in low or "ampere" in low):
        return "cuBLAS GEMV"
    if "fft" in low and ("regular_fft" in low or "vector_fft" in low
                         or "cufft" in low or low.startswith("void fft")):
        m = re.search(r"(regular_fft\w*|vector_fft\w*)", low)
        return f"cuFFT [{m.group(1)[:30]}]" if m else "cuFFT kernel"
    if "reduce_kernel" in low and "at::" in low:
        return "torch/thrust reduce"
    # Generic: take base identifier before template args / parens
    base = re.sub(r"^void\s+", "", n)
    base = base.split("(")[0]
    base = base.split("<")[0]
    base = base.split("::")[-1].strip()
    if not base:
        base = n
    return base[:maxlen] + ("…" if len(base) > maxlen else "")


def categorize_kernel(name):
    low = name.lower()
    if ("sgemm" in low or "gemm" in low or "cublas" in low or "cutlass" in low
            or "gemv" in low):
        return "GEMM"
    if re.search(r"ratio_|update_h|update_w|row_sum|col_sum|normalize", low):
        return "NMF elementwise"
    if "dtw" in low:
        return "DTW"
    if "distance" in low or "znorm" in low:
        return "distance/znorm"
    if "fft" in low or "window_frames" in low or "magnitude" in low:
        return "STFT"
    return "other kernels"


# ------------------------------------------------------------------ wall / idle

def derive_wall_span_ns(report_path, outdir):
    """Export to sqlite and compute the CUDA-active wall span (first runtime/GPU
    event start -> last event end). Returns (wall_ns or None, note string)."""
    sqlite_path = os.path.join(outdir, "export.sqlite")
    cmd = [NSYS, "export", "--type", "sqlite", "--force-overwrite", "true",
           "--output", sqlite_path, report_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except Exception as e:  # noqa: BLE001
        return None, f"sqlite export failed ({e}); idle time not derivable"
    if proc.returncode != 0 or not os.path.exists(sqlite_path):
        return None, f"sqlite export failed: {proc.stderr.strip()[:200]}"
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    spans = []
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY",
              "CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_RUNTIME"):
        if t in tables:
            row = cur.execute(f"SELECT MIN(start), MAX(end) FROM {t}").fetchone()
            if row and row[0] is not None:
                spans.append(row)
    con.close()
    try:
        os.remove(sqlite_path)
    except OSError:
        pass
    if not spans:
        return None, "no CUPTI tables in export; idle time not derivable"
    start = min(s for s, _ in spans)
    end = max(e for _, e in spans)
    return end - start, "wall span = first..last CUDA event (runtime API + GPU ops)"


# ----------------------------------------------------------------------- plots

def plot_kernels(kern_rows, total_kernel_ns, label, outpath, top_n=15):
    rows = sorted(kern_rows, key=lambda r: -r["total_ns"])[:top_n]
    names = [f"{r['short']}\n({r['instances']:,} inst)" for r in rows][::-1]
    times = [ns_to_ms(r["total_ns"]) for r in rows][::-1]
    pcts = [100.0 * r["total_ns"] / total_kernel_ns if total_kernel_ns else 0
            for r in rows][::-1]
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 0.55 * len(rows) + 2))
        bars = ax.barh(range(len(rows)), times, color=BAR_COLOR)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Total GPU time (ms)")
        ax.set_title(f"[{label}] Top {len(rows)} kernels by total GPU time")
        for b, p in zip(bars, pcts):
            ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2,
                    f"{p:.1f}%", va="center", fontsize=8, color=ACCENT)
        ax.margins(x=0.12)
        fig.tight_layout()
        fig.savefig(outpath, dpi=130)
        plt.close(fig)


def plot_categories(cat_ns, label, outpath):
    cats = [c for c in CATEGORY_ORDER if cat_ns.get(c, 0) > 0]
    vals = [ns_to_ms(cat_ns[c]) for c in cats]
    total = sum(vals) or 1.0
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        bars = ax.bar(range(len(cats)), vals,
                      color=[CATEGORY_COLORS.get(c, "#cccccc") for c in cats])
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"[{label}] Time by category "
                     "(GPU kernels by group + memcpy + estimated idle)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:,.0f} ms\n{100 * v / total:.1f}%",
                    ha="center", va="bottom", fontsize=8, color=ACCENT)
        ax.margins(y=0.18)
        fig.tight_layout()
        fig.savefig(outpath, dpi=130)
        plt.close(fig)


def plot_api(api_rows, label, outpath, top_n=12):
    rows = sorted(api_rows, key=lambda r: -r["total_ns"])[:top_n]
    names = [f"{r['name']}\n({r['calls']:,} calls)" for r in rows][::-1]
    times = [ns_to_ms(r["total_ns"]) for r in rows][::-1]
    launch = next((r for r in api_rows if "launchkernel" in r["name"].lower()), None)
    headline = (f"cudaLaunchKernel: {launch['calls']:,} calls, "
                f"{fmt_ms(launch['total_ns'])}" if launch else "no cudaLaunchKernel row")
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(11, 0.5 * len(rows) + 2.2))
        ax.barh(range(len(rows)), times, color="#ce93d8")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Total CPU time in API call (ms)")
        ax.set_title(f"[{label}] Top CUDA API calls by total time\n{headline}")
        fig.tight_layout()
        fig.savefig(outpath, dpi=130)
        plt.close(fig)


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="path to .nsys-rep file")
    ap.add_argument("label", help="iteration label, e.g. 00_baseline")
    ap.add_argument("--candidates", type=int, default=5,
                    help="number of library candidates scanned in the profiled run "
                         "(for per-candidate arithmetic; default 5)")
    ap.add_argument("--plots-root", default=None,
                    help="override plots root dir (default <repo>/plots)")
    args = ap.parse_args()

    if not os.path.exists(args.report):
        sys.exit(f"error: report not found: {args.report}")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_root = args.plots_root or os.path.join(repo_root, "plots")
    outdir = os.path.join(plots_root, args.label)
    os.makedirs(outdir, exist_ok=True)

    tmpdir = tempfile.mkdtemp(prefix="nsys_stats_")
    try:
        csv_kern = run_nsys_stats(args.report, REPORTS["kern"], tmpdir, "k")
        csv_mem = run_nsys_stats(args.report, REPORTS["mem"], tmpdir, "m")
        csv_api = run_nsys_stats(args.report, REPORTS["api"], tmpdir, "a")
        wall_ns, wall_note = derive_wall_span_ns(args.report, tmpdir)

        # ---- parse kernels
        kern_rows = []
        for r in read_csv_rows(csv_kern):
            name = col(r, "Name") or ""
            if not name:
                continue
            kern_rows.append({
                "name": name,
                "short": shorten_kernel_name(name),
                "category": categorize_kernel(name),
                "total_ns": to_float(col(r, "Total Time (ns)", "Total Time")),
                "instances": int(to_float(col(r, "Instances", "Count"))),
                "avg_ns": to_float(col(r, "Avg (ns)", "Average")),
            })
        total_kernel_ns = sum(r["total_ns"] for r in kern_rows)

        # ---- parse memops
        mem_rows = []
        for r in read_csv_rows(csv_mem):
            op = col(r, "Operation", "Name") or ""
            if not op:
                continue
            mem_rows.append({
                "op": op,
                "total_ns": to_float(col(r, "Total Time (ns)", "Total Time")),
                "count": int(to_float(col(r, "Count"))),
            })
        total_mem_ns = sum(r["total_ns"] for r in mem_rows)

        # ---- parse API
        api_rows = []
        for r in read_csv_rows(csv_api):
            name = col(r, "Name") or ""
            if not name:
                continue
            api_rows.append({
                "name": name,
                "total_ns": to_float(col(r, "Total Time (ns)", "Total Time")),
                "calls": int(to_float(col(r, "Num Calls", "Count"))),
            })

        # ---- categories
        cat_ns = {c: 0.0 for c in CATEGORY_ORDER}
        for r in kern_rows:
            cat_ns[r["category"]] += r["total_ns"]
        cat_ns["memcpy/memset"] = total_mem_ns
        gpu_busy_ns = total_kernel_ns + total_mem_ns  # upper bound (ignores overlap)
        idle_ns = None
        if wall_ns is not None:
            idle_ns = max(0.0, wall_ns - gpu_busy_ns)
            cat_ns["GPU idle (est.)"] = idle_ns

        # ---- plots
        if kern_rows:
            plot_kernels(kern_rows, total_kernel_ns, args.label,
                         os.path.join(outdir, "kernels.png"))
        plot_categories(cat_ns, args.label, os.path.join(outdir, "categories.png"))
        if api_rows:
            plot_api(api_rows, args.label, os.path.join(outdir, "api_overhead.png"))

        # ---- summary.csv
        with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["section", "name", "total_ns", "total_ms", "count",
                        "pct_of_section", "category"])
            for r in sorted(kern_rows, key=lambda r: -r["total_ns"]):
                w.writerow(["kernel", r["name"], int(r["total_ns"]),
                            f"{ns_to_ms(r['total_ns']):.3f}", r["instances"],
                            f"{100 * r['total_ns'] / total_kernel_ns:.2f}"
                            if total_kernel_ns else "0", r["category"]])
            for r in mem_rows:
                w.writerow(["memop", r["op"], int(r["total_ns"]),
                            f"{ns_to_ms(r['total_ns']):.3f}", r["count"],
                            f"{100 * r['total_ns'] / total_mem_ns:.2f}"
                            if total_mem_ns else "0", "memcpy/memset"])
            for r in sorted(api_rows, key=lambda r: -r["total_ns"])[:25]:
                w.writerow(["api", r["name"], int(r["total_ns"]),
                            f"{ns_to_ms(r['total_ns']):.3f}", r["calls"], "", ""])
            for c in CATEGORY_ORDER:
                if cat_ns.get(c, 0) > 0:
                    w.writerow(["category", c, int(cat_ns[c]),
                                f"{ns_to_ms(cat_ns[c]):.3f}", "", "", ""])
            if wall_ns is not None:
                w.writerow(["wall", "cuda_active_span", int(wall_ns),
                            f"{ns_to_ms(wall_ns):.3f}", "", "", ""])

        # ---- DIAGNOSTICS.md
        write_diagnostics(outdir, args, kern_rows, mem_rows, api_rows, cat_ns,
                          total_kernel_ns, total_mem_ns, wall_ns, idle_ns, wall_note)
        print(f"wrote {outdir}/: kernels.png categories.png api_overhead.png "
              "summary.csv DIAGNOSTICS.md")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def write_diagnostics(outdir, args, kern_rows, mem_rows, api_rows, cat_ns,
                      total_kernel_ns, total_mem_ns, wall_ns, idle_ns, wall_note):
    n_cand = max(1, args.candidates)
    launch = next((r for r in api_rows if "launchkernel" in r["name"].lower()), None)
    sync = next((r for r in api_rows
                 if "synchronize" in r["name"].lower()
                 or "streamsync" in r["name"].lower()), None)
    top_kern = sorted(kern_rows, key=lambda r: -r["total_ns"])[:10]
    denom = (wall_ns if wall_ns else (total_kernel_ns + total_mem_ns)) or 1.0

    # rank categories for bottleneck list
    ranked = sorted(((c, v) for c, v in cat_ns.items() if v > 0),
                    key=lambda kv: -kv[1])

    L = []
    L.append(f"# Profile diagnostics — {args.label}")
    L.append("")
    L.append(f"Report: `{os.path.abspath(args.report)}` | profiled run: "
             f"{n_cand}-candidate library scan")
    L.append("")
    L.append("## Headline numbers")
    L.append("")
    if wall_ns is not None:
        L.append(f"- CUDA-active wall span: **{fmt_ms(wall_ns)}** ({wall_note})")
    L.append(f"- Total GPU kernel time: **{fmt_ms(total_kernel_ns)}** "
             f"({100 * total_kernel_ns / denom:.1f}% of wall)")
    L.append(f"- Total memcpy/memset time: **{fmt_ms(total_mem_ns)}** "
             f"({100 * total_mem_ns / denom:.1f}% of wall)")
    if idle_ns is not None:
        L.append(f"- Estimated GPU idle: **{fmt_ms(idle_ns)}** "
                 f"({100 * idle_ns / denom:.1f}% of wall) — wall minus GPU busy; "
                 "overlap between kernels/copies would shrink busy time, so idle "
                 "is a lower bound")
    if launch:
        L.append(f"- cudaLaunchKernel: **{launch['calls']:,} calls**, "
                 f"{fmt_ms(launch['total_ns'])} CPU time "
                 f"(~{launch['total_ns'] / launch['calls'] / 1e3:.1f} us/call)")
    if sync:
        L.append(f"- {sync['name']}: {sync['calls']:,} calls, "
                 f"{fmt_ms(sync['total_ns'])} CPU time")
    L.append("")
    L.append("## Category split")
    L.append("")
    L.append("| category | time | % of wall | per candidate |")
    L.append("|---|---:|---:|---:|")
    for c, v in ranked:
        L.append(f"| {c} | {fmt_ms(v)} | {100 * v / denom:.1f}% | "
                 f"{fmt_ms(v / n_cand)} |")
    L.append("")
    L.append("## Top kernels")
    L.append("")
    L.append("| kernel | total | % of kernel time | instances | avg |")
    L.append("|---|---:|---:|---:|---:|")
    for r in top_kern:
        L.append(f"| {r['short']} | {fmt_ms(r['total_ns'])} | "
                 f"{100 * r['total_ns'] / total_kernel_ns:.1f}% | "
                 f"{r['instances']:,} | {r['avg_ns'] / 1e3:.1f} us |")
    L.append("")
    if launch:
        L.append(f"Launches per candidate: ~{launch['calls'] // n_cand:,} "
                 f"({launch['calls']:,} / {n_cand}).")
        L.append("")
    if mem_rows:
        L.append("## Memory operations")
        L.append("")
        L.append("| op | time | count |")
        L.append("|---|---:|---:|")
        for r in sorted(mem_rows, key=lambda r: -r["total_ns"]):
            L.append(f"| {r['op']} | {fmt_ms(r['total_ns'])} | {r['count']:,} |")
        L.append("")
    L.append("## Diagnosis — top 3 bottlenecks")
    L.append("")
    for i, (c, v) in enumerate(ranked[:3], 1):
        L.append(f"{i}. **{c}** — {fmt_ms(v)} ({100 * v / denom:.1f}% of wall; "
                 f"{fmt_ms(v / n_cand)} per candidate).")
    L.append("")
    L.append("## What to optimize next")
    L.append("")
    L.append(_recommendations(ranked, launch, idle_ns, denom, top_kern, n_cand))
    L.append("")

    with open(os.path.join(outdir, "DIAGNOSTICS.md"), "w") as f:
        f.write("\n".join(L))


def _recommendations(ranked, launch, idle_ns, denom, top_kern, n_cand):
    """Heuristic, data-driven recommendation text."""
    recs = []
    cat = dict(ranked)
    idle = cat.get("GPU idle (est.)", 0)
    if idle and idle / denom > 0.25:
        recs.append(
            f"- GPU is idle ~{100 * idle / denom:.0f}% of the wall span. Likely "
            "causes: serialized host loop over candidates, blocking "
            "synchronization between tiny kernels, or CPU-side work between "
            "launches. Consider CUDA streams (one per candidate), CUDA graphs, "
            "or batching candidates together.")
    if launch and launch["calls"] / max(1, n_cand) > 10000:
        recs.append(
            f"- {launch['calls']:,} kernel launches "
            f"(~{launch['calls'] // n_cand:,}/candidate) — launch overhead "
            "dominates if kernels average <10 us. Fuse elementwise NMF kernels "
            "(ratio/update/sum/normalize) and/or capture the NMF iteration loop "
            "in a CUDA graph.")
    small = [r for r in top_kern if r["avg_ns"] < 10_000 and r["instances"] > 1000]
    if small:
        names = ", ".join(r["short"] for r in small[:4])
        recs.append(
            f"- Tiny high-count kernels ({names}) average <10 us each — strong "
            "fusion candidates; each launch costs more CPU time than GPU time.")
    if cat.get("GEMM", 0) / denom > 0.2:
        recs.append(
            "- GEMM is a large share: check matrix shapes per NMF iteration — "
            "batched cublasSgemmStridedBatched across candidates/pitches usually "
            "beats many small GEMMs.")
    if cat.get("DTW", 0) / denom > 0.2:
        recs.append(
            "- DTW is a large share: increase per-launch work (process multiple "
            "diagonals or candidates per launch) and keep the band in "
            "shared memory.")
    if cat.get("memcpy/memset", 0) / denom > 0.15:
        recs.append(
            "- Memcpy share is significant: use pinned host memory, overlap "
            "copies with compute on separate streams, and keep intermediates "
            "device-resident across pipeline stages.")
    if not recs:
        recs.append("- No single dominant bottleneck; profile individual kernels "
                    "with ncu for occupancy/memory-bound analysis.")
    return "\n".join(recs)


if __name__ == "__main__":
    main()
