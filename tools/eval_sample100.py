#!/usr/bin/env python3
"""Batch evaluation of gpu_detect against the Sample100 dataset.

Protocol (per the dataset's intended retrieval setup): each query track is run
against a candidate library containing the ORIGINAL (sampled) tracks only; a
query scores a hit@1 if any of its annotated originals ranks first. Tracks not
yet downloaded are skipped and reported, so this runs fine on a partial scrape.

Usage:
  eval_sample100.py [--data datasets/sample100] [--binary build/gpu_detect]
                    [--iters 60] [--limit N] [--queries T001,T013]
                    [--full-library]

Outputs <data>/results/results.csv (per query: ranks of expected originals,
top-3 listing) and a summary (hit@1, hit@3, MRR) on stdout.
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

RANK_RE = re.compile(r"^\s*(\d+)\.\s+(.+?\.wav)\s+score\s+([0-9.]+)")


def build_library(audio: Path, originals, lib: Path):
    """Symlink the original tracks into a clean candidate dir. gpu_detect's
    self-skip uses fs::equivalent, which resolves symlinks, so a track that is
    both query and original is still excluded from its own run."""
    lib.mkdir(parents=True, exist_ok=True)
    for old in lib.glob("*.wav"):
        old.unlink()
    present = []
    for t in sorted(originals):
        src = audio / t
        if src.exists():
            (lib / t).symlink_to(src.resolve())
            present.append(t)
    return present


def run_query(binary, iters, query_path, lib):
    r = subprocess.run([binary, "--iters", str(iters), str(query_path), str(lib)],
                       capture_output=True, text=True, timeout=3600)
    ranking = []  # [(rank, file, score)]
    in_ranking = False
    for line in r.stdout.splitlines():
        if line.startswith("ranking"):
            in_ranking = True
            continue
        if in_ranking:
            m = RANK_RE.match(line)
            if m:
                ranking.append((int(m.group(1)), m.group(2).strip(), float(m.group(3))))
    return ranking, r.returncode, r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/sample100")
    ap.add_argument("--binary", default="build/gpu_detect")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0, help="evaluate at most N queries")
    ap.add_argument("--queries", default="", help="comma-separated query track ids")
    ap.add_argument("--full-library", action="store_true",
                    help="use the whole audio dir as candidates (adds queries as distractors)")
    args = ap.parse_args()

    data = Path(args.data)
    audio = data / "audio"
    pairs = list(csv.DictReader(open(data / "eval_pairs.csv")))

    # excluded_tracks.txt: ids with wrong/missing audio (see VERIFICATION.md)
    excluded = set()
    excl_file = data / "excluded_tracks.txt"
    if excl_file.exists():
        for line in excl_file.read_text().splitlines():
            tok = line.split("#")[0].strip()
            if tok:
                excluded.add(f"{tok}.wav")
    if excluded:
        print(f"excluding {len(excluded)} tracks: {sorted(t[:-4] for t in excluded)}")

    expected = defaultdict(set)  # query_file -> {original_file, ...}
    for r in pairs:
        if r["query_file"] in excluded or r["expected_file"] in excluded:
            continue
        expected[r["query_file"]].add(r["expected_file"])
    originals = set().union(*expected.values()) - excluded

    if args.full_library:
        lib = audio
        lib_files = {p.name for p in audio.glob("*.wav")}
    else:
        lib = data / "library"
        lib_files = set(build_library(audio, originals, lib))

    only = set(f"{t.strip()}.wav" for t in args.queries.split(",") if t.strip())
    results_dir = data / "results"
    results_dir.mkdir(exist_ok=True)
    out_csv = results_dir / "results.csv"

    n_eval = n_hit1 = n_hit3 = 0
    mrr_sum = 0.0
    skipped = []
    t0 = time.time()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "n_expected", "n_expected_in_lib", "best_expected_rank",
                    "n_candidates", "hit1", "hit3", "top3"])
        for q, exp in sorted(expected.items()):
            if only and q not in only:
                continue
            if not (audio / q).exists():
                skipped.append((q, "query audio missing"))
                continue
            exp_in_lib = exp & lib_files
            if not exp_in_lib:
                skipped.append((q, "no expected original downloaded yet"))
                continue
            if args.limit and n_eval >= args.limit:
                break

            ranking, rc, err = run_query(args.binary, args.iters, audio / q, lib)
            if rc != 0 or not ranking:
                skipped.append((q, f"gpu_detect failed: {err.strip()[:80]}"))
                continue
            rank_of = {name: rank for rank, name, _ in ranking}
            best = min((rank_of.get(e, 10**9) for e in exp_in_lib))
            hit1, hit3 = best == 1, best <= 3
            n_eval += 1
            n_hit1 += hit1
            n_hit3 += hit3
            mrr_sum += 1.0 / best if best < 10**9 else 0.0
            top3 = "; ".join(f"{n}:{s:.3f}" for _, n, s in ranking[:3])
            w.writerow([q, len(exp), len(exp_in_lib), best, len(ranking),
                        int(hit1), int(hit3), top3])
            f.flush()
            print(f"{q}: best expected rank {best}/{len(ranking)} "
                  f"{'HIT' if hit1 else 'miss'}  [{time.time()-t0:.0f}s]")

    print(f"\n=== summary ===")
    print(f"queries evaluated: {n_eval}  (skipped {len(skipped)})")
    if n_eval:
        print(f"hit@1: {n_hit1}/{n_eval} = {n_hit1/n_eval:.1%}")
        print(f"hit@3: {n_hit3}/{n_eval} = {n_hit3/n_eval:.1%}")
        print(f"MRR:   {mrr_sum/n_eval:.3f}")
    for q, why in skipped:
        print(f"  skipped {q}: {why}")
    print(f"per-query detail: {out_csv}")
    return 0 if n_eval else 1


if __name__ == "__main__":
    sys.exit(main())
