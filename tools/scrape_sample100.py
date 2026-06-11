#!/usr/bin/env python3
"""Download audio for the Sample100 dataset (jvbalen/sample_100) via YouTube
search, and build the evaluation manifest for gpu_detect.

The dataset repo ships metadata only (copyright); audio is fetched per track as
`ytsearch1:"<artist> - <title>"` -> bestaudio -> 44.1 kHz 16-bit stereo WAV
named `<track_id>.wav`. Idempotent: existing WAVs are skipped, so re-running
resumes after failures. Every download's actual YouTube title/duration is
logged to download_log.csv for human spot-checking (search can fetch covers or
live versions — that burned us before; verify before trusting results).

Usage:
  scrape_sample100.py --meta ~/sample_100 --out datasets/sample100 [--limit N]
                      [--only T001,T002] [--sleep 5] [--manifest-only]

Requires: ffmpeg (on box) and yt-dlp (`python3 -m pip install --user yt-dlp`).
"""

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def read_rows(path):
    """The dataset CSVs use bare-CR line endings; normalize before parsing."""
    text = Path(path).read_bytes().decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return list(csv.DictReader(io.StringIO(text), skipinitialspace=True))


def ytdlp_cmd():
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    if subprocess.run([sys.executable, "-m", "yt_dlp", "--version"],
                      capture_output=True).returncode == 0:
        return [sys.executable, "-m", "yt_dlp"]
    sys.exit("yt-dlp not found. Install with: python3 -m pip install --user yt-dlp")


def download(track_id, artist, title, outdir, ytdlp):
    wav = outdir / f"{track_id}.wav"
    if wav.exists():
        return "exists", "", 0
    tmp = outdir / f"{track_id}.tmp"
    for stale in outdir.glob(f"{track_id}.tmp*"):
        stale.unlink()
    query = f"ytsearch1:{artist} - {title}"
    r = subprocess.run(
        ytdlp + ["-f", "bestaudio", "--no-playlist", "--print-json",
                 "-o", str(tmp) + ".%(ext)s", query],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return "download_failed", r.stderr.strip().splitlines()[-1] if r.stderr else "", 0
    try:
        info = json.loads(r.stdout.splitlines()[-1])
        yt_title, yt_dur = info.get("title", "?"), int(info.get("duration") or 0)
    except (json.JSONDecodeError, IndexError):
        yt_title, yt_dur = "?", 0
    src = next(iter(outdir.glob(f"{track_id}.tmp*")), None)
    if src is None:
        return "no_output_file", yt_title, yt_dur
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                        "-ac", "2", "-ar", "44100", "-sample_fmt", "s16", str(wav)],
                       capture_output=True, text=True)
    src.unlink()
    if r.returncode != 0 or not wav.exists():
        return "convert_failed", r.stderr.strip(), yt_dur
    return "ok", yt_title, yt_dur


def write_manifest(meta, out):
    """eval_pairs.csv: one row per sample relation usable by our detector.
    Query = the song that USES the sample; expected = the sampled original.
    Interpolations (re-recorded samples) are excluded: the paper's method
    detects reuse of the recording, not the composition."""
    tracks = {r["track_id"]: r for r in read_rows(meta / "tracks.csv")}
    pairs, skipped = [], 0
    for r in read_rows(meta / "samples.csv"):
        if r["interpolation"].strip().lower() == "yes":
            skipped += 1
            continue
        q, o = r["sample_track_id"], r["original_track_id"]
        pairs.append({
            "sample_id": r["sample_id"],
            "query_file": f"{q}.wav",
            "expected_file": f"{o}.wav",
            "query_song": f'{tracks[q]["artist"]} - {tracks[q]["title"]}',
            "expected_song": f'{tracks[o]["artist"]} - {tracks[o]["title"]}',
            "t_original_s": r["t_original"],
            "t_query_s": r["t_sample"],
            "n_repetitions": r["n_repetitions"],
            "sample_type": r["sample_type"],
            "interpolation": r["interpolation"],
            "comments": r.get("comments", "").strip(),
        })
    out_csv = out / "eval_pairs.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        w.writeheader()
        w.writerows(pairs)
    print(f"wrote {out_csv}: {len(pairs)} pairs ({skipped} interpolations excluded)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="path to cloned jvbalen/sample_100")
    ap.add_argument("--out", required=True, help="output dataset dir")
    ap.add_argument("--limit", type=int, default=0, help="download at most N new tracks")
    ap.add_argument("--only", default="", help="comma-separated track ids")
    ap.add_argument("--sleep", type=float, default=5.0, help="seconds between downloads")
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()

    meta, out = Path(args.meta).expanduser(), Path(args.out).expanduser()
    audio = out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    write_manifest(meta, out)
    if args.manifest_only:
        return

    ytdlp = ytdlp_cmd()
    only = set(t.strip() for t in args.only.split(",") if t.strip())
    tracks = read_rows(meta / "tracks.csv")
    log_path = out / "download_log.csv"
    new_log = not log_path.exists()
    done = 0
    with open(log_path, "a", newline="") as logf:
        log = csv.writer(logf)
        if new_log:
            log.writerow(["track_id", "artist", "title", "status",
                          "yt_title", "yt_duration_s"])
        for t in tracks:
            tid = t["track_id"]
            if only and tid not in only:
                continue
            status, detail, dur = download(tid, t["artist"], t["title"], audio, ytdlp)
            if status == "exists":
                continue
            log.writerow([tid, t["artist"], t["title"], status, detail, dur])
            logf.flush()
            print(f"{tid}: {status}  {detail[:70]} ({dur}s)")
            done += 1
            if args.limit and done >= args.limit:
                break
            time.sleep(args.sleep)
    print(f"done: {done} new downloads; audio in {audio}; check {log_path}")


if __name__ == "__main__":
    main()
