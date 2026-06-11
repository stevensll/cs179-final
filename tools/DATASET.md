# Building the evaluation dataset (Sample100)

Instructions for the agent (or human) tasked with downloading the dataset
audio. Read fully before running anything.

## Background

The Gururani & Lerch paper claims its 80-pair dataset + URLs are published at
`github.com/SiddGururani/sample_detection` — **they are not** (the repo is
MATLAB code reading from the author's local disk; wiki empty, no releases, not
in git history). The substitute — and the de-facto public benchmark for this
task — is **Sample100** (Van Balen 2011, `github.com/jvbalen/sample_100`,
already cloned to `~/sample_100`): 105 sample relations between 76 hip-hop
queries and 68 sampled originals, with per-sample annotations (timestamps,
repetition counts, sample type, interpolation flag). Metadata only; audio must
be fetched per track.

## One-time setup

```bash
python3 -m pip install --user yt-dlp     # the only missing dependency
export PATH="$HOME/.local/bin:$PATH"     # if yt-dlp isn't found after install
```

ffmpeg is already on the box. No sudo needed.

## Run

```bash
cd ~/cs179-final
# smoke test: manifest + 3 downloads
python3 tools/scrape_sample100.py --meta ~/sample_100 --out datasets/sample100 --limit 3
# full run (144 tracks x ~5s sleep + download time: expect ~1-2 h unattended)
python3 tools/scrape_sample100.py --meta ~/sample_100 --out datasets/sample100
```

Idempotent: re-run to resume after failures; existing `audio/T###.wav` are
skipped. `--only T001,T013` re-fetches specific tracks.

## Outputs

```
datasets/sample100/
├── eval_pairs.csv     # query_file -> expected_file ground truth (+ annotations)
├── download_log.csv   # per-download status + ACTUAL YouTube title/duration
└── audio/T###.wav     # 44.1 kHz 16-bit stereo
```

`eval_pairs.csv` excludes `interpolation=yes` rows (re-recorded samples — the
paper's method detects recording reuse, not composition reuse; including them
would book guaranteed failures as misses).

## Verification (do not skip — this burned us before)

YouTube search can return covers, live versions, or remasters; a wrong file
makes a test pair silently unwinnable (see CLAUDE.md log, v2/v3 entries).

1. `wc -l datasets/sample100/download_log.csv` and grep for non-`ok` statuses;
   retry or hand-fix those (`--only ...`).
2. Spot-check `yt_title` column against artist/title — flag covers, "live",
   "remix", "sped up", "8D" etc.
3. Duration sanity: compare WAV durations (`ffprobe`) against `yt_duration_s`
   and against plausibility (2–8 min; a 30 s result is a preview/short).
4. Optional strong check for suspicious pairs: clip the annotated sample region
   (`t_query_s` in eval_pairs.csv) and verify with
   `tools/visualize_match.py` or a `gpu_detect --clip` run.

## Evaluating the detector against it

One query at a time (library = all originals + optionally other tracks as
noise, matching the dataset's intended retrieval protocol):

```bash
./build/gpu_detect --iters 60 datasets/sample100/audio/T001.wav datasets/sample100/audio
```

Scoring protocol: a pair counts as a hit if the expected original ranks #1
among candidates (exclude the query itself; the runner should also exclude
other queries from the library or accept them as distractors — record which).
A batch runner script over eval_pairs.csv is TODO (trivial bash loop; ~7 s per
candidate pair on the A5000 means a full 76-query x 68-candidate sweep is
~10 h single-GPU — start with a 10-query subset).
