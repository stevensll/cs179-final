# CS 179 Final Project — GPU Sample Detection

Steven Lei

Goal: Given an input song that samples another song, find the sampled song from a
directory of songs. Implements Gururani & Lerch, *Automatic Sample Detection in
Polyphonic Music* (ISMIR 2017, `docs/`) — NMF/PFNMF source separation + subsequence
DTW — with custom CUDA kernels (two deliberate GPU libraries: cuFFT for the STFT,
cuBLAS for the strided-batched NMF GEMMs; everything else is custom). See
`PROPOSAL.md` for the plan, `docs/TECHNICAL.md` for the design, `TODO.md` for
known limitations and cleanup.

## Build

Requires: CUDA toolkit (12.x), CMake ≥ 3.24, libsndfile dev headers. No other
dependencies.

```
make            # configures + builds into build/ (CMake under the hood)
```

## Usage

```
./build/gpu_detect [--max-seconds S] [--iters I] [--clip] <query.wav> <library_dir>
./build/cpu_demo   [--max-seconds S] [--iters I] [--windows W] <query.wav> <library_dir>
```

Both print the library ranked by match score (lower = better, best match first).
The query file is skipped if it is inside the library directory. Shared flags:
`--max-seconds` truncates the query (essential for the CPU demo — full-length CPU
runs take hours by design), `--iters` NMF iterations (default 100; 60 is plenty).
`gpu_detect --clip` is for a hand-trimmed query that IS the suspected sample
(scores by absolute alignment cost instead of dip depth; implemented, not yet
validated). **`cpu_demo` is the v1 algorithm** (snippet-window hypotheses +
shuffle-null calibration, not the current GPU algorithm — see `TODO.md`); its
`--windows` flag sets the v1 snippet hypotheses per candidate (default 6).

Example (actual `gpu_detect` output format; numbers are the logged v3 results for
the passing pair, other library entries elided):

```
$ ./build/gpu_detect --iters 60 "music/Hung Up.wav" music
...
ranking (best match first):
1. Gimme Gimme.wav             score 0.3000 (shift +0.00 st, cand @97s -> query @48s)
2. <runner-up>                 score 0.6070 ...
...
total ~35 s (5 candidates)
```

## How it works

For each library candidate: NMF (K=40, KL divergence; the paper's K=10 scaled up
because templates model the full song, not a known snippet) extracts spectral
templates from the full candidate song; for all 41 pitch-shift hypotheses (−5..+5 st,
quarter-semitone steps) the frequency-rescaled templates are held fixed in one
batched partially-fixed NMF (L=20 free templates) over the query's spectrogram;
regularized correlation distance between candidate and query activations feeds a
banded subsequence DTW (wavefront-parallel, all shifts and bands batched) that
densely searches every 4 s candidate window; the score is the depth of the best
DTW cost dip, normalized for pitch selectivity (a real match dips at one shift
only) and selection bias, with a path-slope sanity bound (paper §3.3.2).
Matrix multiplies are cuBLAS (strided-batched); STFT is cuFFT; everything else
is custom kernels (windowing, magnitude, NMF update steps, reductions,
pitch-template resampling, normalization, correlation distance, banded
wavefront DTW). **The block-by-block correspondence to the paper — with a
diagram and every deviation justified — is in `docs/PAPER-MAPPING.md`.**

A match can be eyeballed with the visual aid:

```
python3 tools/visualize_match.py "music/Move On Up.wav" 0 "music/Touch The Sky.wav" 2.5 10 out.png
```

renders the two segments' spectrograms stacked for comparison (uses the
`cand @Xs -> query @Ys` locations that `gpu_detect` prints).

## Results

| Test | Expected | Result |
|---|---|---|
| Self-match canary (song vs itself) | top score, shift 0, aligned offsets | **pass** (0.11 @ +0.00 st) |
| Synthetic: 8 s sample mixed into another song | dip @ +0.00 st, found at insert location | **pass** (0.25 @ +0.00 st, locations exact) |
| Synthetic: same, resampled +6% speed | dip @ +1.01 st | **pass** (0.22 @ +1.00 st) |
| Hung Up → Gimme Gimme | rank #1 | **pass, decisive** (0.300 vs 0.607 runner-up) |
| Lucid Dreams → Shape of My Heart | rank #1 | **near miss** (true match found @ +0.00 st but ranks #2: 0.591 vs 0.542) |
| Touch The Sky → Move On Up | rank #1 | **fail** (true −0.50 st alignment found, ranks #3) |

The synthetic controls verify the implementation responds exactly as theory
predicts to pitch shifting and time stretching, and recovers the sample's
location in both songs unaided. In the two failing real pairs the detector
*finds* the true alignment (correct shift, plausible locations) but is edged
out by melodic-resemblance confounds — harmonically sparse candidates whose
clean riff templates match similar minor-key arpeggios in other songs. That
residual confound class is exactly what the paper's 13-feature random-forest
stage exists to separate (out of scope without its labeled training set; see
`docs/PAPER-MAPPING.md`, "What the missing classifier costs", and `TODO.md`).

## Evaluation

To go beyond the n=3 anecdotes above, the repo carries the **Sample100**
benchmark (Van Balen 2011, the public standard for sample detection):
`datasets/sample100/eval_pairs.csv` holds the ground truth (103 query→original
pairs; 3 interpolation rows excluded), and `tools/eval_sample100.py` runs
`gpu_detect` per query against the annotated originals and reports hit@1 /
hit@3 / MRR. Audio: 143/144 tracks downloaded (`tools/scrape_sample100.py`,
protocol in `tools/DATASET.md`; 4 bad/missing tracks excluded via
`excluded_tracks.txt`). First 5-query subset (64-candidate library):
ranks of the true original 2, 3, 6, 17, 57 → hit@1 0/5, hit@3 2/5, MRR 0.215
(random baseline ≈ 0.073). Same picture as the anecdotes, now at scale: the
true original lands far above chance but melodic-resemblance confounds take
#1 — the quantified cost of the paper's missing classifier stage.

## Performance

Full GPU library scan (full songs, 60 NMF iterations, 41 pitch shifts, dense
4 s window search, 3 candidates): **~35 s** (was ~120 s before the cuBLAS
strided-batched PFNMF + banded-DTW redesign). Earlier matched-config benchmark
of the v1 pipeline: CPU 149 s vs GPU 2.2 s (68×); the CPU demo still implements
the v1 algorithm (see `TODO.md`), so v2 CPU numbers don't exist yet — a full v1
CPU library scan extrapolates to hours either way.

## Potential improvements

See `TODO.md` for the full ledger. Headlines: the paper's 13 DTW-path features +
random-forest classifier (fixes the failing test class; the GPU throughput makes
generating its ~500-query training set tractable); fused custom NMF GEMM kernels
for a three-way benchmark against the current cuBLAS path (reimplement from the
v1 design — see CLAUDE.md log); multi-GPU candidate split; tiled DTW kernel.

## Repo layout

```
src/common/   audio I/O + preprocessing (shared)     PROPOSAL.md   proposal + timeline
src/cpu/      single-threaded v1 reference pipeline  docs/         paper, design docs, tests
src/gpu/      custom kernels + cuBLAS wrappers       TODO.md       cleanup + limitations
              + GPU pipeline                         CLAUDE.md     operating notes + build log
music/        song library                           build/        out-of-source build dir
tools/        visualize_match.py, Sample100 scraper + eval runner, DATASET.md
datasets/     sample100/ evaluation set (eval_pairs.csv ground truth + audio/)
```
