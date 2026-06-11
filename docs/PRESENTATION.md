# Presentation companion — GPU sample detection

Speaking notes / slide source for the final presentation, aligned to the v3
implementation (2026-06-10). Every number here is traceable to the CLAUDE.md
build log or `docs/TESTING.md`; nothing below was generated for this document.
Paper-block correspondence: `docs/PAPER-MAPPING.md`. Parameters:
`src/common/params.hpp`. Design detail: `docs/TECHNICAL.md`.

## 1. The problem, and what the demo shows

Sampling reuses a snippet of one recording inside another — usually pitch
shifted, time stretched, chopped, and buried under new instruments. That
defeats fingerprint matchers (Shazam-style), which is why this needs source
separation (NMF) rather than spectrogram lookup.

**Input:** a query WAV and a directory of library songs.
**Output:** the library ranked by match score (lower = better), each line
carrying the estimated pitch shift and the matched locations in both songs.

Output format (this is `gpu_detect`'s actual format; the numbers are the
logged v3 results for the passing real pair, Madonna's *Hung Up* sampling
ABBA's *Gimme Gimme Gimme*; other library entries elided):

```
$ ./build/gpu_detect --iters 60 "music/Hung Up.wav" music
...
ranking (best match first):
1. Gimme Gimme.wav             score 0.3000 (shift +0.00 st, cand @97s -> query @48s)
2. <runner-up>                 score 0.6070 ...
...
total ~35 s (5 candidates)
```

The story in one sentence: the detector not only ranks the right song first by
a 2× margin, it tells you *where* — ABBA at 1:37 maps onto Madonna at 0:48 —
and the claim can be checked by eye with `tools/visualize_match.py`.

## 2. Pipeline walkthrough (with sizes)

The full block diagram, paper-block by paper-block with every deviation
justified, is in `docs/PAPER-MAPPING.md` — present that figure. The narration,
with data sizes for a ~4-minute song (22.05 kHz, FFT 4096, hop 1024,
~21.5 frames/s ≈ 5200 frames):

1. **Preprocess + STFT** (both songs): downmix, RMS-normalize, 2:1 decimate to
   22.05 kHz; magnitude spectrogram V ∈ ℝ^(2049×~5200) ≈ 42 MB float.
2. **Candidate NMF**: KL-divergence NMF on the *full* candidate song → K=40
   spectral templates W_o (2049×40) and activations H_o (40×~5200). (Paper
   uses K=10 on a known ~4.5 s sample; we don't know where the sample is, so
   the rank scales with content — deviation D1.)
3. **Pitch-shifted template bank**: the frequency axis of W_o rescaled by
   2^(p/12) for 41 hypotheses, −5..+5 st in 0.25 st steps (D3), columns
   re-normalized to unit L2 (D2).
4. **PFNMF on the query**: W = [W_o^p fixed | L=20 free templates], all 41
   shifts solved as ONE cuBLAS strided-batched problem. The free templates
   absorb everything in the query that isn't the candidate.
5. **Distance matrices**: H/max(H) (paper §3.2.1), then regularized Pearson
   correlation distance between activation columns, weighted by PFNMF source
   attribution (D4). One ~5200×5200 matrix per shift: 41 × No × Ns floats —
   ~6 GB for the largest current pair (5:33 query), the dominant allocation.
6. **Banded subsequence DTW** (D5): paper eq. 2 run over every dense 4 s band
   of candidate frames (hop 1 s) — the "where is the sample in the candidate"
   search, reusing one distance matrix per shift at zero extra NMF cost.
7. **Scoring**: per (band, shift) dip = min/mean of the path-length-normalized
   last row, with a path-slope sanity bound (paper §3.3.2); then pitch
   selectivity (A1: a real match dips at ONE shift, junk dips at all) and a
   min/median selection-bias correction across all hypotheses (A2).

## 3. The GPU story (for a CS 179 audience)

### What is batched, and why

The workload is independent at every level: candidates × 41 pitch shifts ×
~160 DTW bands. The headline batching decision: **all 41 pitch-shift PFNMFs
run as a single strided-batched problem** — one `cublasSgemmStridedBatched`
call per GEMM per multiplicative-update step, with V shared across the 41
stacked problems (it appears once in memory; the ratio kernel indexes
`i % (M·N)`). Per iteration that is 4 batched GEMMs + custom elementwise and
reduction kernels; PFNMF's "frozen templates" is one branch in the W-update
kernel (`r >= n_fixed`).

### The kernel no library provides: banded wavefront DTW

DTW is a 2-D dependency-carrying recurrence — the textbook hard-to-parallelize
case, and no NVIDIA library implements it. `dtw_band_diag_kernel` exploits the
classic wavefront property: every cell on anti-diagonal d depends only on
diagonals d−1 and d−2, so all cells on a diagonal run in parallel (one launch
per diagonal, one thread per cell). The twist here is the *banding*:
`blockIdx.y` enumerates (shift-matrix, band-slot) pairs, so all 41 shifts and
a chunk of 4 s candidate bands advance in the same launch, all reading from
the same distance matrices. A second matrix L tracks path lengths in lockstep
for the final cost normalization. Band chunking (4 slots) bounds the C/L
working memory. Known limitation, deliberately left on the table: still one
kernel launch per anti-diagonal (a tiled multi-diagonal kernel is in TODO.md —
measure first).

### Kernel inventory (all in `src/gpu/kernels.cu`, strategy comments inline)

| Kernel | Strategy (one line) |
|---|---|
| `window_frames_kernel` | one thread per windowed sample; Hann weight computed inline — frames overlap 4×, cosf hides behind the gather loads |
| `magnitude_kernel` | one thread per (frame, bin); writes the transposed bins×frames layout so NMF's transpose is free |
| `ratio_batched_kernel` | grid-stride elementwise V ⊘ (WH+ε); V shared across the P stacked problems |
| `col_sum_batched_kernel` | one block per (column, problem); two-stage shared-memory tree reduction |
| `row_sum_batched_kernel` | same reduction shape with contiguous reads |
| `update_h_batched_kernel` | grid-stride multiplicative H update |
| `update_w_batched_kernel` | grid-stride W update; PFNMF freezing = skip columns r < n_fixed |
| `pitch_templates_batched_kernel` | linear-interp frequency-axis gather, all 41 shifts in one launch — no library expresses this |
| `normalize_fixed_columns_batched_kernel` | block per (column, problem): L2-norm reduce, then scale to unit norm |
| `normalize_columns_kernel` | P=1 variant; folds the removed norms into H_o so W·H is preserved |
| `max_normalize_batched_kernel` | block per problem: two-stage max reduction over the K fixed rows, then rescale all rows (paper H/max(H)) |
| `znorm_batched_kernel` | thread per (problem, column): z-normalize with Z_REG + PFNMF source-attribution weight folded into the z-vectors |
| `distance_batched_kernel` | thread per D(i,j) cell, grid z over shifts; K-dim dot of z-columns = regularized 1−r |
| `dtw_band_diag_kernel` | wavefront DP: launch per anti-diagonal, thread per cell, blockIdx.y = (matrix, band) — the flagship |

### What is deliberately a library call

- **cuFFT** for the batched R2C STFT: a correct hand-rolled FFT is a weekend,
  a cuFFT-competitive one is a quarter-long project; the custom work
  (windowing, magnitude+transpose) still surrounds it. The CPU demo hand-rolls
  a radix-2 FFT anyway, so an FFT was written once regardless.
- **cuBLAS** for the NMF GEMMs (a v3 decision that *reversed* the original
  plan — see the dated postscript in `docs/gpu-library-vs-custom-kernels.md`).
  Defensible because NMF itself exists in no NVIDIA library: the update loop,
  all elementwise/reduction steps, the template freezing, and everything
  downstream remain custom, and DTW + pitch-shift resampling have no library
  escape hatch at all. The originally planned three-way benchmark
  (CPU / cuBLAS / fused-custom NMF) remains future work (TODO.md).

## 4. Results

### The verification ladder (the correctness argument)

Correctness is argued with controls whose right answer is known *exactly*,
not with vibes (all v3, `docs/TESTING.md`):

| Test | Predicted | Measured |
|---|---|---|
| Self-match canary (song vs itself) | rank 1, shift 0, aligned | 0.108 @ +0.00 st ✓ |
| Synthetic: 8 s of Move On Up mixed into Shape of My Heart @30 s | dip @ +0.00 st, location recovered | 0.247 @ +0.00 st, cand @1s, query @35s ✓ |
| Same insert, resampled +6% speed | dip @ 12·log2(1.06) = **+1.01 st** | 0.218 @ **+1.00 st** ✓ |

The +6% control is the strongest slide: the detector lands on the
theoretically predicted quarter-semitone bin, on a grid the paper doesn't
have, with the location recovered unaided.

### Real pairs — including the honest failures

| Pair | Result |
|---|---|
| Hung Up → Gimme Gimme | **pass, decisive**: 0.300 vs 0.607 runner-up (2× margin), Gimme @97s → Hung Up @48s |
| Lucid Dreams → Shape of My Heart | **near miss**: true match *found* (cand @8s, +0.00 st, the riff) at 0.591 but ranked #2 behind a melodic-resemblance confound at 0.542 |
| Touch The Sky → Move On Up | **fail**: true −0.50 st alignment found consistently, but at 0.727 it ranks #3 behind two confounds (0.607/0.631) |

The failure mode is specific and explainable: in both losses the detector
*finds* the true alignment (correct shift, plausible locations) and is edged
out by harmonically sparse candidates whose clean riff templates match similar
minor-key material anywhere — exactly the precision problem the paper's
13-feature random-forest stage exists to solve. We implement ~4 of its 13
features as heuristics; the classifier needs ~500 labeled queries we don't
have (see `docs/PAPER-MAPPING.md`, "What the missing classifier costs").
For calibration, the paper itself reports 34–50% recall on this task.

### Performance (from the build log)

- Full GPU library scan, full songs, 60 NMF iterations, 41 shifts, dense 4 s
  band search: **~35 s for 5 candidates (~7 s per pair)** — down from ~120 s
  before the v2 cuBLAS strided-batched PFNMF + banded-DTW redesign.
- CPU vs GPU, v1-era matched config (45 s query, 30 iters, 1 window, 41
  shifts, 1 candidate): **149 s CPU vs 2.2 s GPU = 68×**. Caveat stated
  plainly: that measurement is of the v1 algorithm on both sides; the CPU demo
  was never ported to v2/v3, so no current-algorithm CPU number exists. A
  full-config v1 CPU library scan extrapolates to ~8 hours (~230×) — by
  design, not run.
- Memory: removing a dead null-surrogate computation halved the distance-
  matrix footprint (~12 → ~6 GB on the largest pair) and fixed an OOM on
  5½-minute queries.

## 5. Evaluation plan (in progress)

Test pairs so far are n=3; the proper benchmark is **Sample100** (Van Balen
2011) — the public standard for this task: 105 sample relations between 76
hip-hop queries and 68 originals, with timestamps and sample-type annotations.
(The paper's own dataset was supposed to be public; its repo turned out to
contain no annotations — documented in the build log.)

- Ground truth generated: `datasets/sample100/eval_pairs.csv` — 103 pairs
  (3 interpolation rows excluded: recording reuse only).
- Audio download from YouTube is underway (`tools/scrape_sample100.py`;
  ~85/144 tracks at the time of writing), with a verification protocol against
  wrong-version downloads (`tools/DATASET.md`).
- Runner ready: `tools/eval_sample100.py` — per query, library = annotated
  originals; reports **hit@1, hit@3, MRR**, tolerant of a partial scrape.
  A full 76-query sweep is ~10 h single-GPU (~7 s/pair), so it starts with a
  subset; no results yet.

## 6. Live demo script

Verified against the binaries' actual CLIs (sources: `src/gpu/gpu_detect.cu`,
`tools/visualize_match.py`).

```bash
make                                   # CMake under the hood, builds into build/

# 1. The headline: scan the library for what Hung Up samples (~35 s, 5 candidates)
./build/gpu_detect --iters 60 "music/Hung Up.wav" music

# 2. Eyeball the claimed match: candidate offset / query offset from the output line
#    "cand @97s -> query @48s"  (usage: A.wav A_off B.wav B_off duration out.png [max_hz])
python3 tools/visualize_match.py "music/Gimme Gimme.wav" 97 "music/Hung Up.wav" 48 10 match.png

# 3. Under the hood: per-hypothesis score landscape (window x shift) on stderr
SD_DEBUG=1 ./build/gpu_detect --iters 60 "music/Hung Up.wav" music 2> landscape.txt

# 4. If time is short: truncate the query
./build/gpu_detect --max-seconds 60 --iters 60 "music/Hung Up.wav" music
```

The CPU demo (`./build/cpu_demo [--max-seconds S] [--iters I] [--windows W]
<query.wav> <library_dir>`) is the v1 algorithm and is hours-slow at full
config — demo it only with `--max-seconds`, and present it as the historical
baseline, not a parity reference. `gpu_detect` also accepts `--clip` for
hand-trimmed query segments; the alternate scoring path behind it (absolute
normalized cost instead of dip depth) is implemented but untested (TODO.md) —
do not demo it.

## 7. Limitations and future work (from TODO.md)

- **The classifier is the binding constraint.** Both real-pair failures are
  the confound class the paper's random forest separates; the detector already
  finds the true alignments. Needs DTW backtracking + the 13 path features +
  ~500 labeled queries — and the GPU throughput is precisely what makes
  generating that training set tractable.
- **Sample100 evaluation** to replace n=3 anecdotes with hit@1/hit@3/MRR.
- **CPU demo port** to the v3 algorithm (or explicit re-scoping as a per-stage
  reference); CPU/GPU score parity does not currently hold.
- **Performance left on the table:** tiled multi-diagonal DTW (currently ~200k
  small launches per candidate), multi-GPU candidate split across the two
  A5000s (embarrassingly parallel), cuFFT plan reuse, ncu profiling, and the
  three-way NMF benchmark (CPU / cuBLAS / fused custom).
- **Algorithm:** multiple band lengths (real samples span 0.5–25 s; we search
  4 s only), shift-consistency across bands as a corroboration feature.
