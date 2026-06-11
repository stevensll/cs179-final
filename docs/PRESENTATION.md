# Presentation companion — GPU sample detection

Speaking notes / slide source for the final presentation, aligned to the final
build (2026-06-11: log-frequency front end, K=32, RF classifier stage, FIL
inference kernel). Every number is traceable to the CLAUDE.md build log,
`RESULTS.md`, or `plots/PERF-CHARACTERIZATION.md`; nothing below was generated
for this document. Paper correspondence: `docs/PAPER-MAPPING.md` (diagram) and
`docs/PAPER-VS-OURS.md` (narrative). Parameters: `src/common/params.hpp`.

## 1. The problem, and what the demo shows

Sampling reuses a snippet of one recording inside another — usually pitch
shifted, time stretched, chopped, and buried under new instruments. That
defeats fingerprint matchers (Shazam-style), which is why this needs source
separation (NMF) rather than spectrogram lookup.

**Input:** a query WAV and a directory of library songs.
**Output:** the library ranked by match score (lower = better), each line
carrying the estimated pitch shift and the matched locations in both songs.

```
$ ./build/gpu_detect --iters 60 "music/Hung Up.wav" music
...
ranking (best match first):
1. Gimme Gimme.wav              score 0.4149 (shift +0.00 st, cand @96s -> query @47s)
2. Shape of My Heart.wav        score 0.5676 (shift +2.75 st, cand @20s -> query @186s)
...
total 3.0 s (2 gpus)
```

The story in one sentence: the detector ranks the right song first, tells you
*where* — ABBA at 1:36 maps onto Madonna at 0:47, checkable by eye with
`tools/visualize_match.py` — and the full five-song scan takes 3 seconds.

## 2. Pipeline walkthrough (with sizes)

Present the block diagram from `docs/PAPER-MAPPING.md`. Narration, with data
sizes for a ~4-minute song (22.05 kHz, FFT 4096, hop 1024, ~21.5 frames/s ≈
5200 frames):

1. **Preprocess + STFT** (both songs): downmix, RMS-normalize, 2:1 decimate;
   magnitude spectrogram 2049×~5200, then pooled through a triangular
   **log-frequency filterbank** (4 bins/semitone from 55 Hz → 367 bins,
   one GEMM; deviation D6). Two wins: ~5.6× less compute everywhere
   downstream, and a pitch shift becomes an **exact integer translation** of
   a template. V ∈ ℝ^(367×~5200) ≈ 7.6 MB float.
2. **Candidate NMF**: KL-divergence NMF on the *full* candidate song → K=32
   templates W_o (367×32) + activations (32×~5200). (Paper uses K=10 on a
   known ~4.5 s sample; we don't know where the sample is, so the rank scales
   with content — D1. K=32 chosen by config sweep + full-benchmark
   confirmation.) Cached per library song (`.tcache`).
3. **Pitch-shifted template bank**: W_o translated by p·4 bins for 41
   hypotheses, −5..+5 st in 0.25 st steps (D3), columns re-normalized (D2).
4. **PFNMF on the query**: W = [W_o^p fixed | L=20 free templates], all 41
   shifts × all candidates of a group solved as ONE strided-batched problem
   (every problem is query-sized, so candidates batch without padding). The
   free templates absorb everything in the query that isn't the candidate.
5. **Distance matrices**: H/max(H) (paper §3.2.1), then regularized Pearson
   correlation distance, weighted by PFNMF source attribution (D4); written
   in a diagonal-skewed layout so the DTW wavefront reads coalesced. Chunked
   over shifts under a 4 GB budget (the largest pairs would need ~22 GB).
6. **Banded subsequence DTW** (D5): paper eq. 2 over every dense 4 s band of
   candidate frames (hop 1 s) — the "where is the sample in the candidate"
   search, reusing one distance matrix per shift at zero extra NMF cost.
7. **Heuristic scoring**: per (band, shift) dip = min/mean of the
   path-length-normalized last row, slope sanity bound (§3.3.2), pitch
   selectivity (A1: real matches dip at ONE shift), min/median selection-bias
   correction (A2).
8. **Classifier stage (paper §3.3–3.4)**: DTW re-swept with predecessor
   recording for the top hypotheses, paths backtracked into the paper's 13
   path-geometry features (`gpu_detect --features`); a 200-tree random forest
   trained on Sample100 (`tools/train_rf.py`, leakage-safe GroupKFold);
   inference by a FIL-style CUDA kernel (`rf_infer`) with sklearn-exact
   output.

## 3. The GPU story (for a CS 179 audience)

### What is batched, and why

The workload is independent at every level: candidates × 41 pitch shifts ×
~160 DTW bands. Headline decision: **(candidates × shifts) PFNMFs run as one
strided-batched problem** — possible because candidate length never enters
PFNMF (only the frozen templates differ), so there is no padding. Per
iteration: one fused custom kernel + 2 cuBLAS TF32 strided-batched GEMMs +
custom reductions/updates; "frozen templates" is one branch in the W-update
kernel (`r >= n_fixed`). ~70 kernel launches per candidate (project baseline:
~325,000).

### The two kernels no library provides

- **Persistent-band DTW** (`dtw_band_kernel`): DTW is a 2-D
  dependency-carrying recurrence — the textbook hard-to-parallelize case.
  One block per (shift-matrix, band) walks ALL anti-diagonals in-kernel;
  thread t owns band row t; only the two live diagonals exist, in shared
  memory (the cost/length matrices never touch global memory — was ~80 GB of
  traffic per candidate). D is read in the diagonal-skewed layout written by
  the distance kernel, so every wavefront step is one coalesced read. The
  slope filter and min/mean/argmin scoring reduce in-kernel: each band
  returns 4 floats instead of a matrix slab. A predecessor-recording variant
  (`dtw_band_preds_kernel`) feeds the classifier's path backtracking.
- **Forest inference** (`forest_predict_kernel`, `rf_infer`): one thread per
  feature row walks all 200 trees; each node is a packed 16-byte struct so a
  node visit is a single 128-bit load. The forest is 24.9M nodes (~400 MB) at
  depth ~60: ncu shows the purest latency-bound profile in the project
  (long_scoreboard 452 cycles/issue, compute 4%) — packing the node quartered
  the latency chains and bought 3.0×. Output matches sklearn's
  `predict_proba` to 6e-8 (after fixing a real float64-threshold subtlety —
  good war story: sklearn compares float32 features against float64
  thresholds; we export the largest float32 ≤ each threshold to make the
  float32 comparison decision-exact).

### Kernel inventory (strategy comments inline in the sources)

| Kernel | Strategy (one line) |
|---|---|
| `window_frames_kernel` | thread per windowed sample; inline Hann hides behind the gather loads |
| `magnitude_kernel` | thread per (frame, bin); writes transposed so NMF's transpose is free |
| `fused_wh_ratio_kernel` | **the flagship**: W·H + division epilogue in one kernel — R ≤ 60 fits a single shared tile, so WH (the largest per-iteration intermediate) is never materialized; 64×64 block tile, 4×4 per thread |
| `col/row_sum_batched_kernel` | block per (vector, problem); two-stage shared-memory tree reduction |
| `update_h/w_batched_kernel` | coalesced multiplicative updates; PFNMF freezing = skip r < n_fixed |
| `pitch_templates_batched_kernel` | integer-translation gather on the log axis (exact), linear-interp fallback on the linear axis |
| `normalize_*`, `max_normalize`, `znorm` | column-norm / max / regularized z-score reductions; znorm folds Z_REG + source attribution into the z-vectors |
| `distance_batched_kernel` | thread per cell; K-dim dot of z-columns = regularized 1−r; **diagonal-skewed output** for the DTW |
| `dtw_band_kernel` (+`_preds`) | persistent banded wavefront DP in shared memory (above) |
| `forest_predict_kernel` | FIL-style packed-node tree traversal (above) |

### What is deliberately a library call

- **cuFFT** for the batched R2C STFT (the CPU reference hand-rolls a radix-2
  FFT anyway, so an FFT was written once regardless).
- **cuBLAS** for the two large-k NMF GEMMs — TF32 tensor-core strided-batched.
  Defensible because NMF exists in no NVIDIA library, and the skinny-k W·H
  product where cuBLAS underperforms was replaced by the fused custom kernel
  (measured ~8 TFLOPS vs ~430 GFLOPS for cuBLAS on that shape). Rationale doc:
  `docs/gpu-library-vs-custom-kernels.md`.

## 4. Results

### The verification ladder (the correctness argument)

Controls whose right answer is known *exactly* (fresh run, final build;
`tests/run_ladder.sh`):

| Test | Predicted | Measured |
|---|---|---|
| Self-match canary (song vs itself) | rank 1, shift 0, aligned | 0.171 @ +0.00 st ✓ |
| Synthetic: 8 s of Move On Up mixed into Shape of My Heart @30 s | dip @ +0.00 st, location recovered | 0.421 @ +0.00 st, locations exact ✓ |
| Same insert, resampled +6% speed | dip @ 12·log₂(1.06) = **+1.01 st** | 0.346 @ **+1.00 st** ✓ |

The +6% control is the strongest slide: the detector lands on the
theoretically predicted quarter-semitone bin, on a grid the paper doesn't
have, with the location recovered unaided. Bonus correctness card: the CPU
reference reproduces GPU scores **to 4 decimals** on all of these, despite
TF32 + fast-math on the GPU side.

### The headline: Sample100 benchmark vs the paper (RESULTS.md)

70 queries × 64-candidate pools, leakage-safe out-of-fold evaluation:

| | heuristic | + random forest | paper (10-cand pools) |
|---|---|---|---|
| hit@1 | 11.4% | **50.0%** | — |
| MRR | 0.214 | **0.557** | — |
| macro P / R / F | — | 50.0 / 100.0 / **66.7%** | 83.3 / 50.0 / 62.5% |
| simulated 10-pool (paper's exact protocol) | — | top-1 60.3%, **F 75.0%** | F 62.5% |

Talking points: our pools are 6.4× larger and adversarially composed (the
sampled-source canon, not random songs); the RF was trained on public data
with GroupKFold leakage protection; feature importances land on path geometry
(slope, cost, deviation) — independently confirming the paper's §3.3 thesis.
Honest caveats are in RESULTS.md — recall pegs at 100% because Sample100 has
no sample-free queries; precision at the paper's R=50 operating point is
52.2%, and P reaches 85–100% at recalls of 23–41%.

### Real pairs on `music/` (heuristic stage, fresh run)

| Pair | Result |
|---|---|
| Hung Up → Gimme Gimme | **pass, decisive**: 0.415 vs 0.568 runner-up |
| Lucid Dreams → Shape of My Heart | **pass**: rank #1 (0.583) |
| Touch The Sky → Move On Up | **fail at heuristic stage**: true −0.25 st alignment found, ranks #3 behind two confounds — the exact failure class the RF separates on the benchmark |

### Performance

| Workload | before | after | |
|---|---:|---:|---:|
| 5-candidate scan | 42 s | **3.0 s** | 14× |
| full song vs 64 candidates | ~340 s | **19.3 s** | 18× |
| 15 s clip vs 64 candidates | 21.7 s | **2.3 s** | 9.4× |
| full 70-query benchmark | ~8.5 h | **~13 min** | ~40× |
| forest inference | 0.16 M rows/s (sklearn, 32 cores) | **3.1 M rows/s** (1 GPU) | 19× |

Progression figure: `plots/speedup_progression.png` (every kept optimization,
with the accuracy overlay showing the gates *improved* during the campaign).
Kernel-level characterization (occupancy, DRAM, tensor-core utilization, warp
stalls, idle decomposition): `plots/PERF-CHARACTERIZATION.md`. One-line
summary for questions: the fused kernel is compute/memory balanced at its
shared-memory occupancy cap; distance is L1-resident; DTW is latency-bound on
its wavefront; the GEMMs are bandwidth-bound (tensor cores at 32–46% is the
ceiling for these skinny shapes); the forest kernel is pure latency.

## 5. Live demo script

```bash
make                                   # CMake under the hood, builds into build/
tests/run_ladder.sh                    # 4 structural gates, ~30 s warm

# 1. The headline: scan the library for what Hung Up samples (3 s)
./build/gpu_detect --iters 60 "music/Hung Up.wav" music

# 2. Eyeball the claimed match (offsets from the output line)
python3 tools/visualize_match.py "music/Gimme Gimme.wav" 96 "music/Hung Up.wav" 47 10 match.png

# 3. CPU/GPU parity, the correctness card (~50 s)
./build/gpu_detect --max-seconds 30 --iters 30 --no-cache "music/Move On Up.wav" tests/fixtures/probe_selfcopy
./build/cpu_demo   --max-seconds 30 --iters 30 "music/Move On Up.wav" tests/fixtures/probe_selfcopy

# 4. Forest inference: GPU vs sklearn parity + speed (needs plots/rf/ artifacts)
./build/rf_infer plots/rf/forest.bin plots/rf/verify_rows.f32 plots/rf/verify_probs.f32 13

# 5. Under the hood: per-hypothesis score landscape on stderr
SD_DEBUG=1 ./build/gpu_detect --iters 60 "music/Hung Up.wav" music 2> landscape.txt
```

## 6. Limitations and future work (TODO.md)

- Residual ~6% single-GPU idle is pageable H2D staging (measured, §6 of the
  characterization) — pinned buffers + async copies would reclaim it.
- Forest kernel could interleave 2–4 trees per thread to overlap latency
  chains (it is already 19× sklearn).
- Classifier features: multiple band lengths (samples span 0.5–25 s; we
  extract at 4 s), shift-consistency across bands.
- `--clip` mode's accuracy was never validated with a real hand-trimmed clip.
- Location-level ("micro") P/R/F per the paper is not implemented (band-level
  hypotheses exist; the per-occurrence evaluation harness does not).
