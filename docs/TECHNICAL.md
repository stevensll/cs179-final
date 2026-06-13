# Technical Design

Deep reference for algorithm parameters, build, testing, and benchmark
methodology (the README is the primary writeup). Algorithm follows Gururani &
Lerch 2017 (`Automatic-Sample-Detection-in-Polyphonic-Music.pdf` in this
directory); the block-by-block paper ↔ code correspondence, with every
deviation justified, is the mapping table in the README ("Paper → GPU
Mapping") — this file does not duplicate it. Library-vs-custom kernel
rationale: `gpu-library-vs-custom-kernels.md` (see its dated postscript). Test
cases and current status: the README (Evaluation + Errata). History and
measured numbers: CLAUDE.md build log.

## Task formulation

Input: a query song Q and a library of candidate songs {S₁ … Sₙ} (`music/`).
For each candidate Sᵢ, compute the minimum banded subsequence-DTW alignment cost of
Sᵢ's NMF templates being "found" inside Q. Output: candidates ranked by that score,
best match first.

Note the role reversal vs the paper: the paper searches for a known *sample* inside
candidate *songs*; we search for each library song's templates inside the *query*.
Per pair, the candidate Sᵢ plays the paper's "sample/original" role (templates
extracted from it) and the query Q plays the "song" role (PFNMF'd against those
templates). The two structural deviations from the paper (full rationale in
the README's Paper → GPU Mapping section):

- **Random-forest classifier trained on Sample100** (originally out of scope;
  added 2026-06-11): heuristic score normalization (pitch selectivity +
  min/median selection-bias correction, below) ranks; the paper's 13-feature
  forest reranks/decides (tools/train_rf.py, src/gpu/rf_infer.cu).
- **No known sample snippet** — templates come from one NMF over the *full*
  candidate song (rank scaled accordingly), and the "which part of the candidate
  is the sample" search happens inside DTW as dense 4 s bands of the distance
  matrix, at zero extra NMF cost. (An earlier v1 design used 8 s max-RMS snippet
  hypotheses and a shuffle-null calibration; both were replaced — see CLAUDE.md
  log.)

## Parameters

`src/common/params.hpp` is the authoritative copy (each value carries its rationale
there); this table mirrors it. All values from the paper unless marked *deviation*.

| Parameter | Value | Notes |
|---|---|---|
| Analysis sample rate | 22 050 Hz | inputs decimated 2:1 from 44.1 kHz |
| STFT | block 4096, hop 1024, Hann | 2049 bins, ~21.5 frames/s |
| Log-frequency front end | `MEL_BINS`=367, 4 bins/semitone from 55 Hz | *deviation* (D6 in the README mapping table): triangular log-spaced pooling before NMF — ~5.6× less downstream compute AND pitch shifts become exact integer translations of templates; `-DSD_DEFS="SD_MEL_BINS=0"` restores the linear axis |
| `RANK_K` | **32** | *deviation*: paper uses K=10 for a known ~4.5 s sample; we model the full candidate song, so the rank is content-scaled (paper §3.1.1 rationale). 32 locked by config sweep + full-benchmark confirmation (MRR 0.214 vs 0.203 at K=40) |
| `RANK_L` | 20 | free mixture templates in PFNMF |
| `DEFAULT_ITERS` | 100 | NMF multiplicative-update iterations (60 is plenty in practice) |
| Pitch shifts | **41**: −5..+5 st in 0.25 st steps | *deviation* from the paper's 12-step set: resampling speedups give fractional shifts and a 0.5 st grid miss empirically destroys the DTW dip |
| `Z_REG` | 0.01 | correlation regularizer at the paper's H/max(H) scale |
| `WINDOW_SECONDS` | {4} | DTW band length ≈ the paper's average sample length (§4.1); a worst-across-{4,8} variant was tried and regressed |
| `WINDOW_HOP_SECONDS` | 1 | band hop (dense candidate-window search) |

Working-set sizes: the raw spectrogram (2049×~5200 ≈ 42 MB for a 4-min song)
pools to V ≈ 367×~5200 ≈ 7.6 MB. The dominant allocation is the
diagonal-skewed distance stack D (per shift: No×(No+Ns−1) floats — near-
quadratic in candidate length), chunked over the shift axis under a 4 GB
budget; the largest Sample100 pairs would otherwise need ~22 GB.

## Pipeline

See the README mapping table for the stage-by-stage correspondence (preprocess → STFT →
candidate NMF → pitch-shifted template bank → batched PFNMF → regularized
correlation distance → banded subsequence DTW → scoring), including which kernel
implements each block. Scoring summary (host code, `gpu_score_candidate` tail in
`src/gpu/gpu_pipeline.cu`):

1. Per (band, shift): min and mean of the path-length-normalized DTW last row,
   with end points rejected when the path's warp factor L/T falls outside
   [0.7, 1.5] (slope sanity, paper §3.3.2).
2. Raw dip = min/mean; **pitch selectivity**: each band's dip is divided by that
   band's median dip across all 41 shifts (a real match dips at one shift only).
3. **Selection-bias correction**: the best hypothesis is divided by the
   candidate's own median over all (band × shift) hypotheses (a min over ~6800
   hypotheses otherwise favors longer candidates).
4. `--clip` mode: when the query is a hand-trimmed segment that IS the suspected
   sample, there is no non-sample baseline for a dip to stand out from, so step 2's
   raw dip is replaced by the absolute normalized cost (min alone). Implemented;
   not yet validated (README Improvements).

## Repo layout

See the README's **Repo layout** section for the file tree and the
executables' flags. (`tests/` holds `make_fixtures.sh` + `run_ladder.sh`; the
per-stage split of `kernels.cu` / `cpu_pipeline.cpp` is a known cleanup item.)

## Build

- CMake ≥ 3.24 (box has 4.0), `LANGUAGES CXX CUDA`, `find_package(CUDAToolkit)`,
  `CMAKE_CUDA_ARCHITECTURES native` (sm_86 here), pkg-config for sndfile; links
  `CUDA::cufft` and `CUDA::cublas`.
- Compile flags: `-O3 -Wall -Wextra` (host), `-O3 --generate-line-info` (CUDA, for
  profiler/compute-sanitizer friendliness).
- **Code style: the Google C++ Style Guide**, enforced by the checked-in
  `.clang-format` (Google base; documented deviations in `docs/STYLE.md`:
  4-space indent, 100-column limit, repo snake_case naming). Braces on every
  `if`/`for`/`while` body — no single-line control flow, no compact C
  idioms (`strcmp(...) == 0`, not `!strcmp(...)`). Reformat with
  `clang-format -i <files>`.
- Top-level `Makefile` wrapper: `make` = configure (if needed) + build into `build/`;
  `make clean` wipes `build/`. Graders never need to know CMake.
- There are no build options/flags; in particular, no separate cuBLAS-baseline NMF
  build exists (the three-way NMF benchmark is future work — the custom
  GEMM would have to be reimplemented from the v1 design, see CLAUDE.md log).

## Testing

Current practice:

- **Verification ladder**, re-run after every algorithm/scoring change: self-match
  canary (song vs itself → rank 1 @ +0.00 st, aligned locations), two synthetic
  controls with exactly predictable answers (8 s insert → dip @ +0.00 st at the
  insert location; same insert resampled +6% → dip @ +1.00 st, theory +1.01), and
  the real pairs (see the README's Errata).
- **CPU/GPU parity holds**: the CPU reference mirrors the production pipeline
  stage-for-stage (same seeds) and reproduces GPU scores to 4 printed decimals
  on the ladder gates, despite TF32 + fast-math on the GPU side.
- The originally planned golden-file workflow (per-stage CPU dumps compared by GPU
  tests under ~1e-3 relative tolerance) was never built — superseded by the
  end-to-end ladder above, which checks the quantity that matters (final score /
  rank / location) against known-correct answers. If per-stage tests are revived,
  compare NMF at the *activation* level after normalization with fixed seeds (NMF
  is only unique up to scaling/permutation), not raw W/H.
- The ladder is scripted: `tests/make_fixtures.sh` (rebuilds the synthetic
  fixtures with ffmpeg) + `tests/run_ladder.sh` (4 structural gates). No test
  framework, nothing vendored.

## Benchmark methodology and measured results

Measured (CLAUDE.md build log; box: 2× RTX A5000, CUDA 12.5). Current
headline numbers live in README "Performance" (5-cand scan 3.0 s; full-64
19.3 s; clip 2.3 s; benchmark ~13 min; rf_infer 3.1 M rows/s); kernel-level
characterization in plots/PERF-CHARACTERIZATION.md. Historical anchors:

- v2-era scan ~35 s for 5 candidates (~7 s per pair); ~120 s before the
  cuBLAS strided-batched PFNMF + banded-DTW redesign; 42 s at the start of
  the optimization campaign.
- CPU vs GPU at a v1-era matched config: **149 s vs 2.2 s = 68×**; the
  ported (current-algorithm) CPU reference runs the 30 s canary config in
  47.8 s vs ~1 s on GPU. A full-config CPU scan extrapolates to hours — by
  design, use `--max-seconds`/`--iters`.

Conventions for any further measurement: GPU timing with `cudaEvent_t` around each
stage (synchronized), CPU with `std::chrono::steady_clock`; median of 5 runs after 1
warmup; include H2D/D2H in end-to-end numbers (honest accounting) and report
kernel-only times separately. Profiling via Nsight Compute CLI (`ncu`) if present;
otherwise `--generate-line-info` + compute-sanitizer. Planned-but-not-done: per-stage
CPU/GPU table and the three-way NMF benchmark (naive CPU / cuBLAS GEMMs / fused
custom kernels).

## GPU design notes (summary)

Decision rationale in `gpu-library-vs-custom-kernels.md` (note its postscript:
GEMMs moved to cuBLAS on 2026-06-10). As built:

- **Libraries (deliberate):** cuFFT for the batched R2C STFT; cuBLAS
  strided-batched GEMMs (`cublasSgemmStridedBatched` behind row-major wrappers in
  `gemm.cu/.cuh`) for all NMF/PFNMF matrix multiplies — all 41 pitch-shift PFNMFs
  run as one batched problem, with V shared across the stacked problems.
- **Custom kernels (everything else; `kernels.cu` + `rf_infer.cu`):** STFT
  windowing and magnitude(+transpose); the **fused W·H+ratio kernel** (single
  shared-memory tile, division epilogue — WH never materialized); the
  column/row-sum reductions and update steps (PFNMF template freezing = a
  column mask in the W-update kernel); pitch-template integer-translation
  gather (all shifts in one launch; linear-interp fallback on the linear
  axis); column/max/z-normalization kernels (the z-norm folds Z_REG and the
  PFNMF source-attribution weight into the z-vectors); correlation-distance
  kernel writing the **diagonal-skewed layout**; the **persistent banded
  subsequence-DTW kernel** (`dtw_band_kernel`): one block per (shift, band)
  walks all anti-diagonals in shared memory, slope filter + min/mean/argmin
  reduced in-kernel (a predecessor-recording variant feeds the classifier's
  path backtracking); and the **FIL-style forest kernel**
  (`forest_predict_kernel`, packed 16-byte nodes). Eq. 2 boundary: band
  row 0 free, column 0 accumulates.
- **Host:** scoring/ranking + path backtracking on CPU (tiny data: 4 floats
  per band × shift from the DTW kernel).
- Remaining performance headroom is small and measured —
  plots/PERF-CHARACTERIZATION.md §6 (pinned-staging H2D, forest tree
  interleaving).
