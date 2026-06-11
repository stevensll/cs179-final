# Technical Design

Single source of truth for algorithm parameters, repo architecture, build, testing,
and benchmark methodology. Algorithm follows Gururani & Lerch 2017
(`Automatic-Sample-Detection-in-Polyphonic-Music.pdf` in this directory); the
block-by-block paper ↔ code correspondence, with every deviation justified, is in
`PAPER-MAPPING.md` — this file does not duplicate it. Library-vs-custom kernel
rationale: `gpu-library-vs-custom-kernels.md` (see its dated postscript). Test cases
and current status: `TESTING.md`. History and measured numbers: CLAUDE.md build log.

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
`PAPER-MAPPING.md`):

- **No random-forest classifier** — heuristic score normalization (pitch
  selectivity + min/median selection-bias correction, below) replaces it
  (retrieval task, no training data available).
- **No known sample snippet** — templates come from one NMF over the *full*
  candidate song (rank scaled accordingly), and the "which part of the candidate
  is the sample" search happens inside DTW as dense 4 s bands of the distance
  matrix, at zero extra NMF cost. (An earlier v1 design used 8 s max-RMS snippet
  hypotheses and a shuffle-null calibration; both were replaced — see CLAUDE.md
  log. The CPU demo still implements that v1 design; see Testing.)

## Parameters

`src/common/params.hpp` is the authoritative copy (each value carries its rationale
there); this table mirrors it. All values from the paper unless marked *deviation*.

| Parameter | Value | Notes |
|---|---|---|
| Analysis sample rate | 22 050 Hz | inputs decimated 2:1 from 44.1 kHz |
| STFT | block 4096, hop 1024, Hann | 2049 bins, ~21.5 frames/s |
| Log-frequency front end | `MEL_BINS`=367, 4 bins/semitone from 55 Hz | *deviation* (D6 in PAPER-MAPPING.md): triangular log-spaced pooling before NMF — ~5.6× less downstream compute AND pitch shifts become exact integer translations of templates; `-DSD_DEFS="SD_MEL_BINS=0"` restores the linear axis |
| `RANK_K` | **40** | *deviation*: paper uses K=10 for a known ~4.5 s sample; we model the full candidate song, so the rank is content-scaled (paper §3.1.1 rationale) |
| `RANK_L` | 20 | free mixture templates in PFNMF |
| `DEFAULT_ITERS` | 100 | NMF multiplicative-update iterations (60 is plenty in practice) |
| Pitch shifts | **41**: −5..+5 st in 0.25 st steps | *deviation* from the paper's 12-step set: resampling speedups give fractional shifts and a 0.5 st grid miss empirically destroys the DTW dip |
| `Z_REG` | 0.01 | correlation regularizer at the paper's H/max(H) scale |
| `WINDOW_SECONDS` | {4} | DTW band length ≈ the paper's average sample length (§4.1); a worst-across-{4,8} variant was tried and regressed |
| `WINDOW_HOP_SECONDS` | 1 | band hop (dense candidate-window search) |
| `SNIPPET_*`, `DEFAULT_WINDOWS` | legacy | **CPU demo v1 only** — removed once the CPU demo is ported (TODO.md) |

Working-set sizes: V ≈ 2049×5200 ≈ 42 MB for a 4-min song. The dominant allocation
is the distance-matrix stack D: 41 shifts × No × Ns floats ≈ 6 GB for the largest
current pair (5:33 query) — fits one A5000 (24 GB) after the dead null-surrogate
computation was removed (CLAUDE.md log, v3).

## Pipeline

See `PAPER-MAPPING.md` for the stage-by-stage diagram (preprocess → STFT →
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
   not yet validated (TODO.md).

## Repo layout

```
cs179-final/
├── README.md              # final submission readme (per spec)
├── PROPOSAL.md            # proposal + timeline (as proposed; see README for as-built)
├── CLAUDE.md              # operating notes / build log (measured numbers live here)
├── TODO.md                # cleanup + improvements ledger
├── Makefile               # thin wrapper: configure + build via CMake
├── CMakeLists.txt
├── docs/                  # paper PDF, this file, PAPER-MAPPING.md, STYLE.md,
│                          #   TESTING.md, PRESENTATION.md, kernel-scope doc
├── music/                 # song library + queries (WAVs)
├── datasets/sample100/    # Sample100 evaluation set: eval_pairs.csv + audio/
├── tools/                 # visualize_match.py, scrape_sample100.py,
│                          #   eval_sample100.py, DATASET.md
├── src/
│   ├── common/            # compiles WITHOUT CUDA: audio.{hpp,cpp} (WAV load,
│   │                      #   downmix, RMS, 2:1 decimator), params.hpp
│   ├── cpu/               # cpu_demo.cpp + cpu_pipeline.{hpp,cpp} (v1 algorithm)
│   └── gpu/               # gpu_detect.cu, gpu_pipeline.cu(h) (orchestration),
│                          #   kernels.cu(h) (all custom kernels), gemm.cu(h)
│                          #   (cuBLAS wrappers), device_buffer.cuh, error_check.cuh
└── build/                 # out-of-source build dir (not committed)
```

(There is currently no `tests/` or `golden/` directory; see Testing. Splitting
`kernels.cu` / `cpu_pipeline.cpp` into per-stage files is a TODO.md cleanup item.)

Executables:
- `gpu_detect [--max-seconds S] [--iters I] [--clip] <query.wav> <library_dir>` —
  the GPU detector (current algorithm); prints the ranked library with shift and
  matched locations.
- `cpu_demo [--max-seconds S] [--iters I] [--windows W] <query.wav> <library_dir>` —
  single-threaded CPU detector, **still the v1 algorithm** (snippet-window
  hypotheses + shuffle-null calibration). Kept as the historical baseline; port or
  re-scope per TODO.md.

## Build

- CMake ≥ 3.24 (box has 4.0), `LANGUAGES CXX CUDA`, `find_package(CUDAToolkit)`,
  `CMAKE_CUDA_ARCHITECTURES native` (sm_86 here), pkg-config for sndfile; links
  `CUDA::cufft` and `CUDA::cublas`.
- Compile flags: `-O3 -Wall -Wextra` (host), `-O3 --generate-line-info` (CUDA, for
  profiler/compute-sanitizer friendliness).
- Top-level `Makefile` wrapper: `make` = configure (if needed) + build into `build/`;
  `make clean` wipes `build/`. Graders never need to know CMake.
- There are no build options/flags; in particular, no separate cuBLAS-baseline NMF
  build exists (the three-way NMF benchmark is future work, TODO.md — the custom
  GEMM would have to be reimplemented from the v1 design, see CLAUDE.md log).

## Testing

Current practice (status in `TESTING.md`):

- **Verification ladder**, re-run after every algorithm/scoring change: self-match
  canary (song vs itself → rank 1 @ +0.00 st, aligned locations), two synthetic
  controls with exactly predictable answers (8 s insert → dip @ +0.00 st at the
  insert location; same insert resampled +6% → dip @ +1.00 st, theory +1.01), and
  the real pairs in `TESTING.md`.
- **CPU/GPU parity is currently N/A**: the CPU demo implements the v1 algorithm
  (it was held at 4-decimal score parity with the v1 GPU pipeline; that GPU code
  has since been redesigned). Parity testing resumes if/when the CPU demo is
  ported (TODO.md).
- The originally planned golden-file workflow (per-stage CPU dumps compared by GPU
  tests under ~1e-3 relative tolerance) was never built — superseded by the
  end-to-end ladder above, which checks the quantity that matters (final score /
  rank / location) against known-correct answers. If per-stage tests are revived,
  compare NMF at the *activation* level after normalization with fixed seeds (NMF
  is only unique up to scaling/permutation), not raw W/H.
- Tests are run by hand; no test framework, nothing vendored. A scripted ladder
  runner is a TODO.md item.

## Benchmark methodology and measured results

Measured (CLAUDE.md build log, 2026-06-10; box: 2× RTX A5000, CUDA 12.5):

- Full GPU library scan (full songs, 60 NMF iterations, 41 shifts, dense 4 s band
  search): **~35 s for 5 candidates (~7 s per pair)**; was ~120 s before the
  cuBLAS strided-batched PFNMF + banded-DTW redesign.
- CPU vs GPU at a v1-era matched config (45 s query, 30 iters, 1 window, 41
  shifts, 1 candidate): **149 s vs 2.2 s = 68×**. That comparison is v1 algorithm
  on both sides; no current-algorithm CPU number exists (CPU demo not ported). A
  full-config v1 CPU scan extrapolates to ~8 h — do not run it; use
  `--max-seconds`/`--iters`.

Conventions for any further measurement: GPU timing with `cudaEvent_t` around each
stage (synchronized), CPU with `std::chrono::steady_clock`; median of 5 runs after 1
warmup; include H2D/D2H in end-to-end numbers (honest accounting) and report
kernel-only times separately. Profiling via Nsight Compute CLI (`ncu`) if present;
otherwise `--generate-line-info` + compute-sanitizer. Planned-but-not-done: per-stage
CPU/GPU table and the three-way NMF benchmark (naive CPU / cuBLAS GEMMs / fused
custom kernels) — TODO.md.

## GPU design notes (summary)

Decision rationale in `gpu-library-vs-custom-kernels.md` (note its postscript:
GEMMs moved to cuBLAS on 2026-06-10). As built:

- **Libraries (deliberate):** cuFFT for the batched R2C STFT; cuBLAS
  strided-batched GEMMs (`cublasSgemmStridedBatched` behind row-major wrappers in
  `gemm.cu/.cuh`) for all NMF/PFNMF matrix multiplies — all 41 pitch-shift PFNMFs
  run as one batched problem, with V shared across the stacked problems.
- **Custom kernels (everything else, all in `kernels.cu`):** STFT windowing and
  magnitude(+transpose); the NMF elementwise ratio/update steps and the
  column/row-sum reductions (PFNMF template freezing = a column mask in the
  W-update kernel); pitch-template linear-interp frequency gather (all shifts in
  one launch); column/max/z-normalization kernels (the z-norm folds Z_REG and the
  PFNMF source-attribution weight into the z-vectors); correlation-distance kernel
  (grid z over shifts); and the flagship **banded subsequence-DTW wavefront
  kernel** (`dtw_band_diag_kernel`): one launch per anti-diagonal, one thread per
  cell, `blockIdx.y` enumerating (shift-matrix, band-slot) pairs so all shifts and
  a chunk of bands advance together, with a parallel L matrix tracking path length
  for the cost normalization. Eq. 2 boundary: band row 0 free, column 0
  accumulates. No backtracking yet (needed for the paper's path features —
  TODO.md).
- **Host:** scoring/ranking on CPU (tiny data: one last-row min/mean per band ×
  shift, fetched with strided `cudaMemcpy2D`).
- Known performance headroom (TODO.md): tiled multi-diagonal DTW (currently ~200k
  small launches per candidate), multi-GPU candidate split across the two A5000s,
  cuFFT plan reuse.
