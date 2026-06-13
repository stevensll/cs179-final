/* CLAUDE.md — operating notes for this repo */

# cs179-final: GPU sample detection

## Mission

Earn maximum points on the CS 179 final project: given a query song, find the song it
samples from a library, implementing Gururani & Lerch 2017 (NMF + subsequence DTW) with
custom CUDA kernels. Grading deliverables are in `README.md` (proposal, CPU demo, final
submission with performance analysis). The plan of record is `PROPOSAL.md`; algorithm
parameters and architecture live in `docs/TECHNICAL.md`; code style in `docs/STYLE.md`;
the library-vs-custom kernel rationale in `docs/gpu-library-vs-custom-kernels.md`;
test cases in the README (Evaluation + Errata).

## Execution environment

- We are **on the GPU box (titan) directly** — no ssh hop. Repo at
  `/home/slei3/cs179-final`.
- Ubuntu 22.04, 2× RTX A5000 (Ampere, CC 8.6, 24 GB each), CUDA 12.5
  (`/usr/local/cuda/bin/nvcc`), CMake 4.0, g++ 11.4 (C++17).
- **No sudo. Never attempt package installs** (`apt`, `pip --user` for build deps,
  etc.). Use only what is on the box. If something seems missing, check for it under
  `~/local-libs` first, then ask Steven.
- Available and approved: cuFFT (the one deliberate library dependency), cuBLAS (only
  for the benchmark-baseline NMF variant behind a flag), libsndfile (`sndfile.h` is
  present) for WAV decode.
- NOT usable despite runtime .so's being present: FFTW, libsamplerate (no headers).
  The CPU demo hand-rolls its FFT and the 2:1 decimator.
- The WAVs in `music/` are 16-bit stereo 44.1 kHz with a broken RIFF size field
  (0xFFFFFFFF). libsndfile copes; any hand-rolled parsing must tolerate it.
- Python 3.10 + numpy exist — fine for ad-hoc verification scripts, never as a build
  or test dependency.

## Run discipline (shared machine)

- Every executable calls `TA_Utilities::enforce_time_limit(<seconds>)` at startup
  (pattern copied from `~/cs179/lab3-2025-main/src/ta_utilities.*`) so a hung kernel
  can't sit on a GPU. Default limit 120 s; benchmarks may raise it deliberately.
- No unattended long runs. Batch experiments and check results between rounds.
- Compile for Ampere: CMake sets `CMAKE_CUDA_ARCHITECTURES native` (= sm_86 here).
  If a kernel "runs fast and does nothing", check arch flags and
  `checkCuda(cudaGetLastError())` after launches before anything else.
- `make clean` (or wipe `build/`) after ANY header change — stale-object link bugs
  against changed class layouts burned us in lab4.
- Out-of-source builds only (`build/`). Never commit `build/` or golden binaries that
  can be regenerated.

## Testing discipline

- The gate for ANY pipeline change is `tests/run_ladder.sh` (4 structural
  gates: canary + two synthetics with exact ground truth + Hung Up rank-1)
  AND no regression on the Sample100 benchmark (full 70 queries for frontier
  decisions — subsets overestimate ~2×, measured).
- CPU/GPU parity: the CPU reference mirrors the production pipeline with the
  same seeds; scores must match to 4 printed decimals on the ladder gates at
  matched configs. Re-verify after any scoring change.
- The originally planned per-stage golden-file workflow was superseded by the
  end-to-end ladder (it checks the quantity that matters). If revived: compare
  NMF at the activation level after normalization with fixed seeds, not raw
  W/H (NMF is unique only up to scaling/permutation).
- No test frameworks, nothing vendored.

## Style conventions

See `docs/STYLE.md` for the full rules. The non-negotiables: `snake_case` everywhere
with `_kernel` suffix for kernels; `/* ... */` comments, why-only, except every kernel
carries a strategy block comment (decomposition, shared memory, why it beats naive);
magic numbers as named `constexpr`; `checkCuda()` on every CUDA call and after every
launch; device memory through `DeviceBuffer<T>`, no Thrust/CUB.

## Confirmed decisions

- Backend: rank library by min subsequence-DTW cost. The classifier stage was
  originally out of scope ("future-work note only"); Steven extended scope
  2026-06-11 — the paper's 13-feature random forest is now trained
  (tools/train_rf.py) with a CUDA inference kernel (rf_infer).
- Kernel scope: **revised by Steven 2026-06-10 (v3)** — matrix multiplies via
  cuBLAS (NVIDIA built-ins allowed where they exist; enough custom kernels
  remain), cuFFT for STFT; all other stages are custom kernels.
- CPU demo in C++, sharing `src/common/` with the GPU build.
- Build: CMake + thin Makefile wrapper so graders can type `make`.
- Timeline: 3 weeks per `PROPOSAL.md`.

## File map

- `src/common/` — audio I/O, preprocessing, shared types/params (CPU+GPU)
- `src/cpu/` — CPU reference, stage-for-stage mirror of the GPU pipeline
- `src/gpu/` — kernels.cu(h), gpu_pipeline.cu(h), gemm, gpu_detect, rf_infer
- `tests/` — make_fixtures.sh + run_ladder.sh (4 structural gates)
- `music/` — song library + queries; expected matches documented in the README
- `tools/` — eval runner, RF trainer/exporter, pool/profile analysis, plots, scraper
- `build/` — out-of-source build dir (not committed)

## Evaluation detail (ex-RESULTS.md — reader-facing summary lives in README "Results")

Gritty reference for the Sample100 numbers quoted in the README; kept here so
the AI context retains full knowledge without burdening the reader.

- **Corpus**: 17.3M feature rows (70 queries × 64 candidates ×
  top-32-hypothesis path groups), 51,652 positives (0.30%); label = annotated
  (query, candidate) pair AND band_start within 4 s of an annotated
  t_original. Generated by `gpu_detect --features`, ~30 min on 2 GPUs.
- **Leakage protections**: 5-fold GroupKFold over the 52 connected components
  of the query↔original bipartite graph (queries sharing any original
  co-travel); all reported predictions out-of-fold; decision threshold chosen
  on each fold's TRAIN side (0.600, unanimous); features are path geometry
  only (no heuristic scores, no metadata). Training negatives subsampled 20:1
  (test folds predicted in FULL, so reranking is exact).
- **Row-level quality**: out-of-fold AUC 0.629; per-fold 0.55–0.71 (the
  spread is the data-starvation evidence behind
  docs/ACCURACY-OPTIMIZATIONS.md idea #1).
- **Reproducibility**: retraining reproduces MRR to ±0.001 (only stochastic
  step is negative subsampling); quoted figures are the first full run's.
  Per-(query,candidate) OOF scores persisted at plots/rf/rerank_scores.csv —
  every pool-size/threshold analysis is recomputable without retraining
  (tools/pool_analysis.py).
- **Macro caveat mechanics**: recall pegs at 100% because every Sample100
  query HAS a sample (no-detect = FN, and the max-over-64 order statistic
  clears any liberal threshold), so macro precision = top-1 accuracy. The
  10-pool simulation: top-1 = exact hypergeometric expectation
  C(N−b,9)/C(N,9); macro = Monte Carlo, 2,000 pools/query, rng(42).
- **Not implemented**: location-level ("micro") P/R/F per the paper
  (79.4/34.6/48.2 in their table) — band-level hypotheses exist, the
  per-occurrence evaluation harness does not.

## Notes / build log

Append-only log of what was actually done, with enough detail that someone re-running
this on a fresh account could reproduce. Most recent at the bottom. Log: dates,
commands that mattered, perf numbers with the exact scene/input, dead ends and why.

### 2026-06-10 — planning

Environment surveyed; paper read; decisions above locked with Steven. Planning docs
written (`PROPOSAL.md`, `docs/TECHNICAL.md`, `docs/STYLE.md`,
`docs/gpu-library-vs-custom-kernels.md`). No source code yet.

### 2026-06-10 — MVP implemented end-to-end (CPU + GPU), verified, partially passing

Steven locked NMF = fully custom CUDA kernels (no cuBLAS). Built the whole stack in
one pass: `src/common` (libsndfile load, downmix, RMS, hand-rolled 63-tap 2:1
decimator), `src/cpu` (radix-2 FFT, STFT, KL-NMF/PFNMF, pitch templates, correlation
distance, subsequence DTW), `src/gpu` (12 custom kernels + batched cuFFT STFT).
Clean build first try; `make` wrapper works.

**Verification (the part that matters):**
- Self-match canary: song vs itself → score 0.12 at shift +0.00. ✓
- Synthetic controls (ffmpeg-built): MOU 8s intro mixed into Shape @30s → sharp dip
  exactly at +0.00 st; same snippet resampled +6% → dip exactly at +1.00 st
  (theory: 12·log2(1.06)=+1.01). Mathematically correct response to both pitch
  shift and time stretch. ✓
- CPU ↔ GPU agreement: identical scores to 4 printed decimals on matched configs
  (same seeded init, same algorithm). ✓ (re-verified after every scoring change)

**Algorithm deviations from the paper, found necessary during debugging:**
1. Quarter-semitone pitch grid (−5..+5, 41 steps) replaces the paper's 12-step set:
   a 0.5-st grid miss empirically destroys the DTW dip (synth probe: ratio 0.63 vs
   0.13), and real resampling speedups give fractional shifts (TTS/MOU sits at −0.5).
2. Snippet hypotheses: 6 windows of 8 s (max-RMS + evenly spaced), best score wins —
   20 s single-window failed because samples are chopped/looped short.
3. Scoring = (min/mean of DTW cost function), then divided by the same statistic on
   a column-shuffled null (fixed seed) to cancel per-candidate cost bias.
4. Template columns unit-L2-normalized after snippet NMF (norms folded into Ho) and
   after each pitch shift — extreme shifts otherwise shrink norms and bias the
   distance (false dips were piling up at grid edges).

**Test status (`docs/TESTING.md`):**
- Test 1 Lucid Dreams → Shape of My Heart: **PASS** (0.482 vs 0.533 runner-up).
- Test 2 Touch The Sky → Move On Up: **FAIL** — detector finds the true pair's
  consistent −0.5 st alignment, but Shape of My Heart wins via a musical-similarity
  dip that survives null calibration. Chopped/rearranged sample = known-hard class
  (paper recall is 34–50% overall). Improvement paths logged in TODO.md.

**Performance (matched config: 45 s query, 30 iters, 1 window, 41 shifts, 1 cand):**
CPU 149 s vs GPU 2.2 s = **68×**. Full GPU library scan (full songs, 60 iters,
6 windows, 3 candidates): ~120 s. Full-config CPU extrapolates to ~8 h (~230×) —
do not run it; use `--max-seconds/--iters/--windows`.

Debug tooling that exists: `--snippet-offset/--snippet-seconds` flags, `SD_DEBUG=1`
env prints per-shift ratio/null landscapes. Synthetic fixtures in `build/`
(`synth_plain.wav`, `synth_fast.wav`, probe dirs) — regeneration commands in git
history / TODO.md wants them scripted.

Cleanup ledger: `TODO.md` (created this session, per Steven).

### 2026-06-10 — v2: cuBLAS + batched redesign; test-2 root cause found (data, not code)

Steven: move GEMMs to cuBLAS (enough custom kernels remain), GPU only, fix the
algorithm principally (no per-example tuning), add a visual aid.

**Cleanup/redesign (one pass, they were coupled):**
- GEMMs now `cublasSgemmStridedBatched` via row-major wrappers (`src/gpu/gemm.cuh`);
  custom gemm kernel deleted. All 41 pitch-shift PFNMFs run as ONE batched problem.
- Snippet-window heuristic deleted. Templates come from the FULL candidate song
  (one NMF); the "which part of the candidate is the sample" search moved into DTW:
  banded subsequence DTW over row-bands of one distance matrix per shift — dense
  window search at zero extra NMF cost. Band length 4 s = the paper's avg sample
  length (4.2). Full library scan: 120 s → ~35 s.
- NMF workspaces hoisted (`NmfWorkspace`); DTW last rows fetched with one
  `cudaMemcpy2D` per chunk; detector reports match location (cand window + query
  position) — consumed by the new viz tool `tools/visualize_match.py` (numpy+PIL,
  ffmpeg decode; no matplotlib on box).

**Principled scoring fixes (each justified by paper or statistics, not examples):**
- H/max(H) (paper 3.2.1) reinstated as the scale for a REGULARIZED correlation:
  znorm denom = sigma + Z_REG. Kills silent/fade-band degeneracy (false matches had
  clustered at candidate fade-outs).
- Slope sanity from the L matrix: end points with warp L/T outside [0.7, 1.5]
  rejected (paper 3.3.2: real alignments have near-constant slope; bound matches
  the +-33% the +-5 st grid implies). Note: cost/L normalization otherwise REWARDS
  meandering paths.
- Selection-bias correction: best hypothesis judged against the candidate's own
  (band x shift) score distribution (min/median) — a min over ~6800 hypotheses
  otherwise favors longer candidates.
- Pair-level "null" surrogates ABANDONED after both variants backfired: frame
  shuffle breaks activation smoothness (null too expensive -> every smooth match
  scores < 1); time reversal breaks looped samples (quasi-periodic loops survive
  reversal, so the null dips exactly for true pairs). Reversal null still computed
  for SD_DEBUG, unused in scoring.

**Test results:** canary 0.33@+0.00; synth plain 0.28@+0.00; synth +6% 0.24@+1.00
(all locations exact). Test 1 PASSES with best margin yet: 0.360 vs 0.443, match
at Shape@144s -> Lucid@118s, +0.25 st.

**Test 2 root-cause attempt (LATER RETRACTED — see next entry):** three numpy
probes (broadband spectral cosine, chroma rotation, 1.5 s chop-scale) found no
literal-copy peak for MOU-in-TTS (+-6 st, tempo 0.5-2.0x), and I concluded the
files were wrong. Steven subsequently VERIFIED the files are correct. The
probes were fingerprint-class full-spectrum matchers — exactly the class the
paper says fails on samples mixed at low level under other sources (§2.1); my
"validation" used a full-volume synthetic insert, which never tested the buried
case. Lesson: a negative from a fingerprint probe does not establish absence
of a buried sample.

### 2026-06-10 — v3: files verified, Hung Up pair added, K scaled, selectivity scoring

Steven verified TTS/MOU files are correct (v2 entry's conclusion retracted in
place) and added Hung Up.wav / Gimme Gimme.wav (Madonna samples ABBA).

Changes, each A/B'd against the full ladder (canary + 2 synths + 3 real pairs):
- Removed the dead reversal-null computation entirely (it was half the D memory
  and DTW work and unused in scoring) — also fixes the OOM that long queries hit
  (Hung Up is 5:33; D for the largest pair now ~6 GB, was ~12).
- RANK_K 10 → 40: the "sample NMF" block models a full song, so the rank scales
  (paper §3.1.1 rationale). Controls sharpened ~3x (canary 0.33 → 0.10).
  A/B at K=10 confirmed: weaker everywhere; K=10's test-1 "pass" was a
  wrong-location fluke.
- Pitch-selectivity scoring: each window's dip normalized by that window's
  median dip across all 41 shifts (real matches dip at ONE shift, paper fig. 2;
  junk dips at all). Killed the grid-edge degeneracy class for good.
- PFNMF source-attribution weight in znorm (e = |H_fix|/(|H_fix|+|H_free|)),
  max-normalization extended to all rows to preserve the ratio. Neutral on
  ladder, kept for paper fidelity (activations indicate *presence*).
- Tried and REVERTED: worst-across-{4,8}s window lengths (regressed: punishes
  looped true samples at loop boundaries more than confounds).

**Ladder (v3 final):** canary 0.108@+0.00 · synth 0.247@+0.00 · synth+6%
0.218@+1.00 · Hung Up → Gimme Gimme **PASS decisive** (0.300 vs 0.607, @+0.00)
· Lucid → Shape NEAR MISS (true match found @+0.00 cand@8s at 0.591, edged by
Gimme Gimme 0.542 melodic-resemblance confound) · TTS → MOU FAIL (true −0.50
alignment found at 0.727, two confounds above). The remaining confound class
(sparse clean-riff candidates as universal melodic matchers) is precisely what
the paper's classifier stage separates — docs/PAPER-MAPPING.md "What the
missing classifier costs".

Docs: added docs/PAPER-MAPPING.md (1:1 paper-block ↔ code diagram, deviations
table with reasons). Run time ~7 s per candidate pair at --iters 60.

### 2026-06-10 — clip mode + dataset scraping kit (Sample100)

- `--clip` flag wired through gpu_detect/gpu_score_candidate: for human-trimmed
  queries that ARE the suspected sample, score by absolute normalized cost
  (a wall-to-wall clip has no non-sample baseline, so the dip statistic
  self-destructs — measured before the change). NOT yet tested; Steven will
  test with his own clip.
- Paper-dataset hunt: footnote 3's repo (SiddGururani/sample_detection, cloned
  at ~/sample_detection) does NOT contain the promised annotations/URLs — it's
  MATLAB code reading from the author's local D: drive; wiki empty, no
  releases, nothing in git history, nothing on Zenodo (paper PDF only) or the
  GTCMT dataset page. Substitute: **Sample100** (jvbalen/sample_100, cloned at
  ~/sample_100) — the public benchmark: 105 relations, 76 queries, 68
  originals, with timestamps/types/interpolation flags. Metadata only.
- Scraper kit for a download agent: `tools/scrape_sample100.py` (yt-dlp search
  -> 44.1k WAV per track, idempotent, logs actual YT title/duration for
  human verification — wrong-version downloads are the known data trap) +
  `tools/DATASET.md` (setup, run, verification protocol, eval protocol).
  `datasets/sample100/eval_pairs.csv` generated: 103 pairs (3 interpolations
  excluded — recording reuse only). Needs `pip install --user yt-dlp` (not on
  box; left to the agent per no-install discipline).

### 2026-06-11 — first Sample100 numbers (5-query subset)

Scrape agent delivered 143/144 tracks (T157 age-blocked; T108/T109/T115 wrong
audio, excluded via datasets/sample100/excluded_tracks.txt). First eval run
(`tools/eval_sample100.py --limit 5 --iters 60`, 64-candidate original library):

  ranks of true original: 2, 3, 6, 17, 57 (of 64)
  hit@1 0/5 · hit@3 2/5 · MRR 0.215 (random baseline MRR ≈ 0.073, mean rank 32)

Consistent with the music/ anecdotes at scale: the true original is pulled far
above chance (3 of 5 in the top 6) but melodic-resemblance confounds keep
stealing #1 — the missing-classifier gap, now quantified. T006 (rank 57) is an
outlier worth investigating (listen to T006/T007 audio quality first).
GPU utilization during the run: 86% avg SM on GPU1, dips = per-candidate host
setup; GPU0 idle (tools/plot_gpu_load.py, build/gpu_load.png).

### 2026-06-11 — performance push: 42 s -> ~8.8 s scan (4.8x), nsys-gated loop

Steven supplied docs/OPTIMIZATIONS_{CHATGPT,GEMINI}.md to evaluate; every change
was gated on the verification ladder AND the eval-5 ranks (T001/4/6/8/9).
Profiles + plots per iteration in plots/ (00_baseline, 01_tf32_dtw, 02_fused).

Adopted, in order (5-candidate music/ scan, --iters 60):
- TF32 tensor-core GEMMs via cublasGemmStridedBatchedEx + explicit
  CUBLAS_COMPUTE_32F_FAST_TF32 (math-mode alone was ignored for the NN shape);
  per-device handles. 42 -> 37 s.
- Persistent-band DTW kernel: one block per (matrix, band) sweeps all
  anti-diagonals in shared memory over a DIAGONAL-SKEWED distance layout
  (anti-diagonals contiguous -> coalesced); slope filter + min/mean/argmin
  reduce in-kernel to 4 floats/band. Replaced ~325k launches/candidate (was
  33.6% of wall) + 80 GB C/L traffic + 600 MB D2H. Bit-identical scores.
  37 -> 21 s. Skew layout OOM'd 7-min Sample100 candidates (No*(No+Ns) is
  ~quadratic) -> distance+DTW chunked over the shift axis (4 GB budget).
- Multi-GPU: worker thread per device, shared candidate queue (gpu_detect
  --gpus), eval --gpus N pins query subprocesses. 21 -> 12.8 s.
- Interleaved (simultaneous Lee-Seung) NMF update: 3 GEMMs + 1 ratio per iter
  (was 4 + 2). Gate IMPROVED: eval-5 ranks (2,3,6,17,57) -> (12,2,54,1,6),
  first hit@1, MRR 0.215 -> 0.354. 12.8 -> 9.3 s.
- Fused custom W*H+ratio kernel (k=R<=64 single shared tile, division
  epilogue): WH never materialized; ~8 TFLOPS vs cuBLAS's sgemm on the same
  skinny shape, plus two full WH passes eliminated. Bit-identical scores.
  Single-GPU wall 20.2 -> 14.1 s (2-GPU scan masked by 3/2 split).
- Host-init cache (seeded_cached): PFNMF's 23M-float mt19937 init regenerated
  per candidate was ~25% of the idle gap; cached per worker, byte-identical.
  9.3 -> 8.8 s.

Tried and REVERTED (gate failures, machinery kept disabled in params.hpp):
- Two-stage shift screening 15/8 AND 25/16: MRR 0.354 -> 0.186 / 0.221, a
  rank-1 hit fell to rank 5. Partially-converged interim scores mixed into
  the selectivity medians corrupt cross-candidate comparability.
- Plain --iters 30: ladder separation visibly eroded (canary 0.11 -> 0.24).

Rejected on measurement (advice-file items): warp-shuffle reductions, znorm
single-pass, Hann/pitch tables, coalescing fixes for col_sum/magnitude, cuFFT
plan cache (STFT = 0.0% of wall), distance-as-cuBLAS.

Final profile (plots/02_fused): top kernel = our fused custom kernel (54% of
kernel time, compute-dense), then TF32 tensor-core GEMMs — the plan's stop
condition. Remaining idle ~25% is audio decode + per-candidate setup
(prefetch pipelining left as TODO).

Net: 5-candidate scan 42 -> 8.8 s (4.8x); single pair 4.4 -> ~1.6 s effective;
64-candidate eval query ~7 min -> ~70 s (pre-pruning measurement; the final
gate run will refresh); full 73-query eval ~8.5 h -> ~40 min estimate.

### 2026-06-11 — candidate-template cache + decode prefetch

- Candidate-side product (Wo, Zo) split out as gpu_candidate_templates() and
  disk-cached per library song (<library>/.tcache/*.tc; invalidated by audio
  size+mtime + param/iters fingerprint; tmp+rename for concurrent workers;
  --no-cache to bypass). Byte-stable -> scores identical by construction
  (verified: ladder + eval-5 gate MRR 0.354 unchanged).
- One-deep std::async decode prefetch per worker (skips decode when the cache
  probe passes); longest-first candidate ordering; removed redundant
  cudaDeviceSynchronize calls (the latter two measured neutral at small scale,
  kept as hygiene).
- Measured: 15 s clip vs 64-song library 21.7 -> 12.4 s cold -> **7.9 s warm**;
  full-query 5-cand scan 8.8 -> 7.5 s warm. Eval queries vs the Sample100
  library now ~35-50 s each once .tcache is warm.
- eval_sample100.py --gpus N now round-robins N pinned worker processes onto
  the physical GPUs (N=4 -> 2 processes/GPU to fill host-gap idle).

### 2026-06-11 — cross-candidate PFNMF batching + fast-math (goal: speedup vs baseline)

Baseline (warm tcache, --iters 60, 2 GPUs): 5-cand scan 7.5 s · full-song
Hung Up vs 64-cand library 68.8 s · 15 s clip vs 64 4.9... 7.9 s.

Changes, gated (ladder bit-identical; eval-5 MRR 0.354 unchanged throughout):
- gpu_score_candidates(): ONE stacked PFNMF over (candidates x shifts) —
  every problem is query-sized so candidates batch without padding; scoring
  (candidate-length-dependent distance/DTW) stays per-candidate. Group size =
  min(memory budget / V-slab, fair share per device). Two scheduling bugs
  found by measurement: (1) pre-claiming the next group for prefetch starved
  the other GPU on small libraries (a 2-group scan ran entirely on one
  device); (2) the memory-derived group of 4 made a 5-candidate scan split
  4-vs-1 — fixed by the fairness cap.
- Persistent ScoreContext per worker (grow-only device scratch): re-allocating
  the multi-GB workspace per group serialized the two workers on the driver's
  process-wide cudaMalloc/cudaFree lock (and every cudaFree device-syncs).
- --use_fast_math (Steven's suggestion): ladder scores identical to 4
  decimals, ~5% across workloads (the fused kernel's 600M divisions/candidate
  take the approximate-reciprocal path; denormal flush is benign under the
  eps guards).
- Tried and REVERTED: 128x64 fused-kernel tile (Boehm-style traffic cut) —
  measured neutral; each problem's 1.7 MB H slab already lives in L2, so the
  predicted DRAM saving never existed. 64x64 restored.

**Final figures (same protocol as baseline):**
  5-cand scan            7.5 -> 6.9 s   (1.09x)
  full-song vs 64-lib   68.8 -> 60.2 s  (1.14x)
  15 s clip vs 64-lib    7.9 -> 4.9 s   (1.61x)
Batching pays where individual problems are small (clips), as predicted.
Final profile plots/04_final: 93.6% GPU-busy single-GPU (idle 6.4%, was 28%
pre-batching); launches per candidate ~130 (was ~325k at project baseline).
Cumulative since the original implementation: 5-cand scan 42 -> 6.9 s (6.1x);
clip search 21.7 -> 4.9 s (4.4x) on top of the tcache wins.

### 2026-06-11 — log-frequency front end + config-matrix Pareto sweep

**Mel/log-frequency front end (docs/PAPER-MAPPING.md D6):** triangular
log-spaced filterbank (4 bins/semitone from 55 Hz -> 367 analysis bins) pooled
in as one GEMM after the STFT. Two effects: ~5.6x less compute everywhere
downstream of the spectrogram, and pitch shifts become EXACT integer template
translations (the old linear-axis rescale interpolation — the thing that made
fractional-shift misses fatal — is gone when mel is on). Gates: ladder
structurally clean (exact shifts/locations, dips shallower as expected from
pooling), Hung Up pair #1 decisive, eval-5 MRR 0.354 -> 0.363.
Profile plots/05_mel: PFNMF collapsed; distance/znorm (23.5%) + DTW (14.3%)
now co-dominant — both time-axis costs, which motivated the hop knobs below.

**Config matrix:** params.hpp knobs are #ifndef-wrapped (SD_MEL_BINS,
SD_BINS_PER_ST, SD_RANK_K/L, SD_HOP, SD_SHIFT_STEP4, SD_WINDOW_HOP_SECONDS);
cmake -DSD_DEFS="..." stamps variants; tcache filenames carry the fingerprint
so variants' caches coexist. tools/sweep_configs.py builds each variant,
measures warm scan time + Sample100 subset MRR, merges results across runs,
and plots the Pareto frontier (plots/sweep/).

**Sweep results (9-query subset, +-0.05 MRR noise; plots/sweep/pareto.png):**
  config     scan   eval s/q  MRR    note
  linear      6.9    44.1     0.375  pre-mel config — strictly DOMINATED
  mel4        3.2    14.4     0.317  (default at sweep time)
  mel2        2.8    11.6     0.348  2 bins/st holds up
  mel4_k24    2.9    12.5     0.285  K=24 below the rank floor
  mel4_k32    3.0    13.4     0.384  accuracy champion, faster than K=40
  mel2_k24    2.4     9.9     0.309
  hop2048     2.0     6.9     0.357  2.1x measured (2.2x predicted)
  shift05     2.3     8.4     0.318  0.5-st grid safe under exact translations
  bandhop2    3.0    13.2     0.308  ~nothing, as predicted — dead
  hop_shift   1.6     3.6     0.329  4x vs mel4 default, accuracy held
Frontier: hop_shift -> hop2048 -> mel4_k32. Full-73-query confirmation run of
mel4_k32 in flight before any default change.

### 2026-06-11 — default locked (K=32), feature extraction, repo cleanup, RESULTS.md

- Full-70-query confirmations settled the default: mel4_k32 MRR 0.214 /
  hit@1 11.4% vs mel4(K=40) 0.203/8.6% vs hop2048 0.179/7.1%. SD_RANK_K
  default now 32. NOTE: the 9-query sweep subset overestimated MRR ~2x
  (0.384 -> 0.214) — subsets prune dominated configs and measure speed;
  frontier accuracy must come from the full benchmark.
- Final production numbers (K=32): scan 3.0 s · full-64 19.3 s · clip 2.3 s.
- RF critical path built: dtw_band_preds_kernel (predecessor recording for
  selected top-32 hypotheses), host backtracking, the paper's 13 path/cost
  features per unique path start, gpu_detect --features CSV mode. Smoke
  test: 4,532 rows for one query x 5 candidates in 4.5 s; junk hypotheses
  show the expected meandering-path signatures. Trainer script still TODO.
- Cleanup: fixtures + ladder scripted (tests/make_fixtures.sh,
  tests/run_ladder.sh — 4 structural gates, all green); build/ artifacts
  purged; .gitignore (no audio, no binaries); Steven made the initial
  commits (62dd658, 37f5606, 9a87227 — repo finally has history);
  RESULTS.md created (paper-vs-ours metric framing with protocol-differences
  caveats and after-RF placeholders).

### 2026-06-11 — RF stage trained: hit@1 11.4% -> 50.0%, macro F 66.7% (paper: 62.5)

Feature corpus: 70 queries x 64 candidates via gpu_detect --features, both
GPUs, ~30 min -> 17.3M rows, 51,652 positives (0.30%; label = annotated pair
AND band_start within 4 s of an annotated t_original). Corpus sanity-checked
structurally (row counts, value ranges, 0 violations) and by signal (Hung
Up->Gimme truth region: slope 0.98 / dev 0.23 / 12 endpoints vs degenerate
false-candidate geometry — the paper's separability claim, visible raw).

Training (tools/train_rf.py): paper 3.4 config (200 trees, sqrt features),
leakage-safe by construction — GroupKFold over the 52 connected components
of the query<->original graph, out-of-fold predictions only, train-side
threshold, geometry-only features. Training negatives subsampled 20:1
(test folds always predicted in FULL, so reranking is exact). Dry run on a
12-query partial corpus validated mechanics before the full run (~40 min,
the corpus loads single-threaded for ~15 of those).

Results (out-of-fold, 64-candidate pools): row AUC 0.629; rerank by max
prob: MRR 0.214 -> 0.557, hit@1 11.4% -> 50.0%, hit@3 21.4% -> 57.1%.
Macro at the train-chosen threshold 0.600: P 50.0 / R 100.0 / F 66.7 vs
paper 83.3 / 50.0 / 62.5 on 10-candidate pools (caveats in RESULTS.md: no
sample-free queries in Sample100 -> R pegged at 100, P = top-1 accuracy).
Feature importances: path geometry dominates (avg_slope .130, avg_cost .107,
min_cost .100) — the melodic-confound failure mode is what the RF separates.
Model: plots/rf/rf_model.joblib (gitignored, regenerable in ~40 min).

### 2026-06-11 — CPU parity port + CUDA forest inference (FIL-style)

CPU reference (src/cpu/) rewritten as a stage-for-stage mirror of the
production GPU pipeline (was still the v1 snippet-window algorithm): log
filterbank, simultaneous NMF, full-song templates, integer-translation pitch
shifts, max-norm + Z_REG znorm with source attribution, banded DTW + slope
filter, selectivity scoring; same seeds (42/43, 100+p/200+p). MatchInfo
moved to src/common/match_info.hpp (shared); legacy SNIPPET_* params and
best_snippet_offset deleted. Parity vs gpu_detect at matched config
(--max-seconds 30-45 --iters 30): canary 0.2535@+0.00, synth_plain
0.4864@+0.00, synth_fast 0.4334@+1.00 — scores IDENTICAL to 4 decimals,
same locations, despite TF32+fast-math on the GPU side. CPU canary run
47.8 s vs GPU ~1 s at that reduced config.

CUDA forest inference (src/gpu/rf_infer.cu + tools/export_forest.py):
sklearn forest exported to a flat binary (SDRF), kernel = one thread per
row walking all 200 trees, FIL-style 16-byte PackedNode = one 128-bit load
per node visit (3.0x over the 4-array SoA layout on the real 24.9M-node
forest — traversal is DRAM-bound at depth ~60). Numerical parity bug found
and fixed via the harness: sklearn compares float32-cast features against
FLOAT64 thresholds; naive float32 threshold export rounds UP past feature
values and flips decisions (max prob diff 0.02 = 4 flipped trees). Fix:
export floor32(threshold) — largest float32 <= the float64 value — making
the float32 compare decision-exact. Result on 500k real corpus rows:
max |diff| vs predict_proba 5.96e-08, 0 rows above 1e-4 (PARITY OK).
Speed: 3.1M rows/s on one A5000 vs sklearn n_jobs=-1 0.16M rows/s (19x);
full corpus in ~5.6 s. Usage: tools/export_forest.py then
./build/rf_infer plots/rf/forest.bin <rows.f32> <ref_probs.f32> 13.

### 2026-06-11 — final wrap: 10-pool protocol, progression plot, FIL profiling, tidy

- tools/train_rf.py now persists per-(query,candidate) out-of-fold scores
  (plots/rf/rerank_scores.csv) — pool-size/threshold analyses no longer need
  retraining. Retrain reproduced the numbers (MRR 0.558 vs 0.557, the only
  stochastic step is negative subsampling); quoted figures stay the first
  run's.
- tools/pool_analysis.py: the paper's EXACT protocol (1 true + 9 random)
  simulated from the OOF scores — 10-pool top-1 60.3% (exact hypergeometric
  expectation), macro P 60.4 / R 99.0 / F 75.0 (Monte Carlo @ thr 0.600) vs
  paper 83.3 / 50.0 / 62.5. 64-pool P/R sweep: P 100% @ R 22.9, 85.3% @ 41.4,
  52.2% @ the paper's R=50 point. Tables + caveats in RESULTS.md.
- plots/speedup_progression.png (tools/plot_speedup.py): per-iteration scan
  speedup 1x -> 14x with gate-MRR overlay + full-benchmark RF endpoints
  (0.214 -> 0.557); 40x benchmark figure as a callout (different workload).
- Profiling refresh on the final build (plots/06_fil + rewritten
  plots/PERF-CHARACTERIZATION.md): post-mel kernel balance is fused 27.7% /
  distance 26.3% / dtw 19.2% / GEMMs 21.7%; tensor pipes 31.7%/45.8%
  (bandwidth-bound ceiling for these shapes); forest_predict is pure
  latency (long_scoreboard 452 cyc/issue, DRAM 35%, compute 4%) — confirms
  the packed-node 3x rationale. Idle decomposition via sqlite gap
  classification: 6.1% pageable-H2D staging, 3.7% one-time module load,
  0.8% malloc/free, ~0.1% launch overhead, ~6% sub-50us micro-gaps; NOT
  sync, NOT D2H. (Idle share grew from 6.4% pre-mel because mel cut GPU
  work 5.6x; absolute idle similar.)
- Tidy: README/PRESENTATION/TODO/PAPER-MAPPING reconciled to final numbers
  (fresh ladder run: canary 0.171, synth 0.421/0.346, Hung Up 0.415 — all
  PASS; Lucid->Shape now ranks #1 under the production config; TTS->MOU
  still #3 at heuristic stage). TODO.md rewritten: done items closed, only
  small benign items remain. docs/PAPER-VS-OURS.md added (canonical vs ours
  narrative, accuracy + perf rationale). Final commit: 'this is it'.

### 2026-06-11 — documentation restructure: README = single source of truth

Steven's call after scrapping the MkDocs idea: one reader-facing document.
- README.md fully rewritten to the approved outline (Brief / Build / Usage /
  Test / Results / Implementation details / Improvements / Repo layout):
  hero demo block, flags table, ladder + CPU-vs-GPU tables, Sample100 +
  10-pool + perf + per-kernel characterization tables, pipeline walkthrough
  with sizes, kernel inventory, optimization-iterations table 0-12.
- RESULTS.md DELETED — reader content absorbed into README "Results"; gritty
  evaluation detail moved to the "Evaluation detail (ex-RESULTS.md)" section
  above the build log (leakage mechanics, per-fold AUCs, reproducibility,
  10-pool simulation math, micro-eval absence).
- docs/PRESENTATION.md DELETED (no presentation needed; its pipeline/kernel
  content lives in README "Implementation details").
- New figures: plots/match_example.png (tools/visualize_match.py, Gimme@96s
  vs Hung Up@47s), plots/accuracy_comparison.png (tools/plot_accuracy.py),
  plots/pr_curve.png (tools/pool_analysis.py --plot, new flag).
- Staleness sweep: docs/TECHNICAL.md brought to as-built (K=32, classifier
  implemented, persistent DTW kernel + fused kernel + rf_infer in the kernel
  list, tests/ scripted, parity restored, current perf anchors); CLAUDE.md
  "Testing discipline" updated from the never-built golden-file workflow to
  the actual ladder+benchmark gates; all RESULTS.md/PRESENTATION.md
  cross-references repointed (TODO, TESTING, PAPER-MAPPING, PAPER-VS-OURS,
  ACCURACY-OPTIMIZATIONS). Build log entries above remain verbatim
  (append-only; historical mentions of RESULTS.md refer to its commit-era
  state, recoverable from git).
Not committed — left for Steven's review.

### 2026-06-11 — PAPER-MAPPING.md absorbed into the README

docs/PAPER-MAPPING.md DELETED (committed version recoverable at 4c25a3f):
its content is now the "Paper ↔ code mapping" table in README
"Implementation details" — paper block → function/kernel → fidelity marker
(1:1 / adapted D1–D6 / added A1–A2) with changes and optimizations per row.
The D/A labels were preserved so docs/PAPER-VS-OURS.md and TECHNICAL.md
references still resolve; all cross-references repointed to the README table.

### 2026-06-11 — code style: Google C++ standard adopted, mechanical sweep

Steven's call: follow the Google C++ Style Guide; braces on every control-
flow body (no single-line if/for), no compact C tricks.
- Checked-in .clang-format (Google base + InsertBraces; documented
  deviations: 4-space indent, 100-col, snake_case naming kept). Tool:
  clang-format 22.1.5 via pip --user (none on box). IncludeIsMainSourceRegex
  '\.cu$' so .cu files keep their .cuh pair as own-header-first.
- Reformatted all 16 source files (~1,275 insertions / 702 deletions of pure
  formatting). Manual pass: !strcmp -> strcmp()==0 (cpu_demo, gpu_detect);
  paired declarations split in backtrack_path; params.hpp pitch_shift
  qualifier wrapped in an SD_HOST_DEVICE macro (clang-format mangled the
  bare #ifdef form).
- docs/STYLE.md: Google standard is now the base, repo-specific rules
  retained; the four open TODO style items (line width, include order,
  struct-vs-class, int-vs-size_t) resolved per Google. docs/TECHNICAL.md
  build section documents the standard + reformat command.
- Verification on the rebuilt tree: ladder 4/4 PASS with scores
  BIT-IDENTICAL to pre-reformat (0.1710 / 0.4208 / 0.3463 / 0.4149);
  rf_infer parity unchanged (5.96e-08, 0/500k above 1e-4); CPU canary
  parity exact (0.2535 @ +0.00). Zero compiler warnings.
