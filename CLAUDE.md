/* CLAUDE.md — operating notes for this repo */

# cs179-final: GPU sample detection

## Mission

Earn maximum points on the CS 179 final project: given a query song, find the song it
samples from a library, implementing Gururani & Lerch 2017 (NMF + subsequence DTW) with
custom CUDA kernels. Grading deliverables are in `README.md` (proposal, CPU demo, final
submission with performance analysis). The plan of record is `PROPOSAL.md`; algorithm
parameters and architecture live in `docs/TECHNICAL.md`; code style in `docs/STYLE.md`;
the library-vs-custom kernel rationale in `docs/gpu-library-vs-custom-kernels.md`;
test cases in `docs/TESTING.md`.

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

- Golden-file workflow: the CPU demo's per-stage outputs are the reference. GPU stages
  compare against them with the tolerances defined in `docs/TECHNICAL.md` (float
  stages ~1e-3 relative; the final ranking must match exactly).
- Both test pairs in `docs/TESTING.md` must pass end-to-end on CPU before any GPU work,
  and on GPU before any optimization work. Re-run both after every optimization.
- Never overwrite golden files casually — only regenerate them deliberately from a
  CPU demo that passes both test pairs, and say so in the build log.
- Tests are plain C++ executables with asserts; no frameworks, nothing vendored.

## Style conventions

See `docs/STYLE.md` for the full rules. The non-negotiables: `snake_case` everywhere
with `_kernel` suffix for kernels; `/* ... */` comments, why-only, except every kernel
carries a strategy block comment (decomposition, shared memory, why it beats naive);
magic numbers as named `constexpr`; `checkCuda()` on every CUDA call and after every
launch; device memory through `DeviceBuffer<T>`, no Thrust/CUB.

## Confirmed decisions

- Backend: rank library by min subsequence-DTW cost; no classifier (future-work note
  only).
- Kernel scope: **revised by Steven 2026-06-10 (v3)** — matrix multiplies via
  cuBLAS (NVIDIA built-ins allowed where they exist; enough custom kernels
  remain), cuFFT for STFT; all other stages are custom kernels.
- CPU demo in C++, sharing `src/common/` with the GPU build.
- Build: CMake + thin Makefile wrapper so graders can type `make`.
- Timeline: 3 weeks per `PROPOSAL.md`.

## File map

- `src/common/` — audio I/O, preprocessing, pipeline types (shared CPU/GPU)
- `src/cpu/` — CPU reference, one file per stage
- `src/gpu/` — paired `.cu/.cuh` per stage: stft, nmf, pitch_templates, distance, dtw
- `tests/` — per-stage golden tests + end-to-end test
- `golden/` — CPU-generated reference outputs (regenerate deliberately only)
- `music/` — song library + queries; `docs/TESTING.md` maps queries to expected matches
- `build/` — out-of-source build dir (not committed)

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
