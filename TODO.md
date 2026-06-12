# TODO — remaining items

Updated 2026-06-11, end of the RF/classifier session. Everything substantive
landed: production config locked (K=32 log-frequency), RF stage trained and
evaluated (README "Results"), CUDA forest inference verified, CPU parity restored,
profiling characterized (plots/PERF-CHARACTERIZATION.md). History: CLAUDE.md.

## Remaining (small, benign)

- [ ] Pinned staging buffers + async H2D for the PFNMF seeded-init slabs —
      the measured 6.1% single-GPU idle (PERF-CHARACTERIZATION.md §6).
- [ ] Forest kernel: walk 2–4 trees per thread interleaved to overlap the
      dependent-load latency chains (long_scoreboard 452 cyc/issue); bounded
      upside on an already-19×-sklearn kernel.
- [ ] `verify_pair.py`: move the session's literal-copy verification probes
      into tools/ (useful for vetting new test data).
- [ ] Split kernels.cu / cpu_pipeline.cpp into per-stage files per
      docs/TECHNICAL.md (cosmetic).
- [ ] `--clip` accuracy never validated with a real hand-trimmed clip
      (mechanism tested; used in perf benchmarks only).
- [ ] Classifier feature ideas: multiple DTW band lengths (samples range
      0.5–25 s; we extract at 4 s), shift-consistency across bands.
- [ ] Config knobs never swept: SD_RANK_L, WINDOW_SECONDS, accelerated-MU
      iteration schedules.

## Done (this project cycle — see CLAUDE.md log for evidence)

- [x] Production default from the sweep: K=32 locked by full-70-query
      confirmation (MRR 0.214 vs 0.203 at K=40).
- [x] The paper's 13 path features + random-forest classifier: trained on
      Sample100 (the paper's corpus was never published), leakage-safe
      GroupKFold eval; hit@1 11.4% → 50.0%, macro F 66.7% (64-pool) / 75.0%
      (10-pool, paper protocol) vs paper 62.5%.
- [x] CUDA forest inference (rf_infer): sklearn-exact (6e-8), 3.1 M rows/s.
- [x] Sample100 dataset scraped (143/144), eval runner + full benchmark.
- [x] CPU demo ported to the production algorithm; 4-decimal GPU parity.
- [x] Fixtures + ladder scripted (tests/); initial commits made.
- [x] Decode prefetch, longest-first queue, ncu deep-dives (00_baseline …
      06_fil), idle decomposition.
- [x] Legacy v1 machinery removed (SNIPPET_* params, best_snippet_offset).
