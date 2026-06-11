# TODO — cleanup and improvements

Updated 2026-06-10 after v3 (files verified by Steven, K=40, selectivity
scoring, Hung Up pair added). See CLAUDE.md build log for history.

## Config / approximation frontier

- [ ] Decide the production default from the sweep (plots/sweep/pareto.png):
      mel4_k32 (accuracy champion) vs hop2048 (2x faster, -0.03 MRR on n=9).
      Full-73-query confirmation of mel4_k32 pending; rerun frontier points on
      the full set before locking.
- [ ] Knobs swept so far: mel bins/resolution, K, hop, shift grid, band hop.
      Untried: SD_RANK_L, WINDOW_SECONDS, iters axis (accelerated MU).

## Accuracy (the open front)

- [ ] **The paper's 13 path features + random-forest classifier** — now clearly
      the binding constraint: both remaining failures are melodic-resemblance
      confounds the classifier stage exists to separate (the detector already
      finds the true alignments). Needs backtracking + the paper's labeled
      dataset (~500 queries, repo linked in the paper).
- [ ] More test pairs incoming from Steven; re-run the full ladder on each
      (canary, 2 synths, 3 real pairs — commands in CLAUDE.md log).
- [ ] **Sample100 dataset**: spawn an agent to run `tools/scrape_sample100.py`
      per `tools/DATASET.md` (needs `pip install --user yt-dlp`), then verify
      downloads per the protocol there. Manifest already generated
      (`datasets/sample100/eval_pairs.csv`, 103 pairs).
- [ ] Batch eval runner over eval_pairs.csv (bash loop + hit-rate summary;
      start with a 10-query subset — full 76×68 sweep ≈ 10 h single-GPU,
      or split across both A5000s).
- [ ] `--clip` mode is implemented but untested; Steven will test with a
      hand-trimmed clip.

## Code cleanup

- [ ] **Make an initial git commit — the repo currently has zero commits and the
      entire project is untracked** (so there is no history to recover anything
      from; deleted code is gone unless reimplemented).
- [ ] CPU demo still implements the v1 (snippet-window) algorithm — port the v2
      design (full-song templates, banded DTW, min/median scoring) or clearly
      re-scope it as "CPU reference of the GPU stages" for the report.
      CPU/GPU score parity currently does NOT hold.
- [ ] Move the verification probes (/tmp/probe_*.py this session) into tools/
      as a `verify_pair.py` (the validated literal-copy detector is genuinely
      useful for vetting new test data).
- [ ] Script the synthetic fixtures into tests/make_synthetic.sh + a test runner
      for the ladder (canary / synth_plain / synth_fast / test pairs). NOTE: the
      exact ffmpeg commands were never logged and there is no git history to
      recover them from — reconstruct from the recipe in the CLAUDE.md log
      (8 s of Move On Up mixed into Shape of My Heart @30 s; same insert
      resampled +6%).
- [ ] Split kernels.cu / cpu_pipeline.cpp into per-stage files per
      docs/TECHNICAL.md; drop the unused reversal-null computation (or keep
      behind SD_DEBUG only).
- [ ] Remove legacy SNIPPET_* params once the CPU demo is ported.

## Performance

Campaign of 2026-06-11 done: 42 s -> 8.8 s scan; see CLAUDE.md log + plots/.
DTW persistent kernel, multi-GPU, TF32, interleaved NMF, fused W*H+ratio
custom kernel, host-init cache all landed; cuFFT plan reuse measured
irrelevant (STFT = 0.0% of wall). Remaining:

- [ ] Per-worker decode/setup prefetch (GPU idle ~25%, mostly audio decode
      between candidates; overlap next candidate's host work with current
      GPU work).
- [ ] ncu deep-dive on fused_wh_ratio (54% of kernel time at ~8 TFLOPS —
      double-buffering / vectorized smem loads might push it further).
- [ ] Straggler balance: 2-GPU scans split whole candidates; long candidates
      can leave one GPU idle at the tail (sort queue by length, longest first).

## Algorithm

- [ ] Backtracking + the paper's 13 path features; random-forest classifier
      (needs ~500 labeled queries — PROPOSAL.md extensions). The remaining
      false-positive pattern (clean solo-instrument candidates like Shape of My
      Heart act as universal melodic matchers) is exactly what those features +
      classifier are for in the paper.
- [ ] Multiple band lengths (paper samples range 0.5–25 s; we test 4 s only).
- [ ] Shift-consistency across bands as a corroboration feature (true matches
      agree on shift across windows; false ones scatter).
