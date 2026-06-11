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
./build/gpu_detect [--max-seconds S] [--iters I] [--clip] [--gpus N] [--features F.csv] <query.wav> <library_dir>
./build/cpu_demo   [--max-seconds S] [--iters I] [--clip] <query.wav> <library_dir>
```

Both print the library ranked by match score (lower = better, best match first),
and **`cpu_demo` is an algorithmically identical single-threaded mirror of the
GPU pipeline** — same stages, same seeds; scores match to 4 decimals on the
verification gates. The query file is skipped if it is inside the library
directory. Flags: `--max-seconds` truncates the query (essential for the CPU
demo — full-length CPU runs take hours by design), `--iters` NMF iterations
(default 100; 60 is plenty), `--clip` for a hand-trimmed query that IS the
suspected sample (scores by absolute alignment cost instead of dip depth),
`gpu_detect --features` dumps the classifier's 13 path features per hypothesis
instead of ranking.

Example (actual output, production config, warm template cache):

```
$ ./build/gpu_detect --iters 60 "music/Hung Up.wav" music
...
ranking (best match first):
1. Gimme Gimme.wav              score 0.4149 (shift +0.00 st, cand @96s -> query @47s)
2. Shape of My Heart.wav        score 0.5676 (shift +2.75 st, cand @20s -> query @186s)
...
total 3.0 s (2 gpus)
```

## How it works

The spectrogram is pooled to a log-frequency axis (367 bins, 4 per semitone),
so a pitch shift is an exact integer translation of a template. For each
library candidate: NMF (K=32, KL divergence; the paper's K=10 scaled up
because templates model the full song, not a known snippet) extracts spectral
templates from the full candidate song; for all 41 pitch-shift hypotheses
(−5..+5 st, quarter-semitone steps) the translated templates are held fixed in
one batched partially-fixed NMF (L=20 free templates) over the query's
spectrogram; regularized correlation distance between candidate and query
activations feeds a banded subsequence DTW (wavefront-parallel, all shifts and
bands batched) that densely searches every 4 s candidate window; the heuristic
score is the depth of the best DTW cost dip, normalized for pitch selectivity
(a real match dips at one shift only) and selection bias, with a path-slope
sanity bound (paper §3.3.2). On top of that, the paper's classifier stage
(§3.3–3.4): DTW paths are backtracked into 13 path-geometry features
(`--features`), a 200-tree random forest is trained on Sample100
(`tools/train_rf.py`, leakage-safe), and a FIL-style CUDA kernel
(`build/rf_infer`) runs the forest at 3.1 M rows/s with sklearn-exact output.
Matrix multiplies are cuBLAS (strided-batched TF32); STFT is cuFFT; everything
else is custom kernels (windowing, magnitude, fused NMF update, reductions,
pitch-template translation, normalization, correlation distance, banded
wavefront DTW, forest traversal). **The block-by-block correspondence to the
paper — with a diagram and every deviation justified — is in
`docs/PAPER-MAPPING.md`; the narrative of every deviation and optimization is
`docs/PAPER-VS-OURS.md`.**

A match can be eyeballed with the visual aid:

```
python3 tools/visualize_match.py "music/Move On Up.wav" 0 "music/Touch The Sky.wav" 2.5 10 out.png
```

renders the two segments' spectrograms stacked for comparison (uses the
`cand @Xs -> query @Ys` locations that `gpu_detect` prints).

## Results

Heuristic ranking on the `music/` anecdotes (production config, fresh run;
`tests/run_ladder.sh` checks the first four as structural gates):

| Test | Expected | Result |
|---|---|---|
| Self-match canary (song vs itself) | top score, shift 0, aligned offsets | **pass** (0.171 @ +0.00 st) |
| Synthetic: 8 s sample mixed into another song | dip @ +0.00 st, found at insert location | **pass** (0.421 @ +0.00 st, locations exact) |
| Synthetic: same, resampled +6% speed | dip @ +1.01 st | **pass** (0.346 @ +1.00 st) |
| Hung Up → Gimme Gimme | rank #1 | **pass, decisive** (0.415 vs 0.568 runner-up) |
| Lucid Dreams → Shape of My Heart | rank #1 | **pass** (0.583; reported window differs from the known +0.00 alignment) |
| Touch The Sky → Move On Up | rank #1 | **fail at heuristic stage** (true −0.25 st alignment found, ranks #3 behind two confounds) |

The synthetic controls verify the implementation responds exactly as theory
predicts to pitch shifting and time stretching, and recovers the sample's
location in both songs unaided. Where the heuristic fails, the detector still
*finds* the true alignment (correct shift, plausible locations) but is edged
out by melodic-resemblance confounds — harmonically sparse candidates whose
clean riff templates match similar minor-key arpeggios in other songs. That
confound class is exactly what the paper's 13-feature random-forest stage
separates — now implemented and trained (see Evaluation below): on the
Sample100 benchmark it lifts top-1 accuracy from 11.4% to 50.0%.

## Evaluation

To go beyond the n=3 anecdotes above, the repo carries the **Sample100**
benchmark (Van Balen 2011, the public standard for sample detection):
`datasets/sample100/eval_pairs.csv` holds the ground truth (103 query→original
pairs; 3 interpolation rows excluded), and `tools/eval_sample100.py` runs
`gpu_detect` per query against the annotated originals and reports hit@1 /
hit@3 / MRR. Audio: 143/144 tracks downloaded (`tools/scrape_sample100.py`,
protocol in `tools/DATASET.md`; 4 bad/missing tracks excluded via
`excluded_tracks.txt`).

Full benchmark, 70 evaluable queries × 64-candidate pools (details, caveats,
and the paper comparison in **`RESULTS.md`** — the headline document):

- heuristic ranking: MRR 0.214, hit@1 11.4% (random baseline MRR ≈ 0.07)
- **+ random forest (out-of-fold): MRR 0.557, hit@1 50.0%, hit@3 57.1%;
  macro P/R/F 50.0/100.0/66.7%** vs the paper's 83.3/50.0/62.5% on
  10-candidate pools
- under the paper's own protocol (1 true + 9 random, simulated exactly from
  the out-of-fold scores): top-1 60.3%, **macro F 75.0% vs the paper's 62.5%**

## Performance

After the profiled optimization campaign (per-iteration nsys reports and
matplotlib plots in `plots/`; full narrative in the CLAUDE.md build log):

| | time |
|---|---:|
| 5-candidate library scan (full songs, 60 iters, 41 shifts) | **3.0 s** (was 42 s) |
| full-song query vs 64-candidate library (warm template cache) | **19.3 s** (was ~5.7 min) |
| 15 s clip vs 64-candidate library (warm cache) | **2.3 s** (was 21.7 s) |
| full 70-query Sample100 benchmark | **~13 min** (was ~8.5 h est.) |
| random-forest inference (`rf_infer`, 200 trees) | **3.1 M rows/s** (19× sklearn on all CPU cores, sklearn-exact output) |

![optimization progression](plots/speedup_progression.png)

The single largest algorithmic win is the log-frequency front end (D6 in
`docs/PAPER-MAPPING.md`): pooling the spectrum to 367 log-spaced bins cuts all
downstream compute ~5.6× *and* turns pitch shifts into exact integer template
translations — accuracy held (eval MRR 0.354 → 0.363). A config matrix
(`-DSD_DEFS=...`, swept by `tools/sweep_configs.py`) maps the
approximation-vs-accuracy Pareto frontier (`plots/sweep/pareto.png`).

### Optimization iterations

Standard benchmark throughout: the 5-candidate full-song library scan
(`music/`, `--iters 60`, both GPUs unless noted). Every kept change passed the
verification ladder (canary + synthetics: exact shifts/locations) and the
eval-5 rank gate; profile evidence per iteration lives in `plots/`.

| # | Change | Scan | Note |
|---|---|---:|---|
| 0 | v3 algorithm, pre-optimization | 42.0 s | baseline; profile `00_baseline` (DTW = 33.6% of wall, 325k launches/candidate) |
| 1 | TF32 math mode + elementwise kernel restructure | 37.2 s | scores drift ≤1e-4 |
| 2 | Persistent-band DTW kernel over diagonal-skewed distance layout | 21.0 s | bit-identical; DTW 33.6% → 2.2%; killed ~80 GB traffic + 600 MB D2H per candidate |
| 3 | Multi-GPU (worker per device, shared queue) | 12.8 s | bit-identical |
| 4 | Simultaneous-update NMF (3 GEMMs + 1 ratio/iter, was 4+2) | 9.3 s | *improved* eval ranks (MRR 0.215 → 0.354) |
| — | Two-stage shift screening (15/8, then 25/16) | — | **reverted**: eval gate failed both times (MRR → 0.186/0.221) |
| 5 | Fused custom W·H+ratio kernel (k ≤ 64 single tile; `WH` never materialized) | 9.3 s | bit-identical; single-GPU 14.1 → 10.6 s (2-GPU masked by 3/2 split) |
| 6 | Host-init cache + sync removal + longest-first queue | 8.8 s | byte-identical |
| 7 | Candidate-template disk cache (`.tcache`) + decode prefetch | 7.5 s | clip search 21.7 → 7.9 s |
| 8 | Cross-candidate PFNMF batching + persistent per-worker scratch + fair groups | 6.9 s | clip 4.9 s; fixed two scheduling bugs found by measurement |
| 9 | `--use_fast_math` | ~6.9 s | ~5% across workloads; scores identical to 4 decimals |
| — | 128×64 fused tile (Boehm-style) | — | **reverted**: measured neutral (H slabs already L2-resident) |
| 10 | Log-frequency front end (367 bins, shifts = exact translations) | **3.2 s** | MRR 0.354 → 0.363; full-song-vs-64 60.2 → 21.2 s; profile `05_mel` |
| 11 | K=32 default (config-matrix sweep + full-70-query confirmation) | **3.0 s** | full-benchmark MRR 0.214 vs 0.203 at K=40 |
| 12 | FIL-style forest-inference kernel (packed 16-byte nodes) | — | 1.0 → 3.1 M rows/s (3×); output matches sklearn to 6e-8; profile `06_fil` |

Final-build kernel-level characterization (SM/DRAM utilization, occupancy,
tensor-core utilization, warp-stall reasons, idle decomposition):
`plots/PERF-CHARACTERIZATION.md`.

Historical v1 reference: CPU 149 s vs GPU 2.2 s (68×) on a matched config. The
CPU demo now mirrors the production algorithm with 4-decimal score parity
(~48 s for a 30 s canary config the GPU does in ~1 s).

## Potential improvements

See `TODO.md` for the remaining (small) items. Headlines: pinned staging
buffers + async H2D for the residual ~6% single-GPU idle; interleaved
multi-tree traversal in the forest kernel; multiple DTW band lengths and a
shift-consistency feature for the classifier.

## Repo layout

```
src/common/   audio I/O + preprocessing (shared)     RESULTS.md    headline numbers vs paper
src/cpu/      CPU reference (4-decimal GPU parity)   PROPOSAL.md   proposal + timeline
src/gpu/      custom kernels + cuBLAS wrappers       docs/         paper, design docs, tests
              + GPU pipeline + rf_infer              TODO.md       remaining small items
music/        song library                           CLAUDE.md     operating notes + build log
tests/        fixtures + verification ladder         build/        out-of-source build dir
tools/        eval runner, RF trainer/exporter, pool analysis, sweep, plots, DATASET.md
datasets/     sample100/ evaluation set (eval_pairs.csv ground truth + audio/)
```
