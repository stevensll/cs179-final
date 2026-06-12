# Canonical paper vs. this implementation — what changed and why

Companion note to the README's paper ↔ code mapping table and
the README "Results" section (numbers). This is the narrative: every place we deviate from
Gururani & Lerch 2017, split by *why* — some changes exist because our task is
harder than the paper's, some because they measurably improved accuracy, and
some purely for GPU throughput (held to a no-accuracy-loss gate).

## The canonical method, in one paragraph

The paper assumes you already have the **isolated sample audio** and ask only
"is it in this song?": NMF decomposes the sample into K=10 spectral templates
W_o; PFNMF decomposes the query against [W_o frozen | free mixture templates],
at 12 hypothesized pitch shifts; activations are H/max(H)-normalized; a
correlation distance matrix between sample and query activations feeds a
subsequence DTW (free start, cost = last row / path length); 13 path/cost
features from the DTW go into a 200-tree random forest that makes the final
binary call against 10-candidate pools.

## Why our task forced structural changes

We do **not** have the isolated sample — only full songs on both sides — and we
rank a 64-candidate library instead of classifying 10-candidate pools. Three
deviations follow directly:

1. **Full-song templates, K=10 → 32** (D1). The "sample NMF" must model an
   entire candidate song, so the rank scales with content (the paper's own
   §3.1.1 rationale). K=32 was selected by the config sweep *with full-benchmark
   confirmation* — K=24 is below the rank floor (MRR 0.285), K=40 is no better
   and slower (0.203 vs 0.214).
2. **The sample-location search moved into DTW** (D5): subsequence DTW runs
   over dense 4-second bands of candidate frames (1 s hop), every band a
   "maybe the sample is here" hypothesis. 4 s ≈ the paper's average sample
   length. Crucially this dense search reuses one distance matrix per shift —
   it costs no additional NMF.
3. **Hypothesis-distribution calibration** (A2): we take a min over ~6,500
   (shift × band) hypotheses per pair where the paper takes none, so longer
   candidates get more lottery tickets. Each candidate's best hypothesis is
   judged against its own min/median hypothesis distribution.

## Changes made for accuracy (each A/B-gated on the verification ladder)

- **Quarter-semitone pitch grid, 41 steps vs 12** (D3): real resampling-style
  sample speedups land at fractional shifts (Touch The Sky sits at −0.50 st),
  and a 0.25 st grid miss measurably destroys the DTW dip. The paper's coarse
  grid works only because their dataset's shifts were annotated.
- **Log-frequency front end** (D6, also a huge perf win — see below): 2049
  linear FFT bins pooled to 367 log-spaced bins (4 per semitone from 55 Hz).
  Accuracy effect: a pitch shift becomes an **exact integer translation** of
  the template, replacing the paper's lossy interpolated frequency rescale.
- **Regularized, attribution-weighted correlation** (D4): exact Pearson
  amplifies near-silent frames into unit noise vectors (spurious dips at
  fade-outs). We add Z_REG=0.01 to the z-norm denominator at the paper's
  H/max(H) scale, and weight each frame by the PFNMF source attribution
  e = |H_fixed|/(|H_fixed|+|H_free|) — operationalizing the paper's own
  premise that fixed-template activations indicate sample *presence*.
- **Template norm hygiene** (D2): unit-L2 template columns (norms folded into
  H_o), re-normalized after every pitch shift — extreme shifts otherwise
  shrink norms and bias the distance toward grid edges.
- **Pitch-selectivity scoring** (A1): a real match dips at ONE shift (paper
  Fig. 2); broadband junk dips at all of them. Each window's dip is normalized
  by that window's median dip across all 41 shifts.
- **Random forest, faithfully reproduced** (§3.3–3.4): same 13 path-geometry
  features (from GPU DTW with predecessor recording + host backtracking), same
  200-tree/√13 config. Evaluated leakage-safe (GroupKFold over query↔original
  connected components, out-of-fold only, train-side threshold). Effect:
  hit@1 11.4% → 50.0%, MRR 0.214 → 0.557, macro F 66.7% vs the paper's 62.5
  on 6× larger pools. Feature importances confirm the paper's thesis — path
  geometry (slope, cost, deviation) does the separating.

## Changes made for GPU performance

Two classes, with very different evidence requirements.

**Math-identical restructurings** (bit-identical or float-reorder-only; gated
by "scores match to 4 decimals"):

- **Persistent-band DTW kernel**: one block per (shift, band) sweeps all
  anti-diagonals in shared memory over a *diagonal-skewed* distance layout
  (anti-diagonals contiguous → coalesced), with the slope filter and
  min/mean/argmin reduced in-kernel to 4 floats per band. Replaced ~325k
  kernel launches and ~80 GB of cost-matrix traffic per candidate.
- **Fused W·H+ratio kernel**: NMF's inner dimension (R ≤ 52) fits one
  shared-memory tile, so Z = V/(WH+ε) is computed without ever materializing
  WH — eliminating the largest per-iteration intermediate (~1.7 GB across
  shifts). This is exactly the skinny-k shape where cuBLAS underperforms.
- **Cross-candidate PFNMF batching**: every PFNMF problem is query-sized
  (candidate length never enters it), so all candidates × 41 shifts stack
  into one strided batch with zero padding.
- **Caching**: candidate templates disk-cached per library song
  (fingerprint-keyed); the 23M-float seeded PFNMF init cached per worker.
- **Multi-GPU**: worker thread per device with a shared candidate queue and
  fairness-capped group sizes.

**Accuracy-affecting approximations** (gated by the full eval, not just the
ladder):

- **TF32 tensor-core GEMMs** and **--use_fast_math**: ladder scores identical
  to 4 printed decimals; eval ranks unchanged. Kept.
- **Simultaneous (Lee–Seung) NMF updates**: both numerators from one shared
  Z instead of the alternating form's recompute (3 GEMMs + 1 ratio per
  iteration instead of 4 + 2). The gate *improved* (MRR 0.215 → 0.354) —
  kept, and arguably an algorithmic improvement.
- **Log-frequency front end** (D6 again): ~5.6× less compute everywhere
  downstream of the spectrogram. Gated: eval MRR held.
- **Tried and REVERTED** — the honest part of the record: two-stage shift
  screening (twice: partially-converged interim scores corrupt the
  selectivity medians), iters 60→30 (ladder separation eroded), a 128×64
  fused-kernel tile (measured neutral — the H slab already lives in L2),
  worst-across-{4,8}s windows (punishes looped true samples). Machinery for
  the first is kept disabled in `params.hpp` (SCREEN_KEEP).

Net effect (the README "Performance" table): 64-candidate query scan ~340 s → 19.3 s
(18×), full 70-query benchmark ~8.5 h → ~13 min, with *better* accuracy than
the starting point.

## The discipline that made both halves safe

Every change — accuracy or perf — passed the same two gates before merging:
(1) the verification ladder (self-match canary, two synthetic inserts with
exact ground-truth shift/location, one verified real pair), and (2) no
regression on Sample100 eval ranks. The one measurement lesson worth carrying
forward: small eval subsets overestimate (a 9-query sweep subset read MRR ~2×
high); frontier decisions need the full benchmark.
