# Accuracy optimization roadmap (stashed ideas — nothing here is implemented)

Brainstormed 2026-06-11 after the RF stage landed. Constraint: no neural
networks — fundamental mathematical models, protocol changes, and user-input
leverage only. Each idea names the *measured* failure mode it targets;
current state for reference: heuristic MRR 0.214 → RF MRR 0.557 / hit@1 50%
on 64-candidate pools; OOF row AUC 0.629 (folds 0.55–0.71).

Measured failure modes these target:
- **F1 buried samples**: heavily processed/low-level samples are *found*
  (correct shift/location) but outranked (Touch The Sky → Move On Up class).
- **F2 universal matchers**: harmonically sparse clean-riff candidates match
  everything (largely but not fully fixed by the RF).
- **F3 data starvation**: 51,652 positives from 70 queries; per-fold AUC
  spread 0.55–0.71 says the classifier is data-limited, not model-limited.
- **F4 location errors**: the right candidate can win via the wrong window
  (Lucid → Shape ranks #1 at +4.25 st, not the known +0.00 alignment).
- **F5 no abstention signal**: Sample100 has no sample-free queries, so
  recall pegs at 100% and the threshold never learns to say "no sample".

Gating discipline for ANY of these: tests/run_ladder.sh structurally clean +
full-70-query benchmark (subsets overestimate ~2× — measured), leakage rules
of tools/train_rf.py unchanged.

## Tier 1 — highest expected ROI (data and features, classical)

### 1. Synthetic training-data augmentation for the RF  [F3, F5 — biggest lever]
The single most promising non-neural improvement, because it attacks data
scale — the axis neural systems usually win on. Generalize the fixture
machinery (tests/make_fixtures.sh already builds ground-truth inserts) into a
corpus generator: sample a segment (0.5–25 s) from original O, apply random
pitch shift (±5 st, continuous), time-stretch, EQ/filtering, compression, and
mix into an unrelated host track at −5…−25 dB; the label (candidate, location,
shift) is exact by construction. At ~0.9 s/candidate feature extraction,
thousands of labeled queries are an overnight job on the two A5000s.
Also generates **negatives-by-construction** (hosts with no insert) → trains
real abstention and unsticks the pegged recall (F5).
Variant: **hard-negative mining** — re-train including feature rows from the
confound pairs the current RF still ranks #1 wrongly.

### 2. Context features for the RF  [F2 — directly featurize the confound]
The "clean riff universal matcher" is *measurable from W_o itself*: add
candidate-template sparsity/entropy as features. Plus: shift-selectivity
margin (best vs 2nd-best shift), shift-consistency entropy across bands (true
pairs agree on shift; junk scatters), per-query margin/rank features,
candidate length, count of distinct query occurrences — and the existing
`heur_raw`/`heur_sel` columns (already in the corpus CSVs; currently excluded
from FEATS in tools/train_rf.py for purity — including them is legitimate).

### 3. Per-query normalization + ranking objective  [F2, F3]
Z-normalize RF probabilities within each query's candidate pool before
thresholding (removes query-difficulty offsets — some queries are uniformly
"hot"). Bigger step: replace RF with gradient-boosted trees under a
pairwise/LambdaMART-style ranking loss (still trees, no neural nets; sklearn
HistGradientBoosting is on the box). Ranking loss matches the actual task
(rank the true original above 63 negatives) better than pointwise
classification.

## Tier 2 — mathematical upgrades to the factorization

### 4. β-divergence NMF; Itakura–Saito on the power spectrogram  [F1 — most principled single change]
KL on magnitude weights loud cells more; **IS divergence (β=0) is
scale-invariant per time–frequency cell**, so a sample buried 20 dB under new
production contributes the same *relative* reconstruction error as a loud
one — precisely the buried-sample failure. The β-divergence multiplicative
updates generalize the current KL updates with one exponent change (same
kernel structure; a new `SD_BETA` knob fits the existing config matrix and
sweep tooling). Sweep β ∈ {0, 0.5, 1} × {magnitude, power}.

### 5. Sparsity + temporal-continuity penalties on H  [F1, F2]
True sample activations are temporally sparse and smooth; junk co-activation
is diffuse and flickery. Both penalties (Hoyer L1 sparsity; Virtanen 2007
continuity) have closed-form multiplicative-update modifications — small,
gated edits to the update_h kernel, two new knobs.

### 6. Convolutive NMF (NMFD)  [F1, F2 — biggest model upgrade short of neural]
Templates become short spectro-temporal patches (timbre + micro-rhythm)
instead of single-frame spectra; a riff's *rhythm* then has to match, not
just its spectrum — universal matchers lose their universality. Cost: ~T×
the GEMM work (T = patch frames), but batches along the same (cand × shift)
axis and the mel front end bought 5.6× headroom.

### 7. Multiple PFNMF random restarts  [variance reduction]
NMF is non-convex and we run exactly one seeded init. Min-over-3-inits
reduces init variance and is embarrassingly batchable along the existing
stacked-problem axis — purely a throughput question, and throughput is the
thing this project has plenty of.

## Tier 3 — front-end and alignment mathematics

### 8. Harmonic–percussive separation (HPSS) before NMF  [F1]
Classical median-filter separation (time-direction vs frequency-direction
medians on the spectrogram). Drums dominate hip-hop queries and pollute both
templates and activations; run the pipeline on the harmonic component, or
score-fuse harmonic and percussive runs. Two cheap new kernels.

### 9. Diagonal-projection cross-correlation feature  [F2, F4]
Summing the similarity matrix along diagonals = time-lag cross-correlation of
the activation sequences (a Hough transform for slope-1 lines). A sharp peak
is near-proof of a literal loop alignment; flat = resemblance. One cheap
kernel; a strong corroboration feature for the RF and a possible prefilter.

### 10. Hard slope constraints inside DTW (Itakura / step-pattern weights)  [F4]
Replace the *post-hoc* slope filter with local path constraints in the
recurrence, so degenerate paths can never accumulate cheap cost in the first
place — strengthens the whole cost landscape instead of filtering its output.
Step-pattern weighting (favoring diagonal moves) is well-studied DTW theory
and changes what the cost normalization rewards.

### 11. Multiple band lengths with fusion  [F1]
Samples span 0.5–25 s; we search 4 s only. Extract at {2, 4, 8} s and fuse
(max, or as classifier features). The naive worst-across-lengths variant was
tried and REVERTED (punishes looped samples) — fusion must be per-length
features, not a hard min/max across lengths.

### 12. Spectral whitening before NMF  [F1]
Per-band normalization lifts weak harmonics of buried samples before the
factorization sees them; interacts with idea 4 (IS divergence achieves a
related effect more principledly).

## Tier 4 — protocol, ensembling, measurement

### 13. Config ensembling via reciprocal-rank fusion  [cheap accuracy]
mel4_k32 / hop2048 / shift05 have different error patterns and coexisting
template caches; the full benchmark is only ~13 min/config on 2 GPUs.
Reciprocal-rank fusion of 2–3 configs is nearly free and classically robust.

### 14. Per-sample-type analysis  [know where the misses live]
Sample100 annotates sample types; measure where the remaining 50% of top-1
misses concentrate (drum-only samples are known-hard for this entire method
family). Possibly per-type thresholds or per-type expectations in the README results.

### 15. Location-level (micro) evaluation harness  [F4]
The paper reports micro P/R/F; we report none. Needed both to quantify F4 and
to tune anything that claims to fix it.

## Tier 5 — user-input leverage (shorter snippets etc.)

### 16. User-marked query snippets  [F1, F4 — Steven's idea]
`--query-window start,dur` (the dual of `--clip`): score only the suspected
region of the query. Removes dilution from the non-sample majority of the
query, sharpens the dip statistics, and shrinks Ns (faster too). While at it,
validate `--clip` accuracy — implemented but never accuracy-tested.

### 17. User pitch-shift hint  [F2]
Optional `--shift-range lo,hi` when the user can hear the transposition:
collapses the 41-hypothesis grid to a few, removing most of the confound
search space and most of the compute.

### 18. Human-in-the-loop triage  [precision without model changes]
Render the top-5 candidates via tools/visualize_match.py (optionally with
aligned audio snippets) for human confirmation — converts the system's good
hit@3/hit@5 into high-precision decisions through interaction.

## Honest ceilings and known dead ends

- The non-neural method family tops out around 34–50% recall on hard samples
  (the paper's own numbers); expect diminishing returns from any single
  Tier 2–3 item. Tier 1 (data) is the lever most likely to move hit@1
  substantially, because the evidence (AUC fold spread, 0.30% positive rate)
  says the classifier is data-starved, not model-starved.
- Already tried and REVERTED — do not revisit without new reasoning:
  frame-shuffle null, time-reversal null, two-stage shift screening (twice),
  worst-across-{4,8}s windows, --iters 30 (all documented with measurements
  in the CLAUDE.md build log).
