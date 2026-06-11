# Paper ↔ Code Mapping

Block-by-block correspondence between Gururani & Lerch 2017 (Figure 1 / §3) and
this implementation. Every paper block exists in code; each is marked:

- **[1:1]** faithful implementation of the paper block
- **[adapted]** same block, parameters/inputs adapted (reason given)
- **[added]** block not in the paper (needed because we rank a library without
  the paper's trained classifier)
- **[todo]** paper block not yet implemented (see TODO.md)

## Block diagram

Paper Figure 1 on the left; this codebase on the right (`k:` = kernel in
`src/gpu/kernels.cu`, `p:` = host orchestration in `src/gpu/gpu_pipeline.cu`).

```
        PAPER (Fig. 1, §3)                       THIS IMPLEMENTATION
  ─────────────────────────────      ──────────────────────────────────────────────
        Sample      Song               Candidate song          Query song
          │           │                     │                      │
   [pre-process §3.1] │              [1:1]  load_preprocessed()  (audio.cpp)
   downmix·RMS·22kHz  │                     downmix · RMS-normalize · 2:1 decimate
          │           │                     │                      │
   [spectrogram]      │              [1:1]  p: gpu_stft()
   4096/1024 magnit.  │                     k: window_frames + cuFFT + magnitude
          │           │                     4096 / 1024, Hann, 2049 bins
          ▼           │                     │                      │
   ┌────────────┐     │           [adapted] p: gpu_nmf_batched(P=1)   (D1, D2)
   │ NMF        │     │                     KL updates; K=32 over the FULL
   │ → W_o, H_o │     │                     candidate (sample location unknown;
   └────┬───┬───┘     │                     rank content-scaled per §3.1.1);
        │   │         │                     k: normalize_columns (unit-norm
        │   │         │                     templates, norms folded into H_o)
  W_o^p │   │ H_o     │                     │
  [pitch shift §3.2.2]│           [adapted] k: pitch_templates_batched   (D3)
  12 shifts −5..+5    │                     41 shifts −5..+5 in 0.25 st steps;
        │   │         │                     k: normalize_fixed_columns
        ▼   │         ▼                     │                      │
   ┌────────────────────┐            [1:1]  p: gpu_nmf_batched(P=41)
   │ PFNMF §3.1         │                   one cuBLAS strided-batched problem;
   │ W=[W_o^p|W_m] →H_s │                   W=[W_o^p fixed | L=20 free], first K
   └─────────┬──────────┘                   columns frozen (k: update_w skip)
             │                              │
   [activation norm §3.2.1]          [1:1]  k: max_normalize_batched
   H / max(H)                               (all rows scaled: preserves the
             │                              fixed/free ratio used below)
             ▼                              │
   ┌────────────────────┐         [adapted] k: znorm_batched + distance_batched
   │ distance matrix    │                   Pearson via z-norm dot;  (D4)
   │ D = 1 − corr §3.2.2│                   + Z_REG regularizer (silent frames →
   └─────────┬──────────┘                   neutral) + PFNMF source-attribution
             │                              weight e=|H_fix|/(|H_fix|+|H_free|)
             ▼                              │
   ┌────────────────────┐         [adapted] k: dtw_band_diag           (D5)
   │ subsequence DTW    │                   eq. 2 boundary + recurrence exactly;
   │ eq. 2; cost = last │                   run over dense 4 s bands of candidate
   │ row / path length  │                   frames (the sample-location search the
   └─────────┬──────────┘                   paper doesn't need); all shifts and
             │                              bands batched in one grid
             ▼                              │
   [pitch cand. selection §3.2.3]    [1:1]  min over shifts (folded into scoring)
             │                              │
             ▼                              ▼
   ┌────────────────────┐            ┌──────────────────────────────────────┐
   │ feature extraction │    [1:1]   │ heuristic scoring + feature mode     │
   │ 13 path/cost       │            │ · min & mean of cost fn  ≈ §3.3.1    │
   │ features §3.3      │            │ · path-warp bound L/T    ≈ §3.3.2    │
   └─────────┬──────────┘            │ [added] pitch selectivity (per-window│
             │                       │   median over shifts; cf. §3.2.3)    │
             │                       │ [added] min/median selection-bias    │
             │                       │   correction across hypotheses       │
             │                       │ [1:1] k: dtw_band_preds + host       │
             │                       │   backtracking -> the paper's 13     │
             │                       │   features (gpu_detect --features)   │
             ▼                       └──────────────────┬───────────────────┘
   ┌────────────────────┐            ┌──────────────────▼───────────────────┐
   │ random forest §3.4 │    [1:1]   │ tools/train_rf.py (200 trees, √13,   │
   │ classification     │            │ leakage-safe GroupKFold eval) +      │
   └────────────────────┘            │ CUDA inference (src/gpu/rf_infer.cu, │
                                     │ parity ≤6e-8 vs sklearn, 19× faster) │
                                     └──────────────────────────────────────┘
```

## Deviations, each with its reason

| # | Deviation | Reason |
|---|---|---|
| D1 | "Sample NMF" runs on the **full candidate song**, K=10→32 (sweep-selected with full-benchmark confirmation) | The paper has the ground-truth sample audio; we don't. Rank scaled per the paper's own rank-selection rationale (§3.1.1: enough templates to approximate the content). The sample-location search moves to D5. |
| D2 | Template columns unit-L2-normalized (norms folded into H_o), re-normalized after pitch shifting | Extreme shifts change template norms; unnormalized norms bias the correlation distance (observed as grid-edge false positives). |
| D3 | Pitch grid 41×0.25 st vs the paper's 12 steps | Resample-style speedups give fractional shifts; a 0.5 st grid miss empirically destroys the DTW dip (synthetic A/B in CLAUDE.md log). Paper grid was informed by their annotated dataset; ours must cover the continuum. |
| D4 | Correlation regularized (Z_REG) + weighted by PFNMF source attribution | Exact Pearson amplifies near-silent frames into unit noise vectors (spurious dips at fades); the attribution weight realizes the paper's premise that fixed-template activations indicate *presence*. |
| D5 | Subsequence DTW run over dense 4 s candidate bands | Consequence of D1: every candidate window is a sample hypothesis. 4 s = the paper's average sample length (§4.1). Bands reuse one distance matrix, so the dense search costs no extra NMF. |
| D6 | Log-frequency front end: the 2049-bin magnitude spectrogram is pooled through a triangular log-spaced filterbank (4 bins/semitone from 55 Hz, 367 bins) before NMF | Two wins over the paper's linear axis: ~5.6× less compute everywhere downstream, and a pitch shift becomes an EXACT integer translation of the template (the paper's frequency rescale — our old D3 interpolation — is lossy, which made fractional-shift misses fatal). Gated: eval MRR held (0.354 → 0.363). Disable with `-DSD_DEFS="SD_MEL_BINS=0"`. |
| A1 | Pitch-selectivity normalization | Real matches dip at one shift (paper Fig. 2); broadband junk dips at all shifts. Without the classifier, this is the strongest single junk rejector. |
| A2 | Min/median selection-bias correction | We take a min over ~41×160 hypotheses per pair; longer candidates get more lottery tickets. The paper never minimizes over hypotheses, so it doesn't need this. |

## What the classifier bought (formerly "what the missing classifier costs")

The paper's random forest (13 features, §3.3–3.4) is the precision stage: it
separates "deep DTW dip because the sample is present" from "deep dip because
the candidate's templates resemble the query's music" — harmonically sparse
candidates (clean riffs) acting as universal melodic matchers were our
dominant pre-RF failure mode. Trained on Sample100 (the paper's own corpus
was never published; see CLAUDE.md log) with leakage-safe out-of-fold
evaluation: hit@1 11.4% → 50.0%, MRR 0.214 → 0.557, macro F 66.7% vs the
paper's 62.5 on 6× larger candidate pools. Numbers and caveats: RESULTS.md.
Narrative of all deviations: docs/PAPER-VS-OURS.md.
