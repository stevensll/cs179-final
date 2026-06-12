/* Algorithm parameters — single source of truth (see docs/TECHNICAL.md).
 * All values from Gururani & Lerch 2017 unless noted. */

#pragma once

#include <array>
#include <cstddef>

namespace sd {

/* Tunable knobs are #ifndef-wrapped so the config matrix (tools/sweep_configs)
 * can stamp out build variants via -DSD_*=... while keeping everything
 * constexpr. Defaults = the gated production configuration. */
constexpr int SAMPLE_RATE = 22050; /* analysis rate; inputs decimated 2:1 */
constexpr int FFT_SIZE = 4096;
#ifndef SD_HOP
#define SD_HOP 1024
#endif
constexpr int HOP = SD_HOP;
constexpr int N_BINS = FFT_SIZE / 2 + 1; /* 2049 (STFT output) */

/* Log-frequency front end: the magnitude spectrogram is pooled through a
 * triangular log-spaced filterbank (BINS_PER_ST bins per semitone from
 * MEL_FMIN). Two effects: ~5.6x less work everywhere downstream (analysis
 * bins 2049 -> ~367), and a pitch shift becomes an exact integer TRANSLATION
 * of the template (0.25 st grid x 4 bins/st), replacing the lossy
 * interpolated frequency rescale that made fractional-shift misses fatal.
 * SD_MEL_BINS=0 disables (linear-frequency path, the pre-mel pipeline). */
#ifndef SD_MEL_BINS
#define SD_MEL_BINS 367
#endif
#ifndef SD_BINS_PER_ST
#define SD_BINS_PER_ST 4
#endif
constexpr int MEL_BINS = SD_MEL_BINS;
constexpr int BINS_PER_ST = SD_BINS_PER_ST;
constexpr float MEL_FMIN = 55.f;
constexpr int ANALYSIS_BINS = MEL_BINS > 0 ? MEL_BINS : N_BINS;

/* Paper uses K=10 templates for a ~4.5 s sample. Our "sample NMF" block models
 * the FULL candidate song (the sample location is unknown), so the rank scales
 * accordingly to keep templates part-like rather than muddy mixtures — K=10
 * over a dense 3-minute track yields mixture templates whose activations are
 * not discriminative. Same block, same rationale, content-scaled rank. */
/* K=32 selected by the config sweep + full-70-query confirmation (2026-06-11):
 * MRR 0.214 vs 0.203 at K=40, and faster. */
#ifndef SD_RANK_K
#define SD_RANK_K 32
#endif
#ifndef SD_RANK_L
#define SD_RANK_L 20
#endif
constexpr int RANK_K = SD_RANK_K;  /* candidate ("sample") templates */
constexpr int RANK_L = SD_RANK_L;  /* free mixture templates in PFNMF */
constexpr int DEFAULT_ITERS = 100; /* NMF multiplicative-update iterations */
constexpr float NMF_EPS = 1e-8f;

/* Two-stage PFNMF shift screening — DISABLED (SCREEN_KEEP = N_SHIFTS).
 * Tried and REVERTED twice (15/8 and 25/16): interim-score screening degraded
 * the eval-5 gate both times (MRR 0.354 -> 0.186 / 0.221; a rank-1 hit fell
 * to rank 5) — partially-converged scores mixed into the selectivity medians
 * corrupt cross-candidate comparability. Machinery kept for re-evaluation
 * with a future classifier-based scorer. */
constexpr int SCREEN_ITERS = 15;
constexpr int SCREEN_KEEP = 9999; /* >= N_SHIFTS = disabled */

/* Correlation regularizer at the paper's H/max(H) scale: activations are first
 * normalized by their global max (paper 3.2.1), then column z-normalization
 * uses denom = sigma + Z_REG. Frames with activation spread >> Z_REG behave
 * like exact Pearson; near-silent frames shrink toward zero so they contribute
 * a neutral distance instead of amplified noise. */
constexpr float Z_REG = 0.01f;

/* Hypothesized pitch shift of the sample in the query, in semitones.
 * Deviation from the paper's 12-step set: quarter-semitone grid, because
 * resampling-style speedups give fractional shifts and a 0.5-st mismatch
 * empirically destroys the DTW dip (see CLAUDE.md build log). */
/* SD_SHIFT_STEP4 = step in quarter-semitones (1 -> 0.25 st grid of 41;
 * 2 -> 0.5 st grid of 21). Integer so N_SHIFTS stays constexpr-derivable.
 * With the log-frequency front end any multiple of 0.25 st remains an exact
 * integer template translation. */
#ifndef SD_SHIFT_STEP4
#define SD_SHIFT_STEP4 1
#endif
constexpr int N_SHIFTS = 40 / SD_SHIFT_STEP4 + 1; /* -5 .. +5 st inclusive */
constexpr float SHIFT_MIN = -5.f;
constexpr float SHIFT_STEP = 0.25f * SD_SHIFT_STEP4;
#ifdef __CUDACC__
#define SD_HOST_DEVICE __host__ __device__
#else
#define SD_HOST_DEVICE
#endif
SD_HOST_DEVICE constexpr float pitch_shift(int p) {
    return SHIFT_MIN + p * SHIFT_STEP;
}

/* Deviation from the paper (we have no ground-truth sample snippet): templates
 * come from the FULL candidate song; the window search over "which part of the
 * candidate is the sample" happens inside DTW as bands of WINDOW_SECONDS
 * consecutive candidate frames at WINDOW_HOP_SECONDS steps — dense search at
 * no extra NMF cost. Short windows because real samples are chopped/looped. */
/* 4 s ~= the average sample length in the paper's dataset (4.2: "average
 * length of the samples is 4.5 s"); longer windows dilute chopped samples.
 * (A worst-across-{4,8}s variant was tried and REGRESSED: loop-boundary
 * misalignment punishes true looped samples more than confounds.) */
constexpr std::array<int, 1> WINDOW_SECONDS = {4};
#ifndef SD_WINDOW_HOP_SECONDS
#define SD_WINDOW_HOP_SECONDS 1
#endif
constexpr int WINDOW_HOP_SECONDS = SD_WINDOW_HOP_SECONDS;

} /* namespace sd */
