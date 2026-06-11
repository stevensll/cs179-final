/* Algorithm parameters — single source of truth (see docs/TECHNICAL.md).
 * All values from Gururani & Lerch 2017 unless noted. */

#pragma once

#include <array>
#include <cstddef>

namespace sd {

constexpr int   SAMPLE_RATE   = 22050;  /* analysis rate; inputs decimated 2:1 */
constexpr int   FFT_SIZE      = 4096;
constexpr int   HOP           = 1024;
constexpr int   N_BINS        = FFT_SIZE / 2 + 1;   /* 2049 */
/* Paper uses K=10 templates for a ~4.5 s sample. Our "sample NMF" block models
 * the FULL candidate song (the sample location is unknown), so the rank scales
 * accordingly to keep templates part-like rather than muddy mixtures — K=10
 * over a dense 3-minute track yields mixture templates whose activations are
 * not discriminative. Same block, same rationale, content-scaled rank. */
constexpr int   RANK_K        = 40;     /* candidate ("sample") templates */
constexpr int   RANK_L        = 20;     /* free mixture templates in PFNMF */
constexpr int   DEFAULT_ITERS = 100;    /* NMF multiplicative-update iterations */
constexpr float NMF_EPS       = 1e-8f;

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
constexpr int   N_SHIFTS = 41;  /* -5 .. +5 in 0.25-st steps */
constexpr float SHIFT_MIN = -5.f;
constexpr float SHIFT_STEP = 0.25f;
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr float pitch_shift(int p) { return SHIFT_MIN + p * SHIFT_STEP; }

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
constexpr int WINDOW_HOP_SECONDS = 1;

/* Legacy (CPU demo v1 algorithm only — see TODO.md): single-snippet hypotheses. */
constexpr int SNIPPET_SECONDS     = 8;
constexpr int SNIPPET_HOP_SECONDS = 1;
constexpr int DEFAULT_WINDOWS     = 6;

}  /* namespace sd */
