/* CPU reference pipeline (single-threaded), stage-for-stage mirror of the GPU
 * detector (src/gpu/gpu_pipeline.cu): log-frequency STFT, simultaneous-update
 * KL-NMF/PFNMF, pitch-shifted templates, regularized correlation distance,
 * banded subsequence DTW, selectivity scoring. Each stage carries a note on
 * the GPU parallelization strategy (spec requirement). Kept in algorithmic
 * lockstep so GPU scores can be validated against it (float tolerances in
 * docs/TECHNICAL.md; same seeded inits, so differences are summation order). */

#pragma once

#include <vector>

#include "../common/match_info.hpp"
#include "../common/params.hpp"

namespace sd {

/* Row-major matrix. */
struct Mat {
    int rows = 0, cols = 0;
    std::vector<float> v;
    Mat() = default;
    Mat(int r, int c) : rows(r), cols(c), v((size_t)r * c, 0.f) {}
    float& at(int r, int c) { return v[(size_t)r * cols + c]; }
    float at(int r, int c) const { return v[(size_t)r * cols + c]; }
};

/* Magnitude spectrogram (ANALYSIS_BINS x frames): Hann STFT, then the
 * triangular log-frequency filterbank when MEL_BINS > 0 (same filterbank
 * construction as the GPU side). */
Mat stft_magnitude(const std::vector<float>& x);

/* KL-divergence multiplicative-update NMF, SIMULTANEOUS (Lee-Seung) variant:
 * both numerators come from one shared Z = V ./ (WH + eps) and the pre-update
 * W/H — mirrors gpu_nmf_batched, which fuses Z into a single custom kernel.
 * The first n_fixed columns of W are frozen (0 = plain NMF, RANK_K = PFNMF).
 * W and H must be pre-seeded by the caller. */
void nmf(const Mat& V, Mat& W, Mat& H, int iters, int n_fixed);

/* Query-independent candidate-side product (mirrors gpu_candidate_templates):
 * unit-norm spectral templates Wo and z-normalized activations Zo. */
struct CpuCandidateTemplates {
    Mat Wo;  /* ANALYSIS_BINS x RANK_K */
    Mat Zo;  /* RANK_K x No */
    int No = 0;
};
CpuCandidateTemplates candidate_templates(const Mat& Vc, int iters);

/* Per-pair scoring (mirrors gpu_score_candidates for one candidate): PFNMF at
 * every pitch shift against the query, banded subsequence DTW over every
 * candidate window, pitch-selectivity + selection-bias calibration. */
MatchInfo score_candidate(const Mat& Vq, const CpuCandidateTemplates& ct,
                          int iters, bool clip);

/* Deterministic uniform(0.01, 1) init — same mt19937 stream as the GPU host
 * init (seeds 42/43 candidate, 100+p/200+p per PFNMF shift). */
void seed_matrix(Mat& m, unsigned seed);

}  /* namespace sd */
