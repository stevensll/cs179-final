/* CPU reference pipeline (single-threaded). One function per stage; each carries
 * a note on the GPU parallelization strategy (spec requirement). MVP keeps all
 * stages in one translation unit — see TODO.md for the planned split. */

#pragma once

#include <vector>

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

/* Magnitude spectrogram, N_BINS x frames (Hann, FFT_SIZE/HOP from params). */
Mat stft_magnitude(const std::vector<float>& x);

/* KL-divergence multiplicative-update NMF: V ~ W*H. W (M x R), H (R x N) must be
 * pre-initialized (caller seeds them; keeps CPU/GPU runs comparable). The first
 * n_fixed columns of W are frozen (n_fixed = 0 -> plain NMF, = RANK_K -> PFNMF). */
void nmf(const Mat& V, Mat& W, Mat& H, int iters, int n_fixed);

/* Frequency-axis rescale of templates by 2^(semitones/12) (linear interp). */
Mat pitch_shift_templates(const Mat& W, float semitones);

/* Scale each column of W to unit L2 norm; if H is non-null, scale row k of H by
 * the removed norm (preserves W*H). Keeps activation scales comparable across
 * pitch shifts (extreme shifts shrink template norms and bias the distance). */
void normalize_columns(Mat& W, Mat* H);

/* D(i,j) = 1 - Pearson r between column i of Ho and column j of Hs, using the
 * first RANK_K rows of each. */
Mat correlation_distance(const Mat& Ho, const Mat& Hs);

/* Subsequence DTW (paper eq. 2): returns min over end frames of the last row of
 * the cost matrix, normalized by alignment path length. */
float dtw_min_cost(const Mat& D);

/* Deterministic uniform(0.01, 1) init shared by CPU and GPU runs. */
void seed_matrix(Mat& m, unsigned seed);

}  /* namespace sd */
