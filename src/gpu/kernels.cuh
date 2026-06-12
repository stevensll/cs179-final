/* Custom kernels — declarations of the host-side launch wrappers.
 * All matrices row-major. "Batched" kernels operate on P stacked problems
 * (one per pitch-shift hypothesis) at fixed strides; matrix multiplies are
 * cuBLAS (gemm.cuh), everything else is custom. */

#pragma once

#include <cufft.h>

namespace sd {

void launch_window_frames(const float* x, int frames, float* out);
void launch_magnitude(const cufftComplex* cplx, int frames, float* V);

/* NMF steps over P stacked (M x N) problems; V shared across problems. */
/* Fused W*H + ratio epilogue: Z[b] = V ./ (W[b]*H[b] + eps); WH is never
 * materialized. Requires R <= 64 (single shared k-tile). */
void launch_fused_wh_ratio(const float* W, const float* H, const float* V, float* Z, int M, int N,
                           int R, int P);
void launch_col_sum_batched(const float* W, int M, int R, int P, float* out);
void launch_row_sum_batched(const float* H, int R, int N, int P, float* out);
void launch_update_h_batched(float* H, const float* numH, const float* wcol, int R, int N, int P);
void launch_update_w_batched(float* W, const float* numW, const float* hrow, int M, int R, int P,
                             int n_fixed);

/* Writes the pitch-shifted copy of Wo (M x K) into the first K columns of each
 * W_all[b] (M x R), shift factor derived from batch index b. */
void launch_pitch_templates_batched(const float* Wo, float* W_all, int M, int K, int R, int P);
void launch_normalize_fixed_columns_batched(float* W_all, int M, int K, int R, int P);
/* P=1 variant that folds the removed norms into H (candidate NMF). */
void launch_normalize_columns(float* W, int M, int K, float* H, int N);

/* Scale the first K rows of each H[b] by 1/max over those rows (paper 3.2.1). */
void launch_max_normalize_batched(float* H, int R, int N, int K, int P);
void launch_znorm_batched(const float* H, int R, int N, int K, int P, float* Z);
void launch_distance_batched(const float* Zo, int No, const float* Zs, int Ns, int K, int P,
                             float* D);

/* Persistent banded subsequence-DTW: one block per (matrix, band) sweeps all
 * anti-diagonals in-kernel over the SKEWED distance matrices and emits one
 * float4 {min, mean, argmin, nvalid} of the slope-filtered, path-normalized
 * cost function per band. */
void launch_dtw_bands(int NoFull, int T, int Ns, int nmat, int nbands, int band_hop,
                      const float* Dskew, float4* stats);

/* Feature-mode variant: re-sweeps SELECTED bands (band_idx[f], all on the same
 * distance matrix) recording a predecessor byte per cell (0=path start,
 * 1=diagonal, 2=up, 3=left) plus the full last row {cost, len} — consumed by
 * host-side backtracking for the classifier's path features. */
void launch_dtw_band_preds(int NoFull, int T, int Ns, int nsel, const int* band_idx, int band_hop,
                           const float* Dskew, unsigned char* preds, float2* lastrow);

} /* namespace sd */
