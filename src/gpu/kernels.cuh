/* Custom kernels — declarations of the host-side launch wrappers.
 * All matrices row-major. "Batched" kernels operate on P stacked problems
 * (one per pitch-shift hypothesis) at fixed strides; matrix multiplies are
 * cuBLAS (gemm.cuh), everything else is custom. */

#pragma once

#include <cufft.h>

namespace sd {

void launch_window_frames(const float* x, int frames, float* out);
void launch_magnitude(const cufftComplex* cplx, int frames, float* V);

/* NMF elementwise/reduction steps over P stacked (M x N) problems; V shared. */
void launch_ratio_batched(const float* V, const float* WH, float* Z, size_t mn, int P);
void launch_col_sum_batched(const float* W, int M, int R, int P, float* out);
void launch_row_sum_batched(const float* H, int R, int N, int P, float* out);
void launch_update_h_batched(float* H, const float* numH, const float* wcol,
                             int R, int N, int P);
void launch_update_w_batched(float* W, const float* numW, const float* hrow,
                             int M, int R, int P, int n_fixed);

/* Writes the pitch-shifted copy of Wo (M x K) into the first K columns of each
 * W_all[b] (M x R), shift factor derived from batch index b. */
void launch_pitch_templates_batched(const float* Wo, float* W_all, int M, int K, int R, int P);
void launch_normalize_fixed_columns_batched(float* W_all, int M, int K, int R, int P);
/* P=1 variant that folds the removed norms into H (candidate NMF). */
void launch_normalize_columns(float* W, int M, int K, float* H, int N);

/* Scale the first K rows of each H[b] by 1/max over those rows (paper 3.2.1). */
void launch_max_normalize_batched(float* H, int R, int N, int K, int P);
void launch_znorm_batched(const float* H, int R, int N, int K, int P, float* Z);
void launch_distance_batched(const float* Zo, int No, const float* Zs, int Ns,
                             int K, int P, float* D);

/* Banded subsequence-DTW wavefront step over anti-diagonal d. Bands are windows
 * of T consecutive candidate frames at row (band0+slot)*band_hop inside each of
 * the nmat (NoFull x Ns) distance matrices; C/L are (nmat*nslots) stacked
 * (T x Ns) cost/path-length matrices. */
void launch_dtw_band_diag(int d, int NoFull, int T, int Ns, int nmat, int nslots,
                          int band0, int band_hop, const float* D, float* C, float* L);

}  /* namespace sd */
