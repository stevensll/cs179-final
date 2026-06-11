#include "kernels.cuh"

#include <cmath>

#include "../common/params.hpp"
#include "error_check.cuh"

namespace sd {

namespace {

constexpr int BLOCK = 256;

inline int grid_1d(size_t n) { return (int)((n + BLOCK - 1) / BLOCK); }

/* Always check launches: launch errors are silent otherwise. */
inline void post_launch() { checkCuda(cudaGetLastError()); }

/* One thread per windowed output sample. Frames overlap 4x, so this is a
 * gather from x with a Hann weight computed inline (cheaper than staging a
 * window table: the cosf is hidden behind the global loads). */
__global__ void window_frames_kernel(const float* x, int frames, float* out) {
    size_t total = (size_t)frames * FFT_SIZE;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)blockDim.x * gridDim.x) {
        int f = (int)(idx / FFT_SIZE);
        int n = (int)(idx % FFT_SIZE);
        float w = 0.5f * (1.f - cosf(2.f * (float)M_PI * n / (FFT_SIZE - 1)));
        out[idx] = x[(size_t)f * HOP + n] * w;
    }
}

/* One thread per (frame, bin); writes the transposed (bins x frames) layout the
 * NMF stage wants, so the transpose is free here instead of a separate pass. */
__global__ void magnitude_kernel(const cufftComplex* c, int frames, float* V) {
    size_t total = (size_t)frames * N_BINS;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)blockDim.x * gridDim.x) {
        int f = (int)(idx / N_BINS);
        int b = (int)(idx % N_BINS);
        cufftComplex z = c[idx];
        V[(size_t)b * frames + f] = sqrtf(z.x * z.x + z.y * z.y);
    }
}

/* Z[b] = V ./ (WH[b] + eps), V shared across the P stacked problems. */
__global__ void ratio_batched_kernel(const float* V, const float* WH, float* Z,
                                     size_t mn, size_t total) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)blockDim.x * gridDim.x)
        Z[i] = V[i % mn] / (WH[i] + NMF_EPS);
}

/* One block per (column r, problem b); classic two-stage shared-memory
 * reduction (lecture 7): grid-stride partial sums, then tree reduce. */
__global__ void col_sum_batched_kernel(const float* W, int M, int R, float* out) {
    __shared__ float sh[BLOCK];
    int r = blockIdx.x, b = blockIdx.y;
    const float* Wb = W + (size_t)b * M * R;
    float acc = 0.f;
    for (int m = threadIdx.x; m < M; m += blockDim.x) acc += Wb[(size_t)m * R + r];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[(size_t)b * R + r] = sh[0];
}

/* One block per (row r, problem b); same reduction shape, contiguous reads. */
__global__ void row_sum_batched_kernel(const float* H, int R, int N, float* out) {
    __shared__ float sh[BLOCK];
    int r = blockIdx.x, b = blockIdx.y;
    const float* Hb = H + (size_t)b * R * N;
    float acc = 0.f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) acc += Hb[(size_t)r * N + j];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[(size_t)b * R + r] = sh[0];
}

__global__ void update_h_batched_kernel(float* H, const float* numH, const float* wcol,
                                        int R, int N, size_t total) {
    size_t rn = (size_t)R * N;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)blockDim.x * gridDim.x) {
        int b = (int)(i / rn);
        int r = (int)((i % rn) / N);
        H[i] *= numH[i] / (wcol[(size_t)b * R + r] + NMF_EPS);
    }
}

/* PFNMF freezing = skip columns r < n_fixed (the candidate's templates). */
__global__ void update_w_batched_kernel(float* W, const float* numW, const float* hrow,
                                        int M, int R, int n_fixed, size_t total) {
    size_t mr = (size_t)M * R;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)blockDim.x * gridDim.x) {
        int b = (int)(i / mr);
        int r = (int)(i % R);
        if (r >= n_fixed)
            W[i] *= numW[i] / (hrow[(size_t)b * R + r] + NMF_EPS);
    }
}

/* Frequency-axis rescale: Wp[b'][m] = Wo[m / factor(b')], linear interp; one
 * thread per (problem, bin, template) gather, written into the fixed (first K)
 * columns of W_all[b']. No library expresses this — custom by necessity. */
__global__ void pitch_templates_batched_kernel(const float* Wo, float* W_all,
                                               int M, int K, int R, int P) {
    size_t total = (size_t)P * M * K;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)blockDim.x * gridDim.x) {
        int b = (int)(i / ((size_t)M * K));
        int m = (int)((i / K) % M);
        int k = (int)(i % K);
        float factor = exp2f(pitch_shift(b) / 12.f);
        float src = m / factor;
        int lo = (int)src;
        float frac = src - lo;
        float v = (lo + 1 < M)
                      ? (1.f - frac) * Wo[(size_t)lo * K + k] + frac * Wo[(size_t)(lo + 1) * K + k]
                      : 0.f;
        W_all[(size_t)b * M * R + (size_t)m * R + k] = v;
    }
}

/* One block per (fixed column k, problem b): L2-norm reduce, then scale the
 * column to unit norm. Keeps fixed-vs-free template competition and activation
 * scales comparable across pitch shifts — extreme shifts otherwise shrink
 * column norms and bias the correlation distance. */
__global__ void normalize_fixed_columns_batched_kernel(float* W_all, int M, int K, int R) {
    __shared__ float sh[BLOCK];
    int k = blockIdx.x, b = blockIdx.y;
    float* Wb = W_all + (size_t)b * M * R;
    float acc = 0.f;
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        float w = Wb[(size_t)m * R + k];
        acc += w * w;
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float norm = sqrtf(sh[0]) + 1e-12f;
    for (int m = threadIdx.x; m < M; m += blockDim.x) Wb[(size_t)m * R + k] /= norm;
}

/* P=1 candidate-NMF variant: also folds the removed norm into row k of H so
 * W*H is preserved. */
__global__ void normalize_columns_kernel(float* W, int M, int K, float* H, int N) {
    __shared__ float sh[BLOCK];
    int k = blockIdx.x;
    float acc = 0.f;
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        float w = W[(size_t)m * K + k];
        acc += w * w;
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float norm = sqrtf(sh[0]) + 1e-12f;
    for (int m = threadIdx.x; m < M; m += blockDim.x) W[(size_t)m * K + k] /= norm;
    if (H)
        for (int j = threadIdx.x; j < N; j += blockDim.x) H[(size_t)k * N + j] *= norm;
}

/* One block per problem: classic two-stage max-reduction over the first K rows
 * of H[b] (lecture 7), then all threads rescale ALL R rows by 1/max. This is
 * the paper's H/max(H) normalization (3.2.1); it sets the scale at which the
 * Z_REG regularizer in znorm is meaningful. Scaling every row (not just the
 * fixed K) preserves the fixed-vs-free energy ratio that znorm's source
 * attribution uses. */
__global__ void max_normalize_batched_kernel(float* H, int R, int N, int K) {
    __shared__ float sh[BLOCK];
    float* Hb = H + (size_t)blockIdx.x * R * N;
    size_t kn = (size_t)K * N, rn = (size_t)R * N;
    float mx = 0.f;
    for (size_t i = threadIdx.x; i < kn; i += blockDim.x) mx = fmaxf(mx, fabsf(Hb[i]));
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    float inv = 1.f / (sh[0] + 1e-20f);
    for (size_t i = threadIdx.x; i < rn; i += blockDim.x) Hb[i] *= inv;
}

/* One thread per (problem, column): z-normalize the first K rows so a plain dot
 * of two z-columns equals (regularized) Pearson r. Two attenuations fold into
 * the z-vectors so the downstream dot product carries them for free:
 * - Z_REG in the denominator shrinks near-silent frames toward zero instead of
 *   amplifying their noise to a unit vector (degenerate quiet bands otherwise
 *   produce spurious DTW dips);
 * - frames are scaled by the PFNMF source attribution e = |H_fixed|/|H_total|
 *   (rows K..R are the free mixture templates): the paper's premise is that
 *   fixed-template activations indicate sample PRESENCE — a frame whose energy
 *   the free templates won is not explained by the sample and contributes a
 *   neutral distance. R == K (candidate side) leaves e = 1.
 * Adjacent threads hit adjacent columns -> coalesced. */
__global__ void znorm_batched_kernel(const float* H, int R, int N, int K, int P, float* Z) {
    size_t total = (size_t)P * N;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)blockDim.x * gridDim.x) {
        int b = (int)(i / N);
        int j = (int)(i % N);
        const float* Hb = H + (size_t)b * R * N;
        float* Zb = Z + (size_t)b * K * N;
        float mean = 0.f, fix2 = 0.f;
        for (int k = 0; k < K; k++) {
            float h = Hb[(size_t)k * N + j];
            mean += h;
            fix2 += h * h;
        }
        mean /= K;
        float var = 0.f;
        for (int k = 0; k < K; k++) {
            float d = Hb[(size_t)k * N + j] - mean;
            var += d * d;
        }
        float e = 1.f;
        if (R > K) {
            float free2 = 0.f;
            for (int k = K; k < R; k++) {
                float h = Hb[(size_t)k * N + j];
                free2 += h * h;
            }
            e = sqrtf(fix2) / (sqrtf(fix2) + sqrtf(free2) + 1e-12f);
        }
        float scale = e / (sqrtf(var) + Z_REG);
        for (int k = 0; k < K; k++)
            Zb[(size_t)k * N + j] = (Hb[(size_t)k * N + j] - mean) * scale;
    }
}

/* One thread per D(i,j) cell, one z-block per problem; K-dim inner product over
 * z-normalized columns. Adjacent threads share i and walk adjacent j -> Zs reads
 * coalesce, Zo reads broadcast. Zo is shared across problems. */
__global__ void distance_batched_kernel(const float* Zo, int No, const float* Zs,
                                        int Ns, int K, float* D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    int b = blockIdx.z;
    if (j >= Ns) return;
    const float* Zb = Zs + (size_t)b * K * Ns;
    float r = 0.f;
    for (int k = 0; k < K; k++) r += Zo[(size_t)k * No + i] * Zb[(size_t)k * Ns + j];
    D[(size_t)b * No * Ns + (size_t)i * Ns + j] = 1.f - r;
}

/* Banded subsequence-DTW wavefront step: all cells on anti-diagonal d of a band
 * are independent (each depends only on diagonals d-1, d-2), so one launch per
 * diagonal with one thread per cell; blockIdx.y enumerates (matrix, band-slot)
 * pairs. A band treats T consecutive candidate frames (rows i0..i0+T-1 of D) as
 * the sample hypothesis — the dense window search lives here, where bands reuse
 * one distance matrix, instead of one NMF per window. Boundary is paper eq. 2:
 * band row 0 free (C = D), column 0 accumulates. L tracks path length for the
 * final normalization. */
__global__ void dtw_band_diag_kernel(int d, int NoFull, int T, int Ns, int nslots,
                                     int band0, int band_hop,
                                     const float* D, float* C, float* L) {
    int ilo = max(0, d - (Ns - 1));
    int ihi = min(T - 1, d);
    int i = ilo + blockIdx.x * blockDim.x + threadIdx.x;
    if (i > ihi) return;
    int j = d - i;
    int m = blockIdx.y / nslots;             /* which distance matrix */
    int s = blockIdx.y % nslots;             /* band slot within this chunk */
    int i0 = (band0 + s) * band_hop;         /* band's first candidate frame */
    size_t didx = (size_t)m * NoFull * Ns + (size_t)(i0 + i) * Ns + j;
    size_t base = (size_t)blockIdx.y * T * Ns;
    size_t idx = base + (size_t)i * Ns + j;
    float dij = D[didx];
    if (i == 0) { C[idx] = dij; L[idx] = 1.f; return; }
    if (j == 0) { C[idx] = C[idx - Ns] + dij; L[idx] = (float)(i + 1); return; }
    float best = C[idx - Ns - 1], len = L[idx - Ns - 1];   /* diagonal pred */
    if (C[idx - Ns] < best) { best = C[idx - Ns]; len = L[idx - Ns]; }
    if (C[idx - 1] < best) { best = C[idx - 1]; len = L[idx - 1]; }
    C[idx] = best + dij;
    L[idx] = len + 1.f;
}

}  /* namespace */

void launch_window_frames(const float* x, int frames, float* out) {
    window_frames_kernel<<<grid_1d((size_t)frames * FFT_SIZE), BLOCK>>>(x, frames, out);
    post_launch();
}

void launch_magnitude(const cufftComplex* cplx, int frames, float* V) {
    magnitude_kernel<<<grid_1d((size_t)frames * N_BINS), BLOCK>>>(cplx, frames, V);
    post_launch();
}

void launch_ratio_batched(const float* V, const float* WH, float* Z, size_t mn, int P) {
    size_t total = mn * P;
    ratio_batched_kernel<<<grid_1d(total), BLOCK>>>(V, WH, Z, mn, total);
    post_launch();
}

void launch_col_sum_batched(const float* W, int M, int R, int P, float* out) {
    col_sum_batched_kernel<<<dim3(R, P), BLOCK>>>(W, M, R, out);
    post_launch();
}

void launch_row_sum_batched(const float* H, int R, int N, int P, float* out) {
    row_sum_batched_kernel<<<dim3(R, P), BLOCK>>>(H, R, N, out);
    post_launch();
}

void launch_update_h_batched(float* H, const float* numH, const float* wcol,
                             int R, int N, int P) {
    size_t total = (size_t)P * R * N;
    update_h_batched_kernel<<<grid_1d(total), BLOCK>>>(H, numH, wcol, R, N, total);
    post_launch();
}

void launch_update_w_batched(float* W, const float* numW, const float* hrow,
                             int M, int R, int P, int n_fixed) {
    size_t total = (size_t)P * M * R;
    update_w_batched_kernel<<<grid_1d(total), BLOCK>>>(W, numW, hrow, M, R, n_fixed, total);
    post_launch();
}

void launch_pitch_templates_batched(const float* Wo, float* W_all, int M, int K, int R, int P) {
    pitch_templates_batched_kernel<<<grid_1d((size_t)P * M * K), BLOCK>>>(Wo, W_all, M, K, R, P);
    post_launch();
}

void launch_normalize_fixed_columns_batched(float* W_all, int M, int K, int R, int P) {
    normalize_fixed_columns_batched_kernel<<<dim3(K, P), BLOCK>>>(W_all, M, K, R);
    post_launch();
}

void launch_normalize_columns(float* W, int M, int K, float* H, int N) {
    normalize_columns_kernel<<<K, BLOCK>>>(W, M, K, H, N);
    post_launch();
}

void launch_max_normalize_batched(float* H, int R, int N, int K, int P) {
    max_normalize_batched_kernel<<<P, BLOCK>>>(H, R, N, K);
    post_launch();
}

void launch_znorm_batched(const float* H, int R, int N, int K, int P, float* Z) {
    znorm_batched_kernel<<<grid_1d((size_t)P * N), BLOCK>>>(H, R, N, K, P, Z);
    post_launch();
}

void launch_distance_batched(const float* Zo, int No, const float* Zs, int Ns,
                             int K, int P, float* D) {
    dim3 grid((Ns + BLOCK - 1) / BLOCK, No, P);
    distance_batched_kernel<<<grid, BLOCK>>>(Zo, No, Zs, Ns, K, D);
    post_launch();
}

void launch_dtw_band_diag(int d, int NoFull, int T, int Ns, int nmat, int nslots,
                          int band0, int band_hop, const float* D, float* C, float* L) {
    int cells = T < Ns ? T : Ns;  /* upper bound on diagonal length */
    dim3 grid((cells + BLOCK - 1) / BLOCK, nmat * nslots);
    dtw_band_diag_kernel<<<grid, BLOCK>>>(d, NoFull, T, Ns, nslots, band0, band_hop, D, C, L);
    post_launch();
}

}  /* namespace sd */
