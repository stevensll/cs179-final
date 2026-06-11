#include "kernels.cuh"

#include <algorithm>
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

/* Fused W*H + ratio epilogue over P stacked problems sharing V:
 *   Z[b] = V ./ (W[b]*H[b] + eps)   in ONE kernel.
 *
 * 1) The NMF inner dimension (R <= 64) fits a single shared-memory tile, so
 *    there is no k-loop over tiles: one cooperative load of a 64-row W slab
 *    and a 64-column H slab, then each thread accumulates a 4x4 micro-tile
 *    in registers (16x16 threads -> 64x64 output tile per block).
 * 2) This is exactly the skinny-k shape where cuBLAS underperforms (heuristic
 *    keeps it off tensor cores, ~430 GFLOPS measured, plots/01_tf32_dtw) —
 *    and fusing the division means WH, the largest per-iteration
 *    intermediate (~1.7 GB across 41 shifts), is never written or re-read:
 *    two full memory passes per NMF iteration eliminated.
 * 3) V loads and Z stores are coalesced along N; W/H tile loads amortize over
 *    the 64x64 outputs. Padding column on sW avoids bank conflicts. */
/* 64x64 block tile, 4x4 per thread (Boehm's "2D blocktiling" shape; the
 * 128x64 variant halving H-tile re-reads was tried and measured NEUTRAL —
 * each problem's H slab already lives in L2 between block sweeps). */
constexpr int FUSE_BM = 64, FUSE_BN = 64, FUSE_MAX_R = 60;
constexpr int FUSE_TM = 4, FUSE_TN = 4;

__global__ void fused_wh_ratio_kernel(const float* __restrict__ W,
                                      const float* __restrict__ H,
                                      const float* __restrict__ V,
                                      float* __restrict__ Z,
                                      int M, int N, int R) {
    __shared__ float sW[FUSE_BM][FUSE_MAX_R + 1];
    __shared__ float sH[FUSE_MAX_R][FUSE_BN];
    const int b = blockIdx.z;
    const float* Wb = W + (size_t)b * M * R;
    const float* Hb = H + (size_t)b * R * N;
    float* Zb = Z + (size_t)b * (size_t)M * N;
    const int m0 = blockIdx.y * FUSE_BM, n0 = blockIdx.x * FUSE_BN;
    const int tid = threadIdx.y * 16 + threadIdx.x;

    for (int idx = tid; idx < FUSE_BM * R; idx += 256) {
        int m = idx / R, r = idx % R;
        sW[m][r] = (m0 + m < M) ? Wb[(size_t)(m0 + m) * R + r] : 0.f;
    }
    for (int idx = tid; idx < R * FUSE_BN; idx += 256) {
        int r = idx / FUSE_BN, n = idx % FUSE_BN;
        sH[r][n] = (n0 + n < N) ? Hb[(size_t)r * N + (n0 + n)] : 0.f;
    }
    __syncthreads();

    float acc[FUSE_TM][FUSE_TN] = {};
    for (int k = 0; k < R; k++) {
        float wv[FUSE_TM], hv[FUSE_TN];
#pragma unroll
        for (int i = 0; i < FUSE_TM; i++) wv[i] = sW[threadIdx.y * FUSE_TM + i][k];
#pragma unroll
        for (int j = 0; j < FUSE_TN; j++) hv[j] = sH[k][threadIdx.x * FUSE_TN + j];
#pragma unroll
        for (int i = 0; i < FUSE_TM; i++)
#pragma unroll
            for (int j = 0; j < FUSE_TN; j++) acc[i][j] += wv[i] * hv[j];
    }

#pragma unroll
    for (int i = 0; i < FUSE_TM; i++) {
        int m = m0 + threadIdx.y * FUSE_TM + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < FUSE_TN; j++) {
            int n = n0 + threadIdx.x * FUSE_TN + j;
            if (n >= N) continue;
            size_t idx = (size_t)m * N + n;
            Zb[idx] = V[idx] / (acc[i][j] + NMF_EPS);
        }
    }
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

/* One block row (blockIdx.y) per (problem, H row): the divisor wcol[b*R+r] is
 * one scalar per block, rows are contiguous -> fully coalesced, no div/mod. */
__global__ void update_h_batched_kernel(float* __restrict__ H,
                                        const float* __restrict__ numH,
                                        const float* __restrict__ wcol, int N) {
    size_t row = blockIdx.y;          /* row = b * R + r */
    float* h = H + row * N;
    const float* nh = numH + row * N;
    float inv = 1.f / (wcol[row] + NMF_EPS);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x)
        h[i] *= nh[i] * inv;
}

/* PFNMF freezing = skip columns r < n_fixed (the candidate's templates).
 * blockIdx.y selects the problem; W is small (M*R), so the r = i % R for the
 * freeze mask is not on a hot path. */
__global__ void update_w_batched_kernel(float* __restrict__ W,
                                        const float* __restrict__ numW,
                                        const float* __restrict__ hrow,
                                        int M, int R, int n_fixed) {
    size_t mr = (size_t)M * R;
    float* wb = W + blockIdx.y * mr;
    const float* nb = numW + blockIdx.y * mr;
    const float* hb = hrow + (size_t)blockIdx.y * R;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < mr;
         i += (size_t)blockDim.x * gridDim.x) {
        int r = (int)(i % R);
        if (r >= n_fixed)
            wb[i] *= nb[i] / (hb[r] + NMF_EPS);
    }
}

/* Pitch-shifted template generation, one thread per (problem, bin, template),
 * written into the fixed (first K) columns of W_all[b']. No library expresses
 * this — custom by necessity. Two compile-time paths:
 * - log-frequency front end (MEL_BINS > 0): a shift of s semitones is an
 *   EXACT integer translation by s*BINS_PER_ST bins (the 0.25 st grid maps to
 *   whole bins at 4 bins/st) — no interpolation loss at all;
 * - linear axis: Wp[m] = Wo[m / 2^(s/12)], linear interpolation. */
__global__ void pitch_templates_batched_kernel(const float* Wo, float* W_all,
                                               int M, int K, int R, int P) {
    size_t total = (size_t)P * M * K;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)blockDim.x * gridDim.x) {
        int b = (int)(i / ((size_t)M * K));
        int m = (int)((i / K) % M);
        int k = (int)(i % K);
        float v;
        if (MEL_BINS > 0) {
            int src = m - (int)lrintf(pitch_shift(b) * BINS_PER_ST);
            v = (src >= 0 && src < M) ? Wo[(size_t)src * K + k] : 0.f;
        } else {
            float factor = exp2f(pitch_shift(b) / 12.f);
            float src = m / factor;
            int lo = (int)src;
            float frac = src - lo;
            v = (lo + 1 < M)
                    ? (1.f - frac) * Wo[(size_t)lo * K + k] + frac * Wo[(size_t)(lo + 1) * K + k]
                    : 0.f;
        }
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
 * coalesce, Zo reads broadcast. Zo is shared across problems.
 *
 * Output is written in DIAGONAL-SKEWED layout: cell (i,j) lands at
 * [(i+j)*No + i], so each anti-diagonal g = i+j is contiguous in memory and
 * the DTW wavefront kernel reads it fully coalesced (anti-diagonals are
 * contiguous in no conventional layout). The scattered writes here cost one
 * pass; the wavefront reads it T-deep per band. Skewed size: No*(No+Ns-1). */
__global__ void distance_batched_kernel(const float* __restrict__ Zo, int No,
                                        const float* __restrict__ Zs, int Ns,
                                        int K, float* __restrict__ Dskew) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    int b = blockIdx.z;
    if (j >= Ns) return;
    const float* Zb = Zs + (size_t)b * K * Ns;
    float r = 0.f;
    for (int k = 0; k < K; k++) r += Zo[(size_t)k * No + i] * Zb[(size_t)k * Ns + j];
    size_t mat = (size_t)No * (No + Ns - 1);
    Dskew[(size_t)b * mat + (size_t)(i + j) * No + i] = 1.f - r;
}

/* Persistent banded subsequence-DTW: ONE block per (matrix, band) walks all
 * T+Ns-1 anti-diagonals in-kernel, replacing the old one-launch-per-diagonal
 * scheme (~325k launches per candidate, 6.8 us avg — profiled as the top
 * bottleneck at 33.6% of wall, plots/00_baseline).
 *
 * 1) Thread t owns band row i = t for the whole sweep; the recurrence needs
 *    only the previous two diagonals, kept in shared memory (rotating
 *    triple-buffer for C and L) — the cost/path-length matrices never touch
 *    global memory at all (was ~80 GB of traffic + 2.3 GB of workspace).
 * 2) D is read in the skewed layout written by distance_batched_kernel: band
 *    diagonal d lies on global anti-diagonal g = i0 + d, rows i0..i0+T-1 of
 *    which are contiguous -> one fully coalesced read per cell.
 * 3) Band row T-1 (the DTW cost function) is always computed by thread T-1,
 *    so the slope filter (paper 3.3.2: warp L/T in [0.7,1.5]) and the
 *    min/mean/argmin scoring reduce in that thread's registers; each band
 *    outputs 4 floats {min, mean, argmin, nvalid} instead of a T x Ns matrix
 *    slab (kills the 600 MB D2H of last rows).
 * Boundary is paper eq. 2: band row 0 free (C = D), column 0 accumulates. */
constexpr int DTW_MAX_T = 128;

__global__ void dtw_band_kernel(int NoFull, int T, int Ns, int band_hop,
                                const float* __restrict__ Dskew,
                                float4* __restrict__ stats) {
    __shared__ float c0[DTW_MAX_T], c1[DTW_MAX_T], c2[DTW_MAX_T];
    __shared__ float l0[DTW_MAX_T], l1[DTW_MAX_T], l2[DTW_MAX_T];
    float* cbuf[3] = {c0, c1, c2};
    float* lbuf[3] = {l0, l1, l2};

    const int w = blockIdx.x;                /* band index */
    const int m = blockIdx.y;                /* distance matrix (pitch shift) */
    const int i0 = w * band_hop;             /* band's first candidate row */
    const int i = threadIdx.x;               /* this thread's band row */
    const size_t mat = (size_t)NoFull * (NoFull + Ns - 1);
    const float* Dm = Dskew + (size_t)m * mat;

    /* thread T-1 accumulates the band's score over valid end points */
    float mn = 1e30f, mean_sum = 0.f;
    int arg = 0, nvalid = 0;

    for (int d = 0; d < T + Ns - 1; d++) {
        float* cur = cbuf[d % 3];
        float* lcur = lbuf[d % 3];
        const float* p1 = cbuf[(d + 2) % 3];   /* diagonal d-1 */
        const float* lp1 = lbuf[(d + 2) % 3];
        const float* p2 = cbuf[(d + 1) % 3];   /* diagonal d-2 */
        const float* lp2 = lbuf[(d + 1) % 3];

        const int j = d - i;
        if (i < T && j >= 0 && j < Ns) {
            /* coalesced: global anti-diagonal i0+d, consecutive rows i0+i */
            float dij = Dm[(size_t)(i0 + d) * NoFull + (i0 + i)];
            float c, len;
            if (i == 0) {
                c = dij;                       /* free start along the query */
                len = 1.f;
            } else if (j == 0) {
                c = p1[i - 1] + dij;           /* column 0 accumulates */
                len = (float)(i + 1);
            } else {
                float best = p2[i - 1], l = lp2[i - 1];      /* diagonal */
                if (p1[i - 1] < best) { best = p1[i - 1]; l = lp1[i - 1]; }
                if (p1[i] < best) { best = p1[i]; l = lp1[i]; }
                c = best + dij;
                len = l + 1.f;
            }
            cur[i] = c;
            lcur[i] = len;
            if (i == T - 1) {                  /* DTW cost function row */
                float warp = len / (float)T;
                if (warp >= 0.7f && warp <= 1.5f) {
                    float cost = c / len;
                    if (cost < mn) { mn = cost; arg = j; }
                    mean_sum += cost;
                    nvalid++;
                }
            }
        }
        __syncthreads();
    }
    if (i == T - 1) {
        if (nvalid == 0) { mn = 1.f; mean_sum = 1.f; nvalid = 1; }
        stats[(size_t)m * gridDim.x + w] =
            make_float4(mn, mean_sum / nvalid, (float)arg, (float)nvalid);
    }
}

/* Feature-mode DTW sweep over SELECTED bands of ONE distance matrix: same
 * recurrence as dtw_band_kernel, but records a predecessor byte per cell
 * (host backtracking reconstructs alignment paths for the classifier's
 * path-geometry features) and the full {cost, len} last row (host finds the
 * local minima = occurrence end points). One block per selected band; runs
 * on ~F<<nbands hypotheses, so the T*Ns byte map per band stays small. */
__global__ void dtw_band_preds_kernel(int NoFull, int T, int Ns,
                                      const int* __restrict__ band_idx, int band_hop,
                                      const float* __restrict__ Dskew,
                                      unsigned char* __restrict__ preds,
                                      float2* __restrict__ lastrow) {
    __shared__ float c0[DTW_MAX_T], c1[DTW_MAX_T], c2[DTW_MAX_T];
    __shared__ float l0[DTW_MAX_T], l1[DTW_MAX_T], l2[DTW_MAX_T];
    float* cbuf[3] = {c0, c1, c2};
    float* lbuf[3] = {l0, l1, l2};

    const int f = blockIdx.x;
    const int i0 = band_idx[f] * band_hop;
    const int i = threadIdx.x;
    unsigned char* pf = preds + (size_t)f * T * Ns;
    float2* lr = lastrow + (size_t)f * Ns;

    for (int d = 0; d < T + Ns - 1; d++) {
        float* cur = cbuf[d % 3];
        float* lcur = lbuf[d % 3];
        const float* p1 = cbuf[(d + 2) % 3];
        const float* lp1 = lbuf[(d + 2) % 3];
        const float* p2 = cbuf[(d + 1) % 3];
        const float* lp2 = lbuf[(d + 1) % 3];

        const int j = d - i;
        if (i < T && j >= 0 && j < Ns) {
            float dij = Dskew[(size_t)(i0 + d) * NoFull + (i0 + i)];
            float c, len;
            unsigned char pred;
            if (i == 0) {
                c = dij; len = 1.f; pred = 0;            /* free start */
            } else if (j == 0) {
                c = p1[i - 1] + dij; len = (float)(i + 1); pred = 2;  /* up */
            } else {
                float best = p2[i - 1], l = lp2[i - 1];
                pred = 1;                                 /* diagonal */
                if (p1[i - 1] < best) { best = p1[i - 1]; l = lp1[i - 1]; pred = 2; }
                if (p1[i] < best) { best = p1[i]; l = lp1[i]; pred = 3; }
                c = best + dij;
                len = l + 1.f;
            }
            cur[i] = c;
            lcur[i] = len;
            pf[(size_t)i * Ns + j] = pred;
            if (i == T - 1) lr[j] = make_float2(c, len);
        }
        __syncthreads();
    }
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

void launch_fused_wh_ratio(const float* W, const float* H, const float* V,
                           float* Z, int M, int N, int R, int P) {
    if (R > FUSE_MAX_R) {
        std::fprintf(stderr, "fused_wh_ratio: R=%d exceeds FUSE_MAX_R=%d\n", R, FUSE_MAX_R);
        std::exit(1);
    }
    dim3 grid((N + FUSE_BN - 1) / FUSE_BN, (M + FUSE_BM - 1) / FUSE_BM, P);
    fused_wh_ratio_kernel<<<grid, dim3(16, 16)>>>(W, H, V, Z, M, N, R);
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
    dim3 grid((N + BLOCK - 1) / BLOCK, P * R);  /* one grid row per (b, r) */
    update_h_batched_kernel<<<grid, BLOCK>>>(H, numH, wcol, N);
    post_launch();
}

void launch_update_w_batched(float* W, const float* numW, const float* hrow,
                             int M, int R, int P, int n_fixed) {
    size_t mr = (size_t)M * R;
    dim3 grid(grid_1d(mr), P);
    update_w_batched_kernel<<<grid, BLOCK>>>(W, numW, hrow, M, R, n_fixed);
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

void launch_dtw_band_preds(int NoFull, int T, int Ns, int nsel,
                           const int* band_idx, int band_hop, const float* Dskew,
                           unsigned char* preds, float2* lastrow) {
    if (T > DTW_MAX_T) {
        std::fprintf(stderr, "dtw_band_preds: T=%d exceeds DTW_MAX_T=%d\n", T, DTW_MAX_T);
        std::exit(1);
    }
    int threads = ((T + 31) / 32) * 32;
    dtw_band_preds_kernel<<<nsel, threads>>>(NoFull, T, Ns, band_idx, band_hop,
                                             Dskew, preds, lastrow);
    post_launch();
}

void launch_dtw_bands(int NoFull, int T, int Ns, int nmat, int nbands,
                      int band_hop, const float* Dskew, float4* stats) {
    if (T > DTW_MAX_T) {
        std::fprintf(stderr, "dtw_band_kernel: T=%d exceeds DTW_MAX_T=%d\n", T, DTW_MAX_T);
        std::exit(1);
    }
    int threads = ((T + 31) / 32) * 32;   /* whole warps covering the band */
    dtw_band_kernel<<<dim3(nbands, nmat), threads>>>(NoFull, T, Ns, band_hop, Dskew, stats);
    post_launch();
}

}  /* namespace sd */
