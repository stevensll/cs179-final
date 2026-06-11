#include "cpu_pipeline.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>

namespace sd {

namespace {

/* ---------------- FFT ----------------
 * Iterative radix-2 Cooley-Tukey. No FFTW headers on the box, and the GPU side
 * uses cuFFT, so this exists purely as the CPU reference. */
void fft_radix2(std::vector<float>& re, std::vector<float>& im) {
    const size_t n = re.size();
    /* bit-reversal permutation */
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        float ang = -2.f * (float)M_PI / len;
        float wr = cosf(ang), wi = sinf(ang);
        for (size_t i = 0; i < n; i += len) {
            float cr = 1.f, ci = 0.f;
            for (size_t k = 0; k < len / 2; k++) {
                size_t a = i + k, b = i + k + len / 2;
                float tr = re[b] * cr - im[b] * ci;
                float ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
                float ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
}

/* Triangular log-spaced filterbank, identical construction to the GPU side
 * (gpu_pipeline.cu log_filterbank): centers MEL_FMIN * 2^(b/(12*BINS_PER_ST)),
 * triangles spanning neighboring centers, rows normalized to unit sum; rows
 * narrower than one FFT bin take the nearest bin. Built once per process. */
const std::vector<float>& log_filterbank() {
    static const std::vector<float> F = [] {
        std::vector<float> f((size_t)MEL_BINS * N_BINS, 0.f);
        auto center = [](int b) {
            return MEL_FMIN * exp2f((float)b / (12.f * BINS_PER_ST));
        };
        const float bin_hz = (float)SAMPLE_RATE / FFT_SIZE;
        for (int b = 0; b < MEL_BINS; b++) {
            const float lo = center(b - 1), mid = center(b), hi = center(b + 1);
            float sum = 0.f;
            int k0 = std::max(0, (int)(lo / bin_hz));
            int k1 = std::min(N_BINS - 1, (int)(hi / bin_hz) + 1);
            for (int k = k0; k <= k1; k++) {
                float fhz = k * bin_hz;
                float w = 0.f;
                if (fhz > lo && fhz <= mid) w = (fhz - lo) / (mid - lo);
                else if (fhz > mid && fhz < hi) w = (hi - fhz) / (hi - mid);
                f[(size_t)b * N_BINS + k] = w;
                sum += w;
            }
            if (sum > 0.f) {
                for (int k = k0; k <= k1; k++) f[(size_t)b * N_BINS + k] /= sum;
            } else {
                int k = (int)lrintf(mid / bin_hz);
                if (k >= 0 && k < N_BINS) f[(size_t)b * N_BINS + k] = 1.f;
            }
        }
        return f;
    }();
    return F;
}

/* ---------------- small GEMM helpers ----------------
 * Loop orders chosen so the innermost loop streams contiguous memory.
 * GPU strategy: the W*H+ratio product is one fused custom kernel (shared-mem
 * tiles, division epilogue); the large-k numerator GEMMs are cuBLAS
 * strided-batched TF32. */

/* C(MxN) = A(MxK) * B(KxN) */
void gemm_nn(const Mat& A, const Mat& B, Mat& C) {
    assert(A.cols == B.rows && C.rows == A.rows && C.cols == B.cols);
    std::fill(C.v.begin(), C.v.end(), 0.f);
    for (int i = 0; i < A.rows; i++)
        for (int k = 0; k < A.cols; k++) {
            const float a = A.at(i, k);
            const float* brow = &B.v[(size_t)k * B.cols];
            float* crow = &C.v[(size_t)i * C.cols];
            for (int j = 0; j < B.cols; j++) crow[j] += a * brow[j];
        }
}

/* C(RxN) = A(MxR)^T * B(MxN) */
void gemm_tn(const Mat& A, const Mat& B, Mat& C) {
    assert(A.rows == B.rows && C.rows == A.cols && C.cols == B.cols);
    std::fill(C.v.begin(), C.v.end(), 0.f);
    for (int m = 0; m < A.rows; m++)
        for (int r = 0; r < A.cols; r++) {
            const float a = A.at(m, r);
            const float* brow = &B.v[(size_t)m * B.cols];
            float* crow = &C.v[(size_t)r * C.cols];
            for (int j = 0; j < B.cols; j++) crow[j] += a * brow[j];
        }
}

/* C(MxR) = A(MxN) * B(RxN)^T */
void gemm_nt(const Mat& A, const Mat& B, Mat& C) {
    assert(A.cols == B.cols && C.rows == A.rows && C.cols == B.rows);
    for (int m = 0; m < A.rows; m++)
        for (int r = 0; r < B.rows; r++) {
            const float* arow = &A.v[(size_t)m * A.cols];
            const float* brow = &B.v[(size_t)r * B.cols];
            double acc = 0.0;
            for (int j = 0; j < A.cols; j++) acc += (double)arow[j] * brow[j];
            C.at(m, r) = (float)acc;
        }
}

/* Scale each column of W to unit L2 norm; if H is non-null, fold the removed
 * norm into row k of H (preserves W*H). Keeps activation scales comparable
 * between the candidate's Ho and the PFNMF Hs across all pitch shifts.
 * GPU strategy: one block per column, shared-memory norm reduction. */
void normalize_columns(Mat& W, Mat* H) {
    for (int k = 0; k < W.cols; k++) {
        float sq = 0.f;
        for (int m = 0; m < W.rows; m++) sq += W.at(m, k) * W.at(m, k);
        float norm = sqrtf(sq) + 1e-12f;
        for (int m = 0; m < W.rows; m++) W.at(m, k) /= norm;
        if (H)
            for (int j = 0; j < H->cols; j++) H->at(k, j) *= norm;
    }
}

/* Paper 3.2.1 H/max(H): max |H| over the first K rows, ALL rows rescaled so
 * the fixed-vs-free energy ratio (znorm's source attribution) is preserved.
 * Sets the scale at which Z_REG is meaningful.
 * GPU strategy: one block per problem, shared-memory max reduction. */
void max_normalize(Mat& H, int K) {
    float mx = 0.f;
    for (size_t i = 0; i < (size_t)K * H.cols; i++) mx = std::max(mx, fabsf(H.v[i]));
    float inv = 1.f / (mx + 1e-20f);
    for (float& x : H.v) x *= inv;
}

/* Z-normalize the first K rows of each column so a plain dot of two z-columns
 * equals (regularized) Pearson r. Z_REG shrinks near-silent frames toward a
 * neutral distance; rows K..R (the free mixture templates) supply the PFNMF
 * source attribution e = |H_fixed|/(|H_fixed|+|H_free|) — a frame the free
 * templates won is not explained by the sample. R == K leaves e = 1.
 * GPU strategy: one thread per column (znorm_batched_kernel). */
Mat znorm(const Mat& H, int K) {
    const int R = H.rows, N = H.cols;
    Mat Z(K, N);
    for (int j = 0; j < N; j++) {
        float mean = 0.f, fix2 = 0.f;
        for (int k = 0; k < K; k++) {
            float h = H.at(k, j);
            mean += h;
            fix2 += h * h;
        }
        mean /= K;
        float var = 0.f;
        for (int k = 0; k < K; k++) {
            float d = H.at(k, j) - mean;
            var += d * d;
        }
        float e = 1.f;
        if (R > K) {
            float free2 = 0.f;
            for (int k = K; k < R; k++) {
                float h = H.at(k, j);
                free2 += h * h;
            }
            e = sqrtf(fix2) / (sqrtf(fix2) + sqrtf(free2) + 1e-12f);
        }
        float scale = e / (sqrtf(var) + Z_REG);
        for (int k = 0; k < K; k++) Z.at(k, j) = (H.at(k, j) - mean) * scale;
    }
    return Z;
}

/* Pitch-shifted templates into the FIXED (first K) columns of the PFNMF W.
 * Log-frequency axis: shift = exact integer translation by s*BINS_PER_ST
 * bins; linear axis: Wp[m] = Wo[m / 2^(s/12)], linear interpolation.
 * GPU strategy: one thread per (problem, bin, template) gather. */
void pitch_templates_into(const Mat& Wo, Mat& W, int p) {
    const int M = Wo.rows, K = Wo.cols;
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            float v;
            if (MEL_BINS > 0) {
                int src = m - (int)lrintf(pitch_shift(p) * BINS_PER_ST);
                v = (src >= 0 && src < M) ? Wo.at(src, k) : 0.f;
            } else {
                float factor = exp2f(pitch_shift(p) / 12.f);
                float src = m / factor;
                int lo = (int)src;
                float frac = src - lo;
                v = (lo + 1 < M)
                        ? (1.f - frac) * Wo.at(lo, k) + frac * Wo.at(lo + 1, k)
                        : 0.f;
            }
            W.at(m, k) = v;
        }
    /* unit-norm the fixed columns only (no H to fold into; extreme shifts
     * otherwise shrink norms and bias the correlation distance) */
    for (int k = 0; k < K; k++) {
        float sq = 0.f;
        for (int m = 0; m < M; m++) sq += W.at(m, k) * W.at(m, k);
        float norm = sqrtf(sq) + 1e-12f;
        for (int m = 0; m < M; m++) W.at(m, k) /= norm;
    }
}

/* Banded subsequence DTW (paper eq. 2 boundary: band row 0 free along the
 * query, column 0 accumulates) over rows i0..i0+T-1 of D, rolling two rows.
 * Slope filter (paper 3.3.2: warp L/T in [0.7,1.5]) + min/mean/argmin of the
 * normalized last row reduce in place — same outputs as the GPU's float4.
 * GPU strategy: one block per (matrix, band) walks all anti-diagonals of a
 * diagonal-skewed D in shared memory (dtw_band_kernel). */
struct BandStats {
    float mn = 1e30f, mean = 0.f;
    int arg = 0;
};
BandStats dtw_band(const Mat& D, int i0, int T) {
    const int Ns = D.cols;
    std::vector<float> cprev(Ns), lprev(Ns), ccur(Ns), lcur(Ns);
    for (int j = 0; j < Ns; j++) { ccur[j] = D.at(i0, j); lcur[j] = 1.f; }
    for (int i = 1; i < T; i++) {
        std::swap(cprev, ccur);
        std::swap(lprev, lcur);
        const float* drow = &D.v[(size_t)(i0 + i) * Ns];
        ccur[0] = cprev[0] + drow[0];
        lcur[0] = (float)(i + 1);
        for (int j = 1; j < Ns; j++) {
            float best = cprev[j - 1], len = lprev[j - 1];      /* diagonal */
            if (cprev[j] < best) { best = cprev[j]; len = lprev[j]; }    /* up */
            if (ccur[j - 1] < best) { best = ccur[j - 1]; len = lcur[j - 1]; } /* left */
            ccur[j] = best + drow[j];
            lcur[j] = len + 1.f;
        }
    }
    BandStats s;
    float mean_sum = 0.f;
    int nvalid = 0;
    for (int j = 0; j < Ns; j++) {
        float warp = lcur[j] / (float)T;
        if (warp < 0.7f || warp > 1.5f) continue;
        float cost = ccur[j] / lcur[j];
        if (cost < s.mn) { s.mn = cost; s.arg = j; }
        mean_sum += cost;
        nvalid++;
    }
    if (nvalid == 0) { s.mn = 1.f; s.mean = 1.f; }
    else s.mean = mean_sum / nvalid;
    return s;
}

}  /* namespace */

Mat stft_magnitude(const std::vector<float>& x) {
    const int frames = x.size() >= FFT_SIZE ? 1 + (int)((x.size() - FFT_SIZE) / HOP) : 0;
    Mat V(N_BINS, frames);
    std::vector<float> hann(FFT_SIZE);
    for (int n = 0; n < FFT_SIZE; n++)
        hann[n] = 0.5f * (1.f - cosf(2.f * (float)M_PI * n / (FFT_SIZE - 1)));

    std::vector<float> re(FFT_SIZE), im(FFT_SIZE);
    for (int f = 0; f < frames; f++) {
        for (int n = 0; n < FFT_SIZE; n++) {
            re[n] = x[(size_t)f * HOP + n] * hann[n];
            im[n] = 0.f;
        }
        fft_radix2(re, im);
        for (int b = 0; b < N_BINS; b++)
            V.at(b, f) = sqrtf(re[b] * re[b] + im[b] * im[b]);
    }
    if (MEL_BINS <= 0) return V;

    /* log-frequency pooling Vm = F * V (the GPU does this as one GEMM) */
    const std::vector<float>& F = log_filterbank();
    Mat Vm(MEL_BINS, frames);
    for (int b = 0; b < MEL_BINS; b++)
        for (int k = 0; k < N_BINS; k++) {
            const float w = F[(size_t)b * N_BINS + k];
            if (w == 0.f) continue;
            const float* vrow = &V.v[(size_t)k * frames];
            float* orow = &Vm.v[(size_t)b * frames];
            for (int f = 0; f < frames; f++) orow[f] += w * vrow[f];
        }
    return Vm;
}

void nmf(const Mat& V, Mat& W, Mat& H, int iters, int n_fixed) {
    const int M = V.rows, N = V.cols, R = W.cols;
    assert(W.rows == M && H.rows == R && H.cols == N);
    Mat Z(M, N), numH(R, N), numW(M, R);
    std::vector<float> wcol(R), hrow(R);

    for (int it = 0; it < iters; it++) {
        /* one shared Z = V ./ (WH + eps); both numerators and both sums come
         * from the PRE-update W/H (simultaneous variant — the alternating
         * form's WH recompute was ~35% of GPU kernel time) */
        gemm_nn(W, H, Z);
        for (size_t i = 0; i < Z.v.size(); i++) Z.v[i] = V.v[i] / (Z.v[i] + NMF_EPS);
        gemm_tn(W, Z, numH);
        if (n_fixed < R) {
            gemm_nt(Z, H, numW);
            std::fill(hrow.begin(), hrow.end(), 0.f);
            for (int r = 0; r < R; r++)
                for (int j = 0; j < N; j++) hrow[r] += H.at(r, j);
        }
        std::fill(wcol.begin(), wcol.end(), 0.f);
        for (int m = 0; m < M; m++)
            for (int r = 0; r < R; r++) wcol[r] += W.at(m, r);

        for (int r = 0; r < R; r++) {
            const float inv = 1.f / (wcol[r] + NMF_EPS);
            for (int j = 0; j < N; j++) H.at(r, j) *= numH.at(r, j) * inv;
        }
        if (n_fixed < R)
            for (int m = 0; m < M; m++)
                for (int r = n_fixed; r < R; r++)
                    W.at(m, r) *= numW.at(m, r) / (hrow[r] + NMF_EPS);
    }
}

CpuCandidateTemplates candidate_templates(const Mat& Vc, int iters) {
    const int M = ANALYSIS_BINS, No = Vc.cols, K = RANK_K;
    CpuCandidateTemplates ct;
    ct.No = No;
    ct.Wo = Mat(M, K);
    Mat Ho(K, No);
    seed_matrix(ct.Wo, 42);
    seed_matrix(Ho, 43);
    nmf(Vc, ct.Wo, Ho, iters, 0);
    normalize_columns(ct.Wo, &Ho);
    max_normalize(Ho, K);   /* paper 3.2.1 */
    ct.Zo = znorm(Ho, K);   /* R == K -> e = 1 */
    return ct;
}

MatchInfo score_candidate(const Mat& Vq, const CpuCandidateTemplates& ct,
                          int iters, bool clip) {
    const int M = ANALYSIS_BINS, Ns = Vq.cols;
    const int K = RANK_K, R = RANK_K + RANK_L, P = N_SHIFTS, No = ct.No;
    const float fps = (float)SAMPLE_RATE / HOP;
    const int hop = std::max(1, (int)(WINDOW_HOP_SECONDS * fps));
    const int T = std::min(No, (int)(WINDOW_SECONDS[0] * fps));
    const int nbands = No > T ? (No - T) / hop + 1 : 1;

    std::vector<float> rmin((size_t)P * nbands), rmean((size_t)P * nbands);
    std::vector<int> rarg((size_t)P * nbands);

    for (int p = 0; p < P; p++) {
        /* PFNMF: [shifted Wo fixed | RANK_L free templates] against the query
         * (the GPU runs all P shifts as one stacked batch; same seeds) */
        Mat W(M, R), H(R, Ns);
        seed_matrix(W, 100 + p);
        seed_matrix(H, 200 + p);
        pitch_templates_into(ct.Wo, W, p);
        nmf(Vq, W, H, iters, K);

        max_normalize(H, K);
        Mat Zs = znorm(H, K);

        /* D(i,j) = 1 - regularized Pearson r (GPU: distance_batched_kernel,
         * written diagonal-skewed for the wavefront; plain layout here) */
        Mat D(No, Ns);
        for (int i = 0; i < No; i++)
            for (int j = 0; j < Ns; j++) {
                float r = 0.f;
                for (int k = 0; k < K; k++) r += ct.Zo.at(k, i) * Zs.at(k, j);
                D.at(i, j) = 1.f - r;
            }

        for (int w = 0; w < nbands; w++) {
            BandStats s = dtw_band(D, w * hop, T);
            size_t idx = (size_t)p * nbands + w;
            rmin[idx] = s.mn;
            rmean[idx] = s.mean;
            rarg[idx] = s.arg;
        }
    }

    /* ---- score (mirrors gpu score_candidate_from_H) ----
     * Raw dip s(p,w) = min/mean of the band's normalized DTW cost row (clip
     * mode: absolute min — a wall-to-wall clip has no non-sample baseline).
     * 1) Pitch selectivity: each window's dip judged against that window's
     *    median dip across all P shifts (real matches dip at ONE shift).
     * 2) Selection bias: the best hypothesis judged against the candidate's
     *    own hypothesis distribution (min/median over all (p,w)). */
    std::vector<float> s((size_t)P * nbands);
    for (size_t i = 0; i < s.size(); i++)
        s[i] = clip ? rmin[i] : rmin[i] / (rmean[i] + 1e-12f);

    MatchInfo best;
    std::vector<float> col(P), sel((size_t)P * nbands);
    for (int w = 0; w < nbands; w++) {
        for (int p = 0; p < P; p++) col[p] = s[(size_t)p * nbands + w];
        std::nth_element(col.begin(), col.begin() + P / 2, col.end());
        float med = col[P / 2];
        for (int p = 0; p < P; p++) {
            size_t it = (size_t)p * nbands + w;
            sel[it] = s[it] / (med + 1e-12f);
            if (sel[it] < best.score) {
                best.score = sel[it];
                best.shift = pitch_shift(p);
                best.cand_seconds = w * hop / fps;
                best.query_seconds = rarg[it] / fps;
            }
        }
    }
    std::nth_element(sel.begin(), sel.begin() + sel.size() / 2, sel.end());
    best.score /= (sel[sel.size() / 2] + 1e-12f);
    return best;
}

void seed_matrix(Mat& m, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.01f, 1.f);
    for (float& x : m.v) x = u(rng);
}

}  /* namespace sd */
