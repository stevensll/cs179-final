#include "cpu_pipeline.hpp"

#include <cassert>
#include <cmath>
#include <random>

namespace sd {

/* ---------------- FFT ----------------
 * Iterative radix-2 Cooley-Tukey. No FFTW headers on the box, and the GPU side
 * uses cuFFT, so this exists purely as the CPU reference.
 * GPU strategy: batched cuFFT R2C over all frames at once. */
static void fft_radix2(std::vector<float>& re, std::vector<float>& im) {
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

/* ---------------- STFT ----------------
 * GPU strategy: window/frame gather kernel (one thread per output sample, frames
 * overlap 4x) + batched cuFFT + magnitude kernel. */
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
    return V;
}

/* ---------------- small GEMM helpers ----------------
 * Loop orders chosen so the innermost loop streams contiguous memory.
 * GPU strategy: one tiled shared-memory gemm kernel with per-operand transpose
 * indexing; elementwise update steps are separate kernels (fusion = cleanup TODO). */

/* C(MxN) = A(MxK) * B(KxN) */
static void gemm_nn(const Mat& A, const Mat& B, Mat& C) {
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
static void gemm_tn(const Mat& A, const Mat& B, Mat& C) {
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
static void gemm_nt(const Mat& A, const Mat& B, Mat& C) {
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

/* ---------------- NMF / PFNMF ----------------
 * KL-divergence multiplicative updates:
 *   H <- H .* (W^T (V ./ WH)) ./ colsum(W)
 *   W <- W .* ((V ./ WH) H^T) ./ rowsum(H)   (first n_fixed cols of W frozen)
 * GPU strategy: each step is a gemm or elementwise/reduction kernel; all 100
 * iterations stay device-resident. */
void nmf(const Mat& V, Mat& W, Mat& H, int iters, int n_fixed) {
    const int M = V.rows, N = V.cols, R = W.cols;
    assert(W.rows == M && H.rows == R && H.cols == N);
    Mat WH(M, N), Z(M, N), numH(R, N), numW(M, R);
    std::vector<float> wcol(R), hrow(R);

    for (int it = 0; it < iters; it++) {
        /* H update */
        gemm_nn(W, H, WH);
        for (size_t i = 0; i < Z.v.size(); i++) Z.v[i] = V.v[i] / (WH.v[i] + NMF_EPS);
        gemm_tn(W, Z, numH);
        std::fill(wcol.begin(), wcol.end(), 0.f);
        for (int m = 0; m < M; m++)
            for (int r = 0; r < R; r++) wcol[r] += W.at(m, r);
        for (int r = 0; r < R; r++)
            for (int j = 0; j < N; j++)
                H.at(r, j) *= numH.at(r, j) / (wcol[r] + NMF_EPS);

        /* W update (skipped entirely if every column is frozen) */
        if (n_fixed < R) {
            gemm_nn(W, H, WH);
            for (size_t i = 0; i < Z.v.size(); i++) Z.v[i] = V.v[i] / (WH.v[i] + NMF_EPS);
            gemm_nt(Z, H, numW);
            std::fill(hrow.begin(), hrow.end(), 0.f);
            for (int r = 0; r < R; r++)
                for (int j = 0; j < N; j++) hrow[r] += H.at(r, j);
            for (int m = 0; m < M; m++)
                for (int r = n_fixed; r < R; r++)
                    numW.at(m, r) = W.at(m, r) * numW.at(m, r) / (hrow[r] + NMF_EPS);
            for (int m = 0; m < M; m++)
                for (int r = n_fixed; r < R; r++) W.at(m, r) = numW.at(m, r);
        }
    }
}

/* ---------------- pitch-shifted templates ----------------
 * Shifting the sample up by s semitones moves content from bin b/factor to bin b,
 * factor = 2^(s/12); so Wp[b] = W[b/factor], linearly interpolated.
 * GPU strategy: one thread per (bin, template) gather. */
Mat pitch_shift_templates(const Mat& W, float semitones) {
    const float factor = powf(2.f, semitones / 12.f);
    Mat Wp(W.rows, W.cols);
    for (int b = 0; b < W.rows; b++) {
        float src = b / factor;
        int lo = (int)src;
        float frac = src - lo;
        if (lo + 1 >= W.rows) continue;  /* outside source range -> 0 */
        for (int k = 0; k < W.cols; k++)
            Wp.at(b, k) = (1.f - frac) * W.at(lo, k) + frac * W.at(lo + 1, k);
    }
    return Wp;
}

void normalize_columns(Mat& W, Mat* H) {
    for (int k = 0; k < W.cols; k++) {
        double sq = 0.0;
        for (int m = 0; m < W.rows; m++) sq += (double)W.at(m, k) * W.at(m, k);
        float norm = (float)std::sqrt(sq) + 1e-12f;
        for (int m = 0; m < W.rows; m++) W.at(m, k) /= norm;
        if (H)
            for (int j = 0; j < H->cols; j++) H->at(k, j) *= norm;
    }
}

/* ---------------- correlation distance ----------------
 * Columns are RANK_K-dim activation vectors; Pearson r is scale invariant, so the
 * paper's H/max(H) normalization is a no-op here and is skipped (see TODO.md).
 * GPU strategy: z-normalize columns (one thread per column), then one thread per
 * (i,j) cell does a K=10 dot product. */
Mat correlation_distance(const Mat& Ho, const Mat& Hs) {
    const int K = RANK_K;
    auto znorm = [&](const Mat& H) {
        Mat Z(K, H.cols);
        for (int j = 0; j < H.cols; j++) {
            float mean = 0.f, var = 0.f;
            for (int k = 0; k < K; k++) mean += H.at(k, j);
            mean /= K;
            for (int k = 0; k < K; k++) {
                float d = H.at(k, j) - mean;
                var += d * d;
            }
            float denom = sqrtf(var) + 1e-12f;   /* z s.t. dot(za,zb) = Pearson r */
            for (int k = 0; k < K; k++) Z.at(k, j) = (H.at(k, j) - mean) / denom;
        }
        return Z;
    };
    Mat Zo = znorm(Ho), Zs = znorm(Hs);
    Mat D(Ho.cols, Hs.cols);
    for (int i = 0; i < Ho.cols; i++)
        for (int j = 0; j < Hs.cols; j++) {
            float r = 0.f;
            for (int k = 0; k < K; k++) r += Zo.at(k, i) * Zs.at(k, j);
            D.at(i, j) = 1.f - r;
        }
    return D;
}

/* ---------------- subsequence DTW ----------------
 * Paper eq. 2 boundary: free start anywhere in the query (row 0 = D), full
 * accumulation down column 0. L tracks path length for normalization.
 * GPU strategy: anti-diagonal wavefront — all cells on a diagonal are
 * independent; one kernel launch per diagonal, 12 pitch shifts batched in
 * blockIdx.y. */
float dtw_min_cost(const Mat& D) {
    const int No = D.rows, Ns = D.cols;
    Mat C(No, Ns), L(No, Ns);
    for (int j = 0; j < Ns; j++) { C.at(0, j) = D.at(0, j); L.at(0, j) = 1.f; }
    for (int i = 1; i < No; i++) {
        C.at(i, 0) = C.at(i - 1, 0) + D.at(i, 0);
        L.at(i, 0) = (float)(i + 1);
    }
    for (int i = 1; i < No; i++)
        for (int j = 1; j < Ns; j++) {
            float cd = C.at(i - 1, j - 1), cu = C.at(i - 1, j), cl = C.at(i, j - 1);
            float best = cd, len = L.at(i - 1, j - 1);
            if (cu < best) { best = cu; len = L.at(i - 1, j); }
            if (cl < best) { best = cl; len = L.at(i, j - 1); }
            C.at(i, j) = best + D.at(i, j);
            L.at(i, j) = len + 1.f;
        }
    /* Score = min/mean of the cost function: absolute cost level depends on the
     * candidate's snippet character (paper 3.3), so rank by the depth of the
     * minimum relative to this pair's typical cost (flat -> ~1 -> no sample). */
    float mn = 1e30f;
    double sum = 0.0;
    for (int j = 0; j < Ns; j++) {
        float c = C.at(No - 1, j) / L.at(No - 1, j);
        mn = std::min(mn, c);
        sum += c;
    }
    return mn / (float)(sum / Ns + 1e-12);
}

void seed_matrix(Mat& m, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.01f, 1.f);
    for (float& x : m.v) x = u(rng);
}

}  /* namespace sd */
