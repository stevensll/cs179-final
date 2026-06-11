#include "gpu_pipeline.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <cufft.h>

#include "../common/params.hpp"
#include "gemm.cuh"
#include "kernels.cuh"

namespace sd {

namespace {

/* Deterministic init (fixed seeds per problem) so runs are reproducible. */
std::vector<float> seeded(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.01f, 1.f);
    std::vector<float> v(n);
    for (float& x : v) x = u(rng);
    return v;
}

/* Workspaces for one batched NMF; allocated once per call site and reused
 * across iterations (TODO.md cleanup item: was per-call alloc churn). */
struct NmfWorkspace {
    DeviceBuffer<float> WH, Z, numH, numW, wcol, hrow;
    void ensure(int M, int N, int R, int P) {
        size_t mn = (size_t)M * N;
        if (WH.size() < mn * P) WH.alloc(mn * P);
        if (Z.size() < mn * P) Z.alloc(mn * P);
        if (numH.size() < (size_t)P * R * N) numH.alloc((size_t)P * R * N);
        if (numW.size() < (size_t)P * M * R) numW.alloc((size_t)P * M * R);
        if (wcol.size() < (size_t)P * R) wcol.alloc((size_t)P * R);
        if (hrow.size() < (size_t)P * R) hrow.alloc((size_t)P * R);
    }
};

/* KL-divergence multiplicative updates over P stacked problems sharing V:
 *   H <- H .* (W^T (V ./ WH)) ./ colsum(W)
 *   W <- W .* ((V ./ WH) H^T) ./ rowsum(H)   (first n_fixed cols frozen)
 * GEMMs are cuBLAS strided-batched; elementwise/reduction steps are custom. */
void gpu_nmf_batched(const float* dV, int M, int N, float* dW, float* dH,
                     int R, int P, int iters, int n_fixed, NmfWorkspace& ws) {
    ws.ensure(M, N, R, P);
    const size_t mn = (size_t)M * N;
    const long long sW = (long long)M * R, sH = (long long)R * N, sV = (long long)mn;

    for (int it = 0; it < iters; it++) {
        /* H update */
        gemm_nn_batched(M, N, R, dW, sW, dH, sH, ws.WH.ptr(), sV, P);
        launch_ratio_batched(dV, ws.WH.ptr(), ws.Z.ptr(), mn, P);
        gemm_tn_batched(R, N, M, dW, sW, ws.Z.ptr(), sV, ws.numH.ptr(), sH, P);
        launch_col_sum_batched(dW, M, R, P, ws.wcol.ptr());
        launch_update_h_batched(dH, ws.numH.ptr(), ws.wcol.ptr(), R, N, P);

        /* W update (entirely frozen W -> skip) */
        if (n_fixed < R) {
            gemm_nn_batched(M, N, R, dW, sW, dH, sH, ws.WH.ptr(), sV, P);
            launch_ratio_batched(dV, ws.WH.ptr(), ws.Z.ptr(), mn, P);
            gemm_nt_batched(M, R, N, ws.Z.ptr(), sV, dH, sH, ws.numW.ptr(), sW, P);
            launch_row_sum_batched(dH, R, N, P, ws.hrow.ptr());
            launch_update_w_batched(dW, ws.numW.ptr(), ws.hrow.ptr(), M, R, P, n_fixed);
        }
    }
    checkCuda(cudaDeviceSynchronize());
}

}  /* namespace */

GpuMat gpu_stft(const std::vector<float>& x) {
    const int frames = x.size() >= FFT_SIZE ? 1 + (int)((x.size() - FFT_SIZE) / HOP) : 0;

    DeviceBuffer<float> dx(x.size());
    dx.to_device(x.data(), x.size());
    DeviceBuffer<float> dframes((size_t)frames * FFT_SIZE);
    launch_window_frames(dx.ptr(), frames, dframes.ptr());

    DeviceBuffer<cufftComplex> dc((size_t)frames * N_BINS);
    cufftHandle plan;
    checkCufft(cufftPlan1d(&plan, FFT_SIZE, CUFFT_R2C, frames));
    checkCufft(cufftExecR2C(plan, dframes.ptr(), dc.ptr()));
    checkCufft(cufftDestroy(plan));

    GpuMat V;
    V.rows = N_BINS;
    V.cols = frames;
    V.buf.alloc((size_t)N_BINS * frames);
    launch_magnitude(dc.ptr(), frames, V.ptr());
    checkCuda(cudaDeviceSynchronize());
    return V;
}

MatchInfo gpu_score_candidate(const GpuMat& Vq, const GpuMat& Vc, int iters, bool clip) {
    const int M = N_BINS, No = Vc.cols, Ns = Vq.cols;
    const int K = RANK_K, R = RANK_K + RANK_L, P = N_SHIFTS;
    const float fps = (float)SAMPLE_RATE / HOP;

    /* ---- candidate templates: one NMF over the full candidate song ---- */
    DeviceBuffer<float> Wo((size_t)M * K), Ho((size_t)K * No);
    Wo.to_device(seeded((size_t)M * K, 42));
    Ho.to_device(seeded((size_t)K * No, 43));
    DeviceBuffer<float> Zo((size_t)K * No);
    {
        NmfWorkspace ws;
        gpu_nmf_batched(Vc.ptr(), M, No, Wo.ptr(), Ho.ptr(), K, 1, iters, 0, ws);
        /* unit-norm templates (norms folded into Ho) so activation scales are
         * comparable between Ho and the PFNMF Hs across all pitch shifts */
        launch_normalize_columns(Wo.ptr(), M, K, Ho.ptr(), No);
        launch_max_normalize_batched(Ho.ptr(), K, No, K, 1);  /* paper 3.2.1 */
        launch_znorm_batched(Ho.ptr(), K, No, K, 1, Zo.ptr());
    }

    /* ---- one batched PFNMF over all P pitch shifts against the query ---- */
    DeviceBuffer<float> Zs((size_t)P * K * Ns);
    {
        DeviceBuffer<float> W_all((size_t)P * M * R), H_all((size_t)P * R * Ns);
        std::vector<float> init((size_t)P * M * R);
        for (int p = 0; p < P; p++) {
            auto v = seeded((size_t)M * R, 100 + p);
            std::copy(v.begin(), v.end(), init.begin() + (size_t)p * M * R);
        }
        W_all.to_device(init.data(), init.size());
        init.resize((size_t)P * R * Ns);
        for (int p = 0; p < P; p++) {
            auto v = seeded((size_t)R * Ns, 200 + p);
            std::copy(v.begin(), v.end(), init.begin() + (size_t)p * R * Ns);
        }
        H_all.to_device(init.data(), init.size());

        launch_pitch_templates_batched(Wo.ptr(), W_all.ptr(), M, K, R, P);
        launch_normalize_fixed_columns_batched(W_all.ptr(), M, K, R, P);

        NmfWorkspace ws;
        gpu_nmf_batched(Vq.ptr(), M, Ns, W_all.ptr(), H_all.ptr(), R, P, iters, K, ws);
        launch_max_normalize_batched(H_all.ptr(), R, Ns, K, P);  /* paper 3.2.1 */
        launch_znorm_batched(H_all.ptr(), R, Ns, K, P, Zs.ptr());
    }

    /* ---- distance matrices, one per pitch shift ----
     * (Pair-level null surrogates were removed from scoring AND computation:
     * shuffle nulls punish smoothness, reversal nulls punish looped samples —
     * see CLAUDE.md log. min/median across hypotheses is the calibration.) */
    const size_t mat = (size_t)No * Ns;
    DeviceBuffer<float> D((size_t)P * mat);
    launch_distance_batched(Zo.ptr(), No, Zs.ptr(), Ns, K, P, D.ptr());
    Zs.release();

    /* ---- banded subsequence DTW: dense candidate-window search ----
     * Run at every window length in WINDOW_SECONDS; the candidate's score is
     * the WORST across lengths (evidence must persist as the window grows: a
     * true sample keeps matching, a passing melodic resemblance diverges). */
    const int nmat = P;
    /* chunk bands to bound C/L memory (nmat * NSLOTS * T * Ns floats each) */
    const int NSLOTS = 4;
    const int Tmax = std::min(No, (int)(WINDOW_SECONDS.back() * fps));
    DeviceBuffer<float> C((size_t)nmat * NSLOTS * Tmax * Ns), L(C.size());
    std::vector<float> crows((size_t)nmat * NSLOTS * Ns), lrows(crows.size());

    MatchInfo best;
    float worst_of_lengths = 0.f;
    for (int window_seconds : WINDOW_SECONDS) {
    const int T = std::min(No, (int)(window_seconds * fps));
    const int hop = std::max(1, (int)(WINDOW_HOP_SECONDS * fps));
    const int nbands = No > T ? (No - T) / hop + 1 : 1;

    /* per (matrix, band): min of the normalized last row, its mean, its argmin */
    std::vector<float> rmin((size_t)nmat * nbands, 1e30f), rmean((size_t)nmat * nbands, 1.f);
    std::vector<int> rarg((size_t)nmat * nbands, 0);

    for (int band0 = 0; band0 < nbands; band0 += NSLOTS) {
        const int nslots = std::min(NSLOTS, nbands - band0);
        for (int d = 0; d < T + Ns - 1; d++)
            launch_dtw_band_diag(d, No, T, Ns, nmat, nslots, band0, hop,
                                 D.ptr(), C.ptr(), L.ptr());
        checkCuda(cudaDeviceSynchronize());
        /* last rows of every band in the chunk, one strided copy each */
        checkCuda(cudaMemcpy2D(crows.data(), (size_t)Ns * sizeof(float),
                               C.ptr() + (size_t)(T - 1) * Ns, (size_t)T * Ns * sizeof(float),
                               (size_t)Ns * sizeof(float), (size_t)nmat * nslots,
                               cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy2D(lrows.data(), (size_t)Ns * sizeof(float),
                               L.ptr() + (size_t)(T - 1) * Ns, (size_t)T * Ns * sizeof(float),
                               (size_t)Ns * sizeof(float), (size_t)nmat * nslots,
                               cudaMemcpyDeviceToHost));
        for (int m = 0; m < nmat; m++)
            for (int s = 0; s < nslots; s++) {
                const float* cr = crows.data() + ((size_t)m * nslots + s) * Ns;
                const float* lr = lrows.data() + ((size_t)m * nslots + s) * Ns;
                float mn = 1e30f;
                double sum = 0.0;
                int arg = 0, nvalid = 0;
                for (int j = 0; j < Ns; j++) {
                    /* Slope sanity (paper 3.3.2: real alignments have near-
                     * constant slope): L/T estimates the path's time-warp
                     * factor; bound it to the same +-33% the +-5 st pitch grid
                     * implies for resample-style transforms (padded). Without
                     * this, meandering paths dilute their own normalized cost
                     * and melodic-similarity false matches win. */
                    float warp = lr[j] / (float)T;
                    if (warp < 0.7f || warp > 1.5f) continue;
                    float c = cr[j] / lr[j];
                    if (c < mn) { mn = c; arg = j; }
                    sum += c;
                    nvalid++;
                }
                size_t idx = (size_t)m * nbands + band0 + s;
                if (nvalid == 0) { mn = 1.f; sum = 1.0; nvalid = 1; }
                rmin[idx] = mn;
                rmean[idx] = (float)(sum / nvalid);
                rarg[idx] = arg;
            }
    }

    /* ---- score ----
     * Raw dip s(p,w) = min/mean of the band's normalized DTW cost row.
     * 1) Pitch selectivity (paper 3.2.3 / fig. 2: a real match peaks at ONE
     *    shift): each window's dip is judged against that same window's median
     *    dip across all P shifts. Broadband/junk windows dip regardless of
     *    template pitch and self-cancel to ~1; pitched matches stand out.
     * 2) Selection bias: the best hypothesis is then judged against the
     *    candidate's own hypothesis distribution (min/median over all (p,w)),
     *    else a min over nbands*P hypotheses favors longer candidates. */
    MatchInfo lbest;  /* best hypothesis at THIS window length */
    std::vector<float> s((size_t)P * nbands);
    /* clip mode: the query IS the sample (wall-to-wall), so there is no
     * non-sample baseline for a dip to stand out from — score by absolute
     * normalized cost instead of dip depth (min/mean). */
    for (size_t i = 0; i < s.size(); i++)
        s[i] = clip ? rmin[i] : rmin[i] / (rmean[i] + 1e-12f);

    std::vector<float> col(P), sel((size_t)P * nbands);
    for (int w = 0; w < nbands; w++) {
        for (int p = 0; p < P; p++) col[p] = s[(size_t)p * nbands + w];
        std::nth_element(col.begin(), col.begin() + P / 2, col.end());
        float med = col[P / 2];
        for (int p = 0; p < P; p++) {
            size_t it = (size_t)p * nbands + w;
            sel[it] = s[it] / (med + 1e-12f);
            if (getenv("SD_DEBUG") && sel[it] < lbest.score * 1.05f)
                fprintf(stderr, "    T=%ds p=%+.2f w@%.0fs raw=%.4f sel=%.4f\n",
                        window_seconds, pitch_shift(p), w * hop / fps, s[it], sel[it]);
            if (sel[it] < lbest.score) {
                lbest.score = sel[it];
                lbest.shift = pitch_shift(p);
                lbest.cand_seconds = w * hop / fps;
                lbest.query_seconds = rarg[it] / fps;
            }
        }
    }
    std::nth_element(sel.begin(), sel.begin() + sel.size() / 2, sel.end());
    lbest.score /= (sel[sel.size() / 2] + 1e-12f);

    /* worst-across-lengths is the candidate's score; the (finer) shortest-
     * length hit provides the reported match location */
    if (window_seconds == WINDOW_SECONDS.front()) best = lbest;
    worst_of_lengths = std::max(worst_of_lengths, lbest.score);
    }  /* for window_seconds */
    best.score = worst_of_lengths;
    return best;
}

}  /* namespace sd */
