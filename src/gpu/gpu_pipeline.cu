#include "gpu_pipeline.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
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

/* Cached variant: the PFNMF init is ~23M mt19937 floats per candidate but
 * identical across candidates of the same query (W slabs are size-fixed, H
 * slabs depend only on the query frame count) — regenerating them was ~25%
 * of the GPU-idle gap (plots/02_fused). Per-thread: one worker per GPU. */
const std::vector<float>& seeded_cached(size_t n, unsigned seed) {
    thread_local std::map<std::pair<size_t, unsigned>, std::vector<float>> cache;
    auto key = std::make_pair(n, seed);
    auto it = cache.find(key);
    if (it == cache.end()) {
        if (cache.size() > 256) cache.clear();   /* bound growth across queries */
        it = cache.emplace(key, seeded(n, seed)).first;
    }
    return it->second;
}

/* grow-only allocation into persistent ScoreContext scratch */
template <typename T>
void ensure(DeviceBuffer<T>& b, size_t n) {
    if (b.size() < n) b.alloc(n);
}

/* Triangular log-spaced filterbank (MEL_BINS x N_BINS, dense, ~3 MB): center
 * frequencies MEL_FMIN * 2^(b / (12*BINS_PER_ST)), triangles spanning the
 * neighboring centers, each row normalized to unit sum. Rows whose triangle
 * falls between FFT bins (lowest octaves) take the nearest bin instead.
 * Built once per process; applied as a single GEMM in gpu_stft. */
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

/* KL-divergence multiplicative updates over P stacked problems sharing V:
 *   H <- H .* (W^T (V ./ WH)) ./ colsum(W)
 *   W <- W .* ((V ./ WH) H^T) ./ rowsum(H)   (first n_fixed cols frozen)
 * SIMULTANEOUS (Lee-Seung style) variant: both numerators come from the same
 * Z and the pre-update H/W (the alternating form's WH recompute + second
 * ratio pass were ~35% of kernel time, plots/01_tf32_dtw). Z itself comes
 * from the fused custom W*H+ratio kernel (WH never materialized; cuBLAS
 * can't host a division epilogue and underperforms at k<=64 anyway). The two
 * large-k GEMMs (W^T Z, Z H^T) are cuBLAS strided-batched TF32 tensor-core.
 * Accuracy-gated by the ladder and the eval-5 rank check. */
void gpu_nmf_batched(const float* dV, int M, int N, float* dW, float* dH,
                     int R, int P, int iters, int n_fixed, ScoreContext& ctx) {
    const size_t mn = (size_t)M * N;
    ensure(ctx.Z, mn * P);
    ensure(ctx.numH, (size_t)P * R * N);
    ensure(ctx.numW, (size_t)P * M * R);
    ensure(ctx.wcol, (size_t)P * R);
    ensure(ctx.hrow, (size_t)P * R);
    const long long sW = (long long)M * R, sH = (long long)R * N, sV = (long long)mn;

    for (int it = 0; it < iters; it++) {
        launch_fused_wh_ratio(dW, dH, dV, ctx.Z.ptr(), M, N, R, P);
        gemm_tn_batched(R, N, M, dW, sW, ctx.Z.ptr(), sV, ctx.numH.ptr(), sH, P);
        if (n_fixed < R) {  /* numerators from pre-update H and W */
            gemm_nt_batched(M, R, N, ctx.Z.ptr(), sV, dH, sH, ctx.numW.ptr(), sW, P);
            launch_row_sum_batched(dH, R, N, P, ctx.hrow.ptr());
        }
        launch_col_sum_batched(dW, M, R, P, ctx.wcol.ptr());
        launch_update_h_batched(dH, ctx.numH.ptr(), ctx.wcol.ptr(), R, N, P);
        if (n_fixed < R)
            launch_update_w_batched(dW, ctx.numW.ptr(), ctx.hrow.ptr(), M, R, P, n_fixed);
    }
    /* no sync: same-stream ordering covers downstream kernels, and D2H reads
     * synchronize implicitly — a device sync here only stalls host enqueueing */
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
    if (MEL_BINS <= 0) return V;

    /* log-frequency pooling: Vm = F * V (one GEMM; F is ~3 MB, uploaded per
     * call — trivial next to the spectrogram itself) */
    DeviceBuffer<float> F((size_t)MEL_BINS * N_BINS);
    F.to_device(log_filterbank());
    GpuMat Vm;
    Vm.rows = MEL_BINS;
    Vm.cols = frames;
    Vm.buf.alloc((size_t)MEL_BINS * frames);
    gemm_nn_batched(MEL_BINS, frames, N_BINS, F.ptr(), 0, V.ptr(), 0, Vm.ptr(), 0, 1);
    /* no sync needed: consumers are same-stream kernels; dx/dframes/dc free at
     * scope exit via cudaFree, which itself synchronizes with pending work */
    return Vm;
}

CandidateTemplates gpu_candidate_templates(const GpuMat& Vc, int iters) {
    const int M = ANALYSIS_BINS, No = Vc.cols, K = RANK_K;
    CandidateTemplates ct;
    ct.No = No;
    ct.Wo.alloc((size_t)M * K);
    ct.Zo.alloc((size_t)K * No);
    ct.Wo.to_device(seeded_cached((size_t)M * K, 42));
    DeviceBuffer<float> Ho((size_t)K * No);
    Ho.to_device(seeded((size_t)K * No, 43));  /* No varies per candidate */
    ScoreContext scratch;  /* cold path: runs once per candidate ever (tcache) */
    gpu_nmf_batched(Vc.ptr(), M, No, ct.Wo.ptr(), Ho.ptr(), K, 1, iters, 0, scratch);
    /* unit-norm templates (norms folded into Ho) so activation scales are
     * comparable between Ho and the PFNMF Hs across all pitch shifts */
    launch_normalize_columns(ct.Wo.ptr(), M, K, Ho.ptr(), No);
    launch_max_normalize_batched(Ho.ptr(), K, No, K, 1);  /* paper 3.2.1 */
    launch_znorm_batched(Ho.ptr(), K, No, K, 1, ct.Zo.ptr());
    checkCuda(cudaDeviceSynchronize());  /* callers may download for caching */
    return ct;
}

namespace {

struct LengthData {
    int T = 0, nbands = 0;
    std::vector<float> rmin, rmean;
    std::vector<int> rarg;
};

/* znorm + distance + DTW for one candidate's P activation slabs -> per-length
 * band stats. Runs in pitch-shift chunks: the skewed D layout is No*(No+Ns-1)
 * per shift — near-quadratic in candidate length (a 7-minute original needs
 * ~22 GB for 41 shifts at once; that OOM'd the first Sample100 eval).
 * Chunking bounds it at ~4 GB with zero extra work. Works on a COPY of H so
 * the caller's slabs stay pristine; leaves ctx.Zs valid for callers that
 * need the z-activations again (feature extraction). */
std::vector<LengthData> band_stats(const float* dH, const CandidateTemplates& ct,
                                   int Ns, ScoreContext& ctx) {
    const int K = RANK_K, R = RANK_K + RANK_L, P = N_SHIFTS;
    const int No = ct.No;
    const float fps = (float)SAMPLE_RATE / HOP;
    const int hop = std::max(1, (int)(WINDOW_HOP_SECONDS * fps));

    ensure(ctx.Hs, (size_t)P * R * Ns);
    DeviceBuffer<float>& Hs = ctx.Hs;
    checkCuda(cudaMemcpy(Hs.ptr(), dH, (size_t)P * R * Ns * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    launch_max_normalize_batched(Hs.ptr(), R, Ns, K, P);  /* paper 3.2.1 */
    ensure(ctx.Zs, (size_t)P * K * Ns);
    DeviceBuffer<float>& Zs = ctx.Zs;
    launch_znorm_batched(Hs.ptr(), R, Ns, K, P, Zs.ptr());

    const size_t mat = (size_t)No * (No + Ns - 1);
    const size_t budget = (size_t)4 << 30;
    const int pchunk = std::min<int>(P, std::max<size_t>(1, budget / (mat * sizeof(float))));
    ensure(ctx.D, (size_t)pchunk * mat);
    DeviceBuffer<float>& D = ctx.D;

    std::vector<LengthData> lens(WINDOW_SECONDS.size());
    for (size_t li = 0; li < WINDOW_SECONDS.size(); li++) {
        lens[li].T = std::min(No, (int)(WINDOW_SECONDS[li] * fps));
        lens[li].nbands = No > lens[li].T ? (No - lens[li].T) / hop + 1 : 1;
        lens[li].rmin.resize((size_t)P * lens[li].nbands);
        lens[li].rmean.resize((size_t)P * lens[li].nbands);
        lens[li].rarg.resize((size_t)P * lens[li].nbands);
    }
    for (int p0 = 0; p0 < P; p0 += pchunk) {
        const int pc = std::min(pchunk, P - p0);
        launch_distance_batched(ct.Zo.ptr(), No, Zs.ptr() + (size_t)p0 * K * Ns,
                                Ns, K, pc, D.ptr());
        for (auto& ld : lens) {
            ensure(ctx.dstats, (size_t)pc * ld.nbands);
            DeviceBuffer<float4>& dstats = ctx.dstats;
            launch_dtw_bands(No, ld.T, Ns, pc, ld.nbands, hop, D.ptr(), dstats.ptr());
            /* to_host's blocking memcpy orders after the kernel */
            std::vector<float4> stats = dstats.to_host((size_t)pc * ld.nbands);
            for (size_t idx = 0; idx < stats.size(); idx++) {
                size_t out = (size_t)p0 * ld.nbands + idx;
                ld.rmin[out] = stats[idx].x;
                ld.rmean[out] = stats[idx].y;
                ld.rarg[out] = (int)stats[idx].z;
            }
        }
    }
    return lens;
}

/* ---- score ----
 * Raw dip s(p,w) = min/mean of the band's normalized DTW cost row.
 * 1) Pitch selectivity (paper 3.2.3 / fig. 2: a real match peaks at ONE
 *    shift): each window's dip is judged against that same window's median
 *    dip across all P shifts. Broadband/junk windows dip regardless of
 *    template pitch and self-cancel to ~1; pitched matches stand out.
 * 2) Selection bias: the best hypothesis is then judged against the
 *    candidate's own hypothesis distribution (min/median over all (p,w)),
 *    else a min over nbands*P hypotheses favors longer candidates.
 * If sel_out is non-null it receives the per-(p,w) sel scores of the FIRST
 * window length (feature extraction selects hypotheses from it). */
MatchInfo score_candidate_from_H(const float* dH, const CandidateTemplates& ct,
                                 int Ns, bool clip, ScoreContext& ctx,
                                 std::vector<float>* sel_out = nullptr,
                                 std::vector<float>* raw_out = nullptr) {
    const int P = N_SHIFTS;
    const float fps = (float)SAMPLE_RATE / HOP;
    const int hop = std::max(1, (int)(WINDOW_HOP_SECONDS * fps));
    std::vector<LengthData> lens = band_stats(dH, ct, Ns, ctx);

    MatchInfo best;
    float worst_of_lengths = 0.f;
    for (size_t li = 0; li < lens.size(); li++) {
        const int window_seconds = WINDOW_SECONDS[li];
        const int nbands = lens[li].nbands;
        const std::vector<float>& rmin = lens[li].rmin;
        const std::vector<float>& rmean = lens[li].rmean;
        const std::vector<int>& rarg = lens[li].rarg;

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
        if (li == 0 && sel_out) *sel_out = sel;   /* pre-median-normalized copy */
        if (li == 0 && raw_out) *raw_out = s;
        std::nth_element(sel.begin(), sel.begin() + sel.size() / 2, sel.end());
        lbest.score /= (sel[sel.size() / 2] + 1e-12f);

        /* worst-across-lengths is the candidate's score; the (finer) shortest-
         * length hit provides the reported match location */
        if (window_seconds == WINDOW_SECONDS.front()) best = lbest;
        worst_of_lengths = std::max(worst_of_lengths, lbest.score);
    }
    best.score = worst_of_lengths;
    return best;
}

}  /* namespace */

std::vector<MatchInfo> gpu_score_candidates(const GpuMat& Vq,
                                            const std::vector<const CandidateTemplates*>& cts,
                                            int iters, bool clip, ScoreContext& ctx) {
    const int M = ANALYSIS_BINS, Ns = Vq.cols;
    const int K = RANK_K, R = RANK_K + RANK_L, P = N_SHIFTS;
    const int C = (int)cts.size();
    const int CP = C * P;

    /* ---- one PFNMF batch across (candidates x pitch shifts) ----
     * Every problem is (M x Ns) — candidate length never enters PFNMF, only
     * the frozen template columns differ — so C candidates stack into a
     * single strided batch with no padding. This amortizes the 60-iteration
     * launch sequence and its host gaps across the whole group (the gaps were
     * the dominant idle after the kernel-level optimizations; see plots/). */
    ensure(ctx.W_all, (size_t)CP * M * R);
    ensure(ctx.H_all, (size_t)CP * R * Ns);
    for (int c = 0; c < C; c++)
        for (int p = 0; p < P; p++) {
            const auto& w = seeded_cached((size_t)M * R, 100 + p);
            checkCuda(cudaMemcpy(ctx.W_all.ptr() + ((size_t)c * P + p) * M * R, w.data(),
                                 w.size() * sizeof(float), cudaMemcpyHostToDevice));
            const auto& h = seeded_cached((size_t)R * Ns, 200 + p);
            checkCuda(cudaMemcpy(ctx.H_all.ptr() + ((size_t)c * P + p) * R * Ns, h.data(),
                                 h.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    for (int c = 0; c < C; c++) {
        float* Wc = ctx.W_all.ptr() + (size_t)c * P * M * R;
        launch_pitch_templates_batched(cts[c]->Wo.ptr(), Wc, M, K, R, P);
        launch_normalize_fixed_columns_batched(Wc, M, K, R, P);
    }
    gpu_nmf_batched(Vq.ptr(), M, Ns, ctx.W_all.ptr(), ctx.H_all.ptr(), R, CP, iters, K, ctx);

    std::vector<MatchInfo> out(C);
    for (int c = 0; c < C; c++)
        out[c] = score_candidate_from_H(ctx.H_all.ptr() + (size_t)c * P * R * Ns,
                                        *cts[c], Ns, clip, ctx);
    return out;
}

MatchInfo gpu_score_candidate(const GpuMat& Vq, const CandidateTemplates& ct,
                              int iters, bool clip) {
    ScoreContext ctx;
    return gpu_score_candidates(Vq, {&ct}, iters, clip, ctx)[0];
}

namespace {

/* One backtracked alignment path (band row T-1 end point -> row 0 start). */
struct BtPath {
    int js = 0, je = 0;
    float cost = 0.f, len = 0.f, slope = 0.f, dev = 0.f;
};

/* Walk the predecessor map from end point (T-1, je); returns the path with
 * its start column, normalized cost/length, slope (time-warp factor), and
 * mean perpendicular deviation from the idealized straight start->end line
 * (paper 3.3.2: real alignments are straight; resemblances rubber-band). */
BtPath backtrack_path(const unsigned char* pred, int T, int Ns, int je,
                      float cost, float len) {
    std::vector<std::pair<int, int>> cells;
    cells.reserve(3 * T);
    int i = T - 1, j = je;
    while (true) {
        cells.push_back({i, j});
        unsigned char p = pred[(size_t)i * Ns + j];
        if (p == 0 || i == 0) break;
        if (p == 1) { i--; j--; }
        else if (p == 2) { i--; }
        else { j--; }
    }
    BtPath out;
    out.js = cells.back().second;
    out.je = je;
    out.cost = cost;
    out.len = len;
    out.slope = (float)(je - out.js) / (float)std::max(1, T - 1);
    const float dy = (float)(je - out.js), dx = (float)(T - 1);
    const float denom = std::sqrt(dx * dx + dy * dy) + 1e-12f;
    double dev = 0.0;
    for (auto& c : cells)
        dev += std::abs(c.first * dy - (c.second - out.js) * dx) / denom;
    out.dev = (float)(dev / cells.size());
    return out;
}

}  /* namespace */

std::vector<PathFeatures> gpu_extract_features(const GpuMat& Vq,
                                               const CandidateTemplates& ct,
                                               int iters, bool clip, int top_f,
                                               ScoreContext& ctx) {
    const int M = ANALYSIS_BINS, Ns = Vq.cols;
    const int K = RANK_K, R = RANK_K + RANK_L, P = N_SHIFTS;
    const int No = ct.No;
    const float fps = (float)SAMPLE_RATE / HOP;
    const int hop = std::max(1, (int)(WINDOW_HOP_SECONDS * fps));

    /* PFNMF for this one candidate (same init as gpu_score_candidates, C=1) */
    ensure(ctx.W_all, (size_t)P * M * R);
    ensure(ctx.H_all, (size_t)P * R * Ns);
    for (int p = 0; p < P; p++) {
        const auto& w = seeded_cached((size_t)M * R, 100 + p);
        checkCuda(cudaMemcpy(ctx.W_all.ptr() + (size_t)p * M * R, w.data(),
                             w.size() * sizeof(float), cudaMemcpyHostToDevice));
        const auto& h = seeded_cached((size_t)R * Ns, 200 + p);
        checkCuda(cudaMemcpy(ctx.H_all.ptr() + (size_t)p * R * Ns, h.data(),
                             h.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    launch_pitch_templates_batched(ct.Wo.ptr(), ctx.W_all.ptr(), M, K, R, P);
    launch_normalize_fixed_columns_batched(ctx.W_all.ptr(), M, K, R, P);
    gpu_nmf_batched(Vq.ptr(), M, Ns, ctx.W_all.ptr(), ctx.H_all.ptr(), R, P, iters, K, ctx);

    /* heuristic pass: per-hypothesis scores; leaves ctx.Zs valid */
    std::vector<float> sel, raw;
    score_candidate_from_H(ctx.H_all.ptr(), ct, Ns, clip, ctx, &sel, &raw);

    const int T = std::min(No, (int)(WINDOW_SECONDS[0] * fps));
    const int nbands = No > T ? (No - T) / hop + 1 : 1;

    /* top hypotheses by sel score, grouped by shift */
    std::vector<size_t> order(sel.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    const size_t take = std::min<size_t>(top_f, order.size());
    std::partial_sort(order.begin(), order.begin() + take, order.end(),
                      [&](size_t a, size_t b) { return sel[a] < sel[b]; });
    std::map<int, std::vector<int>> by_shift;  /* p -> selected bands w */
    for (size_t k = 0; k < take; k++)
        by_shift[(int)(order[k] / nbands)].push_back((int)(order[k] % nbands));

    const size_t mat = (size_t)No * (No + Ns - 1);
    ensure(ctx.D, mat);
    std::vector<PathFeatures> out;
    for (auto& [p, ws] : by_shift) {
        launch_distance_batched(ct.Zo.ptr(), No, ctx.Zs.ptr() + (size_t)p * K * Ns,
                                Ns, K, 1, ctx.D.ptr());
        const int F = (int)ws.size();
        DeviceBuffer<int> dws(F);
        dws.to_device(ws.data(), F);
        DeviceBuffer<unsigned char> dpred((size_t)F * T * Ns);
        DeviceBuffer<float2> dlast((size_t)F * Ns);
        launch_dtw_band_preds(No, T, Ns, F, dws.ptr(), hop, ctx.D.ptr(),
                              dpred.ptr(), dlast.ptr());
        std::vector<unsigned char> pred = dpred.to_host((size_t)F * T * Ns);
        std::vector<float2> last = dlast.to_host((size_t)F * Ns);

        for (int k = 0; k < F; k++) {
            const unsigned char* pk = pred.data() + (size_t)k * T * Ns;
            const float2* lk = last.data() + (size_t)k * Ns;
            /* slope-filtered normalized cost row; +inf where invalid */
            std::vector<float> c(Ns, 1e30f);
            for (int j = 0; j < Ns; j++) {
                float warp = lk[j].y / (float)T;
                if (warp >= 0.7f && warp <= 1.5f) c[j] = lk[j].x / lk[j].y;
            }
            /* backtrack every local minimum (occurrence end point) */
            std::vector<BtPath> paths;
            for (int j = 0; j < Ns; j++) {
                if (c[j] >= 1e30f) continue;
                if (j > 0 && c[j - 1] < c[j]) continue;
                if (j + 1 < Ns && c[j + 1] <= c[j]) continue;
                paths.push_back(backtrack_path(pk, T, Ns, j, c[j], lk[j].y));
            }
            if (paths.empty()) continue;
            std::sort(paths.begin(), paths.end(),
                      [](const BtPath& a, const BtPath& b) { return a.js < b.js; });

            /* group end points by path start (merge starts within ~1 s) */
            size_t g0 = 0;
            for (size_t i = 1; i <= paths.size(); i++) {
                if (i < paths.size() && paths[i].js - paths[i - 1].js <= (int)fps)
                    continue;
                /* aggregate group [g0, i) into the paper's 13 features */
                PathFeatures f{};
                f.shift = pitch_shift(p);
                f.band_start_s = ws[k] * hop / fps;
                f.n_endpoints = (int)(i - g0);
                f.heur_raw = raw[(size_t)p * nbands + ws[k]];
                f.heur_sel = sel[(size_t)p * nbands + ws[k]];
                const BtPath* bestp = &paths[g0];
                double sc = 0, sc2 = 0, ss = 0, ss2 = 0, sl = 0, sl2 = 0, sd = 0, sd2 = 0;
                for (size_t q = g0; q < i; q++) {
                    const BtPath& b = paths[q];
                    if (b.cost < bestp->cost) bestp = &b;
                    sc += b.cost; sc2 += b.cost * b.cost;
                    ss += b.slope; ss2 += b.slope * b.slope;
                    double ln = b.len / T;
                    sl += ln; sl2 += ln * ln;
                    sd += b.dev; sd2 += b.dev * b.dev;
                }
                const double n = (double)f.n_endpoints;
                auto stddev = [&](double s1, double s2) {
                    double v = s2 / n - (s1 / n) * (s1 / n);
                    return (float)std::sqrt(std::max(0.0, v));
                };
                f.min_cost = bestp->cost;
                f.avg_cost = (float)(sc / n);
                f.std_cost = stddev(sc, sc2);
                f.best_len = bestp->len / T;
                f.best_slope = bestp->slope;
                f.best_dev = bestp->dev;
                f.avg_slope = (float)(ss / n);
                f.std_slope = stddev(ss, ss2);
                f.avg_len = (float)(sl / n);
                f.std_len = stddev(sl, sl2);
                f.avg_dev = (float)(sd / n);
                f.std_dev = stddev(sd, sd2);
                f.query_start_s = bestp->js / fps;
                out.push_back(f);
                g0 = i;
            }
        }
    }
    return out;
}

}  /* namespace sd */
