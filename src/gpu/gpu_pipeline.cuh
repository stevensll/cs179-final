/* GPU pipeline orchestration: STFT, batched NMF/PFNMF, banded-DTW scoring.
 * All heavy data stays device-resident; only DTW last rows come back. */

#pragma once

#include <vector>

#include "../common/match_info.hpp"
#include "device_buffer.cuh"

namespace sd {

struct GpuMat {
    DeviceBuffer<float> buf;
    int rows = 0, cols = 0;
    float* ptr() { return buf.ptr(); }
    const float* ptr() const { return buf.ptr(); }
};

/* Magnitude spectrogram (N_BINS x frames) of a preprocessed signal. */
GpuMat gpu_stft(const std::vector<float>& x);

/* Query-independent candidate-side product: spectral templates Wo (N_BINS x
 * RANK_K) and z-normalized activations Zo (RANK_K x No). Computing it costs a
 * full-song NMF; it is byte-stable for a given (audio, iters), so gpu_detect
 * caches it on disk per library song. */
struct CandidateTemplates {
    DeviceBuffer<float> Wo;  /* N_BINS * RANK_K */
    DeviceBuffer<float> Zo;  /* RANK_K * No */
    int No = 0;
};

/* Full-song NMF on the candidate spectrogram -> templates + z-activations. */
CandidateTemplates gpu_candidate_templates(const GpuMat& Vc, int iters);

/* Persistent per-worker device scratch, grow-only. cudaMalloc/cudaFree are
 * serialized process-wide by the driver (and cudaFree device-syncs), so
 * re-allocating the multi-GB NMF workspace per candidate group made the two
 * GPU workers run mostly mutually exclusive. Allocate once, reuse forever. */
struct ScoreContext {
    DeviceBuffer<float> W_all, H_all;        /* stacked PFNMF problems */
    DeviceBuffer<float> Z, numH, numW, wcol, hrow;  /* NMF workspace */
    DeviceBuffer<float> Hs, Zs, D;           /* scoring scratch */
    DeviceBuffer<float4> dstats;
};

/* Per-pair scoring: one batched PFNMF over all pitch shifts against the query,
 * then banded subsequence DTW searching every candidate window. clip = the
 * query is a human-trimmed segment that IS the suspected sample (wall-to-wall),
 * so score by absolute alignment cost instead of dip depth — a trimmed query
 * has no "surrounding non-sample" baseline for a dip to stand out from. */
MatchInfo gpu_score_candidate(const GpuMat& Vq, const CandidateTemplates& ct,
                              int iters, bool clip);

/* Group variant: ONE stacked PFNMF over (candidates x shifts) — every problem
 * is query-sized, so candidates of any length batch without padding. Scoring
 * (distance/DTW, candidate-length-dependent) runs per candidate afterwards.
 * Group size is the caller's memory call: the NMF workspace holds one V-sized
 * slab per (candidate x shift). ctx is the worker's persistent scratch. */
std::vector<MatchInfo> gpu_score_candidates(const GpuMat& Vq,
                                            const std::vector<const CandidateTemplates*>& cts,
                                            int iters, bool clip, ScoreContext& ctx);

/* One classifier input row: the paper's 13 path/cost features (§3.3) for one
 * unique alignment-path start within one (shift, band) hypothesis. */
struct PathFeatures {
    float shift;          /* semitones */
    float band_start_s;   /* hypothesis window position in the candidate */
    float query_start_s;  /* backtracked path start in the query */
    float heur_raw, heur_sel;  /* the heuristic scores, for comparison */
    int n_endpoints;           /* feature 13: end points mapping to this start */
    float min_cost, avg_cost, std_cost;            /* features 1-3 */
    float best_len, best_slope, best_dev;          /* 4-6 (min-cost path) */
    float avg_slope, std_slope, avg_len, std_len, avg_dev, std_dev;  /* 7-12 */
};

/* Feature-mode scoring of ONE candidate: full pipeline, then for the top
 * `top_f` hypotheses (by heuristic sel score) re-sweeps DTW with predecessor
 * recording, backtracks every cost-row local minimum on the host, groups end
 * points by path start, and emits one PathFeatures row per group. */
std::vector<PathFeatures> gpu_extract_features(const GpuMat& Vq,
                                               const CandidateTemplates& ct,
                                               int iters, bool clip, int top_f,
                                               ScoreContext& ctx);

}  /* namespace sd */
