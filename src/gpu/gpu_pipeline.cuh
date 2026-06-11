/* GPU pipeline orchestration: STFT, batched NMF/PFNMF, banded-DTW scoring.
 * All heavy data stays device-resident; only DTW last rows come back. */

#pragma once

#include <vector>

#include "device_buffer.cuh"

namespace sd {

struct GpuMat {
    DeviceBuffer<float> buf;
    int rows = 0, cols = 0;
    float* ptr() { return buf.ptr(); }
    const float* ptr() const { return buf.ptr(); }
};

/* Best match found for one candidate. */
struct MatchInfo {
    float score = 1e30f;       /* lower = better; null-calibrated dip depth */
    float shift = 0.f;         /* semitones, sample in query vs candidate */
    float cand_seconds = 0.f;  /* start of best-matching candidate window */
    float query_seconds = 0.f; /* end of the best alignment in the query */
};

/* Magnitude spectrogram (N_BINS x frames) of a preprocessed signal. */
GpuMat gpu_stft(const std::vector<float>& x);

/* Full per-candidate scoring: full-song NMF on the candidate, one batched PFNMF
 * over all pitch shifts against the query, then banded subsequence DTW searching
 * every candidate window. clip = the query is a human-trimmed segment that IS
 * the suspected sample (wall-to-wall), so score by absolute alignment cost
 * instead of dip depth — a trimmed query has no "surrounding non-sample"
 * baseline for a dip to stand out from. */
MatchInfo gpu_score_candidate(const GpuMat& Vq, const GpuMat& Vc, int iters, bool clip);

}  /* namespace sd */
