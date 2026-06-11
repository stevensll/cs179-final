/* cpu_demo — single-threaded reference detector.
 * Usage: cpu_demo [--max-seconds S] [--iters I] <query.wav> <library_dir>
 * Prints library songs ranked by min subsequence-DTW cost (best match first). */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "../common/audio.hpp"
#include "../common/params.hpp"
#include "cpu_pipeline.hpp"

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

namespace {

struct Result {
    std::string name;
    float score;
    float best_shift;
};

double secs_since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

}  /* namespace */

int main(int argc, char** argv) {
    float max_seconds = 0.f;
    int iters = sd::DEFAULT_ITERS;
    int windows = sd::DEFAULT_WINDOWS;
    int argi = 1;
    while (argi < argc && argv[argi][0] == '-') {
        if (!strcmp(argv[argi], "--max-seconds") && argi + 1 < argc) max_seconds = atof(argv[++argi]);
        else if (!strcmp(argv[argi], "--iters") && argi + 1 < argc) iters = atoi(argv[++argi]);
        else if (!strcmp(argv[argi], "--windows") && argi + 1 < argc) windows = atoi(argv[++argi]);
        else { fprintf(stderr, "unknown flag %s\n", argv[argi]); return 1; }
        argi++;
    }
    if (argc - argi != 2) {
        fprintf(stderr, "usage: %s [--max-seconds S] [--iters I] <query.wav> <library_dir>\n", argv[0]);
        return 1;
    }
    const std::string query_path = argv[argi];
    const std::string lib_dir = argv[argi + 1];

    sd::enforce_time_limit(7200);  /* CPU demo is slow by design; see PROPOSAL.md */
    auto t0 = clk::now();

    auto query = sd::load_preprocessed(query_path, max_seconds);
    sd::Mat Vq = sd::stft_magnitude(query);
    printf("query: %s (%zu samples, %d frames)\n",
           fs::path(query_path).filename().c_str(), query.size(), Vq.cols);

    std::vector<Result> results;
    for (const auto& entry : fs::directory_iterator(lib_dir)) {
        if (entry.path().extension() != ".wav") continue;
        if (fs::equivalent(entry.path(), fs::path(query_path))) continue;  /* skip self */
        const std::string name = entry.path().filename().string();

        auto cand = sd::load_preprocessed(entry.path().string(), 0.f);
        const size_t wlen = (size_t)sd::SNIPPET_SECONDS * sd::SAMPLE_RATE;

        /* snippet hypotheses: max-RMS window + evenly spaced (same as GPU) */
        std::vector<size_t> offsets;
        offsets.push_back(sd::best_snippet_offset(cand));
        size_t span = cand.size() > wlen ? cand.size() - wlen : 0;
        for (int w = 0; w + 1 < windows; w++) {
            size_t o2 = windows > 2 ? span * w / (windows - 2) : 0;
            bool dup = false;  /* skip windows overlapping an existing one */
            for (size_t o : offsets)
                if ((o > o2 ? o - o2 : o2 - o) < wlen / 2) dup = true;
            if (!dup) offsets.push_back(o2);
        }

        float best = 1e30f, best_shift = 0.f;
        for (size_t off : offsets) {
            size_t win = std::min(wlen, cand.size() - off);
            std::vector<float> snippet(cand.begin() + off, cand.begin() + off + win);
            sd::Mat Vo = sd::stft_magnitude(snippet);

            /* templates of the candidate's sample hypothesis */
            sd::Mat Wo(sd::N_BINS, sd::RANK_K), Ho(sd::RANK_K, Vo.cols);
            sd::seed_matrix(Wo, 42);
            sd::seed_matrix(Ho, 43);
            sd::nmf(Vo, Wo, Ho, iters, 0);
            sd::normalize_columns(Wo, &Ho);  /* see normalize_columns docs */

            for (int p = 0; p < sd::N_SHIFTS; p++) {
                sd::Mat Wp = sd::pitch_shift_templates(Wo, sd::pitch_shift(p));
                sd::normalize_columns(Wp, nullptr);  /* shift changes norms */

                /* PFNMF: [Wp fixed | RANK_L free templates] against the query */
                const int R = sd::RANK_K + sd::RANK_L;
                sd::Mat W(sd::N_BINS, R), H(R, Vq.cols);
                sd::seed_matrix(W, 100 + p);
                sd::seed_matrix(H, 200 + p);
                for (int b = 0; b < sd::N_BINS; b++)
                    for (int k = 0; k < sd::RANK_K; k++) W.at(b, k) = Wp.at(b, k);
                sd::nmf(Vq, W, H, iters, sd::RANK_K);

                sd::Mat D = sd::correlation_distance(Ho, H);

                /* null calibration (matches GPU): same distances with query
                 * frames shuffled — kills temporal structure, keeps marginals.
                 * Scoring by true/null cancels per-candidate cost bias. */
                std::vector<int> perm(D.cols);
                for (int j = 0; j < D.cols; j++) perm[j] = j;
                std::mt19937 prng(7);
                std::shuffle(perm.begin(), perm.end(), prng);
                sd::Mat Dn(D.rows, D.cols);
                for (int i = 0; i < D.rows; i++)
                    for (int j = 0; j < D.cols; j++) Dn.at(i, j) = D.at(i, perm[j]);

                float cost = sd::dtw_min_cost(D) / (sd::dtw_min_cost(Dn) + 1e-12f);
                if (cost < best) { best = cost; best_shift = sd::pitch_shift(p); }
            }
        }
        printf("  %-28s score %.4f (shift %+.2f st)  [t=%.0fs]\n",
               name.c_str(), best, best_shift, secs_since(t0));
        results.push_back({name, best, best_shift});
    }

    std::sort(results.begin(), results.end(),
              [](const Result& a, const Result& b) { return a.score < b.score; });
    printf("\nranking (best match first):\n");
    for (size_t i = 0; i < results.size(); i++)
        printf("%zu. %-28s score %.4f (shift %+.1f st)\n",
               i + 1, results[i].name.c_str(), results[i].score, results[i].best_shift);
    printf("total %.0f s\n", secs_since(t0));
    return 0;
}
