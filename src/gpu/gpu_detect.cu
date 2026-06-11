/* gpu_detect — GPU sample detector.
 * Usage: gpu_detect [--max-seconds S] [--iters I] <query.wav> <library_dir>
 * Prints the library ranked by match score (lower = better) with the matched
 * candidate window and query position. */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "../common/audio.hpp"
#include "../common/params.hpp"
#include "gpu_pipeline.cuh"

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

namespace {

struct Result {
    std::string name;
    sd::MatchInfo m;
};

double secs_since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

}  /* namespace */

int main(int argc, char** argv) {
    float max_seconds = 0.f;
    int iters = sd::DEFAULT_ITERS;
    bool clip = false;  /* query is a human-trimmed segment that IS the sample */
    int argi = 1;
    while (argi < argc && argv[argi][0] == '-') {
        if (!strcmp(argv[argi], "--max-seconds") && argi + 1 < argc) max_seconds = atof(argv[++argi]);
        else if (!strcmp(argv[argi], "--iters") && argi + 1 < argc) iters = atoi(argv[++argi]);
        else if (!strcmp(argv[argi], "--clip")) clip = true;
        else { fprintf(stderr, "unknown flag %s\n", argv[argi]); return 1; }
        argi++;
    }
    if (argc - argi != 2) {
        fprintf(stderr,
                "usage: %s [--max-seconds S] [--iters I] [--clip] <query.wav> <library_dir>\n"
                "  --clip: the query is a trimmed segment that IS the suspected sample\n",
                argv[0]);
        return 1;
    }
    const std::string query_path = argv[argi];
    const std::string lib_dir = argv[argi + 1];

    sd::enforce_time_limit(1800);
    auto t0 = clk::now();

    auto query = sd::load_preprocessed(query_path, max_seconds);
    sd::GpuMat Vq = sd::gpu_stft(query);
    printf("query: %s (%zu samples, %d frames)\n",
           fs::path(query_path).filename().c_str(), query.size(), Vq.cols);

    std::vector<Result> results;
    for (const auto& entry : fs::directory_iterator(lib_dir)) {
        if (entry.path().extension() != ".wav") continue;
        if (fs::equivalent(entry.path(), fs::path(query_path))) continue;  /* skip self */
        const std::string name = entry.path().filename().string();

        auto cand = sd::load_preprocessed(entry.path().string(), 0.f);
        sd::GpuMat Vc = sd::gpu_stft(cand);
        sd::MatchInfo m = sd::gpu_score_candidate(Vq, Vc, iters, clip);
        printf("  %-28s score %.4f (shift %+.2f st, cand @%.0fs -> query @%.0fs)  [t=%.1fs]\n",
               name.c_str(), m.score, m.shift, m.cand_seconds, m.query_seconds,
               secs_since(t0));
        results.push_back({name, m});
    }

    std::sort(results.begin(), results.end(),
              [](const Result& a, const Result& b) { return a.m.score < b.m.score; });
    printf("\nranking (best match first):\n");
    for (size_t i = 0; i < results.size(); i++)
        printf("%zu. %-28s score %.4f (shift %+.2f st, cand @%.0fs -> query @%.0fs)\n",
               i + 1, results[i].name.c_str(), results[i].m.score, results[i].m.shift,
               results[i].m.cand_seconds, results[i].m.query_seconds);
    printf("total %.1f s\n", secs_since(t0));
    return 0;
}
