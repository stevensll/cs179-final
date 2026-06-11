/* gpu_detect — GPU sample detector.
 * Usage: gpu_detect [--max-seconds S] [--iters I] [--clip] [--gpus N]
 *                   <query.wav> <library_dir>
 * Prints the library ranked by match score (lower = better) with the matched
 * candidate window and query position. Candidates are distributed across all
 * visible GPUs (one worker thread per device, shared work queue). */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include <cuda_runtime.h>

#include "../common/audio.hpp"
#include "../common/params.hpp"
#include "gpu_pipeline.cuh"

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

namespace {

struct Result {
    std::string name;
    sd::MatchInfo m;
    int gpu = 0;
};

double secs_since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

/* ---- candidate-template disk cache ----
 * The candidate-side product (Wo, Zo) is query-independent and byte-stable
 * for a given (audio file, iters, analysis params); caching it removes the
 * decode + STFT + full-song NMF from every later query against the same
 * library (~70% of a clip-mode query's cost). Stored in <library>/.tcache/,
 * invalidated by audio size+mtime and the parameter fingerprint. Writes are
 * tmp+rename so concurrent eval workers can't observe torn files. */

struct TcHeader {
    char magic[4];
    int32_t version, fft, hop, k, iters, bins, bps;
    uint64_t audio_size;
    int64_t audio_mtime;
    int32_t No;
};
constexpr int32_t TCACHE_VERSION = 2;  /* v2: log-frequency front end fields */

int64_t mtime_of(const std::string& p) {
    return (int64_t)fs::last_write_time(p).time_since_epoch().count();
}

/* filename carries the config fingerprint so sweep variants coexist instead of
 * endlessly invalidating each other's entries */
fs::path tcache_path(const std::string& cand_path, const std::string& lib_dir) {
    return fs::path(lib_dir) / ".tcache" /
           (fs::path(cand_path).filename().string() + "_b" +
            std::to_string(sd::ANALYSIS_BINS) + "k" + std::to_string(sd::RANK_K) +
            "h" + std::to_string(sd::HOP) + ".tc");
}

bool tcache_header_ok(const TcHeader& h, const std::string& cand_path, int iters) {
    return std::memcmp(h.magic, "SDTC", 4) == 0 && h.version == TCACHE_VERSION &&
           h.fft == sd::FFT_SIZE && h.hop == sd::HOP && h.k == sd::RANK_K &&
           h.bins == sd::ANALYSIS_BINS && h.bps == sd::BINS_PER_ST &&
           h.iters == iters && h.audio_size == (uint64_t)fs::file_size(cand_path) &&
           h.audio_mtime == mtime_of(cand_path) && h.No > 0;
}

/* header-only probe (used by the decode prefetcher to skip needless decodes) */
bool tcache_probe(const std::string& cand_path, const std::string& lib_dir, int iters) {
    std::ifstream f(tcache_path(cand_path, lib_dir), std::ios::binary);
    TcHeader h{};
    if (!f.read((char*)&h, sizeof h)) return false;
    return tcache_header_ok(h, cand_path, iters);
}

bool tcache_load(const std::string& cand_path, const std::string& lib_dir, int iters,
                 sd::CandidateTemplates& ct) {
    std::ifstream f(tcache_path(cand_path, lib_dir), std::ios::binary);
    TcHeader h{};
    if (!f.read((char*)&h, sizeof h)) return false;
    if (!tcache_header_ok(h, cand_path, iters)) return false;
    std::vector<float> wo((size_t)sd::ANALYSIS_BINS * sd::RANK_K);
    std::vector<float> zo((size_t)sd::RANK_K * h.No);
    if (!f.read((char*)wo.data(), wo.size() * sizeof(float))) return false;
    if (!f.read((char*)zo.data(), zo.size() * sizeof(float))) return false;
    ct.No = h.No;
    ct.Wo.alloc(wo.size());
    ct.Wo.to_device(wo.data(), wo.size());
    ct.Zo.alloc(zo.size());
    ct.Zo.to_device(zo.data(), zo.size());
    return true;
}

void tcache_store(const std::string& cand_path, const std::string& lib_dir, int iters,
                  const sd::CandidateTemplates& ct) {
    fs::path path = tcache_path(cand_path, lib_dir);
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    TcHeader h{};
    std::memcpy(h.magic, "SDTC", 4);
    h.version = TCACHE_VERSION;
    h.fft = sd::FFT_SIZE;
    h.hop = sd::HOP;
    h.k = sd::RANK_K;
    h.bins = sd::ANALYSIS_BINS;
    h.bps = sd::BINS_PER_ST;
    h.iters = iters;
    h.audio_size = (uint64_t)fs::file_size(cand_path);
    h.audio_mtime = mtime_of(cand_path);
    h.No = ct.No;
    std::vector<float> wo = ct.Wo.to_host((size_t)sd::ANALYSIS_BINS * sd::RANK_K);
    std::vector<float> zo = ct.Zo.to_host((size_t)sd::RANK_K * ct.No);
    fs::path tmp = path;
    tmp += ".tmp" + std::to_string(::getpid());
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write((const char*)&h, sizeof h);
        f.write((const char*)wo.data(), wo.size() * sizeof(float));
        f.write((const char*)zo.data(), zo.size() * sizeof(float));
        if (!f.good()) { fs::remove(tmp, ec); return; }
    }
    fs::rename(tmp, path, ec);  /* atomic on same fs; loser of a race is fine */
    if (ec) fs::remove(tmp, ec);
}

}  /* namespace */

int main(int argc, char** argv) {
    float max_seconds = 0.f;
    int iters = sd::DEFAULT_ITERS;
    bool clip = false;  /* query is a human-trimmed segment that IS the sample */
    bool use_cache = true;
    int gpus = 0;       /* 0 = all visible devices */
    std::string features_path;  /* classifier feature-dump mode */
    int argi = 1;
    while (argi < argc && argv[argi][0] == '-') {
        if (!strcmp(argv[argi], "--max-seconds") && argi + 1 < argc) max_seconds = atof(argv[++argi]);
        else if (!strcmp(argv[argi], "--iters") && argi + 1 < argc) iters = atoi(argv[++argi]);
        else if (!strcmp(argv[argi], "--clip")) clip = true;
        else if (!strcmp(argv[argi], "--no-cache")) use_cache = false;
        else if (!strcmp(argv[argi], "--gpus") && argi + 1 < argc) gpus = atoi(argv[++argi]);
        else if (!strcmp(argv[argi], "--features") && argi + 1 < argc) features_path = argv[++argi];
        else { fprintf(stderr, "unknown flag %s\n", argv[argi]); return 1; }
        argi++;
    }
    if (argc - argi != 2) {
        fprintf(stderr,
                "usage: %s [--max-seconds S] [--iters I] [--clip] [--no-cache] [--gpus N] <query.wav> <library_dir>\n"
                "  --clip:     the query is a trimmed segment that IS the suspected sample\n"
                "  --no-cache: skip the <library>/.tcache candidate-template cache\n"
                "  --gpus:     limit worker devices (default: all visible GPUs)\n",
                argv[0]);
        return 1;
    }
    const std::string query_path = argv[argi];
    const std::string lib_dir = argv[argi + 1];

    sd::enforce_time_limit(1800);
    auto t0 = clk::now();

    auto query = sd::load_preprocessed(query_path, max_seconds);
    printf("query: %s (%zu samples)\n",
           fs::path(query_path).filename().c_str(), query.size());

    std::vector<std::string> cand_paths;
    for (const auto& entry : fs::directory_iterator(lib_dir)) {
        if (entry.path().extension() != ".wav") continue;
        if (fs::equivalent(entry.path(), fs::path(query_path))) continue;  /* skip self */
        cand_paths.push_back(entry.path().string());
    }
    /* longest-first: a long candidate dealt out last leaves one GPU idle at
     * the tail; deterministic (size, then name) so output order is stable */
    std::sort(cand_paths.begin(), cand_paths.end(),
              [](const std::string& a, const std::string& b) {
                  auto sa = fs::file_size(a), sb = fs::file_size(b);
                  return sa != sb ? sa > sb : a < b;
              });

    /* ---- classifier feature-dump mode: single worker, one CSV ---- */
    if (!features_path.empty()) {
        std::ofstream fout(features_path, std::ios::trunc);
        fout << "query,candidate,shift_st,band_start_s,query_start_s,heur_raw,heur_sel,"
                "n_endpoints,min_cost,avg_cost,std_cost,best_len,best_slope,best_dev,"
                "avg_slope,std_slope,avg_len,std_len,avg_dev,std_dev\n";
        const std::string qname = fs::path(query_path).filename().string();
        sd::GpuMat Vq = sd::gpu_stft(query);
        sd::ScoreContext ctx;
        for (const auto& path : cand_paths) {
            const std::string name = fs::path(path).filename().string();
            sd::CandidateTemplates ct;
            if (!(use_cache && tcache_load(path, lib_dir, iters, ct))) {
                auto cand = sd::load_preprocessed(path, 0.f);
                sd::GpuMat Vc = sd::gpu_stft(cand);
                ct = sd::gpu_candidate_templates(Vc, iters);
                if (use_cache) tcache_store(path, lib_dir, iters, ct);
            }
            auto rows = sd::gpu_extract_features(Vq, ct, iters, clip, 32, ctx);
            for (const auto& r : rows)
                fout << qname << ',' << name << ',' << r.shift << ',' << r.band_start_s
                     << ',' << r.query_start_s << ',' << r.heur_raw << ',' << r.heur_sel
                     << ',' << r.n_endpoints << ',' << r.min_cost << ',' << r.avg_cost
                     << ',' << r.std_cost << ',' << r.best_len << ',' << r.best_slope
                     << ',' << r.best_dev << ',' << r.avg_slope << ',' << r.std_slope
                     << ',' << r.avg_len << ',' << r.std_len << ',' << r.avg_dev
                     << ',' << r.std_dev << '\n';
            printf("  %-28s %zu feature rows  [t=%.1fs]\n", name.c_str(), rows.size(),
                   secs_since(t0));
            fflush(stdout);
        }
        printf("features written to %s (%.1f s)\n", features_path.c_str(), secs_since(t0));
        return 0;
    }

    int n_dev = 0;
    checkCuda(cudaGetDeviceCount(&n_dev));
    if (gpus > 0 && gpus < n_dev) n_dev = gpus;
    if ((int)cand_paths.size() < n_dev) n_dev = std::max(1, (int)cand_paths.size());

    /* group size for cross-candidate PFNMF batching: bounded by (a) memory —
     * the NMF workspace holds one V-sized slab per (candidate x shift), so
     * query length sets the budget (a 15 s clip groups 6, a 5-min song ~4) —
     * and (b) fairness: never let one group exceed an even split across the
     * devices (a 5-candidate scan once ran 4-vs-1 and idled a GPU). */
    const int q_frames = query.size() >= (size_t)sd::FFT_SIZE
                             ? 1 + (int)((query.size() - sd::FFT_SIZE) / sd::HOP) : 1;
    const size_t per_cand = (size_t)sd::N_SHIFTS * sd::ANALYSIS_BINS * (size_t)q_frames * sizeof(float);
    const int mem_group = (int)std::min<size_t>(6, std::max<size_t>(1, ((size_t)9 << 30) / per_cand));
    const int fair_group = std::max(1, (int)((cand_paths.size() + n_dev - 1) / n_dev));
    const int group = std::min(mem_group, fair_group);

    /* one worker per GPU; candidate GROUPS pulled from a shared queue and
     * scored in one stacked PFNMF batch. Each worker computes its own copy of
     * the query spectrogram (device data cannot be shared across devices).
     * The NEXT group's audio decodes run in a one-deep std::async pipeline so
     * host decode overlaps the current group's GPU work; the prefetcher skips
     * decoding when a valid template cache exists. */
    std::vector<Result> results(cand_paths.size());
    std::atomic<size_t> next{0};
    std::mutex print_mu;
    const size_t n_cand = cand_paths.size();
    auto prefetch_audio = [&](size_t idx) {
        return std::async(std::launch::async, [&, idx]() -> std::vector<float> {
            if (use_cache && tcache_probe(cand_paths[idx], lib_dir, iters)) return {};
            return sd::load_preprocessed(cand_paths[idx], 0.f);
        });
    };
    using Group = std::pair<size_t, size_t>;
    auto claim = [&]() -> Group {
        size_t s = next.fetch_add(group);
        return {s, std::min(n_cand, s + (size_t)group)};
    };
    auto prefetch_group = [&](Group g) {
        std::vector<std::future<std::vector<float>>> futs;
        for (size_t i = g.first; i < g.second && i < n_cand; i++)
            futs.push_back(prefetch_audio(i));
        return futs;
    };
    auto worker = [&](int dev) {
        checkCuda(cudaSetDevice(dev));
        sd::GpuMat Vq = sd::gpu_stft(query);
        sd::ScoreContext ctx;  /* persistent device scratch, grows to max size once */
        /* claim ONE group at a time: pre-claiming the next group for prefetch
         * starves the other GPU when there are few groups (a 2-group scan ran
         * entirely on one device). The group's own decodes still prefetch in
         * parallel below, and warm-cache runs skip decode anyway. */
        for (Group cur = claim(); cur.first < n_cand; cur = claim()) {
            auto curf = prefetch_group(cur);
            const size_t gc = cur.second - cur.first;
            std::vector<sd::CandidateTemplates> cts(gc);
            std::vector<bool> from_cache(gc, false);
            for (size_t k = 0; k < gc; k++) {
                const size_t idx = cur.first + k;
                std::vector<float> cand = curf[k].get();
                from_cache[k] = use_cache && tcache_load(cand_paths[idx], lib_dir, iters, cts[k]);
                if (!from_cache[k]) {
                    if (cand.empty())  /* prefetcher trusted a cache that failed full load */
                        cand = sd::load_preprocessed(cand_paths[idx], 0.f);
                    sd::GpuMat Vc = sd::gpu_stft(cand);
                    cts[k] = sd::gpu_candidate_templates(Vc, iters);
                    if (use_cache) tcache_store(cand_paths[idx], lib_dir, iters, cts[k]);
                }
            }
            std::vector<const sd::CandidateTemplates*> ptrs(gc);
            for (size_t k = 0; k < gc; k++) ptrs[k] = &cts[k];
            std::vector<sd::MatchInfo> ms = sd::gpu_score_candidates(Vq, ptrs, iters, clip, ctx);

            for (size_t k = 0; k < gc; k++) {
                const size_t idx = cur.first + k;
                const std::string name = fs::path(cand_paths[idx]).filename().string();
                std::lock_guard<std::mutex> lock(print_mu);
                printf("  %-28s score %.4f (shift %+.2f st, cand @%.0fs -> query @%.0fs)  [gpu%d%s t=%.1fs]\n",
                       name.c_str(), ms[k].score, ms[k].shift, ms[k].cand_seconds,
                       ms[k].query_seconds, dev, from_cache[k] ? " cached" : "",
                       secs_since(t0));
                fflush(stdout);
                results[idx] = {name, ms[k], dev};
            }
        }
    };
    std::vector<std::thread> threads;
    for (int dev = 0; dev < n_dev; dev++) threads.emplace_back(worker, dev);
    for (auto& t : threads) t.join();

    std::sort(results.begin(), results.end(),
              [](const Result& a, const Result& b) { return a.m.score < b.m.score; });
    printf("\nranking (best match first):\n");
    for (size_t i = 0; i < results.size(); i++)
        printf("%zu. %-28s score %.4f (shift %+.2f st, cand @%.0fs -> query @%.0fs)\n",
               i + 1, results[i].name.c_str(), results[i].m.score, results[i].m.shift,
               results[i].m.cand_seconds, results[i].m.query_seconds);
    printf("total %.1f s (%d gpu%s)\n", secs_since(t0), n_dev, n_dev > 1 ? "s" : "");
    return 0;
}
