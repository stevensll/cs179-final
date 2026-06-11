#include "audio.hpp"

#include <sndfile.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "params.hpp"

namespace sd {

/* 63-tap windowed-sinc lowpass at half the input Nyquist, evaluated only at
 * even sample positions (2:1 decimation). Hand-rolled: libsamplerate has no
 * headers on this box. */
static std::vector<float> decimate_2x(const std::vector<float>& x) {
    constexpr int TAPS = 63;
    constexpr float FC = 0.25f;  /* cutoff / input rate */
    float h[TAPS];
    float hsum = 0.f;
    for (int n = 0; n < TAPS; n++) {
        float m = n - (TAPS - 1) / 2.0f;
        float sinc = (m == 0.f) ? 2.f * FC
                                : sinf(2.f * M_PI * FC * m) / (M_PI * m);
        h[n] = sinc * (0.54f - 0.46f * cosf(2.f * M_PI * n / (TAPS - 1)));
        hsum += h[n];
    }
    for (int n = 0; n < TAPS; n++) h[n] /= hsum;  /* unity DC gain */

    std::vector<float> y(x.size() / 2);
    for (size_t i = 0; i < y.size(); i++) {
        float acc = 0.f;
        long center = (long)(2 * i);
        for (int n = 0; n < TAPS; n++) {
            long src = center + n - (TAPS - 1) / 2;
            if (src >= 0 && src < (long)x.size()) acc += h[n] * x[src];
        }
        y[i] = acc;
    }
    return y;
}

std::vector<float> load_preprocessed(const std::string& path, float max_seconds) {
    SF_INFO info = {};
    SNDFILE* f = sf_open(path.c_str(), SFM_READ, &info);
    if (!f) {
        std::fprintf(stderr, "error: cannot open %s: %s\n", path.c_str(), sf_strerror(nullptr));
        std::exit(1);
    }
    std::vector<float> interleaved((size_t)info.frames * info.channels);
    sf_count_t got = sf_readf_float(f, interleaved.data(), info.frames);
    sf_close(f);
    if (got <= 0) {
        std::fprintf(stderr, "error: no audio in %s\n", path.c_str());
        std::exit(1);
    }

    /* downmix to mono */
    std::vector<float> mono((size_t)got);
    for (size_t i = 0; i < mono.size(); i++) {
        float s = 0.f;
        for (int c = 0; c < info.channels; c++) s += interleaved[i * info.channels + c];
        mono[i] = s / info.channels;
    }

    /* RMS-normalize */
    double e = 0.0;
    for (float s : mono) e += (double)s * s;
    float rms = (float)std::sqrt(e / mono.size());
    if (rms > 0.f)
        for (float& s : mono) s /= rms;

    if (info.samplerate == 2 * SAMPLE_RATE) {
        mono = decimate_2x(mono);
    } else if (info.samplerate != SAMPLE_RATE) {
        std::fprintf(stderr, "error: %s is %d Hz; only %d or %d supported (MVP)\n",
                     path.c_str(), info.samplerate, SAMPLE_RATE, 2 * SAMPLE_RATE);
        std::exit(1);
    }

    if (max_seconds > 0.f) {
        size_t n = (size_t)(max_seconds * SAMPLE_RATE);
        if (mono.size() > n) mono.resize(n);
    }
    return mono;
}

size_t best_snippet_offset(const std::vector<float>& x) {
    const size_t win = (size_t)SNIPPET_SECONDS * SAMPLE_RATE;
    const size_t hop = (size_t)SNIPPET_HOP_SECONDS * SAMPLE_RATE;
    if (x.size() <= win) return 0;

    /* prefix sums of energy -> O(1) per window */
    std::vector<double> pre(x.size() + 1, 0.0);
    for (size_t i = 0; i < x.size(); i++) pre[i + 1] = pre[i] + (double)x[i] * x[i];

    size_t best = 0;
    double best_e = -1.0;
    for (size_t off = 0; off + win <= x.size(); off += hop) {
        double e = pre[off + win] - pre[off];
        if (e > best_e) { best_e = e; best = off; }
    }
    return best;
}

void enforce_time_limit(int limit_seconds) {
    std::thread([limit_seconds] {
        std::this_thread::sleep_for(std::chrono::seconds(limit_seconds));
        std::fprintf(stderr, "time limit (%d s) exceeded, exiting\n", limit_seconds);
        _exit(124);
    }).detach();
}

}  /* namespace sd */
