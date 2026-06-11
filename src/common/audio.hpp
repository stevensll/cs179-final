/* Audio loading + preprocessing shared by the CPU demo and the GPU detector.
 * Decode via libsndfile (tolerates the broken RIFF sizes in music/). */

#pragma once

#include <string>
#include <vector>

namespace sd {

/* Load a WAV, downmix to mono, RMS-normalize, and decimate to SAMPLE_RATE.
 * max_seconds > 0 truncates the *decimated* signal (test/debug knob).
 * Exits with a message on failure. */
std::vector<float> load_preprocessed(const std::string& path, float max_seconds);

/* Kill the process after limit_seconds (shared-machine etiquette, CS179 convention). */
void enforce_time_limit(int limit_seconds);

}  /* namespace sd */
