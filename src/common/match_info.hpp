/* Best match found for one candidate — shared by the CPU reference and the
 * GPU detector so their outputs are directly comparable. */

#pragma once

namespace sd {

struct MatchInfo {
    float score = 1e30f;       /* lower = better; selectivity-calibrated dip */
    float shift = 0.f;         /* semitones, sample in query vs candidate */
    float cand_seconds = 0.f;  /* start of best-matching candidate window */
    float query_seconds = 0.f; /* end of the best alignment in the query */
};

}  /* namespace sd */
