# Test Cases

Scan the music directory for all songs; the query is excluded from the library
automatically. Status as of 2026-06-11 (final build: log-frequency front end,
K=32, 5-candidate library; heuristic ranking stage — the trained RF stage is
evaluated on Sample100, see RESULTS.md).

## Real-world pairs (files verified correct by Steven)

1. Input: `Lucid Dreams.wav` → Expected: `Shape of My Heart.wav`
   **Status: PASS** (re-checked 2026-06-11, final build) — Shape ranks #1
   (0.583). Was a near miss in v3 behind the Gimme Gimme confound; the
   log-frequency front end flipped it. Note the reported best window
   (+4.25 st, cand @20s) differs from the known +0.00 alignment — the
   candidate wins, the location report is not the annotated one.

2. Input: `Touch The Sky.wav` → Expected: `Move On Up.wav`
   **Status: FAIL at the heuristic stage** (re-checked 2026-06-11) — true
   match found consistently (−0.25 st, 0.801, rank 3) behind two
   melodic-resemblance confounds (Gimme Gimme 0.557, Shape 0.626).
   NOTE: an earlier session concluded the files didn't share literal audio;
   Steven has verified the files are correct, and that conclusion is
   RETRACTED — the probes used were fingerprint-class full-spectrum matchers,
   which the paper itself (§2.1) says fail on samples mixed at low level under
   other sources. The sample is buried/processed beyond what the heuristic
   ranking separates.

3. Input: `Hung Up.wav` → Expected: `Gimme Gimme.wav`
   **Status: PASS, decisive** — 0.415 @ +0.00 st vs 0.568 runner-up,
   match at Gimme @96s → Hung Up @47s.

The failure mode of 2 is the one the paper's random-forest stage separates
(harmonically sparse candidates act as universal melodic matchers): on the
Sample100 benchmark the trained RF lifts hit@1 from 11.4% to 50.0%
(RESULTS.md). The RF is not wired into the `music/` CLI ranking; these
statuses are the heuristic stage.

## Verification tests (synthetic, ground truth known)

Built by `tests/make_fixtures.sh`, gated by `tests/run_ladder.sh`. All pass
on the final build (fresh run 2026-06-11):

4. Self-match canary: any song queried against a copy of itself must rank it
   first at shift 0 with consistent locations. (Final: 0.171 @ +0.00 st.)

5. Synthetic insert: 8 s of Move On Up mixed into Shape of My Heart at t=30 s.
   Must dip at +0.00 st AND recover the location with no hints.
   (Final: 0.421 @ +0.00 st, locations exact.)

6. Synthetic insert, +6% resample (pitch+tempo): must dip at +1 st
   (12·log2(1.06) = +1.01). (Final: 0.346 @ +1.00 st.)

7. CPU/GPU parity: **PASS** — the CPU reference mirrors the production
   pipeline; scores identical to 4 printed decimals on gates 4–6 at matched
   config (--max-seconds 30–45, --iters 30), same shifts and locations,
   despite TF32 + fast-math on the GPU side.
