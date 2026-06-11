# Test Cases

Scan the music directory for all songs; the query is excluded from the library
automatically. Status as of 2026-06-10 (v3: K=40, pitch-selectivity scoring,
5-candidate library).

## Real-world pairs (files verified correct by Steven)

1. Input: `Lucid Dreams.wav` → Expected: `Shape of My Heart.wav`
   **Status: NEAR MISS** (re-checked 2026-06-11 post-optimization) — Shape
   ranks #2 (0.627) behind Gimme Gimme (0.506, +0.00 st), a genuine
   melodic-resemblance confound (both songs are built on minor-key arpeggio
   riffs). Passed when the library had 3 songs; the confound arrived with the
   new candidates. Since Gimme Gimme joined the library it tops BOTH failing
   queries — the strongest instance yet of the sparse-clean-riff confound
   class the paper's classifier stage exists to separate.

2. Input: `Touch The Sky.wav` → Expected: `Move On Up.wav`
   **Status: FAIL** (re-checked 2026-06-11) — true match found consistently
   (−0.50 st, 0.705, rank 3) behind two melodic-resemblance confounds
   (Gimme Gimme 0.550, Shape 0.595).
   NOTE: an earlier session concluded the files didn't share literal audio;
   Steven has verified the files are correct, and that conclusion is
   RETRACTED — the probes used were fingerprint-class full-spectrum matchers,
   which the paper itself (§2.1) says fail on samples mixed at low level under
   other sources. The sample is simply buried/processed beyond what our
   classifier-less ranking separates.

3. Input: `Hung Up.wav` → Expected: `Gimme Gimme.wav`
   **Status: PASS, decisive** — 0.300 @ +0.00 st vs 0.607 runner-up
   (2× margin), match at Gimme @97s → Hung Up @48s.

The shared failure mode of 1–2 is the one the paper's random-forest stage
exists to fix (see docs/PAPER-MAPPING.md, "What the missing classifier costs"):
harmonically sparse candidates act as universal melodic matchers.

## Verification tests (synthetic, ground truth known)

Built with ffmpeg (commands in the CLAUDE.md build log). All pass as of v3:

4. Self-match canary: any song queried against a copy of itself must rank it
   first at shift 0 with consistent locations. (v3: 0.108 @ +0.00 st.)

5. Synthetic insert: 8 s of Move On Up mixed into Shape of My Heart at t=30 s.
   Must dip at +0.00 st AND recover the location with no hints.
   (v3: 0.247 @ +0.00 st, cand @1s, query @35s.)

6. Synthetic insert, +6% resample (pitch+tempo): must dip at +1 st
   (12·log2(1.06) = +1.01). (v3: 0.218 @ +1.00 st.)

7. CPU/GPU parity: **currently N/A** — the CPU demo still implements the v1
   algorithm (it held 4-decimal parity with GPU v1 when both existed; measured
   numbers in the CLAUDE.md build log — note the repo has no git history).
   Restore once the CPU demo is ported (TODO.md).
