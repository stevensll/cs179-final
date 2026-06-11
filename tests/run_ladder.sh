#!/usr/bin/env bash
# Verification ladder (docs/TESTING.md): structural gates for any pipeline
# change. PASS criteria are about shifts/locations/ranks, not absolute scores
# (representation changes legitimately move score values).
#   1. canary:      self-copy ranks #1 at shift +0.00
#   2. synth_plain: dip at +0.00 st, candidate window near 0 s
#   3. synth_fast:  dip at +1.00 st (+6% resample ground truth: +1.01)
#   4. real pair:   Hung Up -> Gimme Gimme ranks #1
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=build/gpu_detect
ITERS="${ITERS:-60}"
[ -f tests/fixtures/synth_plain.wav ] || tests/make_fixtures.sh

fail=0
check() {  # name, output-line, expected-regex
    if echo "$2" | grep -qE "$3"; then echo "PASS  $1: $2"; else echo "FAIL  $1: $2 (want /$3/)"; fail=1; fi
}

out=$($BIN --iters "$ITERS" "music/Move On Up.wav" tests/fixtures/probe_selfcopy | grep "^1\.")
check canary "$out" "MoveCopy.*shift \+0\.00"

out=$($BIN --iters "$ITERS" tests/fixtures/synth_plain.wav tests/fixtures/probe_mou | grep "^1\.")
check synth_plain "$out" "Move On Up.*shift \+0\.00"

out=$($BIN --iters "$ITERS" tests/fixtures/synth_fast.wav tests/fixtures/probe_mou | grep "^1\.")
check synth_fast "$out" "Move On Up.*shift \+1\.00"

out=$($BIN --iters "$ITERS" "music/Hung Up.wav" music | grep "^1\.")
check hung_up "$out" "Gimme Gimme"

exit $fail
