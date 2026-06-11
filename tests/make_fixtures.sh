#!/usr/bin/env bash
# Regenerate the verification-ladder fixtures (see docs/TESTING.md items 3-6).
# Synthetic inserts: 8 s of Move On Up mixed into Shape of My Heart at t=30 s,
# plain and +6% resampled (ground truth: dips at +0.00 st and +1.01 st).
# Probe dirs: single-candidate libraries for the canary and the synthetics.
set -euo pipefail
cd "$(dirname "$0")"
MUSIC="$(cd ../music && pwd)"
mkdir -p fixtures
cd fixtures

ffmpeg -loglevel error -y -i "$MUSIC/Move On Up.wav" -ss 0 -t 8 -ac 2 snip.wav
ffmpeg -loglevel error -y -i "$MUSIC/Shape of My Heart.wav" -i snip.wav \
  -filter_complex "[1]volume=1.0,adelay=30000|30000[s];[0][s]amix=inputs=2:duration=first:normalize=0" \
  synth_plain.wav
ffmpeg -loglevel error -y -i snip.wav \
  -filter_complex "asetrate=44100*1.06,aresample=44100" snip_fast.wav
ffmpeg -loglevel error -y -i "$MUSIC/Shape of My Heart.wav" -i snip_fast.wav \
  -filter_complex "[1]volume=1.0,adelay=30000|30000[s];[0][s]amix=inputs=2:duration=first:normalize=0" \
  synth_fast.wav
rm -f snip.wav snip_fast.wav

mkdir -p probe_mou probe_selfcopy
ln -sf "$(cd "$MUSIC" && pwd)/Move On Up.wav" probe_mou/
[ -f probe_selfcopy/MoveCopy.wav ] || cp "$MUSIC/Move On Up.wav" probe_selfcopy/MoveCopy.wav
echo "fixtures ready in $(pwd)"
