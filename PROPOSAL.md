# CS 179 Final Project Proposal: GPU-Accelerated Sample Detection

> **Status (2026-06-10):** proposal as submitted; for what was actually built (and where it deviates), see `README.md` and `docs/PAPER-MAPPING.md`.

Steven Lei

## Summary

Given a query song that samples another song, find the sampled song in a library of
songs. I will implement the NMF + dynamic-time-warping sample detection algorithm of
Gururani & Lerch (ISMIR 2017, see `docs/`) as custom CUDA kernels, with a single-threaded
C++ CPU demo as the correctness baseline and benchmark reference.

## Background

Sampling reuses a snippet of an existing recording inside a new song, usually pitch
shifted and/or time stretched and buried under other instruments, so spectrogram or
fingerprint matching (Shazam-style) fails. The paper's approach treats the problem as
source separation: factorize the candidate original's magnitude spectrogram with
Non-negative Matrix Factorization (V ≈ W·H, K=10 spectral templates), then run
*partially fixed* NMF (PFNMF) on the query song with those K templates frozen plus L=20
free templates that absorb everything else in the mix. If the sample is present, the
frozen templates activate in the query with the same temporal pattern as in the
original. Pitch shifts are handled by rescaling the frequency axis of the templates for
12 hypothetical shifts; time stretching is handled by aligning the two activation
sequences with subsequence DTW over a correlation-distance matrix. A low DTW alignment
cost indicates the sample is present.

Computationally, for a ~4 minute song at 22.05 kHz (STFT block 4096, hop 1024) the
spectrogram V is 2049×~5200 (~42 MB as float). Each query/candidate pair requires 12
independent PFNMF runs (~100–200 multiplicative-update iterations each, every iteration
touching V-sized intermediates), then 12 distance matrices (~550×5200) and 12
subsequence-DTW passes. The CPU reference implementation (MATLAB in the original work)
takes on the order of minutes per pair; a library scan multiplies that by the library
size. The workload is parallel at every level: across library songs, across the 12
pitch shifts, and within every stage (element-wise NMF updates, independent distance
cells, anti-diagonal DTW wavefronts), which is what makes it a good GPU target.

Instead of the paper's final random-forest classifier (which needs a labeled training
set), this project ranks library songs by minimum subsequence-DTW cost and returns the
best match — the retrieval formulation matches the project goal and removes the ML
training dependency. (See *Potential extensions* for the classifier path.)

## Questions to address

**Previous GPU implementations?** No end-to-end GPU implementation of sample detection
exists; the reference implementation is MATLAB. The building blocks have been studied
separately — GPU NMF via cuBLAS-backed multiplicative updates, and wavefront-parallel
DTW/Smith-Waterman-style dynamic programming — but composing them, and the
sample-detection-specific pieces (PFNMF with frozen templates, pitch-shifted template
banks, *subsequence* DTW with per-column backtracking), must be custom. No NVIDIA
library provides NMF, DTW, or frequency-axis template resampling
(see `docs/gpu-library-vs-custom-kernels.md` for the full stage-by-stage analysis).

**Technical challenges.**
- *Subsequence DTW* is a 2-D dependency-carrying recurrence — the textbook "hard to
  parallelize" case. The plan is anti-diagonal wavefront parallelism with the previous
  two diagonals staged in shared memory, batched over 12 pitch shifts in the grid
  z-dimension, plus parallel backtracking from every end column.
- *NMF updates are tall-skinny GEMMs* (2049×30 times 30×5200): the inner dimension of
  30 is far from cuBLAS's sweet spot, and each update chains 4 GEMMs with element-wise
  divide/multiply passes over V-sized matrices. Custom kernels that hold the entire
  inner dimension in one tile and fuse the element-wise work eliminate ~3 full
  V-sized memory round-trips per iteration × 200 iterations × 12 shifts.
- *Numerical verification*: floating-point reordering means the GPU cannot match the
  CPU bitwise; per-stage golden-file comparisons need principled tolerances while the
  end-to-end ranking must still be exact.

**Problems to solve.** Audio I/O and resampling with no installable dependencies
(libsndfile is on the box; the 44.1→22.05 kHz decimator and the CPU demo's FFT must be
hand-written); choosing DTW tile/block shapes for ~550×5200 matrices on an A5000
(84 SMs); keeping 12-shift batches resident in 24 GB; overlap of independent
(song, shift) work via streams and, as a stretch goal, across both GPUs.

## Deliverables and goals

1. **CPU demo** (week 1): single-threaded C++ implementing the full pipeline — WAV
   load, preprocess, STFT (hand-rolled radix-2 FFT), NMF/PFNMF, pitch-shifted
   templates, distance matrix, subsequence DTW, ranking — passing both test cases in
   `docs/TESTING.md`, with comments marking each stage's parallelization strategy.
   Its per-stage outputs become the golden files for GPU verification.
2. **GPU implementation** (weeks 2–3): custom kernels for every stage; cuFFT is the
   one deliberate library use (STFT batching). A cuBLAS-backed NMF variant is kept
   behind a build flag purely as a benchmark baseline.
3. **Performance analysis**: per-stage CPU vs GPU timings, plus a three-way NMF
   comparison (naive CPU / cuBLAS variant / fused custom kernels) and end-to-end
   library-scan throughput.
4. **Final README** per spec: install/usage, description, expected results,
   performance tables, potential improvements.

**Stretch goals:** multi-GPU library scan (one A5000 per half of the library);
larger test library beyond the 4 seed songs.

**Potential extensions (noted for the report):** the paper's random-forest classifier
needs ~500 labeled queries' worth of features; all 13 features derive from the DTW cost
matrix last row and backtracked paths this pipeline already produces, so the GPU
throughput is exactly what would make generating that training set tractable.

## Week-by-week timeline (3 weeks)

**Week 1 — proposal + CPU demo.**
Proposal (this document). Project skeleton (CMake + Makefile wrapper, repo layout per
`docs/TECHNICAL.md`). Audio loading via libsndfile, downmix/RMS-normalize/decimate.
Radix-2 FFT and STFT. CPU NMF + PFNMF, pitch-shifted template generation, distance
matrix, subsequence DTW, ranking. Both test pairs pass on CPU; golden files emitted.

**Week 2 — GPU pipeline front half.**
CUDA scaffolding (error checking, `DeviceBuffer`, timers). Window + magnitude kernels
around batched cuFFT. Fused NMF/PFNMF multiplicative-update kernels (+ cuBLAS variant
behind a flag). Pitch-shift template interpolation kernel. Max-reduction normalization
kernel. Per-stage golden tests pass.

**Week 3 — GPU back half + analysis.**
Correlation distance-matrix kernel (12 shifts batched). Wavefront subsequence-DTW
kernel + parallel backtracking. Ranking + end-to-end `gpu_detect` passing both test
pairs. Optimization passes (occupancy, shared-memory shapes, streams). Benchmarks,
performance writeup, final README. Stretch goals as time allows.
