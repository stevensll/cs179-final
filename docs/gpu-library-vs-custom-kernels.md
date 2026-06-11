# GPU Work Breakdown: CUDA Libraries vs. Custom Kernels

> **Postscript (2026-06-10).** The stage-3 verdict below ("fully custom fused NMF,
> cuBLAS only as a benchmark baseline") was **superseded** the same day: per
> Steven's direction, **cuBLAS strided-batched GEMMs were adopted for all NMF/PFNMF
> matrix multiplies** (`src/gpu/gemm.cu(h)`; the elementwise/reduction steps and
> everything downstream remain custom — see CLAUDE.md "Confirmed decisions", v3).
> The rest of the analysis stands as written: stages 4, 6, and 7 (pitch-shift
> templates, distance matrix, subsequence DTW) are custom by necessity and were
> built that way. The original text below is preserved unedited as the
> decision-time record.

Decision doc for where to draw the custom-kernel line. Each pipeline stage from the
paper (Gururani & Lerch 2017) is listed with: what an off-the-shelf CUDA library could
do, what a custom kernel would look like, and how interesting that kernel is as CS179
work. Environment facts baked in: CUDA 12.5, 2× RTX A5000 (sm_86), cuFFT/cuBLAS/Thrust
available, no sudo (nothing new installable — but the toolkit already has everything
relevant).

## Pipeline recap with data sizes

For a ~4 min song at 22.05 kHz, block 4096 / hop 1024:

| Object | Shape | Size |
|---|---|---|
| Spectrogram V (song) | 2049 × ~5200 frames | ~42 MB |
| Spectrogram V (sample/query snippet, 5–25 s) | 2049 × ~100–550 | ~4 MB |
| NMF templates W | 2049 × (K=10 or K+L=30) | tiny |
| Activations H | 30 × ~5200 | tiny |
| Distance matrix D (per pitch shift) | ~550 × ~5200 ≈ 2.8 M entries | ~11 MB |
| Total per (query, library-song) pair | ×12 pitch shifts | ~150 MB working set |

Key parallelism axes: **12 pitch shifts × N library songs** are fully independent
(streams / 2 GPUs), plus fine-grained parallelism inside every stage.

## Stage-by-stage

### 1. Audio load, downmix, RMS-normalize, resample 44.1 → 22.05 kHz
- **Library option:** libsndfile (CPU) for WAV decode — headers present, works with the
  broken RIFF size fields in `music/`. libsamplerate exists as a runtime .so only (no
  header), so it is *not* usable.
- **Custom option:** trivial kernels — stereo downmix, RMS reduction, and a small
  windowed-sinc (or even 2:1 polyphase FIR) decimator.
- **Verdict:** decode on CPU with libsndfile; downmix/normalize/decimate are easy
  warm-up kernels. **Interest: low** (but free to claim as custom work).

### 2. Magnitude spectrogram (STFT)
- **Library option:** cuFFT batched R2C — one plan, all frames in one call. This is the
  canonical "allowed built-in." Even with cuFFT you still write two custom kernels:
  Hann window applied per frame (frames overlap 4×, so it's a gather + multiply) and
  complex → magnitude.
- **Custom option:** hand-rolled radix-2/radix-4 batched FFT kernel. A correct one is
  a weekend; a cuFFT-competitive one is its own quarter-long project.
- **Verdict:** cuFFT + custom window/magnitude kernels. The CPU demo needs a hand-rolled
  radix-2 FFT anyway (no FFTW headers on the box), so you'll have written an FFT once
  regardless — a fair talking point in the report. **Interest of custom FFT: high cost,
  low marginal credit.**

### 3. NMF (sample, K=10) and PFNMF (song, K+L=30) — the compute core
Multiplicative updates, ~100–200 iterations. Each iteration is four GEMMs plus
elementwise divide/multiply:

```
H ← H ⊙ (Wᵀ(V ⊘ WH)) ⊘ (Wᵀ1)      W ← W ⊙ ((V ⊘ WH)Hᵀ) ⊘ (1Hᵀ)   (KL divergence form)
```

(For PFNMF the first K columns of W are frozen — that's just a row/column mask on the
W update, identical kernel structure.)

- **Library option:** **there is no NMF in any NVIDIA library.** Best case, cuBLAS does
  the GEMMs and you still write the elementwise/fusion kernels and the update loop. Note
  the GEMM shapes are extreme: (2049×30)·(30×5200) — inner dimension 30. cuBLAS is
  optimized for large square-ish GEMMs; tall-skinny ones with k=30 leave it far from
  peak, which weakens the usual "you can't beat cuBLAS" argument.
- **Custom option:** a tiled shared-memory matmul specialized for k≤32 (the whole inner
  dimension fits in one tile — no k-loop), **fused** with the elementwise ⊘ and ⊙ so each
  update is 1–2 kernel launches instead of 4 GEMMs + 3 elementwise passes. Fusion saves
  ~3 full read/writes of the 42 MB V-sized intermediate per iteration × 200 iterations
  × 12 pitch shifts — this is where the real speedup over a naive port lives, and it's
  unreachable with cuBLAS.
- **Verdict options (the actual decision):**
  - **(a) Fully custom fused NMF kernels** — most CS179 credit, clean performance story
    (naive CPU → cuBLAS version → fused custom version makes a great 3-way benchmark).
  - **(b) cuBLAS GEMMs + custom elementwise** — ~40% less work, but the report then says
    "the dominant cost is library calls," and the elementwise kernels alone are not very
    interesting.
  - **Recommendation: (a), keeping a cuBLAS variant behind a flag as the benchmark
    baseline** — you get (b) almost for free on the way to (a). **Interest: high.**

### 4. Pitch-shifted template generation (W_o → W_o^p, 12 shifts)
- **Library option:** none. This is resampling the *frequency axis* of each template by
  2^(p/12) — no NVIDIA library expresses this.
- **Custom option:** small linear-interpolation gather kernel; one block per (template,
  shift). Optionally use texture memory for the interpolation — a nice CS179 flourish.
- **Verdict:** must be custom either way. **Interest: medium** (small but unavoidable
  and easy to make elegant).

### 5. Activation normalization (H / max(H))
- **Library option:** Thrust/CUB `max_element` reduce, then a scale pass.
- **Custom option:** standard two-stage parallel max-reduction kernel — a classic CS179
  set-piece (shared memory, warp shuffle, sequential addressing).
- **Verdict:** custom is ~50 lines and exactly what the course teaches; using Thrust
  here would look like dodging. **Interest: medium.**

### 6. Correlation distance matrix D (N_o × N_s per pitch shift)
- **Library option:** partially expressible as cuBLAS: after z-normalizing each
  activation column (custom kernel regardless), Pearson correlation = a single GEMM
  HₒᵀHₛ. But the inner dimension is K=10 — even worse for cuBLAS than stage 3 — and
  you still write the normalize kernel and the 1−r conversion.
- **Custom option:** one thread (or warp) per (i,j) cell; each computes a K=10-dim
  correlation. Hₒ tile lives entirely in shared memory (10×550 floats ≈ 22 KB).
  Memory-bound, coalesced, very teachable.
- **Verdict:** custom — the cuBLAS formulation is contorted and slower. All 12 pitch
  shifts batched in one launch (z-dimension of the grid). **Interest: medium-high.**

### 7. Subsequence DTW (the flagship)
- **Library option:** **none exists.** No NVIDIA library does dynamic programming /
  DTW. This is the centerpiece custom kernel and the strongest answer to "why does this
  project need custom kernels at all."
- **Custom option:** wavefront parallelism — cells along each anti-diagonal of the
  N_o × N_s cost matrix are independent. Design space is rich: one block per
  anti-diagonal sweep with grid-sync, vs. tiled/blocked DTW with halo exchange;
  shared-memory staging of the previous two diagonals; the subsequence variant's free
  first row (Eq. 2 of the paper) is a one-line change to initialization. The 12 pitch
  shifts batch in the grid z-dimension. Backtracking from every end column (for the
  path-start function) is itself parallel across end points.
- **Verdict:** custom by necessity; the most interesting kernel in the project and the
  one to spotlight in the proposal. **Interest: very high.**

### 8. DTW cost function minima, path features, ranking
- Last-row normalization, local-minima extraction, per-start-point aggregation, and the
  final argmin over library songs. Tiny data (one row per pair).
- **Verdict:** CPU, or trivial kernels if convenient. **Interest: low.**

## Summary table

| Stage | Library coverage | Must-write-anyway custom kernels | Decision needed? |
|---|---|---|---|
| 1. Load/resample | libsndfile (CPU decode) | downmix, RMS, decimate | no — easy customs |
| 2. STFT | **cuFFT** (allowed built-in) | window, magnitude | no — cuFFT + small customs |
| 3. NMF/PFNMF | cuBLAS for GEMMs only (poor fit at k=30) | elementwise updates, masking | **yes: fused custom vs cuBLAS** |
| 4. Pitch-shift templates | none | interpolation gather | no — custom only option |
| 5. Normalization | Thrust reduce | max-reduction | no — custom, course classic |
| 6. Distance matrix | contorted cuBLAS (k=10) | z-normalize, 1−r | no — custom recommended |
| 7. Subsequence DTW | **none** | wavefront DP, backtracking | no — custom only option |
| 8. Features/ranking | n/a | n/a | no — CPU |

**Bottom line:** stages 4, 6, 7 have no real library escape hatch, so the project clears
the "interesting enough to require custom kernels" bar no matter what. The only genuine
decision is **stage 3**: fully custom fused NMF (recommended; keeps a cuBLAS build as a
benchmark baseline) vs. cuBLAS-backed NMF (less work, weaker story). cuFFT stays as the
one deliberate built-in either way.

## Aside: scaling to a trained classifier (future work note)

The v1 back end ranks library songs by minimum subsequence-DTW cost (no classifier).
If we later train the paper's random forest (~500 labeled queries from the
whosampled-derived dataset, repo linked in the paper), nothing upstream changes: the
13 features in §3.3 are all computed from the cost-matrix last row and backtracked
paths we already produce. The GPU pipeline's throughput (12 pitch shifts × pairs in
parallel, 2 GPUs) is exactly what makes generating 500 queries' worth of features
tractable — worth one line in the proposal's "potential improvements."
