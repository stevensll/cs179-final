# Profile diagnostics — 00_baseline

Report: `/tmp/baseline.nsys-rep` | profiled run: 5-candidate library scan

## Headline numbers

- CUDA-active wall span: **32,870.62 ms** (wall span = first..last CUDA event (runtime API + GPU ops))
- Total GPU kernel time: **25,699.93 ms** (78.2% of wall)
- Total memcpy/memset time: **227.40 ms** (0.7% of wall)
- Estimated GPU idle: **6,943.29 ms** (21.1% of wall) — wall minus GPU busy; overlap between kernels/copies would shrink busy time, so idle is a lower bound
- cudaLaunchKernel: **1,626,748 calls**, 9,555.80 ms CPU time (~5.9 us/call)
- cudaDeviceSynchronize: 324 calls, 16,484.90 ms CPU time

## Category split

| category | time | % of wall | per candidate |
|---|---:|---:|---:|
| DTW | 11,046.45 ms | 33.6% | 2,209.29 ms |
| GEMM | 9,464.75 ms | 28.8% | 1,892.95 ms |
| GPU idle (est.) | 6,943.29 ms | 21.1% | 1,388.66 ms |
| NMF elementwise | 4,617.29 ms | 14.0% | 923.46 ms |
| distance/znorm | 564.28 ms | 1.7% | 112.86 ms |
| memcpy/memset | 227.40 ms | 0.7% | 45.48 ms |
| STFT | 6.73 ms | 0.0% | 1.35 ms |
| other kernels | 0.42 ms | 0.0% | 0.08 ms |

## Top kernels

| kernel | total | % of kernel time | instances | avg |
|---|---:|---:|---:|---:|
| sd::<unnamed>::dtw_band_diag_kernel(int, int, … | 11,046.45 ms | 43.0% | 1,620,696 | 6.8 us |
| sd::<unnamed>::ratio_batched_kernel(const floa… | 4,434.84 ms | 17.3% | 1,200 | 3695.7 us |
| cuBLAS GEMM [ampere_sgemm_128x128_nn] | 3,594.49 ms | 14.0% | 600 | 5990.8 us |
| cuBLAS GEMM [ampere_sgemm_128x128_tn] | 2,931.11 ms | 11.4% | 300 | 9770.4 us |
| cuBLAS GEMM [ampere_sgemm_128x128_nt] | 2,766.02 ms | 10.8% | 300 | 9220.1 us |
| sd::<unnamed>::distance_batched_kernel(const f… | 563.13 ms | 2.2% | 5 | 112626.4 us |
| cuBLAS GEMM [ampere_sgemm_128x32_nn] | 80.89 ms | 0.3% | 600 | 134.8 us |
| sd::<unnamed>::update_h_batched_kernel(float *… | 79.88 ms | 0.3% | 600 | 133.1 us |
| cuBLAS GEMM [ampere_sgemm_64x64_tn] | 48.38 ms | 0.2% | 300 | 161.3 us |
| sd::<unnamed>::col_sum_batched_kernel(const fl… | 32.54 ms | 0.1% | 600 | 54.2 us |

Launches per candidate: ~325,349 (1,626,748 / 5).

## Memory operations

| op | time | count |
|---|---:|---:|
| [CUDA memcpy Device-to-Host] | 176.94 ms | 616 |
| [CUDA memcpy Host-to-Device] | 50.07 ms | 33 |
| [CUDA memset] | 0.39 ms | 300 |

## Diagnosis — top 3 bottlenecks

1. **DTW** — 11,046.45 ms (33.6% of wall; 2,209.29 ms per candidate).
2. **GEMM** — 9,464.75 ms (28.8% of wall; 1,892.95 ms per candidate).
3. **GPU idle (est.)** — 6,943.29 ms (21.1% of wall; 1,388.66 ms per candidate).

## What to optimize next

- 1,626,748 kernel launches (~325,349/candidate) — launch overhead dominates if kernels average <10 us. Fuse elementwise NMF kernels (ratio/update/sum/normalize) and/or capture the NMF iteration loop in a CUDA graph.
- Tiny high-count kernels (sd::<unnamed>::dtw_band_diag_kernel(int, int, …) average <10 us each — strong fusion candidates; each launch costs more CPU time than GPU time.
- GEMM is a large share: check matrix shapes per NMF iteration — batched cublasSgemmStridedBatched across candidates/pitches usually beats many small GEMMs.
- DTW is a large share: increase per-launch work (process multiple diagonals or candidates per launch) and keep the band in shared memory.
