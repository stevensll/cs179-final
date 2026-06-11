# Profile diagnostics — 03_xcand

Report: `/tmp/iter03.nsys-rep` | profiled run: 5-candidate library scan

## Headline numbers

- CUDA-active wall span: **10,598.80 ms** (wall span = first..last CUDA event (runtime API + GPU ops))
- Total GPU kernel time: **9,886.35 ms** (93.3% of wall)
- Total memcpy/memset time: **44.34 ms** (0.4% of wall)
- Estimated GPU idle: **668.11 ms** (6.3% of wall) — wall minus GPU busy; overlap between kernels/copies would shrink busy time, so idle is a lower bound
- cudaLaunchKernel: **650 calls**, 3.92 ms CPU time (~6.0 us/call)

## Category split

| category | time | % of wall | per candidate |
|---|---:|---:|---:|
| NMF elementwise | 5,570.20 ms | 52.6% | 1,114.04 ms |
| GEMM | 3,029.83 ms | 28.6% | 605.97 ms |
| distance/znorm | 813.52 ms | 7.7% | 162.70 ms |
| GPU idle (est.) | 668.11 ms | 6.3% | 133.62 ms |
| DTW | 470.83 ms | 4.4% | 94.17 ms |
| memcpy/memset | 44.34 ms | 0.4% | 8.87 ms |
| STFT | 1.54 ms | 0.0% | 0.31 ms |
| other kernels | 0.42 ms | 0.0% | 0.08 ms |

## Top kernels

| kernel | total | % of kernel time | instances | avg |
|---|---:|---:|---:|---:|
| sd::<unnamed>::fused_wh_ratio_kernel(const flo… | 5,374.54 ms | 54.4% | 120 | 44787.9 us |
| cuBLAS GEMM [cutlass_80_tensorop_s1688gem] | 1,904.04 ms | 19.3% | 120 | 15867.0 us |
| cuBLAS GEMM [cutlass_80_tensorop_s1688gem] | 1,125.79 ms | 11.4% | 120 | 9381.6 us |
| sd::<unnamed>::distance_batched_kernel(const f… | 811.99 ms | 8.2% | 14 | 57999.3 us |
| sd::<unnamed>::dtw_band_kernel(int, int, int, … | 470.83 ms | 4.8% | 14 | 33630.9 us |
| sd::<unnamed>::update_h_batched_kernel(float *… | 98.38 ms | 1.0% | 120 | 819.8 us |
| sd::<unnamed>::row_sum_batched_kernel(const fl… | 32.46 ms | 0.3% | 120 | 270.5 us |
| sd::<unnamed>::col_sum_batched_kernel(const fl… | 30.12 ms | 0.3% | 120 | 251.0 us |
| sd::<unnamed>::update_w_batched_kernel(float *… | 26.60 ms | 0.3% | 120 | 221.6 us |
| sd::<unnamed>::max_normalize_batched_kernel(fl… | 6.46 ms | 0.1% | 5 | 1292.3 us |

Launches per candidate: ~130 (650 / 5).

## Memory operations

| op | time | count |
|---|---:|---:|
| [CUDA memcpy Host-to-Device] | 43.19 ms | 423 |
| [CUDA memcpy Device-to-Device] | 1.04 ms | 5 |
| [CUDA memcpy Device-to-Host] | 0.11 ms | 14 |

## Diagnosis — top 3 bottlenecks

1. **NMF elementwise** — 5,570.20 ms (52.6% of wall; 1,114.04 ms per candidate).
2. **GEMM** — 3,029.83 ms (28.6% of wall; 605.97 ms per candidate).
3. **distance/znorm** — 813.52 ms (7.7% of wall; 162.70 ms per candidate).

## What to optimize next

- GEMM is a large share: check matrix shapes per NMF iteration — batched cublasSgemmStridedBatched across candidates/pitches usually beats many small GEMMs.
