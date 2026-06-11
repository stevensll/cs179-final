# Profile diagnostics — 05_mel

Report: `/tmp/iter05.nsys-rep` | profiled run: 5-candidate library scan

## Headline numbers

- CUDA-active wall span: **3,461.28 ms** (wall span = first..last CUDA event (runtime API + GPU ops))
- Total GPU kernel time: **2,870.45 ms** (82.9% of wall)
- Total memcpy/memset time: **38.21 ms** (1.1% of wall)
- Estimated GPU idle: **552.62 ms** (16.0% of wall) — wall minus GPU busy; overlap between kernels/copies would shrink busy time, so idle is a lower bound
- cudaLaunchKernel: **350 calls**, 2.45 ms CPU time (~7.0 us/call)

## Category split

| category | time | % of wall | per candidate |
|---|---:|---:|---:|
| NMF elementwise | 974.07 ms | 28.1% | 194.81 ms |
| distance/znorm | 813.67 ms | 23.5% | 162.73 ms |
| GEMM | 584.79 ms | 16.9% | 116.96 ms |
| GPU idle (est.) | 552.62 ms | 16.0% | 110.52 ms |
| DTW | 496.37 ms | 14.3% | 99.27 ms |
| memcpy/memset | 38.21 ms | 1.1% | 7.64 ms |
| STFT | 1.49 ms | 0.0% | 0.30 ms |
| other kernels | 0.07 ms | 0.0% | 0.01 ms |

## Top kernels

| kernel | total | % of kernel time | instances | avg |
|---|---:|---:|---:|---:|
| sd::<unnamed>::fused_wh_ratio_kernel(const flo… | 827.95 ms | 28.8% | 60 | 13799.2 us |
| sd::<unnamed>::distance_batched_kernel(const f… | 812.21 ms | 28.3% | 14 | 58015.2 us |
| sd::<unnamed>::dtw_band_kernel(int, int, int, … | 496.37 ms | 17.3% | 14 | 35454.7 us |
| cuBLAS GEMM [cutlass_80_tensorop_s1688gem] | 357.29 ms | 12.4% | 60 | 5954.8 us |
| cuBLAS GEMM [cutlass_80_tensorop_s1688gem] | 227.06 ms | 7.9% | 60 | 3784.3 us |
| sd::<unnamed>::update_h_batched_kernel(float *… | 97.14 ms | 3.4% | 60 | 1619.0 us |
| sd::<unnamed>::row_sum_batched_kernel(const fl… | 31.53 ms | 1.1% | 60 | 525.5 us |
| sd::<unnamed>::col_sum_batched_kernel(const fl… | 6.86 ms | 0.2% | 60 | 114.3 us |
| sd::<unnamed>::max_normalize_batched_kernel(fl… | 6.40 ms | 0.2% | 5 | 1279.6 us |
| sd::<unnamed>::update_w_batched_kernel(float *… | 3.87 ms | 0.1% | 60 | 64.6 us |

Launches per candidate: ~70 (350 / 5).

## Memory operations

| op | time | count |
|---|---:|---:|
| [CUDA memcpy Host-to-Device] | 37.12 ms | 424 |
| [CUDA memcpy Device-to-Device] | 0.99 ms | 5 |
| [CUDA memcpy Device-to-Host] | 0.09 ms | 14 |

## Diagnosis — top 3 bottlenecks

1. **NMF elementwise** — 974.07 ms (28.1% of wall; 194.81 ms per candidate).
2. **distance/znorm** — 813.67 ms (23.5% of wall; 162.73 ms per candidate).
3. **GEMM** — 584.79 ms (16.9% of wall; 116.96 ms per candidate).

## What to optimize next

- No single dominant bottleneck; profile individual kernels with ncu for occupancy/memory-bound analysis.
