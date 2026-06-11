# Profile diagnostics — 06_fil

Report: `/tmp/iter06.nsys-rep` | profiled run: 5-candidate library scan

## Headline numbers

- CUDA-active wall span: **3,156.18 ms** (wall span = first..last CUDA event (runtime API + GPU ops))
- Total GPU kernel time: **2,587.88 ms** (82.0% of wall)
- Total memcpy/memset time: **33.94 ms** (1.1% of wall)
- Estimated GPU idle: **534.36 ms** (16.9% of wall) — wall minus GPU busy; overlap between kernels/copies would shrink busy time, so idle is a lower bound
- cudaLaunchKernel: **350 calls**, 2.65 ms CPU time (~7.6 us/call)

## Category split

| category | time | % of wall | per candidate |
|---|---:|---:|---:|
| NMF elementwise | 845.53 ms | 26.8% | 169.11 ms |
| distance/znorm | 682.03 ms | 21.6% | 136.41 ms |
| GEMM | 562.23 ms | 17.8% | 112.45 ms |
| GPU idle (est.) | 534.36 ms | 16.9% | 106.87 ms |
| DTW | 496.54 ms | 15.7% | 99.31 ms |
| memcpy/memset | 33.94 ms | 1.1% | 6.79 ms |
| STFT | 1.49 ms | 0.0% | 0.30 ms |
| other kernels | 0.06 ms | 0.0% | 0.01 ms |

## Top kernels

| kernel | total | % of kernel time | instances | avg |
|---|---:|---:|---:|---:|
| sd::<unnamed>::fused_wh_ratio_kernel(const flo… | 718.13 ms | 27.7% | 60 | 11968.8 us |
| sd::<unnamed>::distance_batched_kernel(const f… | 680.90 ms | 26.3% | 14 | 48635.7 us |
| sd::<unnamed>::dtw_band_kernel(int, int, int, … | 496.54 ms | 19.2% | 14 | 35467.2 us |
| cuBLAS GEMM [cutlass_80_tensorop_s1688gem] | 335.92 ms | 13.0% | 60 | 5598.7 us |
| cuBLAS GEMM [cutlass_80_tensorop_s1688gem] | 225.86 ms | 8.7% | 60 | 3764.3 us |
| sd::<unnamed>::update_h_batched_kernel(float *… | 84.91 ms | 3.3% | 60 | 1415.2 us |
| sd::<unnamed>::row_sum_batched_kernel(const fl… | 27.45 ms | 1.1% | 60 | 457.4 us |
| sd::<unnamed>::col_sum_batched_kernel(const fl… | 6.02 ms | 0.2% | 60 | 100.3 us |
| sd::<unnamed>::max_normalize_batched_kernel(fl… | 5.35 ms | 0.2% | 5 | 1069.5 us |
| sd::<unnamed>::update_w_batched_kernel(float *… | 3.43 ms | 0.1% | 60 | 57.1 us |

Launches per candidate: ~70 (350 / 5).

## Memory operations

| op | time | count |
|---|---:|---:|
| [CUDA memcpy Host-to-Device] | 32.98 ms | 424 |
| [CUDA memcpy Device-to-Device] | 0.86 ms | 5 |
| [CUDA memcpy Device-to-Host] | 0.10 ms | 14 |

## Diagnosis — top 3 bottlenecks

1. **NMF elementwise** — 845.53 ms (26.8% of wall; 169.11 ms per candidate).
2. **distance/znorm** — 682.03 ms (21.6% of wall; 136.41 ms per candidate).
3. **GEMM** — 562.23 ms (17.8% of wall; 112.45 ms per candidate).

## What to optimize next

- No single dominant bottleneck; profile individual kernels with ncu for occupancy/memory-bound analysis.
