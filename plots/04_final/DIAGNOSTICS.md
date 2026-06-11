# Profile diagnostics — 04_final

Report: `/tmp/iter04.nsys-rep` | profiled run: 5-candidate library scan

## Headline numbers

- CUDA-active wall span: **9,689.54 ms** (wall span = first..last CUDA event (runtime API + GPU ops))
- Total GPU kernel time: **9,073.43 ms** (93.6% of wall)
- Total memcpy/memset time: **0.00 ms** (0.0% of wall)
- Estimated GPU idle: **616.12 ms** (6.4% of wall) — wall minus GPU busy; overlap between kernels/copies would shrink busy time, so idle is a lower bound

## Category split

| category | time | % of wall | per candidate |
|---|---:|---:|---:|
| NMF elementwise | 4,727.71 ms | 48.8% | 945.54 ms |
| GEMM | 3,033.66 ms | 31.3% | 606.73 ms |
| distance/znorm | 813.05 ms | 8.4% | 162.61 ms |
| GPU idle (est.) | 616.12 ms | 6.4% | 123.22 ms |
| DTW | 497.14 ms | 5.1% | 99.43 ms |
| STFT | 1.49 ms | 0.0% | 0.30 ms |
| other kernels | 0.38 ms | 0.0% | 0.08 ms |

## Top kernels

| kernel | total | % of kernel time | instances | avg |
|---|---:|---:|---:|---:|
| sd::<unnamed>::fused_wh_ratio_kernel(const flo… | 4,538.24 ms | 50.0% | 120 | 37818.7 us |
| cuBLAS GEMM [cutlass_80_tensorop_s1688gem] | 1,908.56 ms | 21.0% | 120 | 15904.6 us |
| cuBLAS GEMM [cutlass_80_tensorop_s1688gem] | 1,125.10 ms | 12.4% | 120 | 9375.9 us |
| sd::<unnamed>::distance_batched_kernel(const f… | 811.60 ms | 8.9% | 14 | 57971.2 us |
| sd::<unnamed>::dtw_band_kernel(int, int, int, … | 497.14 ms | 5.5% | 14 | 35510.0 us |
| sd::<unnamed>::update_h_batched_kernel(float *… | 97.94 ms | 1.1% | 120 | 816.2 us |
| sd::<unnamed>::row_sum_batched_kernel(const fl… | 32.49 ms | 0.4% | 120 | 270.7 us |
| sd::<unnamed>::col_sum_batched_kernel(const fl… | 30.27 ms | 0.3% | 120 | 252.3 us |
| sd::<unnamed>::update_w_batched_kernel(float *… | 20.73 ms | 0.2% | 120 | 172.8 us |
| sd::<unnamed>::max_normalize_batched_kernel(fl… | 6.38 ms | 0.1% | 5 | 1276.7 us |

## Diagnosis — top 3 bottlenecks

1. **NMF elementwise** — 4,727.71 ms (48.8% of wall; 945.54 ms per candidate).
2. **GEMM** — 3,033.66 ms (31.3% of wall; 606.73 ms per candidate).
3. **distance/znorm** — 813.05 ms (8.4% of wall; 162.61 ms per candidate).

## What to optimize next

- GEMM is a large share: check matrix shapes per NMF iteration — batched cublasSgemmStridedBatched across candidates/pitches usually beats many small GEMMs.
