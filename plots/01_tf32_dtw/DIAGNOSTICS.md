# Profile diagnostics — 01_tf32_dtw

Report: `/tmp/iter01.nsys-rep` | profiled run: 5-candidate library scan

## Headline numbers

- CUDA-active wall span: **20,179.99 ms** (wall span = first..last CUDA event (runtime API + GPU ops))
- Total GPU kernel time: **15,990.12 ms** (79.2% of wall)
- Total memcpy/memset time: **0.00 ms** (0.0% of wall)
- Estimated GPU idle: **4,189.87 ms** (20.8% of wall) — wall minus GPU busy; overlap between kernels/copies would shrink busy time, so idle is a lower bound

## Category split

| category | time | % of wall | per candidate |
|---|---:|---:|---:|
| GEMM | 8,194.96 ms | 40.6% | 1,638.99 ms |
| NMF elementwise | 6,525.60 ms | 32.3% | 1,305.12 ms |
| GPU idle (est.) | 4,189.87 ms | 20.8% | 837.97 ms |
| distance/znorm | 813.44 ms | 4.0% | 162.69 ms |
| DTW | 448.95 ms | 2.2% | 89.79 ms |
| STFT | 6.75 ms | 0.0% | 1.35 ms |
| other kernels | 0.42 ms | 0.0% | 0.08 ms |

## Top kernels

| kernel | total | % of kernel time | instances | avg |
|---|---:|---:|---:|---:|
| sd::<unnamed>::ratio_batched_kernel(const floa… | 6,316.44 ms | 39.5% | 1,200 | 5263.7 us |
| cuBLAS GEMM [ampere_sgemm_128x128_nn] | 4,995.71 ms | 31.2% | 600 | 8326.2 us |
| cuBLAS GEMM [cutlass_80_tensorop_s1688gem] | 1,934.73 ms | 12.1% | 300 | 6449.1 us |
| cuBLAS GEMM [cutlass_80_tensorop_s1688gem] | 1,138.30 ms | 7.1% | 300 | 3794.3 us |
| sd::<unnamed>::distance_batched_kernel(const f… | 811.93 ms | 5.1% | 5 | 162385.6 us |
| sd::<unnamed>::dtw_band_kernel(int, int, int, … | 448.95 ms | 2.8% | 5 | 89790.7 us |
| sd::<unnamed>::update_h_batched_kernel(float *… | 99.98 ms | 0.6% | 600 | 166.6 us |
| sd::<unnamed>::row_sum_batched_kernel(const fl… | 38.55 ms | 0.2% | 600 | 64.3 us |
| sd::<unnamed>::col_sum_batched_kernel(const fl… | 32.48 ms | 0.2% | 600 | 54.1 us |
| sd::<unnamed>::update_w_batched_kernel(float *… | 28.09 ms | 0.2% | 600 | 46.8 us |

## Diagnosis — top 3 bottlenecks

1. **GEMM** — 8,194.96 ms (40.6% of wall; 1,638.99 ms per candidate).
2. **NMF elementwise** — 6,525.60 ms (32.3% of wall; 1,305.12 ms per candidate).
3. **GPU idle (est.)** — 4,189.87 ms (20.8% of wall; 837.97 ms per candidate).

## What to optimize next

- GEMM is a large share: check matrix shapes per NMF iteration — batched cublasSgemmStridedBatched across candidates/pitches usually beats many small GEMMs.
