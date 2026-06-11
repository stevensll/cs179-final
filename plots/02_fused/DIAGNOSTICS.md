# Profile diagnostics — 02_fused

Report: `/tmp/iter02.nsys-rep` | profiled run: 5-candidate library scan

## Headline numbers

- CUDA-active wall span: **14,138.56 ms** (wall span = first..last CUDA event (runtime API + GPU ops))
- Total GPU kernel time: **10,089.03 ms** (71.4% of wall)
- Total memcpy/memset time: **61.79 ms** (0.4% of wall)
- Estimated GPU idle: **3,987.74 ms** (28.2% of wall) — wall minus GPU busy; overlap between kernels/copies would shrink busy time, so idle is a lower bound
- cudaLaunchKernel: **3,555 calls**, 22.00 ms CPU time (~6.2 us/call)
- cudaDeviceSynchronize: 30 calls, 10,043.35 ms CPU time

## Category split

| category | time | % of wall | per candidate |
|---|---:|---:|---:|
| NMF elementwise | 5,662.56 ms | 40.1% | 1,132.51 ms |
| GPU idle (est.) | 3,987.74 ms | 28.2% | 797.55 ms |
| GEMM | 3,136.32 ms | 22.2% | 627.26 ms |
| distance/znorm | 813.72 ms | 5.8% | 162.74 ms |
| DTW | 469.27 ms | 3.3% | 93.85 ms |
| memcpy/memset | 61.79 ms | 0.4% | 12.36 ms |
| STFT | 6.74 ms | 0.0% | 1.35 ms |
| other kernels | 0.42 ms | 0.0% | 0.08 ms |

## Top kernels

| kernel | total | % of kernel time | instances | avg |
|---|---:|---:|---:|---:|
| sd::<unnamed>::fused_wh_ratio_kernel(const flo… | 5,453.95 ms | 54.1% | 600 | 9089.9 us |
| cuBLAS GEMM [cutlass_80_tensorop_s1688gem] | 1,938.77 ms | 19.2% | 300 | 6462.6 us |
| cuBLAS GEMM [cutlass_80_tensorop_s1688gem] | 1,136.74 ms | 11.3% | 300 | 3789.1 us |
| sd::<unnamed>::distance_batched_kernel(const f… | 812.21 ms | 8.1% | 14 | 58014.7 us |
| sd::<unnamed>::dtw_band_kernel(int, int, int, … | 469.27 ms | 4.7% | 14 | 33519.0 us |
| sd::<unnamed>::update_h_batched_kernel(float *… | 99.48 ms | 1.0% | 600 | 165.8 us |
| sd::<unnamed>::row_sum_batched_kernel(const fl… | 38.56 ms | 0.4% | 600 | 64.3 us |
| sd::<unnamed>::col_sum_batched_kernel(const fl… | 32.30 ms | 0.3% | 600 | 53.8 us |
| sd::<unnamed>::update_w_batched_kernel(float *… | 28.23 ms | 0.3% | 600 | 47.1 us |
| cuBLAS GEMM [cutlass_80_tensorop_s1688gem] | 24.80 ms | 0.2% | 240 | 103.3 us |

Launches per candidate: ~711 (3,555 / 5).

## Memory operations

| op | time | count |
|---|---:|---:|
| [CUDA memcpy Host-to-Device] | 60.54 ms | 33 |
| [CUDA memcpy Device-to-Device] | 1.01 ms | 5 |
| [CUDA memset] | 0.16 ms | 120 |
| [CUDA memcpy Device-to-Host] | 0.09 ms | 14 |

## Diagnosis — top 3 bottlenecks

1. **NMF elementwise** — 5,662.56 ms (40.1% of wall; 1,132.51 ms per candidate).
2. **GPU idle (est.)** — 3,987.74 ms (28.2% of wall; 797.55 ms per candidate).
3. **GEMM** — 3,136.32 ms (22.2% of wall; 627.26 ms per candidate).

## What to optimize next

- GPU is idle ~28% of the wall span. Likely causes: serialized host loop over candidates, blocking synchronization between tiny kernels, or CPU-side work between launches. Consider CUDA streams (one per candidate), CUDA graphs, or batching candidates together.
- GEMM is a large share: check matrix shapes per NMF iteration — batched cublasSgemmStridedBatched across candidates/pitches usually beats many small GEMMs.
