# Performance characterization (2026-06-11, final optimized build)

Sources: ncu counter captures on the canary workload (Move On Up self-pair,
single GPU, --iters 4 for replay speed; kernel shapes identical to production),
nsys trace /tmp/iter04 (5-candidate music/ scan, --iters 60, single GPU),
wall-clock timings at --iters 60 on 2 GPUs with warm template cache.

## 1. End-to-end latency / throughput per candidate

| Workload (warm cache) | wall | per candidate |
|---|---:|---:|
| Full-song query (Ns=7181) vs 64 candidates, 2 GPUs | 60.2 s | 0.94 s wall (~1.9 s per GPU) |
| 5-candidate scan, 2 GPUs | 6.9 s | ~1.4 s |
| 15 s clip (Ns~320) vs 64 candidates, 2 GPUs | 4.9 s | 77 ms |

Throughput is linear in query frames (PFNMF dominates; all problems are
query-sized), which is why clips are ~12x cheaper than full songs.

## 2-5. Per-kernel counters (ncu)

| Kernel | share of GPU time | SM tput | mem pipe | DRAM | occupancy | dominant stalls (cyc/issue) |
|---|---:|---:|---:|---:|---:|---|
| fused_wh_ratio (PFNMF, z=41) | ~54% | 65.6% | 68.7% | 43.8% | 48.8% (smem-limited, theor. 50%) | long_scoreboard 2.3, not_selected 2.1 (of 8.9) |
| cuBLAS TN tensorop (s1688gemm) | ~19% | 31.7% | 78.4% | 52.7% | n/a | — (memory-bound) |
| distance_batched | ~8% | 61.3% | 95.3% (L1!) | 4.8% | 92.8% | L1/shared-bound, working set cache-resident |
| dtw_band | ~5% | 72.6% | 79.9% | 54.1% | 94.4% | long_scoreboard 11.0, barrier 3.4 (of ~19) |

Readings:
- **Tensor cores: 31.7% pipe utilization** on the TN GEMM — and that is the
  ceiling for this shape, not a deficiency: with n=60 the GEMM is
  bandwidth-bound (mem pipe 78%), so tensor units wait on loads. TF32 is
  engaged and working; feeding it faster would require a different algorithm
  shape, not a better kernel.
- **fused_wh_ratio** sits in the "well-balanced" regime per ncu (compute 66% /
  memory 69%): its occupancy is capped at 50% by the 32 KB of static shared
  memory per block, but the stall profile shows the latency is already mostly
  hidden (not_selected ~ long_scoreboard); the measured-neutral 128x64 tile
  experiment confirms there is little left on the table.
- **dtw_band** is global-latency bound (long_scoreboard 11 cyc/issue: the
  skewed-D reads) with a barrier component from its per-diagonal syncthreads —
  expected for wavefront DP; at ~5% of runtime not worth further work.
- **distance_batched** saturates L1/shared (95%) with DRAM nearly idle — its
  working set is cache-resident; fine.

## 6. The remaining ~6.4% GPU idle (616 ms on the profiled scan)

From the API trace:
- **GPU-side memcpy is NOT the cause**: actual transfer time is 44 ms (0.5%),
  423 H2D ops.
- **Synchronization is NOT the cause**: explicit syncs were removed; the large
  cudaMemcpy/cudaFree CPU times in the API table (7.2 s / 1.9 s) are blocking
  calls absorbing the wait for already-enqueued GPU work, i.e. host idle, not
  GPU idle. (cudaFree max 1.6 s = end-of-process teardown.)
- The idle is **host-side group setup between GPU bursts**: staging ~270 MB of
  pageable seeded-init slabs per candidate group for H2D, host scoring math
  (per-candidate min/median scans), and template-cache file reads. Pinned
  (page-locked) staging buffers + async H2D on a side stream would reclaim
  most of it; bounded upside is the full 6.4%, so it is logged as a TODO, not
  done.
