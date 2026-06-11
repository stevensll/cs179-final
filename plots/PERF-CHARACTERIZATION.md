# Performance characterization (2026-06-11, final build: log-freq + K=32 + RF)

Sources: nsys trace /tmp/iter06 (5-candidate scan, --iters 60, single GPU,
warm tcache; analyzed in plots/06_fil/), nsys /tmp/fil (rf_infer over 500k
rows), ncu counter captures on the canary workload (Move On Up self-pair,
--iters 4 for replay speed — kernel shapes identical to production) and on
rf_infer. Wall-clock figures at --iters 60, 2 GPUs, warm caches.
Previous-build characterization (pre-mel, 04_final) in git history.

## 1. End-to-end latency / throughput

| Workload (warm caches, 2 GPUs) | wall | per candidate |
|---|---:|---:|
| 5-candidate full-song scan | 3.0 s | ~0.6 s |
| Full-song query vs 64 candidates | 19.3 s | 0.30 s wall (~0.6 s per GPU) |
| 15 s clip vs 64 candidates | 2.3 s | 36 ms |
| RF inference (rf_infer, 1 GPU) | 163 ms / 500k rows | **3.1 M rows/s** (200 trees) |
| Full 70-query benchmark | ~13 min | — |

Throughput is linear in query frames (PFNMF dominates and every problem is
query-sized); clips are ~17× cheaper than full songs. RF inference over the
whole 17.3M-row corpus: ~5.6 s — vs sklearn predict_proba at 0.16 M rows/s
on all CPU cores (19×).

## 2–5. Per-kernel counters (ncu)

Share-of-kernel-time from the production-config scan trace; counters from the
canary captures.

| Kernel | share | SM tput | mem pipe | DRAM | occupancy (ach/theor) | top stalls (cyc/issue) |
|---|---:|---:|---:|---:|---:|---|
| fused_wh_ratio (PFNMF) | 27.7% | 64.8% | 69.5% | 50.9% | 48.1 / 50% (smem-limited) | long_scoreboard 2.3, not_selected 2.0 |
| distance_batched | 26.3% | 59.1% | **93.1% (L1!)** | 5.7% | 92.4 / 100% | lg_throttle 10.3, long_scoreboard 8.0 |
| dtw_band | 19.2% | 72.5% | 80.2% | 54.3% | 94.6 / 100% | long_scoreboard 11.0, barrier 3.4 |
| cuBLAS TN tensorop (64×64) | 13.0% | 21.3% | 44.8% | 43.0% | 8.3% (canary-sized grid) | tensor pipe **31.7%** |
| cuBLAS NT tensorop (256×64) | 8.7% | 44.7% | 79.3% | 79.3% | 8.3% | tensor pipe **45.8%** |
| forest_predict (rf_infer) | (own binary) | 4.0% | 35.3% | 35.3% | 82.0 / 100% | **long_scoreboard 452** |

Readings:

- **Post-mel rebalance**: PFNMF's fused kernel (27.7%) now shares the top with
  distance_batched (26.3%) and dtw_band (19.2%) — the time-axis stages the
  log-frequency front end did *not* shrink, exactly as plots/05_mel predicted.
- **fused_wh_ratio** stays "well-balanced" (compute 65% / memory 69%):
  occupancy capped at 50% by its 32 KB static shared memory, but stalls show
  latency mostly hidden (not_selected ≈ long_scoreboard). The measured-neutral
  128×64 tile experiment confirmed nothing is left on this table.
- **distance_batched** saturates L1/shared (93%) with DRAM nearly idle (5.7%)
  — working set cache-resident; the lg_throttle stall (LSU queue) is the
  signature of its skewed-layout scattered stores. Fine at this share.
- **dtw_band** is global-latency bound (long_scoreboard 11: the skewed-D
  anti-diagonal reads) with the expected wavefront barrier component.
- **Tensor cores: 31.7% / 45.8% pipe utilization** — that is the ceiling for
  these skinny-k shapes, not a deficiency: both GEMMs are bandwidth-bound
  (DRAM 43–79%), so tensor units wait on loads. TF32 is engaged and working.
- **forest_predict is the purest latency-bound kernel in the project**:
  long_scoreboard 452 cycles/issue — every node visit is a dependent load
  into a 400 MB forest, so neither compute (4%) nor DRAM bandwidth (35%) is
  the wall; *latency* is. This is why the packed 16-byte node (one 128-bit
  load per visit instead of four scattered 32-bit loads) bought 3.0× — it
  quarters the number of latency chains. Known further upside (not done):
  walking 2–4 trees per thread interleaved would overlap chains; breadth-
  first placement of hot top levels would improve L2 hit rate.

## 6. GPU idle on the production scan (single GPU)

Analyzer estimate: 16.9% of the CUDA-active span (534 ms of 3.16 s) — grown
as a *share* from 6.4% on the pre-mel build because mel cut GPU work ~5.6×
while host-side per-candidate work stayed constant (absolute idle is similar).
Decomposition of the >50 µs gaps (317 ms, 10.8%), classified by the host
activity overlapping each gap:

| cause | time | share of wall |
|---|---:|---:|
| blocking pageable H2D staging (seeded-init slabs, cudaMemcpy) | 179 ms | 6.1% |
| one-time CUDA module load at startup (cuLibraryLoadData) | 110 ms | 3.7% |
| cudaMalloc / cudaFree (grow-only scratch, teardown) | 23 ms | 0.8% |
| kernel-launch overhead (350 launches × ~7.6 µs) | ~3 ms | 0.1% |
| sub-50 µs micro-gaps between the ~70 launches/candidate | ~217 ms | ~6% |

So: **not** synchronization (explicit syncs were removed), **not** D2H (0.1 ms
total), and only trivially launch overhead. The two actionable items remain
the known TODOs: pinned staging buffers + async H2D for the 6.1%, and the
micro-gaps would need CUDA graphs or multi-stream overlap — bounded upside
~12% single-GPU, less end-to-end (the second GPU already overlaps much of it).

## Artifacts

plots/06_fil/: kernels.png, categories.png, api_overhead.png, summary.csv,
DIAGNOSTICS.md (nsys breakdown), ncu_*.txt (raw counter captures).
Progression across all optimization iterations: plots/speedup_progression.png.
