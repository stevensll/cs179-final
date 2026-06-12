# C++/CUDA Style Guide

**Base standard: the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)**,
enforced mechanically by the checked-in `.clang-format` (Google base +
`InsertBraces`). Run `clang-format -i` on touched files. On top of Google:

- **Braces on every control-flow body** — no brace-less or single-line
  `if`/`for`/`while`, no multi-statement lines. Expand it out.
- **No compact C idioms**: `strcmp(a, b) == 0`, never `!strcmp(a, b)`;
  prefer one declaration per line where it aids reading.
- Documented deviations from Google (deliberate): 4-space indent, 100-column
  limit, `snake_case` function names (CS 179 lab convention), `/* ... */`
  comments.

The rest of this file holds the repo-specific conventions, distilled from
Steven's CS 179 lab style (lab3 source, lab4 operating notes). When in doubt:
Google style first, then match what the labs did.

## Naming

- `snake_case` for everything we own: functions, variables, files
  (`nmf_update.cu`, `compute_distance_matrix`).
- Kernels get a `_kernel` suffix: `update_h_batched_kernel`, `dtw_band_diag_kernel`,
  `pitch_templates_batched_kernel`. The suffix makes launch sites obvious.
- Types are `PascalCase` (`DeviceBuffer`, `Spectrogram`, `DtwResult`).
- Constants are `UPPER_SNAKE` `constexpr`: `BLOCK_SIZE`, `STFT_HOP`, `NMF_RANK_K`.
- All magic numbers become named `constexpr` near the top of the file (or in the
  stage's header if shared). No bare `4096` in code.

## Comments

- `/* ... */` for both single- and multi-line comments.
- Comment sparingly — only the *why*, not the *what*. Skip narrating obvious code; do
  call out non-obvious algorithmic choices, GPU hardware alignment, or
  correctness-critical ordering.
- **Exception (spec-graded): every kernel carries a strategy block comment** above or
  at the top of its body covering:
  1. Decomposition — what one thread / one block owns.
  2. Shared memory — what is staged and why it fits.
  3. Why this beats the naive approach (memory traffic, divergence, occupancy).
  4. Lecture references where they apply (e.g. "reduction per lecture 7").

  Template, modeled on lab3's `cudaMaximumKernel`:

  ```c
  /* One block per anti-diagonal tile of the cost matrix C.
   *
   * 1) Each thread owns one cell of the current anti-diagonal; the previous
   *    two diagonals live in shared memory (double-buffered).
   * 2) Dependencies only cross tile borders at the halo columns, exchanged
   *    through global memory between wavefront steps.
   * 3) Beats one-thread-per-row scanning because all cells on a diagonal are
   *    independent (lecture 12, wavefront DP); global traffic drops from
   *    3 reads/cell to ~1 after shared staging.
   */
  __global__ void dtw_antidiagonal_kernel(...)
  ```

- CPU demo stages additionally carry a short comment stating the parallelization
  strategy the GPU version uses (the spec grades the CPU demo on this).

## Error handling

- `checkCuda(...)` wraps **every** CUDA API call. The macro lives in
  `src/gpu/error_check.cuh` (`sd::gpu_assert`: prints `CUDA Error: <msg> file:line`,
  exits 1; pattern from lab3's `ErrorCheck.cuh`, source attributed in the header).
  Companion macros: `checkCufft` (same header) and `checkCublas` (`gemm.cuh`).
- After **every** kernel launch: `checkCuda(cudaGetLastError());` (launch errors are
  silent otherwise). Synchronizing checks (`cudaDeviceSynchronize`) only at stage
  boundaries and in debug paths, not in hot loops.
- Host code: fail fast with a message to `stderr` and nonzero exit. No exceptions
  thrown across the host/device boundary; no exceptions in device code at all.
- Asserts in tests are plain `assert` or a tiny `CHECK(cond, msg)` macro — no
  frameworks.

## Memory management

- Device memory goes through a small RAII `DeviceBuffer<T>` (single header,
  `src/gpu/device_buffer.cuh`, ~60 lines): wraps `cudaMalloc`/`cudaFree`, exposes
  `.ptr()`, `.size()`, `to_device`/`to_host` transfer helpers, move-only, `checkCuda`
  internally. No raw `cudaMalloc` outside it.
- No Thrust, no CUB. Reductions/scans are hand-written kernels (that's the point of
  the course). GEMMs are the exception to "no library math": they go through the
  cuBLAS strided-batched wrappers in `src/gpu/gemm.cu(h)` (Steven's 2026-06-10
  kernel-scope decision; see `gpu-library-vs-custom-kernels.md` postscript) — no
  bare `cublas*` calls outside that pair.
- Host-side: `std::vector` for bulk data; no manual `new`/`delete`.
- Allocate once per pipeline run where possible; no alloc/free inside iteration loops
  (NMF updates reuse their workspaces).

## Kernel conventions

- Grid-stride loops for elementwise kernels (lab style):
  `for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)`.
- Block sizes are named `constexpr` (currently one shared `BLOCK = 256` in
  `kernels.cu`); when a kernel needs a different size, justify it in the strategy
  comment (A5000: 84 SMs, 48 KB default smem/block).
- Batch the 41 pitch shifts in the grid (z-dimension or stacked-problem index)
  rather than looping on the host, wherever the stage allows.
- Floating point is `float` throughout (matches the data; doubles only in test
  reference computations if needed for tolerance headroom).
- No dynamic parallelism, no managed memory; explicit H2D/D2H at pipeline entry/exit
  only — intermediates stay resident on device.

## Host/program structure

- C++17. Compile clean under `-Wall -Wextra` (host) — fix warnings, don't silence
  them.
- Paired `.cu`/`.cuh` files (kernels + launchers in `.cu`, declarations + launch
  wrappers in `.cuh`). Currently all custom kernels share one `kernels.cu(h)` pair
  (the per-stage split is a TODO.md cleanup item); cuBLAS wrappers in `gemm.cu(h)`,
  orchestration in `gpu_pipeline.cu(h)`. Host-only logic in `.cpp`.
- `src/common/` code must compile without CUDA (the CPU demo links it standalone).
- Every executable starts with `sd::enforce_time_limit(...)` (`src/common/audio.cpp`;
  watchdog-thread pattern after lab TA_Utilities — see CLAUDE.md, run discipline).
- Timing: `cudaEvent_t` pairs for GPU stages, `std::chrono::steady_clock` for CPU
  stages; report milliseconds.

## Formerly-open items, resolved by adopting Google style (2026-06-11)

- Line width: **100** (`.clang-format` ColumnLimit).
- Include order: own header first, then system/std, then project headers —
  clang-format sorts and groups automatically (`IncludeIsMainSourceRegex`
  makes `.cu` files claim their `.cuh` pair as own-header).
- `struct` for passive data aggregates, `class` when there are invariants
  (per Google) — matches existing usage (`Mat`, `MatchInfo`, `PathFeatures`
  are structs; `DeviceBuffer` is a class).
- Sizes/indices: `size_t` for memory sizes and offsets (cast once at the
  arithmetic boundary, as the kernels do), `int` for small counts and loop
  bounds that provably fit.
