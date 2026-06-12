/* Row-major strided-batched GEMM wrappers over cuBLAS (NVIDIA built-in; custom
 * GEMM kernels were removed in the cleanup — see TODO.md / CLAUDE.md log).
 * All matrices row-major; "batched" = P independent problems at fixed strides. */

#pragma once

#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>

namespace sd {

#define checkCublas(ans) \
    { sd::cublas_assert((ans), __FILE__, __LINE__); }

inline void cublas_assert(cublasStatus_t code, const char* file, int line) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS Error: code %d %s:%d\n", (int)code, file, line);
        std::exit(1);
    }
}

cublasHandle_t cublas_handle();

/* C[b] (MxN) = A[b] (MxK) * B[b] (KxN) */
void gemm_nn_batched(int M, int N, int K, const float* A, long long sA, const float* B,
                     long long sB, float* C, long long sC, int P);

/* C[b] (MxN) = A[b]^T * B[b],  A stored (KxM), B stored (KxN) */
void gemm_tn_batched(int M, int N, int K, const float* A, long long sA, const float* B,
                     long long sB, float* C, long long sC, int P);

/* C[b] (MxN) = A[b] * B[b]^T,  A stored (MxK), B stored (NxK) */
void gemm_nt_batched(int M, int N, int K, const float* A, long long sA, const float* B,
                     long long sB, float* C, long long sC, int P);

} /* namespace sd */
