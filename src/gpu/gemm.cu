#include "gemm.cuh"

#include <cuda_runtime.h>

#include <mutex>

#include "error_check.cuh"

namespace sd {

/* One handle per device (handles are device-bound), created lazily and
 * thread-safely (one worker thread per GPU in gpu_detect). TF32 tensor-core
 * math is enabled: ~2-3x on Ampere GEMMs; NMF is approximate by nature and
 * the verification ladder gates the precision change. */
cublasHandle_t cublas_handle() {
    static cublasHandle_t handles[16] = {};
    static std::mutex mu;
    int dev = 0;
    checkCuda(cudaGetDevice(&dev));
    if (!handles[dev]) {
        std::lock_guard<std::mutex> lock(mu);
        if (!handles[dev]) {
            cublasHandle_t h;
            checkCublas(cublasCreate(&h));
            checkCublas(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
            handles[dev] = h;
        }
    }
    return handles[dev];
}

/* Row-major C = op(A)*op(B) maps to column-major C^T = op(B)^T * op(A)^T; every
 * row-major matrix viewed as column-major is its own transpose, which fixes the
 * operand order, op flags, and leading dimensions below.
 *
 * GemmEx with an explicit CUBLAS_COMPUTE_32F_FAST_TF32 is used instead of
 * Sgemm: the math-mode setting alone is advisory and the heuristic kept the
 * large NN gemms on fp32 SIMT kernels (profiled: plots/01_tf32_dtw). */

static void gemm_ex(cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k,
                    const float* A, int lda, long long sA, const float* B, int ldb, long long sB,
                    float* C, int ldc, long long sC, int P) {
    const float one = 1.f, zero = 0.f;
    checkCublas(cublasGemmStridedBatchedEx(cublas_handle(), opA, opB, m, n, k, &one, A, CUDA_R_32F,
                                           lda, sA, B, CUDA_R_32F, ldb, sB, &zero, C, CUDA_R_32F,
                                           ldc, sC, P, CUBLAS_COMPUTE_32F_FAST_TF32,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_nn_batched(int M, int N, int K, const float* A, long long sA, const float* B,
                     long long sB, float* C, long long sC, int P) {
    gemm_ex(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, B, N, sB, A, K, sA, C, N, sC, P);
}

void gemm_tn_batched(int M, int N, int K, const float* A, long long sA, const float* B,
                     long long sB, float* C, long long sC, int P) {
    gemm_ex(CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, B, N, sB, A, M, sA, C, N, sC, P);
}

void gemm_nt_batched(int M, int N, int K, const float* A, long long sA, const float* B,
                     long long sB, float* C, long long sC, int P) {
    gemm_ex(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, B, K, sB, A, K, sA, C, N, sC, P);
}

} /* namespace sd */
