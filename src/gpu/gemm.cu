#include "gemm.cuh"

namespace sd {

cublasHandle_t cublas_handle() {
    static cublasHandle_t h = [] {
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle));
        return handle;
    }();
    return h;
}

/* Row-major C = op(A)*op(B) maps to column-major C^T = op(B)^T * op(A)^T; every
 * row-major matrix viewed as column-major is its own transpose, which fixes the
 * operand order, op flags, and leading dimensions below. */

void gemm_nn_batched(int M, int N, int K,
                     const float* A, long long sA,
                     const float* B, long long sB,
                     float* C, long long sC, int P) {
    const float one = 1.f, zero = 0.f;
    checkCublas(cublasSgemmStridedBatched(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                                          N, M, K, &one,
                                          B, N, sB, A, K, sA, &zero, C, N, sC, P));
}

void gemm_tn_batched(int M, int N, int K,
                     const float* A, long long sA,
                     const float* B, long long sB,
                     float* C, long long sC, int P) {
    const float one = 1.f, zero = 0.f;
    checkCublas(cublasSgemmStridedBatched(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                                          N, M, K, &one,
                                          B, N, sB, A, M, sA, &zero, C, N, sC, P));
}

void gemm_nt_batched(int M, int N, int K,
                     const float* A, long long sA,
                     const float* B, long long sB,
                     float* C, long long sC, int P) {
    const float one = 1.f, zero = 0.f;
    checkCublas(cublasSgemmStridedBatched(cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                                          N, M, K, &one,
                                          B, K, sB, A, K, sA, &zero, C, N, sC, P));
}

}  /* namespace sd */
