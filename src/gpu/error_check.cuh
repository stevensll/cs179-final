#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cufft.h>

/* CUDA error check macro, exits on failure. Pattern from CS179 lab3
 * (ErrorCheck.cuh), via https://stackoverflow.com/a/14038590 */
#define checkCuda(ans) { sd::gpu_assert((ans), __FILE__, __LINE__); }
#define checkCufft(ans) { sd::cufft_assert((ans), __FILE__, __LINE__); }

namespace sd {

inline void gpu_assert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error: %s %s:%d\n", cudaGetErrorString(code), file, line);
        std::exit(1);
    }
}

inline void cufft_assert(cufftResult code, const char* file, int line) {
    if (code != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cuFFT Error: code %d %s:%d\n", (int)code, file, line);
        std::exit(1);
    }
}

}  /* namespace sd */
