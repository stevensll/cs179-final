#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "error_check.cuh"

namespace sd {

/* Minimal RAII device allocation. Move-only; all CUDA calls checked. */
template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { alloc(n); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { release(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
        return *this;
    }

    void alloc(size_t n) {
        release();
        checkCuda(cudaMalloc(&p_, n * sizeof(T)));
        n_ = n;
    }
    void release() {
        if (p_) checkCuda(cudaFree(p_));
        p_ = nullptr;
        n_ = 0;
    }

    void to_device(const T* src, size_t n) {
        checkCuda(cudaMemcpy(p_, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void to_device(const std::vector<T>& src) {
        if (n_ < src.size()) alloc(src.size());
        to_device(src.data(), src.size());
    }
    std::vector<T> to_host(size_t n) const {
        std::vector<T> out(n);
        checkCuda(cudaMemcpy(out.data(), p_, n * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }

    T* ptr() { return p_; }
    const T* ptr() const { return p_; }
    size_t size() const { return n_; }

  private:
    T* p_ = nullptr;
    size_t n_ = 0;
};

}  /* namespace sd */
