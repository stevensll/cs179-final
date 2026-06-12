/* rf_infer — FIL-style random-forest inference on GPU, with a verification
 * harness against sklearn (paper stage 3.4; see docs/PAPER-MAPPING.md).
 *
 * Loads the flat forest exported by tools/export_forest.py (SDRF binary:
 * SoA arrays feature/threshold/left/right, leaves carry the class-1
 * probability in the threshold slot), predicts every row of a verification
 * matrix, and compares against sklearn's predict_proba on the same rows.
 *
 * Usage: rf_infer <forest.bin> <rows.f32> <ref_probs.f32> <n_features>
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "../common/audio.hpp"
#include "device_buffer.cuh"
#include "error_check.cuh"

namespace {

constexpr int MAX_FEATURES = 32;
constexpr int BLOCK = 256;

/* One node = one 16-byte struct, loaded as a single int4 (FIL-style AoS).
 * The forest is ~25M nodes (~400 MB) at depth ~60: traversal is DRAM-bound,
 * so the win is one 128-bit transaction per node visit instead of four
 * scattered 32-bit loads from SoA arrays (measured 2x+ on the real model). */
struct __align__(16) PackedNode {
    int feature; /* -1 = leaf */
    float thr;   /* split threshold; for leaves: class-1 probability */
    int left, right;
};

/* One thread per row, each walking every tree root-to-leaf.
 *
 * Strategy: tree traversal is irregular (data-dependent branching), so the
 * win is NOT in the arithmetic — it is in (a) holding the row's features in
 * registers/local so the inner loop touches only node memory, (b) one
 * coalescable 128-bit load per node visit (PackedNode), and (c) the hot
 * upper levels of every tree staying resident in L2 across the whole grid.
 * Threads in a warp diverge per-node but issue overlapping loads; shared
 * memory cannot help because paths are unpredictable. The sklearn split
 * rule x[f] <= thr -> left is replicated exactly (thresholds exported as
 * floor32, see tools/export_forest.py); leaf probabilities are averaged
 * across trees, matching predict_proba. */
__global__ void forest_predict_kernel(const float* __restrict__ rows, int n_rows, int n_features,
                                      const PackedNode* __restrict__ nodes,
                                      const int* __restrict__ roots, int n_trees,
                                      float* __restrict__ probs_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) {
        return;
    }

    float x[MAX_FEATURES];
    for (int f = 0; f < n_features; f++) {
        x[f] = rows[(size_t)i * n_features + f];
    }

    float sum = 0.f;
    for (int t = 0; t < n_trees; t++) {
        int node = roots[t];
        PackedNode nd = nodes[node];
        while (nd.feature >= 0) {
            node = (x[nd.feature] <= nd.thr) ? nd.left : nd.right;
            nd = nodes[node];
        }
        sum += nd.thr;
    }
    probs_out[i] = sum / (float)n_trees;
}

struct Forest {
    int n_trees = 0, n_features = 0, n_nodes = 0;
    std::vector<int> roots, feature, left, right;
    std::vector<float> thr;
};

Forest load_forest(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    char magic[4];
    int32_t version;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    if (memcmp(magic, "SDRF", 4) != 0 || version != 1) {
        fprintf(stderr, "%s: bad magic/version\n", path);
        exit(1);
    }
    Forest fo;
    f.read(reinterpret_cast<char*>(&fo.n_trees), 4);
    f.read(reinterpret_cast<char*>(&fo.n_features), 4);
    f.read(reinterpret_cast<char*>(&fo.n_nodes), 4);
    auto rd = [&](auto& v, size_t n) {
        v.resize(n);
        f.read(reinterpret_cast<char*>(v.data()), n * sizeof(v[0]));
    };
    rd(fo.roots, fo.n_trees);
    rd(fo.feature, fo.n_nodes);
    rd(fo.thr, fo.n_nodes);
    rd(fo.left, fo.n_nodes);
    rd(fo.right, fo.n_nodes);
    if (!f) {
        fprintf(stderr, "%s: truncated\n", path);
        exit(1);
    }
    return fo;
}

std::vector<float> load_f32(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    size_t bytes = f.tellg();
    f.seekg(0);
    std::vector<float> v(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

} /* namespace */

int main(int argc, char** argv) {
    sd::enforce_time_limit(300);
    if (argc != 5) {
        fprintf(stderr, "usage: %s <forest.bin> <rows.f32> <ref_probs.f32> <n_features>\n",
                argv[0]);
        return 1;
    }
    Forest fo = load_forest(argv[1]);
    std::vector<float> rows = load_f32(argv[2]);
    std::vector<float> ref = load_f32(argv[3]);
    int n_features = atoi(argv[4]);
    if (n_features != fo.n_features || n_features > MAX_FEATURES) {
        fprintf(stderr, "feature count mismatch: forest %d, arg %d (max %d)\n", fo.n_features,
                n_features, MAX_FEATURES);
        return 1;
    }
    int n_rows = (int)(rows.size() / n_features);
    if ((size_t)n_rows != ref.size()) {
        fprintf(stderr, "row count mismatch: %d rows vs %zu reference probs\n", n_rows, ref.size());
        return 1;
    }
    printf("forest: %d trees, %d nodes, %d features; %d verification rows\n", fo.n_trees,
           fo.n_nodes, fo.n_features, n_rows);

    std::vector<PackedNode> packed(fo.n_nodes);
    for (int n = 0; n < fo.n_nodes; n++) {
        packed[n] = {fo.feature[n], fo.thr[n], fo.left[n], fo.right[n]};
    }

    sd::DeviceBuffer<float> d_rows, d_out(n_rows);
    sd::DeviceBuffer<PackedNode> d_nodes;
    sd::DeviceBuffer<int> d_roots;
    d_rows.to_device(rows);
    d_nodes.to_device(packed);
    d_roots.to_device(fo.roots);

    int grid = (n_rows + BLOCK - 1) / BLOCK;
    /* warm-up launch, then timed repeats for a stable throughput figure */
    forest_predict_kernel<<<grid, BLOCK>>>(d_rows.ptr(), n_rows, n_features, d_nodes.ptr(),
                                           d_roots.ptr(), fo.n_trees, d_out.ptr());
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    constexpr int REPS = 10;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < REPS; r++) {
        forest_predict_kernel<<<grid, BLOCK>>>(d_rows.ptr(), n_rows, n_features, d_nodes.ptr(),
                                               d_roots.ptr(), fo.n_trees, d_out.ptr());
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() /
        REPS;

    std::vector<float> got = d_out.to_host(n_rows);
    double max_diff = 0, sum_diff = 0;
    int n_above_1e4 = 0, worst = 0;
    for (int i = 0; i < n_rows; i++) {
        double d = fabs((double)got[i] - (double)ref[i]);
        sum_diff += d;
        if (d > max_diff) {
            max_diff = d;
            worst = i;
        }
        if (d > 1e-4) {
            n_above_1e4++;
        }
    }
    printf("kernel: %.3f ms per pass = %.1f M rows/s (%d trees each)\n", ms, n_rows / ms / 1e3,
           fo.n_trees);
    printf(
        "vs sklearn: max |diff| %.3g (row %d: gpu %.6f ref %.6f), "
        "mean %.3g, rows >1e-4: %d/%d\n",
        max_diff, worst, got[worst], ref[worst], sum_diff / n_rows, n_above_1e4, n_rows);
    printf(max_diff <= 1e-4 ? "PARITY OK\n" : "PARITY FAIL\n");
    return max_diff <= 1e-4 ? 0 : 2;
}
