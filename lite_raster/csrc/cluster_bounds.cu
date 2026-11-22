// Exact AABB cache for consecutive 128-row clusters.
// Reuse the production frustum kernel with two AABB endpoints per cluster.
#include "cluster_cull.cu"
#include <cub/block/block_scan.cuh>

namespace {
__global__ void refresh(const float3 *means, const float3 *log_scales, int n, const int *dirty, float3 *points,
                        float3 *scales) {
    using Reduce = cub::BlockReduce<Bounds, Threads>;
    __shared__ typename Reduce::TempStorage shared;
    int block = dirty ? dirty[blockIdx.x] : blockIdx.x;
    int i = block * Threads + threadIdx.x;
    float3 pos = means[min(i, n - 1)], scale = log_scales[min(i, n - 1)];
    Bounds b = Reduce(shared).Reduce({pos, pos, fmaxf(scale.x, fmaxf(scale.y, scale.z))}, Union{});
    if (!threadIdx.x) {
        float maximum = expf(b.scale);
        points[2 * block] = b.lo;
        points[2 * block + 1] = b.hi;
        scales[2 * block] = scales[2 * block + 1] = make_float3(maximum, maximum, maximum);
    }
}

__global__ void compact_blocks(const bool *mask, int blocks, int n, int *selected, int *count) {
    using Scan = cub::BlockScan<int, 256>;
    __shared__ typename Scan::TempStorage shared;
    int base = 0;
    for (int start = 0; start < blocks; start += blockDim.x) {
        int i = start + threadIdx.x;
        int value = i < blocks && mask[i], offset, kept;
        Scan(shared).ExclusiveSum(value, offset, kept);
        if (value)
            selected[base + offset] = i;
        base += kept;
        __syncthreads();
    }
    if (!threadIdx.x)
        *count = base * Threads - (mask[blocks - 1] ? blocks * Threads - n : 0);
}

__global__ void expand_blocks(const int *selected, int n, int64_t *ids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        ids[i] = selected[i / Threads] * Threads + i % Threads;
}
} // namespace
extern "C" int refresh_bounds(const float3 *means, const float3 *scales, int n, const int *dirty, int blocks,
                              float3 *points, float3 *maximum, cudaStream_t stream) {
    if (blocks)
        refresh<<<blocks, Threads, 0, stream>>>(means, scales, n, dirty, points, maximum);
    return cudaGetLastError();
}
extern "C" int select_blocks(const bool *mask, int blocks, int n, int *selected, int *count,
                             cudaStream_t stream) {
    compact_blocks<<<1, 256, 0, stream>>>(mask, blocks, n, selected, count);
    return cudaGetLastError();
}
extern "C" int expand_rows(const int *selected, int n, int64_t *ids, cudaStream_t stream) {
    if (n)
        expand_blocks<<<(n + 255) / 256, 256, 0, stream>>>(selected, n, ids);
    return cudaGetLastError();
}
