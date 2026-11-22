// Conservative cluster frustum test for the circle used by fused_projection.
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cmath>
#include <cstdint>

namespace {
constexpr int Threads = 128;
struct Bounds {
    float3 lo, hi;
    float scale;
};

struct Union {
    __device__ Bounds operator()(const Bounds &a, const Bounds &b) const {
        return {make_float3(fminf(a.lo.x, b.lo.x), fminf(a.lo.y, b.lo.y), fminf(a.lo.z, b.lo.z)),
                make_float3(fmaxf(a.hi.x, b.hi.x), fmaxf(a.hi.y, b.hi.y), fmaxf(a.hi.z, b.hi.z)),
                fmaxf(a.scale, b.scale)};
    }
};

__device__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Upper support of the world-space AABB in a camera-space plane.
__device__ float upper(float3 p, const float *view, float3 c, float3 e) {
    float3 n = make_float3(p.x * view[0] + p.y * view[4] + p.z * view[8],
                           p.x * view[1] + p.y * view[5] + p.z * view[9],
                           p.x * view[2] + p.y * view[6] + p.z * view[10]);
    float t = p.x * view[3] + p.y * view[7] + p.z * view[11];
    float extent = fabsf(n.x) * e.x + fabsf(n.y) * e.y + fabsf(n.z) * e.z;
    // Round outwards, including cancellation between world coordinates and translation.
    float error =
        fabsf(p.x) *
            (fabsf(view[0] * c.x) + fabsf(view[1] * c.y) + fabsf(view[2] * c.z) + fabsf(view[3])) +
        fabsf(p.y) *
            (fabsf(view[4] * c.x) + fabsf(view[5] * c.y) + fabsf(view[6] * c.z) + fabsf(view[7])) +
        fabsf(p.z) *
            (fabsf(view[8] * c.x) + fabsf(view[9] * c.y) + fabsf(view[10] * c.z) + fabsf(view[11]));
    return dot3(n, c) + t + extent + 4.e-6f * (error + extent + 1.f);
}

__global__ void cull(const float3 *means, const float3 *scales, const int64_t *order, const float *view,
                     const float *k, int n, int size, int w, int h, float eps, bool *visible) {
    using Reduce = cub::BlockReduce<Bounds, Threads>;
    __shared__ typename Reduce::TempStorage shared;
    float3 inf = make_float3(INFINITY, INFINITY, INFINITY);
    Bounds b{inf, make_float3(-INFINITY, -INFINITY, -INFINITY), 0.f};
    int start = blockIdx.x * size;
    for (int i = start + threadIdx.x; i < start + min(size, n - start); i += Threads) {
        int row = order[i];
        float3 x = means[row], s = scales[row];
        b = Union{}(b, {x, x, fmaxf(fabsf(s.x), fmaxf(fabsf(s.y), fabsf(s.z)))});
    }
    b = Reduce(shared).Reduce(b, Union{});
    if (threadIdx.x != 0)
        return;
    float3 c = make_float3(.5f * (b.hi.x + b.lo.x), .5f * (b.hi.y + b.lo.y), .5f * (b.hi.z + b.lo.z));
    float3 e = make_float3(.5f * (b.hi.x - b.lo.x), .5f * (b.hi.y - b.lo.y), .5f * (b.hi.z - b.lo.z));
    float fx = k[0], fy = k[4], cx = k[2], cy = k[5];
    if (fx <= 0.f || fy <= 0.f) {
        visible[blockIdx.x] = true;
        return;
    }
    float lx = (fmaxf(fabsf(cx), fabsf(w - cx)) + .15f * w) / fx;
    float ly = (fmaxf(fabsf(cy), fabsf(h - cy)) + .15f * h) / fy;
    // Gershgorin bound for ||R||^2, also valid for slightly non-rigid view matrices.
    float stretch2 = 0.f;
    for (int i = 0; i < 3; ++i) {
        float sum = 0.f;
        for (int j = 0; j < 3; ++j) {
            sum += fabsf(view[4 * i] * view[4 * j] + view[4 * i + 1] * view[4 * j + 1] +
                         view[4 * i + 2] * view[4 * j + 2]);
        }
        stretch2 = fmaxf(stretch2, sum);
    }
    // radius*z <= 3*smax*||R||*max(fx,fy)*sqrt(1+lx^2+ly^2) + pad*z.
    // pad covers the blur, the .01 discriminant floor, and pixel rounding.
    float support = 3.01f * b.scale * fmaxf(fx, fy) * sqrtf(stretch2 * (1.f + lx * lx + ly * ly));
    float pad = 1.01f + 3.01f * sqrtf(fmaxf(0.f, eps) + .1f);
    bool outside = upper(make_float3(0, 0, 1), view, c, e) < .01f ||
                   upper(make_float3(0, 0, -1), view, c, e) < -1.e10f ||
                   upper(make_float3(fx, 0, cx + pad), view, c, e) < -support ||
                   upper(make_float3(-fx, 0, w - cx + pad), view, c, e) < -support ||
                   upper(make_float3(0, fy, cy + pad), view, c, e) < -support ||
                   upper(make_float3(0, -fy, h - cy + pad), view, c, e) < -support;
    visible[blockIdx.x] = !outside;
}
} // namespace

extern "C" int cluster_cull(const float3 *means, const float3 *scales, const int64_t *order,
                            const float *view, const float *k, int n, int size, int w, int h, float eps,
                            bool *visible, cudaStream_t stream) {
    cull<<<1 + (n - 1) / size, Threads, 0, stream>>>(means, scales, order, view, k, n, size, w, h, eps,
                                                     visible);
    return cudaGetLastError();
}
