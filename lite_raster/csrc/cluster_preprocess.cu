// Fused row gathering, vanilla activations, indexed SH and per-view density statistics.
#include "sh_color.cu"
#include <cmath>

namespace {
__global__ void geometry_fwd(const int64_t *ids, int n, const float3 *position, const float3 *scale,
                             const float4 *rot, const float *opacity, float3 *out_pos, float3 *out_scale,
                             float4 *out_rot, float *out_op, float *denom) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int row = ids[i];
    float3 s = scale[row];
    float4 q = rot[row];
    // ATen reduces four contiguous components with shuffle offsets 1, then 2.
    float xy = __fadd_rn(__fmul_rn(q.x, q.x), __fmul_rn(q.y, q.y));
    float zw = __fadd_rn(__fmul_rn(q.z, q.z), __fmul_rn(q.w, q.w));
    float length = sqrtf(__fadd_rn(xy, zw));
    float norm = fmaxf(length, 1.e-12f);
    out_pos[i] = position[row];
    out_scale[i] = make_float3(expf(s.x), expf(s.y), expf(s.z));
    out_rot[i] = make_float4(q.x / norm, q.y / norm, q.z / norm, q.w / norm);
    out_op[i] = 1.f / (1.f + expf(-opacity[row]));
    denom[i] = length;
}

__global__ void geometry_bwd(int n, const float3 *scale, const float4 *rot, const float *opacity,
                             const float *length, const float3 *ds, const float4 *dr, const float *dop,
                             float3 *gs, float4 *gr, float *go) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (ds) {
        float3 s = scale[i], d = ds[i];
        gs[i] = make_float3(s.x * d.x, s.y * d.y, s.z * d.z);
    }
    if (dr) {
        float4 q = rot[i], d = dr[i];
        float norm = fmaxf(length[i], 1.e-12f);
        float dot = length[i] >= 1.e-12f ? q.x * d.x + q.y * d.y + q.z * d.z + q.w * d.w : 0.f;
        gr[i] = make_float4((d.x - q.x * dot) / norm, (d.y - q.y * dot) / norm, (d.z - q.z * dot) / norm,
                            (d.w - q.w * dot) / norm);
    }
    if (dop)
        go[i] = dop[i] * (1.f - opacity[i]) * opacity[i];
}

// Load only the active bands of projected rows; no full compact SH copy.
template <int Degree> __global__ void indexed_sh(Buffers a, const int64_t *ids, int n, int capacity) {
    constexpr int width = ((Degree + 1) * (Degree + 1) - 1) * 3;
    constexpr int pitch = width | 1;
    __shared__ float storage[4][32 * pitch];
    int i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x % 32;
    int first = i - lane;
    if (first >= n)
        return;
    int row = i < n ? ids[i] : 0;
    bool vis = i < n && a.visible[i];
    unsigned mask = __ballot_sync(0xffffffff, vis);
    float *tile = storage[threadIdx.x / 32];
    if constexpr (width) {
        for (int j = lane; j < 32 * width; j += 32) {
            int r = j / width, c = j % width;
            int source = __shfl_sync(0xffffffff, row, r);
            if (mask & (1u << r))
                tile[r * pitch + c] = a.rest[int64_t(source) * (capacity - 1) * 3 + c];
        }
        __syncwarp();
    }
    if (i < n)
        evaluate<Degree, false>(a, i, capacity, a.dc + int64_t(row) * 3, tile + lane * pitch);
}

__global__ void update_stats(const int64_t *ids, int n, const int *tiles, const int *radii, const float2 *grad,
                             const float *scale, float *maxrad, float *acc, float *den) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n || tiles[i] <= 0)
        return;
    int row = ids ? ids[i] : i;
    float dx = grad[i].x * (scale ? scale[0] : 1.f), dy = grad[i].y * (scale ? scale[1] : 1.f);
    maxrad[row] = fmaxf(maxrad[row], float(radii[i]));
    acc[row] += sqrtf(dx * dx + dy * dy);
    den[row] += 1.f;
}
} // namespace
extern "C" int geometry(void **p, int n, bool back, cudaStream_t st) {
    if (!n)
        return 0;
    if (!back)
        geometry_fwd<<<(n + 127) / 128, 128, 0, st>>>(
            (int64_t *)p[0], n, (float3 *)p[1], (float3 *)p[2], (float4 *)p[3], (float *)p[4], (float3 *)p[5],
            (float3 *)p[6], (float4 *)p[7], (float *)p[8], (float *)p[9]);
    else
        geometry_bwd<<<(n + 127) / 128, 128, 0, st>>>(
            n, (float3 *)p[0], (float4 *)p[1], (float *)p[2], (float *)p[3], (float3 *)p[4], (float4 *)p[5],
            (float *)p[6], (float3 *)p[7], (float4 *)p[8], (float *)p[9]);
    return cudaGetLastError();
}
extern "C" int colors(void **p, int n, int degree, int capacity, cudaStream_t st) {
    if (!n)
        return 0;
    Buffers a{(const float3 *)p[0], (float *)p[1], (float *)p[2],
              (const bool *)p[3],   (float *)p[4], (uint8_t *)p[5]};
#define SH(D)                                                                                                  \
    case D:                                                                                                    \
        indexed_sh<D><<<(n + 127) / 128, 128, 0, st>>>(a, (int64_t *)p[6], n, capacity);                       \
        break
    switch (degree) {
        SH(0);
        SH(1);
        SH(2);
        SH(3);
        SH(4);
    default:
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
extern "C" int stats(void **p, int n, cudaStream_t st) {
    if (n)
        update_stats<<<(n + 127) / 128, 128, 0, st>>>((int64_t *)p[0], n, (int *)p[1], (int *)p[2],
                                                      (float2 *)p[3], (float *)p[4], (float *)p[5],
                                                      (float *)p[6], (float *)p[7]);
    return cudaGetLastError();
}

#include "adam.cuh"

namespace {
struct ColorGroups { lite_raster::AdamGroup dc, rest; };

template<int Degree>
__global__ void color_backward_step(Buffers a, const int64_t *ids, ColorGroups groups, int n, int capacity) {
    constexpr int bands = (Degree + 1) * (Degree + 1);
    constexpr int active_width = (bands - 1) * 3, pitch = active_width | 1;
    __shared__ float tile[4][32][pitch];
    int i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32, start = i - lane;
    if (start >= n) return;
    int row = i < n ? ids[i] : 0;
    if (i < n) {
        float dc[3];
        evaluate<Degree, true>(a, i, bands, dc, tile[warp_id][lane]);
        for (int c = 0; c < 3; ++c) adam_update(groups.dc, int64_t(row)*3+c, dc[c]);
    }
    __syncwarp();
    // All lanes participate in the shuffle, including the last partial warp.
    int width = (capacity - 1) * 3;
    for (int j = lane; j < 32*width; j += 32) {
        int r = j / width, c = j % width;
        int dst = __shfl_sync(0xffffffff, row, r);
        if (start + r < n) {
            float grad = c < active_width ? tile[warp_id][r][c] : 0.f;
            adam_update(groups.rest, int64_t(dst)*width+c, grad);
        }
    }
}
} // namespace

extern "C" int colors_adam(void **p, const lite_raster::AdamGroup *groups, int n, int degree,
                           int capacity, cudaStream_t stream) {
    if (!n) return cudaSuccess;
    Buffers a{(const float3 *)p[1], nullptr, nullptr, nullptr, (float *)p[3], (uint8_t *)p[2]};
    ColorGroups sh{groups[0], groups[1]};
#define STEP(d) case d: color_backward_step<d><<<(n+127)/128,128,0,stream>>>(a,(int64_t *)p[0],sh,n,capacity); break
    switch (degree) {
        STEP(0); STEP(1); STEP(2); STEP(3); STEP(4);
        default: return cudaErrorInvalidValue;
    }
#undef STEP
    return cudaGetLastError();
}
