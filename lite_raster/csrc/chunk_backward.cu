// FP32 backward using eight 8x4 pixel chunks per 16x16 tile. The last contributor
// and transmittance allow back-to-front replay without per-splat checkpoints.
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <type_traits>
#include "raster_args.cuh"

namespace {
constexpr unsigned All = 0xffffffffu;
constexpr int Batch = 32;
constexpr int Chunks = 8;
constexpr int Warps = 4;

struct SplatData {
    float2 xy;
    float3 q, rgb;
    float opacity;
    int id;
};

template <typename Pixels> struct Tile {
    Pixels pixels;
    int last[256], last_chunk[Chunks], ticket[Chunks];
    unsigned hits[Chunks];
    SplatData splats[Batch];
};

__device__ float warp_sum(float x) {
#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1)
        x += __shfl_down_sync(All, x, shift);
    return x;
}

__device__ float sigma_at(float3 q, float dx, float dy) {
    float yy = __fmul_rn(__fmul_rn(q.z, dy), dy);
    float diag = __fmaf_rn(__fmul_rn(q.x, dx), dx, yy);
    return __fmaf_rn(__fmul_rn(q.y, dx), dy, __fmul_rn(.5f, diag));
}

// A convex quadratic attains its minimum in the box at its center (if inside)
// or on an edge. For an outside center only the separating edges need testing.
__device__ bool intersects(const SplatData &s, float opacity_bound, float x, float y) {
    float x1 = s.xy.x - x, x0 = x1 - 7.f, y1 = s.xy.y - y, y0 = y1 - 3.f;
    float dx = fmaxf(x0, fminf(x1, 0.f)), dy = fmaxf(y0, fminf(y1, 0.f));
    if (dx == 0.f && dy == 0.f)
        return true;
    float3 q = s.q;
    if (!(q.x > 0.f && q.z > 0.f && fmaf(q.x, q.z, -q.y * q.y) > 2.e-6f * q.x * q.z))
        return true;
    float lower = CUDART_INF_F;
    if (dx != 0.f)
        lower = sigma_at(q, dx, fmaxf(y0, fminf(y1, -q.y * dx / q.z)));
    if (dy != 0.f)
        lower = fminf(lower, sigma_at(q, fmaxf(x0, fminf(x1, -q.y * dy / q.x)), dy));
    float mx = fmaxf(fabsf(x0), fabsf(x1)), my = fmaxf(fabsf(y0), fabsf(y1));
    float slack = 2.e-5f * (1.f + q.x * mx * mx + 2.f * fabsf(q.y) * mx * my + q.z * my * my);
    return !(lower > slack && opacity_bound * __expf(slack - lower) < .99999f / 255.f);
}

template <bool Abs, bool Moments> struct RgbReplay {
    static constexpr bool hard = false;
    struct Pixels {
        float trans[256], suffix[256], upstream[3][256];
    };

    static __device__ int initialize(const Args &a, Pixels &tile, int slot, int x, int y) {
        const int area = a.width * a.height;
        const float *bg = static_cast<const float *>(a.in[Background]);
        int k = y * a.width + x;
        int last = static_cast<const int *>(a.cache[Last])[k];
        float t = static_cast<const float *>(a.cache[Trans])[k], suffix = 0.f;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            float v = static_cast<const float *>(a.dout[RGB])[c * area + k];
            tile.upstream[c][slot] = v;
            suffix += t * v * bg[c];
        }
        tile.trans[slot] = t;
        tile.suffix[slot] = suffix;
        return last;
    }

    // Weighted constant, first and quadratic terms. With Moments, the two first
    // terms hold the actual center gradient, reused for both output buffers.
    float scalars[6] = {};
    float color[3] = {}, abs_x = 0.f, abs_y = 0.f;
    float xx = 0.f, xy_cross = 0.f, yy = 0.f, samples = 0.f;

    __device__ void visit(const Args &, Pixels &tile, const SplatData &s, int k, int idx, float dx, float dy) {
        float sigma = sigma_at(s.q, dx, dy);
        float fall = __expf(-sigma), raw = s.opacity * fall, alpha = fminf(.999f, raw);
        if (sigma >= 0.f && alpha >= 1.f / 255.f) {
            float ra = 1.f / (1.f - alpha), t = tile.trans[k] * ra, w = t * alpha;
            float vx = tile.upstream[0][k], vy = tile.upstream[1][k],
                  vz = tile.upstream[2][k];
            float dot = fmaf(vz, s.rgb.z, fmaf(vy, s.rgb.y, vx * s.rgb.x));
            float suffix = tile.suffix[k];
            tile.trans[k] = t;
            tile.suffix[k] = fmaf(w, dot, suffix);
            color[0] += w * vx;
            color[1] += w * vy;
            color[2] += w * vz;
            if (raw <= .999f) {
                float d = fall * (t * dot - suffix * ra);
                scalars[0] += d;
                if constexpr (!Moments) {
                    scalars[1] += d * dx;
                    scalars[2] += d * dy;
                }
                scalars[3] += (d * dx) * dx;
                scalars[4] += (d * dx) * dy;
                scalars[5] += (d * dy) * dy;
                if constexpr (Abs || Moments) {
                    float ds = -s.opacity * d;
                    float gx = ds * (s.q.x * dx + s.q.y * dy);
                    float gy = ds * (s.q.y * dx + s.q.z * dy);
                    if constexpr (Abs) {
                        abs_x += fabsf(gx);
                        abs_y += fabsf(gy);
                    }
                    if constexpr (Moments) {
                        // These first moments are also the ordinary center gradient.
                        scalars[1] += gx;
                        scalars[2] += gy;
                        xx += gx * gx;
                        xy_cross += gx * gy;
                        yy += gy * gy;
                    }
                }
            }
            // A clamped alpha has zero center derivative but still contributed.
            if constexpr (Moments)
                samples += 1.f;
        }
    }

    __device__ void flush(const Args &a, const Pixels &, const SplatData &s) {
        int lane = threadIdx.x;
#pragma unroll
        for (int j = 0; j < 6; ++j)
            scalars[j] = warp_sum(scalars[j]);
#pragma unroll
        for (int j = 0; j < 3; ++j)
            color[j] = warp_sum(color[j]);
        if constexpr (Abs) {
            abs_x = warp_sum(abs_x);
            abs_y = warp_sum(abs_y);
        }
        if constexpr (Moments) {
            xx = warp_sum(xx);
            xy_cross = warp_sum(xy_cross);
            yy = warp_sum(yy);
            samples = warp_sum(samples);
        }
        if (lane == 0) {
            float *xy = static_cast<float *>(a.din[XY]) + 2 * s.id;
            float *conic = static_cast<float *>(a.din[Conic]) + 3 * s.id;
            float *rgb = static_cast<float *>(a.din[Color]) + 3 * s.id;
            float sign = -s.opacity;
            if constexpr (Moments) {
                float *stats = static_cast<float *>(a.moments) + 6 * s.id;
                atomicAdd(xy, scalars[1]);
                atomicAdd(xy + 1, scalars[2]);
                atomicAdd(stats, scalars[1]);
                atomicAdd(stats + 1, scalars[2]);
                atomicAdd(stats + 2, xx);
                atomicAdd(stats + 3, xy_cross);
                atomicAdd(stats + 4, yy);
                atomicAdd(stats + 5, samples);
            } else {
                atomicAdd(xy, sign * (s.q.x * scalars[1] + s.q.y * scalars[2]));
                atomicAdd(xy + 1, sign * (s.q.y * scalars[1] + s.q.z * scalars[2]));
            }
            atomicAdd(conic, .5f * sign * scalars[3]);
            atomicAdd(conic + 1, sign * scalars[4]);
            atomicAdd(conic + 2, .5f * sign * scalars[5]);
            atomicAdd(static_cast<float *>(a.din[Opacity]) + s.id, scalars[0]);
#pragma unroll
            for (int c = 0; c < 3; ++c)
                atomicAdd(rgb + c, color[c]);
            if constexpr (Abs) {
                atomicAdd(static_cast<float *>(a.absgrad) + 2 * s.id, abs_x);
                atomicAdd(static_cast<float *>(a.absgrad) + 2 * s.id + 1, abs_y);
            }
        }
    }
};

#include "chunk_maps.cuh"

template <typename Replay> __global__ void backward(Args a) {
    __shared__ Tile<typename Replay::Pixels> tile;
    int lane = threadIdx.x, warp = threadIdx.y;
    int x0 = blockIdx.x * 16, y0 = blockIdx.y * 16;
    int tile_index = blockIdx.y * gridDim.x + blockIdx.x;
    int begin = static_cast<const int *>(a.in[Offsets])[tile_index];
    for (int chunk = warp; chunk < Chunks; chunk += Warps) {
        int slot = chunk * 32 + lane;
        int x = x0 + (chunk % 2) * 8 + lane % 8, y = y0 + (chunk / 2) * 4 + lane / 8;
        int last = -1;
        if (x < a.width && y < a.height) {
            last = Replay::initialize(a, tile.pixels, slot, x, y);
        }
        tile.last[slot] = last;
        for (int shift = 16; shift > 0; shift >>= 1)
            last = max(last, __shfl_down_sync(All, last, shift));
        if (lane == 0)
            tile.last_chunk[chunk] = last;
    }
    __syncthreads();
    int end = -1;
#pragma unroll
    for (int chunk = 0; chunk < Chunks; ++chunk)
        end = max(end, tile.last_chunk[chunk]);
    for (int start = end; start >= begin; start -= Batch) {
        if (warp == 0 && start - lane >= begin) {
            int g = static_cast<const int *>(a.in[Ids])[start - lane];
            SplatData &s = tile.splats[lane];
            s.id = g;
            s.xy = static_cast<const float2 *>(a.in[XY])[g];
            s.q = static_cast<const float3 *>(a.in[Conic])[g];
            s.opacity = static_cast<const float *>(a.in[Opacity])[g];
            if constexpr (CHANNELS == 3)
                s.rgb = static_cast<const float3 *>(a.in[Color])[g];
        }
        __syncthreads();
        for (int chunk = warp; chunk < Chunks; chunk += Warps) {
            bool hit = start - lane >= begin && start - lane <= tile.last_chunk[chunk];
            if (hit) {
                const SplatData &s = tile.splats[lane];
                // Hard blending ignores the learned opacity.
                const float opacity_bound = Replay::hard ? fmaxf(1.f, s.opacity) : s.opacity;
                hit = intersects(s, opacity_bound, x0 + .5f + (chunk % 2) * 8,
                                 y0 + .5f + (chunk / 2) * 4);
            }
            unsigned hits = __ballot_sync(All, hit);
            if (lane == 0) {
                tile.hits[chunk] = hits;
                tile.ticket[chunk] = 0;
            }
        }
        __syncthreads();
        for (int i = warp; i < min(Batch, start - begin + 1); i += Warps) {
            SplatData s = tile.splats[i];
            Replay replay{};
            bool visited = false;
            for (int chunk = 0; chunk < Chunks; ++chunk) {
                unsigned hits = tile.hits[chunk];
                if (!(hits & (1u << i)))
                    continue;
                visited = true;
                // Earlier set bits refer to earlier splats in reverse replay.
                // Release/acquire plus the warp barriers publish all 32 pixels.
                int predecessor = 32 - __clz(hits & ((1u << i) - 1));
                cuda::atomic_ref<int, cuda::thread_scope_block> ticket(tile.ticket[chunk]);
                if (lane == 0)
                    while (ticket.load(cuda::memory_order_acquire) != predecessor) {
                    }
                __syncwarp();
                int k = 32 * chunk + lane;
                if (start - i <= tile.last[k]) {
                    float dx = s.xy.x - (x0 + (chunk % 2) * 8 + lane % 8 + .5f);
                    float dy = s.xy.y - (y0 + (chunk / 2) * 4 + lane / 8 + .5f);
                    replay.visit(a, tile.pixels, s, k, start - i, dx, dy);
                }
                __syncwarp();
                if (lane == 0)
                    ticket.store(i + 1, cuda::memory_order_release);
            }
            if (!visited)
                continue;
            replay.flush(a, tile.pixels, s);
        }
        __syncthreads();
    }
}

template <bool Abs, bool Moments>
using Replay = std::conditional_t<CHANNELS == 3 && FLAGS == 0, RgbReplay<Abs, Moments>,
                                  MapReplay<CHANNELS, FLAGS, Abs, Moments>>;

template <bool Moments> void launch_chunks(const Args *a, cudaStream_t stream) {
    dim3 grid((a->width + 15) / 16, (a->height + 15) / 16), block(32, Warps);
    if (a->absgrad)
        backward<Replay<true, Moments>><<<grid, block, 0, stream>>>(*a);
    else
        backward<Replay<false, Moments>><<<grid, block, 0, stream>>>(*a);
}
// The background sees the final transmittance, even on empty tiles.
__global__ void background_backward(Args a) {
    const int area = a.width * a.height, c = blockIdx.y;
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    float grad = pixel < area ? static_cast<const float *>(a.cache[Trans])[pixel] *
                                static_cast<const float *>(a.dout[RGB])[c * area + pixel] : 0.f;
    grad = warp_sum(grad);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(static_cast<float *>(a.din[Background]) + c, grad);
}
} // namespace

extern "C" int launch(const Args *a, cudaStream_t stream, bool backward_pass) {
    if (!backward_pass)
        return cudaErrorInvalidValue;
    if (a->moments)
        launch_chunks<true>(a, stream);
    else
        launch_chunks<false>(a, stream);
    if constexpr (CHANNELS > 0) {
        if (a->din[Background]) {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                return err;
            const int blocks = (a->width * a->height + 255) / 256;
            background_backward<<<dim3(blocks, CHANNELS), 256, 0, stream>>>(*a);
        }
    }
    return cudaGetLastError();
}
