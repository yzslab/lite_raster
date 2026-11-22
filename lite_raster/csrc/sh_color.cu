#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr float dc_basis = 0.2820947917738781f;
constexpr float band1_scale = 0.48860251190292f;

// The first band keeps its common scalar outside the dot product in forward,
// as in gsplat. Higher bands use the same normalized real SH recurrence.
template<int Degree>
__device__ void sh_basis(float3 direction, float *b) {
    if constexpr (Degree == 0) return;
    float inv = rsqrtf(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    float x = direction.x * inv, y = direction.y * inv, z = direction.z * inv;
    b[1] = -y;
    b[2] = z;
    b[3] = -x;
    if constexpr (Degree < 2) return;
    float z2 = z * z;
    float c1 = x * x - y * y, s1 = 2.f * x * y;
    float a = -1.092548430592079f * z;
    b[4] = 0.5462742152960395f * s1;
    b[5] = a * y;
    b[6] = 0.9461746957575601f * z2 - 0.3153915652525201f;
    b[7] = a * x;
    b[8] = 0.5462742152960395f * c1;
    if constexpr (Degree < 3) return;
    float c2 = x * c1 - y * s1, s2 = x * s1 + y * c1;
    float d = -2.285228997322329f * z2 + 0.4570457994644658f;
    float e = 1.445305721320277f * z;
    b[9] = -0.5900435899266435f * s2;
    b[10] = e * s1;
    b[11] = d * y;
    b[12] = z * (1.865881662950577f * z2 - 1.119528997770346f);
    b[13] = d * x;
    b[14] = e * c1;
    b[15] = -0.5900435899266435f * c2;
    if constexpr (Degree < 4) return;
    float c3 = x * c2 - y * s2, s3 = x * s2 + y * c2;
    float f = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float g = 3.31161143515146f * z2 - 0.47308734787878f;
    float h = -1.770130769779931f * z;
    b[16] = 0.6258357354491763f * s3;
    b[17] = h * s2;
    b[18] = g * s1;
    b[19] = f * y;
    b[20] = 1.984313483298443f * z * b[12] - 1.006230589874905f * b[6];
    b[21] = f * x;
    b[22] = g * c1;
    b[23] = h * c2;
    b[24] = 0.6258357354491763f * c3;
}

template<int Band>
__device__ float band_dot(const float *b, const float *rest, int channel) {
    constexpr int start = Band * Band;
    float sum = b[start] * rest[(start - 1) * 3 + channel];
    #pragma unroll
    for (int k = start + 1; k < (Band + 1) * (Band + 1); ++k)
        sum += b[k] * rest[(k - 1) * 3 + channel];
    return sum;
}

struct Buffers {
    const float3 *dirs;
    float *dc, *rest;  // Inputs in forward, destinations in backward.
    const bool *visible;
    float *rgb;        // Output in forward, upstream gradient in backward.
    uint8_t *gate;     // Three clamp-derivative bits.
};

template<int Degree, bool Backward>
__device__ void evaluate(Buffers a, int i, int capacity, float *dc, float *rest) {
    int64_t offset = int64_t(i) * 3;
    constexpr int active = (Degree + 1) * (Degree + 1);
    float b[active];
    uint8_t gate = 0;
    bool visible;
    if constexpr (Backward) {
        gate = a.gate[i];
        visible = gate != 0;
    } else {
        visible = !a.visible || a.visible[i];
    }
    if constexpr (Degree > 0)
        if (visible) sh_basis<Degree>(a.dirs[i], b);

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        if constexpr (Backward) {
            float grad = (gate & (1 << c)) ? a.rgb[offset + c] : 0.f;
            if (dc) dc[c] = dc_basis * grad;
            if (rest) {
                #pragma unroll
                for (int k = 1; k < active; ++k) {
                    float basis = visible ? b[k] : 0.f;
                    rest[(k - 1) * 3 + c] = (k < 4 ? band1_scale * basis : basis) * grad;
                }
            }
        } else {
            float value = 0.f;
            if (visible) {
                value = __fmul_rn(dc_basis, dc[c]);
                if constexpr (Degree >= 1)
                    value += band1_scale * (b[1] * rest[c] + b[2] * rest[3 + c] + b[3] * rest[6 + c]);
                if constexpr (Degree >= 2) value += band_dot<2>(b, rest, c);
                if constexpr (Degree >= 3) value += band_dot<3>(b, rest, c);
                if constexpr (Degree >= 4) value += band_dot<4>(b, rest, c);
                value = __fadd_rn(value, 0.5f);
                if (value >= 0.f) gate |= 1 << c;  // torch.clamp_min passes grad at equality.
            }
            a.rgb[offset + c] = value < 0.f ? 0.f : value;
        }
    }
    if constexpr (Backward) {
        if (rest)
            for (int k = (active - 1) * 3; k < (capacity - 1) * 3; ++k) rest[k] = 0.f;
    } else if (a.gate) {
        a.gate[i] = gate;
    }
}

template<int Degree, bool Backward>
__global__ void direct_kernel(Buffers a, int n, int capacity, bool packed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t offset = int64_t(i) * 3;
    float *dc = a.dc ? a.dc + offset * (packed ? capacity : 1) : nullptr;
    float *rest = packed ? (dc ? dc + 3 : nullptr)
                         : (a.rest ? a.rest + offset * (capacity - 1) : nullptr);
    evaluate<Degree, Backward>(a, i, capacity, dc, rest);
}

constexpr int threads = 128;
constexpr int warp = 32;

// Coalesce global reads/writes without changing parameter or optimizer layout.
// One lane still evaluates every colour and coefficient of one Gaussian.
// The odd shared-memory row pitch permits conflict-free column access.
template<int Width, int Pitch, bool Store>
__device__ void transfer(float *tile, float *global, int rows, unsigned visible) {
    int lane = threadIdx.x % warp;
    if constexpr (Width == Pitch) {
        if (rows == warp && (reinterpret_cast<uintptr_t>(global) & 15) == 0) {
            auto *shared4 = reinterpret_cast<float4 *>(tile);
            auto *global4 = reinterpret_cast<float4 *>(global);
            #pragma unroll
            for (int j = lane; j < warp * Width / 4; j += warp) {
                if constexpr (Store) {
                    global4[j] = shared4[j];
                } else {
                    // A four-float transaction can straddle two Gaussian rows.
                    int first = 4 * j / Width, last = (4 * j + 3) / Width;
                    if (((visible >> first) | (visible >> last)) & 1u)
                        shared4[j] = global4[j];
                }
            }
            return;
        }
    }
    #pragma unroll
    for (int j = lane; j < warp * Width; j += warp) {
        int row = j / Width;
        if constexpr (Store) {
            if (row < rows) global[j] = tile[row * Pitch + j % Width];
        } else if ((visible >> row) & 1u) {
            tile[row * Pitch + j % Width] = global[j];
        }
    }
}

template<int Degree, bool Backward, int Capacity, bool Packed>
__global__ void tiled_kernel(Buffers a, int n) {
    constexpr int width = (Packed ? Capacity : Capacity - 1) * 3;
    constexpr int pitch = width | 1;
    __shared__ __align__(16) float storage[threads / warp][warp * pitch];
    int lane = threadIdx.x % warp;
    int i = blockIdx.x * threads + threadIdx.x;
    int first = i - lane;
    if (first >= n) return;
    int rows = min(warp, n - first);
    float *tile = storage[threadIdx.x / warp];
    float *global = Packed ? a.dc : a.rest;
    if (global) global += int64_t(first) * width;

    if constexpr (!Backward) {
        unsigned visible = __ballot_sync(0xffffffff, i < n && (!a.visible || a.visible[i]));
        transfer<width, pitch, false>(tile, global, rows, visible);
        __syncwarp();
    }
    if (i < n) {
        float *local = tile + lane * pitch;
        float *dc = Packed ? local : (a.dc ? a.dc + int64_t(i) * 3 : nullptr);
        float *rest = global ? (Packed ? local + 3 : local) : nullptr;
        evaluate<Degree, Backward>(a, i, Capacity, dc, rest);
    }
    if constexpr (Backward) {
        __syncwarp();
        if (global) transfer<width, pitch, true>(tile, global, rows, 0);
    }
}

template<int Degree, bool Backward, int Capacity>
bool try_tiled(Buffers a, int n, int blocks, bool packed, cudaStream_t stream) {
    constexpr int active = (Degree + 1) * (Degree + 1);
    // Backward also writes the inactive bands; a contiguous store is valuable
    // during degree warmup. Forward can skip their reads via the direct kernel.
    if constexpr (active == Capacity || (Backward && active < Capacity)) {
        if (packed)
            tiled_kernel<Degree, Backward, Capacity, true><<<blocks, threads, 0, stream>>>(a, n);
        else
            tiled_kernel<Degree, Backward, Capacity, false><<<blocks, threads, 0, stream>>>(a, n);
        return true;
    }
    return false;
}

template<int Degree, bool Backward>
void launch(Buffers a, int n, int capacity, bool packed, cudaStream_t stream) {
    int blocks = (n - 1) / threads + 1;
    switch (capacity) {
        case 4: if (try_tiled<Degree, Backward, 4>(a, n, blocks, packed, stream)) return; break;
        case 9: if (try_tiled<Degree, Backward, 9>(a, n, blocks, packed, stream)) return; break;
        case 16: if (try_tiled<Degree, Backward, 16>(a, n, blocks, packed, stream)) return; break;
        case 25: if (try_tiled<Degree, Backward, 25>(a, n, blocks, packed, stream)) return; break;
    }
    direct_kernel<Degree, Backward><<<blocks, threads, 0, stream>>>(a, n, capacity, packed);
}

template<int Degree>
void dispatch(Buffers a, int n, int capacity, bool packed, cudaStream_t stream, bool backward) {
    if (backward) launch<Degree, true>(a, n, capacity, packed, stream);
    else launch<Degree, false>(a, n, capacity, packed, stream);
}
} // namespace

extern "C" int sh_color(void **p, int n, int degree, int capacity, bool packed, cudaStream_t stream, bool backward) {
    if (!n) return cudaSuccess;
    Buffers a{(const float3 *)p[0], (float *)p[1], (float *)p[2],
              (const bool *)p[3], (float *)p[4], (uint8_t *)p[5]};
    switch (degree) {
        case 0: dispatch<0>(a, n, capacity, packed, stream, backward); break;
        case 1: dispatch<1>(a, n, capacity, packed, stream, backward); break;
        case 2: dispatch<2>(a, n, capacity, packed, stream, backward); break;
        case 3: dispatch<3>(a, n, capacity, packed, stream, backward); break;
        case 4: dispatch<4>(a, n, capacity, packed, stream, backward); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
