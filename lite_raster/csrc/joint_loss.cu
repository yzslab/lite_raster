// 11-tap Gaussian and SSIM derivatives adapted from LiteGS/fused-ssim (LICENSE-fused-ssim).
#include <cuda_runtime.h>
#include "rgb_error.cuh"

namespace {
constexpr int Tile = 16, Radius = 5, Shared = Tile + 2 * Radius;
__constant__ float gaussian[11] = {
    .001028380123898387f, .0075987582094967365f, .036000773310661316f, .10936068743467331f,
    .21300552785396576f, .26601171493530273f, .21300552785396576f, .10936068743467331f,
    .036000773310661316f, .0075987582094967365f, .001028380123898387f
};

// One block per channel and output tile. Out-of-image samples, including
// derivatives outside the loss image, are zero for "same" padding.
template <int N>
__device__ void load_tile(float (&tile)[Shared][Shared][N], const float *const (&src)[N], int h, int w, float scale = 1.f) {
    for (int y = threadIdx.y; y < Shared; y += Tile) {
        int gy = int(blockIdx.y) * Tile + y - Radius;
        for (int x = threadIdx.x; x < Shared; x += Tile) {
            int gx = int(blockIdx.x) * Tile + x - Radius;
            int index = (blockIdx.z * h + gy) * w + gx;
            bool inside = gx >= 0 && gx < w && gy >= 0 && gy < h;
#pragma unroll
            for (int i = 0; i < N; ++i)
                tile[y][x][i] = inside ? src[i][index] * scale : 0.f;
        }
    }
    __syncthreads();
}

template <int N>
__device__ void vertical(const float (&rows)[Shared][Tile][N], float (&result)[N]) {
#pragma unroll
    for (int d = 1; d <= Radius; ++d) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
            const int y = threadIdx.y + Radius;
            result[j] += (rows[y-d][threadIdx.x][j] + rows[y+d][threadIdx.x][j]) * gaussian[Radius-d];
        }
    }
#pragma unroll
    for (int j = 0; j < N; ++j)
        result[j] += rows[threadIdx.y + Radius][threadIdx.x][j] * gaussian[Radius];
}

__device__ float2 block_sum(float2 value, int tid) {
    __shared__ float2 warps[8];
    int lane = tid % 32, warp = tid / 32;
#pragma unroll
    for (int d = 16; d; d /= 2) {
        value.x += __shfl_down_sync(0xffffffff, value.x, d);
        value.y += __shfl_down_sync(0xffffffff, value.y, d);
    }
    if (lane == 0) warps[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < 8 ? warps[lane] : make_float2(0, 0);
#pragma unroll
        for (int d = 4; d; d /= 2) {
            value.x += __shfl_down_sync(0xffffffff, value.x, d);
            value.y += __shfl_down_sync(0xffffffff, value.y, d);
        }
    }
    return value;
}

__global__ void image_statistics(const float *image, const float *target, int h, int w,
                                  float *d_mean, float *d_square, float *d_product, float2 *sums, float *errors,
                                  float lambda_l2) {
    __shared__ float tile[Shared][Shared][2];
    __shared__ float rows[Shared][Tile][5];
    const float *const images[] = {image, target};
    load_tile(tile, images, h, w);
    for (int y = threadIdx.y; y < Shared; y += Tile) {
        float horizontal[5] = {};
#pragma unroll
        for (int d = 1; d <= Radius; ++d) {
            const float *left = tile[y][threadIdx.x + Radius - d];
            const float *right = tile[y][threadIdx.x + Radius + d];
            const float a = left[0], b = left[1], x = right[0], z = right[1];
            const float pair[] = {a+x, a*a+x*x, b+z, b*b+z*z, a*b+x*z};
#pragma unroll
            for (int j = 0; j < 5; ++j) horizontal[j] += pair[j] * gaussian[Radius-d];
        }
        const float a = tile[y][threadIdx.x + Radius][0], b = tile[y][threadIdx.x + Radius][1];
        const float center[] = {a, a*a, b, b*b, a*b};
#pragma unroll
        for (int j = 0; j < 5; ++j) rows[y][threadIdx.x][j] = horizontal[j] + center[j] * gaussian[Radius];
    }
    __syncthreads();
    int x = blockIdx.x * Tile + threadIdx.x, y = blockIdx.y * Tile + threadIdx.y;
    float2 stats = {};
    if (x < w && y < h) {
        float v[5] = {};
        vertical(rows, v);
        const float a = v[0], b = v[2];
        const float A = __fmaf_rn(a, a, __fmul_rn(b, b)) + .0001f;
        // Use one rounding for each variance. Reusing a rounded mean square
        // here changes which low-contrast windows cross the SSIM clamp.
        const float B = fmaxf(0.f, __fmaf_rn(-a, a, v[1])) + fmaxf(0.f, __fmaf_rn(-b, b, v[3])) + .0009f;
        const float C = 2.f*a*b + .0001f;
        const float D = 2.f*(v[4] - a*b) + .0009f;
        const int index = (blockIdx.z * h + y) * w + x;
        // Match the installed fused-ssim convention, including its backward at saturation.
        const float raw = (C * D) / (A * B);
        const bool saturated = raw < -1.f || raw > 1.f;
        stats = {pixel_error<rgb_loss>(image[index], target[index], lambda_l2), fmaxf(-1.f, fminf(1.f, raw))};
        if (errors) errors[index] = stats.x;
        d_mean[index] = saturated ? 0.f : (2.f*b*D) / (A*B) - (2.f*b*C) / (A*B)
                      - (2.f*a*C*D) / (A*A*B) + (2.f*a*C*D) / (A*B*B);
        d_square[index] = saturated ? 0.f : -(C*D) / (A*B*B);
        d_product[index] = saturated ? 0.f : (2.f*C) / (A*B);
    }
    const int tid = threadIdx.y * Tile + threadIdx.x;
    stats = block_sum(stats, tid);
    if (tid == 0) sums[(blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x] = stats;
}

__global__ void finish_mean(const float2 *sums, int count, int pixels, float weight,
                             float *loss, float *rgb_diff, float *ssim) {
    float2 total = {};
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        total.x += sums[i].x;
        total.y += sums[i].y;
    }
    total = block_sum(total, threadIdx.x);
    if (threadIdx.x == 0) {
        *rgb_diff = total.x / pixels;
        *ssim = total.y / pixels;
        *loss = (1.f - weight) * *rgb_diff + weight * (1.f - *ssim);
    }
}

__global__ void image_gradient(const float *image, const float *target, int h, int w,
                               const float *d_mean, const float *d_square, const float *d_product,
                               const float *g_loss, const float *g_rgb, const float *g_ssim,
                               float weight, float lambda_l2, float *gradient, const float *g_errors) {
    __shared__ float tile[Shared][Shared][3];
    __shared__ float rows[Shared][Tile][3];
    const float *const derivatives[] = {d_mean, d_square, d_product};
    const float chain = g_loss ? *g_loss : 0.f;
    const int count = gridDim.z * h * w;
    const float wl = ((1.f - weight) * chain + (g_rgb ? *g_rgb : 0.f)) / count;
    const float ws = (-weight * chain + (g_ssim ? *g_ssim : 0.f)) / count;
    load_tile(tile, derivatives, h, w, ws);
    for (int y = threadIdx.y; y < Shared; y += Tile) {
        float horizontal[3] = {};
#pragma unroll
        for (int d = 1; d <= Radius; ++d) {
            const int x = threadIdx.x + Radius;
#pragma unroll
            for (int j = 0; j < 3; ++j)
                horizontal[j] += (tile[y][x-d][j] + tile[y][x+d][j]) * gaussian[Radius-d];
        }
#pragma unroll
        for (int j = 0; j < 3; ++j)
            rows[y][threadIdx.x][j] = horizontal[j] + tile[y][threadIdx.x + Radius][j] * gaussian[Radius];
    }
    __syncthreads();
    int x = blockIdx.x * Tile + threadIdx.x, y = blockIdx.y * Tile + threadIdx.y;
    if (x < w && y < h) {
        float deriv[3] = {};
        vertical(rows, deriv);
        const int i = (blockIdx.z * h + y) * w + x;
        const float a = image[i], b = target[i];
        const float rgb_grad = pixel_error_grad<rgb_loss>(a, b, lambda_l2);
        gradient[i] = (deriv[0] + 2.f*a*deriv[1] + b*deriv[2]) + wl * rgb_grad
                    + (g_errors ? g_errors[i] * rgb_grad : 0.f);
    }
}
}

extern "C" int joint_loss(float **p, int h, int w, int channels, float weight, float lambda_l2,
                          bool backward, cudaStream_t stream) {
    const dim3 block(Tile, Tile), grid((w + Tile - 1) / Tile, (h + Tile - 1) / Tile, channels);
    if (backward) {
        image_gradient<<<grid, block, 0, stream>>>(p[0], p[1], h, w, p[2], p[3], p[4], p[5], p[6], p[7],
                                                    weight, lambda_l2, p[8], p[9]);
    } else {
        auto *tiles = reinterpret_cast<float2 *>(p[5]);
        image_statistics<<<grid, block, 0, stream>>>(p[0], p[1], h, w, p[2], p[3], p[4], tiles, p[9], lambda_l2);
        if (auto err = cudaGetLastError()) return err;
        finish_mean<<<1, 256, 0, stream>>>(tiles, grid.x * grid.y * channels, h*w*channels, weight, p[6], p[7], p[8]);
    }
    return cudaGetLastError();
}
