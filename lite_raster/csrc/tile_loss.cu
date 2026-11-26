#include <cuda_runtime.h>
#include "rgb_error.cuh"

// A separate block handles one channel of one 16x16 window.
namespace {
constexpr int Tile = 16;
struct Derivative {
    float mean_x, mean_y, luminance_term, contrast, contrast_scale;
};

__device__ float3 warp_sum(float3 v) {
    for (int d = 16; d; d >>= 1) {
        v.x += __shfl_down_sync(0xffffffff, v.x, d);
        v.y += __shfl_down_sync(0xffffffff, v.y, d);
        v.z += __shfl_down_sync(0xffffffff, v.z, d);
    }
    return v;
}

__device__ float3 block_sum(float3 v, float3 *scratch) {
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    v = warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = warp_sum(lane < 8 ? scratch[lane] : make_float3(0, 0, 0));
        if (lane == 0) scratch[0] = v;
    }
    __syncthreads();
    return scratch[0];
}

__device__ float ssim_derivative(const Derivative &d, float x, float y) {
    return d.luminance_term + d.contrast_scale * ((y - d.mean_y) - d.contrast * (x - d.mean_x));
}

__global__ void tile_stats(const float *image, const float *target, int w, int h, int ox, int oy,
                           float weight, float lambda_l2, float inv_pixels, float *joint_grad, float *errors,
                           Derivative *derivatives, float2 *sums) {
    __shared__ float3 moment_scratch[8], centered_scratch[8];
    const int left = int(blockIdx.x * Tile - ox), top = int(blockIdx.y * Tile - oy);
    const unsigned x = left + threadIdx.x % Tile, y = top + threadIdx.x / Tile;
    const bool valid = x < unsigned(w) && y < unsigned(h);
    const unsigned i = (blockIdx.z * h + y) * w + x;
    const float a = valid ? image[i] : 0.f, b = valid ? target[i] : 0.f;
    const float error = pixel_error<rgb_loss>(a, b, lambda_l2);
    const int valid_width = min(Tile + min(left, 0), w - max(left, 0));
    const int valid_height = min(Tile + min(top, 0), h - max(top, 0));
    const int n = valid_width * valid_height;
    const float inv_n = 1.f / n;
    const float3 totals = block_sum(make_float3(a, b, error), moment_scratch);
    const float mx = totals.x * inv_n, my = totals.y * inv_n;
    // Center before squaring to avoid catastrophic cancellation in low-contrast tiles.
    const float da = valid ? a - mx : 0.f, db = valid ? b - my : 0.f;
    const float3 m = block_sum(make_float3(da*da, db*db, da*db), centered_scratch);
    const float lum_den = mx * mx + my * my + .0001f;
    const float lum = 1.f - (mx - my) * (mx - my) / lum_den;
    const float var_den = (m.x + m.y) * inv_n + .0009f;
    const float contrast = (2.f * m.z * inv_n + .0009f) / var_den;
    const Derivative d = {mx, my, 2.f * (my - mx * lum) * contrast / lum_den,
                           contrast, 2.f * lum / var_den};
    if (valid) {
        joint_grad[i] = ((1.f - weight) * pixel_error_grad<rgb_loss>(a, b, lambda_l2)
                         - weight * ssim_derivative(d, a, b)) * inv_pixels;
        if (errors) errors[i] = error;
    }
    if (threadIdx.x == 0) {
        const int t = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        derivatives[t] = d;
        sums[t] = make_float2(totals.z, n * lum * contrast);
    }
}

__global__ void finish(const float2 *sums, int tiles, float inv_pixels, float weight,
                       float *loss, float *rgb_diff, float *ssim) {
    __shared__ float3 scratch[8];
    float3 v = {};
    for (int t = threadIdx.x; t < tiles; t += blockDim.x) {
        const float2 p = sums[t];
        v.x += p.x;
        v.y += p.y;
    }
    v = block_sum(v, scratch);
    if (threadIdx.x == 0) {
        const float l = v.x * inv_pixels, s = v.y * inv_pixels;
        *loss = (1.f - weight) * l + weight * (1.f - s);
        *rgb_diff = l;
        *ssim = s;
    }
}

__global__ void auxiliary_backward(const float *image, const float *target, int w, int h, int ox, int oy,
                                   float inv_pixels, float lambda_l2, const float *joint_grad, const Derivative *d,
                                   const float *g_loss, const float *g_rgb, const float *g_ssim,
                                   const float *g_errors, float *grad) {
    const unsigned x = blockIdx.x * Tile + threadIdx.x % Tile - ox;
    const unsigned y = blockIdx.y * Tile + threadIdx.x / Tile - oy;
    if (x >= unsigned(w) || y >= unsigned(h)) return;
    const unsigned i = (blockIdx.z * h + y) * w + x;
    const int t = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    const float a = image[i], b = target[i];
    float v = g_loss ? *g_loss * joint_grad[i] : 0.f;
    if (g_ssim) v += *g_ssim * inv_pixels * ssim_derivative(d[t], a, b);
    const float rgb_weight = (g_rgb ? *g_rgb * inv_pixels : 0.f) + (g_errors ? g_errors[i] : 0.f);
    grad[i] = v + rgb_weight * pixel_error_grad<rgb_loss>(a, b, lambda_l2);
}
}

extern "C" int tile_loss(float **p, int c, int h, int w, int ox, int oy, float weight, float lambda_l2,
                         bool backward, cudaStream_t stream) {
    const dim3 grid((unsigned(w) + ox + Tile - 1) / Tile, (unsigned(h) + oy + Tile - 1) / Tile, c);
    const float inv_pixels = 1.f / (c * h * w);
    auto *d = reinterpret_cast<Derivative *>(p[3]);
    if (backward) {
        auxiliary_backward<<<grid, 256, 0, stream>>>(p[0], p[1], w, h, ox, oy, inv_pixels, lambda_l2,
                                                     p[2], d, p[4], p[5], p[6], p[7], p[8]);
    } else {
        auto *sums = reinterpret_cast<float2 *>(p[4]);
        tile_stats<<<grid, 256, 0, stream>>>(p[0], p[1], w, h, ox, oy, weight, lambda_l2, inv_pixels, p[2], p[8], d, sums);
        if (auto err = cudaGetLastError()) return err;
        finish<<<1, 256, 0, stream>>>(sums, grid.x * grid.y * c, inv_pixels, weight, p[5], p[6], p[7]);
    }
    return cudaGetLastError();
}
