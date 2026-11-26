#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace lite_raster {
struct AdamGroup {
    float *param, *first, *second;
    float lr, beta1, beta2, eps;
};

struct ProjectionArgs {
    const float *mean, *scale, *quat, *view, *K, *center;
    void *out[10];
    const float *up[10];
    float *gm, *gs, *gq, *gv;
    float *stats[4]; // pixel, world, pixel_ssim, world_ssim
    int n, w, h;
    float eps, clip;
};
struct ProjectionStep {
    AdamGroup groups[4]; // mean, log scale, raw quaternion, opacity logit
    const int64_t *ids;
    const float *alpha, *length;
    float *opacity;
    const float *d_opacity, *d_scale;
    bool geometry_grad;
};

struct TilePairs {
    const float *opacity;
    int *counts;
    int64_t *keys;
    int *ids;
    int tile_size, width, height;  // width and height in tiles
};
} // namespace lite_raster

extern "C" {
int projection(const lite_raster::ProjectionArgs*, int, cudaStream_t, bool);
int projection_adam(const lite_raster::ProjectionArgs*, int, cudaStream_t, bool, const lite_raster::ProjectionStep*);
int intersect_tiles(const lite_raster::TilePairs*, int, const float2*, const int*, const float*, const float*,
                    const int64_t*, cudaStream_t);
int sort_pairs(void*, size_t*, int64_t*, int64_t*, int*, int*, int, int, int*, cudaStream_t);
int encode_offsets(const int64_t*, int, int*, int, cudaStream_t);
}
