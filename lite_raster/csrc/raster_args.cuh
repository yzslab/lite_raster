#pragma once

enum Feature {
    Inverse = 1, Hard = 2, Expected = 4, Median = 8, Normal = 16, Rade = 32, HardDepth = 64,
    ReturnAlpha = 128, ReturnHardAlpha = 256, ReturnHit = 512, RawNormal = 1024
};
enum Input { XY, Conic, Opacity, Color, CenterZ, Plane, Normals, Background, K, Offsets, Ids, NInputs };
enum Output { RGB, Alpha, InvZ, HardInvZ, HardAlpha, ExpZ, MedZ, Nrm, HardZ, NOutputs };
enum Cache { Last, LastHard, MedId, NormLength, Trans, HardTrans, Hit, NCache };
struct Args {
    void *in[NInputs], *out[NOutputs], *cache[NCache], *dout[NOutputs], *din[8], *absgrad, *moments;
    const bool *absgrad_candidates;
    int width, height, count;
};

__device__ inline float3 read_normal(const void *data, int pixel, int area) {
    const float *p = static_cast<const float *>(data) + pixel;
    return {p[0], p[area], p[2 * area]};
}

__device__ inline void write_normal(void *data, int pixel, int area, float3 normal) {
    float *p = static_cast<float *>(data) + pixel;
    p[0] = normal.x;
    p[area] = normal.y;
    p[2 * area] = normal.z;
}

// Match the original FP32 forward's rounding when two plane terms nearly
// cancel. Reversing which product is fused can change inverse-depth gradients.
__device__ __forceinline__ float pixel_depth(float3 plane, float dx, float dy, float ray_z) {
    float slope = __fmaf_rn(plane.x, dx, __fmul_rn(plane.y, dy));
    return __fmul_rn(__fadd_rn(slope, plane.z), ray_z);
}
