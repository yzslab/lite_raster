#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace lite_raster {
struct AdamGroup {
    float *param, *first, *second;
    float lr, beta1, beta2, eps;
};

__device__ inline void adam_update(AdamGroup a, int64_t dst, float g) {
    float m = a.beta1 * a.first[dst] + (1.f - a.beta1) * g;
    float v = a.beta2 * a.second[dst] + (1.f - a.beta2) * g * g;
    // Projection uses fast math. Match standalone Adam's sqrt and division.
    a.param[dst] += __fdiv_rn(-a.lr * m, __fsqrt_rn(v) + a.eps);
    a.first[dst] = m;
    a.second[dst] = v;
}
} // namespace lite_raster
