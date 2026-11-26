#pragma once

#include <cuda_runtime.h>

namespace {
// Selected at build time by loss.py through the backend's FLAGS specialization.
enum class RGBLoss { L1, L2, L1PlusL2 };
constexpr RGBLoss rgb_loss = static_cast<RGBLoss>(FLAGS);

template <RGBLoss Mode>
__device__ float pixel_error(float a, float b, float lambda_l2) {
    const float d = a - b;
    if constexpr (Mode == RGBLoss::L1) return fabsf(d);
    if constexpr (Mode == RGBLoss::L2) return d * d;
    return fmaf(lambda_l2 * d, d, fabsf(d));
}

template <RGBLoss Mode>
__device__ float pixel_error_grad(float a, float b, float lambda_l2) {
    if constexpr (Mode == RGBLoss::L2) return 2.f * (a - b);
    const float sign = float(a > b) - float(a < b);
    if constexpr (Mode == RGBLoss::L1) return sign;
    return fmaf(2.f * lambda_l2, a - b, sign);
}
} // namespace
