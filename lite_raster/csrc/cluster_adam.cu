// LiteGS's uncorrected Adam update for row-major Gaussian parameters.
// rows maps compact gradient rows onto persistent model and moment rows.
#include <cuda_runtime.h>
#include <cstdint>

namespace {
__global__ void update(float* param, const float* grad, float* first, float* second,
                       const int64_t* rows, int64_t count, int64_t total_rows, int width,
                       bool dense_grad, float lr, float beta1, float beta2, float eps) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int64_t compact_row = i / width;
    const int64_t row = rows ? rows[compact_row] : compact_row;
    if (row < 0 || row >= total_rows) return;
    const int64_t dst = row * width + i % width;
    const float g = grad[dense_grad ? dst : i];
    const float m = beta1 * first[dst] + (1.f - beta1) * g;
    const float v = beta2 * second[dst] + (1.f - beta2) * g * g;
    param[dst] += -lr * m / (sqrtf(v) + eps);
    first[dst] = m;
    second[dst] = v;
}
}

extern "C" int cluster_adam(float* param, const float* grad, float* first, float* second,
                            const int64_t* rows, int64_t count, int64_t total_rows, int width,
                            bool dense_grad, float lr, float beta1, float beta2, float eps,
                            cudaStream_t stream) {
    update<<<(count + 255) / 256, 256, 0, stream>>>(param, grad, first, second, rows, count, total_rows,
                                                     width, dense_grad, lr, beta1, beta2, eps);
    return cudaGetLastError();
}
