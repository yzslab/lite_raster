// Host scheduling for dense projection and tile binning on the current PyTorch stream.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <array>
#include <climits>
#include "projection_api.h"

namespace {
using at::Tensor;
using OptionalTensor = std::optional<Tensor>;
using namespace lite_raster;

void check(int status) {
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(static_cast<cudaError_t>(status)));
}

std::array<Tensor, 4> encode(const ProjectionArgs& args, const Tensor& radii, const float* conics,
                             const float* opacity, int tile, cudaStream_t stream) {
    int width = (args.w - 1) / tile + 1, height = (args.h - 1) / tile + 1;
    TORCH_CHECK(int64_t(width) * height <= INT_MAX, "Tile IDs exceed int32");
    Tensor counts = at::empty_like(radii), ends;
    TilePairs pairs{opacity, counts.data_ptr<int>(), nullptr, nullptr, tile, width, height};
    auto intersect = [&](const int64_t* end_ptr) {
        check(intersect_tiles(&pairs, args.n, static_cast<const float2*>(args.out[1]), radii.data_ptr<int>(),
                              static_cast<const float*>(args.out[2]), conics, end_ptr, stream));
    };

    int64_t length = 0;
    if (args.n) {
        intersect(nullptr);
        ends = counts.view({args.n}).cumsum(0, at::kLong);
        // Exact-sized lists need a single host read of the last prefix sum.
        length = ends.select(0, args.n - 1).item<int64_t>();
        TORCH_CHECK(length >= 0 && length <= INT_MAX, "Tile intersections exceed int32");
    }
    auto keys = at::empty({length}, radii.options().dtype(at::kLong));
    auto ids = at::empty({length}, radii.options());
    if (length) {
        pairs.keys = keys.data_ptr<int64_t>();
        pairs.ids = ids.data_ptr<int>();
        intersect(ends.data_ptr<int64_t>());
        auto alt_keys = at::empty_like(keys);
        auto alt_ids = at::empty_like(ids);
        int bits = 32, selected = 0;
        for (unsigned v = width * height - 1; v; v >>= 1) ++bits;
        size_t bytes = 0;
        auto sort = [&](void* scratch) {
            check(sort_pairs(scratch, &bytes, pairs.keys, alt_keys.data_ptr<int64_t>(),
                             pairs.ids, alt_ids.data_ptr<int>(), length, bits, &selected, stream));
        };
        sort(nullptr);
        auto scratch = at::empty({static_cast<int64_t>(bytes)}, radii.options().dtype(at::kByte));
        sort(scratch.data_ptr());
        if (selected) {
            keys = alt_keys;
            ids = alt_ids;
        }
    }
    auto offsets = at::empty({1, height, width}, radii.options());
    check(encode_offsets(keys.data_ptr<int64_t>(), length, offsets.data_ptr<int>(), width * height, stream));
    return {counts, keys, ids, offsets};
}

// Inputs are validated and outputs allocated by the projection autograd nodes.
template<bool Precise>
std::array<Tensor, 4> project_encode(const std::array<Tensor, 6>& inputs, const std::array<OptionalTensor, 10>& out,
                                    const OptionalTensor& alpha, const OptionalTensor& adam_opacity,
                                    int w, int h, float eps, float clip, int mode, int tile) {
    c10::cuda::CUDAGuard guard(inputs[0].device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    TORCH_CHECK(inputs[0].size(0) <= INT_MAX && w > 0 && h > 0 && tile > 0, "Invalid projection dimensions");
    ProjectionArgs args{};
    args.mean = inputs[0].data_ptr<float>();
    args.scale = inputs[1].data_ptr<float>();
    args.quat = inputs[2].data_ptr<float>();
    args.view = inputs[3].data_ptr<float>();
    args.K = inputs[4].data_ptr<float>();
    args.center = inputs[5].data_ptr<float>();
    for (size_t i = 0; i < out.size(); ++i)
        if (out[i]) args.out[i] = out[i]->data_ptr();
    args.n = inputs[0].size(0);
    args.w = w;
    args.h = h;
    args.eps = eps;
    args.clip = clip;

    if (adam_opacity) {
        ProjectionStep step{};
        step.alpha = alpha.value().data_ptr<float>();
        step.opacity = adam_opacity->data_ptr<float>();
        check(projection_adam(&args, mode, stream, false, &step));
    } else {
        check(projection(&args, mode, stream, false));
    }

    if constexpr (Precise) {
        Tensor effective = adam_opacity ? *adam_opacity : alpha.value().view({args.n});
        if (!adam_opacity && out[7]) effective = effective * out[7]->view({args.n});
        return encode(args, *out[0], static_cast<const float*>(args.out[3]), effective.data_ptr<float>(), tile, stream);
    } else {
        return encode(args, *out[0], nullptr, nullptr, tile, stream);
    }
}
} // namespace

PYBIND11_MODULE(_lite_raster_project_encode, m) {
    m.def("precise", &project_encode<true>, pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("bbox", &project_encode<false>, pybind11::call_guard<pybind11::gil_scoped_release>());
}
