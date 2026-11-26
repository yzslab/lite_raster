// Dense single-camera tile encoding, adapted from gsplat (see LICENSE-gsplat).
#include "tile_intersections.cuh"
#include <cub/cub.cuh>
#include <type_traits>

namespace {
using namespace lite_raster;

struct CountSpans {
    __device__ void span(int, int, int, bool) {}
};
struct OrderedWriter {
    TilePairs pairs;
    int row;
    unsigned depth;
    int slot;
    __device__ void span(int u, int start, int end, bool along_y) {
        for (int v = start; v < end; ++v) {
            int tile = along_y ? u * pairs.width + v : v * pairs.width + u;
            pairs.keys[slot] = (int64_t(tile) << 32) | depth;
            pairs.ids[slot++] = row;
        }
    }
};

template<bool Precise, class Writer>
__device__ int enumerate(TilePairs p, int i, float2 xy, int radius, const float *conic, Writer &writer) {
    if constexpr (Precise)
        return precise_tiles(xy, conic[3*i], conic[3*i+1], conic[3*i+2], p.opacity[i],
                             p.tile_size, p.width, p.height, writer);
    float r = float(radius) / p.tile_size;
    float x = xy.x / p.tile_size, y = xy.y / p.tile_size;
    int x0 = max(0, min(p.width, __float2int_rd(x-r))), x1 = max(0, min(p.width, __float2int_ru(x+r)));
    int y0 = max(0, min(p.height, __float2int_rd(y-r))), y1 = max(0, min(p.height, __float2int_ru(y+r)));
    if constexpr (!std::is_same_v<Writer, CountSpans>)
        for (int row = y0; row < y1; ++row) writer.span(row, x0, x1, true);
    return (x1-x0)*(y1-y0);
}

template<bool Precise>
__global__ void intersections(int n, TilePairs pairs, const float2 *means, const int *radii,
                              const float *depths, const float *conics, const int64_t *ends) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (ends) {
        if (pairs.counts[i] == 0) return;
        OrderedWriter writer{pairs, i, __float_as_uint(depths[i]), i ? static_cast<int>(ends[i-1]) : 0};
        enumerate<Precise>(pairs, i, means[i], radii[i], conics, writer);
    } else {
        CountSpans counter;
        pairs.counts[i] = radii[i] > 0 ? enumerate<Precise>(pairs, i, means[i], radii[i], conics, counter) : 0;
    }
}

__global__ void tile_offsets(const int64_t *keys, int n, int *offsets, int tiles) {
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= tiles) return;
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi-lo)/2;
        if ((keys[mid] >> 32) < tile) lo = mid + 1;
        else hi = mid;
    }
    offsets[tile] = lo;
}
}  // namespace

extern "C" int intersect_tiles(const lite_raster::TilePairs *p, int n, const float2 *means, const int *radii,
                               const float *depths, const float *conics, const int64_t *ends, cudaStream_t stream) {
    if (!n) return cudaSuccess;
    int blocks = (n-1)/256+1;
    if (conics) intersections<true><<<blocks,256,0,stream>>>(n,*p,means,radii,depths,conics,ends);
    else intersections<false><<<blocks,256,0,stream>>>(n,*p,means,radii,depths,conics,ends);
    return cudaGetLastError();
}

extern "C" int sort_pairs(void *scratch, size_t *bytes, int64_t *k0, int64_t *k1, int *v0, int *v1,
                          int n, int bits, int *selected, cudaStream_t stream) {
    cub::DoubleBuffer<int64_t> keys(k0,k1);
    cub::DoubleBuffer<int> ids(v0,v1);
    auto err = cub::DeviceRadixSort::SortPairs(scratch,*bytes,keys,ids,n,0,bits,stream);
    *selected = keys.selector;
    return err;
}
extern "C" int encode_offsets(const int64_t *keys, int n, int *offsets, int tiles, cudaStream_t stream) {
    tile_offsets<<<(tiles-1)/256+1,256,0,stream>>>(keys,n,offsets,tiles);
    return cudaGetLastError();
}
