// One specialization for each requested feature mask and color dimension.
// Each block visits the sorted splats of a tile. Backward replays the same
// list in reverse from the last accepted ordinary/hard contribution.
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;
#include "raster_args.cuh"
__device__ inline void add(float *p, float value) {
    auto group = cg::coalesced_threads();
    value = cg::reduce(group, value, cg::plus<float>());
    if (group.thread_rank() == 0)
        atomicAdd(p, value);
}
__device__ inline float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
template <int D, int F, bool Back> __global__ void raster(Args a) {
    constexpr bool inv = F & Inverse, hardInverse = F & Hard, hardZ = F & HardDepth, expz = F & Expected,
                   median = F & Median, normal = F & Normal, rade = F & Rade;
    constexpr bool hard = hardInverse || hardZ || (F & ReturnHardAlpha);
    constexpr bool depth = inv || hardInverse || hardZ || expz || median;
    const int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
    const int pixel = y * a.width + x, tile = blockIdx.y * gridDim.x + blockIdx.x;
    const int area = a.width * a.height;
    const bool inside = x < a.width && y < a.height;
    const int *offset = (int *)a.in[Offsets], *ids = (int *)a.in[Ids];
    const int begin = offset[tile], end = tile + 1 == gridDim.x * gridDim.y ? a.count : offset[tile + 1];
    const float px = x + 0.5f, py = y + 0.5f;
    const float *Kmat = (float *)a.in[K];
    float ray_z = 1;
    if constexpr (rade && depth) {
        float u = (px - Kmat[2]) / Kmat[0], v = (py - Kmat[5]) / Kmat[4];
        ray_z = rsqrtf(1 + u * u + v * v);
    }
    float T = 1, hT = 1, suffix = 0, hSuffix = 0;
    bool done = !inside, hDone = false;
    int last = -1, hLast = -1, medId = -1;
    float rgb[D ? D : 1] = {}, dc[D ? D : 1] = {};
    float3 n = {}, dn = {};
    float di = 0, dh = 0, dhz = 0, de = 0, dm = 0, da = 0, dha = 0;
    float zi = 0, hi = 0, hz = 0, expected = 0, med = 0;
#define OUT(k) ((float *)a.out[k])
#define GRAD(k) ((const float *)a.dout[k])
#define CACHE(k, type) ((type *)a.cache[k])
    const float *bg = (float *)a.in[Background];
    if constexpr (Back)
        if (inside) {
            last = CACHE(Last, int)[pixel];
            T = CACHE(Trans, float)[pixel];
            if constexpr (F & ReturnAlpha)
                da = GRAD(Alpha)[pixel];
            if constexpr (inv)
                di = GRAD(InvZ)[pixel];
            if constexpr (hard) {
                hLast = CACHE(LastHard, int)[pixel];
                hT = CACHE(HardTrans, float)[pixel];
                if constexpr (hardInverse)
                    dh = GRAD(HardInvZ)[pixel];
                if constexpr (hardZ)
                    dhz = GRAD(HardZ)[pixel];
                if constexpr (F & ReturnHardAlpha)
                    dha = GRAD(HardAlpha)[pixel];
                hSuffix = -hT * dha;
            }
            if constexpr (expz) {
                float A = 1 - T;
                de = A > 0 ? GRAD(ExpZ)[pixel] / A : 0;
                da -= de * OUT(ExpZ)[pixel];
            }
            if constexpr (median) {
                medId = CACHE(MedId, int)[pixel];
                dm = GRAD(MedZ)[pixel];
            }
            if constexpr (normal) {
                if constexpr (F & RawNormal) {
                    dn = read_normal(a.dout[Nrm], pixel, area);
                } else {
                    float3 unit = read_normal(a.out[Nrm], pixel, area), g = read_normal(a.dout[Nrm], pixel, area);
                    float len = CACHE(NormLength, float)[pixel], proj = len > 1e-12f ? dot3(g, unit) : 0;
                    float den = fmaxf(len, 1e-12f);
                    dn = {(g.x - unit.x * proj) / den, (g.y - unit.y * proj) / den,
                          (g.z - unit.z * proj) / den};
                }
            }
            suffix = -T * da;
            for (int c = 0; c < D; c++) {
                dc[c] = GRAD(RGB)[c * area + pixel];
                suffix += T * dc[c] * bg[c];
                if (a.din[Background])
                    atomicAdd((float *)a.din[Background] + c, T * dc[c]);
            }
        }
    // Cache the frequently accessed conic and mean once per tile batch.
    __shared__ float4 shape[256];
    __shared__ float2 centers[256];
    __shared__ int batchIds[256];
    int lane = threadIdx.y * 16 + threadIdx.x;
    for (int batch = 0; batch < end - begin; batch += 256) {
        if constexpr (!Back) {
            if (__syncthreads_count(!done) == 0)
                break;
        } else
            __syncthreads();
        int length = min(256, end - begin - batch);
        int start = Back ? end - 1 - batch : begin + batch;
        if (lane < length) {
            int g = ids[start + (Back ? -lane : lane)];
            batchIds[lane] = g;
            centers[lane] = ((float2 *)a.in[XY])[g];
            float3 con = ((float3 *)a.in[Conic])[g];
            shape[lane] = {con.x, con.y, con.z, ((float *)a.in[Opacity])[g]};
        }
        __syncthreads();
        for (int j = 0; !done && j < length; ++j) {
            int idx = start + (Back ? -j : j), g = batchIds[j];
            if constexpr (Back)
                if (idx > max(last, hLast))
                    continue;
            float dx = centers[j].x - px, dy = centers[j].y - py;
            float4 s = shape[j];
            float sigma = .5f * (s.x * dx * dx + s.z * dy * dy) + s.y * dx * dy;
            if (sigma < 0)
                continue;
            float falloff = __expf(-sigma);
            float alpha = fminf(.999f, s.w * falloff), ha = fminf(.999f, falloff);
            bool valid = alpha >= 1.f / 255 && (!Back || idx <= last);
            bool hv = hard && ha >= 1.f / 255 && !hDone && (!Back || idx <= hLast);
            if constexpr (!Back) {
                if (hv && hT * (1 - ha) <= 1e-4f) {
                    hv = false;
                    hDone = true;
                }
                if (valid && T * (1 - alpha) <= 1e-4f) {
                    valid = false;
                    done = true;
                }
            }
            if (!valid && !hv)
                continue;
            float z = 0, inverse = 0;
            float3 plane = {};
            if constexpr (depth) {
                if constexpr (rade) {
                    plane = ((float3 *)a.in[Plane])[g];
                    z = (plane.x * dx + plane.y * dy + plane.z) * ray_z;
                } else
                    z = ((float *)a.in[CenterZ])[g];
                if constexpr (inv || hardInverse) {
                    if constexpr (rade)
                        inverse = z > 1e-8f ? 1.f / z : 0.f;
                    else
                        inverse = 1.f / (z + 1e-8f);
                }
            }
            const float *color = (float *)a.in[Color];
            float3 ng = {};
            if constexpr (normal)
                ng = ((float3 *)a.in[Normals])[g];
            if constexpr (!Back) {
                if (hv) {
                    if constexpr (hardInverse)
                        hi += hT * ha * inverse;
                    if constexpr (hardZ)
                        hz += hT * ha * z;
                    hT *= 1 - ha;
                    hLast = idx;
                }
                if (!valid)
                    continue;
                float w = T * alpha;
                for (int c = 0; c < D; c++)
                    rgb[c] += w * color[g * D + c];
                if constexpr (normal) {
                    n.x += w * ng.x;
                    n.y += w * ng.y;
                    n.z += w * ng.z;
                }
                if constexpr (inv)
                    zi += w * inverse;
                if constexpr (expz)
                    expected += w * z;
                if constexpr (median)
                    if (T > .5f) {
                        med = z;
                        medId = idx;
                    }
                T *= 1 - alpha;
                last = idx;
                if constexpr (F & ReturnHit)
                    CACHE(Hit, bool)[g] = true;
            } else {
                float dz = 0, ds = 0, dop = 0, gx = 0, gy = 0;
                if (valid) {
                    float ra = 1.f / (1 - alpha);
                    T *= ra;
                    float w = T * alpha, featureDot = 0;
                    for (int c = 0; c < D; c++) {
                        featureDot += dc[c] * color[g * D + c];
                        add((float *)a.din[Color] + g * D + c, w * dc[c]);
                    }
                    if constexpr (normal) {
                        featureDot += dot3(ng, dn);
                        float *p = (float *)a.din[Normals] + 3 * g;
                        add(p, w * dn.x);
                        add(p + 1, w * dn.y);
                        add(p + 2, w * dn.z);
                    }
                    if constexpr (expz) {
                        featureDot += de * z;
                        dz += w * de;
                    }
                    if constexpr (inv) {
                        featureDot += di * inverse;
                        dz -= w * di * inverse * inverse;
                    }
                    if constexpr (median)
                        if (idx == medId)
                            dz += dm;
                    float alphaGrad = T * featureDot - suffix * ra;
                    suffix += w * featureDot;
                    if (s.w * falloff <= .999f) {
                        dop = falloff * alphaGrad;
                        ds = -s.w * dop;
                    }
                    gx = ds * (s.x * dx + s.y * dy);
                    gy = ds * (s.y * dx + s.z * dy);
                    if constexpr (rade) {
                        gx += dz * ray_z * plane.x;
                        gy += dz * ray_z * plane.y;
                    }
                    if (a.absgrad) {
                        add((float *)a.absgrad + 2 * g, fabsf(gx));
                        add((float *)a.absgrad + 2 * g + 1, fabsf(gy));
                    }
                    if (a.moments) {
                        float *p = (float *)a.moments + 6 * g;
                        add(p, gx);
                        add(p + 1, gy);
                        add(p + 2, gx * gx);
                        add(p + 3, gx * gy);
                        add(p + 4, gy * gy);
                        add(p + 5, 1);
                    }
                }
                if constexpr (hard)
                    if (hv) {
                        float ra = 1.f / (1 - ha);
                        hT *= ra;
                        float w = hT * ha, val = dh * inverse + dhz * z;
                        float hg = hT * val - hSuffix * ra;
                        hSuffix += w * val;
                        float hdz = w * (dhz - dh * inverse * inverse);
                        dz += hdz;
                        if constexpr (rade) {
                            gx += hdz * ray_z * plane.x;
                            gy += hdz * ray_z * plane.y;
                        }
                        if (falloff <= .999f) {
                            float hs = -falloff * hg;
                            ds += hs;
                            dop -= hs; // gsplat straight-through gradient for hard opacity
                            gx += hs * (s.x * dx + s.y * dy);
                            gy += hs * (s.y * dx + s.z * dy);
                        }
                    }
                if constexpr (depth) {
                    if constexpr (rade) {
                        float *p = (float *)a.din[Plane] + 3 * g;
                        add(p, dz * ray_z * dx);
                        add(p + 1, dz * ray_z * dy);
                        add(p + 2, dz * ray_z);
                    } else
                        add((float *)a.din[CenterZ] + g, dz);
                }
                add((float *)a.din[XY] + 2 * g, gx);
                add((float *)a.din[XY] + 2 * g + 1, gy);
                add((float *)a.din[Opacity] + g, dop);
                float *p = (float *)a.din[Conic] + 3 * g;
                add(p, .5f * ds * dx * dx);
                add(p + 1, ds * dx * dy);
                add(p + 2, .5f * ds * dy * dy);
            }
        }
    }
    if constexpr (!Back)
        if (inside) {
            for (int c = 0; c < D; c++)
                OUT(RGB)[c * area + pixel] = rgb[c] + T * bg[c];
            CACHE(Last, int)[pixel] = last;
            CACHE(Trans, float)[pixel] = T;
            if constexpr (F & ReturnAlpha)
                OUT(Alpha)[pixel] = 1 - T;
            if constexpr (inv)
                OUT(InvZ)[pixel] = zi;
            if constexpr (hard) {
                CACHE(LastHard, int)[pixel] = hLast;
                CACHE(HardTrans, float)[pixel] = hT;
                if constexpr (hardInverse)
                    OUT(HardInvZ)[pixel] = hi;
                if constexpr (hardZ)
                    OUT(HardZ)[pixel] = hz;
                if constexpr (F & ReturnHardAlpha)
                    OUT(HardAlpha)[pixel] = 1 - hT;
            }
            if constexpr (expz)
                OUT(ExpZ)[pixel] = last >= 0 ? expected / (1 - T) : 0;
            if constexpr (median) {
                OUT(MedZ)[pixel] = med;
                CACHE(MedId, int)[pixel] = medId;
            }
            if constexpr (normal) {
                if constexpr (F & RawNormal) {
                    write_normal(a.out[Nrm], pixel, area, n);
                } else {
                    float len = sqrtf(dot3(n, n)), den = fmaxf(1e-12f, len);
                    CACHE(NormLength, float)[pixel] = len;
                    write_normal(a.out[Nrm], pixel, area, {n.x / den, n.y / den, n.z / den});
                }
            }
        }
}
extern "C" int launch(Args *args, cudaStream_t stream, bool backward) {
    dim3 blocks((args->width + 15) / 16, (args->height + 15) / 16), threads(16, 16);
    if (backward)
        raster<CHANNELS, FLAGS, true><<<blocks, threads, 0, stream>>>(*args);
    else
        raster<CHANNELS, FLAGS, false><<<blocks, threads, 0, stream>>>(*args);
    return cudaGetLastError();
}
