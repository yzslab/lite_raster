// Each warp replays the sorted splats for its pixels and reduces their
// per-Gaussian gradients before writing to global memory.
#include <cuda_runtime.h>
#include "raster_args.cuh"

constexpr unsigned FullWarp = 0xffffffffu;
constexpr int WarpsPerBlock = 4;

template <int D, int F>
constexpr bool RGBOnly = D == 3 && (F & ~(ReturnAlpha | ReturnHit)) == 0;
// One warp replays this many pixels in the full RGB backward.
constexpr int RGBWarpWidth = 16, RGBWarpHeight = 8;
constexpr int RGBWarpsPerTile = (16 / RGBWarpWidth) * (16 / RGBWarpHeight);

__device__ __forceinline__ void tile_add(float *dst, float value) {
#pragma unroll
    for (int offset = 16; offset; offset /= 2)
        value += __shfl_down_sync(FullWarp, value, offset);
    if (threadIdx.x == 0 && value != 0.f)
        atomicAdd(dst, value);
}

// Match the original FP32 forward's rounding when two plane terms nearly
// cancel. Reversing which product is fused can change inverse-depth gradients.
__device__ __forceinline__ float pixel_depth(float3 plane, float dx, float dy, float ray_z) {
    float slope = __fmaf_rn(plane.x, dx, __fmul_rn(plane.y, dy));
    return __fmul_rn(__fadd_rn(slope, plane.z), ray_z);
}

template <int D> struct PixelState {
    int last, hard_last, median_id, index;
    float trans, hard_trans, suffix, hard_suffix, ray_z;
    float inverse_grad, hard_inverse_grad, hard_z_grad, expected_grad, median_grad;
    float3 normal_grad;
    // High-dimensional feature maps can be read on demand rather than replicated
    // eight times per lane. RGB and small feature vectors stay in registers.
    float colors[D > 0 && D <= 8 ? D : 1];
};

struct Splat {
    int id;
    float2 xy;
    float3 conic;
    float opacity;
};

template <int D, int F, bool AbsOnly = false> __global__ void warp_backward(Args a) {
    static_assert(!AbsOnly || D == 3);
    constexpr bool split_rgb = RGBOnly<D, F> && !AbsOnly;
    constexpr int parts = split_rgb ? RGBWarpsPerTile : 1;
    constexpr int group_w = split_rgb ? RGBWarpWidth : 16;
    constexpr int group_h = split_rgb ? RGBWarpHeight : 16;
    constexpr int row_stride = 32 / group_w;
    constexpr int PixelsPerLane = 8 / parts;
    constexpr bool stage_rgb = AbsOnly || split_rgb;
    constexpr bool inv = F & Inverse, hard_inv = F & Hard, hard_z = F & HardDepth;
    constexpr bool expected = F & Expected, median = F & Median, normal = F & Normal;
    constexpr bool rade = F & Rade, hard = hard_inv || hard_z || (F & ReturnHardAlpha);
    constexpr bool depth = inv || hard_inv || hard_z || expected || median;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tiles_x = (a.width + 15) / 16, tiles_y = (a.height + 15) / 16;
    const int group = blockIdx.x * WarpsPerBlock + warp;
    const int tile = group / parts;
    const int local = group % parts;
    if (tile >= tiles_x * tiles_y)
        return;
    const int x0 = (tile % tiles_x) * 16 + (local % (16 / group_w)) * group_w;
    const int y0 = (tile / tiles_x) * 16 + (local / (16 / group_w)) * group_h;
    const int x = x0 + lane % group_w;
    const int first_y = y0 + lane / group_w;
    int begin = ((int *)a.in[Offsets])[tile];
    const int *ids = (const int *)a.in[Ids];
    const float *rgb_grad = (const float *)a.dout[RGB];
    PixelState<D> pixels[PixelsPerLane] = {};
    int last_in_tile = -1;
    float bg_grad[D ? D : 1] = {};
#pragma unroll
    for (int i = 0; i < PixelsPerLane; ++i) {
        PixelState<D> &p = pixels[i];
        const int y = first_y + row_stride * i;
        p.last = p.hard_last = -1;
        if (x >= a.width || y >= a.height)
            continue;
        const int k = p.index = y * a.width + x;
        p.last = ((const int *)a.cache[Last])[k];
        p.trans = ((const float *)a.cache[Trans])[k];
        float da = 0.f;
        if constexpr (F & ReturnAlpha)
            da = ((const float *)a.dout[Alpha])[k];
        if constexpr (inv)
            p.inverse_grad = ((const float *)a.dout[InvZ])[k];
        if constexpr (hard) {
            p.hard_last = ((const int *)a.cache[LastHard])[k];
            p.hard_trans = ((const float *)a.cache[HardTrans])[k];
            if constexpr (F & ReturnHardAlpha)
                p.hard_suffix = -p.hard_trans * ((const float *)a.dout[HardAlpha])[k];
            if constexpr (hard_inv)
                p.hard_inverse_grad = ((const float *)a.dout[HardInvZ])[k];
            if constexpr (hard_z)
                p.hard_z_grad = ((const float *)a.dout[HardZ])[k];
        }
        if constexpr (expected) {
            const float alpha = 1.f - p.trans;
            p.expected_grad = alpha > 0.f ? ((const float *)a.dout[ExpZ])[k] / alpha : 0.f;
            da -= p.expected_grad * ((const float *)a.out[ExpZ])[k];
        }
        if constexpr (median) {
            p.median_id = ((const int *)a.cache[MedId])[k];
            p.median_grad = ((const float *)a.dout[MedZ])[k];
        }
        if constexpr (normal) {
            if constexpr (F & RawNormal) {
                p.normal_grad = ((const float3 *)a.dout[Nrm])[k];
            } else {
                const float3 unit = ((const float3 *)a.out[Nrm])[k];
                const float3 grad = ((const float3 *)a.dout[Nrm])[k];
                const float len = ((const float *)a.cache[NormLength])[k];
                const float proj = len > 1e-12f ? grad.x * unit.x + grad.y * unit.y + grad.z * unit.z : 0.f;
                const float den = fmaxf(1e-12f, len);
                p.normal_grad = {(grad.x - unit.x * proj) / den, (grad.y - unit.y * proj) / den,
                                 (grad.z - unit.z * proj) / den};
            }
        }
        p.suffix = -p.trans * da;
        for (int c = 0; c < D; ++c) {
            const float dc = rgb_grad[k * D + c];
            if constexpr (D <= 8)
                p.colors[c] = dc;
            p.suffix += p.trans * dc * ((const float *)a.in[Background])[c];
            if (!AbsOnly && a.din[Background])
                bg_grad[c] += p.trans * dc;
        }
        if constexpr (depth && rade) {
            const float *Kmat = (const float *)a.in[K];
            const float u = (x + .5f - Kmat[2]) / Kmat[0], v = (y + .5f - Kmat[5]) / Kmat[4];
            p.ray_z = rsqrtf(1.f + u * u + v * v);
        }
        last_in_tile = max(last_in_tile, max(p.last, p.hard_last));
    }
    if (!AbsOnly && a.din[Background])
        for (int c = 0; c < D; ++c)
            tile_add((float *)a.din[Background] + c, bg_grad[c]);

#pragma unroll
    for (int offset = 16; offset; offset /= 2)
        last_in_tile = max(last_in_tile, __shfl_down_sync(FullWarp, last_in_tile, offset));
    last_in_tile = __shfl_sync(FullWarp, last_in_tile, 0);
    if constexpr (AbsOnly) {
        // The saved final transmittance includes the front occluders. Once
        // reverse replay reaches the first candidate, its work is complete.
        for (; begin <= last_in_tile; begin += 32) {
            int idx = begin + lane;
            unsigned mask = __ballot_sync(FullWarp, idx <= last_in_tile && a.absgrad_candidates[ids[idx]]);
            if (mask) {
                begin += __ffs(mask) - 1;
                break;
            }
        }
    }
    __shared__ Splat batches[WarpsPerBlock][32];
    __shared__ float batch_rgb[stage_rgb ? WarpsPerBlock * 32 : 1][3];
    Splat *batch = batches[warp];
    for (int start = last_in_tile; start >= begin; start -= 32) {
        bool candidate = false, visit = false;
        if (start - lane >= begin) {
            int g = ids[start - lane];
            if constexpr (AbsOnly)
                candidate = a.absgrad_candidates[g];
            if constexpr (stage_rgb) {
                // Load RGB for the batch together, then broadcast within the warp.
                for (int c = 0; c < 3; ++c)
                    batch_rgb[32 * warp + lane][c] = ((const float *)a.in[Color])[3 * g + c];
            }
            batch[lane] = {g, ((const float2 *)a.in[XY])[g], ((const float3 *)a.in[Conic])[g],
                           ((const float *)a.in[Opacity])[g]};
            if constexpr (split_rgb) {
                const Splat s = batch[lane];
                // Completing the square gives sigma >= det(Q)/2 * max(dx²/c, dy²/a).
                // Use the nearest pixel-center distance on each axis and err toward replay
                // when the FP32 conic is ill-conditioned or the alpha is close to 1/255.
                const float dx0 = s.xy.x - (x0 + .5f), dx1 = s.xy.x - (x0 + group_w - 1 + .5f);
                const float dy0 = s.xy.y - (y0 + .5f), dy1 = s.xy.y - (y0 + group_h - 1 + .5f);
                const float min_x = fmaxf(0.f, fmaxf(dx1, -dx0)), min_y = fmaxf(0.f, fmaxf(dy1, -dy0));
                const float max_x = fmaxf(fabsf(dx0), fabsf(dx1)), max_y = fmaxf(fabsf(dy0), fabsf(dy1));
                const float a = s.conic.x, b = s.conic.y, c = s.conic.z;
                const float det = fmaf(a, c, -b * b) - 2.e-6f * (a * c);
                const float slack = 2.e-5f * (1.f + a * max_x * max_x + c * max_y * max_y
                                              + 2.f * fabsf(b) * max_x * max_y);
                const float lower = .5f * det * fmaxf(min_x * min_x / c, min_y * min_y / a) - slack;
                // For opacity <= 1, a lower bound above log(255) also proves invisibility.
                visit = !(a > 0.f && c > 0.f && det > 0.f && isfinite(lower) && lower > 0.f &&
                          s.opacity <= 1.f && (lower > 5.55f || s.opacity * __expf(-lower) < .99999f / 255.f));
            }
        }
        __syncwarp();
        unsigned mask = 0;
        if constexpr (AbsOnly)
            mask = __ballot_sync(FullWarp, candidate);
        unsigned active = 0;
        if constexpr (split_rgb)
            active = __ballot_sync(FullWarp, visit);
        const int count = min(32, start - begin + 1);
        for (int ordinal = 0; split_rgb ? active != 0 : ordinal < count; ++ordinal) {
            int j = ordinal;
            if constexpr (split_rgb) {
                j = __ffs(active) - 1;
                active &= active - 1;
            }
            const Splat s = batch[j];
            const int g = s.id, idx = start - j;
            const bool collect = !AbsOnly || ((mask >> j) & 1u);
            const float dx = s.xy.x - (x + .5f);
            float3 plane = {}, ng = {};
            float center_z = 0.f;
            if constexpr (depth && rade)
                plane = ((const float3 *)a.in[Plane])[g];
            if constexpr (depth && !rade)
                center_z = ((const float *)a.in[CenterZ])[g];
            if constexpr (normal)
                ng = ((const float3 *)a.in[Normals])[g];
            float color_sum[D ? D : 1] = {}, moments[6] = {};
            float2 xy_sum = {}, absolute = {};
            float3 conic_sum = {}, normal_sum = {}, depth_sum = {};
            float opacity_sum = 0.f;
            bool contributed = false;
#pragma unroll
            for (int i = 0; i < PixelsPerLane; ++i) {
                PixelState<D> &p = pixels[i];
                if (idx > max(p.last, p.hard_last))
                    continue;
                float dy = s.xy.y - (first_y + row_stride * i + .5f);
                float sigma;
                if constexpr (split_rgb) {
                    // Match the original RGB warp's FP32 contraction order at the alpha cutoff.
                    const float yy = __fmul_rn(__fmul_rn(s.conic.z, dy), dy);
                    const float diag = __fmaf_rn(__fmul_rn(s.conic.x, dx), dx, yy);
                    sigma = __fmaf_rn(__fmul_rn(s.conic.y, dx), dy, __fmul_rn(.5f, diag));
                } else {
                    sigma = .5f * (s.conic.x * dx * dx + s.conic.z * dy * dy) + s.conic.y * dx * dy;
                }
                if (sigma < 0.f)
                    continue;
                float fall = __expf(-sigma);
                float alpha = fminf(.999f, s.opacity * fall), ha = fminf(.999f, fall);
                bool valid = idx <= p.last && alpha >= 1.f / 255;
                bool hv = hard && idx <= p.hard_last && ha >= 1.f / 255;
                if (!valid && !hv)
                    continue;
                contributed = true;
                float z = center_z, inverse = 0.f;
                if constexpr (depth && rade)
                    z = pixel_depth(plane, dx, dy, p.ray_z);
                if constexpr (inv || hard_inv)
                    inverse = rade ? (z > 1e-8f ? 1.f / z : 0.f) : 1.f / (z + 1e-8f);
                float dz = 0.f, ds = 0.f, dop = 0.f, gx = 0.f, gy = 0.f;
                if (valid) {
                    const float ra = 1.f / (1.f - alpha);
                    p.trans *= ra;
                    const float w = p.trans * alpha;
                    float dot = 0.f;
                    for (int c = 0; c < D; ++c) {
                        float dc;
                        if constexpr (D <= 8)
                            dc = p.colors[c];
                        else
                            dc = rgb_grad[p.index * D + c];
                        if constexpr (stage_rgb)
                            dot += dc * batch_rgb[32 * warp + j][c];
                        else
                            dot += dc * ((const float *)a.in[Color])[g * D + c];
                        color_sum[c] += w * dc;
                    }
                    if constexpr (normal) {
                        const float3 dn = p.normal_grad;
                        dot += ng.x * dn.x + ng.y * dn.y + ng.z * dn.z;
                        normal_sum.x += w * dn.x;
                        normal_sum.y += w * dn.y;
                        normal_sum.z += w * dn.z;
                    }
                    if constexpr (expected) {
                        dot += p.expected_grad * z;
                        dz += w * p.expected_grad;
                    }
                    if constexpr (inv) {
                        dot += p.inverse_grad * inverse;
                        dz -= w * p.inverse_grad * inverse * inverse;
                    }
                    if constexpr (median)
                        if (idx == p.median_id)
                            dz += p.median_grad;
                    // Every contributing splat updates the suffix behind the next candidate.
                    const float suffix = p.suffix;
                    p.suffix += w * dot;
                    if (!collect)
                        continue;
                    float dalpha = p.trans * dot - suffix * ra;
                    if (s.opacity * fall <= .999f) {
                        dop = fall * dalpha;
                        ds = -s.opacity * dop;
                    }
                    gx = ds * (s.conic.x * dx + s.conic.y * dy);
                    gy = ds * (s.conic.y * dx + s.conic.z * dy);
                    if constexpr (rade) {
                        gx += dz * p.ray_z * plane.x;
                        gy += dz * p.ray_z * plane.y;
                    }
                    // The baseline records ordinary-branch statistics per pixel,
                    // before adding the hard branch or summing across the lane.
                    if (AbsOnly || a.absgrad) {
                        absolute.x += fabsf(gx);
                        absolute.y += fabsf(gy);
                    }
                    if (!AbsOnly && a.moments) {
                        moments[0] += gx;
                        moments[1] += gy;
                        moments[2] += gx * gx;
                        moments[3] += gx * gy;
                        moments[4] += gy * gy;
                        moments[5] += 1.f;
                    }
                }
                if constexpr (hard)
                    if (hv) {
                        float ra = 1.f / (1.f - ha);
                        p.hard_trans *= ra;
                        float w = p.hard_trans * ha, val = p.hard_inverse_grad * inverse + p.hard_z_grad * z;
                        float hg = p.hard_trans * val - p.hard_suffix * ra;
                        p.hard_suffix += w * val;
                        float hdz = w * (p.hard_z_grad - p.hard_inverse_grad * inverse * inverse);
                        dz += hdz;
                        if constexpr (rade) {
                            gx += hdz * p.ray_z * plane.x;
                            gy += hdz * p.ray_z * plane.y;
                        }
                        if (fall <= .999f) {
                            float hs = -fall * hg;
                            ds += hs;
                            dop -= hs;
                            gx += hs * (s.conic.x * dx + s.conic.y * dy);
                            gy += hs * (s.conic.y * dx + s.conic.z * dy);
                        }
                    }
                if constexpr (depth && rade) {
                    depth_sum.x += dz * p.ray_z * dx;
                    depth_sum.y += dz * p.ray_z * dy;
                    depth_sum.z += dz * p.ray_z;
                } else if constexpr (depth)
                    depth_sum.z += dz;
                xy_sum.x += gx;
                xy_sum.y += gy;
                opacity_sum += dop;
                conic_sum.x += .5f * ds * dx * dx;
                conic_sum.y += ds * dx * dy;
                conic_sum.z += .5f * ds * dy * dy;
            }
            if (!collect || !__any_sync(FullWarp, contributed))
                continue;
            // Replay all occluders; only candidate splats own auxiliary statistics.
            if constexpr (AbsOnly) {
                tile_add((float *)a.absgrad + 2 * g, absolute.x);
                tile_add((float *)a.absgrad + 2 * g + 1, absolute.y);
            } else {
                for (int c = 0; c < D; ++c)
                    tile_add((float *)a.din[Color] + g * D + c, color_sum[c]);
                if constexpr (normal) {
                    float *dst = (float *)a.din[Normals] + 3 * g;
                    tile_add(dst, normal_sum.x);
                    tile_add(dst + 1, normal_sum.y);
                    tile_add(dst + 2, normal_sum.z);
                }
                if constexpr (depth && rade) {
                    float *dst = (float *)a.din[Plane] + 3 * g;
                    tile_add(dst, depth_sum.x);
                    tile_add(dst + 1, depth_sum.y);
                    tile_add(dst + 2, depth_sum.z);
                } else if constexpr (depth)
                    tile_add((float *)a.din[CenterZ] + g, depth_sum.z);
                tile_add((float *)a.din[XY] + 2 * g, xy_sum.x);
                tile_add((float *)a.din[XY] + 2 * g + 1, xy_sum.y);
                tile_add((float *)a.din[Opacity] + g, opacity_sum);
                float *dc = (float *)a.din[Conic] + 3 * g;
                tile_add(dc, conic_sum.x);
                tile_add(dc + 1, conic_sum.y);
                tile_add(dc + 2, conic_sum.z);
                if (a.absgrad) {
                    tile_add((float *)a.absgrad + 2 * g, absolute.x);
                    tile_add((float *)a.absgrad + 2 * g + 1, absolute.y);
                }
                if (a.moments)
                    for (int c = 0; c < 6; ++c)
                        tile_add((float *)a.moments + 6 * g + c, moments[c]);
            }
        }
        __syncwarp();
    }
}

extern "C" int launch(Args *args, cudaStream_t stream, bool backward) {
    if (!backward)
        return cudaErrorInvalidValue;
    int tiles = ((args->width + 15) / 16) * ((args->height + 15) / 16);
    const int blocks = (tiles + WarpsPerBlock - 1) / WarpsPerBlock;
    if constexpr (CHANNELS == 3) {
        if (args->absgrad_candidates) {
            warp_backward<CHANNELS, FLAGS, true><<<blocks, dim3(32, WarpsPerBlock), 0, stream>>>(*args);
            return cudaGetLastError();
        }
    }
    constexpr int parts = RGBOnly<CHANNELS, FLAGS> ? RGBWarpsPerTile : 1;
    const int grid_blocks = (tiles * parts + WarpsPerBlock - 1) / WarpsPerBlock;
    warp_backward<CHANNELS, FLAGS><<<grid_blocks, dim3(32, WarpsPerBlock), 0, stream>>>(*args);
    return cudaGetLastError();
}
