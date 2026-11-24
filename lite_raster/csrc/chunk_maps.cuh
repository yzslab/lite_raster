// Pixel adjoints and per-splat gradient accumulation for the chunk scheduler.
#pragma once

__device__ float dot3(float3 u, float3 v) { return u.x * v.x + u.y * v.y + u.z * v.z; }

__device__ void warp_add(float *dst, float value) {
    value = warp_sum(value);
    if (threadIdx.x == 0)
        atomicAdd(dst, value);
}

template <int D, int F, bool Abs, bool Moments> struct MapReplay {
    static constexpr bool hard = F & (Hard | HardDepth | ReturnHardAlpha);
    static constexpr bool depth = F & (Inverse | Expected | Median | Hard | HardDepth);
    static constexpr bool plane_depth = depth && (F & Rade);
    static constexpr bool normal = F & Normal;
    static constexpr bool wide = D > 8;

    struct Pixels {
        float trans[256], suffix[256];
        int last[hard ? 256 : 1], hard_last[hard ? 256 : 1];
        float hard_trans[hard ? 256 : 1], hard_suffix[hard ? 256 : 1];
        float inv[(F & Inverse) ? 256 : 1], expected[(F & Expected) ? 256 : 1];
        float hard_inv[(F & Hard) ? 256 : 1], hard_z[(F & HardDepth) ? 256 : 1];
        float median[(F & Median) ? 256 : 1], ray_z[plane_depth ? 256 : 1];
        int median_id[(F & Median) ? 256 : 1];
        float3 normals[normal ? 256 : 1];
        float colors[D && !wide ? D : 1][D && !wide ? 256 : 1];
        int image_index[wide ? 256 : 1];
    };

    static __device__ int initialize(const Args &a, Pixels &p, int k, int x, int y) {
        const int pixel = y * a.width + x, area = a.width * a.height;
        int last = static_cast<const int *>(a.cache[Last])[pixel];
        const float t = static_cast<const float *>(a.cache[Trans])[pixel];
        p.trans[k] = t;
        float da = 0.f;
        if constexpr (F & ReturnAlpha)
            da = static_cast<const float *>(a.dout[Alpha])[pixel];
        if constexpr (F & Expected) {
            const float alpha = 1.f - t;
            p.expected[k] = alpha > 0.f ? static_cast<const float *>(a.dout[ExpZ])[pixel] / alpha : 0.f;
            da -= p.expected[k] * static_cast<const float *>(a.out[ExpZ])[pixel];
        }
        if constexpr (F & Inverse)
            p.inv[k] = static_cast<const float *>(a.dout[InvZ])[pixel];
        if constexpr (F & Median) {
            p.median[k] = static_cast<const float *>(a.dout[MedZ])[pixel];
            p.median_id[k] = static_cast<const int *>(a.cache[MedId])[pixel];
        }
        if constexpr (normal) {
            float3 dn = read_normal(a.dout[Nrm], pixel, area);
            if constexpr (!(F & RawNormal)) {
                const float3 unit = read_normal(a.out[Nrm], pixel, area);
                const float len = static_cast<const float *>(a.cache[NormLength])[pixel];
                const float proj = len > 1e-12f ? dot3(dn, unit) : 0.f, den = fmaxf(len, 1e-12f);
                dn = {(dn.x - unit.x * proj) / den, (dn.y - unit.y * proj) / den,
                      (dn.z - unit.z * proj) / den};
            }
            p.normals[k] = dn;
        }
        if constexpr (plane_depth) {
            const float *Kmat = static_cast<const float *>(a.in[K]);
            const float u = (x + .5f - Kmat[2]) / Kmat[0], v = (y + .5f - Kmat[5]) / Kmat[4];
            p.ray_z[k] = rsqrtf(1.f + u * u + v * v);
        }
        if constexpr (hard) {
            p.last[k] = last;
            p.hard_last[k] = static_cast<const int *>(a.cache[LastHard])[pixel];
            p.hard_trans[k] = static_cast<const float *>(a.cache[HardTrans])[pixel];
            p.hard_suffix[k] = 0.f;
            if constexpr (F & ReturnHardAlpha)
                p.hard_suffix[k] = -p.hard_trans[k] * static_cast<const float *>(a.dout[HardAlpha])[pixel];
            if constexpr (F & Hard)
                p.hard_inv[k] = static_cast<const float *>(a.dout[HardInvZ])[pixel];
            if constexpr (F & HardDepth)
                p.hard_z[k] = static_cast<const float *>(a.dout[HardZ])[pixel];
            last = max(last, p.hard_last[k]);
        }
        float suffix = -t * da;
        if constexpr (wide)
            p.image_index[k] = pixel;
        for (int c = 0; c < D; ++c) {
            const float dc = static_cast<const float *>(a.dout[RGB])[c * area + pixel];
            suffix += t * dc * static_cast<const float *>(a.in[Background])[c];
            if constexpr (!wide)
                p.colors[c][k] = dc;
        }
        p.suffix[k] = suffix;
        return last;
    }

    float2 xy = {}, absolute = {};
    float3 conic = {}, dz_sum = {}, dn_sum = {};
    float opacity = 0.f, stats[6] = {};
    float colors[D && !wide ? D : 1] = {}, weights[wide ? Chunks : 1] = {};

    __device__ void visit(const Args &a, Pixels &p, const SplatData &s, int k, int idx, float dx, float dy) {
        const float sigma = sigma_at(s.q, dx, dy);
        if (sigma < 0.f)
            return;
        const float fall = __expf(-sigma), raw = s.opacity * fall;
        const float alpha = fminf(.999f, raw), ha = fminf(.999f, fall);
        const bool valid = (!hard || idx <= p.last[k]) && alpha >= 1.f / 255.f;
        const bool hard_valid = hard && idx <= p.hard_last[k] && ha >= 1.f / 255.f;
        if (!valid && !hard_valid)
            return;

        float3 plane = {};
        float z = 0.f, inv = 0.f;
        if constexpr (plane_depth) {
            plane = static_cast<const float3 *>(a.in[Plane])[s.id];
            z = pixel_depth(plane, dx, dy, p.ray_z[k]);
        } else if constexpr (depth)
            z = static_cast<const float *>(a.in[CenterZ])[s.id];
        if constexpr (F & (Inverse | Hard))
            inv = plane_depth ? (z > 1e-8f ? 1.f / z : 0.f) : 1.f / (z + 1e-8f);

        const float qx = s.q.x * dx + s.q.y * dy, qy = s.q.y * dx + s.q.z * dy;
        float ds = 0.f, dz = 0.f;
        if (valid) {
            const float ra = 1.f / (1.f - alpha);
            p.trans[k] *= ra;
            const float w = p.trans[k] * alpha;
            float dot = 0.f;
            for (int c = 0; c < D; ++c) {
                float dc;
                if constexpr (wide)
                    dc = static_cast<const float *>(a.dout[RGB])[c * (a.width * a.height) + p.image_index[k]];
                else {
                    dc = p.colors[c][k];
                    colors[c] += w * dc;
                }
                if constexpr (D != 3)
                    dot += dc * static_cast<const float *>(a.in[Color])[D * s.id + c];
            }
            if constexpr (D == 3)
                dot = dot3(s.rgb, {p.colors[0][k], p.colors[1][k], p.colors[2][k]});
            if constexpr (wide)
                weights[k / 32] = w;
            if constexpr (normal) {
                const float3 dn = p.normals[k];
                dot += dot3(static_cast<const float3 *>(a.in[Normals])[s.id], dn);
                dn_sum.x += w * dn.x;
                dn_sum.y += w * dn.y;
                dn_sum.z += w * dn.z;
            }
            if constexpr (F & Expected) {
                dot += p.expected[k] * z;
                dz += w * p.expected[k];
            }
            if constexpr (F & Inverse) {
                dot += p.inv[k] * inv;
                dz -= w * p.inv[k] * inv * inv;
            }
            if constexpr (F & Median)
                if (idx == p.median_id[k])
                    dz += p.median[k];
            const float da = p.trans[k] * dot - p.suffix[k] * ra;
            p.suffix[k] += w * dot;
            if (raw <= .999f) {
                const float dop = fall * da;
                opacity += dop;
                ds = -s.opacity * dop;
            }
            float gx = ds * qx, gy = ds * qy;
            if constexpr (plane_depth) {
                gx += dz * p.ray_z[k] * plane.x;
                gy += dz * p.ray_z[k] * plane.y;
            }
            xy.x += gx;
            xy.y += gy;
            // Density statistics describe the ordinary blend, before the hard branch.
            if constexpr (Abs) {
                absolute.x += fabsf(gx);
                absolute.y += fabsf(gy);
            }
            if constexpr (Moments) {
                stats[0] += gx;
                stats[1] += gy;
                stats[2] += gx * gx;
                stats[3] += gx * gy;
                stats[4] += gy * gy;
                stats[5] += 1.f;
            }
        }
        if constexpr (hard) {
            if (hard_valid) {
                const float ra = 1.f / (1.f - ha);
                p.hard_trans[k] *= ra;
                const float w = p.hard_trans[k] * ha;
                float dot = 0.f, grad_z = 0.f;
                if constexpr (F & Hard) {
                    dot = p.hard_inv[k] * inv;
                    grad_z = -p.hard_inv[k] * inv * inv;
                }
                if constexpr (F & HardDepth) {
                    dot += p.hard_z[k] * z;
                    grad_z += p.hard_z[k];
                }
                const float da = p.hard_trans[k] * dot - p.hard_suffix[k] * ra;
                p.hard_suffix[k] += w * dot;
                const float hdz = w * grad_z;
                dz += hdz;
                if constexpr (plane_depth) {
                    xy.x += hdz * p.ray_z[k] * plane.x;
                    xy.y += hdz * p.ray_z[k] * plane.y;
                }
                if (fall <= .999f) {
                    const float hs = -fall * da;
                    ds += hs;
                    opacity -= hs; // Match the hard opacity straight-through estimator.
                    xy.x += hs * qx;
                    xy.y += hs * qy;
                }
            }
        }
        conic.x += .5f * ds * dx * dx;
        conic.y += ds * dx * dy;
        conic.z += .5f * ds * dy * dy;
        if constexpr (plane_depth) {
            dz_sum.x += dz * p.ray_z[k] * dx;
            dz_sum.y += dz * p.ray_z[k] * dy;
            dz_sum.z += dz * p.ray_z[k];
        } else if constexpr (depth)
            dz_sum.z += dz;
    }

    __device__ void flush(const Args &a, const Pixels &p, const SplatData &s) {
        warp_add(static_cast<float *>(a.din[XY]) + 2 * s.id, xy.x);
        warp_add(static_cast<float *>(a.din[XY]) + 2 * s.id + 1, xy.y);
        warp_add(static_cast<float *>(a.din[Opacity]) + s.id, opacity);
        float *q = static_cast<float *>(a.din[Conic]) + 3 * s.id;
        warp_add(q, conic.x);
        warp_add(q + 1, conic.y);
        warp_add(q + 2, conic.z);
        if constexpr (plane_depth) {
            float *dst = static_cast<float *>(a.din[Plane]) + 3 * s.id;
            warp_add(dst, dz_sum.x);
            warp_add(dst + 1, dz_sum.y);
            warp_add(dst + 2, dz_sum.z);
        } else if constexpr (depth)
            warp_add(static_cast<float *>(a.din[CenterZ]) + s.id, dz_sum.z);
        if constexpr (normal) {
            float *dst = static_cast<float *>(a.din[Normals]) + 3 * s.id;
            warp_add(dst, dn_sum.x);
            warp_add(dst + 1, dn_sum.y);
            warp_add(dst + 2, dn_sum.z);
        }
        for (int c = 0; c < D; ++c) {
            float grad;
            if constexpr (wide) {
                // Replay visits each of this lane's eight pixels at most once per splat.
                // Retain weights instead of one register or shared-memory plane per channel.
                grad = 0.f;
                for (int chunk = 0; chunk < Chunks; ++chunk) {
                    if (weights[chunk] != 0.f) {
                        int pixel = p.image_index[32 * chunk + threadIdx.x];
                        grad += weights[chunk] * static_cast<const float *>(a.dout[RGB])[c * (a.width * a.height) + pixel];
                    }
                }
            } else
                grad = colors[c];
            warp_add(static_cast<float *>(a.din[Color]) + D * s.id + c, grad);
        }
        if constexpr (Abs) {
            warp_add(static_cast<float *>(a.absgrad) + 2 * s.id, absolute.x);
            warp_add(static_cast<float *>(a.absgrad) + 2 * s.id + 1, absolute.y);
        }
        if constexpr (Moments) {
            for (int i = 0; i < 6; ++i)
                warp_add(static_cast<float *>(a.moments) + 6 * s.id + i, stats[i]);
        }
    }
};
