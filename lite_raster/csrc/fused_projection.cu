// Single-view pinhole projection. Use gsplat's projection,
// quaternion adjoints and center-path moment transform for the basic outputs.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <cooperative_groups.h>
#include "adam.cuh"
#include "pinhole.cuh"
#include "rade_geometry.cuh"

namespace lite_raster {
// NVCC 12.8 requires this argument type to have external linkage to export projection().
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
} // namespace lite_raster

namespace {
using namespace gsplat;
using V = vec3<float>;
using Q = vec4<float>;
using M = mat3<float>;
using M2 = mat2<float>;
enum Output { Radius, XY, Z, Conic, Direction, LegacyNormal, Distance, Compensation, Plane, Normal };
enum Mode { Planes = 1, Normals = 2, Legacy = 4 };

__device__ V grad3(const float *v, int i) { return v ? glm::make_vec3(v + 3*i) : V(0.f); }
__device__ float grad1(const float *v, int i) { return v ? v[i] : 0.f; }
__device__ float *out3(void *p, int i) { return static_cast<float *>(p) + 3*i; }
__device__ void put3(float *p, V v) { p[0] = v.x; p[1] = v.y; p[2] = v.z; }

// Explicit three-term contractions preserve gsplat's FP32 rounding. A changed
// radius at an integer boundary can alter the tile list and every later blend.
__device__ float dot_yxz(V a, V b) {
    return __fmaf_rn(a.z, b.z, __fmaf_rn(a.x, b.x, __fmul_rn(a.y, b.y)));
}
// Adjoint of the symmetric 2D inverse. Thin ellipses amplify rounding in
// these cancelling products, so contract the same pairs as gsplat.
__device__ M2 inverse_adjoint(V p, V g) {
    const float off = .5f * g.y;
    const float shared = __fmul_rn(-p.y, off);
    float aa = __fmaf_rn(-p.x, g.x, shared);
    float ba = __fmaf_rn(-p.y, g.x, __fmul_rn(-p.z, off));
    float ab = __fmaf_rn(-p.x, off, __fmul_rn(-p.y, g.z));
    float bb = __fmaf_rn(-p.z, g.z, shared);
    return M2(__fmaf_rn(p.x, aa, __fmul_rn(p.y, ab)), __fmaf_rn(p.x, ba, __fmul_rn(p.y, bb)),
              __fmaf_rn(p.y, aa, __fmul_rn(p.z, ab)), __fmaf_rn(p.y, ba, __fmul_rn(p.z, bb)));
}
__device__ M covariance(M axes, V s) {
    M rows = glm::transpose(M(axes[0]*s.x, axes[1]*s.y, axes[2]*s.z));
    M cov;
    for (int j = 0; j < 3; ++j) for (int k = j; k < 3; ++k)
        cov[j][k] = cov[k][j] = dot_yxz(rows[j], rows[k]);
    return cov;
}
__device__ M camera_covariance(M R, M cov) {
    M rows = glm::transpose(R), tmp, result;
    for (int c = 0; c < 3; ++c) for (int r = 0; r < 3; ++r) {
        V a = rows[r], b = cov[c];
        tmp[c][r] = c < 2 ? dot_yxz(a,b)
                          : __fmaf_rn(a.z,b.z,__fmaf_rn(a.y,b.y,__fmul_rn(a.x,b.x)));
    }
    tmp = glm::transpose(tmp);
    for (int c = 0; c < 3; ++c) for (int r = 0; r < 3; ++r)
        result[c][r] = dot_yxz(tmp[r], rows[c]);
    return result;
}
__device__ V precision_scale(V s, bool &regular) {
    const int axis = s.x <= s.y && s.x <= s.z ? 0 : (s.y <= s.z ? 1 : 2);
    regular = s[axis] > 1e-7f;
    V inv(0.f);
    if (regular) {
        for (int k = 0; k < 3; ++k) inv[k] = __fdiv_rn(s[axis], s[k]);
    } else {
        inv[axis] = 1.f;
    }
    return inv;
}

// Match the torch matrix products used by the independent RaDe path.
template<bool Batched = false>
__device__ M mm(M a, M b) {
    M result;
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) {
            float head = __fmaf_rn(a[1][r], b[c][1], __fmul_rn(a[0][r], b[c][0]));
            result[c][r] = Batched ? __fadd_rn(head, __fmul_rn(a[2][r], b[c][2]))
                                   : __fmaf_rn(a[2][r], b[c][2], head);
        }
    return result;
}

struct NoStep {};

__device__ void update_geometry(const lite_raster::ProjectionArgs &a, const lite_raster::ProjectionStep &step,
                                int i, V gm, V gs, Q gq) {
    const int64_t row = step.ids[i];
    if (step.geometry_grad) {
        const Q q = glm::make_vec4(a.quat + 4*i);
        float length = step.length[i], norm = fmaxf(length, 1.e-12f);
        float dot = length >= 1.e-12f ? q[0]*gq[0] + q[1]*gq[1] + q[2]*gq[2] + q[3]*gq[3] : 0.f;
        for (int j = 0; j < 3; ++j) adam_update(step.groups[0], row*3+j, gm[j]);
        for (int j = 0; j < 4; ++j)
            adam_update(step.groups[2], row*4+j, __fdiv_rn(gq[j] - q[j]*dot, norm));
    }
    if (step.geometry_grad || step.d_scale) {
        gs += grad3(step.d_scale, i);
        for (int j = 0; j < 3; ++j)
            adam_update(step.groups[1], row*3+j, a.scale[3*i+j] * gs[j]);
    }
    if (step.d_opacity) {
        float alpha = step.alpha[i];
        float grad = step.d_opacity[i];
        if (a.out[Compensation]) grad *= static_cast<const float *>(a.out[Compensation])[i];
        adam_update(step.groups[3], row, grad * (1.f - alpha) * alpha);
    }
}

template<int Mode, bool Back, class Step>
__global__ void project(lite_raster::ProjectionArgs a, Step step) {
    constexpr bool fused = !std::is_same_v<Step, NoStep>;
    constexpr bool plane = Mode & Planes, normal = Mode & Normals, legacy = Mode & Legacy;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.n) return;
    int *radii = static_cast<int *>(a.out[Radius]);
    if constexpr (Back) {
        if constexpr (fused) {
            if (!step.geometry_grad) {
                update_geometry(a, step, i, V(0.f), V(0.f), Q(0.f));
                return;
            }
        }
        if (!radii[i]) {
            if constexpr (fused) {
                update_geometry(a, step, i, V(0.f), V(0.f), Q(0.f));
            } else {
                if (a.gm) put3(a.gm + 3*i, V(0.f));
                if (a.gs) put3(a.gs + 3*i, V(0.f));
                if (a.gq) *reinterpret_cast<float4 *>(a.gq + 4*i) = make_float4(0,0,0,0);
            }
            return;
        }
    } else {
        radii[i] = 0;
        if constexpr (fused) step.opacity[i] = a.out[Compensation] ? 0.f : step.alpha[i];
        put3(out3(a.out[Direction], i), V(0.f));
        if (a.out[Compensation]) static_cast<float *>(a.out[Compensation])[i] = 0.f;
        if constexpr (legacy) {
            put3(out3(a.out[LegacyNormal], i), V(0.f));
            static_cast<float *>(a.out[Distance])[i] = 0.f;
        }
        if constexpr (plane) put3(out3(a.out[Plane], i), V(0.f));
        if constexpr (normal) put3(out3(a.out[Normal], i), V(0.f));
    }

    const float *v = a.view;
    const M R(v[0],v[4],v[8], v[1],v[5],v[9], v[2],v[6],v[10]);
    const V mean = glm::make_vec3(a.mean + 3*i), t(v[3],v[7],v[11]);
    const V m = R * mean + t;
    if constexpr (!Back) if (m.z < 0.01f || m.z > 1e10f) return;
    const V s = glm::make_vec3(a.scale + 3*i);
    const Q quat = glm::make_vec4(a.quat + 4*i);
    const M axes = quat_to_rotmat(quat), cov = covariance(axes, s);
    const M cov_c = camera_covariance(R, cov);
    const float fx = a.K[0], fy = a.K[4], cx = a.K[2], cy = a.K[5];
    const V dir = mean - glm::make_vec3(a.center);

    if constexpr (!Back) {
        M2 c2, inv;
        vec2<float> xy;
        persp_proj(m, cov_c, fx, fy, cx, cy, a.w, a.h, c2, xy);
        float compensation;
        const float det = add_blur(a.eps, c2, compensation);
        if (det <= 0.f) return;
        inverse(c2, inv);
        const float half_trace = .5f * (c2[0][0] + c2[1][1]);
        const float radius = ceilf(3.f * sqrtf(half_trace + sqrtf(fmaxf(.01f, half_trace*half_trace-det))));
        if (radius <= a.clip || xy.x+radius <= 0 || xy.x-radius >= a.w ||
            xy.y+radius <= 0 || xy.y-radius >= a.h) return;
        radii[i] = int(radius);
        if constexpr (fused)
            if (a.out[Compensation]) step.opacity[i] = step.alpha[i] * compensation;
        float *pos = static_cast<float *>(a.out[XY]) + 2*i;
        pos[0] = xy.x; pos[1] = xy.y;
        static_cast<float *>(a.out[Z])[i] = m.z;
        put3(out3(a.out[Conic], i), V(inv[0][0], inv[0][1], inv[1][1]));
        put3(out3(a.out[Direction], i), dir);
        if (a.out[Compensation]) static_cast<float *>(a.out[Compensation])[i] = compensation;
        if constexpr (legacy) {
            float d = glm::dot(dir, axes[2]);
            float sign = d > 0 ? -1.f : 1.f;
            put3(out3(a.out[LegacyNormal], i), sign * axes[2]);
            static_cast<float *>(a.out[Distance])[i] = sign*d;
        }
    }

    V gm(0.f), gs(0.f), gt(0.f);
    Q gq(0.f);
    M gR(0.f);
    if constexpr (plane || normal) {
        if (!Back || a.up[Plane] || a.up[Normal]) {
            bool regular;
            V inv_scale = precision_scale(s, regular);
            M prec = covariance(axes, inv_scale);
            M pc_row = glm::transpose(mm(mm<true>(R, prec), glm::transpose(R)));
            // torch's separate mean GEMM contracts x,y,z in this order,
            // followed by a separately rounded addition of the translation.
            V rade_m;
            for (int j = 0; j < 3; ++j)
                rade_m[j] = __fadd_rn(t[j], __fmaf_rn(R[2][j], mean.z,
                       __fmaf_rn(R[1][j], mean.y, __fmul_rn(R[0][j], mean.x))));
            V gm_r(0.f);
            M gp_row(0.f);
            const V gplane = grad3(a.up[Plane], i), gnormal = grad3(a.up[Normal], i);
            if (Back && !a.gv) {
                const M Rt = glm::transpose(R);
                rade_geometry<plane, normal, true, true>(
                    glm::value_ptr(rade_m), glm::value_ptr(pc_row), a.K, a.w, a.h,
                    nullptr, nullptr, glm::value_ptr(gplane), glm::value_ptr(gnormal), glm::value_ptr(Rt),
                    glm::value_ptr(gm_r), glm::value_ptr(gp_row));
            } else {
                rade_geometry<plane, normal, Back, false>(
                    glm::value_ptr(rade_m), glm::value_ptr(pc_row), a.K, a.w, a.h,
                    !Back && plane ? out3(a.out[Plane], i) : nullptr, !Back && normal ? out3(a.out[Normal], i) : nullptr,
                    glm::value_ptr(gplane), glm::value_ptr(gnormal), nullptr,
                    glm::value_ptr(gm_r), glm::value_ptr(gp_row));
            }
            if constexpr (Back) {
                M gp = glm::transpose(gp_row);
                if (a.gv) {
                    pos_world_to_cam_vjp(R, t, mean, gm_r, gR, gt, gm);
                    gR += gp * R * glm::transpose(prec) + glm::transpose(gp) * R * prec;
                    gp = mm(glm::transpose(R), mm(gp, R));
                } else gm += gm_r;
                V gi(0.f);
                quat_scale_to_covar_vjp(quat, inv_scale, axes, gp, gq, gi);
                if (regular)
                    for (int k = 0; k < 3; ++k) gs[k] -= __fdiv_rn(gi[k]*inv_scale[k], s[k]);
            }
        }
    }

    if constexpr (Back) {
        V conic = glm::make_vec3(out3(a.out[Conic], i)), gc = grad3(a.up[Conic], i);
        const M2 inv(conic.x,conic.y,conic.y,conic.z);
        M2 gc2 = inverse_adjoint(conic, gc);
        float gcomp = grad1(a.up[Compensation], i);
        if constexpr (fused)
            if (step.d_opacity) gcomp += __fmul_rn(step.d_opacity[i], step.alpha[i]);
        if (a.out[Compensation] && (a.up[Compensation] || gcomp != 0.f))
            add_blur_vjp(a.eps, inv, static_cast<float *>(a.out[Compensation])[i], gcomp, gc2);
        for (int j = 0; j < 4; j += 2)
            if (a.stats[j])
                project_position_moments(m, R, fx, fy,
                                         a.stats[j]+6*i, a.stats[j+1]+10*i);
        M gcov_c(0.f), gcov(0.f);
        V gmc(0.f);
        persp_proj_vjp(m, cov_c, fx, fy, cx, cy, a.w, a.h, gc2,
                       a.up[XY] ? glm::make_vec2(a.up[XY]+2*i) : vec2<float>(0.f), gmc, gcov_c);
        gmc.z += grad1(a.up[Z], i);
        pos_world_to_cam_vjp(R, t, mean, gmc, gR, gt, gm);
        covar_world_to_cam_vjp(R, cov, gcov_c, gR, gcov);
        gm += grad3(a.up[Direction], i);
        if constexpr (legacy) {
            const float sign = glm::dot(dir, axes[2]) > 0 ? -1.f : 1.f;
            float gd = sign * grad1(a.up[Distance], i);
            gm += gd * axes[2];
            V gn = sign * grad3(a.up[LegacyNormal], i) + gd * dir;
            quat_scale_to_normal_covar_vjp(quat, s, axes, gcov, gn, gq, gs);
        } else {
            quat_scale_to_covar_vjp(quat, s, axes, gcov, gq, gs);
        }
        if constexpr (fused) {
            update_geometry(a, step, i, gm, gs, gq);
        } else {
            if (a.gm) put3(a.gm + 3*i, gm);
            if (a.gs) put3(a.gs + 3*i, gs);
            if (a.gq) for (int j = 0; j < 4; ++j) a.gq[4*i+j] = gq[j];
        }
        if (a.gv) {
            auto group = cooperative_groups::coalesced_threads();
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) {
                float value = c == 3 ? gt[r] : gR[c][r];
                for (int step = 1; step < group.size(); step *= 2) {
                    float other = group.shfl_down(value, step);
                    if (int(group.thread_rank()) + step < int(group.size())) value += other;
                }
                if (group.thread_rank() == 0) atomicAdd(a.gv + r*4+c, value);
            }
        }
    }
}

template<int Mode, class Step>
void launch(const lite_raster::ProjectionArgs &a, cudaStream_t stream, bool back, Step step) {
    const int blocks = (a.n - 1) / 128 + 1;
    if (back) project<Mode, true><<<blocks,128,0,stream>>>(a, step);
    else project<Mode, false><<<blocks,128,0,stream>>>(a, step);
}
template<class Step>
int dispatch(const lite_raster::ProjectionArgs *a, int mode, cudaStream_t stream, bool back, Step step) {
    if (!a->n) return cudaSuccess;
    switch (mode) {
        case 0: launch<0>(*a, stream, back, step); break;
        case Planes: launch<Planes>(*a, stream, back, step); break;
        case Normals: launch<Normals>(*a, stream, back, step); break;
        case Planes | Normals: launch<Planes | Normals>(*a, stream, back, step); break;
        case Legacy: launch<Legacy>(*a, stream, back, step); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

} // namespace

extern "C" int projection(const lite_raster::ProjectionArgs *a, int mode, cudaStream_t stream, bool back) {
    return dispatch(a, mode, stream, back, NoStep{});
}

extern "C" int projection_adam(const lite_raster::ProjectionArgs *a, int mode, cudaStream_t stream, bool back,
                               const lite_raster::ProjectionStep *step) {
    return dispatch(a, mode, stream, back, *step);
}
