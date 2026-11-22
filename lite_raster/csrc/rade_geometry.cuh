#pragma once
#include <cuda_runtime.h>

// Row-major camera-space precision P. World mode returns parameter adjoints
// in world space, using the same contraction order as the separate kernel.
__device__ inline float ray_product(const float *row, float u, float v) {
    return __fadd_rn(__fmaf_rn(row[0], u, __fmul_rn(row[1], v)), row[2]);
}

template <bool Planes, bool Normals, bool Back, bool World>
__device__ inline void rade_geometry(
    const float *m, const float *P, const float *K, int width, int height,
    float *plane, float *normal, const float *plane_grad, const float *normal_grad,
    const float *R, float *gm, float *gP
) {
    const float fx = K[0], fy = K[4];
    const float cx = K[2], cy = K[5];
    const float u_min = __fdiv_rn(-cx - .15f * width, fx), u_max = __fdiv_rn(width - cx + .15f * width, fx);
    const float v_min = __fdiv_rn(-cy - .15f * height, fy), v_max = __fdiv_rn(height - cy + .15f * height, fy);
    const float x = m[0], y = m[1], z = m[2], raw_u = __fdiv_rn(x, z), raw_v = __fdiv_rn(y, z);
    const float u = fminf(u_max, fmaxf(u_min, raw_u)), v = fminf(v_max, fmaxf(v_min, raw_v));
    float qx = ray_product(P + 0, u, v);
    float qy = ray_product(P + 3, u, v);
    float qz = ray_product(P + 6, u, v);
    const float mag = fmaxf(fabsf(qx), fmaxf(fabsf(qy), fabsf(qz)));
    if (!(mag > 0)) {
        if constexpr (!Back && Normals) normal[2] = -1.f;
        return;  // Other outputs and gradients are initialized to zero.
    }
    qx = __fdiv_rn(qx, mag);
    qy = __fdiv_rn(qy, mag);
    qz = __fdiv_rn(qz, mag);
    // Match the common forward's FMA order even when nvcc reuses qx*u
    // elsewhere in the VJP. One ulp can change the floor on grazing rays.
    const float dot = __fadd_rn(__fmaf_rn(qx, u, __fmul_rn(qy, v)), qz);
    const float q_norm = __fsqrt_rn(qx * qx + qy * qy + qz * qz);
    const float floor = 1e-7f * q_norm;
    const float den = dot < floor ? floor : dot;
    const float sx = __fdiv_rn((1 + v * v) * qx - u * v * qy - u * qz, den);
    const float sy = __fdiv_rn((1 + u * u) * qy - u * v * qx - v * qz, den);

    if constexpr (!Back) {
        if constexpr (Planes) {
            const float len = __fsqrt_rn(1 + u*u + v*v);
            plane[0] = __fdiv_rn(__fdiv_rn(sx * z, len), fx);
            plane[1] = __fdiv_rn(__fdiv_rn(sy * z, len), fy);
            plane[2] = __fsqrt_rn(x*x + y*y + z*z);
        }
        if constexpr (Normals) {
            const float nx = -u - sx, ny = -v - sy, nz = sx*u + sy*v - 1;
            const float len = __fsqrt_rn(nx*nx + ny*ny + nz*nz);
            normal[0] = __fdiv_rn(nx, len);
            normal[1] = __fdiv_rn(ny, len);
            normal[2] = __fdiv_rn(nz, len);
        }
        return;
    }
    float du = 0, dv = 0, dx = 0, dy = 0, dz = 0, dsx = 0, dsy = 0;
    if constexpr (Planes) {
        const float *g = plane_grad;
        const float ray_sq = 1 + u * u + v * v, ray_len = __fsqrt_rn(ray_sq);
        const float gx = __fdiv_rn(__fdiv_rn(g[0], fx), ray_len), gy = __fdiv_rn(__fdiv_rn(g[1], fy), ray_len);
        dsx = gx * z;
        dsy = gy * z;
        const float slope_z = gx * sx + gy * sy;
        const float ray_grad = __fdiv_rn(-slope_z * z, ray_sq);
        du = ray_grad * u;
        dv = ray_grad * v;
        const float mean_grad = __fdiv_rn(g[2], __fsqrt_rn(x * x + y * y + z * z));
        dx = mean_grad * x;
        dy = mean_grad * y;
        dz = mean_grad * z + slope_z;
    }
    if constexpr (Normals) {
        const float *g = normal_grad;
        float nx = -u - sx, ny = -v - sy, nz = sx * u + sy * v - 1;
        const float len = __fsqrt_rn(nx * nx + ny * ny + nz * nz);
        nx = __fdiv_rn(nx, len);
        ny = __fdiv_rn(ny, len);
        nz = __fdiv_rn(nz, len);
        const float radial = nx * g[0] + ny * g[1] + nz * g[2];
        const float gx = __fdiv_rn(g[0] - nx * radial, len);
        const float gy = __fdiv_rn(g[1] - ny * radial, len);
        const float gz = __fdiv_rn(g[2] - nz * radial, len);
        dsx += -gx + u * gz;
        dsy += -gy + v * gz;
        du += -gx + sx * gz;
        dv += -gy + sy * gz;
    }

    // VJP of the two slope numerators, followed by their shared denominator.
    const float ax = __fdiv_rn(dsx, den), ay = __fdiv_rn(dsy, den), dden = -(ax * sx + ay * sy);
    const float uv = u * v;
    du += ax * (-v * qy - qz) + ay * (2 * u * qy - v * qx);
    dv += ax * (2 * v * qx - u * qy) + ay * (-u * qx - qz);
    if (dot >= floor) {
        du += dden * qx;
        dv += dden * qy;
    }
    // The common forward treats mag as a constant scale (slopes are homogeneous in P).
    const float h[3] = {u, v, 1}, q[3] = {qx, qy, qz};
    const float num_x[3] = {1 + v*v, -uv, -u}, num_y[3] = {-uv, 1 + u*u, -v};
    float dq[3];
    for (int k = 0; k < 3; ++k) {
        const float dd = dot < floor ? __fdiv_rn(1e-7f, q_norm) * q[k] : h[k];
        dq[k] = __fdiv_rn(ax * num_x[k] + ay * num_y[k] + dden * dd, mag);
    }
    du += P[0] * dq[0] + P[3] * dq[1] + P[6] * dq[2];
    dv += P[1] * dq[0] + P[4] * dq[1] + P[7] * dq[2];
    if constexpr (Planes && !Normals) {
        // For the precision VJP contract the two plane-Jacobian rows last,
        // preserving the FP32 forward's quotient-rule multiply-add order.
        const float inv_mag = 1.f / mag, ray_len = __fsqrt_rn(1 + u*u + v*v);
        const float *gp = plane_grad;
        for (int k = 0; k < 3; ++k) {
            const float prod = q[k] * inv_mag;
            const float dd = dot < floor ? 1e-7f * ((.5f / q_norm) * (prod + prod)) : h[k] * inv_mag;
            const float slope_x = __fmaf_rn(-sx, dd, __fmul_rn(num_x[k], inv_mag)) / den;
            const float slope_y = __fmaf_rn(-sy, dd, __fmul_rn(num_y[k], inv_mag)) / den;
            const float px = slope_x * z / ray_len / fx, py = slope_y * z / ray_len / fy;
            dq[k] = __fadd_rn(__fmul_rn(px, gp[0]), __fmul_rn(py, gp[1]));
        }
    }
    const float gx = (raw_u >= u_min && raw_u <= u_max) ? __fdiv_rn(du, z) : 0.f;
    const float gy = (raw_v >= v_min && raw_v <= v_max) ? __fdiv_rn(dv, z) : 0.f;
    const float gmean[3] = {dx + gx, dy + gy, dz - gx * raw_u - gy * raw_v};
    if constexpr (World) {
        // With fixed cameras, dP_world = R^T (dq h^T) R and dm_world = R^T dm.
        // There is no sum over splats. Keep the same matrix-product order as
        // the camera-space autograd chain, which matters for very thin splats.
        for (int j = 0; j < 3; ++j) {
            gm[j] = gmean[0] * R[j] + gmean[1] * R[3+j] + gmean[2] * R[6+j];
            float column[3];
            for (int k = 0; k < 3; ++k)
                column[k] = (dq[k] * u) * R[j] + (dq[k] * v) * R[3+j] + dq[k] * R[6+j];
            for (int k = 0; k < 3; ++k)
                gP[3*k+j] = R[k] * column[0] + R[3+k] * column[1] + R[6+k] * column[2];
        }
    } else {
        for (int k = 0; k < 3; ++k) {
            gm[k] = gmean[k];
            for (int j = 0; j < 3; ++j)
                gP[3*k+j] = dq[k] * h[j];
        }
    }
}
