/**
 * Precise Tile Intersect proposed by Speedy-Splat: https://speedysplat.github.io/
 * Some code snippets are copied from:
 *   https://github.com/j-alex-hanson/speedy-splat-rasterizer/blob/b38abb6da7a7217999f8c5ec0e9219284464ffe0/cuda_rasterizer/auxiliary.h
 *   https://github.com/nerfstudio-project/gsplat/commit/3d4f9027f36d70ac8e89536e14c247de377f74b5
 */

// Adapted from the local gsplat implementation for dense single-view FP32.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#include "projection_api.h"

namespace lite_raster {

__device__ inline float2 accutile_ellipse_intersection(
    float A, float B, float C, float disc, float t, float2 p,
    bool isY, float coord
) {
    float p_u   = isY ? p.y : p.x;
    float p_v   = isY ? p.x : p.y;
    float coeff = isY ? A : C;

    float h         = coord - p_u;
    float sqrt_term = sqrtf(disc * h * h + t * coeff);

    return {(-B * h - sqrt_term) / coeff + p_v, (-B * h + sqrt_term) / coeff + p_v};
}

template<class Writer>
__device__ inline uint32_t accutile_process_tiles(
    float A, float B, float C, float disc, float t, float2 p,
    float2 bbox_min, float2 bbox_max, float2 bbox_argmin, float2 bbox_argmax,
    int2 rect_min, int2 rect_max,
    uint32_t tile_size, uint32_t tile_width, bool isY,
    Writer &writer
) {
    float BLOCK = (float)tile_size;

    if (isY) {
        rect_min    = {rect_min.y, rect_min.x};
        rect_max    = {rect_max.y, rect_max.x};
        bbox_min    = {bbox_min.y, bbox_min.x};
        bbox_max    = {bbox_max.y, bbox_max.x};
        bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
        bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
    }

    uint32_t tiles_count = 0;
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    intersect_max_line = {bbox_max.y, bbox_min.y};

    min_line = rect_min.x * BLOCK;
    if (bbox_min.x <= min_line) {
        intersect_min_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, min_line);
    } else {
        intersect_min_line = intersect_max_line;
    }

#pragma unroll 1
    for (int u = rect_min.x; u < rect_max.x; ++u) {
        max_line = min_line + BLOCK;
        if (max_line <= bbox_max.x) {
            intersect_max_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, max_line);
        }

        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
            ellipse_min = bbox_min.y;
        } else {
            ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
            ellipse_max = bbox_max.y;
        } else {
            ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }

        int min_tile_v = max(rect_min.y, min(rect_max.y, (int)(ellipse_min / BLOCK)));
        int max_tile_v = min(rect_max.y, max(rect_min.y, (int)(ellipse_max / BLOCK + 1)));

        tiles_count += max(0, max_tile_v - min_tile_v);

        writer.span(u, min_tile_v, max_tile_v, isY);

        intersect_min_line = intersect_max_line;
        min_line           = max_line;
    }
    return tiles_count;
}

template<class Writer>
__device__ inline int precise_tiles(float2 mean2d, float A, float B, float C, float opacity,
                                    int tile_size, int tile_width, int tile_height, Writer &writer) {
    // Calculate discriminant
    float disc = B * B - A * C;

    // If ill-formed ellipse, return
    if (A <= 0 || C <= 0 || disc >= 0) {
        return 0;
    }

    // Threshold: opacity * Gaussian = 1 / 255
    float t = 2.0f * log(opacity * 255.0f);

    float x_term = sqrt(-(B * B * t) / (disc * A));
    x_term = (B < 0) ? x_term : -x_term;
    float y_term = sqrt(-(B * B * t) / (disc * C));
    y_term = (B < 0) ? y_term : -y_term;

    float2 bbox_argmin = { mean2d.y - y_term, mean2d.x - x_term };
    float2 bbox_argmax = { mean2d.y + y_term, mean2d.x + x_term };

    float2 bbox_min = {
      accutile_ellipse_intersection(A, B, C, disc, t, mean2d, true, bbox_argmin.x).x,
      accutile_ellipse_intersection(A, B, C, disc, t, mean2d, false, bbox_argmin.y).x
    };
    float2 bbox_max = {
      accutile_ellipse_intersection(A, B, C, disc, t, mean2d, true, bbox_argmax.x).y,
      accutile_ellipse_intersection(A, B, C, disc, t, mean2d, false, bbox_argmax.y).y
    };

    // Rectangular tile extent of ellipse
    float tile_size_f = (float)tile_size;
    int2 rect_min = {
        max(0, min((int)tile_width, (int)(bbox_min.x / tile_size_f))),
        max(0, min((int)tile_height, (int)(bbox_min.y / tile_size_f)))
    };
    int2 rect_max = {
        max(0, min((int)tile_width, (int)(bbox_max.x / tile_size_f + 1.f))),
        max(0, min((int)tile_height, (int)(bbox_max.y / tile_size_f + 1.f)))
    };

    int y_span = rect_max.y - rect_min.y;
    int x_span = rect_max.x - rect_min.x;

    // If no tiles are touched, return
    if (y_span == 0 || x_span == 0) {
        return 0;
    }

    bool isY = y_span < x_span;

    return accutile_process_tiles(A, B, C, disc, t, mean2d,
        bbox_min, bbox_max, bbox_argmin, bbox_argmax, rect_min, rect_max,
        tile_size, tile_width, isY, writer);
}

}  // namespace lite_raster
