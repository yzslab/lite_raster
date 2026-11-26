"""Load on-demand CUDA specializations and launch on the calling torch stream."""

import ctypes as ct
import functools

import torch
from ._toolchain import HERE, RASTER_LIBRARIES, _compiler, compiled_path

P = ct.c_void_p


class Args(ct.Structure):
    _fields_ = [
        ("inputs", P * 11),
        ("outputs", P * 9),
        ("cache", P * 7),
        ("upstream", P * 9),
        ("grads", P * 8),
        ("absgrad", P),
        ("moments", P),
        ("absgrad_candidates", P),
        ("width", ct.c_int),
        ("height", ct.c_int),
        ("count", ct.c_int),
    ]


class AdamGroup(ct.Structure):
    _fields_ = [(name, P) for name in ("param", "first", "second")] + [
        (name, ct.c_float) for name in ("lr", "beta1", "beta2", "eps")]


class ProjectionStep(ct.Structure):
    _fields_ = [("groups", AdamGroup * 4)] + [
        (name, P) for name in ("ids", "alpha", "length", "opacity", "d_opacity", "d_scale")
    ] + [("geometry_grad", ct.c_bool)]


class TilePairs(ct.Structure):
    _fields_ = [(name, P) for name in ("opacity", "counts", "keys", "ids")] + [
        (name, ct.c_int) for name in ("tile_size", "width", "height")]


class ProjectionArgs(ct.Structure):
    _fields_ = [
        ("inputs", P * 6), ("outputs", P * 10), ("upstream", P * 10),
        ("grads", P * 4), ("stats", P * 4),
        ("n", ct.c_int), ("width", ct.c_int), ("height", ct.c_int),
        ("eps", ct.c_float), ("radius_clip", ct.c_float),
    ]


@functools.lru_cache(None)
def library(source, cc, channels=0, flags=0):
    lib = ct.CDLL(str(compiled_path(source, cc, channels, flags)))
    if source in RASTER_LIBRARIES:
        lib.launch.argtypes = [ct.POINTER(Args), P, ct.c_bool]
    elif source == "fused_projection":
        lib.projection.argtypes = [ct.POINTER(ProjectionArgs), ct.c_int, P, ct.c_bool]
        lib.projection_adam.argtypes = [*lib.projection.argtypes, ct.POINTER(ProjectionStep)]
    elif source == "intersections":
        lib.intersect_tiles.argtypes = [ct.POINTER(TilePairs), ct.c_int, P, P, P, P, P, P]
        lib.sort_pairs.argtypes = [P, ct.POINTER(ct.c_size_t), P, P, P, P, ct.c_int, ct.c_int,
                                   ct.POINTER(ct.c_int), P]
        lib.encode_offsets.argtypes = [P, ct.c_int, P, ct.c_int, P]
    elif source == "joint_loss":
        lib.joint_loss.argtypes = [ct.POINTER(P), ct.c_int, ct.c_int, ct.c_int, ct.c_float, ct.c_float, ct.c_bool, P]
    elif source == "tile_loss":
        lib.tile_loss.argtypes = [ct.POINTER(P), ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                                  ct.c_float, ct.c_float, ct.c_bool, P]
    elif source == "cluster_adam":
        lib.cluster_adam.argtypes = [P] * 5 + [ct.c_int64, ct.c_int64, ct.c_int, ct.c_bool] + [ct.c_float] * 4 + [P]
    elif source in ("cluster_cull", "cluster_bounds"):
        lib.cluster_cull.argtypes = [P] * 5 + [ct.c_int] * 4 + [ct.c_float, P, P]
        if source == "cluster_bounds":
            lib.refresh_bounds.argtypes = [P, P, ct.c_int, P, ct.c_int, P, P, P]
            lib.select_blocks.argtypes = [P, ct.c_int, ct.c_int, P, P, P]
            lib.expand_rows.argtypes = [P, ct.c_int, P, P]
    elif source == "cluster_preprocess":
        lib.geometry.argtypes = [ct.POINTER(P), ct.c_int, ct.c_bool, P]
        lib.colors.argtypes = [ct.POINTER(P), ct.c_int, ct.c_int, ct.c_int, P]
        lib.colors_adam.argtypes = [ct.POINTER(P), ct.POINTER(AdamGroup), ct.c_int, ct.c_int, ct.c_int, P]
        lib.stats.argtypes = [ct.POINTER(P), ct.c_int, P]
    else:
        lib.sh_color.argtypes = [ct.POINTER(P), ct.c_int, ct.c_int, ct.c_int, ct.c_bool, P, ct.c_bool]
    return lib


def pointer(t):
    return t.data_ptr() if t is not None else None


def gpu_cc(device):
    a, b = torch.cuda.get_device_capability(device)
    return 10 * a + b


def check_launch(err):
    if err:
        raise RuntimeError(f"lite_raster CUDA launch failed with cudaError_t={err}")
