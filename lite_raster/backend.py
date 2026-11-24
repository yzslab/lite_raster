"""Build private CUDA specializations and launch on the calling torch stream."""

import ctypes as ct
import functools
import hashlib
import os
import subprocess
import shutil
from importlib.util import find_spec
from pathlib import Path

import torch
from filelock import FileLock

HERE = Path(__file__).resolve().parent / "csrc"
P = ct.c_void_p
_RASTER_LIBRARIES = ("raster", "warp_backward", "chunk_backward")


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


class ProjectionArgs(ct.Structure):
    _fields_ = [
        ("inputs", P * 6), ("outputs", P * 10), ("upstream", P * 10),
        ("grads", P * 4), ("stats", P * 4),
        ("n", ct.c_int), ("width", ct.c_int), ("height", ct.c_int),
        ("eps", ct.c_float), ("radius_clip", ct.c_float),
    ]

def _compiler():
    nvcc = os.environ.get("LITE_RASTER_NVCC") or shutil.which("nvcc")
    if not nvcc:
        root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
        nvcc = str(Path(root) / "bin/nvcc")
    if not Path(nvcc).is_file():
        raise RuntimeError("lite_raster requires nvcc. Set CUDA_HOME or LITE_RASTER_NVCC.")
    return str(Path(nvcc).resolve())


def _glm_include():
    explicit = os.environ.get("LITE_RASTER_GLM_INCLUDE")
    if explicit:
        candidates = [Path(explicit)]
    else:
        gsplat = find_spec("gsplat")
        candidates = []
        if gsplat and gsplat.origin:
            candidates.append(Path(gsplat.origin).parent / "cuda/csrc/third_party/glm")
        candidates.extend((Path("/usr/local/include"), Path("/usr/include")))
    for root in candidates:
        if (root / "glm/gtc/type_ptr.hpp").is_file():
            return root
    raise RuntimeError("Install GLM headers or point LITE_RASTER_GLM_INCLUDE at the directory containing glm/.")


@functools.lru_cache(None)
def library(source, cc, channels=0, flags=0):
    path = HERE / (source + ".cu")
    digest = hashlib.sha256(path.read_bytes())
    nvcc = _compiler()
    cuda_lib = Path(nvcc).parent.parent / "lib64"
    options = ["-std=c++17", "-O3", "--shared", "-Xcompiler", "-fPIC", "--cudart", "shared",
               "-Xlinker", "-rpath", "-Xlinker", str(cuda_lib), f"-arch=sm_{cc}",
               f"-DCHANNELS={channels}", f"-DFLAGS={flags}"]
    compiler = os.environ.get("LITE_RASTER_CXX") or os.environ.get("CXX")
    if compiler:
        options += ["-ccbin", compiler]
    if source == "fused_projection":
        headers = ("rade_geometry.cuh", "pinhole.cuh")
        glm = _glm_include()
        options += ["--use_fast_math", "--expt-relaxed-constexpr", "-I" + str(glm)]
        digest.update((glm / "glm/detail/setup.hpp").read_bytes())
    elif source == "chunk_backward":
        headers = ("raster_args.cuh", "chunk_maps.cuh")
    elif source in _RASTER_LIBRARIES:
        headers = ("raster_args.cuh",)
    elif source == "joint_loss":
        options += ["--use_fast_math"]
        headers = ()
    elif source == "cluster_preprocess":
        headers = ("sh_color.cu",)
    elif source == "cluster_bounds":
        headers = ("cluster_cull.cu",)
    else:
        headers = ()
    for header in headers:
        digest.update((HERE / header).read_bytes())
    digest.update(repr((nvcc, os.stat(nvcc).st_mtime_ns, options)).encode())
    cache = Path(os.getenv("LITE_RASTER_CACHE_DIR") or
                 Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache") / "lite_raster")
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{source}_{digest.hexdigest()[:20]}.so"
    with FileLock(str(target) + ".lock"):
        if not target.is_file():
            tmp = target.with_suffix(".tmp.so")
            try:
                subprocess.run([nvcc, *options, str(path), "-o", str(tmp)], check=True)
                tmp.replace(target)
            finally:
                tmp.unlink(missing_ok=True)
    lib = ct.CDLL(str(target))
    if source in _RASTER_LIBRARIES:
        lib.launch.argtypes = [ct.POINTER(Args), P, ct.c_bool]
    elif source == "fused_projection":
        lib.projection.argtypes = [ct.POINTER(ProjectionArgs), ct.c_int, P, ct.c_bool]
    elif source == "joint_loss":
        lib.joint_loss.argtypes = [ct.POINTER(P), ct.c_int, ct.c_int, ct.c_int, ct.c_float, ct.c_bool, P]
    elif source == "tile_loss":
        lib.tile_loss.argtypes = [ct.POINTER(P), ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                                  ct.c_float, ct.c_bool, P]
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
