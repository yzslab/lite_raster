"""Host extension for submitting the existing projection and tile kernels in one call."""
import functools
import hashlib
import importlib.util
import os
import sysconfig
from pathlib import Path

import torch

from ._toolchain import HERE, _compiler, _publish, _version, compiled_path
from .backend import gpu_cc
from .intersections import _opacity


@functools.lru_cache(None)
def _extension(cc):
    # Link the exact GPU specializations used by the standalone operators.
    projection = compiled_path("fused_projection", cc)
    intersections = compiled_path("intersections", cc)
    cuda = Path(_compiler()).parents[1]
    torch_root = Path(torch.__file__).resolve().parent
    host = Path(os.getenv("LITE_RASTER_CXX") or os.getenv("CXX") or os.getenv("NVCC_CCBIN") or "c++")
    if host.is_dir():
        host = host / "g++"
    source = HERE / "project_encode.cpp"
    command = [str(host), "-std=c++17", "-O3", "-shared", "-fPIC", "-fvisibility=hidden",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    for name in ("COMPILER_TYPE", "STDLIB", "BUILD_ABI"):
        value = getattr(torch._C, "_PYBIND11_" + name, None)
        if value:
            command.append(f'-DPYBIND11_{name}="{value}"')
    command += [f"-I{torch_root}/include", f"-I{cuda}/include", "-I" + sysconfig.get_path("include"),
                str(source), f"-L{projection.parent}", "-Wl,-rpath,$ORIGIN",
                f"-l:{projection.name}", f"-l:{intersections.name}",
                f"-L{torch_root}/lib", "-ltorch_python", "-ltorch_cpu", "-lc10_cuda", "-lc10",
                f"-L{cuda}/lib64", "-lcudart", f"-Wl,-rpath,{cuda}/lib64"]
    digest = hashlib.sha256(_version(str(host)))
    digest.update(source.read_bytes())
    digest.update((HERE / "projection_api.h").read_bytes())
    digest.update(repr((command, torch.__version__, torch.version.cuda, sysconfig.get_config_var("SOABI"))).encode())
    target = projection.parent / f"project_encode_sm{cc}_{digest.hexdigest()[:24]}.so"
    _publish(target, command)
    spec = importlib.util.spec_from_file_location("_lite_raster_project_encode", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project_encode(inputs, outputs, spec, kind, tile, alpha, adam_opacity=None):
    if kind not in ("precise", "bbox"):
        raise ValueError("chained_isect must be 'precise' or 'bbox'")
    if kind == "precise" and adam_opacity is None:
        if alpha is None:
            raise ValueError("Precise chained tile encoding requires activated isect_opacities")
        alpha = _opacity(alpha, inputs[0], len(inputs[0]))
    module = _extension(gpu_cc(inputs[0].device))
    launch = module.precise if kind == "precise" else module.bbox
    return launch(inputs, outputs, alpha, adam_opacity, *spec, tile)
