"""Compile one CUDA specialization into the shared on-demand cache."""
from contextlib import contextmanager, suppress
import errno
import functools
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import time
from importlib.util import find_spec
from pathlib import Path


HERE = Path(__file__).resolve().parent / "csrc"
RASTER_LIBRARIES = ("raster", "warp_backward", "chunk_backward")
_LOCK_WAIT_SECONDS = 60
# Local includes that affect each translation unit. Some .cu files are included as headers.
_INCLUDES = {
    "raster": ("raster_args.cuh",),
    "warp_backward": ("raster_args.cuh",),
    "chunk_backward": ("raster_args.cuh", "chunk_maps.cuh"),
    "fused_projection": ("pinhole.cuh", "rade_geometry.cuh", "adam.cuh", "projection_api.h"),
    "intersections": ("tile_intersections.cuh", "projection_api.h"),
    "tile_loss": ("rgb_error.cuh",),
    "joint_loss": ("rgb_error.cuh",),
    "cluster_preprocess": ("sh_color.cu", "adam.cuh", "projection_api.h"),
    "cluster_bounds": ("cluster_cull.cu",),
    "cluster_cull": (), "cluster_adam": ("adam.cuh", "projection_api.h"), "sh_color": (),
}


def _compiler():
    nvcc = os.environ.get("LITE_RASTER_NVCC") or shutil.which("nvcc")
    if not nvcc:
        root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
        nvcc = str(Path(root) / "bin/nvcc")
    if not Path(nvcc).is_file():
        raise RuntimeError("Building lite_raster requires nvcc. Set CUDA_HOME or LITE_RASTER_NVCC.")
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
        candidates.extend((Path(__file__).resolve().parents[2] / "gsplat/gsplat/cuda/csrc/third_party/glm",
                           Path("/usr/local/include"), Path("/usr/include")))
    for root in candidates:
        if (root / "glm/gtc/type_ptr.hpp").is_file():
            return root
    raise RuntimeError("Install GLM headers or point LITE_RASTER_GLM_INCLUDE at the directory containing glm/.")


@functools.lru_cache(None)
def _version(compiler):
    return subprocess.check_output([compiler, "--version"])


@contextmanager
def _build_lock(target):
    """Use atomic mkdir across both NFS clients and local access on the server."""
    marker = target.with_suffix(".building")
    deadline = time.monotonic() + _LOCK_WAIT_SECONDS
    owned = False
    while True:
        try:
            marker.mkdir()
            owned = True
            break
        except FileExistsError:
            if target.is_file() or time.monotonic() >= deadline:
                break
            time.sleep(0.2)
        except OSError:
            break
    try:
        yield
    finally:
        if owned:
            with suppress(OSError):
                marker.rmdir()


def _publish(target, command):
    if target.is_file():
        return
    with _build_lock(target):
        if target.is_file():
            return
        # Only the owner cleans this directory. Even two nodes that both believe
        # they hold the lock use separate NVCC intermediates and linker outputs.
        with tempfile.TemporaryDirectory(dir=target.parent, prefix=".build-") as directory:
            output = Path(directory) / target.name
            try:
                subprocess.run([*command, "-o", str(output)], check=True, env={**os.environ, "TMPDIR": directory})
            except subprocess.CalledProcessError:
                if target.is_file():
                    return
                raise
            try:
                # Same-filesystem link creates the final name only once. Losing
                # this race is a successful cache hit.
                os.link(output, target)
            except FileExistsError:
                pass
            except OSError as exc:
                if target.is_file():
                    return
                if exc.errno not in (errno.EOPNOTSUPP, errno.ENOSYS, errno.EPERM):
                    raise
                # Filesystems without hard links can atomically replace one
                # complete binary with another for the same specialization.
                os.replace(output, target)


def compiled_path(source, cc, channels=0, flags=0):
    if source not in _INCLUDES:
        raise ValueError(f"Unknown lite_raster operation: {source}")
    if source in RASTER_LIBRARIES:
        if not 0 <= channels <= 513 or not 0 <= flags < 2048:
            raise ValueError(f"Unsupported raster specialization: channels={channels}, flags={flags}")
    elif source in ("joint_loss", "tile_loss"):
        if flags not in (0, 1, 2):
            raise ValueError(f"Unknown lite_raster loss mode: {flags}")
        channels = 0
    else:
        channels = flags = 0
    nvcc = _compiler()
    options = ["-std=c++17", "-O3", "--shared", "-Xcompiler", "-fPIC", "--cudart", "shared",
               "-Xlinker", "-rpath", "-Xlinker", str(Path(nvcc).parents[1] / "lib64"),
               f"-arch=sm_{cc}", f"-DCHANNELS={channels}", f"-DFLAGS={flags}"]
    compiler = os.environ.get("LITE_RASTER_CXX") or os.environ.get("CXX") or os.environ.get("NVCC_CCBIN")
    if compiler:
        options += ["-ccbin", compiler]
    host = Path(compiler or "gcc")
    if host.is_dir():
        host = host / "gcc"
    digest = hashlib.sha256()
    digest.update(_version(nvcc))
    digest.update(_version(str(host)))
    if source in ("fused_projection", "intersections", "joint_loss"):
        options.append("--use_fast_math")
    if source == "fused_projection":
        glm = _glm_include()
        digest.update((glm / "glm/detail/setup.hpp").read_bytes())
        options += ["--expt-relaxed-constexpr", "-I" + str(glm)]
    source_file = HERE / (source + ".cu")
    for filename in (source_file.name, *_INCLUDES[source]):
        digest.update((HERE / filename).read_bytes())
    env_flags = tuple(os.environ.get(name) for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"))
    digest.update(repr((platform.machine(), nvcc, options, env_flags)).encode())
    cache = Path(os.getenv("LITE_RASTER_CACHE_DIR") or
                 Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache") / "lite_raster").absolute()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{source}_sm{cc}_{channels}_{flags}_{digest.hexdigest()[:24]}.so"
    _publish(target, [nvcc, *options, str(source_file)])
    return target
