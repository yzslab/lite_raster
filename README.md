# lite_raster

PyTorch operators for projecting and alpha-blending FP32 Gaussians in a single pinhole view. The projection returns center Z for tile sorting and optional RaDe ray planes and camera-space normals. The rasterizer uses 16×16 tiles and accepts sorted intersection IDs and offsets supplied by the caller.

## Installation

From the gaussian-splatting-pl checkout, in the CUDA-enabled PyTorch environment:

```sh
python -m pip install ./submodules/lite_raster
```

The wheel contains the CUDA sources. On first use, NVCC builds the requested specialization for the active GPU and caches it under `~/.cache/lite_raster` (or `LITE_RASTER_CACHE_DIR`). Use a CUDA toolkit compatible with the installed PyTorch; point `CUDA_HOME` or `LITE_RASTER_NVCC` at the toolkit when necessary. Set `LITE_RASTER_CXX` to select a supported host compiler. Projection uses the GLM headers from the project's installed gsplat, system GLM, or the directory specified by `LITE_RASTER_GLM_INCLUDE`.

## Project renderer

Select `internal.renderers.lite_raster_renderer.LiteRasterRenderer` as `model.renderer.class_path`. It inherits the v1 anti-aliasing, tile intersection, and separate SH settings. The defaults enable fused projection, warp raster backward and fused RGB SH, with `rade: false`. Fused SH propagates gradients to the coefficients. Select `sh_impl: gsplat` to also differentiate the viewing direction; select `rade: true` to render per-pixel RaDe depth and normals. Both modes sort by Gaussian center Z. Basic-mode normals preserve the length of the v1 alpha-blended vector; RaDe outputs unit normals.

The specified second-moment evaluation script accepts `--lite-raster` alongside its usual project name and training arguments. Use a fresh project name for a separate set of checkpoints:

```sh
python utils/eval_mipnerf360-second_moment_guided-opacity_growth_reg-log-rho_85-eigen_max_cancellation-ssim_grad_max-mul_split.py lite-raster-eval --lite-raster
```

The renderer uses the project's gsplat for tile intersections and sorting. In basic mode, v1 handles its specialized LiDAR, minimum-axis and unbiased-depth requests, alternative camera models, and tile sizes other than 16. Ordinary SSIM loss uses RGB. The target density controller reads full-loss position moments and runs a separate auxiliary RGB backward for candidate AbsGrad. Controllers requiring a dedicated `render_ssim` gradient output should use the v1 renderer.

## Operators

The ordinary RGB warp backward defaults to two 16×8 regions per 16×16 tile. Each warp handles one region, skips splats whose alpha stays below the rendering threshold throughout it, and reduces gradients, AbsGrad and position moments before atomic accumulation.

For auxiliary AbsGrad sampling, the warp backward takes the candidate mask from the screen-space points. Each tile replays the occlusion needed by its candidates and accumulates absolute screen-position derivatives only for those candidates. A loss applied only to RGB selects the RGB backward specialization even when the forward returned additional maps.

* `project(pre, means, scales, quats, ...)` takes the v1 camera tuple (world-to-camera matrix, intrinsics, image size, camera center) and returns the v1 projection tuple, an optional RaDe plane, and an optional RaDe normal.
* `rasterize(..., flags=Mode.ALPHA, backward_impl="warp")` returns a fixed `RasterResult` with HWC images and optional per-Gaussian hit flags. Colors and normals have separate inputs. Unrequested outputs are `None`. `backward_impl="pixel"` selects the reference backward.
* `sh_color(degree, dirs, dc, rest=None, visible=None)` evaluates SH degrees 0–4, adds 0.5 and clamps the result to nonnegative RGB.
