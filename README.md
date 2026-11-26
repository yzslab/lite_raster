# lite_raster

PyTorch operators for projecting and alpha-blending FP32 Gaussians in a single pinhole view. The projection returns center Z for tile sorting and optional RaDe ray planes and camera-space normals. The rasterizer uses 16×16 tiles and accepts sorted intersection IDs and offsets supplied by the caller.

## Installation

From the gaussian-splatting-pl checkout, in the CUDA-enabled PyTorch environment:

```sh
python -m pip install ./submodules/lite_raster
```

The wheel contains the CUDA sources. Installation only packages these files. On first use, the worker compiles the requested operator for the calling GPU's compute capability. Raster kernels specialize on the exact channel count and output flags. For example, an RGB-only run builds its RGB variants, and the first request for a RaDe depth map builds the additional variant it needs. Later processes reuse the compiled libraries.

Workers need NVCC, a compatible host compiler and, for fused projection, GLM headers. Set `CUDA_HOME` or `LITE_RASTER_NVCC` to select the toolkit and `LITE_RASTER_CXX` to select the host compiler. GLM is discovered from gsplat or the system include directories. You can also set `LITE_RASTER_GLM_INCLUDE` to the directory containing `glm/`.

By default libraries are cached under `$XDG_CACHE_HOME/lite_raster` or `~/.cache/lite_raster`. For CLI processes and Slurm jobs on different nodes to share builds, point `LITE_RASTER_CACHE_DIR` at the same writable directory or use a shared home. The cache key includes the kernel, included project headers, GPU architecture, compiler versions and build options.

Each compilation gets a private `.build-*` directory inside the cache for NVCC's temporary files and output. After NVCC succeeds, the worker atomically creates the final `.so` name with a hard link. Concurrent builders reuse the first published binary. On filesystems without hard links, publication uses an atomic rename of the complete binary. Each worker cleans only its own directory. A killed worker can leave a private directory behind, which subsequent jobs ignore.

On a cache miss, workers coordinate by atomically creating a per-specialization `.building` directory. This works across NFS clients and processes accessing the same export directly on the server. The owner removes its marker when finished. If creating it fails, a worker can compile independently. If the marker already exists, a worker waits for the published result or retries acquiring the marker for up to 60 seconds before proceeding independently. A warm cache hit skips this coordination, including any marker left by a killed process.

## Project renderer

Select `internal.renderers.lite_raster_renderer.LiteRasterRenderer` as `model.renderer.class_path`. It inherits the v1 anti-aliasing, tile intersection, and separate SH settings. The defaults enable fused projection, warp raster backward and fused RGB SH, with `rade: false`. Fused SH propagates gradients to the coefficients. Select `sh_impl: gsplat` to also differentiate the viewing direction; select `rade: true` to render per-pixel RaDe depth and normals. Both modes sort by Gaussian center Z. Basic-mode normals preserve the length of the v1 alpha-blended vector; RaDe outputs unit normals.

The specified second-moment evaluation script accepts `--lite-raster` alongside its usual project name and training arguments. Use a fresh project name for a separate set of checkpoints:

```sh
python utils/eval_mipnerf360-second_moment_guided-opacity_growth_reg-log-rho_85-eigen_max_cancellation-ssim_grad_max-mul_split.py lite-raster-eval --lite-raster
```

The renderer uses the project's gsplat for tile intersections and sorting. In basic mode, v1 handles its specialized LiDAR, minimum-axis and unbiased-depth requests, alternative camera models, and tile sizes other than 16. Ordinary SSIM loss uses RGB. The target density controller reads full-loss position moments and runs a separate auxiliary RGB backward for candidate AbsGrad. Controllers requiring a dedicated `render_ssim` gradient output should use the v1 renderer.

Set `--model.renderer.cluster_size=128` to group Gaussian means in Morton order and cull clusters before dense pinhole projection. The default is `0`. The renderer refreshes bounds from current positions and scales on each view and rebuilds membership when densification replaces the model rows. Selected parameters form ordinary contiguous tensors ordered by original row, including equal-depth ties. With this option, per-Gaussian renderer outputs use compact row indices and `outputs["gaussian_ids"]` maps each row back to the model. GNS and SelectiveAdam consume this mapping. The existing v1 fallback renders the full scene.

To train with the separate LiteGS Adam optimizer, append `-c configs/litegs_adam.yaml --model.renderer.cluster_size=128` after the lite_raster renderer config. Adam reads compact gradients and updates global FP32 parameters and moments using the model row IDs. It updates all rows in selected clusters, including zero-gradient rows with previous momentum. An unclustered render with tile intersections selects primitives intersecting a tile. The compact renderer gathers trainable raw parameters before activation and supplies row-sparse gradients to LiteGSAdam parameter groups. A dense regularizer can contribute to the same parameter. Model and optimizer checkpoints retain their usual row-major layout.

Clustered rendering requires the FasterGS MortonEncoding CUDA extension. Install it in the training environment with `python -m pip install --no-build-isolation ~/src/nerficg/src/CudaUtils/MortonEncoding`, adjusting the path to your nerficg checkout.

## Operators

The ordinary RGB warp backward defaults to two 16×8 regions per 16×16 tile. Each warp handles one region, skips splats whose alpha stays below the rendering threshold throughout it, and reduces gradients, AbsGrad and position moments before atomic accumulation.

Append `-c configs/snippets/lite_raster_chunk_backward.yaml` to select `backward_impl="chunk"`. This PennyGS-style FP32 backward partitions each 16×16 tile into eight 8×4 pixel chunks and lets NVCC allocate registers. RGB or feature channels, alpha, inverse/expected/median depth, hard depth/inverse depth/alpha, and raw or normalized normals can contribute gradients in the same backward. Both center-Z and RaDe depth support this mode, including depth-only rendering. The kernel specializes on the images used in the loss, so returning auxiliary images for logging still allows the RGB-only fast path. Trainable backgrounds, AbsGrad and position moments are supported. Candidate-only AbsGrad sampling uses its dedicated warp kernel.

For auxiliary AbsGrad sampling, the warp backward takes the candidate mask from the screen-space points. Each tile replays the occlusion needed by its candidates and accumulates absolute screen-position derivatives only for those candidates. A loss applied only to RGB selects the RGB backward specialization even when the forward returned additional maps.

* `project(pre, means, scales, quats, ...)` takes the v1 camera tuple (world-to-camera matrix, intrinsics, image size, camera center) and returns the v1 projection tuple, an optional RaDe plane, and an optional RaDe normal.
* `rasterize(..., flags=Mode.ALPHA, backward_impl="warp")` returns a fixed `RasterResult` with contiguous CHW images and optional per-Gaussian hit flags. Colors and normals have separate inputs. Unrequested outputs are `None`. `backward_impl="pixel"` selects the reference backward.
* `sh_color(degree, dirs, dc, rest=None, visible=None)` evaluates SH degrees 0–4, adds 0.5 and clamps the result to nonnegative RGB.

### Optimized clustered training in Improved-GS

Append `-c configs/lite_raster-clustered.yaml` to an Improved-GS training configuration with a GNS budget. This selects 128-row Morton clusters, fused visible-row geometry and SH, LiteGSAdam and fused GNS statistics. The original projection, tile ordering and alpha-blending kernels consume the resulting compact rows. GNS can also use `fused_density_stats: true` with `cluster_size: 0`.

The fused training path supports the vanilla FP32 Gaussian model with all six raw parameters managed by LiteGSAdam. The renderer reorders parameters, optimizer moments and registered per-Gaussian state together. Its bounds cache recognizes single visible-row optimizer updates and refreshes all clusters for other tracked parameter changes. Custom operations that write directly through CUDA pointers must increment PyTorch's tensor version after writing. The existing renderer path remains available for other model or optimizer combinations.

To reproduce the 10k bicycle setup, append `configs/lite_raster-clustered-bicycle.yaml` after the clustered preset. This additional file sets the measured LiteGS budget curve, GPU image caching, downsampling and training length.

```sh
python main.py fit -c configs/improved-gs.yaml -c configs/lite_raster-clustered.yaml \
  -c configs/lite_raster-clustered-bicycle.yaml --data.path="$HOME/data/Mip-NeRF360/bicycle" -n bicycle-clustered
```

Since 0.2, all raster image fields use contiguous `[C, H, W]` storage, including RGB, normals and single-channel depth or alpha. Per-Gaussian arrays retain their original layout. Callers that need HWC can use `result.render.permute(1, 2, 0)`.

Append `configs/lite_raster-fused-rgb.yaml` to the clustered bicycle preset for joint L1 + SSIM training. The preset caches targets as contiguous CHW. For ordinary RGB training with GNS or vanilla density control and compatible plugins, it skips the alpha image. Adam and LiteGSAdam also omit the per-Gaussian hit buffer. SelectiveAdam retains it for its optimizer update. Explicit auxiliary requests and evaluation use the full renderer. The warp backward selects an FP32 specialization when training requires only signed gradients or signed gradients plus AbsGrad.

`lite_raster.loss.fused_l1_ssim(image, target, weight)` returns the weighted loss, mean RGB error and SSIM as differentiable scalars. Both fused SSIM functions accept the same `rgb_diff_loss` and `lambda_l2` options as `VanillaMetrics`. For residual `d`, the default `l1` uses `abs(d)`, `l2` uses `d²`, and `l1+l2` uses `abs(d) + lambda_l2 * d²`. Pass `return_pixel_errors=True` to also receive the per-channel RGB error map. Each input is one FP32 CUDA image with a fixed target.

Append `-c configs/snippets/lite_raster_tile_ssim.yaml` after the fused RGB preset to use PennyGS-style Tile SSIM during training. It shares the existing `lambda_dssim` weight and validation metrics. The corresponding Python function is `lite_raster.loss.fused_tile_l1_ssim(image, target, weight, step=global_step)`. It has the same outputs and computes one SSIM value per channel per 16×16 tile. The grid shifts with the training step and partial tiles contribute in proportion to their valid pixel counts.
