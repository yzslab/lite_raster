import ctypes as ct
from enum import IntFlag
from typing import NamedTuple

import torch

from .backend import Args, P, check_launch, gpu_cc, library, pointer


class Mode(IntFlag):
    INVERSE = 1
    HARD = 2
    EXPECTED = 4
    MEDIAN = 8
    NORMAL = 16
    RADE = 32
    HARD_DEPTH = 64
    ALPHA = 128
    HARD_ALPHA = 256
    HIT = 512
    RAW_NORMAL = 1024


_ALL_FLAGS = sum(Mode)


class RasterResult(NamedTuple):
    render: torch.Tensor | None
    alpha: torch.Tensor | None
    inverse_depth: torch.Tensor | None
    hard_inverse_depth: torch.Tensor | None
    hard_alpha: torch.Tensor | None
    exp_depth: torch.Tensor | None
    median_depth: torch.Tensor | None
    normal: torch.Tensor | None
    hard_depth: torch.Tensor | None
    hit: torch.Tensor | None


def _run(inputs, outputs, cache, spec, upstream=(), grads=(), absolute=None, moments=None, candidates=None):
    width, height, flags, backward_impl = spec
    xy, colors = inputs[0], inputs[3]
    args = Args(
        (P * 11)(*map(pointer, inputs)),
        (P * 9)(*map(pointer, outputs)),
        (P * 7)(*map(pointer, cache)),
        (P * 9)(*map(pointer, upstream)),
        (P * 8)(*map(pointer, grads)),
        pointer(absolute),
        pointer(moments),
        pointer(candidates),
        width,
        height,
        inputs[-1].numel(),
    )
    with torch.cuda.device(xy.device):
        source = "warp_backward" if grads and backward_impl == "warp" else "raster"
        lib = library(source, gpu_cc(xy.device), colors.shape[-1], int(flags))
        check_launch(
            lib.launch(ct.byref(args), torch.cuda.current_stream().cuda_stream, bool(grads))
        )


class _Raster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, conic, opacity, colors, z, plane, normals, bg, K, offsets, ids, spec):
        w, h, f, _ = spec
        ctx.spec = spec
        inputs = (xy, conic, opacity, colors, z, plane, normals, bg, K, offsets, ids)

        def image(channels, enabled=True, dtype=None):
            if enabled:
                return torch.empty((h, w, channels), dtype=dtype or xy.dtype, device=xy.device)
            return None

        hard = f & (Mode.HARD | Mode.HARD_DEPTH | Mode.HARD_ALPHA)
        outputs = (
            image(colors.shape[1], colors.shape[1] > 0),
            image(1, f & Mode.ALPHA),
            image(1, f & Mode.INVERSE),
            image(1, f & Mode.HARD),
            image(1, f & Mode.HARD_ALPHA),
            image(1, f & Mode.EXPECTED),
            image(1, f & Mode.MEDIAN),
            image(3, f & Mode.NORMAL),
            image(1, f & Mode.HARD_DEPTH),
        )
        cache = (
            image(1, dtype=torch.int),
            image(1, hard, torch.int),
            image(1, f & Mode.MEDIAN, torch.int),
            image(1, f & Mode.NORMAL and not f & Mode.RAW_NORMAL),
            image(1),
            image(1, hard),
            torch.zeros(len(xy), device=xy.device, dtype=torch.bool) if f & Mode.HIT else None,
        )
        _run(inputs, outputs, cache, spec)
        ctx.save_for_backward(*inputs, *outputs, *cache)
        if cache[-1] is not None:
            ctx.mark_non_differentiable(cache[-1])
        ctx.set_materialize_grads(False)
        return (*outputs, cache[-1])

    @staticmethod
    def backward(ctx, *out_grads):
        saved = ctx.saved_tensors
        inputs, outputs, cache = saved[:11], saved[11:20], saved[20:]
        xy = inputs[0]
        candidates = getattr(xy, "auxiliary_absgrad_candidates", None)
        collect_moments = getattr(xy, "compute_position_moments", False)
        if candidates is not None:
            if candidates.dtype != torch.bool or candidates.device != xy.device or candidates.shape != (len(xy),):
                raise ValueError("Expected a boolean auxiliary candidate mask on the projection device")
            candidates = candidates.contiguous()
        abs_only = (candidates is not None and not collect_moments and ctx.spec[-1] == "warp"
                    and inputs[3].shape[1] == 3)
        grads = [None] * 8 if abs_only else [torch.zeros_like(t) for t in inputs[:8]]
        if not ctx.needs_input_grad[7]:
            grads[7] = None
        spec = ctx.spec
        if abs_only and out_grads[0] is not None and all(g is None for g in out_grads[1:]):
            # SSIM supplies only dL/dRGB, even if this forward also returned maps.
            width, height, _, impl = spec
            spec = (width, height, Mode(0), impl)
            upstream = [out_grads[0].contiguous()]
        else:
            upstream = []
            for g, img in zip(out_grads, outputs):
                if g is not None:
                    g = g.contiguous()
                elif img is not None:
                    g = torch.zeros_like(img)
                upstream.append(g)
        absolute = torch.zeros_like(xy) if getattr(xy, "lite_absgrad", False) or candidates is not None else None
        moments = xy.new_zeros((len(xy), 6)) if collect_moments else None
        _run(inputs, outputs, cache, spec, upstream, grads, absolute, moments, candidates if abs_only else None)
        if absolute is not None:
            previous = getattr(xy, "absgrad", None)
            xy.absgrad = absolute if previous is None else previous + absolute
        if moments is not None:
            state = getattr(xy, "position_moment_state", None)
            previous = state.get("pixel") if state is not None else getattr(xy, "position_moments", None)
            xy.position_moments = moments if previous is None else previous + moments
            if state is not None:
                state["pixel"] = xy.position_moments
        return (*grads, None, None, None, None)


def rasterize(
    xy, conics, opacity, colors, z, planes, normals, background, K, offsets, ids, size, flags,
    *, backward_impl="warp",
) -> RasterResult:
    """Single pinhole view, 16-pixel tiles, float32.

    A positive color channel count requests render. Mode selects each other output
    independently, including ALPHA, HARD_ALPHA and the per-Gaussian HIT mask.
    Image tensors are HWC. Unrequested fields in the returned tuple are None.
    backward_impl selects one thread per pixel ("pixel") or warp-reduced
    gradients ("warp"). Both use the same FP32 forward and saved tensors.
    """
    w, h = size
    if backward_impl not in ("pixel", "warp"):
        raise ValueError("backward_impl must be 'pixel' or 'warp'")
    n = len(xy)
    if not xy.is_cuda or xy.dtype != torch.float32 or xy.shape != (n, 2) or not xy.is_contiguous():
        raise ValueError("xy must be a contiguous CUDA float32 [N,2] tensor")
    if w <= 0 or h <= 0 or int(flags) & ~_ALL_FLAGS:
        raise ValueError("Invalid image size or output flags")
    if (
        offsets.shape not in [((h + 15) // 16, (w + 15) // 16), (1, (h + 15) // 16, (w + 15) // 16)]
        or ids.ndim != 1
    ):
        raise ValueError("Expected tile offsets for a single view and flat intersection IDs")
    for t in (offsets, ids):
        if t.device != xy.device or t.dtype != torch.int32:
            raise ValueError("Intersections must be int32 tensors on the same GPU")
    if K.requires_grad:
        raise ValueError("Intrinsic gradients are unsupported")

    def feature(t, shape, required=True):
        if t is None and not required:
            return xy.new_empty(0)
        if t is None or t.shape != shape or t.device != xy.device or t.dtype != xy.dtype:
            raise ValueError(f"Expected float32 tensor of shape {shape} on {xy.device}")
        return t.contiguous()

    if colors is None:
        colors = xy.new_empty((n, 0))
    channels = colors.shape[-1]
    if channels > 513:
        raise ValueError("At most 513 feature channels are supported")
    if background is None:
        background = xy.new_zeros(channels)
    active_depth = flags & (
        Mode.INVERSE | Mode.HARD | Mode.HARD_DEPTH | Mode.EXPECTED | Mode.MEDIAN
    )
    result = _Raster.apply(
        xy,
        feature(conics, (n, 3)),
        feature(opacity, (n,)),
        feature(colors, (n, channels)),
        feature(z, (n,), active_depth and not flags & Mode.RADE),
        feature(planes, (n, 3), active_depth and flags & Mode.RADE),
        feature(normals, (n, 3), flags & Mode.NORMAL),
        feature(background, (channels,)),
        feature(K, (3, 3)),
        offsets.contiguous(),
        ids.contiguous(),
        (w, h, flags, backward_impl),
    )
    return RasterResult(*result)
