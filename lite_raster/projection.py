"""Dense FP32 projection for one pinhole camera, with optional RaDe geometry."""
import ctypes as ct

import torch

from ._chained import project_encode
from .backend import P, ProjectionArgs, check_launch, gpu_cc, library, pointer


def _run(inputs, outputs, upstream, grads, stats, spec, backward, step=None):
    w, h, eps, radius_clip, mode = spec
    means = inputs[0]
    if not len(means):
        return
    args = ProjectionArgs(
        (P * 6)(*map(pointer, inputs)), (P * 10)(*map(pointer, outputs)),
        (P * 10)(*map(pointer, upstream)), (P * 4)(*map(pointer, grads)),
        (P * 4)(*map(pointer, stats)), len(means), w, h, eps, radius_clip,
    )
    with torch.cuda.device(means.device):
        lib = library("fused_projection", gpu_cc(means.device))
        stream = torch.cuda.current_stream().cuda_stream
        if step is None:
            check_launch(lib.projection(ct.byref(args), mode, stream, backward))
        else:
            check_launch(lib.projection_adam(ct.byref(args), mode, stream, backward, ct.byref(step)))


def _outputs(means, mode, compensate):
    n = len(means)
    radii = means.new_empty((1, n), dtype=torch.int32)
    xy = means.new_empty((1, n, 2))
    z = means.new_empty((1, n))
    conic = means.new_empty((1, n, 3))
    comp = torch.empty_like(z) if compensate else None
    outputs = (radii, xy, z, conic, torch.empty_like(conic),
               torch.empty_like(conic) if mode == 4 else None,
               torch.empty_like(z) if mode == 4 else None, comp,
               torch.empty_like(means) if mode & 1 else None,
               torch.empty_like(means) if mode & 2 else None)
    return outputs


def _position_stats(means, state):
    stats = []
    for source, target in (("pixel", "world"), ("pixel_ssim", "world_ssim")):
        pixel = state.get(source)
        if pixel is None:
            stats.extend((None, None))
        else:
            if pixel.dtype != means.dtype or pixel.device != means.device:
                raise ValueError("Position statistics must use the projection's FP32 device")
            if source == "pixel_ssim" and stats[0] is None:
                raise RuntimeError("SSIM position statistics require full statistics too")
            world = means.new_zeros((1, len(means), 10))
            state[target] = world
            stats.extend((pixel.reshape(1, len(means), 6).contiguous(), world))
    return stats


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, scales, quats, view, K, center, spec, compensate, isect):
        ctx.set_materialize_grads(False)
        ctx.spec = spec
        inputs = means, scales, quats, view, K, center
        outputs = _outputs(means, spec[-1], compensate)
        radii, xy, _, conic, _, _, _, comp = outputs[:8]
        if isect is None:
            _run(inputs, outputs, (), (), (), spec, False)
            intersections = ()
        else:
            intersections = project_encode(inputs, outputs, spec, *isect)
        ctx.save_for_backward(*inputs, radii, conic, comp)
        ctx.position_moment_state = {}
        xy.position_moment_state = ctx.position_moment_state
        return (*outputs, *intersections)

    @staticmethod
    def backward(ctx, *upstream):
        *inputs, radii, conic, comp = ctx.saved_tensors
        means, scales, quats, view = inputs[:4]
        grads = [torch.empty_like(t) if needed else None
                 for t, needed in zip((means, scales, quats), ctx.needs_input_grad[:3])]
        grads.append(torch.zeros_like(view) if ctx.needs_input_grad[3] else None)
        outputs = (radii, None, None, conic, None, None, None, comp, None, None)
        stats = _position_stats(means, ctx.position_moment_state)
        _run(inputs, outputs, [t.contiguous() if t is not None else None for t in upstream[:10]],
             grads, stats, ctx.spec, True)
        return tuple(grads) + (None,) * (len(ctx.needs_input_grad) - len(grads))


def project(pre, means, scales, quats, eps=0.3, radius_clip=0., compensate=False, planes=False, normals=False,
            legacy=False, *, isect_opacities=None, tile_size=16, chained_isect=None):
    """Optionally return tile lists. chained_isect='bbox' or 'precise' runs separate kernels
    in one C++ call. Precise encoding takes activated, pre-AA isect_opacities.
    Passing only isect_opacities selects chained precise encoding.
    """
    view, K, (w, h), center = pre
    inputs = (means, scales, quats, view, K, center)
    n = len(means)
    shapes = ((n, 3), (n, 3), (n, 4), (1, 4, 4), (1, 3, 3), (1, 3))
    if not means.is_cuda or any(t.shape != shape or t.device != means.device or t.dtype != torch.float32
                                for t, shape in zip(inputs, shapes)):
        raise ValueError("Expected FP32 Gaussian parameters and one CUDA pinhole camera")
    if K.requires_grad and (planes or normals):
        raise ValueError("RaDe intrinsic gradients are unsupported")
    if legacy and (planes or normals):
        raise ValueError("Select basic normals or RaDe geometry")
    mode = int(planes) | (int(normals) << 1) | (int(legacy) << 2)
    if chained_isect is None and isect_opacities is not None:
        chained_isect = "precise"
    isect = (chained_isect, tile_size, isect_opacities) if chained_isect is not None else None
    result = _Projection.apply(*(t.contiguous() for t in inputs), (w, h, eps, radius_clip, mode),
                               compensate, isect)
    if isect is not None:
        return result[:8], result[8], result[9], result[10:]
    return result[:8], result[8], result[9]
