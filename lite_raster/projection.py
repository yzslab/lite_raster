"""Dense FP32 projection for one pinhole camera, with optional RaDe geometry."""
import ctypes as ct

import torch

from .backend import P, ProjectionArgs, check_launch, gpu_cc, library, pointer


def _run(inputs, outputs, upstream, grads, stats, spec, backward):
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
        launch = library("fused_projection", gpu_cc(means.device)).projection
        check_launch(launch(ct.byref(args), mode, torch.cuda.current_stream().cuda_stream, backward))


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, scales, quats, view, K, center, spec, compensate):
        ctx.set_materialize_grads(False)
        ctx.spec = spec
        n, mode = len(means), spec[-1]
        inputs = means, scales, quats, view, K, center
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
        _run(inputs, outputs, (), (), (), spec, False)
        ctx.save_for_backward(*inputs, radii, conic, comp)
        ctx.position_moment_state = {}
        xy.position_moment_state = ctx.position_moment_state
        return outputs

    @staticmethod
    def backward(ctx, *upstream):
        *inputs, radii, conic, comp = ctx.saved_tensors
        means, scales, quats, view = inputs[:4]
        grads = [torch.empty_like(t) if needed else None
                 for t, needed in zip((means, scales, quats), ctx.needs_input_grad[:3])]
        grads.append(torch.zeros_like(view) if ctx.needs_input_grad[3] else None)
        outputs = (radii, None, None, conic, None, None, None, comp, None, None)
        state = ctx.position_moment_state
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
        _run(inputs, outputs, [t.contiguous() if t is not None else None for t in upstream],
             grads, stats, ctx.spec, True)
        return (*grads, None, None, None, None)


def project(pre, means, scales, quats, eps=0.3, radius_clip=0., compensate=False, planes=False, normals=False, legacy=False):
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
    result = _Projection.apply(*(t.contiguous() for t in inputs), (w, h, eps, radius_clip, mode), compensate)
    return result[:8], result[8], result[9]
