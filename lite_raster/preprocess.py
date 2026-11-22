"""FP32 operations on unique, sorted Gaussian rows produced by cluster culling."""
import torch

from .backend import P, check_launch, gpu_cc, library, pointer
from .sh_color import _launch as launch_sh


def _launch(name, reference, tensors, *args):
    with torch.cuda.device(reference.device):
        kernel = getattr(library("cluster_preprocess", gpu_cc(reference.device)), name)
        buffers = (P * len(tensors))(*map(pointer, tensors))
        check_launch(kernel(buffers, *args, torch.cuda.current_stream().cuda_stream))


def _row_gradient(ids, values, shape):
    if values is not None:
        return torch.sparse_coo_tensor(ids[None], values.contiguous(), shape, is_coalesced=True)


def _check_rows(ids, reference):
    if (not reference.is_cuda or ids.device != reference.device or ids.dtype != torch.long
            or ids.ndim != 1 or not ids.is_contiguous()):
        raise ValueError("Expected contiguous int64 row IDs on the input CUDA device")


class _Geometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, means, log_scales, quats, logits):
        ctx.set_materialize_grads(False)
        inputs = means, log_scales, quats, logits
        outputs = [t.new_empty((len(ids), *t.shape[1:])) for t in inputs]
        lengths = means.new_empty(len(ids))
        buffers = (ids, *(t.contiguous() for t in inputs), *outputs, lengths)
        _launch("geometry", means, buffers, len(ids), False)
        ctx.save_for_backward(ids, *outputs[1:], lengths)
        ctx.shapes = tuple(t.shape for t in inputs)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, d_means, d_scales, d_quats, d_opacities):
        ids, scales, quats, opacities, lengths = ctx.saved_tensors
        upstream = [g.contiguous() if g is not None else None for g in (d_scales, d_quats, d_opacities)]
        grads = [torch.empty_like(g) if g is not None else None for g in upstream]
        _launch("geometry", scales, (scales, quats, opacities, lengths, *upstream, *grads), len(ids), True)
        return None, *(_row_gradient(ids, g, shape) for g, shape in zip((d_means, *grads), ctx.shapes))


def compact_geometry(ids, means, log_scales, quats, logits):
    """Gather model rows and apply exp, quaternion normalization and sigmoid."""
    _check_rows(ids, means)
    n = len(means)
    for tensor, width in zip((means, log_scales, quats, logits), (3, 3, 4, 1)):
        if tensor.shape != (n, width) or tensor.device != means.device or tensor.dtype != torch.float32:
            raise ValueError("Expected FP32 means, log scales, quaternions and logits on one CUDA device")
    return _Geometry.apply(ids, means, log_scales, quats, logits)


class _IndexedSH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, dirs, visible, dc, rest, degree):
        colors = torch.empty_like(dirs)
        gate = ids.new_empty(len(ids), dtype=torch.uint8)
        buffers = (dirs, dc.contiguous(), rest.contiguous(), visible, colors, gate, ids)
        _launch("colors", dirs, buffers, len(ids), degree, 1 + rest.shape[1])
        ctx.save_for_backward(ids, dirs, gate)
        ctx.dc_shape, ctx.rest_shape, ctx.degree = dc.shape, rest.shape, degree
        return colors

    @staticmethod
    def backward(ctx, d_colors):
        ids, dirs, gate = ctx.saved_tensors
        dc = dirs.new_empty((len(ids), *ctx.dc_shape[1:]))
        rest = dirs.new_empty((len(ids), *ctx.rest_shape[1:]))
        launch_sh(dirs, dc, rest, None, d_colors.contiguous(), gate, ctx.degree, rest.shape[1] + 1, False, True)
        return None, None, None, _row_gradient(ids, dc, ctx.dc_shape), _row_gradient(ids, rest, ctx.rest_shape), None


def indexed_sh_color(ids, dirs, visible, dc, rest, degree):
    """Evaluate active SH bands directly from the original coefficient tables."""
    _check_rows(ids, dirs)
    if (dirs.shape != (len(ids), 3) or visible.shape != (len(ids),) or visible.dtype != torch.bool
            or visible.device != dirs.device or dc.shape != (len(rest), 1, 3) or rest.ndim != 3
            or rest.shape[2] != 3 or degree not in range(5) or (degree + 1) ** 2 > 1 + rest.shape[1]
            or any(t.device != dirs.device or t.dtype != torch.float32 for t in (dc, rest, dirs))):
        raise ValueError("Invalid indexed SH inputs")
    return _IndexedSH.apply(ids, dirs.detach().contiguous(), visible.contiguous(), dc, rest, degree)


@torch.no_grad()
def try_update_density_stats(ids, tiles, radii, grad, scale, max_radii, accumulated, counts):
    """Update supported FP32 statistics in place. False lets other renderers use their native layout."""
    n = len(grad)
    total = len(max_radii)
    if (not grad.is_cuda or (ids is not None and len(ids) != n) or (ids is None and total != n)
            or grad.shape != (n, 2) or tiles.shape != (n,) or radii.shape != (n,)
            or tiles.dtype != torch.int32 or radii.dtype != torch.int32
            or max_radii.shape != (total,) or accumulated.numel() != total or counts.numel() != total
            or not all(t.is_contiguous() for t in (max_radii, accumulated, counts))
            or (scale is not None and (not isinstance(scale, torch.Tensor) or scale.shape not in ((2,), (1, 2))
                                       or scale.dtype != torch.float32 or scale.device != grad.device))
            or any(t.device != grad.device for t in (tiles, radii, max_radii, accumulated, counts))
            or any(t.dtype != torch.float32 for t in (grad, max_radii, accumulated, counts))):
        return False
    if ids is not None:
        _check_rows(ids, grad)
    buffers = (ids, tiles.contiguous(), radii.contiguous(), grad.contiguous(),
               scale.contiguous() if scale is not None else None, max_radii, accumulated, counts)
    _launch("stats", grad, buffers, n)
    torch.autograd.graph.increment_version((max_radii, accumulated, counts))
    return True
