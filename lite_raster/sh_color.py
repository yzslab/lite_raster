"""Single-thread-per-Gaussian FP32 SH colour with coefficient-only gradients."""
import torch

from .backend import P, check_launch, gpu_cc, library, pointer


def _launch(dirs, dc, rest, visible, rgb, gate, degree, capacity, packed, backward):
    if dirs.shape[0] == 0:
        return
    buffers = (P * 6)(*map(pointer, (dirs, dc, rest, visible, rgb, gate)))
    with torch.cuda.device(dirs.device):
        kernel = library("sh_color", gpu_cc(dirs.device)).sh_color
        stream = torch.cuda.current_stream().cuda_stream
        check_launch(kernel(buffers, len(dirs), degree, capacity, packed, stream, backward))


class _SHColor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dirs, dc, rest, visible, degree, save_gate):
        packed = rest is None
        capacity = dc.shape[1] if packed else 1 + rest.shape[1]
        rgb = torch.empty_like(dirs)
        gate = torch.empty(len(dirs), dtype=torch.uint8, device=dirs.device) if save_gate else None
        _launch(dirs, dc, rest, visible, rgb, gate, degree, capacity, packed, False)
        if save_gate:
            ctx.save_for_backward(dirs, gate)
            ctx.degree = degree
            ctx.capacity = capacity
            ctx.dc_shape = dc.shape
            ctx.rest_shape = None if packed else rest.shape
        return rgb

    @staticmethod
    def backward(ctx, grad_rgb):
        dirs, gate = ctx.saved_tensors
        grad_dc = dirs.new_empty(ctx.dc_shape) if ctx.needs_input_grad[1] else None
        grad_rest = dirs.new_empty(ctx.rest_shape) if ctx.needs_input_grad[2] else None
        _launch(dirs, grad_dc, grad_rest, None, grad_rgb.contiguous(), gate,
                ctx.degree, ctx.capacity, ctx.rest_shape is None, True)
        return None, grad_dc, grad_rest, None, None, None


def sh_color(degree, dirs, dc, rest=None, visible=None):
    """Accept [N,K,3] coefficients, or separate [N,1,3] and [N,K-1,3] tensors."""
    if dirs.ndim != 2 or dirs.shape[1] != 3 or not dirs.is_cuda:
        raise ValueError("dirs must be a CUDA tensor of shape [N,3]")
    n = len(dirs)
    coeffs = (dc,) if rest is None else (dc, rest)
    for t in coeffs:
        if t.ndim != 3 or t.shape[0] != n or t.shape[2] != 3:
            raise ValueError("SH coefficients must have shape [N,K,3]")
    if any(t.device != dirs.device or t.dtype != torch.float32 for t in (dirs, *coeffs)):
        raise ValueError("directions and coefficients must be FP32 on the same CUDA device")
    if rest is not None and dc.shape[1] != 1:
        raise ValueError("separate DC coefficients must have shape [N,1,3]")
    capacity = dc.shape[1] if rest is None else 1 + rest.shape[1]
    if degree not in range(5) or capacity < (degree + 1) ** 2:
        raise ValueError("SH degree must be 0–4 and fit the coefficient capacity")
    if visible is not None:
        if visible.shape != (n,) or visible.device != dirs.device or visible.dtype != torch.bool:
            raise ValueError("visibility must be a boolean tensor of shape [N] on the same device")
        visible = visible.contiguous()
    save_gate = torch.is_grad_enabled() and any(t.requires_grad for t in coeffs)
    return _SHColor.apply(dirs.detach().contiguous(), dc.contiguous(),
                          rest.contiguous() if rest is not None else None, visible, degree, save_gate)
