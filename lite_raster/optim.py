"""FP32 LiteGS Adam for compact or dense per-Gaussian gradients."""
import torch

from .backend import check_launch, gpu_cc, library, pointer


@torch.no_grad()
def update_rows(param, grad, first, second, rows, lr, betas, eps, *, dense_grad):
    """Update unique global rows in place. A missing row list selects every row."""
    n = len(param)
    m = n if rows is None else len(rows)
    if grad.shape != (n if dense_grad else m, *param.shape[1:]):
        raise ValueError("Adam gradient and selected rows have incompatible shapes")
    if first.shape != param.shape or second.shape != param.shape:
        raise ValueError("Adam moment shape must match its parameter")
    if not (param.is_cuda and all(t.device == param.device and t.dtype == torch.float32
                                  for t in (param, grad, first, second))):
        raise ValueError("Expected FP32 parameter, gradient and moments on one CUDA device")
    if not all(t.is_contiguous() for t in (param, first, second)):
        raise ValueError("Expected contiguous row-major Gaussian parameters and moments")
    if rows is not None and (rows.ndim != 1 or rows.device != param.device or rows.dtype != torch.long):
        raise ValueError("Expected a vector of int64 row IDs on the parameter device")
    if not m or not param.numel():
        return
    width = param.numel() // n
    grad = grad.contiguous()
    if rows is not None:
        rows = rows.contiguous()
    with torch.cuda.device(param.device):
        launch = library("cluster_adam", gpu_cc(param.device)).cluster_adam
        check_launch(launch(*map(pointer, (param, grad, first, second, rows)), m * width, n, width,
                            dense_grad, lr, *betas, eps, torch.cuda.current_stream().cuda_stream))
    torch.autograd.graph.increment_version((param, first, second))
