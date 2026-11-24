"""Fused mean L1 and 11×11 zero-padded SSIM for a single float32 CUDA image."""

import torch

from .backend import P, check_launch, gpu_cc, library, pointer


def _launch(image, target, derivatives, args, weight, backward):
    c, h, w = image.shape
    pointers = (P * 10)(*map(pointer, (image, target, *derivatives, *args)))
    with torch.cuda.device(image.device):
        lib = library("joint_loss", gpu_cc(image.device))
        check_launch(lib.joint_loss(pointers, h, w, c, weight, backward, torch.cuda.current_stream().cuda_stream))


class _JointLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, target, weight, return_pixel_errors):
        c, h, w = image.shape
        derivatives = [torch.empty_like(image) for _ in range(3)]
        tiles = image.new_empty((c, (h + 15) // 16, (w + 15) // 16, 2))
        outputs = tuple(image.new_empty(()) for _ in range(3))
        errors = torch.empty_like(image) if return_pixel_errors else None
        _launch(image, target, derivatives, (tiles, *outputs, errors), weight, False)
        ctx.save_for_backward(image, target, *derivatives)
        ctx.set_materialize_grads(False)
        ctx.weight = weight
        return (*outputs, errors)

    @staticmethod
    def backward(ctx, grad_loss, grad_l1, grad_ssim, grad_errors):
        image, target, *derivatives = ctx.saved_tensors
        grad_image = torch.empty_like(image)
        if grad_errors is not None:
            grad_errors = grad_errors.contiguous()
        _launch(image, target, derivatives, (grad_loss, grad_l1, grad_ssim, grad_image, grad_errors), ctx.weight, True)
        return grad_image, None, None, None


def fused_l1_ssim(image, target, weight=0.2, *, return_pixel_errors=False):
    """Return weighted loss, mean L1 and mean SSIM, optionally followed by the CHW L1 map.

    All outputs support gradients with respect to the image. The target is fixed.
    """
    if (image.ndim != 3 or not image.numel() or image.shape != target.shape or not image.is_cuda
            or image.device != target.device or image.dtype != torch.float32 or target.dtype != image.dtype
            or target.requires_grad):
        raise ValueError("Expected matching nonempty float32 CUDA [C,H,W] images and a fixed target")
    outputs = _JointLoss.apply(image.contiguous(), target.contiguous(), weight, return_pixel_errors)
    return outputs if return_pixel_errors else outputs[:3]
