"""Fused mean L1 with sliding-window or tile SSIM for a single FP32 CUDA image."""

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


def _validate_images(image, target):
    if (image.ndim != 3 or not image.numel() or image.shape != target.shape or not image.is_cuda
            or image.device != target.device or image.dtype != torch.float32 or target.dtype != image.dtype
            or target.requires_grad):
        raise ValueError("Expected matching nonempty float32 CUDA [C,H,W] images and a fixed target")


def fused_l1_ssim(image, target, weight=0.2, *, return_pixel_errors=False):
    """Return weighted loss, mean L1 and mean 11×11 SSIM, optionally followed by the CHW L1 map.

    All outputs support gradients with respect to the image. The target is fixed.
    """
    _validate_images(image, target)
    outputs = _JointLoss.apply(image.contiguous(), target.contiguous(), weight, return_pixel_errors)
    return outputs if return_pixel_errors else outputs[:3]


def _launch_tile(tensors, offset, weight, backward):
    image = tensors[0]
    with torch.cuda.device(image.device):
        pointers = (P * len(tensors))(*map(pointer, tensors))
        lib = library("tile_loss", gpu_cc(image.device))
        check_launch(lib.tile_loss(pointers, *image.shape, *offset, weight, backward,
                                   torch.cuda.current_stream().cuda_stream))


class _TileLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, target, weight, offset, return_pixel_errors):
        channels, height, width = image.shape
        ox, oy = offset
        tile_count = channels * ((height + oy + 15) // 16) * ((width + ox + 15) // 16)
        joint_gradient = torch.empty_like(image)
        derivatives = image.new_empty((tile_count, 5))
        sums = image.new_empty((tile_count, 2))
        outputs = tuple(image.new_empty(()) for _ in range(3))
        errors = torch.empty_like(image) if return_pixel_errors else None
        _launch_tile((image, target, joint_gradient, derivatives, sums, *outputs, errors), offset, weight, False)
        ctx.save_for_backward(image, target, joint_gradient, derivatives)
        ctx.offset = offset
        ctx.set_materialize_grads(False)
        return (*outputs, errors)

    @staticmethod
    def backward(ctx, grad_loss, grad_l1, grad_ssim, grad_errors):
        image, target, joint_gradient, derivatives = ctx.saved_tensors
        if grad_l1 is None and grad_ssim is None and grad_errors is None:
            return joint_gradient * grad_loss, None, None, None, None
        grad_image = torch.empty_like(image)
        if grad_errors is not None:
            grad_errors = grad_errors.contiguous()
        _launch_tile((image, target, joint_gradient, derivatives, grad_loss, grad_l1, grad_ssim, grad_errors,
                      grad_image), ctx.offset, 0., True)
        return grad_image, None, None, None, None


def fused_tile_l1_ssim(image, target, weight=0.2, *, step=0, return_pixel_errors=False):
    """Like fused_l1_ssim, using disjoint 16×16 windows with equal weight per valid pixel.

    Pass the training step to shift the window grid through all 256 offsets.
    """
    _validate_images(image, target)
    # 73 is coprime to 256, so this visits every origin without using training RNG.
    phase = (int(step) * 73) % 256
    offset = (phase % 16, phase // 16)
    if image.numel() >= 2**31 or image.shape[0] > 65535 or (image.shape[1] + offset[1] + 15) // 16 > 65535:
        raise ValueError("Image exceeds CUDA tile loss indexing or grid limits")
    outputs = _TileLoss.apply(image.contiguous(), target.contiguous(), weight, offset, return_pixel_errors)
    return outputs if return_pixel_errors else outputs[:3]
