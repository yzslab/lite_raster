"""Fused RGB reconstruction loss with sliding-window or tile SSIM for one FP32 CUDA image."""

import torch

from .backend import P, check_launch, gpu_cc, library, pointer


_RGB_MODES = {"l1": 0, "l2": 1, "l1+l2": 2}  # RGBLoss in rgb_error.cuh


def _rgb_spec(rgb_diff_loss, lambda_l2):
    if rgb_diff_loss not in _RGB_MODES:
        raise ValueError(f"Unknown rgb_diff_loss {rgb_diff_loss!r}, expected one of {tuple(_RGB_MODES)}")
    return _RGB_MODES[rgb_diff_loss], lambda_l2


def _launch(image, target, derivatives, args, weight, rgb_spec, backward):
    c, h, w = image.shape
    mode, lambda_l2 = rgb_spec
    pointers = (P * 10)(*map(pointer, (image, target, *derivatives, *args)))
    with torch.cuda.device(image.device):
        lib = library("joint_loss", gpu_cc(image.device), flags=mode)
        check_launch(lib.joint_loss(pointers, h, w, c, weight, lambda_l2, backward, torch.cuda.current_stream().cuda_stream))


class _JointLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, target, weight, return_pixel_errors, rgb_spec):
        c, h, w = image.shape
        derivatives = [torch.empty_like(image) for _ in range(3)]
        tiles = image.new_empty((c, (h + 15) // 16, (w + 15) // 16, 2))
        outputs = tuple(image.new_empty(()) for _ in range(3))
        errors = torch.empty_like(image) if return_pixel_errors else None
        _launch(image, target, derivatives, (tiles, *outputs, errors), weight, rgb_spec, False)
        ctx.save_for_backward(image, target, *derivatives)
        ctx.set_materialize_grads(False)
        ctx.weight = weight
        ctx.rgb_spec = rgb_spec
        return (*outputs, errors)

    @staticmethod
    def backward(ctx, grad_loss, grad_rgb, grad_ssim, grad_errors):
        image, target, *derivatives = ctx.saved_tensors
        grad_image = torch.empty_like(image)
        if grad_errors is not None:
            grad_errors = grad_errors.contiguous()
        _launch(image, target, derivatives, (grad_loss, grad_rgb, grad_ssim, grad_image, grad_errors),
                ctx.weight, ctx.rgb_spec, True)
        return grad_image, None, None, None, None


def _validate_images(image, target):
    if (image.ndim != 3 or not image.numel() or image.shape != target.shape or not image.is_cuda
            or image.device != target.device or image.dtype != torch.float32 or target.dtype != image.dtype
            or target.requires_grad):
        raise ValueError("Expected matching nonempty float32 CUDA [C,H,W] images and a fixed target")


def fused_l1_ssim(image, target, weight=0.2, *, return_pixel_errors=False, rgb_diff_loss="l1", lambda_l2=2.5):
    """Return weighted loss, mean RGB error and mean 11×11 SSIM, optionally followed by the CHW error map.

    For residual d, RGB error is |d| ("l1"), d² ("l2"), or |d| + lambda_l2 * d² ("l1+l2").
    All outputs support gradients with respect to the image. The target is fixed.
    """
    _validate_images(image, target)
    outputs = _JointLoss.apply(image.contiguous(), target.contiguous(), weight, return_pixel_errors,
                               _rgb_spec(rgb_diff_loss, lambda_l2))
    return outputs if return_pixel_errors else outputs[:3]


def _launch_tile(tensors, offset, weight, rgb_spec, backward):
    image = tensors[0]
    mode, lambda_l2 = rgb_spec
    with torch.cuda.device(image.device):
        pointers = (P * len(tensors))(*map(pointer, tensors))
        lib = library("tile_loss", gpu_cc(image.device), flags=mode)
        check_launch(lib.tile_loss(pointers, *image.shape, *offset, weight, lambda_l2, backward,
                                   torch.cuda.current_stream().cuda_stream))


class _TileLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, target, weight, offset, return_pixel_errors, rgb_spec):
        channels, height, width = image.shape
        ox, oy = offset
        tile_count = channels * ((height + oy + 15) // 16) * ((width + ox + 15) // 16)
        joint_gradient = torch.empty_like(image)
        derivatives = image.new_empty((tile_count, 5))
        sums = image.new_empty((tile_count, 2))
        outputs = tuple(image.new_empty(()) for _ in range(3))
        errors = torch.empty_like(image) if return_pixel_errors else None
        _launch_tile((image, target, joint_gradient, derivatives, sums, *outputs, errors), offset, weight, rgb_spec, False)
        ctx.save_for_backward(image, target, joint_gradient, derivatives)
        ctx.offset = offset
        ctx.rgb_spec = rgb_spec
        ctx.set_materialize_grads(False)
        return (*outputs, errors)

    @staticmethod
    def backward(ctx, grad_loss, grad_rgb, grad_ssim, grad_errors):
        image, target, joint_gradient, derivatives = ctx.saved_tensors
        if grad_rgb is None and grad_ssim is None and grad_errors is None:
            return joint_gradient * grad_loss, None, None, None, None, None
        grad_image = torch.empty_like(image)
        if grad_errors is not None:
            grad_errors = grad_errors.contiguous()
        _launch_tile((image, target, joint_gradient, derivatives, grad_loss, grad_rgb, grad_ssim, grad_errors,
                      grad_image), ctx.offset, 0., ctx.rgb_spec, True)
        return grad_image, None, None, None, None, None


def fused_tile_l1_ssim(image, target, weight=0.2, *, step=0, return_pixel_errors=False,
                       rgb_diff_loss="l1", lambda_l2=2.5):
    """Like fused_l1_ssim, using disjoint 16×16 windows with equal weight per valid pixel.

    Pass the training step to shift the window grid through all 256 offsets.
    """
    _validate_images(image, target)
    # 73 is coprime to 256, so this visits every origin without using training RNG.
    phase = (int(step) * 73) % 256
    offset = (phase % 16, phase // 16)
    if image.numel() >= 2**31 or image.shape[0] > 65535 or (image.shape[1] + offset[1] + 15) // 16 > 65535:
        raise ValueError("Image exceeds CUDA tile loss indexing or grid limits")
    outputs = _TileLoss.apply(image.contiguous(), target.contiguous(), weight, offset, return_pixel_errors,
                              _rgb_spec(rgb_diff_loss, lambda_l2))
    return outputs if return_pixel_errors else outputs[:3]
