"""Single-view projection and indexed SH with a deferred, fused LiteGSAdam step."""
import ctypes as ct
from weakref import ref

import torch

from .backend import AdamGroup, P, ProjectionStep, check_launch, gpu_cc, library, pointer
from .preprocess import _check_rows, _gather_geometry
from .projection import _outputs, _position_stats, _run


class BackwardAdam:
    """Bind this render to the optimizer's current groups (including restored or replaced parameters)."""
    geometry = ("means", "scales", "rotations", "opacities")
    colors = ("shs_dc", "shs_rest")

    def __init__(self, optimizers, bounds_updates):
        self.groups = {group["name"]: (opt, group) for opt in optimizers for group in opt.param_groups}
        self.bounds_updates = bounds_updates
        self._pending_geometry = self._pending_colors = None
        self._applied = False

    def _queue_geometry(self, ctx, upstream, d_opacity, d_scale):
        if self._applied or self._pending_geometry is not None:
            raise RuntimeError("Expected one projection backward per fused Adam render")
        # Autograd releases saved_tensors when this callback returns.
        self._pending_geometry = (ctx.saved_tensors, ctx.spec, ctx.position_moment_state, upstream, d_opacity, d_scale)

    def _queue_colors(self, ids, dirs, gate, params, grad, degree):
        if self._applied or self._pending_colors is not None:
            raise RuntimeError("Expected one indexed SH backward per fused Adam render")
        self._pending_colors = (ids, dirs, gate, params, grad, degree)

    @torch.no_grad()
    def apply_pending_adam(self):
        """Consume this render's upstream gradients once, after backward and before densification."""
        if self._applied:
            return
        self._applied = True
        geometry, colors = self._pending_geometry, self._pending_colors
        self._pending_geometry = self._pending_colors = None
        # Reject a direct parameter loss before either kernel can change the model.
        if any(p.grad is not None for _, group in self.groups.values() for p in group["params"]):
            raise RuntimeError("fused_backward_adam received an additional raw parameter gradient. "
                               "Use ordinary LiteGSAdam for parameter regularizers.")
        if geometry is not None:
            self._step_geometry(*geometry)
        if colors is not None:
            self._step_colors(*colors)

    def _bind(self, names, params, active):
        groups = (AdamGroup * len(names))()
        written = []
        for i, (name, param, needed) in enumerate(zip(names, params, active)):
            if not needed:
                continue
            opt, group = self.groups[name]
            if len(group["params"]) != 1 or group["params"][0] is not param or not param.requires_grad:
                raise RuntimeError(f"Expected the current trainable {name} parameter")
            state = opt.row_state(param)
            first, second = state["exp_avg"], state["exp_avg_sq"]
            if not all(t.shape == param.shape and t.device == param.device and t.dtype == torch.float32
                       and t.is_contiguous() for t in (param, first, second)):
                raise ValueError(f"Expected contiguous FP32 {name} and matching Adam moments")
            groups[i] = AdamGroup(*map(pointer, (param, first, second)), group["lr"], *group["betas"], group["eps"])
            if param.numel():
                written.extend((param, first, second))
        return groups, written

    def _step_colors(self, ids, dirs, gate, params, grad, degree):
        groups, written = self._bind(self.colors, params, (True, True))
        if len(ids):
            grad = grad.contiguous()
            with torch.cuda.device(ids.device):
                kernel = library("cluster_preprocess", gpu_cc(ids.device)).colors_adam
                buffers = (P * 4)(*map(pointer, (ids, dirs, gate, grad)))
                check_launch(kernel(buffers, groups, len(ids), degree, 1 + params[1].shape[1],
                                    torch.cuda.current_stream().cuda_stream))
            torch.autograd.graph.increment_version(written)

    def _step_geometry(self, saved_tensors, spec, moment_state, upstream, d_opacity, d_scale):
        ids, alpha, length, *saved = saved_tensors
        inputs, (radii, conic, comp), params = saved[:6], saved[6:9], saved[9:]
        projected_grad = any(g is not None for g in upstream) or (comp is not None and d_opacity is not None)
        active = (projected_grad, projected_grad or d_scale is not None, projected_grad, d_opacity is not None)
        groups, written = self._bind(self.geometry, params, active)
        d_opacity = d_opacity.contiguous() if d_opacity is not None else None
        d_scale = d_scale.contiguous() if d_scale is not None else None
        step = ProjectionStep(groups, *map(pointer, (ids, alpha, length, None, d_opacity, d_scale)), projected_grad)
        stats = _position_stats(inputs[0], moment_state) if projected_grad else ()
        outputs = (radii, None, None, conic, None, None, None, comp, None, None)
        _run(inputs, outputs, [g.contiguous() if g is not None else None for g in upstream],
             (), stats, spec, True, step)
        if len(ids):
            torch.autograd.graph.increment_version(written)
            for name, param, updated in zip(self.geometry[:2], params, active):
                if updated:
                    self.bounds_updates[name] = (ref(param), param._version - 1, param._version, ref(ids))


class _ProjectAdam(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, means, log_scales, quats, logits, view, K, center, spec, compensate, adam):
        ctx.set_materialize_grads(False)
        raw = means, log_scales, quats, logits
        (pos, scales, rot, alpha), length = _gather_geometry(ids, raw)
        inputs = pos, scales, rot, view, K, center
        outputs = _outputs(pos, spec[-1], compensate)
        opacity = pos.new_empty(len(ids))
        step = ProjectionStep(ids=pointer(ids), alpha=pointer(alpha), opacity=pointer(opacity))
        _run(inputs, outputs, (), (), (), spec, False, step)
        ctx.spec, ctx.adam = spec, adam
        radii, xy, _, conic, _, _, _, comp = outputs[:8]
        ctx.save_for_backward(ids, alpha, length, *inputs, radii, conic, comp, *raw)
        ctx.position_moment_state = {}
        xy.position_moment_state = ctx.position_moment_state
        return *outputs, opacity, scales

    @staticmethod
    def backward(ctx, *grads):
        ctx.adam._queue_geometry(ctx, grads[:10], *grads[10:])
        return (None,) * 11


def project_adam(pre, ids, means, scales, rotations, opacities, adam, *, eps, radius_clip,
                 compensate=False, planes=False, normals=False, legacy=False):
    """Gather raw rows, project and return the opacity consumed by tile encoding and blending."""
    _check_rows(ids, means)
    view, K, (w, h), center = pre
    n = len(means)
    inputs = means, scales, rotations, opacities, view, K, center
    shapes = ((n, 3), (n, 3), (n, 4), (n, 1), (1, 4, 4), (1, 3, 3), (1, 3))
    if any(t.shape != shape or t.dtype != torch.float32 or t.device != means.device for t, shape in zip(inputs, shapes)):
        raise ValueError("Expected FP32 vanilla Gaussians and one pinhole camera")
    if legacy and (planes or normals):
        raise ValueError("Select basic normals or RaDe geometry")
    mode = int(planes) | (int(normals) << 1) | (int(legacy) << 2)
    result = _ProjectAdam.apply(ids, *inputs[:4], *(t.detach().contiguous() for t in inputs[4:]),
                                (w, h, eps, radius_clip, mode), compensate, adam)
    return result[:8], *result[8:]
