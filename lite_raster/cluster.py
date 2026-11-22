"""Cull Morton clusters and compact differentiable Gaussian rows for dense kernels."""
import torch

from .backend import check_launch, gpu_cc, library, pointer


@torch.no_grad()
def visible_rows(pre, means, scales, order, row_clusters, size, eps):
    """Return unique global row IDs in the original order, preserving equal-depth ties."""
    view, intrinsics, (width, height), _ = pre
    n = means.shape[0]
    if n == 0:
        return order
    if not (means.is_cuda and scales.shape == means.shape == (n, 3)
            and means.dtype == scales.dtype == view.dtype == intrinsics.dtype == torch.float32
            and order.shape == row_clusters.shape == (n,) and order.dtype == torch.int64
            and row_clusters.dtype == torch.int32 and view.shape == (1, 4, 4)
            and intrinsics.shape == (1, 3, 3) and size > 0
            and all(t.device == means.device for t in (scales, order, row_clusters, view, intrinsics))):
        raise ValueError("Invalid dense cluster inputs")
    buffers = [t.contiguous() for t in (means, scales, order, view, intrinsics)]
    visible = means.new_empty((1 + (n - 1) // size,), dtype=torch.bool)
    with torch.cuda.device(means.device):
        cull = library("cluster_cull", gpu_cc(means.device)).cluster_cull
        check_launch(cull(*map(pointer, buffers), n, size, width, height, eps, pointer(visible),
                          torch.cuda.current_stream().cuda_stream))
    # Materializing the exact size incurs one device-to-host size synchronization.
    return visible.index_select(0, row_clusters).nonzero(as_tuple=True)[0]


class _Compact(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, ids, sparse_grad):
        ctx.shape = data.shape
        ctx.sparse_grad = sparse_grad
        ctx.save_for_backward(ids)
        return data.index_select(0, ids)

    @staticmethod
    def backward(ctx, grad):
        ids, = ctx.saved_tensors
        # IDs are unique. Scatter straight into the model gradient without atomic adds.
        if ctx.sparse_grad:
            sparse = torch.sparse_coo_tensor(ids.unsqueeze(0), grad.contiguous(), ctx.shape, is_coalesced=True)
            return sparse, None, None
        dense = grad.new_zeros(ctx.shape)
        return dense.index_copy_(0, ids, grad), None, None


def compact_rows(data, ids, sparse_grad=False):
    """Gather IDs produced by visible_rows (unique and in ascending order)."""
    if ids is None or len(ids) == len(data):
        return data
    return _Compact.apply(data, ids, sparse_grad)


class ClusterBounds:
    """Exact bounds for physical, consecutive 128-row clusters on one CUDA stream."""
    size = 128

    def __init__(self, means, scales):
        import weakref
        self.model = (weakref.ref(means), weakref.ref(scales))
        self.n = len(means)
        self.device = means.device
        self.stream = torch.cuda.current_stream(means.device).cuda_stream
        self.blocks = (self.n + self.size - 1) // self.size
        self.endpoints = means.new_empty((2 * self.blocks, 3))
        self.max_scales = torch.empty_like(self.endpoints)
        self.order = torch.arange(2 * self.blocks, device=means.device)
        self.visible = means.new_empty(self.blocks, dtype=torch.bool)
        self.selected = means.new_empty(self.blocks, dtype=torch.int32)
        self.count = means.new_empty((), dtype=torch.int32)
        self.versions = None
        self.last_rows = None

    def matches(self, means, scales):
        return (self.model[0]() is means and self.model[1]() is scales and self.n == len(means)
                and means.device == self.device and torch.cuda.current_stream(means.device).cuda_stream == self.stream)

    def _refresh_range(self, params, versions, updates):
        # Only trust a single, observed optimizer update from the cached version
        # to the current version, and only for the exact rows last selected here.
        if self.versions is None:
            return None, self.blocks
        if self.versions == versions:
            return None, 0
        for param, old, new, update in zip(params, self.versions, versions, updates):
            if old == new:
                continue
            if (update is None or update[0]() is not param or update[1:3] != (old, new)
                    or update[3]() is not self.last_rows):
                return None, self.blocks
        return self.selected, (len(self.last_rows) + self.size - 1) // self.size

    @torch.no_grad()
    def visible_rows(self, pre, means, scales, eps, updates=(None, None)):
        """scales are raw logs. Unrecognized changes trigger a full bounds rebuild."""
        view, intrinsics, (width, height), _ = pre
        if (not self.matches(means, scales) or not means.is_cuda or means.shape != scales.shape
                or means.shape != (self.n, 3) or view.shape != (1, 4, 4) or intrinsics.shape != (1, 3, 3)
                or any(t.device != means.device or t.dtype != torch.float32 for t in (means, scales, view, intrinsics))):
            raise ValueError("Invalid contiguous-cluster bounds inputs")
        if self.n == 0:
            return self.order
        versions = (means._version, scales._version)
        dirty, refresh_count = self._refresh_range((means, scales), versions, updates)
        with torch.cuda.device(means.device):
            lib = library("cluster_bounds", gpu_cc(means.device))
            if refresh_count:
                positions, log_scales = means.contiguous(), scales.contiguous()
                check_launch(lib.refresh_bounds(pointer(positions), pointer(log_scales), self.n,
                                                pointer(dirty), refresh_count, pointer(self.endpoints),
                                                pointer(self.max_scales), self.stream))
            matrices = (view.contiguous(), intrinsics.contiguous())
            check_launch(lib.cluster_cull(*map(pointer, (self.endpoints, self.max_scales, self.order, *matrices)),
                                          2 * self.blocks, 2, width, height, eps, pointer(self.visible), self.stream))
            check_launch(lib.select_blocks(pointer(self.visible), self.blocks, self.n, pointer(self.selected),
                                          pointer(self.count), self.stream))
            rows = means.new_empty(int(self.count.item()), dtype=torch.long)
            check_launch(lib.expand_rows(pointer(self.selected), len(rows), pointer(rows), self.stream))
        self.versions, self.last_rows = versions, rows
        return rows
