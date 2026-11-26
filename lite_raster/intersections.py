"""Single-view tile lists in gsplat order. Tile IDs and FP32 center Z form a 64-bit key."""
import ctypes as ct

import torch

from .backend import TilePairs, check_launch, gpu_cc, library, pointer

_INT_MAX = 2**31 - 1


def _grid(image_size, tile_size, n):
    w, h = image_size
    if min(w, h, tile_size) <= 0 or max(w, h, tile_size, n) > _INT_MAX:
        raise ValueError("Expected positive int32 image and tile dimensions and an int32 Gaussian count")
    width, height = (w-1)//tile_size+1, (h-1)//tile_size+1
    if width * height > _INT_MAX:
        raise OverflowError("Tile IDs exceed the int32 renderer layout")
    return tile_size, width, height


def _opacity(tensor, reference, n):
    if (tensor.device != reference.device or tensor.dtype != torch.float32 or
            tensor.shape not in ((n,), (1, n), (n, 1))):
        raise ValueError("Expected one FP32 opacity per projected Gaussian on the projection device")
    return tensor.detach().contiguous()


def _sorted_list(counts, keys, ids, grid):
    _, width, height = grid
    lib = library("intersections", gpu_cc(counts.device))
    stream = torch.cuda.current_stream(counts.device).cuda_stream
    n = keys.numel()
    if n:
        alt_keys, alt_ids = torch.empty_like(keys), torch.empty_like(ids)
        size, selected = ct.c_size_t(), ct.c_int()
        args = (ct.byref(size), *map(pointer, (keys, alt_keys, ids, alt_ids)), n,
                32 + (width * height - 1).bit_length(), ct.byref(selected), stream)
        check_launch(lib.sort_pairs(None, *args))
        scratch = counts.new_empty(size.value, dtype=torch.uint8)
        check_launch(lib.sort_pairs(pointer(scratch), *args))
        if selected.value:
            keys, ids = alt_keys, alt_ids
    offsets = counts.new_empty((1, height, width))
    check_launch(lib.encode_offsets(pointer(keys), n, pointer(offsets), width*height, stream))
    return counts, keys, ids, offsets


def _buffers(counts, capacity):
    return counts.new_empty(capacity, dtype=torch.int64), counts.new_empty(capacity)


def _encode(pre, projection, opacity, tile_size, precise):
    radii, xy, z, conics = projection[:4]
    n = radii.numel()
    grid = _grid(pre[2], tile_size, n)
    if radii.shape != (1, n) or not radii.is_cuda or radii.dtype != torch.int32:
        raise ValueError("Expected dense int32 CUDA radii [1, N]")
    floats = [(xy, (1, n, 2)), (z, (1, n))]
    if precise:
        floats.append((conics, (1, n, 3)))
        opacity = _opacity(opacity, radii, n)
    if any(t.shape != shape or t.dtype != torch.float32 or t.device != radii.device for t, shape in floats):
        raise ValueError("Expected dense FP32 single-view projection tensors on the radii device")
    inputs = [t.contiguous() for t in (xy, radii, z)]
    conics = conics.contiguous() if precise else None
    with torch.cuda.device(radii.device):
        counts = radii.new_empty((1, n))
        keys, ids = _buffers(counts, 0)
        if n:
            lib = library("intersections", gpu_cc(radii.device))
            stream = torch.cuda.current_stream().cuda_stream
            args = (n, *map(pointer, inputs), pointer(conics))
            pairs = TilePairs(pointer(opacity), pointer(counts), None, None, *grid)
            check_launch(lib.intersect_tiles(ct.byref(pairs), *args, None, stream))
            ends = counts.reshape(-1).cumsum(0, dtype=torch.int64)
            length = int(ends[-1].item())
            if length > _INT_MAX:
                raise OverflowError("Tile intersections exceed the renderer's int32 offsets")
            if length:
                keys, ids = _buffers(counts, length)
                pairs.keys, pairs.ids = pointer(keys), pointer(ids)
                check_launch(lib.intersect_tiles(ct.byref(pairs), *args, pointer(ends), stream))
        return _sorted_list(counts, keys, ids, grid)


def isect_encode_precise(preprocessed_camera, projection_results, opacities, tile_size=16):
    """Encode exact ellipse–tile intersections using effective (AA-compensated) opacity."""
    return _encode(preprocessed_camera, projection_results, opacities, tile_size, True)


def isect_encode(preprocessed_camera, projection_results, opacities=None, tile_size=16):
    """Encode radius bounding boxes, as in the default gsplat v1 renderer."""
    return _encode(preprocessed_camera, projection_results, None, tile_size, False)
