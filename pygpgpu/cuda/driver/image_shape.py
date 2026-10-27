from __future__ import annotations

from typing import Tuple


def split_channels(shape:Tuple[int, ...])->Tuple[Tuple[int, ...], int]:
    """Split a trailing channel axis from an image shape.

    A last axis of 1, 2 or 4 is a channel count, which is what CUDA arrays accept.
    Anything else is spatial, so (8, 16, 3) reads as a 2D image with 3 channels while
    (8, 16, 7) stays a 3D image. Both ``cudaTextureObject_t`` and
    ``cudaSurfaceObject_t`` use this so the two accept exactly the same shapes.
    """
    if len(shape) > 1 and shape[-1] in (1, 2, 4):
        return tuple(shape[:-1]), int(shape[-1])
    return tuple(shape), 1


def extents_from_data(shape:Tuple[int, ...])->Tuple[int, int, int]:
    """Split the shape of a host image into (width, height, depth).

    A numpy image is row-major, so its axes are (depth, height, width) for a 3D image
    and (height, width) for a 2D one. Missing axes are 1, which is how CUDA stores a
    1D or 2D image: as a 2D array with one row, or a 3D array with one slice.
    """
    spatial, _ = split_channels(shape)
    return extents_from_spatial(spatial, data_order=True)


def extents_from_shape(shape:Tuple[int, ...])->Tuple[int, int, int]:
    """Split a user-supplied shape into (width, height, depth).

    A shape given to the constructor reads as (width[, height[, depth]]) -- the order
    a kernel's coordinates are written in -- which is the opposite of the data layout.
    """
    spatial, _ = split_channels(shape)
    return extents_from_spatial(spatial, data_order=False)


def extents_from_spatial(spatial:Tuple[int, ...], data_order:bool)->Tuple[int, int, int]:
    """Map 1, 2 or 3 spatial extents onto (width, height, depth)."""
    if len(spatial) == 1:
        return int(spatial[0]), 1, 1
    if len(spatial) == 2:
        # Both orders agree for two axes: (height, width) in data order and
        # (width, height) in shape order.
        width, height = (spatial[1], spatial[0]) if data_order else (spatial[0], spatial[1])
        return int(width), int(height), 1
    if len(spatial) == 3:
        if data_order:
            return int(spatial[2]), int(spatial[1]), int(spatial[0])
        return int(spatial[0]), int(spatial[1]), int(spatial[2])
    raise ValueError(
        "an image must have 1, 2 or 3 spatial dimensions, got %d" % len(spatial))
