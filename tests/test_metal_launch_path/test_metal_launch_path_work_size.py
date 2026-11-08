"""The work_size tests, split out of `test_metal_launch_path.py`.

The subject is work_size.
"""
from __future__ import annotations

from _helpers.test_metal_launch_path_helpers import (
    _NoDeviceProgram,
)

import pytest


@pytest.mark.parametrize("bracket,expected", [
    # Automatic: one group of one thread, so an id-reading body gets 0.
    (None, ((1, 1, 1), (1, 1, 1))),
    # A bare item count and its one-tuple spelling are the same launch.
    (64, ((64, 1, 1), (1, 1, 1))),
    ((64,), ((64, 1, 1), (1, 1, 1))),
    # The ordinary CUDA pair. This used to raise TypeError: 'int' object is not
    # iterable, which made every plain `k[(4, 256)]` launch fail.
    ((4, 256), ((4, 1, 1), (256, 1, 1))),
    # Mixed spellings: a scalar side and a tuple side describe one launch.
    ((4, (256,)), ((4, 1, 1), (256, 1, 1))),
    (((4,), (256,)), ((4, 1, 1), (256, 1, 1))),
    # The fully spelled three-axis pair passes through.
    (((4, 1, 1), (256, 1, 1)), ((4, 1, 1), (256, 1, 1))),
    (((2, 3, 4), (8, 4, 2)), ((2, 3, 4), (8, 4, 2))),
])
def test_the_work_size_bracket_maps_onto_mtl_size(bracket, expected) -> None:
    """Every bracket spelling reaches Metal as two three-axis `MTLSize`s.

    The bracket is CUDA's `(blocks_per_grid, threads_per_block)` on every backend,
    and `dispatchThreadgroups:threadsPerThreadgroup:` takes that pair under its own
    names -- so the mapping is a normalisation, not a conversion. Either side may be
    written as a scalar or as a per-axis tuple, and `MTLSize` is always three
    dimensional, so a missing axis is padded with 1 (never 0, which would ask for an
    empty dispatch).
    """
    from pygpgpu.metal.oop.kernel import BoundKernel, MetalKernel

    kernel = MetalKernel(_NoDeviceProgram(), "k")
    assert BoundKernel(kernel, bracket)._dispatch_shape() == expected


def test_a_bracket_with_more_than_three_axes_is_refused() -> None:
    """Metal dispatches at most three axes, and the message says which side."""
    from pygpgpu.metal.oop.kernel import BoundKernel, MetalKernel

    kernel = MetalKernel(_NoDeviceProgram(), "k")
    with pytest.raises(ValueError) as refused:
        BoundKernel(kernel, ((1, 2, 3, 4), 256))._dispatch_shape()
    assert "blocks_per_grid" in str(refused.value)
    assert "at most three" in str(refused.value)
