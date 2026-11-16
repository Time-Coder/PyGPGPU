"""Bugs in the numpy-compatible layer's view bookkeeping: a byte offset used as an index.

`_member_bytes` answers ``(host_array, byte_offset)`` -- the slice of the family's buffer a
view occupies, and where it starts, **in bytes**. The slice was then taken with that number
as an *element* index:

    return family.data[offset:offset + view.nbytes], offset   # offset is BYTES

For a float32 family a view at byte 8 is element 2, but `data[8:28]` addresses elements 8
through 27. A write through `a[2:7]` therefore landed at `a[8:10]` and the caller's real
slice was never touched -- a silent wrong answer, and the values that did change were far
outside the view.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


@kernel
def square_in_place(a: np.ndarray):
    for i in range(a.shape[0]):
        a[i] = a[i] * a[i]


def _gnp():
    """The numpy-compatible layer, imported lazily so a missing one skips clearly."""
    return pytest.importorskip("pygpgpu.numpy")


def test_a_plain_numpy_slice_is_written_in_place(gpu_backend):
    """The control: a real numpy view works, so the bug is in the gnp view path."""
    base = np.arange(10, dtype=np.float32)
    view = base[2:7]
    square_in_place[gpu_backend][(1, 5)](view)
    np.testing.assert_array_equal(base, np.array([0, 1, 4, 9, 16, 25, 36, 7, 8, 9],
                                                dtype=np.float32))


def test_a_gnp_slice_view_writes_the_right_elements(gpu_backend):
    """`a[2:7]` must update elements 2..6 of the root, in bytes 8..27.

    The slice was taken with the byte offset as an element index, so a float32 view at byte
    8 addressed elements 8..27: the write landed at `a[8:10]` (the only two in range) and
    the view itself never changed.
    """
    gnp = _gnp()
    root = gnp.asarray(np.arange(10, dtype=np.float32))
    root.to_device()
    view = root[2:7]

    square_in_place[gpu_backend][(1, 5)](view)
    view.to_host(wait=True)

    expected = np.array([0, 1, 4, 9, 16, 25, 36, 7, 8, 9], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(root), expected)


def test_a_gnp_slice_view_reads_back_its_own_values(gpu_backend):
    """The same fix from the read side: the view reports what the kernel computed."""
    gnp = _gnp()
    root = gnp.asarray(np.arange(10, dtype=np.float32))
    root.to_device()
    view = root[2:7]

    square_in_place[gpu_backend][(1, 5)](view)
    view.to_host(wait=True)

    np.testing.assert_array_equal(np.asarray(view),
                                  np.array([4, 9, 16, 25, 36], dtype=np.float32))


def test_a_gnp_view_at_offset_zero_still_works(gpu_backend):
    """The control: a view whose byte offset is 0 looked right even when the bug was live."""
    gnp = _gnp()
    root = gnp.asarray(np.arange(10, dtype=np.float32))
    root.to_device()
    view = root[0:3]

    square_in_place[gpu_backend][(1, 3)](view)
    view.to_host(wait=True)

    np.testing.assert_array_equal(np.asarray(root),
                                  np.array([0, 1, 4, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32))


def test_a_gnp_row_of_a_2d_array_writes_the_right_row(gpu_backend):
    """A row view's offset is measured in a *multi-dimensional* family buffer.

    `root[1]` of a (3, 4) float32 array starts at byte 16, i.e. flat element 4. Slicing the
    2-D `family.data` by that element index (`data[4:8]`) walks axis 0 and answers an empty
    array, so the launch asked the driver for a zero-byte buffer and `cuMemAlloc` refused.
    """
    gnp = _gnp()
    root = gnp.asarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    root.to_device()
    row = root[1]

    square_in_place[gpu_backend][(1, 4)](row)
    row.to_host(wait=True)

    expected = np.array([[0, 1, 2, 3], [16, 25, 36, 49], [8, 9, 10, 11]],
                        dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(root), expected)


def test_a_gnp_column_view_is_not_contiguous_and_stays_correct(gpu_backend):
    """A strided view takes the staging branch; its writes still land in the root.

    The read-back is deferred, so the host bytes are current only after `to_host()` -- the
    same contract a whole-array launch has.
    """
    gnp = _gnp()
    root = gnp.asarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    root.to_device()
    column = root[:, 1]

    square_in_place[gpu_backend][(1, 3)](column)
    root.to_host(wait=True)

    expected = np.array([[0, 1, 2, 3], [4, 25, 6, 7], [8, 81, 10, 11]], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(root), expected)
