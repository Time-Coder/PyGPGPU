"""The buffers tests, split out of `test_metal_launch_path.py`.

The subject is buffers.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_a_data_size_mismatch_is_refused_before_the_copy() -> None:
    """The copy length is the buffer's own size, so the data must match it.

    `set_data` copies `self._size` bytes with `memmove`, which reads past the end of a
    shorter array and silently truncates a longer one. `_host_pointer_of` is the single
    place both `set_data` and `bind_data` go through, and it refuses the mismatch
    before either touches memory.
    """
    from pygpgpu.metal.oop.buffer import Buffer


    class FakeBuffer:
        """Just enough of a Buffer to run `_host_pointer_of` without a device."""

        _size = 8
        _contiguous_with_size = staticmethod(Buffer._contiguous_with_size)

    check = Buffer._host_pointer_of

    mismatched = np.zeros(4, np.uint8)
    with pytest.raises(ValueError) as refused:
        check(FakeBuffer(), mismatched)
    assert "not equal to buffer size" in str(refused.value)

    # A matching array is accepted, and the pointer is the array's own storage.
    matching = np.zeros(8, np.uint8)
    pointer, array = check(FakeBuffer(), matching)
    assert pointer == int(matching.ctypes.data)
    assert array.nbytes == 8


def test_bytes_and_bytearray_are_accepted_as_data() -> None:
    """A `bytes` payload is a valid buffer source, sized by its own length."""
    from pygpgpu.metal.oop.buffer import Buffer

    array, size = Buffer._contiguous_with_size(b"abcd")
    assert size == 4 and array.nbytes == 4
    array, size = Buffer._contiguous_with_size(bytearray(b"abc"))
    assert size == 3 and array.nbytes == 3
