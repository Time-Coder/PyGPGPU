"""An `Image` argument is a texture handle, not an object to flatten into a structure.

Split out as its own subject because the bug was a *classification* one: `looks_like_object`
claims anything the translator does not already recognize, and an `Image` is neither an
array nor a ctypes record nor a device handle, so it was claimed, scanned for members the
body reads (`src[y, x]` is a subscript, not a member access), found to read none, and
refused. The interpreter ran the same kernel correctly, because it needs no device type --
so the feature worked on exactly the backend that does not need it and was refused on both
that do.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_an_image_argument_is_not_an_object_argument() -> None:
    """The predicate that decides the two paths answers "not an object" for an `Image`."""
    from pygpgpu.image import Image
    from pygpgpu.kernel.objects.scan_paths import looks_like_object

    assert not looks_like_object(Image(np.zeros((2, 2), np.float32)))
    # The neighbouring answers must not have moved: an array is not an object either, and
    # a plain class instance still is (that is the path a real object argument takes).
    assert not looks_like_object(np.zeros(2, np.float32))

    class Plain:
        pass

    assert looks_like_object(Plain())


def test_a_kernel_sampling_an_image_compiles_on_every_device_backend(gpu_backend) -> None:
    """The whole point: a kernel that takes a texture compiles and answers.

    Before the fix every device backend raised `SyntaxError: '<name>' is a Image, but the
    kernel body reads none of its attributes` -- the object-argument refusal -- so this
    pins the behaviour end to end rather than only the predicate.
    """
    from pygpgpu import Image, kernel

    if gpu_backend == "hip":
        pytest.skip("HIP refuses every image kernel by name (see the image refusal)")

    @kernel
    def shift_right(src: Image, dst: Image):
        for y, x in np.ndindex(dst.shape):
            dst[y, x] = src[y, x]

    source = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    dst = Image(shape=(4, 5), dtype=np.float32)
    shift_right[gpu_backend](Image(source), dst)
    np.testing.assert_allclose(dst.to_numpy(), source)
