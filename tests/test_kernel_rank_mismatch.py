"""A magic loop whose index count disagrees with the array's rank.

The loop states its rank in the source (`for i, j, k in ...`) while an array's shape is
only known from the value passed at call time, so the mismatch can only be caught at
launch. The python backend reports it as the interpreter's own
``not enough values to unpack (expected 3, got 2)``; a device launch used to raise a bare
``IndexError: tuple index out of range``, which names neither the kernel nor the array.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


@kernel
def merge_gray_images(image1: np.ndarray, image2: np.ndarray,
                      weight1: float, weight2: float):
    """The correct body for 2-D images: one index per dimension."""
    dest: np.ndarray = np.zeros_like(image1)
    for i, j in np.ndindex(dest.shape):
        dest[i, j] = np.clip(
            weight1 * float(image1[i, j]) + weight2 * float(image2[i, j]), 0, 255)
    return dest


@kernel
def merge_rgb_images(image1: np.ndarray, image2: np.ndarray,
                     weight1: float, weight2: float):
    """A 3-D body: wrong for 2-D images, which is what the error test needs."""
    dest: np.ndarray = np.zeros_like(image1)
    for i, j, k in np.ndindex(dest.shape):
        dest[i, j, k] = np.clip(
            weight1 * float(image1[i, j, k]) + weight2 * float(image2[i, j, k]), 0, 255)
    return dest


@kernel
def partial_subscript(a: np.ndarray, out: np.ndarray):
    """`out[i, j] = a[i, j]` on 3-D data selects a slice in python, not one element."""
    for i, j in np.ndindex(out.shape[0], out.shape[1]):
        out[i, j] = a[i, j]
    return None


@kernel
def copy_every_element(a: np.ndarray, out: np.ndarray):
    """The correct spelling of the same kernel: one index per dimension."""
    for i, j, k in np.ndindex(out.shape):
        out[i, j, k] = a[i, j, k]
    return None


def test_a_matching_rank_still_works(runner):
    moon = np.arange(4 * 5, dtype=np.uint8).reshape(4, 5) * 3
    camera = np.arange(4 * 5, dtype=np.uint8).reshape(4, 5) * 2
    result = runner(merge_gray_images, moon, camera, 1.0, -1.0)
    expected = np.clip(moon.astype(np.int32) - camera.astype(np.int32), 0, 255)
    np.testing.assert_array_equal(result, expected)


def test_too_many_indices_names_the_array(gpu_backend, runner):
    moon = np.arange(4 * 5, dtype=np.uint8).reshape(4, 5)
    with pytest.raises(ValueError) as caught:
        runner(merge_rgb_images, moon, moon, 1.0, -1.0)
    message: str = str(caught.value)
    # The wording the interpreter uses, so the two backends read alike.
    assert "not enough values to unpack" in message
    assert "expected 3, got 2" in message
    # ...plus what the device path knows and python's message cannot say.
    assert "merge_rgb_images()" in message
    assert "shape (4, 5)" in message


def test_the_error_is_not_a_bare_index_error(gpu_backend, runner):
    """The old failure was `IndexError: tuple index out of range` with no context."""
    moon = np.arange(4 * 5, dtype=np.uint8).reshape(4, 5)
    with pytest.raises(ValueError) as caught:
        runner(merge_rgb_images, moon, moon, 1.0, -1.0)
    assert not isinstance(caught.value, IndexError)


def test_a_partial_subscript_is_refused(gpu_backend, runner):
    """`out[i, j]` on a 3-D array means a slice in python, and used to write one element."""
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 100
    out = np.zeros((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError) as caught:
        runner(partial_subscript, a, out)
    message: str = str(caught.value)
    assert "sub-array" in message
    assert "shape (2, 3, 4)" in message
    assert "indexed with 2 indices" in message
    # Nothing was written, so a caller cannot mistake a partial result for success.
    np.testing.assert_allclose(out, np.zeros((2, 3, 4), dtype=np.float32))


def test_the_python_backend_still_accepts_a_partial_subscript(runner):
    """The interpreter can represent a slice, so the python backend must keep working."""
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 100
    out = np.zeros((2, 3, 4), dtype=np.float32)
    partial_subscript["python"](a, out)
    np.testing.assert_allclose(out, a)


def test_a_full_subscript_on_the_same_data_works(runner):
    """The fix must not reject the correct spelling of the same kernel."""
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 100
    out = np.zeros((2, 3, 4), dtype=np.float32)
    runner(copy_every_element, a, out)
    np.testing.assert_allclose(out, a)


def test_the_python_backend_reports_the_same_kind_of_error(runner):
    """The interpreter's own message is the reference the device path mirrors."""
    moon = np.arange(4 * 5, dtype=np.uint8).reshape(4, 5)
    with pytest.raises(ValueError) as caught:
        merge_rgb_images["python"](moon, moon, 1.0, -1.0)
    assert "not enough values to unpack" in str(caught.value)
