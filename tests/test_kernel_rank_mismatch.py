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
    """`out[i, j]` on a 3-D array means a slice in python, and used to write one element.

    Python raises nothing here -- `out[i, j]` selects a sub-array -- so this is a *device* limit
    rather than a mirror of the interpreter, and it is reported as a ValueError to keep it
    apart from the IndexError a real indexing mistake raises.
    """
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 100
    out = np.zeros((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError) as caught:
        runner(partial_subscript, a, out)
    assert not isinstance(caught.value, IndexError)
    message: str = str(caught.value)
    assert "sub-array" in message
    assert "shape (2, 3, 4)" in message
    assert "Index every dimension" in message
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


@kernel
def over_indexed(a: np.ndarray, out: np.ndarray):
    """`a[i, i]` on a 1-D array: python raises from the subscript, not from the loop."""
    for i in range(out.shape[0]):
        out[i] = a[i, i] + 1.0


def test_a_too_wide_subscript_reads_exactly_like_python(runner):
    """The message is numpy's own sentence, with nothing appended.

    An extra clause naming the array and its shape used to be added here. It made the
    decorated and undecorated readings differ, and the caller's own line already names the
    array -- `a` is visible in `out[i] = a[i, i] + 1.0`.
    """
    a = np.arange(9, dtype=np.float32)
    out = np.zeros(9, dtype=np.float32)
    with pytest.raises(IndexError) as caught:
        runner(over_indexed, a, out)
    assert str(caught.value) == ("too many indices for array: array is 1-dimensional, "
                                 "but 2 were indexed")


def test_the_failing_subscript_is_the_one_python_would_blame(runner):
    """Two frames -- the caller and the failing line -- and the FIRST offending subscript.

    The body reaches `a[i, i]` before anything else, so that is where python raises; blaming
    the last subscript of the body (what recording "the widest one" used to do when several
    tie) pointed a reader at a statement that had not run yet.
    """
    import linecache
    import traceback

    a = np.arange(9, dtype=np.float32)
    out = np.zeros(9, dtype=np.float32)
    with pytest.raises(IndexError) as caught:
        runner(over_indexed, a, out)

    frames = []
    cursor = caught.value.__traceback__
    while cursor is not None:
        frames.append(cursor)
        cursor = cursor.tb_next
    # The kernel's own line is the last frame, and no frame of this library is anywhere in
    # the chain -- the caller's frames (the test, the fixture) come first, as they do for an
    # undecorated call.
    assert frames[-1].tb_frame.f_code.co_name == "over_indexed"
    for node in frames:
        assert "pygpgpu" not in node.tb_frame.f_code.co_filename.replace("\\", "/").split("/"), \
            node.tb_frame.f_code.co_filename
    body = frames[-1]
    assert body.tb_frame.f_code.co_filename == __file__
    assert "a[i, i]" in linecache.getline(__file__, body.tb_lineno)

    # ...and CPython draws its caret under that subscript, as it does without the decorator.
    rendered = "".join(traceback.format_exception(caught.value))
    assert "^" in rendered.splitlines()[-1] or "^" in rendered.splitlines()[-2], rendered
