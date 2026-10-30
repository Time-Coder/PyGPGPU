"""The np.ndindex magic loop accepts every spelling numpy accepts.

numpy treats these as the same iteration domain, so a kernel body may use any of them:

    for i, j in np.ndindex(arr.shape):
    for i, j in np.ndindex(arr.shape[0], arr.shape[1]):
    for i, j in np.ndindex(2, 3):

Only the first used to be recognised. The other two were not seen as the magic loop at
all, so every statement above them -- including a host-created array -- was translated as
device code and failed with a message about the creation rather than the loop.

Recognition is not enough on its own: the launch sizes dimensions fastest-first while the
loop unpacks them slowest-first, so a domain given in loop order would launch transposed
and silently leave the last row or column unwritten. The correctness tests below cover
both halves.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# One spelling per shape, each writing every element it claims
# ---------------------------------------------------------------------------


@kernel
def whole_shape_loop(image: np.ndarray):
    """`np.ndindex(dest.shape)`: the spelling that already worked."""
    dest: np.ndarray = np.zeros_like(image)
    for i, j, k in np.ndindex(dest.shape):
        dest[i, j, k] = image[i, j, k] + 1
    return dest


@kernel
def explicit_extents_loop(image: np.ndarray):
    """`np.ndindex(dest.shape[0], dest.shape[1])`: two dimensions of a 3-D image."""
    dest: np.ndarray = np.zeros_like(image)
    for i, j in np.ndindex(dest.shape[0], dest.shape[1]):
        dest[i, j, 0] = image[i, j, 2]
        dest[i, j, 1] = image[i, j, 1]
        dest[i, j, 2] = image[i, j, 0]
    return dest


@kernel
def literal_extents_loop(image: np.ndarray):
    """`np.ndindex(2, 3)`: the domain stated outright, with no array involved."""
    dest: np.ndarray = np.zeros((2, 3), dtype=np.float32)
    for i, j in np.ndindex(2, 3):
        dest[i, j] = float(i * 10 + j)
    return dest


@kernel
def extents_from_two_arrays(a: np.ndarray, b: np.ndarray):
    """Each extent from a different array: the sources are tracked per dimension."""
    for i, j in np.ndindex(a.shape[0], b.shape[1]):
        b[i, j] = a[i, j] * 2.0
    return None


def test_whole_shape_domain(runner):
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    result = runner(whole_shape_loop, image)
    np.testing.assert_array_equal(result, image + 1)


def test_explicit_extents_domain(runner):
    """Writing all three channels of every pixel: a transposed launch would miss column 2."""
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    result = runner(explicit_extents_loop, image)
    np.testing.assert_array_equal(result, image[:, :, ::-1])


def test_literal_extents_domain(runner):
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    result = runner(literal_extents_loop, image)
    assert result.shape == (2, 3)
    expected = np.array([[i * 10 + j for j in range(3)] for i in range(2)], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_extents_from_two_arrays(runner):
    """A non-square domain from two arrays: a transposed launch leaves half of it unwritten."""
    a = np.ones((2, 4), dtype=np.float32)
    b = np.zeros((2, 4), dtype=np.float32)
    runner(extents_from_two_arrays, a, b)
    np.testing.assert_allclose(b, a * 2.0)


# ---------------------------------------------------------------------------
# The creation above the loop is lifted out for every spelling
# ---------------------------------------------------------------------------


@kernel
def creation_above_explicit_extents(image: np.ndarray):
    scratch: np.ndarray = np.ones_like(image)
    for i, j in np.ndindex(scratch.shape[0], scratch.shape[1]):
        scratch[i, j, 0] = image[i, j, 0] * 2.0
    return scratch


def test_a_creation_above_an_explicit_extent_loop_works(runner):
    image = np.full((2, 3, 3), 5, dtype=np.float32)
    result = runner(creation_above_explicit_extents, image)
    expected = np.ones((2, 3, 3), dtype=np.float32)
    expected[:, :, 0] = image[:, :, 0] * 2.0
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Mismatches are named rather than silently reshaped
# ---------------------------------------------------------------------------


@kernel
def mismatched_extents(x: np.ndarray, out: np.ndarray):
    for i, j, k in np.ndindex(x.shape[0], x.shape[1]):
        out[i] = x[i]


def test_a_domain_that_does_not_match_the_unpacking_is_rejected(gpu_backend, runner):
    with pytest.raises(SyntaxError) as caught:
        runner(mismatched_extents, np.zeros((2, 3), np.float32), np.zeros(2, np.float32))
    message = str(caught.value)
    assert "3" in message and "2" in message
