"""Every numpy array constructor works inside a kernel body, above the magic loop.

The user-facing promise is "any function that creates an ndarray is supported": the host
runs the statement, so the array has exactly the shape and dtype numpy gives it, and the
kernel receives it as a parameter. These tests exercise the constructors by category
(direct allocation, `*_like`, ranges, layout transforms, combining) rather than trusting
that a name list stays complete.

The functions are real module-level definitions because a kernel needs retrievable
source: `inspect.getsource` cannot read a function built with `exec`.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# Direct allocation
# ---------------------------------------------------------------------------


@kernel
def make_zeros(x: np.ndarray):
    a = np.zeros((2, 2), dtype=np.float32)
    for i, j in np.ndindex(a.shape):
        a[i, j] = 1.0
    return a


@kernel
def make_ones(x: np.ndarray):
    a = np.ones((2, 2), dtype=np.float32)
    for i, j in np.ndindex(a.shape):
        a[i, j] = a[i, j] + 0.0
    return a


@kernel
def make_empty(x: np.ndarray):
    a = np.empty((2, 2), dtype=np.float32)
    for i, j in np.ndindex(a.shape):
        a[i, j] = 7.0
    return a


@kernel
def make_full(x: np.ndarray):
    a = np.full((2, 2), 3.0, dtype=np.float32)
    for i, j in np.ndindex(a.shape):
        a[i, j] = a[i, j] + 0.0
    return a


@kernel
def make_eye(x: np.ndarray):
    a = np.eye(3, dtype=np.float32)
    for i, j in np.ndindex(a.shape):
        a[i, j] = a[i, j] + 0.0
    return a


@kernel
def make_identity(x: np.ndarray):
    a = np.identity(3, dtype=np.float32)
    for i, j in np.ndindex(a.shape):
        a[i, j] = a[i, j] + 0.0
    return a


@kernel
def make_tri(x: np.ndarray):
    a = np.tri(3, dtype=np.float32)
    for i, j in np.ndindex(a.shape):
        a[i, j] = a[i, j] + 0.0
    return a


@kernel
def make_diag(x: np.ndarray):
    a = np.diag(np.arange(3, dtype=np.float32))
    for i, j in np.ndindex(a.shape):
        a[i, j] = a[i, j] + 0.0
    return a


@kernel
def make_like(x: np.ndarray):
    zeros = np.zeros_like(x)
    full = np.full_like(x, 2.0)
    for i in range(zeros.shape[0]):
        zeros[i] = x[i] + full[i]
    return zeros


@pytest.mark.parametrize(
    "kernel_func, expected",
    [
        (make_zeros, np.ones((2, 2), dtype=np.float32)),
        (make_ones, np.ones((2, 2), dtype=np.float32)),
        (make_empty, np.full((2, 2), 7.0, dtype=np.float32)),
        (make_full, np.full((2, 2), 3.0, dtype=np.float32)),
        (make_eye, np.eye(3, dtype=np.float32)),
        (make_identity, np.identity(3, dtype=np.float32)),
        (make_tri, np.tri(3, dtype=np.float32)),
        (make_diag, np.diag(np.arange(3, dtype=np.float32))),
    ],
)
def test_a_direct_allocation_constructor(runner, kernel_func, expected):
    x = np.arange(4, dtype=np.float32)
    result = runner(kernel_func, x)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_zeros_like_and_full_like(runner):
    x = np.arange(4, dtype=np.float32)
    np.testing.assert_allclose(runner(make_like, x), x + 2.0)


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------


@kernel
def make_arange(x: np.ndarray):
    a = np.arange(4, dtype=np.float32)
    for i in range(a.shape[0]):
        a[i] = a[i] * 2.0
    return a


@kernel
def make_linspace(x: np.ndarray):
    a = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    for i in range(a.shape[0]):
        a[i] = a[i] + 0.0
    return a


def test_arange_constructor(runner):
    np.testing.assert_allclose(runner(make_arange, np.zeros(4, np.float32)),
                               np.arange(4, dtype=np.float32) * 2.0)


def test_linspace_constructor(runner):
    np.testing.assert_allclose(runner(make_linspace, np.zeros(4, np.float32)),
                               np.linspace(0.0, 1.0, 4, dtype=np.float32), rtol=1e-6)


# ---------------------------------------------------------------------------
# Layout and shape transforms
# ---------------------------------------------------------------------------


@kernel
def make_flipped(x: np.ndarray):
    a = np.flip(np.arange(4, dtype=np.float32))
    for i in range(a.shape[0]):
        a[i] = a[i] + 0.0
    return a


@kernel
def make_rolled(x: np.ndarray):
    a = np.roll(np.arange(4, dtype=np.float32), 1)
    for i in range(a.shape[0]):
        a[i] = a[i] + 0.0
    return a


@kernel
def make_repeated(x: np.ndarray):
    a = np.repeat(np.arange(2, dtype=np.float32), 2)
    for i in range(a.shape[0]):
        a[i] = a[i] + 0.0
    return a


@kernel
def make_reshaped(x: np.ndarray):
    a = np.reshape(np.arange(4, dtype=np.float32), (2, 2))
    for i, j in np.ndindex(a.shape):
        a[i, j] = a[i, j] + 0.0
    return a


@kernel
def make_clipped(x: np.ndarray):
    a = np.clip(np.arange(4, dtype=np.float32), 0.0, 1.0)
    for i in range(a.shape[0]):
        a[i] = a[i] + 0.0
    return a


@kernel
def make_cumsum(x: np.ndarray):
    a = np.cumsum(np.arange(4, dtype=np.float32))
    for i in range(a.shape[0]):
        a[i] = a[i] + 0.0
    return a


def test_flip_constructor(runner):
    np.testing.assert_allclose(runner(make_flipped, np.zeros(4, np.float32)),
                               np.flip(np.arange(4, dtype=np.float32)))


def test_roll_constructor(runner):
    np.testing.assert_allclose(runner(make_rolled, np.zeros(4, np.float32)),
                               np.roll(np.arange(4, dtype=np.float32), 1))


def test_repeat_constructor(runner):
    np.testing.assert_allclose(runner(make_repeated, np.zeros(4, np.float32)),
                               np.repeat(np.arange(2, dtype=np.float32), 2))


def test_reshape_constructor(runner):
    np.testing.assert_allclose(runner(make_reshaped, np.zeros(4, np.float32)),
                               np.arange(4, dtype=np.float32).reshape(2, 2))


def test_clip_constructor(runner):
    np.testing.assert_allclose(runner(make_clipped, np.zeros(4, np.float32)),
                               np.clip(np.arange(4, dtype=np.float32), 0.0, 1.0))


def test_cumsum_constructor(runner):
    np.testing.assert_allclose(runner(make_cumsum, np.zeros(4, np.float32)),
                               np.cumsum(np.arange(4, dtype=np.float32)))


# ---------------------------------------------------------------------------
# Combining
# ---------------------------------------------------------------------------


@kernel
def make_concatenated(x: np.ndarray):
    a = np.concatenate((np.arange(2, dtype=np.float32), np.arange(2, dtype=np.float32)))
    for i in range(a.shape[0]):
        a[i] = a[i] + 0.0
    return a


@kernel
def make_stacked(x: np.ndarray):
    a = np.stack((np.arange(2, dtype=np.float32), np.arange(2, dtype=np.float32)))
    for i, j in np.ndindex(a.shape):
        a[i, j] = a[i, j] + 0.0
    return a


@kernel
def make_hstacked(x: np.ndarray):
    a = np.hstack((np.arange(2, dtype=np.float32), np.arange(2, dtype=np.float32)))
    for i in range(a.shape[0]):
        a[i] = a[i] + 0.0
    return a


def test_concatenate_constructor(runner):
    expected = np.concatenate((np.arange(2, dtype=np.float32),
                               np.arange(2, dtype=np.float32)))
    np.testing.assert_allclose(runner(make_concatenated, np.zeros(4, np.float32)), expected)


def test_stack_constructor(runner):
    expected = np.stack((np.arange(2, dtype=np.float32), np.arange(2, dtype=np.float32)))
    np.testing.assert_allclose(runner(make_stacked, np.zeros(4, np.float32)), expected)


def test_hstack_constructor(runner):
    expected = np.hstack((np.arange(2, dtype=np.float32), np.arange(2, dtype=np.float32)))
    np.testing.assert_allclose(runner(make_hstacked, np.zeros(4, np.float32)), expected)


# ---------------------------------------------------------------------------
# The dtype constructors are casts, not creations
# ---------------------------------------------------------------------------


@kernel
def cast_with_numpy_dtype(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.float32(x[i] * 2.0)


def test_numpy_dtype_cast(runner):
    x = np.arange(1, 4, dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(cast_with_numpy_dtype, x, out)
    np.testing.assert_allclose(out, x * 2.0)
