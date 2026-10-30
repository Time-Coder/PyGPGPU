"""Feature 2: creating arrays inside the kernel body, above the magic loop.

Python builds a value before the loop; the device never runs that statement. Instead the
host executes the creation (it is ordinary python), keeps the object for the call, and
passes it to the kernel as an extra parameter whenever the body uses it. A created object
the body returns reuses that same parameter as its return slot, so the caller gets the
host's own object, not a copy.

Only statements above the magic loop may create: below it they would run once per work
item, which is not what a python function would do. Everything the device cannot build --
a list, a dict, a python object -- is refused at translation time rather than silently
producing a different result.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# Created arrays used inside the kernel
# ---------------------------------------------------------------------------


@kernel
def fill_with_zeros(x: np.ndarray, out: np.ndarray):
    scratch = np.zeros(4, dtype=np.float32)
    for i in range(out.shape[0]):
        out[i] = scratch[0] + x[i]


@kernel
def filled_with_a_value(x: np.ndarray, out: np.ndarray):
    table = np.full(2, 7.0, dtype=np.float32)
    for i in range(out.shape[0]):
        out[i] = x[i] + table[1]


@kernel
def created_from_arange(x: np.ndarray, out: np.ndarray):
    ramp = np.arange(4, dtype=np.float32)
    for i in range(out.shape[0]):
        out[i] = ramp[i] * x[i]


# ---------------------------------------------------------------------------
# The annotated spelling of an assignment creates just the same
# ---------------------------------------------------------------------------


@kernel
def annotated_creation(x: np.ndarray):
    """`name: T = np.zeros_like(...)` is the same statement to python as `name = ...`."""
    result: np.ndarray = np.zeros_like(x)
    for i in range(result.shape[0]):
        result[i] = x[i] * 2.0
    return result


def test_an_annotated_creation_is_host_built(runner):
    x = np.arange(4, dtype=np.float32)
    result = runner(annotated_creation, x)
    np.testing.assert_allclose(result, x * 2.0)


@kernel
def annotated_creation_used_internally(x: np.ndarray, out: np.ndarray):
    """The annotated form also has to feed a loop domain and read back normally."""
    scratch: np.ndarray = np.zeros_like(x)
    for i in range(scratch.shape[0]):
        scratch[i] = x[i] + 1.0
    for i in range(out.shape[0]):
        out[i] = scratch[i] * 3.0


def test_an_annotated_creation_used_inside_the_kernel(runner):
    x = np.arange(3, dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(annotated_creation_used_internally, x, out)
    np.testing.assert_allclose(out, (x + 1.0) * 3.0)


@kernel
def annotated_creation_below_the_loop(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        scratch: np.ndarray = np.zeros(2, dtype=np.float32)
        out[i] = scratch[0]


def test_an_annotated_creation_below_the_loop_is_a_private_array(runner):
    """A constant shape below the loop is per-work-item storage, which is what python means."""
    out = np.ones(2, dtype=np.float32)
    runner(annotated_creation_below_the_loop, np.zeros(2, np.float32), out)
    np.testing.assert_allclose(out, np.zeros(2, np.float32))


@kernel
def annotated_declaration_only(x: np.ndarray, out: np.ndarray):
    """`name: np.ndarray` with no value declares nothing; it must not confuse the scan."""
    for i in range(out.shape[0]):
        out[i] = x[i]


def test_an_annotation_without_a_value_is_harmless(runner):
    x = np.arange(3, dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(annotated_declaration_only, x, out)
    np.testing.assert_allclose(out, x)


@kernel
def created_by_copy(x: np.ndarray, out: np.ndarray):
    stage = np.empty(4, dtype=np.float32)
    for i in range(out.shape[0]):
        stage[i] = x[i]
        out[i] = stage[i] * 2.0


@kernel
def created_like_an_argument(x: np.ndarray, out: np.ndarray):
    scratch = np.zeros_like(x)
    for i in range(out.shape[0]):
        scratch[i] = x[i] * 2.0
        out[i] = scratch[i]


def test_zeros_created_above_the_loop(runner):
    x = np.ones(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(fill_with_zeros, x, out)
    np.testing.assert_allclose(out, np.ones(4, dtype=np.float32))


def test_full_created_above_the_loop(runner):
    x = np.zeros(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(filled_with_a_value, x, out)
    np.testing.assert_allclose(out, np.full(4, 7.0, dtype=np.float32))


def test_arange_created_above_the_loop(runner):
    x = np.ones(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(created_from_arange, x, out)
    np.testing.assert_allclose(out, np.arange(4, dtype=np.float32))


def test_created_array_is_written_inside_the_loop(runner):
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(created_by_copy, x, out)
    np.testing.assert_allclose(out, x * 2.0)


def test_zeros_like_an_argument(runner):
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(created_like_an_argument, x, out)
    np.testing.assert_allclose(out, x * 2.0)


# ---------------------------------------------------------------------------
# Created values that are scalars
# ---------------------------------------------------------------------------


@kernel
def scalar_creation(np_value: np.float32, out: np.ndarray):
    scale = np.float32(2.5)
    for i in range(out.shape[0]):
        out[i] = np_value * scale


def test_created_scalar_value(runner):
    out = np.zeros(3, dtype=np.float32)
    runner(scalar_creation, np.float32(2.0), out)
    np.testing.assert_allclose(out, np.full(3, 5.0, dtype=np.float32))


# ---------------------------------------------------------------------------
# Returning a created object
# ---------------------------------------------------------------------------


@kernel
def return_created_array(x: np.ndarray):
    result = np.zeros(x.shape[0], dtype=np.float32)
    for i in range(x.shape[0]):
        result[i] = x[i] * 2.0
    return result


def test_created_array_can_be_returned(runner):
    x = np.arange(4, dtype=np.float32)
    result = runner(return_created_array, x)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, x * 2.0)


@kernel
def return_created_vector(x: np.ndarray):
    for i in range(x.shape[0]):
        if i == 0:
            return np.float32(x[i] * 3.0)


def test_created_scalar_can_be_returned(runner):
    x = np.array([2.0, 5.0], dtype=np.float32)
    assert float(runner(return_created_vector, x)) == 6.0


# ---------------------------------------------------------------------------
# Several created objects, and one used by a helper
# ---------------------------------------------------------------------------


@kernel
def two_created_arrays(x: np.ndarray, out: np.ndarray):
    first = np.zeros(2, dtype=np.float32)
    second = np.ones(2, dtype=np.float32)
    for i in range(out.shape[0]):
        out[i] = first[0] + second[1] + x[i]


def test_two_created_arrays_in_one_kernel(runner):
    x = np.arange(3, dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(two_created_arrays, x, out)
    np.testing.assert_allclose(out, x + 1.0)


@kernel
def helper_uses_created_array(x: np.ndarray, out: np.ndarray):
    table = np.arange(3, dtype=np.float32)
    for i in range(out.shape[0]):
        out[i] = lookup_created(table, i)


def lookup_created(table, index):
    return table[index] * 10.0


def test_a_created_array_can_reach_a_helper(runner):
    x = np.zeros(3, dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(helper_uses_created_array, x, out)
    np.testing.assert_allclose(out, np.arange(3, dtype=np.float32) * 10.0)


# ---------------------------------------------------------------------------
# A created array as the iteration domain
# ---------------------------------------------------------------------------


def sum_over_block(src: np.ndarray, i: int, j: int, ratio: int):
    """A plain helper: no decorator, translated into a device function at the call site."""
    total: float = 0.0
    for sub_i in range(i * ratio, (i + 1) * ratio):
        total = total + src[sub_i, j]
    return total


@kernel
def downsample(src: np.ndarray, ratio: int):
    """The loop's domain is a *created* array, and its shape is a tuple expression."""
    scaled = np.zeros(
        (int(src.shape[0] / ratio), int(src.shape[1] / ratio)),
        dtype=np.float32,
    )
    for i, j in np.ndindex(scaled.shape):
        scaled[i, j] = sum_over_block(src, i, j, ratio)
    return scaled


def test_created_array_can_define_the_magic_loop(runner):
    """`for i, j in np.ndindex(created.shape)` must be recognised as the magic loop."""
    src = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    ratio = 2
    result = runner(downsample, src, ratio)
    assert result.shape == (4, 4)
    expect = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            total = 0.0
            for sub_i in range(i * ratio, (i + 1) * ratio):
                total += src[sub_i, j]
            expect[i, j] = total
    np.testing.assert_allclose(result, expect, rtol=1e-6)


@kernel
def fill_from_created_like(x: np.ndarray):
    """A `*_like` creation with an argument-dependent shape, used as the domain."""
    scratch = np.zeros_like(x)
    for i in range(scratch.shape[0]):
        scratch[i] = x[i] + 1.0
    return scratch


def test_created_like_defines_the_magic_loop(runner):
    x = np.arange(4, dtype=np.float32)
    result = runner(fill_from_created_like, x)
    np.testing.assert_allclose(result, x + 1.0)


@kernel
def enumerate_created(x: np.ndarray):
    """`for i, v in enumerate(created)` -- the other 1-D magic-loop form."""
    scratch = np.arange(x.shape[0], dtype=np.float32)
    for i, v in enumerate(scratch):
        scratch[i] = v + x[i]
    return scratch


def test_enumerate_over_a_created_array(runner):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = runner(enumerate_created, x)
    np.testing.assert_allclose(result, np.array([1.0, 3.0, 5.0], dtype=np.float32))


@kernel
def range_over_created_shape(x: np.ndarray, out: np.ndarray):
    """`range(created.shape[0])` -- a 1-D magic loop reading a created array's extent."""
    scratch = np.full(x.shape[0], 3.0, dtype=np.float32)
    for i in range(scratch.shape[0]):
        out[i] = x[i] * scratch[i]
    return None


def test_range_over_a_created_shape(runner):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(range_over_created_shape, x, out)
    np.testing.assert_allclose(out, x * 3.0)


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


@kernel
def creation_below_the_loop(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        scratch = np.zeros(2, dtype=np.float32)
        out[i] = scratch[0]


def test_creation_below_the_magic_loop_is_a_private_array(runner):
    out = np.ones(2, dtype=np.float32)
    runner(creation_below_the_loop, np.zeros(2, np.float32), out)
    np.testing.assert_allclose(out, np.zeros(2, np.float32))


@kernel
def creates_a_list(x: np.ndarray):
    values = [1, 2, 3]
    for i in range(x.shape[0]):
        if i == 0:
            x[i] = values[0]


def test_creating_an_unsupported_value_is_rejected(gpu_backend, runner):
    with pytest.raises(SyntaxError):
        runner(creates_a_list, np.zeros(3, dtype=np.float32))
