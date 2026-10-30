"""Device helpers must be emitted with this backend's own type names.

CUDA C defines neither `uchar` nor `ushort`/`uint`/`ulong` -- those are OpenCL spellings
the translator uses internally, and `FunctionTranslator._device_type` maps them at every
place that writes a declaration. A cast written straight from the internal name compiled
on OpenCL and failed on CUDA, which is what these tests pin down.

The trigger is a helper whose parameter type is a narrow integer but whose `return`
promotes it: `def f(v: float, ...)` called with a `uint8` element infers `uchar` as the
return type from its first `return`, and later returns have to cast to it.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# A helper that returns a narrow integer type
# ---------------------------------------------------------------------------


def narrow_identity(value: float, alpha: float) -> float:
    """The shape that exposed the bug: an early return fixes a non-float return type."""
    if abs(alpha) < 1e-6:
        return value
    return (1 - np.exp(-alpha * value)) / (1 - np.exp(-alpha))


@kernel
def brightness_by_uint8(image: np.ndarray, alpha: float):
    """Assigning a helper's float result into a uint8 array casts to uchar."""
    dest: np.ndarray = np.zeros_like(image)
    for i, j, k in np.ndindex(image.shape):
        dest[i, j, k] = narrow_identity(image[i, j, k], alpha)
    return dest


def test_a_helper_returning_a_narrow_int_type_compiles(gpu_backend, runner):
    """uchar is an OpenCL spelling; the CUDA backend has to emit unsigned char."""
    image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
    result = runner(brightness_by_uint8, image, 0.0)
    assert result.dtype == np.uint8
    np.testing.assert_allclose(result, image)


# ---------------------------------------------------------------------------
# The other translator-only spellings
# ---------------------------------------------------------------------------


def narrow_pass(value: float) -> float:
    if value > 1e9:
        return value
    return value * 2.0


@kernel
def uint16_helper(values: np.ndarray):
    dest = np.zeros_like(values)
    for i in range(values.shape[0]):
        dest[i] = narrow_pass(values[i])
    return dest


@kernel
def uint32_helper(values: np.ndarray):
    dest = np.zeros_like(values)
    for i in range(values.shape[0]):
        dest[i] = narrow_pass(values[i])
    return dest


@kernel
def uint64_helper(values: np.ndarray):
    dest = np.zeros_like(values)
    for i in range(values.shape[0]):
        dest[i] = narrow_pass(values[i])
    return dest


@pytest.mark.parametrize("kernel_func, dtype", [
    (uint16_helper, np.uint16),
    (uint32_helper, np.uint32),
    (uint64_helper, np.uint64),
])
def test_every_narrow_integer_spelling_compiles(gpu_backend, runner, kernel_func, dtype):
    values = np.array([2, 3, 4], dtype=dtype)
    result = runner(kernel_func, values)
    assert result.dtype == dtype
    np.testing.assert_allclose(result, values * 2)


# ---------------------------------------------------------------------------
# An integer cast inside the loop stays a device cast, not a host creation
# ---------------------------------------------------------------------------


@kernel
def cast_a_device_value(values: np.ndarray, out: np.ndarray):
    """`np.uint8(x)` runs on the device: it converts a value, it builds no array."""
    for i in range(out.shape[0]):
        out[i] = float(np.uint8(values[i]))
    return None


def test_a_numpy_dtype_cast_inside_the_loop_is_device_code(runner):
    values = np.array([1.5, 2.7, 3.9], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(cast_a_device_value, values, out)
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0], dtype=np.float32))
