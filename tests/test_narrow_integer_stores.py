"""Narrow integer arrays must answer what python/numpy answer for the same source.

Two divergences, both silent (no error, a different number than the interpreter gives):

* **item 1**: assigning a negative float into an unsigned array. numpy 2 stores the modulo
  (``uint8(-1.0) == 255``); C converts a negative float to an unsigned type as undefined
  behaviour and the compiler answers 0.
* **item 2**: arithmetic on a narrow element. ``uint8[255] + 1`` is 0 with numpy's uint8
  wrap; the device read the element as a 32-bit ``int`` and answered 256.

The chosen scope for item 2 is the **store**: the value written *into* a narrow array is
wrapped to that array's element width. The element's own read width is unchanged (it is
``int`` by the documented `infer_ctype` contract), so only the assignment is affected.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# Item 1: a float assigned into an integer array, modulo numpy's rules
# ---------------------------------------------------------------------------


@kernel
def store_from(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i]


@kernel
def store_constant(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = -1.0


@pytest.mark.parametrize("dtype, value, expected", [
    (np.uint8, -1.0, 255),
    (np.uint16, -1.0, 65535),
    (np.int8, -1.0, -1),
    (np.int8, 130.0, -126),          # numpy: int8(130.0) wraps to -126
    (np.uint8, 300.0, 44),           # numpy: uint8(300.0) == 44
])
def test_a_float_stored_into_an_integer_array_wraps_like_numpy(gpu_backend, dtype, value, expected):
    """Matches what numpy computes for the same assignment (the interpreter's answer)."""
    out = np.zeros(1, dtype=dtype)
    store_from[gpu_backend](np.array([value], dtype=np.float32), out)
    assert int(out[0]) == expected, f"{np.dtype(dtype).name} <- {value} answered {int(out[0])}"


def test_a_negative_constant_stored_into_an_unsigned_array(gpu_backend):
    """A literal goes through the same conversion as a value read from an array."""
    out = np.zeros(1, dtype=np.uint8)
    store_constant[gpu_backend](out)
    assert int(out[0]) == 255


# ---------------------------------------------------------------------------
# Item 2: the value written back into a narrow array is wrapped to its width
# ---------------------------------------------------------------------------


@kernel
def increment_into(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] + 1


@kernel
def double_into(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] * 2


@pytest.mark.parametrize("dtype, value, expected", [
    (np.uint8, 255, 0),              # numpy: uint8(255) + 1 wraps to 0
    (np.int8, 127, -128),            # numpy: int8(127) + 1 wraps to -128
    (np.uint16, 65535, 0),
    (np.int16, 32767, -32768),
])
def test_an_increment_stored_into_a_narrow_array_wraps(gpu_backend, dtype, value, expected):
    """The store wraps to the destination's element width, like the interpreter's numpy."""
    out = np.zeros(1, dtype=dtype)
    increment_into[gpu_backend](np.array([value], dtype=dtype), out)
    assert int(out[0]) == expected, f"{np.dtype(dtype).name} {value} + 1 answered {int(out[0])}"


def test_a_double_stored_into_a_narrow_array_wraps(gpu_backend):
    """`* 2` overflows as readily as `+ 1`; the wrap belongs to the store, not one operator."""
    out = np.zeros(1, dtype=np.uint8)
    double_into[gpu_backend](np.array([200], dtype=np.uint8), out)
    assert int(out[0]) == 144        # 400 % 256


def test_a_value_that_fits_is_unchanged(gpu_backend):
    """The control: no overflow, so the stored number is the computed one."""
    out = np.zeros(1, dtype=np.uint8)
    increment_into[gpu_backend](np.array([7], dtype=np.uint8), out)
    assert int(out[0]) == 8
