"""Narrow scalar arguments: the numpy name is a width, not a vector type.

`np.int8` and the registry's eight-component `int8` vector share a class name, so a signature
lookup by name alone turned a byte-valued argument into a vector parameter the device could
not bind. Both backends also have to *carry* the byte correctly: OpenCL binds `char` to
`ctypes.c_char`, whose numpy view is a one-byte string, so casting an integer array to it
converted the numbers to characters -- a 7 arrived as the byte for "7" (0x37, read back as
55).
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel
from pygpgpu.kernel.translator import infer_ctype


@pytest.mark.parametrize("scalar_type, expected", [
    (np.int8, "int"),
    (np.int16, "int"),
    (np.uint8, "int"),
    (np.uint16, "int"),
    (np.float16, "float"),
    (np.float32, "float"),
    (np.float64, "float"),
])
def test_a_numpy_scalar_is_a_scalar_not_a_vector_of_that_name(scalar_type, expected):
    assert infer_ctype(scalar_type(1)) == expected


def test_the_vector_type_of_the_same_name_still_resolves():
    """`int8` as a vector type is a real type, so the class itself must still map."""
    from pygpgpu import vec_types

    assert infer_ctype(vec_types.int8(*range(8))) == "int8"


def test_a_vector_alias_still_resolves():
    from pygf import color3f

    assert infer_ctype(color3f(1.0, 2.0, 3.0)) == "float3"


@kernel
def read_byte(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = int(x[i])


@pytest.mark.parametrize("element_type", [np.int8, np.uint8])
@pytest.mark.parametrize("value", [7, -7])
def test_a_byte_array_reaches_the_device_as_its_numbers(runner, element_type, value):
    if element_type is np.uint8 and value < 0:
        pytest.skip("an unsigned byte cannot hold a negative value")
    x = np.array([value], dtype=element_type)
    out = np.zeros(1, dtype=np.float32)
    runner(read_byte, x, out)
    assert float(out[0]) == float(value)


@kernel
def read_short(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = float(x[i])


@pytest.mark.parametrize("element_type, value", [
    (np.int16, -1234),
    (np.uint16, 40000),
])
def test_the_other_narrow_arrays_carry_their_values(runner, element_type, value):
    """fp16 is out of scope on CUDA/HIP (see AGENTS.md), so the integer widths are what
    this pins."""
    x = np.array([value], dtype=element_type)
    out = np.zeros(1, dtype=np.float32)
    runner(read_short, x, out)
    assert float(out[0]) == pytest.approx(float(value), rel=1e-3)
