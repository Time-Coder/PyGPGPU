"""numpy serialization and the dtype surface.

`np.array` reads a vector as its components, a vector list as an array of vectors, and an
explicit structured dtype as records; the class answers the structured dtype and an instance
the component's ctypes type.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from pygf import float2, float4

from pygpgpu.vec_types import (
    char2,
    char4,
    uchar2,
    ushort2,
)
from pygpgpu.opencl.driver.vec_types import (
    char16,
    double16,
    float8,
)

from _helpers.genvec_classes import (
    NARROW_CLASSES,
    NARROW_FLOAT_CLASSES,
    WIDE_CLASSES,
    width_of,
)



# ---------------------------------------------------------------------------
# numpy serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [float2, float4, char2, char4, uchar2, ushort2,
                                 float8, char16, double16])
def test_a_vector_builds_a_flat_array(cls: type) -> None:
    n: int = width_of(cls)
    value: object = cls(*range(n))
    array: np.ndarray = np.array(value)
    assert array.shape == (n,)
    assert array.tolist() == list(range(n))


@pytest.mark.parametrize("cls", [float2, float4, char2, char4, ushort2, float8,
                                 char16])
def test_a_vector_list_builds_an_array_of_vectors(cls: type) -> None:
    n: int = width_of(cls)
    array: object = np.array([cls(*range(n)), cls(*range(n, 2 * n))])
    assert array.shape == (2,)
    assert array.dtype == cls.numpy_dtype()
    assert list(array[0]) == list(range(n))
    assert list(array[1]) == list(range(n, 2 * n))


@pytest.mark.parametrize("cls", [float2, char2, uchar2, ushort2, float8, char16])
def test_a_structured_dtype_builds_records(cls: type) -> None:
    n: int = width_of(cls)
    value: object = cls(*range(n))
    array: np.ndarray = np.array([value], dtype=cls.numpy_dtype())
    assert array.dtype == cls.numpy_dtype()
    assert list(array[0]) == list(range(n))

# ---------------------------------------------------------------------------
# The dtype surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES)
def test_class_dtype_is_the_structured_numpy_dtype(cls: type) -> None:
    assert isinstance(cls.dtype, np.dtype)
    assert cls.dtype == np.dtype(cls._fields_)


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES)
def test_instance_dtype_is_the_component_ctypes_type(cls: type) -> None:
    value: object = cls()
    component: object = value.dtype
    assert isinstance(component, type)
    assert issubclass(component, ctypes._SimpleCData)
    assert ctypes.sizeof(component) * len(cls._fields_) == ctypes.sizeof(cls)
