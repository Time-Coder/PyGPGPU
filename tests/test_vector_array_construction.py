"""Building a vector array on the host.

A vector is iterable (it yields its components), so numpy reads a list of them as a 2-D block
of numbers: `np.array([float3(...), float3(...)])` used to become a `(2, 3)` float64 matrix,
which is a different value from an array of two vectors -- and it silently doubled the
element width. The constructors are patched (`pygpgpu/vec_types/array.py`) so a sequence of
vector instances is an array of vectors, and so an array built with a vector dtype is the same
`VectorArray` view the other constructors produce.
"""
from __future__ import annotations

import numpy as np
import pytest
from pygf import float3, float4

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------


def test_a_list_of_vectors_is_an_array_of_vectors():
    array = np.array([float3(1.0, 2.0, 3.0), float3(4.0, 5.0, 6.0)])
    assert array.shape == (2,)
    assert array.dtype.itemsize == 12
    assert array[0].z == pytest.approx(3.0)
    assert array[1].x == pytest.approx(4.0)


def test_the_element_type_is_the_vector_type_not_wider():
    """`float3` is float32; the matrix spelling used to widen it to float64."""
    array = np.array([float3(1.0, 2.0, 3.0)])
    assert array.dtype["x"] == np.float32
    assert array.dtype.itemsize == 12


def test_a_nested_list_is_a_multi_dimensional_array():
    array = np.array([[float3(1, 2, 3), float3(4, 5, 6)],
                      [float3(7, 8, 9), float3(1, 1, 1)]])
    assert array.shape == (2, 2)
    assert array[1, 0].y == pytest.approx(8.0)


def test_an_explicit_vector_dtype_gives_the_same_view():
    array = np.array([float3(1.0, 2.0, 3.0)], dtype=float3.dtype)
    assert array.shape == (1,)
    assert array[0].x == pytest.approx(1.0)


def test_asarray_behaves_like_array():
    array = np.asarray([float3(1.0, 2.0, 3.0), float3(4.0, 5.0, 6.0)])
    assert array.shape == (2,)
    assert array[1].z == pytest.approx(6.0)


def test_a_wide_vector_keeps_its_width():
    array = np.array([float4(1.0, 2.0, 3.0, 4.0)])
    assert array.dtype.itemsize == 16
    assert array[0].w == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# What must not change
# ---------------------------------------------------------------------------


def test_a_plain_sequence_keeps_numpy_semantics():
    numbers = np.array([1, 2, 3])
    assert numbers.dtype == np.int64
    assert numbers.shape == (3,)


def test_an_explicit_scalar_dtype_is_respected():
    array = np.array([1.0, 2.0], dtype=np.float32)
    assert array.dtype == np.float32


def test_a_sequence_of_records_is_untouched():
    dtype = np.dtype([("i", "<i4"), ("f", "<f4")])
    array = np.array([(1, 0.5), (2, 1.5)], dtype=dtype)
    assert array.dtype == dtype
    assert array["i"].tolist() == [1, 2]


def test_a_numpy_scalar_is_not_mistaken_for_a_vector():
    """`np.uint8` shares its class name with an eight-component vector type in the registry."""
    from pygpgpu.vec_types.layout import vector_name_of

    assert vector_name_of(np.uint8(1)) is None
    array = np.asarray(np.uint8(1))
    assert array.shape == ()
    assert array.dtype == np.uint8
    assert int(array[()]) == 1


# ---------------------------------------------------------------------------
# The point of it: passing the array straight to a kernel
# ---------------------------------------------------------------------------


@kernel
def read_z(points: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = points[i].z


def test_the_constructed_array_is_a_kernel_argument(runner):
    """No dtype spelling and no view conversion: the list of vectors travels as vectors."""
    points = np.array([float3(1.0, 2.0, 3.0), float3(4.0, 5.0, 6.0)])
    out = np.zeros(1, dtype=np.float32)
    runner(read_z, points, out)
    assert float(out[0]) == pytest.approx(3.0)
