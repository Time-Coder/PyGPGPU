"""Layout and construction: the two halves of the vector contract that must not move.

Every class keeps its structured dtype byte-identical (itemsize, field list, class/registry
identity), and every class builds the same way: zeros, broadcast, positional, a flattened
sequence, another vector's components -- with the C conversion rules a narrow slot applies.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

import pygf

from pygpgpu.opencl.driver import CLInfo
from pygpgpu.cuda.driver import CUInfo

from pygf import float2, float3, float4

from pygpgpu.vec_types import (
    char2, char4, uchar2,
    genVec2,
    genVec3,
    genVec4,
    genVec8,
    genVec16,
)

from _helpers.genvec_classes import (
    NARROW_CLASSES,
    NARROW_FLOAT_CLASSES,
    WIDE_CLASSES,
    width_of,
)


# ---------------------------------------------------------------------------
# Layout: byte-identical dtypes before and after the switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES)
def test_numpy_dtype_matches_fields(cls: type) -> None:
    """`numpy_dtype()` and `np.dtype(_fields_)` answer the same structured dtype."""
    dtype: np.dtype = cls.numpy_dtype()
    assert dtype == np.dtype(cls._fields_)
    assert dtype.itemsize == ctypes.sizeof(cls)
    assert list(dtype.fields) == [name for name, _ in cls._fields_]


@pytest.mark.parametrize("name", ["char2", "uchar3", "short4", "ushort8", "int2",
                                  "uint4", "float2", "float4", "float8", "float16",
                                  "double2", "double16", "long3", "long8"])
def test_registry_classes_keep_their_layout(name: str) -> None:
    """Every registry class answers a structured dtype sized like its class."""
    cls: type = CLInfo.vec_types[name]
    assert cls.numpy_dtype().itemsize == ctypes.sizeof(cls)


def test_the_shared_classes_are_still_pygf_objects() -> None:
    """The borrowed classes are pygf's, and a padded vec3 is not."""
    assert float2 is CLInfo.vec_types["float2"]
    assert float2 is CUInfo.vec_types["float2"]
    assert float4 is CUInfo.vec_types["float4"]
    assert float3 is not CLInfo.vec_types["float3"]
    assert CLInfo.vec_types["float3"].numpy_dtype().itemsize == 16


def test_every_vector_class_inherits_pygf_genvec() -> None:
    """The parallel base is gone: every class hangs off pygf's genVec."""
    for cls in NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES:
        assert issubclass(cls, pygf.genVec), cls.__name__
        assert issubclass(cls, pygf.genType), cls.__name__
    for name in ("char3", "float3", "long3", "long8"):
        assert issubclass(CLInfo.vec_types[name], pygf.genVec), name


def test_the_width_bases_declare_their_width() -> None:
    """The genVecN intermediates are thin: they name the width over the padding."""
    for base, width in ((genVec2, 2), (genVec3, 3), (genVec4, 4),
                        (genVec8, 8), (genVec16, 16)):
        assert base.__len__(base()) == width
    assert issubclass(genVec2, pygf.genVec2)
    assert issubclass(genVec3, pygf.genVec3)
    assert issubclass(genVec4, pygf.genVec4)
    assert issubclass(genVec8, pygf.genVec)
    assert issubclass(genVec16, pygf.genVec)

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES)
def test_zero_and_broadcast_construction(cls: type) -> None:
    assert list(cls()) == [0] * width_of(cls)
    assert list(cls(7)) == [7] * width_of(cls)


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES)
def test_positional_construction(cls: type) -> None:
    values: list[int] = list(range(width_of(cls)))
    assert list(cls(*values)) == values


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES)
def test_a_sequence_argument_flattens(cls: type) -> None:
    values: list[int] = list(range(width_of(cls)))
    assert list(cls(values)) == values
    assert list(cls(tuple(values))) == values
    assert list(cls(values[:2], *values[2:])) == values


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES)
def test_a_vector_argument_gives_its_components(cls: type) -> None:
    values: list[int] = list(range(width_of(cls)))
    assert list(cls(cls(*values))) == values


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES)
def test_wrong_arity_is_refused(cls: type) -> None:
    with pytest.raises(ValueError):
        cls(*range(width_of(cls) + 1))
    with pytest.raises(ValueError):
        cls(*range(width_of(cls)), 0)
    if width_of(cls) > 2:
        # two components fill only the narrowest widths; beyond that it is too few
        with pytest.raises(ValueError):
            cls(1, 2)


def test_a_float_into_a_narrow_component_truncates_like_c() -> None:
    """char2(1.9, -1.5) is (1, -1): the C float-to-integer conversion."""
    assert list(char2(1.9, -1.5)) == [1, -1]
    assert list(uchar2(2.9, 0.5)) == [2, 0]
    value: object = char2()
    value.x = 3.7
    assert list(value) == [3, 0]


def test_an_out_of_range_integer_wraps_like_ctypes() -> None:
    assert list(char2(-1, 300)) == [-1, 44]
    assert list(uchar2(256, 255)) == [0, 255]


def test_a_float_vector_builds_a_narrow_vector() -> None:
    """char2(float2(1.9, 2.9)) truncates, it does not refuse."""
    assert list(char2(float2(1.9, 2.9))) == [1, 2]


def test_a_mixed_width_vector_argument_fills_components() -> None:
    assert list(float4(float2(1.0, 2.0), 3.0, 4.0)) == [1.0, 2.0, 3.0, 4.0]
    assert list(char4(char2(1, 2), 3, 4)) == [1, 2, 3, 4]

# ---------------------------------------------------------------------------
# The sequence protocol and text forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES)
def test_len_iter_and_index(cls: type) -> None:
    value: object = cls(*range(width_of(cls)))
    assert len(value) == width_of(cls)
    assert list(value) == list(range(width_of(cls)))
    for index in range(width_of(cls)):
        assert value[index] == index


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES)
def test_repr_and_str_name_the_class(cls: type) -> None:
    value: pygf.genVec = cls(*range(width_of(cls)))
    expected: str = f"{cls.__name__}({', '.join(str(v) for v in value)})"
    assert repr(value) == expected
    assert str(value) == expected
