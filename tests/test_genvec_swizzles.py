"""Swizzles: the GLSL letter spellings and OpenCL's numeric s0-style ones.

Reads answer a family vector of the swizzle's length; writes store through the component's
own accessor, refuse repeats, and broadcast a scalar. The numeric names are gated on for
the PyGPGPU classes at every width and off for pygf's own surface.
"""
from __future__ import annotations

import pytest

from pygf import float2, float4, int2

from pygpgpu.vec_types import (
    char16,
    char2,
    char4,
    char8,
    float16,
    float8,
    short2,
    short4,
    uchar2,
    ushort2,
)

from _helpers.genvec_classes import (
    NARROW_CLASSES,
    WIDE_CLASSES,
    family_base,
    width_of,
)



# ---------------------------------------------------------------------------
# Letter swizzles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [float2, int2, char2, uchar2, short2, ushort2,
                                 float8, char8])
def test_a_single_letter_reads_one_component(cls: type) -> None:
    n: int = width_of(cls)
    value: object = cls(*range(n))
    assert value.x == 0
    assert value.r == 0
    assert value.s == 0
    if n > 1:
        assert value.y == 1
        assert value.g == 1
        assert value.t == 1
    if n > 2:
        assert value.z == 2
        assert value.b == 2
    if n > 3:
        assert value.w == 3
        assert value.a == 3


@pytest.mark.parametrize("cls", [float2, float4, int2, char2, char4, uchar2,
                                 short4, ushort2, float8, char16])
def test_a_letter_swizzle_reads_a_family_vector(cls: type) -> None:
    n: int = width_of(cls)
    value: object = cls(*range(n))
    xy: object = value.xy
    assert family_base(type(xy)) == family_base(cls)
    assert len(xy) == 2
    assert list(xy) == [0, 1]
    assert list(value.yx) == [1, 0]
    assert list(value.xx) == [0, 0]
    # a three-component result needs a 3-wide class in the family: pygf's
    # float/int families have one, the narrow families do not (their vec3s are
    # device classes in the driver packages)
    if n > 2 and family_base(cls) in ("float", "double", "int", "uint"):
        assert len(value.xyz) == 3
        assert list(value.xyz) == [0, 1, 2]
        assert list(value.rgb) == [0, 1, 2]
    elif n > 2:
        with pytest.raises(AttributeError):
            value.xyz
    if n == 4:
        assert len(value.xyzw) == 4
        assert list(value.rgba) == [0, 1, 2, 3]
        assert list(value.stpq) == [0, 1, 2, 3]


def test_a_wide_vector_swizzles_within_its_first_four_components() -> None:
    value: object = float8(*range(8))
    assert list(value.xy) == [0.0, 1.0]
    assert list(value.wz) == [3.0, 2.0]
    assert len(value.xyzw) == 4


def test_an_unknown_swizzle_is_refused() -> None:
    with pytest.raises(AttributeError):
        char2(1, 2).q
    with pytest.raises(AttributeError):
        float4(1, 2, 3, 4).xyzzz
    with pytest.raises(AttributeError):
        char2(1, 2).xyzzz

# ---------------------------------------------------------------------------
# Numeric swizzles (OpenCL's s0-style component spelling)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES)
def test_a_numeric_component_reads_one_slot(cls: type) -> None:
    n: int = width_of(cls)
    value: object = cls(*range(n))
    assert value.s0 == 0
    if n > 1:
        assert value.s1 == 1
    if n > 2:
        assert value.s2 == 2
        assert value.s3 == 3


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES)
def test_a_numeric_swizzle_reads_a_family_vector(cls: type) -> None:
    n: int = width_of(cls)
    value: object = cls(*range(n))
    pair: object = value.s01
    assert family_base(type(pair)) == family_base(cls)
    assert len(pair) == 2
    assert list(pair) == [0, 1]
    assert list(value.s10) == [1, 0]
    assert list(value.s00) == [0, 0]


def test_numeric_swizzles_reach_every_width() -> None:
    assert list(char4(1, 2, 3, 4).s0123) == [1, 2, 3, 4]
    assert list(char4(1, 2, 3, 4).s0101) == [1, 2, 1, 2]
    value: object = float8(*range(8))
    assert list(value.s04) == [0.0, 4.0]
    assert list(value.s0123) == [0.0, 1.0, 2.0, 3.0]
    assert list(value.s01234567) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert type(value.s01234567) is float8
    wide: object = float16(*range(16))
    assert wide.s0 == 0.0
    assert wide.sA == 10.0
    assert wide.sa == 10.0
    assert wide.sF == 15.0
    assert list(wide.sA0) == [10.0, 0.0]
    assert list(wide.saf) == [10.0, 15.0]
    assert list(wide.s0123456789ABCDEF) == [float(i) for i in range(16)]
    assert type(wide.s0123456789ABCDEF) is float16
    chars: object = char16(*range(16))
    assert chars.sF == 15
    assert list(chars.sAF) == [10, 15]
    assert list(chars.sfa) == [15, 10]
    # on a 16-wide vector the lowercase hex digits shadow rgba's a and b
    assert chars.a == 10
    assert list(chars.rgba) == [0, 1, 11, 10]


def test_a_numeric_swizzle_beyond_the_charset_is_refused() -> None:
    with pytest.raises(AttributeError):
        char2(1, 2).s2
    with pytest.raises(AttributeError):
        float8(*range(8)).s8
    with pytest.raises(AttributeError):
        float8(*range(8)).sF
    with pytest.raises(AttributeError):
        char2(1, 2).sxyz
    with pytest.raises(AttributeError):
        char2(1, 2).s01234


def test_pygf_classes_keep_their_letter_only_surface() -> None:
    """The gate stays off for the classes pygf owns: no numeric swizzles there."""
    value: object = float2(1.0, 2.0)
    with pytest.raises(AttributeError):
        value.s0

# ---------------------------------------------------------------------------
# Swizzle writes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [float2, float4, int2, char2, char4, uchar2,
                                 short4, ushort2])
def test_a_letter_swizzle_write_reaches_the_fields(cls: type) -> None:
    n: int = width_of(cls)
    value: object = cls(*range(n))
    value.xy = (9, 8)
    assert list(value) == [9, 8] + list(range(2, n))
    value.x = 5
    assert value[0] == 5
    with pytest.raises(AttributeError):
        value.xx = (1, 1)
    if n == 2:
        with pytest.raises(AttributeError):
            value.xyz = (1, 2, 3)
    value.xy = 7
    assert value[0] == 7 and value[1] == 7


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES)
def test_a_numeric_swizzle_write_reaches_the_fields(cls: type) -> None:
    n: int = width_of(cls)
    value: object = cls(*range(n))
    value.s01 = (9, 8)
    assert value[0] == 9 and value[1] == 8
    value.s0 = 5
    assert value[0] == 5
    with pytest.raises(AttributeError):
        value.s00 = (1, 1)
    value.s01 = 7
    assert value[0] == 7 and value[1] == 7


def test_a_swizzle_write_accepts_a_vector_value() -> None:
    value: object = char4(1, 2, 3, 4)
    value.xy = char2(9, 8)
    assert list(value) == [9, 8, 3, 4]
    value.s01 = char2(7, 6)
    assert list(value) == [7, 6, 3, 4]


def test_a_float_into_a_narrow_swizzle_write_truncates() -> None:
    value: object = char2(0, 0)
    value.x = 1.9
    assert list(value) == [1, 0]


def test_a_swizzle_write_with_a_wrong_length_is_refused() -> None:
    value: object = char2(1, 2)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        value.xy = (1, 2, 3)
