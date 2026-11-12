"""Arithmetic, comparison and unary operators under the unified semantics.

Result types promote through pygf's dtype order and resolve through the operand's own
family; comparisons answer the int vector of the operand's package; in-place arithmetic
mutates and returns self.
"""
from __future__ import annotations

import pytest

from pygf import double2, float2, float4, int2, int4, uint2

from pygpgpu.vec_types import (
    char2,
    char4,
    char8,
    double16,
    float16,
    float8,
    int8,
    short2,
    uchar2,
    uint8,
    ushort2,
    ushort4,
)

from _helpers.genvec_classes import (
    NARROW_CLASSES,
    NARROW_FLOAT_CLASSES,
    WIDE_CLASSES,
    family_base,
    width_of,
)



# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES
                         + [int2, int4, uint2, double2])
def test_homogeneous_arithmetic(cls: type) -> None:
    n: int = width_of(cls)
    a: object = cls(*range(2, n + 2))
    b: object = cls(*range(1, n + 1))
    assert type(a + b) is cls
    assert list(a + b) == [2 * i + 3 for i in range(n)]
    assert type(a - b) is cls
    assert list(a - b) == [1] * n
    assert type(a * b) is cls
    # small factors: a narrow slot wraps what it cannot hold
    assert list(cls(*[2] * n) * cls(*[3] * n)) == [6] * n


@pytest.mark.parametrize("cls", [float2, float4, double2, float8, double16])
def test_float_division(cls: type) -> None:
    n: int = width_of(cls)
    a: object = cls(*[3.0] * n)
    b: object = cls(*[2.0] * n)
    assert type(a / b) is cls
    assert list(a / b) == [1.5] * n


@pytest.mark.parametrize("cls", [int2, int4, char2, char4, short2])
def test_integer_division_answers_a_float_vector(cls: type) -> None:
    """`/` between integer types below `uint` is a true division into a float vector."""
    n: int = width_of(cls)
    a: object = cls(*[7] * n)
    b: object = cls(*[2] * n)
    result: object = a / b
    assert family_base(type(result)) == "float"
    assert len(result) == n
    assert list(result) == [3.5] * n


@pytest.mark.parametrize("cls", [ushort2, ushort4])
def test_unsigned_short_division_truncates_like_c(cls: type) -> None:
    """`ushort` ranks above `uint` -- it is the `half` storage dtype -- so its division
    stays in the family and truncates toward zero, the C unsigned division."""
    n: int = width_of(cls)
    a: object = cls(*[7] * n)
    b: object = cls(*[2] * n)
    assert type(a / b) is cls
    assert list(a / b) == [3] * n


@pytest.mark.parametrize("cls", [int2, char2, char4, short2, ushort2, float4, float8])
def test_floor_division_and_modulo(cls: type) -> None:
    n: int = width_of(cls)
    a: object = cls(*[7] * n)
    b: object = cls(*[2] * n)
    assert type(a // b) is cls
    assert list(a // b) == [3] * n
    assert type(a % b) is cls
    assert list(a % b) == [1] * n


@pytest.mark.parametrize("cls", [int2, char2, ushort2, float4, double2, int8, float16])
def test_power(cls: type) -> None:
    n: int = width_of(cls)
    a: object = cls(*[3] * n)
    assert type(a ** cls(*[2] * n)) is cls
    assert list(a ** cls(*[2] * n)) == [9] * n
    assert type(a ** 2) is cls
    assert list(a ** 2) == [9] * n


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES)
def test_scalar_arithmetic_broadcasts(cls: type) -> None:
    n: int = width_of(cls)
    a: object = cls(*[2] * n)
    assert type(a + 1) is cls
    assert list(a + 1) == [3] * n
    assert list(1 + a) == [3] * n
    assert type(a * 2) is cls
    assert list(a * 2) == [4] * n
    assert list(2 * a) == [4] * n
    assert list(a - 1) == [1] * n
    assert list(3 - a) == [1] * n


def test_arithmetic_promotes_to_the_wider_family() -> None:
    """char2 + int2 is an int2; short2 + ushort2 is the unsigned short2."""
    result: object = char2(1, 2) + int2(10, 20)
    assert type(result) is int2
    assert list(result) == [11, 22]
    assert type(int2(10, 20) + char2(1, 2)) is int2
    assert type(short2(1, 2) + ushort2(1, 1)) is ushort2
    assert type(uchar2(1, 2) + char2(1, 2)) is uchar2
    assert type(char2(1, 2) + float2(1.0, 2.0)) is float2


def test_arithmetic_across_the_two_families_now_interops() -> None:
    """A pygf class and a PyGPGPU class are one hierarchy: no broadcast bug."""
    result: object = float2(1.0, 2.0) + char2(1, 1)
    assert type(result) is float2
    assert list(result) == [2.0, 3.0]
    result = char2(1, 1) + float2(1.0, 2.0)
    assert type(result) is float2
    assert list(result) == [2.0, 3.0]
    result = int8(*[1] * 8) + char8(*[1] * 8)
    assert type(result) is int8
    assert list(result) == [2] * 8
    with pytest.raises(TypeError):
        # different widths have no element-wise meaning
        float8(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) + float4(1, 1, 1, 1)


def test_a_foreign_operand_is_refused_by_python() -> None:
    with pytest.raises(TypeError):
        char2(1, 2) + "x"
    with pytest.raises(TypeError):
        float2(1.0, 2.0) * (1 + 2j)

# ---------------------------------------------------------------------------
# Unary operators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [int2, char2, char4, short2, float2, float4,
                                 float8, int8])
def test_negation(cls: type) -> None:
    n: int = width_of(cls)
    a: object = cls(*range(1, n + 1))
    assert list(-a) == [-i for i in range(1, n + 1)]


@pytest.mark.parametrize("cls", [int2, char2, short2, float2, float8])
def test_pos_is_a_copy(cls: type) -> None:
    n: int = width_of(cls)
    a: object = cls(*range(1, n + 1))
    assert list(+a) == list(range(1, n + 1))
    assert +a is not a


@pytest.mark.parametrize("cls", [int2, char2, char4, short2, ushort2, int8, uint8])
def test_invert_is_component_wise_on_integer_vectors(cls: type) -> None:
    n: int = width_of(cls)
    a: object = cls(*[5] * n)
    # the slot stores what C stores: a signed slot keeps -6, an unsigned one wraps
    expected = [cls._fields_[0][1](~5).value] * n
    assert list(~a) == expected


@pytest.mark.parametrize("cls", [float2, float4, float8, double2])
def test_invert_is_refused_on_float_vectors(cls: type) -> None:
    with pytest.raises(TypeError):
        ~cls(*[1.0] * width_of(cls))


@pytest.mark.parametrize("cls", [int2, char2, char4, float2, float8])
def test_abs_is_component_wise(cls: type) -> None:
    n: int = width_of(cls)
    a: object = cls(*[-value for value in range(1, n + 1)])
    assert type(abs(a)) is cls
    assert list(abs(a)) == list(range(1, n + 1))

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES
                         + [int2, int4, double2])
def test_ordering_answers_an_int_vector(cls: type) -> None:
    n: int = width_of(cls)
    a: object = cls(*range(0, n))
    b: object = cls(*[n - 1 - i for i in range(n)])
    result: object = a < b
    assert family_base(type(result)) == "int"
    assert len(result) == n
    assert list(result) == [1 if i < n - 1 - i else 0 for i in range(n)]
    assert list(a > b) == [1 if i > n - 1 - i else 0 for i in range(n)]


@pytest.mark.parametrize("cls", [int2, char2, char4, float2, float8, ushort2])
def test_equality_answers_an_int_vector(cls: type) -> None:
    n: int = width_of(cls)
    a: object = cls(*[1] * n)
    assert list(a == cls(*[1] * n)) == [1] * n
    assert list(a == cls(*[0] * n)) == [0] * n
    assert list(a != cls(*[0] * n)) == [1] * n
    assert list(a == 1) == [1] * n
    assert list(a == 0) == [0] * n


def test_equality_with_a_foreign_type_is_a_bool() -> None:
    """`==`/`!=` against something that is not a vector or number stay bool."""
    assert (char2(1, 2) == "x") is False
    assert (char2(1, 2) != "x") is True
    assert (float2(1.0, 2.0) == "x") is False

# ---------------------------------------------------------------------------
# In-place arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [int2, char2, char4, ushort2, float2, float8, int8])
def test_inplace_arithmetic_mutates_and_returns_self(cls: type) -> None:
    n: int = width_of(cls)
    a: object = cls(*[2] * n)
    before: object = a
    a += cls(*[1] * n)
    assert a is before
    assert list(a) == [3] * n
    a *= 2
    assert list(a) == [6] * n
    a -= 1
    assert list(a) == [5] * n
    a //= 2
    assert list(a) == [2] * n
    a %= 2
    assert list(a) == [0] * n


def test_inplace_on_a_narrow_field_truncates_and_wraps() -> None:
    value: object = char2(1, 1)
    value += 0.9
    assert list(value) == [1, 1]
    value += 300
    assert list(value) == [45, 45]
