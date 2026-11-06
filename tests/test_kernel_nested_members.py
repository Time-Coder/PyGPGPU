"""A structure that holds another structure, a quaternion, or a complex value.

The three cases are one feature seen three ways, and they had three different states before
this file existed. A nested structure worked on python/cuda/hip and failed on OpenCL (the
member's layout coincided with a vector, and OpenCL aligns a vector more strictly than the
host packs it). A quaternion member failed everywhere: the generated source never included
the quaternion header, and a backend reading the structure back met a type name it did not
know (`KeyError: 'quatf'`). A complex member was refused outright.

A complex member is spelled `c64`/`c128` -- the fixed names the complex header declares --
which is what makes `.real`/`.imag` and the arithmetic apply to it rather than to two
unrelated floats.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

import pygf
from pygpgpu import kernel


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class Holder(ctypes.Structure):
    _fields_ = [("origin", Point), ("scale", ctypes.c_float)]


@kernel
def read_nested(holder: Holder, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = holder.origin.x + holder.origin.y + holder.scale


class QuaternionHolder(ctypes.Structure):
    _fields_ = [("q", pygf.quatf), ("bias", ctypes.c_float)]


@kernel
def read_quaternion(holder: QuaternionHolder, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = holder.q.w * 1000 + holder.q.x * 100 + holder.q.y * 10 + holder.q.z


class PlainPoint:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class PlainHolder:
    def __init__(self, origin: PlainPoint, scale: float):
        self.origin = origin
        self.scale = scale


@kernel
def read_plain_nested(holder: PlainHolder, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = holder.origin.x + holder.origin.y + holder.scale


class ComplexHolder:
    def __init__(self, center: complex, scale: float):
        self.center = center
        self.scale = scale


@kernel
def read_complex(holder: ComplexHolder, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = holder.center.real * holder.scale + holder.center.imag


@kernel
def complex_arithmetic(holder: ComplexHolder, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = abs(holder.center * holder.center) + holder.scale


class ComplexMagnitude:
    def __init__(self, value: complex):
        self.value = value


@kernel
def write_complex(holder: ComplexMagnitude):
    for i in range(1):
        holder.value = complex(5.0, 6.0)


class ComplexSample:
    def __init__(self, point: complex, gain: float):
        self.point = point
        self.gain = gain


@kernel
def numpy_complex_member(sample: ComplexSample, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = sample.point.real + sample.gain


class WidthHolder:
    def __init__(self, wide: complex, narrow: complex):
        self.wide = wide
        self.narrow = narrow


@kernel
def both_complex_widths(holder: WidthHolder, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = holder.narrow.real + holder.wide.imag


# ---------------------------------------------------------------------------
# A structure inside a structure
# ---------------------------------------------------------------------------


def test_a_ctypes_structure_member(runner):
    """Two floats inside a structure.

    OpenCL spells this member `float2` (its layout matches one) and aligns it to 8, so the
    containing structure is 16 bytes while the caller's class is 12. The packed value is
    rebuilt with the tail padding the device declaration implies.
    """
    out = np.zeros(2, np.float32)
    runner(read_nested, Holder(Point(1.0, 2.0), 3.0), out)
    np.testing.assert_allclose(out, np.array([6.0, 6.0], np.float32))


def test_a_plain_class_member_becomes_a_structure(runner):
    """The synthesized case of the same layout, with the same padding rule."""
    out = np.zeros(2, np.float32)
    runner(read_plain_nested, PlainHolder(PlainPoint(1.0, 2.0), 3.0), out)
    np.testing.assert_allclose(out, np.array([6.0, 6.0], np.float32))


# ---------------------------------------------------------------------------
# A quaternion member
# ---------------------------------------------------------------------------


def test_a_quaternion_member_is_read(runner):
    """The generated source has to include the quaternion header for this declaration."""
    out = np.zeros(2, np.float32)
    runner(read_quaternion, QuaternionHolder(pygf.quatf(1.0, 2.0, 3.0, 4.0), 0.0), out)
    np.testing.assert_allclose(out, np.array([1234.0, 1234.0], np.float32))


# ---------------------------------------------------------------------------
# A complex member
# ---------------------------------------------------------------------------


def test_a_python_complex_member_is_read(runner):
    """`center.imag` proves the member is a complex value, not an unrelated structure."""
    out = np.zeros(2, np.float32)
    runner(read_complex, ComplexHolder(2.0 + 3.0j, 4.0), out)
    np.testing.assert_allclose(out, np.array([11.0, 11.0], np.float32))


def test_a_complex_member_supports_the_complex_operations(runner):
    out = np.zeros(2, np.float32)
    runner(complex_arithmetic, ComplexHolder(1.0 + 2.0j, 3.0), out)
    expected = abs((1 + 2j) ** 2) + 3.0
    np.testing.assert_allclose(out, np.array([expected, expected], np.float32), rtol=1e-6)


def test_a_numpy_complex_member_keeps_its_width(runner):
    sample = ComplexSample(np.complex64(1.0 + 2.0j), 3.0)
    out = np.zeros(2, np.float32)
    runner(numpy_complex_member, sample, out)
    np.testing.assert_allclose(out, np.array([4.0, 4.0], np.float32))


def test_both_complex_widths_travel_together(runner):
    """A `c64` and a `c128` member in one structure: both header blocks are needed."""
    holder = WidthHolder(np.complex128(1.0 + 2.0j), np.complex64(3.0 + 4.0j))
    out = np.zeros(2, np.float32)
    runner(both_complex_widths, holder, out)
    np.testing.assert_allclose(out, np.array([5.0, 5.0], np.float32))


def test_a_complex_member_written_by_the_body_comes_home_as_a_complex(runner):
    """The caller's object gets a plain `complex` back, not the record that stored it."""
    holder = ComplexMagnitude(0 + 0j)
    runner(write_complex, holder)
    assert isinstance(holder.value, complex)
    assert holder.value == pytest.approx(5 + 6j)


# ---------------------------------------------------------------------------
# The generated declaration
# ---------------------------------------------------------------------------


def test_a_complex_member_is_declared_with_the_fixed_name():
    """The device type is `c64`/`c128`, so the class and the header agree on one spelling."""
    from pygpgpu.kernel import structs
    from pygpgpu.kernel.complex import (Complex128Record, Complex64Record,
                                        record_complex_type_name)
    from pygpgpu.kernel.objects import member_type

    narrow = member_type(1 + 2j, "center", None)
    assert narrow is Complex64Record
    wide = member_type(1 + 2j, "center", {"float": "double"})
    assert wide is Complex128Record

    assert record_complex_type_name(np.dtype(Complex64Record)) == "c64"
    assert record_complex_type_name(np.dtype(Complex128Record)) == "c128"
    # The declaration names that fixed type, and the typedef block leaves it to the header.
    assert structs.struct_name(np.dtype(Complex64Record)) == "c64"
    rendered = structs.render_structures([np.dtype(Complex64Record)], "cuda")
    assert "c64" not in rendered


def test_a_quaternion_member_is_declared_with_the_fixed_name():
    from pygpgpu.kernel import structs

    assert structs.struct_name(np.dtype(pygf.quatf)) == "quatf"
    assert "quatf" not in structs.render_structures([np.dtype(pygf.quatf)], "cuda")
