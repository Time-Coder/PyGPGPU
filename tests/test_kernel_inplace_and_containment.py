"""In-place operators and containment, dispatched through the class's own methods.

C has no in-place operator for a structure, so `v += w` used to emit `v += w` and fail to
compile; python answers it by asking `v.__iadd__(w)` and falling back to `v.__add__(w)`. The
translation now follows that and stores the returned value back. `x in box` / `x not in box`
reach `box.__contains__(x)`, whose receiver is the *right* operand.

Every class here has its own layout on purpose: two classes with one layout share a dtype,
hence one C type, so their names and methods would come from whichever registered first.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from pygpgpu import kernel


class Acc(ctypes.Structure):
    """Two floats: knows `__iadd__`, and answers `in` too."""

    _fields_ = [("value", ctypes.c_float), ("spare", ctypes.c_float)]

    def __iadd__(self, other):
        return Acc(self.value + other.value, 0.0)

    def __contains__(self, item):
        return item == self.value


class Plain(ctypes.Structure):
    """Three floats: only `__add__`, so `+=` has to fall back to it."""

    _fields_ = [("value", ctypes.c_float), ("a", ctypes.c_float), ("b", ctypes.c_float)]

    def __add__(self, other):
        return Plain(self.value + other.value, 0.0, 0.0)


@kernel
def in_place_add(lhs, rhs, answer: np.ndarray):
    for i in range(answer.shape[0]):
        total = Acc(lhs.value, 0.0)
        step = Acc(rhs.value, 0.0)
        total += step
        answer[i] = total.value


@kernel
def in_place_fallback(lhs, rhs, answer: np.ndarray):
    for i in range(answer.shape[0]):
        total = Plain(lhs.value, 0.0, 0.0)
        step = Plain(rhs.value, 0.0, 0.0)
        total += step
        answer[i] = total.value


@kernel
def contained(holder, answer: np.ndarray):
    for i in range(answer.shape[0]):
        answer[i] = 1.0 if 2.0 in holder else 0.0


@kernel
def not_contained(carrier, answer: np.ndarray):
    for i in range(answer.shape[0]):
        answer[i] = 1.0 if 5.0 not in carrier else 0.0


def test_an_in_place_operator_uses_the_class_method(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(in_place_add, Acc(1.0, 0.0), Acc(2.0, 0.0), out)
    assert float(out[0]) == pytest.approx(3.0)


def test_an_in_place_operator_falls_back_to_the_plain_one(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(in_place_fallback, Plain(1.0, 0.0, 0.0), Plain(2.0, 0.0, 0.0), out)
    assert float(out[0]) == pytest.approx(3.0)


def test_containment_asks_the_right_operand(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(contained, Acc(2.0, 0.0), out)
    assert float(out[0]) == pytest.approx(1.0)
    runner(contained, Acc(3.0, 0.0), out)
    assert float(out[0]) == pytest.approx(0.0)


def test_not_containment_negates_the_same_call(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(not_contained, Acc(2.0, 0.0), out)
    assert float(out[0]) == pytest.approx(1.0)
    runner(not_contained, Acc(5.0, 0.0), out)
    assert float(out[0]) == pytest.approx(0.0)


class Converter(ctypes.Structure):
    """A class that defines every conversion the translator supports.

    A distinct layout again: one layout per class, or they share a dtype and a C type.
    """

    _fields_ = [("value", ctypes.c_float), ("spare", ctypes.c_int)]

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return self.value

    def __abs__(self):
        return Converter(abs(self.value), 0)

    def __round__(self):
        return round(self.value)

    def __call__(self, factor):
        return Converter(self.value * factor, 0)


@kernel
def converts_to_int(boxed, answer: np.ndarray):
    for i in range(answer.shape[0]):
        answer[i] = float(int(boxed))


@kernel
def converts_to_float(boxed, answer: np.ndarray):
    for i in range(answer.shape[0]):
        answer[i] = float(boxed)


@kernel
def converts_with_abs(boxed, answer: np.ndarray):
    for i in range(answer.shape[0]):
        answer[i] = abs(boxed).value


@kernel
def converts_with_round(boxed, answer: np.ndarray):
    for i in range(answer.shape[0]):
        answer[i] = float(round(boxed))


@kernel
def calls_the_value(boxed, answer: np.ndarray):
    for i in range(answer.shape[0]):
        answer[i] = boxed(3.0).value


def test_a_conversion_asks_the_class(runner):
    """`int(box)` is `box.__int__()` and `float(box)` is `box.__float__()`.

    Without the dispatch the translator cast the aggregate itself, which the compiler
    rejected ("invalid type conversion").
    """
    out = np.zeros(1, dtype=np.float32)
    runner(converts_to_int, Converter(-2.7, 0), out)
    assert float(out[0]) == pytest.approx(-2.0)
    runner(converts_to_float, Converter(2.5, 0), out)
    assert float(out[0]) == pytest.approx(2.5)


def test_a_single_argument_builtin_asks_the_class(runner):
    """`abs(box)` is `box.__abs__()` and `round(box)` is `box.__round__()`."""
    out = np.zeros(1, dtype=np.float32)
    runner(converts_with_abs, Converter(-2.5, 0), out)
    assert float(out[0]) == pytest.approx(2.5)
    runner(converts_with_round, Converter(2.4, 0), out)
    assert float(out[0]) == pytest.approx(2.0)


def test_calling_a_value_asks_the_class(runner):
    """`box(args)` is `box.__call__(args)` -- the callee is a value, not a function."""
    out = np.zeros(1, dtype=np.float32)
    runner(calls_the_value, Converter(2.0, 0), out)
    assert float(out[0]) == pytest.approx(6.0)


class Measured(ctypes.Structure):
    """A class whose computed member is a `@property` with a setter.

    A distinct layout, like every class here: one layout is one dtype, hence one C type.
    """

    _fields_ = [("raw", ctypes.c_float), ("scale", ctypes.c_float), ("tag", ctypes.c_int)]

    @property
    def doubled(self):
        return self.raw * self.scale

    @doubled.setter
    def doubled(self, value):
        self.raw = value / self.scale


@kernel
def reads_a_property(measured, answer: np.ndarray):
    for i in range(answer.shape[0]):
        answer[i] = measured.doubled


@kernel
def writes_a_property(measured, answer: np.ndarray):
    for i in range(answer.shape[0]):
        measured.doubled = 8.0
        answer[i] = measured.raw


def test_a_property_read_calls_its_getter(runner):
    """`box.total` is a call to the property's getter, not a field read.

    Without the dispatch the translator looked for a member of that name and refused with
    "structure ... has no member '.total'".
    """
    out = np.zeros(1, dtype=np.float32)
    runner(reads_a_property, Measured(3.0, 2.0, 0), out)
    assert float(out[0]) == pytest.approx(6.0)


def test_a_property_write_calls_its_setter(runner):
    """`box.total = v` calls the setter with `v`, and the store it makes reaches the host."""
    out = np.zeros(1, dtype=np.float32)
    runner(writes_a_property, Measured(0.0, 2.0, 0), out)
    assert float(out[0]) == pytest.approx(4.0)


@kernel
def accumulates_a_vector(value, answer: np.ndarray):
    """`pixel += color`: C has no `+=` for a vector type."""
    for i in range(answer.shape[0]):
        total = value * 0.0
        total += value
        total += value
        answer[i] = total.x


@kernel
def accumulates_a_call(color, answer: np.ndarray):
    """The operand is a call, which must be evaluated once, not once per component."""
    for i in range(answer.shape[0]):
        total = color * 0.0
        total += color * 2.0
        answer[i] = total.y


def test_a_vector_in_place_operator_is_written_out(runner):
    """`v += w` on a vector becomes `v = v + w`, because the vector has no `+=`.

    CUDA rejects `v += w` for a vector type outright ("no operator \"+=\" matches these
    operands"), and the per-component `+` the vector dispatch produces is the form the
    store needs. An operand that is a call is evaluated into a temporary first: the
    component-wise spelling names it once per component.
    """
    from pygf import float3

    out = np.zeros(1, dtype=np.float32)
    runner(accumulates_a_vector, float3(1.0, 2.0, 3.0), out)
    assert float(out[0]) == pytest.approx(2.0)   # 1.0 accumulated twice
    runner(accumulates_a_call, float3(1.0, 2.0, 3.0), out)
    assert float(out[0]) == pytest.approx(4.0)   # 2.0 doubled
