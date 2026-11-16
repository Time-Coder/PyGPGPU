"""Regression tests for the divergences black-box probing found.

Each of these was found by writing the code a user writes and *running* it on both the
interpreter and a device, then comparing against plain python. They share one shape: the
two backends answered differently, or the device answered a number python never computes.

What is deliberately **not** here (design choices, not bugs -- raised with the maintainer):

* a `raise` inside a body loses its exception type on a device (there is no way to carry a
  python exception through the driver's launch status);
* `assert` is refused by the translator while the interpreter executes it;
* an f-string `:format` spec is refused (needs python's formatting machinery on the device);
* int32 overflow inside a body wraps on the device and does not in the interpreter -- the
  width is part of the `int=` contract, and the body's own arithmetic is the caller's.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pygpgpu import get_global_id, kernel

# ---------------------------------------------------------------------------
# `and` / `or` used as a VALUE, not as a condition
# ---------------------------------------------------------------------------


@kernel(domain="out")
def and_as_value(out: np.ndarray, x: np.ndarray):
    i = get_global_id(0)
    out[i] = x[0] and x[1]


@kernel(domain="out")
def or_as_value(out: np.ndarray, x: np.ndarray):
    i = get_global_id(0)
    out[i] = x[0] or x[1]


@kernel(domain="out")
def and_as_value_scaled(out: np.ndarray, x: np.ndarray):
    i = get_global_id(0)
    out[i] = (x[0] and x[1]) * 10.0


def test_and_answers_the_operand_not_a_bool(gpu_backend):
    """`2.0 and 3.0` is 3.0 in python; `&&` answered 1.0.

    The condition position was always right, so only a body that *uses* the result as a
    number showed it -- `(a and b) * 10.0` computed 10.0 where python computes 30.0.
    """
    out = np.zeros(1, dtype=np.float32)
    and_as_value[gpu_backend](out, np.array([2.0, 3.0], dtype=np.float32))
    assert float(out[0]) == pytest.approx(2.0 and 3.0)


def test_or_answers_the_operand_not_a_bool(gpu_backend):
    """`-5.0 or 3.0` is -5.0 in python (the first operand is truthy)."""
    out = np.zeros(1, dtype=np.float32)
    or_as_value[gpu_backend](out, np.array([-5.0, 3.0], dtype=np.float32))
    assert float(out[0]) == pytest.approx(-5.0 or 3.0)


def test_a_boolean_value_feeds_the_arithmetic_that_follows(gpu_backend):
    """The whole point: the operand keeps its numeric meaning in the next expression."""
    out = np.zeros(1, dtype=np.float32)
    and_as_value_scaled[gpu_backend](out, np.array([2.0, 3.0], dtype=np.float32))
    assert float(out[0]) == pytest.approx((2.0 and 3.0) * 10.0)


def test_a_false_left_operand_short_circuits_in_value_position(gpu_backend):
    """`0.0 and 9.0` is 0.0 -- the left operand is returned, not the right one."""
    out = np.zeros(1, dtype=np.float32)
    and_as_value[gpu_backend](out, np.array([0.0, 9.0], dtype=np.float32))
    assert float(out[0]) == pytest.approx(0.0)


def test_a_truthy_left_operand_short_circuits_or(gpu_backend):
    """`4.0 or 9.0` is 4.0 -- `or` returns the first operand when it is truthy."""
    out = np.zeros(1, dtype=np.float32)
    or_as_value[gpu_backend](out, np.array([4.0, 9.0], dtype=np.float32))
    assert float(out[0]) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# min / max against a NaN operand
# ---------------------------------------------------------------------------


@kernel(domain="out")
def builtin_min(out: np.ndarray, x: np.ndarray, y: np.ndarray):
    i = get_global_id(0)
    out[i] = min(x[0], y[0])


@kernel(domain="out")
def builtin_max(out: np.ndarray, x: np.ndarray, y: np.ndarray):
    i = get_global_id(0)
    out[i] = max(x[0], y[0])


def _nan_case(kernel_object, gpu_backend, first: float, second: float) -> float:
    out = np.zeros(1, dtype=np.float32)
    a = np.array([first], dtype=np.float32)
    b = np.array([second], dtype=np.float32)
    kernel_object[gpu_backend](out, a, b)
    return float(out[0])


@pytest.mark.parametrize("name, candidate", [
    ("min", (math.nan, 1.0)), ("min", (1.0, math.nan)),
    ("max", (math.nan, 1.0)), ("max", (1.0, math.nan)),
])
def test_min_and_max_follow_pythons_operand_order(gpu_backend, name, candidate):
    """python answers the *second* operand when it compares strictly better, else the first.

    `fmin`/`fmax` answer the operand that is not NaN instead, which is a different rule: it
    turned `min(nan, 1.0)` into 1.0 where python answers nan, and `min(1.0, nan)` into nan
    where python answers 1.0. The two families are both wanted, so `min`/`max` emit the
    comparison and `fmin`/`fmax` keep theirs.
    """
    first, second = candidate
    expected = (min if name == "min" else max)(first, second)
    actual = _nan_case(builtin_min if name == "min" else builtin_max,
                       gpu_backend, first, second)
    if math.isnan(expected):
        assert math.isnan(actual), f"{name}({first}, {second}) answered {actual}"
    else:
        assert actual == pytest.approx(expected)


def test_min_of_two_plain_numbers_still_works(gpu_backend):
    """The control: no NaN involved, both orders answer python's number."""
    assert _nan_case(builtin_min, gpu_backend, 2.0, 1.0) == pytest.approx(1.0)
    assert _nan_case(builtin_min, gpu_backend, 1.0, 2.0) == pytest.approx(1.0)
    assert _nan_case(builtin_max, gpu_backend, 2.0, 1.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# A negative range() step
# ---------------------------------------------------------------------------


@kernel(domain="out")
def count_down(out: np.ndarray):
    i = get_global_id(0)
    total = 0
    for j in range(5, 0, -1):
        total = total * 10 + j
    out[i] = total


def test_a_negative_range_step_is_a_constant_integer(gpu_backend):
    """`range(5, 0, -1)` is a constant step, so refusing it contradicted the message.

    The step arrives as translated text and a negative literal is wrapped in the
    parentheses the unary-minus visitor emits (`(-1)`), so the `-?\\d+` test missed it and
    raised "step must be a constant integer" -- while the positive `2` worked.
    """
    out = np.zeros(1, dtype=np.float32)
    count_down[gpu_backend](out)
    expected = 0
    for j in range(5, 0, -1):
        expected = expected * 10 + j
    assert float(out[0]) == pytest.approx(expected)
