"""The returns tests, split out of `test_kernel_complex.py`.

The subject is returns.
"""
from __future__ import annotations

from _helpers.test_kernel_complex import (
    make_scope_rebinder,
    multiply_by_scope,
    passed_scalar,
    read_scope_complex128,
    read_scope_complex64,
    read_scope_python,
    return_scope,
    return_scope_complex64,
    returned_computed,
    returned_constant,
    returned_element,
)

import numpy as np
import pytest


@pytest.mark.parametrize("value", [np.complex64(1 + 2j), np.complex128(1 + 2j), 1 + 2j],
                         ids=["complex64", "complex128", "python"])
def test_a_complex_scalar_argument(runner, value):
    """A python complex is a `c64` under the default mapping, so the value arrives narrowed."""
    out = np.zeros(2, np.complex128)
    runner(passed_scalar, value, out)
    np.testing.assert_allclose(out, np.array([value, value], np.complex128))


def test_a_complex_element_is_returned(runner):
    result = runner(returned_element, np.array([1 + 2j], np.complex64))
    assert result == pytest.approx(1 + 2j)


def test_a_computed_complex_is_returned(runner):
    result = runner(returned_computed, np.array([1 + 2j], np.complex64))
    assert result == pytest.approx((1 + 2j) ** 2, rel=1e-6)


def test_a_complex_constant_is_returned(runner):
    """A returned literal crosses the loop never, so the slot answers it directly."""
    result = runner(returned_constant, np.zeros(1, np.complex64))
    assert result == pytest.approx(3 + 4j)


@pytest.mark.parametrize("function, dtype", [
    (read_scope_python, np.complex128),
    (read_scope_complex64, np.complex64),
    (read_scope_complex128, np.complex128),
], ids=["python", "complex64", "complex128"])
def test_an_outer_complex_scalar_is_read(runner, function, dtype):
    out = np.zeros(2, dtype=dtype)
    runner(function, out)
    np.testing.assert_allclose(out, np.array([3 + 4j, 3 + 4j], dtype=dtype))


def test_an_outer_complex_scalar_takes_part_in_arithmetic(runner):
    a = np.array([1 + 1j, 2 + 2j], np.complex64)
    out = np.zeros(2, np.complex128)
    runner(multiply_by_scope, a, out)
    np.testing.assert_allclose(out, a * (3 + 4j), rtol=1e-6)


def test_an_outer_complex_scalar_is_returned(runner):
    assert runner(return_scope, np.zeros(1, np.complex64)) == pytest.approx(3 + 4j)
    assert runner(return_scope_complex64, np.zeros(1, np.complex64)) == pytest.approx(3 + 4j)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128],
                         ids=["complex64", "complex128"])
def test_an_outer_complex_scalar_can_be_rebound(runner, dtype):
    function, peek = make_scope_rebinder(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(function, dtype(5 + 6j), out)
    np.testing.assert_allclose(out, np.array([5 + 6j, 5 + 6j], dtype=dtype))
    # The `nonlocal` slot is copied back into the scope after the launch, so the closure
    # holds the value the kernel put there -- for a complex value just as for an int.
    assert complex(peek()) == pytest.approx(5 + 6j)
