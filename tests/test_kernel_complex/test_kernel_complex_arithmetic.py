"""The arithmetic tests, split out of `test_kernel_complex.py`.

The subject is arithmetic.
"""
from __future__ import annotations

from _helpers import test_kernel_complex_helpers as _HELPERS
from _helpers.test_kernel_complex_helpers import (
    DOUBLE,
    FLOAT,
    FLOAT_SCALAR_FORMS,
    add,
    cube,
    divide,
    identity,
    multiply,
    negate,
    other,
    scale,
    scale_div,
    square_by_operator,
    subtract,
    values,
)

import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_addition_matches_numpy(runner, dtype):
    a, b = values(dtype), other(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(add, a, b, out)
    np.testing.assert_allclose(out, a + b)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_subtraction_matches_numpy(runner, dtype):
    a, b = values(dtype), other(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(subtract, a, b, out)
    np.testing.assert_allclose(out, a - b)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_multiplication_is_the_full_product(runner, dtype):
    """`z * w` is the full product, not a per-component scale."""
    a, b = values(dtype), other(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(multiply, a, b, out)
    np.testing.assert_allclose(out, a * b, rtol=1e-6)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_division_matches_numpy(runner, dtype):
    a, b = values(dtype), other(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(divide, a, b, out)
    np.testing.assert_allclose(out, a / b, rtol=1e-6)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_a_real_scalar_scales_and_divides(runner, dtype):
    a = values(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(scale, a, np.float32(2.0), out)
    np.testing.assert_allclose(out, a * 2.0, rtol=1e-6)
    runner(scale_div, a, np.float32(2.0), out)
    np.testing.assert_allclose(out, a / 2.0, rtol=1e-6)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
@pytest.mark.parametrize("name, expected", FLOAT_SCALAR_FORMS,
                         ids=[name for name, _ in FLOAT_SCALAR_FORMS])
def test_a_float_literal_beside_a_complex_value(runner, dtype, name, expected):
    """A python float is a double, so these widen to c128 at either array width.

    The widening is what makes them worth their own test: the emitted call has to be the
    header's `c128_scale` / `c128_rsub` / ..., and not a `_c128`-suffixed name -- that
    suffix marks an *operand* of that width (`c64_scale_c128(c64, double)`), and a c128
    operand has no such helper. Every form here used to fail to compile with
    "identifier c128_scale_c128 is undefined", because the name was built from the
    already-widened result width. The kernels sit with the others at the bottom of the
    file, so the form is named here and looked up when the test runs.
    """
    a = values(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(getattr(_HELPERS, name), a, out)
    np.testing.assert_allclose(out, np.asarray(expected(a), dtype=dtype), rtol=1e-6)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_negation_and_the_unary_plus(runner, dtype):
    a = values(dtype)
    negative = np.zeros(2, dtype=dtype)
    positive = np.zeros(2, dtype=dtype)
    runner(negate, a, negative)
    runner(identity, a, positive)
    np.testing.assert_allclose(negative, -a)
    np.testing.assert_allclose(positive, a)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_powers(runner, dtype):
    a = values(dtype)
    squared = np.zeros(2, dtype=dtype)
    runner(square_by_operator, a, squared)
    np.testing.assert_allclose(squared, a ** 2, rtol=1e-6)
    cubed = np.zeros(2, dtype=dtype)
    runner(cube, a, cubed)
    np.testing.assert_allclose(cubed, a ** 3, rtol=1e-6)
