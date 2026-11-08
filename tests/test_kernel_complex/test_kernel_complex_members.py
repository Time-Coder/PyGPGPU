"""The members tests, split out of `test_kernel_complex.py`.

The subject is members.
"""
from __future__ import annotations

from _helpers.test_kernel_complex_helpers import (
    COMPLEX_FUNCTIONS,
    DOUBLE,
    FLOAT,
    conjugate_call,
    conjugate_method,
    magnitude,
    phase,
    read_members,
    values,
)

import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_the_real_and_imaginary_members(runner, dtype):
    a = values(dtype)
    out = np.zeros((2, 2), dtype=np.float64 if dtype is DOUBLE else np.float32)
    runner(read_members, a, out)
    np.testing.assert_allclose(out[:, 0], a.real)
    np.testing.assert_allclose(out[:, 1], a.imag)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_the_conjugate_method(runner, dtype):
    a = values(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(conjugate_method, a, out)
    np.testing.assert_allclose(out, np.conjugate(a))


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_the_free_conjugate_function(runner, dtype):
    """`conjugate(z)` is the same operation under its module-level spelling."""
    a = values(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(conjugate_call, a, out)
    np.testing.assert_allclose(out, np.conjugate(a))


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
@pytest.mark.parametrize("name", sorted((
    "sqrt", "exp", "log", "log2", "log10", "sin", "cos", "tan",
    "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "square", "reciprocal", "exp2",
)))
def test_each_complex_function_matches_numpy(runner, dtype, name):
    a = values(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(COMPLEX_FUNCTIONS[name], a, out)
    expected = getattr(np, name)(a)
    np.testing.assert_allclose(out, expected,
                               rtol=1e-6 if dtype is FLOAT else 1e-12,
                               atol=1e-6 if dtype is FLOAT else 1e-12)


def test_the_magnitude_is_a_real(runner):
    """`abs(complex)` is the modulus, a real -- numpy's own rule."""
    a = values(DOUBLE)
    out = np.zeros(2, dtype=np.float64)
    runner(magnitude, a, out)
    np.testing.assert_allclose(out, np.abs(a), rtol=1e-12)


def test_the_phase_is_a_real(runner):
    a = values(DOUBLE)
    out = np.zeros(2, dtype=np.float64)
    runner(phase, a, out)
    np.testing.assert_allclose(out, np.angle(a), rtol=1e-12)
