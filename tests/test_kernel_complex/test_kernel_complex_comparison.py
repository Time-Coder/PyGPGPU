"""The comparison tests, split out of `test_kernel_complex.py`.

The subject is comparison.
"""
from __future__ import annotations

from _helpers.test_kernel_complex_helpers import (
    DOUBLE,
    FLOAT,
    equality,
    equality_with_real,
    less_than,
    mixed_add,
    values,
)

import numpy as np
import pytest


def test_a_float_and_a_double_value_answer_a_double(runner):
    """The double operand widens the whole operation, as the quaternions do."""
    single = values(FLOAT)
    double = values(DOUBLE)
    out = np.zeros(2, dtype=DOUBLE)
    runner(mixed_add, single, double, out)
    np.testing.assert_allclose(out, single.astype(DOUBLE) + double, rtol=1e-6)


def test_a_complex_value_compares_with_a_real_one(runner):
    """A real operand is widened, exactly as the arithmetic widens it."""
    a = np.array([2.0 + 0.0j, 1.0 + 1.0j], dtype=FLOAT)
    out = np.zeros(2, dtype=np.float32)
    runner(equality_with_real, a, np.float32(2.0), out)
    np.testing.assert_array_equal(out, np.array([1.0, 0.0], dtype=np.float32))


def test_equality_compares_both_parts(runner):
    a = np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=FLOAT)
    b = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=FLOAT)
    out = np.zeros(2, dtype=np.float32)
    runner(equality, a, b, out)
    np.testing.assert_array_equal(out, np.array([1.0, 0.0], dtype=np.float32))


def test_ordering_a_complex_value_is_refused(gpu_backend, runner):
    """Python orders no complex number, so the device refuses the comparison by name."""
    a = values(FLOAT)
    with pytest.raises(SyntaxError) as caught:
        runner(less_than, a, a, np.zeros(2, dtype=np.float32))
    message = str(caught.value)
    assert "complex" in message
    assert "==" in message
