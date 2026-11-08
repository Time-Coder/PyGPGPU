"""The creation tests, split out of `test_kernel_complex.py`.

The subject is creation.
"""
from __future__ import annotations

from _helpers.test_kernel_complex_helpers import (
    constructed_from_args,
    constructed_from_complex,
    constructed_from_constants,
    constructed_from_real,
    constructed_from_string,
    constructed_then_narrowed,
    created_constant,
    created_in_loop,
    values,
)

import numpy as np
import pytest


def test_a_complex_created_above_the_loop(runner):
    """`3 + 4j` parses as `3 + (4j)`, so it goes through the arithmetic, not a constant."""
    out = np.zeros(2, np.complex128)
    runner(created_constant, out)
    np.testing.assert_allclose(out, np.array([3 + 4j, 3 + 4j], np.complex128))


def test_a_complex_local_inside_the_loop(runner):
    a = values(np.complex64)
    out = np.zeros(2, np.complex64)
    runner(created_in_loop, a, out)
    np.testing.assert_allclose(out, a * a, rtol=1e-6)


def test_complex_of_two_arguments(runner):
    """`complex(re, im)` is python's own spelling, and the result is the mapped width."""
    out = np.zeros(2, dtype=np.complex128)
    runner(constructed_from_args, np.float32(3.0), np.float32(4.0), out)
    np.testing.assert_allclose(out, np.array([3 + 4j, 3 + 4j], dtype=np.complex128))


def test_complex_of_two_constants(runner):
    out = np.zeros(2, dtype=np.complex128)
    runner(constructed_from_constants, out)
    np.testing.assert_allclose(out, np.array([3 + 4j, 3 + 4j], dtype=np.complex128))


def test_complex_of_a_complex_value_is_a_copy_at_the_mapped_width(runner):
    """`complex(z)` of a complex64 is widened under the default mapping before the store."""
    out = np.zeros(2, dtype=np.complex128)
    runner(constructed_from_complex, np.array([1 + 2j, 1 + 2j], np.complex64), out)
    np.testing.assert_allclose(out, np.array([1 + 2j, 1 + 2j], dtype=np.complex128))


def test_complex_of_a_real_has_a_zero_imaginary_part(runner):
    out = np.zeros(2, dtype=np.complex128)
    runner(constructed_from_real, np.float32(5.0), out)
    np.testing.assert_allclose(out, np.array([5 + 0j, 5 + 0j], dtype=np.complex128))


def test_a_constructed_value_narrows_through_numpy_cast(runner):
    out = np.zeros(2, dtype=np.complex64)
    runner(constructed_then_narrowed, out)
    np.testing.assert_allclose(out, np.array([3 + 4j, 3 + 4j], dtype=np.complex64))


def test_complex_of_a_string_is_refused(gpu_backend, runner):
    """A string has no device value, and python would parse it on the host."""
    out = np.zeros(2, dtype=np.complex128)
    with pytest.raises(SyntaxError) as caught:
        runner(constructed_from_string, out)
    assert "complex(" in str(caught.value)
