"""The widths tests, split out of `test_kernel_complex.py`.

The subject is widths.
"""
from __future__ import annotations

from _helpers.test_kernel_complex import (
    DOUBLE,
    FLOAT,
    _generated_source,
    created_from_element,
    created_from_real,
    double_complex_argument,
    passed_scalar,
    python_constructed,
    python_literal,
    scalar_scale,
    values,
)

import numpy as np
import pytest
from pygpgpu.kernel import complex as complex_support


def test_the_two_widths_are_the_ones_numpy_has():
    assert complex_support.complex_type_name(np.dtype("complex64")) == "c64"
    assert complex_support.complex_type_name(np.dtype("complex128")) == "c128"
    assert complex_support.needs(np.dtype("complex64")) == "COMPLEX_NEED_FLOAT"
    assert complex_support.needs(np.dtype("complex128")) == "COMPLEX_NEED_DOUBLE"
    # A non-complex dtype is not one, so a caller's own two-float structure is unaffected.
    assert complex_support.complex_type_name(np.dtype("float32")) is None
    assert complex_support.needs(np.dtype([("re", "<f4"), ("im", "<f4")])) is None


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_a_complex_scalar_argument_carries_both_parts(runner, dtype):
    """A scalar argument travels by value, and the imaginary part must survive the trip."""
    out = np.zeros(2, dtype=dtype)
    runner(scalar_scale, dtype(1.0 + 2.0j), out)
    np.testing.assert_allclose(out, np.array([3.0 + 6.0j, 3.0 + 6.0j], dtype=dtype))


def test_a_python_complex_argument_is_a_c64_under_the_default_mapping(gpu_backend):
    """The device declares the 32-bit pair, so the value arrives narrowed, as a float does."""
    out = np.zeros(2, np.complex128)
    passed_scalar[gpu_backend](0.1 + 0.2j, out)
    # A `c128` parameter would have carried 0.1 + 0.2j exactly; a `c64` carries the pair
    # `np.complex64` rounds to, and the double array holds that without changing it again.
    assert out[0] == complex(np.complex64(0.1 + 0.2j))
    assert out[0] != 0.1 + 0.2j


def test_a_double_mapping_carries_a_python_complex_at_128_bits(gpu_backend):
    """`float=np.float64` is what lets a python complex hold a value float32 cannot."""
    out = np.zeros(2, np.complex128)
    double_complex_argument[gpu_backend](1e40 + 1e40j, out)
    # A `c64` parameter would have answered infinity: float32 stops at ~3.4e38.
    assert out[0] == 1e40 + 1e40j


@pytest.mark.parametrize("type_map, kind", [
    (None, "c64"),
    ({"float": "float"}, "c64"),
    ({"float": "double"}, "c128"),
], ids=["no-mapping", "float", "double"])
def test_a_python_complex_literal_is_the_mapped_width(type_map, kind):
    """`3 + 4j` is built at the width the python float beside it travels as."""
    source = _generated_source(python_literal, type_map)
    assert f"{kind}_make(" in source
    other = "c128" if kind == "c64" else "c64"
    assert f"{other}_make(" not in source


@pytest.mark.parametrize("type_map, kind", [
    (None, "c64"),
    ({"float": "float"}, "c64"),
    ({"float": "double"}, "c128"),
], ids=["no-mapping", "float", "double"])
def test_the_complex_constructor_is_the_mapped_width(type_map, kind):
    """`complex(re, im)` answers the same width a python complex travels as."""
    source = _generated_source(python_constructed, type_map)
    assert f"{kind}_make(" in source
    other = "c128" if kind == "c64" else "c64"
    assert f"{other}_make(" not in source


def test_a_casted_complex_keeps_both_parts(runner):
    a = values(np.complex128)
    out = np.zeros(2, np.complex64)
    runner(created_from_element, a, out)
    np.testing.assert_allclose(out, a.astype(np.complex64), rtol=1e-6)


def test_a_real_cast_to_complex_has_a_zero_imaginary_part(runner):
    a = np.array([1.0, 3.0], np.float32)
    out = np.zeros(2, np.complex64)
    runner(created_from_real, a, out)
    np.testing.assert_allclose(out, (a * 2.0).astype(np.complex64))
