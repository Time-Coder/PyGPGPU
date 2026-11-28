"""The blas tests, split out of `test_hip_libraries.py`.

The subject is blas.
"""
from __future__ import annotations

from _helpers.test_hip_libraries_helpers import (
    _require_library,
)

import numpy as np
import pytest


def test_gemm_matches_numpy(hipblas):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    np.testing.assert_allclose(hipblas.gemm(a, b), a @ b, rtol=1e-6)


def test_gemm_in_double_precision(hipblas):
    a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    b = np.array([[5.5, 6.5], [7.5, 8.5]], dtype=np.float64)
    np.testing.assert_allclose(hipblas.gemm(a, b), a @ b, rtol=1e-12)


def test_gemm_follows_numpys_one_dimensional_rules(hipblas):
    """A 1-D operand is a row on the left and a column on the right, as `@` says."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    vector = np.array([1.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(hipblas.gemm(matrix, vector), matrix @ vector, rtol=1e-6)
    np.testing.assert_allclose(hipblas.gemm(vector, matrix), vector @ matrix, rtol=1e-6)


def test_gemm_honours_the_transpose_flags(hipblas):
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(hipblas.gemm(a, b, transpose_b=True), a @ b.T, rtol=1e-6)
    np.testing.assert_allclose(hipblas.gemm(a, b, transpose_a=True), a.T @ b, rtol=1e-6)


def test_gemm_honours_alpha_and_beta(hipblas):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.eye(2, dtype=np.float32)
    start = np.full((2, 2), 10.0, dtype=np.float32)
    expected = 2.0 * (a @ b) + 3.0 * start
    np.testing.assert_allclose(
        hipblas.gemm(a, b, out=start.copy(), alpha=2.0, beta=3.0), expected, rtol=1e-6)


def test_axpy_matches_the_definition(hipblas):
    """`axpy` writes into `y`, so the expectation is computed before the call."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    expected = 2.0 * x + y
    np.testing.assert_allclose(hipblas.axpy(2.0, x, y), expected, rtol=1e-6)


def test_scal_matches_the_definition(hipblas):
    """`scal` writes into `x`, so the expectation is computed before the call."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    expected = 2.0 * x
    np.testing.assert_allclose(hipblas.scal(2.0, x), expected, rtol=1e-6)


def test_dot_matches_the_definition(hipblas):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    assert hipblas.dot(x, y) == pytest.approx(float(x @ y), rel=1e-6)


def test_a_bad_dtype_is_refused(hipblas):
    """The library takes float32 and float64; an int array is refused by name."""
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    with pytest.raises(TypeError) as caught:
        hipblas.gemm(a, a)
    assert "float32" in str(caught.value)


def test_shapes_that_do_not_multiply_are_refused(hipblas):
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError) as caught:
        hipblas.gemm(a, b)
    assert "do not multiply" in str(caught.value)


def test_lt_gemm_matches_gemm_or_says_why_it_cannot(hipblas):
    """hipBLASLt is bound, but rocBLASLt needs a kernel library for the device.

    ROCm ships its Tensile kernels for the discrete GPUs only, so on an integrated one
    every matmul fails inside rocBLASLt while the binding itself is fine. The test
    accepts that, because it is a fact about the machine and not about the code -- but
    when the call does work its answer has to be gemm's.
    """
    _require_library("hipblaslt")
    from pygpgpu.libraries import LibraryError

    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    try:
        result = hipblas.lt_gemm(a, b)
    except LibraryError as error:
        assert "Tensile" in str(error), (
            f"hipBLASLt failed for a reason other than its kernel library: {error}")
        pytest.skip("rocBLASLt has no kernel library for this device")
        return
    np.testing.assert_allclose(result, a @ b, rtol=1e-5)

def test_gemm_refuses_mixed_dtypes(hipblas):
    """One dtype for both operands, with the reason named."""
    a = np.ones((2, 2), np.float32)
    b = np.ones((2, 2), np.float64)
    with pytest.raises(TypeError, match="one dtype"):
        hipblas.gemm(a, b)


def test_gemm_out_casts_the_dtype(hipblas):
    """An `out=` of another dtype receives the product, the way numpy's does.

    Regression: `ascontiguousarray(out)` kept the caller's dtype, so BLAS wrote
    float32 bytes into a float64-sized buffer and the caller's array came back
    wrong -- silently.
    """
    a = np.ones((2, 3), np.float32)
    b = np.ones((3, 2), np.float32)
    out = np.zeros((2, 2), np.float64)

    result = hipblas.gemm(a, b, out=out)
    assert result is out
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-6)


def test_gemm_out_follows_a_strided_view(hipblas):
    """A non-contiguous `out=` is written at its strided positions, not scattered."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(6, dtype=np.float32).reshape(3, 2)
    holder = np.zeros((2, 4), np.float32)
    out = holder[:, :2]          # shape (2, 2), strides make it non-contiguous

    result = hipblas.gemm(a, b, out=out)
    assert result is out
    np.testing.assert_allclose(holder[:, :2], a @ b, rtol=1e-5, atol=1e-6)
    assert np.all(holder[:, 2:] == 0.0), "the store spilled past the view"
