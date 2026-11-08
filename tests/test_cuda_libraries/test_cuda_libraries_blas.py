"""The blas tests, split out of `test_cuda_libraries.py`.

The subject is blas.
"""
from __future__ import annotations

from _helpers.test_cuda_libraries_helpers import (
    _require_library,
)

import ctypes
import numpy as np
import pytest
from pygpgpu.cuda.libraries._base import LibraryError


def test_the_error_table_is_the_librarys_own():
    """Each library maps its own codes; none shares the driver's numbering."""
    from pygpgpu.cuda.driver.cuinfo import CUInfo
    from pygpgpu.cuda.libraries.cublas import CUBLASInfo
    from pygpgpu.cuda.libraries.cufft import CUFFTInfo
    from pygpgpu.cuda.libraries.cusparse import CUSPARSEInfo

    # The same number means different things in different libraries: cuBLAS's 7 is
    # "invalid value" and the driver's 7 is "mapping error", so the tables must be
    # separate rather than shared.
    assert CUBLASInfo.error_codes[7] != CUInfo.error_codes[7]
    assert CUFFTInfo.error_codes[3] != CUInfo.error_codes[3]

    # Every library's own "success" is 0, and every table carries a message for it.
    for table in (CUBLASInfo.error_codes, CUFFTInfo.error_codes,
                  CUSPARSEInfo.error_codes):
        assert table[0]

    # A code the driver does not have at all is still mapped by its own library.
    assert 10 in CUSPARSEInfo.error_codes
    assert 10 not in CUInfo.error_codes

def test_cublas_gemm_matches_numpy(device):
    """A float32 and a float64 gemm both agree with numpy."""
    _require_library("cublas")
    from pygpgpu.cuda.libraries.wrappers import gemm

    rng = np.random.default_rng(7)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((4, 2)).astype(np.float32)

    result = gemm(a, b)
    assert result.dtype == np.float32
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result, a @ b, rtol=1e-5, atol=1e-6)

    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    result64 = gemm(a64, b64)
    assert result64.dtype == np.float64
    np.testing.assert_allclose(result64, a64 @ b64, rtol=1e-12, atol=1e-12)

def test_cublas_gemm_writes_into_the_output(device):
    """The output argument is filled in and returned."""
    _require_library("cublas")
    from pygpgpu.cuda.libraries.wrappers import gemm

    rng = np.random.default_rng(11)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((3, 2)).astype(np.float32)
    out = np.zeros((2, 2), np.float32)

    result = gemm(a, b, out=out)
    assert result is out
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-6)

def test_cublas_gemm_follows_numpy_shape_rules(device):
    """A 1-D operand follows numpy's rules for `@`."""
    _require_library("cublas")
    from pygpgpu.cuda.libraries.wrappers import gemm

    rng = np.random.default_rng(13)
    vector = rng.standard_normal(4).astype(np.float32)
    other = rng.standard_normal(4).astype(np.float32)

    # Two 1-D arrays make a scalar.
    product = gemm(vector, other)
    assert product.shape == ()
    np.testing.assert_allclose(float(product), float(vector @ other), rtol=1e-5, atol=1e-6)

    # A matrix and a 1-D array make a 1-D array.
    matrix = rng.standard_normal((3, 4)).astype(np.float32)
    result = gemm(matrix, vector)
    assert result.shape == (3,)
    np.testing.assert_allclose(result, matrix @ vector, rtol=1e-5, atol=1e-6)

def test_cublas_gemm_refuses_a_mismatched_shape(device):
    """Two matrices that do not multiply are refused with a message naming the shapes."""
    _require_library("cublas")
    from pygpgpu.cuda.libraries.wrappers import gemm

    a = np.zeros((2, 3), np.float32)
    b = np.zeros((4, 5), np.float32)
    with pytest.raises(ValueError, match="cannot multiply"):
        gemm(a, b)

def test_cublas_axpy_scal_and_dot_match_numpy(device):
    """The level-1 operations agree with numpy."""
    _require_library("cublas")
    from pygpgpu.cuda.libraries.wrappers import axpy, dot, scal

    rng = np.random.default_rng(17)
    x = rng.standard_normal(8).astype(np.float32)
    y = rng.standard_normal(8).astype(np.float32)

    original = y.copy()
    result = axpy(2.5, x, y)
    np.testing.assert_allclose(result, original + 2.5 * x, rtol=1e-5, atol=1e-6)

    original = x.copy()
    result = scal(3.0, x)
    np.testing.assert_allclose(result, 3.0 * original, rtol=1e-5, atol=1e-6)

    value = dot(x, y)
    np.testing.assert_allclose(value, float(x @ y), rtol=1e-5, atol=1e-6)

def test_cublas_handle_lifecycle(device):
    """A handle is created, used and destroyed."""
    _require_library("cublas")
    from pygpgpu.cuda.libraries.cublas import CUBLAS, cublasHandle_t
    from pygpgpu.cuda.libraries.wrappers import destroy_handle

    handle = cublasHandle_t()
    CUBLAS.cublasCreate_v2(ctypes.byref(handle))
    assert handle.value

    version = ctypes.c_int(0)
    CUBLAS.cublasGetVersion(ctypes.byref(version))
    assert version.value > 0

    destroy_handle(handle)

def test_a_failing_cublas_call_raises_the_librarys_error(device):
    """A refused call raises a RuntimeError naming the library and the code."""
    _require_library("cublas")
    from pygpgpu.cuda.libraries.cublas import CUBLAS, cublasHandle_t

    handle = cublasHandle_t()
    CUBLAS.cublasCreate_v2(ctypes.byref(handle))

    # A gemm whose leading dimension is smaller than the matrix it describes is
    # refused by the library.
    with pytest.raises(LibraryError) as raised:
        CUBLAS.cublasSgemm_v2(
            handle, 0, 0, 4, 4, 4,
            ctypes.byref(ctypes.c_float(1.0)),
            ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_float)), 1,
            ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_float)), 1,
            ctypes.byref(ctypes.c_float(0.0)),
            ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_float)), 1)

    message = str(raised.value)
    assert "cuBLAS" in message
    assert "cublasSgemm_v2" in message

def test_cublaslt_gemm_matches_numpy(device):
    """The newer cuBLAS path gives the same numbers as the classic one."""
    _require_library("cublaslt")
    from pygpgpu.cuda.libraries.wrappers import lt_gemm

    rng = np.random.default_rng(19)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((4, 2)).astype(np.float32)

    result = lt_gemm(a, b)
    np.testing.assert_allclose(result, a @ b, rtol=1e-5, atol=1e-6)
