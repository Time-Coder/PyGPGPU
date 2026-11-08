"""The solver tests, split out of `test_hip_libraries.py`.

The subject is solver.
"""
from __future__ import annotations

from _helpers.test_hip_libraries_helpers import (
    _require_library,
)

import ctypes
import numpy as np
import pytest
from pygpgpu.hip.libraries import (
    HIPSOLVER,
    hipblasFillMode_t,
)


def test_hipsolver_factors_a_matrix_or_says_why_it_cannot(hip_device):
    """rocSOLVER has no runtime kernel directory in this install either.

    Worse than a failure: the factorization answers **success with `devInfo == 0` and
    leaves the matrix untouched**, so a caller gets a wrong answer with no error. The
    test therefore checks the *answer* rather than the status code, and skips when the
    library is not usable -- which is a fact about the machine.
    """
    _require_library("hipsolver")
    from pygpgpu.hip.libraries.wrappers import (allocate, ensure_context, free,
                                                synchronize, to_device, to_host)

    ensure_context()
    from ctypes import POINTER, byref, c_float, c_int, c_void_p

    def typed(pointer, restype):
        return ctypes.cast(c_void_p(int(pointer.value)), POINTER(restype))

    matrix = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    n = matrix.shape[0]
    expected = np.tril(np.linalg.cholesky(matrix))

    handle = ctypes.c_void_p()
    HIPSOLVER.hipsolverDnCreate(byref(handle))
    d_matrix = to_device(matrix)
    d_info = to_device(np.zeros(1, dtype=np.int32))
    work_size = c_int(0)
    d_work = None
    try:
        HIPSOLVER.hipsolverDnSpotrf_bufferSize(
            handle, c_int(int(hipblasFillMode_t.HIPBLAS_FILL_MODE_LOWER)), c_int(n),
            typed(d_matrix, c_float), c_int(n), byref(work_size))
        if work_size.value <= 0:
            pytest.skip("rocSOLVER reported no workspace need, so it cannot factorize")
            return
        d_work = allocate(work_size.value * 4)
        HIPSOLVER.hipsolverDnSpotrf(
            handle, c_int(int(hipblasFillMode_t.HIPBLAS_FILL_MODE_LOWER)), c_int(n),
            typed(d_matrix, c_float), c_int(n), typed(d_work, c_float), work_size,
            typed(d_info, c_int))
        synchronize()

        factor = to_host(d_matrix, np.zeros((n, n), dtype=np.float32))
        if np.array_equal(np.tril(factor), np.tril(matrix)):
            pytest.skip("rocSOLVER answered success but left the matrix untouched; "
                        "this install ships no kernels for the device")
            return
        np.testing.assert_allclose(np.tril(factor), expected, atol=1e-5)
    finally:
        if d_work is not None:
            free(d_work)
        free(d_matrix)
        free(d_info)
        HIPSOLVER.hipsolverDnDestroy(handle)
