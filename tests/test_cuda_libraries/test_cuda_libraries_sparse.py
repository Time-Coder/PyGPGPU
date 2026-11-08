"""The sparse tests, split out of `test_cuda_libraries.py`.

The subject is sparse.
"""
from __future__ import annotations

from _helpers.test_cuda_libraries_helpers import (
    _require_library,
)

import ctypes
import numpy as np


def test_cusparse_csrmv_matches_the_dense_product(device):
    """A sparse matrix-vector multiply agrees with the dense product."""
    _require_library("cusparse")
    from pygpgpu.cuda.libraries.wrappers import csrmv

    # The diagonal matrix diag(1, 2, 3): three rows, so four row offsets.
    row_offsets = np.array([0, 1, 2, 3], np.int32)
    column_indices = np.array([0, 1, 2], np.int32)
    values = np.array([1.0, 2.0, 3.0], np.float32)
    vector = np.array([1.0, 1.0, 1.0], np.float32)

    result = csrmv(row_offsets, column_indices, values, vector)
    np.testing.assert_allclose(result, values * vector, rtol=1e-5, atol=1e-6)

def test_cusparse_csrmv_on_a_real_matrix(device):
    """A matrix with more than one entry per row is multiplied correctly."""
    _require_library("cusparse")
    from pygpgpu.cuda.libraries.wrappers import csrmv

    # The matrix [[1, 2], [0, 3]] in CSR form.
    row_offsets = np.array([0, 2, 3], np.int32)
    column_indices = np.array([0, 1, 1], np.int32)
    values = np.array([1.0, 2.0, 3.0], np.float32)
    vector = np.array([4.0, 5.0], np.float32)

    result = csrmv(row_offsets, column_indices, values, vector)
    np.testing.assert_allclose(result, np.array([14.0, 15.0]), rtol=1e-5, atol=1e-6)

def test_cusparse_handle_lifecycle(device):
    """A handle is created and destroyed through the bound entry points."""
    _require_library("cusparse")
    from pygpgpu.cuda.libraries.cusparse import CUSPARSE

    handle = ctypes.c_void_p()
    CUSPARSE.cusparseCreate(ctypes.byref(handle))
    assert handle.value
    CUSPARSE.cusparseDestroy(handle)
