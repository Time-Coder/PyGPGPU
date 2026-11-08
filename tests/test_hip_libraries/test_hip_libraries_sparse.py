"""The sparse tests, split out of `test_hip_libraries.py`.

The subject is sparse.
"""
from __future__ import annotations

from _helpers.test_hip_libraries_helpers import (
    _require_library,
)

import ctypes
import pytest
from pygpgpu.hip.libraries import (
    HIPSPARSE,
)


def test_hipsparse_creates_a_handle_or_says_why_it_cannot(hip_device):
    """rocSPARSE resolves its kernels out of a runtime directory beside the DLL.

    ROCm 7.2's Windows install creates one for rocBLAS (`bin/rocblas/library`) and for
    rocFFT, and none for rocSPARSE -- so `hipsparseCreate` answers
    `HIPSPARSE_STATUS_INTERNAL_ERROR` and leaves a null handle. The binding is verified;
    the library cannot run here.
    """
    _require_library("hipsparse")
    from pygpgpu.libraries import LibraryError
    from pygpgpu.hip.libraries.wrappers import ensure_context

    ensure_context()
    handle = ctypes.c_void_p()
    try:
        status = HIPSPARSE.hipsparseCreate(ctypes.byref(handle))
    except LibraryError as error:
        assert "hipSPARSE" in str(error), error
        pytest.skip(f"rocSPARSE has no kernels for this device: {error}")
        return
    try:
        assert status == 0
        assert handle.value, "hipsparseCreate reported success but returned no handle"
    finally:
        HIPSPARSE.hipsparseDestroy(handle)
