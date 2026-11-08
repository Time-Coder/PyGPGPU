"""The runtime tests, split out of `test_cuda_libraries.py`.

The subject is runtime.
"""
from __future__ import annotations

from _helpers.test_cuda_libraries_helpers import (
    _require_library,
)

import ctypes
import numpy as np


def test_cudart_reports_the_device(device):
    """The runtime's own view of the device agrees with the driver's."""
    _require_library("cudart")
    from pygpgpu.cuda.libraries.cudart import CUDART

    count = ctypes.c_int(0)
    CUDART.cudaGetDeviceCount(ctypes.byref(count))
    assert count.value >= 1

    ordinal = ctypes.c_int(-1)
    CUDART.cudaGetDevice(ctypes.byref(ordinal))
    assert 0 <= ordinal.value < count.value

    version = ctypes.c_int(0)
    CUDART.cudaRuntimeGetVersion(ctypes.byref(version))
    assert version.value > 0

def test_cudart_allocates_and_copies(device):
    """The runtime's allocation and copy work."""
    _require_library("cudart")
    from pygpgpu.cuda.libraries.cudart import CUDART, cudaMemcpyKind

    host = np.arange(16, dtype=np.float32)

    pointer = ctypes.c_void_p()
    CUDART.cudaMalloc(ctypes.byref(pointer), host.nbytes)
    try:
        CUDART.cudaMemcpy(pointer, host.ctypes.data, host.nbytes,
                          int(cudaMemcpyKind.cudaMemcpyHostToDevice))

        back = np.zeros(16, np.float32)
        CUDART.cudaMemcpy(back.ctypes.data, pointer, back.nbytes,
                          int(cudaMemcpyKind.cudaMemcpyDeviceToHost))
        np.testing.assert_array_equal(back, host)
    finally:
        CUDART.cudaFree(pointer)
