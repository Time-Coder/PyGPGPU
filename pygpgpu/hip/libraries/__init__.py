"""The host-side HIP math libraries, bound with ctypes.

hipBLAS, hipBLASLt, hipFFT, hipSPARSE and hipSOLVER are the vendor's ports of the CUDA
libraries. How much each one shares with its CUDA counterpart is a fact about that
library, not a policy:

* **hipBLAS and hipBLASLt** -- the API *and* the argument lists are cuBLAS's, so their
  signature tables are **derived from the CUDA ones** (see ``_symbols.py``) and cannot
  drift. Their **values** are not: ``HIPBLAS_OP_N`` is 111 where ``CUBLAS_OP_N`` is 0, so
  the enums and the status messages are hipBLAS's own (``_enums.py``). Passing a cuBLAS
  value to a hipBLAS call is what makes it answer ``HIPBLAS_STATUS_INVALID_ENUM``.
* **hipFFT, hipSPARSE and hipSOLVER** -- the shape is cuFFT's / cuSPARSE's / cuSOLVER's
  but the signatures are **not** in every case, so their tables are written here from
  ROCm's own headers. hipFFT's plan handle is a **pointer** on the AMD platform where
  cuFFT's is an `int`; `hipfftExecC2C` takes a direction argument the cuFFT table here
  omits; `hipsparseGetVersion` takes a handle and `hipsparseCreateDnMat` an order; and
  hipSOLVER has no version query at all. A table copied from the CUDA side binds symbols
  it cannot then call -- which is exactly what an earlier attempt produced (an access
  violation inside `hipfftExecR2C`).

A thin, numpy-friendly layer sits on top (``wrappers.py``), so a call takes arrays rather
than device pointers::

    from pygpgpu import hip_libraries

    c = hip_libraries.gemm(a, b)          # a @ b, on the GPU
    d = hip_libraries.axpy(2.0, x, y)     # y = 2*x + y
    f = hip_libraries.rfft(x)             # n//2+1 complex numbers, like np.fft.rfft

These are **host-side** operations, not device builtins: a gemm moves whole arrays to the
device and runs one library call, while a ``@kernel`` body is per-thread arithmetic. A
kernel body cannot call any of this.
"""
from __future__ import annotations

import importlib
from typing import Dict, List

from ._enums import (
    HIPFFT_BACKWARD,
    HIPFFT_FORWARD,
    hipblasComputeType_t,
    hipblasDiagType_t,
    hipblasFillMode_t,
    hipblasOperation_t,
    hipblasPointerMode_t,
    hipblasSideMode_t,
    hipblasStatus_t,
    hipDataType,
    hipfftResult,
    hipfftType,
    hipsolverStatus_t,
    hipsparseIndexBase_t,
    hipsparseIndexType_t,
    hipsparseOperation_t,
    hipsparseOrder_t,
    hipsparseSpMVAlg_t,
    hipsparseStatus_t,
)
from .hipblas import HIPBLAS, HIPBLASLt
from .hipfft import HIPFFT
from .hipsolver import HIPSOLVER
from .hipsparse import HIPSPARSE

__all__ = [
    "HIPBLAS",
    "HIPBLASLt",
    "HIPFFT",
    "HIPSPARSE",
    "HIPSOLVER",
    "hipblasStatus_t",
    "hipblasOperation_t",
    "hipblasComputeType_t",
    "hipblasPointerMode_t",
    "hipblasFillMode_t",
    "hipblasDiagType_t",
    "hipblasSideMode_t",
    "hipDataType",
    "hipfftResult",
    "hipfftType",
    "HIPFFT_FORWARD",
    "HIPFFT_BACKWARD",
    "hipsparseStatus_t",
    "hipsparseOperation_t",
    "hipsparseIndexBase_t",
    "hipsparseIndexType_t",
    "hipsparseOrder_t",
    "hipsparseSpMVAlg_t",
    "hipsolverStatus_t",
    "library_names",
    "driver_for",
    # the numpy-friendly wrappers, imported on first use
    "ensure_context",
    "allocate",
    "free",
    "to_device",
    "to_host",
    "synchronize",
    "create_handle",
    "destroy_handle",
    "set_stream",
    "gemm",
    "lt_gemm",
    "axpy",
    "scal",
    "dot",
    "rfft",
    "irfft",
]


# The library classes, by the name a caller knows them by.
LIBRARY_CLASSES:Dict[str, type] = {
    "hipblas": HIPBLAS,
    "hipblaslt": HIPBLASLt,
    "hipfft": HIPFFT,
    "hipsparse": HIPSPARSE,
    "hipsolver": HIPSOLVER,
}


def library_names()->List[str]:
    """The names of the bound HIP libraries, in a stable order."""
    return list(LIBRARY_CLASSES.keys())


def driver_for(name:str)->type:
    """The driver class of a named library (``"hipblas"``, ``"hipfft"``, ...)."""
    key = name.lower()
    if key not in LIBRARY_CLASSES:
        raise KeyError(
            f"unknown HIP library '{name}'; bound: {', '.join(LIBRARY_CLASSES)}")
    return LIBRARY_CLASSES[key]


# The wrappers are imported on demand rather than at the import of this package, because
# they need the HIP driver (and therefore start its background initialization), and a
# caller who only wants the raw entry points should not pay for that.
_WRAPPER_NAMES:List[str] = [
    "ensure_context", "allocate", "free", "to_device", "to_host", "synchronize",
    "create_handle", "destroy_handle", "set_stream",
    "gemm", "lt_gemm", "axpy", "scal", "dot", "rfft", "irfft",
]


def __getattr__(name:str)->object:
    """The numpy-friendly wrappers, imported on first use."""
    if name in _WRAPPER_NAMES:
        module = importlib.import_module(".wrappers", __package__)
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
