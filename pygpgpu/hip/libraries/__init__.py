"""The host-side HIP math libraries, bound with ctypes.

hipBLAS and hipBLASLt are the vendor's ports of cuBLAS and cuBLASLt: the API is the same,
the argument lists are the same, and only the names differ -- so the signature tables here
are **derived from the CUDA ones** (see ``_symbols.py``) instead of being written out
again, and the two sides cannot drift apart.

The **values** are not the same, though. ``HIPBLAS_OP_N`` is 111 where ``CUBLAS_OP_N`` is
0, and hipBLAS numbers its status codes 0..11 where cuBLAS's run to 16, so the enums and
the status messages are hipBLAS's own (``_enums.py``). Passing a cuBLAS value to a hipBLAS
call is exactly what makes it answer ``HIPBLAS_STATUS_INVALID_ENUM``.

A thin, numpy-friendly layer sits on top (``wrappers.py``), so a call takes arrays rather
than device pointers::

    from pygpgpu import hip_libraries

    c = hip_libraries.gemm(a, b)          # a @ b, on the GPU
    dl = hip_libraries.axpy(2.0, x, y)    # y = 2*x + y

These are **host-side** operations, not device builtins: a gemm moves whole arrays to the
device and runs one library call, while a ``@kernel`` body is per-thread arithmetic. A
kernel body cannot call any of this.

hipFFT, hipSPARSE and hipSOLVER are **not bound yet**. Their entry points exist and load,
but their signatures are not cuFFT's / cuSPARSE's / cuSOLVER's in every case (hipFFT's
plan handle and its exec call in particular), and a table copied from the CUDA side binds
symbols it cannot then call. They will come with their own tables.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List

from ._enums import hipblasComputeType_t, hipblasOperation_t, hipblasStatus_t
from .hipblas import HIPBLAS, HIPBLASLt

__all__ = [
    "HIPBLAS",
    "HIPBLASLt",
    "hipblasStatus_t",
    "hipblasOperation_t",
    "hipblasComputeType_t",
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
]


# The library classes, by the name a caller knows them by.
LIBRARY_CLASSES:Dict[str, Any] = {
    "hipblas": HIPBLAS,
    "hipblaslt": HIPBLASLt,
}


def library_names()->List[str]:
    """The names of the bound HIP libraries, in a stable order."""
    return list(LIBRARY_CLASSES.keys())


def driver_for(name:str)->Any:
    """The driver class of a named library (``"hipblas"``, ``"hipblaslt"``)."""
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
    "gemm", "lt_gemm", "axpy", "scal", "dot",
]


def __getattr__(name:str)->Any:
    """The numpy-friendly wrappers, imported on first use."""
    if name in _WRAPPER_NAMES:
        module = importlib.import_module(".wrappers", __package__)
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
