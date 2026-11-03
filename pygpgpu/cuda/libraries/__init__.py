"""Host-side CUDA math libraries, bound with ctypes.

The CUDA *driver* (``pygpgpu/cuda/driver``) is the low-level API: contexts, streams,
modules, memory. The libraries bound here are the ones that ship as separate DLLs
next to it and do the actual mathematics:

    cuBLAS      -- dense linear algebra (gemm, axpy, scal, dot, ...)
    cuBLASLt    -- the newer cuBLAS path (heuristics, mixed precision)
    cuSOLVER    -- dense linear solves and factorizations (potrf, gesvd, ...)
    cuFFT       -- fast fourier transforms
    cuSPARSE    -- sparse linear algebra (csrmv, ...)
    CUDA runtime-- allocation and host/device copies

Every library follows the shape of the CUDA driver class (``MetaCUDA`` / ``CUDA``):
a metaclass holds the DLL lazily, a ``Func`` wrapper binds the argument types and the
return type on first use, ``init()`` pre-binds the entry points, and a returned error
code is turned into a ``RuntimeError`` carrying the library's own message.

The DLL is loaded on first use, never at import time, and a machine without the CUDA
toolkit still imports this package: the failure surfaces at the call.

Nothing here is a device-side builtin. A ``@kernel`` body cannot call any of it --
these are array-level operations the host performs, not per-thread arithmetic. See the
README section "Host-side CUDA libraries" for the difference.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List

# The module that owns each library's driver class, and the class inside it. The
# two tables are public because they are the honest answer to "what can I bind
# here?": the value is the module path, and the class is inside it.
LIBRARY_MODULES:Dict[str, str] = {
    "cublas": "pygpgpu.cuda.libraries.cublas",
    "cublaslt": "pygpgpu.cuda.libraries.cublaslt",
    "cusolver": "pygpgpu.cuda.libraries.cusolver",
    "cufft": "pygpgpu.cuda.libraries.cufft",
    "cusparse": "pygpgpu.cuda.libraries.cusparse",
    "cudart": "pygpgpu.cuda.libraries.cudart",
}

LIBRARY_CLASSES:Dict[str, str] = {
    "cublas": "CUBLAS",
    "cublaslt": "CUBLASLt",
    "cusolver": "CUSOLVER",
    "cufft": "CUFFT",
    "cusparse": "CUSPARSE",
    "cudart": "CUDART",
}


def library_names()->List[str]:
    """The names of the bound libraries, in a stable order."""
    return list(LIBRARY_MODULES.keys())


def driver_for(name:str)->Any:
    """The driver class of a named library (``"cublas"``, ``"cufft"``, ...).

    The module is imported on demand, so asking for a library never loads the DLLs
    of the ones before it -- and asking for a library this build does not carry
    raises ``KeyError`` naming the ones it does.
    """
    key = name.lower()
    if key not in LIBRARY_MODULES:
        raise KeyError(f"unknown CUDA library '{name}'; bound: {', '.join(LIBRARY_MODULES)}")

    module = importlib.import_module(LIBRARY_MODULES[key])
    return getattr(module, LIBRARY_CLASSES[key])


def init_all()->None:
    """Pre-bind every bound library's entry points.

    A library whose DLL is missing is recorded rather than raised, so this is safe
    to call on a machine without the CUDA toolkit; the failure surfaces at the call
    that needs it.
    """
    for name in LIBRARY_MODULES:
        driver_for(name).init()


# The pythonic wrappers. They are imported on demand rather than at the import of
# this package, because the wrappers need the CUDA driver (and therefore start its
# background initialization), and a caller who only wants the raw entry points
# should not pay for that.
_WRAPPER_NAMES:List[str] = [
    "gemm", "lt_gemm", "axpy", "scal", "dot", "rfft", "irfft", "csrmv",
    "ensure_context", "allocate", "free", "to_device", "to_host", "synchronize",
]


def __getattr__(name:str)->Any:
    """The pythonic wrappers, imported on first use.

    ``pygpgpu.cuda_libraries.gemm(a, b)`` reaches the wrapper this way: the
    wrappers module is imported when the name is first read, not when this package
    is imported.
    """
    if name in _WRAPPER_NAMES:
        module = importlib.import_module(".wrappers", __package__)
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LIBRARY_MODULES",
    "LIBRARY_CLASSES",
    "library_names",
    "driver_for",
    "init_all",
    *_WRAPPER_NAMES,
]
