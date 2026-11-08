"""A numpy-friendly layer over hipBLAS and hipBLASLt.

The raw entry points take device pointers and a column-major layout; these take numpy
arrays and return the same layout the caller passed, so ``gemm(a, b)`` is ``a @ b``.

Device memory goes through the HIP runtime the backend already uses
(``hipMalloc``/``hipMemcpy``), not through a second allocator: the buffers a caller
builds with ``to_device`` are the ones the backend's own kernels read.

These are **host-side** operations, the same way the CUDA ones are: a ``gemm`` moves whole
arrays to the device and runs one library call, so it is not expressible inside a
``@kernel`` body."""
# This module is the facade of a split: the wrappers moved to the part module named
# after their library, and __all__ stays here because hip/libraries/__init__ resolves
# each wrapper from this module by name. HIP has no csrmv: rocSPARSE needs a runtime
# directory ROCm 7.2 does not ship, so there is no sparse module to mirror.
from __future__ import annotations



__all__ = [
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


from ._wrappers_common import (
    _cast,
    _device_pointer,
    _element_type,
    _is_double,
    _is_single,
    _scalar_pointer,
    _stream_pointer,
    _void,
    allocate,
    create_handle,
    destroy_handle,
    ensure_context,
    free,
    set_stream,
    synchronize,
    to_device,
    to_host,
)
from ._wrappers_blas import (
    _double_pointer,
    _gemm_call,
    axpy,
    dot,
    gemm,
    scal,
)
from ._wrappers_lt import (
    lt_gemm,
)
from ._wrappers_fft import (
    irfft,
    rfft,
)
