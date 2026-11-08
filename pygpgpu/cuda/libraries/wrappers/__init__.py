"""The pythonic layer over the bound CUDA libraries.

The raw entry points are awkward to call: they take device pointers, they take the
shape of a matrix in four separate numbers, and they want the matrix in the
column-major layout BLAS uses. The wrappers here take numpy arrays and work out the
rest, because the point of the bindings is to be usable from python.

Everything here is an **array-level** operation: it moves whole arrays to the device,
runs one library call, and brings the result back. None of it is expressible inside
a ``@kernel`` body -- a kernel body is per-thread arithmetic, and a gemm is a
library call the host makes."""
# This module is the facade of a split: the wrappers moved to the part module named
# after their library, and __all__ stays here because cuda/libraries/__init__ resolves
# each wrapper from this module by name. The row-major/column-major convention inside
# gemm/lt_gemm (A gets b's memory, B gets a's, lda = N) is load-bearing; it moved
# verbatim with the code.
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
    "csrmv",
]


from ._wrappers_common import (
    _FLOAT32,
    _FLOAT64,
    _as_matrix,
    _context_is_usable,
    _data_type,
    _device_pointer,
    _element_type,
    _stream_pointer,
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
    axpy,
    dot,
    gemm,
    scal,
)
from ._wrappers_fft import (
    irfft,
    rfft,
)
from ._wrappers_sparse import (
    csrmv,
)
from ._wrappers_lt import (
    lt_gemm,
)
