"""hipBLAS's own enumerations.

The API shape is cuBLAS's, but the **values** are the vendor's own, and they are not
cuBLAS's numbering:

* ``HIPBLAS_OP_N`` is **111**, where cuBLAS's ``CUBLAS_OP_N`` is 0;
* the status codes are numbered separately -- ``HIPBLAS_STATUS_ALLOC_FAILED`` is 2 where
  ``CUBLAS_STATUS_ALLOC_FAILED`` is 3, and ``INTERNAL_ERROR`` is 6 where cuBLAS's is 14;
* the compute types are a small dense numbering (0..10) where cuBLAS's are 64..77.

So a hipBLAS enum cannot be the cuBLAS one under another name, and passing cuBLAS's value
to a hipBLAS call is exactly what makes it answer ``HIPBLAS_STATUS_INVALID_ENUM`` for an
ordinary gemm. The values here are read from ROCm's
``include/hipblas-common/hipblas-common.h``.
"""
from __future__ import annotations

from ...constants import IntEnum

__all__ = ["hipblasStatus_t", "hipblasOperation_t", "hipblasComputeType_t"]


class hipblasStatus_t(IntEnum):
    """hipBLAS's status codes. Not cuBLAS's: only SUCCESS agrees."""
    HIPBLAS_STATUS_SUCCESS           = 0
    HIPBLAS_STATUS_NOT_INITIALIZED   = 1
    HIPBLAS_STATUS_ALLOC_FAILED      = 2
    HIPBLAS_STATUS_INVALID_VALUE     = 3
    HIPBLAS_STATUS_MAPPING_ERROR     = 4
    HIPBLAS_STATUS_EXECUTION_FAILED  = 5
    HIPBLAS_STATUS_INTERNAL_ERROR    = 6
    HIPBLAS_STATUS_NOT_SUPPORTED     = 7
    HIPBLAS_STATUS_ARCH_MISMATCH     = 8
    HIPBLAS_STATUS_HANDLE_IS_NULLPTR = 9
    HIPBLAS_STATUS_INVALID_ENUM      = 10
    HIPBLAS_STATUS_UNKNOWN           = 11


class hipblasOperation_t(IntEnum):
    """Which transpose a gemm applies. 111/112/113, not cuBLAS's 0/1/2."""
    HIPBLAS_OP_N = 111
    HIPBLAS_OP_T = 112
    HIPBLAS_OP_C = 113


class hipblasComputeType_t(IntEnum):
    """The type a gemm computes in. A small dense numbering, not cuBLAS's 64..77."""
    HIPBLAS_COMPUTE_16F           = 0
    HIPBLAS_COMPUTE_16F_PEDANTIC  = 1
    HIPBLAS_COMPUTE_32F           = 2
    HIPBLAS_COMPUTE_32F_PEDANTIC  = 3
    HIPBLAS_COMPUTE_32F_FAST_16F  = 4
    HIPBLAS_COMPUTE_32F_FAST_16BF = 5
    HIPBLAS_COMPUTE_32F_FAST_TF32 = 6
    HIPBLAS_COMPUTE_64F           = 7
    HIPBLAS_COMPUTE_64F_PEDANTIC  = 8
    HIPBLAS_COMPUTE_32I           = 9
    HIPBLAS_COMPUTE_32I_PEDANTIC  = 10
