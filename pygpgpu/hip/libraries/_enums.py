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

__all__ = ["hipblasStatus_t", "hipblasOperation_t", "hipblasComputeType_t",
           "hipblasPointerMode_t", "hipblasFillMode_t", "hipblasDiagType_t",
           "hipblasSideMode_t",
           "hipDataType",
           "hipfftResult", "hipfftType", "HIPFFT_FORWARD", "HIPFFT_BACKWARD",
           "hipsparseStatus_t", "hipsparseOperation_t", "hipsparseIndexBase_t",
           "hipsparseIndexType_t", "hipsparseOrder_t", "hipsparseSpMVAlg_t",
           "hipsolverStatus_t"]


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


class hipblasPointerMode_t(IntEnum):
    """Where a scalar argument of a level-1 call lives. 0/1, as cuBLAS has it."""
    HIPBLAS_POINTER_MODE_HOST   = 0
    HIPBLAS_POINTER_MODE_DEVICE = 1


class hipblasFillMode_t(IntEnum):
    """Which triangle of a matrix a factorization reads.

    **121/122/123, where cuBLAS's are 0/1/2.** This is the enum hipSOLVER takes as its
    `uplo` argument, so passing cuBLAS's value makes it answer `INVALID_ENUM`.
    """
    HIPBLAS_FILL_MODE_UPPER = 121
    HIPBLAS_FILL_MODE_LOWER = 122
    HIPBLAS_FILL_MODE_FULL  = 123


class hipblasDiagType_t(IntEnum):
    """Whether a triangular matrix is unit triangular. 131/132, not cuBLAS's 0/1."""
    HIPBLAS_DIAG_NON_UNIT = 131
    HIPBLAS_DIAG_UNIT     = 132


class hipblasSideMode_t(IntEnum):
    """Which side a symmetric or triangular matrix multiplies on. 141/142/143."""
    HIPBLAS_SIDE_LEFT  = 141
    HIPBLAS_SIDE_RIGHT = 142
    HIPBLAS_SIDE_BOTH  = 143


# ---------------------------------------------------------------------------
# hipDataType -- the element type a descriptor names
# ---------------------------------------------------------------------------


class hipDataType(IntEnum):
    """The element type, from `hip/library_types.h`.

    Not cuBLAS's `cudaDataType_t`: the two produce the same numbers for the common types
    but not for all of them, and this is the enum HIP's own headers name.
    """
    HIP_R_32F  = 0
    HIP_R_64F  = 1
    HIP_R_16F  = 2
    HIP_R_8I   = 3
    HIP_C_32F  = 4
    HIP_C_64F  = 5
    HIP_C_16F  = 6
    HIP_C_8I   = 7
    HIP_R_8U   = 8
    HIP_C_8U   = 9
    HIP_R_32I  = 10
    HIP_C_32I  = 11
    HIP_R_32U  = 12
    HIP_C_32U  = 13
    HIP_R_16BF = 14
    HIP_C_16BF = 15


# ---------------------------------------------------------------------------
# hipFFT -- from hipfft/hipfft.h
# ---------------------------------------------------------------------------


class hipfftResult(IntEnum):
    """hipFFT's result codes. Not cuFFT's numbering, though both start at 0."""
    HIPFFT_SUCCESS                   = 0
    HIPFFT_INVALID_PLAN              = 1
    HIPFFT_ALLOC_FAILED              = 2
    HIPFFT_INVALID_TYPE              = 3
    HIPFFT_INVALID_VALUE             = 4
    HIPFFT_INTERNAL_ERROR            = 5
    HIPFFT_EXEC_FAILED               = 6
    HIPFFT_SETUP_FAILED              = 7
    HIPFFT_INVALID_SIZE              = 8
    HIPFFT_UNALIGNED_DATA            = 9
    HIPFFT_INCOMPLETE_PARAMETER_LIST = 10
    HIPFFT_INVALID_DEVICE            = 11
    HIPFFT_PARSE_ERROR               = 12
    HIPFFT_NO_WORKSPACE              = 13
    HIPFFT_NOT_IMPLEMENTED           = 14
    HIPFFT_NOT_SUPPORTED             = 16


class hipfftType(IntEnum):
    """The transform kind. These DO carry cuFFT's values (`0x2a` and friends)."""
    HIPFFT_R2C = 0x2a
    HIPFFT_C2R = 0x2c
    HIPFFT_C2C = 0x29
    HIPFFT_D2Z = 0x6a
    HIPFFT_Z2D = 0x6c
    HIPFFT_Z2Z = 0x69


# The direction argument `hipfftExecC2C` takes. These are macros in the header, not enum
# members, which is why they are plain constants here.
HIPFFT_FORWARD = -1
HIPFFT_BACKWARD = 1


# ---------------------------------------------------------------------------
# hipSPARSE -- from hipsparse/hipsparse-types.h
# ---------------------------------------------------------------------------


class hipsparseStatus_t(IntEnum):
    """hipSPARSE's status codes. Not cuSPARSE's."""
    HIPSPARSE_STATUS_SUCCESS                   = 0
    HIPSPARSE_STATUS_NOT_INITIALIZED           = 1
    HIPSPARSE_STATUS_ALLOC_FAILED              = 2
    HIPSPARSE_STATUS_INVALID_VALUE             = 3
    HIPSPARSE_STATUS_ARCH_MISMATCH             = 4
    HIPSPARSE_STATUS_MAPPING_ERROR             = 5
    HIPSPARSE_STATUS_EXECUTION_FAILED          = 6
    HIPSPARSE_STATUS_INTERNAL_ERROR            = 7
    HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8
    HIPSPARSE_STATUS_ZERO_PIVOT                = 9
    HIPSPARSE_STATUS_NOT_SUPPORTED             = 10
    HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES    = 11


class hipsparseOperation_t(IntEnum):
    """Which transpose a sparse operation applies."""
    HIPSPARSE_OPERATION_NON_TRANSPOSE       = 0
    HIPSPARSE_OPERATION_TRANSPOSE           = 1
    HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2


class hipsparseIndexBase_t(IntEnum):
    """Whether the row and column indices start at zero or one."""
    HIPSPARSE_INDEX_BASE_ZERO = 0
    HIPSPARSE_INDEX_BASE_ONE  = 1


class hipsparseIndexType_t(IntEnum):
    """The width of the index arrays a sparse descriptor names."""
    HIPSPARSE_INDEX_16U = 1
    HIPSPARSE_INDEX_32I = 2
    HIPSPARSE_INDEX_64I = 3


class hipsparseOrder_t(IntEnum):
    """Which layout a dense descriptor names."""
    HIPSPARSE_ORDER_COL = 1
    HIPSPARSE_ORDER_ROW = 2


class hipsparseSpMVAlg_t(IntEnum):
    """The algorithm a sparse matrix-vector multiply runs."""
    HIPSPARSE_SPMV_ALG_DEFAULT = 0
    HIPSPARSE_SPMV_COO_ALG1    = 1
    HIPSPARSE_SPMV_CSR_ALG1    = 2
    HIPSPARSE_SPMV_CSR_ALG2    = 3
    HIPSPARSE_SPMV_COO_ALG2    = 4


# ---------------------------------------------------------------------------
# hipSOLVER -- from hipsolver/internal/hipsolver-types.h
# ---------------------------------------------------------------------------


class hipsolverStatus_t(IntEnum):
    """hipSOLVER's status codes.

    It has two of its own (`ZERO_PIVOT`, `MATRIX_TYPE_NOT_SUPPORTED`) that hipBLAS's
    numbering does not, which is why the two tables cannot be one.
    """
    HIPSOLVER_STATUS_SUCCESS                   = 0
    HIPSOLVER_STATUS_NOT_INITIALIZED           = 1
    HIPSOLVER_STATUS_ALLOC_FAILED              = 2
    HIPSOLVER_STATUS_INVALID_VALUE             = 3
    HIPSOLVER_STATUS_MAPPING_ERROR             = 4
    HIPSOLVER_STATUS_EXECUTION_FAILED          = 5
    HIPSOLVER_STATUS_INTERNAL_ERROR            = 6
    HIPSOLVER_STATUS_NOT_SUPPORTED             = 7
    HIPSOLVER_STATUS_ARCH_MISMATCH             = 8
    HIPSOLVER_STATUS_HANDLE_IS_NULLPTR         = 9
    HIPSOLVER_STATUS_INVALID_ENUM              = 10
    HIPSOLVER_STATUS_UNKNOWN                   = 11
    HIPSOLVER_STATUS_ZERO_PIVOT                = 12
    HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 13
