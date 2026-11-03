"""cuBLAS -- dense linear algebra, bound with ctypes.

The shape is the CUDA driver's shape: a metaclass holds the DLL lazily, a ``Func``
wrapper binds the argument types and the return type on first use, ``init()``
pre-binds the entry points, and a returned status code becomes a ``RuntimeError``
carrying cuBLAS's own description of it.

The entry points bound here are the ones a GPU maths user actually calls: the general
matrix multiply (``gemm``), the level-1 operations (``axpy``, ``scal``, ``dot``) and
the handle and stream management around them.

The ``cublasStatus_t`` values are cuBLAS's own, not the CUDA driver's: cuBLAS has its
own numbering, and a code that means "success" in one library does not in the other.
"""
from __future__ import annotations

from ctypes import c_char_p, c_int, c_float, c_double, c_void_p, POINTER
from typing import Any, Dict, List

from ...constants import IntEnum
from ._base import Library


class cublasStatus_t(IntEnum):
    """cuBLAS's own status codes.

    These are cuBLAS's numbering, not the CUDA driver's: the same number means
    something else in each library, so the two tables never share an entry.
    """
    CUBLAS_STATUS_SUCCESS          = 0
    CUBLAS_STATUS_NOT_INITIALIZED  = 1
    CUBLAS_STATUS_ALLOC_FAILED     = 3
    CUBLAS_STATUS_INVALID_VALUE    = 7
    CUBLAS_STATUS_ARCH_MISMATCH    = 8
    CUBLAS_STATUS_MAPPING_ERROR    = 11
    CUBLAS_STATUS_EXECUTION_FAILED = 13
    CUBLAS_STATUS_INTERNAL_ERROR   = 14
    CUBLAS_STATUS_NOT_SUPPORTED    = 15
    CUBLAS_STATUS_LICENSE_ERROR    = 16


class cublasOperation_t(IntEnum):
    """Which transpose a gemm applies to its operands."""
    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1
    CUBLAS_OP_C = 2


class cublasFillMode_t(IntEnum):
    """Which triangle of a matrix a factorization reads."""
    CUBLAS_FILL_MODE_LOWER = 0
    CUBLAS_FILL_MODE_UPPER = 1
    CUBLAS_FILL_MODE_FULL  = 2


class cublasPointerMode_t(IntEnum):
    """Where a scalar argument of a level-1 call lives."""
    CUBLAS_POINTER_MODE_HOST   = 0
    CUBLAS_POINTER_MODE_DEVICE = 1


class cublasComputeType_t(IntEnum):
    """The type a gemm computes in."""
    CUBLAS_COMPUTE_16F           = 64
    CUBLAS_COMPUTE_16F_PEDANTIC  = 65
    CUBLAS_COMPUTE_32F           = 68
    CUBLAS_COMPUTE_32F_PEDANTIC  = 69
    CUBLAS_COMPUTE_64F           = 70
    CUBLAS_COMPUTE_64F_PEDANTIC  = 71
    CUBLAS_COMPUTE_32F_FAST_16F  = 74
    CUBLAS_COMPUTE_32F_FAST_16BF = 75
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77


class cublasGemmAlgo_t(IntEnum):
    """The algorithm a gemm runs."""
    CUBLAS_GEMM_DEFAULT = -1
    CUBLAS_GEMM_ALGO0   = 0
    CUBLAS_GEMM_ALGO1   = 1
    CUBLAS_GEMM_ALGO2   = 2
    CUBLAS_GEMM_ALGO3   = 3
    CUBLAS_GEMM_ALGO4   = 4
    CUBLAS_GEMM_ALGO5   = 5
    CUBLAS_GEMM_ALGO6   = 6
    CUBLAS_GEMM_ALGO7   = 7
    CUBLAS_GEMM_ALGO8   = 8
    CUBLAS_GEMM_ALGO9   = 9
    CUBLAS_GEMM_ALGO10  = 10
    CUBLAS_GEMM_ALGO11  = 11
    CUBLAS_GEMM_ALGO12  = 12
    CUBLAS_GEMM_ALGO13  = 13
    CUBLAS_GEMM_ALGO14  = 14
    CUBLAS_GEMM_ALGO15  = 15
    CUBLAS_GEMM_ALGO16  = 16
    CUBLAS_GEMM_ALGO17  = 17
    CUBLAS_GEMM_ALGO18  = 18
    CUBLAS_GEMM_ALGO19  = 19
    CUBLAS_GEMM_ALGO20  = 20
    CUBLAS_GEMM_ALGO21  = 21
    CUBLAS_GEMM_ALGO22  = 22
    CUBLAS_GEMM_ALGO23  = 23


# The handle is an opaque pointer, and so is every device pointer it is given.
cublasHandle_t = c_void_p
ptr_cublasHandle_t = POINTER(cublasHandle_t)

# The stream a handle's work is ordered on. The CUDA runtime's own stream type is
# the same opaque pointer the driver uses.
cudaStream_t = c_void_p
ptr_cudaStream_t = POINTER(cudaStream_t)


class CUBLASInfo:
    """cuBLAS's signature table and its status codes.

    The table maps a symbol name to the argument names and types it takes and the
    type it returns, which is what the ``Func`` wrapper binds on first use.
    """

    func_signatures:Dict[str, Dict[str, Any]] = {
        # cublasStatus_t cublasCreate_v2(cublasHandle_t *handle)
        "cublasCreate_v2": {
            "args": {
                "handle": ptr_cublasHandle_t,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasDestroy_v2(cublasHandle_t handle)
        "cublasDestroy_v2": {
            "args": {
                "handle": cublasHandle_t,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId)
        "cublasSetStream_v2": {
            "args": {
                "handle": cublasHandle_t,
                "streamId": cudaStream_t,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *streamId)
        "cublasGetStream_v2": {
            "args": {
                "handle": cublasHandle_t,
                "streamId": ptr_cudaStream_t,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode)
        "cublasSetPointerMode_v2": {
            "args": {
                "handle": cublasHandle_t,
                "mode": c_int,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode)
        "cublasGetPointerMode_v2": {
            "args": {
                "handle": cublasHandle_t,
                "mode": POINTER(c_int),
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version)
        #
        # The _v2 form takes the handle; the older cublasGetVersion does not and
        # answers the version of the library itself.
        "cublasGetVersion_v2": {
            "args": {
                "handle": cublasHandle_t,
                "version": POINTER(c_int),
            },
            "restype": c_int,
        },

        # int cublasGetVersion(void)
        "cublasGetVersion": {
            "args": {},
            "restype": c_int,
        },

        # const char* cublasGetStatusString(cublasStatus_t status)
        #
        # The one function whose return value is a message rather than a status, so
        # it is exempt from the error check.
        "cublasGetStatusString": {
            "args": {
                "status": c_int,
            },
            "restype": c_char_p,
        },

        # cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
        #     cublasOperation_t transb, int m, int n, int k, const float* alpha,
        #     const float* A, int lda, const float* B, int ldb, const float* beta,
        #     float* C, int ldc)
        "cublasSgemm_v2": {
            "args": {
                "handle": cublasHandle_t,
                "transa": c_int,
                "transb": c_int,
                "m": c_int,
                "n": c_int,
                "k": c_int,
                "alpha": POINTER(c_float),
                "A": POINTER(c_float),
                "lda": c_int,
                "B": POINTER(c_float),
                "ldb": c_int,
                "beta": POINTER(c_float),
                "C": POINTER(c_float),
                "ldc": c_int,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasDgemm_v2(... the double form ...)
        "cublasDgemm_v2": {
            "args": {
                "handle": cublasHandle_t,
                "transa": c_int,
                "transb": c_int,
                "m": c_int,
                "n": c_int,
                "k": c_int,
                "alpha": POINTER(c_double),
                "A": POINTER(c_double),
                "lda": c_int,
                "B": POINTER(c_double),
                "ldb": c_int,
                "beta": POINTER(c_double),
                "C": POINTER(c_double),
                "ldc": c_int,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
        #     cublasOperation_t transb, int m, int n, int k, const void* alpha,
        #     const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype,
        #     int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc,
        #     cublasComputeType_t computeType, cublasGemmAlgo_t algo)
        #
        # The generic form: one entry point covers every combination of operand and
        # compute type, so a caller that wants a mixed-precision product does not
        # need a separate binding per type.
        "cublasGemmEx": {
            "args": {
                "handle": cublasHandle_t,
                "transa": c_int,
                "transb": c_int,
                "m": c_int,
                "n": c_int,
                "k": c_int,
                "alpha": c_void_p,
                "A": c_void_p,
                "Atype": c_int,
                "lda": c_int,
                "B": c_void_p,
                "Btype": c_int,
                "ldb": c_int,
                "beta": c_void_p,
                "C": c_void_p,
                "Ctype": c_int,
                "ldc": c_int,
                "computeType": c_int,
                "algo": c_int,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n, const float* alpha,
        #     const float* x, int incx, float* y, int incy)
        "cublasSaxpy_v2": {
            "args": {
                "handle": cublasHandle_t,
                "n": c_int,
                "alpha": POINTER(c_float),
                "x": POINTER(c_float),
                "incx": c_int,
                "y": POINTER(c_float),
                "incy": c_int,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasDaxpy_v2(... the double form ...)
        "cublasDaxpy_v2": {
            "args": {
                "handle": cublasHandle_t,
                "n": c_int,
                "alpha": POINTER(c_double),
                "x": POINTER(c_double),
                "incx": c_int,
                "y": POINTER(c_double),
                "incy": c_int,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float* alpha,
        #     float* x, int incx)
        "cublasSscal_v2": {
            "args": {
                "handle": cublasHandle_t,
                "n": c_int,
                "alpha": POINTER(c_float),
                "x": POINTER(c_float),
                "incx": c_int,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasDscal_v2(... the double form ...)
        "cublasDscal_v2": {
            "args": {
                "handle": cublasHandle_t,
                "n": c_int,
                "alpha": POINTER(c_double),
                "x": POINTER(c_double),
                "incx": c_int,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float* x,
        #     int incx, const float* y, int incy, float* result)
        #
        # The result is a device pointer in the default pointer mode, so the caller
        # reads it back from the device.
        "cublasSdot_v2": {
            "args": {
                "handle": cublasHandle_t,
                "n": c_int,
                "x": POINTER(c_float),
                "incx": c_int,
                "y": POINTER(c_float),
                "incy": c_int,
                "result": POINTER(c_float),
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasDdot_v2(... the double form ...)
        "cublasDdot_v2": {
            "args": {
                "handle": cublasHandle_t,
                "n": c_int,
                "x": POINTER(c_double),
                "incx": c_int,
                "y": POINTER(c_double),
                "incy": c_int,
                "result": POINTER(c_double),
            },
            "restype": c_int,
        },
    }

    error_codes:Dict[int, str] = {
        cublasStatus_t.CUBLAS_STATUS_SUCCESS:
            "no errors.",
        cublasStatus_t.CUBLAS_STATUS_NOT_INITIALIZED:
            "the library was not initialized.",
        cublasStatus_t.CUBLAS_STATUS_ALLOC_FAILED:
            "the library could not allocate enough memory to perform the requested operation.",
        cublasStatus_t.CUBLAS_STATUS_INVALID_VALUE:
            "one of the parameters passed to the API call is not within an acceptable range of values.",
        cublasStatus_t.CUBLAS_STATUS_ARCH_MISMATCH:
            "the architecture of the device does not support the requested operation.",
        cublasStatus_t.CUBLAS_STATUS_MAPPING_ERROR:
            "an internal error occurred while mapping the memory of the operands.",
        cublasStatus_t.CUBLAS_STATUS_EXECUTION_FAILED:
            "the requested operation could not be performed on the device.",
        cublasStatus_t.CUBLAS_STATUS_INTERNAL_ERROR:
            "an internal error occurred.",
        cublasStatus_t.CUBLAS_STATUS_NOT_SUPPORTED:
            "the requested operation is not supported on this device.",
        cublasStatus_t.CUBLAS_STATUS_LICENSE_ERROR:
            "the license for the library is not valid on this device.",
    }

    no_need_check_error_funcs:List[str] = [
        "cublasGetStatusString",
        # The version is a version, not a status: a non-zero one is not an error.
        "cublasGetVersion",
    ]


class CUBLAS(Library):
    """cuBLAS's entry points.

    The class is used through its metaclass, so ``CUBLAS.cublasSgemm_v2`` is the
    bound function and ``CUBLAS.cublasCreate_v2(ptr)`` is the call.
    """

    _library_name = "cuBLAS"
    _dll_name = "cublas64_12.dll"
    _so_name = "libcublas.so"
    _signatures = CUBLASInfo.func_signatures
    _error_codes = CUBLASInfo.error_codes
    _no_error_check = CUBLASInfo.no_need_check_error_funcs


__all__ = [
    "CUBLAS",
    "CUBLASInfo",
    "cublasStatus_t",
    "cublasOperation_t",
    "cublasFillMode_t",
    "cublasPointerMode_t",
    "cublasComputeType_t",
    "cublasGemmAlgo_t",
    "cublasHandle_t",
    "cudaStream_t",
]
