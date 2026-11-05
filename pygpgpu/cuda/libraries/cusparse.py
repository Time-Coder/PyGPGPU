"""cuSPARSE -- sparse linear algebra, bound with ctypes.

The shape is the CUDA driver's shape: a metaclass holds the DLL lazily, a ``Func``
wrapper binds the argument types and the return type on first use, ``init()``
pre-binds the entry points, and a returned status code becomes a ``RuntimeError``
carrying cuSPARSE's own description of it.

The operations bound here are the sparse matrix-vector multiply (``csrmv``) and the
descriptors it needs. The generic API is the one bound: a matrix is described by its
format, its shape and its data type, so the same entry point serves CSR, COO and
other formats.
"""
from __future__ import annotations

from ctypes import c_int, c_int64, c_size_t, c_void_p, POINTER
from typing import Dict, List

from ...constants import IntEnum
from ._base import Library


class cusparseStatus_t(IntEnum):
    """cuSPARSE's own status codes.

    These are cuSPARSE's numbering, not the CUDA driver's.
    """
    CUSPARSE_STATUS_SUCCESS                = 0
    CUSPARSE_STATUS_NOT_INITIALIZED        = 1
    CUSPARSE_STATUS_ALLOC_FAILED           = 2
    CUSPARSE_STATUS_INVALID_VALUE          = 3
    CUSPARSE_STATUS_ARCH_MISMATCH          = 4
    CUSPARSE_STATUS_MAPPING_ERROR          = 5
    CUSPARSE_STATUS_EXECUTION_FAILED       = 6
    CUSPARSE_STATUS_INTERNAL_ERROR         = 7
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8
    CUSPARSE_STATUS_ZERO_PIVOT             = 9
    CUSPARSE_STATUS_NOT_SUPPORTED          = 10
    CUSPARSE_STATUS_INSUFFICIENT_RESOURCES = 11


class cusparseIndexType_t(IntEnum):
    """The integer type a matrix's indices are stored in."""
    CUSPARSE_INDEX_16I = 1
    CUSPARSE_INDEX_32I = 2
    CUSPARSE_INDEX_64I = 3


class cusparseIndexBase_t(IntEnum):
    """Whether a matrix's indices count from zero or from one."""
    CUSPARSE_INDEX_BASE_ZERO = 0
    CUSPARSE_INDEX_BASE_ONE  = 1


class cusparseOperation_t(IntEnum):
    """Whether a sparse product transposes its matrix."""
    CUSPARSE_OPERATION_NON_TRANSPOSE      = 0
    CUSPARSE_OPERATION_TRANSPOSE          = 1
    CUSPARSE_OPERATION_HERMITIAN_TRANSPOSE = 2


class cusparseSpMVAlg_t(IntEnum):
    """The algorithm a sparse matrix-vector multiply runs."""
    CUSPARSE_SPMV_ALG_DEFAULT = 0
    CUSPARSE_SPMV_COO_ALG1   = 1
    CUSPARSE_SPMV_CSR_ALG1   = 2
    CUSPARSE_SPMV_CSR_ALG2   = 3
    CUSPARSE_SPMV_COO_ALG2   = 4
    CUSPARSE_SPMV_SELL_ALG1  = 5


class cudaDataType_t(IntEnum):
    """The element type of a matrix or a vector.

    The same numbering the CUDA runtime uses, so one table serves both.
    """
    CUDA_R_16F = 2
    CUDA_C_16F = 6
    CUDA_R_32F = 0
    CUDA_C_32F = 4
    CUDA_R_64F = 1
    CUDA_C_64F = 5
    CUDA_R_8I  = 3
    CUDA_C_8I  = 7
    CUDA_R_16I = 20
    CUDA_C_16I = 21
    CUDA_R_32I = 10
    CUDA_C_32I = 11
    CUDA_R_64I = 24
    CUDA_C_64I = 25


# Every descriptor is an opaque pointer.
cusparseHandle_t = c_void_p
cusparseSpMatDescr_t = c_void_p
cusparseConstSpMatDescr_t = c_void_p
cusparseDnVecDescr_t = c_void_p
cusparseConstDnVecDescr_t = c_void_p
cusparseMatDescr_t = c_void_p
cusparseDnMatDescr_t = c_void_p
cusparseConstDnMatDescr_t = c_void_p

ptr_cusparseHandle_t = POINTER(cusparseHandle_t)
ptr_cusparseSpMatDescr_t = POINTER(cusparseSpMatDescr_t)
ptr_cusparseDnVecDescr_t = POINTER(cusparseDnVecDescr_t)
ptr_cusparseMatDescr_t = POINTER(cusparseMatDescr_t)
ptr_cusparseDnMatDescr_t = POINTER(cusparseDnMatDescr_t)

# The stream a handle's work is ordered on.
cudaStream_t = c_void_p


class CUSPARSEInfo:
    """cuSPARSE's signature table and its status codes."""

    func_signatures:Dict[str, Dict[str, object]] = {
        # cusparseStatus_t cusparseCreate(cusparseHandle_t *handle)
        "cusparseCreate": {
            "args": {
                "handle": ptr_cusparseHandle_t,
            },
            "restype": c_int,
        },

        # cusparseStatus_t cusparseDestroy(cusparseHandle_t handle)
        "cusparseDestroy": {
            "args": {
                "handle": cusparseHandle_t,
            },
            "restype": c_int,
        },

        # cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId)
        "cusparseSetStream": {
            "args": {
                "handle": cusparseHandle_t,
                "streamId": cudaStream_t,
            },
            "restype": c_int,
        },

        # cusparseStatus_t cusparseGetVersion(int *version)
        "cusparseGetVersion": {
            "args": {
                "version": POINTER(c_int),
            },
            "restype": c_int,
        },

        # const char* cusparseGetErrorString(cusparseStatus_t status)
        #
        # The one function whose return value is a message rather than a status, so
        # it is exempt from the error check.
        "cusparseGetErrorString": {
            "args": {
                "status": c_int,
            },
            "restype": c_void_p,
        },

        # cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr,
        #     int64_t rows, int64_t cols, int64_t nnz, void* csrRowOffsets,
        #     void* csrColInd, void* csrValues, cusparseIndexType_t csrRowOffsetsType,
        #     cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase,
        #     cudaDataType valueType)
        "cusparseCreateCsr": {
            "args": {
                "spMatDescr": ptr_cusparseSpMatDescr_t,
                "rows": c_int64,
                "cols": c_int64,
                "nnz": c_int64,
                "csrRowOffsets": c_void_p,
                "csrColInd": c_void_p,
                "csrValues": c_void_p,
                "csrRowOffsetsType": c_int,
                "csrColIndType": c_int,
                "idxBase": c_int,
                "valueType": c_int,
            },
            "restype": c_int,
        },

        # cusparseStatus_t cusparseDestroySpMat(cusparseConstSpMatDescr_t spMatDescr)
        "cusparseDestroySpMat": {
            "args": {
                "spMatDescr": cusparseConstSpMatDescr_t,
            },
            "restype": c_int,
        },

        # cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
        #     int64_t size, void* values, cudaDataType valueType)
        "cusparseCreateDnVec": {
            "args": {
                "dnVecDescr": ptr_cusparseDnVecDescr_t,
                "size": c_int64,
                "values": c_void_p,
                "valueType": c_int,
            },
            "restype": c_int,
        },

        # cusparseStatus_t cusparseDestroyDnVec(cusparseConstDnVecDescr_t dnVecDescr)
        "cusparseDestroyDnVec": {
            "args": {
                "dnVecDescr": cusparseConstDnVecDescr_t,
            },
            "restype": c_int,
        },

        # cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t handle,
        #     cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA,
        #     cusparseConstDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY,
        #     cudaDataType computeType, cusparseSpMVAlg_t alg, size_t* bufferSize)
        "cusparseSpMV_bufferSize": {
            "args": {
                "handle": cusparseHandle_t,
                "opA": c_int,
                "alpha": c_void_p,
                "matA": cusparseConstSpMatDescr_t,
                "vecX": cusparseConstDnVecDescr_t,
                "beta": c_void_p,
                "vecY": cusparseDnVecDescr_t,
                "computeType": c_int,
                "alg": c_int,
                "bufferSize": POINTER(c_size_t),
            },
            "restype": c_int,
        },

        # cusparseStatus_t cusparseSpMV_preprocess(... the same arguments, plus the
        #     buffer the size query reported ...)
        "cusparseSpMV_preprocess": {
            "args": {
                "handle": cusparseHandle_t,
                "opA": c_int,
                "alpha": c_void_p,
                "matA": cusparseConstSpMatDescr_t,
                "vecX": cusparseConstDnVecDescr_t,
                "beta": c_void_p,
                "vecY": cusparseDnVecDescr_t,
                "computeType": c_int,
                "alg": c_int,
                "externalBuffer": c_void_p,
            },
            "restype": c_int,
        },

        # cusparseStatus_t cusparseSpMV(... the same arguments, plus the buffer ...)
        "cusparseSpMV": {
            "args": {
                "handle": cusparseHandle_t,
                "opA": c_int,
                "alpha": c_void_p,
                "matA": cusparseConstSpMatDescr_t,
                "vecX": cusparseConstDnVecDescr_t,
                "beta": c_void_p,
                "vecY": cusparseDnVecDescr_t,
                "computeType": c_int,
                "alg": c_int,
                "externalBuffer": c_void_p,
            },
            "restype": c_int,
        },

        # cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr,
        #     int64_t rows, int64_t cols, int64_t ld, cudaDataType valueType,
        #     cusparseMatrixLayout_t layout)
        "cusparseCreateDnMat": {
            "args": {
                "dnMatDescr": ptr_cusparseDnMatDescr_t,
                "rows": c_int64,
                "cols": c_int64,
                "ld": c_int64,
                "valueType": c_int,
                "layout": c_int,
            },
            "restype": c_int,
        },

        # cusparseStatus_t cusparseDestroyDnMat(cusparseConstDnMatDescr_t dnMatDescr)
        "cusparseDestroyDnMat": {
            "args": {
                "dnMatDescr": cusparseConstDnMatDescr_t,
            },
            "restype": c_int,
        },
    }

    error_codes:Dict[int, str] = {
        cusparseStatus_t.CUSPARSE_STATUS_SUCCESS:
            "no errors.",
        cusparseStatus_t.CUSPARSE_STATUS_NOT_INITIALIZED:
            "the library was not initialized.",
        cusparseStatus_t.CUSPARSE_STATUS_ALLOC_FAILED:
            "the library could not allocate enough memory to perform the requested operation.",
        cusparseStatus_t.CUSPARSE_STATUS_INVALID_VALUE:
            "one of the parameters passed to the API call is not within an acceptable range of values.",
        cusparseStatus_t.CUSPARSE_STATUS_ARCH_MISMATCH:
            "the architecture of the device does not support the requested operation.",
        cusparseStatus_t.CUSPARSE_STATUS_MAPPING_ERROR:
            "an internal error occurred while mapping the memory of the operands.",
        cusparseStatus_t.CUSPARSE_STATUS_EXECUTION_FAILED:
            "the requested operation could not be performed on the device.",
        cusparseStatus_t.CUSPARSE_STATUS_INTERNAL_ERROR:
            "an internal error occurred.",
        cusparseStatus_t.CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            "the matrix format is not supported.",
        cusparseStatus_t.CUSPARSE_STATUS_ZERO_PIVOT:
            "the factorization encountered a zero pivot.",
        cusparseStatus_t.CUSPARSE_STATUS_NOT_SUPPORTED:
            "the requested operation is not supported on this device.",
        cusparseStatus_t.CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
            "the resources to perform the requested operation are not sufficient.",
    }

    no_need_check_error_funcs:List[str] = [
        "cusparseGetErrorString",
    ]


class CUSPARSE(Library):
    """cuSPARSE's entry points."""

    _library_name = "cuSPARSE"
    _dll_name = "cusparse64_12.dll"
    _so_name = "libcusparse.so"
    _signatures = CUSPARSEInfo.func_signatures
    _error_codes = CUSPARSEInfo.error_codes
    _no_error_check = CUSPARSEInfo.no_need_check_error_funcs


__all__ = [
    "CUSPARSE",
    "CUSPARSEInfo",
    "cusparseStatus_t",
    "cusparseIndexType_t",
    "cusparseIndexBase_t",
    "cusparseOperation_t",
    "cusparseSpMVAlg_t",
    "cudaDataType_t",
    "cusparseHandle_t",
    "cudaStream_t",
]
