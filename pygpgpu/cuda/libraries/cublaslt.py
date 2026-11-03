"""cuBLASLt -- the newer cuBLAS path, bound with ctypes.

The shape is the CUDA driver's shape: a metaclass holds the DLL lazily, a ``Func``
wrapper binds the argument types and the return type on first use, ``init()``
pre-binds the entry points, and a returned status code becomes a ``RuntimeError``
carrying cuBLASLt's own description of it.

cuBLASLt is the layer under cuBLAS: where the classic API takes a type-specific
entry point (``cublasSgemm_v2``), cuBLASLt takes one generic entry point plus a set
of descriptors that say what the operands are. That is what makes mixed precision
and the fused epilogues possible, and it is also why a call needs a workspace and a
heuristically chosen algorithm.

The descriptors are opaque pointers, but two of them are small enough to be passed by
value: the algorithm descriptor and the heuristic result are plain byte arrays with
a fixed size, so they are declared as ctypes structures here.
"""
from __future__ import annotations

from ctypes import (c_char_p, c_float, c_int, c_int64, c_size_t, c_uint64, c_void_p,
                    POINTER, Structure)
from typing import Any, Dict, List

from ...constants import IntEnum
from ._base import Library


class cublasLtOrder_t(IntEnum):
    """The memory order of a matrix layout."""
    CUBLASLT_ORDER_COL   = 0
    CUBLASLT_ORDER_ROW   = 1
    CUBLASLT_ORDER_COL32 = 2
    CUBLASLT_ORDER_COL4_4R2_8C = 3
    CUBLASLT_ORDER_COL32_2R_4R4 = 4


class cublasLtMatrixLayoutAttribute_t(IntEnum):
    """The attributes a matrix layout carries."""
    CUBLASLT_MATRIX_LAYOUT_TYPE  = 0
    CUBLASLT_MATRIX_LAYOUT_ORDER = 1
    CUBLASLT_MATRIX_LAYOUT_ROWS = 2
    CUBLASLT_MATRIX_LAYOUT_COLS = 3
    CUBLASLT_MATRIX_LAYOUT_LD   = 4


class cublasLtMatmulDescAttributes_t(IntEnum):
    """The attributes a matrix-multiply description carries."""
    CUBLASLT_MATMUL_DESC_TRANSA = 3
    CUBLASLT_MATMUL_DESC_TRANSB = 4
    CUBLASLT_MATMUL_DESC_EPILOGUE = 7


class cublasLtMatmulPreferenceAttributes_t(IntEnum):
    """The attributes a heuristic preference carries."""
    CUBLASLT_MATMUL_PREF_SEARCH_MODE = 0
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1


class cublasLtEpilogue_t(IntEnum):
    """The operation applied to the result of a matrix multiply."""
    CUBLASLT_EPILOGUE_DEFAULT = 0
    CUBLASLT_EPILOGUE_RELU    = 1
    CUBLASLT_EPILOGUE_BIAS    = 2
    CUBLASLT_EPILOGUE_GELU    = 3


class cudaDataType_t(IntEnum):
    """The element type of a matrix.

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


# The handle and the descriptors are opaque pointers.
cublasLtHandle_t = c_void_p
cublasLtMatrixLayout_t = c_void_p
cublasLtMatmulDesc_t = c_void_p
cublasLtMatmulPreference_t = c_void_p
ptr_cublasLtHandle_t = POINTER(cublasLtHandle_t)
ptr_cublasLtMatrixLayout_t = POINTER(cublasLtMatrixLayout_t)
ptr_cublasLtMatmulDesc_t = POINTER(cublasLtMatmulDesc_t)
ptr_cublasLtMatmulPreference_t = POINTER(cublasLtMatmulPreference_t)

# The stream the work is ordered on.
cudaStream_t = c_void_p


class cublasLtMatmulAlgo_t(Structure):
    """The algorithm descriptor, passed by value.

    cuBLASLt declares it as a plain byte array of a fixed size, and the library
    reads it by value, so it is declared the same way here.
    """
    _fields_ = [
        ("data", c_uint64 * 8),
    ]


class cublasLtMatmulHeuristicResult_t(Structure):
    """One algorithm the heuristic chose, passed by value.

    The layout is the one cuBLASLt declares: the algorithm, the workspace it
    needs, the status it reported, a utilization measure and some reserved space.
    """
    _fields_ = [
        ("algo", cublasLtMatmulAlgo_t),
        ("workspaceSize", c_size_t),
        ("state", c_int),
        ("wavesCount", c_float),
        ("reserved", c_int * 4),
    ]


class CUBLASLtInfo:
    """cuBLASLt's signature table and its status codes."""

    func_signatures:Dict[str, Dict[str, Any]] = {
        # cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle)
        "cublasLtCreate": {
            "args": {
                "lightHandle": ptr_cublasLtHandle_t,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle)
        "cublasLtDestroy": {
            "args": {
                "lightHandle": cublasLtHandle_t,
            },
            "restype": c_int,
        },

        # size_t cublasLtGetVersion(void)
        #
        # The version itself, not a status: the return value is a number, so it is
        # exempt from the error check.
        "cublasLtGetVersion": {
            "args": {},
            "restype": c_size_t,
        },

        # const char* cublasLtGetStatusString(cublasStatus_t status)
        #
        # The one function whose return value is a message rather than a status, so
        # it is exempt from the error check.
        "cublasLtGetStatusString": {
            "args": {
                "status": c_int,
            },
            "restype": c_char_p,
        },

        # cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout,
        #     cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld)
        "cublasLtMatrixLayoutCreate": {
            "args": {
                "matLayout": ptr_cublasLtMatrixLayout_t,
                "type": c_int,
                "rows": c_uint64,
                "cols": c_uint64,
                "ld": c_int64,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout)
        "cublasLtMatrixLayoutDestroy": {
            "args": {
                "matLayout": cublasLtMatrixLayout_t,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
        #     cublasLtMatrixLayoutAttribute_t attr, const void* buf, size_t sizeInBytes)
        "cublasLtMatrixLayoutSetAttribute": {
            "args": {
                "matLayout": cublasLtMatrixLayout_t,
                "attr": c_int,
                "buf": c_void_p,
                "sizeInBytes": c_size_t,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
        #     cublasComputeType_t computeType, cudaDataType_t scaleType)
        "cublasLtMatmulDescCreate": {
            "args": {
                "matmulDesc": ptr_cublasLtMatmulDesc_t,
                "computeType": c_int,
                "scaleType": c_int,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc)
        "cublasLtMatmulDescDestroy": {
            "args": {
                "matmulDesc": cublasLtMatmulDesc_t,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
        #     cublasLtMatmulDescAttributes_t attr, const void* buf, size_t sizeInBytes)
        "cublasLtMatmulDescSetAttribute": {
            "args": {
                "matmulDesc": cublasLtMatmulDesc_t,
                "attr": c_int,
                "buf": c_void_p,
                "sizeInBytes": c_size_t,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref)
        "cublasLtMatmulPreferenceCreate": {
            "args": {
                "pref": ptr_cublasLtMatmulPreference_t,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref)
        "cublasLtMatmulPreferenceDestroy": {
            "args": {
                "pref": cublasLtMatmulPreference_t,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasLtMatmulPreferenceSetAttribute(
        #     cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr,
        #     const void* buf, size_t sizeInBytes)
        "cublasLtMatmulPreferenceSetAttribute": {
            "args": {
                "pref": cublasLtMatmulPreference_t,
                "attr": c_int,
                "buf": c_void_p,
                "sizeInBytes": c_size_t,
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
        #     cublasLtMatmulDesc_t computeDesc, cublasLtMatrixLayout_t Adesc,
        #     cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc,
        #     cublasLtMatrixLayout_t Ddesc, cublasLtMatmulPreference_t preference,
        #     int requestedAlgoCount, cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
        #     int* returnAlgoCount)
        "cublasLtMatmulAlgoGetHeuristic": {
            "args": {
                "lightHandle": cublasLtHandle_t,
                "computeDesc": cublasLtMatmulDesc_t,
                "Adesc": cublasLtMatrixLayout_t,
                "Bdesc": cublasLtMatrixLayout_t,
                "Cdesc": cublasLtMatrixLayout_t,
                "Ddesc": cublasLtMatrixLayout_t,
                "preference": cublasLtMatmulPreference_t,
                "requestedAlgoCount": c_int,
                "heuristicResultsArray": POINTER(cublasLtMatmulHeuristicResult_t),
                "returnAlgoCount": POINTER(c_int),
            },
            "restype": c_int,
        },

        # cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle,
        #     cublasLtMatmulDesc_t computeDesc, const void* alpha, const void* A,
        #     cublasLtMatrixLayout_t Adesc, const void* B, cublasLtMatrixLayout_t Bdesc,
        #     const void* beta, const void* C, cublasLtMatrixLayout_t Cdesc, void* D,
        #     cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo,
        #     void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream)
        "cublasLtMatmul": {
            "args": {
                "lightHandle": cublasLtHandle_t,
                "computeDesc": cublasLtMatmulDesc_t,
                "alpha": c_void_p,
                "A": c_void_p,
                "Adesc": cublasLtMatrixLayout_t,
                "B": c_void_p,
                "Bdesc": cublasLtMatrixLayout_t,
                "beta": c_void_p,
                "C": c_void_p,
                "Cdesc": cublasLtMatrixLayout_t,
                "D": c_void_p,
                "Ddesc": cublasLtMatrixLayout_t,
                "algo": POINTER(cublasLtMatmulAlgo_t),
                "workspace": c_void_p,
                "workspaceSizeInBytes": c_size_t,
                "stream": cudaStream_t,
            },
            "restype": c_int,
        },
    }

    error_codes:Dict[int, str] = {
        0: "no errors.",
        1: "the library was not initialized.",
        3: "the library could not allocate enough memory to perform the requested operation.",
        7: "one of the parameters passed to the API call is not within an acceptable range of values.",
        8: "the architecture of the device does not support the requested operation.",
        11: "an internal error occurred while mapping the memory of the operands.",
        13: "the requested operation could not be performed on the device.",
        14: "an internal error occurred.",
        15: "the requested operation is not supported on this device.",
        16: "the license for the library is not valid on this device.",
    }

    no_need_check_error_funcs:List[str] = [
        "cublasLtGetStatusString",
        # The version is a version, not a status: a non-zero one is not an error.
        "cublasLtGetVersion",
    ]


class CUBLASLt(Library):
    """cuBLASLt's entry points."""

    _library_name = "cuBLASLt"
    _dll_name = "cublasLt64_12.dll"
    _so_name = "libcublasLt.so"
    _signatures = CUBLASLtInfo.func_signatures
    _error_codes = CUBLASLtInfo.error_codes
    _no_error_check = CUBLASLtInfo.no_need_check_error_funcs


__all__ = [
    "CUBLASLt",
    "CUBLASLtInfo",
    "cublasLtOrder_t",
    "cublasLtMatrixLayoutAttribute_t",
    "cublasLtMatmulDescAttributes_t",
    "cublasLtMatmulPreferenceAttributes_t",
    "cublasLtEpilogue_t",
    "cudaDataType_t",
    "cublasLtMatmulAlgo_t",
    "cublasLtMatmulHeuristicResult_t",
    "cublasLtHandle_t",
    "cudaStream_t",
]
