"""cuSOLVER -- dense linear solves and factorizations, bound with ctypes.

The shape is the CUDA driver's shape: a metaclass holds the DLL lazily, a ``Func``
wrapper binds the argument types and the return type on first use, ``init()``
pre-binds the entry points, and a returned status code becomes a ``RuntimeError``
carrying cuSOLVER's own description of it.

The operations bound here are the Cholesky factorization (``potrf``), the singular
value decomposition (``gesvd``) and the LU factorization (``getrf``), each with the
workspace query that has to precede it.

**A note about the consumer cards.** cuSOLVER's dense factorizations are implemented
as device kernels that NVIDIA does not ship for every architecture, and a card the
kernels do not support fails at execution rather than at setup: the call returns a
failure code and the context reports an illegal memory access. On the machine this
was developed on (an RTX 4070 Laptop, compute capability 8.9, CUDA 12.6) every one
of ``potrf``, ``getrf``, ``gesvd``, ``gesvdj`` and ``syevj`` fails that way, and
``compute-sanitizer`` puts the fault inside cuSOLVER's own DLL rather than in the
arguments. The bindings below are therefore correct but not runnable on that card;
the tests skip with that reason instead of failing.
"""
from __future__ import annotations

from ctypes import c_char, c_int, c_float, c_double, c_void_p, POINTER
from typing import Dict, List

from ...constants import IntEnum
from ._base import Library


class cusolverStatus_t(IntEnum):
    """cuSOLVER's own status codes.

    These are cuSOLVER's numbering, not the CUDA driver's.
    """
    CUSOLVER_STATUS_SUCCESS                  = 0
    CUSOLVER_STATUS_NOT_INITIALIZED          = 1
    CUSOLVER_STATUS_ALLOC_FAILED             = 2
    CUSOLVER_STATUS_INVALID_VALUE            = 3
    CUSOLVER_STATUS_ARCH_MISMATCH            = 4
    CUSOLVER_STATUS_MAPPING_ERROR            = 5
    CUSOLVER_STATUS_EXECUTION_FAILED         = 6
    CUSOLVER_STATUS_INTERNAL_ERROR            = 7
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8
    CUSOLVER_STATUS_NOT_SUPPORTED            = 9
    CUSOLVER_STATUS_ZERO_PIVOT               = 10
    CUSOLVER_STATUS_INVALID_LICENSE          = 11
    CUSOLVER_STATUS_INVALID_WORKSPACE        = 31


# The handle and the parameters object are opaque pointers.
cusolverDnHandle_t = c_void_p
cusolverDnParams_t = c_void_p
ptr_cusolverDnHandle_t = POINTER(cusolverDnHandle_t)
ptr_cusolverDnParams_t = POINTER(cusolverDnParams_t)

# The stream a handle's work is ordered on.
cudaStream_t = c_void_p


class CUSOLVERInfo:
    """cuSOLVER's signature table and its status codes."""

    func_signatures:Dict[str, Dict[str, object]] = {
        # cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle)
        "cusolverDnCreate": {
            "args": {
                "handle": ptr_cusolverDnHandle_t,
            },
            "restype": c_int,
        },

        # cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle)
        "cusolverDnDestroy": {
            "args": {
                "handle": cusolverDnHandle_t,
            },
            "restype": c_int,
        },

        # cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId)
        "cusolverDnSetStream": {
            "args": {
                "handle": cusolverDnHandle_t,
                "streamId": cudaStream_t,
            },
            "restype": c_int,
        },

        # cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t *params)
        "cusolverDnCreateParams": {
            "args": {
                "params": ptr_cusolverDnParams_t,
            },
            "restype": c_int,
        },

        # cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t params)
        "cusolverDnDestroyParams": {
            "args": {
                "params": cusolverDnParams_t,
            },
            "restype": c_int,
        },

        # cusolverStatus_t cusolverGetVersion(int *version)
        #
        # The version belongs to the library, not to a handle: cuSOLVER exports
        # `cusolverGetVersion`, and there is no `cusolverDnGetVersion`.
        "cusolverGetVersion": {
            "args": {
                "version": POINTER(c_int),
            },
            "restype": c_int,
        },

        # ---------- Cholesky ----------
        #
        # The workspace query comes first: it reports how many elements the
        # factorization needs, and the factorization refuses a smaller workspace.

        # cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle,
        #     cublasFillMode_t uplo, int n, float* A, int lda, int* Lwork)
        "cusolverDnSpotrf_bufferSize": {
            "args": {
                "handle": cusolverDnHandle_t,
                "uplo": c_int,
                "n": c_int,
                "A": POINTER(c_float),
                "lda": c_int,
                "Lwork": POINTER(c_int),
            },
            "restype": c_int,
        },

        # cusolverStatus_t cusolverDnDpotrf_bufferSize(... the double form ...)
        "cusolverDnDpotrf_bufferSize": {
            "args": {
                "handle": cusolverDnHandle_t,
                "uplo": c_int,
                "n": c_int,
                "A": POINTER(c_double),
                "lda": c_int,
                "Lwork": POINTER(c_int),
            },
            "restype": c_int,
        },

        # cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle,
        #     cublasFillMode_t uplo, int n, float* A, int lda, float* Workspace,
        #     int Lwork, int* devInfo)
        "cusolverDnSpotrf": {
            "args": {
                "handle": cusolverDnHandle_t,
                "uplo": c_int,
                "n": c_int,
                "A": POINTER(c_float),
                "lda": c_int,
                "Workspace": POINTER(c_float),
                "Lwork": c_int,
                "devInfo": POINTER(c_int),
            },
            "restype": c_int,
        },

        # cusolverStatus_t cusolverDnDpotrf(... the double form ...)
        "cusolverDnDpotrf": {
            "args": {
                "handle": cusolverDnHandle_t,
                "uplo": c_int,
                "n": c_int,
                "A": POINTER(c_double),
                "lda": c_int,
                "Workspace": POINTER(c_double),
                "Lwork": c_int,
                "devInfo": POINTER(c_int),
            },
            "restype": c_int,
        },

        # ---------- LU ----------
        #
        # The workspace query takes the matrix itself, because the factorization
        # pivots and the workspace depends on the pivoting it chooses.

        # cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle,
        #     int m, int n, float* A, int lda, int* Lwork)
        "cusolverDnSgetrf_bufferSize": {
            "args": {
                "handle": cusolverDnHandle_t,
                "m": c_int,
                "n": c_int,
                "A": POINTER(c_float),
                "lda": c_int,
                "Lwork": POINTER(c_int),
            },
            "restype": c_int,
        },

        # cusolverStatus_t cusolverDnDgetrf_bufferSize(... the double form ...)
        "cusolverDnDgetrf_bufferSize": {
            "args": {
                "handle": cusolverDnHandle_t,
                "m": c_int,
                "n": c_int,
                "A": POINTER(c_double),
                "lda": c_int,
                "Lwork": POINTER(c_int),
            },
            "restype": c_int,
        },

        # cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n,
        #     float* A, int lda, float* Workspace, int Lwork, int* devInfo)
        "cusolverDnSgetrf": {
            "args": {
                "handle": cusolverDnHandle_t,
                "m": c_int,
                "n": c_int,
                "A": POINTER(c_float),
                "lda": c_int,
                "Workspace": POINTER(c_float),
                "Lwork": c_int,
                "devInfo": POINTER(c_int),
            },
            "restype": c_int,
        },

        # cusolverStatus_t cusolverDnDgetrf(... the double form ...)
        "cusolverDnDgetrf": {
            "args": {
                "handle": cusolverDnHandle_t,
                "m": c_int,
                "n": c_int,
                "A": POINTER(c_double),
                "lda": c_int,
                "Workspace": POINTER(c_double),
                "Lwork": c_int,
                "devInfo": POINTER(c_int),
            },
            "restype": c_int,
        },

        # ---------- singular value decomposition ----------
        #
        # The workspace query takes only the shape: the legacy gesvd reads the
        # matrix again on the device, and its workspace does not depend on the
        # values.

        # cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle,
        #     int m, int n, int* lwork)
        "cusolverDnSgesvd_bufferSize": {
            "args": {
                "handle": cusolverDnHandle_t,
                "m": c_int,
                "n": c_int,
                "lwork": POINTER(c_int),
            },
            "restype": c_int,
        },

        # cusolverStatus_t cusolverDnDgesvd_bufferSize(... the double form ...)
        "cusolverDnDgesvd_bufferSize": {
            "args": {
                "handle": cusolverDnHandle_t,
                "m": c_int,
                "n": c_int,
                "lwork": POINTER(c_int),
            },
            "restype": c_int,
        },

        # cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t handle,
        #     signed char jobu, signed char jobvt, int m, int n, float* A, int lda,
        #     float* S, float* U, int ldu, float* VT, int ldvt, float* work,
        #     int lwork, float* rwork, int* devInfo)
        "cusolverDnSgesvd": {
            "args": {
                "handle": cusolverDnHandle_t,
                "jobu": c_char,
                "jobvt": c_char,
                "m": c_int,
                "n": c_int,
                "A": POINTER(c_float),
                "lda": c_int,
                "S": POINTER(c_float),
                "U": POINTER(c_float),
                "ldu": c_int,
                "VT": POINTER(c_float),
                "ldvt": c_int,
                "work": POINTER(c_float),
                "lwork": c_int,
                "rwork": POINTER(c_float),
                "devInfo": POINTER(c_int),
            },
            "restype": c_int,
        },

        # cusolverStatus_t cusolverDnDgesvd(... the double form ...)
        "cusolverDnDgesvd": {
            "args": {
                "handle": cusolverDnHandle_t,
                "jobu": c_char,
                "jobvt": c_char,
                "m": c_int,
                "n": c_int,
                "A": POINTER(c_double),
                "lda": c_int,
                "S": POINTER(c_double),
                "U": POINTER(c_double),
                "ldu": c_int,
                "VT": POINTER(c_double),
                "ldvt": c_int,
                "work": POINTER(c_double),
                "lwork": c_int,
                "rwork": POINTER(c_double),
                "devInfo": POINTER(c_int),
            },
            "restype": c_int,
        },
    }

    error_codes:Dict[int, str] = {
        cusolverStatus_t.CUSOLVER_STATUS_SUCCESS:
            "no errors.",
        cusolverStatus_t.CUSOLVER_STATUS_NOT_INITIALIZED:
            "the library was not initialized.",
        cusolverStatus_t.CUSOLVER_STATUS_ALLOC_FAILED:
            "the library could not allocate enough memory to perform the requested operation.",
        cusolverStatus_t.CUSOLVER_STATUS_INVALID_VALUE:
            "one of the parameters passed to the API call is not within an acceptable range of values.",
        cusolverStatus_t.CUSOLVER_STATUS_ARCH_MISMATCH:
            "the architecture of the device does not support the requested operation.",
        cusolverStatus_t.CUSOLVER_STATUS_MAPPING_ERROR:
            "an internal error occurred while mapping the memory of the operands.",
        cusolverStatus_t.CUSOLVER_STATUS_EXECUTION_FAILED:
            "the requested operation could not be performed on the device.",
        cusolverStatus_t.CUSOLVER_STATUS_INTERNAL_ERROR:
            "an internal error occurred.",
        cusolverStatus_t.CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            "the matrix type is not supported by the requested operation.",
        cusolverStatus_t.CUSOLVER_STATUS_NOT_SUPPORTED:
            "the requested operation is not supported on this device.",
        cusolverStatus_t.CUSOLVER_STATUS_ZERO_PIVOT:
            "the factorization encountered a zero pivot.",
        cusolverStatus_t.CUSOLVER_STATUS_INVALID_LICENSE:
            "the license for the library is not valid on this device.",
        cusolverStatus_t.CUSOLVER_STATUS_INVALID_WORKSPACE:
            "the workspace is not large enough for the requested operation.",
    }

    no_need_check_error_funcs:List[str] = []


class CUSOLVER(Library):
    """cuSOLVER's entry points."""

    _library_name = "cuSOLVER"
    _dll_name = "cusolver64_11.dll"
    _so_name = "libcusolver.so"
    _signatures = CUSOLVERInfo.func_signatures
    _error_codes = CUSOLVERInfo.error_codes
    _no_error_check = CUSOLVERInfo.no_need_check_error_funcs


__all__ = [
    "CUSOLVER",
    "CUSOLVERInfo",
    "cusolverStatus_t",
    "cusolverDnHandle_t",
    "cusolverDnParams_t",
    "cudaStream_t",
]
