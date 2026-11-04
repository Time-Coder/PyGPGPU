"""hipSOLVER -- the dense linear solves and factorizations, bound with ctypes.

The signature table is written here rather than derived from cuSOLVER's, for three
reasons:

* ``hipsolverGetVersion`` does not exist -- ROCm's hipSOLVER declares no version query at
  all, where cuSOLVER's table lists one;
* the cuSOLVER table here disagrees with cuSOLVER's own API for ``getrf`` (it lists a
  ``lwork`` the API does not take and leaves out ``devIpiv``), so deriving from it would
  carry that shape into hipSOLVER;
* ``hipsolverDnCreateParams``/``DestroyParams`` exist in HIP (the cuSOLVER table here
  omits them), so they are bound.

The handle is an opaque pointer, and the ``uplo`` argument is hipBLAS's
``hipblasFillMode_t`` -- the same enum hipBLAS uses, **and not cuBLAS's values**:
``HIPBLAS_FILL_MODE_LOWER`` is 122 where ``CUBLAS_FILL_MODE_LOWER`` is 0, so passing
cuBLAS's value makes every factorization answer ``INVALID_ENUM``. The declarations these
come from are ROCm's ``include/hipsolver/hipsolver.h`` and ``internal/hipsolver-dense.h``.

**This ROCm install ships no rocSOLVER kernels, and the calls fail silently.** The DLL
loads, every entry point binds, `hipsolverDnCreate` succeeds, and a factorization then
answers **success with `devInfo == 0` while leaving the matrix untouched** -- there is no
runtime directory for rocSOLVER beside the DLL (`bin/rocsolver` does not exist, while
`bin/rocblas/library` does). A caller therefore gets a wrong answer with no error, which
is worse than a failure: check the result, or ask
``hipsolverDnSgetrf_bufferSize`` for a matrix that needs workspace and treat a zero as
"this library is not usable here". `tests/test_hip_libraries.py` probes exactly that and
skips its hipSOLVER case with the reason.
"""
from __future__ import annotations

from ctypes import POINTER, c_char, c_double, c_float, c_int, c_void_p
from typing import Any, Dict

from ...libraries import Library, resolve_dll
from ._enums import hipsolverStatus_t
from .hipblas import HIP_DLL_ROOTS

__all__ = ["HIPSOLVER"]


# The handle and the parameters object. Both opaque pointers.
hipsolverHandle_t = c_void_p
ptr_hipsolverHandle_t = POINTER(hipsolverHandle_t)
hipsolverDnParams_t = c_void_p
ptr_hipsolverDnParams_t = POINTER(hipsolverDnParams_t)


_HIPSOLVER_STATUS_MESSAGES:Dict[str, str] = {
    "HIPSOLVER_STATUS_SUCCESS": "no errors.",
    "HIPSOLVER_STATUS_NOT_INITIALIZED": "the library was not initialized.",
    "HIPSOLVER_STATUS_ALLOC_FAILED": "the library could not allocate enough memory.",
    "HIPSOLVER_STATUS_INVALID_VALUE":
        "one of the parameters is not within an acceptable range.",
    "HIPSOLVER_STATUS_MAPPING_ERROR":
        "an internal error occurred while mapping the memory of the operands.",
    "HIPSOLVER_STATUS_EXECUTION_FAILED":
        "the requested operation could not be performed on the device.",
    "HIPSOLVER_STATUS_INTERNAL_ERROR": "an internal error occurred.",
    "HIPSOLVER_STATUS_NOT_SUPPORTED": "the requested operation is not supported.",
    "HIPSOLVER_STATUS_ARCH_MISMATCH":
        "the architecture of the device does not support the requested operation.",
    "HIPSOLVER_STATUS_HANDLE_IS_NULLPTR": "the library handle is a null pointer.",
    "HIPSOLVER_STATUS_INVALID_ENUM": "an unsupported enum value was passed to the function.",
    "HIPSOLVER_STATUS_UNKNOWN": "the back-end returned an unsupported status code.",
    "HIPSOLVER_STATUS_ZERO_PIVOT": "a zero pivot was encountered during the factorization.",
    "HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED": "the matrix type is not supported.",
}


def _status_messages()->Dict[int, str]:
    return {int(member.value): _HIPSOLVER_STATUS_MESSAGES.get(
        member.name, f"{member.name.replace('_', ' ').lower()}.")
        for member in hipsolverStatus_t}


class HIPSOLVER(Library):
    """hipSOLVER's entry points.

    The class is used through its metaclass, so ``HIPSOLVER.hipsolverDnSpotrf`` is the
    bound function and ``HIPSOLVER.hipsolverDnCreate(byref(handle))`` is the call.
    """

    _library_name = "hipSOLVER"
    _dll_name = "hipsolver.dll"
    _so_name = "libhipsolver.so"

    _signatures:Dict[str, Dict[str, Any]] = {
        # hipsolverStatus_t hipsolverDnCreate(hipsolverHandle_t* handle)
        "hipsolverDnCreate": {
            "args": {"handle": ptr_hipsolverHandle_t},
            "restype": c_int,
        },
        # hipsolverStatus_t hipsolverDnDestroy(hipsolverHandle_t handle)
        "hipsolverDnDestroy": {
            "args": {"handle": hipsolverHandle_t},
            "restype": c_int,
        },
        # hipsolverStatus_t hipsolverDnSetStream(hipsolverHandle_t handle,
        #     hipStream_t streamId)
        "hipsolverDnSetStream": {
            "args": {"handle": hipsolverHandle_t, "streamId": c_void_p},
            "restype": c_int,
        },
        # hipsolverStatus_t hipsolverDnCreateParams(hipsolverDnParams_t* params)
        "hipsolverDnCreateParams": {
            "args": {"params": ptr_hipsolverDnParams_t},
            "restype": c_int,
        },
        # hipsolverStatus_t hipsolverDnDestroyParams(hipsolverDnParams_t params)
        "hipsolverDnDestroyParams": {
            "args": {"params": hipsolverDnParams_t},
            "restype": c_int,
        },

        # Cholesky: hipsolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, int* lwork)
        "hipsolverDnSpotrf_bufferSize": {
            "args": {"handle": hipsolverHandle_t, "uplo": c_int, "n": c_int,
                     "A": POINTER(c_float), "lda": c_int, "lwork": POINTER(c_int)},
            "restype": c_int,
        },
        "hipsolverDnDpotrf_bufferSize": {
            "args": {"handle": hipsolverHandle_t, "uplo": c_int, "n": c_int,
                     "A": POINTER(c_double), "lda": c_int, "lwork": POINTER(c_int)},
            "restype": c_int,
        },
        # hipsolverDnSpotrf(handle, uplo, n, A, lda, work, lwork, int* devInfo)
        "hipsolverDnSpotrf": {
            "args": {"handle": hipsolverHandle_t, "uplo": c_int, "n": c_int,
                     "A": POINTER(c_float), "lda": c_int, "work": POINTER(c_float),
                     "lwork": c_int, "devInfo": POINTER(c_int)},
            "restype": c_int,
        },
        "hipsolverDnDpotrf": {
            "args": {"handle": hipsolverHandle_t, "uplo": c_int, "n": c_int,
                     "A": POINTER(c_double), "lda": c_int, "work": POINTER(c_double),
                     "lwork": c_int, "devInfo": POINTER(c_int)},
            "restype": c_int,
        },

        # LU: hipsolverDnSgetrf_bufferSize(handle, m, n, A, lda, int* lwork)
        "hipsolverDnSgetrf_bufferSize": {
            "args": {"handle": hipsolverHandle_t, "m": c_int, "n": c_int,
                     "A": POINTER(c_float), "lda": c_int, "lwork": POINTER(c_int)},
            "restype": c_int,
        },
        "hipsolverDnDgetrf_bufferSize": {
            "args": {"handle": hipsolverHandle_t, "m": c_int, "n": c_int,
                     "A": POINTER(c_double), "lda": c_int, "lwork": POINTER(c_int)},
            "restype": c_int,
        },
        # hipsolverDnSgetrf(handle, m, n, A, lda, work, int* devIpiv, int* devInfo)
        #
        # No `lwork`: the workspace size comes from the _bufferSize query, and the pivot
        # array is a real argument. cuSOLVER's table here lists neither correctly.
        "hipsolverDnSgetrf": {
            "args": {"handle": hipsolverHandle_t, "m": c_int, "n": c_int,
                     "A": POINTER(c_float), "lda": c_int, "work": POINTER(c_float),
                     "devIpiv": POINTER(c_int), "devInfo": POINTER(c_int)},
            "restype": c_int,
        },
        "hipsolverDnDgetrf": {
            "args": {"handle": hipsolverHandle_t, "m": c_int, "n": c_int,
                     "A": POINTER(c_double), "lda": c_int, "work": POINTER(c_double),
                     "devIpiv": POINTER(c_int), "devInfo": POINTER(c_int)},
            "restype": c_int,
        },

        # SVD: hipsolverDnSgesvd_bufferSize(handle, m, n, int* lwork)
        "hipsolverDnSgesvd_bufferSize": {
            "args": {"handle": hipsolverHandle_t, "m": c_int, "n": c_int,
                     "lwork": POINTER(c_int)},
            "restype": c_int,
        },
        "hipsolverDnDgesvd_bufferSize": {
            "args": {"handle": hipsolverHandle_t, "m": c_int, "n": c_int,
                     "lwork": POINTER(c_int)},
            "restype": c_int,
        },
        # hipsolverDnSgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv,
        #     work, lwork, rwork, int* devInfo)
        #
        # `jobu`/`jobv` are `signed char`, not int.
        "hipsolverDnSgesvd": {
            "args": {"handle": hipsolverHandle_t, "jobu": c_char, "jobv": c_char,
                     "m": c_int, "n": c_int, "A": POINTER(c_float), "lda": c_int,
                     "S": POINTER(c_float), "U": POINTER(c_float), "ldu": c_int,
                     "V": POINTER(c_float), "ldv": c_int, "work": POINTER(c_float),
                     "lwork": c_int, "rwork": POINTER(c_float),
                     "devInfo": POINTER(c_int)},
            "restype": c_int,
        },
        "hipsolverDnDgesvd": {
            "args": {"handle": hipsolverHandle_t, "jobu": c_char, "jobv": c_char,
                     "m": c_int, "n": c_int, "A": POINTER(c_double), "lda": c_int,
                     "S": POINTER(c_double), "U": POINTER(c_double), "ldu": c_int,
                     "V": POINTER(c_double), "ldv": c_int, "work": POINTER(c_double),
                     "lwork": c_int, "rwork": POINTER(c_double),
                     "devInfo": POINTER(c_int)},
            "restype": c_int,
        },
    }

    _error_codes = _status_messages()

    @classmethod
    def dll(cls)->Any:
        """The library's DLL, loaded on first use."""
        if cls._dll_instance is None:
            from ctypes import CDLL

            cls._dll_instance = CDLL(resolve_dll(cls._dll_name, cls._so_name,
                                                 HIP_DLL_ROOTS))
        return cls._dll_instance
