"""hipFFT -- the fast fourier transforms, bound with ctypes.

The shape is the CUDA libraries' shape, and the **signature table is written here** rather
than derived from cuFFT's, because two things differ in a way ctypes cares about:

* **the plan handle is a pointer.** HIP spells it `struct hipfftHandle_t*` on the AMD
  platform and `int` on NVIDIA's (`#ifdef __HIP_PLATFORM_NVIDIA__`), while cuFFT's
  `cufftHandle` is an `int`. Reusing cuFFT's table therefore passes a 4-byte integer where
  the library reads an 8-byte pointer -- which is an access violation inside the first
  call, not a wrong answer;
* **an entry point that seems to match does not**: `hipfftExecC2C` takes the transform
  direction as its fourth argument (the cuFFT table here lists three), and
  `hipfftGetSize1d` takes `(plan, nx, type, batch, workSize)` (the cuFFT table lists four,
  without the type).

The declarations these come from are ROCm's `include/hipfft/hipfft.h`.
"""
from __future__ import annotations

import ctypes
from ctypes import POINTER, c_int, c_size_t, c_void_p
from typing import Dict

from ...libraries import Library, resolve_dll
from ._enums import hipfftResult
from .hipblas import HIP_DLL_ROOTS

__all__ = ["HIPFFT", "hipfftHandle"]


# The plan handle. A pointer on the AMD platform: the header defines it as
# `typedef struct hipfftHandle_t* hipfftHandle` unless the NVIDIA platform is selected,
# where it is an `int` for cuFFT compatibility. The DLL is ROCm's, so it is a pointer.
hipfftHandle = c_void_p
ptr_hipfftHandle = POINTER(hipfftHandle)


# The result codes and the descriptions, by the enum's own member names.
_HIPFFT_STATUS_MESSAGES:Dict[str, str] = {
    "HIPFFT_SUCCESS": "no errors.",
    "HIPFFT_INVALID_PLAN": "the plan is not valid.",
    "HIPFFT_ALLOC_FAILED": "the library could not allocate enough memory.",
    "HIPFFT_INVALID_TYPE": "the transform type is not valid.",
    "HIPFFT_INVALID_VALUE": "one of the parameters is not within an acceptable range.",
    "HIPFFT_INTERNAL_ERROR": "an internal error occurred.",
    "HIPFFT_EXEC_FAILED": "the transform could not be executed on the device.",
    "HIPFFT_SETUP_FAILED": "the library could not be initialized.",
    "HIPFFT_INVALID_SIZE": "the transform size is not valid.",
    "HIPFFT_UNALIGNED_DATA": "the data is not aligned as the transform requires.",
    "HIPFFT_INCOMPLETE_PARAMETER_LIST": "one of the parameters is missing.",
    "HIPFFT_INVALID_DEVICE": "the device is not valid.",
    "HIPFFT_PARSE_ERROR": "an internal parse error occurred.",
    "HIPFFT_NO_WORKSPACE": "no workspace was set.",
    "HIPFFT_NOT_IMPLEMENTED": "the requested operation is not implemented.",
    "HIPFFT_NOT_SUPPORTED": "the requested operation is not supported.",
}


def _status_messages()->Dict[int, str]:
    return {int(member.value): _HIPFFT_STATUS_MESSAGES.get(
        member.name, f"{member.name.replace('_', ' ').lower()}.")
        for member in hipfftResult}


class HIPFFT(Library):
    """hipFFT's entry points.

    The class is used through its metaclass, so ``HIPFFT.hipfftPlan1d`` is the bound
    function and ``HIPFFT.hipfftExecR2C(plan, in, out)`` is the call.
    """

    _library_name = "hipFFT"
    _dll_name = "hipfft.dll"
    _so_name = "libhipfft.so"

    _signatures:Dict[str, Dict[str, object]] = {
        # hipfftResult hipfftCreate(hipfftHandle* plan)
        "hipfftCreate": {
            "args": {"plan": ptr_hipfftHandle},
            "restype": c_int,
        },
        # hipfftResult hipfftDestroy(hipfftHandle plan)
        "hipfftDestroy": {
            "args": {"plan": hipfftHandle},
            "restype": c_int,
        },
        # hipfftResult hipfftPlan1d(hipfftHandle* plan, int nx, hipfftType type,
        #     int batch)
        "hipfftPlan1d": {
            "args": {"plan": ptr_hipfftHandle, "nx": c_int, "type": c_int,
                     "batch": c_int},
            "restype": c_int,
        },
        # hipfftResult hipfftMakePlan1d(hipfftHandle plan, int nx, hipfftType type,
        #     int batch, size_t* workSize)
        "hipfftMakePlan1d": {
            "args": {"plan": hipfftHandle, "nx": c_int, "type": c_int, "batch": c_int,
                     "workSize": POINTER(c_size_t)},
            "restype": c_int,
        },
        # hipfftResult hipfftGetSize1d(hipfftHandle plan, int nx, hipfftType type,
        #     int batch, size_t* workSize)
        "hipfftGetSize1d": {
            "args": {"plan": hipfftHandle, "nx": c_int, "type": c_int, "batch": c_int,
                     "workSize": POINTER(c_size_t)},
            "restype": c_int,
        },
        # hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream)
        "hipfftSetStream": {
            "args": {"plan": hipfftHandle, "stream": c_void_p},
            "restype": c_int,
        },
        # hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal* idata,
        #     hipfftComplex* odata)
        #
        # The buffers are device pointers; `hipfftReal` is a float and `hipfftComplex` two
        # of them, so a caller passes the address either way.
        "hipfftExecR2C": {
            "args": {"plan": hipfftHandle, "idata": c_void_p, "odata": c_void_p},
            "restype": c_int,
        },
        # hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex* idata,
        #     hipfftReal* odata)
        "hipfftExecC2R": {
            "args": {"plan": hipfftHandle, "idata": c_void_p, "odata": c_void_p},
            "restype": c_int,
        },
        # hipfftResult hipfftExecC2C(hipfftHandle plan, hipfftComplex* idata,
        #     hipfftComplex* odata, int direction)
        #
        # The direction is the fourth argument -- HIPFFT_FORWARD is -1 and
        # HIPFFT_BACKWARD is 1.
        "hipfftExecC2C": {
            "args": {"plan": hipfftHandle, "idata": c_void_p, "odata": c_void_p,
                     "direction": c_int},
            "restype": c_int,
        },
        # hipfftResult hipfftGetVersion(int* version)
        "hipfftGetVersion": {
            "args": {"version": POINTER(c_int)},
            "restype": c_int,
        },
    }

    _error_codes = _status_messages()
    # The version is a version, not a status: a non-zero one is not an error.
    _no_error_check = ["hipfftGetVersion"]

    @classmethod
    def dll(cls)->ctypes.CDLL:
        """The library's DLL, loaded on first use."""
        if cls._dll_instance is None:
            from ctypes import CDLL

            cls._dll_instance = CDLL(resolve_dll(cls._dll_name, cls._so_name,
                                                 HIP_DLL_ROOTS))
        return cls._dll_instance
