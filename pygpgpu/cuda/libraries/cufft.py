"""cuFFT -- fast fourier transforms, bound with ctypes.

The shape is the CUDA driver's shape: a metaclass holds the DLL lazily, a ``Func``
wrapper binds the argument types and the return type on first use, ``init()``
pre-binds the entry points, and a returned result code becomes a ``RuntimeError``
carrying cuFFT's own description of it.

Two things about cuFFT are worth knowing before using it, because they are not what
the names suggest:

* **A plan is tied to one transform direction.** A plan made for a real-to-complex
  transform refuses a complex-to-real execution (and the other way round) with
  ``CUFFT_INVALID_TYPE``, so the two directions need two plans.
* **The real-to-complex output is full complex, not packed.** An ``n``-point
  real-to-complex transform writes ``n/2 + 1`` complex numbers, each a real and an
  imaginary part, not the packed "half-complex" layout the older API described.
"""
from __future__ import annotations

from ctypes import c_int, c_size_t, c_void_p, POINTER
from typing import Dict, List

from ...constants import IntEnum
from ._base import Library


class cufftResult(IntEnum):
    """cuFFT's own result codes.

    These are cuFFT's numbering, not the CUDA driver's.
    """
    CUFFT_SUCCESS           = 0x0
    CUFFT_INVALID_PLAN      = 0x1
    CUFFT_ALLOC_FAILED      = 0x2
    CUFFT_INVALID_TYPE      = 0x3
    CUFFT_INVALID_VALUE     = 0x4
    CUFFT_INTERNAL_ERROR    = 0x5
    CUFFT_EXEC_FAILED       = 0x6
    CUFFT_SETUP_FAILED      = 0x7
    CUFFT_INVALID_SIZE      = 0x8
    CUFFT_INVALID_DEVICE    = 0xB
    CUFFT_NOT_IMPLEMENTED   = 0xE
    CUFFT_LICENSE_ERROR     = 0xF
    CUFFT_NOT_SUPPORTED     = 0x10


class cufftType(IntEnum):
    """The transform a plan performs."""
    CUFFT_R2C = 0x2A
    CUFFT_C2R = 0x2C
    CUFFT_C2C = 0x29
    CUFFT_D2Z = 0x2A
    CUFFT_Z2D = 0x2C
    CUFFT_Z2Z = 0x29


# A plan is an integer handle, not a pointer.
cufftHandle = c_int
ptr_cufftHandle = POINTER(cufftHandle)

# The stream a plan's executions are ordered on.
cudaStream_t = c_void_p


class CUFFTInfo:
    """cuFFT's signature table and its result codes."""

    func_signatures:Dict[str, Dict[str, object]] = {
        # cufftResult cufftCreate(cufftHandle *handle)
        "cufftCreate": {
            "args": {
                "handle": ptr_cufftHandle,
            },
            "restype": c_int,
        },

        # cufftResult cufftDestroy(cufftHandle plan)
        "cufftDestroy": {
            "args": {
                "plan": cufftHandle,
            },
            "restype": c_int,
        },

        # cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch)
        "cufftPlan1d": {
            "args": {
                "plan": ptr_cufftHandle,
                "nx": c_int,
                "type": c_int,
                "batch": c_int,
            },
            "restype": c_int,
        },

        # cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type,
        #     int batch, size_t *workSize)
        "cufftMakePlan1d": {
            "args": {
                "plan": cufftHandle,
                "nx": c_int,
                "type": c_int,
                "batch": c_int,
                "workSize": POINTER(c_size_t),
            },
            "restype": c_int,
        },

        # cufftResult cufftGetSize1d(cufftHandle plan, int *nx, int *batch, size_t *workSize)
        "cufftGetSize1d": {
            "args": {
                "plan": cufftHandle,
                "nx": POINTER(c_int),
                "batch": POINTER(c_int),
                "workSize": POINTER(c_size_t),
            },
            "restype": c_int,
        },

        # cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream)
        "cufftSetStream": {
            "args": {
                "plan": cufftHandle,
                "stream": cudaStream_t,
            },
            "restype": c_int,
        },

        # cufftResult cufftExecR2C(cufftHandle plan, const float *input, cufftComplex *output)
        "cufftExecR2C": {
            "args": {
                "plan": cufftHandle,
                "input": c_void_p,
                "output": c_void_p,
            },
            "restype": c_int,
        },

        # cufftResult cufftExecC2R(cufftHandle plan, const cufftComplex *input, float *output)
        "cufftExecC2R": {
            "args": {
                "plan": cufftHandle,
                "input": c_void_p,
                "output": c_void_p,
            },
            "restype": c_int,
        },

        # cufftResult cufftExecC2C(cufftHandle plan, const cufftComplex *input,
        #     cufftComplex *output)
        "cufftExecC2C": {
            "args": {
                "plan": cufftHandle,
                "input": c_void_p,
                "output": c_void_p,
            },
            "restype": c_int,
        },

        # cufftResult cufftGetVersion(int *version)
        "cufftGetVersion": {
            "args": {
                "version": POINTER(c_int),
            },
            "restype": c_int,
        },
    }

    error_codes:Dict[int, str] = {
        cufftResult.CUFFT_SUCCESS:
            "no errors.",
        cufftResult.CUFFT_INVALID_PLAN:
            "the plan is invalid (it was destroyed, or it was made for another transform).",
        cufftResult.CUFFT_ALLOC_FAILED:
            "the library could not allocate enough memory to perform the requested operation.",
        cufftResult.CUFFT_INVALID_TYPE:
            "the transform type does not match the one the plan was made for.",
        cufftResult.CUFFT_INVALID_VALUE:
            "one of the parameters passed to the API call is not within an acceptable range of values.",
        cufftResult.CUFFT_INTERNAL_ERROR:
            "an internal error occurred.",
        cufftResult.CUFFT_EXEC_FAILED:
            "the execution of the transform failed.",
        cufftResult.CUFFT_SETUP_FAILED:
            "the plan could not be set up.",
        cufftResult.CUFFT_INVALID_SIZE:
            "the size of the transform is not supported (it must be a power of two, two, three, five or seven times a power of two).",
        cufftResult.CUFFT_INVALID_DEVICE:
            "the device is not supported.",
        cufftResult.CUFFT_NOT_IMPLEMENTED:
            "the transform is not implemented.",
        cufftResult.CUFFT_LICENSE_ERROR:
            "the license for the library is not valid on this device.",
        cufftResult.CUFFT_NOT_SUPPORTED:
            "the transform is not supported on this device.",
    }

    no_need_check_error_funcs:List[str] = []


class CUFFT(Library):
    """cuFFT's entry points."""

    _library_name = "cuFFT"
    _dll_name = "cufft64_11.dll"
    _so_name = "libcufft.so"
    _signatures = CUFFTInfo.func_signatures
    _error_codes = CUFFTInfo.error_codes
    _no_error_check = CUFFTInfo.no_need_check_error_funcs


__all__ = [
    "CUFFT",
    "CUFFTInfo",
    "cufftResult",
    "cufftType",
    "cufftHandle",
    "cudaStream_t",
]
