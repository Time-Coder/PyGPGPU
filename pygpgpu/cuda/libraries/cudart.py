"""The CUDA runtime (cudart) -- allocation and host/device copies, bound with ctypes.

The shape is the CUDA driver's shape: a metaclass holds the DLL lazily, a ``Func``
wrapper binds the argument types and the return type on first use, ``init()``
pre-binds the entry points, and a returned error code becomes a ``RuntimeError``
carrying the runtime's own description of it.

**The CUDA backend already has a device copy.** The driver backend
(``pygpgpu/cuda/driver/cuda.py``) binds ``cuMemAlloc_v2``, ``cuMemcpyHtoD_v2`` and
``cuMemcpyDtoH_v2``, and the ``oop`` layer builds its buffers on them -- so the
CUDA backend does not need cudart to move data. What cudart adds is the runtime's
own view of the device: ``cudaSetDevice``, the device count and the properties, and
the runtime's allocation and copy entry points for a caller that wants to use the
runtime API directly.

The two APIs share one context (the runtime's primary context is the driver's primary
context), so a pointer allocated through one is usable by the other. That was measured
on this machine: a buffer allocated with ``cuMemAlloc_v2`` and copied with
``cudaMemcpy`` behaves exactly like a buffer allocated with ``cudaMalloc``.
"""
from __future__ import annotations

from ctypes import c_char_p, c_int, c_size_t, c_void_p, POINTER
from typing import Any, Dict, List

from ...constants import IntEnum
from ._base import Library


class cudaError_t(IntEnum):
    """The CUDA runtime's own error codes.

    These are the runtime's numbering, which is also the driver's: the driver API and
    the runtime API share one error space (``cudaErrorMemoryAllocation`` is
    ``CUDA_ERROR_OUT_OF_MEMORY``), so one table serves both.
    """
    cudaSuccess                           = 0
    cudaErrorInvalidValue                 = 1
    cudaErrorMemoryAllocation             = 2
    cudaErrorInitializationError          = 3
    cudaErrorCudartUnloading              = 4
    cudaErrorProfilerDisabled             = 5
    cudaErrorProfilerNotInitialized       = 6
    cudaErrorProfilerAlreadyStarted       = 7
    cudaErrorProfilerAlreadyStopped       = 8
    cudaErrorInvalidConfiguration         = 9
    cudaErrorInvalidPitchValue            = 12
    cudaErrorInvalidSymbol                = 13
    cudaErrorInvalidHostPointer           = 16
    cudaErrorInvalidDevicePointer         = 17
    cudaErrorInvalidTexture               = 18
    cudaErrorInvalidTextureBinding        = 19
    cudaErrorInvalidChannelDescriptor     = 20
    cudaErrorInvalidMemcpyDirection       = 21
    cudaErrorAddressOfConstant            = 22
    cudaErrorTextureFetchFailed           = 23
    cudaErrorTextureNotBound              = 24
    cudaErrorSynchronizationError         = 25
    cudaErrorInvalidFilterSetting         = 26
    cudaErrorInvalidNormSetting           = 27
    cudaErrorMixedDeviceExecution         = 28
    cudaErrorNotYetImplemented            = 31
    cudaErrorMemoryValueTooLarge          = 32
    cudaErrorStubLibrary                  = 34
    cudaErrorInsufficientDriver           = 35
    cudaErrorCallRequiresNewerDriver      = 36
    cudaErrorInvalidSurface               = 37
    cudaErrorDuplicateVariableName        = 43
    cudaErrorDuplicateTextureName         = 44
    cudaErrorDuplicateSurfaceName         = 45
    cudaErrorDevicesUnavailable           = 46
    cudaErrorIncompatibleDriverContext    = 49
    cudaErrorMissingConfiguration         = 52
    cudaErrorPriorLaunchFailure           = 53
    cudaErrorLaunchMaxDepthExceeded       = 65
    cudaErrorLaunchFileScopedTex          = 66
    cudaErrorLaunchFileScopedSurf         = 67
    cudaErrorSyncDepthExceeded            = 68
    cudaErrorLaunchPendingCountExceeded   = 69
    cudaErrorInvalidDeviceFunction        = 98
    cudaErrorNoDevice                     = 100
    cudaErrorInvalidDevice                = 101
    cudaErrorDeviceNotLicensed            = 102
    cudaErrorSoftwareValidityNotEstablished = 103
    cudaErrorStartupFailure               = 127
    cudaErrorInvalidKernelImage           = 200
    cudaErrorDeviceUninitialized          = 201
    cudaErrorMapBufferObjectFailed        = 205
    cudaErrorUnmapBufferObjectFailed      = 206
    cudaErrorArrayIsMapped                = 207
    cudaErrorAlreadyMapped                = 208
    cudaErrorNoKernelImageForDevice       = 209
    cudaErrorAlreadyAcquired              = 210
    cudaErrorNotMapped                    = 211
    cudaErrorNotMappedAsArray             = 212
    cudaErrorNotMappedAsPointer           = 213
    cudaErrorECCUncorrectable             = 214
    cudaErrorUnsupportedLimit             = 215
    cudaErrorDeviceAlreadyInUse           = 216
    cudaErrorPeerAccessUnsupported        = 217
    cudaErrorInvalidPtx                   = 218
    cudaErrorInvalidGraphicsContext       = 219
    cudaErrorNvlinkUncorrectable          = 220
    cudaErrorJitCompilerNotFound          = 221
    cudaErrorUnsupportedPtxVersion        = 222
    cudaErrorJitCompilationDisabled       = 223
    cudaErrorUnsupportedExecAffinity      = 224
    cudaErrorUnsupportedDevSideSync       = 225
    cudaErrorInvalidSource                = 300
    cudaErrorFileNotFound                 = 301
    cudaErrorSharedObjectSymbolNotFound   = 302
    cudaErrorSharedObjectInitFailed       = 303
    cudaErrorOperatingSystem               = 304
    cudaErrorInvalidResourceHandle        = 400
    cudaErrorIllegalState                 = 401
    cudaErrorLossyQuery                   = 402
    cudaErrorSymbolNotFound               = 500
    cudaErrorNotReady                     = 600
    cudaErrorIllegalAddress               = 700
    cudaErrorLaunchOutOfResources         = 701
    cudaErrorLaunchTimeout                = 702
    cudaErrorLaunchIncompatibleTexturing  = 703
    cudaErrorIllegalInstruction            = 715
    cudaErrorMisalignedAddress            = 716
    cudaErrorInvalidAddressSpace          = 717
    cudaErrorInvalidPc                    = 718
    cudaErrorLaunchFailure                = 719
    cudaErrorNotSupported                 = 801
    cudaErrorSystemNotReady               = 802
    cudaErrorSystemDriverMismatch         = 803
    cudaErrorCompatNotSupportedOnDevice   = 804
    cudaErrorStreamCaptureUnsupported     = 900
    cudaErrorTimeout                      = 909
    cudaErrorUnknown                      = 999


class cudaMemcpyKind(IntEnum):
    """Which way a copy goes."""
    cudaMemcpyHostToHost     = 0
    cudaMemcpyHostToDevice   = 1
    cudaMemcpyDeviceToHost   = 2
    cudaMemcpyDeviceToDevice = 3
    cudaMemcpyDefault        = 4


class cudaDeviceAttr(IntEnum):
    """The device properties the runtime reports."""
    cudaDevAttrClockRate           = 13
    cudaDevAttrComputeCapabilityMajor = 75
    cudaDevAttrComputeCapabilityMinor = 76
    cudaDevAttrGlobalMemoryBusWidth = 37
    cudaDevAttrMultiProcessorCount = 16
    cudaDevAttrMaxThreadsPerBlock  = 1
    cudaDevAttrMemoryClockRate     = 36
    cudaDevAttrTotalMemory         = 2
    cudaDevAttrWarpSize            = 10


# A device pointer is an opaque address; the runtime's malloc hands one out.
cudaDeviceptr = c_void_p
ptr_cudaDeviceptr = POINTER(cudaDeviceptr)

# A stream is an opaque pointer.
cudaStream_t = c_void_p
ptr_cudaStream_t = POINTER(cudaStream_t)


class CUDARTInfo:
    """The CUDA runtime's signature table and its error codes."""

    func_signatures:Dict[str, Dict[str, Any]] = {
        # cudaError_t cudaMalloc(void **devPtr, size_t size)
        "cudaMalloc": {
            "args": {
                "devPtr": ptr_cudaDeviceptr,
                "size": c_size_t,
            },
            "restype": c_int,
        },

        # cudaError_t cudaFree(void *devPtr)
        "cudaFree": {
            "args": {
                "devPtr": c_void_p,
            },
            "restype": c_int,
        },

        # cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
        #     enum cudaMemcpyKind kind)
        "cudaMemcpy": {
            "args": {
                "dst": c_void_p,
                "src": c_void_p,
                "count": c_size_t,
                "kind": c_int,
            },
            "restype": c_int,
        },

        # cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
        #     enum cudaMemcpyKind kind, cudaStream_t stream)
        "cudaMemcpyAsync": {
            "args": {
                "dst": c_void_p,
                "src": c_void_p,
                "count": c_size_t,
                "kind": c_int,
                "stream": cudaStream_t,
            },
            "restype": c_int,
        },

        # cudaError_t cudaMemset(void *devPtr, int value, size_t count)
        "cudaMemset": {
            "args": {
                "devPtr": c_void_p,
                "value": c_int,
                "count": c_size_t,
            },
            "restype": c_int,
        },

        # cudaError_t cudaSetDevice(int device)
        "cudaSetDevice": {
            "args": {
                "device": c_int,
            },
            "restype": c_int,
        },

        # cudaError_t cudaGetDevice(int *device)
        "cudaGetDevice": {
            "args": {
                "device": POINTER(c_int),
            },
            "restype": c_int,
        },

        # cudaError_t cudaGetDeviceCount(int *count)
        "cudaGetDeviceCount": {
            "args": {
                "count": POINTER(c_int),
            },
            "restype": c_int,
        },

        # cudaError_t cudaDeviceSynchronize(void)
        "cudaDeviceSynchronize": {
            "args": {},
            "restype": c_int,
        },

        # cudaError_t cudaStreamSynchronize(cudaStream_t stream)
        "cudaStreamSynchronize": {
            "args": {
                "stream": cudaStream_t,
            },
            "restype": c_int,
        },

        # cudaError_t cudaStreamCreate(cudaStream_t *pStream)
        "cudaStreamCreate": {
            "args": {
                "pStream": ptr_cudaStream_t,
            },
            "restype": c_int,
        },

        # cudaError_t cudaStreamDestroy(cudaStream_t stream)
        "cudaStreamDestroy": {
            "args": {
                "stream": cudaStream_t,
            },
            "restype": c_int,
        },

        # cudaError_t cudaRuntimeGetVersion(int *version)
        "cudaRuntimeGetVersion": {
            "args": {
                "version": POINTER(c_int),
            },
            "restype": c_int,
        },

        # cudaError_t cudaDriverGetVersion(int *version)
        "cudaDriverGetVersion": {
            "args": {
                "version": POINTER(c_int),
            },
            "restype": c_int,
        },

        # const char* cudaGetErrorString(cudaError_t error)
        #
        # The one function whose return value is a message rather than a status, so
        # it is exempt from the error check.
        "cudaGetErrorString": {
            "args": {
                "error": c_int,
            },
            "restype": c_char_p,
        },
    }

    error_codes:Dict[int, str] = {
        cudaError_t.cudaSuccess:
            "no errors.",
        cudaError_t.cudaErrorInvalidValue:
            "one of the parameters passed to the API call is not within an acceptable range of values.",
        cudaError_t.cudaErrorMemoryAllocation:
            "the API call failed because it was unable to allocate enough memory.",
        cudaError_t.cudaErrorInitializationError:
            "the driver could not be initialized.",
        cudaError_t.cudaErrorInvalidConfiguration:
            "the device has an invalid configuration.",
        cudaError_t.cudaErrorInvalidDevicePointer:
            "the device pointer is not valid.",
        cudaError_t.cudaErrorInvalidMemcpyDirection:
            "the copy direction is not supported for the two pointers.",
        cudaError_t.cudaErrorInvalidResourceHandle:
            "a resource handle is not valid.",
        cudaError_t.cudaErrorIllegalState:
            "a resource required by the API call is not in a valid state.",
        cudaError_t.cudaErrorNotReady:
            "an asynchronous operation has not completed yet.",
        cudaError_t.cudaErrorIllegalAddress:
            "the device encountered a load or store on an invalid memory address.",
        cudaError_t.cudaErrorLaunchOutOfResources:
            "a launch did not occur because it did not have appropriate resources.",
        cudaError_t.cudaErrorLaunchTimeout:
            "the device kernel took too long to execute.",
        cudaError_t.cudaErrorIllegalInstruction:
            "the device encountered an illegal instruction.",
        cudaError_t.cudaErrorMisalignedAddress:
            "the device encountered a load or store on a misaligned address.",
        cudaError_t.cudaErrorInvalidAddressSpace:
            "the device encountered an instruction that can only operate on memory in another address space.",
        cudaError_t.cudaErrorInvalidPc:
            "the device program counter wrapped its address space.",
        cudaError_t.cudaErrorLaunchFailure:
            "an exception occurred on the device while executing a kernel.",
        cudaError_t.cudaErrorNoDevice:
            "no CUDA-capable device was detected.",
        cudaError_t.cudaErrorInvalidDevice:
            "the device ordinal is not valid, or the device is not available.",
        cudaError_t.cudaErrorNotSupported:
            "the operation is not supported on the current system or device.",
        cudaError_t.cudaErrorSystemNotReady:
            "the system is not yet ready to start any CUDA work.",
        cudaError_t.cudaErrorSystemDriverMismatch:
            "there is a mismatch between the versions of the display driver and the CUDA driver.",
        cudaError_t.cudaErrorCompatNotSupportedOnDevice:
            "the system was upgraded to run with forward compatibility but the hardware does not support it.",
        cudaError_t.cudaErrorStreamCaptureUnsupported:
            "the operation is not permitted when the stream is capturing.",
        cudaError_t.cudaErrorTimeout:
            "the timeout specified for the wait operation has lapsed.",
        cudaError_t.cudaErrorUnknown:
            "an unknown error has occurred.",
    }

    no_need_check_error_funcs:List[str] = [
        "cudaGetErrorString",
    ]


class CUDART(Library):
    """The CUDA runtime's entry points."""

    _library_name = "CUDA runtime"
    _dll_name = "cudart64_12.dll"
    _so_name = "libcudart.so"
    _signatures = CUDARTInfo.func_signatures
    _error_codes = CUDARTInfo.error_codes
    _no_error_check = CUDARTInfo.no_need_check_error_funcs

    @classmethod
    def _preheat(cls):
        """Ask the runtime how many devices there are.

        The runtime has a device of its own to set up, and doing it here (rather than
        at the first real call) keeps the first call cheap. The call is harmless on a
        machine with no device: it answers zero.
        """
        from ctypes import byref

        def _ask() -> None:
            count = c_int(0)
            cls.cudaGetDeviceCount(byref(count))

        return _ask


__all__ = [
    "CUDART",
    "CUDARTInfo",
    "cudaError_t",
    "cudaMemcpyKind",
    "cudaDeviceAttr",
    "cudaDeviceptr",
    "cudaStream_t",
]
