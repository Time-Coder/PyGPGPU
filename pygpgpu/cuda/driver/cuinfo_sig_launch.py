"""CUDA function signatures: the launch family."""
from typing import Any, Dict
from ctypes import (
    c_float, c_uint, c_int,
    POINTER, c_void_p
)

from .cutypes import (
    CUstream,
    CUfunction, CUevent, CUstreamCallback,
    CUDA_RESOURCE_DESC,
    CUsurfObject
)


SIGNATURES:Dict[str, Any] = {
            "cuSurfObjectCreate": {
                "args": {
                    "pSurfObject": POINTER(CUsurfObject),
                    "pResDesc": POINTER(CUDA_RESOURCE_DESC)
                },
                "restype": c_int
            },

            "cuSurfObjectDestroy": {
                "args": {
                    "surfObject": CUsurfObject
                },
                "restype": c_int
            },

            "cuEventQuery": {
                "args": {"hEvent": CUevent},
                "restype": c_int
            },

            "cuLaunchKernel": {
                "args": {
                    "f": CUfunction,
                    "gridDimX": c_uint,
                    "gridDimY": c_uint,
                    "gridDimZ": c_uint,
                    "blockDimX": c_uint,
                    "blockDimY": c_uint,
                    "blockDimZ": c_uint,
                    "sharedMemBytes": c_uint,
                    "hStream": CUstream,
                    "kernelParams": POINTER(c_void_p),
                    "extra": POINTER(c_void_p)
                },
                "restype": c_int
            },

            "cuEventCreate": {
                "args": {
                    "phEvent": POINTER(CUevent),
                    "Flags": c_uint
                },
                "restype": c_int
            },

            "cuEventRecord": {
                "args": {
                    "hEvent": CUevent,
                    "hStream": CUstream
                },
                "restype": c_int
            },

            "cuEventSynchronize": {
                "args": {
                    "hEvent": CUevent
                },
                "restype": c_int
            },

            "cuStreamCreate": {
                "args": {
                    "phStream": POINTER(CUstream),
                    "Flags": c_uint
                },
                "restype": c_int
            },

            "cuStreamWaitEvent": {
                "args": {
                    "hStream": CUstream,
                    "hEvent": CUevent,
                    "Flags": c_uint
                },
                "restype": c_int
            },

            "cuStreamSynchronize": {
                "args": {
                    "hStream": CUstream
                },
                "restype": c_int
            },

            "cuStreamAddCallback": {
                "args": {
                    "hStream": CUstream,
                    "callback": CUstreamCallback,
                    "userData": c_void_p,
                    "flags": c_uint
                },
                "restype": c_int
            },

            "cuStreamQuery": {
                "args": {
                    "hStream": CUstream
                },
                "restype": c_int
            },

            "cuEventElapsedTime": {
                "args": {
                    "pMilliseconds": POINTER(c_float),
                    "hStart": CUevent,
                    "hEnd": CUevent
                },
                "restype": c_int
            },

            "cuEventDestroy": {
                "args": {
                    "hEvent": CUevent
                },
                "restype": c_int
            },

            "cuStreamDestroy": {
                "args": {
                    "hStream": CUstream
                },
                "restype": c_int
            },
}
