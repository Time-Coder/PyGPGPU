"""CUDA function signatures: the device family."""
from typing import Any, Dict
from ctypes import (
    c_uint, c_int,
    c_size_t, c_char_p, POINTER, c_void_p
)

from .cutypes import (
    CUdevice, CUuuid, CUcontext,
    ptr_CUdeviceptr, CUevent
)


SIGNATURES:Dict[str, Any] = {
            "cuInit": {
                "args": {
                    "Flags": c_uint,
                },
                "restype": c_int
            },

            "cuDeviceGetCount": {
                "args": {
                    "count": POINTER(c_int)
                },
                "restype": c_int
            },

            "cuDeviceGet": {
                "args": {
                    "device": POINTER(CUdevice),
                    "ordinal": c_int
                },
                "restype": c_int
            },

            "cuDeviceGetName": {
                "args": {
                    "name": c_char_p,
                    "len": c_int,
                    "dev": CUdevice
                },
                "restype": c_int
            },

            "cuDeviceGetAttribute": {
                "args": {
                    "pi": POINTER(c_int),
                    "attrib": c_int,
                    "dev": CUdevice
                },
                "restype": c_int
            },

            "cuDeviceTotalMem": {
                "args": {
                    "bytes": POINTER(c_size_t),
                    "dev": CUdevice
                },
                "restype": c_int
            },

            "cuDeviceGetUuid": {
                "args": {
                    "uuid": POINTER(CUuuid),
                    "dev": CUdevice
                },
                "restype": c_int
            },

            "cuCtxCreate_v2": {
                "args": {
                    "pctx": POINTER(CUcontext),
                    "flags": c_uint,
                    "dev": CUdevice
                },
                "restype": c_int
            },

            "cuDevicePrimaryCtxRetain": {
                "args": {
                    "pctx": POINTER(CUcontext),
                    "dev": CUdevice
                },
                "restype": c_int
            },

            "cuDevicePrimaryCtxRelease_v2": {
                "args": {"dev": CUdevice},
                "restype": c_int
            },

            "cuDevicePrimaryCtxGetState": {
                "args": {
                    "dev": CUdevice,
                    "flags": POINTER(c_uint),
                    "active": POINTER(c_int)
                },
                "restype": c_int
            },

            "cuDevicePrimaryCtxSetFlags_v2": {
                "args": {
                    "dev": CUdevice,
                    "flags": c_uint
                },
                "restype": c_int
            },

            "cuCtxDestroy": {
                "args": {
                    "ctx": CUcontext
                },
                "restype": c_int
            },

            "cuCtxGetApiVersion": {
                "args": {
                    "ctx": CUcontext,
                    "version": POINTER(c_uint)
                },
                "restype": c_int
            },

            "cuCtxGetCacheConfig": {
                "args": {
                    "pconfig": POINTER(c_int)
                },
                "restype": c_int
            },

            "cuCtxGetLimit": {
                "args": {
                    "pvalue": POINTER(c_size_t),
                    "limit": c_int
                },
                "restype": c_int
            },

            "cuCtxPopCurrent": {
                "args": {
                    "pctx": POINTER(CUcontext)
                },
                "restype": c_int
            },

            "cuCtxPushCurrent_v2": {
                "args": {
                    "ctx": CUcontext
                },
                "restype": c_int
            },

            "cuCtxSetCacheConfig": {
                "args": {
                    "config": c_int
                },
                "restype": c_int
            },

            "cuCtxSetLimit": {
                "args": {
                    "limit": c_int,
                    "value": c_size_t
                },
                "restype": c_int
            },

            "cuCtxSynchronize": {
                "args": {},
                "restype": c_int
            },

            "cuCtxGetCurrent": {
                "args": {
                    "pctx": POINTER(CUcontext)
                },
                "restype": c_int
            },

            "cuCtxSetCurrent": {
                "args": {
                    "ctx": CUcontext
                },
                "restype": c_int
            },

            "cuCtxWaitEvent": {
                "args": {
                    "hCtx": CUcontext,
                    "hEvent": CUevent
                },
                "restype": c_int
            },

            "cuMemHostGetDevicePointer_v2": {
                "args": {
                    "pdptr": ptr_CUdeviceptr,
                    "p": c_void_p,
                    "Flags": c_uint
                },
                "restype": c_int
            },
}
