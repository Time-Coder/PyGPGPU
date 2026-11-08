"""CUDA function signatures: the module family."""
from typing import Any, Dict
from ctypes import (
    c_uint, c_int,
    c_size_t, c_char_p, POINTER, c_void_p
)

from .cutypes import (
    ptr_CUmodule, CUmodule, ptr_CUfunction,
    ptr_CUdeviceptr, CUdeviceptr,
    CUstream,
    CUfunction, CUDA_MEMCPY2D, CUDA_MEMCPY3D
)


SIGNATURES:Dict[str, Any] = {
            "cuMemcpy2D_v2": {
                "args": {
                    "pCopy": POINTER(CUDA_MEMCPY2D)
                },
                "restype": c_int
            },

            "cuMemcpy2DAsync_v2": {
                "args": {
                    "pCopy": POINTER(CUDA_MEMCPY2D),
                    "hStream": CUstream
                },
                "restype": c_int
            },

            "cuMemcpy3D_v2": {
                "args": {
                    "pCopy": POINTER(CUDA_MEMCPY3D)
                },
                "restype": c_int
            },

            "cuMemcpy3DAsync_v2": {
                "args": {
                    "pCopy": POINTER(CUDA_MEMCPY3D),
                    "hStream": CUstream
                },
                "restype": c_int
            },

            "cuModuleLoadData": {
                "args": {
                    "module": ptr_CUmodule,
                    "image": c_void_p
                },
                "restype": c_int
            },

            "cuModuleGetFunction": {
                "args": {
                    "hfunc": ptr_CUfunction,
                    "hmod": CUmodule,
                    "name": c_char_p
                },
                "restype": c_int
            },

            "cuModuleUnload": {
                "args": {
                    "hmod": CUmodule
                },
                "restype": c_int
            },

            "cuFuncGetAttribute": {
                "args": {
                    "pi": POINTER(c_int),
                    "attrib": c_int,
                    "hfunc": CUfunction
                },
                "restype": c_int
            },

            "cuFuncSetAttribute": {
                "args": {
                    "hfunc": CUfunction,
                    "attrib": c_int,
                    "value": c_int
                },
                "restype": c_int
            },

            "cuMemAlloc_v2": {
                "args": {
                    "dptr": ptr_CUdeviceptr,
                    "bytesize": c_size_t
                },
                "restype": c_int
            },

            "cuMemcpyHtoD_v2": {
                "args": {
                    "dstDevice": CUdeviceptr,
                    "srcHost": c_void_p,
                    "ByteCount": c_size_t
                },
                "restype": c_int
            },

            "cuMemcpyDtoH_v2": {
                "args": {
                    "dstHost": c_void_p,
                    "srcDevice": CUdeviceptr,
                    "ByteCount": c_size_t
                },
                "restype": c_int
            },

            "cuMemcpyHtoDAsync_v2": {
                "args": {
                    "dstDevice": CUdeviceptr,
                    "srcHost": c_void_p,
                    "ByteCount": c_size_t,
                    "hStream": CUstream
                },
                "restype": c_int
            },

            "cuMemcpyDtoHAsync_v2": {
                "args": {
                    "dstHost": c_void_p,
                    "srcDevice": CUdeviceptr,
                    "ByteCount": c_size_t,
                    "hStream": CUstream
                },
                "restype": c_int
            },

            "cuMemFree_v2": {
                "args": {
                    "dptr": CUdeviceptr
                },
                "restype": c_int
            },

            "cuMemHostRegister_v2": {
                "args": {
                    "p": c_void_p,
                    "bytesize": c_size_t,
                    "Flags": c_uint
                },
                "restype": c_int
            },

            "cuMemHostUnregister": {
                "args": {"p": c_void_p},
                "restype": c_int
            },
}
