"""CUDA function signatures: the array family."""
from typing import Any, Dict
from ctypes import (
    c_int,
    POINTER
)

from .cutypes import (
    CUarray, CUtexObject, CU_ARRAY_DESC, CUDA_RESOURCE_DESC,
    CUDA_TEXTURE_DESC, CUDA_RESOURCE_VIEW_DESC, CUDA_ARRAY3D_DESCRIPTOR
)


SIGNATURES:Dict[str, Any] = {
            "cuArrayCreate_v2": {
                "args": {
                    "pHandle": POINTER(CUarray),
                    "pAllocateArray": POINTER(CU_ARRAY_DESC)
                },
                "restype": c_int
            },

            "cuArrayDestroy": {
                "args": {
                    "hArray": CUarray
                },
                "restype": c_int
            },

            "cuArrayGetDescriptor_v2": {
                "args": {
                    "pArrayDescriptor": POINTER(CU_ARRAY_DESC),
                    "hArray": CUarray
                },
                "restype": c_int
            },

            "cuTexObjectCreate": {
                "args": {
                    "pTexObject": POINTER(CUtexObject),
                    "pResDesc": POINTER(CUDA_RESOURCE_DESC),
                    "pTexDesc": POINTER(CUDA_TEXTURE_DESC),
                    "pResViewDesc": POINTER(CUDA_RESOURCE_VIEW_DESC)
                },
                "restype": c_int
            },

            "cuTexObjectDestroy": {
                "args": {
                    "texObject": CUtexObject
                },
                "restype": c_int
            },

            "cuArray3DCreate_v2": {
                "args": {
                    "pHandle": POINTER(CUarray),
                    "pAllocateArray": POINTER(CUDA_ARRAY3D_DESCRIPTOR)
                },
                "restype": c_int
            },

            "cuArray3DGetDescriptor_v2": {
                "args": {
                    "pArrayDescriptor": POINTER(CUDA_ARRAY3D_DESCRIPTOR),
                    "hArray": CUarray
                },
                "restype": c_int
            },
}
