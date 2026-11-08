# The memcpy descriptors and the memory types.
from ctypes import (
    c_uint, c_size_t, Structure, c_void_p
)
from .core import CUarray, CUdeviceptr
from ....constants import IntEnum



class CUmemorytype(IntEnum):
    CU_MEMORYTYPE_HOST                = 0x01
    CU_MEMORYTYPE_DEVICE              = 0x02
    CU_MEMORYTYPE_ARRAY               = 0x03
    CU_MEMORYTYPE_UNIFIED             = 0x04


class CUDA_MEMCPY3D(Structure):
    """A 3D copy description, mirroring CUDA_MEMCPY3D_v2 from cuda.h.

    The pitch/height fields are needed whenever one side is linear memory; they are
    ignored when the corresponding side is an array. reserved0/reserved1 must be
    NULL, so the struct is zeroed before use.
    """
    _fields_ = [
        ("srcXInBytes", c_size_t),
        ("srcY", c_size_t),
        ("srcZ", c_size_t),
        ("srcLOD", c_size_t),
        ("srcMemoryType", c_uint),
        ("srcHost", c_void_p),
        ("srcDevice", CUdeviceptr),
        ("srcArray", CUarray),
        ("reserved0", c_void_p),
        ("srcPitch", c_size_t),
        ("srcHeight", c_size_t),

        ("dstXInBytes", c_size_t),
        ("dstY", c_size_t),
        ("dstZ", c_size_t),
        ("dstLOD", c_size_t),
        ("dstMemoryType", c_uint),
        ("dstHost", c_void_p),
        ("dstDevice", CUdeviceptr),
        ("dstArray", CUarray),
        ("reserved1", c_void_p),
        ("dstPitch", c_size_t),
        ("dstHeight", c_size_t),

        ("WidthInBytes", c_size_t),
        ("Height", c_size_t),
        ("Depth", c_size_t),
    ]


# CUDA_ARRAY_FLAGS values. A surface needs CUDA_ARRAY_FLAG_SURFACE_LDST set at
# creation time, and a texture does not.
CUDA_ARRAY_FLAG_SURFACE_LDST = 0x02


class CUDA_MEMCPY2D(Structure):
    """2D copy descriptor used by cuMemcpy2D to move image rows in and out of an array.

    Which source/destination pair is used is chosen by the srcMemoryType and
    dstMemoryType fields.
    """
    _fields_ = [
        ("srcXInBytes", c_size_t),
        ("srcY", c_size_t),
        ("srcMemoryType", c_uint),
        ("srcHost", c_void_p),
        ("srcDevice", CUdeviceptr),
        ("srcArray", CUarray),
        ("srcPitch", c_size_t),
        ("dstXInBytes", c_size_t),
        ("dstY", c_size_t),
        ("dstMemoryType", c_uint),
        ("dstHost", c_void_p),
        ("dstDevice", CUdeviceptr),
        ("dstArray", CUarray),
        ("dstPitch", c_size_t),
        ("WidthInBytes", c_size_t),
        ("Height", c_size_t),
    ]
