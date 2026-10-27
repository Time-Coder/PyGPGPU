from ctypes import (c_int, c_uint, c_size_t, c_float, c_void_p, c_ulonglong,
                    c_char_p, POINTER, CFUNCTYPE, Structure, Union)
from enum import IntEnum


class hipError(IntEnum):
    SUCCESS = 0
    INVALID_VALUE = 1
    OUT_OF_MEMORY = 2
    NOT_INITIALIZED = 3


# Type aliases named like the cuda backend's, to ease porting of the oop layer
CUdevice = c_int
CUdeviceptr = c_ulonglong
CUmodule = c_void_p
CUfunction = c_void_p
CUstream = c_void_p
CUevent = c_void_p

ptr_CUmodule = POINTER(CUmodule)
ptr_CUfunction = POINTER(CUfunction)
ptr_CUdeviceptr = POINTER(CUdeviceptr)
ptr_CUstream = POINTER(CUstream)
ptr_CUevent = POINTER(CUevent)

CUstreamCallback = CFUNCTYPE(None, CUstream, c_int, c_void_p)

ptr_ptr_char = POINTER(POINTER(c_char_p))


# ---------- texture and array objects ----------
#
# HIP mirrors the CUDA driver API here, and the ROCm headers keep the same field
# order for these descriptors, so the layouts below are shared with the cuda backend.
# Only the entry points that reading needs are bound: ROCm exports no
# hipSurfObjectCreate, so writing an image has no host-side object to bind.

CUarray = c_void_p
CUtexObject = c_ulonglong
CUsurfObject = c_ulonglong

ptr_CUarray = POINTER(CUarray)
ptr_CUtexObject = POINTER(CUtexObject)
ptr_CUsurfObject = POINTER(CUsurfObject)


class CUarray_format(IntEnum):
    CU_AD_FORMAT_UNSIGNED_INT8 = 0x01
    CU_AD_FORMAT_UNSIGNED_INT16 = 0x02
    CU_AD_FORMAT_UNSIGNED_INT32 = 0x03
    CU_AD_FORMAT_SIGNED_INT8 = 0x08
    CU_AD_FORMAT_SIGNED_INT16 = 0x09
    CU_AD_FORMAT_SIGNED_INT32 = 0x0A
    CU_AD_FORMAT_HALF = 0x10
    CU_AD_FORMAT_FLOAT = 0x20


class CUresourcetype(IntEnum):
    CU_RESOURCE_TYPE_ARRAY = 0x00
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01
    CU_RESOURCE_TYPE_LINEAR = 0x02
    CU_RESOURCE_TYPE_PITCH2D = 0x03


class CUaddress_mode(IntEnum):
    CU_TR_ADDRESS_MODE_WRAP = 0
    CU_TR_ADDRESS_MODE_CLAMP = 1
    CU_TR_ADDRESS_MODE_MIRROR = 2
    CU_TR_ADDRESS_MODE_BORDER = 3


class CUfilter_mode(IntEnum):
    CU_TR_FILTER_MODE_POINT = 0
    CU_TR_FILTER_MODE_LINEAR = 1


# The array flag a surface needs; unused here because surfaces cannot be created, but
# kept next to the other array constants for symmetry with the cuda backend.
CUDA_ARRAY_FLAG_SURFACE_LDST = 0x02


class CU_ARRAY_DESC(Structure):
    """The description of a 2D array (``CUDA_ARRAY_DESCRIPTOR``)."""
    _fields_ = [("Width", c_size_t), ("Height", c_size_t),
                ("Format", c_uint), ("NumChannels", c_uint)]


class CUDA_ARRAY3D_DESCRIPTOR(Structure):
    """The description of a 3D array."""
    _fields_ = [("Width", c_size_t), ("Height", c_size_t), ("Depth", c_size_t),
                ("Format", c_uint), ("NumChannels", c_uint), ("Flags", c_uint)]


class _CUDA_RESOURCE_DESC_RES_UNION(Union):
    _fields_ = [("hArray", CUarray),
                ("reserved", c_int * 32)]


class CUDA_RESOURCE_DESC(Structure):
    _fields_ = [("resType", c_uint), ("_pad", c_uint),
                ("res", _CUDA_RESOURCE_DESC_RES_UNION),
                ("flags", c_uint), ("_pad2", c_uint)]


class CUDA_TEXTURE_DESC(Structure):
    _fields_ = [("addressMode", c_uint * 3), ("filterMode", c_uint),
                ("flags", c_uint), ("maxAnisotropy", c_uint),
                ("mipmapFilterMode", c_uint), ("mipmapLevelBias", c_float),
                ("minMipmapLevelClamp", c_float), ("maxMipmapLevelClamp", c_float),
                ("borderColor", c_float * 4), ("reserved", c_int * 12)]
