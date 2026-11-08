# The texture/surface resource descriptors.
from ctypes import (
    c_int, c_uint, c_float, c_size_t, Structure, c_void_p
)
import ctypes as _ctypes
from .core import CUarray
from ....constants import IntEnum



class _CUDA_RESOURCE_DESC_ARRAY(Structure):
    """The ``array`` member of the resource union: a single array handle.

    cuda.h declares only ``CUarray hArray`` here. The format and channel count are
    properties of the array itself, set when it is created, and writing them into
    this struct would overlap the other union members (they land where ``linear``
    keeps its format/size fields).
    """
    _fields_ = [
        ("hArray", CUarray),
    ]


class _CUDA_RESOURCE_DESC_RES_UNION(_ctypes.Union):
    """The 64-byte union that follows resType in CUDA_RESOURCE_DESC."""
    _fields_ = [
        ("array", _CUDA_RESOURCE_DESC_ARRAY),
        ("reserved", c_void_p * 8),
    ]


class _CUDA_RESOURCE_DESC_HEAD(Structure):
    """resType plus its alignment padding; the union starts 8 bytes in."""
    _fields_ = [
        ("resType", c_uint),
        ("_pad", c_uint),
    ]


class CUDA_RESOURCE_DESC(Structure):
    """Describes where a texture's storage lives. Only the array form is modelled
    here, which is what the Array class produces.

    Mirrors CUDA_RESOURCE_DESC_v1 from cuda.h: resType, an 64-byte union of the
    resource kinds, then a trailing flags word that must be zero. Omitting that
    trailing field makes the driver reject the descriptor with INVALID_VALUE.
    """
    _fields_ = [
        ("resType", c_uint),
        ("_pad", c_uint),
        ("res", _CUDA_RESOURCE_DESC_RES_UNION),
        ("flags", c_uint),
        ("_pad2", c_uint),
    ]


class CUDA_TEXTURE_DESC(Structure):
    """Sampling rules for a texture object.

    Field order and trailing padding follow CUDA_TEXTURE_DESC_v1 from cuda.h:
    after the mipmap clamps come borderColor[4] and reserved[12]. The driver reads
    every member by offset, so the tail cannot be shortened.
    """
    _fields_ = [
        ("addressMode", c_uint * 3),
        ("filterMode", c_uint),
        ("flags", c_uint),
        ("maxAnisotropy", c_uint),
        ("mipmapFilterMode", c_uint),
        ("mipmapLevelBias", c_float),
        ("minMipmapLevelClamp", c_float),
        ("maxMipmapLevelClamp", c_float),
        ("borderColor", c_float * 4),
        ("reserved", c_int * 12),
    ]


class CUresourcetype(IntEnum):
    CU_RESOURCE_TYPE_ARRAY           = 0x00
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01
    CU_RESOURCE_TYPE_LINEAR          = 0x02
    CU_RESOURCE_TYPE_PITCH2D         = 0x03


class CUaddress_mode(IntEnum):
    CU_TR_ADDRESS_MODE_WRAP   = 0
    CU_TR_ADDRESS_MODE_CLAMP  = 1
    CU_TR_ADDRESS_MODE_MIRROR = 2


class CUfilter_mode(IntEnum):
    CU_TR_FILTER_MODE_POINT  = 0
    CU_TR_FILTER_MODE_LINEAR = 1


CU_TRSF_READ_AS_INTEGER     = 0x01
CU_TRSF_NORMALIZED_COORDINATES = 0x02


class CUDA_RESOURCE_VIEW_DESC(Structure):
    """Optional view onto an existing resource; passing NULL (the usual case) makes
    the texture see the whole array in its own format."""
    _fields_ = [
        ("format", c_uint),
        ("width", c_size_t),
        ("height", c_size_t),
        ("depth", c_size_t),
        ("firstMipmapLevel", c_uint),
        ("lastMipmapLevel", c_uint),
        ("firstLayer", c_uint),
        ("lastLayer", c_uint),
        ("_reserved", c_uint * 16),
    ]
