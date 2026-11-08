# The array descriptors.
from ctypes import (
    c_uint, c_size_t, Structure
)
from ....constants import IntEnum



class CUarray_format(IntEnum):
    """Element format stored in an array."""
    CU_AD_FORMAT_UNSIGNED_INT8  = 0x01
    CU_AD_FORMAT_UNSIGNED_INT16 = 0x02
    CU_AD_FORMAT_UNSIGNED_INT32 = 0x03
    CU_AD_FORMAT_SIGNED_INT8    = 0x08
    CU_AD_FORMAT_SIGNED_INT16   = 0x09
    CU_AD_FORMAT_SIGNED_INT32   = 0x0A
    CU_AD_FORMAT_HALF           = 0x10
    CU_AD_FORMAT_FLOAT          = 0x20


class CUarray_height_align(IntEnum):
    CU_ARRAY_HEIGHT_ALIGN_NONE   = 0
    CU_ARRAY_HEIGHT_ALIGN_16     = 16
    CU_ARRAY_HEIGHT_ALIGN_32     = 32
    CU_ARRAY_HEIGHT_ALIGN_64     = 64
    CU_ARRAY_HEIGHT_ALIGN_128    = 128


class CU_ARRAY_DESC(Structure):
    """Array layout, as returned by cuArrayGetDescriptor.

    Mirrors CUDA_ARRAY_DESCRIPTOR_v2 from cuda.h: two size_t dimensions followed by
    the format and channel count. Declaring Width/Height as c_uint would shift
    Format by 8 bytes and hand the driver garbage.
    """
    _fields_ = [
        ("Width", c_size_t),
        ("Height", c_size_t),
        ("Format", c_uint),
        ("NumChannels", c_uint),
    ]


# The CUDA headers name it CUDA_ARRAY_DESCRIPTOR; both spellings appear in docs.
CUDA_ARRAY_DESCRIPTOR = CU_ARRAY_DESC


class CUDA_ARRAY3D_DESCRIPTOR(Structure):
    """3D array layout, mirroring CUDA_ARRAY3D_DESCRIPTOR_v2 from cuda.h."""
    _fields_ = [
        ("Width", c_size_t),
        ("Height", c_size_t),
        ("Depth", c_size_t),
        ("Format", c_uint),
        ("NumChannels", c_uint),
        ("Flags", c_uint),
    ]
