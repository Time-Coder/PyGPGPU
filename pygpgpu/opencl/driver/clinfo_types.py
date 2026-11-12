"""The value tables: which C type a CL type name names, and which vector class carries it."""
from ctypes import (
    c_char, c_ubyte,
    c_int16, c_uint16,
    c_int32, c_uint32,
    c_uint64,
    c_float, c_double,
    c_int64,
)
from .vec_types import (
    char2, char3, char4,
    uchar2, uchar3, uchar4,
    short2, short3, short4,
    ushort2, ushort3, ushort4,
    int2, int3, int4,
    uint2, uint3, uint4,
    long2, long3, long4,
    ulong2, ulong3, ulong4,
    float2, float3, float4,
    double2, double3, double4,
)

# The 8/16-wide families are defined in this package: OpenCL C is the only
# backend that declares them (a kernel refuses the width everywhere; see
# translator.vectors.KERNEL_VECTOR_WIDTHS).
from .vec_types import (
    char8, char16,
    uchar8, uchar16,
    short8, short16,
    ushort8, ushort16,
    int8, int16,
    uint8, uint16,
    long8, long16,
    ulong8, ulong16,
    float8, float16,
    double8, double16,
)

from .image_types import (
    image1d_t,
    image2d_t,
    image3d_t,
    image1d_array_t,
    image2d_array_t
)


image_types = {
    "image2d_t": image2d_t,
    "image3d_t": image3d_t,
    "image2d_array_t": image2d_array_t,
    "image1d_t": image1d_t,
    # "image1d_buffer_t",
    "image1d_array_t": image1d_array_t,
    # "image2d_depth_t",
    # "image2d_array_depth_t"
}


basic_types = {
    'char': c_char,
    'char2': char2,
    'char3': char3,
    'char4': char4,
    'char8': char8,
    'char16': char16,

    'uchar': c_ubyte,
    'uchar2': uchar2,
    'uchar3': uchar3,
    'uchar4': uchar4,
    'uchar8': uchar8,
    'uchar16': uchar16,

    'short': c_int16,
    'short2': short2,
    'short3': short3,
    'short4': short4,
    'short8': short8,
    'short16': short16,

    'ushort': c_uint16,
    'ushort2': ushort2,
    'ushort3': ushort3,
    'ushort4': ushort4,
    'ushort8': ushort8,
    'ushort16': ushort16,

    'int': c_int32,
    'int2': int2,
    'int3': int3,
    'int4': int4,
    'int8': int8,
    'int16': int16,

    'uint': c_uint32,
    'uint2': uint2,
    'uint3': uint3,
    'uint4': uint4,
    'uint8': uint8,
    'uint16': uint16,

    'long': c_int64,
    'long2': long2,
    'long3': long3,
    'long4': long4,
    'long8': long8,
    'long16': long16,

    'ulong': c_uint64,
    'ulong2': ulong2,
    'ulong3': ulong3,
    'ulong4': ulong4,
    'ulong8': ulong8,
    'ulong16': ulong16,

    'float': c_float,
    'float2': float2,
    'float3': float3,
    'float4': float4,
    'float8': float8,
    'float16': float16,

    'double': c_double,
    'double2': double2,
    'double3': double3,
    'double4': double4,
    'double8': double8,
    'double16': double16,
}


scalar_types = {
    'char': c_char,
    'uchar': c_ubyte,
    'short': c_int16,
    'ushort': c_uint16,
    'int': c_int32,
    'uint': c_uint32,
    'long': c_int64,
    'ulong': c_uint64,
    'float': c_float,
    'double': c_double
}


vec_types = {
    'char2': char2,
    'char3': char3,
    'char4': char4,
    'char8': char8,
    'char16': char16,

    'uchar2': uchar2,
    'uchar3': uchar3,
    'uchar4': uchar4,
    'uchar8': uchar8,
    'uchar16': uchar16,

    'short2': short2,
    'short3': short3,
    'short4': short4,
    'short8': short8,
    'short16': short16,

    'ushort2': ushort2,
    'ushort3': ushort3,
    'ushort4': ushort4,
    'ushort8': ushort8,
    'ushort16': ushort16,

    'int2': int2,
    'int3': int3,
    'int4': int4,
    'int8': int8,
    'int16': int16,

    'uint2': uint2,
    'uint3': uint3,
    'uint4': uint4,
    'uint8': uint8,
    'uint16': uint16,

    'long2': long2,
    'long3': long3,
    'long4': long4,
    'long8': long8,
    'long16': long16,

    'ulong2': ulong2,
    'ulong3': ulong3,
    'ulong4': ulong4,
    'ulong8': ulong8,
    'ulong16': ulong16,

    'float2': float2,
    'float3': float3,
    'float4': float4,
    'float8': float8,
    'float16': float16,

    'double2': double2,
    'double3': double3,
    'double4': double4,
    'double8': double8,
    'double16': double16,
}


alter_types = {
    'char': (c_char, int),
    'uchar': (c_ubyte, int),
    'short': (c_int16, int),
    'ushort': (c_uint16, int),
    'int': (c_int32, int),
    'uint': (c_uint32, int),
    'long': (c_int64, int),
    'ulong': (c_uint64, int),
    'float': (c_float, float),
    'double': (c_double, float)
}
