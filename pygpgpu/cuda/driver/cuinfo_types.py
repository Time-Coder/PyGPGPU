"""The value tables: which C type a CUDA type name names, and which vector class carries it."""
from ctypes import (
    c_int8, c_uint8, c_int16, c_uint16,
    c_int32, c_uint32, c_int64, c_uint64,
    c_float, c_double
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
    double2, double3, double4
)


basic_types = {
    'char': c_int8,
    'char2': char2,
    'char3': char3,
    'char4': char4,

    'uchar': c_uint8,
    'uchar2': uchar2,
    'uchar3': uchar3,
    'uchar4': uchar4,

    'short': c_int16,
    'short2': short2,
    'short3': short3,
    'short4': short4,

    'ushort': c_uint16,
    'ushort2': ushort2,
    'ushort3': ushort3,
    'ushort4': ushort4,

    'int': c_int32,
    'int2': int2,
    'int3': int3,
    'int4': int4,

    'uint': c_uint32,
    'uint2': uint2,
    'uint3': uint3,
    'uint4': uint4,

    'long': c_int64,
    'long2': long2,
    'long3': long3,
    'long4': long4,

    'ulong': c_uint64,
    'ulong2': ulong2,
    'ulong3': ulong3,
    'ulong4': ulong4,

    'float': c_float,
    'float2': float2,
    'float3': float3,
    'float4': float4,

    'double': c_double,
    'double2': double2,
    'double3': double3,
    'double4': double4,
}


scalar_types = {
    'char': c_int8,
    'uchar': c_uint8,
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

    'uchar2': uchar2,
    'uchar3': uchar3,
    'uchar4': uchar4,

    'short2': short2,
    'short3': short3,
    'short4': short4,

    'ushort2': ushort2,
    'ushort3': ushort3,
    'ushort4': ushort4,

    'int2': int2,
    'int3': int3,
    'int4': int4,

    'uint2': uint2,
    'uint3': uint3,
    'uint4': uint4,

    'long2': long2,
    'long3': long3,
    'long4': long4,

    'ulong2': ulong2,
    'ulong3': ulong3,
    'ulong4': ulong4,

    'float2': float2,
    'float3': float3,
    'float4': float4,

    'double2': double2,
    'double3': double3,
    'double4': double4
}


alter_types = {
    'char': (c_int8, int),
    'uchar': (c_uint8, int),
    'short': (c_int16, int),
    'ushort': (c_uint16, int),
    'int': (c_int32, int),
    'uint': (c_uint32, int),
    'long': (c_int64, int),
    'ulong': (c_uint64, int),
    'float': (c_float, float),
    'double': (c_double, float)
}


no_need_check_error_funcs = {
    "cuStreamQuery"
}


no_cached_info = set()
