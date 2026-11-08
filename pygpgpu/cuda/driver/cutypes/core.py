# The scalar aliases, the opaque handles, and the callback/pointer types.
from ctypes import (
    c_int, c_uint, c_float, POINTER,
    c_size_t, Structure, c_char, c_void_p, c_uint64,
    CFUNCTYPE
)
from typing_extensions import TypeAlias



CUdevice_v1 = c_int
CUdevice = CUdevice_v1
CUcontext = c_void_p
CUmodule = c_void_p
CUfunction = c_void_p
CUdeviceptr = c_uint64
CUstream = c_void_p
CUevent = c_void_p

ptr_float: TypeAlias = POINTER(c_float)
ptr_ptr_void: TypeAlias = POINTER(c_void_p)
ptr_uint: TypeAlias = POINTER(c_uint)
ptr_int: TypeAlias = POINTER(c_int)
ptr_CUdevice: TypeAlias = POINTER(CUdevice)
ptr_size_t: TypeAlias = POINTER(c_size_t)
ptr_CUcontext: TypeAlias = POINTER(CUcontext)
ptr_CUmodule: TypeAlias = POINTER(CUmodule)
ptr_CUfunction: TypeAlias = POINTER(CUfunction)
ptr_CUdeviceptr: TypeAlias = POINTER(CUdeviceptr)
ptr_CUstream: TypeAlias = POINTER(CUstream)
ptr_CUevent: TypeAlias = POINTER(CUevent)

CUstreamCallback:TypeAlias = CFUNCTYPE(None, CUstream, c_uint, c_void_p)


CUarray = c_void_p
CUtexObject = c_uint64
CUsurfObject = c_uint64

ptr_CUarray: TypeAlias = POINTER(CUarray)
ptr_CUtexObject: TypeAlias = POINTER(CUtexObject)


class CUuuid(Structure):
    _fields_ = [
        ("bytes", c_char * 16)
    ]

ptr_CUuuid:TypeAlias = POINTER(CUuuid)
