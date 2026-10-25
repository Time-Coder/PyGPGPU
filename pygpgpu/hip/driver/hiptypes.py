from ctypes import c_int, c_void_p, c_ulonglong, c_char_p, POINTER, CFUNCTYPE
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
