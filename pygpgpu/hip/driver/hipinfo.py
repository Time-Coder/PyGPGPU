from ctypes import (
    c_int8, c_uint8, c_int16, c_uint16,
    c_int32, c_uint32, c_int64, c_uint64,
    c_float, c_double, c_uint, c_int,
    c_size_t, c_char_p, POINTER, c_void_p
)

from .hiptypes import (
    CUdeviceptr, CUmodule, CUfunction, CUstream, CUevent,
    ptr_CUmodule, ptr_CUfunction, ptr_CUdeviceptr, ptr_CUstream, ptr_CUevent,
    CUstreamCallback
)


class HIPInfo:

    # Error strings are obtained dynamically via hipGetErrorString, so no static table is needed
    error_codes = {}
    no_need_check_error_funcs = {"hipGetErrorString", "hipEventQuery", "hipStreamQuery", "hipGetLastError"}

    # Vector types reuse the opencl registry directly (50 of them, including float3 etc.)
    vec_types = None  # filled lazily (see _vec_types())

    basic_types = {
        'char': c_int8, 'uchar': c_uint8,
        'short': c_int16, 'ushort': c_uint16,
        'int': c_int32, 'uint': c_uint32,
        'long': c_int64, 'ulong': c_uint64,
        'float': c_float, 'double': c_double,
        'bool': c_int, 'void': None,
    }

    alter_types = {
        'char': (c_int8, int), 'uchar': (c_uint8, int),
        'short': (c_int16, int), 'ushort': (c_uint16, int),
        'int': (c_int32, int), 'uint': (c_uint32, int),
        'long': (c_int64, int), 'ulong': (c_uint64, int),
        'float': (c_float, float), 'double': (c_double, float),
    }

    @classmethod
    def _scalar_types(cls):
        from ...opencl.driver.clinfo import CLInfo
        return dict(CLInfo.scalar_types)

    @classmethod
    def _vec_types(cls):
        if cls.vec_types is None:
            from ...opencl.driver.clinfo import CLInfo
            cls.vec_types = dict(CLInfo.vec_types)
            for name, ctype in cls.basic_types.items():
                cls.vec_types.setdefault(name, ctype)
        return cls.vec_types

    def __getattr__(self, name:str):
        # Lazy access to vec_types: does HIPInfo.vec_types trigger __getattr__? A class attribute of None -> instances never trigger it;
        # this is a safety net for class-level attribute access
        raise AttributeError(name)


# Class-attribute access compatibility: once _vec_types fills HIPInfo.vec_types it is a plain dict
def __getattr__(name:str):
    if name == "vec_types":
        return HIPInfo._vec_types()
    if name == "scalar_types":
        return HIPInfo._scalar_types()
    raise AttributeError(name)


func_signatures = {
    # hipError_t hipInit(unsigned int flags)
    "hipInit": {"args": {"flags": c_uint}, "restype": c_int},

    # hipError_t hipGetDeviceCount(int* count)
    "hipGetDeviceCount": {"args": {"count": POINTER(c_int)}, "restype": c_int},

    # hipError_t hipGetDevice(int* device, int ordinal)
    "hipGetDevice": {"args": {"device": POINTER(c_int), "ordinal": c_int}, "restype": c_int},

    # hipError_t hipSetDevice(int device)
    "hipSetDevice": {"args": {"device": c_int}, "restype": c_int},

    # hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device)
    "hipGetDeviceProperties": {"args": {"prop": c_void_p, "device": c_int}, "restype": c_int},

    "hipDeviceSynchronize": {"args": {}, "restype": c_int},

    # hipError_t hipMalloc(void* dev, size_t size)
    "hipMalloc": {"args": {"dev": ptr_CUdeviceptr, "size": c_size_t}, "restype": c_int},
    "hipFree": {"args": {"dev": CUdeviceptr}, "restype": c_int},

    "hipMemcpyHtoD": {"args": {"dst": CUdeviceptr, "src": c_void_p, "size": c_size_t}, "restype": c_int},
    "hipMemcpyDtoH": {"args": {"dst": c_void_p, "src": CUdeviceptr, "size": c_size_t}, "restype": c_int},
    "hipMemcpyHtoDAsync": {"args": {"dst": CUdeviceptr, "src": c_void_p, "size": c_size_t, "stream": CUstream}, "restype": c_int},
    "hipMemcpyDtoHAsync": {"args": {"dst": c_void_p, "src": CUdeviceptr, "size": c_size_t, "stream": CUstream}, "restype": c_int},

    # hipError_t hipModuleLoadData(hipModule_t* module, const void* image)
    "hipModuleLoadData": {"args": {"module": ptr_CUmodule, "image": c_void_p}, "restype": c_int},
    "hipModuleUnload": {"args": {"module": CUmodule}, "restype": c_int},

    # hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* name)
    "hipModuleGetFunction": {"args": {"function": ptr_CUfunction, "module": CUmodule, "name": c_char_p}, "restype": c_int},

    "hipModuleLaunchKernel": {
        "args": {
            "function": CUfunction,
            "gridDimX": c_uint, "gridDimY": c_uint, "gridDimZ": c_uint,
            "blockDimX": c_uint, "blockDimY": c_uint, "blockDimZ": c_uint,
            "sharedMemBytes": c_uint,
            "stream": CUstream,
            "kernelParams": POINTER(c_void_p),
            "extra": POINTER(c_void_p),
        },
        "restype": c_int,
    },

    "hipStreamCreate": {"args": {"stream": ptr_CUstream, "flags": c_uint}, "restype": c_int},
    "hipStreamDestroy": {"args": {"stream": CUstream}, "restype": c_int},
    "hipStreamSynchronize": {"args": {"stream": CUstream}, "restype": c_int},
    "hipStreamQuery": {"args": {"stream": CUstream}, "restype": c_int},
    "hipStreamAddCallback": {"args": {"stream": CUstream, "callback": CUstreamCallback, "userData": c_void_p, "arg": c_uint}, "restype": c_int},

    "hipEventCreate": {"args": {"event": ptr_CUevent}, "restype": c_int},
    "hipEventDestroy": {"args": {"event": CUevent}, "restype": c_int},
    "hipEventRecord": {"args": {"event": CUevent, "stream": CUstream}, "restype": c_int},
    "hipEventSynchronize": {"args": {"event": CUevent}, "restype": c_int},
    "hipEventQuery": {"args": {"event": CUevent}, "restype": c_int},

    "hipEventElapsedTime": {"args": {"ms": POINTER(c_float), "start": CUevent, "end": CUevent}, "restype": c_int},

    "hipGetErrorString": {"args": {"error": c_int}, "restype": c_char_p},
    "hipGetLastError": {"args": {}, "restype": c_int},
}


# Filled at import time (avoids circular imports, so it sits at the end of the module)
HIPInfo.scalar_types = HIPInfo._scalar_types()
HIPInfo.vec_types = HIPInfo._vec_types()
