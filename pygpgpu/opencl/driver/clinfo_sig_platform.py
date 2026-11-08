"""CL function signatures: the platform family."""
from ctypes import (
    c_uint, c_int, c_void_p, c_ulong, c_size_t,
)
from typing import Any, Dict

from .cltypes import (
    cl_uint,
    cl_platform_id,
    ptr_cl_device_id,
    cl_int,
    ptr_size_t,
    ptr_cl_platform_id,
    ptr_cl_uint,
    ErrorCode
)

SIGNATURES:Dict[str, Any] = {
        "clGetPlatformInfo": {
            "args": {
                "platform": cl_platform_id,
                "param_name": cl_uint,
                "param_value_size": c_size_t,
                "param_value": c_void_p,
                "param_value_size_ret": ptr_size_t
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_PLATFORM: "platform is not a valid platform.",
                ErrorCode.CL_INVALID_VALUE: "param_name is not one of the supported values, or the size in bytes specified by param_value_size is less than size of the return type specified in the Platform Queries table and param_value is not NULL.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clGetPlatformIDs": {
            "args": {
                "num_entries": cl_uint,
                "platforms": ptr_cl_platform_id,
                "num_platforms": ptr_cl_uint
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_PLATFORM_NOT_FOUND_KHR: "cl_khr_icd extension is supported and zero platforms are available.",
                ErrorCode.CL_INVALID_VALUE: "num_entries is equal to zero and platforms is not NULL or both num_platforms and platforms are NULL.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clGetDeviceIDs": {
            "args": {
                "platform": c_void_p,
                "device_type": c_ulong,
                "num_entries": c_uint,
                "devices": ptr_cl_device_id,
                "num_devices": ptr_cl_uint
            },
            "restype": c_int,
            "errors": {
                ErrorCode.CL_INVALID_PLATFORM: "platform is not a valid platform.",
                ErrorCode.CL_INVALID_DEVICE_TYPE: "device_type is not a valid value.",
                ErrorCode.CL_INVALID_VALUE: "num_entries is equal to zero and devices is not NULL or both num_devices and devices are NULL.",
                ErrorCode.CL_DEVICE_NOT_FOUND: "no OpenCL devices that matched device_type were found.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
}
