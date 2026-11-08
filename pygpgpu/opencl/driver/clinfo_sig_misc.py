"""CL function signatures: the misc family."""
from ctypes import (
    c_void_p, c_size_t,
)
from typing import Any, Dict

from .cltypes import (
    ptr_cl_int,
    ptr_cl_ulong,
    cl_context,
    cl_bitfield,
    cl_mem,
    ErrorCode
)

SIGNATURES:Dict[str, Any] = {
        "clCreateBuffer": {
            "args": {
                "context": cl_context,
                "flags": cl_bitfield,
                "size": c_size_t,
                "host_ptr": c_void_p,
                "errcode_ret": ptr_cl_int
            },
            "restype": cl_mem,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: "context is not a valid context",
                ErrorCode.CL_INVALID_PROPERTY: """one of following case happend:
    * a property name in properties is not a supported property name
    * the value specified for a supported property name is not valid
    * the same property name is specified more than once
    * properties does not include a supported external memory handle and CL_MEM_DEVICE_HANDLE_LIST_KHR is specified as part of properties
    * properties includes more than one external memory handle""",
                ErrorCode.CL_INVALID_DEVICE: """one of following case happend:
    * a device identified by the property CL_MEM_DEVICE_HANDLE_LIST_KHR is not a valid device or is not associated with context
    * a device identified by property CL_MEM_DEVICE_HANDLE_LIST_KHR cannot import the requested external memory object type
    * CL_MEM_DEVICE_HANDLE_LIST_KHR is not specified as part of properties and one or more devices in context cannot import the requested external memory object type""",
                ErrorCode.CL_INVALID_VALUE: """one of following case happend:
    * values specified in flags are not valid as defined in the Memory Flags table
    * properties includes a supported external memory handle and flags includes CL_MEM_USE_HOST_PTR, CL_MEM_ALLOC_HOST_PTR, or CL_MEM_COPY_HOST_PTR
    * CL_MEM_IMMUTABLE_EXT is set in flags and CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, or CL_MEM_HOST_WRITE_ONLY is set in flags
    * CL_MEM_IMMUTABLE_EXT is set in flags and none of the following conditions are met:
    * CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR is set in flags
    * properties includes an external memory handle""",
                ErrorCode.CL_INVALID_BUFFER_SIZE: """one of following case happend:
    * size is zero and properties does not include an AHardwareBuffer external memory handle
    * size is non-zero and properties includes an AHardwareBuffer external memory handle
    * size is greater than CL_DEVICE_MAX_MEM_ALLOC_SIZE for all devices in context
    * CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR is set in flags, and host_ptr is a pointer returned by clSVMAlloc, and size is greater than the size passed to clSVMAlloc""",
                ErrorCode.CL_INVALID_HOST_PTR: """one of following case happend:
    * host_ptr is NULL, and CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR are set in flags
    * host_ptr is not NULL but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in flags
    * properties includes a supported external memory handle and host_ptr is not NULL""",
                ErrorCode.CL_INVALID_OPERATION: """one of following case happend:
    * properties includes an AHardwareBuffer external memory handle and the AHardwareBuffer format is not AHARDWAREBUFFER_FORMAT_BLOB
    * properties includes CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT and there are no devices in the context that support the cl_ext_buffer_device_address extension""",
                ErrorCode.CL_MEM_OBJECT_ALLOCATION_FAILURE: "there is a failure to allocate memory for the buffer object",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host"
            }
        },
    
        "clCreateBufferWithProperties": {
            "args": {
                "context": cl_context,
                "properties": ptr_cl_ulong,
                "flags": cl_bitfield,
                "size": c_size_t,
                "host_ptr": c_void_p,
                "errcode_ret": ptr_cl_int
            },
            "restype": cl_mem,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: "context is not a valid context",
                ErrorCode.CL_INVALID_PROPERTY: """one of following case happend:
    * a property name in properties is not a supported property name
    * the value specified for a supported property name is not valid
    * the same property name is specified more than once
    * properties does not include a supported external memory handle and CL_MEM_DEVICE_HANDLE_LIST_KHR is specified as part of properties
    * properties includes more than one external memory handle""",
                ErrorCode.CL_INVALID_DEVICE: """one of following case happend:
    * a device identified by the property CL_MEM_DEVICE_HANDLE_LIST_KHR is not a valid device or is not associated with context
    * a device identified by property CL_MEM_DEVICE_HANDLE_LIST_KHR cannot import the requested external memory object type
    * CL_MEM_DEVICE_HANDLE_LIST_KHR is not specified as part of properties and one or more devices in context cannot import the requested external memory object type""",
                ErrorCode.CL_INVALID_VALUE: """one of following case happend:
    * values specified in flags are not valid as defined in the Memory Flags table
    * properties includes a supported external memory handle and flags includes CL_MEM_USE_HOST_PTR, CL_MEM_ALLOC_HOST_PTR, or CL_MEM_COPY_HOST_PTR
    * CL_MEM_IMMUTABLE_EXT is set in flags and CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, or CL_MEM_HOST_WRITE_ONLY is set in flags
    * CL_MEM_IMMUTABLE_EXT is set in flags and none of the following conditions are met:
    * CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR is set in flags
    * properties includes an external memory handle""",
                ErrorCode.CL_INVALID_BUFFER_SIZE: """one of following case happend:
    * size is zero and properties does not include an AHardwareBuffer external memory handle
    * size is non-zero and properties includes an AHardwareBuffer external memory handle
    * size is greater than CL_DEVICE_MAX_MEM_ALLOC_SIZE for all devices in context
    * CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR is set in flags, and host_ptr is a pointer returned by clSVMAlloc, and size is greater than the size passed to clSVMAlloc""",
                ErrorCode.CL_INVALID_HOST_PTR: """one of following case happend:
    * host_ptr is NULL, and CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR are set in flags
    * host_ptr is not NULL but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in flags
    * properties includes a supported external memory handle and host_ptr is not NULL""",
                ErrorCode.CL_INVALID_OPERATION: """one of following case happend:
    * properties includes an AHardwareBuffer external memory handle and the AHardwareBuffer format is not AHARDWAREBUFFER_FORMAT_BLOB
    * properties includes CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT and there are no devices in the context that support the cl_ext_buffer_device_address extension""",
                ErrorCode.CL_MEM_OBJECT_ALLOCATION_FAILURE: "there is a failure to allocate memory for the buffer object",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host"
            }
        },
}
