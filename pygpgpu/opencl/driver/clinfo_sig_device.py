"""CL function signatures: the device family."""
from ctypes import (
    c_int, c_void_p, c_size_t,
)
from typing import Any, Dict

from .cltypes import (
    cl_ulong,
    cl_uint,
    cl_device_id,
    ptr_cl_int,
    ptr_cl_ulong,
    cl_context,
    ptr_size_t,
    cl_mem,
    ErrorCode,
    ptr_cl_image_format,
    ptr_cl_image_desc
)

SIGNATURES:Dict[str, Any] = {
        "clGetDeviceInfo": {
            "args": {
                "device": cl_device_id,
                "param_name": cl_uint,
                "param_value_size": c_size_t,
                "param_value": c_void_p,
                "param_value_size_ret": ptr_size_t
            },
            "restype": c_int,
            "errors": {
                ErrorCode.CL_INVALID_DEVICE: "device is not a valid device.",
                ErrorCode.CL_INVALID_VALUE: "param_name is not one of the supported values, or the size in bytes specified by param_value_size is less than size of the return type specified in the Device Queries table and param_value is not NULL.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clCreateImage": {
            "args": {
                "context": cl_context,
                "flags": cl_ulong,
                "image_format": ptr_cl_image_format,
                "image_desc": ptr_cl_image_desc,
                "host_ptr": c_void_p,
                "errcode_ret": ptr_cl_int
            },
            "restype": cl_mem,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: "context is not a valid context",
                ErrorCode.CL_INVALID_OPERATION: "there are no devices in context that support images (i.e. CL_DEVICE_IMAGE_SUPPORT specified in the Device Queries table is CL_FALSE)",
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
    * an image is being created from another memory object (buffer or image) under one of the following circumstances:
    * mem_object was created with CL_MEM_WRITE_ONLY and flags specifies CL_MEM_READ_WRITE or CL_MEM_READ_ONLY
    * mem_object was created with CL_MEM_READ_ONLY and flags specifies CL_MEM_READ_WRITE or CL_MEM_WRITE_ONLY
    * flags specifies CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR or CL_MEM_COPY_HOST_PTR
    * an image is being created from another memory object (buffer or image) and mem_object was created with CL_MEM_HOST_WRITE_ONLY and flags specifies CL_MEM_HOST_READ_ONLY
    * mem_object was created with CL_MEM_HOST_READ_ONLY and flags specifies CL_MEM_HOST_WRITE_ONLY
    * mem_object was created with CL_MEM_HOST_NO_ACCESS and_flags_ specifies CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_WRITE_ONLY
    * properties includes a supported external memory handle and flags includes CL_MEM_USE_HOST_PTR, CL_MEM_ALLOC_HOST_PTR, or CL_MEM_COPY_HOST_PTR
    * CL_MEM_IMMUTABLE_EXT is set in flags and CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, or CL_MEM_HOST_WRITE_ONLY is set in flags
    * CL_MEM_IMMUTABLE_EXT is not set in flags and mem_object was created with CL_MEM_IMMUTABLE_EXT
    * CL_MEM_IMMUTABLE_EXT is set in flags and none of the following conditions are met:
    * CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR is set in flags
    * the image is being created from another memory object (buffer or image)
    * properties includes an external memory handle""",
                ErrorCode.CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: """one of following case happend:
    * image_format is NULL
    * values specified in image_format are not valid
    * properties includes an AHardwareBuffer external memory handle and image_format is not NULL
    * an image is created from a buffer and the row pitch, or slice pitch, if the cl_ext_image_from_buffer extension is supported, or base address alignment do not follow the rules described for creating an image from a buffer""",
                ErrorCode.CL_INVALID_IMAGE_DESCRIPTOR: "properties includes an AHardwareBuffer external memory handle and image_desc is not NULL",
                ErrorCode.CL_INVALID_IMAGE_SIZE: """one of following case happend:
    * the image dimensions specified in image_desc are not valid or exceed the maximum image dimensions described in the Device Queries table for all devices in context
    * the cl_ext_image_from_buffer extension is supported and an image is created from a buffer and the buffer passed in mem_object is too small to be used as a data store for the image, e.g. if its size is smaller than the value returned for CL_IMAGE_REQUIREMENTS_SIZE_EXT for the parameters used to create the image""",
                ErrorCode.CL_INVALID_HOST_PTR: """one of following case happend:
    * host_ptr is NULL and CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR are set in flags
    * host_ptr is not NULL but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in flags
    * properties includes a supported external memory handle and host_ptr is not NULL""",
                ErrorCode.CL_IMAGE_FORMAT_NOT_SUPPORTED: """one of following case happend:
    * there are no devices in context that support image_format
    * properties includes an AHardwareBuffer external memory handle and the AHardwareBuffer format is not supported""",
                ErrorCode.CL_MEM_OBJECT_ALLOCATION_FAILURE: "there is a failure to allocate memory for the image object",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host"
            }
        },
    
        "clCreateImageWithProperties": {
            "args": {
                "context": cl_context,
                "properties": ptr_cl_ulong,
                "flags": cl_ulong,
                "image_format": ptr_cl_image_format,
                "image_desc": ptr_cl_image_desc,
                "host_ptr": c_void_p,
                "errcode_ret": ptr_cl_int
            },
            "restype": cl_mem,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: "context is not a valid context",
                ErrorCode.CL_INVALID_OPERATION: "there are no devices in context that support images (i.e. CL_DEVICE_IMAGE_SUPPORT specified in the Device Queries table is CL_FALSE)",
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
    * an image is being created from another memory object (buffer or image) under one of the following circumstances:
    * mem_object was created with CL_MEM_WRITE_ONLY and flags specifies CL_MEM_READ_WRITE or CL_MEM_READ_ONLY
    * mem_object was created with CL_MEM_READ_ONLY and flags specifies CL_MEM_READ_WRITE or CL_MEM_WRITE_ONLY
    * flags specifies CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR or CL_MEM_COPY_HOST_PTR
    * an image is being created from another memory object (buffer or image) and mem_object was created with CL_MEM_HOST_WRITE_ONLY and flags specifies CL_MEM_HOST_READ_ONLY
    * mem_object was created with CL_MEM_HOST_READ_ONLY and flags specifies CL_MEM_HOST_WRITE_ONLY
    * mem_object was created with CL_MEM_HOST_NO_ACCESS and_flags_ specifies CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_WRITE_ONLY
    * properties includes a supported external memory handle and flags includes CL_MEM_USE_HOST_PTR, CL_MEM_ALLOC_HOST_PTR, or CL_MEM_COPY_HOST_PTR
    * CL_MEM_IMMUTABLE_EXT is set in flags and CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, or CL_MEM_HOST_WRITE_ONLY is set in flags
    * CL_MEM_IMMUTABLE_EXT is not set in flags and mem_object was created with CL_MEM_IMMUTABLE_EXT
    * CL_MEM_IMMUTABLE_EXT is set in flags and none of the following conditions are met:
    * CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR is set in flags
    * the image is being created from another memory object (buffer or image)
    * properties includes an external memory handle""",
                ErrorCode.CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: """one of following case happend:
    * image_format is NULL
    * values specified in image_format are not valid
    * properties includes an AHardwareBuffer external memory handle and image_format is not NULL
    * an image is created from a buffer and the row pitch, or slice pitch, if the cl_ext_image_from_buffer extension is supported, or base address alignment do not follow the rules described for creating an image from a buffer""",
                ErrorCode.CL_INVALID_IMAGE_DESCRIPTOR: "properties includes an AHardwareBuffer external memory handle and image_desc is not NULL",
                ErrorCode.CL_INVALID_IMAGE_SIZE: """one of following case happend:
    * the image dimensions specified in image_desc are not valid or exceed the maximum image dimensions described in the Device Queries table for all devices in context
    * the cl_ext_image_from_buffer extension is supported and an image is created from a buffer and the buffer passed in mem_object is too small to be used as a data store for the image, e.g. if its size is smaller than the value returned for CL_IMAGE_REQUIREMENTS_SIZE_EXT for the parameters used to create the image""",
                ErrorCode.CL_INVALID_HOST_PTR: """one of following case happend:
    * host_ptr is NULL and CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR are set in flags
    * host_ptr is not NULL but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in flags
    * properties includes a supported external memory handle and host_ptr is not NULL""",
                ErrorCode.CL_IMAGE_FORMAT_NOT_SUPPORTED: """one of following case happend:
    * there are no devices in context that support image_format
    * properties includes an AHardwareBuffer external memory handle and the AHardwareBuffer format is not supported""",
                ErrorCode.CL_MEM_OBJECT_ALLOCATION_FAILURE: "there is a failure to allocate memory for the image object",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host"
            }
        },
}
