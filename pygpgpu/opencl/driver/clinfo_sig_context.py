"""CL function signatures: the context family."""
from ctypes import (
    c_void_p, c_size_t,
)
from typing import Any, Dict

from .cltypes import (
    cl_command_queue,
    cl_uint,
    cl_device_id,
    CL_CONTEXT_NOTIFY_CALLBACK,
    ptr_int64,
    ptr_cl_device_id,
    ptr_cl_int,
    ptr_cl_ulong,
    cl_context,
    cl_int,
    cl_bitfield,
    ptr_size_t,
    ErrorCode
)

SIGNATURES:Dict[str, Any] = {
        "clCreateContext": {
            "args": {
                "properties": ptr_int64,
                "num_devices": cl_uint,
                "devices": ptr_cl_device_id,
                "pfn_notify": CL_CONTEXT_NOTIFY_CALLBACK,
                "user_data": c_void_p,
                "errcode_ret": ptr_cl_int
            },
            "restype": c_void_p,
            "errors": {
                ErrorCode.CL_INVALID_PLATFORM: "no platform is specified in properties and no platform could be selected, or the platform specified in properties is not a valid platform.",
                ErrorCode.CL_INVALID_PROPERTY: "a context property name in properties is not a supported property name, if the value specified for a supported property name is not valid, or the same property name is specified more than once. This error code is missing before version 1.1.",
                ErrorCode.CL_INVALID_VALUE: """one of following case happends:
    * devices is NULL.
    * num_devices is equal to zero.
    * pfn_notify is NULL but user_data is not NULL.""",
                ErrorCode.CL_INVALID_DEVICE: "any device in devices is not a valid device.",
                ErrorCode.CL_DEVICE_NOT_AVAILABLE: "a device in devices is currently not available even though the device was returned by clGetDeviceIDs.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host.",
                ErrorCode.CL_INVALID_DX9_MEDIA_ADAPTER_KHR: "any of the values of the properties CL_CONTEXT_ADAPTER_D3D9_KHR, CL_CONTEXT_ADAPTER_D3D9EX_KHR or CL_CONTEXT_ADAPTER_DXVA_KHR is non-NULL and does not specify a valid media adapter with which the cl_device_ids against which this context is to be created may interoperate.",
                ErrorCode.CL_INVALID_D3D10_DEVICE_KHR: "the value of the property CL_CONTEXT_D3D10_DEVICE_KHR is non-NULL and does not specify a valid Direct3D 10 device with which the cl_device_ids against which this context is to be created may interoperate.",
                ErrorCode.CL_INVALID_D3D11_DEVICE_KHR: "the value of the property CL_CONTEXT_D3D11_DEVICE_KHR is non-NULL and does not specify a valid Direct3D 11 device with which the cl_device_ids against which this context is to be created may interoperate.",
                ErrorCode.CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR: """one of following case happend:
    * a share group was specified for a CGL-based OpenGL implementation by setting the property CL_CGL_SHAREGROUP_KHR, and the specified share group does not identify a valid CGL share group object.
    * a context was specified for an OpenGL or OpenGL ES implementation using the EGL, GLX, or WGL binding APIs, as described above; and any of the following conditions hold:
    * The specified display and context properties do not identify a valid OpenGL or OpenGL ES context.
    * The specified context does not support buffer and renderbuffer objects.
    * The specified context is not compatible with the OpenCL context being created (for example, it exists in a physically distinct address space, such as another hardware device; or it does not support sharing data with OpenCL due to implementation restrictions).""",
                ErrorCode.CL_INVALID_OPERATION: """one of following case happened:
    * Direct3D 11 interoperability is specified by setting CL_INVALID_D3D11_DEVICE_KHR to a non-NULL value, and interoperability with another graphics API is also specified.
    * Direct3D 10 interoperability is specified by setting CL_INVALID_D3D10_DEVICE_KHR to a non-NULL value, and interoperability with another graphics API is also specified.
    * a context was specified as described above and any of the following conditions hold:
    * A context or share group object was specified for one of CGL, EGL, GLX, or WGL and the OpenGL implementation does not support that window-system binding API.
    * More than one of the properties CL_CGL_SHAREGROUP_KHR, CL_EGL_DISPLAY_KHR, CL_GLX_DISPLAY_KHR, and CL_WGL_HDC_KHR is set to a non-default value.
    * Both of the properties CL_CGL_SHAREGROUP_KHR and CL_GL_CONTEXT_KHR are set to non-default values.
    * Any of the devices specified in the devices argument cannot support OpenCL objects which share the data store of an OpenGL object.""",
                ErrorCode.CL_INVALID_PROPERTY: """one of following case happend:
    * both CL_CONTEXT_INTEROP_USER_SYNC, and any of the properties defined by the cl_khr_gl_sharing extension are defined in properties.",
    * the cl_khr_terminate_context extension is supported and CL_CONTEXT_TERMINATE_KHR is set to CL_TRUE in properties, but not all of the devices associated with the context support the ability to support context termination (i.e. CL_DEVICE_TERMINATE_CAPABILITY_CONTEXT_KHR is set for CL_DEVICE_TERMINATE_CAPABILITY_KHR)."""
            }
        },
    
        "clReleaseContext": {
            "args": {
                "context": cl_context
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: "context is not a valid OpenCL context.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clGetContextInfo": {
            "args": {
                "context": cl_context,
                "param_name": cl_uint,
                "param_value_size": c_size_t,
                "param_value": c_void_p,
                "param_value_size_ret": ptr_size_t
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: "context is not a valid context.",
                ErrorCode.CL_INVALID_VALUE: "param_name is not one of the supported values, or the size in bytes specified by param_value_size is less than size of the return type specified in the Context Queries table and param_value is not NULL.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clCreateCommandQueue": {
            "args": {
                "context": cl_context,
                "device": cl_device_id,
                "properties": cl_bitfield,
                "errcode_ret": ptr_cl_int
            },
            "restype": cl_command_queue,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: "context is not a valid context.",
                ErrorCode.CL_INVALID_DEVICE: "device is not a valid device or is not associated with context.",
                ErrorCode.CL_INVALID_VALUE: "values specified in properties are not valid.",
                ErrorCode.CL_INVALID_QUEUE_PROPERTIES: "values specified in properties are valid but are not supported by the device.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clCreateCommandQueueWithProperties": {
            "args": {
                "context": cl_context,
                "device": cl_device_id,
                "properties": ptr_cl_ulong,
                "errcode_ret": ptr_cl_int
            },
            "restype": cl_command_queue,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: "context is not a valid context.",
                ErrorCode.CL_INVALID_DEVICE: "device is not a valid device or is not associated with context.",
                ErrorCode.CL_INVALID_VALUE: "values specified in properties are not valid.",
                ErrorCode.CL_INVALID_QUEUE_PROPERTIES: """one of following case happend:
    * values specified in properties are valid but are not supported by the device.
    * the cl_khr_priority_hints extension is supported, the CL_QUEUE_PRIORITY_KHR property is specified, and the queue is a CL_QUEUE_ON_DEVICE.
    * the cl_khr_throttle_hints extension is supported, the CL_QUEUE_THROTTLE_KHR property is specified, and the queue is a CL_QUEUE_ON_DEVICE.""",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clGetCommandQueueInfo": {
            "args": {
                "command_queue": cl_command_queue,
                "param_name": cl_uint,
                "param_value_size": c_size_t,
                "param_value": c_void_p,
                "param_value_size_ret": ptr_size_t
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_COMMAND_QUEUE: "command_queue is not a valid command-queue, or command_queue is not a valid command-queue for param_name.",
                ErrorCode.CL_INVALID_VALUE: "param_name is not one of the supported values, or the size in bytes specified by param_value_size is less than size of the return type specified in the Command-Queue Queries table and param_value is not NULL.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clReleaseCommandQueue": {
            "args": {
                "command_queue": cl_command_queue
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_COMMAND_QUEUE: "command_queue is not a valid command-queue.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
}
