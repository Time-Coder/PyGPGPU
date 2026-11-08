"""CL function signatures: the memobj family."""
from ctypes import (
    c_void_p, c_size_t,
)
from typing import Any, Dict

from .cltypes import (
    cl_ulong,
    cl_command_queue,
    cl_uint,
    ptr_int64,
    ptr_cl_int,
    ptr_cl_ulong,
    cl_context,
    cl_int,
    ptr_size_t,
    cl_mem,
    ErrorCode,
    ptr_cl_event,
    cl_sampler
)

SIGNATURES:Dict[str, Any] = {
        "clGetMemObjectInfo": {
            "args": {
                "memobj": cl_mem,
                "param_name": cl_uint,
                "param_value_size": c_size_t,
                "param_value": c_void_p,
                "param_value_size_ret": ptr_size_t
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_MEM_OBJECT: "memobj is a not a valid memory object.",
                ErrorCode.CL_INVALID_OPERATION: "the cl_ext_buffer_device_address is not supported or the buffer was not allocated with CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT.",
                ErrorCode.CL_INVALID_VALUE: "param_name is not one of the supported values, or the size in bytes specified by param_value_size is less than size of the return type specified in the Memory Object Queries table and param_value is not NULL.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host.",
                ErrorCode.CL_INVALID_DX9_MEDIA_SURFACE_KHR: "param_name is CL_MEM_DX9_MEDIA_SURFACE_INFO_KHR and memobj was not created by calling clCreateFromDX9MediaSurfaceKHR from a Direct3D9 surface.",
                ErrorCode.CL_INVALID_D3D10_RESOURCE_KHR: "param_name is CL_MEM_D3D10_RESOURCE_KHR and memobj was not created by calling clCreateFromD3D10BufferKHR, clCreateFromD3D10Texture2DKHR, or clCreateFromD3D10Texture3DKHR.",
                ErrorCode.CL_INVALID_D3D11_RESOURCE_KHR: "param_name is CL_MEM_D3D11_RESOURCE_KHR and memobj was not created by calling clCreateFromD3D11BufferKHR, clCreateFromD3D11Texture2DKHR, or clCreateFromD3D11Texture3DKHR."
            }
        },
    
        "clReleaseMemObject": {
            "args": {
                "memobj": cl_mem
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_MEM_OBJECT: "memobj is not a valid memory object.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clSVMAlloc": {
            "args": {
                "context": cl_context,
                "flags": cl_uint,
                "size": c_size_t,
                "alignment": cl_uint
            },
            "restype": c_void_p,
            "errors": {}
        },
    
        "clSVMFree": {
            "args": {
                "context": cl_context,
                "svm_pointer": c_void_p
            },
            "restype": None,
            "errors": {}
        },
    
        "clEnqueueSVMMemcpy": {
            "args": {
                "command_queue": cl_command_queue,
                "blocking_copy": cl_uint,
                "dst_ptr": c_void_p,
                "src_ptr": c_void_p,
                "size": c_size_t,
                "num_events_in_wait_list": cl_uint,
                "event_wait_list": ptr_cl_event,
                "event": ptr_cl_event
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_COMMAND_QUEUE: "command_queue is not a valid host command-queue.",
                ErrorCode.CL_INVALID_CONTEXT: "the context associated with command_queue is not the same as the context associated with events in event_wait_list.",
                ErrorCode.CL_INVALID_VALUE: "dst_ptr or src_ptr is NULL, or size is 0.",
                ErrorCode.CL_INVALID_EVENT_WAIT_LIST: "event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_list is not NULL and num_events_in_wait_list is 0.",
                ErrorCode.CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: "the copy is blocking and the execution status of any of the events in event_wait_list is a negative integer value.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host.",
            }
        },
    
        "clCreateSampler": {
            "args": {
                "context": cl_context,
                "normalized_coords": cl_uint,
                "addressing_mode": cl_uint,
                "filter_mode": cl_uint,
                "errcode_ret": ptr_cl_int
            },
            "restype": cl_sampler,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: "context is not a valid context.",
                ErrorCode.CL_INVALID_VALUE: "addressing_mode, filter_mode, normalized_coords or a combination of these arguements are not valid.",
                ErrorCode.CL_INVALID_OPERATION: "images are not supported by any device associated with context (i.e. CL_DEVICE_IMAGE_SUPPORT specified in the Device Queries table is CL_FALSE).",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clCreateSamplerWithProperties": {
            "args": {
                "context": cl_context,
                "sampler_properties": ptr_cl_ulong,
                "errcode_ret": ptr_cl_int
            },
            "restype": cl_sampler,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: "context is not a valid context.",
                ErrorCode.CL_INVALID_VALUE: "the property name in sampler_properties is not a supported property name, if the value specified for a supported property name is not valid, or the same property name is specified more than once.",
                ErrorCode.CL_INVALID_OPERATION: "images are not supported by any device associated with context (i.e. CL_DEVICE_IMAGE_SUPPORT specified in the Device Queries table is CL_FALSE).",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clGetSamplerInfo": {
            "args": {
                "sampler": cl_sampler,
                "param_name": cl_uint,
                "param_value_size": c_size_t,
                "param_value": c_void_p,
                "param_value_size_ret": ptr_size_t
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_SAMPLER: "sampler is a not a valid sampler object.",
                ErrorCode.CL_INVALID_VALUE: "param_name is not one of the supported values, or the size in bytes specified by param_value_size is less than size of the return type specified in the Sampler Object Queries table and param_value is not NULL.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clReleaseSampler": {
            "args": {
                "sampler": cl_sampler
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_SAMPLER: "sampler is not a valid sampler object.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clCreatePipe": {
            "args": {
                "context": cl_context,
                "flags": cl_ulong,
                "pipe_packet_size": cl_uint,
                "pipe_max_packets": cl_uint,
                "properties": ptr_int64,
                "errcode_ret": ptr_cl_int
            },
            "restype": cl_mem,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: "context is not a valid context.",
                ErrorCode.CL_INVALID_OPERATION: "no devices in context support pipes.",
                ErrorCode.CL_INVALID_VALUE: """one of following case happend:
    * values specified in flags are not as defined above.,
    * properties is not NULL.,
    * CL_MEM_IMMUTABLE_EXT is set in flags.""",
                ErrorCode.CL_INVALID_PIPE_SIZE: "pipe_packet_size is 0 or the pipe_packet_size exceeds CL_DEVICE_PIPE_MAX_PACKET_SIZE value specified in the Device Queries table for all devices in context or pipe_max_packets is 0.",
                ErrorCode.CL_MEM_OBJECT_ALLOCATION_FAILURE: "there is a failure to allocate memory for the pipe object.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        }
}
