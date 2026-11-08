"""CL function signatures: the enqueue family."""
from ctypes import (
    c_void_p, c_size_t,
)
from typing import Any, Dict

from .cltypes import (
    cl_kernel,
    cl_command_queue,
    cl_uint,
    cl_int,
    ptr_size_t,
    cl_mem,
    ErrorCode,
    ptr_cl_event
)

SIGNATURES:Dict[str, Any] = {
        "clEnqueueWriteBuffer": {
            "args": {
                "command_queue": cl_command_queue,
                "buffer": cl_mem,
                "blocking_write": cl_uint,
                "offset": c_size_t,
                "size": c_size_t,
                "ptr": c_void_p,
                "num_events_in_wait_list": cl_uint,
                "event_wait_list": ptr_cl_event,
                "event": ptr_cl_event
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_COMMAND_QUEUE: "command_queue is not a valid host command-queue.",
                ErrorCode.CL_INVALID_CONTEXT: "the context associated with command_queue and buffer are not the same or the context associated with command_queue and events in event_wait_list are not the same.",
                ErrorCode.CL_INVALID_MEM_OBJECT: "buffer is not a valid buffer object.",
                ErrorCode.CL_INVALID_VALUE: "the region being read or written specified by (offset, size) is out of bounds or ptr is a NULL value.",
                ErrorCode.CL_INVALID_EVENT_WAIT_LIST: "event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_list is not NULL and num_events_in_wait_list is 0, or event objects in event_wait_list are not valid events.",
                ErrorCode.CL_MISALIGNED_SUB_BUFFER_OFFSET: "buffer is a sub-buffer object and offset specified when the sub-buffer object is created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue. This error code is missing before version 1.1.",
                ErrorCode.CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: "the read and write operations are blocking and the execution status of any of the events in event_wait_list is a negative integer value. This error code is missing before version 1.1.",
                ErrorCode.CL_MEM_OBJECT_ALLOCATION_FAILURE: "there is a failure to allocate memory for data store associated with buffer.",
                ErrorCode.CL_INVALID_OPERATION: """one of following case happend:
    * clEnqueueWriteBuffer is called on buffer which has been created with CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_NO_ACCESS.
    * clEnqueueWriteBuffer is called on buffer which has been created with CL_MEM_IMMUTABLE_EXT.""",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host.",
            }
        },
    
        "clEnqueueReadBuffer": {
            "args": {
                "command_queue": cl_command_queue,
                "buffer": cl_mem,
                "blocking_read": cl_uint,
                "offset": c_size_t,
                "size": c_size_t,
                "ptr": c_void_p,
                "num_events_in_wait_list": cl_uint,
                "event_wait_list": ptr_cl_event,
                "event": ptr_cl_event
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_COMMAND_QUEUE: "command_queue is not a valid host command-queue.",
                ErrorCode.CL_INVALID_CONTEXT: "the context associated with command_queue and buffer are not the same or the context associated with command_queue and events in event_wait_list are not the same.",
                ErrorCode.CL_INVALID_MEM_OBJECT: "buffer is not a valid buffer object.",
                ErrorCode.CL_INVALID_VALUE: "the region being read or written specified by (offset, size) is out of bounds or ptr is a NULL value.",
                ErrorCode.CL_INVALID_EVENT_WAIT_LIST: "event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_list is not NULL and num_events_in_wait_list is 0, or event objects in event_wait_list are not valid events.",
                ErrorCode.CL_MISALIGNED_SUB_BUFFER_OFFSET: "buffer is a sub-buffer object and offset specified when the sub-buffer object is created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue. This error code is missing before version 1.1.",
                ErrorCode.CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: "the read and write operations are blocking and the execution status of any of the events in event_wait_list is a negative integer value. This error code is missing before version 1.1.",
                ErrorCode.CL_MEM_OBJECT_ALLOCATION_FAILURE: "there is a failure to allocate memory for data store associated with buffer.",
                ErrorCode.CL_INVALID_OPERATION: "clEnqueueReadBuffer is called on buffer which has been created with CL_MEM_HOST_WRITE_ONLY or CL_MEM_HOST_NO_ACCESS.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host.",
            }
        },
    
        "clEnqueueNDRangeKernel": {
            "args": {
                "command_queue": cl_command_queue,
                "kernel": cl_kernel,
                "work_dim": cl_uint,
                "global_work_offset": ptr_size_t,
                "global_work_size": ptr_size_t,
                "local_work_size": ptr_size_t,
                "num_events_in_wait_list": cl_uint,
                "event_wait_list": ptr_cl_event,
                "event": ptr_cl_event
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_PROGRAM_EXECUTABLE: "there is no successfully built program executable available for the device associated with command_queue.",
                ErrorCode.CL_INVALID_COMMAND_QUEUE: "command_queue is not a valid host command-queue.",
                ErrorCode.CL_INVALID_KERNEL: "kernel is not a valid kernel object.",
                ErrorCode.CL_INVALID_CONTEXT: "context associated with command_queue and kernel are not the same or the context associated with command_queue and events in event_wait_list are not the same.",
                ErrorCode.CL_INVALID_KERNEL_ARGS: "the kernel argument values have not been specified.",
                ErrorCode.CL_INVALID_WORK_DIMENSION: "work_dim is not a valid value (i.e. a value between 1 and CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS).",
                ErrorCode.CL_INVALID_GLOBAL_WORK_SIZE: "global_work_size is NULL or any of the values specified in global_work_size[0], …​global_work_size[work_dim - 1] are zero. This error condition does not apply when the device associated with command_queue supports OpenCL 2.1 or newer.",
                ErrorCode.CL_INVALID_GLOBAL_WORK_SIZE: "any of the values specified in global_work_size[0], …​ global_work_size[work_dim - 1] exceed the maximum value representable by size_t on the device on which the kernel-instance will be enqueued.",
                ErrorCode.CL_INVALID_GLOBAL_OFFSET: "the value specified in global_work_size + the corresponding values in global_work_offset for any dimensions is greater than the maximum value representable by size t on the device on which the kernel-instance will be enqueued, or global_work_offset is non-NULL before version 1.1.",
                ErrorCode.CL_INVALID_WORK_GROUP_SIZE: "local_work_size is specified and does not match the required work-group size for kernel in the program source.",
                ErrorCode.CL_INVALID_WORK_GROUP_SIZE: "local_work_size is specified and is not consistent with the required number of sub-groups for kernel in the program source.",
                ErrorCode.CL_INVALID_WORK_GROUP_SIZE: "local_work_size is specified and the total number of work-items in the work-group computed as local_work_size[0] × …​ local_work_size[work_dim - 1] is greater than the value specified by CL_KERNEL_WORK_GROUP_SIZE in the Kernel Object Device Queries table.",
                ErrorCode.CL_INVALID_WORK_GROUP_SIZE: "the work-group size must be uniform and the local_work_size is not NULL, is not equal to the required work-group size specified in the kernel source, or the global_work_size is not evenly divisible by the local_work_size.",
                ErrorCode.CL_INVALID_WORK_ITEM_SIZE: "the number of work-items specified in any of local_work_size[0], …​ local_work_size[work_dim - 1] is greater than the corresponding values specified by CL_DEVICE_MAX_WORK_ITEM_SIZES[0], …​, CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1].",
                ErrorCode.CL_MISALIGNED_SUB_BUFFER_OFFSET: "a sub-buffer object is specified as the value for an argument that is a buffer object and the offset specified when the sub-buffer object is created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue. This error code is missing before version 1.1.",
                ErrorCode.CL_INVALID_IMAGE_SIZE: "an image object is specified as an argument value and the image dimensions (image width, height, specified or compute row and/or slice pitch) are not supported by device associated with queue.",
                ErrorCode.CL_IMAGE_FORMAT_NOT_SUPPORTED: "an image object is specified as an argument value and the image format (image channel order and data type) is not supported by device associated with queue.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to queue the execution instance of kernel on the command-queue because of insufficient resources needed to execute the kernel. For example, the explicitly specified local_work_size causes a failure to execute the kernel because of insufficient resources such as registers or local memory. Another example would be the number of read-only image args used in kernel exceed the CL_DEVICE_MAX_READ_IMAGE_ARGS value for device or the number of write-only and read-write image args used in kernel exceed the CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS value for device or the number of samplers used in kernel exceed CL_DEVICE_MAX_SAMPLERS for device.",
                ErrorCode.CL_MEM_OBJECT_ALLOCATION_FAILURE: "there is a failure to allocate memory for data store associated with image or buffer objects specified as arguments to kernel.",
                ErrorCode.CL_INVALID_EVENT_WAIT_LIST: "event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_list is not NULL and num_events_in_wait_list is 0, or event objects in event_wait_list are not valid events.",
                ErrorCode.CL_INVALID_OPERATION: "SVM pointers are passed as arguments to a kernel and the device does not support SVM, or system pointers are passed as arguments to a kernel and the device does not support fine-grain system SVM.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clFinish": {
            "args": {
                "command_queue": cl_command_queue
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_COMMAND_QUEUE: "command_queue is not a valid host command-queue.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clWaitForEvents": {
            "args": {
                "num_events": cl_uint,
                "event_list": ptr_cl_event
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_VALUE: "num_events is zero or event_list is NULL.",
                ErrorCode.CL_INVALID_CONTEXT: "events specified in event_list do not belong to the same context.",
                ErrorCode.CL_INVALID_EVENT: "event objects specified in event_list are not valid event objects.",
                ErrorCode.CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: "the execution status of any of the events in event_list is a negative integer value. This error code is missing before version 1.1.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clEnqueueReadImage": {
            "args": {
                "command_queue": cl_command_queue,
                "image": cl_mem,
                "blocking_read": cl_uint,
                "origin": ptr_size_t,
                "region": ptr_size_t,
                "row_pitch": c_size_t,
                "slice_pitch": c_size_t,
                "ptr": c_void_p,
                "num_events_in_wait_list": cl_uint,
                "event_wait_list": ptr_cl_event,
                "event": ptr_cl_event
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_COMMAND_QUEUE: "command_queue is not a valid host command-queue.",
                ErrorCode.CL_INVALID_CONTEXT: "the context associated with command_queue and image are not the same or the context associated with command_queue and events in event_wait_list are not the same.",
                ErrorCode.CL_INVALID_MEM_OBJECT: "image is not a valid image object.",
                ErrorCode.CL_INVALID_VALUE: """one of following case happend:
    * origin or region is NULL.
    * the region being read or written specified by origin and region is out of bounds.
    * values in origin and region do not follow rules described in the argument description for origin and region.
    * image is a 1D or 2D image and slice_pitch or input_slice_pitch is not 0.
    * ptr is NULL.""",
                ErrorCode.CL_INVALID_EVENT_WAIT_LIST: "event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_list is not NULL and num_events_in_wait_list is 0, or event objects in event_wait_list are not valid events.",
                ErrorCode.CL_INVALID_IMAGE_SIZE: "image dimensions (image width, height, specified or compute row and/or slice pitch) for image are not supported by device associated with queue.",
                ErrorCode.CL_IMAGE_FORMAT_NOT_SUPPORTED: "image format (image channel order and data type) for image are not supported by device associated with queue.",
                ErrorCode.CL_MEM_OBJECT_ALLOCATION_FAILURE: "there is a failure to allocate memory for data store associated with image.",
                ErrorCode.CL_INVALID_OPERATION: """one of following case happend:
    * the device associated with command_queue does not support images (i.e. CL_DEVICE_IMAGE_SUPPORT specified in the Device Queries table is CL_FALSE).
    * clEnqueueReadImage is called on image which has been created with CL_MEM_HOST_WRITE_ONLY or CL_MEM_HOST_NO_ACCESS.
    * clEnqueueWriteImage is called on image which has been created with CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_NO_ACCESS.
    * clEnqueueWriteImage is called on image which has been created with CL_MEM_IMMUTABLE_EXT.""",
                ErrorCode.CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: "the read and write operations are blocking and the execution status of any of the events in event_wait_list is a negative integer value. This error code is missing before version 1.1.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host.",
                ErrorCode.CL_INVALID_MIP_LEVEL: "the cl_khr_mipmap_image extension is supported, and the mip level specified in origin is not a valid level for image,"
            }
        },
    
        "clEnqueueWriteImage": {
            "args": {
                "command_queue": cl_command_queue,
                "image": cl_mem,
                "blocking_write": cl_uint,
                "origin": ptr_size_t,
                "region": ptr_size_t,
                "input_row_pitch": c_size_t,
                "input_slice_pitch": c_size_t,
                "ptr": c_void_p,
                "num_events_in_wait_list": cl_uint,
                "event_wait_list": ptr_cl_event,
                "event": ptr_cl_event
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_COMMAND_QUEUE: "command_queue is not a valid host command-queue.",
                ErrorCode.CL_INVALID_CONTEXT: "the context associated with command_queue and image are not the same or the context associated with command_queue and events in event_wait_list are not the same.",
                ErrorCode.CL_INVALID_MEM_OBJECT: "image is not a valid image object.",
                ErrorCode.CL_INVALID_VALUE: """one of following case happend:
    * origin or region is NULL.
    * the region being read or written specified by origin and region is out of bounds.
    * values in origin and region do not follow rules described in the argument description for origin and region.
    * image is a 1D or 2D image and slice_pitch or input_slice_pitch is not 0.
    * ptr is NULL.""",
                ErrorCode.CL_INVALID_EVENT_WAIT_LIST: "event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_list is not NULL and num_events_in_wait_list is 0, or event objects in event_wait_list are not valid events.",
                ErrorCode.CL_INVALID_IMAGE_SIZE: "image dimensions (image width, height, specified or compute row and/or slice pitch) for image are not supported by device associated with queue.",
                ErrorCode.CL_IMAGE_FORMAT_NOT_SUPPORTED: "image format (image channel order and data type) for image are not supported by device associated with queue.",
                ErrorCode.CL_MEM_OBJECT_ALLOCATION_FAILURE: "there is a failure to allocate memory for data store associated with image.",
                ErrorCode.CL_INVALID_OPERATION: """one of following case happend:
    * the device associated with command_queue does not support images (i.e. CL_DEVICE_IMAGE_SUPPORT specified in the Device Queries table is CL_FALSE).
    * clEnqueueReadImage is called on image which has been created with CL_MEM_HOST_WRITE_ONLY or CL_MEM_HOST_NO_ACCESS.
    * clEnqueueWriteImage is called on image which has been created with CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_NO_ACCESS.
    * clEnqueueWriteImage is called on image which has been created with CL_MEM_IMMUTABLE_EXT.""",
                ErrorCode.CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: "the read and write operations are blocking and the execution status of any of the events in event_wait_list is a negative integer value. This error code is missing before version 1.1.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host.",
                ErrorCode.CL_INVALID_MIP_LEVEL: "the cl_khr_mipmap_image extension is supported, and the mip level specified in origin is not a valid level for image,"
            }
        },
}
