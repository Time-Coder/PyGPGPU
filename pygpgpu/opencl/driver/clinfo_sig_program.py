"""CL function signatures: the program family."""
from ctypes import (
    c_void_p, c_size_t, POINTER, c_char_p,
)
from typing import Any, Dict

from .cltypes import (
    cl_program,
    cl_kernel,
    cl_uint,
    cl_device_id,
    ptr_cl_device_id,
    ptr_cl_int,
    ptr_ptr_ubyte,
    cl_context,
    cl_int,
    ptr_size_t,
    ptr_cl_uint,
    ptr_ptr_char,
    ptr_cl_kernel,
    ErrorCode
)

SIGNATURES:Dict[str, Any] = {
        "clCreateProgramWithSource": {
            "args": {
                "context": cl_context,
                "count": cl_uint,
                "strings": ptr_ptr_char,
                "lengths": ptr_size_t,
                "errcode_ret": ptr_cl_int
            },
            "restype": cl_program,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: "context is not a valid context.",
                ErrorCode.CL_INVALID_VALUE: "count is zero or strings or any entry in strings is NULL.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clCreateProgramWithBinary": {
            "args": {
                "context": cl_context,
                "num_devices": cl_uint,
                "device_list": ptr_cl_device_id,
                "lengths": ptr_size_t,
                "binaries": ptr_ptr_ubyte,
                "binary_status": ptr_cl_int,
                "errcode_ret": ptr_cl_int
            },
            "restype": cl_program,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: "context is not a valid context.",
                ErrorCode.CL_INVALID_VALUE: "device_list is NULL or num_devices is zero.",
                ErrorCode.CL_INVALID_DEVICE: "any device in device_list is not in the list of devices associated with context.",
                ErrorCode.CL_INVALID_VALUE: "lengths or binaries is NULL or any entry in lengths[i] is zero or binaries[i] is NULL.",
                ErrorCode.CL_INVALID_BINARY: "an invalid program binary was encountered for any device. binary_status will return specific status for each device.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clBuildProgram": {
            "args": {
                "program": cl_program,
                "num_devices": cl_uint,
                "device_list": POINTER(cl_device_id),
                "options": c_char_p,
                "pfn_notify": c_void_p,
                "user_data": c_void_p
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_PROGRAM: "program is not a valid program object.",
                ErrorCode.CL_INVALID_VALUE: """one of following case happend:
    * device_list is NULL and num_devices is greater than zero, or device_list is not NULL and num_devices is zero.
    * pfn_notify is NULL but user_data is not NULL.""",
                ErrorCode.CL_INVALID_DEVICE: "any device in device_list is not in the list of devices associated with program.",
                ErrorCode.CL_INVALID_BINARY: "program is created with clCreateProgramWithBinary and devices listed in device_list do not have a valid program binary loaded.",
                ErrorCode.CL_INVALID_BUILD_OPTIONS: "the build options specified by options are invalid.",
                ErrorCode.CL_COMPILER_NOT_AVAILABLE: "program is created with clCreateProgramWithILKHR, clCreateProgramWithSource or clCreateProgramWithIL and a compiler is not available, i.e. CL_DEVICE_COMPILER_AVAILABLE specified in the Device Queries table is set to CL_FALSE.",
                ErrorCode.CL_BUILD_PROGRAM_FAILURE: "there is a failure to build the program executable. This error will be returned if clBuildProgram does not return until the build has completed.",
                ErrorCode.CL_INVALID_OPERATION: """one of following case happend:
    * the build of a program executable for any of the devices listed in device_list by a previous call to clBuildProgram for program has not completed.
    * there are kernel objects attached to program.
    * program was not created with clCreateProgramWithSource, clCreateProgramWithIL or clCreateProgramWithBinary.""",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host.",
            }
        },
    
        "clGetProgramInfo": {
            "args": {
                "program": cl_program,
                "param_name": cl_uint,
                "param_value_size": c_size_t,
                "param_value": c_void_p,
                "param_value_size_ret": ptr_size_t
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_PROGRAM: "program is a not a valid program object.",
                ErrorCode.CL_INVALID_VALUE: "param_name is not one of the supported values, or the size in bytes specified by param_value_size is less than size of the return type specified in the Program Object Queries table and param_value is not NULL.",
                ErrorCode.CL_INVALID_PROGRAM_EXECUTABLE: "param_name is CL_PROGRAM_NUM_KERNELS, CL_PROGRAM_KERNEL_NAMES, CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT, or CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT and a successful program executable has not been built for at least one device in the list of devices associated with program.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clGetProgramBuildInfo": {
            "args": {
                "program": cl_program,
                "device": cl_device_id,
                "param_name": cl_uint,
                "param_value_size": c_size_t,
                "param_value": c_void_p,
                "param_value_size_ret": ptr_size_t
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_PROGRAM: "program is a not a valid program object.",
                ErrorCode.CL_INVALID_DEVICE: "device is not in the list of devices associated with program.",
                ErrorCode.CL_INVALID_VALUE: "param_name is not one of the supported values, or the size in bytes specified by param_value_size is less than size of the return type specified in the Program Build Queries table and param_value is not NULL.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clReleaseProgram": {
            "args": {
                "program": cl_program
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_PROGRAM: "program is not a valid program object.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clCreateKernel": {
            "args": {
                "program": cl_program,
                "kernel_name": c_char_p,
                "errcode_ret": ptr_cl_int
            },
            "restype": cl_kernel,
            "errors": {
                ErrorCode.CL_INVALID_PROGRAM: "program is not a valid program object.",
                ErrorCode.CL_INVALID_PROGRAM_EXECUTABLE: "there is no successfully built executable for program.",
                ErrorCode.CL_INVALID_KERNEL_NAME: "kernel_name is not found in program.",
                ErrorCode.CL_INVALID_KERNEL_DEFINITION: "the definition of the function for kernel_name is not the same for all devices for which the program executable has been built.",
                ErrorCode.CL_INVALID_VALUE: "kernel_name is NULL.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clCreateKernelsInProgram": {
            "args": {
                "program": cl_program,
                "num_kernels": cl_uint,
                "kernels": ptr_cl_kernel,
                "num_kernels_ret": ptr_cl_uint
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_PROGRAM: "program is not a valid program object.",
                ErrorCode.CL_INVALID_PROGRAM_EXECUTABLE: "there is no successfully built executable for any device in program.",
                ErrorCode.CL_INVALID_VALUE: "kernels is not NULL and num_kernels is less than the number of kernels in program.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clGetKernelInfo": {
            "args": {
                "kernel": cl_kernel,
                "param_name": cl_uint,
                "param_value_size": c_size_t,
                "param_value": c_void_p,
                "param_value_size_ret": ptr_size_t
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_KERNEL: "kernel is a not a valid kernel object.",
                ErrorCode.CL_INVALID_VALUE: "param_name is not one of the supported values, or the size in bytes specified by param_value_size is less than size of the return type specified in the Kernel Object Queries table and param_value is not NULL.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clReleaseKernel": {
            "args": {
                "kernel": cl_kernel
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_KERNEL: "kernel is not a valid kernel object.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
    
        "clSetKernelArg": {
            "args": {
                "kernel": cl_kernel,
                "arg_index": cl_uint,
                "arg_size": c_size_t,
                "arg_value": c_void_p
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_KERNEL: "kernel is not a valid kernel object.",
                ErrorCode.CL_INVALID_ARG_INDEX: "arg_index is not a valid argument index.",
                ErrorCode.CL_INVALID_ARG_VALUE: "arg_value specified is not a valid value.",
                ErrorCode.CL_INVALID_MEM_OBJECT: "for an argument declared to be a memory object when the specified arg_value is not a valid memory object.",
                ErrorCode.CL_INVALID_MEM_OBJECT: "for an argument declared to be a depth image, depth image array, multi-sample image, multi-sample image array, multi-sample depth image, or a multi-sample depth image array when the specified arg_value does not follow the rules described above for a depth memory object or memory array object argument.",
                ErrorCode.CL_INVALID_SAMPLER: "for an argument declared to be of type sampler_t when the specified arg_value is not a valid sampler object.",
                ErrorCode.CL_INVALID_DEVICE_QUEUE: "for an argument declared to be of type queue_t when the specified arg_value is not a valid device queue object. This error code is missing before version 2.0.",
                ErrorCode.CL_INVALID_ARG_SIZE: """one of following case happend:
    * arg_size does not match the size of the data type for an argument that is not a memory object, or
    * the argument is a memory object and arg_size != sizeof(cl_mem), or
    * arg_size is zero and the argument is declared with the local qualifier, or
    * the argument is a sampler and arg_size != sizeof(cl_sampler).""",
                ErrorCode.CL_MAX_SIZE_RESTRICTION_EXCEEDED: "the size in bytes of the memory object (if the argument is a memory object) or arg_size (if the argument is declared with local qualifier) exceeds a language- specified maximum size restriction for this argument, such as the MaxByteOffset SPIR-V decoration. This error code is missing before version 2.2.",
                ErrorCode.CL_INVALID_ARG_VALUE: """one of following case happend:
    * the argument is an image declared with the read_only qualifier and arg_value refers to an image object created with cl_mem_flags of CL_MEM_WRITE_ONLY
    * the image argument is declared with the write_only qualifier and arg_value refers to an image object created with cl_mem_flags of CL_MEM_READ_ONLY.""",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },
}
