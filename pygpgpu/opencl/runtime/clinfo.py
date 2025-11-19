from ctypes import (
    c_char, c_ubyte,
    c_int16, c_uint16,
    c_int32, c_uint32,
    c_int64, c_uint64,
    c_float, c_double,
    c_uint, c_int, c_void_p, c_ulong, c_size_t, POINTER, c_int64, c_char_p
)
from ...vectorization.clmath import (
    char2, char3, char4, char8, char16,
    uchar2, uchar3, uchar4, uchar8, uchar16,
    short2, short3, short4, short8, short16,
    ushort2, ushort3, ushort4, ushort8, ushort16,
    int2, int3, int4, int8, int16,
    uint2, uint3, uint4, uint8, uint16,
    long2, long3, long4, long8, long16,
    ulong2, ulong3, ulong4, ulong8, ulong16,
    float2, float3, float4, float8, float16,
    double2, double3, double4, double8, double16
)
import numpy as np
from typing import List

from .cltypes import (
    cl_platform_info,
    cl_program,
    cl_version,
    cl_version_khr,
    cl_name_version,
    cl_name_version_khr,
    cl_kernel,
    cl_ulong,
    cl_command_queue,
    cl_platform_command_buffer_capabilities_khr,
    cl_external_memory_handle_type_khr,
    cl_semaphore_type_khr,
    cl_external_semaphore_handle_type_khr,
    cl_device_info,
    cl_device_type,
    cl_device_mem_cache_type,
    cl_uint,
    cl_bool,
    cl_device_fp_config,
    cl_device_local_mem_type,
    cl_device_exec_capabilities,
    cl_command_queue_properties,
    cl_platform_id,
    cl_device_id,
    cl_device_partition_property,
    cl_device_affinity_domain,
    cl_device_svm_capabilities,
    cl_device_atomic_capabilities,
    cl_device_device_enqueue_capabilities,
    cl_device_command_buffer_capabilities_khr,
    cl_mutable_dispatch_fields_khr,
    cl_device_integer_dot_product_capabilities_khr,
    cl_device_integer_dot_product_acceleration_properties_khr,
    cl_device_pci_bus_info_khr,
    cl_device_terminate_capability_khr,
    cl_context_info,
    cl_context_properties,
    cl_program_info,
    cl_program_build_info,
    cl_build_status,
    cl_program_binary_type,
    cl_kernel_info,
    cl_command_queue_info,
    cl_queue_properties,
    cl_mem_info,
    cl_event_info,
    cl_command_type,
    cl_command_execution_status,
    CL_CONTEXT_NOTIFY_CALLBACK,
    CL_BULD_PROGRAM_CALLBACK,
    CL_EVENT_NOTIFY_CALLBACK,
    ptr_int64,
    ptr_cl_device_id,
    ptr_cl_int,
    ptr_ptr_ubyte,
    ptr_cl_ulong,
    cl_context,
    cl_int,
    cl_bitfield,
    ptr_size_t,
    ptr_cl_platform_id,
    ptr_cl_uint,
    ptr_ptr_char,
    ptr_cl_kernel,
    cl_mem,
    ErrorCode,
    cl_mem_flags,
    cl_mem_properties,
    cl_mem_object_type,
    ptr_cl_event,
    cl_event,
    ptr_cl_image_format,
    ptr_cl_image_desc,
    cl_sampler,
    cl_sampler_info,
    cl_addressing_mode,
    cl_filter_mode,
    cl_sampler_properties
)


class CLInfo:

    image_types = {
        "image2d_t",
        "image3d_t",
        "image2d_array_t",
        "image1d_t",
        "image1d_buffer_t",
        "image1d_array_t",
        "image2d_depth_t",
        "image2d_array_depth_t"
    }

    basic_types = {
        'char': c_char,
        'char2': char2,
        'char3': char3,
        'char4': char4,
        'char8': char8,
        'char16': char16,

        'uchar': c_ubyte,
        'uchar2': uchar2,
        'uchar3': uchar3,
        'uchar4': uchar4,
        'uchar8': uchar8,
        'uchar16': uchar16,

        'short': c_int16,
        'short2': short2,
        'short3': short3,
        'short4': short4,
        'short8': short8,
        'short16': short16,

        'ushort': c_uint16,
        'ushort2': ushort2,
        'ushort3': ushort3,
        'ushort4': ushort4,
        'ushort8': ushort8,
        'ushort16': ushort16,

        'int': c_int32,
        'int2': int2,
        'int3': int3,
        'int4': int4,
        'int8': int8,
        'int16': int16,

        'uint': c_uint32,
        'uint2': uint2,
        'uint3': uint3,
        'uint4': uint4,
        'uint8': uint8,
        'uint16': uint16,

        'long': c_int64,
        'long2': long2,
        'long3': long3,
        'long4': long4,
        'long8': long8,
        'long16': long16,

        'ulong': c_uint64,
        'ulong2': ulong2,
        'ulong3': ulong3,
        'ulong4': ulong4,
        'ulong8': ulong8,
        'ulong16': ulong16,

        'float': c_float,
        'float2': float2,
        'float3': float3,
        'float4': float4,
        'float8': float8,
        'float16': float16,

        'double': c_double,
        'double2': double2,
        'double3': double3,
        'double4': double4,
        'double8': double8,
        'double16': double16,
    }

    scalar_types = {
        'char': c_char,
        'uchar': c_ubyte,
        'short': c_int16,
        'ushort': c_uint16,
        'int': c_int32,
        'uint': c_uint32,
        'long': c_int64,
        'ulong': c_uint64,
        'float': c_float,
        'double': c_double
    }

    vec_types = {
        'char2': char2,
        'char3': char3,
        'char4': char4,
        'char8': char8,
        'char16': char16,

        'uchar2': uchar2,
        'uchar3': uchar3,
        'uchar4': uchar4,
        'uchar8': uchar8,
        'uchar16': uchar16,

        'short2': short2,
        'short3': short3,
        'short4': short4,
        'short8': short8,
        'short16': short16,

        'ushort2': ushort2,
        'ushort3': ushort3,
        'ushort4': ushort4,
        'ushort8': ushort8,
        'ushort16': ushort16,

        'int2': int2,
        'int3': int3,
        'int4': int4,
        'int8': int8,
        'int16': int16,

        'uint2': uint2,
        'uint3': uint3,
        'uint4': uint4,
        'uint8': uint8,
        'uint16': uint16,

        'long2': long2,
        'long3': long3,
        'long4': long4,
        'long8': long8,
        'long16': long16,

        'ulong2': ulong2,
        'ulong3': ulong3,
        'ulong4': ulong4,
        'ulong8': ulong8,
        'ulong16': ulong16,

        'float2': float2,
        'float3': float3,
        'float4': float4,
        'float8': float8,
        'float16': float16,

        'double2': double2,
        'double3': double3,
        'double4': double4,
        'double8': double8,
        'double16': double16,
    }

    dtypes = {
        'char': np.dtype('int8'),
        'char2': np.dtype('int8'),
        'char3': np.dtype('int8'),
        'char4': np.dtype('int8'),
        'char8': np.dtype('int8'),
        'char16': np.dtype('int8'),

        'uchar': np.dtype('uint8'),
        'uchar2': np.dtype('uint8'),
        'uchar3': np.dtype('uint8'),
        'uchar4': np.dtype('uint8'),
        'uchar8': np.dtype('uint8'),
        'uchar16': np.dtype('uint8'),

        'short': np.dtype('int16'),
        'short2': np.dtype('int16'),
        'short3': np.dtype('int16'),
        'short4': np.dtype('int16'),
        'short8': np.dtype('int16'),
        'short16': np.dtype('int16'),

        'ushort': np.dtype('uint16'),
        'ushort2': np.dtype('uint16'),
        'ushort3': np.dtype('uint16'),
        'ushort4': np.dtype('uint16'),
        'ushort8': np.dtype('uint16'),
        'ushort16': np.dtype('uint16'),

        'int': np.dtype('int32'),
        'int2': np.dtype('int32'),
        'int3': np.dtype('int32'),
        'int4': np.dtype('int32'),
        'int8': np.dtype('int32'),
        'int16': np.dtype('int32'),

        'uint': np.dtype('uint32'),
        'uint2': np.dtype('uint32'),
        'uint3': np.dtype('uint32'),
        'uint4': np.dtype('uint32'),
        'uint8': np.dtype('uint32'),
        'uint16': np.dtype('uint32'),

        'long': np.dtype('int64'),
        'long2': np.dtype('int64'),
        'long3': np.dtype('int64'),
        'long4': np.dtype('int64'),
        'long8': np.dtype('int64'),
        'long16': np.dtype('int64'),

        'ulong': np.dtype('uint64'),
        'ulong2': np.dtype('uint64'),
        'ulong3': np.dtype('uint64'),
        'ulong4': np.dtype('uint64'),
        'ulong8': np.dtype('uint64'),
        'ulong16': np.dtype('uint64'),

        'float': np.dtype('float32'),
        'float2': np.dtype('float32'),
        'float3': np.dtype('float32'),
        'float4': np.dtype('float32'),
        'float8': np.dtype('float32'),
        'float16': np.dtype('float32'),

        'double': np.dtype('float64'),
        'double2': np.dtype('float64'),
        'double3': np.dtype('float64'),
        'double4': np.dtype('float64'),
        'double8': np.dtype('float64'),
        'double16': np.dtype('float64'),
    }

    alter_types = {
        'char': (c_char, int),
        'uchar': (c_ubyte, int),
        'short': (c_int16, int),
        'ushort': (c_uint16, int),
        'int': (c_int32, int),
        'uint': (c_uint32, int),
        'long': (c_int64, int),
        'ulong': (c_uint64, int),
        'float': (c_float, float),
        'double': (c_double, float)
    }

    func_signatures = {
        # cl_int clGetPlatformInfo(
        #   cl_platform_id platform,
        #   cl_platform_info param_name,
        #   size_t param_value_size,
        #   void* param_value,
        #   size_t* param_value_size_ret
        # );
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

        # cl_int clGetPlatformIDs(
        #   cl_uint num_entries,
        #   cl_platform_id* platforms,
        #   cl_uint* num_platforms
        # );
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

        # cl_int clGetDeviceIDs(
        #   cl_platform_id platform,
        #   cl_device_type device_type,
        #   cl_uint num_entries,
        #   cl_device_id* devices,
        #   cl_uint* num_devices
        # );
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

        # cl_int clGetDeviceInfo(
        #   cl_device_id device,
        #   cl_device_info param_name,
        #   size_t param_value_size,
        #   void* param_value,
        #   size_t* param_value_size_ret
        # );
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

        # cl_context clCreateContext(
        #   const cl_context_properties* properties,
        #   cl_uint num_devices,
        #   const cl_device_id* devices,
        #   void (CL_CONTEXT_NOTIFY_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
        #   void* user_data,
        #   cl_int* errcode_ret
        # );
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

        # cl_int clReleaseContext(cl_context context);
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

        # cl_int clGetContextInfo(
        #   cl_context context,
        #   cl_context_info param_name,
        #   size_t param_value_size,
        #   void* param_value,
        #   size_t* param_value_size_ret
        # );
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

        # cl_program clCreateProgramWithSource(
        #     cl_context context,
        #     cl_uint count,
        #     const char** strings,
        #     const size_t* lengths,
        #     cl_int* errcode_ret
        # );
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

        # cl_program clCreateProgramWithBinary(
        #     cl_context context,
        #     cl_uint num_devices,
        #     const cl_device_id* device_list,
        #     const size_t* lengths,
        #     const unsigned char** binaries,
        #     cl_int* binary_status,
        #     cl_int* errcode_ret
        # );
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

        # cl_int clBuildProgram(
        #     cl_program program,
        #     cl_uint num_devices,
        #     const cl_device_id* device_list,
        #     const char* options,
        #     void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
        #     void* user_data
        # );
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

        # cl_int clGetProgramInfo(
        #     cl_program program,
        #     cl_program_info param_name,
        #     size_t param_value_size,
        #     void* param_value,
        #     size_t* param_value_size_ret
        # );
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

        # cl_int clGetProgramBuildInfo(
        #     cl_program program,
        #     cl_device_id device,
        #     cl_program_build_info param_name,
        #     size_t param_value_size,
        #     void* param_value,
        #     size_t* param_value_size_ret
        # );
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

        # cl_int clReleaseProgram(cl_program program);
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

        # cl_int clCreateKernelsInProgram(
        #     cl_program program,
        #     cl_uint num_kernels,
        #     cl_kernel* kernels,
        #     cl_uint* num_kernels_ret
        # );
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

        # cl_int clGetKernelInfo(
        #     cl_kernel kernel,
        #     cl_kernel_info param_name,
        #     size_t param_value_size,
        #     void* param_value,
        #     size_t* param_value_size_ret
        # );
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

        # cl_int clReleaseKernel(cl_kernel kernel);
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

        # cl_command_queue clCreateCommandQueue(
        #     cl_context context,
        #     cl_device_id device,
        #     cl_command_queue_properties properties,
        #     cl_int* errcode_ret
        # );
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

        # cl_command_queue clCreateCommandQueueWithProperties(
        #     cl_context context,
        #     cl_device_id device,
        #     const cl_queue_properties* properties,
        #     cl_int* errcode_ret
        # );
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

        # cl_int clGetCommandQueueInfo(
        #     cl_command_queue command_queue,
        #     cl_command_queue_info param_name,
        #     size_t param_value_size,
        #     void* param_value,
        #     size_t* param_value_size_ret
        # );
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

        # cl_int clReleaseCommandQueue(cl_command_queue command_queue);
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

        # cl_mem clCreateBuffer(
        #     cl_context context,
        #     cl_mem_flags flags,
        #     size_t size,
        #     void* host_ptr,
        #     cl_int* errcode_ret
        # );
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

        # cl_mem clCreateBufferWithProperties(
        #     cl_context context,
        #     const cl_mem_properties* properties,
        #     cl_mem_flags flags,
        #     size_t size,
        #     void* host_ptr,
        #     cl_int* errcode_ret
        # );
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

        # cl_int clGetMemObjectInfo(
        #     cl_mem memobj,
        #     cl_mem_info param_name,
        #     size_t param_value_size,
        #     void* param_value,
        #     size_t* param_value_size_ret
        # );
        "clGetMemObjectInfo": {
            "args": {
                "memobj": cl_mem,
                "param_name": cl_mem_info,
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

        # cl_int clReleaseMemObject(cl_mem memobj);
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

        # cl_int clEnqueueWriteBuffer(
        #     cl_command_queue command_queue,
        #     cl_mem buffer,
        #     cl_bool blocking_write,
        #     size_t offset,
        #     size_t size,
        #     const void *ptr,
        #     cl_uint num_events_in_wait_list,
        #     const cl_event *event_wait_list,
        #     cl_event *event
        # );
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

        # cl_int clEnqueueReadBuffer(
        #     cl_command_queue command_queue,
        #     cl_mem buffer,
        #     cl_bool blocking_read,
        #     size_t offset,
        #     size_t size,
        #     void* ptr,
        #     cl_uint num_events_in_wait_list,
        #     const cl_event* event_wait_list,
        #     cl_event* event
        # );
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

        # cl_int clSetKernelArg(
        #     cl_kernel kernel,
        #     cl_uint arg_index,
        #     size_t arg_size,
        #     const void* arg_value
        # );
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

        # cl_int clEnqueueNDRangeKernel(
        #     cl_command_queue command_queue,
        #     cl_kernel kernel,
        #     cl_uint work_dim,
        #     const size_t* global_work_offset,
        #     const size_t* global_work_size,
        #     const size_t* local_work_size,
        #     cl_uint num_events_in_wait_list,
        #     const cl_event* event_wait_list,
        #     cl_event* event
        # );
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

        # cl_int clFinish(cl_command_queue command_queue);
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

        # cl_int clGetEventInfo(
        #     cl_event event,
        #     cl_event_info param_name,
        #     size_t param_value_size,
        #     void* param_value,
        #     size_t* param_value_size_ret
        # );
        "clGetEventInfo": {
            "args": {
                "event": cl_event,
                "param_name": cl_uint,
                "param_value_size": c_size_t,
                "param_value": c_void_p,
                "param_value_size_ret": ptr_size_t
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_EVENT: "event is a not a valid event object.",
                ErrorCode.CL_INVALID_VALUE: "param_name is not one of the supported values, or the size in bytes specified by param_value_size is less than size of the return type specified in the Event Object Queries table and param_value is not NULL.",
                ErrorCode.CL_INVALID_VALUE: "the information to query given in param_name cannot be queried for event.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },

        # cl_int clWaitForEvents(
        #     cl_uint num_events,
        #     const cl_event* event_list
        # );
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

        # cl_event clCreateUserEvent(
        #     cl_context context,
        #     cl_int* errcode_ret
        # );
        "clCreateUserEvent": {
            "args": {
                "context": cl_context,
                "errcode_ret": ptr_cl_int
            },
            "restype": cl_event,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: "if context is not a valid context.",
                ErrorCode.CL_OUT_OF_RESOURCES: "if there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "if there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },

        # cl_int clSetUserEventStatus(
        #     cl_event event,
        #     cl_int execution_status
        # );
        "clSetUserEventStatus": {
            "args": {
                "event": cl_event,
                "execution_status": cl_int
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_EVENT: "event is not a valid user event object.",
                ErrorCode.CL_INVALID_VALUE: "the execution_status is not CL_COMPLETE or a negative integer value.",
                ErrorCode.CL_INVALID_OPERATION: "the execution_status for event has already been changed by a previous call to clSetUserEventStatus.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },

        # cl_int clSetEventCallback(
        #     cl_event event,
        #     cl_int command_exec_callback_type,
        #     void (CL_CALLBACK* pfn_notify)(cl_event event, cl_int event_command_status, void *user_data),
        #     void* user_data
        # );
        "clSetEventCallback": {
            "args": {
                "event": cl_event,
                "command_exec_callback_type": cl_int,
                "pfn_notify": CL_EVENT_NOTIFY_CALLBACK,
                "user_data": c_void_p
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_EVENT: "event is not a valid event object.",
                ErrorCode.CL_INVALID_VALUE: "pfn_event_notify is NULL or command_exec_callback_type is not CL_SUBMITTED, CL_RUNNING, or CL_COMPLETE.",
                ErrorCode.CL_OUT_OF_RESOURCES: "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },

        # cl_int clReleaseEvent(cl_event event);
        "clReleaseEvent": {
            "args": {
                "event": cl_event
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_EVENT: "if event is not a valid event object.",
                ErrorCode.CL_OUT_OF_RESOURCES: "if there is a failure to allocate resources required by the OpenCL implementation on the device.",
                ErrorCode.CL_OUT_OF_HOST_MEMORY: "if there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },

        # cl_mem clCreateImage(
        #     cl_context context,
        #     cl_mem_flags flags,
        #     const cl_image_format* image_format,
        #     const cl_image_desc* image_desc,
        #     void* host_ptr,
        #     cl_int* errcode_ret
        # );
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

        # cl_mem clCreateImageWithProperties(
        #     cl_context context,
        #     const cl_mem_properties* properties,
        #     cl_mem_flags flags,
        #     const cl_image_format* image_format,
        #     const cl_image_desc* image_desc,
        #     void* host_ptr,
        #     cl_int* errcode_ret
        # );
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

        # cl_int clEnqueueReadImage(
        #     cl_command_queue command_queue,
        #     cl_mem image,
        #     cl_bool blocking_read,
        #     const size_t* origin,
        #     const size_t* region,
        #     size_t row_pitch,
        #     size_t slice_pitch,
        #     void* ptr,
        #     cl_uint num_events_in_wait_list,
        #     const cl_event* event_wait_list,
        #     cl_event* event
        # );
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

        # cl_int clEnqueueWriteImage(
        #     cl_command_queue command_queue,
        #     cl_mem image,
        #     cl_bool blocking_write,
        #     const size_t* origin,
        #     const size_t* region,
        #     size_t input_row_pitch,
        #     size_t input_slice_pitch,
        #     const void* ptr,
        #     cl_uint num_events_in_wait_list,
        #     const cl_event* event_wait_list,
        #     cl_event* event
        # );
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

        # cl_sampler clCreateSampler(
        #     cl_context context,
        #     cl_bool normalized_coords,
        #     cl_addressing_mode addressing_mode,
        #     cl_filter_mode filter_mode,
        #     cl_int* errcode_ret
        # );
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

        # cl_sampler clCreateSamplerWithProperties(
        #     cl_context context,
        #     const cl_sampler_properties* sampler_properties,
        #     cl_int* errcode_ret
        # );
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

        # cl_int clGetSamplerInfo(
        #     cl_sampler sampler,
        #     cl_sampler_info param_name,
        #     size_t param_value_size,
        #     void* param_value,
        #     size_t* param_value_size_ret
        # );
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

        # cl_int clReleaseSampler(cl_sampler sampler);
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
        }
    }

    platform_info_types = {
        cl_platform_info.CL_PLATFORM_PROFILE: str,
        cl_platform_info.CL_PLATFORM_VERSION: str,
        cl_platform_info.CL_PLATFORM_NUMERIC_VERSION: cl_version,
        cl_platform_info.CL_PLATFORM_NUMERIC_VERSION_KHR: cl_version_khr,
        cl_platform_info.CL_PLATFORM_NAME: str,
        cl_platform_info.CL_PLATFORM_VENDOR: str,
        cl_platform_info.CL_PLATFORM_EXTENSIONS: str,
        cl_platform_info.CL_PLATFORM_EXTENSIONS_WITH_VERSION: List[cl_name_version],
        cl_platform_info.CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR: List[cl_name_version_khr],
        cl_platform_info.CL_PLATFORM_HOST_TIMER_RESOLUTION: List[cl_ulong],
        cl_platform_info.CL_PLATFORM_COMMAND_BUFFER_CAPABILITIES_KHR: cl_platform_command_buffer_capabilities_khr,
        cl_platform_info.CL_PLATFORM_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR: List[cl_external_memory_handle_type_khr],
        cl_platform_info.CL_PLATFORM_SEMAPHORE_TYPES_KHR: List[cl_semaphore_type_khr],
        cl_platform_info.CL_PLATFORM_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR: List[cl_external_semaphore_handle_type_khr],
        cl_platform_info.CL_PLATFORM_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR: List[cl_external_semaphore_handle_type_khr],
        cl_platform_info.CL_PLATFORM_ICD_SUFFIX_KHR: str
    }

    device_info_types = {
        cl_device_info.CL_DEVICE_TYPE: cl_device_type,
        cl_device_info.CL_DEVICE_VENDOR_ID: cl_uint,
        cl_device_info.CL_DEVICE_MAX_COMPUTE_UNITS: cl_uint,
        cl_device_info.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: cl_uint,
        cl_device_info.CL_DEVICE_MAX_WORK_ITEM_SIZES: List[c_size_t],
        cl_device_info.CL_DEVICE_MAX_WORK_GROUP_SIZE: c_size_t,
        cl_device_info.CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: cl_uint,
        cl_device_info.CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: cl_uint,
        cl_device_info.CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: cl_uint,
        cl_device_info.CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: cl_uint,
        cl_device_info.CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: cl_uint,
        cl_device_info.CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: cl_uint,
        cl_device_info.CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF: cl_uint,
        cl_device_info.CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF: cl_uint,
        cl_device_info.CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR: cl_uint,
        cl_device_info.CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT: cl_uint,
        cl_device_info.CL_DEVICE_NATIVE_VECTOR_WIDTH_INT: cl_uint,
        cl_device_info.CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG: cl_uint,
        cl_device_info.CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT: cl_uint,
        cl_device_info.CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE: cl_uint,
        cl_device_info.CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF: cl_uint,
        cl_device_info.CL_DEVICE_MAX_CLOCK_FREQUENCY: cl_uint,
        cl_device_info.CL_DEVICE_ADDRESS_BITS: cl_uint,
        cl_device_info.CL_DEVICE_MAX_MEM_ALLOC_SIZE: cl_ulong,
        cl_device_info.CL_DEVICE_IMAGE_SUPPORT: cl_bool,
        cl_device_info.CL_DEVICE_MAX_READ_IMAGE_ARGS: cl_uint,
        cl_device_info.CL_DEVICE_MAX_WRITE_IMAGE_ARGS: cl_uint,
        cl_device_info.CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS: cl_uint,
        cl_device_info.CL_DEVICE_IL_VERSION: str,
        cl_device_info.CL_DEVICE_IL_VERSION_KHR: str,
        cl_device_info.CL_DEVICE_ILS_WITH_VERSION: List[cl_name_version],
        cl_device_info.CL_DEVICE_ILS_WITH_VERSION_KHR: List[cl_name_version_khr],
        # cl_device_info.CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR: List[str],
        # cl_device_info.CL_DEVICE_SPIRV_EXTENSIONS_KHR: List[str],
        # cl_device_info.CL_DEVICE_SPIRV_CAPABILITIES_KHR: List[cl_uint],
        cl_device_info.CL_DEVICE_IMAGE2D_MAX_WIDTH: c_size_t,
        cl_device_info.CL_DEVICE_IMAGE2D_MAX_HEIGHT: c_size_t,
        cl_device_info.CL_DEVICE_IMAGE3D_MAX_WIDTH: c_size_t,
        cl_device_info.CL_DEVICE_IMAGE3D_MAX_HEIGHT: c_size_t,
        cl_device_info.CL_DEVICE_IMAGE3D_MAX_DEPTH: c_size_t,
        cl_device_info.CL_DEVICE_IMAGE_MAX_BUFFER_SIZE: c_size_t,
        cl_device_info.CL_DEVICE_IMAGE_MAX_ARRAY_SIZE: c_size_t,
        cl_device_info.CL_DEVICE_MAX_SAMPLERS: cl_uint,
        cl_device_info.CL_DEVICE_IMAGE_PITCH_ALIGNMENT: cl_uint,
        cl_device_info.CL_DEVICE_IMAGE_PITCH_ALIGNMENT_KHR: cl_uint,
        cl_device_info.CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT: cl_uint,
        cl_device_info.CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT_KHR: cl_uint,
        cl_device_info.CL_DEVICE_MAX_PIPE_ARGS: cl_uint,
        cl_device_info.CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS: cl_uint,
        cl_device_info.CL_DEVICE_PIPE_MAX_PACKET_SIZE: cl_uint,
        cl_device_info.CL_DEVICE_MAX_PARAMETER_SIZE: c_size_t,
        cl_device_info.CL_DEVICE_MEM_BASE_ADDR_ALIGN: cl_uint,
        cl_device_info.CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: cl_uint,
        cl_device_info.CL_DEVICE_SINGLE_FP_CONFIG: cl_device_fp_config,
        cl_device_info.CL_DEVICE_DOUBLE_FP_CONFIG: cl_device_fp_config,
        cl_device_info.CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: cl_device_mem_cache_type,
        cl_device_info.CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: cl_uint,
        cl_device_info.CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: cl_ulong,
        cl_device_info.CL_DEVICE_GLOBAL_MEM_SIZE: cl_ulong,
        cl_device_info.CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: cl_ulong,
        cl_device_info.CL_DEVICE_MAX_CONSTANT_ARGS: cl_uint,
        cl_device_info.CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE: c_size_t,
        cl_device_info.CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE: c_size_t,
        cl_device_info.CL_DEVICE_LOCAL_MEM_TYPE: cl_device_local_mem_type,
        cl_device_info.CL_DEVICE_LOCAL_MEM_SIZE: cl_ulong,
        cl_device_info.CL_DEVICE_ERROR_CORRECTION_SUPPORT: cl_bool,
        cl_device_info.CL_DEVICE_HOST_UNIFIED_MEMORY: cl_bool,
        cl_device_info.CL_DEVICE_PROFILING_TIMER_RESOLUTION: c_size_t,
        cl_device_info.CL_DEVICE_ENDIAN_LITTLE: cl_bool,
        cl_device_info.CL_DEVICE_AVAILABLE: cl_bool,
        cl_device_info.CL_DEVICE_COMPILER_AVAILABLE: cl_bool,
        cl_device_info.CL_DEVICE_LINKER_AVAILABLE: cl_bool,
        cl_device_info.CL_DEVICE_EXECUTION_CAPABILITIES: cl_device_exec_capabilities,
        cl_device_info.CL_DEVICE_QUEUE_PROPERTIES: cl_command_queue_properties,
        cl_device_info.CL_DEVICE_QUEUE_ON_HOST_PROPERTIES: cl_command_queue_properties,
        cl_device_info.CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES: cl_command_queue_properties,
        cl_device_info.CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE: cl_uint,
        cl_device_info.CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE: cl_uint,
        cl_device_info.CL_DEVICE_MAX_ON_DEVICE_QUEUES: cl_uint,
        cl_device_info.CL_DEVICE_MAX_ON_DEVICE_EVENTS: cl_uint,
        cl_device_info.CL_DEVICE_BUILT_IN_KERNELS: str,
        cl_device_info.CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION: List[cl_name_version],
        cl_device_info.CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION_KHR: List[cl_name_version_khr],
        cl_device_info.CL_DEVICE_PLATFORM: cl_platform_id,
        cl_device_info.CL_DEVICE_NAME: str,
        cl_device_info.CL_DEVICE_VENDOR: str,
        cl_device_info.CL_DRIVER_VERSION: str,
        cl_device_info.CL_DEVICE_PROFILE: str,
        cl_device_info.CL_DEVICE_VERSION: str,
        cl_device_info.CL_DEVICE_NUMERIC_VERSION: cl_version,
        cl_device_info.CL_DEVICE_NUMERIC_VERSION_KHR: cl_version_khr,
        cl_device_info.CL_DEVICE_OPENCL_C_VERSION: str,
        cl_device_info.CL_DEVICE_OPENCL_C_ALL_VERSIONS: List[cl_name_version],
        cl_device_info.CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR: cl_version_khr,
        cl_device_info.CL_DEVICE_OPENCL_C_FEATURES: List[cl_name_version],
        cl_device_info.CL_DEVICE_EXTENSIONS: str,
        cl_device_info.CL_DEVICE_EXTENSIONS_WITH_VERSION: List[cl_name_version],
        cl_device_info.CL_DEVICE_EXTENSIONS_WITH_VERSION_KHR: List[cl_name_version_khr],
        cl_device_info.CL_DEVICE_PRINTF_BUFFER_SIZE: c_size_t,
        cl_device_info.CL_DEVICE_PREFERRED_INTEROP_USER_SYNC: cl_bool,
        cl_device_info.CL_DEVICE_PARENT_DEVICE: cl_device_id,
        cl_device_info.CL_DEVICE_PARTITION_MAX_SUB_DEVICES: cl_uint,
        cl_device_info.CL_DEVICE_PARTITION_PROPERTIES: List[cl_device_partition_property],
        cl_device_info.CL_DEVICE_PARTITION_AFFINITY_DOMAIN: cl_device_affinity_domain,
        cl_device_info.CL_DEVICE_PARTITION_TYPE: List[cl_device_partition_property],
        cl_device_info.CL_DEVICE_REFERENCE_COUNT: cl_uint,
        cl_device_info.CL_DEVICE_SVM_CAPABILITIES: cl_device_svm_capabilities,
        cl_device_info.CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT: cl_uint,
        cl_device_info.CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT: cl_uint,
        cl_device_info.CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT: cl_uint,
        cl_device_info.CL_DEVICE_MAX_NUM_SUB_GROUPS: cl_uint,
        cl_device_info.CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: cl_bool,
        cl_device_info.CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES: cl_device_atomic_capabilities,
        cl_device_info.CL_DEVICE_ATOMIC_FENCE_CAPABILITIES: cl_device_atomic_capabilities,
        cl_device_info.CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT: cl_bool,
        cl_device_info.CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT: cl_bool,
        cl_device_info.CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT: cl_bool,
        cl_device_info.CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES: cl_device_device_enqueue_capabilities,
        cl_device_info.CL_DEVICE_PIPE_SUPPORT: cl_bool,
        cl_device_info.CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: c_size_t,
        cl_device_info.CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED: str,
        cl_device_info.CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR: cl_device_command_buffer_capabilities_khr,
        cl_device_info.CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR: cl_command_queue_properties,
        # cl_device_info.CL_DEVICE_COMMAND_BUFFER_SUPPORTED_QUEUE_PROPERTIES_KHR: cl_command_queue_properties,
        cl_device_info.CL_DEVICE_COMMAND_BUFFER_NUM_SYNC_DEVICES_KHR: cl_uint,
        cl_device_info.CL_DEVICE_COMMAND_BUFFER_SYNC_DEVICES_KHR: List[cl_device_id],
        cl_device_info.CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR: cl_mutable_dispatch_fields_khr,
        cl_device_info.CL_DEVICE_UUID_KHR: bytes,
        cl_device_info.CL_DRIVER_UUID_KHR: bytes,
        cl_device_info.CL_DEVICE_LUID_VALID_KHR: cl_bool,
        cl_device_info.CL_DEVICE_LUID_KHR: bytes,
        cl_device_info.CL_DEVICE_NODE_MASK_KHR: cl_uint,
        cl_device_info.CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR: List[cl_external_memory_handle_type_khr],
        cl_device_info.CL_DEVICE_EXTERNAL_MEMORY_IMPORT_ASSUME_LINEAR_IMAGES_HANDLE_TYPES_KHR: List[cl_external_memory_handle_type_khr],
        cl_device_info.CL_DEVICE_HALF_FP_CONFIG: cl_device_fp_config,
        cl_device_info.CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR: cl_device_integer_dot_product_capabilities_khr,
        cl_device_info.CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR: cl_device_integer_dot_product_acceleration_properties_khr,
        cl_device_info.CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR: cl_device_integer_dot_product_acceleration_properties_khr,
        # cl_device_info.CL_DEVICE_KERNEL_CLOCK_CAPABILITIES_KHR: cl_device_kernel_clock_capabilities_khr,
        cl_device_info.CL_DEVICE_PCI_BUS_INFO_KHR: cl_device_pci_bus_info_khr,
        cl_device_info.CL_DEVICE_SEMAPHORE_TYPES_KHR: List[cl_semaphore_type_khr],
        cl_device_info.CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR: List[cl_external_semaphore_handle_type_khr],
        cl_device_info.CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR: List[cl_external_semaphore_handle_type_khr],
        cl_device_info.CL_DEVICE_SPIR_VERSIONS: str,
        cl_device_info.CL_DEVICE_MAX_NAMED_BARRIER_COUNT_KHR: cl_uint,
        cl_device_info.CL_DEVICE_TERMINATE_CAPABILITY_KHR: cl_device_terminate_capability_khr,
        cl_device_info.CL_DEVICE_CXX_FOR_OPENCL_NUMERIC_VERSION_EXT: cl_version
    }

    context_info_types = {
        cl_context_info.CL_CONTEXT_REFERENCE_COUNT: cl_uint,
        cl_context_info.CL_CONTEXT_NUM_DEVICES: cl_uint,
        cl_context_info.CL_CONTEXT_DEVICES: List[cl_device_id],
        cl_context_info.CL_CONTEXT_PROPERTIES: List[cl_context_properties],
        cl_context_info.CL_CONTEXT_D3D10_PREFER_SHARED_RESOURCES_KHR: cl_bool,
        cl_context_info.CL_CONTEXT_D3D11_PREFER_SHARED_RESOURCES_KHR: cl_bool
    }

    program_info_types = {
        cl_program_info.CL_PROGRAM_REFERENCE_COUNT: cl_uint,
        cl_program_info.CL_PROGRAM_CONTEXT: cl_context,
        cl_program_info.CL_PROGRAM_NUM_DEVICES: cl_uint,
        cl_program_info.CL_PROGRAM_DEVICES: List[cl_device_id],
        cl_program_info.CL_PROGRAM_SOURCE: str,
        cl_program_info.CL_PROGRAM_IL: str,
        cl_program_info.CL_PROGRAM_IL_KHR: str,
        cl_program_info.CL_PROGRAM_BINARY_SIZES: List[c_size_t],
        cl_program_info.CL_PROGRAM_BINARIES: List[bytes],
        cl_program_info.CL_PROGRAM_NUM_KERNELS: c_size_t,
        cl_program_info.CL_PROGRAM_KERNEL_NAMES: str,
        cl_program_info.CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT: cl_bool,
        cl_program_info.CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT: cl_bool
    }

    program_build_info_types = {
        cl_program_build_info.CL_PROGRAM_BUILD_STATUS: cl_build_status,
        cl_program_build_info.CL_PROGRAM_BUILD_OPTIONS: str,
        cl_program_build_info.CL_PROGRAM_BUILD_LOG: str,
        cl_program_build_info.CL_PROGRAM_BINARY_TYPE: cl_program_binary_type,
        cl_program_build_info.CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE: c_size_t
    }

    kernel_info_types = {
        cl_kernel_info.CL_KERNEL_FUNCTION_NAME: str,
        cl_kernel_info.CL_KERNEL_NUM_ARGS: cl_uint,
        cl_kernel_info.CL_KERNEL_REFERENCE_COUNT: cl_uint,
        cl_kernel_info.CL_KERNEL_CONTEXT: cl_context,
        cl_kernel_info.CL_KERNEL_PROGRAM: cl_program,
        cl_kernel_info.CL_KERNEL_ATTRIBUTES: str
    }

    command_queue_info_types = {
        cl_command_queue_info.CL_QUEUE_CONTEXT: cl_context,
        cl_command_queue_info.CL_QUEUE_DEVICE: cl_device_id,
        cl_command_queue_info.CL_QUEUE_REFERENCE_COUNT: cl_uint,
        cl_command_queue_info.CL_QUEUE_PROPERTIES: cl_command_queue_properties,
        cl_command_queue_info.CL_QUEUE_PROPERTIES_ARRAY: List[cl_queue_properties],
        cl_command_queue_info.CL_QUEUE_SIZE: cl_uint,
        cl_command_queue_info.CL_QUEUE_DEVICE_DEFAULT: cl_command_queue
    }

    mem_info_types = {
        cl_mem_info.CL_MEM_TYPE: cl_mem_object_type,
        cl_mem_info.CL_MEM_FLAGS: cl_mem_flags,
        cl_mem_info.CL_MEM_SIZE: c_size_t,
        cl_mem_info.CL_MEM_HOST_PTR: c_void_p,
        cl_mem_info.CL_MEM_MAP_COUNT: cl_uint,
        cl_mem_info.CL_MEM_REFERENCE_COUNT: cl_uint,
        cl_mem_info.CL_MEM_CONTEXT: cl_context,
        cl_mem_info.CL_MEM_ASSOCIATED_MEMOBJECT: cl_mem,
        cl_mem_info.CL_MEM_OFFSET: c_size_t,
        cl_mem_info.CL_MEM_USES_SVM_POINTER: cl_bool,
        cl_mem_info.CL_MEM_PROPERTIES: List[cl_mem_properties],
        cl_mem_info.CL_MEM_DX9_MEDIA_ADAPTER_TYPE_KHR: cl_uint,
        # cl_mem_info.CL_MEM_DX9_MEDIA_SURFACE_INFO_KHR: cl_dx9_surface_info_khr,
        cl_mem_info.CL_MEM_D3D10_RESOURCE_KHR: c_void_p,
        cl_mem_info.CL_MEM_D3D11_RESOURCE_KHR: c_void_p,
        # cl_mem_info.CL_MEM_DEVICE_ADDRESS_EXT: List[cl_mem_device_address_ext]
    }

    event_info_types = {
        cl_event_info.CL_EVENT_COMMAND_QUEUE: cl_command_queue,
        cl_event_info.CL_EVENT_CONTEXT: cl_context,
        cl_event_info.CL_EVENT_COMMAND_TYPE: cl_command_type,
        cl_event_info.CL_EVENT_COMMAND_EXECUTION_STATUS: cl_int,
        cl_event_info.CL_EVENT_REFERENCE_COUNT: cl_uint
    }

    sampler_info_types = {
        cl_sampler_info.CL_SAMPLER_REFERENCE_COUNT: cl_uint,
        cl_sampler_info.CL_SAMPLER_CONTEXT: cl_context,
        cl_sampler_info.CL_SAMPLER_NORMALIZED_COORDS: cl_bool,
        cl_sampler_info.CL_SAMPLER_ADDRESSING_MODE: cl_addressing_mode,
        cl_sampler_info.CL_SAMPLER_FILTER_MODE: cl_filter_mode,
        cl_sampler_info.CL_SAMPLER_PROPERTIES: List[cl_sampler_properties]
    }

    no_cached_info = {
        cl_device_info.CL_DEVICE_REFERENCE_COUNT,
        cl_context_info.CL_CONTEXT_REFERENCE_COUNT,
        cl_program_info.CL_PROGRAM_REFERENCE_COUNT,
        cl_kernel_info.CL_KERNEL_REFERENCE_COUNT,
        cl_event_info.CL_EVENT_REFERENCE_COUNT,
        cl_event_info.CL_EVENT_COMMAND_EXECUTION_STATUS,
        cl_sampler_info.CL_SAMPLER_REFERENCE_COUNT
    }