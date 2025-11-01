from ctypes import c_uint, c_int, c_void_p, c_ulong, c_size_t, POINTER
from typing import List

from .cltypes import (
    cl_platform_info,
    cl_version,
    cl_version_khr,
    cl_name_version,
    cl_name_version_khr,
    cl_ulong,
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
    cl_device_terminate_capability_khr
)


class CLInfo:
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
                "platform": c_void_p,
                "param_name": c_uint,
                "param_value_size": c_size_t,
                "param_value": c_void_p,
                "param_value_size_ret": POINTER(c_size_t)
            },
            "restype": c_int,
            "errors": {
                "CL_INVALID_PLATFORM": "platform is not a valid platform.",
                "CL_INVALID_VALUE": "param_name is not one of the supported values, or if the size in bytes specified by param_value_size is less than size of the return type specified in the Platform Queries table and param_value is not NULL.",
                "CL_OUT_OF_HOST_MEMORY": "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        },

        # cl_int clGetPlatformIDs(
        #   cl_uint num_entries,
        #   cl_platform_id* platforms,
        #   cl_uint* num_platforms
        # );
        "clGetPlatformIDs": {
            "args": {
                "num_entries": c_uint,
                "platforms": POINTER(c_void_p),
                "num_platforms": POINTER(c_uint)
            },
            "restype": c_int,
            "errors": {
                "CL_PLATFORM_NOT_FOUND_KHR": "cl_khr_icd extension is supported and zero platforms are available.",
                "CL_INVALID_VALUE": "num_entries is equal to zero and platforms is not NULL or if both num_platforms and platforms are NULL.",
                "CL_OUT_OF_HOST_MEMORY": "there is a failure to allocate resources required by the OpenCL implementation on the host."
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
                "devices": POINTER(c_void_p),
                "num_devices": POINTER(c_uint)
            },
            "restype": c_int,
            "errors": {
                "CL_INVALID_PLATFORM": "platform is not a valid platform.",
                "CL_INVALID_DEVICE_TYPE": "device_type is not a valid value.",
                "CL_INVALID_VALUE": "num_entries is equal to zero and devices is not NULL or both num_devices and devices are NULL.",
                "CL_DEVICE_NOT_FOUND": "no OpenCL devices that matched device_type were found.",
                "CL_OUT_OF_RESOURCES": "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                "CL_OUT_OF_HOST_MEMORY": "there is a failure to allocate resources required by the OpenCL implementation on the host."
            }
        }, 

        # cl_context clCreateContext(
        #   const cl_context_properties* properties,
        #   cl_uint num_devices,
        #   const cl_device_id* devices,
        #   void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
        #   void* user_data,
        #   cl_int* errcode_ret
        # );
        "clCreateContext": {
            "args": {
                "properties": POINTER(POINTER(c_int)),
                "num_devices": c_uint,
                "devices": POINTER(c_void_p),
                "pfn_notify": c_void_p,
                "user_data": c_void_p,
                "errcode_ret": POINTER(c_int)
            },
            "restype": c_void_p,
            "errors": {
                "CL_INVALID_PLATFORM": "no platform is specified in properties and no platform could be selected, or if the platform specified in properties is not a valid platform.",
                "CL_INVALID_PROPERTY": "a context property name in properties is not a supported property name, if the value specified for a supported property name is not valid, or if the same property name is specified more than once. This error code is missing before version 1.1.",
                "CL_INVALID_VALUE": """one of following case happends:
    * devices is NULL.
    * num_devices is equal to zero.
    * pfn_notify is NULL but user_data is not NULL.""",
                "CL_INVALID_DEVICE": "any device in devices is not a valid device.",
                "CL_DEVICE_NOT_AVAILABLE": "a device in devices is currently not available even though the device was returned by clGetDeviceIDs.",
                "CL_OUT_OF_RESOURCES": "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                "CL_OUT_OF_HOST_MEMORY": "there is a failure to allocate resources required by the OpenCL implementation on the host.",
                "CL_INVALID_DX9_MEDIA_ADAPTER_KHR": "any of the values of the properties CL_CONTEXT_ADAPTER_D3D9_KHR, CL_CONTEXT_ADAPTER_D3D9EX_KHR or CL_CONTEXT_ADAPTER_DXVA_KHR is non-NULL and does not specify a valid media adapter with which the cl_device_ids against which this context is to be created may interoperate.",
                "CL_INVALID_D3D10_DEVICE_KHR": "the value of the property CL_CONTEXT_D3D10_DEVICE_KHR is non-NULL and does not specify a valid Direct3D 10 device with which the cl_device_ids against which this context is to be created may interoperate.",
                "CL_INVALID_OPERATION": "Direct3D 10 interoperability is specified by setting CL_INVALID_D3D10_DEVICE_KHR to a non-NULL value, and interoperability with another graphics API is also specified.",
                "CL_INVALID_D3D11_DEVICE_KHR": "the value of the property CL_CONTEXT_D3D11_DEVICE_KHR is non-NULL and does not specify a valid Direct3D 11 device with which the cl_device_ids against which this context is to be created may interoperate.",
                "CL_INVALID_OPERATION": "Direct3D 11 interoperability is specified by setting CL_INVALID_D3D11_DEVICE_KHR to a non-NULL value, and interoperability with another graphics API is also specified.",
                "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR": """a context was specified for an OpenGL or OpenGL ES implementation using the EGL, GLX, or WGL binding APIs, as described above; and any of the following conditions hold:
    * The specified display and context properties do not identify a valid OpenGL or OpenGL ES context.
    * The specified context does not support buffer and renderbuffer objects.
    * The specified context is not compatible with the OpenCL context being created (for example, it exists in a physically distinct address space, such as another hardware device; or it does not support sharing data with OpenCL due to implementation restrictions).""",
                "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR": "a share group was specified for a CGL-based OpenGL implementation by setting the property CL_CGL_SHAREGROUP_KHR, and the specified share group does not identify a valid CGL share group object.",
                "CL_INVALID_OPERATION": """a context was specified as described above and any of the following conditions hold:
    * A context or share group object was specified for one of CGL, EGL, GLX, or WGL and the OpenGL implementation does not support that window-system binding API.
    * More than one of the properties CL_CGL_SHAREGROUP_KHR, CL_EGL_DISPLAY_KHR, CL_GLX_DISPLAY_KHR, and CL_WGL_HDC_KHR is set to a non-default value.
    * Both of the properties CL_CGL_SHAREGROUP_KHR and CL_GL_CONTEXT_KHR are set to non-default values.
    * Any of the devices specified in the devices argument cannot support OpenCL objects which share the data store of an OpenGL object.""",
                "CL_INVALID_PROPERTY": "both CL_CONTEXT_INTEROP_USER_SYNC, and any of the properties defined by the cl_khr_gl_sharing extension are defined in properties.",
                "CL_INVALID_PROPERTY": "the cl_khr_terminate_context extension is supported and CL_CONTEXT_TERMINATE_KHR is set to CL_TRUE in properties, but not all of the devices associated with the context support the ability to support context termination (i.e. CL_DEVICE_TERMINATE_CAPABILITY_CONTEXT_KHR is set for CL_DEVICE_TERMINATE_CAPABILITY_KHR)."
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
                "device": c_void_p,
                "param_name": c_uint,
                "param_value_size": c_size_t,
                "param_value": c_void_p,
                "param_value_size_ret": POINTER(c_size_t)
            },
            "restype": c_int,
            "errors": {
                "CL_INVALID_DEVICE": "device is not a valid device.",
                "CL_INVALID_VALUE": "param_name is not one of the supported values, or if the size in bytes specified by param_value_size is less than size of the return type specified in the Device Queries table and param_value is not NULL.",
                "CL_OUT_OF_RESOURCES": "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                "CL_OUT_OF_HOST_MEMORY": "there is a failure to allocate resources required by the OpenCL implementation on the host."
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