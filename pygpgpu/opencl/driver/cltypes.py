from ctypes import c_char, c_uint64, c_ubyte, c_uint32, c_int32, c_int64, c_void_p, c_size_t, c_char_p, POINTER, WINFUNCTYPE, Structure, CFUNCTYPE
from ...constants import IntConstant, IntEnum, IntFlag
from typing import TypeAlias
import sys

cl_int = c_int32
cl_uint = c_uint32
cl_ulong = c_uint64
cl_long = c_int64

cl_platform_id = c_void_p
cl_device_id = c_void_p
cl_context = c_void_p
cl_program = c_void_p
cl_kernel = c_void_p
cl_command_queue = c_void_p
cl_mem = c_void_p
cl_sampler = c_void_p

cl_bitfield = cl_ulong
cl_semaphore_type_khr = cl_uint
cl_properties = cl_ulong
cl_event = c_void_p

ptr_cl_platform_id:TypeAlias = POINTER(cl_platform_id)
ptr_cl_uint:TypeAlias = POINTER(cl_uint)
ptr_cl_int:TypeAlias = POINTER(cl_int)
ptr_size_t:TypeAlias = POINTER(c_size_t)
ptr_cl_device_id:TypeAlias = POINTER(cl_device_id)
ptr_int64:TypeAlias = POINTER(c_int64)
ptr_ptr_char:TypeAlias = POINTER(c_char_p)
ptr_ubyte:TypeAlias = POINTER(c_ubyte)
ptr_ptr_ubyte:TypeAlias = POINTER(ptr_ubyte)
ptr_cl_kernel:TypeAlias = POINTER(cl_kernel)
ptr_cl_ulong:TypeAlias = POINTER(cl_ulong)
ptr_cl_event:TypeAlias = POINTER(cl_event)
ptr_cl_mem:TypeAlias = POINTER(cl_mem)
ptr_ptr_void:TypeAlias = POINTER(c_void_p)
CL_CONTEXT_NOTIFY_CALLBACK:TypeAlias = CFUNCTYPE(None, c_char_p, c_void_p, c_size_t, c_void_p)
CL_BULD_PROGRAM_CALLBACK:TypeAlias = CFUNCTYPE(None, cl_program, c_void_p)
CL_EVENT_NOTIFY_CALLBACK:TypeAlias = CFUNCTYPE(None, cl_event, cl_int, c_void_p)

setattr(sys.modules[__name__], "LP_c_void_p", ptr_ptr_void)


class ErrorCode(IntEnum):
    CL_SUCCESS = 0
    CL_DEVICE_NOT_FOUND = -1
    CL_DEVICE_NOT_AVAILABLE = -2
    CL_COMPILER_NOT_AVAILABLE = -3
    CL_MEM_OBJECT_ALLOCATION_FAILURE = -4
    CL_OUT_OF_RESOURCES = -5
    CL_OUT_OF_HOST_MEMORY = -6
    CL_PROFILING_INFO_NOT_AVAILABLE = -7
    CL_MEM_COPY_OVERLAP = -8
    CL_IMAGE_FORMAT_MISMATCH = -9
    CL_IMAGE_FORMAT_NOT_SUPPORTED = -10
    CL_BUILD_PROGRAM_FAILURE = -11
    CL_MAP_FAILURE = -12
    CL_MISALIGNED_SUB_BUFFER_OFFSET = -13
    CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14
    CL_COMPILE_PROGRAM_FAILURE = -15
    CL_LINKER_NOT_AVAILABLE = -16
    CL_LINK_PROGRAM_FAILURE = -17
    CL_DEVICE_PARTITION_FAILED = -18
    CL_KERNEL_ARG_INFO_NOT_AVAILABLE = -19
    CL_INVALID_VALUE = -30
    CL_INVALID_DEVICE_TYPE = -31
    CL_INVALID_PLATFORM = -32
    CL_INVALID_DEVICE = -33
    CL_INVALID_CONTEXT = -34
    CL_INVALID_QUEUE_PROPERTIES = -35
    CL_INVALID_COMMAND_QUEUE = -36
    CL_INVALID_HOST_PTR = -37
    CL_INVALID_MEM_OBJECT = -38
    CL_INVALID_IMAGE_FORMAT_DESCRIPTOR = -39
    CL_INVALID_IMAGE_SIZE = -40
    CL_INVALID_SAMPLER = -41
    CL_INVALID_BINARY = -42
    CL_INVALID_BUILD_OPTIONS = -43
    CL_INVALID_PROGRAM = -44
    CL_INVALID_PROGRAM_EXECUTABLE = -45
    CL_INVALID_KERNEL_NAME = -46
    CL_INVALID_KERNEL_DEFINITION = -47
    CL_INVALID_KERNEL = -48
    CL_INVALID_ARG_INDEX = -49
    CL_INVALID_ARG_VALUE = -50
    CL_INVALID_ARG_SIZE = -51
    CL_INVALID_KERNEL_ARGS = -52
    CL_INVALID_WORK_DIMENSION = -53
    CL_INVALID_WORK_GROUP_SIZE = -54
    CL_INVALID_WORK_ITEM_SIZE = -55
    CL_INVALID_GLOBAL_OFFSET = -56
    CL_INVALID_EVENT_WAIT_LIST = -57
    CL_INVALID_EVENT = -58
    CL_INVALID_OPERATION = -59
    CL_INVALID_GL_OBJECT = -60
    CL_INVALID_BUFFER_SIZE = -61
    CL_INVALID_MIP_LEVEL = -62
    CL_INVALID_GLOBAL_WORK_SIZE = -63
    CL_INVALID_PROPERTY = -64
    CL_INVALID_IMAGE_DESCRIPTOR = -65
    CL_INVALID_COMPILER_OPTIONS = -66
    CL_INVALID_LINKER_OPTIONS = -67
    CL_INVALID_DEVICE_PARTITION_COUNT = -68
    CL_INVALID_PIPE_SIZE = -69
    CL_INVALID_DEVICE_QUEUE = -70
    CL_INVALID_SPEC_ID = -71
    CL_MAX_SIZE_RESTRICTION_EXCEEDED = -72
    CL_INVALID_D3D9_DEVICE_NV = -1010
    CL_INVALID_D3D9_RESOURCE_NV = -1011
    CL_D3D9_RESOURCE_ALREADY_ACQUIRED_NV = -1012
    CL_D3D9_RESOURCE_NOT_ACQUIRED_NV = -1013
    CL_INVALID_D3D10_DEVICE_NV = -1002
    CL_INVALID_D3D10_RESOURCE_NV = -1003
    CL_D3D10_RESOURCE_ALREADY_ACQUIRED_NV = -1004
    CL_D3D10_RESOURCE_NOT_ACQUIRED_NV = -1005
    CL_INVALID_D3D11_DEVICE_NV = -1006
    CL_INVALID_D3D11_RESOURCE_NV = -1007
    CL_D3D11_RESOURCE_ALREADY_ACQUIRED_NV = -1008
    CL_D3D11_RESOURCE_NOT_ACQUIRED_NV = -1009
    CL_PLATFORM_NOT_FOUND_KHR = -1001
    CL_INVALID_DX9_MEDIA_ADAPTER_KHR          = -1010
    CL_INVALID_DX9_MEDIA_SURFACE_KHR          = -1011
    CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR = -1012
    CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR     = -1013
    CL_INVALID_D3D10_DEVICE_KHR               = -1002
    CL_INVALID_D3D10_RESOURCE_KHR             = -1003
    CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR    = -1004
    CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR        = -1005
    CL_INVALID_D3D11_DEVICE_KHR               = -1006
    CL_INVALID_D3D11_RESOURCE_KHR             = -1007
    CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR    = -1008
    CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR        = -1009
    CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR    = -1000

    @classmethod
    def dtype(cls):
        return cl_int

class cl_device_type(IntFlag):
    CL_DEVICE_TYPE_DEFAULT = (1 << 0)
    CL_DEVICE_TYPE_CPU = (1 << 1)
    CL_DEVICE_TYPE_GPU = (1 << 2)
    CL_DEVICE_TYPE_ACCELERATOR = (1 << 3)
    CL_DEVICE_TYPE_CUSTOM = (1 << 4)
    CL_DEVICE_TYPE_ALL = 0xFFFFFFFF

class cl_platform_info(IntEnum):
    CL_PLATFORM_PROFILE = 0x0900
    CL_PLATFORM_VERSION = 0x0901
    CL_PLATFORM_NUMERIC_VERSION = 0x0906
    CL_PLATFORM_NUMERIC_VERSION_KHR = 0x0906
    CL_PLATFORM_NAME = 0x0902
    CL_PLATFORM_VENDOR = 0x0903
    CL_PLATFORM_EXTENSIONS = 0x0904
    CL_PLATFORM_EXTENSIONS_WITH_VERSION = 0x0907
    CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR = 0x0907
    CL_PLATFORM_HOST_TIMER_RESOLUTION = 0x0905
    CL_PLATFORM_COMMAND_BUFFER_CAPABILITIES_KHR = 0x0908
    CL_PLATFORM_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR = 0x2044
    CL_PLATFORM_SEMAPHORE_TYPES_KHR = 0x2036
    CL_PLATFORM_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR = 0x2037
    CL_PLATFORM_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR = 0x2038
    CL_PLATFORM_ICD_SUFFIX_KHR = 0x0920

class cl_device_info(IntEnum):
    CL_DEVICE_TYPE                                    = 0x1000
    CL_DEVICE_VENDOR_ID                               = 0x1001
    CL_DEVICE_MAX_COMPUTE_UNITS                       = 0x1002
    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS                = 0x1003
    CL_DEVICE_MAX_WORK_GROUP_SIZE                     = 0x1004
    CL_DEVICE_MAX_WORK_ITEM_SIZES                     = 0x1005
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR             = 0x1006
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT            = 0x1007
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT              = 0x1008
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG             = 0x1009
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT            = 0x100A
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE           = 0x100B
    CL_DEVICE_MAX_CLOCK_FREQUENCY                     = 0x100C
    CL_DEVICE_ADDRESS_BITS                            = 0x100D
    CL_DEVICE_MAX_READ_IMAGE_ARGS                     = 0x100E
    CL_DEVICE_MAX_WRITE_IMAGE_ARGS                    = 0x100F
    CL_DEVICE_MAX_MEM_ALLOC_SIZE                      = 0x1010
    CL_DEVICE_IMAGE2D_MAX_WIDTH                       = 0x1011
    CL_DEVICE_IMAGE2D_MAX_HEIGHT                      = 0x1012
    CL_DEVICE_IMAGE3D_MAX_WIDTH                       = 0x1013
    CL_DEVICE_IMAGE3D_MAX_HEIGHT                      = 0x1014
    CL_DEVICE_IMAGE3D_MAX_DEPTH                       = 0x1015
    CL_DEVICE_IMAGE_SUPPORT                           = 0x1016
    CL_DEVICE_MAX_PARAMETER_SIZE                      = 0x1017
    CL_DEVICE_MAX_SAMPLERS                            = 0x1018
    CL_DEVICE_MEM_BASE_ADDR_ALIGN                     = 0x1019
    CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE                = 0x101A
    CL_DEVICE_SINGLE_FP_CONFIG                        = 0x101B
    CL_DEVICE_GLOBAL_MEM_CACHE_TYPE                   = 0x101C
    CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE               = 0x101D
    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE                   = 0x101E
    CL_DEVICE_GLOBAL_MEM_SIZE                         = 0x101F
    CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE                = 0x1020
    CL_DEVICE_MAX_CONSTANT_ARGS                       = 0x1021
    CL_DEVICE_LOCAL_MEM_TYPE                          = 0x1022
    CL_DEVICE_LOCAL_MEM_SIZE                          = 0x1023
    CL_DEVICE_ERROR_CORRECTION_SUPPORT                = 0x1024
    CL_DEVICE_PROFILING_TIMER_RESOLUTION              = 0x1025
    CL_DEVICE_ENDIAN_LITTLE                           = 0x1026
    CL_DEVICE_AVAILABLE                               = 0x1027
    CL_DEVICE_COMPILER_AVAILABLE                      = 0x1028
    CL_DEVICE_EXECUTION_CAPABILITIES                  = 0x1029
    CL_DEVICE_QUEUE_PROPERTIES                        = 0x102A
    CL_DEVICE_QUEUE_ON_HOST_PROPERTIES                = 0x102A
    CL_DEVICE_NAME                                    = 0x102B
    CL_DEVICE_VENDOR                                  = 0x102C
    CL_DRIVER_VERSION                                 = 0x102D
    CL_DEVICE_PROFILE                                 = 0x102E
    CL_DEVICE_VERSION                                 = 0x102F
    CL_DEVICE_EXTENSIONS                              = 0x1030
    CL_DEVICE_PLATFORM                                = 0x1031
    CL_DEVICE_DOUBLE_FP_CONFIG                        = 0x1032
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF             = 0x1034
    CL_DEVICE_HOST_UNIFIED_MEMORY                     = 0x1035
    CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR                = 0x1036
    CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT               = 0x1037
    CL_DEVICE_NATIVE_VECTOR_WIDTH_INT                 = 0x1038
    CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG                = 0x1039
    CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT               = 0x103A
    CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE              = 0x103B
    CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF                = 0x103C
    CL_DEVICE_OPENCL_C_VERSION                        = 0x103D
    CL_DEVICE_LINKER_AVAILABLE                        = 0x103E
    CL_DEVICE_BUILT_IN_KERNELS                        = 0x103F
    CL_DEVICE_IMAGE_MAX_BUFFER_SIZE                   = 0x1040
    CL_DEVICE_IMAGE_MAX_ARRAY_SIZE                    = 0x1041
    CL_DEVICE_PARENT_DEVICE                           = 0x1042
    CL_DEVICE_PARTITION_MAX_SUB_DEVICES               = 0x1043
    CL_DEVICE_PARTITION_PROPERTIES                    = 0x1044
    CL_DEVICE_PARTITION_AFFINITY_DOMAIN               = 0x1045
    CL_DEVICE_PARTITION_TYPE                          = 0x1046
    CL_DEVICE_REFERENCE_COUNT                         = 0x1047
    CL_DEVICE_PREFERRED_INTEROP_USER_SYNC             = 0x1048
    CL_DEVICE_PRINTF_BUFFER_SIZE                      = 0x1049
    CL_DEVICE_IMAGE_PITCH_ALIGNMENT                   = 0x104A
    CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT            = 0x104B
    CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS               = 0x104C
    CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE                = 0x104D
    CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES              = 0x104E
    CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE          = 0x104F
    CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE                = 0x1050
    CL_DEVICE_MAX_ON_DEVICE_QUEUES                    = 0x1051
    CL_DEVICE_MAX_ON_DEVICE_EVENTS                    = 0x1052
    CL_DEVICE_SVM_CAPABILITIES                        = 0x1053
    CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE    = 0x1054
    CL_DEVICE_MAX_PIPE_ARGS                           = 0x1055
    CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS            = 0x1056
    CL_DEVICE_PIPE_MAX_PACKET_SIZE                    = 0x1057
    CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT     = 0x1058
    CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT       = 0x1059
    CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT        = 0x105A
    CL_DEVICE_IL_VERSION                              = 0x105B
    CL_DEVICE_MAX_NUM_SUB_GROUPS                      = 0x105C
    CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS  = 0x105D
    CL_DEVICE_NUMERIC_VERSION                         = 0x105E
    CL_DEVICE_EXTENSIONS_WITH_VERSION                 = 0x1060
    CL_DEVICE_ILS_WITH_VERSION                        = 0x1061
    CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION           = 0x1062
    CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES              = 0x1063
    CL_DEVICE_ATOMIC_FENCE_CAPABILITIES               = 0x1064
    CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT          = 0x1065
    CL_DEVICE_OPENCL_C_ALL_VERSIONS                   = 0x1066
    CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE      = 0x1067
    CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT = 0x1068
    CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT           = 0x1069
    CL_DEVICE_OPENCL_C_FEATURES                       = 0x106F
    CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES             = 0x1070
    CL_DEVICE_PIPE_SUPPORT                            = 0x1071
    CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED       = 0x1072
    CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR            = 0x12A9
    CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR  = 0x12AA
    CL_DEVICE_COMMAND_BUFFER_NUM_SYNC_DEVICES_KHR        = 0x12AB
    CL_DEVICE_COMMAND_BUFFER_SYNC_DEVICES_KHR            = 0x12AC
    CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR          = 0x12B0
    CL_DEVICE_HALF_FP_CONFIG                             = 0x1033
    CL_DEVICE_IL_VERSION_KHR                             = 0x105B
    CL_DEVICE_IMAGE_PITCH_ALIGNMENT_KHR                  = 0x104A
    CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT_KHR           = 0x104B
    CL_DEVICE_TERMINATE_CAPABILITY_KHR                   = 0x2031
    CL_DEVICE_SPIR_VERSIONS                              = 0x40E0
    CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV                = 0x4000
    CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV                = 0x4001
    CL_DEVICE_REGISTERS_PER_BLOCK_NV                     = 0x4002
    CL_DEVICE_WARP_SIZE_NV                               = 0x4003
    CL_DEVICE_GPU_OVERLAP_NV                             = 0x4004
    CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV                     = 0x4005
    CL_DEVICE_INTEGRATED_MEMORY_NV                       = 0x4006
    CL_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT_NV            = 0x4007
    CL_DEVICE_PCI_BUS_ID_NV                              = 0x4008
    CL_DEVICE_PCI_SLOT_ID_NV                             = 0x4009
    CL_DEVICE_PCI_DOMAIN_ID_NV                           = 0x400A
    CL_DEVICE_MAX_LOCAL_MEMORY_PER_SM_NV                 = 0x400B
    CL_DEVICE_PROFILING_TIMER_OFFSET_AMD                 = 0x4036
    CL_DEVICE_TOPOLOGY_AMD                               = 0x4037
    CL_DEVICE_BOARD_NAME_AMD                             = 0x4038
    CL_DEVICE_GLOBAL_FREE_MEMORY_AMD                     = 0x4039
    CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD                  = 0x4040
    CL_DEVICE_SIMD_WIDTH_AMD                             = 0x4041
    CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD                 = 0x4042
    CL_DEVICE_WAVEFRONT_WIDTH_AMD                        = 0x4043
    CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD                    = 0x4044
    CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD               = 0x4045
    CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD          = 0x4046
    CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD        = 0x4047
    CL_DEVICE_LOCAL_MEM_BANKS_AMD                        = 0x4048
    CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD                 = 0x4049
    CL_DEVICE_GFXIP_MAJOR_AMD                            = 0x404A
    CL_DEVICE_GFXIP_MINOR_AMD                            = 0x404B
    CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD                 = 0x404C
    CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD              = 0x4030
    CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD                    = 0x4031
    CL_DEVICE_PREFERRED_CONSTANT_BUFFER_SIZE_AMD         = 0x4033
    CL_DEVICE_PCIE_ID_AMD                                = 0x4034
    CL_DEVICE_PARENT_DEVICE_EXT                          = 0x4054
    CL_DEVICE_PARTITION_TYPES_EXT                        = 0x4055
    CL_DEVICE_AFFINITY_DOMAINS_EXT                       = 0x4056
    CL_DEVICE_REFERENCE_COUNT_EXT                        = 0x4057
    CL_DEVICE_PARTITION_STYLE_EXT                        = 0x4058
    CL_DEVICE_CXX_FOR_OPENCL_NUMERIC_VERSION_EXT         = 0x4230
    CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM              = 0x40A0
    CL_DEVICE_PAGE_SIZE_QCOM                             = 0x40A1
    CL_DEVICE_MAX_NAMED_BARRIER_COUNT_KHR                = 0x2035
    CL_DEVICE_NUMERIC_VERSION_KHR                        = 0x105E
    CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR               = 0x105F
    CL_DEVICE_EXTENSIONS_WITH_VERSION_KHR                = 0x1060
    CL_DEVICE_ILS_WITH_VERSION_KHR                       = 0x1061
    CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION_KHR          = 0x1062
    CL_DEVICE_UUID_KHR                                   = 0x106A
    CL_DRIVER_UUID_KHR                                   = 0x106B
    CL_DEVICE_LUID_VALID_KHR                             = 0x106C
    CL_DEVICE_LUID_KHR                                   = 0x106D
    CL_DEVICE_NODE_MASK_KHR                              = 0x106E
    CL_DEVICE_PCI_BUS_INFO_KHR                           = 0x410F
    CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR       = 0x1073
    CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR  = 0x1074
    CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR  = 0x1075
    CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR    = 0x204F
    CL_DEVICE_EXTERNAL_MEMORY_IMPORT_ASSUME_LINEAR_IMAGES_HANDLE_TYPES_KHR  = 0x2052
    CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR          = 0x204D
    CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR          = 0x204E
    CL_DEVICE_SEMAPHORE_TYPES_KHR                        = 0x204C
    CL_DEVICE_SVM_CAPABILITIES_ARM                       = 0x40B6
    CL_DEVICE_COMPUTE_UNITS_BITFIELD_ARM                 = 0x40BF
    CL_DEVICE_JOB_SLOTS_ARM                              = 0x41E0
    CL_DEVICE_SCHEDULING_CONTROLS_CAPABILITIES_ARM       = 0x41E4
    CL_DEVICE_SUPPORTED_REGISTER_ALLOCATIONS_ARM         = 0x41EB
    CL_DEVICE_MAX_WARP_COUNT_ARM                         = 0x41EA
    CL_DEVICE_CONTROLLED_TERMINATION_CAPABILITIES_ARM    = 0x41EE
    CL_DEVICE_IP_VERSION_INTEL                           = 0x4250
    CL_DEVICE_ID_INTEL                                   = 0x4251
    CL_DEVICE_NUM_SLICES_INTEL                           = 0x4252
    CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL             = 0x4253
    CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL                = 0x4254
    CL_DEVICE_NUM_THREADS_PER_EU_INTEL                   = 0x4255
    CL_DEVICE_FEATURE_CAPABILITIES_INTEL                 = 0x4256
    CL_DEVICE_ME_VERSION_INTEL                           = 0x407E
    CL_DEVICE_SIMULTANEOUS_INTEROPS_INTEL                = 0x4104
    CL_DEVICE_NUM_SIMULTANEOUS_INTEROPS_INTEL            = 0x4105
    CL_DEVICE_SUB_GROUP_SIZES_INTEL                      = 0x4108
    CL_DEVICE_PLANAR_YUV_MAX_WIDTH_INTEL                 = 0x417E
    CL_DEVICE_PLANAR_YUV_MAX_HEIGHT_INTEL                = 0x417F
    CL_DEVICE_AVC_ME_VERSION_INTEL                       = 0x410B
    CL_DEVICE_AVC_ME_SUPPORTS_TEXTURE_SAMPLER_USE_INTEL  = 0x410C
    CL_DEVICE_AVC_ME_SUPPORTS_PREEMPTION_INTEL           = 0x410D
    CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL                = 0x4190
    CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL              = 0x4191
    CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL  = 0x4192
    CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL  = 0x4193
    CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL       = 0x4194
    CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL              = 0x418B
    CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT          = 0x4231
    CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT          = 0x4232
    CL_DEVICE_HALF_FP_ATOMIC_CAPABILITIES_EXT            = 0x4233

class cl_device_mem_cache_type(IntEnum):
    CL_NONE                                    = 0x0
    CL_READ_ONLY_CACHE                         = 0x1
    CL_READ_WRITE_CACHE                        = 0x2

class cl_device_fp_config(IntFlag):
    CL_FP_DENORM                               = (1 << 0)
    CL_FP_INF_NAN                              = (1 << 1)
    CL_FP_ROUND_TO_NEAREST                     = (1 << 2)
    CL_FP_ROUND_TO_ZERO                        = (1 << 3)
    CL_FP_ROUND_TO_INF                         = (1 << 4)
    CL_FP_FMA                                  = (1 << 5)
    CL_FP_SOFT_FLOAT                           = (1 << 6)
    CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT        = (1 << 7)

class cl_platform_command_buffer_capabilities_khr(IntFlag):
    CL_COMMAND_BUFFER_PLATFORM_UNIVERSAL_SYNC_KHR        = (1 << 0)
    CL_COMMAND_BUFFER_PLATFORM_REMAP_QUEUES_KHR          = (1 << 1)
    CL_COMMAND_BUFFER_PLATFORM_AUTOMATIC_REMAP_KHR       = (1 << 2)

class cl_device_local_mem_type(IntEnum):
    CL_LOCAL  = 0x1
    CL_GLOBAL = 0x2

class cl_device_exec_capabilities(IntFlag):
    CL_EXEC_KERNEL                             = (1 << 0)
    CL_EXEC_NATIVE_KERNEL                      = (1 << 1)

class cl_command_queue_properties(IntFlag):
    CL_QUEUE_NO_SYNC_OPERATIONS_INTEL          = (1 << 29)
    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE     = (1 << 0)
    CL_QUEUE_PROFILING_ENABLE                  = (1 << 1)
    CL_QUEUE_ON_DEVICE                         = (1 << 2)
    CL_QUEUE_ON_DEVICE_DEFAULT                 = (1 << 3)

class cl_device_affinity_domain(IntFlag):
    CL_DEVICE_AFFINITY_DOMAIN_NUMA               = (1 << 0)
    CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE           = (1 << 1)
    CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE           = (1 << 2)
    CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE           = (1 << 3)
    CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE           = (1 << 4)
    CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE = (1 << 5)

class cl_device_svm_capabilities(IntFlag):
    CL_DEVICE_SVM_COARSE_GRAIN_BUFFER           = (1 << 0)
    CL_DEVICE_SVM_FINE_GRAIN_BUFFER             = (1 << 1)
    CL_DEVICE_SVM_FINE_GRAIN_SYSTEM             = (1 << 2)
    CL_DEVICE_SVM_ATOMICS                       = (1 << 3)

class cl_device_atomic_capabilities(IntFlag):
    CL_DEVICE_ATOMIC_ORDER_RELAXED          = (1 << 0)
    CL_DEVICE_ATOMIC_ORDER_ACQ_REL          = (1 << 1)
    CL_DEVICE_ATOMIC_ORDER_SEQ_CST          = (1 << 2)
    CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM        = (1 << 3)
    CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP       = (1 << 4)
    CL_DEVICE_ATOMIC_SCOPE_DEVICE           = (1 << 5)
    CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES      = (1 << 6)

class cl_device_device_enqueue_capabilities(IntFlag):
    CL_DEVICE_QUEUE_SUPPORTED               = (1 << 0)
    CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT     = (1 << 1)

class cl_device_command_buffer_capabilities_khr(IntFlag):
    CL_COMMAND_BUFFER_CAPABILITY_KERNEL_PRINTF_KHR       = (1 << 0)
    CL_COMMAND_BUFFER_CAPABILITY_DEVICE_SIDE_ENQUEUE_KHR = (1 << 1)
    CL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR    = (1 << 2)
    CL_COMMAND_BUFFER_CAPABILITY_OUT_OF_ORDER_KHR        = (1 << 3)
    CL_COMMAND_BUFFER_CAPABILITY_MULTIPLE_QUEUE_KHR      = (1 << 4)

class cl_mutable_dispatch_fields_khr(IntFlag):
    CL_MUTABLE_DISPATCH_GLOBAL_OFFSET_KHR               = (1 << 0)
    CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR                 = (1 << 1)
    CL_MUTABLE_DISPATCH_LOCAL_SIZE_KHR                  = (1 << 2)
    CL_MUTABLE_DISPATCH_ARGUMENTS_KHR                   = (1 << 3)
    CL_MUTABLE_DISPATCH_EXEC_INFO_KHR                   = (1 << 4)

class cl_external_memory_handle_type_khr(IntEnum):
    CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR               = 0x2067
    CL_EXTERNAL_MEMORY_HANDLE_D3D11_TEXTURE_KHR         = 0x2063
    CL_EXTERNAL_MEMORY_HANDLE_D3D11_TEXTURE_KMT_KHR     = 0x2064
    CL_EXTERNAL_MEMORY_HANDLE_D3D12_HEAP_KHR            = 0x2065
    CL_EXTERNAL_MEMORY_HANDLE_D3D12_RESOURCE_KHR        = 0x2066
    CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR             = 0x2060
    CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR          = 0x2061
    CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR      = 0x2062
    CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_NAME_KHR     = 0x2069

class cl_device_integer_dot_product_capabilities_khr(IntFlag):
    CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR = (1 << 0)
    CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR        = (1 << 1)

class cl_external_semaphore_handle_type_khr(IntEnum):
    CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR                = 0x2059
    CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR                  = 0x2055
    CL_SEMAPHORE_HANDLE_SYNC_FD_KHR                    = 0x2058
    CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR               = 0x2056
    CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR           = 0x2057
    CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_NAME_KHR          = 0x2068

class cl_device_terminate_capability_khr(IntFlag):
    CL_DEVICE_TERMINATE_CAPABILITY_CONTEXT_KHR         = (1 << 0)

class cl_bool(IntEnum):
    CL_FALSE        = 0
    CL_TRUE         = 1
    CL_BLOCKING     = 1
    CL_NON_BLOCKING = 0

class cl_device_partition_property(IntEnum):
    CL_DEVICE_PARTITION_EQUALLY                = 0x1086
    CL_DEVICE_PARTITION_BY_COUNTS              = 0x1087
    CL_DEVICE_PARTITION_BY_COUNTS_LIST_END     = 0x0
    CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN     = 0x1088

    @classmethod
    def dtype(cls):
        return c_int64
    
class cl_context_info(IntEnum):
    CL_CONTEXT_REFERENCE_COUNT                 = 0x1080
    CL_CONTEXT_DEVICES                         = 0x1081
    CL_CONTEXT_PROPERTIES                      = 0x1082
    CL_CONTEXT_NUM_DEVICES                     = 0x1083
    CL_CONTEXT_D3D9_DEVICE_NV                  = 0x4026
    CL_CONTEXT_D3D10_DEVICE_NV                 = 0x4014
    CL_CONTEXT_D3D10_DEVICE_KHR                        = 0x4014
    CL_CONTEXT_D3D10_PREFER_SHARED_RESOURCES_KHR       = 0x402C
    CL_CONTEXT_D3D11_DEVICE_NV                         = 0x401D
    CL_CONTEXT_D3D11_DEVICE_KHR                        = 0x401D
    CL_CONTEXT_D3D11_PREFER_SHARED_RESOURCES_KHR       = 0x402D
    CL_CONTEXT_ADAPTER_D3D9_KHR                        = 0x2025
    CL_CONTEXT_ADAPTER_D3D9EX_KHR                      = 0x2026
    CL_CONTEXT_ADAPTER_DXVA_KHR                        = 0x2027
    CL_CONTEXT_D3D9_DEVICE_INTEL                       = 0x4026
    CL_CONTEXT_D3D9EX_DEVICE_INTEL                     = 0x4072
    CL_CONTEXT_DXVA_DEVICE_INTEL                       = 0x4073

class cl_context_properties(IntEnum):
    CL_CONTEXT_PLATFORM                                = 0x1084
    CL_CONTEXT_INTEROP_USER_SYNC                       = 0x1085
    CL_CONTEXT_TERMINATE_KHR                           = 0x2032
    CL_PRINTF_CALLBACK_ARM                             = 0x40B0
    CL_PRINTF_BUFFERSIZE_ARM                           = 0x40B1
    CL_CONTEXT_SHOW_DIAGNOSTICS_INTEL                  = 0x4106
    CL_CONTEXT_DIAGNOSTICS_LEVEL_ALL_INTEL             = 0xff
    CL_CONTEXT_DIAGNOSTICS_LEVEL_GOOD_INTEL            = (1 << 0)
    CL_CONTEXT_DIAGNOSTICS_LEVEL_BAD_INTEL             = (1 << 1)
    CL_CONTEXT_DIAGNOSTICS_LEVEL_NEUTRAL_INTEL         = (1 << 2)
    CL_GL_CONTEXT_KHR                                  = 0x2008
    CL_EGL_DISPLAY_KHR                                 = 0x2009
    CL_GLX_DISPLAY_KHR                                 = 0x200A
    CL_WGL_HDC_KHR                                     = 0x200B
    CL_CGL_SHAREGROUP_KHR                              = 0x200C

    @classmethod
    def dtype(cls):
        return c_int64
    
class cl_program_info(IntEnum):
    CL_PROGRAM_REFERENCE_COUNT                 = 0x1160
    CL_PROGRAM_CONTEXT                         = 0x1161
    CL_PROGRAM_NUM_DEVICES                     = 0x1162
    CL_PROGRAM_DEVICES                         = 0x1163
    CL_PROGRAM_SOURCE                          = 0x1164
    CL_PROGRAM_BINARY_SIZES                    = 0x1165
    CL_PROGRAM_BINARIES                        = 0x1166
    CL_PROGRAM_NUM_KERNELS                     = 0x1167
    CL_PROGRAM_KERNEL_NAMES                    = 0x1168
    CL_PROGRAM_IL                              = 0x1169
    CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT      = 0x116A
    CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT      = 0x116B
    CL_PROGRAM_IL_KHR                          = 0x1169

class cl_program_build_info(IntEnum):
    CL_PROGRAM_BUILD_STATUS                     = 0x1181
    CL_PROGRAM_BUILD_OPTIONS                    = 0x1182
    CL_PROGRAM_BUILD_LOG                        = 0x1183
    CL_PROGRAM_BINARY_TYPE                      = 0x1184
    CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE = 0x1185

class cl_build_status(IntEnum):
    CL_BUILD_SUCCESS                            = 0
    CL_BUILD_NONE                               = -1
    CL_BUILD_ERROR                              = -2
    CL_BUILD_IN_PROGRESS                        = -3

    @classmethod
    def dtype(cls):
        return cl_int
    
class cl_program_binary_type(IntEnum):
    CL_PROGRAM_BINARY_TYPE_NONE                 = 0x0
    CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT      = 0x1
    CL_PROGRAM_BINARY_TYPE_LIBRARY              = 0x2
    CL_PROGRAM_BINARY_TYPE_EXECUTABLE           = 0x4
    CL_PROGRAM_BINARY_TYPE_INTERMEDIATE         = 0x40E1

class cl_kernel_arg_address_qualifier(IntEnum):
    CL_KERNEL_ARG_ADDRESS_GLOBAL                = 0x119B
    CL_KERNEL_ARG_ADDRESS_LOCAL                 = 0x119C
    CL_KERNEL_ARG_ADDRESS_CONSTANT              = 0x119D
    CL_KERNEL_ARG_ADDRESS_PRIVATE               = 0x119E

class cl_kernel_arg_access_qualifier(IntEnum):
    CL_KERNEL_ARG_ACCESS_READ_ONLY              = 0x11A0
    CL_KERNEL_ARG_ACCESS_WRITE_ONLY             = 0x11A1
    CL_KERNEL_ARG_ACCESS_READ_WRITE             = 0x11A2
    CL_KERNEL_ARG_ACCESS_NONE                   = 0x11A3

class cl_kernel_arg_type_qualifier(IntFlag):
    CL_KERNEL_ARG_TYPE_NONE                     = 0
    CL_KERNEL_ARG_TYPE_CONST                    = (1 << 0)
    CL_KERNEL_ARG_TYPE_RESTRICT                 = (1 << 1)
    CL_KERNEL_ARG_TYPE_VOLATILE                 = (1 << 2)
    CL_KERNEL_ARG_TYPE_PIPE                     = (1 << 3)

class cl_kernel_info(IntEnum):
    CL_KERNEL_FUNCTION_NAME                     = 0x1190
    CL_KERNEL_NUM_ARGS                          = 0x1191
    CL_KERNEL_REFERENCE_COUNT                   = 0x1192
    CL_KERNEL_CONTEXT                           = 0x1193
    CL_KERNEL_PROGRAM                           = 0x1194
    CL_KERNEL_ATTRIBUTES                        = 0x1195
    CL_KERNEL_MAX_WARP_COUNT_ARM                = 0x41E9

class cl_queue_properties(IntEnum):
    CL_QUEUE_PRIORITY_KHR                       = 0x1096
    CL_QUEUE_THROTTLE_KHR                       = 0x1097
    CL_QUEUE_JOB_SLOT_ARM                       = 0x41E1
    CL_QUEUE_KERNEL_BATCHING_ARM                = 0x41E7
    CL_QUEUE_DEFERRED_FLUSH_ARM                 = 0x41EC
    CL_QUEUE_FAMILY_INTEL                       = 0x418C
    CL_QUEUE_INDEX_INTEL                        = 0x418D

    @classmethod
    def dtype(cls):
        return cl_ulong
    
class cl_command_queue_info(IntEnum):
    CL_QUEUE_CONTEXT                           = 0x1090
    CL_QUEUE_DEVICE                            = 0x1091
    CL_QUEUE_REFERENCE_COUNT                   = 0x1092
    CL_QUEUE_PROPERTIES                        = 0x1093
    CL_QUEUE_SIZE                              = 0x1094
    CL_QUEUE_DEVICE_DEFAULT                    = 0x1095
    CL_QUEUE_PROPERTIES_ARRAY                  = 0x1098

class cl_mem_flags(IntFlag):
    CL_MEM_EXT_HOST_PTR_QCOM                    = (1 << 29)
    CL_MEM_USE_UNCACHED_CPU_MEMORY_IMG          = (1 << 26)
    CL_MEM_USE_CACHED_CPU_MEMORY_IMG            = (1 << 27)
    CL_MEM_USE_GRALLOC_PTR_IMG                  = (1 << 28)
    CL_MEM_NO_ACCESS_INTEL                      = (1 << 24)
    CL_MEM_ACCESS_FLAGS_UNRESTRICTED_INTEL      = (1 << 25)
    CL_MEM_FORCE_HOST_MEMORY_INTEL              = (1 << 20)
    CL_MEM_READ_WRITE                           = (1 << 0)
    CL_MEM_WRITE_ONLY                           = (1 << 1)
    CL_MEM_READ_ONLY                            = (1 << 2)
    CL_MEM_USE_HOST_PTR                         = (1 << 3)
    CL_MEM_ALLOC_HOST_PTR                       = (1 << 4)
    CL_MEM_COPY_HOST_PTR                        = (1 << 5)
    CL_MEM_HOST_WRITE_ONLY                      = (1 << 7)
    CL_MEM_HOST_READ_ONLY                       = (1 << 8)
    CL_MEM_HOST_NO_ACCESS                       = (1 << 9)
    CL_MEM_KERNEL_READ_AND_WRITE                = (1 << 12)

class cl_mem_properties(IntEnum):
    CL_MEM_ALLOC_FLAGS_IMG                             = 0x40D7
    CL_MEM_DEVICE_HANDLE_LIST_KHR                      = 0x2051
    CL_MEM_DEVICE_HANDLE_LIST_END_KHR                  = 0
    CL_MEM_LOCALLY_UNCACHED_RESOURCE_INTEL             = 0x4218
    CL_MEM_DEVICE_ID_INTEL                             = 0x4219

    @classmethod
    def dtype(cls):
        return cl_ulong
    
class cl_mem_info(IntEnum):
    CL_MEM_TYPE                                = 0x1100
    CL_MEM_FLAGS                               = 0x1101
    CL_MEM_SIZE                                = 0x1102
    CL_MEM_HOST_PTR                            = 0x1103
    CL_MEM_MAP_COUNT                           = 0x1104
    CL_MEM_REFERENCE_COUNT                     = 0x1105
    CL_MEM_CONTEXT                             = 0x1106
    CL_MEM_ASSOCIATED_MEMOBJECT                = 0x1107
    CL_MEM_OFFSET                              = 0x1108
    CL_MEM_USES_SVM_POINTER                    = 0x1109
    CL_MEM_PROPERTIES                          = 0x110A
    CL_MEM_D3D9_RESOURCE_NV                    = 0x4027
    CL_MEM_D3D10_RESOURCE_NV                   = 0x4015
    CL_MEM_D3D10_RESOURCE_KHR                  = 0x4015
    CL_MEM_D3D11_RESOURCE_NV                   = 0x401E
    CL_MEM_D3D11_RESOURCE_KHR                  = 0x401E
    CL_MEM_DX9_MEDIA_ADAPTER_TYPE_KHR          = 0x2028
    CL_MEM_DX9_MEDIA_SURFACE_INFO_KHR          = 0x2029
    CL_MEM_DX9_RESOURCE_INTEL                  = 0x4027
    CL_MEM_DX9_SHARED_HANDLE_INTEL             = 0x4074
    CL_MEM_USES_SVM_POINTER_ARM                = 0x40B7

class cl_mem_object_type(IntEnum):
    CL_MEM_OBJECT_BUFFER                       = 0x10F0
    CL_MEM_OBJECT_IMAGE2D                      = 0x10F1
    CL_MEM_OBJECT_IMAGE3D                      = 0x10F2
    CL_MEM_OBJECT_IMAGE2D_ARRAY                = 0x10F3
    CL_MEM_OBJECT_IMAGE1D                      = 0x10F4
    CL_MEM_OBJECT_IMAGE1D_ARRAY                = 0x10F5
    CL_MEM_OBJECT_IMAGE1D_BUFFER               = 0x10F6
    CL_MEM_OBJECT_PIPE                         = 0x10F7

class cl_event_info(IntEnum):
    CL_EVENT_COMMAND_QUEUE                     = 0x11D0
    CL_EVENT_COMMAND_TYPE                      = 0x11D1
    CL_EVENT_REFERENCE_COUNT                   = 0x11D2
    CL_EVENT_COMMAND_EXECUTION_STATUS          = 0x11D3
    CL_EVENT_CONTEXT                           = 0x11D4
    CL_EVENT_COMMAND_TERMINATION_REASON_ARM    = 0x41ED

class cl_command_execution_status(IntEnum):
    CL_COMPLETE                                = 0x0
    CL_RUNNING                                 = 0x1
    CL_SUBMITTED                               = 0x2
    CL_QUEUED                                  = 0x3

    @classmethod
    def dtype(cls):
        return cl_int
    
class cl_command_type(IntEnum):
    CL_COMMAND_NDRANGE_KERNEL                  = 0x11F0
    CL_COMMAND_TASK                            = 0x11F1
    CL_COMMAND_NATIVE_KERNEL                   = 0x11F2
    CL_COMMAND_READ_BUFFER                     = 0x11F3
    CL_COMMAND_WRITE_BUFFER                    = 0x11F4
    CL_COMMAND_COPY_BUFFER                     = 0x11F5
    CL_COMMAND_READ_IMAGE                      = 0x11F6
    CL_COMMAND_WRITE_IMAGE                     = 0x11F7
    CL_COMMAND_COPY_IMAGE                      = 0x11F8
    CL_COMMAND_COPY_IMAGE_TO_BUFFER            = 0x11F9
    CL_COMMAND_COPY_BUFFER_TO_IMAGE            = 0x11FA
    CL_COMMAND_MAP_BUFFER                      = 0x11FB
    CL_COMMAND_MAP_IMAGE                       = 0x11FC
    CL_COMMAND_UNMAP_MEM_OBJECT                = 0x11FD
    CL_COMMAND_MARKER                          = 0x11FE
    CL_COMMAND_ACQUIRE_GL_OBJECTS              = 0x11FF
    CL_COMMAND_RELEASE_GL_OBJECTS              = 0x1200
    CL_COMMAND_READ_BUFFER_RECT                = 0x1201
    CL_COMMAND_WRITE_BUFFER_RECT               = 0x1202
    CL_COMMAND_COPY_BUFFER_RECT                = 0x1203
    CL_COMMAND_USER                            = 0x1204
    CL_COMMAND_BARRIER                         = 0x1205
    CL_COMMAND_MIGRATE_MEM_OBJECTS             = 0x1206
    CL_COMMAND_FILL_BUFFER                     = 0x1207
    CL_COMMAND_FILL_IMAGE                      = 0x1208
    CL_COMMAND_SVM_FREE                        = 0x1209
    CL_COMMAND_SVM_MEMCPY                      = 0x120A
    CL_COMMAND_SVM_MEMFILL                     = 0x120B
    CL_COMMAND_SVM_MAP                         = 0x120C
    CL_COMMAND_SVM_UNMAP                       = 0x120D
    CL_COMMAND_SVM_MIGRATE_MEM                 = 0x120E
    CL_COMMAND_GL_FENCE_SYNC_OBJECT_KHR                = 0x200D
    CL_COMMAND_COMMAND_BUFFER_KHR                      = 0x12A8
    CL_COMMAND_MIGRATE_MEM_OBJECT_EXT                  = 0x4040
    CL_COMMAND_ACQUIRE_GRALLOC_OBJECTS_IMG             = 0x40D2
    CL_COMMAND_RELEASE_GRALLOC_OBJECTS_IMG             = 0x40D3
    CL_COMMAND_GENERATE_MIPMAP_IMG                     = 0x40D6
    CL_COMMAND_ACQUIRE_EXTERNAL_MEM_OBJECTS_KHR        = 0x2047
    CL_COMMAND_RELEASE_EXTERNAL_MEM_OBJECTS_KHR        = 0x2048
    CL_COMMAND_SEMAPHORE_WAIT_KHR                      = 0x2042
    CL_COMMAND_SEMAPHORE_SIGNAL_KHR                    = 0x2043
    CL_COMMAND_SVM_FREE_ARM                            = 0x40BA
    CL_COMMAND_SVM_MEMCPY_ARM                          = 0x40BB
    CL_COMMAND_SVM_MEMFILL_ARM                         = 0x40BC
    CL_COMMAND_SVM_MAP_ARM                             = 0x40BD
    CL_COMMAND_SVM_UNMAP_ARM                           = 0x40BE
    CL_COMMAND_MEMFILL_INTEL                           = 0x4204
    CL_COMMAND_MEMCPY_INTEL                            = 0x4205
    CL_COMMAND_MIGRATEMEM_INTEL                        = 0x4206
    CL_COMMAND_MEMADVISE_INTEL                         = 0x4207
    CL_COMMAND_ACQUIRE_DX9_MEDIA_SURFACES_KHR          = 0x202B
    CL_COMMAND_RELEASE_DX9_MEDIA_SURFACES_KHR          = 0x202C
    CL_COMMAND_ACQUIRE_DX9_OBJECTS_INTEL               = 0x402A
    CL_COMMAND_RELEASE_DX9_OBJECTS_INTEL               = 0x402B
    CL_COMMAND_ACQUIRE_D3D11_OBJECTS_KHR               = 0x4020
    CL_COMMAND_RELEASE_D3D11_OBJECTS_KHR               = 0x4021
    CL_COMMAND_ACQUIRE_D3D11_OBJECTS_NV                = 0x4020
    CL_COMMAND_RELEASE_D3D11_OBJECTS_NV                = 0x4021
    CL_COMMAND_ACQUIRE_D3D10_OBJECTS_KHR               = 0x4017
    CL_COMMAND_RELEASE_D3D10_OBJECTS_KHR               = 0x4018
    CL_COMMAND_ACQUIRE_D3D10_OBJECTS_NV                = 0x4017
    CL_COMMAND_RELEASE_D3D10_OBJECTS_NV                = 0x4018
    CL_COMMAND_ACQUIRE_D3D9_OBJECTS_NV                 = 0x402A
    CL_COMMAND_RELEASE_D3D9_OBJECTS_NV                 = 0x402B

class cl_channel_order(IntEnum):
    CL_R                                       = 0x10B0
    CL_A                                       = 0x10B1
    CL_RG                                      = 0x10B2
    CL_RA                                      = 0x10B3
    CL_RGB                                     = 0x10B4
    CL_RGBA                                    = 0x10B5
    CL_BGRA                                    = 0x10B6
    CL_ARGB                                    = 0x10B7
    CL_INTENSITY                               = 0x10B8
    CL_LUMINANCE                               = 0x10B9
    CL_Rx                                      = 0x10BA
    CL_RGx                                     = 0x10BB
    CL_RGBx                                    = 0x10BC
    CL_DEPTH                                   = 0x10BD
    CL_sRGB                                    = 0x10BF
    CL_sRGBx                                   = 0x10C0
    CL_sRGBA                                   = 0x10C1
    CL_sBGRA                                   = 0x10C2
    CL_ABGR                                    = 0x10C3
    CL_DEPTH_STENCIL                           = 0x10BE
    CL_NV21_IMG                                = 0x40D0
    CL_YV12_IMG                                = 0x40D1
    CL_YUYV_INTEL                              = 0x4076
    CL_UYVY_INTEL                              = 0x4077
    CL_YVYU_INTEL                              = 0x4078
    CL_VYUY_INTEL                              = 0x4079
    CL_NV12_INTEL                              = 0x410E

class cl_channel_type(IntEnum):
    CL_SNORM_INT8                              = 0x10D0
    CL_SNORM_INT16                             = 0x10D1
    CL_UNORM_INT8                              = 0x10D2
    CL_UNORM_INT16                             = 0x10D3
    CL_UNORM_SHORT_565                         = 0x10D4
    CL_UNORM_SHORT_555                         = 0x10D5
    CL_UNORM_INT_101010                        = 0x10D6
    CL_SIGNED_INT8                             = 0x10D7
    CL_SIGNED_INT16                            = 0x10D8
    CL_SIGNED_INT32                            = 0x10D9
    CL_UNSIGNED_INT8                           = 0x10DA
    CL_UNSIGNED_INT16                          = 0x10DB
    CL_UNSIGNED_INT32                          = 0x10DC
    CL_HALF_FLOAT                              = 0x10DD
    CL_FLOAT                                   = 0x10DE
    CL_UNORM_INT_101010_2                      = 0x10E0
    CL_UNORM_INT24                             = 0x10DF
    CL_UNSIGNED_INT_RAW10_EXT                  = 0x10E3
    CL_UNSIGNED_INT_RAW12_EXT                  = 0x10E4

class cl_addressing_mode(IntEnum):
    CL_ADDRESS_NONE                           = 0x1130
    CL_ADDRESS_CLAMP_TO_EDGE                  = 0x1131
    CL_ADDRESS_CLAMP                          = 0x1132
    CL_ADDRESS_REPEAT                         = 0x1133
    CL_ADDRESS_MIRRORED_REPEAT                = 0x1134

class cl_filter_mode(IntEnum):
    CL_FILTER_NEAREST                         = 0x1140
    CL_FILTER_LINEAR                          = 0x1141

class cl_sampler_info(IntEnum):
    CL_SAMPLER_REFERENCE_COUNT                 = 0x1150
    CL_SAMPLER_CONTEXT                         = 0x1151
    CL_SAMPLER_NORMALIZED_COORDS               = 0x1152
    CL_SAMPLER_ADDRESSING_MODE                 = 0x1153
    CL_SAMPLER_FILTER_MODE                     = 0x1154
    CL_SAMPLER_MIP_FILTER_MODE                 = 0x1155
    CL_SAMPLER_LOD_MIN                         = 0x1156
    CL_SAMPLER_LOD_MAX                         = 0x1157
    CL_SAMPLER_PROPERTIES                      = 0x1158

class cl_sampler_properties(IntEnum):
    CL_SAMPLER_MIP_FILTER_MODE_KHR                     = 0x1155
    CL_SAMPLER_LOD_MIN_KHR                             = 0x1156
    CL_SAMPLER_LOD_MAX_KHR                             = 0x1157

    @classmethod
    def dtype(cls)->type:
        return cl_ulong

CL_VERSION_MAJOR_BITS = IntConstant("CL_VERSION_MAJOR_BITS", 10)
CL_VERSION_MINOR_BITS = IntConstant("CL_VERSION_MINOR_BITS", 10)
CL_VERSION_PATCH_BITS = IntConstant("CL_VERSION_PATCH_BITS", 12)

CL_VERSION_MAJOR_MASK = IntConstant("CL_VERSION_MAJOR_MASK", (1 << CL_VERSION_MAJOR_BITS) - 1)
CL_VERSION_MINOR_MASK = IntConstant("CL_VERSION_MINOR_MASK", (1 << CL_VERSION_MINOR_BITS) - 1)
CL_VERSION_PATCH_MASK = IntConstant("CL_VERSION_PATCH_MASK", (1 << CL_VERSION_PATCH_BITS) - 1)

class cl_version(cl_uint):

    @property
    def major(self)->int:
        return (self.value >> (CL_VERSION_MINOR_BITS + CL_VERSION_PATCH_BITS))

    @property
    def minor(self)->int:
        return ((self.value >> CL_VERSION_PATCH_BITS) & CL_VERSION_MINOR_MASK)

    @property
    def patch(self)->int:
        return (self.value & CL_VERSION_PATCH_MASK)
    
    def __str__(self)->str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __repr__(self)->str:
        return f"{self.major}.{self.minor}.{self.patch}"
        
cl_version_khr = cl_version


CL_NAME_VERSION_MAX_NAME_SIZE = IntConstant("CL_NAME_VERSION_MAX_NAME_SIZE", 64)
CL_NAME_VERSION_MAX_NAME_SIZE_KHR = IntConstant("CL_NAME_VERSION_MAX_NAME_SIZE_KHR", 64)

class cl_name_version(Structure):
    _fields_ = [
        ("version", cl_version),
        ("name", c_char * CL_NAME_VERSION_MAX_NAME_SIZE)
    ]

    def __repr__(self)->str:
        return str((self.name.decode("utf-8"), self.version))


class cl_name_version_khr(Structure):
    _fields_ = [
        ("version", cl_version_khr),
        ("name", c_char * CL_NAME_VERSION_MAX_NAME_SIZE_KHR)
    ]

    def __repr__(self)->str:
        return str((self.name.decode("utf-8"), self.version))
    
class cl_device_integer_dot_product_acceleration_properties_khr(Structure):
    _fields_ = [
        ("signed_accelerated", cl_uint),
        ("unsigned_accelerated", cl_uint),
        ("mixed_signedness_accelerated", cl_uint),
        ("accumulating_saturating_signed_accelerated", cl_uint),
        ("accumulating_saturating_unsigned_accelerated", cl_uint),
        ("accumulating_saturating_mixed_signedness_accelerated", cl_uint)
    ]
    
class cl_device_pci_bus_info_khr(Structure):
    _fields_ = [
        ("pci_domain", cl_uint),
        ("pci_bus", cl_uint),
        ("pci_device", cl_uint),
        ("pci_function", cl_uint)
    ]

class cl_image_format(Structure):
    _fields_ = [
        ("image_channel_order", cl_uint),
        ("image_channel_data_type", cl_uint)
    ]

ptr_cl_image_format:TypeAlias = POINTER(cl_image_format)

class cl_image_desc(Structure):
    _fields_ = [
        ("image_type", cl_uint),
        ("image_width", c_size_t),
        ("image_height", c_size_t),
        ("image_depth", c_size_t),
        ("image_array_size", c_size_t),
        ("image_row_pitch", c_size_t),
        ("image_slice_pitch", c_size_t),
        ("num_mip_levels", cl_uint),
        ("num_samples", cl_uint),
        ("buffer", cl_mem)
    ]

    @property
    def mem_object(self)->cl_mem:
        return self.buffer
    
    @mem_object.setter
    def mem_object(self, value:cl_mem)->None:
        self.buffer = value

ptr_cl_image_desc:TypeAlias = POINTER(cl_image_desc)


class queue_t:

    def __init__(self, properties:cl_command_queue_properties):
        self.properties = (cl_command_queue_properties.CL_QUEUE_ON_DEVICE | properties)