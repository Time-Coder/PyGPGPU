from ctypes import c_char, c_int64, c_void_p, c_int, c_uint, c_ulong, c_size_t, c_char_p, POINTER, LittleEndianStructure, CFUNCTYPE
from .clconstantes import IntConstante, IntEnum, IntFlag
from typing import TypeAlias

cl_int = c_int
cl_uint = c_uint
cl_ulong = c_ulong

cl_platform_id = c_void_p
cl_device_id = c_void_p
cl_context = c_void_p
cl_program = c_void_p
cl_bitfield = cl_ulong
cl_semaphore_type_khr = cl_uint

ptr_cl_platform_id:TypeAlias = POINTER(cl_platform_id)
ptr_cl_uint:TypeAlias = POINTER(cl_uint)
ptr_cl_int:TypeAlias = POINTER(cl_int)
ptr_size_t:TypeAlias = POINTER(c_size_t)
ptr_cl_device_id:TypeAlias = POINTER(cl_device_id)
ptr_int64:TypeAlias = POINTER(c_int64)
ptr_ptr_char:TypeAlias = POINTER(c_char_p)
CL_CONTEXT_NOTIFY_CALLBACK:TypeAlias = CFUNCTYPE(None, c_char_p, c_void_p, c_size_t, c_void_p)
CL_BULD_PROGRAM_CALLBACK:TypeAlias = CFUNCTYPE(None, cl_program, c_void_p)


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
        return c_int

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

    @property
    def dtype(self):
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

    @property
    def dtype(self):
        return cl_int
    
class cl_program_binary_type(IntEnum):
    CL_PROGRAM_BINARY_TYPE_NONE                 = 0x0
    CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT      = 0x1
    CL_PROGRAM_BINARY_TYPE_LIBRARY              = 0x2
    CL_PROGRAM_BINARY_TYPE_EXECUTABLE           = 0x4
    CL_PROGRAM_BINARY_TYPE_INTERMEDIATE         = 0x40E1

CL_VERSION_MAJOR_BITS = IntConstante("CL_VERSION_MAJOR_BITS", 10)
CL_VERSION_MINOR_BITS = IntConstante("CL_VERSION_MINOR_BITS", 10)
CL_VERSION_PATCH_BITS = IntConstante("CL_VERSION_PATCH_BITS", 12)

CL_VERSION_MAJOR_MASK = IntConstante("CL_VERSION_MAJOR_MASK", (1 << CL_VERSION_MAJOR_BITS) - 1)
CL_VERSION_MINOR_MASK = IntConstante("CL_VERSION_MINOR_MASK", (1 << CL_VERSION_MINOR_BITS) - 1)
CL_VERSION_PATCH_MASK = IntConstante("CL_VERSION_PATCH_MASK", (1 << CL_VERSION_PATCH_BITS) - 1)

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


CL_NAME_VERSION_MAX_NAME_SIZE = IntConstante("CL_NAME_VERSION_MAX_NAME_SIZE", 64)
CL_NAME_VERSION_MAX_NAME_SIZE_KHR = IntConstante("CL_NAME_VERSION_MAX_NAME_SIZE_KHR", 64)

class cl_name_version(LittleEndianStructure):
    _fields_ = [
        ("version", cl_version),
        ("name", c_char * CL_NAME_VERSION_MAX_NAME_SIZE)
    ]

    def __repr__(self)->str:
        return str((self.name.decode("utf-8"), self.version))


class cl_name_version_khr(LittleEndianStructure):
    _fields_ = [
        ("version", cl_version_khr),
        ("name", c_char * CL_NAME_VERSION_MAX_NAME_SIZE_KHR)
    ]

    def __repr__(self)->str:
        return str((self.name.decode("utf-8"), self.version))
    
class cl_device_integer_dot_product_acceleration_properties_khr(LittleEndianStructure):
    _fields_ = [
        ("signed_accelerated", cl_uint),
        ("unsigned_accelerated", cl_uint),
        ("mixed_signedness_accelerated", cl_uint),
        ("accumulating_saturating_signed_accelerated", cl_uint),
        ("accumulating_saturating_unsigned_accelerated", cl_uint),
        ("accumulating_saturating_mixed_signedness_accelerated", cl_uint)
    ]
    
class cl_device_pci_bus_info_khr(LittleEndianStructure):
    _fields_ = [
        ("pci_domain", cl_uint),
        ("pci_bus", cl_uint),
        ("pci_device", cl_uint),
        ("pci_function", cl_uint)
    ]