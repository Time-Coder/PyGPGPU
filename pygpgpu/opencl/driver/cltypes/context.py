# Context, program and kernel queries and their properties.
from ctypes import c_int64
from ....constants import IntEnum, IntFlag
from .core import cl_ulong



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
