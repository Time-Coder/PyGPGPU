import enum
from ctypes import c_char, c_void_p, c_int, c_uint, c_ulong, c_size_t, POINTER, LittleEndianStructure
from .clconstantes import CL_NAME_VERSION_MAX_NAME_SIZE, CL_NAME_VERSION_MAX_NAME_SIZE_KHR
from typing import TypeAlias

cl_int = c_int
cl_uint = c_uint
cl_ulong = c_ulong

cl_platform_id = c_void_p
cl_platform_info = cl_uint
cl_device_type = cl_ulong
cl_device_id = c_void_p
cl_context_properties = POINTER(cl_int)
cl_version_khr = cl_uint
cl_version = cl_uint
cl_bitfield = cl_ulong
cl_platform_command_buffer_capabilities_khr = cl_bitfield
cl_external_memory_handle_type_khr = cl_uint
cl_semaphore_type_khr = cl_uint
cl_external_semaphore_handle_type_khr = cl_uint

ptr_cl_platform_id:TypeAlias = POINTER(cl_platform_id)
ptr_cl_uint:TypeAlias = POINTER(cl_uint)
ptr_size_t:TypeAlias = POINTER(c_size_t)
ptr_cl_device_id:TypeAlias = POINTER(cl_device_id)


class cl_device_type(enum.IntFlag):
    CL_DEVICE_TYPE_DEFAULT = (1 << 0)
    CL_DEVICE_TYPE_CPU = (1 << 1)
    CL_DEVICE_TYPE_GPU = (1 << 2)
    CL_DEVICE_TYPE_ACCELERATOR = (1 << 3)
    CL_DEVICE_TYPE_CUSTOM = (1 << 4)
    CL_DEVICE_TYPE_ALL = 0xFFFFFFFF

class cl_platform_info(enum.IntEnum):
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

class cl_device_info(enum.IntEnum):
    CL_DEVICE_NUMERIC_VERSION_KHR = 0x105E
    CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR = 0x105F
    CL_DEVICE_EXTENSIONS_WITH_VERSION_KHR = 0x1060
    CL_DEVICE_ILS_WITH_VERSION_KHR = 0x1061
    CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION_KHR = 0x1062
    CL_DEVICE_UUID_KHR = 0x106A
    CL_DRIVER_UUID_KHR = 0x106B
    CL_DEVICE_LUID_VALID_KHR = 0x106C
    CL_DEVICE_LUID_KHR = 0x106D
    CL_DEVICE_NODE_MASK_KHR = 0x106E


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