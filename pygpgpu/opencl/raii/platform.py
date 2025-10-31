from ctypes import c_void_p, c_size_t, byref, create_string_buffer, cast, POINTER, c_char, sizeof, _SimpleCData
from typing import Any, Optional, List

from ..runtime import CL
from ..runtime.cltypes import (
    cl_platform_info,
    cl_name_version,
    cl_platform_id,
    cl_ulong,
    cl_platform_command_buffer_capabilities_khr,
    cl_version,
    cl_version_khr,
    cl_name_version_khr,
    cl_external_memory_handle_type_khr,
    cl_semaphore_type_khr,
    cl_external_semaphore_handle_type_khr
)


class Platform:

    def __init__(self, id:cl_platform_id):
        self.__id:cl_platform_id = id
        self.__profile:Optional[str] = None
        self.__version:Optional[str] = None
        self.__numeric_version:Optional[int] = None
        self.__numeric_version_khr:Optional[int] = None
        self.__name:Optional[str] = None
        self.__vendor:Optional[str] = None
        self.__extensions:Optional[str] = None
        self.__extensions_with_version:Optional[List[cl_name_version]] = None
        self.__extensions_with_version_khr:Optional[List[cl_name_version_khr]] = None
        self.__host_timer_resolution:Optional[int] = None
        self.__command_buffer_capabilities:Optional[int] = None
        self.__external_memory_import_handle_types:Optional[List[cl_external_memory_handle_type_khr]] = None
        self.__semaphore_types:Optional[List[cl_semaphore_type_khr]] = None
        self.__semaphore_import_handle_types:Optional[List[cl_external_semaphore_handle_type_khr]] = None
        self.__semaphore_export_handle_types:Optional[List[cl_external_semaphore_handle_type_khr]] = None
        self.__icd_suffix:str = None

    @property
    def id(self)->c_void_p:
        return self.__id
    
    @property
    def profile(self)->str:
        if self.__profile is None:
            self.__profile = self.__fetch_info(cl_platform_info.CL_PLATFORM_PROFILE)

        return self.__profile
    
    @property
    def version(self)->str:
        if self.__version is None:
            self.__version = self.__fetch_info(cl_platform_info.CL_PLATFORM_VERSION)

        return self.__version
    
    @property
    def numeric_version(self)->int:
        if self.__numeric_version is None:
            self.__numeric_version = self.__fetch_info(cl_platform_info.CL_PLATFORM_NUMERIC_VERSION)

        return self.__numeric_version
    
    @property
    def numeric_version_khr(self)->int:
        if self.__numeric_version_khr is None:
            self.__numeric_version_khr = self.__fetch_info(cl_platform_info.CL_PLATFORM_NUMERIC_VERSION)

        return self.__numeric_version_khr
    
    @property
    def name(self)->str:
        if self.__name is None:
            self.__name = self.__fetch_info(cl_platform_info.CL_PLATFORM_NAME)

        return self.__name
    
    @property
    def vendor(self)->str:
        if self.__vendor is None:
            self.__vendor = self.__fetch_info(cl_platform_info.CL_PLATFORM_VENDOR)

        return self.__vendor
    
    @property
    def extensions(self)->List[str]:
        if self.__extensions is None:
            self.__extensions = self.__fetch_info(cl_platform_info.CL_PLATFORM_EXTENSIONS).split(" ")

        return self.__extensions
    
    @property
    def extensions_with_version(self)->List[cl_name_version]:
        if self.__extensions_with_version is None:
            self.__extensions_with_version = self.__fetch_info(cl_platform_info.CL_PLATFORM_EXTENSIONS_WITH_VERSION)

        return self.__extensions_with_version
    
    @property
    def extensions_with_version_khr(self)->List[cl_name_version]:
        if self.__extensions_with_version_khr is None:
            self.__extensions_with_version_khr = self.__fetch_info(cl_platform_info.CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR)

        return self.__extensions_with_version_khr
    
    @property
    def host_timer_resolution(self)->int:
        if self.__host_timer_resolution is None:
            self.__host_timer_resolution = self.__fetch_info(cl_platform_info.CL_PLATFORM_HOST_TIMER_RESOLUTION)

        return self.__host_timer_resolution
    
    @property
    def command_buffer_capabilities(self)->int:
        if self.__command_buffer_capabilities is None:
            self.__command_buffer_capabilities = self.__fetch_info(cl_platform_info.CL_PLATFORM_COMMAND_BUFFER_CAPABILITIES_KHR)

        return self.__command_buffer_capabilities
    
    @property
    def external_memory_import_handle_types(self)->List[cl_external_memory_handle_type_khr]:
        if self.__external_memory_import_handle_types is None:
            self.__external_memory_import_handle_types = self.__fetch_info(cl_platform_info.CL_PLATFORM_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR)

        return self.__external_memory_import_handle_types
    
    @property
    def semaphore_types(self)->List[cl_semaphore_type_khr]:
        if self.__semaphore_types is None:
            self.__semaphore_types = self.__fetch_info(cl_platform_info.CL_PLATFORM_SEMAPHORE_TYPES_KHR)

        return self.__semaphore_types
    
    @property
    def semaphore_import_handle_types(self)->List[cl_external_semaphore_handle_type_khr]:
        if self.__semaphore_import_handle_types is None:
            self.__semaphore_import_handle_types = self.__fetch_info(cl_platform_info.CL_PLATFORM_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR)

        return self.__semaphore_import_handle_types
    
    @property
    def semaphore_export_handle_types(self)->List[cl_external_semaphore_handle_type_khr]:
        if self.__semaphore_export_handle_types is None:
            self.__semaphore_export_handle_types = self.__fetch_info(cl_platform_info.CL_PLATFORM_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR)

        return self.__semaphore_export_handle_types
    
    @property
    def icd_suffix(self)->str:
        if self.__icd_suffix is None:
            self.__icd_suffix = self.__fetch_info(cl_platform_info.CL_PLATFORM_ICD_SUFFIX_KHR)

        return self.__icd_suffix
    
    def __repr__(self)->str:
        return f"Platform('{self.name}')"
    
    def __parse_array(self, buffer, cls):
        step:int = sizeof(cls)
        n:int = len(buffer) // step
        result = []
        offset:int = 0
        is_prim:bool = issubclass(cls, _SimpleCData)
        for i in range(n):
            value = cls.from_buffer_copy(buffer[offset:offset+step])
            if is_prim:
                value = value.value

            result.append(value)
            offset += step
        return result

    def __fetch_info(self, key:cl_platform_info)->Any:
        result_size = c_size_t()
        CL.clGetPlatformInfo(self.__id, key, 0, None, byref(result_size))

        result_bytes = (c_char * result_size.value)()
        CL.clGetPlatformInfo(self.__id, key, result_size, result_bytes, None)

        if key in [
            cl_platform_info.CL_PLATFORM_PROFILE,
            cl_platform_info.CL_PLATFORM_VERSION,
            cl_platform_info.CL_PLATFORM_NAME,
            cl_platform_info.CL_PLATFORM_VENDOR,
            cl_platform_info.CL_PLATFORM_EXTENSIONS,
            cl_platform_info.CL_PLATFORM_ICD_SUFFIX_KHR
        ]:
            result = result_bytes.value.decode("utf-8")
        elif key == cl_platform_info.CL_PLATFORM_NUMERIC_VERSION:
            result = cl_version.from_buffer_copy(result_bytes).value
        elif key == cl_platform_info.CL_PLATFORM_NUMERIC_VERSION_KHR:
            result = cl_version_khr.from_buffer_copy(result_bytes).value
        elif key == cl_platform_info.CL_PLATFORM_EXTENSIONS_WITH_VERSION:
            result = self.__parse_array(result_bytes, cl_name_version)
        elif key == cl_platform_info.CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR:
            result = self.__parse_array(result_bytes, cl_name_version_khr)
        elif key == cl_platform_info.CL_PLATFORM_HOST_TIMER_RESOLUTION:
            result = cl_ulong.from_buffer_copy(result_bytes).value
        elif key == cl_platform_info.CL_PLATFORM_COMMAND_BUFFER_CAPABILITIES_KHR:
            result = cl_platform_command_buffer_capabilities_khr.from_buffer_copy(result_bytes).value
        elif key == cl_platform_info.CL_PLATFORM_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR:
            result = self.__parse_array(result_bytes, cl_external_memory_handle_type_khr)
        elif key == cl_platform_info.CL_PLATFORM_SEMAPHORE_TYPES_KHR:
            result = self.__parse_array(result_bytes, cl_semaphore_type_khr)
        elif key in [
            cl_platform_info.CL_PLATFORM_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR,
            cl_platform_info.CL_PLATFORM_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR
        ]:
            result = self.__parse_array(result_bytes, cl_external_semaphore_handle_type_khr)

        return result