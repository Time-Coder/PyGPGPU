from __future__ import annotations

from ctypes import c_void_p
from typing import List, TYPE_CHECKING, Iterator

from ..runtime.cltypes import (
    cl_version,
    cl_version_khr,
    cl_name_version,
    cl_name_version_khr,
    cl_ulong,
    cl_platform_command_buffer_capabilities_khr,
    cl_external_memory_handle_type_khr,
    cl_semaphore_type_khr,
    cl_external_semaphore_handle_type_khr
)

if TYPE_CHECKING:
    from .device import Device


class Platform:

    @property
    def id(self)->c_void_p: ...
    
    @property
    def extensions(self)->List[str]: ...

    @property
    def n_devices(self)->int: ...

    def device(self, index:int)->Device: ...

    def __iter__(self)->Iterator[Device]: ...

    def __contains__(self, device:Device)->bool: ...

    @property
    def profile(self)->str: ...

    @property
    def version(self)->str: ...

    @property
    def numeric_version(self)->cl_version: ...

    @property
    def numeric_version_khr(self)->cl_version_khr: ...

    @property
    def name(self)->str: ...
    
    @property
    def vendor(self)->str: ...

    @property
    def extensions_with_version(self)->List[cl_name_version]: ...

    @property
    def extensions_with_version_khr(self)->List[cl_name_version_khr]: ...

    @property
    def host_timer_resolution(self)->List[cl_ulong]: ...

    @property
    def command_buffer_capabilities(self)->cl_platform_command_buffer_capabilities_khr: ...

    @property
    def external_memory_import_handle_types(self)->List[cl_external_memory_handle_type_khr]: ...

    @property
    def semaphore_types(self)->List[cl_semaphore_type_khr]: ...

    @property
    def semaphore_import_handle_types(self)->List[cl_external_semaphore_handle_type_khr]: ...

    @property
    def semaphore_export_handle_types(self)->List[cl_external_semaphore_handle_type_khr]: ...

    @property
    def icd_suffix(self)->str: ...