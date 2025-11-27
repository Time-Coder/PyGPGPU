from __future__ import annotations

from ctypes import c_void_p
from typing import List, TYPE_CHECKING, Iterator, Tuple, Optional, Dict, Any

from ..driver import (
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

from .build_options import BuildOptions
if TYPE_CHECKING:
    from .device import Device
    from .program import Program
    from .context import Context


class Platform:

    def compile(self,
        file_name:str,
        includes:Optional[List[str]] = None,
        defines:Optional[Dict[str, Any]] = None,
        single_precision_constant:bool=False,
        denorms_are_zero:bool=False,
        fp32_correctly_rounded_divide_sqrt:bool=False,
        opt_disable:bool=False,
        strict_aliasing:bool=False,
        uniform_work_group_size:bool=False,
        no_subgroup_ifp:bool=False,
        mad_enable:bool=False,
        no_signed_zeros:bool=False,
        unsafe_math_optimizations:bool=False,
        finite_math_only:bool=False,
        fast_relaxed_math:bool=False,
        w:bool=False,
        Werror:bool=False,
        cl_std:Optional[float]=None,
        kernel_arg_info:bool=False,
        g:bool=False,
        create_library:bool=False,
        enable_link_options:bool=False,
        x_spir:bool=False,
        spir_std:Optional[float]=None,
        type_checked:bool=False,
        options:Optional[BuildOptions]=None
    )->Program: ...

    @property
    def id(self)->c_void_p: ...
    
    @property
    def extensions(self)->List[str]: ...

    @property
    def n_devices(self)->int: ...

    @property
    def devices(self)->Tuple[Device]: ...

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

    @property
    def default_context(self)->Context: ...

    @property
    def create_context(self, *devices)->Context: ...