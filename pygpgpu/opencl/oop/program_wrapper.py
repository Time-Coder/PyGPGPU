from __future__ import annotations
from ctypes import c_char_p
from typing import Dict, List, Any, Optional, Set
from types import ModuleType

from .device import Device
from .build_options import BuildOptions
from .kernel_parser import KernelParser
from .kernel_info import KernelInfo
from .kernel_wrapper import KernelWrapper
from .program import Program
from .platform import Platform


class ProgramWrapper:

    def __init__(self, file_name:str, includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[BuildOptions]=None, type_checked:bool=False):
        self._binaries:Dict[Device, bytes] = {}
        self._kernel_wrappers:Dict[str, KernelWrapper] = {}
        self._type_checked:bool = type_checked
        self._kernel_parser:KernelParser = KernelParser()
        self._kernel_parser.parse(file_name, includes, defines, options)
        self._programs:Dict[Platform, Program] = {}

    def program(self, device:Device)->Program:
        if device.platform not in self._programs:
            self._programs[device.platform] = Program(device.platform.default_context, type_checked=self._type_checked, kernel_parser=self._kernel_parser)

        return self._programs[device.platform]

    @property
    def file_name(self)->str:
        return self._kernel_parser.file_name
    
    @property
    def base_name(self)->str:
        return self._kernel_parser.base_name
    
    @property
    def includes(self)->List[str]:
        return self._kernel_parser.includes
    
    @property
    def defines(self)->Dict[str, Any]:
        return self._kernel_parser.defines
    
    @property
    def options(self)->BuildOptions:
        return self._kernel_parser.options
    
    @property
    def options_ptr(self)->c_char_p:
        return self._kernel_parser.options_ptr
    
    @property
    def clean_code(self)->str:
        return self._kernel_parser.clean_code

    @property
    def line_map(self)->Dict[int, str]:
        return self._kernel_parser.line_map
    
    @property
    def related_files(self)->Set[str]:
        return self._kernel_parser.related_files
    
    @property
    def kernel_infos(self)->Dict[str, KernelInfo]:
        return self._kernel_parser.kernel_infos
    
    @property
    def kernel_names(self)->List[str]:
        return list(self.kernel_infos.keys())
    
    @property
    def n_kernels(self)->int:
        return len(self.kernel_infos)
    
    @property
    def kernels_wrappers(self)->Dict[str, KernelWrapper]:
        self._fetch_kernel_wrappers()
        return self._kernel_wrappers
    
    def _fetch_kernel_wrappers(self):
        if self._kernel_wrappers:
            return
        
        for name, kernel_info in self.kernel_infos.items():
            kernel = KernelWrapper(self, name, kernel_info.args, self._type_checked)
            self._kernel_wrappers[kernel.name] = kernel

    def __getattr__(self, name:str)->KernelWrapper:
        return self.kernels_wrappers[name]
        
    def __getitem__(self, name:str)->KernelWrapper:
        return self.kernels_wrappers[name]
