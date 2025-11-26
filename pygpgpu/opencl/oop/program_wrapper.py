from __future__ import annotations
from ctypes import c_char_p
from typing import Dict, List, Any, Optional, Set
from types import ModuleType

from .device import Device
from .build_options import BuildOptions
from .program_parser import ProgramParser
from .kernel_info import KernelInfo
from .kernel_wrapper import KernelWrapper
from .program import Program
from .platform import Platform


class ProgramWrapper:

    def __init__(self, file_name:str, includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[BuildOptions]=None, type_checked:bool=False):
        self._binaries:Dict[Device, bytes] = {}
        self._kernel_wrappers:Dict[str, KernelWrapper] = {}
        self._type_checked:bool = type_checked
        self._program_parser:ProgramParser = ProgramParser()
        self._program_parser.parse(file_name, includes, defines, options)
        self._programs:Dict[Platform, Program] = {}

    def program(self, device:Device)->Program:
        if device.platform not in self._programs:
            self._programs[device.platform] = Program(device.platform.default_context, type_checked=self._type_checked, program_parser=self._program_parser)

        return self._programs[device.platform]

    @property
    def file_name(self)->str:
        return self._program_parser.file_name
    
    @property
    def base_name(self)->str:
        return self._program_parser.base_name
    
    @property
    def includes(self)->List[str]:
        return self._program_parser.includes
    
    @property
    def defines(self)->Dict[str, Any]:
        return self._program_parser.defines
    
    @property
    def options(self)->BuildOptions:
        return self._program_parser.options
    
    @property
    def options_ptr(self)->c_char_p:
        return self._program_parser.options_ptr
    
    @property
    def clean_code(self)->str:
        return self._program_parser.clean_code

    @property
    def line_map(self)->Dict[int, str]:
        return self._program_parser.line_map
    
    @property
    def related_files(self)->Set[str]:
        return self._program_parser.related_files
    
    @property
    def kernel_infos(self)->Dict[str, KernelInfo]:
        return self._program_parser.kernel_infos
    
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
    
    @property
    def structs(self)->Dict[str, type]:
        return self._program_parser._struct_types
    
    def _fetch_kernel_wrappers(self):
        if self._kernel_wrappers:
            return
        
        for name, kernel_info in self.kernel_infos.items():
            kernel = KernelWrapper(self, name, kernel_info.args, self._type_checked)
            self._kernel_wrappers[kernel.name] = kernel

    def __getattr__(self, name:str)->KernelWrapper:
        if name in self.kernels_wrappers:
            return self.kernels_wrappers[name]
        elif name in self.structs:
            return self.structs[name]
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
        
    def __getitem__(self, name:str)->KernelWrapper:
        if name in self.kernels_wrappers:
            return self.kernels_wrappers[name]
        elif name in self.structs:
            return self.structs[name]
        else:
            raise KeyError(name)
