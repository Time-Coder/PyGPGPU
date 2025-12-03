from __future__ import annotations
from ctypes import c_char_p, pointer
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Set, override

from .cuobject import CUObject
from .device import Device
from .build_options import BuildOptions
from ..driver import (
    CUDA,
    CUmodule
)
from .program_info import KernelInfo
from .nvrtc_program import NVRTCProgram
from .kernel import Kernel
from .context import Context
from .device import Device


class Program(CUObject):

    def __init__(self, file_name:str="", includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[BuildOptions]=None, type_checked:bool=False, nvrtc_program:NVRTCProgram=None):
        self._kernels:Dict[str, Kernel] = {}
        self._type_checked:bool = type_checked
        if nvrtc_program is None:
            self._nvrtc_program:NVRTCProgram = NVRTCProgram(file_name, includes, defines, options)
        else:
            self._nvrtc_program:NVRTCProgram = nvrtc_program

        program_id = CUmodule()
        ptr_program_id = pointer(program_id)
        CUDA.cuModuleLoadData(ptr_program_id, self._nvrtc_program.cubin(Context.current().device))
        CUObject.__init__(self, program_id)

    @property
    def file_name(self)->str:
        return self._nvrtc_program.file_name
    
    @property
    def base_name(self)->str:
        return self._nvrtc_program.base_name
    
    @property
    def includes(self)->List[str]:
        return self._nvrtc_program.includes
    
    @property
    def defines(self)->Dict[str, Any]:
        return self._nvrtc_program.defines
    
    @property
    def options(self)->BuildOptions:
        return self._nvrtc_program.options
    
    @property
    def options_ptr(self)->c_char_p:
        return self._nvrtc_program.options_ptr
    
    @property
    def clean_code(self)->str:
        return self._nvrtc_program.clean_code

    @property
    def line_map(self)->Dict[int, str]:
        return self._nvrtc_program.line_map
    
    @property
    def related_files(self)->Set[str]:
        return self._nvrtc_program.related_files
    
    @property
    def kernel_infos(self)->Dict[str, KernelInfo]:
        return self._nvrtc_program.kernel_infos
    
    @property
    def kernel_names(self)->List[str]:
        return list(self.kernel_infos.keys())
    
    @property
    def n_kernels(self)->int:
        return len(self.kernel_infos)
    
    @property
    def kernels(self)->Dict[str, Kernel]:
        self._fetch_kernels()
        return self._kernels
    
    @property
    def structs(self)->Dict[str, type]:
        return self._nvrtc_program.structs
    
    def _fetch_kernels(self):
        if self._kernels:
            return
        
        kernel_ids = (cl_kernel * self.n_kernels)()
        CL.clCreateKernelsInProgram(self.id, self.n_kernels, kernel_ids, None)
        for kernel_id in kernel_ids:
            kernel = Kernel(self, kernel_id)
            kernel.args = self.kernel_infos[kernel.name].args
            kernel.type_checked = self._type_checked
            self._kernels[kernel.name] = kernel

    def __getattr__(self, name:str):
        if name in self.kernel_infos:
            return self.kernels[name]
        elif name in self.structs:
            return self.structs[name]
        else:
            return CUObject.__getattr__(self, name)
        
    def __getitem__(self, name:str)->Kernel:
        if name in self.kernel_infos:
            return self.kernels[name]
        elif name in self.structs:
            return self.structs[name]
        else:
            raise KeyError(name)

    @override    
    @staticmethod
    def _prefix()->str:
        return ""

    @override    
    @staticmethod
    def _get_info_func():
        return None


    @override    
    @staticmethod
    def _info_enum()->type:
        return None

    @override    
    @staticmethod
    def _release_func():
        return CUDA.cuModuleUnload