from __future__ import annotations
from ctypes import c_char_p, pointer
from typing import Dict, List, Any, Optional, Set, override

from .hipobject import HIPObject
from .build_options import BuildOptions
from ..driver import (
    HIP,
    CUmodule
)
from ctypes import c_char
from .program_info import KernelInfo
from .hip_compiler import HipCompiler
from .kernel import Kernel
from .context import Context


class Program(HIPObject):

    def __init__(self, file_name:str="", includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[BuildOptions]=None, type_checked:bool=False, compiler:Optional[HipCompiler]=None):
        self._kernels:Dict[str, Kernel] = {}
        self._type_checked:bool = type_checked
        if compiler is None:
            self._compiler:HipCompiler = HipCompiler(file_name, includes, defines, options)
        else:
            self._compiler:HipCompiler = compiler

        ctx = Context.current()
        if ctx is None:
            raise RuntimeError("no current HIP context; call device.default_context.make_current() first")
        ctx.make_current()
        image:bytes = self._compiler.code_object(ctx.device)
        image_buf = (c_char * len(image)).from_buffer_copy(image)
        program_id = CUmodule()
        ptr_program_id = pointer(program_id)
        HIP.hipModuleLoadData(ptr_program_id, image_buf)
        HIPObject.__init__(self, program_id)

    @property
    def file_name(self)->str:
        return self._compiler.file_name
    
    @property
    def base_name(self)->str:
        return self._compiler.base_name
    
    @property
    def includes(self)->List[str]:
        return self._compiler.includes
    
    @property
    def defines(self)->Dict[str, Any]:
        return self._compiler.defines
    
    @property
    def options(self)->BuildOptions:
        return self._compiler.options
    
    @property
    def options_ptr(self)->c_char_p:
        return self._compiler.options_ptr
    
    @property
    def clean_code(self)->str:
        return self._compiler.clean_code

    @property
    def line_map(self)->Dict[int, str]:
        return self._compiler.line_map
    
    @property
    def related_files(self)->Set[str]:
        return self._compiler.related_files
    
    @property
    def kernel_infos(self)->Dict[str, KernelInfo]:
        return self._compiler.kernel_infos
    
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
        return self._compiler.structs
    
    def _fetch_kernels(self):
        if self._kernels:
            return

        for kernel_name, kernel_info in self.kernel_infos.items():
            kernel = Kernel(self, kernel_name)
            kernel.args = kernel_info.args
            kernel.type_checked = self._type_checked
            self._kernels[kernel_name] = kernel

    def __getattr__(self, name:str):
        if name in self.kernel_infos:
            return self.kernels[name]
        elif name in self.structs:
            return self.structs[name]
        else:
            return HIPObject.__getattr__(self, name)
        
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
    def _info_enum()->type|None:
        return None

    @override    
    @staticmethod
    def _release_func():
        return HIP.hipModuleUnload