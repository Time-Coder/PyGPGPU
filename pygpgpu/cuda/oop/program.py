from __future__ import annotations
from ctypes import c_char_p, pointer, c_size_t, c_char, POINTER, c_ubyte, string_at
import warnings
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Union, Set, override

from .clobject import CLObject
from .device import Device
from .build_options import BuildOptions
from ..driver import (
    CL,
    IntEnum,
    CLInfo,
    cl_program_info,
    cl_int,
    cl_uint,
    cl_program_build_info,
    ErrorCode,
    cl_program,
    cl_kernel
)
from .program_parser import ProgramParser
from .program_info import KernelInfo
from .nvrtc_program import NVRTCProgram
from .kernel import Kernel
from ...exceptions import CompileError, CompileWarning
from ...utils import save_bin, load_bin, modify_time

if TYPE_CHECKING:
    from .context import Context


class Program(CLObject):

    def __init__(self, file_name:str="", includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[BuildOptions]=None, type_checked:bool=False, program_parser:ProgramParser=None):
        self._kernels:Dict[str, Kernel] = {}
        self._type_checked:bool = type_checked
        self._nvrtc_program:NVRTCProgram = NVRTCProgram(file_name, includes, defines, options, program_parser)


        

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
    def kernels(self)->Dict[str, Kernel]:
        self._fetch_kernels()
        return self._kernels
    
    @property
    def structs(self)->Dict[str, type]:
        return self._program_parser._struct_types
    
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
        if name in ["build_status", "build_options", "build_log", "binary_type", "global_variable_total_size"]:
            return self._get_build_attr(name)
        elif name in self.kernel_infos:
            return self.kernels[name]
        elif name in self.structs:
            return self.structs[name]
        else:
            return CLObject.__getattr__(self, name)
        
    def __getitem__(self, name:str)->Kernel:
        return self.kernels[name]

    def _get_build_attr(self, name:str)->Any:
        if name in self._info:
            return self._info[name]
        
        key = None
        if name == "build_status":
            key = cl_program_build_info.CL_PROGRAM_BUILD_STATUS
        elif name == "build_options":
            key = cl_program_build_info.CL_PROGRAM_BUILD_OPTIONS
        elif name == "build_log":
            key = cl_program_build_info.CL_PROGRAM_BUILD_LOG
        elif name == "binary_type":
            key = cl_program_build_info.CL_PROGRAM_BINARY_TYPE
        elif name == "global_variable_total_size":
            key = cl_program_build_info.CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE

        if key is None:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        self._info[name] = self._fetch_build_info(key)

        return self._info[name]
    
    def _fetch_build_info(self, key:cl_program_build_info)->Dict[Device, Any]:
        result:Dict[Device, Any] = {}
        for device in self.devices:
            result_size = c_size_t()
            CL.clGetProgramBuildInfo(self.id, device.id, key, 0, None, pointer(result_size))

            result_bytes = (c_char * result_size.value)()
            CL.clGetProgramBuildInfo(self.id, device.id, key, result_size, result_bytes, None)

            cls = CLInfo.program_build_info_types[key]
            result[device] = self._parse_result(result_bytes, cls)

        return result

    @property
    def context(self)->Context:
        return self._context
    
    @property
    def devices(self)->List[Device]:
        return self._context.devices
    
    @property
    def device_ids(self):
        return self._context.device_ids
    
    @property
    def n_devices(self)->int:
        return self._context.n_devices

    @override    
    @staticmethod
    def _prefix()->str:
        return "CL_PROGRAM"

    @override    
    @staticmethod
    def _get_info_func()->CL.Func:
        return CL.clGetProgramInfo

    @override    
    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]:
        return CLInfo.program_info_types

    @override    
    @staticmethod
    def _info_enum()->type:
        return cl_program_info

    @override    
    @staticmethod
    def _release_func():
        return CL.clReleaseProgram