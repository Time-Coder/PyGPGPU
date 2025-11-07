from __future__ import annotations
from ctypes import c_char_p, pointer, c_size_t, c_char, POINTER, c_ubyte, string_at
import re
import os
import json
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from .clobject import CLObject
from .device import Device
from ..runtime import CL, IntEnum, CLInfo, cl_program_info, cl_int, cl_uint, cl_program_build_info
from ...kernel_parser import CPreprocessor
from ...compile_error import CompileError
from ...utils import md5

if TYPE_CHECKING:
    from .context import Context


class Program(CLObject):

    def __init__(self, context:Context, file_name:str, includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[List[str]]=None):
        self._context:Context = context
        self._file_name:str = file_name
        self._includes:Optional[List[str]] = includes if includes is not None else []
        self._defines:Optional[Dict[str, Any]] = defines if defines is not None else []
        self._options:Optional[List[str]] = options if options is not None else []
        self._options_ptr:c_char_p = c_char_p(" ".join(self._options).encode("utf-8"))
        (
            self._clean_code,
            self._line_map,
            self._related_files
        ) = CPreprocessor.macros_expand_file(file_name, includes, defines)
        source_len:int = c_size_t(len(self._clean_code))
        error_code = cl_int(0)
        source = (c_char_p * 1)(self._clean_code.encode("utf-8"))
        program_id = CL.clCreateProgramWithSource(context.id, cl_uint(1), source, pointer(source_len), pointer(error_code))
        CLObject.__init__(self, program_id)

    def build(self):
        error_messages = []
        try:
            CL.clBuildProgram(self.id, self.n_devices, self.device_ids, self._options_ptr, None, None)
        except RuntimeError as e:
            for device, message in self.build_log.items():
                error_message = f"{device} reports:\n{self._format_error(message)}"
                error_messages.append(error_message)

        if error_messages:
            raise CompileError("\n" + "\n\n".join(error_messages))

    def _format_error(self, error_message:str)->str:
        def replace_handler(match:re.Match):
            old_line_number = int(match.group(1))
            file_name, new_line_number = self._line_map[old_line_number]
            return f"{file_name}:{new_line_number}"

        return re.sub(r'<kernel>:(\d+)', replace_handler, error_message.strip("\r\n"))

    @staticmethod
    def _md5(file_name:str, includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[List[str]]=None)->str:
        if includes is None:
            includes = []

        if defines is None:
            defines = {}

        if options is None:
            options = []
        
        file_name = os.path.abspath(file_name).replace("\\", "/")
        clean_includes:List[str] = list(set(os.path.abspath(include).replace("\\", "/") for include in includes))
        clean_defines:List[str] = []
        keys = set(defines.keys())
        for key in keys:
            clean_defines.append(f"{key}={defines[key]}")
        clean_options:List[str] = list(set(options))
        content = {
            "file_name": file_name,
            "includes": clean_includes,
            "defines": clean_defines,
            "options": clean_options
        }
        return md5(json.dumps(content, separators=(',', ':'), indent=None))

    @property
    def binary_sizes(self)->Dict[Device, int]:
        binary_sizes = CLObject.__getattr__(self, "binary_sizes")
        result:Dict[Device, int] = {}
        for i in range(self.n_devices):
            result[self.devices[i]] = binary_sizes[i]

        return result

    @property
    def binaries(self)->Dict[Device, bytes]:
        result:Dict[Device, int] = {}
        binary_sizes = self.binary_sizes
        binaries = (POINTER(c_ubyte) * self.n_devices)()
        for i in range(self.n_devices):
            binaries[i] = (c_ubyte * binary_sizes[self.devices[i]])()

        result_size = c_size_t()
        CL.clGetProgramInfo(self.id, cl_program_info.CL_PROGRAM_BINARIES, 0, None, pointer(result_size))
        CL.clGetProgramInfo(self.id, cl_program_info.CL_PROGRAM_BINARIES, result_size, binaries, None)

        for i in range(self.n_devices):
            binary = string_at(binaries[0], binary_sizes[self.devices[i]])
            result[self.devices[i]] = binary

        return result
    
    @property
    def clean_code(self)->str:
        return self._clean_code
    
    @property
    def options(self)->List[str]:
        return self._options

    def __getattr__(self, name:str):        
        if name in ["build_status", "build_options", "build_log", "binary_type", "global_variable_total_size"]:
            return self._get_build_attr(name)
        else:
            return CLObject.__getattr__(self, name)

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

    @property
    def _prefix(self)->str:
        return "CL_PROGRAM"

    @property
    def _get_info_func(self)->CL.Func:
        return CL.clGetProgramInfo

    @property
    def _info_types_map(self)->Dict[IntEnum, type]:
        return CLInfo.program_info_types

    @property
    def _info_enum(self)->type:
        return cl_program_info
    
    @staticmethod
    def _release(program_id):
        if not program_id:
            return
        
        CL.clReleaseProgram(program_id)