from __future__ import annotations
from ctypes import c_char_p, pointer, c_size_t, c_char, POINTER, c_ubyte, string_at
import re
import os
import json
import warnings
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Union

from .clobject import CLObject
from .device import Device
from .build_options import BuildOptions
from ..runtime import CL, IntEnum, CLInfo, cl_program_info, cl_int, cl_uint, cl_program_build_info, ErrorCode
from ...kernel_parser import CPreprocessor
from ...exceptions import CompileError, CompileWarning
from ...utils import md5sums, save_var, load_var, save_bin, load_bin, modify_time

if TYPE_CHECKING:
    from .context import Context


class Program(CLObject):

    def __init__(self, context:Context, file_name:str, includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[BuildOptions]=None):
        self._context:Context = context
        self._file_name:str = file_name
        self._base_name:str = os.path.basename(file_name)
        self._includes:List[str] = includes if includes is not None else []
        self._defines:Dict[str, Any] = defines if defines is not None else []
        self._options:List[str] = options if options is not None else BuildOptions()
        self._options_ptr:c_char_p = c_char_p(str(self._options).encode("utf-8"))
        self._binaries:Dict[Device, bytes] = {}

        newest_mtime = self._load_meta()

        if not newest_mtime:
            if CL.print_info:
                print(f"preprocessing {self._base_name}... ", end="", flush=True)

            (
                self._clean_code,
                self._line_map,
                self._related_files
            ) = CPreprocessor.macros_expand_file(file_name, includes, defines)
            self._save_meta()

            if CL.print_info:
                print(f"done.", flush=True)

        else:
            if CL.print_info:
                print(f"load {self._base_name}'s meta info from cache.")
        
        if not self._load_bin(newest_mtime):
            source_len:int = c_size_t(len(self._clean_code))
            error_code = cl_int(0)
            source = (c_char_p * 1)(self._clean_code.encode("utf-8"))
            program_id = CL.clCreateProgramWithSource(context.id, cl_uint(1), source, pointer(source_len), pointer(error_code))
        else:
            binary_sizes = (c_size_t * self.n_devices)()
            binaries = (POINTER(c_ubyte) * self.n_devices)()
            for i, device in enumerate(self.devices):
                binary = self._binaries[device]
                binary_sizes[i] = len(binary)
                binaries[i] = (c_ubyte * binary_sizes[i]).from_buffer_copy(binary)

            binary_status = (cl_int * self.n_devices)()
            error_code = cl_int(0)

            try:
                program_id = CL.clCreateProgramWithBinary(context.id, cl_uint(self.n_devices), self.device_ids, binary_sizes, binaries, binary_status, pointer(error_code))
            except BaseException as e:
                pass

            if CL.check_error:
                error_messages = []
                for i in range(self.n_devices):
                    device = self.devices[i]
                    status = binary_status[i]
                    if status == ErrorCode.CL_INVALID_VALUE:
                        error_messages.append(f"binary for {device} error: {ErrorCode.CL_INVALID_VALUE}: lengths[{i}] is zero or if binaries[{i}] is a NULL value")
                    elif status == ErrorCode.CL_INVALID_BINARY:
                        error_messages.append(f"binary for {device} error: {ErrorCode.CL_INVALID_VALUE}: program binary is not a valid binary for this device")

                if error_messages:
                    raise RuntimeError(f"{error_code}:\n{'\n'.join(error_messages)}")

                if error_code.value != ErrorCode.CL_SUCCESS:
                    raise e
                
            if CL.print_info:
                print(f"load {self._base_name}'s binary from cache.")

        CLObject.__init__(self, program_id)

    def build(self):
        try:
            if CL.print_info:
                print(f"building {self._base_name}... ", end="", flush=True)

            error_code = CL.clBuildProgram(self.id, self.n_devices, self.device_ids, self._options_ptr, None, None)
            success = (error_code == ErrorCode.CL_SUCCESS)

            if CL.print_info:
                print(f"done.", flush=True)
        except RuntimeError as e:
            success = False

        if success:
            self._save_bin()

        if CL.check_error:
            error_messages = []
            for device, message in self.build_log.items():
                message = message.strip("\n")
                if not message:
                    continue

                error_message = f"{device} reports:\n{self._format_error(message)}"
                error_messages.append(error_message)

            final_message:str = "\n" + "\n\n".join(error_messages)

            if not success:
                if error_messages:
                    raise CompileError(final_message)
                else:
                    raise e
            else:
                if error_messages:
                    warnings.warn(final_message, CompileWarning)

    def _format_error(self, error_message:str)->str:
        def replace_handler(match:re.Match):
            old_line_number = int(match.group(1))
            file_name, new_line_number = self._line_map[old_line_number]
            return f"{file_name}:{new_line_number}"

        return re.sub(r'<kernel>:(\d+)', replace_handler, error_message.strip("\r\n"))
    
    @property
    def _cache_folder(self)->str:
        self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        return self_folder + "/__clcache__"
    
    @property
    def _meta_file_name(self)->str:
        return f"{self._cache_folder}/{self._base_name}_{self._md5}.meta"
    
    def _bin_file_name(self, device:Device)->str:
        return f"{self._cache_folder}/{device.unique_key}/{self._base_name}_{self._md5}.bin"

    def _save_meta(self)->None:
        meta:Dict[str, Any] = {
            "clean_code": self._clean_code,
            "related_files": self._related_files,
            "line_map": self._line_map
        }
        save_var(meta, self._meta_file_name)

    def _save_bin(self)->None:
        for device in self.devices:
            save_bin(self.binaries[device], self._bin_file_name(device))

    def _load_meta(self)->Union[bool, float]:
        meta_mtime = modify_time(self._meta_file_name)
        if modify_time(self._file_name) > meta_mtime:
            return False
        
        newest_mtime:float = 0
        meta:Dict[str, Any] = load_var(self._meta_file_name)
        for related_file in meta["related_files"]:
            if not os.path.isfile(related_file):
                return False

            related_file_mtime = modify_time(related_file)
            if related_file_mtime > newest_mtime:
                newest_mtime = related_file_mtime

            if related_file_mtime > meta_mtime:
                return False
            
        self._clean_code = meta["clean_code"]
        self._related_files = meta["related_files"]
        self._line_map = meta["line_map"]
        return newest_mtime
            
    def _load_bin(self, newest_mtime:Union[float, bool])->bool:
        if not newest_mtime:
            return False
        
        for device in self.devices:
            bin_file_name = self._bin_file_name(device)            
            if newest_mtime > modify_time(bin_file_name):
                return False
            
        for device in self.devices:
            bin_file_name = self._bin_file_name(device)            
            self._binaries[device] = load_bin(bin_file_name)

        return True

    @property
    def _md5(self)->str:
        return Program._md5_of(self._file_name, self._includes, self._defines, self._options)

    @staticmethod
    def _md5_of(file_name:str, includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[BuildOptions]=None)->str:
        if includes is None:
            includes = []

        if defines is None:
            defines = {}

        if options is None:
            options = BuildOptions()
        
        file_name = os.path.abspath(file_name).replace("\\", "/")
        clean_includes:List[str] = []
        for include in includes:
            include = os.path.abspath(include).replace("\\", "/")
            if include not in clean_includes:
                clean_includes.append(include)
                
        content = {
            "file_name": file_name,
            "includes": clean_includes,
            "defines": defines,
            "options": str(options)
        }
        return md5sums(json.dumps(content, separators=(',', ':'), indent=None))

    @property
    def binary_sizes(self)->Dict[Device, int]:
        binary_sizes = CLObject.__getattr__(self, "binary_sizes")
        result:Dict[Device, int] = {}
        for i in range(self.n_devices):
            result[self.devices[i]] = binary_sizes[i]

        return result

    @property
    def binaries(self)->Dict[Device, bytes]:
        if self._binaries:
            return self._binaries
        
        binary_sizes = self.binary_sizes
        binaries = (POINTER(c_ubyte) * self.n_devices)()
        for i in range(self.n_devices):
            binaries[i] = (c_ubyte * binary_sizes[self.devices[i]])()

        result_size = c_size_t()
        CL.clGetProgramInfo(self.id, cl_program_info.CL_PROGRAM_BINARIES, 0, None, pointer(result_size))
        CL.clGetProgramInfo(self.id, cl_program_info.CL_PROGRAM_BINARIES, result_size, binaries, None)

        for i in range(self.n_devices):
            binary = string_at(binaries[0], binary_sizes[self.devices[i]])
            self._binaries[self.devices[i]] = binary

        return self._binaries
    
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