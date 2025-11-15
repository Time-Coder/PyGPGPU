from __future__ import annotations
from ctypes import c_char_p, pointer, c_size_t, c_char, POINTER, c_ubyte, string_at
import os
import warnings
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Union, Set

from .clobject import CLObject
from .device import Device
from .build_options import BuildOptions
from ..runtime import (
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
from .kernel_parser import KernelParser, KernelInfo
from .kernel import Kernel
from ...exceptions import CompileError, CompileWarning
from ...utils import save_bin, load_bin, modify_time

if TYPE_CHECKING:
    from .context import Context


class Program(CLObject):

    def __init__(self, context:Context, file_name:str, includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[BuildOptions]=None):
        self._context:Context = context
        self._binaries:Dict[Device, bytes] = {}
        self._kernel_parser:KernelParser = KernelParser()
        self._kernels:Dict[str, Kernel] = {}
        newest_mtime = self._kernel_parser.parse(file_name, includes, defines, options)

        old_check_error = CL.check_error
        if not self._load_bin(newest_mtime):
            self._create_with_source()
        else:
            try:
                CL.check_error = True
                self._create_with_binary()
            except:
                CL.check_error = old_check_error
                self._binaries.clear()
                self._create_with_source()
            finally:
                CL.check_error = old_check_error

    def _create_with_source(self)->cl_program:
        source_len:int = c_size_t(len(self._kernel_parser.clean_code))
        error_code = cl_int(0)
        source = (c_char_p * 1)(self._kernel_parser.clean_code.encode("utf-8"))
        program_id = CL.clCreateProgramWithSource(self.context.id, cl_uint(1), source, pointer(source_len), pointer(error_code))
        CLObject.__init__(self, program_id)
        self._build(True)
        return program_id
    
    def _create_with_binary(self)->cl_program:
        binary_sizes = (c_size_t * self.n_devices)()
        binaries = (POINTER(c_ubyte) * self.n_devices)()
        for i, device in enumerate(self.devices):
            binary = self._binaries[device]
            binary_sizes[i] = len(binary)
            binaries[i] = (c_ubyte * binary_sizes[i]).from_buffer_copy(binary)

        binary_status = (cl_int * self.n_devices)()
        error_code = cl_int(0)

        try:
            program_id = CL.clCreateProgramWithBinary(self.context.id, cl_uint(self.n_devices), self.device_ids, binary_sizes, binaries, binary_status, pointer(error_code))
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
            print(f"load {self._kernel_parser.base_name}'s binary from cache.")

        CLObject.__init__(self, program_id)
        self._build(False)
        return program_id

    def _build(self, save:bool):
        try:
            if CL.print_info:
                print(f"building {self._kernel_parser.base_name} ... ", end="", flush=True)

            error_code = CL.clBuildProgram(self.id, self.n_devices, self.device_ids, self._kernel_parser.options_ptr, None, None)
            success = (error_code == ErrorCode.CL_SUCCESS)

            if CL.print_info:
                print(f"done.", flush=True)
        except RuntimeError as e:
            success = False

        if save and success:
            self._save_bin()

        if CL.print_info and not success:
            print(f"error.", flush=True)

        if CL.check_error:
            error_messages = []
            for device, message in self.build_log.items():
                message = message.strip("\n")
                if not message:
                    continue

                error_message = f"{device} reports:\n{self._kernel_parser.format_error(message)}"
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
    
    @property
    def _cache_folder(self)->str:
        self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        return self_folder + "/__clcache__"
    
    def _bin_file_name(self, device:Device)->str:
        return f"{self._cache_folder}/{device.unique_key}/{self._kernel_parser.base_name}_{self._kernel_parser.md5}.bin"

    def _save_bin(self)->None:
        for device in self.devices:
            save_bin(self.binaries[device], self._bin_file_name(device))
            
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
    def kernels(self)->Dict[str, Kernel]:
        self._fetch_kernels()
        return self._kernels
    
    def _fetch_kernels(self):
        if self._kernels:
            return
        
        kernel_ids = (cl_kernel * self.n_kernels)()
        CL.clCreateKernelsInProgram(self.id, self.n_kernels, kernel_ids, None)
        for kernel_id in kernel_ids:
            kernel = Kernel(kernel_id, self)
            kernel.args = self.kernel_infos[kernel.name].args
            self._kernels[kernel.name] = kernel

    def __getattr__(self, name:str):
        if name in ["build_status", "build_options", "build_log", "binary_type", "global_variable_total_size"]:
            return self._get_build_attr(name)
        elif name in self.kernel_infos:
            return self.kernels[name]
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

    @staticmethod
    def _prefix()->str:
        return "CL_PROGRAM"

    @staticmethod
    def _get_info_func()->CL.Func:
        return CL.clGetProgramInfo

    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]:
        return CLInfo.program_info_types

    @staticmethod
    def _info_enum()->type:
        return cl_program_info
    
    @staticmethod
    def _release_func():
        return CL.clReleaseProgram