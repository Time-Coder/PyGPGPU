from __future__ import annotations
from ctypes import c_char_p, pointer, c_size_t, c_char
import warnings
from typing import Dict, List, Any, Optional, Union, Set, override

from .cuobject import CUObject
from .build_options import BuildOptions
from ..driver import (
    NVRTC,
    nvrtcResult,
    nvrtcProgram
)
from .program_parser import ProgramParser
from .program_info import KernelInfo
from .device import Device
from ...exceptions import CompileError, CompileWarning
from ...utils import save_bin, load_bin, modify_time


class NVRTCProgram(CUObject):

    def __init__(self, file_name:str="", includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[BuildOptions]=None, program_parser:ProgramParser=None):
        self._ptx:Dict[Device, bytes] = {}
        self._cubin:Dict[Device, bytes] = {}

        if program_parser is None:
            self._program_parser:ProgramParser = ProgramParser()
            self._program_parser.parse(file_name, includes, defines, options)
        else:
            self._program_parser:ProgramParser = program_parser

    def _create_with_source(self)->nvrtcProgram:
        program_id = nvrtcProgram(0)
        ptr_program_id = pointer(program_id)
        source = c_char_p(self._program_parser.clean_code.encode("utf-8") + b"\0")
        name = c_char_p(b"<kernel>")
        NVRTC.nvrtcCreateProgram(ptr_program_id, source, name, 0, None, None)
        CUObject.__init__(self, program_id)
        return program_id

    def _build(self, device:Device, save:bool):
        try:
            if NVRTC.print_info:
                print(f"building {self._program_parser.base_name} ... ", end="", flush=True)

            error_code = NVRTC.nvrtcCompileProgram(self.id, self._program_parser.n_options, self._program_parser.options_ptr)
            success = (error_code == nvrtcResult.NVRTC_SUCCESS)

            if NVRTC.print_info:
                print(f"done.", flush=True)
        except RuntimeError as e:
            success = False

        if success:
            ptx_size = c_size_t(0)
            ptr_ptx_size = pointer(ptx_size)
            NVRTC.nvrtcGetPTXSize(self.id, ptr_ptx_size)

            ptx = (c_char * ptx_size.value)()
            NVRTC.nvrtcGetPTX(self.id, ptx)
            self._ptx[device] = ptx.raw

            cubin_size = c_size_t(0)
            ptr_cubin_size = pointer(cubin_size)
            NVRTC.nvrtcGetCUBINSize(self.id, ptr_cubin_size)

            cubin = (c_char * cubin_size.value)()
            NVRTC.nvrtcGetCUBIN(self.id, cubin)
            self._cubin[device] = cubin.raw

            if save:
                self._save_ptx(device)
                self._save_cubin(device)

        if NVRTC.print_info and not success:
            print(f"error.", flush=True)

        if NVRTC.check_error:
            log_size = c_size_t(0)
            ptr_log_size = pointer(log_size)
            NVRTC.nvrtcGetProgramLogSize(self.id, ptr_log_size)

            if log_size.value > 0:
                log = (c_char * log_size.value)()
                NVRTC.nvrtcGetProgramLog(self.id, log)

                error_message = self._program_parser.format_error(log.value)
                if success:
                    warnings.warn(error_message, CompileWarning)
                else:
                    raise CompileError(error_message)
    
    def _cubin_file_name(self, device:Device)->str:
        return f"{self._program_parser.cache_folder}/{device.unique_key}/{self._program_parser.base_name}_{self._program_parser.md5}.cubin"
    
    def _ptx_file_name(self, device:Device)->str:
        return f"{self._program_parser.cache_folder}/{device.unique_key}/{self._program_parser.base_name}_{self._program_parser.md5}.ptx"

    def _save_cubin(self, device:Device)->None:
        save_bin(self._cubin[device], self._cubin_file_name(device))
    
    def _save_ptx(self, device:Device)->None:
        save_bin(self._ptx[device], self._ptx_file_name(device))

    def _load_cubin(self, device:Device)->bool:
        if not self._program_parser.newest_mtime:
            return False
        
        cubin_file_name = self._cubin_file_name(device)
        if self._program_parser.newest_mtime > modify_time(cubin_file_name):
            return False
                       
        self._cubin[device] = load_bin(cubin_file_name)

        return True
    
    def _load_ptx(self, device:Device)->bool:
        if not self._program_parser.newest_mtime:
            return False
        
        ptx_file_name = self._ptx_file_name(device)
        if self._program_parser.newest_mtime > modify_time(ptx_file_name):
            return False
        
        self._ptx[device] = load_bin(ptx_file_name)

        return True
    
    def ptx(self, device:Device)->bytes:
        if device not in self._ptx:
            if not self._load_cubin(device):
                if not self.id:
                    self._create_with_source()
                    self._build(device, True)

        return self._ptx[device]

    def cubin(self, device:Device)->bytes:
        if device not in self._ptx:
            if not self._load_cubin(device):
                if not self.id:
                    self._create_with_source()
                    self._build(device, True)

        return self._cubin[device]
    
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
    def structs(self)->Dict[str, type]:
        return self._program_parser._struct_types

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
        return NVRTC.nvrtcDestroyProgram