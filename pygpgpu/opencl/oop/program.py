from __future__ import annotations
from ctypes import c_char_p, pointer, c_size_t, c_char
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from .clobject import CLObject
from .device import Device
from ..runtime import CL, IntEnum, CLInfo, cl_platform_info, cl_int, cl_uint, cl_device_id, cl_program_build_info
from ...kernel_parser import CPreprocessor
from ...compile_error import CompileError

if TYPE_CHECKING:
    from .context import Context


class Program(CLObject):

    def __init__(self, context:Context, file_name:str, includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None):
        self._context:Context = context
        self._file_name:str = file_name
        self._includes:Optional[List[str]] = includes
        self._defines:Optional[Dict[str, Any]] = defines
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
        n_devices = len(self.devices)
        devices_ids = (cl_device_id * n_devices)()
        for i in range(n_devices):
            devices_ids[i] = self.devices[i].id

        error_messages = []
        try:
            CL.clBuildProgram(self.id, n_devices, devices_ids, None, None, None)
        except RuntimeError as e:
            for device_id in devices_ids:
                log_size = c_size_t()
                CL.clGetProgramBuildInfo(self.id, device_id, cl_program_build_info.CL_PROGRAM_BUILD_LOG, 0, None, pointer(log_size))
                log = (c_char * log_size.value)()
                CL.clGetProgramBuildInfo(self.id, device_id, cl_program_build_info.CL_PROGRAM_BUILD_LOG, log_size, log, None)
                error_message = log.value.decode("utf-8").strip("\n")
                error_messages.append(error_message)

        if error_messages:
            raise CompileError("\n" + "\n\n".join(error_messages))

    @property
    def context(self)->Context:
        return self._context
    
    @property
    def devices(self)->List[Device]:
        return self._context.devices

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
        return cl_platform_info
    
    @staticmethod
    def _release(program_id):
        if not program_id:
            return
        
        CL.clReleaseProgram(program_id)