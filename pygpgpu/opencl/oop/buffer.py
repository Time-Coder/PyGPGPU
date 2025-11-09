from __future__ import annotations

from ctypes import c_void_p, pointer
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .context import Context

from ..runtime import (
    cl_mem_flags,
    cl_mem,
    CL,
    cl_int,
    CLInfo,
    IntEnum,
    cl_mem_info
)
from .clobject import CLObject


class Buffer(CLObject):

    def __init__(self, context:Context, flags:cl_mem_flags, size:int, host_ptr:c_void_p):
        error_code = cl_int(0)
        buffer_id:cl_mem = CL.clCreateBuffer(context.id, flags, size, host_ptr, pointer(error_code))

        self._context:Context = context
        self._flags:cl_mem_flags = flags
        self._size:int = size
        self._host_ptr:c_void_p = host_ptr
        CLObject.__init__(self, buffer_id)

    @property
    def context(self)->Context:
        return self._context
    
    @property
    def flags(self)->cl_mem_flags:
        return self._flags

    @property
    def size(self)->int:
        return self._size
    
    @property
    def host_ptr(self)->c_void_p:
        return self._host_ptr

    def __len__(self)->int:
        return self._size

    @property
    def _prefix(self)->str:
        return "CL_MEM"

    @property
    def _get_info_func(self)->CL.Func:
        return CL.clGetMemObjectInfo

    @property
    def _info_types_map(self)->Dict[IntEnum, type]:
        return CLInfo.mem_info_types

    @property
    def _info_enum(self)->type:
        return cl_mem_info

    @staticmethod
    def _release(buffer_id):
        if not buffer_id:
            return
        
        CL.clReleaseMemObject(buffer_id)