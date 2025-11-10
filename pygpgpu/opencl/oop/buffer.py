from __future__ import annotations

from ctypes import c_void_p, pointer, c_ubyte
import ctypes
from typing import TYPE_CHECKING, Dict, Union

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
from .command_queue import CommandQueue


class Buffer(CLObject):

    def __init__(self, context:Context, flags:cl_mem_flags, data:Union[bytes, int]):
        error_code = cl_int(0)
        size = len(data) if isinstance(data, bytes) else data
        host_ptr = ctypes.cast(data, ctypes.c_void_p) if isinstance(data, bytes) else None
        buffer_id:cl_mem = CL.clCreateBuffer(context.id, flags, size, host_ptr, pointer(error_code))

        self._context:Context = context
        self._flags:cl_mem_flags = flags
        self._size:int = size
        self._host_ptr:c_void_p = host_ptr
        self._data:bytes = data
        CLObject.__init__(self, buffer_id)

    def write(self, cmd_queue:CommandQueue, offset:int, data:bytes):
        size = len(data)
        ptr = ctypes.cast(data, ctypes.c_void_p)
        CL.clEnqueueWriteBuffer(cmd_queue.id, self.id, True, offset, size, ptr, 0, None, None)

    def read(self, cmd_queue:CommandQueue, offset:int, size:int)->bytes:
        result = (c_ubyte * size)()
        CL.clEnqueueReadBuffer(cmd_queue.id, self.id, True, offset, size, result, 0, None, None)
        return result.raw

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
    
    @property
    def data(self)->bytes:
        return self._data

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