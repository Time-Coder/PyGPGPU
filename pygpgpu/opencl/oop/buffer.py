from __future__ import annotations

from ctypes import c_void_p, pointer, c_ubyte
import ctypes
from typing import TYPE_CHECKING, Dict, Union, Optional

import numpy as np

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

if TYPE_CHECKING:
    from .context import Context


class Buffer(CLObject):

    def __init__(self, context:Context, data_or_size:Union[bytes, bytearray, np.ndarray, int], flags:Optional[cl_mem_flags]=None, auto_share:bool=True):
        error_code = cl_int(0)

        data = data_or_size
        if isinstance(data_or_size, int):
            data = None
            host_ptr = None
            size = data_or_size
        elif isinstance(data, bytes):
            size = len(data)
            host_ptr = (ctypes.c_ubyte * size).from_buffer_copy(data)
        elif isinstance(data, bytearray):
            size = len(data)
            host_ptr = (ctypes.c_ubyte * size).from_buffer(data)
        elif isinstance(data, np.ndarray):
            used_data = data
            if not used_data.flags['C_CONTIGUOUS']:
                used_data = np.ascontiguousarray(used_data)

            host_ptr = used_data.ctypes.data_as(c_void_p)
            size = used_data.nbytes

        if flags is None:
            flags = cl_mem_flags.CL_MEM_READ_WRITE

        if auto_share:
            if size > 4*1024*1024:
                flags = (flags & ~cl_mem_flags.CL_MEM_COPY_HOST_PTR) | cl_mem_flags.CL_MEM_USE_HOST_PTR
            else:
                flags = (flags & ~cl_mem_flags.CL_MEM_USE_HOST_PTR) | cl_mem_flags.CL_MEM_COPY_HOST_PTR

        buffer_id:cl_mem = CL.clCreateBuffer(context.id, flags, size, host_ptr, pointer(error_code))

        self._context:Context = context
        self._flags:cl_mem_flags = flags
        self._size:int = size
        self._host_ptr:c_void_p = host_ptr
        self._data:Union[bytes, bytearray, np.ndarray, None] = data
        CLObject.__init__(self, buffer_id)

    def write(self, offset:int, data:bytes, cmd_queue:CommandQueue):
        size = len(data)
        ptr = ctypes.cast(data, ctypes.c_void_p)
        CL.clEnqueueWriteBuffer(cmd_queue.id, self.id, True, offset, size, ptr, 0, None, None)

    def read(self, offset:int, size:int, cmd_queue:CommandQueue)->bytes:
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
    def data(self)->Union[bytes, bytearray, np.ndarray, None]:
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