from __future__ import annotations

from ctypes import c_void_p, pointer, c_ubyte
import ctypes
from typing import TYPE_CHECKING, Dict, Union, Optional, List

import numpy as np

from ..runtime import (
    cl_mem_flags,
    cl_mem,
    CL,
    cl_int,
    CLInfo,
    IntEnum,
    cl_mem_info,
    cl_event
)
from .clobject import CLObject
from .command_queue import CommandQueue
from .event import Event

if TYPE_CHECKING:
    from .context import Context


class Buffer(CLObject):

    def __init__(self, context:Context, data_or_size:Union[bytes, bytearray, np.ndarray, int], flags:Optional[cl_mem_flags]=None):
        
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
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)

            host_ptr = data.ctypes.data_as(c_void_p)
            size = data.nbytes

        if flags is None:
            flags = cl_mem_flags.CL_MEM_READ_WRITE

        buffer_id:cl_mem = CL.clCreateBuffer(context.id, flags, size, host_ptr, pointer(error_code))
        self._context:Context = context
        self._flags:cl_mem_flags = flags
        self._size:int = size
        self._host_ptr:c_void_p = host_ptr
        self._data:Union[bytes, bytearray, np.ndarray, None] = data
        CLObject.__init__(self, buffer_id)

    def write(self, cmd_queue:CommandQueue, offset:int=0, size:int=None, host_ptr:c_void_p=None, after_events:List[Event]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as buffer")
        
        if size is None:
            size = self._size

        if host_ptr is None:
            host_ptr = self._host_ptr

        if after_events is None:
            after_events = []

        wait_list = Event.wait_list(after_events)
        event_id = cl_event(0)
        CL.clEnqueueWriteBuffer.record_call_stack()
        CL.clEnqueueWriteBuffer(cmd_queue.id, self.id, False, offset, size, host_ptr, len(after_events), wait_list, pointer(event_id))
        return Event(self.context, event_id, CL.clEnqueueWriteBuffer)

    def read(self, cmd_queue:CommandQueue, offset:int=0, size:int=None, host_ptr:c_void_p=None, after_events:List[Event]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as buffer")
        
        if size is None:
            size = self._size

        if host_ptr is None:
            host_ptr = self._host_ptr

        if after_events is None:
            after_events = []

        wait_list = Event.wait_list(after_events)
        event_id = cl_event(0)
        CL.clEnqueueReadBuffer.record_call_stack()
        CL.clEnqueueReadBuffer(cmd_queue.id, self.id, True, offset, size, host_ptr, len(after_events), wait_list, pointer(event_id))
        return Event(self.context, event_id, CL.clEnqueueReadBuffer)

    def set_data(self, cmd_queue:CommandQueue, data:Union[bytes, bytearray, np.ndarray], after_events:List[Event]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as buffer")
        
        if isinstance(data, bytes):
            size = len(data)
            host_ptr = (ctypes.c_ubyte * size).from_buffer_copy(data)
        elif isinstance(data, bytearray):
            size = len(data)
            host_ptr = (ctypes.c_ubyte * size).from_buffer(data)
        elif isinstance(data, np.ndarray):
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)

            host_ptr = data.ctypes.data_as(c_void_p)
            size = data.nbytes

        if size != self._size:
            raise ValueError("data size is not equal to buffer size")

        event = self.write(cmd_queue, 0, size, host_ptr, after_events)
        self._host_ptr = host_ptr
        self._data = data
        return event

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

    @staticmethod
    def _prefix()->str:
        return "CL_MEM"

    @staticmethod
    def _get_info_func()->CL.Func:
        return CL.clGetMemObjectInfo

    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]:
        return CLInfo.mem_info_types

    @staticmethod
    def _info_enum()->type:
        return cl_mem_info

    @staticmethod
    def _release_func()->CL.Func:
        return CL.clReleaseMemObject