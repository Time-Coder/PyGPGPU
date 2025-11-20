from __future__ import annotations

from ctypes import c_void_p, pointer
import ctypes
from typing import TYPE_CHECKING, Union, Optional, List, override

import numpy as np

from ..runtime import (
    cl_mem_flags,
    cl_mem,
    CL,
    cl_int,
    cl_event
)
from .command_queue import CommandQueue
from .event import Event

if TYPE_CHECKING:
    from .context import Context

from .mem_object import MemObject


class Buffer(MemObject):

    def __init__(self, context:Context, data:Union[bytes, bytearray, np.ndarray, None]=None, size:int=0, flags:Optional[cl_mem_flags]=None):
        data_is_valid = (data is not None)
        if data_is_valid:
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
        else:
            host_ptr = None

        if flags is None:
            flags = cl_mem_flags.CL_MEM_READ_WRITE

        if (
            data_is_valid and 
            not (flags & cl_mem_flags.CL_MEM_USE_HOST_PTR) and 
            not (flags & cl_mem_flags.CL_MEM_COPY_HOST_PTR)
        ):
            if size < 4*1024*1024:
                flags |= cl_mem_flags.CL_MEM_COPY_HOST_PTR
            else:
                flags |= cl_mem_flags.CL_MEM_USE_HOST_PTR

        error_code = cl_int(0)
        buffer_id:cl_mem = CL.clCreateBuffer(context.id, flags, size, host_ptr, pointer(error_code))
        MemObject.__init__(self, context, buffer_id, data, host_ptr, size, flags)

    @override
    def write(self, cmd_queue:CommandQueue, origin:int=0, region:Optional[int]=None, host_ptr:c_void_p=None, after_events:List[Event]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as buffer")
        
        if region is None:
            region = self._size

        if host_ptr is None:
            host_ptr = self._host_ptr

        if after_events is None:
            after_events = []

        events_ptr = Event.events_ptr(after_events)
        event_id = cl_event(0)
        CL.clEnqueueWriteBuffer(cmd_queue.id, self.id, False, origin, region, host_ptr, len(after_events), events_ptr, pointer(event_id))
        return Event(self.context, event_id, CL.clEnqueueWriteBuffer)

    @override
    def read(self, cmd_queue:CommandQueue, origin:int=0, region:Optional[int]=None, host_ptr:c_void_p=None, after_events:List[Event]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as buffer")
        
        if region is None:
            region = self._size

        if host_ptr is None:
            host_ptr = self._host_ptr

        if after_events is None:
            after_events = []

        events_ptr = Event.events_ptr(after_events)
        event_id = cl_event(0)
        CL.clEnqueueReadBuffer(cmd_queue.id, self.id, True, origin, region, host_ptr, len(after_events), events_ptr, pointer(event_id))
        return Event(self.context, event_id, CL.clEnqueueReadBuffer)

    @override
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
