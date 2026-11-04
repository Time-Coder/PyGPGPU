from __future__ import annotations

from ctypes import c_void_p, pointer
import ctypes
from typing import Any, List, Optional, TYPE_CHECKING, Tuple, Union
from typing_extensions import override

import numpy as np

from ..driver import (
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

    def __init__(self, context:Context, data:Union[bytes, bytearray, np.ndarray, None]=None, size:int=0, flags:cl_mem_flags=cl_mem_flags.CL_MEM_READ_WRITE):
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
    def write(self, cmd_queue:CommandQueue, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:Optional[c_void_p]=None, after_events:Optional[List[Event]]=None)->Event:
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
    def read(self, cmd_queue:CommandQueue, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:Optional[c_void_p]=None, after_events:Optional[List[Event]]=None)->Event:
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
        # A non-blocking read, like the write above: the caller waits on the returned
        # event (Event.wait) or registers a completion callback. A *blocking* read would
        # block until the queue drained, which deadlocks whenever the enqueue happens
        # inside a driver event callback -- and that is exactly where a deferred read-back
        # runs (see Kernel.__after_call).
        CL.clEnqueueReadBuffer(cmd_queue.id, self.id, False, origin, region, host_ptr, len(after_events), events_ptr, pointer(event_id))
        return Event(self.context, event_id, CL.clEnqueueReadBuffer)

    def _host_pointer_of(self, data:Union[bytes, bytearray, np.ndarray])->Tuple[ctypes.c_void_p, Any]:
        """The host pointer and the (contiguous) data a copy would read from."""
        if isinstance(data, bytes):
            size: int = len(data)
            host_ptr = (ctypes.c_ubyte * size).from_buffer_copy(data)
        elif isinstance(data, bytearray):
            size = len(data)
            host_ptr = (ctypes.c_ubyte * size).from_buffer(data)
        else:
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)
            host_ptr = data.ctypes.data_as(c_void_p)
            size = data.nbytes
        if size != self._size:
            raise ValueError("data size is not equal to buffer size")
        return host_ptr, data

    def bind_data(self, data:Union[bytes, bytearray, np.ndarray])->None:
        """Remember the caller's data **without** copying it to the device.

        Some arguments are never read by the kernel, so their initial contents do not have
        to travel -- but the read-back still needs a host destination, and `read` takes it
        from `_host_ptr`, which only the copy path used to set. Binding the pointer while
        skipping the transfer is what keeps a skipped upload from turning into a bad
        device-to-host copy at read-back time.
        """
        self._host_ptr, self._data = self._host_pointer_of(data)

    @override
    def set_data(self, cmd_queue:CommandQueue, data:Union[bytes, bytearray, np.ndarray], after_events:Optional[List[Event]]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as buffer")

        host_ptr, data = self._host_pointer_of(data)
        self._host_ptr = host_ptr
        self._data = data

        event = None
        # Either direction needs the real contents (a read uses them; a read-back copies the
        # whole buffer home, so an upload that was skipped would come back as zeros in every
        # element the kernel left alone). Only an argument with neither flag skips the copy,
        # and that decision is made before this method is called (see need_upload).
        if not (self.flags & cl_mem_flags.CL_MEM_WRITE_ONLY) and self.kernel_flags:
            event = self.write(cmd_queue, 0, self._size, host_ptr, after_events)
        
        return event
