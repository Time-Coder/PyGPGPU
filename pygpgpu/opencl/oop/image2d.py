from __future__ import annotations

from ctypes import c_void_p, pointer, c_size_t
from typing import TYPE_CHECKING, List, Tuple, Optional, override

import numpy as np

from ..runtime import (
    CL,
    cl_event,
    cl_mem_flags,
    image2d_t
)
from .command_queue import CommandQueue
from .event import Event

if TYPE_CHECKING:
    from .context import Context

from .imagend import imagend


class image2d(imagend):

    def __init__(self, context:Context, image:image2d_t):
        imagend.__init__(self, context, image)

    @override
    def write(self, cmd_queue:CommandQueue, origin:Tuple[int, int]=(0, 0), region:Optional[Tuple[int, int]]=None, host_ptr:c_void_p=None, after_events:List[Event]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as current image2d")
        
        if region is None:
            region = self._image.shape

        if host_ptr is None:
            host_ptr = self._image.host_ptr

        if after_events is None:
            after_events = []

        ptr_origin = (c_size_t * 3)(origin[1], origin[0], 0)
        ptr_region = (c_size_t * 3)(region[1], region[0], 1)

        events_ptr = Event.events_ptr(after_events)
        event_id = cl_event(0)
        CL.clEnqueueWriteImage(cmd_queue.id, self.id, False, ptr_origin, ptr_region, 0, 0, host_ptr, len(after_events), events_ptr, pointer(event_id))
        return Event(self.context, event_id, CL.clEnqueueWriteImage)

    @override
    def read(self, cmd_queue:CommandQueue, origin:Tuple[int, int]=(0, 0), region:Optional[Tuple[int, int]]=None, host_ptr:c_void_p=None, after_events:List[Event]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as current image2d")
        
        if region is None:
            region = self._image.shape

        if host_ptr is None:
            host_ptr = self._image.host_ptr

        if after_events is None:
            after_events = []

        ptr_origin = (c_size_t * 3)(origin[1], origin[0], 0)
        ptr_region = (c_size_t * 3)(region[1], region[0], 1)

        events_ptr = Event.events_ptr(after_events)
        event_id = cl_event(0)
        CL.clEnqueueReadImage(cmd_queue.id, self.id, False, ptr_origin, ptr_region, 0, 0, host_ptr, len(after_events), events_ptr, pointer(event_id))
        return Event(self.context, event_id, CL.clEnqueueReadImage)

    @override
    def set_data(self, cmd_queue:CommandQueue, data:np.ndarray, after_events:List[Event]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as current image2d")
        
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        host_ptr = data.ctypes.data_as(c_void_p)
        shape = data.shape
        dtype = data.dtype

        if shape != self._image.shape:
            raise ValueError("data shape is not equal to image shape")
        
        if dtype != self._image.dtype:
            raise ValueError("data dtype is not equal to image dtype")

        self._host_ptr = host_ptr
        self._data = data
        self._image.data = data

        event = None
        if not (self.flags & cl_mem_flags.CL_MEM_WRITE_ONLY):
            event = self.write(cmd_queue, (0, 0), None, host_ptr, after_events)
        
        return event
    