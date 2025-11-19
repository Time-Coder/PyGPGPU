from __future__ import annotations

from ctypes import c_void_p, pointer, c_size_t
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from ..runtime import (
    cl_mem,
    CL,
    cl_int,
    cl_event,
    cl_image_format,
    cl_image_desc,
    cl_channel_order,
    cl_channel_type,
    cl_mem_flags
)
from .command_queue import CommandQueue
from .event import Event

if TYPE_CHECKING:
    from .context import Context

from .mem import Mem
from .image2d_t import image2d_t


class image2d(Mem):

    def __init__(self, context:Context, image2d_t_:image2d_t):
        self._image2d_t:image2d_t = image2d_t_
        error_code = cl_int(0)
        image_id:cl_mem = CL.clCreateImage(context.id, self._image2d_t.flags, pointer(self._image2d_t.format), pointer(self._image2d_t.desc), self._image2d_t.host_ptr, pointer(error_code))
        Mem.__init__(self, context, image_id, self._image2d_t.data, self._image2d_t.host_ptr, self._image2d_t.size, self._image2d_t.flags)

    def write(self, cmd_queue:CommandQueue, origin:Tuple[int, int]=(0, 0), region:Tuple[int, int]=(0, 0), host_ptr:c_void_p=None, after_events:List[Event]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as image2d_t")
        
        if region == (0, 0):
            region = self._image2d_t.shape

        if host_ptr is None:
            host_ptr = self._image2d_t.host_ptr

        if after_events is None:
            after_events = []

        ptr_origin = (c_size_t * 3)(origin[1], origin[0], 0)
        ptr_region = (c_size_t * 3)(region[1], region[0], 1)

        events_ptr = Event.events_ptr(after_events)
        event_id = cl_event(0)
        CL.clEnqueueWriteImage.record_call_stack()
        CL.clEnqueueWriteImage(cmd_queue.id, self.id, False, ptr_origin, ptr_region, 0, 0, host_ptr, len(after_events), events_ptr, pointer(event_id))
        return Event(self.context, event_id, CL.clEnqueueWriteImage)

    def read(self, cmd_queue:CommandQueue, origin:Tuple[int, int]=(0, 0), region:Tuple[int, int]=(0, 0), host_ptr:c_void_p=None, after_events:List[Event]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as buffer")
        
        if region == (0, 0):
            region = self.shape

        if host_ptr is None:
            host_ptr = self.host_ptr

        if after_events is None:
            after_events = []

        ptr_origin = (c_size_t * 3)(origin[1], origin[0], 0)
        ptr_region = (c_size_t * 3)(region[1], region[0], 1)

        events_ptr = Event.events_ptr(after_events)
        event_id = cl_event(0)
        CL.clEnqueueReadImage.record_call_stack()
        CL.clEnqueueReadImage(cmd_queue.id, self.id, False, ptr_origin, ptr_region, 0, 0, host_ptr, len(after_events), events_ptr, pointer(event_id))
        return Event(self.context, event_id, CL.clEnqueueReadImage)

    def set_data(self, cmd_queue:CommandQueue, data:np.ndarray, after_events:List[Event]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as buffer")
        
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        host_ptr = data.ctypes.data_as(c_void_p)
        shape = data.shape
        dtype = data.dtype

        if shape != self._image2d_t.shape:
            raise ValueError("data shape is not equal to image shape")
        
        if dtype != self._image2d_t.dtype:
            raise ValueError("data dtype is not equal to image dtype")

        self._host_ptr = host_ptr
        self._data = data
        self._image2d_t.data = data

        event = None
        if not (self.flags & cl_mem_flags.CL_MEM_WRITE_ONLY):
            event = self.write(cmd_queue, (0, 0), (0, 0), host_ptr, after_events)
        
        return event
    
    @property
    def format(self)->cl_image_format:
        return self._image2d_t.format
    
    @property
    def desc(self)->cl_image_desc:
        return self._image2d_t.desc
    
    @property
    def channel_order(self)->cl_channel_order:
        return self._image2d_t.channel_order
    
    @property
    def channel_type(self)->cl_channel_type:
        return self._image2d_t.channel_type
    
    @property
    def shape(self)->Tuple[int, int, int]:
        return self._image2d_t.shape
    
    @property
    def dtype(self)->np.dtype:
        return self._image2d_t.dtype