from __future__ import annotations

from ctypes import c_void_p, pointer, c_size_t
from typing import TYPE_CHECKING, Optional, List, Tuple

import numpy as np
import math

from ..runtime import (
    cl_mem_flags,
    cl_mem,
    CL,
    cl_int,
    cl_event,
    cl_image_format,
    cl_image_desc,
    cl_channel_order,
    cl_channel_type,
    cl_mem_object_type
)
from .command_queue import CommandQueue
from .event import Event

if TYPE_CHECKING:
    from .context import Context

from .mem import Mem


class image2d(Mem):

    def __init__(self, context:Context, data:Optional[np.ndarray]=None, shape:Optional[Tuple[int,...]]=None, dtype:Optional[type]=None, flags:Optional[cl_mem_flags]=None):
        data_is_valid = (data is not None)
        if data_is_valid:
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)

            host_ptr = data.ctypes.data_as(c_void_p)
            shape = data.shape
            dtype = data.dtype
            size = data.nbytes
        else:
            data = None
            host_ptr = None
            size = math.prod(shape) * np.dtype(dtype).itemsize

        if flags is None:
            flags = cl_mem_flags.CL_MEM_READ_WRITE

        if len(shape) == 2:
            self._channel_order = cl_channel_order.CL_LUMINANCE
        elif len(shape) == 3:
            if shape[2] == 1:
                self._channel_order = cl_channel_order.CL_LUMINANCE
            elif shape[2] == 2:
                self._channel_order = cl_channel_order.CL_RG
            elif shape[2] == 3:
                self._channel_order = cl_channel_order.CL_RGB
            elif shape[2] == 4:
                self._channel_order == cl_channel_order.CL_RGBA
            else:
                raise ValueError(f"image shape {shape} is not supported")
        else:
            raise ValueError(f"image shape {shape} is not supported")
        
        if dtype == np.int8:
            self._channel_type = cl_channel_type.CL_SIGNED_INT8
        elif dtype == np.uint8:
            self._channel_type = cl_channel_type.CL_UNSIGNED_INT8
        elif dtype == np.int16:
            self._channel_type = cl_channel_type.CL_SIGNED_INT16
        elif dtype == np.uint16:
            self._channel_type = cl_channel_type.CL_UNSIGNED_INT16
        elif dtype == np.int32:
            self._channel_type = cl_channel_type.CL_SIGNED_INT32
        elif dtype == np.uint32:
            self._channel_type = cl_channel_type.CL_UNSIGNED_INT32
        elif dtype == np.float16:
            self._channel_type = cl_channel_type.CL_HALF_FLOAT
        elif dtype == np.float32:
            self._channel_type = cl_channel_type.CL_FLOAT
        else:
            raise ValueError("channel type is not supported")
        
        self._shape = shape
        self._dtype = dtype

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

        self._format:cl_image_format = cl_image_format(self._channel_order, self._channel_type)
        self._desc:cl_image_desc = cl_image_desc(cl_mem_object_type.CL_MEM_OBJECT_IMAGE2D, shape[1], shape[0], 0, 0, 0, 0, 0, 0)
        error_code = cl_int(0)
        image_id:cl_mem = CL.clCreateImage(context.id, flags, pointer(self._format), pointer(self._desc), host_ptr, pointer(error_code))
        Mem.__init__(self, context, image_id, data, host_ptr, size, flags)

    def write(self, cmd_queue:CommandQueue, origin:Tuple[int, int]=(0, 0), region:Tuple[int, int]=(0, 0), host_ptr:c_void_p=None, after_events:List[Event]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as image2d_t")
        
        if region == (0, 0):
            region = self._shape

        if host_ptr is None:
            host_ptr = self._host_ptr

        if after_events is None:
            after_events = []

        ptr_origin = (c_size_t * 3)(origin[0], origin[1], 0)
        ptr_region = (c_size_t * 3)(region[0], region[1], 0)

        events_ptr = Event.events_ptr(after_events)
        event_id = cl_event(0)
        CL.clEnqueueWriteImage.record_call_stack()
        CL.clEnqueueWriteImage(cmd_queue.id, self.id, False, ptr_origin, ptr_region, 0, 0, host_ptr, len(after_events), events_ptr, pointer(event_id))
        return Event(self.context, event_id, CL.clEnqueueWriteImage)

    def read(self, cmd_queue:CommandQueue, origin:Tuple[int, int]=(0, 0), region:Tuple[int, int]=(0, 0), host_ptr:c_void_p=None, after_events:List[Event]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as buffer")
        
        if region == (0, 0):
            region = self._shape

        if host_ptr is None:
            host_ptr = self._host_ptr

        if after_events is None:
            after_events = []

        ptr_origin = (c_size_t * 3)(origin[0], origin[1], 0)
        ptr_region = (c_size_t * 3)(region[0], region[1], 0)

        events_ptr = Event.events_ptr(after_events)
        event_id = cl_event(0)
        CL.clEnqueueReadImage.record_call_stack()
        CL.clEnqueueReadImage(cmd_queue.id, self.id, True, ptr_origin, ptr_region, 0, 0, host_ptr, len(after_events), events_ptr, pointer(event_id))
        return Event(self.context, event_id, CL.clEnqueueReadImage)

    def set_data(self, cmd_queue:CommandQueue, data:np.ndarray, after_events:List[Event]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as buffer")
        
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        host_ptr = data.ctypes.data_as(c_void_p)
        shape = data.shape
        dtype = data.dtype

        if shape != self._shape:
            raise ValueError("data shape is not equal to image shape")
        
        if dtype != self._dtype:
            raise ValueError("data dtype is not equal to image dtype")

        event = self.write(cmd_queue, (0, 0), (0, 0), host_ptr, after_events)
        self._host_ptr = host_ptr
        self._data = data
        return event
    
    @property
    def format(self)->cl_image_format:
        return self._format
    
    @property
    def desc(self)->cl_image_desc:
        return self._desc
    
    @property
    def channel_order(self)->cl_channel_order:
        return self._channel_order
    
    @property
    def channel_type(self)->cl_channel_type:
        return self._channel_type
    
    @property
    def shape(self)->Tuple[int, int, int]:
        return self._shape
    
    @property
    def dtype(self)->np.dtype:
        return self._dtype