from __future__ import annotations

from ctypes import c_void_p, pointer
import ctypes
from typing import Union, Optional, override

import numpy as np

from ..driver import CUDA, CUdeviceptr
from .stream import Stream
from .event import Event
from .mem_object import MemObject, DetectFlags


class Buffer(MemObject):

    def __init__(self, size:int):
        buffer_id = CUdeviceptr(0)
        ptr_buffer_id = pointer(buffer_id)
        CUDA.cuMemAlloc(ptr_buffer_id, size)
        MemObject.__init__(self, buffer_id, None, None, size)

    @override
    def write(self, stream:Stream, origin:int=0, region:Optional[int]=None, host_ptr:c_void_p=None)->Event:        
        if region is None:
            region = self._size

        if host_ptr is None:
            host_ptr = self._host_ptr

        CUDA.cuMemcpyHtoDAsync(self.id + origin, host_ptr, region, stream.id)
        return stream.record_event()

    @override
    def read(self, stream:Stream, origin:int=0, region:Optional[int]=None, host_ptr:c_void_p=None)->Event:        
        if region is None:
            region = self._size

        if host_ptr is None:
            host_ptr = self._host_ptr

        CUDA.cuMemcpyDtoHAsync(host_ptr, self.id + origin, region, stream.id)
        return stream.record_event()

    @override
    def set_data(self, stream:Stream, data:Union[bytes, bytearray, np.ndarray])->Optional[Event]:        
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

        self._host_ptr = host_ptr
        self._data = data

        event = None
        if self.kernel_flags != DetectFlags.Writable:
            event = self.write(stream, 0, size, host_ptr)

        return event
        