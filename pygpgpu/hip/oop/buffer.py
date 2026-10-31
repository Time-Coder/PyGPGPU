from __future__ import annotations

from ctypes import c_void_p, pointer
import ctypes
from typing import Optional, Tuple, Union
from typing_extensions import override

import numpy as np

from ..driver import HIP, CUdeviceptr
from .stream import Stream
from .event import Event
from .mem_object import MemObject


class Buffer(MemObject):

    def __init__(self, size:int):
        buffer_id = CUdeviceptr(0)
        ptr_buffer_id = pointer(buffer_id)
        HIP.hipMalloc(ptr_buffer_id, size)
        MemObject.__init__(self, buffer_id, None, None, size)

    @override
    def write(self, stream:Stream, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:Optional[c_void_p]=None)->Event:        
        if region is None:
            region = self._size

        if host_ptr is None:
            host_ptr = self._host_ptr

        HIP.hipMemcpyHtoDAsync(CUdeviceptr(self.id.value + self._byte_offset(origin)), host_ptr, region, stream.id)
        return stream.record_event()

    @override
    def read(self, stream:Stream, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:Optional[c_void_p]=None)->Event:        
        if region is None:
            region = self._size

        if host_ptr is None:
            host_ptr = self._host_ptr

        HIP.hipMemcpyDtoHAsync(host_ptr, CUdeviceptr(self.id.value + self._byte_offset(origin)), region, stream.id)
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

        # The caller hands over data the host owns (a fresh argument upload, or a gnp array
        # whose dirty_on_host flag the caller already checked), so it is always copied in:
        # gating on Readed here dropped a host-side write to a write-only argument, and the
        # device then saw whatever the buffer pool had left in it.
        return self.write(stream, 0, size, host_ptr)
        
