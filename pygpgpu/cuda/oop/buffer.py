from __future__ import annotations

from ctypes import c_void_p, pointer
import ctypes
from typing import Optional, Tuple, Union
from typing_extensions import override

import numpy as np

from ..driver import CUDA, CUdeviceptr
from .stream import Stream
from .event import Event
from .mem_object import MemObject


class Buffer(MemObject):

    def __init__(self, size:int):
        buffer_id = CUdeviceptr(0)
        ptr_buffer_id = pointer(buffer_id)
        CUDA.cuMemAlloc_v2(ptr_buffer_id, size)
        MemObject.__init__(self, buffer_id, None, None, size)

    @override
    def write(self, stream:Stream, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:Optional[c_void_p]=None)->Event:        
        if region is None:
            region = self._size

        if host_ptr is None:
            host_ptr = self._host_ptr

        CUDA.cuMemcpyHtoDAsync_v2(CUdeviceptr(self.id.value + self._byte_offset(origin)), host_ptr, region, stream.id)
        return stream.record_event()

    @override
    def read(self, stream:Stream, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:Optional[c_void_p]=None)->Event:        
        if region is None:
            region = self._size

        if host_ptr is None:
            host_ptr = self._host_ptr

        CUDA.cuMemcpyDtoHAsync_v2(host_ptr, CUdeviceptr(self.id.value + self._byte_offset(origin)), region, stream.id)
        return stream.record_event()

    def _host_pointer_of(self, data:Union[bytes, bytearray, np.ndarray])->Tuple[ctypes.c_void_p, np.ndarray]:
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
        skipping the transfer is what keeps a skipped upload from turning into a
        `CUDA_ERROR_INVALID_VALUE` at read-back time.
        """
        self._host_ptr, self._data = self._host_pointer_of(data)

    @override
    def set_data(self, stream:Stream, data:Union[bytes, bytearray, np.ndarray])->Optional[Event]:
        host_ptr, data = self._host_pointer_of(data)
        self._host_ptr = host_ptr
        self._data = data

        # The caller hands over data the host owns (a fresh argument upload, or a gnp array
        # whose dirty_on_host flag the caller already checked), so it is always copied in:
        # gating on Readed here dropped a host-side write to a write-only argument, and the
        # device then saw whatever the buffer pool had left in it.
        return self.write(stream, 0, self._size, host_ptr)
        
