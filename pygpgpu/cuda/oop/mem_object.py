from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Union, Tuple, Optional, Any
from typing_extensions import override
from ctypes import c_void_p
from abc import abstractmethod
import time

from ..driver import (
    CUdeviceptr,
    IntEnum,
    CUDA
)
from ...constants import HostPointer, KernelFlags
from .cuobject import CUObject

if TYPE_CHECKING:
    from .stream import Stream
    from .event import Event

import numpy as np


class MemObject(CUObject):

    def __init__(self, mem_id:CUdeviceptr, data:Union[bytes, bytearray, np.ndarray, None], host_ptr:HostPointer, size:int):
        self._data:Union[bytes, bytearray, np.ndarray, None] = data
        self._host_ptr:HostPointer = host_ptr
        self._size:int = size
        self._using:bool = False
        self._dirty_on_device:bool = False
        self._dirty_on_host:bool = False
        self._device_dirty_time:float = 0.0
        self._kernel_flags:KernelFlags = (KernelFlags.Readed | KernelFlags.Writed)
        CUObject.__init__(self, mem_id)

    @abstractmethod
    def write(self, stream:Stream, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:Optional[c_void_p]=None)->Event:
        pass

    @abstractmethod
    def read(self, stream:Stream, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:Optional[c_void_p]=None)->Event:
        pass

    @abstractmethod
    def set_data(self, stream:Stream, data:Union[bytes, bytearray, np.ndarray])->Event:
        pass
    
    @property
    def data(self)->Union[bytes, bytearray, np.ndarray, None]:
        return self._data
    
    @property
    def host_ptr(self)->HostPointer:
        return self._host_ptr
    
    @property
    def size(self)->int:
        return self._size

    @staticmethod
    def _byte_offset(origin:Union[int, Tuple[int, ...]])->int:
        """Return the byte offset for a flat element offset.

        CUDA/HIP buffers are 1-D allocations, so `origin` must be a flat element index.
        A per-dimension tuple (used by image reads/writes on the OpenCL side) has no
        meaningful flat byte address here; reject it instead of silently mis-addressing.
        """
        if isinstance(origin, tuple):
            raise TypeError(
                f"origin must be a flat element offset for this backend, got a tuple {origin}")
        return int(origin)

    @property
    def using(self)->bool:
        return self._using
    
    @property
    def flags(self)->int:
        # gnp._to_device tests the opencl CL_MEM_READ_ONLY bit to decide the device dirty flag;
        # CUDA device memory is always readable and writable, so returning 0 means "not read-only"
        return 0

    @property
    def dirty_on_device(self)->bool:
        return self._dirty_on_device
    
    @dirty_on_device.setter
    def dirty_on_device(self, dirty:bool)->None:
        self._dirty_on_device = dirty
        self._device_dirty_time = (time.time() if dirty else 0.0)
    
    @property
    def dirty_on_host(self)->bool:
        return self._dirty_on_host
    
    @property
    def device_dirty_time(self)->float:
        return self._device_dirty_time
    
    @dirty_on_host.setter
    def dirty_on_host(self, dirty:bool)->None:
        self._dirty_on_host = dirty

    @property
    def kernel_flags(self)->KernelFlags:
        return self._kernel_flags

    @kernel_flags.setter
    def kernel_flags(self, flags:KernelFlags)->None:
        self._kernel_flags = flags
    
    def use(self)->None:
        if self._using:
            raise RuntimeError("mem obj is still in using")

        self._using = True

    def unuse(self)->None:
        if not self._using:
            raise RuntimeError("mem obj is not in using")

        self._using = False
    
    def __len__(self)->int:
        return self._size

    @override
    @staticmethod
    def _prefix()->str:
        return ""

    @override
    @staticmethod
    def _get_info_func()->Any:
        return None

    @override
    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]|None:
        return None

    @override
    @staticmethod
    def _info_enum()->type|None:
        return None

    @override
    @staticmethod
    def _release_func()->Any:
        return CUDA.cuMemFree_v2
