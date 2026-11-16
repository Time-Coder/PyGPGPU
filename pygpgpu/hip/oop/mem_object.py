from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Dict, Union, Tuple, Optional
from typing_extensions import override
from ctypes import c_void_p
from abc import abstractmethod
import time
import threading

from ..driver import CUdeviceptr, HIP
from ...constants import HostPointer, KernelFlags
from .hipobject import HIPObject

if TYPE_CHECKING:
    from .stream import Stream
    from .event import Event

import numpy as np


class MemObject(HIPObject):

    def __init__(self, mem_id:CUdeviceptr, data:Union[bytes, bytearray, np.ndarray, None], host_ptr:HostPointer, size:int):
        self._data:Union[bytes, bytearray, np.ndarray, None] = data
        self._host_ptr:HostPointer = host_ptr
        self._size:int = size
        self._using:bool = False
        # The claim's serial and the lock that guards it: a release arrives from a driver
        # callback thread while the caller's next launch may already hold a new claim (see
        # `use`/`unuse`). Same contract as the OpenCL pool's.
        self._claim_serial:int = 0
        self._claim_lock:threading.Lock = threading.Lock()
        self._dirty_on_device:bool = False
        self._dirty_on_host:bool = False
        self._device_dirty_time:float = 0.0
        self._kernel_flags:KernelFlags = (KernelFlags.Readed | KernelFlags.Writed)
        HIPObject.__init__(self, mem_id)

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

        HIP buffers are 1-D allocations, so `origin` must be a flat element index. A
        per-dimension tuple has no meaningful flat byte address here; reject it instead of
        silently mis-addressing.
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
    
    def use(self)->int:
        """Claim the buffer and return the claim's serial for its release.

        The serial is what makes a *late* release harmless: a pooled buffer is handed back
        from a driver callback, and a rapid caller can have the same buffer handed out again
        before that callback fires. A release carrying the old serial belongs to a claim that
        has already ended, and returning it would free the new holder's claim. The OpenCL pool
        already carried this; HIP's `unuse()` raised from a driver thread instead.
        """
        with self._claim_lock:
            if self._using:
                raise RuntimeError("mem obj is still in using")

            self._using = True
            self._claim_serial += 1
            return self._claim_serial

    def unuse(self, claim_serial:Optional[int]=None)->None:
        """Return the claim made by `use`, identified by its serial.

        A stale serial -- or none, from the synchronous path -- is handled the way the
        OpenCL pool handles it: a mismatch is a no-op, because the claim it names has already
        been returned and the current holder must keep the buffer.
        """
        with self._claim_lock:
            if not self._using:
                return
            if claim_serial is not None and claim_serial != self._claim_serial:
                return

            self._using = False

    @property
    def claim_serial(self)->int:
        """The serial of the claim currently held, for a release that validates it."""
        with self._claim_lock:
            return self._claim_serial
    
    def __len__(self)->int:
        return self._size

    @override
    @staticmethod
    def _prefix()->str:
        return ""

    @override
    @staticmethod
    def _get_info_func()->object:
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
    def _release_func()->object:
        return HIP.hipFree
