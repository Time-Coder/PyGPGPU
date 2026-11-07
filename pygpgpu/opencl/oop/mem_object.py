from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Union, Tuple, Optional, List
from typing_extensions import override
from ctypes import c_void_p
from abc import abstractmethod
import threading
import time

from ..driver import (
    cl_mem,
    CL,
    CLInfo,
    IntEnum,
    cl_mem_info,
    cl_mem_flags
)
from .clobject import CLObject

if TYPE_CHECKING:
    from .context import Context
    from .command_queue import CommandQueue
    from .event import Event

import numpy as np

from ...constants import HostPointer


class MemObject(CLObject):

    def __init__(self, context:Context, mem_id:cl_mem, data:Union[bytes, bytearray, np.ndarray, None], host_ptr:HostPointer, size:int, flags:cl_mem_flags):
        self._context:Context = context
        self._flags:cl_mem_flags = flags
        self._kernel_flags:cl_mem_flags = cl_mem_flags.CL_MEM_READ_WRITE
        self._data:Union[bytes, bytearray, np.ndarray, None] = data
        self._host_ptr:HostPointer = host_ptr
        self._size:int = size
        self._using:bool = False
        # Guards the claim above. A pooled buffer is handed out under the context's lock,
        # but it is handed *back* from a driver callback, which holds no such lock -- and
        # that callback can still be queued while the buffer has already been handed out
        # again (see unuse).
        self._claim_lock:threading.Lock = threading.Lock()
        # The claim's serial, handed out by `use` and required by `unuse`. A release that
        # arrives holding a serial the buffer no longer carries is stale -- its own claim
        # has already been returned -- and must not touch the holder's claim, whoever
        # that is.
        self._claim_serial:int = 0
        self._dirty_on_device:bool = False
        self._dirty_on_host:bool = False
        self._device_dirty_time:float = 0.0
        CLObject.__init__(self, mem_id)

    @abstractmethod
    def write(self, cmd_queue:CommandQueue, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:Optional[c_void_p]=None, after_events:Optional[List[Event]]=None)->Event:
        pass

    @abstractmethod
    def read(self, cmd_queue:CommandQueue, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:Optional[c_void_p]=None, after_events:Optional[List[Event]]=None)->Event:
        pass

    @abstractmethod
    def set_data(self, cmd_queue:CommandQueue, data:Union[bytes, bytearray, np.ndarray], after_events:Optional[List[Event]]=None)->Event:
        pass

    @property
    def context(self)->Context:
        return self._context
    
    @property
    def data(self)->Union[bytes, bytearray, np.ndarray, None]:
        return self._data
    
    @property
    def flags(self)->cl_mem_flags:
        return self._flags
    
    @property
    def kernel_flags(self)->cl_mem_flags:
        return self._kernel_flags

    @kernel_flags.setter
    def kernel_flags(self, flags:cl_mem_flags)->None:
        self._kernel_flags = flags

    @property
    def claim_serial(self)->int:
        """The serial of the claim currently held, for a release that validates it.

        Read under the claim lock: the release callback captures the serial it was
        registered for, and a driver thread can be handing a new claim out while the
        capture runs.
        """
        with self._claim_lock:
            return self._claim_serial

    @property
    def host_ptr(self)->HostPointer:
        return self._host_ptr
    
    @property
    def size(self)->int:
        return self._size

    @property
    def using(self)->bool:
        return self._using
    
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
    
    def use(self)->int:
        """Claim the buffer and return the claim's serial for its release."""
        with self._claim_lock:
            if self._using:
                raise RuntimeError("mem obj is still in using")

            self._using = True
            self._claim_serial += 1
            return self._claim_serial

    def unuse(self, claim_serial:Optional[int]=None)->None:
        """Return the claim made by `use`, identified by its serial.

        Two paths release a buffer -- the argument's own `unuse()` after its read-back,
        and the launch event's release callback for an argument that is not read back --
        and both are registered on events whose order the driver decides. A rapid caller
        can hand the same pooled buffer out again while one of them is still queued, so a
        queued release fires after a NEW claim exists. The serial settles it: a release
        whose serial is not the current holder's belongs to an older claim that has
        already been returned, and returning it again would free the new holder's claim
        -- handing the buffer out twice. So a mismatched (or absent) serial is a no-op,
        not an error; the double claim itself still refuses, because two users of one
        buffer is a real bug, while a stale release is an artifact of when the driver
        reports completion.
        """
        with self._claim_lock:
            if not self._using:
                return
            if claim_serial is not None and claim_serial != self._claim_serial:
                return

            self._using = False
    
    def __len__(self)->int:
        return self._size

    @override
    @staticmethod
    def _prefix()->str:
        return "CL_MEM"

    @override
    @staticmethod
    def _get_info_func()->object:
        return CL.clGetMemObjectInfo

    @override
    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]|None:
        return CLInfo.mem_info_types

    @override
    @staticmethod
    def _info_enum()->type|None:
        return cl_mem_info

    @override
    @staticmethod
    def _release_func()->object:
        return CL.clReleaseMemObject
