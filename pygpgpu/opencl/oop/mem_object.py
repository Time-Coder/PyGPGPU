from __future__ import annotations

from typing import TYPE_CHECKING, Dict, override, Union, Tuple, Optional, List
from ctypes import c_void_p
from abc import abstractmethod
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


class MemObject(CLObject):

    def __init__(self, context:Context, mem_id:cl_mem, data:Union[bytes, bytearray, np.ndarray, None], host_ptr:c_void_p, size:int, flags:cl_mem_flags):
        self._context:Context = context
        self._flags:cl_mem_flags = flags
        self._kernel_flags:cl_mem_flags = cl_mem_flags.CL_MEM_READ_WRITE
        self._data:Union[bytes, bytearray, np.ndarray, None] = data
        self._host_ptr:c_void_p = host_ptr
        self._size:int = size
        self._using:bool = False
        self._dirty_on_device:bool = False
        self._dirty_on_host:bool = False
        self._device_dirty_time:float = 0.0
        CLObject.__init__(self, mem_id)

    @abstractmethod
    def write(self, cmd_queue:CommandQueue, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:c_void_p=None, after_events:List[Event]=None)->Event:
        pass

    @abstractmethod
    def read(self, cmd_queue:CommandQueue, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:c_void_p=None, after_events:List[Event]=None)->Event:
        pass

    @abstractmethod
    def set_data(self, cmd_queue:CommandQueue, data:Union[bytes, bytearray, np.ndarray], after_events:List[Event]=None)->Event:
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
    def host_ptr(self)->c_void_p:
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
        return "CL_MEM"

    @override
    @staticmethod
    def _get_info_func()->CL.Func:
        return CL.clGetMemObjectInfo

    @override
    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]:
        return CLInfo.mem_info_types

    @override
    @staticmethod
    def _info_enum()->type:
        return cl_mem_info

    @override
    @staticmethod
    def _release_func()->CL.Func:
        return CL.clReleaseMemObject