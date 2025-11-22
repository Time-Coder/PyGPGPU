from __future__ import annotations

from typing import TYPE_CHECKING, Dict, override, Union, Tuple, Optional, List
from ctypes import c_void_p
from abc import abstractmethod

from ..runtime import (
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
        self._data:Union[bytes, bytearray, np.ndarray, None] = data
        self._host_ptr:c_void_p = host_ptr
        self._size:int = size
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
    def host_ptr(self)->c_void_p:
        return self._host_ptr
    
    @property
    def size(self)->int:
        return self._size
    
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