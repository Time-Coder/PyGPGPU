from __future__ import annotations

from typing import TYPE_CHECKING, Union, Tuple, Optional, List
from ctypes import c_void_p
from abc import abstractmethod

from ..driver import (
    CUdeviceptr
)
from .cuobject import CUObject

if TYPE_CHECKING:
    from .stream import Stream
    from .event import Event

import numpy as np


class MemObject(CUObject):

    def __init__(self, mem_id:CUdeviceptr, data:Union[bytes, bytearray, np.ndarray, None], host_ptr:c_void_p, size:int): ...

    @abstractmethod
    def write(self, stream:Stream, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:c_void_p=None, after_events:List[Event]=None)->Event: ...

    @abstractmethod
    def read(self, stream:Stream, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:c_void_p=None, after_events:List[Event]=None)->Event: ...

    @abstractmethod
    def set_data(self, stream:Stream, data:Union[bytes, bytearray, np.ndarray], after_events:List[Event]=None)->Event: ...
    
    @property
    def data(self)->Union[bytes, bytearray, np.ndarray, None]: ...
    
    @property
    def host_ptr(self)->c_void_p: ...
    
    @property
    def size(self)->int: ...

    @property
    def using(self)->bool: ...
    
    @property
    def dirty_on_device(self)->bool: ...
    
    @dirty_on_device.setter
    def dirty_on_device(self, dirty:bool)->None: ...
    
    @property
    def dirty_on_host(self)->bool: ...
    
    @property
    def device_dirty_time(self)->float: ...
    
    @dirty_on_host.setter
    def dirty_on_host(self, dirty:bool)->None: ...
    
    def use(self)->None: ...

    def unuse(self)->None: ...
    
    def __len__(self)->int: ...
