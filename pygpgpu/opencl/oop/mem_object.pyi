from __future__ import annotations

from ctypes import c_void_p
from typing import TYPE_CHECKING, List, Union, Tuple, Optional

if TYPE_CHECKING:
    from .context import Context

from ..runtime import (
    cl_mem_flags,
    cl_mem,
    cl_mem_object_type,
    cl_mem_properties
)
from .clobject import CLObject
from .command_queue import CommandQueue
from .event import Event
import numpy as np


class MemObject(CLObject):

    def __init__(self, context:Context, mem_id:cl_mem, data:Union[bytes, bytearray, np.ndarray, None], host_ptr:c_void_p, size:int, flags:cl_mem_flags): ...

    @property
    def context(self)->Context: ...

    @property
    def data(self)->Union[bytes, bytearray, np.ndarray, None]: ...
    
    @property
    def flags(self)->cl_mem_flags: ...

    @property
    def host_ptr(self)->c_void_p: ...

    @property
    def size(self)->int: ...

    @property
    def dirty_on_device(self)->bool: ...
    
    @dirty_on_device.setter
    def dirty_on_device(self, dirty:bool)->None: ...
    
    @property
    def dirty_on_host(self)->bool: ...
    
    @property
    def device_dirty_time(self)->float: ...

    @property
    def using(self)->bool: ...

    def use(self)->None: ...

    def unuse(self)->None: ...
    
    def __len__(self)->int: ...
    
    @property
    def type(self)->cl_mem_object_type: ...

    @property
    def flags(self)->cl_mem_flags: ...

    @property
    def map_count(self)->int: ...

    @property
    def reference_count(self)->int: ...

    @property
    def associated_memobject(self)->cl_mem: ...

    @property
    def offset(self)->int: ...

    @property
    def uses_svm_pointer(self)->bool: ...

    @property
    def properties(self)->List[cl_mem_properties]: ...

    @property
    def dx9_media_adapter_type(self)->int: ...

    @property
    def d3d10_resource(self)->c_void_p: ...

    @property
    def d3d11_resource(self)->c_void_p: ...

    def write(self, cmd_queue:CommandQueue, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:c_void_p=None, after_events:List[Event]=None)->Event: ...

    def read(self, cmd_queue:CommandQueue, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:c_void_p=None, after_events:List[Event]=None)->Event: ...

    def set_data(self, cmd_queue:CommandQueue, data:Union[bytes, bytearray, np.ndarray], after_events:List[Event]=None)->Event: ...