from __future__ import annotations

from ctypes import c_void_p
from typing import TYPE_CHECKING, List, Union, Optional
import numpy as np

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


class Buffer(CLObject):

    def __init__(self, context:Context, data_or_size:Union[bytes, bytearray, np.ndarray, int], flags:Optional[cl_mem_flags]=None): ...

    def write(self, cmd_queue:CommandQueue, offset:int, size:int, host_ptr:c_void_p, after_events:List[Event])->Event: ...

    def read(self, cmd_queue:CommandQueue, offset:int, size:int, host_ptr:c_void_p, after_events:List[Event])->Event: ...

    def set_data(self, cmd_queue:CommandQueue, data:Union[bytes, bytearray, np.ndarray], after_events:List[Event])->Event: ...

    @property
    def context(self)->Context: ...
    
    @property
    def flags(self)->cl_mem_flags: ...

    @property
    def size(self)->int: ...
    
    @property
    def host_ptr(self)->c_void_p: ...

    @property
    def data(self)->bytes: ...

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
