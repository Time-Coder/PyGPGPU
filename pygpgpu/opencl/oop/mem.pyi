from __future__ import annotations

from ctypes import c_void_p
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from .context import Context

from ..runtime import (
    cl_mem_flags,
    cl_mem,
    cl_mem_object_type,
    cl_mem_properties
)
from .clobject import CLObject
import numpy as np


class Mem(CLObject):

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
