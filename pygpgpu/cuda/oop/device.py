from ctypes import create_string_buffer, c_size_t, pointer
from typing import override, Optional
from uuid import UUID

from .cuobject import CUObject
from .context import Context
from ..driver import (
    CUDA,
    CUdevice_attribute,
    CUuuid,
    CUcontext,
    ptr_CUcontext,
    CUctx_flags
)


class Device(CUObject):

    def __init__(self, device_id:int)->None:
        CUObject.__init__(self, device_id)
        self._name:str = ""
        self._total_memory:int = 0
        self._uuid:Optional[UUID] = None
        self._default_context:Optional[Context] = None

    @property
    def name(self)->str:
        if not self._name:
            name_buf = create_string_buffer(256)
            CUDA.cuDeviceGetName(name_buf, 256, self.id)
            self._name = name_buf.value.decode('utf-8')

        return self._name
    
    @property
    def total_memory(self)->int:
        if not self._total_memory:
            nbytes = c_size_t()
            ptr_nbytes = pointer(nbytes)
            CUDA.cuDeviceTotalMem(ptr_nbytes, self.id)
            self._total_memory = nbytes.value
        return self._total_memory
    
    @property
    def uuid(self)->UUID:
        if not self._uuid:
            result = CUuuid()
            ptr_result = pointer(result)
            CUDA.cuDeviceGetUuid(ptr_result, self.id)
            self._uuid = UUID(bytes=bytes(result))

        return self._uuid
    
    def create_context(self, flags:CUctx_flags=CUctx_flags.CU_CTX_SCHED_AUTO)->Context:
        return Context(self, flags)
    
    @property
    def default_context(self)->Context:
        if not self._default_context:
            self._default_context = Context(self)

        return self._default_context
    
    @staticmethod
    @override
    def _prefix()->str:
        return "CU_DEVICE_ATTRIBUTE"

    @staticmethod
    @override
    def _get_info_func()->CUDA.Func:
        return CUDA.cuDeviceGetAttribute

    @staticmethod
    @override
    def _info_enum()->type:
        return CUdevice_attribute

    @staticmethod
    @override
    def _release_func()->CUDA.Func:
        return None