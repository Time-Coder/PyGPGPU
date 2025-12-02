from __future__ import annotations
from ctypes import pointer, c_float
from typing import override

from .cuobject import CUObject

from ..driver import CUevent, CUDA, CUevent_flags


class Event(CUObject):

    def __init__(self, flags:CUevent_flags=CUevent_flags.CU_EVENT_DEFAULT):
        self._flags:CUevent_flags = flags

        event_id = CUevent()
        ptr_event_id = pointer(event_id)
        CUDA.cuEventCreate(ptr_event_id, flags)
        CUObject.__init__(self, event_id)

    @property
    def flags(self)->CUevent_flags:
        return self._flags
    
    def sync(self):
        CUDA.cuEventSynchronize(self.id)

    def msecs_to(self, event:Event)->float:
        msecs = c_float()
        ptr_msecs = pointer(msecs)
        CUDA.cuEventElapsedTime(ptr_msecs, self.id, event.id)
        return msecs.value

    @staticmethod
    @override
    def _prefix()->str:
        return ""

    @staticmethod
    @override
    def _get_info_func()->CUDA.Func:
        return None

    @staticmethod
    @override
    def _info_enum()->type:
        return None

    @staticmethod
    @override
    def _release_func()->CUDA.Func:
        return CUDA.cuEventDestroy