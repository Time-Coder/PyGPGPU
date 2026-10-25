from __future__ import annotations
from ctypes import pointer, c_float
from typing import override

from .hipobject import HIPObject
from ..driver import CUevent, HIP, hipError


class Event(HIPObject):

    def __init__(self, flags:int=0):
        self._flags:int = flags

        event_id = CUevent()
        ptr_event_id = pointer(event_id)
        HIP.hipEventCreate(ptr_event_id)
        HIPObject.__init__(self, event_id)

    @property
    def flags(self)->int:
        return self._flags

    def sync(self):
        HIP.hipEventSynchronize(self.id)

    def wait(self)->None:
        self.sync()

    def msecs_to(self, event:"Event")->float:
        msecs = c_float(0)
        HIP.hipEventElapsedTime(pointer(msecs), event.id, self.id)
        return msecs.value

    @property
    def status(self)->int:
        return int(HIP.hipEventQuery(self.id))

    @property
    def finished(self)->bool:
        return (self.status == hipError.SUCCESS)

    @override
    @staticmethod
    def _prefix()->str:
        return ""

    @override
    @staticmethod
    def _get_info_func():
        return None

    @override
    @staticmethod
    def _info_enum()->type|None:
        return None

    @override
    @staticmethod
    def _release_func():
        return HIP.hipEventDestroy
