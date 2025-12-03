from ctypes import pointer
from typing import override

from .cuobject import CUObject
from .event import Event
from ..driver import CUevent, CUDA, CUstream_flags, CUevent_wait_flags


class Stream(CUObject):

    def __init__(self, flags:CUstream_flags=CUstream_flags.CU_STREAM_DEFAULT):
        self._flags:CUstream_flags = flags

        stream_id = CUevent()
        ptr_stream_id = pointer(stream_id)
        CUDA.cuStreamCreate(ptr_stream_id, flags)
        CUObject.__init__(self, stream_id)

    @property
    def flags(self)->CUstream_flags:
        return self._flags

    def record_event(self)->Event:
        event = Event()
        CUDA.cuEventRecord(event.id, self.id)
        return event
    
    def wait_event(self, event:Event, flags:CUevent_wait_flags=CUevent_wait_flags.CU_EVENT_WAIT_DEFAULT):
        CUDA.cuStreamWaitEvent(self.id, event.id, flags)

    def sync(self)->None:
        CUDA.cuStreamSynchronize(self.id)

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
        return CUDA.cuStreamDestroy