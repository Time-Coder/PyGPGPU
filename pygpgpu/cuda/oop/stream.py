from __future__ import annotations
from ctypes import pointer, c_uint, c_void_p, cast, POINTER, py_object
from typing import override, Union, List, Callable, Set
import asyncio
from concurrent import futures

from .cuobject import CUObject
from .event import Event
from ..driver import CUstream, CUDA, CUstream_flags, CUevent_wait_flags, CUresult, CUstreamCallback, CUInfo


class Stream(CUObject):

    def __init__(self, flags:CUstream_flags=CUstream_flags.CU_STREAM_DEFAULT):
        self._py_obj = py_object(self)
        self._ptr_py_obj = pointer(self._py_obj)
        self._user_data = cast(self._ptr_py_obj, c_void_p)

        self._flags:CUstream_flags = flags
        self._on_completed_callbacks:Set[Callable[[Stream, CUresult], None]] = set()

        stream_id = CUstream()
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

    @property
    def finished(self)->bool:
        return (self.status == CUresult.CUDA_SUCCESS)
    
    @property
    def status(self)->CUresult:
        return CUresult(CUDA.cuStreamQuery(self.id))

    def create_future(self, future_type:type)->Union[asyncio.Future, futures.Future]:
        future:Union[asyncio.Future, futures.Future] = future_type()

        def on_completed(stream:Stream, error_code:CUresult):
            if error_code != CUresult.CUDA_SUCCESS:
                future.set_exception(RuntimeError(str(error_code) + ": " + CUInfo.error_codes[error_code]))
            else:
                future.set_result(None)
        
        if self.finished:
            on_completed(self, self.status)
        else:
            self.on_completed_callbacks.append(on_completed)

        return future

    def add_completed_callback(self, callback:Callable[[Stream, CUresult], None])->None:
        self._on_completed_callbacks.add(callback)
        CUDA.cuStreamAddCallback(self.id, Stream._callback, self._user_data)

    @CUstreamCallback
    def _callback(stream_id:CUstream, error_code:c_uint, user_data:c_void_p):
        error_code = CUresult(error_code)
        self:Stream = Stream.instance(stream_id)
        on_completed = cast(user_data, POINTER(py_object)).contents.value
        on_completed(self, error_code)
        self._on_completed_callbacks.remove(on_completed)

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