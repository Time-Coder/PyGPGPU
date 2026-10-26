from __future__ import annotations
from ctypes import pointer, c_uint, c_void_p
from typing import Union, Callable, Set, Any
from typing_extensions import override
import asyncio
from concurrent import futures

from .cuobject import CUObject
from .event import Event
from ..driver import CUstream, CUDA, CUstream_flags, CUevent_wait_flags, CUresult, CUstreamCallback, CUInfo


class Stream(CUObject):

    def __init__(self, flags:CUstream_flags=CUstream_flags.CU_STREAM_DEFAULT):
        self._flags:CUstream_flags = flags
        self._on_completed_callbacks:Set[Callable[[Stream, CUresult], None]] = set()
        # Set while a launch owns the stream; the Context pool only hands out
        # streams whose flag is clear and only destroys idle ones.
        self.busy:bool = False
        # The legacy NULL stream is not created (and must not be destroyed); it is
        # tracked so the pool can treat it like any other stream.
        self._legacy:bool = False

        stream_id = CUstream()
        ptr_stream_id = pointer(stream_id)
        CUDA.cuStreamCreate(ptr_stream_id, flags)
        CUObject.__init__(self, stream_id)
        # Resolved through CUObject._callback_targets rather than a py_object(self),
        # which would make this object reference itself and only be freed by the
        # cyclic collector (see CUObject._callback_data).
        self._user_data = self._callback_data()

    @staticmethod
    def legacy()->Stream:
        """Wrap the legacy NULL stream, which is CUDA's default stream (id 0).

        Every launch on it is implicitly ordered with every other stream that is
        not CU_STREAM_NON_BLOCKING, which is exactly what a synchronous call
        wants: sequential execution without creating a stream per call.
        """
        stream = Stream.__new__(Stream)
        stream._flags = CUstream_flags.CU_STREAM_DEFAULT
        stream._on_completed_callbacks = set()
        stream.busy = True
        stream._legacy = True
        CUObject.__init__(stream, CUstream(0))
        stream._user_data = stream._callback_data()
        # CUObject._release skips a null id, so the NULL stream is never destroyed
        # even if this wrapper is collected; no finalizer override is needed.
        return stream

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
            # Register through add_callback, not by appending to the set: the set is
            # only drained by the driver callback that add_callback queues, so a
            # callback added here without one would never run and the future would
            # never complete.
            self.add_callback(on_completed)

        return future

    def add_callback(self, callback:Callable[[Stream, CUresult], None])->None:
        self._on_completed_callbacks.add(callback)
        CUDA.cuStreamAddCallback(self.id, Stream._callback, self._user_data, 0)

    @CUstreamCallback
    def _callback(stream_id:CUstream, error_code:c_uint, user_data:c_void_p):
        error_code = CUresult(error_code)
        self:Stream = CUObject._from_callback(user_data)
        if self is None:
            # The wrapper was collected while the callback was still queued; there
            # is nothing left to notify.
            return
        for on_completed in list(self._on_completed_callbacks):
            on_completed(self, error_code)
            self._on_completed_callbacks.remove(on_completed)

    @staticmethod
    @override
    def _prefix()->str:
        return ""

    @staticmethod
    @override
    def _get_info_func()->Any:
        return None

    @staticmethod
    @override
    def _info_enum()->type|None:
        return None

    @staticmethod
    @override
    def _release_func()->Any:
        return CUDA.cuStreamDestroy
