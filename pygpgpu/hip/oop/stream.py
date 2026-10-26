from __future__ import annotations
from ctypes import pointer, c_uint, c_void_p
from typing import Union, Callable, Set, Any
from typing_extensions import override
import asyncio
from concurrent import futures

from .hipobject import HIPObject
from .event import Event
from ..driver import CUstream, CUstreamCallback
from ..driver import HIP, HIPInfo, hipError
from ...utils import settle_future


class Stream(HIPObject):

    def __init__(self, flags:int=0):
        self._flags:int = flags
        self._on_completed_callbacks:Set[Callable[[Stream, int], None]] = set()
        # Set while a launch owns the stream; the Context pool only hands out
        # streams whose flag is clear and only destroys idle ones.
        self.busy:bool = False
        # The legacy NULL stream is not created (and must not be destroyed).
        self._legacy:bool = False

        stream_id = CUstream()
        ptr_stream_id = pointer(stream_id)
        HIP.hipStreamCreate(ptr_stream_id, flags)
        HIPObject.__init__(self, stream_id)
        # Resolved through the weak registry rather than a py_object(self), which
        # would make the object reference itself and only be freed by the cyclic
        # collector (see the CUDA Stream for the full explanation).
        self._user_data = self._callback_data()

    @staticmethod
    def legacy()->Stream:
        """Wrap the legacy NULL stream, which is HIP's default stream (id 0).

        Launches on it are implicitly ordered with every other blocking stream,
        which is what a synchronous call wants: sequential execution without
        creating a stream per call.
        """
        stream = Stream.__new__(Stream)
        stream._flags = 0
        stream._on_completed_callbacks = set()
        stream.busy = True
        stream._legacy = True
        HIPObject.__init__(stream, CUstream(0))
        stream._user_data = stream._callback_data()
        return stream

    @property
    def flags(self)->int:
        return self._flags

    def record_event(self)->Event:
        event = Event()
        HIP.hipEventRecord(event.id, self.id)
        return event
    
    def wait_event(self, event:Event, flags:int=0):
        HIP.hipStreamWaitEvent(self.id, event.id, flags)

    def sync(self)->None:
        HIP.hipStreamSynchronize(self.id)

    @property
    def finished(self)->bool:
        return (self.status == hipError.SUCCESS)
    
    @property
    def status(self)->int:
        return int(HIP.hipStreamQuery(self.id))

    def create_future(self, future_type:type)->Union[asyncio.Future, futures.Future]:
        future:Union[asyncio.Future, futures.Future] = future_type()

        def on_completed(stream:Stream, error_code:int):
            # Runs on a driver thread; settle_future hands the completion to the
            # event loop so a bare `await future` wakes up.
            if error_code != hipError.SUCCESS:
                settle_future(future, RuntimeError(
                    str(error_code) + ": " + HIPInfo.error_codes[error_code]))
            else:
                settle_future(future)
        
        if self.finished:
            on_completed(self, self.status)
        else:
            # Register through add_callback, not by appending to the set: the set is
            # only drained by the driver callback that add_callback queues, so a
            # callback added here without one would never run and the future would
            # never complete.
            self.add_callback(on_completed)

        return future

    def add_callback(self, callback:Callable[[Stream, int], None])->None:
        self._on_completed_callbacks.add(callback)
        HIP.hipStreamAddCallback(self.id, Stream._callback, self._user_data, 0)

    @CUstreamCallback
    def _callback(stream_id:CUstream, error_code:c_uint, user_data:c_void_p):
        error_code = int(error_code)
        self:Stream = HIPObject._from_callback(user_data)
        if self is None:
            # The wrapper was collected while the callback was still queued.
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
        return HIP.hipStreamDestroy
