from __future__ import annotations
import threading
from collections import OrderedDict

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..driver import HIP

if TYPE_CHECKING:
    from .device import Device
    from .buffer import Buffer
    from .stream import Stream


# Upper bound on the streams a Context keeps alive; mirrors the CUDA backend.
STREAM_POOL_SIZE:int = 16


class Context:
    """A HIP context is simply the "current device" (hipSetDevice); make_current switches the current device."""

    _current_context:"Context" = None

    def __init__(self, device:"Device"):
        self._device:"Device" = device
        self._buffers:Dict[int, List[Buffer]] = {}

        # One cached stream per thread for blocking calls, evicted least-recently
        # used past STREAM_POOL_SIZE; the main thread uses the NULL stream instead.
        # Each entry pairs the stream with its creating thread, since a stream may
        # only be reused on that thread.
        self._sync_streams:OrderedDict[int, Tuple[Stream, threading.Thread]] = OrderedDict()
        self._legacy_stream:Optional[Stream] = None

        # Non-blocking streams for asynchronous launches, capped by a semaphore.
        self._async_streams:List[Tuple[Stream, threading.Thread]] = []
        self._async_slots = threading.Semaphore(STREAM_POOL_SIZE)
        self._stream_lock = threading.RLock()

    @staticmethod
    def current()->Optional["Context"]:
        return Context._current_context

    def make_current(self)->None:
        if Context._current_context is self:
            return
        HIP.hipSetDevice(self._device.id)
        Context._current_context = self

    def sync(self)->None:
        self.make_current()
        HIP.hipDeviceSynchronize()

    def get_buffer(self, size:int=0)->"Buffer":
        available_buffer = None
        for buffer in self._buffers.get(size, []):
            if not buffer.using:
                available_buffer = buffer
                break

        if available_buffer is None:
            from .buffer import Buffer
            available_buffer = Buffer(size)
            self._buffers.setdefault(size, []).append(available_buffer)

        available_buffer.use()
        return available_buffer

    # ---------- stream pools ----------

    def _take_idle_sync(self) -> Optional["Stream"]:
        """Pop the least-recently-used idle stream from the synchronous cache.

        Caller must hold _stream_lock.
        """
        for ident, stream in self._sync_streams.items():
            if not stream.busy:
                del self._sync_streams[ident]
                return stream
        return None

    def _take_idle_async(self) -> Optional["Stream"]:
        """Pop an idle stream from the asynchronous pool (caller holds the lock)."""
        for index, stream in enumerate(self._async_streams):
            if not stream.busy:
                del self._async_streams[index]
                return stream
        return None

    def get_sync_stream(self) -> "Stream":
        """The stream a blocking call should use.

        The main thread runs on the NULL stream, which is never created and never
        destroyed, so the common case allocates nothing.

        Other threads share a least-recently-used cache capped at STREAM_POOL_SIZE,
        keyed by thread id. A HIP stream belongs to the context that was current
        when it was created, not to the thread, so it can be handed to another
        thread as long as this context is made current there first. Three things
        avoid needless create/destroy cycles: a new thread takes over the
        least-recently-used idle entry; a full cache re-keys its victim onto the
        current thread instead of destroying it; and the asynchronous pool is
        drained before a new stream is created, so both pools share one budget.
        """
        if threading.current_thread() is threading.main_thread():
            with self._stream_lock:
                if self._legacy_stream is None:
                    from .stream import Stream
                    self._legacy_stream = Stream.legacy()
                return self._legacy_stream

        ident = threading.get_ident()
        with self._stream_lock:
            self.make_current()

            cached = self._sync_streams.get(ident)
            if cached is not None and not cached.busy:
                self._sync_streams.move_to_end(ident)
                cached.busy = True
                return cached

            stream = self._take_idle_sync()
            if stream is None:
                stream = self._take_idle_async()
                if stream is not None:
                    self._async_slots.release()
            if stream is not None:
                stream.busy = True
                self._sync_streams[ident] = stream
                return stream

            from .stream import Stream
            stream = Stream()
            stream.busy = True
            self._sync_streams[ident] = stream

            if len(self._sync_streams) > STREAM_POOL_SIZE:
                self._shrink_sync_cache(ident)

            return stream

    def _shrink_sync_cache(self, keep_ident:int)->None:
        """Bring the synchronous cache back to STREAM_POOL_SIZE (caller holds lock).

        The least-recently-used entry is re-keyed onto ``keep_ident`` instead of
        being destroyed, so nothing is allocated or released. If every other entry
        is busy, the newest is left out of the cache.
        """
        while len(self._sync_streams) > STREAM_POOL_SIZE:
            for victim_ident, victim in self._sync_streams.items():
                if victim_ident == keep_ident:
                    continue
                if victim.busy:
                    continue
                del self._sync_streams[victim_ident]
                del self._sync_streams[keep_ident]
                self._sync_streams[keep_ident] = victim
                return
            del self._sync_streams[keep_ident]
            return

    def release_sync_stream(self, stream:"Stream")->None:
        """Return a synchronous stream to the cache (the NULL stream is kept)."""
        if stream is self._legacy_stream:
            return
        with self._stream_lock:
            stream.busy = False

    def get_async_stream(self, blocking:bool=True)->Optional["Stream"]:
        """Take a non-blocking stream from the asynchronous pool.

        ``blocking`` decides what happens when all STREAM_POOL_SIZE streams are in
        flight: a caller willing to wait blocks here, while an asynchronous launch
        asks with ``blocking=False`` and waits on its own worker. An idle stream
        from the synchronous cache is taken over before creating a new one.
        """
        if blocking:
            self._async_slots.acquire()
        elif not self._async_slots.acquire(blocking=False):
            return None

        with self._stream_lock:
            self.make_current()

            stream = self._take_idle_async()
            if stream is None:
                stream = self._take_idle_sync()
            if stream is not None:
                stream.busy = True
                self._async_streams.append(stream)
                return stream

            from .stream import Stream
            stream = Stream(flags=1)   # hipStreamNonBlocking
            stream.busy = True
            self._async_streams.append(stream)
            return stream

    def release_async_stream(self, stream:"Stream")->None:
        """Return a non-blocking stream to the pool and free its slot."""
        with self._stream_lock:
            stream.busy = False
        self._async_slots.release()

    def create_buffer(self, size:int=0, flags:Any=None)->"Buffer":
        """Aligns with the gnp ndarray._to_device interface."""
        from .buffer import Buffer
        return Buffer(size)

    @property
    def device(self)->"Device":
        return self._device
