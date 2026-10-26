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

    def create_buffer(self, size:int=0, flags:Any=None)->"Buffer":
        """Aligns with the gnp ndarray._to_device interface."""
        from .buffer import Buffer
        return Buffer(size)

    # ---------- stream pools ----------

    def get_sync_stream(self)->"Stream":
        """The stream a blocking call should use.

        The main thread runs on the NULL stream, which is never created and never
        destroyed. Other threads share a least-recently-used cache keyed by thread
        id and capped at STREAM_POOL_SIZE, so a program that keeps spawning and
        joining threads reuses a bounded set of streams instead of leaking one per
        thread. A cached stream is only reused by the thread that created it: HIP,
        like CUDA, ties a stream to the context current on the creating thread, so
        a recycled thread id gets a fresh stream rather than a foreign handle.
        """
        if threading.current_thread() is threading.main_thread():
            with self._stream_lock:
                if self._legacy_stream is None:
                    from .stream import Stream
                    self._legacy_stream = Stream.legacy()
                return self._legacy_stream

        ident = threading.get_ident()
        victims:List[Stream] = []
        with self._stream_lock:
            self.make_current()

            entry = self._sync_streams.get(ident)
            if entry is not None:
                stream, owner = entry
                if owner is threading.current_thread() and not stream.busy:
                    self._sync_streams.move_to_end(ident)
                    stream.busy = True
                    return stream
                del self._sync_streams[ident]
                if not stream.busy:
                    victims.append(stream)

            from .stream import Stream
            stream = Stream()
            stream.busy = True
            self._sync_streams[ident] = (stream, threading.current_thread())

            while len(self._sync_streams) > STREAM_POOL_SIZE:
                evicted = False
                for victim_ident, (victim, _owner) in self._sync_streams.items():
                    if victim_ident == ident:
                        continue
                    if not victim.busy:
                        del self._sync_streams[victim_ident]
                        victims.append(victim)
                        evicted = True
                        break
                if not evicted:
                    del self._sync_streams[ident]
                    break

        for victim in victims:
            try:
                victim._release(victim.id)
            except BaseException:
                pass

        return stream

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
        asks with ``blocking=False`` and waits on its own worker instead. Returns
        None when ``blocking=False`` and no stream is free.
        """
        if blocking:
            self._async_slots.acquire()
        elif not self._async_slots.acquire(blocking=False):
            return None

        with self._stream_lock:
            self.make_current()

            current = threading.current_thread()
            for stream, owner in self._async_streams:
                if not stream.busy and owner is current:
                    stream.busy = True
                    return stream

            from .stream import Stream
            stream = Stream(flags=1)   # hipStreamNonBlocking
            stream.busy = True
            self._async_streams.append((stream, current))
            return stream

    def release_async_stream(self, stream:"Stream")->None:
        """Return a non-blocking stream to the pool and free its slot."""
        with self._stream_lock:
            stream.busy = False
        self._async_slots.release()

    @property
    def device(self)->"Device":
        return self._device
