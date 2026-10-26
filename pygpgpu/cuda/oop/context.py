from __future__ import annotations
import threading
from collections import OrderedDict
from ctypes import pointer, c_uint, c_size_t
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from typing_extensions import override

from .cuobject import CUObject
from .event import Event
from ..driver import CUDA, CUctx_flags, CUcontext, CUfunc_cache, CUlimit, CUstream_flags

if TYPE_CHECKING:
    from .device import Device
    from .buffer import Buffer
    from .stream import Stream


# Upper bound on the streams a Context keeps alive. Synchronous calls share one
# cached stream per thread and asynchronous calls draw from a separate pool of
# non-blocking streams; capping both keeps a process that creates and drops
# threads (or keeps launching asynchronously) from growing streams forever.
STREAM_POOL_SIZE:int = 16


class Context(CUObject):

    def __init__(self, device:Device, flags:CUctx_flags=CUctx_flags.CU_CTX_SCHED_AUTO):
        self._device:Device = device
        self._flags:CUctx_flags = flags
        self._api_version:int = 0
        self._buffers:Dict[int, List[Buffer]] = {}

        # Synchronous streams: one per thread, evicted least-recently-used once the
        # pool is full. Keyed by thread id; the value pairs the stream with the
        # thread that created it, because a stream may only be reused by that same
        # thread (a CUDA context is thread-local). The main thread does not take a
        # slot -- it runs on CUDA's legacy NULL stream, whose implicit ordering is
        # what sequential code wants.
        self._sync_streams:OrderedDict[int, Tuple[Stream, threading.Thread]] = OrderedDict()
        self._legacy_stream:Optional[Stream] = None

        # Asynchronous streams must not use the NULL stream: a blocking stream
        # would serialize behind it and defeat the point of launching without
        # waiting, so the pool is created with CU_STREAM_NON_BLOCKING. Each entry
        # pairs the stream with its creating thread for the same reason as above.
        self._async_streams:List[Tuple[Stream, threading.Thread]] = []
        self._async_slots = threading.Semaphore(STREAM_POOL_SIZE)
        self._stream_lock = threading.RLock()

        context_id = CUcontext()
        ptr_context_id = pointer(context_id)
        CUDA.cuCtxCreate(ptr_context_id, flags, device.id)
        CUObject.__init__(self, context_id)

    @staticmethod
    def current()->Context:
        context_id = CUcontext()
        ptr_context_id = pointer(context_id)
        CUDA.cuCtxGetCurrent(ptr_context_id)
        return Context.instance(context_id)

    def get_buffer(self, size:int=0)->Buffer:
        available_buffer:Optional[Buffer] = None
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

    def get_sync_stream(self)->Stream:
        """The stream a blocking call should use.

        The main thread gets CUDA's legacy NULL stream, which is never created and
        never destroyed. Other threads share a least-recently-used cache keyed by
        thread id, capped at STREAM_POOL_SIZE: without the cache a program that
        keeps spawning and joining threads would leak one stream per thread, and
        without the cap the cache itself would grow without bound.

        A cached stream is only reused by the thread that created it. A CUDA
        context is thread-local, and a cuStream belongs to the context current on
        the thread that created it, so handing one to a different thread fails
        with CUDA_ERROR_INVALID_HANDLE. Thread ids can be recycled, so an entry
        whose id was taken over by a new thread is rebuilt rather than reused.
        """
        if threading.current_thread() is threading.main_thread():
            with self._stream_lock:
                self.make_current()
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
                # A recycled thread id or a stream still in flight: drop it and
                # build a fresh one for this thread.
                del self._sync_streams[ident]
                if not stream.busy:
                    victims.append(stream)

            from .stream import Stream
            stream = Stream()
            stream.busy = True
            self._sync_streams[ident] = (stream, threading.current_thread())

            # Evict the least-recently-used idle stream. A busy one is running a
            # kernel and must not be destroyed; if every entry is busy, real
            # concurrency exceeds the pool size and this new stream simply stays
            # out of the cache rather than evicting live work.
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

        # Destroy outside the lock: cuStreamDestroy may block on the driver.
        # A stream cached by a thread whose context has since been destroyed is a
        # dead handle, so a failed destroy is expected and ignored.
        for victim in victims:
            try:
                victim._release(victim.id)
            except BaseException:
                pass

        return stream

    def release_sync_stream(self, stream:Stream)->None:
        """Return a synchronous stream to the cache (the NULL stream is kept)."""
        if stream is self._legacy_stream:
            return
        with self._stream_lock:
            stream.busy = False

    def get_async_stream(self, blocking:bool=True)->Optional[Stream]:
        """Take a non-blocking stream from the asynchronous pool.

        ``blocking`` selects what a caller does when all STREAM_POOL_SIZE streams
        are in flight: a thread that is willing to wait blocks here, while an
        asynchronous launch asks with ``blocking=False`` and waits on its own
        worker instead of stalling the caller. Returns None when
        ``blocking=False`` and no stream is free.
        """
        if blocking:
            self._async_slots.acquire()
        elif not self._async_slots.acquire(blocking=False):
            return None

        with self._stream_lock:
            # See get_sync_stream: a context is thread-local, and a stream belongs to
            # the context of the thread that created it, so only the creating thread
            # may pick it up again.
            self.make_current()

            current = threading.current_thread()
            for stream, owner in self._async_streams:
                if not stream.busy and owner is current:
                    stream.busy = True
                    return stream

            from .stream import Stream
            stream = Stream(CUstream_flags.CU_STREAM_NON_BLOCKING)
            stream.busy = True
            self._async_streams.append((stream, current))
            return stream

    def release_async_stream(self, stream:Stream)->None:
        """Return a non-blocking stream to the pool and free its slot."""
        with self._stream_lock:
            stream.busy = False
        self._async_slots.release()

    def create_buffer(self, size:int=0, flags:Any=None)->Buffer:
        """Aligns with the gnp ndarray._to_device interface (flags are only recorded under CUDA)."""
        from .buffer import Buffer
        return Buffer(size)
    
    def make_current(self):
        if Context.current() == self:
            return
        
        CUDA.cuCtxSetCurrent(self.id)

    def push_current(self):
        CUDA.cuCtxPushCurrent(self.id)

    @staticmethod
    def pop_current()->Context:
        context_id = CUcontext()
        ptr_context_id = pointer(context_id)
        CUDA.cuCtxPopCurrent(ptr_context_id)
        return Context.instance(context_id)

    def sync(self)->None:
        self.make_current()
        CUDA.cuCtxSynchronize()

    def wait_event(self, event:Event)->None:
        CUDA.cuCtxWaitEvent(self.id, event.id)

    @property
    def device(self)->Device:
        return self._device
    
    @property
    def flags(self)->CUctx_flags:
        return self._flags
    
    @property
    def api_version(self)->int:
        if not self._api_version:
            version = c_uint()
            ptr_version = pointer(version)
            CUDA.cuCtxGetApiVersion(self.id, ptr_version)
            self._api_version = version.value

        return self._api_version
    
    @property
    def cache_config(self)->CUfunc_cache:
        self.make_current()
        config = CUfunc_cache()
        ptr_config = pointer(config)
        CUDA.cuCtxGetCacheConfig(ptr_config)
        return config
    
    @cache_config.setter
    def cache_config(self, config:CUfunc_cache):
        self.make_current()
        CUDA.cuCtxSetCacheConfig(config)

    @property
    def stack_size(self)->int:
        return self._get_limit(CUlimit.CU_LIMIT_STACK_SIZE)
    
    @stack_size.setter
    def stack_size(self, size:int):
        self._set_limit(CUlimit.CU_LIMIT_STACK_SIZE, size)

    @property
    def printf_FIFO_size(self)->int:
        return self._get_limit(CUlimit.CU_LIMIT_PRINTF_FIFO_SIZE)
    
    @printf_FIFO_size.setter
    def printf_FIFO_size(self, size:int):
        self._set_limit(CUlimit.CU_LIMIT_PRINTF_FIFO_SIZE, size)

    @property
    def malloc_heap_size(self)->int:
        return self._get_limit(CUlimit.CU_LIMIT_MALLOC_HEAP_SIZE)
    
    @malloc_heap_size.setter
    def malloc_heap_size(self, size:int):
        self._set_limit(CUlimit.CU_LIMIT_MALLOC_HEAP_SIZE, size)

    @property
    def dev_runtime_sync_depth(self)->int:
        return self._get_limit(CUlimit.CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH)
    
    @dev_runtime_sync_depth.setter
    def dev_runtime_sync_depth(self, depth:int):
        self._set_limit(CUlimit.CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH, depth)

    @property
    def dev_runtime_pending_launch_count(self)->int:
        return self._get_limit(CUlimit.CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT)
    
    @dev_runtime_pending_launch_count.setter
    def dev_runtime_pending_launch_count(self, count:int):
        self._set_limit(CUlimit.CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT, count)

    @property
    def max_L2_fetch_granularity(self)->int:
        return self._get_limit(CUlimit.CU_LIMIT_MAX_L2_FETCH_GRANULARITY)
    
    @max_L2_fetch_granularity.setter
    def max_L2_fetch_granularity(self, granularity:int):
        self._set_limit(CUlimit.CU_LIMIT_MAX_L2_FETCH_GRANULARITY, granularity)

    @property
    def persisting_L2_cache_size(self)->int:
        return self._get_limit(CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE)
    
    @persisting_L2_cache_size.setter
    def persisting_L2_cache_size(self, size:int):
        self._set_limit(CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE, size)

    @property
    def shared_memory_size(self)->int:
        return self._get_limit(CUlimit.CU_LIMIT_SHMEM_SIZE)

    @property
    def CIG_enabled(self)->bool:
        return (self._get_limit(CUlimit.CU_LIMIT_CIG_ENABLED) != 0)

    @property
    def CIG_shmem_fallback_enabled(self)->bool:
        return (self._get_limit(CUlimit.CU_LIMIT_CIG_SHMEM_FALLBACK_ENABLED) != 0)
    
    @CIG_shmem_fallback_enabled.setter
    def CIG_shmem_fallback_enabled(self, enabled:bool):
        self._set_limit(CUlimit.CU_LIMIT_CIG_SHMEM_FALLBACK_ENABLED, int(enabled))

    def _get_limit(self, limit:CUlimit)->int:
        self.make_current()
        value = c_size_t()
        ptr_value = pointer(value)
        CUDA.cuCtxGetLimit(ptr_value, limit)
        return value.value

    def _set_limit(self, limit:CUlimit, value:int):
        self.make_current()
        CUDA.cuCtxSetLimit(limit, value)

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
        return CUDA.cuCtxDestroy
