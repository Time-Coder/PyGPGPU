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
    from .array import Array
    from .texture import TextureObject


# Upper bound on the streams a Context keeps alive. Synchronous calls share one
# cached stream per thread and asynchronous calls draw from a separate pool of
# non-blocking streams; capping both keeps a process that creates and drops
# threads (or keeps launching asynchronously) from growing streams forever.
STREAM_POOL_SIZE:int = 16

# How many times create_texture rebuilds an image before giving up. The driver
# rejects texture creation intermittently, and a refused array stays refused, so
# each attempt allocates a replacement rather than retrying the same handle.
TEXTURE_RETRY_ATTEMPTS:int = 24


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
        # cuCtxCreate_v2, not the 1.0-era cuCtxCreate: the legacy symbol yields a
        # context on which cuArrayCreate fails with CUDA_ERROR_INVALID_CONTEXT,
        # which would make the Array/TextureObject support unusable.
        CUDA.cuCtxCreate_v2(ptr_context_id, flags, device.id)
        CUObject.__init__(self, context_id)

    @staticmethod
    def current()->Context:
        context_id = CUcontext()
        ptr_context_id = pointer(context_id)
        CUDA.cuCtxGetCurrent(ptr_context_id)
        return Context.instance(context_id)

    def create_array(self, width:int, height:int, dtype:Any=None, num_channels:int=1)->Array:
        """Allocate a 2D device-side image (the CUDA counterpart of image2d_t)."""
        import numpy as np
        from .array import Array

        self.make_current()
        return Array(width, height, np.float32 if dtype is None else dtype, num_channels)

    def create_texture(self, array:Array, **options:Any)->TextureObject:
        """Build a sampled view over ``array`` for use as a kernel argument.

        On the tested machine the driver intermittently rejects texture creation with
        CUDA_ERROR_INVALID_VALUE and accepts a later request, and rejecting an array
        seems to stick to that array: retrying the same handle kept failing while a
        freshly allocated one succeeded. So a refusal is answered by re-creating the
        image from the array's contents and building a texture over the new handle,
        which the caller never sees.
        """
        from .texture import TextureObject

        self.make_current()
        try:
            return TextureObject(array, **options)
        except BaseException as first_error:
            last_error:BaseException = first_error

        contents = array.to_numpy()
        for _ in range(TEXTURE_RETRY_ATTEMPTS):
            fresh = self.create_array(array.width, array.height, array.dtype,
                                      array.num_channels)
            fresh.set_data(contents)
            try:
                texture = TextureObject(fresh, **options)
            except BaseException as error:
                last_error = error
                continue
            # Hand the caller an object that keeps the replacement image alive.
            texture._array = fresh
            return texture

        raise last_error

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

    def _take_idle_sync(self) -> Optional[Stream]:
        """Pop the least-recently-used idle stream from the synchronous cache.

        Caller must hold _stream_lock. The entry is returned without being marked
        busy; whoever takes it does that.
        """
        for ident, stream in self._sync_streams.items():
            if not stream.busy:
                del self._sync_streams[ident]
                return stream
        return None

    def _take_idle_async(self) -> Optional[Stream]:
        """Pop an idle stream from the asynchronous pool (caller holds the lock)."""
        for index, stream in enumerate(self._async_streams):
            if not stream.busy:
                del self._async_streams[index]
                return stream
        return None

    def get_sync_stream(self)->Stream:
        """The stream a blocking call should use.

        The main thread runs on CUDA's legacy NULL stream: it is never created and
        never destroyed, and its implicit ordering with every other blocking stream
        is exactly what sequential code wants, so the common case allocates nothing.

        Other threads share a least-recently-used cache capped at STREAM_POOL_SIZE,
        keyed by thread id. A stream belongs to the context that was current when it
        was created, not to the thread, so it can be handed to another thread as
        long as this context is made current there first (which the pool always
        does). Three things are done to avoid needless create/destroy cycles:

        - A new thread takes over the least-recently-used idle entry rather than
          creating a stream, then evicting it and creating another.
        - When the cache is full, the victim entry is re-keyed onto the current
          thread instead of being destroyed, so no stream is created at all.
        - Only when the cache and the asynchronous pool hold no idle stream is a
          new stream created -- and even then the asynchronous pool is drained
          first, so the two pools share one budget.
        """
        if threading.current_thread() is threading.main_thread():
            with self._stream_lock:
                self.make_current()
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

            # A new thread (or one whose cached stream is still in flight): reuse
            # whatever is already idle before creating anything.
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

        The least-recently-used entry is re-keyed onto ``keep_ident`` rather than
        destroyed, so the cache stays full but nothing is allocated or released.
        Only if every entry is busy -- real concurrency above the pool size -- is
        the new entry left out of the cache.
        """
        while len(self._sync_streams) > STREAM_POOL_SIZE:
            for victim_ident, victim in self._sync_streams.items():
                if victim_ident == keep_ident:
                    continue
                if victim.busy:
                    continue
                # Re-key instead of destroy: the victim becomes the entry for the
                # thread that just asked, so no stream is created or destroyed.
                del self._sync_streams[victim_ident]
                del self._sync_streams[keep_ident]
                self._sync_streams[keep_ident] = victim
                return
            # Everything else is in flight; drop the newest entry from the cache
            # and let its finalizer reclaim it when it goes idle.
            del self._sync_streams[keep_ident]
            return

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

        An idle stream from the synchronous cache is taken over before creating a
        new one, so the two pools share a single budget of live streams.
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
            stream = Stream(CUstream_flags.CU_STREAM_NON_BLOCKING)
            stream.busy = True
            self._async_streams.append(stream)
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
        CUDA.cuCtxPushCurrent_v2(self.id)

    @staticmethod
    def pop_current()->Context:
        context_id = CUcontext()
        ptr_context_id = pointer(context_id)
        CUDA.cuCtxPopCurrent_v2(ptr_context_id)
        return Context.instance(context_id)

    def sync(self)->None:
        self.make_current()
        CUDA.cuCtxSynchronize_v2()

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
        return CUDA.cuCtxDestroy_v2
