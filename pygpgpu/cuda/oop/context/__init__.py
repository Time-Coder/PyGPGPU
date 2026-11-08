from __future__ import annotations
import threading
from collections import OrderedDict
from ctypes import pointer, c_uint, c_size_t
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import override
from ..cuobject import CUObject
from ..event import Event
from ...driver import CUDA, CUctx_flags, CUcontext, CUfunc_cache, CUlimit, CUstream_flags
from .context_stream import _ContextStreamMixin
from .context_image import _ContextImageMixin


if TYPE_CHECKING:
    # numpy is imported inside the methods that use it; the annotation needs the name.
    import numpy as np

    from ..surface import SurfaceObject

    from ..device import Device
    from ..buffer import Buffer
    from ..stream import Stream
    from ..array import Array
    from ..texture import TextureObject


STREAM_POOL_SIZE:int = 16


TEXTURE_RETRY_ATTEMPTS:int = 40


class Context(_ContextStreamMixin, _ContextImageMixin, CUObject):

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


    def get_buffer(self, size:int=0)->Buffer:
        available_buffer:Optional[Buffer] = None
        for buffer in self._buffers.get(size, []):
            if not buffer.using:
                available_buffer = buffer
                break

        if available_buffer is None:
            from ..buffer import Buffer
            available_buffer = Buffer(size)
            self._buffers.setdefault(size, []).append(available_buffer)

        available_buffer.use()
        return available_buffer


    def create_buffer(self, size:int=0, flags:object=None)->Buffer:
        """Aligns with the gnp ndarray._to_device interface (flags are only recorded under CUDA)."""
        from ..buffer import Buffer
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
    def _get_info_func()->object:
        return None


    @staticmethod
    @override
    def _info_enum()->type|None:
        return None


    @staticmethod
    @override
    def _release_func()->object:
        return CUDA.cuCtxDestroy_v2


__all__ = [
    "CIG_enabled",
    "CIG_shmem_fallback_enabled",
    "Context",
    "__init__",
    "_get_info_func",
    "_get_limit",
    "_info_enum",
    "_prefix",
    "_release_func",
    "_set_limit",
    "_shrink_sync_cache",
    "_take_idle_async",
    "_take_idle_sync",
    "api_version",
    "cache_config",
    "create_array",
    "create_buffer",
    "create_image",
    "create_surface",
    "create_texture",
    "current",
    "dev_runtime_pending_launch_count",
    "dev_runtime_sync_depth",
    "device",
    "flags",
    "get_async_stream",
    "get_buffer",
    "get_sync_stream",
    "make_current",
    "malloc_heap_size",
    "max_L2_fetch_granularity",
    "persisting_L2_cache_size",
    "pop_current",
    "printf_FIFO_size",
    "push_current",
    "release_async_stream",
    "release_sync_stream",
    "shared_memory_size",
    "stack_size",
    "sync",
    "texture",
    "wait_event",
]
