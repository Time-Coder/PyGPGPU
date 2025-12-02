from __future__ import annotations
from ctypes import pointer, c_uint, c_size_t
from typing import override, TYPE_CHECKING

from .cuobject import CUObject
from .event import Event
from ..driver import CUDA, CUctx_flags, CUcontext, CUfunc_cache, CUlimit

if TYPE_CHECKING:
    from .device import Device


class Context(CUObject):

    def __init__(self, device:Device, flags:CUctx_flags=CUctx_flags.CU_CTX_SCHED_AUTO):
        self._device:Device = device
        self._flags:CUctx_flags = flags
        self._api_version:int = 0

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
    def _get_info_func()->CUDA.Func:
        return None

    @staticmethod
    @override
    def _info_enum()->type:
        return None

    @staticmethod
    @override
    def _release_func()->CUDA.Func:
        return CUDA.cuCtxDestroy