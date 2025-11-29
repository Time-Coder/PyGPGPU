from ctypes import c_char_p, c_uint, c_size_t
from typing import Any

from .cutypes import (
    CUresult,
    CUdevice,
    ptr_size_t,
    ptr_int,
    ptr_CUdevice,
    CUdevice_attribute,
    ptr_CUuuid,
    ptr_CUcontext,
    CUctx_flags,
    CUcontext,
    ptr_uint,
    CUlimit,
    CUfunc_cache
)


class CUDA:

    check_error: bool
    print_call: bool
    print_info: bool

    class Func:

        def __call__(self, *args, **kwargs)->Any: ...

    @staticmethod
    def init(): ...
    
    @staticmethod
    def cuInit(Flags: int)->CUresult: ...
    
    @staticmethod
    def cuDeviceGetCount(count: ptr_int)->CUresult: ...
    
    @staticmethod
    def cuDeviceGet(device: ptr_CUdevice, ordinal: int)->CUresult: ...
    
    @staticmethod
    def cuDeviceGetName(name:c_char_p, len: int, dev: CUdevice)->CUresult: ...
    
    @staticmethod
    def cuDeviceGetAttribute(pi: ptr_int, attrib: CUdevice_attribute, dev: CUdevice)->CUresult: ...
    
    @staticmethod
    def cuDeviceTotalMem(bytes: ptr_size_t, dev: CUdevice)->CUresult: ...
    
    @staticmethod
    def cuDeviceGetUuid(uuid: ptr_CUuuid, dev: CUdevice)->CUresult: ...
    
    @staticmethod
    def cuCtxCreate(pctx: ptr_CUcontext, flags: CUctx_flags,  dev: CUdevice)->CUresult: ...
    
    @staticmethod
    def cuCtxDestroy(ctx: CUcontext)->CUresult: ...
    
    @staticmethod
    def cuCtxGetApiVersion(ctx:CUcontext, version:ptr_uint)->CUresult: ...
    
    @staticmethod
    def cuCtxGetCacheConfig(pconfig: ptr_int)->CUresult: ...
    
    @staticmethod
    def cuCtxGetLimit(pvalue: ptr_size_t, limit: CUlimit)->CUresult: ...
    
    @staticmethod
    def cuCtxPopCurrent(pctx: ptr_CUcontext)->CUresult: ...
    
    @staticmethod
    def cuCtxPushCurrent(ctx: CUcontext)->CUresult: ...

    @staticmethod
    def cuCtxGetCurrent(pctx: ptr_CUcontext)->CUresult: ...

    @staticmethod
    def cuCtxSetCurrent(ctx: CUcontext)->CUresult: ...
    
    @staticmethod
    def cuCtxSetCacheConfig(config: CUfunc_cache)->CUresult: ...
    
    @staticmethod
    def cuCtxSetLimit(limit: CUlimit, value: c_size_t)->CUresult: ...
    
    @staticmethod
    def cuCtxSynchronize()->CUresult: ...
