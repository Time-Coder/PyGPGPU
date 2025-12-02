from ctypes import c_char_p, c_uint, c_size_t, c_void_p
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
    CUfunc_cache,
    ptr_CUmodule,
    ptr_CUfunction,
    CUmodule,
    CUfunction_attribute,
    CUfunction,
    ptr_CUdeviceptr,
    CUdeviceptr,
    CUstream,
    ptr_ptr_void,
    ptr_CUevent,
    CUevent_flags,
    CUevent,
    ptr_CUstream,
    CUstream_flags,
    CUevent_wait_flags,
    ptr_float
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

    @staticmethod
    def cuCtxWaitEvent(hCtx:CUcontext, hEvent:CUevent)->CUresult: ...

    @staticmethod
    def cuModuleLoadData(module:ptr_CUmodule, image:c_void_p)->CUresult: ...

    @staticmethod
    def cuModuleGetFunction(hfunc:ptr_CUfunction, hmod:CUmodule, name:c_char_p)->CUresult: ...

    @staticmethod
    def cuModuleUnload(hmod:CUmodule)->CUresult: ...

    @staticmethod
    def cuFuncGetAttribute(pi:ptr_int, attrib:CUfunction_attribute, hfunc:CUfunction)->CUresult: ...

    @staticmethod
    def cuFuncSetAttribute(hfunc:CUfunction, attrib:CUfunction_attribute, value:int)->CUresult: ...

    @staticmethod
    def cuMemAlloc(dptr:ptr_CUdeviceptr, bytesize:int)->CUresult: ...

    @staticmethod
    def cuMemcpyHtoD(dstDevice:CUdeviceptr, srcHost:c_void_p, ByteCount:int)->CUresult: ...

    @staticmethod
    def cuMemcpyDtoH(dstHost:c_void_p, srcDevice:CUdeviceptr, ByteCount:int)->CUresult: ...

    @staticmethod
    def cuMemcpyHtoDAsync(dstDevice:CUdeviceptr, srcHost:c_void_p, ByteCount:int, hStream:CUstream)->CUresult: ...

    @staticmethod
    def cuMemcpyDtoHAsync(dstHost:c_void_p, srcDevice:CUdeviceptr, ByteCount:int, hStream:CUstream)->CUresult: ...

    @staticmethod
    def cuMemFree(dptr:CUdeviceptr)->CUresult: ...

    @staticmethod
    def cuLaunchKernel(
        f: CUfunction,
        gridDimX: int, gridDimY: int, gridDimZ: int,
        blockDimX: int, blockDimY: int, blockDimZ: int,
        sharedMemBytes: int, hStream: CUstream, kernelParams: ptr_ptr_void, extra: ptr_ptr_void
    )->CUresult: ...

    @staticmethod
    def cuEventCreate(phEvent:ptr_CUevent, Flags:CUevent_flags)->CUresult: ...

    @staticmethod
    def cuEventRecord(hEvent:CUevent, hStream:CUstream)->CUresult: ...

    @staticmethod
    def cuEventSynchronize(hEvent:CUevent)->CUresult: ...

    @staticmethod
    def cuStreamCreate(phStream:ptr_CUstream, Flags:CUstream_flags)->CUresult: ...

    @staticmethod
    def cuStreamWaitEvent(hStream:CUstream, hEvent:CUevent, Flags:CUevent_wait_flags)->CUresult: ...

    @staticmethod
    def cuEventElapsedTime(pMilliseconds:ptr_float, hStart:CUevent, hEnd:CUevent)->CUresult: ...

    @staticmethod
    def cuEventDestroy(hEvent:CUevent)->CUresult: ...

    @staticmethod
    def cuStreamDestroy(hStream:CUstream)->CUresult: ...
