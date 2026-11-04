"""The Metal backend's ctypes driver layer, shaped like the OpenCL and CUDA drivers.

The runtime (Metal.framework / libobjc) is initialized on a background thread
started at import: `import pygpgpu.metal` does not block on the framework load.
Every driver call goes through `Func.__call__`, which waits for that
initialization to finish before touching the driver -- the wait is the whole
synchronization story, and nothing else needs to know. Both `_start_init` and
`_wait_ready` are private: the backend package starts the thread when it is
imported, and a caller never asks for the initialization or waits for it
explicitly.

A machine without Metal (any non-Apple machine, and Mac virtual machines) fails
the CDLL load in `init()`; the failure is recorded in `MetaMetal.last_init_error`
instead of raised, exactly like the CUDA/HIP/OpenCL drivers record theirs. That
is what lets `import pygpgpu.metal` succeed everywhere and the backend probe
answer "unavailable" without an exception escaping the import.

Metal's device API is Objective-C, so device/queue/buffer/library calls do not
go through a flat signature table: the oop layer sends them as messages through
the `objc` bridge (objc_msgSend). The C entry points Metal does export
(`MTLCreateSystemDefaultDevice`, `MTLCopyAllDevices`) are the ones that follow
the `Func` pattern here.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional
from ctypes import CDLL, c_ulonglong, c_void_p

from .objc import ObjC

# The same background-initialization contract as every other driver (see the
# module docstring): one event, one thread, one lock.
_INIT_DONE = threading.Event()
_INIT_THREAD:Optional[threading.Thread] = None
_INIT_LOCK = threading.Lock()

# Metal.framework lives at a fixed path on macOS.
METAL_FRAMEWORK:str = "/System/Library/Frameworks/Metal.framework/Metal"


class MetaMetal(type):
    """The driver entry points and the recorded init error, per the driver contract."""

    __dll = None
    # The exception init() hit when the machine has no runtime (the CDLL OSError
    # for Metal.framework on a non-Apple machine). Diagnostic only: every real
    # call re-raises whatever the driver itself raises, so nothing here can make
    # a broken runtime look usable.
    last_init_error:Optional[BaseException] = None

    @staticmethod
    def dll():
        if MetaMetal.__dll is None:
            MetaMetal.__dll = CDLL(METAL_FRAMEWORK)
        return MetaMetal.__dll

    def __getattr__(self, name:str)->Callable:
        """Resolve an unknown name into a lazily bound C entry point."""
        if name not in MTL.func_signatures:
            MTL.func_signatures[name] = {"args": {}, "restype": c_void_p}
        return MTL.Func(name)


class MTL(metaclass=MetaMetal):
    """The Metal C entry points, named after the symbols of Metal.framework.

    Device/queue/buffer/library handles are plain c_void_p at this layer; the
    oop layer wraps them and speaks the Objective-C API through `objc`.
    """

    print_call:bool = False
    check_error:bool = True

    # The C entry points, with their real signatures. `MTLCopyAllDevices` exists
    # on macOS only; a probe on another platform fails earlier, at the CDLL.
    func_signatures:Dict[str, Dict[str, Any]] = {
        # id<MTLDevice> MTLCreateSystemDefaultDevice(void)
        "MTLCreateSystemDefaultDevice": {"args": {}, "restype": c_void_p},
        # id<MTLDevice>* MTLCopyAllDevices(size_t* count)  (macOS only)
        "MTLCopyAllDevices": {"args": {"count": c_void_p}, "restype": c_void_p},
    }

    class Func:
        """One C entry point of Metal.framework, resolved and cached like CL.Func."""

        def __init__(self, name:str):
            self.name = name

        def __call__(self, *args, **kwargs):
            # The one synchronization point with the background initialization: a
            # call made before it finished simply waits for it, then runs (or
            # fails with the driver's own error, if the runtime is missing).
            _wait_ready()

            func_info = MTL.func_signatures[self.name]
            if "dll_func" not in func_info:
                func = getattr(MetaMetal.dll(), self.name)
                func.argtypes = list(func_info["args"].values())
                func.restype = func_info["restype"]
                func_info["dll_func"] = func
            else:
                func = func_info["dll_func"]

            current_n_args:int = len(args)
            target_n_args:int = len(func_info["args"])
            if current_n_args < target_n_args:
                args = list(args)
                args_names = list(func_info["args"].keys())
                for i in range(current_n_args, target_n_args):
                    args.append(kwargs[args_names[i]])

            return func(*args)

    @staticmethod
    def device_ids()->list:
        """Every Metal device, by handle, in the order MTLCopyAllDevices reports them.

        `MTLCopyAllDevices` lists every GPU; when that entry point is absent (an
        older runtime) the system-default device is the only one.
        """
        from ctypes import POINTER, byref, c_void_p
        count = c_ulonglong(0)
        try:
            array = MTL.MTLCopyAllDevices(byref(count))
        except AttributeError:
            array = 0
        if array:
            handles = POINTER(c_void_p)(array)
            return [int(handles[index]) for index in range(int(count.value))]
        default = MTL.MTLCreateSystemDefaultDevice()
        return [default] if default else []


def init()->None:
    """Load the frameworks and preheat the runtime, ON THE BACKGROUND THREAD.

    This is the only initializer. It catches its own failure (a machine without
    Metal) and records it, so the import and the backend probe stay
    exception-free -- a broken runtime fails exactly where it is used.

    The preheat goes through the raw framework, never through the oop layer:
    init runs ON the background thread, and the oop layer's calls wait for init,
    so calling back into it here would wait for itself -- a deadlock.
    """
    try:
        ObjC.load()
        MetaMetal.dll()  # raises OSError where Metal.framework is absent
        # Register the selectors the backend's oop layer uses, once. SEL values
        # are interned strings, so an early registration costs nothing at call
        # time and keeps every later send to a dict lookup.
        for name in (
            "alloc", "init", "name", "retain", "release",
            "newCommandQueue",
            "newBufferWithLength:options:",
            "newBufferWithBytesNoCopy:length:options:deallocator:",
            "newLibraryWithSource:options:completionHandler:",
            "newFunctionWithName:",
            "newComputePipelineStateWithFunction:error:",
            "setComputePipelineState:", "setBuffer:offset:atIndex:",
            "dispatchThreadgroups:threadsPerThreadgroup:",
            "commit", "waitUntilCompleted",
            "pushDebugGroup:", "popDebugGroup",
            "maxThreadsPerThreadgroup", "threadExecutionWidth",
            "maxThreadgroupMemoryLength", "maxBufferLength",
            "currentAllocatedSize", "registryID", "isHeadless",
            "hasUnifiedMemory", "recommendedMaxWorkingSetSize",
        ):
            ObjC.sel(name)
    except BaseException as error:  # noqa: BLE001  (the contract: record, never raise)
        MetaMetal.last_init_error = error


def _start_init()->None:
    """Run `init()` once, on a background thread (idempotent)."""
    global _INIT_THREAD
    with _INIT_LOCK:
        if _INIT_THREAD is not None:
            return

        def _run()->None:
            try:
                init()
            finally:
                _INIT_DONE.set()

        _INIT_THREAD = threading.Thread(target=_run, name="pygpgpu-metal-init")
        _INIT_THREAD.start()


def _wait_ready()->None:
    """Block until the background runtime initialization has finished (or failed)."""
    _INIT_DONE.wait()
