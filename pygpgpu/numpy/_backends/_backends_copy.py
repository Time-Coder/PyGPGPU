# The copies between host and device, and the events they settle through.
from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from .._devices import Target, context_for



def wait_event(target: Target, event: object) -> None:
    """Block until one of this backend's copy events has completed."""
    if event is None:
        return
    if target.backend == "opencl":
        from ...opencl.oop.event import Event

        Event.wait([event])
        return
    if target.backend == "metal":
        event.wait()
        return
    event.wait()


def wait_target(target: Target) -> None:
    """Block until the device has finished every copy and launch queued for it."""
    if target.backend == "cuda":
        from ...cuda.driver import CUDA

        context_for(target).make_current()
        CUDA.cuCtxSynchronize()
    elif target.backend == "hip":
        from ...hip.driver import HIP

        context_for(target).make_current()
        HIP.hipDeviceSynchronize()
    elif target.backend == "opencl":
        context_for(target).sync()
    else:
        # Metal's launch waits on its own command buffer, so the queue is already idle.
        context_for(target).sync()


def sync_queue(target: Target, queue: object) -> None:
    """Block until one execution channel is idle."""
    if target.backend == "metal":
        # A Metal copy is a memcpy on the host; there is no queue to wait for.
        return
    if queue is None:
        wait_target(target)
        return
    if target.backend == "opencl":
        queue.wait()
    else:
        queue.sync()


def event_callback(target: Target, queue: object, event: object,
                   callback: Callable[[int], None]) -> None:
    """Run ``callback(error_code)`` once ``event`` completes, without blocking."""
    if target.backend == "opencl":
        # The event may already have completed by the time the transfer is registered --
        # a small copy is often finished before the enqueue returns -- so the callback
        # list is only consulted for a still-running event. Registering without this
        # check loses the completion entirely and the waiter blocks forever.
        if event.finished:
            callback(event.status)
            return
        event.on_completed_callbacks.append(
            lambda completed, error_code: callback(error_code))
        return
    if target.backend == "metal":
        # A Metal copy is finished by the time it returns (a memcpy, or a command buffer
        # the launch already waited on).
        callback(None)
        return
    # CUDA/HIP: a stream callback is ordered after everything already queued on the stream,
    # which includes the copy whose event was just recorded.
    queue.add_callback(lambda _stream, error_code: callback(error_code))


def error_code_ok(backend: str, error_code: int) -> bool:
    """Whether a completion callback's status means success."""
    if error_code is None:
        return True
    if backend == "cuda":
        from ...cuda.driver import CUresult

        return error_code == CUresult.CUDA_SUCCESS
    if backend == "hip":
        from ...hip.driver import hipError

        return error_code == hipError.SUCCESS
    value = getattr(error_code, "value", error_code)
    return int(value) == 0


def _require_entry_point(backend: str, name: str) -> None:
    """Raise if the driver binding has no such entry point."""
    if backend == "cuda":
        from ...cuda.driver.cuinfo import CUInfo

        present = name in CUInfo.func_signatures
    else:
        # HIP's signature table is module level (`hipinfo.func_signatures`), unlike CUDA's
        # class attribute: the two drivers were written at different times.
        from ...hip.driver import hipinfo

        present = name in hipinfo.func_signatures
    if not present:
        raise NotImplementedError(
            f"the {backend} driver binding has no {name}, so host memory cannot be "
            f"registered for zero-copy; use zero_copy=False")


def is_host_mapped(mem_obj: object) -> bool:
    """Whether a binding's device memory *is* the host array, with no copy involved."""
    return bool(getattr(mem_obj, "_gnp_host_mapped", False))


def release(mem_obj: object) -> None:
    """Release whatever extra hold the binding took (a host registration)."""
    unregister = getattr(mem_obj, "_gnp_unregister", None)
    if unregister is None:
        return
    try:
        unregister()
    finally:
        mem_obj._gnp_unregister = None
        mem_obj._gnp_host_mapped = False


def device_pointer(target: Target, mem_obj: object) -> int:
    """The address the device uses for this binding's memory."""
    handle = mem_obj.id
    if isinstance(handle, int):
        return handle
    return int(getattr(handle, "value", handle))


def upload(target: Target, queue: object, mem_obj: object,
           host_array: np.ndarray) -> Optional[object]:
    """Copy the host array's bytes into the binding, on ``queue``."""
    if is_host_mapped(mem_obj):
        # The device memory already contains these bytes: nothing travels.
        return None
    return mem_obj.set_data(queue, host_array)


def download(target: Target, queue: object, mem_obj: object,
             host_array: np.ndarray) -> Optional[object]:
    """Copy the binding's bytes into the host array, on ``queue``."""
    if is_host_mapped(mem_obj):
        return None
    return mem_obj.read(queue)


def settle(target: Target, queue: object, event: object) -> None:
    """Block until a copy made on ``queue`` has landed (the sync-copy path)."""
    if event is not None:
        wait_event(target, event)
    else:
        sync_queue(target, queue)
