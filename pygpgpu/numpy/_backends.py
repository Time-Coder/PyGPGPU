"""The per-backend half of a gnp copy: allocate, transfer, wait, release.

Everything here is about *one* operation on *one* device. ``ndarray.py`` decides which
operation to perform; this module knows how each backend spells it.

Three things are load-bearing:

* **Every wait is explicit and local.** A copy returns the backend's own event and the
  caller registers a completion callback on it; nothing blocks "just in case". The
  places that genuinely depend on a result block through :func:`wait_event`,
  :func:`sync_queue` / :func:`wait_target`.
* **Zero-copy is only offered where the API guarantees it.** Metal's ``StorageModeShared``
  buffer *is* host memory (the code checks that the addresses agree), and CUDA/HIP
  mapped pinned memory is host memory at a device address that
  ``cuMemHostGetDevicePointer``/``hipHostGetDevicePointer`` answers for. OpenCL's
  ``CL_MEM_USE_HOST_PTR`` is explicitly *not* that: the runtime is free to keep a private
  copy, so a request there raises instead of handing back an array whose writes might
  never reach the device.
* **A refusal names the backend and the missing capability.** A capability the runtime
  does not have must fail here, where the reason is known, rather than as a fault on the
  device later.
"""
from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional, Tuple

import numpy as np

from ._devices import Target, context_for


# The zero-copy flags the drivers use, spelled once because both CUDA and HIP name them
# the same way: bit 0 makes the registration portable between contexts, bit 1 asks for a
# device mapping (which is what zero-copy needs).
_HOST_REGISTER_PORTABLE: int = 0x01
_HOST_REGISTER_DEVICEMAP: int = 0x02

# The largest page a host registration has to start on. A registration covers a page
# range, and the array's own address is somewhere inside it; the device pointer for the
# array is the registered range's device address plus that offset.
_PAGE_SIZE: int = 4096

# Page ranges currently registered with a backend, keyed by (backend, page_start): a
# registration covers a whole page, so two arrays that start on the same page share one
# pin. Registering the same range twice makes the driver refuse the second call
# (CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED), which is why this table exists.
_registered_ranges: dict = {}
_registration_lock = None


def _lock():
    """The lock guarding `_registered_ranges` (created lazily, like the table)."""
    global _registration_lock
    if _registration_lock is None:
        import threading

        _registration_lock = threading.RLock()
    return _registration_lock


def _register_range(backend: str, start: int, size: int) -> None:
    """Register a page range, sharing one pin between arrays that land on the same page.

    Two things make this fiddly, and both are driver behaviour rather than ours:

    * registering a range the driver already has makes it refuse the call
      (``CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED``), so the table above exists to say
      "this page is already pinned" before the driver is asked;
    * that refusal is also the only reliable signal that a registration *outside this
      table* still exists -- one a caller made directly, or one this process made before
      the table was consulted. The answer there is not to give up: the driver is told to
      drop it and the registration is made again over the range this caller needs, which
      is the state the caller asked for either way.

    A range already pinned over fewer bytes is widened the same way; the holder of the
    narrower range keeps its device pointer valid, because the page itself never moves.
    """
    with _lock():
        key = (backend, start)
        entry = _registered_ranges.get(key)
        if entry is not None and entry["size"] >= size:
            entry["count"] += 1
            return

        if entry is not None:
            _driver_unregister(backend, start)

        try:
            _driver_register(backend, start, size)
        except BaseException:
            # The page is pinned by something this table does not know about. Drop that
            # registration and take the page over, because a caller asking for zero-copy
            # has to end up with *a* registration over its own bytes.
            try:
                _driver_unregister(backend, start)
                _driver_register(backend, start, size)
            except BaseException:
                raise

        _registered_ranges[key] = {"count": 1, "size": size}


def _driver_register(backend: str, start: int, size: int) -> None:
    """Make the driver pin one host page range (the call that can refuse)."""
    if backend == "cuda":
        from ..cuda.driver import CUDA

        CUDA.cuMemHostRegister_v2(ctypes.c_void_p(start), size,
                                  _HOST_REGISTER_PORTABLE | _HOST_REGISTER_DEVICEMAP)
        return
    from ..hip.driver import HIP

    HIP.hipHostRegister(ctypes.c_void_p(start), size,
                        _HOST_REGISTER_PORTABLE | _HOST_REGISTER_DEVICEMAP)


def _driver_unregister(backend: str, start: int) -> None:
    """Release one host page range, swallowing a refusal (it may not be pinned at all)."""
    try:
        if backend == "cuda":
            from ..cuda.driver import CUDA

            CUDA.cuMemHostUnregister(ctypes.c_void_p(start))
            return
        from ..hip.driver import HIP

        HIP.hipHostUnregister(ctypes.c_void_p(start))
    except BaseException:
        pass


def _unregister_range(backend: str, start: int) -> None:
    """Release one hold on a page range, unregistering it with the last holder."""
    with _lock():
        key = (backend, start)
        entry = _registered_ranges.get(key)
        if entry is None:
            return
        entry["count"] -= 1
        if entry["count"] > 0:
            return
        del _registered_ranges[key]
        _driver_unregister(backend, start)


# ---------- waiting ----------

def wait_event(target: Target, event: Any) -> None:
    """Block until one of this backend's copy events has completed."""
    if event is None:
        return
    if target.backend == "opencl":
        from ..opencl.oop.event import Event

        Event.wait([event])
        return
    if target.backend == "metal":
        event.wait()
        return
    event.wait()


def wait_target(target: Target) -> None:
    """Block until the device has finished every copy and launch queued for it."""
    if target.backend == "cuda":
        from ..cuda.driver import CUDA

        context_for(target).make_current()
        CUDA.cuCtxSynchronize()
    elif target.backend == "hip":
        from ..hip.driver import HIP

        context_for(target).make_current()
        HIP.hipDeviceSynchronize()
    elif target.backend == "opencl":
        context_for(target).sync()
    else:
        # Metal's launch waits on its own command buffer, so the queue is already idle.
        context_for(target).sync()


def sync_queue(target: Target, queue: Any) -> None:
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


def event_callback(target: Target, queue: Any, event: Any,
                   callback: Callable[[Any], None]) -> None:
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


def error_code_ok(backend: str, error_code: Any) -> bool:
    """Whether a completion callback's status means success."""
    if error_code is None:
        return True
    if backend == "cuda":
        from ..cuda.driver import CUresult

        return error_code == CUresult.CUDA_SUCCESS
    if backend == "hip":
        from ..hip.driver import hipError

        return error_code == hipError.SUCCESS
    value = getattr(error_code, "value", error_code)
    return int(value) == 0


# ---------- allocation ----------

def allocate(target: Target, size: int, zero_copy: bool = False,
             host_ptr: Optional[int] = None) -> Any:
    """Create the device object a binding copies through."""
    context = context_for(target)
    context.make_current()
    if zero_copy:
        return _allocate_zero_copy(target, context, size, host_ptr)
    buffer = _create_buffer(target, context, size)
    buffer._gnp_host_mapped = False
    return buffer


def _create_buffer(target: Target, context: Any, size: int) -> Any:
    """Allocate a device buffer, in whatever spelling this backend's API needs.

    OpenCL's ``clCreateBuffer`` has no default flags -- passing None reaches ctypes as a
    missing value and fails as an ArgumentError -- so it is given the explicit
    read/write flag every other backend defaults to.
    """
    if target.backend == "opencl":
        from ..opencl.driver import cl_mem_flags

        return context.create_buffer(size=size, flags=cl_mem_flags.CL_MEM_READ_WRITE)
    return context.create_buffer(size=size)


def _allocate_zero_copy(target: Target, context: Any, size: int,
                        host_ptr: Optional[int]) -> Any:
    """Device memory that *is* the host array's own bytes."""
    if host_ptr is None:
        raise ValueError(
            "a zero-copy binding needs the host array it maps, but no host address was "
            "supplied")
    if target.backend == "metal":
        return _allocate_zero_copy_metal(context, size, host_ptr)
    if target.backend == "cuda":
        return _allocate_zero_copy_cuda(context, size, host_ptr)
    if target.backend == "hip":
        return _allocate_zero_copy_hip(context, size, host_ptr)
    supported, reason = zero_copy_supported(target)
    raise NotImplementedError(reason)


def _allocate_zero_copy_metal(context: Any, size: int, host_ptr: int) -> Any:
    from ..metal.oop.buffer import Buffer

    buffer = Buffer(context, size=size)
    contents = int(buffer._contents())
    if contents == 0:
        raise RuntimeError("Metal returned a shared buffer with no host address")
    # StorageModeShared already is host memory (the CPU writes the GPU's bytes), so an
    # exact address match is what proves the array's own memory is what the device sees.
    buffer._gnp_host_mapped = (contents == int(host_ptr))
    if not buffer._gnp_host_mapped:
        # A Metal buffer of our own still needs the array's bytes; the binding keeps a
        # copy in that case, which the caller asked not to happen.
        raise NotImplementedError(
            "Metal could not hand back a shared buffer at the array's own address, so "
            "zero-copy cannot be guaranteed for this array")
    return buffer


def _aligned_range(host_ptr: int, size: int) -> Tuple[int, int, int]:
    """``(page_start, registered_size, offset)`` for an array inside a page range."""
    start = (int(host_ptr) // _PAGE_SIZE) * _PAGE_SIZE
    offset = int(host_ptr) - start
    registered = ((offset + int(size) + _PAGE_SIZE - 1) // _PAGE_SIZE) * _PAGE_SIZE
    return start, registered, offset


def _allocate_zero_copy_cuda(context: Any, size: int, host_ptr: int) -> Any:
    from ..cuda.driver import CUDA, CUdeviceptr

    # A registration is made on the *current* context, so this one has to be current
    # before the driver sees the request.
    context.make_current()

    # The device has to be able to map host memory at all; asking first turns a
    # would-be fault into a sentence. `cuDeviceGetAttribute` fills an int through a
    # pointer, and its attribute argument is the enum's numeric value.
    try:
        from ctypes import byref, c_int

        from ..cuda.driver.cutypes import CUdevice_attribute

        can_map = c_int(0)
        CUDA.cuDeviceGetAttribute(byref(can_map),
                                  int(CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY),
                                  context.device.id)
        can_map = can_map.value
    except BaseException:
        # A driver too old to answer the question is assumed to map host memory, and the
        # registration itself is the real test.
        can_map = 1
    if can_map != 1:
        raise NotImplementedError(
            f"device '{context.device.name}' reports "
            f"CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY == 0, so it cannot read host memory "
            f"directly; zero-copy is unavailable on it")

    start, registered, offset = _aligned_range(host_ptr, size)
    _register_range("cuda", start, registered)
    device_base = CUdeviceptr(0)
    try:
        CUDA.cuMemHostGetDevicePointer_v2(device_base, ctypes.c_void_p(start), 0)
    except BaseException:
        _unregister_range("cuda", start)
        raise
    device_address = int(device_base.value) + offset

    from ..cuda.oop.buffer import Buffer
    from ..cuda.oop.mem_object import MemObject

    buffer = Buffer.__new__(Buffer)
    MemObject.__init__(buffer, CUdeviceptr(device_address), None, None, size)
    _replace_free_with_unregister(buffer, "cuda", start, _unregister_range)
    return buffer


def _replace_free_with_unregister(buffer: Any, backend: str, start: int,
                                  unregister_range: Callable[[str, int], None]) -> None:
    """Make a registered buffer's release unregister its page range, not free its pointer.

    A zero-copy binding's handle is a *device address of host memory*: it was never
    allocated with ``cuMemAlloc``/``hipMalloc``, so the buffer class's own finalizer
    (which frees the pointer) must not run. Its release is re-pointed at the
    registration instead, which is the resource the binding actually holds -- and the
    buffer keeps its class, so a launch treats it exactly like any other buffer.
    """
    buffer._gnp_unregister = lambda: unregister_range(backend, start)
    buffer._gnp_host_mapped = True
    # `weakref.finalize` holds the release callable it was given at construction, so the
    # one that was installed has to be replaced rather than shadowed.
    try:
        buffer._finalizer.detach()
    except BaseException:
        pass
    import weakref

    weakref.finalize(buffer, _safe_release, buffer._gnp_unregister)


def _safe_release(action: Callable[[], None]) -> None:
    try:
        action()
    except BaseException:
        pass


def _allocate_zero_copy_hip(context: Any, size: int, host_ptr: int) -> Any:
    from ..hip.driver import HIP, CUdeviceptr

    start, registered, offset = _aligned_range(host_ptr, size)
    _register_range("hip", start, registered)
    device_base = CUdeviceptr(0)
    try:
        HIP.hipHostGetDevicePointer(device_base, ctypes.c_void_p(start), 0)
    except BaseException:
        _unregister_range("hip", start)
        raise
    device_address = int(device_base.value) + offset

    from ..hip.oop.buffer import Buffer
    from ..hip.oop.mem_object import MemObject

    buffer = Buffer.__new__(Buffer)
    MemObject.__init__(buffer, CUdeviceptr(device_address), None, None, size)
    _replace_free_with_unregister(buffer, "hip", start, _unregister_range)
    return buffer


def zero_copy_supported(target: Target) -> Tuple[bool, str]:
    """Whether this backend can bind host memory to the device, and why not."""
    return zero_copy_backend_support(target.backend)


def zero_copy_backend_support(backend: str) -> Tuple[bool, str]:
    """Whether a backend *can* bind host memory, independent of this machine's devices.

    A capability question rather than a device question: ``tests`` and a caller choosing
    a backend both need to know before a device is resolved -- and resolving one can fail
    for an unrelated reason (Metal on a non-Mac, a backend with no devices at all).
    """
    if backend == "metal":
        # Metal's StorageModeShared *is* host memory; the allocation checks the address.
        return True, ""
    if backend == "cuda":
        try:
            _require_entry_point("cuda", "cuMemHostRegister_v2")
            _require_entry_point("cuda", "cuMemHostGetDevicePointer_v2")
        except NotImplementedError as error:
            return False, str(error)
        return True, ""
    if backend == "hip":
        try:
            _require_entry_point("hip", "hipHostRegister")
            _require_entry_point("hip", "hipHostGetDevicePointer")
        except NotImplementedError as error:
            return False, str(error)
        return True, ""
    if backend == "opencl":
        return False, (
            "OpenCL cannot guarantee zero-copy: CL_MEM_USE_HOST_PTR lets the runtime keep "
            "a private copy of the host data, so a kernel write might never reach the "
            "caller's array. Use zero_copy=False (the default) on this backend, or run the "
            "kernel on cuda/hip/metal")
    return False, f"zero-copy is not implemented for the {backend} backend"


# The public spelling of zero_copy_supported, which the array re-exports.
zero_copy_support = zero_copy_supported


def allocate_image(target: Target, queue: Any, image_info: Any) -> Any:
    """Create the backend's image object for an image-typed kernel argument."""
    context = context_for(target)
    context.make_current()
    return context.create_image(image_info)


def _require_entry_point(backend: str, name: str) -> None:
    """Raise if the driver binding has no such entry point."""
    if backend == "cuda":
        from ..cuda.driver.cuinfo import CUInfo

        present = name in CUInfo.func_signatures
    else:
        # HIP's signature table is module level (`hipinfo.func_signatures`), unlike CUDA's
        # class attribute: the two drivers were written at different times.
        from ..hip.driver import hipinfo

        present = name in hipinfo.func_signatures
    if not present:
        raise NotImplementedError(
            f"the {backend} driver binding has no {name}, so host memory cannot be "
            f"registered for zero-copy; use zero_copy=False")


def is_host_mapped(mem_obj: Any) -> bool:
    """Whether a binding's device memory *is* the host array, with no copy involved."""
    return bool(getattr(mem_obj, "_gnp_host_mapped", False))


def release(mem_obj: Any) -> None:
    """Release whatever extra hold the binding took (a host registration)."""
    unregister = getattr(mem_obj, "_gnp_unregister", None)
    if unregister is None:
        return
    try:
        unregister()
    finally:
        mem_obj._gnp_unregister = None
        mem_obj._gnp_host_mapped = False


def device_pointer(target: Target, mem_obj: Any) -> int:
    """The address the device uses for this binding's memory."""
    handle = mem_obj.id
    if isinstance(handle, int):
        return handle
    return int(getattr(handle, "value", handle))


# ---------- the copies ----------

def upload(target: Target, queue: Any, mem_obj: Any, host_array: np.ndarray) -> Any:
    """Copy the host array's bytes into the binding, on ``queue``."""
    if is_host_mapped(mem_obj):
        # The device memory already contains these bytes: nothing travels.
        return None
    return mem_obj.set_data(queue, host_array)


def download(target: Target, queue: Any, mem_obj: Any, host_array: np.ndarray) -> Any:
    """Copy the binding's bytes into the host array, on ``queue``."""
    if is_host_mapped(mem_obj):
        return None
    return mem_obj.read(queue)


def settle(target: Target, queue: Any, event: Any) -> None:
    """Block until a copy made on ``queue`` has landed (the sync-copy path)."""
    if event is not None:
        wait_event(target, event)
    else:
        sync_queue(target, queue)


# ---------- host memory for a torch tensor ----------

def page_locked(host_ptr: int, size: int) -> Optional[Callable[[], None]]:
    """Page-lock a host range for CUDA/HIP, returning a release callable.

    Only used by the torch bridge: a torch tensor's storage is ordinary host memory and
    pinning it lets an upload run as a real asynchronous DMA instead of a staged copy.
    A backend that cannot pin (or that refuses this particular range) answers None, and
    the transfer falls back to an ordinary host buffer.
    """
    start, registered, _offset = _aligned_range(host_ptr, size)
    attempt = _pin_range(start, registered)
    return attempt


def _pin_range(start: int, registered: int) -> Optional[Callable[[], None]]:
    """Try to pin a range on every available backend; None when nothing can."""
    pins: list = []
    from ..kernel.decorator import DEFAULT_BACKEND_ORDER, _probe_backend

    for backend in DEFAULT_BACKEND_ORDER:
        if _probe_backend(backend) is None:
            continue
        if backend == "cuda":
            try:
                from ..cuda.driver import CUDA

                CUDA.cuMemHostRegister_v2(ctypes.c_void_p(start), registered,
                                          _HOST_REGISTER_PORTABLE)
            except BaseException:
                continue

            def unpin_cuda(start: int = start) -> None:
                try:
                    from ..cuda.driver import CUDA

                    CUDA.cuMemHostUnregister(ctypes.c_void_p(start))
                except BaseException:
                    pass

            pins.append(unpin_cuda)
        elif backend == "hip":
            try:
                from ..hip.driver import HIP

                HIP.hipHostRegister(ctypes.c_void_p(start), registered, 0)
            except BaseException:
                continue

            def unpin_hip(start: int = start) -> None:
                try:
                    from ..hip.driver import HIP

                    HIP.hipHostUnregister(ctypes.c_void_p(start))
                except BaseException:
                    pass

            pins.append(unpin_hip)
        break

    if not pins:
        return None

    def unpin() -> None:
        for release_pin in pins:
            release_pin()

    return unpin
