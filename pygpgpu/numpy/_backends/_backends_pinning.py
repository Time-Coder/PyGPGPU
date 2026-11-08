# Pinning host memory so a device can read it in place: the registration the
# driver keeps, and the page-locked range a zero-copy buffer is built on.
from __future__ import annotations
import ctypes
from typing import Callable, Optional, Tuple



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
        from ...cuda.driver import CUDA

        CUDA.cuMemHostRegister_v2(ctypes.c_void_p(start), size,
                                  _HOST_REGISTER_PORTABLE | _HOST_REGISTER_DEVICEMAP)
        return
    from ...hip.driver import HIP

    HIP.hipHostRegister(ctypes.c_void_p(start), size,
                        _HOST_REGISTER_PORTABLE | _HOST_REGISTER_DEVICEMAP)


def _driver_unregister(backend: str, start: int) -> None:
    """Release one host page range, swallowing a refusal (it may not be pinned at all)."""
    try:
        if backend == "cuda":
            from ...cuda.driver import CUDA

            CUDA.cuMemHostUnregister(ctypes.c_void_p(start))
            return
        from ...hip.driver import HIP

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


def _aligned_range(host_ptr: int, size: int) -> Tuple[int, int, int]:
    """``(page_start, registered_size, offset)`` for an array inside a page range."""
    start = (int(host_ptr) // _PAGE_SIZE) * _PAGE_SIZE
    offset = int(host_ptr) - start
    registered = ((offset + int(size) + _PAGE_SIZE - 1) // _PAGE_SIZE) * _PAGE_SIZE
    return start, registered, offset


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
    from ...kernel.decorator import DEFAULT_BACKEND_ORDER, _probe_backend

    for backend in DEFAULT_BACKEND_ORDER:
        if _probe_backend(backend) is None:
            continue
        if backend == "cuda":
            try:
                from ...cuda.driver import CUDA

                CUDA.cuMemHostRegister_v2(ctypes.c_void_p(start), registered,
                                          _HOST_REGISTER_PORTABLE)
            except BaseException:
                continue

            def unpin_cuda(start: int = start) -> None:
                try:
                    from ...cuda.driver import CUDA

                    CUDA.cuMemHostUnregister(ctypes.c_void_p(start))
                except BaseException:
                    pass

            pins.append(unpin_cuda)
        elif backend == "hip":
            try:
                from ...hip.driver import HIP

                HIP.hipHostRegister(ctypes.c_void_p(start), registered, 0)
            except BaseException:
                continue

            def unpin_hip(start: int = start) -> None:
                try:
                    from ...hip.driver import HIP

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
