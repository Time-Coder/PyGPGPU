"""Pinning host memory so a device can read it in place.

A zero-copy binding is a *registration* the driver holds over the caller's own bytes plus the
device address that registration maps them to. Two arrays may share pages -- same-size arrays
allocated back to back make one of them straddle a page -- so the table below is keyed by the
registration the driver actually holds, and a range that an existing registration covers
*shares* it instead of asking the driver again:

* the driver refuses a second registration that overlaps one it holds
  (``CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED``), and ``...HostUnregister`` only drops a range
  whose start is the registered one, so the refusal cannot be resolved at the caller's own
  start;
* registering page by page is not an answer either: two separate registrations do not map
  contiguously, so an array straddling the boundary would read through an unmapped device
  address.

A range that only *partly* overlaps has to become one registration over the union, which means
unregistering and registering again -- and a re-registration may move the device address the
existing holders are using (measured on HIP). So every holder is remembered, and a merge
refreshes each one's buffer handle; a holder that is gone (collected) is simply dropped.
"""
from __future__ import annotations
import ctypes
import weakref
from typing import Callable, Dict, List, Optional, Tuple

from typing_extensions import TypedDict


# The zero-copy flags the drivers use, spelled once because both CUDA and HIP name them the
# same way: bit 0 makes the registration portable between contexts, bit 1 asks for a device
# mapping (which is what zero-copy needs).
_HOST_REGISTER_PORTABLE: int = 0x01
_HOST_REGISTER_DEVICEMAP: int = 0x02

# The largest page a host registration has to start on. A registration covers a page range,
# and the array's own address is somewhere inside it; the device pointer for the array is the
# registered range's device address plus the host offset from the range's base.
_PAGE_SIZE: int = 4096

class Registration(TypedDict):
    """One registration the driver holds, and the bindings that rely on it.

    `count` is how many zero-copy bindings use it, and `holders` exists only so a merge can
    re-point the device address of every binding already using it -- held weakly, because a
    binding owns its buffer and the buffer must stay collectable.
    """

    size: int
    count: int
    holders: List[Tuple[int, "weakref.ref"]]


_registered_ranges: Dict[Tuple[str, int], Registration] = {}
# Which registration each caller's own (start, size) landed on. A caller whose range an earlier
# registration covers is not a key of its own, so its release -- and a merge -- has to find the
# entry that holds it.
_attachments: Dict[Tuple[str, int, int], Tuple[str, int]] = {}
_registration_lock = None


def _lock():
    """The lock guarding the tables above (created lazily, like the tables)."""
    global _registration_lock
    if _registration_lock is None:
        import threading

        _registration_lock = threading.RLock()
    return _registration_lock


def _covering_entry(backend: str, start: int, size: int) -> Optional[Tuple[str, int]]:
    """The key of the registration that covers `[start, start + size)` entirely, or None."""
    end: int = start + size
    for key, entry in _registered_ranges.items():
        if key[0] != backend:
            continue
        if key[1] <= start and end <= key[1] + entry["size"]:
            return key
    return None


def _overlapping_entries(backend: str, start: int, end: int) -> List[Tuple[str, int]]:
    """The keys of every registration of this backend that overlaps `[start, end)`."""
    overlapping: List[Tuple[str, int]] = []
    for key, entry in _registered_ranges.items():
        if key[0] != backend:
            continue
        other_start: int = key[1]
        other_end: int = other_start + entry["size"]
        if start < other_end and other_start < end:
            overlapping.append(key)
    return overlapping


def _device_pointer(backend: str, base: int) -> int:
    """The device address a registration starting at `base` maps its host bytes to."""
    if backend == "cuda":
        from ...cuda.driver import CUDA, CUdeviceptr

        device = CUdeviceptr(0)
        CUDA.cuMemHostGetDevicePointer_v2(device, ctypes.c_void_p(base), 0)
        return int(device.value)
    from ...hip.driver import HIP, CUdeviceptr

    device = CUdeviceptr(0)
    HIP.hipHostGetDevicePointer(device, ctypes.c_void_p(base), 0)
    return int(device.value)


def _set_buffer_address(backend: str, buffer: object, address: int) -> None:
    """Move a registered buffer's handle to the address a re-registration gave it.

    `CUObject.id` reads `_id`, and `_kernel_args` reads `id` per launch, so the next launch
    uses the new mapping. The object's entry in the driver's handle registry is keyed by the
    old address and is left alone: it is a weak reference, nothing looks a buffer up by
    address, and the entry disappears with the buffer.
    """
    if backend == "cuda":
        from ...cuda.driver import CUdeviceptr
    else:
        from ...hip.driver import CUdeviceptr

    buffer._id = CUdeviceptr(address)  # type: ignore[attr-defined]


def _refresh_holders(backend: str, key: Tuple[str, int],
                     holders: List[Tuple[int, "weakref.ref"]]) -> None:
    """Re-point every live holder of a registration that had to be made again."""
    alive: List[Tuple[int, "weakref.ref"]] = []
    device_base: Optional[int] = None
    for host_ptr, holder in holders:
        buffer: object = holder()
        if buffer is None:
            continue
        if device_base is None:
            device_base = _device_pointer(backend, key[1])
        _set_buffer_address(backend, buffer, device_base + (int(host_ptr) - key[1]))
        alive.append((host_ptr, holder))
    holders[:] = alive


def _replace_range(backend: str, start: int, size: int) -> None:
    """Register a range, taking over a registration this table does not know about.

    A caller's own ``cuMemHostRegister`` or the torch bridge's `page_locked` can hold the
    page; the refusal is the only reliable "already pinned" signal, so the range is dropped
    and taken over -- the page itself never moves, and this table now owns it.
    """
    try:
        _driver_register(backend, start, size)
    except BaseException:
        _driver_unregister(backend, start)
        _driver_register(backend, start, size)


def _register_range(backend: str, start: int, size: int) -> int:
    """Ensure one registration covers `[start, start + size)`, and answer its base address.

    A covering registration gains a holder, which is the common case and asks the driver
    nothing. A range that overlaps but is not covered is merged into one registration over
    the union, and every existing holder's device address is refreshed; an unrelated range is
    registered as it stands.
    """
    with _lock():
        end: int = start + size
        covering: Optional[Tuple[str, int]] = _covering_entry(backend, start, size)
        if covering is not None:
            entry: Registration = _registered_ranges[covering]
            entry["count"] += 1
            _attachments[(backend, start, size)] = covering
            return covering[1]

        overlapping: List[Tuple[str, int]] = _overlapping_entries(backend, start, end)
        if not overlapping:
            _replace_range(backend, start, size)
            key: Tuple[str, int] = (backend, start)
            _registered_ranges[key] = {"size": size, "count": 1, "holders": []}
            _attachments[(backend, start, size)] = key
            return start

        old: List[Tuple[Tuple[str, int], Registration]] = [
            (key, dict(_registered_ranges[key])) for key in overlapping]  # type: ignore[typeddict-item]
        union_start: int = min([start] + [key[1] for key, _entry in old])
        union_end: int = max([end] + [key[1] + _entry["size"] for key, _entry in old])
        holders: List[Tuple[int, "weakref.ref"]] = [
            holder for _key, entry in old for holder in entry["holders"]]
        count: int = sum(entry["count"] for _key, entry in old) + 1
        for key, _entry in old:
            _driver_unregister(backend, key[1])
        try:
            _replace_range(backend, union_start, union_end - union_start)
        except BaseException:
            # The union could not be registered (an outside pin that starts elsewhere). Put
            # the ranges back so the holders keep working, then let the caller see the failure.
            for key, entry in old:
                try:
                    _driver_register(backend, key[1], entry["size"])
                except BaseException:
                    pass
            raise
        for key, _entry in old:
            del _registered_ranges[key]
        new_key: Tuple[str, int] = (backend, union_start)
        _registered_ranges[new_key] = {
            "size": union_end - union_start, "count": count, "holders": holders}
        for attachment, attached in list(_attachments.items()):
            if attached in overlapping:
                _attachments[attachment] = new_key
        _attachments[(backend, start, size)] = new_key
        _refresh_holders(backend, new_key, holders)
        return union_start


def _attach_holder(backend: str, start: int, size: int, host_ptr: int,
                   buffer: object) -> None:
    """Remember `buffer` as a holder of the registration covering `[start, start + size)`."""
    with _lock():
        key: Optional[Tuple[str, int]] = _attachments.get((backend, start, size))
        if key is None:
            return
        entry: Optional[Registration] = _registered_ranges.get(key)
        if entry is None:
            return
        entry["holders"].append((int(host_ptr), weakref.ref(buffer)))


def _unregister_range(backend: str, start: int, size: int,
                      buffer: Optional[object]=None) -> None:
    """Release one hold on a range, unpinning the registration with its last holder.

    `buffer` is optional and only prunes the holder list early: a dead buffer's weak
    reference is harmless, and `_refresh_holders` drops it anyway. The release callable must
    **not** capture the buffer -- a closure over it would keep the buffer alive through its
    own finalizer, so no zero-copy binding would ever be collected or unpinned.
    """
    with _lock():
        key: Optional[Tuple[str, int]] = _attachments.get((backend, start, size))
        if key is None:
            # A release with no recorded attachment (a caller that never registered through
            # this table) names its own start, which is the only registration it could own.
            key = (backend, start)
        entry: Optional[Registration] = _registered_ranges.get(key)
        if entry is None:
            return
        if buffer is not None:
            entry["holders"] = [holder for holder in entry["holders"]
                                if holder[1]() is not None and holder[1]() is not buffer]
        entry["count"] -= 1
        if entry["count"] > 0:
            return
        del _registered_ranges[key]
        for attachment, attached in list(_attachments.items()):
            if attached == key:
                del _attachments[attachment]
        _driver_unregister(backend, key[1])


def _driver_register(backend: str, start: int, size: int) -> None:
    """Make the driver pin one host range (the call that can refuse)."""
    if backend == "cuda":
        from ...cuda.driver import CUDA

        CUDA.cuMemHostRegister_v2(ctypes.c_void_p(start), size,
                                  _HOST_REGISTER_PORTABLE | _HOST_REGISTER_DEVICEMAP)
        return
    from ...hip.driver import HIP

    HIP.hipHostRegister(ctypes.c_void_p(start), size,
                        _HOST_REGISTER_PORTABLE | _HOST_REGISTER_DEVICEMAP)


def _driver_unregister(backend: str, start: int) -> None:
    """Release one host range, swallowing a refusal (it may not be pinned at all)."""
    try:
        if backend == "cuda":
            from ...cuda.driver import CUDA

            CUDA.cuMemHostUnregister(ctypes.c_void_p(start))
            return
        from ...hip.driver import HIP

        HIP.hipHostUnregister(ctypes.c_void_p(start))
    except BaseException:
        pass


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
