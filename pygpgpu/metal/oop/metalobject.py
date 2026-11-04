"""A base class for objects that wrap one Objective-C handle.

Metal has no release-style destroy beyond dropping the last reference, so this
type mirrors the shape of the other backends' `CUObject`/`CLObject` without
their info-query machinery: an object owns one raw handle (`id`) and releases it
when collected.

**`id` answers a handle wrapper, not a bare `int`.** The numpy layer keys the
per-device copies of an array on `cmd_queue.context.id.value`
(`pygpgpu/numpy/ndarray.py`), which is the ctypes shape OpenCL and CUDA answer
with; a plain `int` there raises `AttributeError: 'int' object has no attribute
'value'`. The wrapper below gives Metal the same protocol while staying a real
integer for the driver, so `obj.id == 0` and `obj.id.value == 0` both work and a
handle can still be passed straight to `objc_msgSend`.
"""
from __future__ import annotations

from typing import Any, Optional

from ..driver.objc import ObjC


class Handle(int):
    """An Objective-C handle that reads like the ctypes value objects.

    It is an `int` subclass, so it can be handed to a ctypes `c_void_p` argument
    and compared against zero without unwrapping; `.value` exists because the
    shared numpy layer reads it.
    """

    @property
    def value(self)->int:
        return int(self)

    @classmethod
    def of(cls, handle:Any)->"Handle":
        """Wrap a handle that may already be a `Handle`, an int or a ctypes value."""
        if isinstance(handle, Handle):
            return handle
        inner:Any = getattr(handle, "value", handle)
        return cls(int(inner))


class MTLObject:
    """One Objective-C object, retained on construction and released on collection."""

    def __init__(self, object_id:int)->None:
        # `alloc` hands back a +1 reference, while handles arriving from the
        # framework (a device from MTLCopyAllDevices, a buffer from a queue) are
        # +0 autoreleased. Retaining unconditionally makes the python wrapper own
        # its handle, so collecting the wrapper is what frees it; the extra
        # reference the framework already held is released by its own pool.
        self._object_id:int = int(ObjC.retain(int(object_id))) if object_id else 0

    @property
    def id(self)->Handle:
        return Handle.of(self._object_id)

    def __del__(self)->None:
        object_id:Optional[int] = getattr(self, "_object_id", None)
        if object_id:
            try:
                ObjC.release(object_id)
            except Exception:  # noqa: BLE001  (a finalizer must not raise)
                pass
            self._object_id = 0

    def __repr__(self)->str:
        return f"{self.__class__.__name__}({self._object_id:#x})"
