# The launch binding: which array is whose argument, the family a kernel object
# belongs to, and which queue a launch lands on. These records carry state in
# self.__dict__ (`_parent`, `_bindings`), which is what _Family.newest_binding
# reads -- so no module here may declare __slots__.
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import ndarray
    from . import ImageDescriptor
import threading
import weakref
from typing import Dict, List, Optional, Tuple
import numpy as np
from .._backends import (is_host_mapped)
from .._devices import Target
from ..transfer import Transfer



class _Binding:
    """One device's copy of one element's bytes."""

    __slots__ = ("mem_obj", "epoch", "pending", "target", "zero_copy", "image_info")

    def __init__(self, mem_obj: object, epoch: int, target: Target, zero_copy: bool,
                 image_info: Optional["ImageDescriptor"] = None) -> None:
        self.mem_obj = mem_obj
        self.epoch = epoch
        self.target = target
        self.zero_copy = zero_copy
        self.image_info = image_info
        self.pending: List[Transfer] = []

    @property
    def mapped(self) -> bool:
        """Whether the device sees the host bytes directly (zero-copy)."""
        return is_host_mapped(self.mem_obj)

    def add_pending(self, transfer: Transfer) -> None:
        self.pending.append(transfer)
        transfer._on_complete(lambda _error, entry=self: entry.drop_pending(transfer))

    def drop_pending(self, transfer: Transfer) -> None:
        try:
            self.pending.remove(transfer)
        except ValueError:
            pass


class _Family:
    """The shared state of one numpy buffer and all of its views."""

    def __init__(self, root: "ndarray", data: np.ndarray, live: bool) -> None:
        self.lock = threading.RLock()
        # Signals a copy or launch currently reading the family's host bytes.
        self.cond = threading.Condition(self.lock)
        self.readers = 0
        self.root = weakref.ref(root)
        self.data = data
        self.live = live
        self.epoch = 1
        # The epoch the host bytes reflect; it lags `epoch` while a device is ahead.
        self.current_epoch = 1
        # Members whose bindings may hold device content (weak, so views may be collected).
        self.writers: List[weakref.ReferenceType["ndarray"]] = []

    # -- reader/writer gate --

    def begin_read(self) -> None:
        with self.lock:
            self.readers += 1

    def end_read(self) -> None:
        with self.cond:
            self.readers -= 1
            if self.readers <= 0:
                self.cond.notify_all()

    def wait_readers(self) -> None:
        """Block until nothing is reading the family's host bytes."""
        with self.cond:
            while self.readers > 0:
                self.cond.wait()

    # -- content bookkeeping --

    def bump(self) -> int:
        """Record a content change: every device copy becomes stale at once."""
        with self.lock:
            self.epoch += 1
            return self.epoch

    def refresh_data(self) -> None:
        """Re-capture ``data`` from the caller's memory after a host write."""
        if self.live:
            return
        root = self.root()
        if root is None:
            return
        np.copyto(self.data, root.view(np.ndarray))

    def note_writer(self, member: "ndarray") -> None:
        with self.lock:
            for reference in self.writers:
                if reference() is member:
                    return
            self.writers.append(weakref.ref(member))

    def newest_binding(self) -> Optional[Tuple["ndarray", _Binding]]:
        """The (member, binding) pair holding the newest device content."""
        best: Optional[Tuple["ndarray", _Binding]] = None
        alive: List[weakref.ReferenceType["ndarray"]] = []
        for reference in self.writers:
            member = reference()
            if member is None:
                continue
            alive.append(reference)
            for binding in member.__dict__.get("_bindings", {}).values():
                if best is None or binding.epoch > best[1].epoch:
                    best = (member, binding)
        self.writers = alive
        return best


def _queue_backend(cmd_queue: object) -> str:
    """Which backend an execution channel belongs to, from its own class."""
    mapping = _queue_classes()
    for cls in type(cmd_queue).__mro__:
        name = mapping.get(cls)
        if name is not None:
            return name
    raise TypeError(
        f"cannot tell which backend a {type(cmd_queue).__name__} belongs to; pass "
        f"backend= explicitly")


# Filled in on first use by _queue_classes (see why it is lazy there).
_QUEUE_CLASSES: Optional[Dict[type, str]] = None


def _queue_classes() -> Dict[type, str]:
    """Map execution-channel classes to backend names, built on first use.

    This cannot be a module-scope table: ``pygpgpu.numpy`` is imported by ``pygpgpu``
    itself, and ``import pygpgpu`` must load no GPU runtime -- importing a backend's
    channel class would load that backend's whole driver stack.
    """
    global _QUEUE_CLASSES
    if _QUEUE_CLASSES is None:
        from ...cuda.oop.stream import Stream as CUDAStream
        from ...hip.oop.stream import Stream as HIPStream
        from ...metal.oop.context import CommandQueue as MetalQueue
        from ...opencl.oop.command_queue import CommandQueue as OpenCLQueue

        _QUEUE_CLASSES = {CUDAStream: "cuda", HIPStream: "hip",
                          MetalQueue: "metal", OpenCLQueue: "opencl"}
    return _QUEUE_CLASSES

# The old per-image dirty flag scheme lived on MemObject; a gnp image binding is
# re-uploaded per launch instead, because an image's device object is created from its
# host pointer and there is no cheap way to tell which of the two is current.
_IMAGE_ALWAYS_UPLOAD: bool = True
