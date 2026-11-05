"""The device-resident numpy array (``pygpgpu.numpy.ndarray``, "gnp").

A gnp array is a numpy array whose contents may live on one or more devices instead of in
host memory. Passing it to a kernel does **not** copy it back to the CPU afterwards: the
bytes stay where the device put them, and a copy back happens only when something actually
needs them on the host -- a numpy operation, an element read, ``to_host()``, or a
transfer to another device.

How the state is tracked
-----------------------
A *family* is one numpy buffer plus every view of it (``a``, ``a[1]``, ``a.T``, ...). The
array at the top of the family is its **root**, and it owns a :class:`_Family`:

* ``data`` -- the root's bytes as one contiguous numpy array. When the root owns its
  memory this *is* that memory, so no staging copy exists; when it is a strided view of
  something else, ``data`` is a contiguous copy and ``live`` is False.
* ``epoch`` -- a counter bumped whenever the family's contents change, on the host or on
  a device. A device copy records the epoch its buffer holds, so "is this binding
  current?" is one integer comparison and **one** bump invalidates every other device
  copy at once. ``current_epoch`` records which epoch the host bytes reflect: it lags
  ``epoch`` exactly while a device holds content the host does not have.
* ``readers`` -- a reader/writer gate held while a copy or launch reads the family's host
  bytes. It is what makes "the caller wrote this array while an upload of it was still
  queued" safe instead of silently racing.

Each element (each view object) holds at most one :class:`_Binding` per (backend, device,
layout): the device buffer, the epoch it holds, and the host bytes it was filled from. A
binding is *current* when its epoch equals the family's and *stale* otherwise. Refreshing
a stale binding means "bring the host bytes up to date, then upload", and bringing the
host up to date means downloading from whichever binding holds the newest epoch. That one
rule answers every copy question the API has:

================  ===========================================================
situation         result
================  ===========================================================
same device       current -> nothing moves; stale -> upload only
other device      download to host, upload to the other device (host staging)
another backend   the same, because each backend has its own buffer type
host write        every binding is stale; the next launch uploads
device write      the host bytes are stale; a host read downloads (``to_host``)
================  ===========================================================

Waiting
-------
``to_device``/``to_host`` return a :class:`Transfer` immediately, and the copy runs on the
backend's asynchronous stream/queue. Nothing waits "just in case"; the engine waits exactly
where a result is depended on:

* before reading host bytes another in-flight copy is still writing;
* before launching on a binding whose upload is still queued;
* before any host-side read of the array (``arr[i]``, ``a + 1``, ``tolist()``),
  which is what makes an un-awaited ``to_host()`` still correct.

Cross-device correctness
------------------------
A copy between two devices always goes through the host: two backends' buffers are
unrelated types, and even two devices of one backend may live in different contexts. What
the engine cannot make correct it raises instead of guessing -- a zero-copy request on a
backend whose API does not guarantee it raises ``NotImplementedError`` naming the backend
(see ``_backends.zero_copy_supported``), rather than handing back memory the device might
not be looking at.
"""
from __future__ import annotations

import functools
import threading
import weakref
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import DTypeLike

from ..constants import KernelFlags
from ..utils import get_print_info
from ._backends import (allocate, allocate_image, download, error_code_ok, event_callback,
                        is_host_mapped, release, sync_queue, upload, wait_event,
                        zero_copy_support)
from ._devices import Target, copy_queue, release_copy_queue, target_of
from .transfer import Transfer


if TYPE_CHECKING:
    from typing_extensions import Protocol

    class ImageDescriptor(Protocol):
        """What a launch needs to know about an image argument.

        The concrete descriptor belongs to whichever backend owns the image, so the
        array layer describes only the three facts it reads: the shape it must match,
        the flags the texture declares, and the dtype.
        """
        shape: Tuple[int, ...]
        flags: int
        dtype: DTypeLike


# ---------- per-device binding ----------

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


# ---------- the family ----------

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
        from ..cuda.oop.stream import Stream as CUDAStream
        from ..hip.oop.stream import Stream as HIPStream
        from ..metal.oop.context import CommandQueue as MetalQueue
        from ..opencl.oop.command_queue import CommandQueue as OpenCLQueue

        _QUEUE_CLASSES = {CUDAStream: "cuda", HIPStream: "hip",
                          MetalQueue: "metal", OpenCLQueue: "opencl"}
    return _QUEUE_CLASSES

# The old per-image dirty flag scheme lived on MemObject; a gnp image binding is
# re-uploaded per launch instead, because an image's device object is created from its
# host pointer and there is no cheap way to tell which of the two is current.
_IMAGE_ALWAYS_UPLOAD: bool = True


class ndarray(np.ndarray):
    """A numpy array whose bytes may live on a device.

    It behaves like ``numpy.ndarray`` everywhere: indexing, arithmetic, ``np.sum`` and
    friends all work, because each of those operations first brings the bytes back to the
    host. What it adds is that a kernel call does not: the bytes stay on the device across
    any number of launches, and only a host read -- or an explicit transfer -- brings them
    back.
    """

    # ---------- construction ----------

    def __new__(cls, input_array: object) -> "ndarray":
        obj = np.asarray(input_array).view(cls)
        obj._parent = None
        obj._children = []
        obj._bindings = {}
        obj._family = None
        obj._staging = None
        obj._staging_epoch = 0
        obj._arg_name = ""
        obj._should_copy = True
        return obj

    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        if obj is None:
            return
        self._bindings = {}
        self._family = None
        self._staging = None
        self._staging_epoch = 0
        self._arg_name = ""
        self._should_copy = True
        self._children = []
        # A plain numpy array is never a family root: what a view over it shares is a
        # buffer numpy already owns, and the gnp array built over it *is* the root. Only a
        # view that aliases another gnp element joins that element's family.
        parent = obj if isinstance(obj, ndarray) and _aliases(obj, self) else None
        self._parent = parent
        if parent is not None:
            parent._children.append(self)

    # ---------- family plumbing ----------

    def _root(self) -> "ndarray":
        current: "ndarray" = self
        for _ in range(64):
            parent = getattr(current, "_parent", None)
            if parent is None:
                return current
            current = parent
        return current

    def _make_family(self) -> "_Family":
        """Create this family's state, choosing the layout its root can express."""
        root = self._root()
        root_array = root.view(np.ndarray)
        if root.flags["C_CONTIGUOUS"]:
            family = _Family(root, root_array, live=True)
        else:
            # A strided root cannot be handed to a device as one contiguous buffer, so the
            # family keeps a contiguous copy and refreshes it on every host write.
            family = _Family(root, np.ascontiguousarray(root_array), live=False)
        root._family = family
        if root is not self:
            self._family = family
        return family

    @property
    def _state(self) -> "_Family":
        """The family this element belongs to, created on first use."""
        family = self.__dict__.get("_family")
        if family is not None:
            return family
        root = self._root()
        family = root.__dict__.get("_family")
        if family is None:
            return self._make_family()
        self._family = family
        return family

    def _member_bytes(self) -> Tuple[np.ndarray, int]:
        """``(host_array, byte_offset)``: the bytes this element occupies in the family.

        For the root (and a view whose layout is the family's own) the array is
        ``family.data`` itself; for any other view it is a contiguous copy kept on the
        element, rebuilt whenever the family moves on. The offset is where those bytes
        start inside ``family.data``, which is how a device write to a view is folded back
        into the whole buffer.
        """
        family = self._state
        root = family.root()
        if root is None or self is root:
            return family.data, 0
        view = self.view(np.ndarray)
        if view.flags["C_CONTIGUOUS"] and root is not None:
            offset = int(view.__array_interface__["data"][0]) - int(
                root.view(np.ndarray).__array_interface__["data"][0])
            if 0 <= offset <= family.data.nbytes - view.nbytes:
                return family.data[offset:offset + view.nbytes], offset
        # A non-contiguous or unrepresentable view: keep its own contiguous copy.
        if self._staging is None or self._staging_epoch < family.current_epoch:
            self._staging = np.ascontiguousarray(view)
            self._staging_epoch = family.current_epoch
        return self._staging, -1

    def _absorb(self, host_array: np.ndarray, offset: int) -> None:
        """Fold the bytes just downloaded for one element back into the family."""
        if offset >= 0:
            # The bytes came down straight into the family's buffer: nothing to fold.
            return
        # This element owns a private contiguous copy (its view is not contiguous, so a
        # device buffer for it is a copy rather than a window into the family). The
        # downloaded values are the *view's* values, so they have to be scattered back
        # through the view's own layout -- writing them contiguously into the family
        # would shift every row and lose the assignment entirely.
        view = self.view(np.ndarray)
        try:
            np.copyto(view, np.ascontiguousarray(host_array).reshape(view.shape))
        except (ValueError, TypeError):
            # A shape/dtype the copy cannot express: the host keeps the staged copy,
            # which is already current, so there is nothing else to do here.
            return
        finally:
            self._staging = None
            self._staging_epoch = 0

    # ---------- host side ----------

    def _print_copy_info(self) -> bool:
        """Whether the backends in use currently log progress."""
        return get_print_info()

    def _wait_pending(self, bindings: Optional[List[_Binding]] = None) -> None:
        """Wait for the copies these bindings still have in flight."""
        entries = (list(self._bindings.values()) if bindings is None else bindings)
        for binding in entries:
            if binding.pending:
                pending, binding.pending = list(binding.pending), []
                for transfer in pending:
                    transfer.wait()

    def _ensure_host_bytes(self) -> None:
        """Make the host bytes current, waiting for the copy that carries them.

        Every host-side operation goes through here, which is what lets ``to_host()`` be
        asynchronous: the next actual read waits for the copy.
        """
        if not self.__dict__.get("_should_copy", True):
            return
        family = self._state
        self._wait_pending()
        with family.lock:
            if family.current_epoch < family.epoch:
                newest = family.newest_binding()
                if newest is not None:
                    self._hydrate_from(family, newest)

    def _hydrate_from(self, family: "_Family", newest: Tuple["ndarray", _Binding]) -> None:
        """Download the newest device content into the family's host bytes, now."""
        member, binding = newest
        target = binding.target
        host_array, offset = member._member_bytes()
        queue = copy_queue(target)
        try:
            event = download(target, queue, binding.mem_obj, host_array)
            if event is not None:
                wait_event(target, event)
            else:
                sync_queue(target, queue)
        finally:
            release_copy_queue(target, queue)
        member._absorb(host_array, offset)
        family.current_epoch = binding.epoch

    def _after_host_write(self) -> None:
        """Record that this element's bytes changed on the host."""
        family = self.__dict__.get("_family")
        if family is None:
            return
        family.note_writer(self)
        family.refresh_data()
        with family.lock:
            family.epoch += 1
            family.current_epoch = family.epoch
            self._staging = None
            self._staging_epoch = 0

    def _set_host_dirty(self) -> None:
        """Mark the whole family as changed on the host."""
        self._root()._after_host_write()

    def _set_tree_host_dirty(self) -> None:
        """Kept for the launch path: marking the family covers every view of it."""
        self._set_host_dirty()

    # ---------- transfers ----------

    def to_host(self, stream: object = None, *, backend: Optional[str] = None,
                device: object = None, wait: bool = False) -> Transfer:
        """Start copying this array's contents back to host memory.

        The copy is asynchronous: the returned :class:`Transfer` is already valid, and
        anything that later *reads* the array waits for it automatically. Pass
        ``wait=True`` to block now, or use ``transfer.wait()`` / ``await transfer`` to
        block explicitly.

        The host bytes the array already has are what the copy lands in, so a device that
        wrote the array does not disturb the rest of the family.
        """
        family = self._state
        newest = family.newest_binding()
        if newest is None or newest[1].epoch <= family.current_epoch:
            transfer = Transfer.completed("to_host")
            if wait:
                transfer.wait()
            return transfer

        member, binding = newest
        target = binding.target
        host_array, offset = member._member_bytes()
        if binding.mapped:
            # The binding *is* the host bytes, so the device's writes are already here.
            with family.lock:
                family.current_epoch = binding.epoch
            transfer = Transfer.completed(f"to_host {target}")
            if wait:
                transfer.wait()
            return transfer

        queue = copy_queue(target)
        transfer = Transfer(f"to_host {family.data.shape} from {target}")
        try:
            event = download(target, queue, binding.mem_obj, host_array)
        except BaseException:
            release_copy_queue(target, queue)
            raise

        def complete(error_code: Optional[int] = None) -> None:
            try:
                if not error_code_ok(target.backend, error_code):
                    transfer._complete(RuntimeError(
                        f"the copy from {target} failed ({error_code})"))
                    return
                with family.lock:
                    family.current_epoch = binding.epoch
                member._absorb(host_array, offset)
                if self._print_copy_info():
                    print(f"copy data to host for argument '{self._arg_name}'")
                transfer._complete()
            finally:
                release_copy_queue(target, queue)

        if event is None:
            complete()
        else:
            binding.add_pending(transfer)
            event_callback(target, queue, event, complete)
        if wait:
            transfer.wait()
        return transfer

    def to_device(self, target: object = None, stream: object = None, *,
                  backend: Optional[str] = None, device: object = None,
                  zero_copy: Optional[bool] = None, wait: bool = False) -> Transfer:
        """Start copying this array's contents to a device.

        ``target`` is a selector string (``"cuda"``, ``"opencl:1"``), a
        ``(backend, device)`` pair, or a Device object; ``None`` means the default device,
        which is the one a plain ``kernel(...)`` call would run on. The ``backend=`` /
        ``device=`` keywords are the same thing spelled in two arguments.

        When the array's bytes are already current on that device nothing is copied and the
        transfer is complete at once -- that is what makes a second kernel call on the same
        device free.

        ``zero_copy=True`` asks the device to read and write the array's own host memory
        instead of a copy of it. That is honoured only where the backend's API guarantees
        it (see :func:`pygpgpu.numpy.zero_copy_support`); elsewhere it raises
        ``NotImplementedError`` naming the backend. ``None`` (the default) keeps whatever
        choice the array already made for that device.
        """
        if target is not None and (backend is not None or device is not None):
            raise TypeError(
                "to_device() takes either a positional target selector or the "
                "backend=/device= keywords, not both")
        if target is None and (backend is not None or device is not None):
            target = (backend, device)
        return self._to_device_target(target_of(target), zero_copy=zero_copy, wait=wait)

    def _to_device_target(self, target: Target, zero_copy: Optional[bool] = None,
                          wait: bool = False) -> Transfer:
        """The engine's device copy: reuse a current binding, or upload."""
        family = self._state
        key = (target.key, self._layout_key(target.backend))
        binding = self._bindings.get(key)
        if binding is not None and binding.epoch == family.epoch and \
                (zero_copy is None or binding.zero_copy == bool(zero_copy)):
            transfer = Transfer.completed(f"to_device {target}")
            if wait:
                transfer.wait()
            return transfer

        if binding is not None and zero_copy is not None and \
                binding.zero_copy != bool(zero_copy):
            # Changing the memory model for this device: a mapped buffer and a copy do not
            # hold the same bytes at the same address, so the old binding is dropped.
            self._release_binding(key, binding)
            binding = None

        if binding is None:
            use_zero_copy = bool(zero_copy)
            if use_zero_copy:
                supported, reason = zero_copy_support(target)
                if not supported:
                    raise NotImplementedError(reason)
            host_array, offset = self._member_bytes()
            host_ptr = (int(host_array.__array_interface__["data"][0]) if use_zero_copy
                        else None)
            mem_obj = allocate(target, int(host_array.nbytes), use_zero_copy, host_ptr)
            binding = _Binding(mem_obj, 0, target, use_zero_copy)
            self._bindings[key] = binding
        return self._upload_into(target, binding, family, wait)

    def _upload_into(self, target: Target, binding: _Binding, family: "_Family",
                     wait: bool) -> Transfer:
        """Bring a stale binding up to date on the device."""
        # A queued copy on this binding has to land before a new one starts on it.
        self._wait_pending([binding])

        host_array, offset = self._member_bytes()
        if binding.mapped:
            # The binding is the host bytes, so it is current the moment the host is.
            self._ensure_host_bytes()
            binding.epoch = family.epoch
            transfer = Transfer.completed(f"to_device {target} (zero-copy)")
            if wait:
                transfer.wait()
            return transfer

        if family.current_epoch < family.epoch:
            newest = family.newest_binding()
            if newest is not None and newest[1] is not binding:
                # Another device holds newer content: stage through the host, which is the
                # only route between two devices and between two backends.
                self._hydrate_from(family, newest)

        queue = copy_queue(target)
        transfer = Transfer(f"to_device {target} for {family.data.shape}")
        family.begin_read()
        try:
            event = upload(target, queue, binding.mem_obj, host_array)
        except BaseException:
            family.end_read()
            release_copy_queue(target, queue)
            raise

        def complete(error_code: Optional[int] = None) -> None:
            try:
                family.end_read()
                if not error_code_ok(target.backend, error_code):
                    transfer._complete(RuntimeError(
                        f"the copy to {target} failed ({error_code})"))
                    return
                with family.lock:
                    binding.epoch = family.epoch
                if self._print_copy_info():
                    print(f"copy data to device for argument '{self._arg_name}'")
                transfer._complete()
            finally:
                release_copy_queue(target, queue)

        if event is None:
            complete()
        else:
            binding.add_pending(transfer)
            family.note_writer(self)
            event_callback(target, queue, event, complete)
        if wait:
            transfer.wait()
        return transfer

    def _release_binding(self, key: tuple, binding: _Binding) -> None:
        """Drop one device binding, waiting for what it still has queued."""
        self._wait_pending([binding])
        self._bindings.pop(key, None)
        release(binding.mem_obj)
        binding.mem_obj = None

    def release(self, target: object = None) -> None:
        """Release the device copies of this array (the host bytes stay untouched).

        Without a target it releases every copy; with one it releases that device's copy.
        Any content the device holds that the host does not have is downloaded first --
        releasing a copy is not a way to lose the result -- and a copy still in flight is
        waited for, because the device is writing into it.
        """
        family = self._state
        if target is None:
            self._wait_pending()
            with family.lock:
                if family.current_epoch < family.epoch:
                    newest = family.newest_binding()
                    if newest is not None:
                        self._hydrate_from(family, newest)
            for key in list(self._bindings.keys()):
                self._release_binding(key, self._bindings[key])
            return
        resolved = target_of(target)
        keys = [key for key in self._bindings if key[0] == resolved.key]
        # Only downloading from this device is pointless if another one holds newer bytes;
        # the family-wide pull happens first so nothing is lost either way.
        self._wait_pending([self._bindings[key] for key in keys])
        with family.lock:
            if family.current_epoch < family.epoch:
                newest = family.newest_binding()
                if newest is not None:
                    self._hydrate_from(family, newest)
        for key in keys:
            self._release_binding(key, self._bindings[key])

    def _layout_key(self, backend: str) -> tuple:
        """What a device copy's bytes depend on, besides the contents."""
        view = self.view(np.ndarray)
        return (backend, view.dtype.str, view.shape, view.strides)

    # ---------- the launch path's entry point ----------

    def _to_device(self, cmd_queue: object, arg_name: str,
                   flags: KernelFlags = KernelFlags(1 << 0),
                   kernel_flags: KernelFlags = KernelFlags(1 << 0),
                   image_info: Optional["ImageDescriptor"] = None,
                   *, backend: Optional[str] = None,
                   device: object = None) -> Tuple[object, object]:
        """Bind this argument to the device a launch runs on; returns ``(mem_obj, event)``.

        This is the interface the backends' ``use_buffer``/``use_image`` call. ``event`` is
        the backend's copy event (None when nothing had to travel); the caller keeps it
        alive for the launch and for the read-back.
        """
        used_backend = backend if backend is not None else _queue_backend(cmd_queue)
        used_device = device
        if used_device is None:
            used_device = getattr(cmd_queue, "device", None)
        if used_device is None:
            raise ValueError(
                f"cannot tell which device the {type(cmd_queue).__name__} runs on; pass "
                f"device= explicitly")
        target = target_of((used_backend, used_device))
        self._arg_name = arg_name
        if image_info is not None:
            return self._bind_image(target, cmd_queue, arg_name, image_info, kernel_flags)
        return self._bind_for_launch(target, cmd_queue, kernel_flags)

    def _bind_for_launch(self, target: Target, cmd_queue: object,
                         kernel_flags: KernelFlags) -> Tuple[object, object]:
        """Bind this element's bytes to ``target`` on the caller's own channel."""
        family = self._state
        key = (target.key, self._layout_key(target.backend))
        binding = self._bindings.get(key)

        if binding is not None and binding.epoch == family.epoch:
            # The device already holds exactly these bytes: the launch is free of copies.
            self._wait_pending([binding])
            return binding.mem_obj, None

        if binding is None:
            host_array, _offset = self._member_bytes()
            mem_obj = allocate(target, int(host_array.nbytes))
            binding = _Binding(mem_obj, 0, target, False)
            self._bindings[key] = binding

        self._wait_pending([binding])
        host_array, _offset = self._member_bytes()
        if family.current_epoch < family.epoch:
            newest = family.newest_binding()
            if newest is not None and newest[1] is not binding:
                self._hydrate_from(family, newest)

        event = upload(target, cmd_queue, binding.mem_obj, host_array)
        family.note_writer(self)
        if event is None:
            with family.lock:
                binding.epoch = family.epoch
            return binding.mem_obj, None

        # The launch reads these bytes; keep anything from changing them until it has.
        family.begin_read()
        event_callback(target, cmd_queue, event, lambda _code: family.end_read())
        with family.lock:
            binding.epoch = family.epoch
        if self._print_copy_info():
            print(f"copy data to device for argument '{self._arg_name}'")
        return binding.mem_obj, event

    def _bind_image(self, target: Target, cmd_queue: object, arg_name: str,
                    image_info: "ImageDescriptor",
                    kernel_flags: KernelFlags) -> Tuple[object, object]:
        """Bind an image argument: images are re-uploaded per launch, as they always were."""
        host_array, _offset = self._member_bytes()
        view = self.view(np.ndarray)
        if view.shape != tuple(image_info.shape):
            raise ValueError(
                f"the image argument '{arg_name}' has host shape {view.shape} but its "
                f"image is declared with shape {tuple(image_info.shape)}")
        flags = image_info.flags
        key = (target.key, self._layout_key(target.backend) + ("image", id(image_info)),
               image_info.shape, image_info.dtype, flags)
        binding = self._bindings.get(key)
        if binding is None:
            mem_obj = allocate_image(target, cmd_queue, image_info)
            binding = _Binding(mem_obj, 0, target, False, image_info)
            self._bindings[key] = binding
            always_upload = True
        else:
            always_upload = _IMAGE_ALWAYS_UPLOAD

        if binding.epoch == self._state.epoch and not always_upload:
            return binding.mem_obj, None
        if always_upload:
            event = binding.mem_obj.set_data(cmd_queue, view)
        else:
            event = upload(target, cmd_queue, binding.mem_obj, host_array)
        binding.epoch = self._state.epoch
        self._state.note_writer(self)
        if self._print_copy_info() and event is not None:
            print(f"copy data to device for argument '{arg_name}'")
        return binding.mem_obj, event

    def _mark_device_written(self, mem_obj: object = None,
                             target: Optional[Target] = None) -> None:
        """Record that a kernel wrote this argument on the device.

        The binding just launched on is the freshest device copy of this element, so it
        takes the family's next epoch. That makes the host bytes stale (the next host
        access downloads) while leaving that binding current (the next launch on the same
        device reuses the buffer untouched).

        ``mem_obj`` names the buffer the launch actually wrote, which is what distinguishes
        "this device" from another one the array was also copied to: an array resident on
        two devices has a binding for each, and only the one the kernel ran on holds the
        new content.

        The binding may belong to a *different* view object than this one. Two views of
        one buffer (`k(a[0], a[0], a[0])` naming three separately-created views) share a
        single device buffer -- the launch binds the family's buffer once (`seen_arrays`) --
        so the binding lives wherever that first view put it, and searching only this
        object's own `_bindings` would find nothing and silently drop the write.
        """
        family = self._state
        self._wait_pending()
        candidates = [binding for binding in self._bindings.values()
                      if (mem_obj is None or binding.mem_obj is mem_obj)
                      and (target is None or binding.target.key == target.key)]
        if not candidates and mem_obj is not None:
            # Same buffer, held by another member of the family.
            root = family.root()
            for reference in list(family.writers):
                member = reference()
                if member is None or (root is not None and member is root and member is self):
                    continue
                for binding in member.__dict__.get("_bindings", {}).values():
                    if binding.mem_obj is mem_obj and (
                            target is None or binding.target.key == target.key):
                        candidates.append(binding)
        if not candidates:
            return
        with family.lock:
            epoch = family.epoch + 1
            family.epoch = epoch
            for binding in candidates:
                binding.epoch = epoch
            family.note_writer(self)

    # ---------- zero-copy ----------

    def to_device_zero_copy(self, target: object = None) -> Transfer:
        """``to_device(..., zero_copy=True)`` as its own spelling.

        Equivalent to ``to_device(target, zero_copy=True)``: the device reads and writes
        the array's own host memory. Raises ``NotImplementedError`` naming the backend when
        its API cannot guarantee that.
        """
        return self._to_device_target(target_of(target), zero_copy=True)

    def is_zero_copy(self, target: object = None) -> bool:
        """Whether the array's bytes are bound to that device's own memory.

        ``target=None`` asks about any device: True when at least one binding is zero-copy.
        """
        if target is None:
            return any(binding.mapped for binding in self._bindings.values())
        resolved = target_of(target)
        return any(binding.mapped for key, binding in self._bindings.items()
                   if key[0] == resolved.key)

    # ---------- PyTorch interop ----------

    @classmethod
    def from_torch(cls, tensor: object, zero_copy: bool = False) -> "ndarray":
        """A gnp array sharing (or holding a copy of) a ``torch.Tensor``'s values.

        A CPU tensor's storage is ordinary host memory, so with ``zero_copy=True`` the
        returned array *is* the tensor's buffer: writing through one is visible in the
        other. That needs the tensor to be contiguous and page-locked first, and the
        returned array keeps the pin alive for its lifetime. A CUDA tensor is uploaded
        once, and the two copies are then independent.
        """
        from ._torch import from_torch

        return from_torch(cls, tensor, zero_copy=zero_copy)

    def to_torch(self, zero_copy: bool = False) -> object:
        """A ``torch.Tensor`` holding this array's values.

        With ``zero_copy=True`` a host-side array hands its own buffer to torch, so the
        tensor and the array are one storage. A device-resident array has no host storage
        to lend, so it is copied to the host first and the tensor owns that copy.
        """
        from ._torch import to_torch

        return to_torch(self, zero_copy=zero_copy)

    # ---------- numpy protocol ----------

    def __array__(self, dtype: Optional[DTypeLike] = None,
                  copy: Optional[bool] = None) -> np.ndarray:
        self._ensure_host_bytes()
        plain = self.view(np.ndarray)
        if dtype is not None and plain.dtype != np.dtype(dtype):
            return plain.astype(dtype)
        return plain

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: object,
                        **kwargs: object) -> object:
        self._ensure_host_bytes()
        result = _call_ufunc(ufunc, method, _to_plain_args(inputs),
                             _to_plain_mapping(kwargs))
        out = kwargs.get("out")
        if out is not None:
            for value in (out if isinstance(out, tuple) else (out,)):
                if isinstance(value, ndarray):
                    value._after_host_write()
        return self._change_result(result)

    def __array_function__(self, func: Callable[..., object],
                           types: Tuple[type, ...], args: Tuple[object, ...],
                           kwargs: Mapping[str, object]) -> object:
        self._ensure_host_bytes()
        result = func(*_to_plain_args(args), **_to_plain_mapping(kwargs))
        return self._change_result(result)

    def _change_to_ndarray(self, value: object) -> object:
        if value is self:
            return value
        if not isinstance(value, ndarray):
            value = value.view(ndarray)
        value._bindings = {}
        value._family = None
        value._staging = None
        value._staging_epoch = 0
        value._arg_name = ""
        value._should_copy = True
        value._children = []
        if _aliases(value, self):
            value._parent = self
            self._children.append(value)
        else:
            value._parent = None
        return value

    def _change_result(self, result: object) -> object:
        if isinstance(result, np.ndarray):
            return self._change_to_ndarray(result)
        if isinstance(result, tuple):
            return tuple(self._change_to_ndarray(item) if isinstance(item, np.ndarray)
                         else item for item in result)
        return result

    # ---------- the array surface ----------

    def __getitem__(self, key: object) -> object:
        self._ensure_host_bytes()
        return self._change_result(np.ndarray.__getitem__(self.view(np.ndarray), key))

    def __setitem__(self, key: object, value: object) -> None:
        self._ensure_host_bytes()
        if isinstance(value, ndarray):
            value._ensure_host_bytes()
            value = value.view(np.ndarray)
        np.ndarray.__setitem__(self.view(np.ndarray), key, value)
        self._after_host_write()

    def __repr__(self) -> str:
        self._ensure_host_bytes()
        return np.ndarray.__repr__(self.view(np.ndarray))

    def __str__(self) -> str:
        self._ensure_host_bytes()
        return np.ndarray.__str__(self.view(np.ndarray))

    def tolist(self) -> object:
        self._ensure_host_bytes()
        return np.ndarray.tolist(self.view(np.ndarray))

    def tobytes(self) -> bytes:
        self._ensure_host_bytes()
        return np.ndarray.tobytes(self.view(np.ndarray))

    def tofile(self, fid: object, sep: str = "", format: str = "%s") -> None:
        self._ensure_host_bytes()
        np.ndarray.tofile(self.view(np.ndarray), fid, sep, format)

    def copy(self, order: str = "C") -> "ndarray":
        self._ensure_host_bytes()
        return self._change_result(np.ndarray.copy(self.view(np.ndarray), order))

    def dumps(self) -> bytes:
        self._ensure_host_bytes()
        return np.ndarray.dumps(self.view(np.ndarray))

    def dump(self, file: object) -> None:
        self._ensure_host_bytes()
        np.ndarray.dump(self.view(np.ndarray), file)

    def item(self, *args: int) -> object:
        self._ensure_host_bytes()
        return np.ndarray.item(self.view(np.ndarray), *args)

    def itemset(self, *args: object) -> None:
        self._ensure_host_bytes()
        np.ndarray.itemset(self.view(np.ndarray), *args)
        self._after_host_write()

    def _inplace(func):
        @functools.wraps(func)
        def wrapper(self: "ndarray", *args: object, **kwargs: object) -> object:
            self._ensure_host_bytes()
            result = func(self, *args, **kwargs)
            self._after_host_write()
            return result
        return wrapper

    __iadd__ = _inplace(np.ndarray.__iadd__)
    __isub__ = _inplace(np.ndarray.__isub__)
    __imul__ = _inplace(np.ndarray.__imul__)
    __itruediv__ = _inplace(np.ndarray.__itruediv__)
    __ifloordiv__ = _inplace(np.ndarray.__ifloordiv__)
    __imod__ = _inplace(np.ndarray.__imod__)
    __ipow__ = _inplace(np.ndarray.__ipow__)
    __ilshift__ = _inplace(np.ndarray.__ilshift__)
    __irshift__ = _inplace(np.ndarray.__irshift__)
    __iand__ = _inplace(np.ndarray.__iand__)
    __ixor__ = _inplace(np.ndarray.__ixor__)
    __ior__ = _inplace(np.ndarray.__ior__)

    @property
    def T(self) -> "ndarray":
        self._ensure_host_bytes()
        return self._change_to_ndarray(np.transpose(self.view(np.ndarray)))

    @property
    def data(self) -> memoryview:
        self._ensure_host_bytes()
        return self.view(np.ndarray).data

    @property
    def real(self) -> "ndarray":
        self._ensure_host_bytes()
        return self._change_result(self.view(np.ndarray).real)

    @property
    def imag(self) -> "ndarray":
        self._ensure_host_bytes()
        return self._change_result(self.view(np.ndarray).imag)

    @property
    def flat(self) -> np.flatiter:
        self._ensure_host_bytes()
        return self.view(np.ndarray).flat

    @property
    def ctypes(self) -> object:
        self._ensure_host_bytes()
        return self.view(np.ndarray).ctypes

    @property
    def base(self) -> Optional["ndarray"]:
        return self.__dict__.get("_parent")


# ---------- helpers for the numpy protocol ----------

def _to_plain(values: object) -> object:
    """Replace a gnp array by its plain numpy view, recursing into a tuple or list."""
    if isinstance(values, ndarray):
        values._ensure_host_bytes()
        return values.view(np.ndarray)
    if isinstance(values, tuple):
        return tuple(_to_plain(item) for item in values)
    if isinstance(values, list):
        return [_to_plain(item) for item in values]
    return values


def _to_plain_args(values: Tuple[object, ...]) -> Tuple[object, ...]:
    """Every positional argument's plain view, as a tuple a call can be splatted with."""
    return tuple(_to_plain(item) for item in values)


def _to_plain_mapping(mapping: Mapping[str, object]) -> Dict[str, object]:
    return {key: _to_plain(value) for key, value in mapping.items()}


def _aliases(value: object, original: object) -> bool:
    """Whether ``value`` overlaps ``original``'s memory.

    Both are gnp elements, so the question is asked of the plain buffers they index:
    asking numpy directly would dispatch ``np.shares_memory`` back into
    ``__array_function__``, which is the very method that may be asking.
    """
    return np.shares_memory(value.view(np.ndarray), original.view(np.ndarray),
                            max_work=2 ** 20)


def _call_ufunc(ufunc: np.ufunc, method: str, inputs: Tuple[object, ...],
                kwargs: Mapping[str, object]) -> object:
    """Run a ufunc with plain arrays, keeping the method (``__call__``, ``reduce``, ...).

    numpy refuses to dispatch a ufunc whose inputs are its own subclass back to the same
    subclass a second time, so the operation is evaluated on the plain views instead of
    through ``super().__array_ufunc__``.
    """
    if method == "__call__":
        return ufunc(*inputs, **kwargs)
    return getattr(ufunc, method)(*inputs, **kwargs)
