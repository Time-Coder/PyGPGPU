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
not be looking at."""
from __future__ import annotations
from typing import TYPE_CHECKING
from .binding import (
    _Binding,
    _Family,
    _IMAGE_ALWAYS_UPLOAD,
    _QUEUE_CLASSES,
    _queue_backend,
    _queue_classes,
)
from .plain import (
    _aliases,
    _call_ufunc,
    _to_plain,
    _to_plain_args,
    _to_plain_mapping,
)
import functools
from typing import Callable, Dict, List, Mapping, Optional, Tuple
import numpy as np
from numpy.typing import DTypeLike
from ...constants import KernelFlags
from .._backends import (allocate, allocate_image, download, error_code_ok, event_callback,
                        is_host_mapped, release, sync_queue, upload, wait_event,
                        zero_copy_support)
from .._devices import Target, copy_queue, release_copy_queue, target_of
from ..transfer import Transfer
from ._host_state import _HostStateMixin
from ._transfer import _TransferMixin
from ._launch_binding import _LaunchBindingMixin
from ._numpy_api import _NumpyApiMixin, _inplace

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
not be looking at."""

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

class ndarray(_HostStateMixin, _TransferMixin, _LaunchBindingMixin, _NumpyApiMixin, np.ndarray):

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

__all__ = [
    "T",
    "__array__",
    "__array_finalize__",
    "__array_function__",
    "__array_ufunc__",
    "__getitem__",
    "__iadd__",
    "__iand__",
    "__ifloordiv__",
    "__ilshift__",
    "__imod__",
    "__imul__",
    "__ior__",
    "__ipow__",
    "__irshift__",
    "__isub__",
    "__itruediv__",
    "__ixor__",
    "__new__",
    "__repr__",
    "__setitem__",
    "__str__",
    "_absorb",
    "_after_host_write",
    "_bind_for_launch",
    "_bind_image",
    "_change_result",
    "_change_to_ndarray",
    "_ensure_host_bytes",
    "_hydrate_from",
    "_inplace",
    "_layout_key",
    "_make_family",
    "_mark_device_written",
    "_member_bytes",
    "_print_copy_info",
    "_release_binding",
    "_root",
    "_set_host_dirty",
    "_set_tree_host_dirty",
    "_state",
    "_to_device",
    "_to_device_target",
    "_upload_into",
    "_wait_pending",
    "base",
    "copy",
    "ctypes",
    "data",
    "dump",
    "dumps",
    "flat",
    "from_torch",
    "imag",
    "is_zero_copy",
    "item",
    "itemset",
    "ndarray",
    "real",
    "release",
    "to_device",
    "to_device_zero_copy",
    "to_host",
    "to_torch",
    "tobytes",
    "tofile",
    "tolist",
]
