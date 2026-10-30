"""Host-side support for the dual-sided builtins in ``builtins.py``.

The builtin functions are the user-facing API and nothing else lives there. What they need
in order to run in the interpreter is here:

* the per-thread frame the ``python`` backend installs for one launch, so two concurrent
  launches never see each other's work item, and
* the small helpers their bodies are made of (reading a work item, a launch extent, and
  one flat read-modify-write for the atomics).

A device backend never executes any of this: it emits the corresponding OpenCL C or CUDA
call instead (see ``device_builtins.py``).
"""
from __future__ import annotations

import threading
from typing import Any, Optional

# Installed by the python backend for the duration of one kernel call, on the calling
# thread only.
_FRAME = threading.local()


def set_host_frame(frame: Optional[Any]) -> None:
    """Install the active python-backend frame for this thread (None clears it)."""
    _FRAME.frame = frame


def get_host_frame() -> Optional[Any]:
    """The frame installed by the python backend on this thread, or None."""
    return getattr(_FRAME, "frame", None)


def _dimension(dim: Any, what: str) -> int:
    """Validate a work-size dimension argument."""
    if not isinstance(dim, int) or isinstance(dim, bool) or dim not in (0, 1, 2):
        raise SyntaxError(f"{what}(dim) requires a dimension of 0, 1 or 2") from None
    return dim


def require_frame(what: str) -> Any:
    """The current launch's frame, or a message explaining why there is none."""
    frame: Optional[Any] = get_host_frame()
    if frame is None:
        raise SyntaxError(
            f"{what}() is only available inside a @kernel function body: on a device "
            f"backend it maps to a device builtin, and on the python backend it needs the "
            f"work item of the launch being run serially") from None
    return frame


def work_item(dim: Any, what: str) -> int:
    """The global id of the current work item along ``dim``."""
    frame: Any = require_frame(what)
    axis: int = _dimension(dim, what)
    if frame.coords is None:
        raise SyntaxError(
            f"{what}({axis}) has no meaning in a serial python run: the python backend "
            f"executes a magic-loop kernel as one plain call, so there is no current work "
            f"item. Use the loop variable instead (the magic loop gives the same index).") from None
    if axis >= len(frame.coords):
        raise ValueError(
            f"{what}({axis}) is out of range: this launch has {len(frame.coords)} "
            f"work-size dimension(s)")
    return frame.coords[axis]


def local_extent(dim: Any, what: str) -> int:
    """How many work items a group holds along ``dim``."""
    frame: Any = require_frame(what)
    axis: int = _dimension(dim, what)
    return frame.local_size[axis] if axis < len(frame.local_size) else 1


def work_size_extent(dim: Any, what: str) -> Any:
    """The (global, local) extents of the launch along ``dim``."""
    frame: Any = require_frame(what)
    axis: int = _dimension(dim, what)
    if axis >= len(frame.global_size):
        raise ValueError(
            f"{what}({axis}) is out of range: this launch has {len(frame.global_size)} "
            f"work-size dimension(s); declare the domain with work_size(array) or pass a "
            f"(global, local) bracket")
    return (frame.global_size[axis],
            frame.local_size[axis] if axis < len(frame.local_size) else 1)


def atomic_in_place(array: Any, index: Any, apply: Any) -> Any:
    """Read-modify-write one element of ``array`` and return the previous value.

    ``index`` is a *flat* offset, because that is what the device code means: the emitted
    call is ``atomicAdd(&arr[i], v)`` on a raw pointer, with no notion of the array's
    declared rank. Indexing a multi-dimensional array directly would address axis 0
    instead, so the buffer is flattened first -- and, being a view of a contiguous array,
    the write still lands in the caller's data.
    """
    store: Any = array
    if getattr(array, "ndim", 1) > 1:
        store = array.reshape(-1)
    position: int = int(index)
    previous: Any = store[position]
    store[position] = apply(previous)
    return previous
