"""A command queue and the command buffers a launch runs on.

Metal has no context object separate from the device: a `MTLCommandQueue` belongs
to a device, so the Context here is a thin holder of the device plus the one
queue its launches run on. That is enough for the shape the rest of the package
expects (`context.id`, `context.create_buffer`, `context.device`), and it avoids
inventing a second level of state Metal does not have.

A launch is: make a command buffer, set the pipeline, bind the buffers, dispatch
the threadgroups, commit, and (for a synchronous call) wait. The queue is
serial, so consecutive launches run in order, which is what the multi-pass
bodies depend on.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..driver.metal import _wait_ready
from ..driver.objc import ObjC
from .metalobject import MTLObject


class CommandQueue(MTLObject):
    """One `MTLCommandQueue`: a serial stream of command buffers."""

    def __init__(self, device:Any)->None:
        _wait_ready()
        queue_id = ObjC.send_ptr(device.id, "newCommandQueue")
        if not queue_id:
            raise RuntimeError("Metal could not create a command queue")
        MTLObject.__init__(self, queue_id)
        self._device:Any = device

    @property
    def device(self)->Any:
        return self._device

    def new_command_buffer(self)->int:
        """A fresh `MTLCommandBuffer`, ready for encoding."""
        command_buffer = ObjC.send_ptr(self.id, "commandBuffer")
        if not command_buffer:
            raise RuntimeError("Metal could not create a command buffer")
        return command_buffer


class Context(MTLObject):
    """The device plus the queue its launches run on.

    `id` is the *device's* handle, not the queue's. Every backend's Context answers `id`
    with something that identifies the device + context pair, and the array layer compares
    those values to tell "the same device" from "another device". A queue handle is
    recreated per Context, so two Contexts over one GPU would compare unequal and the
    array layer would treat them as different devices -- re-uploading, and never reusing
    the buffer that is already there. The device handle is the same object for the life of
    the process, so it is the honest answer.
    """

    def __init__(self, device:Any)->None:
        _wait_ready()
        self._device:Any = device
        self._queue:CommandQueue = CommandQueue(device)
        MTLObject.__init__(self, device.id)
        self._buffers:Dict[int, List[Any]] = {}

    @property
    def device(self)->Any:
        return self._device

    @property
    def command_queue(self)->CommandQueue:
        return self._queue

    # ---- the surface shared with the other backends' contexts ----

    def make_current(self)->None:
        """Metal has no per-thread context to bind; this exists for interface parity.

        Unlike CUDA's `cuCtxSetCurrent` and HIP's `hipSetDevice`, Metal has no per-thread
        current context: a command queue is usable from any thread, and the objects
        built here carry their own device. The shared copy code calls this
        unconditionally, so the method exists and does nothing -- exactly as OpenCL's
        does, and for the same reason.
        """

    def create_buffer(self, size:int=0, flags:Any=None)->Any:
        from .buffer import Buffer
        return Buffer(self, size=size)

    def get_buffer(self, size:int=0)->Any:
        """A reusable device buffer of `size` bytes, per the shared allocation path."""
        from .buffer import Buffer
        available:Optional[Any] = None
        for buffer in self._buffers.get(size, []):
            if not buffer.using:
                available = buffer
                break
        if available is None:
            available = Buffer(self, size=size)
            self._buffers.setdefault(size, []).append(available)
        available.use()
        return available

    def sync(self)->None:
        """Wait for everything queued so far (the queue is serial, so one wait covers it)."""
        # Metal has no queue-level wait; each launch waits for its own command
        # buffer, so a completed launch is already a completed queue.
        return

    def wait_event(self, event:Any)->None:
        event.wait()
