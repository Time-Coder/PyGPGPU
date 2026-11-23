"""The launch made on a Metal entry point.

`BoundKernel` is what a caller holds after `kernel[work_sizes]`: the kernel plus its fixed
work size. It owns the three things a launch does -- bind and encode, commit and wait (or
hand the wait to a poller), and copy the written arguments home -- and it is the module's
subject because every path through it is about one of those three.

The synchronous `__call__` waits inline (`waitUntilCompleted`) and copies back on the
calling thread. `submit`/`async_call` commit and return a future at once, and a poller
thread watches the command buffer's `status` until the work is done, because Metal reports
completion through that status rather than through a separate event object.
"""
from __future__ import annotations

import ctypes
import concurrent.futures
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ...driver.metal import MetaMetal, _wait_ready
from ...driver.objc import ObjC
from .copyback import (_MTCOMMAND_BUFFER_STATUS_COMPLETED, _POLL_INTERVAL,
                       _finish_buffer)
from .scalars import BoundEntry, Extents, MTLSize, _axis_extents, _mtl_size

if TYPE_CHECKING:
    import asyncio

    from ..buffer import Buffer
    from . import MetalKernel


class BoundKernel:
    """A kernel with its work size fixed: the object a launch is made on."""

    def __init__(self, kernel:"MetalKernel",
                 work_sizes:Optional[Extents])->None:
        self._kernel = kernel
        self._work_sizes = work_sizes

    def submit(self, *args:object, **kwargs:object)->concurrent.futures.Future:
        """Launch without waiting, returning a ``concurrent.futures.Future``.

        The bracket form `k["metal"].submit(...)` reaches the launch through this
        object -- the same surface the other backends' `Kernel.__getitem__`
        answers -- so the two future styles have to forward from here as well.
        """
        return self._launch_async(args, kwargs, concurrent.futures.Future)

    def async_call(self, *args:object, **kwargs:object)->"asyncio.Future":
        """Launch without waiting, returning an ``asyncio.Future`` to ``await``."""
        import asyncio

        from ....utils import _asyncio_future_of
        return self._launch_async(
            args, kwargs, lambda: _asyncio_future_of(asyncio))

    def __call__(self, *args:object, **kwargs:object)->None:
        """The synchronous launch: encode, commit, wait, copy back, release."""
        _wait_ready()
        kernel = self._kernel
        bound = kernel._bind_args(args, kwargs)

        command_buffer:int = kernel.program.context.command_queue.new_command_buffer()
        try:
            self._encode(command_buffer, bound)
            self._log_launch(kernel)
            ObjC.send(command_buffer, "commit", (), ctypes.c_void_p)
            ObjC.send(command_buffer, "waitUntilCompleted", (), ctypes.c_void_p)
            # The launch has waited, so the copies back can happen inline. The read-back
            # flag decides the copy home; the claim serial validates the release.
            error_obj:int = ObjC.send_ptr(command_buffer, "error")
            encode_error:Optional[BaseException] = None
            if error_obj:
                encode_error = RuntimeError(
                    f"the metal launch failed: {ObjC.describe(error_obj)}")
            entries = self._claim_entries()
            for buffer, host_value, need, claim in entries:
                _finish_buffer(buffer, host_value, need, encode_error,
                               claim_serial=claim)
            if MetaMetal.print_info:
                # One line per argument copied home, after the copy-backs ran -- the
                # same per-argument accounting OpenCL's `on_completed` reports.
                arg_names = list(kernel._args.keys())
                for index, value, is_buffer in bound:
                    if is_buffer and kernel._args[arg_names[index]].need_read_back:
                        print(f"copy data to host for argument "
                              f"'{arg_names[index]}'", flush=True)
        finally:
            ObjC.release(int(command_buffer))

    def _log_launch(self, kernel:"MetalKernel")->None:
        """The launch line the other backends print, when verbose output is on.

        `_dispatch_shape` is the one place that normalises every bracket spelling (a
        bare count, `(4, 256)`, a per-axis pair), so the log matches what the encoder
        actually dispatches rather than what the caller wrote.
        """
        if not MetaMetal.print_info:
            return
        grid3, threads3 = self._dispatch_shape()
        print(f"launch Metal Kernel('{kernel._name}') on Device("
              f"'{kernel.program.context.device.name}') with grid_dim={grid3}, "
              f"block_dim={threads3}", flush=True)

    def _launch_async(self, args:Tuple[object, ...],
                      kwargs:Dict[str, object], future_type:type)->Any:
        """Launch without waiting; the future settles when the work has finished.

        Metal reports completion through the command buffer's `status`, so a poller
        thread watches it and settles the future when the buffer completes -- the
        shape the other backends' stream callbacks give them, reached through the
        one completion mechanism this driver surface exposes without blocks. An
        encode-time failure never starts a poller: it settles the future directly.
        """
        from ....utils import settle_future

        future:Any = future_type()
        try:
            command_buffer, entries = self._enqueue(args, kwargs)
        except BaseException as encode_error:
            settle_future(future, encode_error)
            return future

        def watch() -> None:
            while ObjC.send_int(command_buffer, "status") \
                    < _MTCOMMAND_BUFFER_STATUS_COMPLETED:
                time.sleep(_POLL_INTERVAL)
            error_obj:int = ObjC.send_ptr(command_buffer, "error")
            if error_obj:
                message:str = ObjC.describe(error_obj)
                settle_future(future, RuntimeError(
                    f"the metal launch failed: {message}"))
                return
            for buffer, host_value, need, claim in entries:
                _finish_buffer(buffer, host_value, need, claim_serial=claim)
            settle_future(future)
            # The buffer's work is done and the launch holds nothing else against it.
            ObjC.release(int(command_buffer))

        threading.Thread(target=watch, name="pygpgpu-metal", daemon=True).start()
        return future

    def _enqueue(self, args:Tuple[object, ...],
                 kwargs:Dict[str, object])->Tuple[int, List[Tuple["Buffer", object, object, int]]]:
        """Bind, encode and commit without waiting; answer what completion needs.

        `(command_buffer, entries)`: the buffer a completion waits on or polls, and
        one `(buffer, host_value, need_read_back, claim_serial)` per pointer
        argument -- the flag decides the copy home, the serial the release must
        carry so it cannot free a claim the buffer no longer holds
        (see `Buffer.unuse`). An encode-time failure releases the buffer here and
        re-raises, so a caller's `finally` never sees a half-encoded buffer.
        """
        _wait_ready()
        kernel = self._kernel
        bound = kernel._bind_args(args, kwargs)
        command_buffer:int = kernel.program.context.command_queue.new_command_buffer()
        try:
            self._encode(command_buffer, bound)
            self._log_launch(kernel)
            ObjC.send(command_buffer, "commit", (), ctypes.c_void_p)
        except BaseException:
            ObjC.release(int(command_buffer))
            raise
        return command_buffer, self._claim_entries()

    def _encode(self, command_buffer:int, bound:List[BoundEntry])->None:
        """The encode half of a launch, shared by the synchronous and async paths."""
        kernel = self._kernel
        encoder:int = ObjC.send_ptr(command_buffer, "computeCommandEncoder")
        if not encoder:
            raise RuntimeError("Metal could not create a compute command encoder")
        ObjC.send(encoder, "setComputePipelineState:", (ctypes.c_void_p,),
                  ctypes.c_void_p, kernel.pipeline)
        # Only the host-bound indices are set: a dispatch-supplied parameter has
        # nothing for the host to write (see the module docstring). A buffer takes
        # `setBuffer:offset:atIndex:` and a scalar takes `setBytes:length:atIndex:`;
        # binding a scalar as a buffer would present a pointer where MSL declared a
        # value, and the pipeline rejects that launch.
        for index, value, is_buffer in bound:
            if is_buffer:
                ObjC.send(encoder, "setBuffer:offset:atIndex:",
                          (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
                          ctypes.c_void_p, value.id, 0, index)
            else:
                payload:bytes = value
                # `setBytes:` reads from the pointer for the duration of the call
                # only, which is what the encoder copies from -- so a temporary
                # ctypes array is the right holder, not a buffer the launch must
                # keep alive.
                data = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
                ObjC.send(encoder, "setBytes:length:atIndex:",
                          (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
                          ctypes.c_void_p,
                          ctypes.cast(data, ctypes.c_void_p), len(payload), index)
        # The threadgroup-space parameters (`threadgroup float* _local0`) are
        # neither buffers nor bytes: Metal allocates them per threadgroup, and the
        # encoder only learns the size through `setThreadgroupMemoryLength:atIndex:`.
        # Skipping the call leaves the allocation at 0 bytes -- reads and writes
        # through the parameter fall off it, and the kernel runs "fine" with nothing
        # behind the pointer. The byte count is the element width times the count
        # the translator rendered into the declaration (`local_sizes` carries it).
        for name, byte_length in kernel.threadgroup_allocations():
            index:int = list(kernel.args).index(name)
            ObjC.send(encoder, "setThreadgroupMemoryLength:atIndex:",
                      (ctypes.c_size_t, ctypes.c_size_t), ctypes.c_void_p,
                      int(byte_length), int(index))
        threadgroups, threads = self._dispatch_shape()
        ObjC.send(encoder, "dispatchThreadgroups:threadsPerThreadgroup:",
                  (MTLSize, MTLSize), ctypes.c_void_p,
                  _mtl_size(threadgroups), _mtl_size(threads))
        ObjC.send(encoder, "endEncoding", (), ctypes.c_void_p)

    def _claim_entries(self)->List[Tuple["Buffer", object, object, int]]:
        """One `(buffer, host_value, need_read_back, claim_serial)` per pointer argument.

        The read-back flag and the claim serial travel together but answer different
        questions: the flag decides whether the bytes are copied home, the serial
        validates the release (`Buffer.unuse`). Conflating them -- passing the serial
        where the flag belongs -- silently skips every copy-back, which is exactly
        what a launch that "ran fine" but left the caller's array untouched looks
        like from the outside.
        """
        return [(buffer, host_value, need, buffer.claim_serial)
                for buffer, host_value, need in self._kernel.take_bound_entries()]

    def _dispatch_shape(self)->Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """The `(threadgroups, threads_per_threadgroup)` pair for this launch."""
        work_sizes:Optional[Extents] = self._work_sizes
        if work_sizes is None:
            # Automatic: the caller did not bracket, so one threadgroup of one
            # thread is the safe launch -- work items that ask for an id get 0.
            return (1, 1, 1), (1, 1, 1)
        if isinstance(work_sizes, int):
            work_sizes = (work_sizes,)
        extents:List[Extents] = list(work_sizes)
        if len(extents) == 1:
            # A bare item count: that many one-thread groups, so every item is one
            # thread. The shared work-size code derived the same total on the other
            # backends; Metal expresses it as a one-axis grid.
            return (int(extents[0]), 1, 1), (1, 1, 1)
        if len(extents) != 2:
            raise ValueError(
                "the work size bracket must be n, (blocks,), or "
                "(blocks_per_grid, threads_per_block)")
        # The pair is CUDA's `(blocks_per_grid, threads_per_block)`, and either side
        # may be written as a scalar (`(4, 256)`) or as a per-axis tuple
        # (`((4, 1, 1), (256, 1, 1))`). Both spellings are accepted here, because the
        # bracket is shared with the other backends and `k[(4, 256)]` is the common
        # form -- the scalar spelling used to raise `TypeError: 'int' object is not
        # iterable`, which made the ordinary case fail.
        threadgroups = _axis_extents(extents[0], "blocks_per_grid")
        threads = _axis_extents(extents[1], "threads_per_block")
        return threadgroups, threads
