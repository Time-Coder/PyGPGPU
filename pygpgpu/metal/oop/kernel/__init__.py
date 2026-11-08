"""One Metal entry point, and the launch made on it.

A launch binds the argument buffers by index, sets the pipeline state, dispatches
the threadgroups, and commits the command buffer. The synchronous call waits for it
(`waitUntilCompleted`) and copies the written arguments back inline; an asynchronous
call (`submit`/`async_call`) returns a future at once, and a poller watches the
command buffer's `status` until the work is done -- Metal reports completion through
the command buffer's status, not through a separate event object. Metal's command
queue is serial, so a multi-pass body's consecutive launches run in order with the
arrays staying on the device between them, exactly as the other backends' streams do.

The work size is CUDA's pair, `(blocks_per_grid, threads_per_block)`, as AGENTS.md
requires of every backend: `blocks` is literally the threadgroup count and
`threads` literally the threadgroup size, which is why Metal needs no conversion
at all -- `dispatchThreadgroups:threadsPerThreadgroup:` takes that same pair under
its own names.

Two bindings are Metal's own and are load-bearing:

* **A pointer argument travels through a buffer, a scalar does not.** A `device`/`constant`
  pointer (`device float* a`) is bound with `setBuffer:offset:atIndex:` and the device
  reads it by indexing; a *scalar* (`int n`, `float scale`) lives in MSL's thread address
  space, so it is bound with `setBytes:length:atIndex:` and the device reads the value
  directly. Binding a scalar as a buffer presents a pointer where the kernel declared a
  value, and the pipeline rejects the launch with a type mismatch. `_bind_args` answers
  `(index, value, is_buffer)` so the encoder can branch; the width comes from the
  declared type (`_SCALAR_SIZES`), not from python's own. That is why the parser's
  argument layout, not the C signature, is what a launch binds.
* **A dispatch-supplied parameter is not bound at all.** The work-item numbers
  (`[[thread_position_in_grid]]`, ...) are filled in by the threadgroup dispatch;
  binding a buffer there would overwrite the value the GPU is about to write. The
  encoder therefore sets only the host-bound indices, which are the first ones --
  the translator emits the attribute parameters last."""
# Re-export shim: the scalar layer moved to `scalars` and every name it published
# stays importable from here (the binding tests import _SCALAR_SIZES and
# _scalar_bytes from this module). The classes stay: four tests read this file's
# source text -- the module docstring, `def _finish_buffer` before
# `Kernel = MetalKernel`, the dispatch selector tuple, and the release sequence.
from __future__ import annotations
import ctypes
import concurrent.futures
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...driver.metal import _wait_ready
from ...driver.objc import ObjC
from ....utils import settle_future

if TYPE_CHECKING:
    from ..buffer import Buffer
    from ..program import Program
    from ..program_info import ArgInfo

from .scalars import (
    BoundEntry,
    Extents,
    MTLSize,
    _COMPLEX_FORMATS,
    _SCALAR_RANGES,
    _SCALAR_SIZES,
    _axis_extents,
    _complex_bytes,
    _mtl_size,
    _pack_integer,
    _pack_real,
    _scalar_bytes,
)



class MetalKernel:
    """One Metal entry point: its argument layout and its compute pipeline state."""

    def __init__(self, program:"Program", name:str)->None:
        self._program:"Program" = program
        self._name:str = name
        self.args:Dict[str, "ArgInfo"] = {}
        # The device signature's trailing attribute parameters, which the threadgroup
        # dispatch fills in. They are not bindable, and `args` therefore stops before
        # them -- which is what makes a host-bound parameter's position in `args` equal
        # its index in the device signature.
        self.dispatch_parameters:List[str] = []
        self.type_checked:bool = False
        self._pipeline:Optional[int] = None

    def take_bound_entries(self)->List[Tuple["Buffer", object, object]]:
        """The argument buffers the last `_bind_args` claimed, and consume the list.

        A launch needs, after the device has finished, one copy-back and one claim
        release per pointer argument. The entries travel with the launch rather than
        living on the kernel: two launches of one kernel (two threads, or a caller
        still reading a future while the next `submit()` starts) must not overwrite
        each other's list.
        """
        entries = self._bound_entries
        self._bound_entries = []
        return entries

    def __repr__(self)->str:
        return f"MetalKernel({self._name})"

    @property
    def name(self)->str:
        return self._name

    @property
    def program(self)->"Program":
        return self._program

    @property
    def pipeline(self)->int:
        """The `MTLComputePipelineState`, built on first launch."""
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()
        return self._pipeline

    def _build_pipeline(self)->int:
        function_id = ObjC.send_ptr(self._program.id, "newFunctionWithName:",
                                    (ctypes.c_char_p,), self._name.encode("utf-8"))
        if not function_id:
            raise RuntimeError(
                f"{self._program.file_name} declares no function {self._name!r}")
        error = ctypes.c_void_p(0)
        try:
            pipeline = ObjC.send(
                self._program.device.id, "newComputePipelineStateWithFunction:error:",
                (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
                ctypes.c_void_p, function_id, ctypes.byref(error))
            if not pipeline and error:
                # Read the description before the object is released below: `describe`
                # sends it a message, and a released object may already be gone.
                message:str = ObjC.describe(int(error.value or 0))
                raise RuntimeError(
                    f"building a pipeline for {self._name} failed: {message}")
            return int(pipeline or 0)
        finally:
            # `newFunctionWithName:` answers a +1 reference that only this call uses;
            # the pipeline keeps its own. The `NSError` is a +1 reference too.
            ObjC.release(function_id)
            if error:
                ObjC.release(int(error.value or 0))

    def _host_bound_names(self)->List[str]:
        """The parameters the host binds, in declaration order.

        These are the first entries of the signature: the translator appends the
        attribute parameters last (see `parameter_declarations`), so slicing off the
        dispatch-supplied tail is all an ordinary generated kernel needs. A
        handwritten source may order them differently, which is why the test is per
        parameter rather than positional.
        """
        return [name for name in self.args
                if not getattr(self.args[name], "dispatch_supplied", False)]

    def _bind_args(self, args:Tuple[object, ...],
                   kwargs:Dict[str, object])->List[BoundEntry]:
        """Upload every host-bound argument, paired with its index and its kind.

        Each entry is `(index, value, is_buffer)`. A pointer argument becomes a
        `MTLBuffer`; a scalar becomes its **bytes**, which MSL binds with
        `setBytes:length:atIndex:` rather than `setBuffer:offset:atIndex:`.

        That distinction is load-bearing, not stylistic: a scalar kernel parameter
        lives in the thread address space, and a buffer bound there would present
        the device a pointer where it declared a value -- the pipeline rejects the
        launch with a type mismatch. Only a `device`/`constant` pointer takes a
        buffer.
        """

        if kwargs:
            raise TypeError(
                f"{self._name}() takes positional arguments only, got "
                f"{', '.join(sorted(kwargs))}")
        names:List[str] = self._host_bound_names()
        if len(args) != len(names):
            raise TypeError(
                f"{self._name}() takes {len(names)} arguments but {len(args)} "
                f"were given ({', '.join(names) or 'none'})")
        # A host-bound parameter's position in `args` IS its buffer index -- that is true
        # only while `args` excludes the attribute parameters, because the device signature
        # declares them at the end. This check is what keeps it true: if a parser change
        # ever let an attribute slip into `args`, every index after it would shift and the
        # launch would bind arrays to the wrong slots -- silently, on the device.
        overlap:List[str] = [name for name in self.args
                             if name in self.dispatch_parameters]
        if overlap:
            raise RuntimeError(
                f"{self._name}: the attribute parameter(s) {', '.join(overlap)} appear "
                f"among the bindable arguments; they are filled in by the dispatch and "
                f"must not be bound, so every buffer index after them would be wrong")
        self._bound_entries:List[Tuple["Buffer", object, object]] = []
        bound:List[BoundEntry] = []
        for position, name in enumerate(names):
            index:int = list(self.args).index(name)
            argument:object = args[position]
            arg_info:"ArgInfo" = self.args[name]
            if arg_info.is_ptr:
                if not isinstance(argument, np.ndarray):
                    raise TypeError(
                        f"argument '{name}' must be np.ndarray, got "
                        f"{type(argument).__name__}")
                if self.type_checked:
                    arg_info.check_type(argument)
                buffer, _event = arg_info.use_buffer(
                    self._program.context.command_queue, argument)
                # The flag travels with the buffer rather than being re-derived from
                # `buffer.kernel_flags` after the launch: `need_read_back` is the
                # shared `ArgInfo` contract every backend reads, and a second
                # derivation here could drift from it.
                self._bound_entries.append((buffer, argument, arg_info.need_read_back))
                bound.append((index, buffer, True))
            else:
                if self.type_checked:
                    arg_info.check_type(argument)
                bound.append((index, _scalar_bytes(arg_info.base_type_str, argument),
                              False))
        return bound

    def __getitem__(self, work_sizes:Optional[Extents])->"BoundKernel":
        """Bind a CUDA-style `(blocks, threads_per_block)` work size."""
        return BoundKernel(self, work_sizes)

    def __call__(self, *args:object, **kwargs:object)->None:
        self.__getitem__(None)(*args, **kwargs)

    def submit(self, *args:object, **kwargs:object)->concurrent.futures.Future:
        """Launch without waiting, returning a ``concurrent.futures.Future``."""
        return self.__getitem__(None)._launch_async(
            args, kwargs, concurrent.futures.Future)

    def async_call(self, *args:object, **kwargs:object)->Any:
        """Launch without waiting, returning an ``asyncio.Future`` to ``await``."""
        import asyncio

        from ....utils import _asyncio_future_of
        return self.__getitem__(None)._launch_async(
            args, kwargs, lambda: _asyncio_future_of(asyncio))


class BoundKernel:
    """A kernel with its work size fixed: the object a launch is made on."""

    def __init__(self, kernel:MetalKernel,
                 work_sizes:Optional[Extents])->None:
        self._kernel = kernel
        self._work_sizes = work_sizes

    def __call__(self, *args:object, **kwargs:object)->None:
        """The synchronous launch: encode, commit, wait, copy back, release."""
        _wait_ready()
        kernel = self._kernel
        bound = kernel._bind_args(args, kwargs)

        command_buffer:int = kernel.program.context.command_queue.new_command_buffer()
        try:
            self._encode(command_buffer, bound)
            ObjC.send(command_buffer, "commit", (), ctypes.c_void_p)
            ObjC.send(command_buffer, "waitUntilCompleted", (), ctypes.c_void_p)
            # The launch has waited, so the copies back can happen inline.
            error_obj:int = ObjC.send_ptr(command_buffer, "error")
            encode_error:Optional[BaseException] = None
            if error_obj:
                encode_error = RuntimeError(
                    f"the metal launch failed: {ObjC.describe(error_obj)}")
            for buffer, host_value, claim in self._claim_entries():
                _finish_buffer(buffer, host_value, claim, encode_error)
        finally:
            ObjC.release(int(command_buffer))

    def _launch_async(self, args:Tuple[object, ...],
                      kwargs:Dict[str, object], future_type:type)->Any:
        """Launch without waiting; the future settles when the work has finished.

        Metal reports completion through the command buffer's `status`, so a poller
        thread watches it and settles the future when the buffer completes -- the
        shape the other backends' stream callbacks give them, reached through the
        one completion mechanism this driver surface exposes without blocks. An
        encode-time failure never starts a poller: it settles the future directly.
        """
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
            for buffer, host_value, claim in entries:
                _finish_buffer(buffer, host_value, claim)
            settle_future(future)
            # The buffer's work is done and the launch holds nothing else against it.
            ObjC.release(int(command_buffer))

        threading.Thread(target=watch, name="pygpgpu-metal", daemon=True).start()
        return future

    def _enqueue(self, args:Tuple[object, ...],
                 kwargs:Dict[str, object])->Tuple[int, List[Tuple["Buffer", object, int]]]:
        """Bind, encode and commit without waiting; answer what completion needs.

        `(command_buffer, entries)`: the buffer a completion waits on or polls, and
        one `(buffer, host_value, claim_serial)` per pointer argument -- the serial
        the release must carry so it cannot free a claim the buffer no longer holds
        (see `Buffer.unuse`). An encode-time failure releases the buffer here and
        re-raises, so a caller's `finally` never sees a half-encoded buffer.
        """
        _wait_ready()
        kernel = self._kernel
        bound = kernel._bind_args(args, kwargs)
        command_buffer:int = kernel.program.context.command_queue.new_command_buffer()
        try:
            self._encode(command_buffer, bound)
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
        threadgroups, threads = self._dispatch_shape()
        ObjC.send(encoder, "dispatchThreadgroups:threadsPerThreadgroup:",
                  (MTLSize, MTLSize), ctypes.c_void_p,
                  _mtl_size(threadgroups), _mtl_size(threads))
        ObjC.send(encoder, "endEncoding", (), ctypes.c_void_p)

    def _claim_entries(self)->List[Tuple["Buffer", object, int]]:
        """One `(buffer, host_value, claim_serial)` per pointer argument.

        The serial is captured now, at claim time: a release that fires after the
        buffer was handed out again carries a stale serial and must not free the new
        holder's claim.
        """
        return [(buffer, host_value, buffer.claim_serial)
                for buffer, host_value, _need in self._kernel.take_bound_entries()]

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


def _finish_buffer(buffer:"Buffer", host_value:object,
                   need_read_back:object, error:Optional[BaseException]=None)->None:
    """Copy a written buffer back into the caller's array and release its claim.

    `need_read_back` comes from the argument's `ArgInfo` -- the shared read/write
    analysis every backend consults -- rather than from a mask tested here, so a
    buffer the body never writes is not copied back and a written one always is.
    The synchronous launch passes the read-back flag and no error; the asynchronous
    poller passes the claim's serial instead, so the release cannot free a claim the
    buffer no longer carries.

    StorageModeShared makes the device's bytes the host's bytes, so the read-back is
    a copy from the buffer's own contents.
    """
    try:
        if need_read_back is True and error is None and host_value is not None and \
                isinstance(host_value, np.ndarray):
            source = ctypes.string_at(buffer._contents(), buffer.size)
            host_value.reshape(-1).view(np.uint8)[:buffer.size] = np.frombuffer(
                source, dtype=np.uint8)[:host_value.nbytes]
    finally:
        # The claim is released on every path; a release carrying a serial the buffer
        # no longer holds is a no-op (see MemObject.unuse), so it never shadows the
        # read-back's own error. A raise from the release itself (a buffer that never
        # had a claim) is absorbed: the read-back's failure is the one worth seeing.
        try:
            buffer.unuse(need_read_back if isinstance(need_read_back, int) else None)
        except RuntimeError:
            pass

# `MTLCommandBufferStatus`: the value `status` answers once every command in the
# buffer has finished. The poller waits for the buffer to reach it.
_MTCOMMAND_BUFFER_STATUS_COMPLETED:int = 3
# How long the completion poller sleeps between status reads. A launch takes
# microseconds to tens of milliseconds; this is short enough to feel immediate and
# long enough that a sleeping poller costs nothing.
_POLL_INTERVAL:float = 0.0002

# The shared `oop` spelling for a launchable entry point. Every other backend's
# `program`/`kernel` module publishes `Kernel`, and a caller that walks a backend's
# modules generically (`for backend in (...): backend.oop.kernel.Kernel`) would
# otherwise hit an `AttributeError` only on Metal. The class keeps its own name
# where the Metal-specific machinery is described; this is the cross-backend one.
Kernel = MetalKernel




