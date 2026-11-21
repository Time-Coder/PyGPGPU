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
from ...driver.metal import MetaMetal
from ...driver.metal import _wait_ready
from .copyback import (_MTCOMMAND_BUFFER_STATUS_COMPLETED, _POLL_INTERVAL,
                      _finish_buffer)
from ...driver.objc import ObjC
from ....utils import settle_future

if TYPE_CHECKING:
    import asyncio

    from ..buffer import Buffer
    from ..program import Program
    from ..program_info import ArgInfo

from .launch import BoundKernel as BoundKernel
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
        # The launch path reads `kernel._args` on every backend the decorator drives
        # (the OpenCL kernel publishes the same attribute through its `args` property
        # pair below), so Metal carries the private spelling and keeps the public one
        # an alias of it.
        self._args:Dict[str, "ArgInfo"] = {}
        self._arg_list:List["ArgInfo"] = []
        self._arg_names:List[str] = []
        # The device signature's trailing attribute parameters, which the threadgroup
        # dispatch fills in. They are not bindable, and `args` therefore stops before
        # them -- which is what makes a host-bound parameter's position in `args` equal
        # its index in the device signature.
        self.dispatch_parameters:List[str] = []
        self.type_checked:bool = False
        self._pipeline:Optional[int] = None
        self._bound_entries:List[Tuple["Buffer", object, object]] = []

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

    @property
    def args(self)->Dict[str, "ArgInfo"]:
        """The bindable parameters, keyed by name (the shared launch surface)."""
        return self._args

    @args.setter
    def args(self, args:Dict[str, "ArgInfo"])->None:
        self._args = args
        self._arg_list = list(args.values())
        self._arg_names = list(args.keys())

    def set_threadgroup_allocations(self, byte_sizes:List[int])->None:
        """Record the byte size of every `threadgroup`-space parameter, in order.

        The launch metadata carries one entry per `local_buffers` record (element
        width times count), and the parser read the parameters in the same
        declaration order -- so the two lists zip positionally.
        """
        self._threadgroup_allocations:List[Tuple[str, int]] = list(zip(
            (name for name, info in self._args.items()
             if getattr(info, "address", None) == "local"),
            byte_sizes))

    def threadgroup_allocations(self)->List[Tuple[str, int]]:
        """The `(parameter name, byte length)` pairs `_encode` announces."""
        return getattr(self, "_threadgroup_allocations", [])

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
        # `newFunctionWithName:` takes an `NSString *`, so the bridge passes a real
        # string object -- a bare `c_char_p` is a garbage object pointer there.
        function_id = ObjC.send_ptr(self._program.id, "newFunctionWithName:",
                                    (ctypes.c_void_p,), ObjC.string(self._name))
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

        A `threadgroup`-space pointer (`_local0`, what a `local_array_per_group`
        buffer declares) is excluded as well: the dispatch allocates that memory per
        threadgroup and no host buffer exists to bind -- the same slot the OpenCL
        launch skips when it walks its argument list.
        """
        return [name for name in self.args
                if not getattr(self.args[name], "dispatch_supplied", False)
                and getattr(self.args[name], "address", None) != "local"]

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
                is_struct_param:bool = (self._program.structs.get(arg_info.base_type_str)
                                        is not None)
                if not is_struct_param and not isinstance(argument, np.ndarray):
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

    def async_call(self, *args:object, **kwargs:object)->"asyncio.Future":
        """Launch without waiting, returning an ``asyncio.Future`` to ``await``."""
        import asyncio

        from ....utils import _asyncio_future_of
        return self.__getitem__(None)._launch_async(
            args, kwargs, lambda: _asyncio_future_of(asyncio))



# The shared `oop` spelling for a launchable entry point. Every other backend's
# `program`/`kernel` module publishes `Kernel`, and a caller that walks a backend's
# modules generically (`for backend in (...): backend.oop.kernel.Kernel`) would
# otherwise hit an `AttributeError` only on Metal. The class keeps its own name
# where the Metal-specific machinery is described; this is the cross-backend one.
Kernel = MetalKernel




