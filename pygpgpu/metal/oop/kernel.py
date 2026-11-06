"""One Metal entry point, and the launch made on it.

A launch binds the argument buffers by index, sets the pipeline state, dispatches
the threadgroups, commits the command buffer, and -- for the synchronous call this
package makes -- waits for it. Metal's command queue is serial, so a multi-pass
body's consecutive launches run in order with the arrays staying on the device
between them, exactly as the other backends' streams do.

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
  the translator emits the attribute parameters last.
"""
from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Dict, List, NoReturn, Optional, Tuple, Union

import numpy as np

from ..driver.metal import _wait_ready
from ..driver.objc import ObjC

if TYPE_CHECKING:
    from .buffer import Buffer
    from .program import Program
    from .program_info import ArgInfo

# A bracket side: one axis count, or one per axis.
Extents = Union[int, Tuple[int, ...]]

# What `_bind_args` produces per argument: the index to bind at, the buffer a pointer
# argument uploaded (or the bytes a scalar occupies), and whether it is a buffer.
BoundEntry = Tuple[int, Union["Buffer", bytes], bool]


# MTLSize is a three-NSUInteger struct, passed by value to the dispatch calls.
class MTLSize(ctypes.Structure):
    # `MTLSize`'s three fields are `NSUInteger`, which is pointer-sized -- 8 bytes on
    # every Mac. `c_ulong` is that size on macOS but 4 bytes on Windows, so naming
    # `c_size_t` keeps the struct the right size regardless of the host.
    _fields_ = [("width", ctypes.c_size_t), ("height", ctypes.c_size_t),
                ("depth", ctypes.c_size_t)]


def _axis_extents(value:Extents, what:str)->Tuple[int, int, int]:
    """One side of the bracket as Metal's three axes, padded with 1.

    `MTLSize` is always three-dimensional, while a bracket may name one axis (`4`)
    or three (`(4, 1, 1)`). A scalar becomes `(4, 1, 1)` and a shorter tuple is
    padded, because a missing axis means one group along it -- zero would ask for
    an empty dispatch, which is never what a caller wrote.
    """
    if isinstance(value, int):
        return (int(value), 1, 1)
    extents:Tuple[int, ...] = tuple(int(v) for v in value)
    if len(extents) > 3:
        raise ValueError(
            f"the {what} side of a work size bracket has {len(extents)} axes; "
            f"Metal dispatches at most three")
    return extents + (1,) * (3 - len(extents))


def _mtl_size(extents:Tuple[int, ...])->MTLSize:
    """One, two or three axes as the struct Metal's dispatch calls take."""
    values:List[int] = [int(v) for v in extents] + [1, 1, 1]
    return MTLSize(values[0], values[1], values[2])


# The bytes each scalar kernel parameter occupies, in MSL's own layout. This is
# what `setBytes:length:atIndex:` has to copy, so it must be the *device's* size
# rather than the host's: a python `int` is 8 bytes, while `int` in MSL is 4.
_SCALAR_SIZES:Dict[str, int] = {
    "bool": 1, "char": 1, "uchar": 1,
    "short": 2, "ushort": 2, "half": 2,
    "int": 4, "uint": 4, "float": 4,
    "long": 8, "ulong": 8, "double": 8,
}

# A complex value is a *structure* of two reals rather than a scalar, and it is bound the way
# a scalar is: the generated signature declares `c64 value`, a value in MSL's thread address
# space, so the host writes the record's bytes with `setBytes:`. The two names are the fixed
# ones the complex header declares (see `program_info.COMPLEX_TYPE_NAMES`), and each spells
# the format of *one* component -- `re` then `im`, in declaration order, which is numpy's own
# complex layout, so a `complex64`/`complex128` array argument and this scalar share one
# memory shape.
_COMPLEX_FORMATS:Dict[str, str] = {"c64": "<f", "c128": "<d"}


def _scalar_bytes(device_type:str, value:object)->bytes:
    """One scalar argument as the bytes the device's own type would hold.

    A kernel's extent parameter is declared `const int`, so the host has to send
    four bytes even though python's `int` is wider -- the length passed to
    `setBytes:` is what the device copies, and a mismatch makes the pipeline read
    half a value.
    """
    if device_type in _COMPLEX_FORMATS:
        return _complex_bytes(device_type, value)
    size:int = _SCALAR_SIZES.get(device_type, 4)
    if device_type in ("float",):
        return _pack_real(device_type, "<f", float(value), 3.4028235e38)
    if device_type in ("double",):
        return _pack_real(device_type, "<d", float(value), None)
    if device_type in ("half",):
        # `<e` is IEEE half in little endian, which is what `setBytes:` expects and
        # what Metal's own storage layout is. `numpy.float16(...).tobytes()` answers
        # the *host's* byte order, so on a big-endian host it would disagree with
        # every other branch here -- all of which spell the order explicitly.
        return _pack_real(device_type, "<e", float(value), 65504.0)
    if device_type in ("bool", "uchar", "char"):
        return _pack_integer(device_type, size, int(bool(value) if device_type == "bool"
                                                    else value),
                             signed=device_type == "char")
    return _pack_integer(device_type, size, int(value),
                         signed=device_type in ("char", "short", "int", "long"))


# The inclusive range each declared scalar type can hold. `to_bytes` raises a bare
# `OverflowError` ("int too big to convert") for a value outside it, which says
# nothing about which parameter was wrong or what the type accepts.
_SCALAR_RANGES:Dict[str, Tuple[int, int]] = {
    "bool": (0, 1),
    "char": (-128, 127), "uchar": (0, 255),
    "short": (-32768, 32767), "ushort": (0, 65535),
    "int": (-2147483648, 2147483647), "uint": (0, 4294967295),
    "long": (-9223372036854775808, 9223372036854775807),
    "ulong": (0, 18446744073709551615),
}


def _pack_real(device_type:str, code:str, value:float,
               largest:Optional[float])->bytes:
    """One floating-point scalar, with an overflow spelled out.

    `struct.pack` raises `OverflowError: float too large to pack with f format -- the
    format letter names neither the kernel parameter's type nor the largest value it
    holds. ±infinity and NaN are accepted (they are valid IEEE values, not overflow).
    """
    import math
    import struct
    if largest is not None and math.isfinite(value) and abs(value) > largest:
        raise OverflowError(
            f"a kernel parameter declared {device_type} cannot hold {value!r}: "
            f"its largest finite value is {largest}")
    return struct.pack(code, value)


def _complex_bytes(device_type:str, value:object)->bytes:
    """One complex argument as the two real components MSL's record holds.

    The components are packed through `_pack_real`, so a `c64` refuses a value its `float`
    component cannot hold with the same message a `float` parameter gives (and keeps ±inf
    and NaN, which IEEE allows). The record is what `setBytes:` copies, so its length is the
    structure's -- 8 bytes for a `c64`, 16 for a `c128`.
    """
    code:str = _COMPLEX_FORMATS[device_type]
    largest:Optional[float] = 3.4028235e38 if device_type == "c64" else None
    real:float = float(np.real(value))
    imaginary:float = float(np.imag(value))
    return (_pack_real(device_type, code, real, largest)
            + _pack_real(device_type, code, imaginary, largest))

def _pack_integer(device_type:str, size:int, value:int, signed:bool)->bytes:
    """One integer scalar as `size` little-endian bytes, with the range checked."""
    limits:Optional[Tuple[int, int]] = _SCALAR_RANGES.get(device_type)
    if limits is not None and not (limits[0] <= value <= limits[1]):
        raise OverflowError(
            f"a kernel parameter declared {device_type} cannot hold {value}: "
            f"its range is {limits[0]}..{limits[1]}")
    return value.to_bytes(size, "little", signed=signed)


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
        self._bound_entries:List[Tuple["Buffer", object, bool]] = []

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
        self._bound_entries = []
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

    # `@kernel` offers `submit`/`async_call` on every backend, and a Metal kernel
    # must answer them with a clear refusal rather than an AttributeError: the
    # asynchronous form needs a completion handler on the command buffer, which is
    # the one thing the synchronous launch here does not build. The message says
    # what to use instead, which is what a user reaching for a future needs.
    def submit(self, *args:object, **kwargs:object)->NoReturn:
        raise NotImplementedError(
            "asynchronous launches are not supported on the metal backend yet: the "
            "launch path waits on its command buffer (`waitUntilCompleted`), and the "
            "future-based form needs a completion handler. Call the kernel directly "
            "(it is synchronous), or run it on the opencl or cuda backend")

    def async_call(self, *args:object, **kwargs:object)->NoReturn:
        raise NotImplementedError(
            "asynchronous launches are not supported on the metal backend yet: the "
            "launch path waits on its command buffer (`waitUntilCompleted`), and the "
            "asyncio form needs a completion handler. Call the kernel directly "
            "(it is synchronous), or run it on the opencl or cuda backend")


class BoundKernel:
    """A kernel with its work size fixed: the object a launch is made on."""

    def __init__(self, kernel:MetalKernel,
                 work_sizes:Optional[Extents])->None:
        self._kernel = kernel
        self._work_sizes = work_sizes

    def __call__(self, *args:object, **kwargs:object)->None:
        _wait_ready()
        kernel = self._kernel
        bound = kernel._bind_args(args, kwargs)

        command_buffer:int = kernel.program.context.command_queue.new_command_buffer()
        try:
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
            ObjC.send(command_buffer, "commit", (), ctypes.c_void_p)
            ObjC.send(command_buffer, "waitUntilCompleted", (), ctypes.c_void_p)
            # The launch has waited, so the copies back can happen inline.
            for buffer, host_value, need_read_back in kernel._bound_entries:
                _finish_buffer(buffer, host_value, need_read_back)
            return None
        finally:
            ObjC.release(int(command_buffer))

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
                   need_read_back:bool)->None:
    """Copy a written buffer back into the caller's array and release its claim.

    `need_read_back` comes from the argument's `ArgInfo` -- the shared read/write
    analysis every backend consults -- rather than from a mask tested here, so a
    buffer the body never writes is not copied back and a written one always is.

    StorageModeShared makes the device's bytes the host's bytes, so the read-back is
    a copy from the buffer's own contents.
    """
    try:
        if need_read_back and host_value is not None and isinstance(
                host_value, np.ndarray):
            source = ctypes.string_at(buffer._contents(), buffer.size)
            host_value.reshape(-1).view(np.uint8)[:buffer.size] = np.frombuffer(
                source, dtype=np.uint8)[:host_value.nbytes]
    finally:
        # `unuse()` raises when the buffer was never claimed, and a raise here would
        # replace whatever exception the read-back was already unwinding. The claim is
        # released on every path; a buffer that was somehow not in use has nothing to
        # release, so the failure is not worth shadowing the real one.
        try:
            buffer.unuse()
        except RuntimeError:
            pass

# The shared `oop` spelling for a launchable entry point. Every other backend's
# `program`/`kernel` module publishes `Kernel`, and a caller that walks a backend's
# modules generically (`for backend in (...): backend.oop.kernel.Kernel`) would
# otherwise hit an `AttributeError` only on Metal. The class keeps its own name
# where the Metal-specific machinery is described; this is the cross-backend one.
Kernel = MetalKernel
