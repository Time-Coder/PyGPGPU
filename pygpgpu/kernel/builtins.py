"""The dual-sided device builtins: one function per name a kernel body may call.

This module holds the builtins and nothing else. On a device backend the translator
recognizes a call by *which function it resolves to* (see ``builtin_registry.py``) and
emits the matching OpenCL C or CUDA call, never looking at the body below. On the
``python`` backend the body is what runs, so it implements the same operation on host numpy
values -- that is why each one is written out rather than left as a placeholder.

The two sides must agree, which is why the implementations live here next to the names:
the signatures are what an editor shows, and the bodies are what the interpreter does.
Anything a body needs but a user should not see lives in ``host_runtime.py``; the device
side lives in ``device_builtins.py``.

Outside a kernel, the functions that depend on a work item (``get_global_id`` and friends)
raise a ``SyntaxError`` explaining that; the ones that act on a value (``atomic_add``,
``dot``, ...) simply operate on it.

Usage::

    from pygpgpu import atomic_add, get_global_id

    @kernel
    def k(out: np.ndarray):
        i = get_global_id(0)
        atomic_add(out, i, 1)
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# The vector half of the namespace (dot/length/normalize/clamp/mix/...) is dual-sided
# already: `pygpgpu.vec_types.funcs` holds one implementation that serves both sides, so
# the names are re-exported here to give them a signature in this namespace too. The
# explicit `as` spelling marks them as deliberate re-exports rather than unused imports.
from ..vec_types.funcs import abs_diff as abs_diff
from ..vec_types.funcs import all as all
from ..vec_types.funcs import any as any
from ..vec_types.funcs import clamp as clamp
from ..vec_types.funcs import cross as cross
from ..vec_types.funcs import distance as distance
from ..vec_types.funcs import dot as dot
from ..vec_types.funcs import exp2 as exp2
from ..vec_types.funcs import hadd as hadd
from ..vec_types.funcs import isequal as isequal
from ..vec_types.funcs import isgreater as isgreater
from ..vec_types.funcs import isgreaterequal as isgreaterequal
from ..vec_types.funcs import isless as isless
from ..vec_types.funcs import islessequal as islessequal
from ..vec_types.funcs import isnotequal as isnotequal
from ..vec_types.funcs import isordered as isordered
from ..vec_types.funcs import isunordered as isunordered
from ..vec_types.funcs import length as length
from ..vec_types.funcs import mad as mad
from ..vec_types.funcs import mad24 as mad24
from ..vec_types.funcs import mix as mix
from ..vec_types.funcs import mul24 as mul24
from ..vec_types.funcs import normalize as normalize
from ..vec_types.funcs import rhadd as rhadd
from ..vec_types.funcs import rsqrt as rsqrt
from ..vec_types.funcs import saturate as saturate
from ..vec_types.funcs import select as select
from ..vec_types.funcs import step as step
from ..vec_types.funcs import sqrt as sqrt
from ..vec_types.funcs import sin as sin
from ..vec_types.funcs import cos as cos
from ..vec_types.funcs import tan as tan
from ..vec_types.funcs import asin as asin
from ..vec_types.funcs import acos as acos
from ..vec_types.funcs import atan as atan
from ..vec_types.funcs import atan2 as atan2
from ..vec_types.funcs import sinh as sinh
from ..vec_types.funcs import cosh as cosh
from ..vec_types.funcs import tanh as tanh
from ..vec_types.funcs import exp as exp
from ..vec_types.funcs import log as log
from ..vec_types.funcs import log2 as log2
from ..vec_types.funcs import log10 as log10
from ..vec_types.funcs import pow as pow
from ..vec_types.funcs import floor as floor
from ..vec_types.funcs import ceil as ceil
from ..vec_types.funcs import trunc as trunc
from ..vec_types.funcs import fabs as fabs
from ..vec_types.funcs import fmod as fmod
from ..vec_types.funcs import hypot as hypot
from ..vec_types.funcs import cbrt as cbrt

from .host_runtime import atomic_in_place, local_extent, work_item, work_size_extent

# Everything a kernel body may call, so `from pygpgpu import *` and an editor completing
# `pygpgpu.` show exactly the supported set and nothing else.
__all__ = [
    # work-item indexing
    "get_global_id", "get_local_id", "get_group_id",
    "get_global_size", "get_local_size", "get_num_groups",
    # CUDA's index vectors (three axes each)
    "blockIdx", "threadIdx", "blockDim", "gridDim",
    # synchronisation and launch shape
    "barrier", "work_size", "zeros_per_group", "zeros_per_block",
    # atomics
    "atomic_add", "atomic_sub", "atomic_min", "atomic_max",
    "atomic_and", "atomic_or", "atomic_xor", "atomic_xchg", "atomic_cas",
    # math and vector builtins (defined in pygpgpu.vec_types.funcs)
    "rsqrt", "exp2", "clamp", "mix", "mad", "saturate", "step", "select",
    "any", "all",
    "isequal", "isnotequal", "isgreater", "isgreaterequal",
    "isless", "islessequal", "isordered", "isunordered",
    "abs_diff", "hadd", "rhadd", "mad24", "mul24",
    "dot", "cross", "normalize", "length", "distance",
    # scalar math intrinsics (also from pygpgpu.vec_types.funcs)
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "log2",
    "log10",
    "pow",
    "floor",
    "ceil",
    "trunc",
    "fabs",
    "fmod",
    "hypot",
    "cbrt",
]


# ---------------------------------------------------------------------------
# Work-item indexing
# ---------------------------------------------------------------------------


def get_global_id(dim: int) -> int:
    """The global id of this work item along ``dim``."""
    return work_item(dim, "get_global_id")


def get_local_id(dim: int) -> int:
    """The id of this work item inside its group along ``dim``."""
    return work_item(dim, "get_local_id") % local_extent(dim, "get_local_id")


def get_group_id(dim: int) -> int:
    """The id of this work item's group along ``dim``."""
    return work_item(dim, "get_group_id") // local_extent(dim, "get_group_id")


def get_global_size(dim: int) -> int:
    """The total number of work items along ``dim``."""
    global_extent, _ = work_size_extent(dim, "get_global_size")
    return global_extent


def get_local_size(dim: int) -> int:
    """The number of work items per group along ``dim``."""
    _, local_size = work_size_extent(dim, "get_local_size")
    return local_size


def get_num_groups(dim: int) -> int:
    """The number of groups along ``dim``."""
    global_extent, local_size = work_size_extent(dim, "get_num_groups")
    return -(-global_extent // local_size)


# ---------------------------------------------------------------------------
# CUDA's index vectors
# ---------------------------------------------------------------------------
#
# A CUDA programmer reaches for `blockIdx`, `threadIdx`, `blockDim` and `gridDim`, and these
# are **vectors** there, not scalars: `blockIdx.x` / `.y` / `.z` are the three axes. Each is
# published here as an `int3` **value**, not a function, so `buf[blockIdx][j]` reads like the
# CUDA original.
#
# A value cannot hold a per-launch number, so it is a lazy one: the attributes are computed
# from the current launch frame when they are read, exactly the numbers the OpenCL spellings
# answer per axis (`blockIdx.x` is `get_group_id(0)`, `threadIdx.x` is `get_local_id(0)`,
# `blockDim.x` is `get_local_size(0)`, `gridDim.x` is `get_num_groups(0)`). Reading the same
# name twice in one launch therefore gives the same answer, and a serial run gives the serial
# work item's -- there is no state to go stale.
#
# Indexing a `local()` buffer takes the **whole vector** -- `buf[blockIdx][j]`, never
# `buf[blockIdx.x][j]` -- because the slice belongs to the work group, and a group is named
# by all three of its axes at once.


class _IndexVector:
    """An `int3` whose components are read from the current launch frame.

    Subscripting it (`blockIdx[0]`) and taking a component (`blockIdx.x`) both give the
    axis's number, so a kernel can use it however it used to spell that axis. The device
    side does not see this class at all: the translator recognizes the *name* and emits the
    vector directly (see `INDEX_VECTOR_BUILTINS`), and this only has to be right on the
    interpreter backend, where the body runs unchanged.
    """

    __slots__ = ("_readers",)

    def __init__(self, **readers:Any)->None:
        object.__setattr__(self, "_readers", readers)

    def _value(self)->Any:
        from pygf import int3

        return int3(*[reader() for reader in self._readers.values()])

    def __getitem__(self, axis:int)->int:
        readers:Any = object.__getattribute__(self, "_readers")
        if not isinstance(axis, int) or axis < 0 or axis >= len(readers):
            raise IndexError(
                f"an index vector has three axes; {axis} is out of range")
        return list(readers.values())[axis]()

    def __getattr__(self, name:str)->int:
        readers:Any = object.__getattribute__(self, "_readers")
        if name not in readers:
            raise AttributeError(
                f"an index vector has .x, .y and .z; it has no '{name}'")
        return readers[name]()

    def __setattr__(self, name:str, value:Any)->None:
        raise TypeError(
            "an index vector is read-only: it reports the launch, and cannot be assigned to")

    def __repr__(self)->str:
        # Never evaluates the readers: this object is a registry key (see
        # `builtin_registry.BUILTIN_NAMES_BY_FUNCTION`), and hashing or printing it during
        # import must not need a launch frame.
        readers:Any = object.__getattribute__(self, "_readers")
        axes:str = ", ".join(readers)
        return f"<index vector: {axes}>"

    def __hash__(self)->int:
        return id(self)


blockIdx:Any = _IndexVector(x=lambda: get_group_id(0), y=lambda: get_group_id(1),
                            z=lambda: get_group_id(2))
threadIdx:Any = _IndexVector(x=lambda: get_local_id(0), y=lambda: get_local_id(1),
                             z=lambda: get_local_id(2))
blockDim:Any = _IndexVector(x=lambda: get_local_size(0), y=lambda: get_local_size(1),
                            z=lambda: get_local_size(2))
gridDim:Any = _IndexVector(x=lambda: get_num_groups(0), y=lambda: get_num_groups(1),
                           z=lambda: get_num_groups(2))


# ---------------------------------------------------------------------------
# Synchronisation and launch shape
# ---------------------------------------------------------------------------


def barrier() -> None:
    """Wait for every work item of the group. A serial run has nothing to wait for."""
    return None


def work_size(*arrays: Any) -> None:
    """Declare the iteration domain of a gid-style kernel.

    Host-side only: the translator erases the statement, and the launch derived from it is
    what the device backends use. Here it only needs to exist so the call is a no-op.
    """
    return None


def zeros_per_group(shape: Any, dtype: Any = "float32") -> Any:
    """Shared (work-group) memory: a zeroed array of ``shape``/``dtype``, one slice per group.

    The arguments are ``np.zeros``' own, in its own order: the **shape first** (an int or a
    tuple of ints) and the **dtype second** (defaulting to float32, as an array of no declared
    dtype would be). So ``zeros_per_group(256, np.int32)`` and
    ``zeros_per_group((64, 64), np.float32)`` read like the numpy call they replace.

    The name says what it builds -- zeros, one per group -- so it cannot be confused with a
    kernel-local variable, which ``local()`` invited. ``zeros_per_block`` is the same function
    under CUDA's word for a group; both spellings are published because a CUDA reader looks
    for "block" and an OpenCL reader for "group", and requiring one of them would make half
    the readers hunt for an alias.

    The shape has to be a **compile-time constant**: the buffer is a fixed-size
    ``__shared__`` array (CUDA/HIP) or one ``__local`` parameter (OpenCL), so a size known
    only at launch cannot declare it.

    This is CUDA's ``__shared__`` / OpenCL's ``__local``, and the value stands for **all
    groups' slices at once**, keyed by group:

    * on the **host** it really is a mapping -- block id -> that group's array -- so
      ``buf[blockIdx]`` selects the slice, and the model is honest about what a shared
      buffer is;
    * on the **device** no indexing happens at all: the emitted code is a plain
      ``__shared__`` array (CUDA/HIP) or ``__local`` parameter (OpenCL), which the hardware
      already gives each group privately, so ``buf[blockIdx][j]`` folds to ``buf[j]``.

    That is why the translator requires the block id for the first subscript but emits
    nothing for it: the index is a statement about *which* group's memory is meant, and
    inside a work item only one answer is reachable.

    On the interpreter backend this returns the mapping itself, with each group's array
    created on first use, so a body that reads ``buf[blockIdx][j]`` works unchanged and a
    group never sees another group's storage. A serial run has no real parallelism, so a
    ``barrier()`` cannot order anything -- a body that *depends* on another work item having
    written first still gets a serial answer, which is the answer a device run would give
    only if the ordering happened to be the same.
    """
    from .host_runtime import require_frame

    frame: Any = require_frame("zeros_per_group")
    return _GroupSlices(shape, dtype, frame)


class _GroupSlices:
    """One group's zeroed array per block id, created on demand.

    Subscripting is the whole point -- ``buf[blockIdx]`` -- and a block id outside the launch
    is refused rather than silently allocated, because that would hand out storage for a
    group the launch does not have.
    """

    __slots__ = ("_shape", "_dtype", "_frame", "_slices")

    def __init__(self, shape: Any, dtype: Any, frame: Any)->None:
        if isinstance(shape, int):
            shape = (shape, )
        self._shape: Any = tuple(int(s) for s in shape)
        self._dtype: Any = dtype
        self._frame: Any = frame
        self._slices: Dict[Any, Any] = {}

    def __getitem__(self, group: Any)->Any:
        import numpy as np

        key: Tuple[int, ...] = _group_key(group)
        groups: List[int] = []
        for axis in range(len(self._frame.global_size)):
            local: int = self._frame.local_size[axis] if axis < len(self._frame.local_size) else 1
            groups.append(-(-self._frame.global_size[axis] // max(1, local)))
        for axis, count in enumerate(groups):
            if axis < len(key) and key[axis] >= count:
                raise IndexError(
                    f"no work group {key} in this launch: it has {tuple(groups)} group(s), so "
                    f"a zeros_per_group() buffer has no storage for it")
        if key not in self._slices:
            self._slices[key] = np.zeros(self._shape, dtype=self._dtype)
        return self._slices[key]

    def __repr__(self)->str:
        return f"<zeros_per_group{self._shape} dtype={self._dtype}>"


def _group_key(group: Any)->Tuple[int, ...]:
    """The block id a `buf[blockIdx]` subscript names, as a tuple of axis indices."""
    if isinstance(group, int):
        return (int(group), )
    # `blockIdx`/`get_group_id(0)` and a plain list of axes are all acceptable: what matters
    # is the three numbers, not the container.
    if isinstance(group, _IndexVector):
        from .host_runtime import get_host_frame

        frame: Any = get_host_frame()
        axes: int = len(frame.global_size) if frame is not None else 3
        return tuple(group[axis] for axis in range(max(1, axes)))
    try:
        return tuple(int(v) for v in group)
    except TypeError:
        raise TypeError(
            f"a zeros_per_group() buffer is indexed by a work group, not by "
            f"{type(group).__name__}: use `buf[blockIdx]`") from None


# One object, two names: a CUDA reader looks for "block" and an OpenCL reader for "group".
# They are the same function, so the host and the device cannot disagree about them however
# they are spelled (see the one-object-per-name rule in AGENTS.md).
zeros_per_block: Any = zeros_per_group


# ---------------------------------------------------------------------------
# Atomics
# ---------------------------------------------------------------------------


def atomic_add(array: Any, index: int, value: Any) -> Any:
    """Add ``value`` to ``array[index]``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: current + value)


def atomic_sub(array: Any, index: int, value: Any) -> Any:
    """Subtract ``value`` from ``array[index]``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: current - value)


def atomic_min(array: Any, index: int, value: Any) -> Any:
    """Keep the smaller of ``array[index]`` and ``value``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: min(current, value))


def atomic_max(array: Any, index: int, value: Any) -> Any:
    """Keep the larger of ``array[index]`` and ``value``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: max(current, value))


def atomic_and(array: Any, index: int, value: Any) -> Any:
    """Bitwise-and ``value`` into ``array[index]``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: current & value)


def atomic_or(array: Any, index: int, value: Any) -> Any:
    """Bitwise-or ``value`` into ``array[index]``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: current | value)


def atomic_xor(array: Any, index: int, value: Any) -> Any:
    """Bitwise-xor ``value`` into ``array[index]``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: current ^ value)


def atomic_xchg(array: Any, index: int, value: Any) -> Any:
    """Store ``value`` into ``array[index]``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: value)


def atomic_cas(array: Any, index: int, compare: Any, value: Any) -> Any:
    """Store ``value`` into ``array[index]`` when it still holds ``compare``.

    Returns the previous value, so a caller can tell whether the swap happened.
    """
    return atomic_in_place(
        array, index, lambda current: value if current == compare else current)
