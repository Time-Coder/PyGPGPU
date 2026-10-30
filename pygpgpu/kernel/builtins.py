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

from typing import Any

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

from .host_runtime import atomic_in_place, local_extent, work_item, work_size_extent

# Everything a kernel body may call, so `from pygpgpu import *` and an editor completing
# `pygpgpu.` show exactly the supported set and nothing else.
__all__ = [
    # work-item indexing
    "get_global_id", "get_local_id", "get_group_id",
    "get_global_size", "get_local_size", "get_num_groups",
    # synchronisation and launch shape
    "barrier", "work_size", "local",
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


def local(dtype: Any, count: int) -> Any:
    """Allocate ``count`` elements of ``dtype`` in local (shared) memory.

    Device-only, and the interpreter says so rather than guessing: a shared buffer is
    meaningless without barriers, and ``barrier()`` is a no-op in a serial run, so an
    algorithm that counts into shared memory and then merges would silently produce a
    different number there. Run such a kernel on a device backend, or write the same
    result with atomics straight into an array argument.
    """
    raise SyntaxError(
        "local() is not supported by the python backend: it needs a work-group barrier to "
        "separate the phases that share the buffer, and a serial run cannot honor one. Use "
        "a cuda/opencl/hip backend for this kernel, or write the same result with "
        "atomic_add straight into an array argument") from None


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
