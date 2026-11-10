"""Work-group storage a gid-style body indexes directly, with no block subscript.

`local_array` / `shared_array` build the same device declaration the per-group pair does
(`__shared__ T name[N]` on CUDA/HIP, one `__local` parameter on OpenCL), but they are for one
work item of one known block, so there is nothing to select: the value *is* the caller's block
and `buf[j]` is the element. That is only meaningful where every work item of the body runs as
its own work item, which is what the gid style is; a magic-loop body stands for the whole
launch, so it keeps the per-group spelling (`local_array_per_group`), whose value is the
mapping `buf[blockIdx][j]`. The translator refuses the block-local pair in a magic-loop body
for that reason, and the host body below says the same thing.

The interpreter still has one array per block, not one per work item: a serial run executes the
work items of a group one after another, and the storage they share has to be shared for the
body to mean anything.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Tuple, Union

if TYPE_CHECKING:
    import numpy as np
    from ..python_backend import PythonFrame


def local_array(shape: object, dtype: object = "float32") -> object:
    """This work group's own zeroed array of ``shape``/``dtype``, indexed directly.

    ``local_array(256, np.int32)`` is CUDA's ``__shared__ int buf[256]`` (OpenCL's
    ``__local int*``) for the block the calling work item belongs to: ``buf[j]`` is its
    j-th element, and every work item of the same block reads and writes the same array.
    There is no ``blockIdx`` subscript, because the value already *is* that block's memory.

    The arguments are ``np.zeros``' own and in its own order -- **shape first, dtype
    second** -- and the shape has to be a compile-time constant, exactly as it does for
    `local_array_per_group`: the declaration's size is baked into the generated source.

    **This spelling is for the gid style.** A magic-loop body runs once and stands for every
    work item of the launch, so which block a bare array would belong to has no answer
    there; that body keeps `local_array_per_group`, whose value is a mapping and is indexed
    by its group (`buf[blockIdx][j]`). The translator refuses this call in a magic-loop body,
    and the interpreter refuses it for the same reason -- the two sides have to agree.

    On the interpreter backend the array is created on the first work item of each block and
    reused by the rest, so a body that reads a slot another work item wrote still sees it.
    A ``barrier()`` cannot order anything in a serial run, so a body that depends on a
    *later* work item's write reads what the serial order has produced so far.
    """
    return _block_array(shape, dtype, "local_array")


def shared_array(shape: object, dtype: object = "float32") -> object:
    """CUDA's spelling of `local_array`: this block's own zeroed array, indexed directly.

    ``shared_array(256, np.int32)`` is `local_array(256, np.int32)` -- the same device
    declaration, the same direct indexing, and the same rule that it belongs to the gid
    style. Both names are published because a CUDA reader looks for "shared" and an OpenCL
    reader for "local", exactly as `local_array_per_group` and `shared_array_per_block`
    serve that pair.

    The host model, the dtype and shape rules and the gid-only restriction are documented on
    `local_array`.
    """
    return _block_array(shape, dtype, "shared_array")


def _block_array(shape: object, dtype: object, what: str) -> object:
    """The caller's block storage for this creation site, created on first use."""
    import sys

    from ..host_runtime import require_frame

    frame: "PythonFrame" = require_frame(what)
    if not frame.per_work_item:
        raise SyntaxError(
            f"{what}() is for the gid style only: a magic-loop body runs once for the whole "
            f"launch, so a bare block array would have no work group to belong to. Use "
            f"local_array_per_group(...) and index it by its group "
            f"(`buf[blockIdx][j]`)") from None
    # Two `local_array(...)` calls in one body are two arrays, so the storage is keyed by
    # the *call site*: the statement's file, its first line and the line being executed are
    # stable for the life of the process and tell two creations apart. Frame 0 is this
    # function, frame 1 the builtin, frame 2 the kernel body.
    caller = sys._getframe(2)
    site: str = f"{caller.f_code.co_filename}:{caller.f_code.co_firstlineno}:{caller.f_lineno}"
    extents: Tuple[int, ...] = _extents(shape, what)
    block: Tuple[int, ...] = (frame.block_of(frame.coords)
                              if frame.coords is not None else (0, ))
    return _BlockArray(frame.shared_array(site, block, extents, dtype), extents, what)


def _extents(shape: object, what: str) -> Tuple[int, ...]:
    """The shape as extents, refusing what a fixed-size declaration cannot carry.

    A shape is what ``np.zeros`` accepts: an int, or a tuple/list of ints (an array's own
    ``.shape`` is a tuple). Anything else is refused here rather than reaching the
    declaration as something the host would have to guess at.
    """
    extents: Tuple[int, ...]
    if isinstance(shape, int):
        extents = (int(shape), )
    elif isinstance(shape, (tuple, list)):
        try:
            extents = tuple(int(extent) for extent in shape)
        except (TypeError, ValueError):
            raise SyntaxError(
                f"{what}() shape must be a tuple of ints, and this one is {shape!r}") from None
    else:
        raise SyntaxError(
            f"{what}() takes a shape: an int or a tuple of ints, not "
            f"{type(shape).__name__}") from None
    if not extents or any(extent <= 0 for extent in extents):
        raise SyntaxError(
            f"{what}() shape must be one or more positive extents, and this one is "
            f"{extents}: the array is a fixed-size work-group declaration") from None
    return extents


class _BlockArray:
    """One work group's array, standing in for it so a body indexes it directly.

    Reads and writes go straight to the array the frame holds for this block, which is what
    makes two work items of one block see each other's stores. The rest of the numpy
    interface a body may reach for (``shape``, ``reshape``, ...) is delegated, so an atomic
    taking this value works exactly as it does on a plain array.
    """

    __slots__ = ("_store", "_shape", "_what")

    def __init__(self, store: "np.ndarray", shape: Tuple[int, ...], what: str)->None:
        self._store: "np.ndarray" = store
        self._shape: Tuple[int, ...] = shape
        self._what: str = what

    @property
    def shape(self)->Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self)->object:
        return self._store.dtype

    @property
    def ndim(self)->int:
        return len(self._shape)

    def __getitem__(self, index: Union[int, Tuple[int, ...], slice])->"np.generic":
        """One element of this block's array: a numpy scalar, as a plain array answers."""
        return self._store[index]

    def __setitem__(self, index: Union[int, Tuple[int, ...], slice], value: object)->None:
        self._store[index] = value

    def __len__(self)->int:
        return int(self._shape[0])

    def __iter__(self)->Iterator["np.generic"]:
        return iter(self._store)

    def __getattr__(self, name: str)->object:
        # Everything else is the array's own: `reshape` is what the flat read-modify-write
        # of an atomic reaches for, and it must answer a view of the same storage.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._store, name)

    def __repr__(self)->str:
        return f"<{self._what}{self._shape} dtype={self.dtype}>"
